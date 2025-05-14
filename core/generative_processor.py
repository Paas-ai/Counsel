# -*- coding: utf-8 -*-
"""
Updated Generative Processor with LlamaIndex and Elasticsearch Integration

This module integrates the PDF processing capabilities with LlamaIndex for
improved document retrieval and LLM orchestration.
"""

import json
import logging
import os
import boto3
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# LlamaIndex imports
from llama_index import (
    VectorStoreIndex,
    ServiceContext
)
from llama_index.llms import HuggingFaceLLM
from llama_index.vector_stores import ElasticsearchStore
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.prompts import PromptTemplate
from llama_index.schema import NodeWithScore, TextNode
from transformers import AutoTokenizer, AutoModelForCausalLM
from elasticsearch import Elasticsearch
import torch

# Local imports
from ..config import Config
from .database import update_processing_status
from .redis_connection import get_redis_service

logger = logging.getLogger(__name__)

# Define custom prompt templates
QA_TEMPLATE = """You are a helpful, friendly AI assistant. 
Use the following context information to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context information:
{context_str}

User question: {query_str}

Your answer:"""

CHAT_TEMPLATE = """You are a helpful, friendly AI assistant for audio chat services. 
The following is a conversation between a human and you. 
Use the following context information and conversation history to respond to the user.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context information:
{context_str}

Conversation history:
{chat_history}

User: {query_str}
Assistant:"""

class LlmProcessor:
    """Process user queries with LlamaIndex and Elasticsearch integration
    
    This processor combines text search in Elasticsearch with semantic search
    using LlamaIndex and provides fallback mechanisms for model loading.
    """
    
    def __init__(self):
        """Initialize LlamaIndex processor"""
        # S3 client for model downloading if needed
        self.s3_client = boto3.client('s3')
        
        # Load model with fallback options
        self.model_path, self.tokenizer, self.model = self._load_model_with_fallback()
        
        # Initialize LlamaIndex LLM
        self.llm = HuggingFaceLLM(
            context_window=4096,
            max_new_tokens=Config.LLAMA_MAX_TOKENS,
            generate_kwargs={
                "temperature": Config.LLAMA_TEMPERATURE,
                "top_p": Config.LLAMA_TOP_P,
                "repetition_penalty": Config.LLAMA_REPEAT_PENALTY
            },
            tokenizer=self.tokenizer,
            model=self.model,
            model_name=self.model_path,
            device_map="auto"
        )
        
        # Initialize embeddings
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Initialize Elasticsearch client
        self.es_client = Elasticsearch(
            hosts=Config.ELASTICSEARCH_CONFIG['hosts'],
            basic_auth=(
                Config.ELASTICSEARCH_CONFIG['username'],
                Config.ELASTICSEARCH_CONFIG['password']
            ),
            retry_on_timeout=True,
            max_retries=3
        )
        
        # Initialize vector store
        self.vector_store = ElasticsearchStore(
            es_client=self.es_client,
            index_name=Config.ELASTICSEARCH_CONFIG['vector_index'],
            embed_dim=self.embed_model.get_embedding_dimension()
        )
        
        # Create custom prompt templates
        self.qa_prompt = PromptTemplate(QA_TEMPLATE)
        self.chat_prompt = PromptTemplate(CHAT_TEMPLATE)
        
        # Initialize ServiceContext
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model
        )
        
        # Redis service for conversation management
        self.redis_service = get_redis_service()
    
    def _load_model_with_fallback(self) -> Tuple[str, AutoTokenizer, AutoModelForCausalLM]:
        """
        Load LLM model with fallback options
        
        This method tries several approaches to load the model:
        1. Try loading from the configured local path
        2. If not found, try downloading from S3
        3. If S3 download fails, try loading a smaller backup model
        4. If all else fails, raise an informative error
        
        Returns:
            Tuple of (model_path, tokenizer, model)
        """
        try:
            # First attempt: Load from configured path
            model_path = Config.LLAMA_MODEL_PATH
            logger.info(f"Attempting to load model from: {model_path}")
            
            if not os.path.exists(model_path):
                logger.info(f"Model not found at {model_path}, checking if directory exists")
                # Check if it's a directory path that exists but needs to be downloaded
                model_dir = os.path.dirname(model_path)
                if not os.path.exists(model_dir):
                    logger.info(f"Creating model directory: {model_dir}")
                    os.makedirs(model_dir, exist_ok=True)
                
                # Try downloading from S3 if configured
                if hasattr(Config, 'S3_BUCKET') and hasattr(Config, 'S3_MODEL_KEY'):
                    logger.info(f"Downloading model from S3: {Config.S3_BUCKET}/{Config.S3_MODEL_KEY}")
                    try:
                        self.s3_client.download_file(
                            Config.S3_BUCKET,
                            Config.S3_MODEL_KEY,
                            model_path
                        )
                        logger.info("Model downloaded successfully from S3")
                    except Exception as e:
                        logger.error(f"Failed to download model from S3: {str(e)}")
                        # Try fallback model path if defined
                        if hasattr(Config, 'FALLBACK_MODEL_PATH'):
                            model_path = Config.FALLBACK_MODEL_PATH
                            logger.info(f"Trying fallback model path: {model_path}")
                            if not os.path.exists(model_path):
                                raise ValueError(f"Fallback model not found at: {model_path}")
                        else:
                            raise ValueError("Model not found and no fallback configured")
            
            # Check available device
            device_map = "auto"
            if torch.cuda.is_available():
                logger.info("CUDA available, using GPU")
                # Check VRAM and adjust loading strategy if needed
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"Available VRAM: {vram_gb:.2f} GB")
                
                # For very low VRAM, use CPU instead
                if vram_gb < 4:
                    logger.warning("Low VRAM detected, falling back to CPU")
                    device_map = "cpu"
            else:
                logger.info("CUDA not available, using CPU")
                device_map = "cpu"
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Load model with appropriate settings
            logger.info(f"Loading model from {model_path} to {device_map}")
            model_kwargs = {
                "device_map": device_map,
                "trust_remote_code": True
            }
            
            # Add low_cpu_mem_usage if available memory is limited
            if device_map == "cpu":
                import psutil
                available_memory_gb = psutil.virtual_memory().available / (1024**3)
                logger.info(f"Available system memory: {available_memory_gb:.2f} GB")
                if available_memory_gb < 16:
                    model_kwargs["low_cpu_mem_usage"] = True
            
            # Load model with appropriate settings
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            logger.info("Model loaded successfully")
            return model_path, tokenizer, model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            
            # Final fallback: try loading a small model from HF Hub if allowed
            if hasattr(Config, 'ALLOW_REMOTE_FALLBACK') and Config.ALLOW_REMOTE_FALLBACK:
                try:
                    fallback_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                    logger.info(f"Attempting to load small fallback model from HuggingFace: {fallback_model_name}")
                    
                    # Load tokenizer and model from HF Hub
                    tokenizer = AutoTokenizer.from_pretrained(fallback_model_name)
                    model = AutoModelForCausalLM.from_pretrained(
                        fallback_model_name,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    
                    logger.info("Fallback model loaded successfully from HuggingFace Hub")
                    return fallback_model_name, tokenizer, model
                except Exception as hub_error:
                    logger.error(f"Failed to load fallback model from HuggingFace Hub: {str(hub_error)}")
            
            # If all fallbacks fail, raise the original error
            raise ValueError(f"Failed to load model: {str(e)}")
        
    async def _perform_text_search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Perform text-based search in Elasticsearch document index
        
        Args:
            query: User query
            limit: Maximum number of results to return
            
        Returns:
            List of document chunks from text search
        """
        try:
            # Text search query
            text_query = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content^3", "section_title^2"],
                        "type": "most_fields",
                        "operator": "and",
                        "fuzziness": "AUTO"
                    }
                },
                "size": limit
            }
            
            results = self.es_client.search(
                index=Config.ELASTICSEARCH_CONFIG['document_index'],
                body=text_query
            )
            
            # Extract results
            documents = []
            for hit in results["hits"]["hits"]:
                doc = hit["_source"]
                doc["score"] = hit["_score"]
                documents.append(doc)
            
            logger.info(f"Text search returned {len(documents)} results")
            return documents
            
        except Exception as e:
            logger.error(f"Error in text search: {str(e)}")
            raise
    
    def _create_nodes_from_docs(self, docs: List[Dict]) -> List[TextNode]:
        """
        Convert Elasticsearch document chunks to LlamaIndex nodes
        
        Args:
            docs: List of document chunks from Elasticsearch
            
        Returns:
            List of TextNode objects for LlamaIndex
        """
        nodes = []
        for doc in docs:
            # Create metadata
            metadata = {
                "document_id": doc.get("document_id"),
                "chunk_id": doc.get("chunk_id"),
                "section_title": doc.get("section_title", ""),
                "section_level": doc.get("section_level", 0),
                "score": doc.get("score", 0)
            }
            
            # Create node
            node = TextNode(
                text=doc.get("content", ""),
                metadata=metadata,
                id_=doc.get("chunk_id")
            )
            nodes.append(node)
        
        return nodes
    
    async def _vector_search(self, query: str, docs: List[Dict], limit: int = 5) -> List[NodeWithScore]:
        """
        Perform vector search using pre-computed embeddings from Elasticsearch
        
        Args:
            query: User query
            docs: List of document chunks from text search
            limit: Maximum number of results
            
        Returns:
            List of nodes with similarity scores
        """
        try:
            # Extract chunk IDs from text search results
            chunk_ids = [doc.get("chunk_id") for doc in docs if "chunk_id" in doc]
            
            if not chunk_ids:
                raise ValueError("No valid chunk IDs found in text search results")
            
            # Generate embedding for the query
            query_embedding = self.embed_model.get_text_embedding(query)
            
            # Fetch pre-computed embeddings for these chunks from vector index
            vector_query = {
                "query": {
                    "terms": {
                        "_id": chunk_ids
                    }
                },
                "size": len(chunk_ids)  # Get all matched chunks
            }
            
            vector_results = self.es_client.search(
                index=Config.ELASTICSEARCH_CONFIG['vector_index'],
                body=vector_query
            )
            
            # Create nodes with pre-computed embeddings
            nodes_with_embeddings = []
            
            # Map doc_id to document content
            doc_map = {doc.get("chunk_id"): doc for doc in docs if "chunk_id" in doc}
            
            # Process vector results
            for hit in vector_results["hits"]["hits"]:
                chunk_id = hit["_id"]
                vector_source = hit["_source"]
                
                # Get corresponding document content
                if chunk_id in doc_map:
                    doc = doc_map[chunk_id]
                    
                    # Create metadata
                    metadata = {
                        "document_id": doc.get("document_id"),
                        "chunk_id": chunk_id,
                        "section_title": doc.get("section_title", ""),
                        "section_level": doc.get("section_level", 0),
                        "text_search_score": doc.get("score", 0)
                    }
                    
                    # Create node with embedding and text
                    node = TextNode(
                        text=doc.get("content", ""),
                        metadata=metadata,
                        id_=chunk_id,
                        embedding=vector_source.get("embedding")
                    )
                    
                    nodes_with_embeddings.append(node)
            
            logger.info(f"Retrieved {len(nodes_with_embeddings)} nodes with pre-computed embeddings")
            
            # Calculate cosine similarity between query embedding and document embeddings
            nodes_with_scores = []
            for node in nodes_with_embeddings:
                if node.embedding:
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, node.embedding)
                    nodes_with_scores.append(NodeWithScore(node=node, score=similarity))
            
            # Sort by similarity score
            nodes_with_scores.sort(key=lambda x: x.score, reverse=True)
            
            # Return top k results
            top_results = nodes_with_scores[:limit]
            logger.info(f"Vector search returned {len(top_results)} results")
            return top_results
            
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            # If vector search fails, return nodes sorted by text search score
            logger.info("Falling back to text search results")
            
            # Create nodes from docs without embeddings
            nodes = []
            for doc in docs:
                metadata = {
                    "document_id": doc.get("document_id"),
                    "chunk_id": doc.get("chunk_id"),
                    "section_title": doc.get("section_title", ""),
                    "section_level": doc.get("section_level", 0),
                    "score": doc.get("score", 0)
                }
                
                node = TextNode(
                    text=doc.get("content", ""),
                    metadata=metadata,
                    id_=doc.get("chunk_id")
                )
                nodes.append(node)
            
            # Return nodes sorted by text search score
            sorted_nodes = sorted(nodes, key=lambda x: x.metadata.get("score", 0), reverse=True)
            return [NodeWithScore(node=node, score=node.metadata.get("score", 0)) 
                    for node in sorted_nodes[:limit]]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (between -1 and 1)
        """
        # Ensure numpy is imported at the top of the file if not already
        import numpy as np
        
        # Convert to numpy arrays
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # Handle zero division
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        similarity = dot_product / (norm_v1 * norm_v2)
        
        # Ensure result is within bounds
        return max(min(similarity, 1.0), -1.0)
    
    async def _prepare_conversation_context(self, conversation_id: str) -> List[Dict]:
        """
        Retrieve conversation history from Redis
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            List of conversation messages
        """
        try:
            conversation_key = f"conversation_history:{conversation_id}"
            history = self.redis_service.get_value(conversation_key)
            return json.loads(history) if history else []
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            return []
    
    async def _update_conversation_history(self, 
                                          conversation_id: str, 
                                          user_message: str, 
                                          assistant_response: str) -> None:
        """
        Update conversation history in Redis
        
        Args:
            conversation_id: Conversation identifier
            user_message: User's message
            assistant_response: Assistant's response
        """
        try:
            conversation_key = f"conversation_history:{conversation_id}"
            history = await self._prepare_conversation_context(conversation_id)
            
            # Add new messages
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": assistant_response})
            
            # Keep only last N messages to manage context window
            max_history = Config.LLAMA_MAX_HISTORY or 10
            if len(history) > max_history * 2:
                history = history[-max_history * 2:]
            
            # Store updated history
            self.redis_service.set_with_expiry(
                conversation_key,
                json.dumps(history),
                3600  # 1 hour expiry
            )
        except Exception as e:
            logger.error(f"Error updating conversation history: {str(e)}")
            raise
    
    def _format_document_for_prompt(self, nodes: List[NodeWithScore]) -> str:
        """
        Format retrieved document nodes for insertion into prompt
        
        Args:
            nodes: List of retrieved nodes with similarity scores
            
        Returns:
            Formatted document text for prompt context
        """
        if not nodes:
            return "No relevant documents found."
        
        formatted_docs = []
        for i, node_with_score in enumerate(nodes):
            node = node_with_score.node
            score = node_with_score.score
            
            # Extract metadata
            doc_id = node.metadata.get("document_id", "Unknown")
            section = node.metadata.get("section_title", "Unknown section")
            
            # Format document with metadata and content
            doc_text = f"Document {i+1} [ID: {doc_id} | Section: {section} | Relevance: {score:.2f}]\n"
            doc_text += f"{node.text}\n"
            
            formatted_docs.append(doc_text)
        
        # Join all formatted documents with separators
        return "\n---\n".join(formatted_docs)
    
    async def process_text(self, 
                          text: str, 
                          user_id: str, 
                          conversation_id: str,
                          status_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process text query with document retrieval and LLM
        
        Args:
            text: User query text
            user_id: User identifier
            conversation_id: Conversation identifier
            status_data: Optional status tracking data
            
        Returns:
            Processing response with LLM answer
        """
        try:
            # Update status if provided
            if status_data:
                status_data['steps'].append({
                    'step': 'document_retrieval',
                    'status': 'in_progress',
                    'timestamp': datetime.utcnow().isoformat()
                })
                await update_processing_status(user_id, status_data)
            
            # Step 1: Perform text search in Elasticsearch
            text_search_results = await self._perform_text_search(text)
            
            # Step 2: Perform semantic search with pre-computed vector embeddings
            retrieved_nodes = await self._vector_search(text, text_search_results)
            
            # Update status for LLM processing
            if status_data:
                status_data['steps'][-1].update({
                    'status': 'completed',
                    'completed_at': datetime.utcnow().isoformat()
                })
                status_data['steps'].append({
                    'step': 'llm_processing',
                    'status': 'in_progress',
                    'timestamp': datetime.utcnow().isoformat()
                })
                await update_processing_status(user_id, status_data)
            
            # Get conversation history
            history = await self._prepare_conversation_context(conversation_id)
            
            # Select prompt template based on conversation state
            if history:
                # Conversation mode
                prompt_template = self.chat_prompt
                chat_history = self._format_chat_history(history)
                
                # Create query engine with chat prompt
                query_engine = index = VectorStoreIndex(
                    [node.node for node in retrieved_nodes], 
                    service_context=self.service_context
                ).as_query_engine(
                    text_qa_template=prompt_template,
                    similarity_top_k=len(retrieved_nodes)
                )
                
                # Format document context (for logging or debugging)
                _ = self._format_document_for_prompt(retrieved_nodes)
                
                # Generate response with conversation context
                response = query_engine.query(
                    text,
                    extra_info={"chat_history": chat_history}
                )
            else:
                # QA mode for first interaction
                prompt_template = self.qa_prompt
                
                # Create query engine with QA prompt
                query_engine = index = VectorStoreIndex(
                    [node.node for node in retrieved_nodes], 
                    service_context=self.service_context
                ).as_query_engine(
                    text_qa_template=prompt_template,
                    similarity_top_k=len(retrieved_nodes)
                )
                
                # Format document context (for logging or debugging if needed)
                _ = self._format_document_for_prompt(retrieved_nodes)
                
                # Generate response for first question
                response = query_engine.query(text)
            
            # Extract answer
            answer = str(response).strip()
            
            # Update conversation history
            await self._update_conversation_history(
                conversation_id,
                text,
                answer
            )
            
            # Update status if provided
            if status_data:
                status_data['steps'][-1].update({
                    'status': 'completed',
                    'completed_at': datetime.utcnow().isoformat()
                })
                await update_processing_status(user_id, status_data)
            
            # Prepare source documents metadata with formatted context
            source_docs = []
            for node in retrieved_nodes:
                source_docs.append({
                    'document_id': node.node.metadata.get('document_id'),
                    'section_title': node.node.metadata.get('section_title'),
                    'score': node.score
                })
            
            return {
                'status': 'success',
                'response': answer,
                'metadata': {
                    'source_documents': source_docs,
                    'tokens_generated': len(self.tokenizer.encode(answer))
                }
            }
        
        except Exception as e:
            error_message = str(e)
            if status_data:
                status_data['steps'][-1].update({
                    'status': 'error',
                    'error': error_message,
                    'error_timestamp': datetime.utcnow().isoformat()
                })
                await update_processing_status(user_id, status_data)
            
            logger.error(f"Processing error: {error_message}")
            return {
                'status': 'error',
                'error': error_message
            }