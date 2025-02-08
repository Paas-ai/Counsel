# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:40:38 2025

@author: Kumanan
"""

from langchain.vectorstores import ElasticVectorSearch
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.vector_stores import ElasticsearchStore
from datetime import datetime
import logging
import numpy as np
from typing import Dict, Optional, Any, List
from ..config import Config
from .database import update_processing_status
from .redis_connection import get_redis_service

logger = logging.getLogger(__name__)

class LangchainProcessor:
    def __init__(self):
        # Initialize LlamaCpp embeddings
        self.embeddings = LlamaCppEmbeddings(
            model_path=Config.LLAMA_MODEL_PATH,
            n_ctx=Config.LLAMA_MAX_TOKENS
        )
        
        # Initialize LLM
        self.llm = LlamaCpp(
            model_path=Config.LLAMA_MODEL_PATH,
            temperature=Config.LLAMA_TEMPERATURE,
            max_tokens=Config.LLAMA_MAX_TOKENS,
            top_p=Config.LLAMA_TOP_P,
            n_ctx=4096
        )
        
        # Initialize Elasticsearch client
        self.es_client = Elasticsearch(**Config.ELASTICSEARCH_CONFIG)
        
        self.redis_service = get_redis_service()

    async def _retrieve_relevant_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Two-stage document retrieval:
        1. Get initial candidates using Elasticsearch keyword search
        2. Rank candidates using semantic similarity with embeddings
        """
        try:
            # Stage 1: Elasticsearch keyword search
            es_results = await self._elasticsearch_search(query, top_k * 3)
            if not es_results:
                return []

            # Stage 2: Semantic ranking using embeddings
            ranked_results = await self._semantic_ranking(query, es_results, top_k)
            return ranked_results

        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}")
            return []

    async def _elasticsearch_search(self, query: str, size: int) -> List[Dict]:
        """
        Perform initial keyword search using Elasticsearch
        """
        try:
            # Build Elasticsearch query
            es_query = {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "chunks.text"],
                    "fuzziness": "AUTO"
                }
            }
            
            # Execute search
            response = await self.es_client.search(
                index=Config.ELASTICSEARCH_CONFIG['document_index'],
                query=es_query,
                size=size
            )
            
            # Extract and format results
            candidates = []
            for hit in response['hits']['hits']:
                for chunk in hit['_source'].get('chunks', []):
                    candidates.append({
                        'text': chunk['text'],
                        'source': hit['_source'].get('file_name', 'Unknown'),
                        'es_score': hit['_score']
                    })
            
            return candidates

        except Exception as e:
            logger.error(f"Elasticsearch search error: {str(e)}")
            raise

    async def _semantic_ranking(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """
        Rank candidates using semantic similarity
        """
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            
            # Calculate semantic similarity for each candidate
            for candidate in candidates:
                # Generate embedding for candidate text
                doc_embedding = await self._generate_embedding(candidate['text'])
                
                # Calculate semantic similarity
                semantic_score = self._calculate_cosine_similarity(
                    query_embedding,
                    doc_embedding
                )
                
                # Combine scores (30% ES score, 70% semantic score)
                candidate['semantic_score'] = semantic_score
                candidate['final_score'] = (
                    0.3 * self._normalize_score(candidate['es_score']) +
                    0.7 * semantic_score
                )
            
            # Sort by final score and return top_k
            ranked_results = sorted(
                candidates,
                key=lambda x: x['final_score'],
                reverse=True
            )[:top_k]
            
            return ranked_results

        except Exception as e:
            logger.error(f"Semantic ranking error: {str(e)}")
            raise

    async def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for text using LlamaCpp
        """
        try:
            # Generate raw embedding
            raw_embedding = self.embeddings.embed_query(text)
            
            # Convert to numpy array for efficient computation
            return np.array(raw_embedding, dtype=np.float32)

        except Exception as e:
            logger.error(f"Embedding generation error: {str(e)}")
            raise

    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        """
        try:
            # Calculate dot product
            dot_product = np.dot(vec1, vec2)
            
            # Calculate L2 norms (magnitudes)
            query_norm = np.linalg.norm(vec1)
            doc_norm = np.linalg.norm(vec2)
            
            # Ensure non-zero division
            if query_norm == 0.0 or doc_norm == 0.0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = dot_product / (query_norm * doc_norm)
            
            # Ensure result is between -1 and 1
            return float(np.clip(similarity, -1.0, 1.0))

        except Exception as e:
            logger.error(f"Cosine similarity calculation error: {str(e)}")
            raise

    async def process_text(self, 
                          text: str, 
                          user_id: str, 
                          conversation_id: str,
                          status_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process user query with document retrieval and LLM
        """
        try:
            if status_data:
                status_data['steps'].append({
                    'step': 'document_retrieval',
                    'status': 'in_progress',
                    'timestamp': datetime.utcnow().isoformat()
                })
                await update_processing_status(user_id, status_data)

            # Retrieve relevant documents
            relevant_docs = await self._retrieve_relevant_documents(text)
            
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
            conversation_history = await self._prepare_conversation_context(conversation_id)
            
            # Format context from retrieved documents
            context = "\n\n".join([doc['text'] for doc in relevant_docs])
            
            # Format prompt with context and history
            prompt = self._format_prompt(conversation_history, text, context)
            
            # Generate response using LLM
            response = self.llm(prompt)
            
            # Update conversation history
            await self._update_conversation_history(
                conversation_id,
                text,
                response
            )

            if status_data:
                status_data['steps'][-1].update({
                    'status': 'completed',
                    'completed_at': datetime.utcnow().isoformat()
                })
                await update_processing_status(user_id, status_data)

            return {
                'status': 'success',
                'response': response,
                'metadata': {
                    'documents_used': [
                        {'source': doc['source'], 'relevance_score': doc['final_score']}
                        for doc in relevant_docs
                    ]
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

    async def _prepare_conversation_context(self, conversation_id: str) -> list:
        """
        Retrieve conversation history from Redis
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
        """
        try:
            conversation_key = f"conversation_history:{conversation_id}"
            history = await self._prepare_conversation_context(conversation_id)
            
            # Add new messages
            history.extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_response}
            ])
            
            # Maintain context window
            max_history = Config.LLAMA_MAX_HISTORY
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

    def _format_prompt(self, 
                      conversation_history: list, 
                      current_message: str,
                      context: str) -> str:
        """
        Format the prompt with retrieved context and conversation history
        
        Args:
            conversation_history (list): List of previous conversation messages
            current_message (str): Current user query
            context (str): Retrieved document content
            
        Returns:
            str: Formatted prompt for LLM
        """
        # Format previous conversation
        formatted_history = ""
        for msg in conversation_history:
            if msg["role"] == "user":
                formatted_history += f"User: {msg['content']}\n"
            else:
                formatted_history += f"Assistant: {msg['content']}\n"

        # Create prompt with context
        prompt = f"""Use the following retrieved documents to help answer the user's question:

                Retrieved Documents:
                    {context}
                
                Previous Conversation:
                    {formatted_history}
                    
                Current Question:
                    User: {current_message}
                    Assistant: """
                        
        return prompt