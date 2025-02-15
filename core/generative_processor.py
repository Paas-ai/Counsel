from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.callbacks import AsyncCallbackHandler
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    Document as LlamaDocument
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.node_parser import SimpleNodeParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from elasticsearch import AsyncElasticsearch
from datetime import datetime
import logging
import json
from typing import Dict, Optional, Any, List
from ..config import Config
from .database import update_processing_status
from .redis_connection import get_redis_service

logger = logging.getLogger(__name__)

class StatusUpdateCallback(AsyncCallbackHandler):
    """Callback handler for updating processing status"""
    
    def __init__(self, user_id: str, status_data: Dict):
        self.user_id = user_id
        self.status_data = status_data

    async def on_chain_start(self, serialized: Dict, inputs: Dict, **kwargs):
        """Called when chain starts running"""
        self.status_data['steps'].append({
            'step': serialized.get('name', 'unknown'),
            'status': 'in_progress',
            'timestamp': datetime.utcnow().isoformat()
        })
        await update_processing_status(self.user_id, self.status_data)

    async def on_chain_end(self, outputs: Dict, **kwargs):
        """Called when chain finishes running"""
        self.status_data['steps'][-1].update({
            'status': 'completed',
            'completed_at': datetime.utcnow().isoformat()
        })
        await update_processing_status(self.user_id, self.status_data)

    async def on_chain_error(self, error: Exception, **kwargs):
        """Called when chain errors"""
        self.status_data['steps'][-1].update({
            'status': 'error',
            'error': str(error),
            'error_timestamp': datetime.utcnow().isoformat()
        })
        await update_processing_status(self.user_id, self.status_data)

class OrchestrationProcessor:
    def __init__(self):
        # Initialize Elasticsearch client
        self.es_client = AsyncElasticsearch(
            hosts=Config.ELASTICSEARCH_CONFIG['hosts'],
            basic_auth=(
                Config.ELASTICSEARCH_CONFIG['username'],
                Config.ELASTICSEARCH_CONFIG['password']
            )
        )
        
        # Initialize model and tokenizer
        self.model_name = Config.LLAMA_MODEL_NAME
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=Config.HUGGINGFACE_TOKEN
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=Config.HUGGINGFACE_TOKEN,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Initialize LLM
        text_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=Config.LLAMA_MAX_TOKENS,
            temperature=Config.LLAMA_TEMPERATURE,
            top_p=Config.LLAMA_TOP_P,
            repetition_penalty=Config.LLAMA_REPEAT_PENALTY
        )
        self.llm = HuggingFacePipeline(pipeline=text_pipeline)
        
        # Initialize LlamaIndex components
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model,
            node_parser=SimpleNodeParser()
        )
        
        # Initialize chains
        self._initialize_chains()
        
        self.redis_service = get_redis_service()
        logger.info("Orchestration processor initialized successfully")

    def _initialize_chains(self):
        """Initialize LangChain chains for orchestration"""
        
        # Document Search Chain
        search_prompt = PromptTemplate(
            input_variables=["query"],
            template="Perform a search for documents relevant to: {query}"
        )
        self.search_chain = LLMChain(
            llm=self.llm,
            prompt=search_prompt,
            output_key="search_results",
            verbose=True
        )
        
        # Ranking Chain
        ranking_prompt = PromptTemplate(
            input_variables=["search_results", "query"],
            template="""
            Rank the following search results for the query: {query}
            
            Search Results:
            {search_results}
            
            Return the top ranked documents.
            """
        )
        self.ranking_chain = LLMChain(
            llm=self.llm,
            prompt=ranking_prompt,
            output_key="ranked_documents",
            verbose=True
        )
        
        # Response Generation Chain
        response_prompt = PromptTemplate(
            input_variables=["ranked_documents", "query"],
            template="""
            Based on the following ranked documents:
            {ranked_documents}
            
            Generate a comprehensive response to: {query}
            """
        )
        self.response_chain = LLMChain(
            llm=self.llm,
            prompt=response_prompt,
            output_key="final_response",
            verbose=True
        )
        
        # Combine chains
        self.full_chain = SequentialChain(
            chains=[self.search_chain, self.ranking_chain, self.response_chain],
            input_variables=["query"],
            output_variables=["search_results", "ranked_documents", "final_response"],
            verbose=True
        )

    async def _perform_text_search(self, query: str) -> List[Dict]:
        """Elasticsearch text search"""
        try:
            response = await self.es_client.search(
                index=Config.ELASTICSEARCH_CONFIG['document_index'],
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["content", "title"],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    },
                    "size": 10
                }
            )
            
            return [hit['_source'] for hit in response['hits']['hits']]
            
        except Exception as e:
            logger.error(f"Elasticsearch search error: {str(e)}")
            raise

    async def _perform_semantic_ranking(self, documents: List[Dict], query: str) -> List[LlamaDocument]:
        """LlamaIndex semantic ranking"""
        try:
            llama_documents = [
                LlamaDocument(
                    text=doc.get('content', ''),
                    metadata={
                        'title': doc.get('title', ''),
                        'source': doc.get('source', '')
                    }
                ) for doc in documents
            ]
            
            index = VectorStoreIndex.from_documents(
                llama_documents,
                service_context=self.service_context
            )
            
            query_engine = index.as_query_engine(
                similarity_top_k=len(llama_documents)
            )
            
            response = query_engine.query(query)
            return response.source_nodes
            
        except Exception as e:
            logger.error(f"Semantic ranking error: {str(e)}")
            raise

    async def process_text(self, 
                          text: str, 
                          user_id: str, 
                          conversation_id: str,
                          status_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Process query through the orchestrated pipeline"""
        try:
            # Initialize callback handler for status updates
            callbacks = []
            if status_data:
                callbacks.append(StatusUpdateCallback(user_id, status_data))

            # Step 1: Text Search using Elasticsearch
            documents = await self._perform_text_search(text)
            
            # Step 2: Semantic Ranking using LlamaIndex
            ranked_nodes = await self._perform_semantic_ranking(documents, text)
            
            # Step 3: Process through LangChain orchestration
            result = await self.full_chain.arun(
                query=text,
                callbacks=callbacks
            )
            
            # Update conversation history
            await self._update_conversation_history(
                conversation_id,
                text,
                result['final_response']
            )

            return {
                'status': 'success',
                'response': result['final_response'],
                'metadata': {
                    'documents_searched': len(documents),
                    'documents_ranked': len(ranked_nodes),
                    'top_documents': [
                        {
                            'title': node.metadata.get('title', ''),
                            'source': node.metadata.get('source', ''),
                            'score': node.score if hasattr(node, 'score') else 0
                        }
                        for node in ranked_nodes[:3]
                    ],
                    'intermediate_results': {
                        'search_results': result['search_results'],
                        'ranked_documents': result['ranked_documents']
                    }
                }
            }

        except Exception as e:
            error_message = str(e)
            logger.error(f"Processing error: {error_message}")
            return {
                'status': 'error',
                'error': error_message
            }