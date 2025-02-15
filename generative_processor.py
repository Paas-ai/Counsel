from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import ElasticsearchStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datetime import datetime
import logging
import os
from typing import Dict, Optional, Any, List
from ..config import Config
from .database import update_processing_status
from .redis_connection import get_redis_service

logger = logging.getLogger(__name__)

class LangChainProcessor:
    def __init__(self):
        # Initialize S3 client
        self.s3_client = boto3.client('s3')
        
        # Initialize model and tokenizer
        self.model_path = self._get_model_path()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",  # Automatically choose best device (CPU/GPU)
            trust_remote_code=True
        )
        
        # Create text generation pipeline
        text_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=Config.LLAMA_MAX_TOKENS,
            temperature=Config.LLAMA_TEMPERATURE,
            top_p=Config.LLAMA_TOP_P,
            repetition_penalty=Config.LLAMA_REPEAT_PENALTY
        )
        
        # Initialize LangChain LLM
        self.llm = HuggingFacePipeline(pipeline=text_pipeline)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Initialize Elasticsearch store
        self.vector_store = ElasticsearchStore(
            es_url=Config.ELASTICSEARCH_CONFIG['hosts'][0],
            index_name=Config.ELASTICSEARCH_CONFIG['vector_index'],
            embedding=self.embeddings,
            es_user=Config.ELASTICSEARCH_CONFIG['username'],
            es_password=Config.ELASTICSEARCH_CONFIG['password']
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize conversation chain
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            memory=self.memory,
            verbose=True
        )
        
        self.redis_service = get_redis_service()

    def _get_model_path(self) -> str:
        """
        Get model path - downloads from S3 if not present locally
        """
        local_model_path = Config.LLAMA_MODEL_PATH
        
        # Check if model exists locally
        if not os.path.exists(local_model_path):
            logger.info("Model not found locally, downloading from S3...")
            
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
            
            # Download model from S3
            try:
                self.s3_client.download_file(
                    Config.S3_BUCKET,
                    Config.S3_MODEL_KEY,
                    local_model_path
                )
                logger.info("Model downloaded successfully")
            except Exception as e:
                logger.error(f"Error downloading model: {str(e)}")
                raise
        
        return local_model_path

    async def _prepare_conversation_context(self, conversation_id: str) -> list:
        """Retrieve conversation history from Redis"""
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
        """Update conversation history in Redis"""
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

            # Get conversation history
            history = await self._prepare_conversation_context(conversation_id)
            
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

            # Process with LangChain conversation chain
            response = self.conversation_chain({
                "question": text,
                "chat_history": history
            })
            
            # Extract answer from response
            answer = response['answer']
            
            # Update conversation history
            await self._update_conversation_history(
                conversation_id,
                text,
                answer
            )

            if status_data:
                status_data['steps'][-1].update({
                    'status': 'completed',
                    'completed_at': datetime.utcnow().isoformat()
                })
                await update_processing_status(user_id, status_data)

            return {
                'status': 'success',
                'response': answer,
                'metadata': {
                    'source_documents': response.get('source_documents', []),
                    'tokens_used': len(self.tokenizer.encode(text + answer))
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