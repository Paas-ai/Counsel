import requests
import uuid
import time
from datetime import datetime
from typing import Dict, Optional, Any
import logging
from app.config import Config
from .database import update_processing_status
from .generative_processor import LangchainProcessor
from celery import Celery
from celery.result import AsyncResult
import aiohttp
import asyncio
import json
import hashlib
from cryptography.fernet import Fernet
from .redis_connection import get_redis_service

logger = logging.getLogger(__name__)


# Initialize Celery and Redis
celery = Celery('audio_processing', broker='redis://localhost:6379/0')

class ConversationManager:
    def __init__(self):
        self.redis_service = get_redis_service()

    async def create_conversation(self, user_id):
        """Create a new conversation and return conversation_id"""
        try:
            conversation_id = str(uuid.uuid4())
            conversation_data = {
                'user_id': user_id,
                'started_at': datetime.utcnow().isoformat(),
                'status': 'active'
            }
            
            # Store conversation data in Redis
            self.redis_service.set_with_expiry(
                f"conversation:{conversation_id}",
                json.dumps(conversation_data),
                3600  # Expire after 1 hour of inactivity
            )
            
            logger.info(f"Created new conversation {conversation_id} for user {user_id}")
            return conversation_id
            
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            raise

    async def end_conversation(self, conversation_id):
        """Mark conversation as completed"""
        try:
            conversation_key = f"conversation:{conversation_id}"
            conversation_data = self.redis_service.get_value(conversation_key)
            
            if conversation_data:
                data = json.loads(conversation_data)
                data['status'] = 'completed'
                data['ended_at'] = datetime.utcnow().isoformat()
                
                # Update conversation data
                self.redis_service.set_with_expiry(
                    conversation_key,
                    json.dumps(data),
                    3600  # Keep completed conversation data for 1 hour
                )
                
                # Clean up any associated config data
                self.redis_service.delete_key(f"service_config:{conversation_id}")
                
                logger.info(f"Ended conversation {conversation_id}")
                
        except Exception as e:
            logger.error(f"Error ending conversation: {str(e)}")
            raise

    async def validate_conversation(self, conversation_id):
        """Check if conversation exists and is active"""
        try:
            conversation_key = f"conversation:{conversation_id}"
            conversation_data = self.redis_service.get_value(conversation_key)
            
            if not conversation_data:
                raise ValueError("Conversation not found")
                
            data = json.loads(conversation_data)
            if data['status'] != 'active':
                raise ValueError("Conversation is not active")
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating conversation: {str(e)}")
            raise
            
    async def get_service_config(self, conversation_id):
        """Get cached service configuration parameters"""
        try:
            config_key = f"service_config:{conversation_id}"
            config_data = self.redis_service.get_value(config_key)
            return json.loads(config_data) if config_data else None
        except Exception as e:
            logger.error(f"Error getting service config: {str(e)}")
            return None

    async def store_service_config(self, conversation_id, config_data):
        """Store service configuration parameters"""
        try:
            config_key = f"service_config:{conversation_id}"
            self.redis_service.set_with_expiry(
                config_key,
                json.dumps(config_data),
                3600  # Cache for 1 hour
            )
        except Exception as e:
            logger.error(f"Error storing service config: {str(e)}")
            raise
            
class WebhookManager:
    def __init__(self):
        self.max_retries = Config.WEBHOOK_MAX_RETRIES
        self.base_delay = Config.WEBHOOK_BASE_DELAY
        self.timeout = Config.WEBHOOK_TIMEOUT
        self.api_key = Config.API_KEY
        self.access_id = Config.Access_ID
        self.base_url = Config.URL
        
    def prepare_webhook_headers(self, auth_key=None, auth_value=None, is_config=False):
        """Prepare headers for configuration/authentication calls"""
        try:
            if is_config:
                if not self.api_key:
                    raise ValueError("API key not found")
                    
                if not self.access_id:
                    raise ValueError("Access id not found")
                
                logger.info("Preparing config header")
                return {
                    'Content-Type': 'application/json',
                    'Content-Length': 'application/json',
                    'Host': 'application/json',
                    'Accept-Post': '*/*',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'User-Agent': 'AudioChatApp/1.0',
                    'userID': self.access_id,
                    'ulcaApiKey': f'Bearer {self.api_key}'
                    }
            else:
                logger.info("Preparing compute header")
                return {
                    'Content-Type': 'application/json',
                    'Content-Length': 'application/json',
                    'Host': 'application/json',
                    'Accept-Post': '*/*',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'User-Agent': 'AudioChatApp/1.0',
                    f'{auth_key}': f'Bearer {auth_value}'
                    }
        except Exception as e:
            logger.error(f"Error preparing headers: {str(e)}")
            raise
    
    async def send_webhook(self, webhook_type, payload, session, callbackUrl=None, auth_key=None, auth_value=None, is_config=False):
        """Send webhook with appropriate headers"""
        try:
            url = callbackUrl if callbackUrl else self.base_url
            if not url:
                raise ValueError(f"No webhook URL configured for {webhook_type}")
            
            headers = self.prepare_webhook_headers(auth_key, auth_value, is_config)
                
            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()
                        logger.info(f"Webhook {webhook_type} sent successfully")
                        return result
                    
                except aiohttp.ClientError as e:
                    delay = self.base_delay * (2 ** attempt)
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(delay)

        except Exception as e:
            logger.error(f"Error sending webhook {webhook_type}: {str(e)}")
            raise
            
class WebhookEvents:
    def __init__(self):
        self.webhook_manager = WebhookManager()
        self.conversation_manager = ConversationManager()
        self.encryption_manager = EncryptionManager()
        self.redis_service = get_redis_service()
    
    async def fetch_service_config(self, session, conversation_id, config_arguments):
        """Fetch configuration parameters from external service"""
        try:
            PipeLineId = getattr(Config, "PipeLineId")
            if config_arguments['process_type'] == 'asr_nmt':
                config_payload = {
                                    "pipelineTasks": [
                                        {
                                            "taskType": "asr",
                                            "config": {
                                                "language": {
                                                    "sourceLanguage": config_arguments['source_language']
                                                    }
                                                }
                                            },
                                        {
                                            "taskType": "translation",
                                            "config": {
                                                "language": {
                                                    "sourceLanguage": config_arguments['source_language'],
                                                    "targetLanguage": config_arguments['target_language']
                                                            }
                                                    }
                                            }
                                    ],  
                                    "pipelineRequestConfig": {
                                        "pipelineId": PipeLineId
                                        }
                                }
            elif config_arguments['process_type'] == 'nmt_tts':
                config_payload = {
                                    "pipelineTasks": [
                                        {
                                            "taskType": "translation",
                                            "config": {
                                                "language": {
                                                    "sourceLanguage": config_arguments['source_language'],
                                                    "targetLanguage": config_arguments['target_language']
                                                            }
                                                    }
                                            },
                                        {
                                            "taskType": "tts",
                                            "config": {
                                                "language": {
                                                    "sourceLanguage": config_arguments['target_language']
                                                            }
                                                    }
                                            }
                                    ],  
                                    "pipelineRequestConfig": {
                                        "pipelineId": PipeLineId
                                        }
                                }
            else:
                raise ValueError(f"Invalid process_type: {config_arguments['process_type']}")

            config_response = await self.webhook_manager.send_webhook(
                'asr_nmt',
                config_payload,
                session,
                is_config=True
            )

            await self.conversation_manager.store_service_config(
                conversation_id,
                config_response
            )

            return config_response

        except Exception as e:
            logger.error(f"Error fetching service config: {str(e)}")
            raise
            
    async def process_asr_nmt(self, audio_path, user_id, session, status_data, conversation_id, config_arguments) -> Dict[str, Any]:
        """Handle ASR_NMT processing"""
        try:
            status_data['steps'].append({
                'step': 'asr_nmt',
                'status': 'in_progress',
                'timestamp': datetime.utcnow().isoformat()
            })
            
            await WebhookEvents.update_status(user_id, status_data)
            
            config_params = None
            
            # Get or fetch service configuration
            config_params = await self.conversation_manager.get_service_config(conversation_id, config_arguments)
            if not config_params:
                config_params = await self.fetch_service_config(session, conversation_id, config_arguments)
            
            if config_params:
                try:
                    #Get the callbackUrl and Authvalue Pair
                    callbackUrl = config_params.get('pipelineInferenceAPIEndPoint',{}).get('callbackUrl')
                    auth_key = config_params.get('pipelineInferenceAPIEndPoint',{}).get('inferenceApiKey',{}).get('name')
                    auth_value = config_params.get('pipelineInferenceAPIEndPoint',{}).get('inferenceApiKey',{}).get('value')
                    
                    # Find the ASR task in pipelineResponseConfig
                    asr_task = next((task for task in config_params.get('pipelineResponseConfig', [])
                                     if task['taskType'] == 'asr'), None)                    
                    
                    # Find the NMT task in pipelineResponseConfig
                    nmt_task = next((task for task in config_params.get('pipelineResponseConfig', [])
                                     if task['taskType'] == 'translation'), None)
                    
                    if asr_task and asr_task.get('config'):
                        source_language = asr_task['config'][0].get('language', {}).get('sourceLanguage')
                        asr_service = asr_task['config'][0].get('serviceId', None)
                    
                    if nmt_task and nmt_task.get('config'):
                        nmt_service = nmt_task['config'][0].get('serviceId', None)       
                
                except Exception as e:
                    logger.error(f"Error extracting source language: {str(e)}")
                    
            stt_payload = {
                            "pipelineTasks": [
                                {
                                    "taskType": "asr",
                                    "config": {
                                        "language": {
                                            "sourceLanguage": source_language
                                            },
                                            "serviceId": asr_service,
                                            "audioFormat": "flac",
                                            "samplingRate": 16000
                                        }
                                },
                                {
                                    "taskType": "translation",
                                    "config": {
                                        "language": {
                                            "sourceLanguage": source_language,
                                            "targetLanguage": "en"
                                            },
                                            "serviceId": nmt_service
                                        }
                                }
                            ],
                            "inputData": {
                                "audio": [
                                    {
                                        "audioContent": "{{generated_base64_content}}"
                                    }
                                ]
                            }
                        } 
              
            translated_response = await WebhookManager.send_webhook(
                'asr_nmt',
                stt_payload,
                session,
                callbackUrl,
                auth_key,
                auth_value,
                is_config=False)
            
            status_data['steps'][-1].update({
                'status': 'asr_nmt_completed',
                'completed_at': datetime.utcnow().isoformat()
            })
            
            await WebhookEvents.update_status(user_id, status_data)
            
            return translated_response
            
        except Exception as e:
            status_data['steps'][-1].update({
                'status': 'error',
                'error': str(e),
                'error_timestamp': datetime.utcnow().isoformat()
            })
            await WebhookEvents.update_status(user_id, status_data)
            raise

    async def process_nmt_tts(self, response_text, user_id, session, status_data, conversation_id, config_arguments) -> Dict[str, Any]:
        """Handle NMT_TTS processing"""
        try:
            status_data['steps'].append({
                'step': 'nmt_tts',
                'status': 'in_progress',
                'timestamp': datetime.utcnow().isoformat()
            })
            
            await WebhookEvents.update_status(user_id, status_data)
            
            config_params = None
            
            # Get or fetch service configuration
            config_params = await self.conversation_manager.get_service_config(conversation_id, config_arguments)
            if not config_params:
                config_params = await self.fetch_service_config(session, conversation_id, config_arguments)
            
            
            if config_params:
                try:
                    #Get the callbackUrl and Authvalue Pair
                    callbackUrl = config_params.get('pipelineInferenceAPIEndPoint',{}).get('callbackUrl')
                    auth_key = config_params.get('pipelineInferenceAPIEndPoint',{}).get('inferenceApiKey',{}).get('name')
                    auth_value = config_params.get('pipelineInferenceAPIEndPoint',{}).get('inferenceApiKey',{}).get('value')
                    
                    # Find the NMT task in pipelineResponseConfig
                    nmt_task = next((task for task in config_params.get('pipelineResponseConfig', [])
                                     if task['taskType'] == 'translation'), None)
                    
                    # Find the TTS task in pipelineResponseConfig
                    tts_task = next((task for task in config_params.get('pipelineResponseConfig', [])
                                     if task['taskType'] == 'tts'), None)
                    
                    if nmt_task and nmt_task.get('config'):
                        nmt_service = nmt_task['config'][0].get('serviceId', None)       
                    
                    if tts_task and tts_task.get('config'):
                        tts_service = tts_task['config'][0].get('serviceId', None)
                    
                except Exception as e:
                    logger.error(f"Error extracting source language: {str(e)}")
                    
            tts_payload = {
                            "pipelineTasks": [
                                {
                                    "taskType": "translation",
                                    "config": {
                                        "language": {
                                            "sourceLanguage": "en",
                                            "targetLanguage": user_language
                                            },
                                            "serviceId": nmt_service
                                    }
                                },
                                {
                                    "taskType": "tts",
                                    "config": {
                                        "language": {
                                            "sourceLanguage": user_language
                                            },
                                        "serviceId": tts_service,
                                        "gender": "female"
                                    }
                                }       
                            ],
                            "inputData": {
                                "input": [
                                    {
                                        "source": response_text
                                    }
                                ]
                            }
                        } 
              
            nmt_tts_response = await WebhookManager.send_webhook(
                'asr_nmt',
                tts_payload,
                session,
                callbackUrl,
                auth_key,
                auth_value,
                is_config=False)
            
            status_data['steps'][-1].update({
                'status': 'asr_nmt_completed',
                'completed_at': datetime.utcnow().isoformat()
            })
            
            await WebhookEvents.update_status(user_id, status_data)
            
            return nmt_tts_response
            
        except Exception as e:
            status_data['steps'][-1].update({
                'status': 'error',
                'error': str(e),
                'error_timestamp': datetime.utcnow().isoformat()
            })
            await WebhookEvents.update_status(user_id, status_data)
            raise
    
    @staticmethod
    @celery.task
    async def process_audio(self, audio_path, user_id, conversation_id=None):
        status_data = {
            'user_id': user_id,
            'status': 'processing_started',
            'timestamp': datetime.utcnow().isoformat(),
            'steps': []
        }
        
        config_arguments = dict.fromkeys(['process_type', 'source_language', 'target_language'])

        try:
            # Create or validate conversation
            if not conversation_id:
                conversation_id = await self.conversation_manager.create_conversation(user_id)
                status_data['conversation_id'] = conversation_id
            else:
                await self.conversation_manager.validate_conversation(conversation_id)

            async with aiohttp.ClientSession() as session:
                
                config_arguments.update({
                        'process_type': 'asr_nmt',
                        'source_language': user_language,
                        'target_language': 'en'
                        })
                
                # Process ASR_NMT with config parameters
                asr_nmt_response = await self.process_asr_nmt(
                    audio_path, 
                    user_id, 
                    session, 
                    status_data, 
                    conversation_id,
                    config_arguments
                )

                '''status_data.update({
                    'status': 'completed',
                    'completed_at': datetime.utcnow().isoformat()
                })
                await self.update_status(user_id, status_data)
                '''
                
                # Find the Translation task in asr_nmt_response
                translation_task = next((task for task in asr_nmt_response.get('pipelineResponse', [])
                                     if task['taskType'] == 'translation'), None)
                    
                if translation_task and translation_task.get('output'):
                    prompt = translation_task['output'][0].get('target', None)
                
                # Initialize LLM processor
                llm_processor = LangchainProcessor()
                
                #get llm_response
                llm_response = await llm_processor.process_text(
                    prompt,
                    user_id,
                    conversation_id,
                    status_data
                )
                
                if llm_response.get('status') == 'error':
                    raise Exception(f"LLM processing failed: {llm_response.get('error')}")

                if not llm_response.get('response'):
                    raise Exception("No response received from LLM processor")
                else:
                    response_text = llm_response.get('response')
                
                config_arguments.update({
                        'process_type': 'nmt_tts',
                        'source_language': 'en',
                        'target_language': user_language
                        })
                
                # Process NMT_TTS with config parameters
                nmt_tts_response = await self.process_nmt_tts(
                    response_text,
                    user_id, 
                    session, 
                    status_data, 
                    conversation_id,
                    config_arguments
                )

                return {
                    'status': 'success',
                    'conversation_id': conversation_id,
                    'processing_status': status_data,
                    'results': {
                    'original_text': prompt,
                    'llm_response': llm_response.get('response', ''),
                    'metadata': llm_response.get('metadata', {})
                    }
                }

        except Exception as e:
            error_message = str(e)
            status_data.update({
                'status': 'error',
                'error': error_message,
                'error_timestamp': datetime.utcnow().isoformat()
            })
            await self.update_status(user_id, status_data)
            raise
    
    @staticmethod()
    async def update_status(user_id, status_data):
        """Update status in both Redis and database"""
        try:
            redis_service = get_redis_service()
            # Update Redis
            redis_service.set_with_expiry(
                f"status:{user_id}", 
                json.dumps(status_data),
                3600  # 1 hour expiry
            )
        
            # Update database
            update_processing_status(user_id, status_data)
        except Exception as e:
            logger.error(f"Status update error: {str(e)}")
            raise
            
class EncryptionManager():
    def load_key():
        return open("secret.key", "rb").read()  # Read the key from the saved file
    
    def encrypt_api_key(api_key, key):
        f = Fernet(key)
        encrypted_api_key = f.encrypt(api_key.encode())  # Encrypt the API key (must be in bytes)
        return encrypted_api_key

    def decrypt_api_key(encrypted_api_key, key):
        f = Fernet(key)
        decrypted_api_key = f.decrypt(encrypted_api_key).decode()  # Decrypt and convert back to string
        return decrypted_api_key
    