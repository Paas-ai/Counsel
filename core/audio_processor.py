# -*- coding: utf-8 -*-
"""
Enhanced Audio Processing Manager

This module handles the complete audio processing pipeline including:
- Audio file validation and preprocessing
- Conversion to base64 for API transmission
- Interaction with external speech-to-text/translation APIs
- Communication with LLM for response generation
- Text-to-speech conversion for final response
- Comprehensive error handling and retry mechanisms
"""

import os
import base64
import uuid
import json
import logging
import time
import asyncio
import aiohttp
import magic
import re
import struct
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from datetime import datetime, timedelta
import tempfile

# Local imports
from ..config import Config
from .database import update_processing_status
from .redis_connection import get_redis_service
from .file_handler import FileHandler
from .webhook_manager import WebhookManager
from .conversation_manager import ConversationManager

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Comprehensive audio processing pipeline manager.
    
    This class orchestrates the entire audio processing flow from 
    receiving the audio file to returning the audio response.
    """
    
    # Audio file signatures (magic numbers) mapping
    AUDIO_SIGNATURES = {
        # WAV file signatures
        'wav': [
            b'RIFF....WAVE',  # Standard WAV
            b'RIFX....WAVE',  # Big-endian WAV
        ],
        # MP3 file signatures
        'mp3': [
            b'\xFF\xFB',      # MPEG-1 Layer 3
            b'\xFF\xF3',      # MPEG-2 Layer 3
            b'\xFF\xF2',      # MPEG-2.5 Layer 3
            b'ID3',           # ID3 tag (often at start of MP3)
        ],
        # FLAC file signatures
        'flac': [
            b'fLaC',          # Standard FLAC signature
        ],
        # M4A/AAC file signatures
        'm4a': [
            b'ftypM4A ',      # M4A file signature
            b'ftypmp42',      # Another possible M4A signature
        ],
    }

    # MIME types mapping for additional validation
    AUDIO_MIME_TYPES = {
        'wav': ['audio/wav', 'audio/x-wav', 'audio/wave', 'audio/vnd.wave'],
        'mp3': ['audio/mpeg', 'audio/mp3', 'audio/mpeg3'],
        'flac': ['audio/flac', 'audio/x-flac'],
        'm4a': ['audio/mp4', 'audio/x-m4a', 'audio/m4a'],
    }
    
    def __init__(self):
        """Initialize the audio processor with necessary services"""
        self.webhook_manager = WebhookManager()
        self.conversation_manager = ConversationManager()
        self.redis_service = get_redis_service()
        
        # Security and validation settings
        self.allowed_audio_formats = Config.ALLOWED_EXTENSIONS
        self.max_file_size = Config.MAX_CONTENT_LENGTH
        
        # Initialize magic library for MIME type detection
        self.mime_magic = magic.Magic(mime=True)
        
        # Target format for API processing
        self.target_audio_format = "flac"  # Optimal format for STT APIs
        
        # API configuration
        self.max_retries = Config.WEBHOOK_MAX_RETRIES
        self.retry_delay = Config.WEBHOOK_BASE_DELAY
        
        # Language settings
        self.default_language = "en"
        self.supported_languages = ["en", "hi", "ta", "te", "kn", "ml", "bn", "gu", "mr", "pa", "ur"]
        
        # Cache for temporary files that need cleanup
        self._temp_files = []
    
    def __del__(self):
        """Clean up any temporary files on object destruction"""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.error(f"Error cleaning up temp file {temp_file}: {str(e)}")
    
    async def validate_audio_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Comprehensively validate an audio file with multiple security checks
        
        Args:
            file_path: Path to the audio file to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # 1. Basic existence check and sanitize path
            sanitized_path = self._sanitize_path(file_path)
            if not sanitized_path:
                return False, "Invalid file path"
                
            if not os.path.exists(sanitized_path):
                return False, "Audio file not found"
                
            if not os.path.isfile(sanitized_path):
                return False, "Path is not a file"
            
            # 2. File size check
            file_size = os.path.getsize(sanitized_path)
            if file_size == 0:
                return False, "Audio file is empty"
                
            if file_size > self.max_file_size:
                return False, f"File too large ({file_size/1024/1024:.1f}MB). Maximum size: {self.max_file_size/1024/1024:.1f}MB"
            
            # 3. File extension check
            file_ext = self._get_file_extension(sanitized_path)
            if not file_ext or file_ext not in self.allowed_audio_formats:
                return False, f"Invalid file format. Allowed formats: {', '.join(self.allowed_audio_formats)}"
            
            # 4. MIME type validation
            mime_type = self._get_mime_type(sanitized_path)
            if not self._validate_mime_type(file_ext, mime_type):
                return False, f"File MIME type ({mime_type}) doesn't match expected audio format"
            
            # 5. File signature/magic number validation
            if not self._validate_file_signature(sanitized_path, file_ext):
                return False, "File signature verification failed. File may be corrupted or not an audio file."
            
            # 6. Audio file structure validation (basic check for corrupted files)
            if not self._validate_audio_structure(sanitized_path, file_ext):
                return False, "Audio file structure is invalid or corrupted"
            
            # All checks passed
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating audio file: {str(e)}")
            return False, f"Error validating audio file: {str(e)}"

    def _sanitize_path(self, file_path: str) -> Optional[str]:
        """
        Sanitize file path to prevent path traversal attacks
        
        Args:
            file_path: Original file path
            
        Returns:
            Sanitized absolute path or None if path is suspicious
        """
        try:
            # Convert to absolute path and normalize
            abs_path = os.path.abspath(file_path)
            
            # Check for suspicious patterns that might indicate path traversal
            if '..' in abs_path or '/tmp/' in abs_path or re.search(r'\/\.', abs_path):
                logger.warning(f"Suspicious path detected: {file_path}")
                return None
                
            return abs_path
        except Exception as e:
            logger.error(f"Path sanitization error: {str(e)}")
            return None
    
    def _get_file_extension(self, file_path: str) -> Optional[str]:
        """
        Get lowercase file extension without the dot
        
        Args:
            file_path: Path to file
            
        Returns:
            File extension or None if no extension
        """
        _, ext = os.path.splitext(file_path)
        if ext:
            return ext[1:].lower()  # Remove the dot and convert to lowercase
        return None
    
    def _get_mime_type(self, file_path: str) -> str:
        """
        Get MIME type of file using magic library
        
        Args:
            file_path: Path to file
            
        Returns:
            MIME type string
        """
        return self.mime_magic.from_file(file_path)
    
    def _validate_mime_type(self, file_ext: str, detected_mime: str) -> bool:
        """
        Validate that the detected MIME type matches the expected type for the extension
        
        Args:
            file_ext: File extension
            detected_mime: Detected MIME type
            
        Returns:
            True if valid, False otherwise
        """
        # Check if extension has registered MIME types
        if file_ext not in self.AUDIO_MIME_TYPES:
            return False
            
        # Check if detected MIME type is in the list of valid types for this extension
        return detected_mime in self.AUDIO_MIME_TYPES[file_ext]
    
    def _validate_file_signature(self, file_path: str, file_ext: str) -> bool:
        """
        Validate file signature (magic numbers) against known audio formats
        
        Args:
            file_path: Path to file
            file_ext: File extension to check against
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Check if we have signatures for this format
            if file_ext not in self.AUDIO_SIGNATURES:
                return False
                
            # Read first 12 bytes (enough for most signatures)
            with open(file_path, 'rb') as f:
                file_start = f.read(12)
            
            # Check against known signatures
            for signature in self.AUDIO_SIGNATURES[file_ext]:
                # Convert signature pattern to regex pattern
                pattern = signature.replace(b'....', b'....')  # Handle 4-byte wildcards
                pattern = re.escape(pattern).replace(b'\\.\\.\\.\\.\\.', b'....')  # Unescape wildcards
                
                # Replace wildcards with regex pattern that matches any 4 bytes
                pattern = pattern.replace(b'....', b'.{4}')
                
                if re.match(pattern, file_start):
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error validating file signature: {str(e)}")
            return False
    
    def _validate_audio_structure(self, file_path: str, file_ext: str) -> bool:
        """
        Perform basic validation of audio file structure.
        Different checks for different formats.
        
        Args:
            file_path: Path to file
            file_ext: File extension
            
        Returns:
            True if structure seems valid, False otherwise
        """
        try:
            if file_ext == 'wav':
                return self._validate_wav_structure(file_path)
            elif file_ext == 'mp3':
                return self._validate_mp3_structure(file_path)
            elif file_ext == 'flac':
                return self._validate_flac_structure(file_path)
            elif file_ext == 'm4a':
                # Basic check for m4a (more complex validation would require specialized libraries)
                return True
                
            # For unsupported formats, return True to not block the process
            return True
            
        except Exception as e:
            logger.error(f"Error validating audio structure: {str(e)}")
            return False
    
    def _validate_wav_structure(self, file_path: str) -> bool:
        """
        Validate basic WAV file structure
        
        Args:
            file_path: Path to WAV file
            
        Returns:
            True if structure is valid, False otherwise
        """
        try:
            with open(file_path, 'rb') as f:
                # Check RIFF header
                chunk_id = f.read(4)
                if chunk_id != b'RIFF' and chunk_id != b'RIFX':
                    return False
                
                # Read chunk size (4 bytes)
                chunk_size = f.read(4)
                
                # Check format
                format_id = f.read(4)
                if format_id != b'WAVE':
                    return False
                
                # Find fmt subchunk (should be present in valid WAV)
                while True:
                    subchunk_id = f.read(4)
                    if not subchunk_id or len(subchunk_id) < 4:
                        return False  # Unexpected EOF
                    
                    if subchunk_id == b'fmt ':
                        break
                        
                    # Skip this subchunk
                    subchunk_size = struct.unpack('<I', f.read(4))[0]
                    f.seek(subchunk_size, 1)  # Seek relative to current position
                
                # Read fmt subchunk size
                subchunk_size = struct.unpack('<I', f.read(4))[0]
                
                # Read audio format code (1 is PCM)
                audio_format = struct.unpack('<H', f.read(2))[0]
                
                # Basic validation of audio format
                if audio_format not in [1, 3]:  # 1=PCM, 3=IEEE float
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"Error validating WAV structure: {str(e)}")
            return False
    
    def _validate_mp3_structure(self, file_path: str) -> bool:
        """
        Perform basic validation of MP3 file structure
        
        Args:
            file_path: Path to MP3 file
            
        Returns:
            True if structure seems valid, False otherwise
        """
        try:
            with open(file_path, 'rb') as f:
                # Check for ID3 tag
                header = f.read(3)
                if header == b'ID3':
                    # Skip ID3 tag
                    f.seek(6, 1)  # Skip version and flags
                    size_bytes = f.read(4)
                    size = ((size_bytes[0] & 0x7F) << 21) | ((size_bytes[1] & 0x7F) << 14) | \
                           ((size_bytes[2] & 0x7F) << 7) | (size_bytes[3] & 0x7F)
                    f.seek(size, 1)  # Skip ID3 tag content
                else:
                    # Rewind if no ID3
                    f.seek(0)
                
                # Look for MP3 frame header
                # MP3 frames start with 0xFF followed by 0xE0 or higher
                found_frame = False
                for _ in range(1024):  # Check first 1KB for MP3 frame
                    byte = f.read(1)
                    if not byte:
                        break  # EOF
                    
                    if byte == b'\xFF':
                        next_byte = f.read(1)
                        if not next_byte:
                            break  # EOF
                        
                        if (ord(next_byte) & 0xE0) == 0xE0:
                            found_frame = True
                            break
                
                return found_frame
                
        except Exception as e:
            logger.error(f"Error validating MP3 structure: {str(e)}")
            return False
    
    def _validate_flac_structure(self, file_path: str) -> bool:
        """
        Perform basic validation of FLAC file structure
        
        Args:
            file_path: Path to FLAC file
            
        Returns:
            True if structure seems valid, False otherwise
        """
        try:
            with open(file_path, 'rb') as f:
                # Check FLAC signature "fLaC"
                if f.read(4) != b'fLaC':
                    return False
                
                # Read first metadata block header
                header = f.read(4)
                if len(header) < 4:
                    return False
                
                # First bit is last-metadata-block flag, next 7 bits are block type
                block_type = header[0] & 0x7F
                
                # Validate block type (0=STREAMINFO, which must be first)
                if block_type != 0:
                    return False
                
                # Valid FLAC structure detected
                return True
                
        except Exception as e:
            logger.error(f"Error validating FLAC structure: {str(e)}")
            return False
    
    async def preprocess_audio(self, file_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Preprocess the audio file for optimal API processing.
        
        This converts audio to the ideal format for STT API:
        - Convert to FLAC format if not already
        - Resample to 16kHz (optimal for speech recognition)
        - Set to mono channel (simplifies processing)
        - Normalize audio levels
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (success, processed_file_path, error_message)
        """
        try:
            # Get file extension
            file_ext = os.path.splitext(file_path)[1][1:].lower()
            
            # If already in target format, verify it has correct parameters
            if file_ext == self.target_audio_format:
                # In a real implementation, you would verify:
                # - Sample rate (should be 16kHz)
                # - Channels (should be mono)
                # - Bit depth (should be 16-bit)
                # But for simplicity, we'll just return the file as-is
                return True, file_path, None
            
            # Create temp file path for converted audio
            temp_dir = tempfile.gettempdir()
            output_filename = f"{uuid.uuid4()}.{self.target_audio_format}"
            output_path = os.path.join(temp_dir, output_filename)
            
            # Add to temp files list for cleanup
            self._temp_files.append(output_path)
            
            # Log the conversion
            logger.info(f"Converting audio from {file_ext} to {self.target_audio_format}")
            
            # Use ffmpeg to convert the audio with appropriate parameters
            # This is executed in a separate process to avoid blocking
            import subprocess
            
            try:
                # Command to convert audio to FLAC with optimal STT parameters
                cmd = [
                    'ffmpeg',
                    '-i', file_path,                 # Input file
                    '-ar', '16000',                  # Sample rate: 16kHz
                    '-ac', '1',                      # Mono channel
                    '-sample_fmt', 's16',            # 16-bit samples
                    '-af', 'dynaudnorm',             # Normalize audio levels
                    '-c:a', 'flac',                  # FLAC codec
                    '-q:a', '8',                     # Quality setting
                    output_path                      # Output file
                 ]
                
                # Execute the command
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Wait for the process to complete
                stdout, stderr = await process.communicate()
                
                # Check the return code
                if process.returncode != 0:
                    error_msg = stderr.decode().strip()
                    logger.error(f"FFmpeg error: {error_msg}")
                    return False, None, f"Audio conversion failed: {error_msg}"
                
                # Verify the output file was created
                if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                    return False, None, "Audio conversion resulted in empty file"
                
                return True, output_path, None
                
            except FileNotFoundError:
                # FFmpeg not installed or not in PATH
                logger.error("FFmpeg not found. Falling back to direct file usage.")
                
                # For fallback, just copy the file
                import shutil
                shutil.copy(file_path, output_path)
                return True, output_path, None
                
            except Exception as e:
                logger.error(f"Error during audio conversion: {str(e)}")
                return False, None, f"Error during audio conversion: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
            return False, None, f"Error preprocessing audio: {str(e)}"
       
    async def encode_audio_to_base64(self, file_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Convert audio file to base64 for API transmission.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (success, base64_string, error_message)
        """
        try:
            with open(file_path, "rb") as audio_file:
                encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
                return True, encoded_string, None
        except Exception as e:
            logger.error(f"Error encoding audio to base64: {str(e)}")
            return False, None, f"Error encoding audio to base64: {str(e)}"
    
    async def process_audio_to_text(
        self, 
        audio_base64: str, 
        user_language: str,
        session: aiohttp.ClientSession,
        user_id: str,
        conversation_id: str,
        status_data: Dict[str, Any]
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Process audio through speech-to-text API and translation if needed.
        
        Args:
            audio_base64: Base64 encoded audio data
            user_language: User's preferred language code
            session: HTTP session for API calls
            user_id: User identifier
            conversation_id: Conversation identifier
            status_data: Status tracking data
            
        Returns:
            Tuple of (success, response_data, error_message)
        """
        try:
            # Update status
            status_data['steps'].append({
                'step': 'speech_to_text',
                'status': 'in_progress',
                'timestamp': datetime.utcnow().isoformat()
            })
            await update_processing_status(user_id, status_data)
            
            # Validate language
            if user_language not in self.supported_languages:
                logger.warning(f"Unsupported language: {user_language}, using default: {self.default_language}")
                user_language = self.default_language
            
            # Get API configuration for current conversation
            config_arguments = {
                'process_type': 'asr_nmt',
                'source_language': user_language,
                'target_language': 'en'
            }
            
            # Fetch or get cached service configuration
            config_params = await self.conversation_manager.get_service_config(conversation_id)
            if not config_params:
                config_params = await self.webhook_manager.fetch_service_config(
                    session, 
                    conversation_id, 
                    config_arguments
                )
            
            # Extract necessary configuration details
            if not config_params:
                raise ValueError("Failed to get API configuration")
            
            try:
                # Extract callback URL and auth details
                callback_url = config_params.get('pipelineInferenceAPIEndPoint', {}).get('callbackUrl')
                auth_key = config_params.get('pipelineInferenceAPIEndPoint', {}).get('inferenceApiKey', {}).get('name')
                auth_value = config_params.get('pipelineInferenceAPIEndPoint', {}).get('inferenceApiKey', {}).get('value')
                
                # Find ASR task configuration
                asr_task = next((task for task in config_params.get('pipelineResponseConfig', [])
                               if task['taskType'] == 'asr'), None)                    
                
                # Find NMT task configuration
                nmt_task = next((task for task in config_params.get('pipelineResponseConfig', [])
                               if task['taskType'] == 'translation'), None)
                
                if asr_task and asr_task.get('config'):
                    source_language = asr_task['config'][0].get('language', {}).get('sourceLanguage')
                    asr_service = asr_task['config'][0].get('serviceId', None)
                
                if nmt_task and nmt_task.get('config'):
                    nmt_service = nmt_task['config'][0].get('serviceId', None)
                    
            except Exception as e:
                logger.error(f"Error extracting API configuration: {str(e)}")
                raise ValueError(f"Error in API configuration: {str(e)}")
            
            # Prepare payload for ASR+NMT
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
                            "audioContent": audio_base64
                        }
                    ]
                }
            }
            
            # Send request to ASR+NMT API
            response = await self.webhook_manager.send_webhook(
                'asr_nmt',
                stt_payload,
                session,
                callback_url,
                auth_key,
                auth_value
            )
            
            # Update status
            status_data['steps'][-1].update({
                'status': 'completed',
                'completed_at': datetime.utcnow().isoformat()
            })
            await update_processing_status(user_id, status_data)
            
            return True, response, None
            
        except Exception as e:
            # Update status with error
            if status_data and 'steps' in status_data and status_data['steps']:
                status_data['steps'][-1].update({
                    'status': 'error',
                    'error': str(e),
                    'error_timestamp': datetime.utcnow().isoformat()
                })
                await update_processing_status(user_id, status_data)
            
            logger.error(f"Error in speech-to-text processing: {str(e)}")
            return False, None, f"Error in speech-to-text processing: {str(e)}"
    
    async def extract_text_from_response(self, response: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Extract the translated text from API response.
        
        Args:
            response: API response data
            
        Returns:
            Tuple of (success, extracted_text, error_message)
        """
        try:
            # Find the translation task in response
            translation_tasks = [task for task in response.get('pipelineResponse', [])
                                if task['taskType'] == 'translation']
            
            if not translation_tasks:
                # If no translation tasks, try to get ASR output directly
                asr_tasks = [task for task in response.get('pipelineResponse', [])
                           if task['taskType'] == 'asr']
                
                if not asr_tasks or not asr_tasks[0].get('output'):
                    raise ValueError("No speech-to-text or translation output found in response")
                
                # Extract text from ASR output
                text = asr_tasks[0]['output'][0].get('source', '')
            else:
                # Extract text from translation output
                translation_task = translation_tasks[0]
                if not translation_task.get('output'):
                    raise ValueError("No translation output found in response")
                
                text = translation_task['output'][0].get('target', '')
            
            if not text:
                raise ValueError("Empty text result from API")
            
            return True, text, None
            
        except Exception as e:
            logger.error(f"Error extracting text from response: {str(e)}")
            return False, None, f"Error extracting text from response: {str(e)}"
    
    async def process_text_with_llm(
        self, 
        text: str, 
        user_id: str, 
        conversation_id: str,
        status_data: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Process the extracted text with LLM to generate a response.
        
        Args:
            text: The input text to process
            user_id: User identifier
            conversation_id: Conversation identifier
            status_data: Status tracking data
            
        Returns:
            Tuple of (success, llm_response, error_message)
        """
        try:
            # Update status
            status_data['steps'].append({
                'step': 'llm_processing',
                'status': 'in_progress',
                'timestamp': datetime.utcnow().isoformat()
            })
            await update_processing_status(user_id, status_data)
            
            # Import here to avoid circular imports
            from .llm_processor import LlmProcessor
            
            # Initialize LLM processor
            llm_processor = LlmProcessor()
            
            # Process text with LLM
            result = await llm_processor.process_text(
                text,
                user_id,
                conversation_id,
                status_data
            )
            
            if result.get('status') == 'error':
                raise Exception(f"LLM processing failed: {result.get('error')}")
            
            response_text = result.get('response')
            if not response_text:
                raise Exception("No response received from LLM processor")
            
            # Update status
            status_data['steps'][-1].update({
                'status': 'completed',
                'completed_at': datetime.utcnow().isoformat()
            })
            await update_processing_status(user_id, status_data)
            
            return True, response_text, None
            
        except Exception as e:
            # Update status with error
            if status_data and 'steps' in status_data and status_data['steps']:
                status_data['steps'][-1].update({
                    'status': 'error',
                    'error': str(e),
                    'error_timestamp': datetime.utcnow().isoformat()
                })
                await update_processing_status(user_id, status_data)
            
            logger.error(f"Error in LLM processing: {str(e)}")
            return False, None, f"Error in LLM processing: {str(e)}"
    
    async def convert_text_to_speech(
        self, 
        text: str, 
        user_language: str,
        session: aiohttp.ClientSession,
        user_id: str,
        conversation_id: str,
        status_data: Dict[str, Any]
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Convert the LLM response text to speech in user's language.
        
        Args:
            text: Text to convert to speech
            user_language: User's preferred language code
            session: HTTP session for API calls
            user_id: User identifier
            conversation_id: Conversation identifier
            status_data: Status tracking data
            
        Returns:
            Tuple of (success, tts_response, error_message)
        """
        try:
            # Update status
            status_data['steps'].append({
                'step': 'text_to_speech',
                'status': 'in_progress',
                'timestamp': datetime.utcnow().isoformat()
            })
            await update_processing_status(user_id, status_data)
            
            # Validate language
            if user_language not in self.supported_languages:
                logger.warning(f"Unsupported language: {user_language}, using default: {self.default_language}")
                user_language = self.default_language
            
            # Get API configuration for current conversation
            config_arguments = {
                'process_type': 'nmt_tts',
                'source_language': 'en',
                'target_language': user_language
            }
            
            # Fetch or get cached service configuration
            config_params = await self.conversation_manager.get_service_config(conversation_id)
            if not config_params:
                config_params = await self.webhook_manager.fetch_service_config(
                    session, 
                    conversation_id, 
                    config_arguments
                )
            
            # Extract necessary configuration details
            if not config_params:
                raise ValueError("Failed to get API configuration")
            
            try:
                # Extract callback URL and auth details
                callback_url = config_params.get('pipelineInferenceAPIEndPoint', {}).get('callbackUrl')
                auth_key = config_params.get('pipelineInferenceAPIEndPoint', {}).get('inferenceApiKey', {}).get('name')
                auth_value = config_params.get('pipelineInferenceAPIEndPoint', {}).get('inferenceApiKey', {}).get('value')
                
                # Find NMT task configuration
                nmt_task = next((task for task in config_params.get('pipelineResponseConfig', [])
                               if task['taskType'] == 'translation'), None)
                
                # Find TTS task configuration
                tts_task = next((task for task in config_params.get('pipelineResponseConfig', [])
                               if task['taskType'] == 'tts'), None)
                
                if nmt_task and nmt_task.get('config'):
                    nmt_service = nmt_task['config'][0].get('serviceId', None)
                
                if tts_task and tts_task.get('config'):
                    tts_service = tts_task['config'][0].get('serviceId', None)
                    
            except Exception as e:
                logger.error(f"Error extracting API configuration: {str(e)}")
                raise ValueError(f"Error in API configuration: {str(e)}")
            
            # Prepare payload for NMT+TTS
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
                            "gender": "female"  # Or get from user preferences
                        }
                    }       
                ],
                "inputData": {
                    "input": [
                        {
                            "source": text
                        }
                    ]
                }
            }
            
            # Send request to NMT+TTS API
            response = await self.webhook_manager.send_webhook(
                'nmt_tts',
                tts_payload,
                session,
                callback_url,
                auth_key,
                auth_value
            )
            
            # Update status
            status_data['steps'][-1].update({
                'status': 'completed',
                'completed_at': datetime.utcnow().isoformat()
            })
            await update_processing_status(user_id, status_data)
            
            return True, response, None
            
        except Exception as e:
            # Update status with error
            if status_data and 'steps' in status_data and status_data['steps']:
                status_data['steps'][-1].update({
                    'status': 'error',
                    'error': str(e),
                    'error_timestamp': datetime.utcnow().isoformat()
                })
                await update_processing_status(user_id, status_data)
            
            logger.error(f"Error in text-to-speech processing: {str(e)}")
            return False, None, f"Error in text-to-speech processing: {str(e)}"
    
    async def decode_audio_from_base64(
        self, 
        response_audio_base64: str
    ) -> Tuple[bool, Optional[bytes], Optional[str]]:
        """
        Decode base64 audio string to binary data.
    
        Args:
            response_audio_base64: Base64 encoded audio string
        
        Returns:
            Tuple of (success, audio_data, error_message)
        """
        try:
            # Decode base64 to binary
            reponse_binary = base64.b64decode(response_audio_base64)
            
            return True, reponse_binary, None
            
        except Exception as e:
            logger.error(f"Error decoding audio from base64: {str(e)}")
            return False, None, f"Error decoding audio from base64: {str(e)}"
    
    async def process_audio(
        self, 
        audio_file, 
        user_id: str, 
        user_language: str,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process audio through the complete pipeline.
        
        Args:
            audio_path: Path to the input audio file
            user_id: User identifier
            user_language: User's preferred language
            conversation_id: Optional conversation identifier
            
        Returns:
            Dictionary with processing results
        """
        # Initialize status tracking
        status_data = {
            'user_id': user_id,
            'status': 'processing_started',
            'timestamp': datetime.utcnow().isoformat(),
            'steps': []
        }
        
        try:
            # Step 0: Save the uploaded file
            file_handler = FileHandler(Config.UPLOAD_FOLDER, Config.ALLOWED_EXTENSIONS)

            file_result = file_handler.secure_save_file(
                audio_file, 
                user_id, 
                conversation_id,
                'query'
            )
            audio_path = file_result['full_path']

            # Create or validate conversation
            if not conversation_id:
                try:
                    conversation_id = await self.conversation_manager.create_conversation(user_id)
                    status_data['conversation_id'] = conversation_id
                    
                    # Move the file to the new conversation directory
                    user_dir = os.path.join(Config.UPLOAD_FOLDER, f"user_{user_id}")
                    conv_dir = os.path.join(user_dir, f"conv_{conversation_id}")
                    os.makedirs(conv_dir, exist_ok=True)
                    
                    # New filename in conversation directory
                    old_filename = os.path.basename(audio_path)
                    new_path = os.path.join(conv_dir, old_filename)
                    
                    try:
                        os.rename(audio_path, new_path)
                        audio_path = new_path
                    except OSError as e:
                        logger.warning(f"Could not move file to conversation directory: {str(e)}")
                        # Continue with original path if move fails
                except Exception as e:
                    # Cleanup: remove the temporary file if something goes wrong
                    if os.path.exists(audio_path):
                        try:
                            os.remove(audio_path)
                        except:
                            logger.warning(f"Failed to clean up temporary file: {audio_path}")
                    
                    # Re-raise the exception
                    raise ValueError(f"Error creating conversation: {str(e)}")
            else:
                await self.conversation_manager.validate_conversation(conversation_id)
            
            # Step 1: Validate audio file
            valid, error = await self.validate_audio_file(audio_path)
            if not valid:
                raise ValueError(f"Invalid audio file: {error}")
            
            # Step 2: Preprocess audio
            success, processed_path, error = await self.preprocess_audio(audio_path)
            if not success:
                raise ValueError(f"Audio preprocessing failed: {error}")
            
            # Step 3: Encode audio to base64
            success, query_audio_base64, error = await self.encode_audio_to_base64(processed_path)
            if not success:
                raise ValueError(f"Audio encoding failed: {error}")
            
            webhook_manager = WebhookManager()

            #Process audio through WebhookEvents
            async with aiohttp.ClientSession() as session:
                # Step 4: Webhook call
                response = await webhook_manager.process_api_chain(
                    query_audio_base64,
                    user_id,
                    user_language,
                    conversation_id,
                    status_data,
                    session
                )

                if response.get('response_audio_base64'):
                    # Step 5: Decode base64 back to audio data
                    success, response_binary, error = await self.decode_audio_from_base64(response['response_audio_base64'])
                    if not success:
                        raise ValueError(f"Audio decoding failed: {error}")
                    
                    # Step 6: Save the decoded audio response
                    response_result = file_handler.secure_save_file(
                        response_binary,
                        user_id,
                        conversation_id,
                        'response'
                    )
                    
                    response['audio_response'] = {
                        'url': f'/audio/{response_result["relative_path"]}'
    
                    }

            # Update final status
            status_data['status'] = 'completed'
            status_data['completed_at'] = datetime.utcnow().isoformat()
            await update_processing_status(user_id, status_data)
            
            return {
            'status': 'success',
            'conversation_id': conversation_id,
            'processing_status': status_data,
            'audio_response': response.get('audio_response')
            }
            
        except Exception as e:
            # Update error status
            error_message = str(e)
            status_data['status'] = 'error'
            status_data['error'] = error_message
            status_data['error_timestamp'] = datetime.utcnow().isoformat()
            await update_processing_status(user_id, status_data)
            
            logger.error(f"Audio processing error: {error_message}")
            return {
                'status': 'error',
                'error': error_message,
                'processing_status': status_data
            }
        finally:
            # Cleanup any temporary files
            pass