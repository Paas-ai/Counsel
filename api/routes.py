# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:32:14 2024

@author: shanmugananth
"""

from flask import Blueprint, request, jsonify, send_from_directory
import os
import time
import logging
from celery import Celery
from celery.result import AsyncResult
import asyncio
import aiohttp
from datetime import datetime
from ..core.file_handler import FileHandler
from ..core.webhook_manager import WebhookEvents, ConversationManager
from ..core.database import get_db, update_processing_status
from ..core.redis_connection import get_redis_service
from ..core.auth import handle_registration
from ..core.rate_limiter import rate_limit
from ..config import Config
import json

logger = logging.getLogger(__name__)
api_bp = Blueprint('api', __name__)

# Initialize Celery
celery = Celery('audio_processing', broker='redis://localhost:6379/0')

# Initialize Redis
redis_service = get_redis_service()

# Initialize handlers
file_handler = FileHandler(Config.UPLOAD_FOLDER, Config.ALLOWED_EXTENSIONS)

#Global instance of CoversationManager
conversation_manager = ConversationManager()

@api_bp.route('/register', methods=['POST'])
@rate_limit(requests_per_minute=5, limit_type='ip')  # Limit per IP since users aren't registered yet
def register_user():
    """
    Registers a new user in the system
    
    Route: /register
    Method: POST
    Expected Input JSON:
        {
            "mobile number": "+91-xxxxxxxxxx"
        }
    Returns:
        - Success: JSON with user_id and success message
        - Error: JSON with error message
    """
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Delegate the actual registration logic
        task = handle_registration.delay(data)
        
        return jsonify({
            'task_id': task.id,
            'message': 'Registration processing started',
            'status': 'pending'
        })
    
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': 'Registration failed',
            'details': str(e)
        }), 500

@api_bp.route('/registration/status/<user_id>', methods=['GET'])
def get_registration_status(task_id):
    """
    Check the status of a registration task
    """
    try:
        task = handle_registration.AsyncResult(task_id)
        
        if task.ready():
            result = task.get()
            return jsonify(result)
        else:
            return jsonify({
                'status': 'pending',
                'message': 'Registration is still processing'
            })
            
    except Exception as e:
        logger.error(f"Error checking registration status: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': 'Failed to check registration status'
        }), 500

@api_bp.route('/status/<int:user_id>', methods=['GET'])
async def get_processing_status(user_id):
    try:
        # Try Redis first
        status = redis_service.get_value(f"status:{user_id}")
        if status:
            return jsonify({'status': json.loads(status)})
            
        # Fallback to database
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT processing_status FROM users WHERE id = ?', (user_id,))
        result = c.fetchone()
        
        if result and result['processing_status']:
            return jsonify({
                'status': json.loads(result['processing_status'])
            })
        return jsonify({'error': 'Status not found'}), 404
        
    except Exception as e:
        logger.error(f"Error fetching status: {str(e)}")
        return jsonify({'error': 'Failed to fetch status'}), 500

@api_bp.route('/upload_audio', methods=['POST'])
@rate_limit(requests_per_minute=15, limit_type='user')  # Limit per user
async def upload_audio():
    """
    Handles audio file uploads, processes them, and returns results
    
    Route: /upload_audio
    Method: POST
    Expected Input:
        - audio: File in request.files (audio file)
        - user_id: String in request.form (user identifier)
    Returns:
        - JSON response with status and results
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        user_id = request.form.get('user_id')
        conversation_id = request.form.get('conversation_id')  # will be None for new conversations
        
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400

        # Save file with conversation context
        file_result = file_handler.secure_save_file(
            file, 
            user_id, 
            conversation_id,
            'query'
        )
        
        #Process audio through WebhookEvents
        result = await WebhookEvents.process_audio(file_result['full_path'], request.form['user_id'],conversation_id)
        
        # After getting the response from WebhookEvents, save the response audio
        if result.get('audio_response'):
            response_audio = result['audio_response']  # This should be your audio data
            response_file_result = file_handler.secure_save_file(
                response_audio,
                user_id,
                conversation_id,
                'response'
            )
            
            response_data = {
                'status': 'success',
                'conversation_id': result.get('conversation_id') or conversation_id,
                'audio_response': {
                    'url': f'/audio/{response_file_result["relative_path"]}'
                }
            }
        else:
            response_data = {
                'status': 'success',
                'conversation_id': result.get('conversation_id') or conversation_id
            }
        
        # Cleanup old files
        file_handler.cleanup_old_files()
        
        # Return success response
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/audio/<path:filepath>')
def serve_audio(filepath):
    try:
        # Split the filepath into directory and filename
        directory = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        
        # Construct the full directory path
        full_directory = os.path.join(Config.UPLOAD_FOLDER, directory)
        
        return send_from_directory(full_directory, filename)
    
    except Exception as e:
        logger.error(f"Error serving audio file: {str(e)}")
        return jsonify({'error': 'Audio file not found'}), 404

@api_bp.route('/conversation/end', methods=['POST'])
@rate_limit(requests_per_minute=6, limit_type='both')  # Limit both per user and per IP
async def end_conversation():
    """End an active conversation"""
    try:
        data = request.json
        conversation_id = data.get('conversation_id')
        
        if not conversation_id:
            return jsonify({'error': 'Conversation ID is required'}), 400
            
        await conversation_manager.end_conversation(conversation_id)
        
        return jsonify({'status': 'success', 'message': 'Conversation ended'})
        
    except Exception as e:
        logger.error(f"Error ending conversation: {str(e)}")
        return jsonify({'error': str(e)}), 500

# user-friendly error messages
@api_bp.errorhandler(429)
def ratelimit_handler(e):
    """
    Handles rate limit exceeded errors with dynamic retry times
    
    Args:
        e: The error object passed by Flask
        
    Returns:
        JSON response with error details and retry information
    """
    # Get the rate limit info if available
    limit_info = getattr(e, 'description', {}).get('limit_info', {})
    
    # Calculate retry time
    current_time = int(time.time())
    window_end = current_time + 60  # Default to 60 seconds if no specific info
    
    if limit_info:
        # Get the most restrictive wait time from all limits
        retry_seconds = 60
        for identifier, info in limit_info.items():
            if 'window_size' in info:
                retry_seconds = min(retry_seconds, info['window_size'])
        window_end = current_time + retry_seconds
    
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Please wait a moment before trying again',
        'retry_after': {
            'seconds': retry_seconds,
            'timestamp': window_end,
            'formatted': f'{retry_seconds} seconds'
        },
        'limit_details': limit_info
    }), 429, {
        'Retry-After': str(retry_seconds)  # Standard HTTP header
    }