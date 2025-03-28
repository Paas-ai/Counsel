# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:12:32 2024

@author: Kumanan
"""

import os
import uuid
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
from celery import Celery

# Initialize Celery
celery = Celery('file_handler', broker='redis://localhost:6379/0')
logger = logging.getLogger(__name__)

class FileHandler:
    def __init__(self, upload_folder, allowed_extensions):
        self.upload_folder = upload_folder
        self.allowed_extensions = allowed_extensions

    def allowed_file(self, filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in self.allowed_extensions

    def secure_save_file(self, file_input, user_id, conversation_id, message_type='query'):
        """
        Securely saves the file with conversation context and validation
        
        Args:
            file_input: Either a FileStorage object from Flask or bytes data
            user_id: ID of the user
            conversation_id: Current conversation ID
            message_type: 'query' for user messages, 'response' for system responses
        """
        try:
            # Validate file_input is provided
            if not file_input:
                raise ValueError("No file provided")
                
            # Ensure upload folder exists
            os.makedirs(self.upload_folder, exist_ok=True)
            
            # Create user-specific directory
            user_dir = os.path.join(self.upload_folder, f"user_{user_id}")
            os.makedirs(user_dir, exist_ok=True)
            
            # Create conversation-specific directory if needed
            if conversation_id:
                conv_dir = os.path.join(user_dir, f"conv_{conversation_id}")
                os.makedirs(conv_dir, exist_ok=True)
                file_dir = conv_dir
            else:
                file_dir = user_dir

            # Add timestamp and UUID for uniqueness
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            
            if hasattr(file_input, 'filename'):
                # It's a Flask FileStorage object
                original_filename = file_input.filename
                if not original_filename:
                    raise ValueError("No filename provided")

                safe_filename = secure_filename(original_filename)
                if not safe_filename:
                    raise ValueError("Invalid filename")

                if not self.allowed_file(safe_filename):
                    raise ValueError(f"File type not allowed. Allowed types: {', '.join(self.allowed_extensions)}")
                    
                final_filename = f"{timestamp}_{unique_id}_{safe_filename}"
                # Save file to path
                safe_path = os.path.join(file_dir, final_filename)
                file_input.save(safe_path)
            else:
                # Check if bytes data is valid
                if len(file_input) == 0:
                    raise ValueError("Empty file data provided")
                    
                # Determine extension based on message_type
                extension = 'mp3' if message_type == 'response' else 'wav'
                final_filename = f"{timestamp}_{unique_id}.{extension}"
                safe_path = os.path.join(file_dir, final_filename)
                
                # Write bytes to file
                with open(safe_path, 'wb') as f:
                    f.write(file_input)

            # Calculate relative path from upload folder root
            relative_path = os.path.relpath(safe_path, self.upload_folder)

            return {
                'success': True,
                'original_name': getattr(file_input, 'filename', final_filename),
                'saved_as': final_filename,
                'full_path': safe_path,
                'relative_path': relative_path
            }

        except Exception as e:
            logger.error(f"File save error: {str(e)}")
            raise
    
    @celery.task(name='file_handler.cleanup_old_files')
    def cleanup_old_files(self, max_age_hours=24):
        """
        Cleanup files older than specified hours
        """
        try:
            current_time = datetime.now()
            files_cleaned = 0
            
            # Walk through all directories
            for root, dirs, files in os.walk(self.upload_folder):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    file_age = current_time - datetime.fromtimestamp(
                        os.path.getctime(file_path)
                    )
                    
                    if file_age.total_seconds() > (max_age_hours * 3600):
                        os.remove(file_path)
                        files_cleaned += 1
                        logger.info(f"Removed old file: {file_path}")
            
            # Cleanup empty directories
            for root, dirs, files in os.walk(self.upload_folder, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                        logger.info(f"Removed empty directory: {dir_path}")

            logger.info(f"Cleanup completed. Removed {files_cleaned} files.")
            
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")