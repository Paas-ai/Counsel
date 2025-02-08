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

    def secure_save_file(self, file, user_id, conversation_id, message_type='query'):
        """
        Securely saves the uploaded file with conversation context and validation
        
        Args:
            file: The uploaded file
            user_id: ID of the user
            conversation_id: Current conversation ID
            message_type: 'query' for user messages, 'response' for system responses
        """
        try:
            if not file:
                raise ValueError("No file provided")

            original_filename = file.filename
            if not original_filename:
                raise ValueError("No filename provided")

            safe_filename = secure_filename(original_filename)
            if not safe_filename:
                raise ValueError("Invalid filename")

            if not self.allowed_file(safe_filename):
                raise ValueError(f"File type not allowed. Allowed types: {', '.join(self.allowed_extensions)}")
            
            # Create conversation directory if it doesn't exist
            conversation_path = os.path.join(
                self.upload_folder,
                str(user_id),
                str(conversation_id)
            )
            # Ensure upload folder exists
            os.makedirs(conversation_path, exist_ok=True)

            # Add timestamp and UUID for uniqueness
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            final_filename = f"{timestamp}_{unique_id}_{safe_filename}"
            
            # Save file in conversation directory
            safe_path = os.path.join(conversation_path, final_filename)
            file.save(safe_path)

            return {
                'success': True,
                'original_name': file.filename,
                'saved_as': final_filename,
                'full_path': safe_path,
                'relative_path': os.path.join(str(user_id), str(conversation_id), final_filename)
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