# -*- coding: utf-8 -*-
"""
Conversation Manager

This module provides a robust conversation management system that handles
conversation state, persistence, and lifecycle operations.

Features:
- Creation, retrieval, and termination of conversations
- Support for conversation metadata and configuration
- Redis-based storage for quick access
- Database persistence for long-term storage
- Automatic cleanup of stale conversations
"""

import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Union

# Local imports
from .redis_connection import get_redis_service
from .database import get_db, update_processing_status
from ..config import Config

logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Manages conversation lifecycle, state, and persistence.
    """
    
    def __init__(self):
        """Initialize the conversation manager"""
        self.redis_service = get_redis_service()
        self.max_inactivity = Config.CONVERSATION_MAX_INACTIVITY or 3600  # 1 hour default
        self.max_conversations_per_user = Config.MAX_CONVERSATIONS_PER_USER or 10
        self.max_messages_per_conversation = Config.MAX_MESSAGES_PER_CONVERSATION or 10
        
    async def create_conversation(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new conversation for a user.
        
        Args:
            user_id: User identifier
            metadata: Optional metadata for the conversation
            
        Returns:
            Conversation ID
        """
        try:
            # Check if user has reached max conversations
            user_conversations = await self.get_user_conversations(user_id)
            active_conversations = [c for c in user_conversations if c['status'] == 'active']
            
            if len(active_conversations) >= self.max_conversations_per_user:
                # Auto-end oldest conversation
                oldest_conversation = min(
                    active_conversations, 
                    key=lambda c: datetime.fromisoformat(c['started_at'])
                )
                await self.end_conversation(oldest_conversation['id'])
            
            # Generate a new conversation ID
            conversation_id = str(uuid.uuid4())
            
            # Create conversation data
            conversation_data = {
                'id': conversation_id,
                'user_id': user_id,
                'started_at': datetime.utcnow().isoformat(),
                'last_activity': datetime.utcnow().isoformat(),
                'status': 'active',
                'message_count': 0,
                'metadata': metadata or {}
            }
            
            # Store in Redis for quick access
            await self.redis_service.set_with_expiry(
                f"conversation:{conversation_id}",
                json.dumps(conversation_data),
                self.max_inactivity  # Expire after inactivity period
            )
            
            # Store user-conversation mapping
            user_conversations_key = f"user:{user_id}:conversations"
            current_conversations = json.loads(
                await self.redis_service.get_value(user_conversations_key) or '[]'
            )
            current_conversations.append(conversation_id)
            await self.redis_service.set_with_expiry(
                user_conversations_key,
                json.dumps(current_conversations),
                86400 * 30  # 30 days expiry
            )
            
            # Also store in database for persistence
            conn = await get_db()
            try:
                async with conn.cursor() as cursor:
                    # Prepare JSON fields
                    if metadata:
                        metadata_json = json.dumps(metadata)
                    else:
                        metadata_json = '{}'
                    
                    # Insert conversation record
                    await cursor.execute('''
                        INSERT INTO conversations (
                            id, user_id, started_at, last_activity, 
                            status, message_count, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ''', (
                        conversation_id,
                        user_id,
                        datetime.utcnow(),
                        datetime.utcnow(),
                        'active',
                        0,
                        metadata_json
                    ))
                    await conn.commit()
            finally:
                conn.close()
            
            logger.info(f"Created new conversation {conversation_id} for user {user_id}")
            return conversation_id
            
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            raise
    
    async def validate_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Check if conversation exists and is active.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            Conversation data if valid
            
        Raises:
            ValueError: If conversation not found or not active
        """
        try:
            # Try to get from Redis first
            conversation_key = f"conversation:{conversation_id}"
            conversation_data = await self.redis_service.get_value(conversation_key)
            
            if not conversation_data:
                # Try to get from database
                conn = await get_db()
                try:
                    async with conn.cursor() as cursor:
                        await cursor.execute('''
                            SELECT * FROM conversations
                            WHERE id = %s
                        ''', (conversation_id,))
                        result = await cursor.fetchone()
                        
                        if not result:
                            raise ValueError(f"Conversation {conversation_id} not found")
                        
                        # Convert database result to dict
                        conversation_data = {
                            'id': result['id'],
                            'user_id': result['user_id'],
                            'started_at': result['started_at'].isoformat(),
                            'last_activity': result['last_activity'].isoformat(),
                            'status': result['status'],
                            'message_count': result['message_count'],
                            'metadata': json.loads(result['metadata'])
                        }
                        
                        # Cache in Redis
                        await self.redis_service.set_with_expiry(
                            conversation_key,
                            json.dumps(conversation_data),
                            self.max_inactivity
                        )
                finally:
                    conn.close()
            else:
                conversation_data = json.loads(conversation_data)
            
            # Check if conversation is active
            if conversation_data['status'] != 'active':
                raise ValueError(f"Conversation {conversation_id} is not active")
            
            # Update last activity time
            await self._update_activity_time(conversation_id)
            
            return conversation_data
                
        except ValueError as e:
            # Re-raise expected validation errors
            raise
        except Exception as e:
            logger.error(f"Error validating conversation: {str(e)}")
            raise ValueError(f"Error validating conversation: {str(e)}")
    
    async def _update_activity_time(self, conversation_id: str) -> None:
        """
        Update the last activity time for a conversation.
        
        Args:
            conversation_id: Conversation identifier
        """
        try:
            # Get current conversation data
            conversation_key = f"conversation:{conversation_id}"
            conversation_data = await self.redis_service.get_value(conversation_key)
            
            if conversation_data:
                # Update last activity time
                data = json.loads(conversation_data)
                data['last_activity'] = datetime.utcnow().isoformat()
                
                # Store updated data
                await self.redis_service.set_with_expiry(
                    conversation_key,
                    json.dumps(data),
                    self.max_inactivity
                )
            
            # Also update in database
            conn = await get_db()
            try:
                async with conn.cursor() as cursor:
                    await cursor.execute('''
                        UPDATE conversations
                        SET last_activity = %s
                        WHERE id = %s
                    ''', (datetime.utcnow(), conversation_id))
                    await conn.commit()
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Error updating activity time: {str(e)}")
            # Non-critical error, just log it
    
    async def end_conversation(self, conversation_id: str) -> None:
        """
        Mark conversation as completed.
        
        Args:
            conversation_id: Conversation identifier
        """
        try:
            # Get current conversation data
            conversation_key = f"conversation:{conversation_id}"
            conversation_data = await self.redis_service.get_value(conversation_key)
            
            if conversation_data:
                data = json.loads(conversation_data)
                
                # Update status
                data['status'] = 'completed'
                data['ended_at'] = datetime.utcnow().isoformat()
                
                # Store updated data with extended expiry (keep completed conversations for a while)
                await self.redis_service.set_with_expiry(
                    conversation_key,
                    json.dumps(data),
                    self.max_inactivity * 2  # Keep completed conversations longer
                )
                
                # Clean up any associated config data
                await self.redis_service.delete_key(f"service_config:{conversation_id}")
                
                # Clean up conversation history
                await self.redis_service.delete_key(f"conversation_history:{conversation_id}")
            
            # Update in database
            conn = await get_db()
            try:
                async with conn.cursor() as cursor:
                    await cursor.execute('''
                        UPDATE conversations
                        SET status = 'completed', ended_at = %s
                        WHERE id = %s
                    ''', (datetime.utcnow(), conversation_id))
                    await conn.commit()
            finally:
                conn.close()
                
            logger.info(f"Ended conversation {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error ending conversation: {str(e)}")
            raise
    
    async def get_user_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all conversations for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of conversation data dictionaries
        """
        try:
            # Try to get conversation IDs from Redis
            user_conversations_key = f"user:{user_id}:conversations"
            conversation_ids = json.loads(
                await self.redis_service.get_value(user_conversations_key) or '[]'
            )
            
            if not conversation_ids:
                # Try to get from database
                conn = await get_db()
                try:
                    async with conn.cursor() as cursor:
                        await cursor.execute('''
                            SELECT id FROM conversations
                            WHERE user_id = %s
                            ORDER BY last_activity DESC
                        ''', (user_id,))
                        rows = await cursor.fetchall()
                        conversation_ids = [row['id'] for row in rows]
                        
                        # Cache in Redis
                        if conversation_ids:
                            await self.redis_service.set_with_expiry(
                                user_conversations_key,
                                json.dumps(conversation_ids),
                                86400  # 1 day
                            )
                finally:
                    conn.close()
            
            # Get details for each conversation
            conversations = []
            for conv_id in conversation_ids:
                try:
                    # First try Redis
                    conversation_key = f"conversation:{conv_id}"
                    conversation_data = await self.redis_service.get_value(conversation_key)
                    
                    if conversation_data:
                        conversations.append(json.loads(conversation_data))
                    else:
                        # Try database
                        conn = await get_db()
                        try:
                            async with conn.cursor() as cursor:
                                await cursor.execute('''
                                    SELECT * FROM conversations
                                    WHERE id = %s
                                ''', (conv_id,))
                                result = await cursor.fetchone()
                                
                                if result:
                                    # Convert database result to dict
                                    conversation = {
                                        'id': result['id'],
                                        'user_id': result['user_id'],
                                        'started_at': result['started_at'].isoformat(),
                                        'last_activity': result['last_activity'].isoformat(),
                                        'status': result['status'],
                                        'message_count': result['message_count'],
                                        'metadata': json.loads(result['metadata'])
                                    }
                                    
                                    if result['ended_at']:
                                        conversation['ended_at'] = result['ended_at'].isoformat()
                                    
                                    conversations.append(conversation)
                                    
                                    # Cache in Redis
                                    await self.redis_service.set_with_expiry(
                                        conversation_key,
                                        json.dumps(conversation),
                                        self.max_inactivity
                                    )
                        finally:
                            conn.close()
                except Exception as e:
                    logger.error(f"Error getting conversation {conv_id}: {str(e)}")
                    # Skip this conversation
            
            return conversations
            
        except Exception as e:
            logger.error(f"Error getting user conversations: {str(e)}")
            return []
    
    async def get_conversation_messages(self, 
                                     conversation_id: str, 
                                     limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get messages for a conversation.
        
        Args:
            conversation_id: Conversation identifier
            limit: Maximum number of messages to return
            
        Returns:
            List of message dictionaries
        """
        try:
            # Try to get from Redis
            conversation_history_key = f"conversation_history:{conversation_id}"
            history = await self.redis_service.get_value(conversation_history_key)
            
            if history:
                messages = json.loads(history)
                
                # Return limited number of messages, most recent first
                return messages[-limit:] if len(messages) > limit else messages
            
            # Try to get from database
            conn = await get_db()
            try:
                async with conn.cursor() as cursor:
                    await cursor.execute('''
                        SELECT * FROM conversation_messages
                        WHERE conversation_id = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    ''', (conversation_id, limit))
                    rows = await cursor.fetchall()
                    
                    messages = []
                    for row in rows:
                        messages.append({
                            'id': row['id'],
                            'role': row['role'],
                            'content': row['content'],
                            'timestamp': row['timestamp'].isoformat(),
                            'metadata': json.loads(row['metadata'])
                        })
                    
                    # Cache in Redis
                    if messages:
                        # Reverse to chronological order
                        messages.reverse()
                        await self.redis_service.set_with_expiry(
                            conversation_history_key,
                            json.dumps(messages),
                            self.max_inactivity
                        )
                    
                    return messages
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Error getting conversation messages: {str(e)}")
            return []
    
    async def add_message(self, 
                       conversation_id: str, 
                       role: str, 
                       content: str,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Optional message metadata
            
        Returns:
            Message ID
        """
        try:
            # Validate conversation
            conversation_data = await self.validate_conversation(conversation_id)
            
            # Check if conversation has reached message limit
            if conversation_data['message_count'] >= self.max_messages_per_conversation:
                raise ValueError(f"Conversation {conversation_id} has reached the maximum message limit")
            
            # Generate message ID
            message_id = str(uuid.uuid4())
            
            # Create message data
            message_data = {
                'id': message_id,
                'role': role,
                'content': content,
                'timestamp': datetime.utcnow().isoformat(),
                'metadata': metadata or {}
            }
            
            # Update conversation history in Redis
            conversation_history_key = f"conversation_history:{conversation_id}"
            history = json.loads(
                await self.redis_service.get_value(conversation_history_key) or '[]'
            )
            
            # Add new message
            history.append(message_data)
            
            # Store updated history
            await self.redis_service.set_with_expiry(
                conversation_history_key,
                json.dumps(history),
                self.max_inactivity
            )
            
            # Update conversation message count
            conversation_key = f"conversation:{conversation_id}"
            conversation_data = json.loads(
                await self.redis_service.get_value(conversation_key) or '{}'
            )
            
            if conversation_data:
                conversation_data['message_count'] = conversation_data.get('message_count', 0) + 1
                conversation_data['last_activity'] = datetime.utcnow().isoformat()
                
                await self.redis_service.set_with_expiry(
                    conversation_key,
                    json.dumps(conversation_data),
                    self.max_inactivity
                )
            
            # Store in database
            conn = await get_db()
            try:
                async with conn.cursor() as cursor:
                    # Insert message
                    if metadata:
                        metadata_json = json.dumps(metadata)
                    else:
                        metadata_json = '{}'
                        
                    await cursor.execute('''
                        INSERT INTO conversation_messages (
                            id, conversation_id, role, content, 
                            timestamp, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                    ''', (
                        message_id,
                        conversation_id,
                        role,
                        content,
                        datetime.utcnow(),
                        metadata_json
                    ))
                    
                    # Update conversation message count
                    await cursor.execute('''
                        UPDATE conversations
                        SET message_count = message_count + 1,
                            last_activity = %s
                        WHERE id = %s
                    ''', (datetime.utcnow(), conversation_id))
                    
                    await conn.commit()
            finally:
                conn.close()
            
            return message_id
            
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            raise
    
    async def get_service_config(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached service configuration for a conversation.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            Service configuration dictionary or None if not found
        """
        try:
            config_key = f"service_config:{conversation_id}"
            config_data = await self.redis_service.get_value(config_key)
            
            return json.loads(config_data) if config_data else None
            
        except Exception as e:
            logger.error(f"Error getting service config: {str(e)}")
            return None
    
    async def store_service_config(self, 
                               conversation_id: str, 
                               config_data: Dict[str, Any]) -> None:
        """
        Store service configuration for a conversation.
        
        Args:
            conversation_id: Conversation identifier
            config_data: Service configuration data
        """
        try:
            config_key = f"service_config:{conversation_id}"
            await self.redis_service.set_with_expiry(
                config_key,
                json.dumps(config_data),
                3600  # Cache for 1 hour
            )
        except Exception as e:
            logger.error(f"Error storing service config: {str(e)}")
            raise
    
    async def cleanup_stale_conversations(self) -> int:
        """
        Cleanup stale conversations that have been inactive.
        
        Returns:
            Number of conversations cleaned up
        """
        try:
            # Get inactive cutoff time
            inactive_cutoff = datetime.utcnow() - timedelta(seconds=self.max_inactivity)
            
            # Get from database
            conn = await get_db()
            try:
                # Find stale conversations
                stale_conversations = []
                async with conn.cursor() as cursor:
                    await cursor.execute('''
                        SELECT id FROM conversations
                        WHERE status = 'active' AND last_activity < %s
                    ''', (inactive_cutoff,))
                    rows = await cursor.fetchall()
                    stale_conversations = [row['id'] for row in rows]
                
                # End each stale conversation
                for conv_id in stale_conversations:
                    try:
                        await self.end_conversation(conv_id)
                    except Exception as e:
                        logger.error(f"Error ending stale conversation {conv_id}: {str(e)}")
                
                return len(stale_conversations)
                
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Error cleaning up stale conversations: {str(e)}")
            return 0