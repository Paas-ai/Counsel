# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 12:08:30 2024

@author: Kumanan
"""

import redis
from redis.connection import ConnectionPool
from redis.exceptions import ConnectionError, RedisError
from typing import Optional
import logging
import os
from ..config import Config

logger = logging.getLogger(__name__)

class RedisConnectionManager:
    _pool = None
    _instance = None

    def __init__(self):
        if not RedisConnectionManager._pool:
            self._initialize_pool()

    @classmethod
    def _initialize_pool(cls):
        """Initialize Redis connection pool with production settings"""
        try:
            # Get configuration from environment or config
            redis_host = Config.REDIS_HOST or 'localhost'
            redis_port = int(Config.REDIS_PORT or 6379)
            redis_db = int(Config.REDIS_DB or 0)
            redis_password = Config.REDIS_PASSWORD
            max_connections = int(Config.REDIS_MAX_CONNECTIONS or 100)
            socket_timeout = float(Config.REDIS_SOCKET_TIMEOUT or 5.0)
            socket_connect_timeout = float(Config.REDIS_CONNECT_TIMEOUT or 2.0)
            retry_on_timeout = Config.REDIS_RETRY_ON_TIMEOUT or True

            # Create connection pool with production settings
            cls._pool = ConnectionPool(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                max_connections=max_connections,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                retry_on_timeout=retry_on_timeout,
                decode_responses=True,
                health_check_interval=30
            )
            
            logger.info(f"Redis connection pool initialized with max {max_connections} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection pool: {str(e)}")
            raise

    @classmethod
    def get_connection(cls) -> redis.Redis:
        """Get a Redis connection from the pool"""
        if not cls._pool:
            cls._initialize_pool()
            
        try:
            return redis.Redis(connection_pool=cls._pool)
        except Exception as e:
            logger.error(f"Failed to get Redis connection from pool: {str(e)}")
            raise

    @classmethod
    def close_all(cls):
        """Close all connections in the pool"""
        if cls._pool:
            try:
                cls._pool.disconnect()
                logger.info("All Redis connections closed")
            except Exception as e:
                logger.error(f"Error closing Redis connections: {str(e)}")

    @classmethod
    def get_pool_stats(cls) -> dict:
        """Get current connection pool statistics"""
        if not cls._pool:
            return {"status": "not_initialized"}
            
        return {
            "max_connections": cls._pool.max_connections,
            "current_connections": len(cls._pool._connections),
            "pid": os.getpid()
        }

class RedisService:
    """Service class for Redis operations with retries and error handling"""
    
    def __init__(self):
        self.redis = RedisConnectionManager.get_connection()
        self.max_retries = int(Config.REDIS_MAX_RETRIES or 3)
        
    def _handle_operation(self, operation, *args, **kwargs):
        """Execute Redis operation with retries and error handling"""
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except ConnectionError as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Redis connection failed after {self.max_retries} attempts: {str(e)}")
                    raise
                logger.warning(f"Redis connection attempt {attempt + 1} failed, retrying...")
                self.redis = RedisConnectionManager.get_connection()  # Get fresh connection
            except RedisError as e:
                logger.error(f"Redis operation error: {str(e)}")
                raise

    def set_with_expiry(self, key: str, value: str, expiry: int = 3600) -> bool:
        """Set key with expiry"""
        return self._handle_operation(self.redis.setex, key, expiry, value)

    def get_value(self, key: str) -> Optional[str]:
        """Get value for key"""
        return self._handle_operation(self.redis.get, key)

    def delete_key(self, key: str) -> bool:
        """Delete key"""
        return self._handle_operation(self.redis.delete, key)

# Global accessor function
def get_redis_service() -> RedisService:
    return RedisService()