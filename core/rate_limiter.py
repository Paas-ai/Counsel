# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 19:40:59 2024

@author: Kumanan
"""

from functools import wraps
from flask import request, jsonify
import time
from .redis_connection import get_redis_service
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiting implementation using Redis with support for both IP and user-based limiting"""
    
    def __init__(self, requests_per_minute=60, limit_type='ip'):
        self.redis_service = get_redis_service()
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # 1 minute in seconds
        self.limit_type = limit_type  # 'ip', 'user', or 'both'
        
    def _get_identifiers(self):
        """Get identifiers based on limit type"""
        identifiers = []
        
        # IP-based identifier
        if self.limit_type in ['ip', 'both']:
            ip_address = request.remote_addr
            identifiers.append(f"ip:{ip_address}")
            
        # User-based identifier
        if self.limit_type in ['user', 'both']:
            # Try to get user_id from different possible sources
            user_id = None
            if request.is_json:
                user_id = request.json.get('user_id')
            elif request.form:
                user_id = request.form.get('user_id')
            elif request.args:
                user_id = request.args.get('user_id')
                
            if user_id:
                identifiers.append(f"user:{user_id}")
            elif self.limit_type == 'user':
                # If we're only tracking by user but can't find a user_id, log warning
                logger.warning("User-based rate limiting requested but no user_id found")
                
        return identifiers
        
    def is_rate_limited(self):
        """Check if the request should be rate limited"""
        try:
            identifiers = self._get_identifiers()
            if not identifiers:
                logger.warning("No valid identifiers found for rate limiting")
                return False
                
            current_time = int(time.time())
            is_limited = False
            limit_info = {}

            # Check each identifier against its limit
            for identifier in identifiers:
                key = f"rate_limit:{identifier}"
                
                # Get the current request count
                request_history = self.redis_service.get_value(key)
                if not request_history:
                    request_history = []
                else:
                    request_history = eval(request_history)  # Convert string to list
                    
                # Remove requests outside current window
                current_window = current_time - self.window_size
                request_history = [ts for ts in request_history if ts > current_window]
                
                # Add current request
                request_history.append(current_time)
                
                # Update Redis with new history
                self.redis_service.set_with_expiry(
                    key,
                    str(request_history),
                    self.window_size
                )
                
                # Check if limit exceeded
                requests_made = len(request_history)
                if requests_made > self.requests_per_minute:
                    is_limited = True
                    limit_info[identifier] = {
                        'requests_made': requests_made,
                        'limit': self.requests_per_minute,
                        'window_size': self.window_size
                    }
            
            return is_limited, limit_info
            
        except Exception as e:
            logger.error(f"Rate limiting error: {str(e)}")
            return False, {}  # On error, allow request to proceed
            
def rate_limit(requests_per_minute=60, limit_type='ip'):
    """
    Decorator for rate limiting endpoints
    
    Args:
        requests_per_minute (int): Number of requests allowed per minute
        limit_type (str): Type of limiting - 'ip', 'user', or 'both'
    """
    limiter = RateLimiter(requests_per_minute, limit_type)
    
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            is_limited, limit_info = limiter.is_rate_limited()
            if is_limited:
                response = {
                    'error': 'Rate limit exceeded',
                    'message': f'Maximum {requests_per_minute} requests per minute allowed',
                    'limit_info': limit_info
                }
                return jsonify(response), 429
            return f(*args, **kwargs)
        return wrapped
    return decorator