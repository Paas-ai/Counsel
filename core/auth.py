# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 18:57:53 2025

@author: Kumanan
"""

from flask import Blueprint, request, jsonify, session, redirect, url_for
from authlib.integrations.flask_client import OAuth
from functools import wraps
import jwt
import re
import secrets
from psycopg2.extras import RealDictCursor
from jwt import PyJWKClient
from jwt.exceptions import PyJWTError
from datetime import datetime
import logging
from typing import Dict, Optional, Tuple, Any
from .database import get_db
from .redis_connection import get_redis_service
from ..config import Config
from .. import celery

logger = logging.getLogger(__name__)
auth_bp = Blueprint('auth', __name__)
oauth = OAuth()

def init_oidc(app):
    """Initialize OIDC client with application context"""
    oauth.init_app(app)
    
    # Register OIDC client
    oauth.register(
        name='cognito',
        client_id=Config.OIDC_CLIENT_ID,
        client_secret=Config.OIDC_CLIENT_SECRET,
        server_metadata_url=f'{Config.OIDC_PROVIDER_URL}/.well-known/openid-configuration',
        client_kwargs={
            'scope': 'openid phone profile',
            'token_endpoint_auth_method': 'client_secret_basic'
        }
    )

def validate_and_format_mobile(mobile_number: str) -> Tuple[bool, str, str]:
    """
    Validate and format a mobile number
    
    Args:
        mobile_number: The mobile number to validate
    
    Returns:
        Tuple of (is_valid, formatted_number, error_message)
    """
    # Strip any non-digit characters
    digits_only = re.sub(r'\D', '', mobile_number)
    
    # Check if we have exactly 10 digits
    if len(digits_only) != 10:
        return False, "", "Mobile number must be exactly 10 digits"
    
    # Check if all characters are digits in the original
    if len(digits_only) != len(mobile_number):
        return False, "", "Input must be exactly 10 digits, no other characters allowed"
    
    # Validate first digit (must be 6-9 for Indian mobile numbers)
    if digits_only[0] not in ['6', '7', '8', '9']:
        return False, "", "Mobile number must start with 6, 7, 8 or 9"
    
    # Format the number with country code
    formatted_number = f"+91-{digits_only}"
    return True, formatted_number, ""

@celery.task(name='auth.handle_registration')
def handle_registration(data):
    """Initial registration with mobile number and OIDC flow initiation"""
    try:
        mobile_number = data.get('mobile_number')
        if not mobile_number:
            return jsonify({'error': 'Mobile number is required'}), 400
            
        # Validate and format mobile number
        is_valid, formatted_mobile, error = validate_and_format_mobile(mobile_number)
        if not is_valid:
            return {
                'error': error,
                'valid_format': False,
                'status': 'error'
            }, 400
            
        conn = get_db()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check if user exists and get their status
                cur.execute(
                    'SELECT id, is_active FROM users WHERE mobile_number = %s',
                    (formatted_mobile,)
                )
                existing_user = cur.fetchone()
                
                if existing_user:
                    if existing_user['is_active']:
                        return {'error': 'User already registered', 'status': 'error'}, 409
                    user_id = existing_user['id']
                else:
                    # Create new user
                    cur.execute('''
                        INSERT INTO users (mobile_number, username, created_at)
                        VALUES (%s, %s, NOW())
                        RETURNING id
                    ''', (formatted_mobile, formatted_mobile))
                    user_id = cur.fetchone()['id']
                
                # Generate OIDC state and nonce with improved security
                state = secrets.token_urlsafe(32)
                nonce = secrets.token_urlsafe(32)
                
                # Store OIDC session with additional security metadata
                cur.execute('''
                    INSERT INTO oidc_auth_sessions 
                    (mobile_number, state, nonce, created_at, expires_at, ip_address, user_agent)
                    VALUES (%s, %s, %s, NOW(), NOW() + INTERVAL '10 minutes', %s, %s)
                ''', (
                    formatted_mobile,
                    state,
                    nonce,
                    request.remote_addr if hasattr(request, 'remote_addr') else 'unknown',
                    request.user_agent.string if hasattr(request, 'user_agent') else 'unknown'
                ))
                
                conn.commit()
                
                # Store state in Redis for faster validation
                redis_service = get_redis_service()
                redis_service.set_with_expiry(
                    f'oidc_state:{state}',
                    json.dumps({
                        'mobile_number': formatted_mobile,
                        'user_id': str(user_id),
                        'nonce': nonce
                    }),
                    600  # 10 minutes expiry
                )
                
                # Get OIDC authorization URL
                redirect_url = url_for('api.callback', _external=True)
                auth_params = {
                    'redirect_uri': redirect_url,
                    'state': state,
                    'nonce': nonce,
                    'scope': 'openid profile phone',
                    'response_type': 'code'
                }
                
                auth_url = oauth.cognito.create_authorization_url(**auth_params)
                
                return {
                    'status': 'success',
                    'auth_url': auth_url['url'],
                    'state': state,
                    'mobile_number': formatted_mobile,
                    'user_id': str(user_id)
                }
                
        finally:
            conn.close()

    except psycopg2.Error as e:
        logger.error(f"Database error during registration: {str(e)}")
        return {'error': 'Registration failed due to database error'}, 500
    except redis.RedisError as e:
        logger.error(f"Redis error during registration: {str(e)}")
        return {'error': 'Registration failed due to session error'}, 500        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return {'error': 'Registration failed', 'details': str(e), 'status': 'error'}, 500

def get_token_from_header() -> Optional[str]:
    """Extract token from Authorization header"""
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return auth_header.split(' ')[1]
    return None

def verify_token(token: str) -> Dict:
    """
    Verify and decode JWT token with comprehensive validation
    """
    try:
        # Decode token without verification first to get kid
        unverified_headers = jwt.get_unverified_header(token)
        kid = unverified_headers.get('kid')
        if not kid:
            raise jwt.InvalidTokenError('No key ID in token header')

        # Get public key for this kid from OIDC provider
        jwks_client = jwt.PyJWKClient(f'{Config.OIDC_PROVIDER_URL}/.well-known/jwks.json')
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        
        # Verify token with proper key
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=['RS256'],
            audience=Config.OIDC_CLIENT_ID,
            options={
                'verify_exp': True,
                'verify_iat': True,
                'verify_nbf': True,
                'verify_iss': True,
                'verify_aud': True,
                'require': ['exp', 'iat', 'nbf', 'iss', 'aud', 'sub']
            }
        )

        # Check if token is blacklisted in Redis
        redis_service = get_redis_service()
        if redis_service.get_value(f"blacklisted_token:{payload['jti']}"):
            raise jwt.InvalidTokenError('Token has been blacklisted')

        # Verify user exists and is active in database
        conn = get_db()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute('''
                    SELECT id, is_active 
                    FROM users 
                    WHERE sub = %s
                ''', (payload['sub'],))
                
                user = cur.fetchone()
                if not user:
                    raise jwt.InvalidTokenError('User not found')
                if not user['is_active']:
                    raise jwt.InvalidTokenError('User is not active')
                
                # Add user ID to payload
                payload['user_id'] = user['id']
                
        finally:
            conn.close()

        return payload

    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        raise
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {str(e)}")
        raise
    except PyJWTError as e:
        # This will catch any other JWT-related errors not caught above
        logger.error(f"JWT validation error: {str(e)}")
        raise jwt.InvalidTokenError(f"JWT validation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        raise jwt.InvalidTokenError(str(e))

def require_auth(f):
    """Decorator to protect routes requiring authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_token_from_header()
        if not token:
            return jsonify({'error': 'No authorization token provided'}), 401

        try:
            # Verify and decode token
            payload = verify_token(token)
            
            # Store user info in request context
            request.user = payload
            
            # Log access attempt
            logger.info(f"Authenticated access by user {payload['sub']}")
            
            return f(*args, **kwargs)
            
        except jwt.ExpiredSignatureError:
            return jsonify({
                'error': 'Token has expired',
                'code': 'token_expired'
            }), 401
        except jwt.InvalidTokenError as e:
            return jsonify({
                'error': str(e),
                'code': 'invalid_token'
            }), 401
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return jsonify({
                'error': 'Authentication failed',
                'code': 'auth_error'
            }), 401

    return decorated

def process_auth_callback(state: str, code: str) -> Dict[str, Any]:
    """
    Process OIDC callback with code and state
    
    Args:
        state: The state parameter from the callback
        code: The authorization code from the callback
        
    Returns:
        Dictionary with token and user information
    """
    if not state:
        raise ValueError("Invalid state")
    
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get session info that matches the state from registration
        cur.execute('''
            SELECT mobile_number, nonce 
            FROM oidc_auth_sessions 
            WHERE state = %s AND expires_at > NOW()
        ''', (state,))
        
        session = cur.fetchone()
        if not session:
            raise ValueError("Invalid or expired session")
            
        # Get tokens from Cognito
        token = oauth.cognito.authorize_access_token()
        if not token:
            raise ValueError("Failed to get access token")
            
        userinfo = oauth.cognito.parse_id_token(token, nonce=session['nonce'])
        
        # Update existing user with Cognito identity
        cur.execute('''
            UPDATE users 
            SET sub = %s,
                last_login = NOW(),
                is_active = TRUE
            WHERE mobile_number = %s
            RETURNING id
        ''', (
            userinfo['sub'],
            session['mobile_number']
        ))
        
        user = cur.fetchone()
        if not user:
            raise ValueError("User not found")
        
        conn.commit()
        
        # Store tokens in Redis
        redis_service = get_redis_service()
        redis_service.set_with_expiry(
            f"user_tokens:{userinfo['sub']}",
            {
                'access_token': token['access_token'],
                'refresh_token': token.get('refresh_token'),
                'id_token': token['id_token']
            },
            token['expires_in']
        )
        
        return {
            'status': 'success',
            'access_token': token['access_token'],
            'id_token': token['id_token'],
            'user_info': {
                'sub': userinfo['sub'],
                'mobile_number': session['mobile_number'],
                'user_id': user['id']
            }
        }
        
    finally:
        conn.close()

def process_token_refresh(refresh_token: str) -> Dict[str, Any]:
    """
    Refresh access token using refresh token
    
    Args:
        refresh_token: The refresh token
        
    Returns:
        Dictionary with new tokens
    """
    if not refresh_token:
        raise ValueError("Refresh token required")

    # Exchange refresh token for new tokens
    token = oauth.oidc.refresh_token(refresh_token)
    
    # Update tokens in Redis
    user_id = jwt.decode(token['id_token'], verify=False)['sub']
    redis_service = get_redis_service()
    redis_service.set_with_expiry(
        f"user_tokens:{user_id}",
        {
            'access_token': token['access_token'],
            'refresh_token': token.get('refresh_token'),
            'id_token': token['id_token']
        },
        token['expires_in']
    )
    
    return {
        'access_token': token['access_token'],
        'expires_in': token['expires_in']
    }

def process_logout(user_id: str) -> Dict[str, Any]:
    """
    Process user logout
    
    Args:
        user_id: The user ID (sub)
        
    Returns:
        Dictionary with logout status
    """
    # Clear tokens from Redis
    redis_service = get_redis_service()
    redis_service.delete_key(f"user_tokens:{user_id}")
    
    # Clear session
    if 'session' in globals():
        session.clear()
    
    # Build logout URL
    logout_url = oauth.oidc.load_server_metadata().get('end_session_endpoint')
    
    return {
        'status': 'success',
        'message': 'Successfully logged out',
        'logout_url': logout_url
    }

def get_user_information(user_id: str) -> Dict[str, Any]:
    """
    Get user information
    
    Args:
        user_id: The user ID (sub)
        
    Returns:
        Dictionary with user information
    """
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE sub = ?', (user_id,))
        user = c.fetchone()
        
        if not user:
            raise ValueError("User not found")
            
        return {
            'id': user['id'],
            'email': user.get('email'),
            'mobile_number': user['mobile_number'],
            'last_login': user['last_login'],
            'is_active': user['is_active']
        }
    finally:
        conn.close()