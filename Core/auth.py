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
from typing import Dict, Optional
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

@celery.task(name='auth.handle_registration')
def handle_registration(data):
    """Initial registration with mobile number and OIDC flow initiation"""
    try:
        mobile_number = data.get('mobile_number')
        if not mobile_number:
            return jsonify({'error': 'Mobile number is required'}), 400
            
        # Improved mobile number validation with specific regex
        if not re.match(r'^\+?[1-9]\d{1,14}$', mobile_number):
            return jsonify({
                'error': 'Invalid mobile number format',
                'details': 'Number must start with + and contain 7-15 digits'
            }), 400
            
        conn = get_db()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check if user exists and get their status
                cur.execute(
                    'SELECT id, is_active FROM users WHERE mobile_number = %s',
                    (mobile_number,)
                )
                existing_user = cur.fetchone()
                
                if existing_user:
                    if existing_user['is_active']:
                        return jsonify({'error': 'User already registered'}), 409
                    user_id = existing_user['id']
                else:
                    # Create new user
                    cur.execute('''
                        INSERT INTO users (mobile_number, username, created_at)
                        VALUES (%s, %s, NOW())
                        RETURNING id
                    ''', (mobile_number, mobile_number))
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
                    mobile_number,
                    state,
                    nonce,
                    request.remote_addr,
                    request.user_agent.string
                ))
                
                conn.commit()
                
                # Store state in Redis for faster validation
                redis_service = get_redis_service()
                redis_service.set_with_expiry(
                    f'oidc_state:{state}',
                    json.dumps({
                        'mobile_number': mobile_number,
                        'user_id': str(user_id),
                        'nonce': nonce
                    }),
                    600  # 10 minutes expiry
                )
                
                # Get OIDC authorization URL
                redirect_url = url_for('auth.callback', _external=True)
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
                    'user_id': str(user_id)
                }
                
        finally:
            conn.close()
            
    except psycopg2.Error as e:
        logger.error(f"Database error during registration: {str(e)}")
        return jsonify({'error': 'Registration failed due to database error'}), 500
    except redis.RedisError as e:
        logger.error(f"Redis error during registration: {str(e)}")
        return jsonify({'error': 'Registration failed due to session error'}), 500
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': 'Registration failed'}), 500

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

@auth_bp.route('/login')
def login():
    """Initiate Cognito OIDC login flow"""
    try:
        # Generate state and nonce for PKCE
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(32)
        
        # Store state and nonce in session
        session['oauth_state'] = state
        session['oauth_nonce'] = nonce
        
        # Get redirect URI
        redirect_uri = url_for('auth.callback', _external=True)
        
        # Store return URL if provided
        if 'return_to' in request.args:
            session['return_to'] = request.args['return_to']
        
        return oauth.cognito.authorize_redirect(
            redirect_uri=redirect_uri,
            state=state,
            nonce=nonce
        )
        
    except Exception as e:
        logger.error(f"Login initiation error: {str(e)}")
        return jsonify({'error': 'Failed to initiate login'}), 500

@auth_bp.route('/callback')
def callback():
    """Handle Cognito OIDC callback"""
    try:
        state = request.args.get('state')
        if not state:
            return jsonify({'error': 'Invalid state'}), 400
        
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
                return jsonify({'error': 'Invalid or expired session'}), 400
                
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
                return jsonify({'error': 'User not found'}), 404
            
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
            
            return jsonify({
                'status': 'success',
                'access_token': token['access_token'],
                'id_token': token['id_token'],
                'user_info': {
                    'sub': userinfo['sub'],
                    'mobile_number': session['mobile_number'],
                    'user_id': user['id']
                }
            })
            
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Callback error: {str(e)}")
        return jsonify({'error': 'Authentication failed'}), 401

@auth_bp.route('/refresh', methods=['POST'])
def refresh_token():
    """Refresh access token using refresh token"""
    try:
        refresh_token = request.json.get('refresh_token')
        if not refresh_token:
            return jsonify({'error': 'Refresh token required'}), 400

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
        
        return jsonify({
            'access_token': token['access_token'],
            'expires_in': token['expires_in']
        })

    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        return jsonify({'error': 'Token refresh failed'}), 401

@auth_bp.route('/logout')
@require_auth
def logout():
    """Handle user logout"""
    try:
        user_id = request.user['sub']
        
        # Clear tokens from Redis
        redis_service = get_redis_service()
        redis_service.delete_key(f"user_tokens:{user_id}")
        
        # Clear session
        session.clear()
        
        # Build logout URL
        logout_url = oauth.oidc.load_server_metadata().get('end_session_endpoint')
        redirect_uri = url_for('auth.login', _external=True)
        
        return redirect(f"{logout_url}?post_logout_redirect_uri={redirect_uri}")

    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return jsonify({'error': 'Logout failed'}), 500

@auth_bp.route('/userinfo')
@require_auth
def get_userinfo():
    """Get authenticated user information"""
    try:
        user_id = request.user['sub']
        conn = get_db()
        try:
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE external_id = ?', (user_id,))
            user = c.fetchone()
            
            if not user:
                return jsonify({'error': 'User not found'}), 404
                
            return jsonify({
                'id': user['id'],
                'email': user['email'],
                'name': user['name'],
                'last_login': user['last_login']
            })
        finally:
            conn.close()

    except Exception as e:
        logger.error(f"Error fetching user info: {str(e)}")
        return jsonify({'error': 'Failed to fetch user info'}), 500