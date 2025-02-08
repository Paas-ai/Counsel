# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:11:56 2024

@author: Kumanan
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API URL
    URL = os.environ.get('URL')
    
    # API User ID
    Access_ID = os.environ.get('ACCESS_ID')
    
    # API Keys
    API_KEY = os.environ.get('STT_API_KEY')
    
    # PipeLineId
    PipeLineId = os.environ.get('PipelineId')
    
    #STT_API_URL = os.environ.get('STT_API_URL')
    #TRANSLATION_API_URL = os.environ.get('TRANSLATION_API_URL')

    # App Configuration
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'audio_uploads')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_FILE_SIZE', 16 * 1024 * 1024))
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    PORT = int(os.environ.get('PORT', 5000))
    
    # Elasticsearch Configuration
    ELASTICSEARCH_CONFIG = {
        'hosts': os.environ.get('ELASTICSEARCH_HOSTS', 'http://localhost:9200').split(','),
        'username': os.environ.get('ELASTICSEARCH_USERNAME', 'elastic'),
        'password': os.environ.get('ELASTICSEARCH_PASSWORD', ''),
        'document_index': os.environ.get('ELASTICSEARCH_DOC_INDEX', 'documents'),
        'vector_index': os.environ.get('ELASTICSEARCH_VECTOR_INDEX', 'document_vectors')
    }
    
    # Llama Configuration
    LLAMA_MODEL_PATH = os.environ.get('LLAMA_MODEL_PATH', './models/llama-3.2.bin')
    LLAMA_MAX_TOKENS = int(os.environ.get('LLAMA_MAX_TOKENS', 512))
    LLAMA_TEMPERATURE = float(os.environ.get('LLAMA_TEMPERATURE', 0.7))
    LLAMA_TOP_P = float(os.environ.get('LLAMA_TOP_P', 0.95))
    LLAMA_MAX_HISTORY = int(os.environ.get('LLAMA_MAX_HISTORY', 10))
    LLAMA_TOP_K = int(os.environ.get('LLAMA_TOP_K', 40))
    LLAMA_REPEAT_PENALTY = float(os.environ.get('LLAMA_REPEAT_PENALTY', 1.1))
        
    # Database Configuration
    POSTGRES_USER = os.environ.get('POSTGRES_USER', 'your_user')
    POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD', 'your_password')
    POSTGRES_HOST = os.environ.get('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = os.environ.get('POSTGRES_PORT', '5432')
    POSTGRES_DB = os.environ.get('POSTGRES_DB', 'your_db_name')
    SQLALCHEMY_DATABASE_URI = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    
    # Redis Configuration
    REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
    REDIS_DB = int(os.environ.get('REDIS_DB', 0))
    REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')
    REDIS_MAX_CONNECTIONS = int(os.environ.get('REDIS_MAX_CONNECTIONS', 100))
    REDIS_SOCKET_TIMEOUT = float(os.environ.get('REDIS_SOCKET_TIMEOUT', 5.0))
    REDIS_CONNECT_TIMEOUT = float(os.environ.get('REDIS_CONNECT_TIMEOUT', 2.0))
    REDIS_RETRY_ON_TIMEOUT = os.environ.get('REDIS_RETRY_ON_TIMEOUT', 'True').lower() == 'true'
    REDIS_MAX_RETRIES = int(os.environ.get('REDIS_MAX_RETRIES', 3))

    # Webhook Configuration
    WEBHOOK_MAX_RETRIES = int(os.environ.get('WEBHOOK_MAX_RETRIES', 3))
    WEBHOOK_BASE_DELAY = float(os.environ.get('WEBHOOK_BASE_DELAY', 1.0))
    WEBHOOK_TIMEOUT = int(os.environ.get('WEBHOOK_TIMEOUT', 30))
    
    # OIDC Configuration
    OIDC_CLIENT_ID = os.environ.get('TEST_OIDC_CLIENT_ID')
    OIDC_CLIENT_SECRET = os.environ.get('TEST_OIDC_CLIENT_SECRET')
    OIDC_PROVIDER_URL = os.environ.get('OIDC_PROVIDER_URL')
    OIDC_REDIRECT_URI = os.environ.get('OIDC_REDIRECT_URI')
    
    # OIDC Endpoints
    #Auth URL - Initial authentication
    #Token URL - Getting tokens
    #UserInfo URL - Getting user details
    #JWKS URL - Token verification
    COGNITO_JWKS_URL = f"{OIDC_PROVIDER_URL}/.well-known/jwks.json"
    # This endpoint provides the JSON Web Key Set (JWKS) that contains the public keys 
    # used to verify the JWT tokens issued by Cognito
    
    COGNITO_AUTH_URL = f"{OIDC_PROVIDER_URL}/oauth2/authorize"
    # Authorization endpoint where users are redirected to authenticate
    # Used when starting the OIDC authentication flow
    
    COGNITO_TOKEN_URL = f"{OIDC_PROVIDER_URL}/oauth2/token"
    # Token endpoint used to exchange authorization code for access tokens and ID tokens
    # Called by your backend after successful authentication
    
    COGNITO_USERINFO_URL = f"{OIDC_PROVIDER_URL}/oauth2/userInfo"
    # UserInfo endpoint to get additional user information
    # Can be called with a valid access token
    
    # JWT Configuration
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')
    JWT_ACCESS_TOKEN_EXPIRES = int(os.environ.get('JWT_ACCESS_TOKEN_EXPIRES', 3600))  # 1 hour
    JWT_REFRESH_TOKEN_EXPIRES = int(os.environ.get('JWT_REFRESH_TOKEN_EXPIRES', 2592000))  # 30 days
    
    # Session Configuration
    SESSION_TYPE = 'redis'
    SESSION_REDIS = f"redis://{':' + REDIS_PASSWORD + '@' if REDIS_PASSWORD else ''}{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    PERMANENT_SESSION_LIFETIME = int(os.environ.get('SESSION_LIFETIME', 1800))  # 30 minute

    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}

    @staticmethod
    def validate():
        required_vars = [
            'API_KEY',
            'URL',
            'LLAMA_MODEL_PATH',
            'ELASTICSEARCH_HOSTS'
        ]
        
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

