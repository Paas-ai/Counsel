# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:37:17 2025

@author: Kumanan
"""

import pytest
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
from typing import Dict, Generator
from flask import Flask
from unittest.mock import patch

# Load environment variables
load_dotenv()

# Database fixtures
@pytest.fixture(scope="session")
def db_connection() -> Generator:
    """Create test database connection"""
    conn = psycopg2.connect(
        dbname=os.getenv('TEST_DB_NAME'),
        user=os.getenv('TEST_DB_USER'),
        password=os.getenv('TEST_DB_PASSWORD'),
        host=os.getenv('TEST_DB_HOST', 'localhost'),
        port=int(os.getenv('TEST_DB_PORT', 5432))
    )
    yield conn
    conn.close()

@pytest.fixture(scope="function")
def db_cursor(db_connection) -> Generator:
    """Create test database cursor"""
    cur = db_connection.cursor(cursor_factory=RealDictCursor)
    yield cur
    cur.close()

# Redis fixture
@pytest.fixture(scope="session")
def redis_client() -> Generator:
    """Create Redis test client"""
    client = redis.Redis(
        host=os.getenv('TEST_REDIS_HOST', 'localhost'),
        port=int(os.getenv('TEST_REDIS_PORT', 6379)),
        db=int(os.getenv('TEST_REDIS_DB', 0)),
        decode_responses=True
    )
    yield client
    client.close()

# OIDC Configuration
@pytest.fixture(scope="session")
def oidc_config() -> Dict:
    """OIDC test configuration"""
    return {
        'client_id': os.getenv('TEST_OIDC_CLIENT_ID'),
        'client_secret': os.getenv('TEST_OIDC_CLIENT_SECRET'),
        'provider_url': os.getenv('TEST_OIDC_PROVIDER_URL'),
        'redirect_uri': os.getenv('TEST_OIDC_REDIRECT_URI')
    }

# Test Environment Setup/Teardown
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> None:
    """Setup test environment before all tests"""
    # Setup operations before tests
    try:
        # Any initialization needed
        pass
        
    except Exception as e:
        pytest.fail(f"Failed to setup test environment: {str(e)}")

    yield  # Run tests

    # Cleanup operations after all tests
    try:
        # Any cleanup needed
        pass
    except Exception as e:
        pytest.fail(f"Failed to cleanup test environment: {str(e)}")

# Test User Data
@pytest.fixture(scope="session")
def test_user() -> Dict:
    """Test user data"""
    return {
        'mobile_number': os.getenv('TEST_MOBILE_NUMBER', '+1234567890'),
        'username': 'test_user'
    }

# API Client Configuration
@pytest.fixture(scope="session")
def api_config() -> Dict:
    """API test configuration"""
    return {
        'base_url': os.getenv('TEST_API_URL', 'http://localhost:5000'),
        'timeout': int(os.getenv('TEST_API_TIMEOUT', 30)),
        'headers': {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    }

# Error handling helper
@pytest.fixture(scope="session")
def assert_error_response():
    """Helper to assert error response structure"""
    def _assert_error(response, expected_status: int, expected_code: str = None):
        assert response.status_code == expected_status
        data = response.json()
        assert 'error' in data
        if expected_code:
            assert data.get('code') == expected_code
    return _assert_error

@pytest.fixture(scope="session")
def app():
    """Create test Flask application"""
    app = Flask(__name__)
    app.config.update({
        'TESTING': True,
        'SERVER_NAME': 'test.local',
        'OIDC_CLIENT_ID': os.getenv('TEST_OIDC_CLIENT_ID'),
        'OIDC_CLIENT_SECRET': os.getenv('TEST_OIDC_CLIENT_SECRET'),
        'OIDC_PROVIDER_URL': os.getenv('TEST_OIDC_PROVIDER_URL')
    })
    
    # Register blueprints and initialize extensions
    from app.auth import auth_bp, init_oidc
    app.register_blueprint(auth_bp)
    init_oidc(app)
    
    return app

@pytest.fixture(scope="session")
def client(app):
    """Create test client"""
    return app.test_client()

@pytest.fixture
def mock_cognito():
    """Mock Cognito client for testing"""
    with patch('app.auth.oauth.cognito') as mock:
        mock.create_authorization_url.return_value = {
            'url': 'https://cognito-domain/login',
            'state': 'test-state'
        }
        yield mock