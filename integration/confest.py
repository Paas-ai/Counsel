# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 19:00:23 2025

@author: Kumanan
"""

import pytest
from unittest.mock import patch, MagicMock
import json
from typing import Dict

@pytest.fixture
def mock_auth_flow():
    """Mock complete authentication flow"""
    with patch('app.auth.oauth.cognito') as mock_cognito:
        # Setup mock responses
        mock_cognito.create_authorization_url.return_value = {
            'url': 'https://cognito-domain/login',
            'state': 'test-state-123'
        }
        
        mock_cognito.authorize_access_token.return_value = {
            'access_token': 'test-access-token',
            'id_token': 'test-id-token',
            'refresh_token': 'test-refresh-token',
            'expires_in': 3600
        }
        
        mock_cognito.parse_id_token.return_value = {
            'sub': 'test-user-id',
            'phone_number': '+91-9876543210',
            'email': 'test@example.com'
        }
        
        yield mock_cognito

@pytest.fixture
def integration_test_data() -> Dict:
    """Test data for integration tests"""
    return {
        'valid_mobile': '9876543210',
        'formatted_mobile': '+91-9876543210',
        'invalid_mobile': '123456',
        'test_state': 'test-state-123',
        'test_code': 'test-auth-code'
    }

@pytest.fixture
def mock_webhook_calls():
    """Mock external webhook calls"""
    with patch('app.core.webhook_manager.WebhookManager.send_webhook') as mock:
        mock.return_value = {'status': 'success'}
        yield mock

@pytest.fixture
def mock_celery_task():
    """Mock Celery task execution"""
    with patch('app.core.tasks.process_registration.delay') as mock:
        mock.return_value = MagicMock(id='test-task-id')
        yield mock

@pytest.fixture
def mock_cognito_tokens():
    """Provides standardized mock tokens and user info for OIDC testing"""
    return {
        'tokens': {
            'access_token': 'eyJ0eXAi...test_access_token',
            'id_token': 'eyJ0eXAi...test_id_token',
            'refresh_token': 'eyJ0eXAi...test_refresh_token',
            'expires_in': 3600,
            'token_type': 'Bearer'
        },
        'userinfo': {
            'sub': 'test-user-id',
            'phone_number': '+91-9876543210',
            'phone_number_verified': True,
            'email': 'test@example.com',
            'email_verified': True
        },
        'state': 'test-state-123',
        'nonce': 'test-nonce-456',
        'code': 'test-authorization-code'
    }

@pytest.fixture
def mock_oidc_endpoints(requests_mock):
    """Mock OIDC endpoint responses"""
    def _setup_endpoints(config):
        # Mock authorization endpoint
        requests_mock.get(
            f"{config['provider_url']}/oauth2/authorize",
            status_code=302,
            headers={'Location': f"http://localhost:5000/auth/callback?code=test-code&state={config['state']}"}
        )
        
        # Mock token endpoint
        requests_mock.post(
            f"{config['provider_url']}/oauth2/token",
            json={
                'access_token': config['tokens']['access_token'],
                'id_token': config['tokens']['id_token'],
                'refresh_token': config['tokens']['refresh_token'],
                'expires_in': config['tokens']['expires_in'],
                'token_type': 'Bearer'
            }
        )
        
        # Mock userinfo endpoint
        requests_mock.get(
            f"{config['provider_url']}/oauth2/userInfo",
            json=config['userinfo']
        )
    
    return _setup_endpoints

@pytest.fixture
def verify_oidc_flow():
    """Helper to verify OIDC flow steps"""
    def _verify_flow(response, mock_cognito, mock_redis, mock_db, expected_mobile):
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert 'auth_url' in data
        assert 'state' in data
        
        # Verify Cognito interaction
        mock_cognito.create_authorization_url.assert_called_once()
        
        # Verify Redis state storage
        assert mock_redis.set_with_expiry.called
        redis_call_args = mock_redis.set_with_expiry.call_args[0]
        assert 'oidc_state' in redis_call_args[0]
        
        # Verify DB operations
        mock_db.cursor().execute.assert_called()
        call_args = mock_db.cursor().execute.call_args_list
        assert any(expected_mobile in str(args) for args in call_args)
        
        return data
    
    return _verify_flow