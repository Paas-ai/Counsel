# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 18:50:29 2025

@author: Kumanan
"""

import pytest
from unittest.mock import patch, MagicMock
import json
from flask import url_for

class TestRegistrationFlow:
    @pytest.fixture
    def mock_cognito(self):
        """Mock Cognito client"""
        with patch('app.auth.oauth.cognito') as mock:
            mock.create_authorization_url.return_value = {
                'url': 'https://cognito-domain/login',
                'state': 'test-state'
            }
            yield mock

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis service"""
        with patch('app.auth.get_redis_service') as mock:
            mock_redis = MagicMock()
            mock.return_value = mock_redis
            yield mock_redis

    def test_successful_registration(self, client, mock_cognito, mock_redis, mock_db):
        """Test successful registration flow"""
        # Arrange
        test_mobile = "9876543210"
        
        # Act
        response = client.post('/auth/register',
            json={'mobile_number': test_mobile},
            headers={'Content-Type': 'application/json'}
        )
        
        # Assert
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check response structure
        assert 'status' in data
        assert 'auth_url' in data
        assert 'state' in data
        assert 'mobile_number' in data
        
        # Verify formatted mobile number
        assert data['mobile_number'] == "+91-9876543210"
        
        # Verify Cognito interaction
        mock_cognito.create_authorization_url.assert_called_once()
        
        # Verify Redis storage
        mock_redis.set_with_expiry.assert_called_once()
        redis_call_args = mock_redis.set_with_expiry.call_args[0]
        assert 'oidc_state' in redis_call_args[0]
        
        # Verify database interaction
        mock_db.cursor().execute.assert_called()  # Verify DB was called
        
    def test_registration_with_existing_user(self, client, mock_db):
        """Test registration attempt with already registered number"""
        # Arrange
        test_mobile = "9876543210"
        mock_db.cursor().fetchone.return_value = {
            'id': '123',
            'is_active': True
        }
        
        # Act
        response = client.post('/auth/register',
            json={'mobile_number': test_mobile}
        )
        
        # Assert
        assert response.status_code == 409
        data = json.loads(response.data)
        assert 'error' in data
        assert 'already registered' in data['error'].lower()

    def test_registration_with_invalid_mobile(self, client):
        """Test registration with invalid mobile number"""
        invalid_numbers = [
            "123456789",    # Too short
            "12345678901",  # Too long
            "5876543210",   # Invalid start digit
            "98765-43210",  # Contains hyphen
            "abcdefghij",   # Non-numeric
        ]
        
        for number in invalid_numbers:
            response = client.post('/auth/register',
                json={'mobile_number': number}
            )
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
            assert 'valid_format' in data

    @pytest.mark.asyncio
    async def test_registration_redis_failure(self, client, mock_redis, mock_cognito):
        """Test handling of Redis failures during registration"""
        # Arrange
        mock_redis.set_with_expiry.side_effect = Exception("Redis connection failed")
        
        # Act
        response = client.post('/auth/register',
            json={'mobile_number': "9876543210"}
        )
        
        # Assert
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        assert 'failed' in data['error'].lower()

    @pytest.mark.asyncio
    async def test_registration_cognito_failure(self, client, mock_cognito):
        """Test handling of Cognito failures during registration"""
        # Arrange
        mock_cognito.create_authorization_url.side_effect = Exception("Cognito error")
        
        # Act
        response = client.post('/auth/register',
            json={'mobile_number': "9876543210"}
        )
        
        # Assert
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data

    def test_registration_state_generation(self, client, mock_cognito, mock_redis):
        """Test OIDC state generation and storage"""
        # Act
        response = client.post('/auth/register',
            json={'mobile_number': "9876543210"}
        )
        
        # Assert
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify state in response
        assert 'state' in data
        assert len(data['state']) > 32  # State should be sufficiently long
        
        # Verify state storage in Redis
        mock_redis.set_with_expiry.assert_called_once()
        redis_key = mock_redis.set_with_expiry.call_args[0][0]
        assert data['state'] in redis_key
        
    def test_empty_request_body(self, client):
        """Test registration with empty request body"""
        response = client.post('/auth/register',
            json={},
            headers={'Content-Type': 'application/json'}
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Mobile number is required' in data['error']
    
    def test_missing_mobile_number(self, client):
        """Test registration with missing mobile_number field"""
        response = client.post('/auth/register',
            json={'some_field': 'some_value'},
            headers={'Content-Type': 'application/json'}
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Mobile number is required' in data['error']

    def test_invalid_content_type(self, client):
        """Test registration with invalid content type"""
        response = client.post('/auth/register',
            data='9876543210',  # Not JSON
            headers={'Content-Type': 'text/plain'}
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    #OIDC state tests
    def test_state_data_structure(self, client, mock_cognito, mock_redis):
        """Test the structure of stored OIDC state data"""
        # Arrange
        test_mobile = "9876543210"
        
        # Act
        response = client.post('/auth/register',
            json={'mobile_number': test_mobile}
        )
        
        # Assert
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify Redis storage call
        mock_redis.set_with_expiry.assert_called_once()
        stored_data = json.loads(mock_redis.set_with_expiry.call_args[0][1])
        
        # Check required fields in stored state
        assert 'mobile_number' in stored_data
        assert 'nonce' in stored_data
        assert 'user_id' in stored_data
        assert stored_data['mobile_number'] == f"+91-{test_mobile}"

    def test_state_expiry_setting(self, client, mock_cognito, mock_redis):
        """Test that OIDC state is stored with correct expiry"""
        # Act
        response = client.post('/auth/register',
            json={'mobile_number': "9876543210"}
        )
        
        # Assert
        assert response.status_code == 200
        
        # Verify Redis expiry setting
        mock_redis.set_with_expiry.assert_called_once()
        expiry_time = mock_redis.set_with_expiry.call_args[0][2]
        assert expiry_time == 600  # 10 minutes expiry
    
    #Database operation tests
    def test_db_user_creation(self, client, mock_cognito, mock_db):
        """Test user creation in database"""
        # Arrange
        test_mobile = "9876543210"
        mock_db.cursor().fetchone.return_value = None  # No existing user
        
        # Act
        response = client.post('/auth/register',
            json={'mobile_number': test_mobile}
        )
        
        # Assert
        assert response.status_code == 200
        
        # Verify correct SQL execution
        mock_db.cursor().execute.assert_called()
        insert_call = [call for call in mock_db.cursor().execute.call_args_list 
                      if 'INSERT INTO users' in call[0][0]][0]
        assert f"+91-{test_mobile}" in insert_call[0][1]

    def test_db_connection_failure(self, client, mock_db):
        """Test handling of database connection failure"""
        # Arrange
        mock_db.cursor.side_effect = Exception("Database connection failed")
        
        # Act
        response = client.post('/auth/register',
            json={'mobile_number': "9876543210"}
        )
        
        # Assert
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        assert 'failed' in data['error'].lower()
    
    #Concurrent registration test
    @pytest.mark.asyncio
    async def test_concurrent_registrations(self, client, mock_cognito, mock_db):
        """Test handling of concurrent registration requests"""
        # Arrange
        test_mobile = "9876543210"
        mock_db.cursor().fetchone.return_value = None  # No existing user
        
        # Act - Make multiple concurrent requests
        import asyncio
        responses = await asyncio.gather(
            *[
                asyncio.create_task(
                    client.post('/auth/register',
                        json={'mobile_number': test_mobile}
                    )
                )
                for _ in range(3)
            ]
        )
        
        # Assert - Only one should succeed, others should fail
        success_count = sum(1 for r in responses if r.status_code == 200)
        assert success_count == 1
        
        error_responses = [r for r in responses if r.status_code != 200]
        for error_response in error_responses:
            assert error_response.status_code == 409  # Conflict