# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:34:38 2025

@author: Kumanan
"""

import pytest
import requests
import json
from datetime import datetime
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class TestTokenValidation:
    """Test suite for token validation functionality"""
    
    @pytest.fixture(scope="class")
    def api_client(self):
        """Fixture to provide base URL and common headers"""
        return {
            'base_url': 'http://your-api-url',
            'headers': {
                'Content-Type': 'application/json'
            }
        }
    
    @pytest.fixture(scope="class")
    def test_user(self):
        """Fixture to provide test user data"""
        return {
            'mobile_number': '+1234567890'  # Test mobile number
        }
    
    @pytest.fixture(scope="class")
    def auth_tokens(self, api_client, test_user) -> Dict:
        """Fixture to get authentication tokens through registration and OIDC flow"""
        try:
            # 1. Register user
            register_response = requests.post(
                f"{api_client['base_url']}/auth/register",
                headers=api_client['headers'],
                json={'mobile_number': test_user['mobile_number']}
            )
            assert register_response.status_code in [200, 409]  # 409 if user exists
            
            # 2. Get auth URL and state
            auth_data = register_response.json()
            assert 'auth_url' in auth_data
            assert 'state' in auth_data
            
            # 3. Simulate OIDC authentication (this would normally be done through browser)
            # For testing, you might need to mock this or use a test OIDC provider
            oidc_response = self._simulate_oidc_auth(auth_data['auth_url'])
            
            # 4. Exchange code for tokens
            token_response = requests.post(
                f"{api_client['base_url']}/auth/callback",
                params={'state': auth_data['state'], 'code': oidc_response['code']}
            )
            assert token_response.status_code == 200
            
            tokens = token_response.json()
            assert 'access_token' in tokens
            assert 'id_token' in tokens
            
            return tokens
            
        except Exception as e:
            logger.error(f"Failed to get auth tokens: {str(e)}")
            raise

    def _simulate_oidc_auth(self, auth_url: str) -> Dict:
        """
        Helper method to simulate OIDC authentication flow
        This would need to be implemented based on your OIDC provider
        """
        # This is a placeholder - implement based on your OIDC provider
        pass

    def test_valid_token(self, api_client, auth_tokens):
        """Test validation with a valid token"""
        headers = {
            **api_client['headers'],
            'Authorization': f"Bearer {auth_tokens['access_token']}"
        }
        
        response = requests.get(
            f"{api_client['base_url']}/auth/test/validate-token",
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert 'user_info' in data
        
        # Validate user info structure
        user_info = data['user_info']
        assert 'user_id' in user_info
        assert 'sub' in user_info
        assert 'aud' in user_info
        assert 'iss' in user_info

    def test_expired_token(self, api_client):
        """Test validation with an expired token"""
        expired_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ0ZXN0X3VzZXIiLCJleHAiOjE1MTYyMzkwMjJ9.signature"
        headers = {
            **api_client['headers'],
            'Authorization': f'Bearer {expired_token}'
        }
        
        response = requests.get(
            f"{api_client['base_url']}/auth/test/validate-token",
            headers=headers
        )
        
        assert response.status_code == 401
        data = response.json()
        assert data['error'] == 'Token has expired'
        assert data['code'] == 'token_expired'

    def test_invalid_token(self, api_client):
        """Test validation with an invalid token"""
        headers = {
            **api_client['headers'],
            'Authorization': 'Bearer invalid.token.format'
        }
        
        response = requests.get(
            f"{api_client['base_url']}/auth/test/validate-token",
            headers=headers
        )
        
        assert response.status_code == 401
        data = response.json()
        assert 'error' in data
        assert data['code'] == 'invalid_token'

    def test_no_token(self, api_client):
        """Test validation without a token"""
        response = requests.get(
            f"{api_client['base_url']}/auth/test/validate-token",
            headers=api_client['headers']
        )
        
        assert response.status_code == 401
        data = response.json()
        assert data['error'] == 'No authorization token provided'

    def test_blacklisted_token(self, api_client, auth_tokens):
        """Test validation with a blacklisted token"""
        headers = {
            **api_client['headers'],
            'Authorization': f"Bearer {auth_tokens['access_token']}"
        }
        
        # First verify token is valid
        pre_check = requests.get(
            f"{api_client['base_url']}/auth/test/validate-token",
            headers=headers
        )
        assert pre_check.status_code == 200
        
        # Logout to blacklist token
        logout_response = requests.post(
            f"{api_client['base_url']}/auth/logout",
            headers=headers
        )
        assert logout_response.status_code == 200
        
        # Try to use the blacklisted token
        response = requests.get(
            f"{api_client['base_url']}/auth/test/validate-token",
            headers=headers
        )
        
        assert response.status_code == 401
        data = response.json()
        assert 'error' in data
        assert 'blacklisted' in data['error'].lower()

    def test_malformed_token(self, api_client):
        """Test validation with malformed token"""
        headers = {
            **api_client['headers'],
            'Authorization': 'Bearer not.even.close.to.valid'
        }
        
        response = requests.get(
            f"{api_client['base_url']}/auth/test/validate-token",
            headers=headers
        )
        
        assert response.status_code == 401
        data = response.json()
        assert 'error' in data
        assert data['code'] == 'invalid_token'

    def test_token_wrong_audience(self, api_client):
        """Test validation with token having wrong audience"""
        # This would need a token with wrong audience claim
        pass

    def test_token_wrong_issuer(self, api_client):
        """Test validation with token having wrong issuer"""
        # This would need a token with wrong issuer claim
        pass

    @pytest.mark.asyncio
    async def test_concurrent_token_validation(self, api_client, auth_tokens):
        """Test concurrent token validation requests"""
        import asyncio
        import aiohttp
        
        async def validate_token(session):
            headers = {
                **api_client['headers'],
                'Authorization': f"Bearer {auth_tokens['access_token']}"
            }
            async with session.get(
                f"{api_client['base_url']}/auth/test/validate-token",
                headers=headers
            ) as response:
                return await response.json()
        
        async with aiohttp.ClientSession() as session:
            tasks = [validate_token(session) for _ in range(5)]
            results = await asyncio.gather(*tasks)
            
        for result in results:
            assert result['status'] == 'success'

def main():
    """Manual test runner"""
    base_url = 'http://your-api-url'  # Configure your API URL
    
    # Create test instance
    test_instance = TestTokenValidation()
    
    # Setup fixtures manually
    api_client = test_instance.api_client()
    test_user = test_instance.test_user()
    auth_tokens = test_instance.auth_tokens(api_client, test_user)
    
    # Run tests
    print("\nRunning token validation tests...")
    
    try:
        print("\n1. Testing valid token...")
        test_instance.test_valid_token(api_client, auth_tokens)
        print("✓ Valid token test passed")
        
        print("\n2. Testing expired token...")
        test_instance.test_expired_token(api_client)
        print("✓ Expired token test passed")
        
        print("\n3. Testing invalid token...")
        test_instance.test_invalid_token(api_client)
        print("✓ Invalid token test passed")
        
        print("\n4. Testing no token...")
        test_instance.test_no_token(api_client)
        print("✓ No token test passed")
        
        print("\n5. Testing blacklisted token...")
        test_instance.test_blacklisted_token(api_client, auth_tokens)
        print("✓ Blacklisted token test passed")
        
        print("\n6. Testing malformed token...")
        test_instance.test_malformed_token(api_client)
        print("✓ Malformed token test passed")
        
        print("\nAll tests completed successfully!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {str(e)}")
    except Exception as e:
        print(f"\n❌ Error running tests: {str(e)}")

if __name__ == "__main__":
    main()