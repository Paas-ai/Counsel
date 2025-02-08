# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 19:01:15 2025

@author: Kumanan
"""

import pytest
from typing import Dict, List

@pytest.fixture
def valid_mobile_numbers() -> List[str]:
    """Valid mobile number test cases"""
    return [
        "9876543210",
        "8876543210",
        "7876543210",
        "6876543210"
    ]

@pytest.fixture
def invalid_mobile_numbers() -> List[str]:
    """Invalid mobile number test cases"""
    return [
        "987654321",     # Too short
        "98765432100",   # Too long
        "5876543210",    # Invalid start digit
        "1234567890",    # Invalid start digit
        "abcdefghij",    # Non-numeric
        "98765-4321",    # Contains hyphen
        "9876 54321",    # Contains space
        ""              # Empty string
    ]

@pytest.fixture
def mock_redis_operations():
    """Mock Redis operations for unit tests"""
    mock_data = {}
    
    def mock_set(key: str, value: str, ex: int = None):
        mock_data[key] = value
        return True
    
    def mock_get(key: str) -> str:
        return mock_data.get(key)
    
    def mock_delete(key: str) -> int:
        if key in mock_data:
            del mock_data[key]
            return 1
        return 0
    
    return {
        'set': mock_set,
        'get': mock_get,
        'delete': mock_delete,
        'data': mock_data
    }

@pytest.fixture
def sample_oidc_state_data() -> Dict:
    """Sample OIDC state data"""
    return {
        'state': 'test-state-123',
        'nonce': 'test-nonce-456',
        'mobile_number': '+91-9876543210',
        'user_id': 'test-user-id'
    }

@pytest.fixture
def sample_token_data() -> Dict:
    """Sample token data for validation tests"""
    return {
        'access_token': 'test-access-token',
        'id_token': 'test-id-token',
        'refresh_token': 'test-refresh-token',
        'expires_in': 3600,
        'token_type': 'Bearer'
    }