# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:32:25 2025

@author: Kumanan
"""

import pytest
import os
from dotenv import load_dotenv
from typing import Dict

# Load environment variables for tests
load_dotenv()

# Common test configuration
@pytest.fixture(scope="session")
def test_config() -> Dict:
    """Global test configuration fixture"""
    return {
        'api_url': os.getenv('TEST_API_URL', 'http://localhost:5000'),
        'oidc_provider_url': os.getenv('OIDC_PROVIDER_URL'),
        'test_mobile_number': os.getenv('TEST_MOBILE_NUMBER'),
        'test_timeout': int(os.getenv('TEST_TIMEOUT', '30'))
    }

# Common test user data
@pytest.fixture(scope="session")
def test_user_data() -> Dict:
    """Test user data fixture"""
    return {
        'mobile_number': os.getenv('TEST_MOBILE_NUMBER', '+1234567890'),
        'username': 'test_user'
    }

# Common headers for API requests
@pytest.fixture(scope="session")
def api_headers() -> Dict:
    """Common API headers fixture"""
    return {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }