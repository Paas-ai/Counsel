# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:33:08 2025

@author: Kumanan
"""

import pytest
from typing import Dict

@pytest.fixture(scope="session")
def auth_headers(api_headers) -> Dict:
    """Auth-specific headers fixture"""
    return {
        **api_headers,
        'X-Test-Auth': 'true'  # Any auth-specific headers
    }

@pytest.fixture(scope="session")
def mock_oidc_config() -> Dict:
    """OIDC test configuration"""
    return {
        'client_id': 'test_client',
        'response_type': 'code',
        'scope': 'openid profile email'
    }