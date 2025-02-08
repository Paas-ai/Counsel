# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 18:46:24 2025

@author: Kumanan
"""

import pytest
from app.auth import validate_and_format_mobile

class TestMobileValidation:
    @pytest.mark.parametrize("input_number,expected", [
        ("9876543210", (True, "+91-9876543210", "")),  # Valid number
        ("8876543210", (True, "+91-8876543210", "")),  # Valid number starting with 8
        ("7876543210", (True, "+91-7876543210", "")),  # Valid number starting with 7
        ("6876543210", (True, "+91-6876543210", "")),  # Valid number starting with 6
    ])
    def test_valid_mobile_numbers(self, input_number, expected):
        """Test various valid mobile number formats"""
        assert validate_and_format_mobile(input_number) == expected

    @pytest.mark.parametrize("input_number,expected_error", [
        ("987654321", "Mobile number must be exactly 10 digits"),  # Too short
        ("98765432100", "Mobile number must be exactly 10 digits"),  # Too long
        ("5876543210", "Mobile number must start with 6, 7, 8 or 9"),  # Invalid start digit
        ("1234567890", "Mobile number must start with 6, 7, 8 or 9"),  # Invalid start digit
        ("abcdefghij", "Input must be exactly 10 digits, no other characters allowed"),  # Non-numeric
        ("98765-4321", "Input must be exactly 10 digits, no other characters allowed"),  # Contains hyphen
        ("9876 54321", "Input must be exactly 10 digits, no other characters allowed"),  # Contains space
        ("", "Mobile number must be exactly 10 digits"),  # Empty string
    ])
    def test_invalid_mobile_numbers(self, input_number, expected_error):
        """Test various invalid mobile number formats"""
        is_valid, formatted, error = validate_and_format_mobile(input_number)
        assert not is_valid
        assert error == expected_error
        assert formatted == ""

    def test_edge_cases(self):
        """Test edge cases and potential issues"""
        # Test with None
        with pytest.raises(AttributeError):
            validate_and_format_mobile(None)
        
        # Test with non-string number
        with pytest.raises(AttributeError):
            validate_and_format_mobile(1234567890)
        
        # Test with whitespace-only string
        result = validate_and_format_mobile("          ")
        assert not result[0]  # should be invalid
        assert "Input must be exactly 10 digits" in result[2]