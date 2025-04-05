"""Tests for regex pattern matching functionality."""

from unittest import TestCase

from llmshield.matchers.regex import (
    EMAIL_ADDRESS_PATTERN,
    CREDIT_CARD_PATTERN,
    IP_ADDRESS_PATTERN,
    URL_PATTERN,
    PHONE_NUMBER_PATTERN
)


class TestRegexMatchers(TestCase):
    """Test suite for regex pattern matching."""

    def test_email_pattern_valid(self):
        """Test email pattern matching."""
        valid_emails = [
            "john.doe@example.com",
            "user+tag@domain.co.uk",
            "first.last@subdomain.company.org"
        ]
        for email in valid_emails:
            with self.subTest(email=email):
                match = EMAIL_ADDRESS_PATTERN.fullmatch(email)
                self.assertIsNotNone(match, f"Email should match: {email}")
                self.assertEqual(match.group(), email)

    def test_email_pattern_invalid(self):
        """Test email pattern matching."""
        invalid_emails = [
            "plainaddress",
            "john.doe@.com",
            "john.doe@example",
            "john..doe@example.com"  # double dots not allowed
        ]
        for email in invalid_emails:
            with self.subTest(email=email):
                match = EMAIL_ADDRESS_PATTERN.fullmatch(email)
                self.assertIsNone(match, f"Email should not match: {email}")

    def test_credit_card_pattern_valid(self):
        """Test credit card pattern matching."""
        valid_cards = [
            "4532015112345678",      # Visa
            "5425233456788790",      # Mastercard
            "347352358990016",       # American Express
            "3530111333300000",      # JCB
            "6011000990139424",      # Discover
            # If your regex supports spaces/dashes, you can add:
            # "4532 0151 1234 5678",
            # "5425-2334-5678-8790",
        ]
        for card in valid_cards:
            with self.subTest(card=card):
                match = CREDIT_CARD_PATTERN.search(card)
                self.assertIsNotNone(match, f"Credit card should match: {card}")
                self.assertEqual(match.group(), card)

    def test_credit_card_pattern_invalid(self):
        """Test credit card pattern matching."""
        invalid_cards = [
            "123456789012",   # Too short
            "abcdefg",        # Non-digits
            "453201511234567"  # One digit short for Visa
        ]
        for card in invalid_cards:
            with self.subTest(card=card):
                match = CREDIT_CARD_PATTERN.search(card)
                self.assertIsNone(match, f"Credit card should not match: {card}")

    def test_ip_address_pattern_valid(self):
        """Test IP address pattern matching."""
        valid_ips = [
            "192.168.1.1",
            "10.0.0.0",
            "172.16.254.1"
        ]
        for ip in valid_ips:
            with self.subTest(ip=ip):
                match = IP_ADDRESS_PATTERN.search(ip)
                self.assertIsNotNone(match, f"IP address should match: {ip}")
                self.assertEqual(match.group(), ip)

    def test_ip_address_pattern_invalid(self):
        """Test IP address pattern matching."""
        invalid_ips = [
            "256.1.2.3",      # 256 is invalid
            "192.168.1",      # Not enough octets
            "123.456.789.0"   # Each octet must be 0-255
        ]
        for ip in invalid_ips:
            with self.subTest(ip=ip):
                match = IP_ADDRESS_PATTERN.search(ip)
                self.assertIsNone(match, f"IP address should not match: {ip}")

    def test_url_pattern_valid(self):
        """Test URL pattern matching."""
        valid_urls = [
            "https://example.com",
            "http://subdomain.example.com/path",
            "https://my-site.org/path?query=value",
            "http://domain.anything/path#fragment"
        ]
        for url in valid_urls:
            with self.subTest(url=url):
                match = URL_PATTERN.search(url)
                self.assertIsNotNone(match, f"URL should match: {url}")
                self.assertEqual(match.group(), url)

    def test_url_pattern_invalid(self):
        """Test URL pattern matching."""
        invalid_urls = [
            "ftp://example.com",    # Wrong protocol
            "justtext",
            "http//example.com"     # Missing colon after http
        ]
        for url in invalid_urls:
            with self.subTest(url=url):
                match = URL_PATTERN.search(url)
                self.assertIsNone(match, f"URL should not match: {url}")

    def test_phone_number_pattern_valid(self):
        """Test phone number pattern matching."""
        valid_numbers = [
            "123-456-7890",
            "(123) 456-7890",
            "123 456 7890",
            "123.456.7890",
            "+44 (123) 456-7890",
            "+1 123-456-7890",
            "+44 84491234567"
        ]
        for number in valid_numbers:
            with self.subTest(number=number):
                # Use fullmatch on the trimmed candidate.
                match = PHONE_NUMBER_PATTERN.fullmatch(number.strip())
                self.assertIsNotNone(match, f"Phone number should match: {number}")
                self.assertEqual(match.group().strip(), number.strip())

    def test_phone_number_pattern_invalid(self):
        """Test phone number pattern matching."""
        invalid_numbers = [
            "1234567",          # Too short
            "phone: 1234567890", # Contains alphabetical characters
            "12-3456-7890"       # Incorrect grouping
        ]
        for number in invalid_numbers:
            with self.subTest(number=number):
                match = PHONE_NUMBER_PATTERN.fullmatch(number.strip())
                self.assertIsNone(match, f"Phone number should not match: {number}")


if __name__ == '__main__':
    import unittest
    unittest.main(verbosity=2)
