"""Test uncloaking functionality for restoring entities.

Description:
    This test module provides testing for the uncloaking functionality that
    restores original entities from cloaked placeholders in LLM responses,
    with focus on edge cases and error handling.

Test Classes:
    - TestUncloak: Tests uncloaking edge cases and validation

Author:
    LLMShield by brainpolo, 2025-2026
"""

import unittest

from llmshield import LLMShield


class TestUncloak(unittest.TestCase):
    """Tests for the uncloak functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.shield = LLMShield()

    def test_uncloak_empty_input_raises(self):
        """Test that empty input raises ValueError."""
        with self.assertRaises(ValueError):
            self.shield.uncloak("", {})

    def test_uncloak_partial_replacements(self):
        """Test uncloaking when entity map is incomplete."""
        result = self.shield.uncloak(
            "Hello [PERSON_0] and [PERSON_1]",
            {"[PERSON_0]": "John"},
        )
        self.assertEqual(result, "Hello John and [PERSON_1]")

    def test_uncloak_repeated_placeholder(self):
        """Test uncloaking with the same placeholder appearing twice."""
        result = self.shield.uncloak(
            "[PERSON_0] [PERSON_0] [PERSON_1]",
            {"[PERSON_0]": "John", "[PERSON_1]": "Smith"},
        )
        self.assertEqual(result, "John John Smith")

    def test_recursive_dict_uncloaking(self):
        """Test recursive uncloaking of deeply nested structures."""
        nested_response = {
            "header": "Message from [PERSON_0]",
            "body": "Hello from [ORGANISATION_0]",
            "metadata": {
                "sender": {
                    "name": "[PERSON_0]",
                    "email": "[EMAIL_0]",
                    "company": "[ORGANISATION_0]",
                },
                "recipients": [
                    {"name": "[PERSON_1]", "contact": "[PHONE_0]"},
                    {"name": "[PERSON_2]", "contact": "[EMAIL_1]"},
                ],
                "confidential": True,
                "nested": {"deeply": {"secret": ("Address: [IP_ADDRESS_0]")}},
            },
        }

        entity_map = {
            "[PERSON_0]": "John Doe",
            "[PERSON_1]": "Jane Smith",
            "[PERSON_2]": "Bob Johnson",
            "[ORGANISATION_0]": "Acme Corp",
            "[EMAIL_0]": "john.doe@example.com",
            "[EMAIL_1]": "bob@example.com",
            "[PHONE_0]": "+1-555-123-4567",
            "[IP_ADDRESS_0]": "192.168.1.1",
        }

        uncloaked = self.shield.uncloak(nested_response, entity_map)

        # Top-level strings
        self.assertEqual(uncloaked["header"], "Message from John Doe")

        # Nested dicts
        sender = uncloaked["metadata"]["sender"]
        self.assertEqual(sender["name"], "John Doe")
        self.assertEqual(sender["email"], "john.doe@example.com")

        # Lists of dicts
        recipients = uncloaked["metadata"]["recipients"]
        self.assertEqual(recipients[0]["name"], "Jane Smith")
        self.assertEqual(recipients[0]["contact"], "+1-555-123-4567")

        # Deep nesting
        self.assertEqual(
            uncloaked["metadata"]["nested"]["deeply"]["secret"],
            "Address: 192.168.1.1",
        )

        # Non-string values preserved
        self.assertTrue(uncloaked["metadata"]["confidential"])


if __name__ == "__main__":
    unittest.main()
