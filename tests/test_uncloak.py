"""Test uncloaking functionality for restoring entities.

Description:
    This test module provides testing for the uncloaking functionality that
    restores original entities from cloaked placeholders in LLM responses,
    with focus on edge cases and error handling.

Test Classes:
    - TestUncloak: Tests uncloaking edge cases and validation

Author:
    LLMShield by brainpolo, 2025
"""

from unittest import TestCase

from llmshield import LLMShield


class TestUncloak(TestCase):
    """Tests for the uncloak functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.shield = LLMShield()

    def test_uncloak_edge_cases(self):
        """Test edge cases in uncloaking."""
        # Test empty inputs
        with self.assertRaises(ValueError):
            self.shield.uncloak("", {})

        # Test partial replacements
        self.assertEqual(
            self.shield.uncloak(
                "Hello [PERSON_0] and [PERSON_1]", {"[PERSON_0]": "John"}
            ),
            "Hello John and [PERSON_1]",
        )

        # Test multiple replacements
        self.assertEqual(
            self.shield.uncloak(
                "[PERSON_0] [PERSON_0] [PERSON_1]",
                {"[PERSON_0]": "John", "[PERSON_1]": "Smith"},
            ),
            "John John Smith",
        )

    def test_recursive_dict_uncloaking(self):
        """Test recursive uncloaking of nested dictionary structures."""
        # Create a nested dictionary response (like structured output from an
        # LLM)
        nested_response = {
            "header": "Message from [PERSON_0]",
            "body": "Hello, I'm [PERSON_0] from [ORGANISATION_0]",
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
                "nested": {
                    "deeply": {
                        "secret": "This is [PERSON_0]'s secret address: "
                        "[IP_ADDRESS_0]"
                    }
                },
            },
        }

        # Entity map with all placeholders
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

        # Apply uncloaking
        uncloaked = self.shield.uncloak(nested_response, entity_map)

        # Verify top-level strings are uncloaked
        self.assertEqual(uncloaked["header"], "Message from John Doe")
        self.assertEqual(
            uncloaked["body"], "Hello, I'm John Doe from Acme Corp"
        )

        # Verify nested dictionary is uncloaked
        self.assertEqual(uncloaked["metadata"]["sender"]["name"], "John Doe")
        self.assertEqual(
            uncloaked["metadata"]["sender"]["email"], "john.doe@example.com"
        )
        self.assertEqual(
            uncloaked["metadata"]["sender"]["company"], "Acme Corp"
        )

        # Verify arrays of dictionaries are uncloaked
        self.assertEqual(
            uncloaked["metadata"]["recipients"][0]["name"], "Jane Smith"
        )
        self.assertEqual(
            uncloaked["metadata"]["recipients"][0]["contact"],
            "+1-555-123-4567",
        )
        self.assertEqual(
            uncloaked["metadata"]["recipients"][1]["name"], "Bob Johnson"
        )
        self.assertEqual(
            uncloaked["metadata"]["recipients"][1]["contact"],
            "bob@example.com",
        )

        # Verify deeply nested structures are uncloaked
        self.assertEqual(
            uncloaked["metadata"]["nested"]["deeply"]["secret"],
            "This is John Doe's secret address: 192.168.1.1",
        )

        # Verify non-string values are preserved
        self.assertEqual(uncloaked["metadata"]["confidential"], True)

    def test_uncloak_with_llmshield(self):
        """Test recursive uncloaking with the LLMShield class."""
        shield = LLMShield(start_delimiter="[", end_delimiter="]")

        # Create structured response with nested placeholders
        structured_response = {
            "answer": "My name is [PERSON_0] and I work at [ORGANISATION_0]",
            "metadata": {
                "entities": {
                    "person": "[PERSON_0]",
                    "org": "[ORGANISATION_0]",
                    "location": {
                        "address": "123 Main St, [PLACE_0]",
                        "coordinates": "[IP_ADDRESS_0]",
                    },
                }
            },
        }

        # Create entity map
        entity_map = {
            "[PERSON_0]": "John Doe",
            "[ORGANISATION_0]": "Acme Inc",
            "[PLACE_0]": "New York",
            "[IP_ADDRESS_0]": "192.168.1.1",
        }

        # Apply uncloaking
        uncloaked = shield.uncloak(structured_response, entity_map)

        # Verify all levels are uncloaked
        self.assertEqual(
            uncloaked["answer"], "My name is John Doe and I work at Acme Inc"
        )
        self.assertEqual(
            uncloaked["metadata"]["entities"]["person"], "John Doe"
        )
        self.assertEqual(uncloaked["metadata"]["entities"]["org"], "Acme Inc")
        self.assertEqual(
            uncloaked["metadata"]["entities"]["location"]["address"],
            "123 Main St, New York",
        )
        self.assertEqual(
            uncloaked["metadata"]["entities"]["location"]["coordinates"],
            "192.168.1.1",
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
