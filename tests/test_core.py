"""
Tests for the core functionality of LLMShield.

! Module is intended for internal use only.
"""

import re
import time
import random

from unittest import TestCase, main

from llmshield import LLMShield
from llmshield.entity_detector import EntityType
from llmshield.utils import wrap_entity


class TestCoreFunctionality(TestCase):
    """Test core functionality of LLMShield."""

    def setUp(self):
        """Set up test cases."""
        self.start_delimiter = '['
        self.end_delimiter = ']'
        self.llm_func = lambda prompt: "Thanks [PERSON_0], I'll send details to [EMAIL_0]"
        self.shield = LLMShield(
            llm_func=self.llm_func,
            start_delimiter=self.start_delimiter,
            end_delimiter=self.end_delimiter
        )

        # Updated test prompt with proper spacing
        self.test_prompt = (
            "Hi, I'm John Doe.\n"
            "You can reach me at john.doe@example.com.\n"
            "Some numbers are 192.168.1.1 and 378282246310005\n"
        )
        self.test_entity_map = {
            wrap_entity(EntityType.EMAIL, 0, self.start_delimiter, self.end_delimiter):
                "john.doe@example.com",
            wrap_entity(EntityType.PERSON, 0, self.start_delimiter, self.end_delimiter):
                "John Doe",
            wrap_entity(EntityType.IP_ADDRESS, 0, self.start_delimiter, self.end_delimiter):
                "192.168.1.1",
            wrap_entity(EntityType.CREDIT_CARD, 0, self.start_delimiter, self.end_delimiter):
                "378282246310005"
        }
        self.test_llm_response = "Thanks " + self.test_entity_map[
            wrap_entity(EntityType.PERSON, 0, self.start_delimiter, self.end_delimiter)
        ] + ", I'll send details to " + self.test_entity_map[
            wrap_entity(EntityType.EMAIL, 0, self.start_delimiter, self.end_delimiter)
        ]

    def test_cloak_sensitive_info(self):
        """Test that sensitive information is properly cloaked."""
        cloaked_prompt, entity_map = self.shield.cloak(self.test_prompt)

        # Check that sensitive information is removed
        self.assertNotIn("john.doe@example.com", cloaked_prompt)
        self.assertNotIn("John Doe", cloaked_prompt)
        self.assertNotIn("192.168.1.1", cloaked_prompt)
        self.assertNotIn("378282246310005", cloaked_prompt)
        self.assertTrue(len(entity_map) == 4, f"Entity map should have 4 items: {entity_map}")

    def test_uncloak(self):
        """Test that cloaked entities are properly restored."""
        cloaked_prompt, entity_map = self.shield.cloak(self.test_prompt)
        uncloaked = self.shield.uncloak(cloaked_prompt, entity_map)
        self.assertEqual(uncloaked, self.test_prompt,
            f"Uncloaked response is not equal to test prompt: {uncloaked} != {self.test_prompt}")

    def test_end_to_end(self):
        """Test end-to-end flow with mock LLM function."""
        def mock_llm(prompt: str) -> str:
            time.sleep(float(random.randint(1, 10)) / 10)
            person_match = re.search(r'\[PERSON_\d+\]', prompt)
            email_match = re.search(r'\[EMAIL_\d+\]', prompt)
            return f"Thanks {person_match.group()}, I'll send details to {email_match.group()}"

        shield = LLMShield(
            llm_func=mock_llm,
            start_delimiter=self.start_delimiter,
            end_delimiter=self.end_delimiter
        )

        # Updated test input
        test_input = "Hi, I'm John Doe (john.doe@example.com)"
        response = shield.ask(prompt=test_input)

        # Test the entity map - use _ for intentionally unused variable
        _, _ = self.shield.cloak(test_input)

        self.assertIn("John Doe", response)
        self.assertIn("john.doe@example.com", response)

    def test_delimiter_customization(self):
        """Test custom delimiter functionality."""
        shield = LLMShield(start_delimiter='[[', end_delimiter=']]')
        cloaked_prompt, _ = shield.cloak("Hi, I'm John Doe")
        self.assertIn("[[PERSON_0]]", cloaked_prompt)
        self.assertNotIn("<PERSON_0>", cloaked_prompt)

    def test_entity_detection_accuracy(self):
        """Test accuracy of entity detection with complex examples."""
        test_cases = [
            # Test case 1: Proper Nouns
            {
                "input": "Dr. John Smith from Microsoft Corporation visited New York. "
                        "The CEO of Apple Inc met with IBM executives at UNESCO headquarters.",
                "expected_entities": {
                    "John Smith": EntityType.PERSON,
                    "Microsoft Corporation": EntityType.ORGANISATION,
                    "New York": EntityType.PLACE,
                    "Apple Inc": EntityType.ORGANISATION,
                    "IBM": EntityType.ORGANISATION,
                    "UNESCO": EntityType.ORGANISATION
                }
            },
            # Test case 2: Numbers and Locators
            {
                "input": "Contact us at support@company.com or call 44 (555) 123-4567. "
                        "Visit https://www.company.com. "
                        "Server IP: 192.168.1.1. "
                        "Credit card: 378282246310005",
                "expected_entities": {
                    "support@company.com": EntityType.EMAIL,
                    "https://www.company.com": EntityType.URL,
                    "192.168.1.1": EntityType.IP_ADDRESS,
                    "378282246310005": EntityType.CREDIT_CARD
                }
            }
        ]

        for i, test_case in enumerate(test_cases, 1):
            input_text = test_case["input"]
            expected = test_case["expected_entities"]

            # Get cloaked text and entity map - use the result to verify entities
            _, entity_map = self.shield.cloak(input_text)

            # Verify each expected entity is found
            for entity_text, entity_type in expected.items():
                found = False
                for placeholder, value in entity_map.items():
                    if value == entity_text and entity_type.name in placeholder:
                        found = True
                        break
                self.assertTrue(
                    found,
                    f"Failed to detect {entity_type.name}: '{entity_text}' in test case {i}"
                )

    def test_error_handling(self):
        """Test error handling in core functions."""
        shield = LLMShield(start_delimiter='[[', end_delimiter=']]')

        # Test invalid inputs - these should cover lines 59, 61, 63
        with self.assertRaises(ValueError):
            shield.ask(prompt=None)  # Line 59
        with self.assertRaises(ValueError):
            shield.ask(prompt="")    # Line 61
        with self.assertRaises(ValueError):
            shield.ask(prompt="   ") # Line 63

        # Test LLM errors
        def failing_llm(**kwargs):
            raise ValueError("LLM failed")  # Use specific exception type

        shield_with_failing_llm = LLMShield(
            llm_func=failing_llm,
            start_delimiter='[[',
            end_delimiter=']]'
        )

        with self.assertRaises(ValueError):
            shield_with_failing_llm.ask(prompt="Hello John Doe")

        # Test empty responses
        shield_empty = LLMShield(
            llm_func=lambda **kwargs: "No entity found",
            start_delimiter='[[',
            end_delimiter=']]'
        )
        response = shield_empty.ask(prompt="Hello John Doe")
        self.assertEqual(response, "No entity found")

        # Test dict response
        shield_dict = LLMShield(
            llm_func=lambda **kwargs: {"content": "test"},
            start_delimiter='[[',
            end_delimiter=']]'
        )
        response = shield_dict.ask(prompt="Hello John Doe")
        self.assertEqual(response, {"content": "test"})


    def test_error_propagation(self):
        """Test specific error propagation in ask method to cover lines 113-115."""
        # Create a custom exception to ensure we're testing the right pathway
        class CustomError(Exception):
            """Custom exception for testing."""
            pass

        # This LLM function raises the custom exception during processing
        # specifically to test lines 113-115
        def llm_with_specific_error(**kwargs):  # Accept keyword arguments
            raise CustomError("Test exception")

        shield = LLMShield(
            llm_func=llm_with_specific_error,
            start_delimiter='<<',
            end_delimiter='>>'
        )

        # This should propagate the exception through lines 113-115
        with self.assertRaises(CustomError):
            shield.ask(prompt="Test prompt")

    def test_constructor_validation(self):
        """Test constructor validation (lines 59, 61, 63)."""
        # Test invalid start delimiter (line 59)
        with self.assertRaises(ValueError):
            LLMShield(start_delimiter="", end_delimiter="]")

        # Test invalid end delimiter (line 61)
        with self.assertRaises(ValueError):
            LLMShield(start_delimiter="[", end_delimiter="")

        # Test non-callable llm_func (line 63)
        with self.assertRaises(ValueError):
            LLMShield(start_delimiter="[", end_delimiter="]", llm_func="not_callable")

    def test_uncloak_with_stored_entity_map(self):
        """Test uncloaking with stored entity map from previous cloak (line 115)."""
        shield = LLMShield(
            start_delimiter='[[',
            end_delimiter=']]'
        )

        # First cloak something to store the entity map internally
        test_text = "Hello John Doe"
        cloaked_text, _ = shield.cloak(test_text)

        # Now uncloak without providing an entity map - should use stored one from _last_entity_map
        uncloaked = shield.uncloak(cloaked_text, entity_map=None)

        # Should successfully uncloak using the stored map
        self.assertEqual(uncloaked, test_text)

    def test_ask_missing_required_param(self):
        """Test ValueError when neither 'prompt' nor 'message' is provided to ask."""
        shield = LLMShield(
            llm_func=lambda **kwargs: "Response",
            start_delimiter='[[',
            end_delimiter=']]'
        )

        # Call ask without providing either 'prompt' or 'message'
        with self.assertRaises(ValueError):
            shield.ask()  # No prompt or message provided

    def test_uncloak_invalid_response_type(self):
        """Test ValueError when trying to uncloak invalid response types."""
        shield = LLMShield(
            start_delimiter='[[',
            end_delimiter=']]'
        )

        # Create a mock entity map
        entity_map = {"[[PERSON_0]]": "John Doe"}

        # Try to uncloak various invalid response types
        invalid_responses = [
            123,  # int
            3.14,  # float
            True,  # bool
            (1, 2, 3),  # tuple
        ]

        for response in invalid_responses:
            with self.assertRaises(TypeError) as context:
                shield.uncloak(response, entity_map)

            # Verify the correct error message
            self.assertIn("Response must be ", str(context.exception))


if __name__ == '__main__':
    main(verbosity=2)