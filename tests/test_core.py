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
            "My IP is 192.168.1.1.\n"
            "Credit card: 378282246310005\n"
        )
        self.test_entity_map = {
            wrap_entity(EntityType.EMAIL, 0, self.start_delimiter, self.end_delimiter): "john.doe@example.com",
            wrap_entity(EntityType.PERSON, 0, self.start_delimiter, self.end_delimiter): "John Doe",
            wrap_entity(EntityType.IP_ADDRESS, 0, self.start_delimiter, self.end_delimiter): "192.168.1.1",
            wrap_entity(EntityType.CREDIT_CARD, 0, self.start_delimiter, self.end_delimiter): "378282246310005"
        }
        self.test_llm_response = "Thanks " + self.test_entity_map[wrap_entity(EntityType.PERSON, 0, self.start_delimiter, self.end_delimiter)] + ", I'll send details to " + self.test_entity_map[wrap_entity(EntityType.EMAIL, 0, self.start_delimiter, self.end_delimiter)]

    def test_cloak_sensitive_info(self):
        """Test that sensitive information is properly cloaked."""
        cloaked_prompt, entity_map = self.shield.cloak(self.test_prompt)

        # Check that sensitive information is removed
        self.assertNotIn("john.doe@example.com", cloaked_prompt)
        self.assertNotIn("John Doe", cloaked_prompt)
        self.assertNotIn("192.168.1.1", cloaked_prompt)
        self.assertNotIn("378282246310005", cloaked_prompt)
        # Print for debugging
        print("\nCloaked prompt:", cloaked_prompt)
        print("\nEntity map:", entity_map)

    def test_uncloak(self):
        """Test that cloaked entities are properly restored."""
        cloaked_prompt, entity_map = self.shield.cloak(self.test_prompt)
        uncloaked = self.shield.uncloak(cloaked_prompt, entity_map)
        self.assertEqual(uncloaked, self.test_prompt,
            f"Uncloaked response is not equal to test prompt: {uncloaked} != {self.test_prompt}")

    def test_end_to_end(self):
        """Test end-to-end flow with mock LLM function."""
        def MockLLM(prompt: str, start_delimiter: str, end_delimiter: str) -> str:
            time.sleep(float(random.randint(1, 10)) / 10)
            person_match = re.search(f"{re.escape(start_delimiter)}PERSON_\\d+{re.escape(end_delimiter)}", prompt)
            email_match = re.search(f"{re.escape(start_delimiter)}EMAIL_\\d+{re.escape(end_delimiter)}", prompt)
            return f"Thanks {person_match.group()}, I'll send details to {email_match.group()}"

        shield = LLMShield(
            llm_func=MockLLM,
            start_delimiter=self.start_delimiter,
            end_delimiter=self.end_delimiter
        )

        # Updated test input
        test_input = "Hi, I'm John Doe (john.doe@example.com)"
        response = shield.ask(
            test_input,
            start_delimiter=self.start_delimiter,
            end_delimiter=self.end_delimiter
        )

        # Test the entity map
        _, entity_map = shield.cloak(test_input)

        # Print for debugging
        print("\nEntity map:", entity_map)
        print("\nResponse:", response)

        self.assertIn("John Doe", response)
        self.assertIn("john.doe@example.com", response)

    def test_delimiter_customization(self):
        """Test custom delimiter functionality."""
        shield = LLMShield(start_delimiter='[[', end_delimiter=']]  ')
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

            # Get cloaked text and entity map
            cloaked, entity_map = self.shield.cloak(input_text)

            # Print debug information
            print(f"\nTest case {i}:")
            print(f"Input: {input_text}")
            print(f"Cloaked: {cloaked}")
            print(f"Entity map: {entity_map}")
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


if __name__ == '__main__':
    main(verbosity=2)