"""Test utility functions and helper methods.

Description:
    This test module provides testing for utility functions including
    conversation hashing, entity wrapping, input processing, and various
    helper functions used throughout the library.

Test Classes:
    - TestConversationHash: Tests conversation hashing functionality
    - TestWrapEntity: Tests entity wrapping with delimiters
    - TestUtils: Tests miscellaneous utility functions
    - TestShouldCloakInput: Tests input cloaking logic
    - TestProcessAskHelperInput: Tests input processing helpers

Author: LLMShield by brainpolo, 2025
"""

# Standard library Imports
import unittest
from pathlib import Path

# Third party Imports
from parameterized import parameterized

# Local Imports
from llmshield import LLMShield
from llmshield.entity_detector import EntityType
from llmshield.utils import (
    PydanticLike,
    _should_cloak_input,
    ask_helper,
    conversation_hash,
    is_valid_delimiter,
    wrap_entity,
)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.start_delimiter = "["
        self.end_delimiter = "]"

    def test_is_valid_delimiter(self):
        """Test delimiter validation function."""
        # Valid cases
        self.assertTrue(is_valid_delimiter("["))
        self.assertTrue(is_valid_delimiter("]]"))
        self.assertTrue(is_valid_delimiter("<<<"))
        self.assertTrue(is_valid_delimiter("#"))

        # Invalid cases
        self.assertFalse(is_valid_delimiter(""))
        self.assertFalse(is_valid_delimiter(None))
        self.assertFalse(is_valid_delimiter(123))
        self.assertFalse(is_valid_delimiter(["["]))

    def test_wrap_entity(self):
        """Test entity wrapping function."""
        # Test with different entity types
        self.assertEqual(
            wrap_entity(EntityType.PERSON, 0, "[", "]"), "[PERSON_0]"
        )
        self.assertEqual(
            wrap_entity(EntityType.EMAIL, 1, "<", ">"), "<EMAIL_1>"
        )

        # Test with multi-character delimiters
        self.assertEqual(
            wrap_entity(EntityType.PHONE, 2, "[[", "]]"),
            "[[PHONE_2]]",
        )

    @parameterized.expand(
        [
            # (description, param_name, input_value, expected_str)
            ("integer_prompt", "prompt", 123, "123"),
            (
                "path_message",
                "message",
                "Path('/test/path')",
                "/test/path",
            ),  # Path string representation is just the path
            (
                "bytes_prompt",
                "prompt",
                b"binary data",
                "b'binary data'",
            ),  # Bytes string representation
            (
                "tuple_message",
                "message",
                ("file.txt", b"content"),
                "('file.txt', b'content')",
            ),
            ("none_prompt", "prompt", None, "None"),
        ]
    )
    def test_ask_helper_no_cloaking_scenarios(
        self, description, param_name, input_value, expected_str
    ):
        """Test ask_helper when input doesn't need cloaking (line 180).

        Parameterized test for various input types that don't require cloaking.
        """
        # Handle Path object specially
        if description == "path_message":
            input_value = Path("/test/path")

        def mock_llm(**kwargs):
            param_name_actual = "prompt" if "prompt" in kwargs else "message"
            value = kwargs.get(param_name_actual, "unknown")
            return f"Response to: {value}"

        shield = LLMShield(llm_func=mock_llm)
        kwargs = {param_name: input_value, "stream": False}
        result = ask_helper(shield=shield, **kwargs)
        expected = f"Response to: {expected_str}"
        self.assertEqual(result, expected)

    @parameterized.expand(
        [
            # (description, input_obj, expected_tuple_for_hash)
            ("empty_content", {"role": "user", "content": ""}, ("user", "")),
            ("missing_content", {"role": "user"}, ("user", "")),
            ("empty_dict", {}, ("", "")),
            ("missing_role", {"content": "hello"}, ("", "hello")),
        ]
    )
    def test_conversation_hash_edge_cases(
        self, description, message, expected_tuple
    ):
        """Test conversation_hash with edge cases - parameterized."""
        result = conversation_hash(message)
        expected = hash(expected_tuple)
        self.assertEqual(result, expected)

    def test_conversation_hash_list_of_messages(self):
        """Test conversation_hash with list of messages."""
        # Test list of messages
        messages = [{}, {"role": "assistant"}]
        result = conversation_hash(messages)
        expected = hash(frozenset([("", ""), ("assistant", "")]))
        self.assertEqual(result, expected)

    @parameterized.expand(
        [
            # (description, input_value, should_cloak)
            ("empty_string", "", True),
            ("normal_string", "hello", True),
            ("empty_list", [], True),
            ("string_list", ["hello", "world"], True),
            (
                "mixed_list",
                ["string", 123],
                True,
            ),  # Still a list, so should cloak
            ("integer", 123, False),
            ("dict", {"key": "value"}, False),
            ("bytes", b"bytes", False),
            ("none", None, False),
            ("float", 3.14, False),
        ]
    )
    def test_should_cloak_input_comprehensive(
        self, description, input_value, expected_result
    ):
        """Test _should_cloak_input with various input types.

        Parameterized test for comprehensive input type validation.
        """
        result = _should_cloak_input(input_value)
        self.assertEqual(result, expected_result)

    @parameterized.expand(
        [
            # (description, model_class_name, should_match_protocol)
            ("complete_model", "GoodModel", True),
            ("empty_model", "BadModel", False),
            ("missing_validate", "PartialModel", False),
            ("missing_dump", "AnotherPartialModel", False),
        ]
    )
    def test_pydantic_like_protocol_cases(
        self, description, model_class_name, should_match
    ):
        """Test PydanticLike protocol runtime checking - parameterized."""

        # Create test classes with different method combinations
        class GoodModel:
            @staticmethod
            def model_dump() -> dict:
                return {}

            @classmethod
            def model_validate(cls, data: dict):
                return cls()

        class BadModel:
            pass

        class PartialModel:
            @staticmethod
            def model_dump() -> dict:
                return {}

            # Missing model_validate

        class AnotherPartialModel:
            @classmethod
            def model_validate(cls, data: dict):
                return cls()

            # Missing model_dump

        # Get the class by name
        model_classes = {
            "GoodModel": GoodModel,
            "BadModel": BadModel,
            "PartialModel": PartialModel,
            "AnotherPartialModel": AnotherPartialModel,
        }

        model_class = model_classes[model_class_name]
        instance = model_class()
        result = isinstance(instance, PydanticLike)
        self.assertEqual(result, should_match)


if __name__ == "__main__":
    unittest.main(verbosity=2)
