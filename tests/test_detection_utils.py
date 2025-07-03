"""Test detection utility functions.

Description:
    This test module provides comprehensive testing for the detection
    utility functions, including ChatCompletion-like object detection
    and content extraction.

Test Classes:
    - TestDetectionUtils: Tests detection utility functions

Author:
    LLMShield by brainpolo, 2025
"""

import unittest
from unittest.mock import Mock

from llmshield.detection_utils import (
    extract_anthropic_content,
    extract_chatcompletion_content,
    is_anthropic_message_like,
    is_chatcompletion_like,
)


class TestDetectionUtils(unittest.TestCase):
    """Test detection utility functions."""

    def test_is_chatcompletion_like_valid(self):
        """Test is_chatcompletion_like with valid objects."""
        # Valid ChatCompletion-like object
        obj = Mock()
        obj.choices = [Mock()]
        obj.model = "gpt-4"
        self.assertTrue(is_chatcompletion_like(obj))

        # Empty choices still valid
        obj = Mock()
        obj.choices = []
        obj.model = "gpt-4"
        self.assertTrue(is_chatcompletion_like(obj))

    def test_is_chatcompletion_like_invalid(self):
        """Test is_chatcompletion_like with invalid objects."""
        # Test missing attributes
        for attrs in [{"model": "gpt-4"}, {"choices": []}]:
            obj = Mock(**attrs)
            for attr in ["choices", "model"]:
                if attr not in attrs:
                    delattr(obj, attr)
            self.assertFalse(is_chatcompletion_like(obj))

        # Test non-object types
        for invalid in ["string", 123, None, [], {}]:
            self.assertFalse(is_chatcompletion_like(invalid))

    def test_extract_chatcompletion_content_valid(self):
        """Test extract_chatcompletion_content with valid content."""
        # Regular message content
        obj = Mock(
            choices=[Mock(message=Mock(content="Hello world"))], model="gpt-4"
        )
        self.assertEqual(extract_chatcompletion_content(obj), "Hello world")

        # Streaming delta content
        choice = Mock(delta=Mock(content="Streaming content"))
        delattr(choice, "message")
        obj = Mock(choices=[choice], model="gpt-4")
        content = extract_chatcompletion_content(obj)
        self.assertEqual(content, "Streaming content")

    def test_extract_chatcompletion_content_none(self):
        """Test extract_chatcompletion_content with None content."""
        # Test None content in both message and delta
        for attr_name in ["message", "delta"]:
            choice = Mock(**{attr_name: Mock(content=None)})
            if attr_name == "delta":
                delattr(choice, "message")
            obj = Mock(choices=[choice], model="gpt-4")
            self.assertIsNone(extract_chatcompletion_content(obj))

    def test_extract_chatcompletion_content_invalid(self):
        """Test extract_chatcompletion_content with invalid objects."""
        test_cases = [
            Mock(),  # Missing model attribute
            Mock(choices=[], model="gpt-4"),  # Empty choices
            Mock(choices=[Mock()], model="gpt-4"),  # No message/delta
            "string",
            123,
            None,  # Non-object types
        ]

        # Cleanup Mock objects
        if hasattr(test_cases[0], "model"):
            delattr(test_cases[0], "model")
        for attr in ["message", "delta"]:
            if hasattr(test_cases[2].choices[0], attr):
                delattr(test_cases[2].choices[0], attr)

        for obj in test_cases:
            self.assertIsNone(extract_chatcompletion_content(obj))

    def test_extract_chatcompletion_content_missing_attributes(self):
        """Test extract_chatcompletion_content with missing attributes."""

        # Create a proper object without content attribute
        class MessageWithoutContent:
            pass

        class ChoiceWithMessage:
            def __init__(self, message):
                self.message = message

        message = MessageWithoutContent()
        choice = ChoiceWithMessage(message)
        obj = Mock(choices=[choice], model="gpt-4")

        content = extract_chatcompletion_content(obj)
        self.assertIsNone(content)

        # Create delta without content attribute
        class DeltaWithoutContent:
            pass

        class ChoiceWithDelta:
            def __init__(self, delta):
                self.delta = delta

        delta = DeltaWithoutContent()
        choice = ChoiceWithDelta(delta)
        obj = Mock(choices=[choice], model="gpt-4")

        content = extract_chatcompletion_content(obj)
        self.assertIsNone(content)

    def test_extract_anthropic_content_attribute_error(self):
        """Test extract_anthropic_content with AttributeError."""

        class BadAnthropicMessage:
            role = "assistant"
            model = "claude-3"
            content = None

            def __getattribute__(self, name):
                if name == "content" and hasattr(self, "_accessing_content"):
                    raise AttributeError("content attribute error")
                return object.__getattribute__(self, name)

        bad_msg = BadAnthropicMessage()
        self.assertTrue(is_anthropic_message_like(bad_msg))

        bad_msg._accessing_content = True
        self.assertIsNone(extract_anthropic_content(bad_msg))


if __name__ == "__main__":
    unittest.main(verbosity=2)
