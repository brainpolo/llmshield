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
    extract_chatcompletion_content,
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
        # Missing choices
        obj = Mock()
        obj.model = "gpt-4"
        delattr(obj, "choices")
        self.assertFalse(is_chatcompletion_like(obj))

        # Missing model
        obj = Mock()
        obj.choices = []
        delattr(obj, "model")
        self.assertFalse(is_chatcompletion_like(obj))

        # Non-object types
        self.assertFalse(is_chatcompletion_like("string"))
        self.assertFalse(is_chatcompletion_like(123))
        self.assertFalse(is_chatcompletion_like(None))
        self.assertFalse(is_chatcompletion_like([]))
        self.assertFalse(is_chatcompletion_like({}))

    def test_extract_chatcompletion_content_valid(self):
        """Test extract_chatcompletion_content with valid content."""
        # Regular message content
        message = Mock(content="Hello world")
        choice = Mock(message=message)
        obj = Mock(choices=[choice], model="gpt-4")

        content = extract_chatcompletion_content(obj)
        self.assertEqual(content, "Hello world")

        # Streaming delta content
        delta = Mock(content="Streaming content")
        choice = Mock(delta=delta)
        delattr(choice, "message")
        obj = Mock(choices=[choice], model="gpt-4")

        content = extract_chatcompletion_content(obj)
        self.assertEqual(content, "Streaming content")

    def test_extract_chatcompletion_content_none(self):
        """Test extract_chatcompletion_content with None content."""
        # None content in message
        message = Mock(content=None)
        choice = Mock(message=message)
        obj = Mock(choices=[choice], model="gpt-4")

        content = extract_chatcompletion_content(obj)
        self.assertIsNone(content)

        # None content in delta
        delta = Mock(content=None)
        choice = Mock(delta=delta)
        delattr(choice, "message")
        obj = Mock(choices=[choice], model="gpt-4")

        content = extract_chatcompletion_content(obj)
        self.assertIsNone(content)

    def test_extract_chatcompletion_content_invalid(self):
        """Test extract_chatcompletion_content with invalid objects."""
        # Not ChatCompletion-like
        obj = Mock()
        delattr(obj, "model")
        content = extract_chatcompletion_content(obj)
        self.assertIsNone(content)

        # Empty choices
        obj = Mock(choices=[], model="gpt-4")
        content = extract_chatcompletion_content(obj)
        self.assertIsNone(content)

        # Choice without message or delta
        choice = Mock()
        delattr(choice, "message")
        delattr(choice, "delta")
        obj = Mock(choices=[choice], model="gpt-4")
        content = extract_chatcompletion_content(obj)
        self.assertIsNone(content)

        # Non-object types
        self.assertIsNone(extract_chatcompletion_content("string"))
        self.assertIsNone(extract_chatcompletion_content(123))
        self.assertIsNone(extract_chatcompletion_content(None))

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
