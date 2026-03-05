"""Test detection utility functions.

Description:
    This test module provides comprehensive testing for the detection
    utility functions, including ChatCompletion-like object detection
    and content extraction.

Test Classes:
    - TestDetectionUtils: Tests detection utility functions

Author:
    LLMShield by brainpolo, 2025-2026
"""

import unittest
from unittest.mock import Mock

from llmshield.detection_utils import (
    extract_anthropic_content,
    extract_chatcompletion_content,
    extract_cohere_content,
    extract_google_content,
    extract_xai_content,
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
            """Mock message without content attribute."""

        class ChoiceWithMessage:
            """Mock choice containing a message."""

            def __init__(self, message):
                """Initialise instance."""
                self.message = message

        message = MessageWithoutContent()
        choice = ChoiceWithMessage(message)
        obj = Mock(choices=[choice], model="gpt-4")

        content = extract_chatcompletion_content(obj)
        self.assertIsNone(content)

        # Create delta without content attribute
        class DeltaWithoutContent:
            """Mock delta without content attribute."""

        class ChoiceWithDelta:
            """Mock choice containing a delta."""

            def __init__(self, delta):
                """Initialise instance."""
                self.delta = delta

        delta = DeltaWithoutContent()
        choice = ChoiceWithDelta(delta)
        obj = Mock(choices=[choice], model="gpt-4")

        content = extract_chatcompletion_content(obj)
        self.assertIsNone(content)

    def test_extract_anthropic_content_attribute_error(self):
        """Test extract_anthropic_content with AttributeError."""

        class FailOnSecondAccess:
            """Mock that fails on second content access."""

            role = "assistant"
            model = "claude-3"
            _count = 0

            @property
            def content(self):
                """Return content once, then raise AttributeError."""
                self._count += 1
                if self._count > 1:
                    raise AttributeError("boom")
                return "initial"

        obj = FailOnSecondAccess()
        self.assertIsNone(extract_anthropic_content(obj))

    def test_extract_xai_content_attribute_error(self):
        """Test extract_xai_content with broken content attr."""

        class BrokenXAI:
            """Mock xAI response with broken content."""

            @property
            def content(self):
                """Raise AttributeError to simulate broken access."""
                raise AttributeError("boom")

        BrokenXAI.__module__ = "xai_sdk.response"
        result = extract_xai_content(BrokenXAI())
        self.assertIsNone(result)

    def test_extract_xai_content_not_xai(self):
        """Test extract_xai_content returns None for non-xAI."""
        self.assertIsNone(extract_xai_content("not xai"))

    def test_extract_google_content_attribute_error(self):
        """Test extract_google_content with broken text attr."""

        class BrokenGoogle:
            """Mock Google response with broken text."""

            candidates = []
            usage_metadata = {}

            @property
            def text(self):
                """Raise ValueError to simulate missing text."""
                raise ValueError("no text")

        result = extract_google_content(BrokenGoogle())
        self.assertIsNone(result)

    def test_extract_google_content_not_google(self):
        """Test extract_google_content returns None for non-Google."""
        self.assertIsNone(extract_google_content("not google"))

    def test_extract_cohere_content_string(self):
        """Test extract_cohere_content with string content."""
        obj = Mock()
        obj.message = Mock()
        obj.message.content = "Hello world"
        obj.finish_reason = "COMPLETE"
        del obj.choices
        del obj.candidates
        del obj.role
        result = extract_cohere_content(obj)
        self.assertEqual(result, "Hello world")

    def test_extract_cohere_content_attribute_error(self):
        """Test extract_cohere_content with AttributeError."""

        class BrokenCohere:
            """Mock Cohere response with broken message."""

            @property
            def message(self):
                """Raise AttributeError to simulate failure."""
                raise AttributeError("no message")

        BrokenCohere.__module__ = "cohere.types"
        result = extract_cohere_content(BrokenCohere())
        self.assertIsNone(result)

    def test_extract_cohere_content_not_cohere(self):
        """Test extract_cohere_content returns None for non-Cohere."""
        self.assertIsNone(extract_cohere_content("not cohere"))

    def test_extract_anthropic_content_list_blocks(self):
        """Test extract_anthropic_content with list content blocks."""
        obj = Mock()
        obj.role = "assistant"
        obj.model = "claude-3"
        block = Mock()
        block.type = "text"
        block.text = "Hello from blocks"
        obj.content = [block]
        del obj.choices
        result = extract_anthropic_content(obj)
        self.assertEqual(result, "Hello from blocks")


if __name__ == "__main__":
    unittest.main(verbosity=2)
