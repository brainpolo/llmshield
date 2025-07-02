"""Test Anthropic response handling and uncloaking.

Description:
    This test module validates proper handling of Anthropic Message objects,
    including content extraction and uncloaking of text and tool use blocks.

Test Classes:
    - TestAnthropicResponse: Tests Anthropic response uncloaking

Author: LLMShield by brainpolo, 2025
"""

import unittest
from unittest.mock import Mock

from llmshield.detection_utils import (
    extract_anthropic_content,
    is_anthropic_message_like,
)
from llmshield.uncloak_response import _uncloak_anthropic_message


class TestAnthropicResponse(unittest.TestCase):
    """Test Anthropic response handling."""

    def test_is_anthropic_message_like(self):
        """Test detection of Anthropic Message objects."""
        # Valid Anthropic Message
        anthropic_msg = Mock()
        anthropic_msg.content = "Hello world"
        anthropic_msg.model = "claude-3-5-haiku-20241022"
        anthropic_msg.role = "assistant"

        self.assertTrue(is_anthropic_message_like(anthropic_msg))

        # Invalid object (missing model and role attributes)
        invalid_obj = Mock()
        invalid_obj.content = "Hello"
        del invalid_obj.model  # Remove the default Mock attributes
        del invalid_obj.role

        self.assertFalse(is_anthropic_message_like(invalid_obj))

        # Regular dict (not Anthropic Message)
        regular_dict = {"content": "Hello", "role": "user"}
        self.assertFalse(is_anthropic_message_like(regular_dict))

    def test_extract_anthropic_content_simple_string(self):
        """Test extracting simple string content."""
        anthropic_msg = Mock()
        anthropic_msg.content = "Hello world"
        anthropic_msg.model = "claude-3-5-haiku-20241022"
        anthropic_msg.role = "assistant"

        content = extract_anthropic_content(anthropic_msg)
        self.assertEqual(content, "Hello world")

    def test_extract_anthropic_content_non_anthropic_message(self):
        """Test extracting content from non-Anthropic message returns None."""
        regular_dict = {"content": "Hello", "role": "user"}
        content = extract_anthropic_content(regular_dict)
        self.assertIsNone(content)

    def test_extract_anthropic_content_no_content_blocks(self):
        """Test extracting content when no text blocks are found."""
        anthropic_msg = Mock()
        anthropic_msg.model = "claude-3-5-haiku-20241022"
        anthropic_msg.role = "assistant"
        anthropic_msg.content = [
            {"type": "tool_use", "id": "call_123", "name": "get_weather"},
        ]

        content = extract_anthropic_content(anthropic_msg)
        self.assertIsNone(content)

    def test_extract_anthropic_content_attribute_error(self):
        """Test extracting content handles AttributeError gracefully."""
        anthropic_msg = Mock()
        anthropic_msg.model = "claude-3-5-haiku-20241022"
        anthropic_msg.role = "assistant"
        del anthropic_msg.content  # This will cause AttributeError

        content = extract_anthropic_content(anthropic_msg)
        self.assertIsNone(content)

    def test_extract_anthropic_content_blocks(self):
        """Test extracting content from content blocks."""
        anthropic_msg = Mock()
        anthropic_msg.model = "claude-3-5-haiku-20241022"
        anthropic_msg.role = "assistant"
        anthropic_msg.content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "world"},
        ]

        content = extract_anthropic_content(anthropic_msg)
        self.assertEqual(content, "Hello world")

    def test_extract_anthropic_content_mixed_blocks(self):
        """Test extracting content with mixed block types."""
        anthropic_msg = Mock()
        anthropic_msg.model = "claude-3-5-haiku-20241022"
        anthropic_msg.role = "assistant"
        anthropic_msg.content = [
            {"type": "text", "text": "Here's the weather:"},
            {"type": "tool_use", "id": "call_123", "name": "get_weather"},
            {"type": "text", "text": "It's sunny!"},
        ]

        content = extract_anthropic_content(anthropic_msg)
        self.assertEqual(content, "Here's the weather: It's sunny!")

    def test_extract_anthropic_content_object_blocks(self):
        """Test extracting content from object-style blocks."""
        anthropic_msg = Mock()
        anthropic_msg.model = "claude-3-5-haiku-20241022"
        anthropic_msg.role = "assistant"

        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Hello from object block"

        anthropic_msg.content = [text_block]

        content = extract_anthropic_content(anthropic_msg)
        self.assertEqual(content, "Hello from object block")

    def test_uncloak_anthropic_message_simple_text(self):
        """Test uncloaking simple text content."""
        anthropic_msg = Mock()
        anthropic_msg.content = "Hello <PERSON_0>"
        anthropic_msg.model = "claude-3-5-haiku-20241022"
        anthropic_msg.role = "assistant"

        entity_map = {"<PERSON_0>": "John"}

        result = _uncloak_anthropic_message(anthropic_msg, entity_map)
        self.assertEqual(result.content, "Hello John")

    def test_uncloak_anthropic_message_text_blocks(self):
        """Test uncloaking text in content blocks."""
        anthropic_msg = Mock()
        anthropic_msg.model = "claude-3-5-haiku-20241022"
        anthropic_msg.role = "assistant"
        anthropic_msg.content = [
            {"type": "text", "text": "Hello <PERSON_0>"},
            {"type": "text", "text": "Email: <EMAIL_0>"},
        ]

        entity_map = {"<PERSON_0>": "John", "<EMAIL_0>": "john@example.com"}

        result = _uncloak_anthropic_message(anthropic_msg, entity_map)

        self.assertEqual(result.content[0]["text"], "Hello John")
        self.assertEqual(result.content[1]["text"], "Email: john@example.com")

    def test_uncloak_anthropic_message_tool_use(self):
        """Test uncloaking tool use blocks."""
        anthropic_msg = Mock()
        anthropic_msg.model = "claude-3-5-haiku-20241022"
        anthropic_msg.role = "assistant"
        anthropic_msg.content = [
            {
                "type": "tool_use",
                "id": "call_123",
                "name": "send_email",
                "input": {
                    "to": "<EMAIL_0>",
                    "subject": "Meeting with <PERSON_0>",
                },
            }
        ]

        entity_map = {"<EMAIL_0>": "john@example.com", "<PERSON_0>": "John"}

        result = _uncloak_anthropic_message(anthropic_msg, entity_map)

        tool_block = result.content[0]
        self.assertEqual(tool_block["input"]["to"], "john@example.com")
        self.assertEqual(tool_block["input"]["subject"], "Meeting with John")

    def test_uncloak_anthropic_message_object_blocks(self):
        """Test uncloaking object-style blocks."""
        anthropic_msg = Mock()
        anthropic_msg.model = "claude-3-5-haiku-20241022"
        anthropic_msg.role = "assistant"

        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Hello <PERSON_0>"

        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.input = {"email": "<EMAIL_0>"}

        anthropic_msg.content = [text_block, tool_block]

        entity_map = {"<PERSON_0>": "John", "<EMAIL_0>": "john@example.com"}

        result = _uncloak_anthropic_message(anthropic_msg, entity_map)

        self.assertEqual(result.content[0].text, "Hello John")
        self.assertEqual(result.content[1].input["email"], "john@example.com")

    def test_uncloak_anthropic_message_preserves_structure(self):
        """Test that uncloaking preserves message structure."""
        anthropic_msg = Mock()
        anthropic_msg.content = "Hello world"
        anthropic_msg.model = "claude-3-5-haiku-20241022"
        anthropic_msg.role = "assistant"
        anthropic_msg.id = "msg_123"
        anthropic_msg.stop_reason = "end_turn"

        entity_map = {}

        result = _uncloak_anthropic_message(anthropic_msg, entity_map)

        # Check that other attributes are preserved
        self.assertEqual(result.model, "claude-3-5-haiku-20241022")
        self.assertEqual(result.role, "assistant")
        self.assertEqual(result.id, "msg_123")
        self.assertEqual(result.stop_reason, "end_turn")

    def test_uncloak_anthropic_message_attribute_error(self):
        """Test that uncloaking handles AttributeError gracefully."""
        # Create a malformed message object that will cause AttributeError
        anthropic_msg = Mock()
        del anthropic_msg.content  # Remove content attribute

        entity_map = {"<PERSON_0>": "John"}

        # Should not raise an exception
        result = _uncloak_anthropic_message(anthropic_msg, entity_map)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
