"""Test Anthropic response handling and uncloaking.

Description:
    This test module validates proper handling of Anthropic Message objects,
    including content extraction and uncloaking of text and tool use blocks.

Test Classes:
    - TestAnthropicResponse: Tests Anthropic response uncloaking

Author: LLMShield by brainpolo, 2025-2026
"""

import unittest
from unittest.mock import Mock

from llmshield.detection_utils import (
    extract_anthropic_content,
    is_anthropic_message_like,
)
from llmshield.uncloak_response import _uncloak_anthropic_message
from tests.helpers import make_anthropic_msg


class TestAnthropicResponse(unittest.TestCase):
    """Test Anthropic response handling."""

    def test_is_anthropic_message_like(self):
        """Test detection of Anthropic Message objects."""
        self.assertTrue(is_anthropic_message_like(make_anthropic_msg()))

        # Missing model and role
        invalid = Mock()
        invalid.content = "Hello"
        del invalid.model
        del invalid.role
        self.assertFalse(is_anthropic_message_like(invalid))

        # Dict is not Anthropic Message
        self.assertFalse(
            is_anthropic_message_like({"content": "Hello", "role": "user"})
        )

    def test_extract_content_simple_string(self):
        """Test extracting simple string content."""
        msg = make_anthropic_msg(content="Hello world")
        self.assertEqual(extract_anthropic_content(msg), "Hello world")

    def test_extract_content_non_anthropic(self):
        """Test non-Anthropic message returns None."""
        self.assertIsNone(
            extract_anthropic_content({"content": "Hello", "role": "user"})
        )

    def test_extract_content_no_text_blocks(self):
        """Test no text blocks returns None."""
        msg = make_anthropic_msg(
            content=[
                {
                    "type": "tool_use",
                    "id": "call_123",
                    "name": "get_weather",
                },
            ]
        )
        self.assertIsNone(extract_anthropic_content(msg))

    def test_extract_content_missing_content_attr(self):
        """Test missing content attribute returns None."""
        msg = make_anthropic_msg()
        del msg.content
        self.assertIsNone(extract_anthropic_content(msg))

    def test_extract_content_blocks(self):
        """Test extracting content from multiple text blocks."""
        msg = make_anthropic_msg(
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "world"},
            ]
        )
        self.assertEqual(extract_anthropic_content(msg), "Hello world")

    def test_extract_content_mixed_blocks(self):
        """Test extracting content with mixed block types."""
        msg = make_anthropic_msg(
            content=[
                {"type": "text", "text": "Here's the weather:"},
                {
                    "type": "tool_use",
                    "id": "call_123",
                    "name": "get_weather",
                },
                {"type": "text", "text": "It's sunny!"},
            ]
        )
        self.assertEqual(
            extract_anthropic_content(msg),
            "Here's the weather: It's sunny!",
        )

    def test_extract_content_object_blocks(self):
        """Test extracting content from object-style blocks."""
        block = Mock()
        block.type = "text"
        block.text = "Hello from object block"
        msg = make_anthropic_msg(content=[block])
        self.assertEqual(
            extract_anthropic_content(msg),
            "Hello from object block",
        )

    def test_uncloak_simple_text(self):
        """Test uncloaking simple text content."""
        msg = make_anthropic_msg(content="Hello <PERSON_0>")
        result = _uncloak_anthropic_message(msg, {"<PERSON_0>": "John"})
        self.assertEqual(result.content, "Hello John")

    def test_uncloak_text_blocks(self):
        """Test uncloaking text in content blocks."""
        msg = make_anthropic_msg(
            content=[
                {"type": "text", "text": "Hello <PERSON_0>"},
                {"type": "text", "text": "Email: <EMAIL_0>"},
            ]
        )
        entity_map = {
            "<PERSON_0>": "John",
            "<EMAIL_0>": "john@example.com",
        }
        result = _uncloak_anthropic_message(msg, entity_map)
        self.assertEqual(result.content[0]["text"], "Hello John")
        self.assertEqual(
            result.content[1]["text"],
            "Email: john@example.com",
        )

    def test_uncloak_tool_use(self):
        """Test uncloaking tool use blocks."""
        msg = make_anthropic_msg(
            content=[
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
        )
        entity_map = {
            "<EMAIL_0>": "john@example.com",
            "<PERSON_0>": "John",
        }
        result = _uncloak_anthropic_message(msg, entity_map)
        tool = result.content[0]
        self.assertEqual(tool["input"]["to"], "john@example.com")
        self.assertEqual(tool["input"]["subject"], "Meeting with John")

    def test_uncloak_object_blocks(self):
        """Test uncloaking object-style blocks."""
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Hello <PERSON_0>"

        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.input = {"email": "<EMAIL_0>"}

        msg = make_anthropic_msg(content=[text_block, tool_block])
        entity_map = {
            "<PERSON_0>": "John",
            "<EMAIL_0>": "john@example.com",
        }
        result = _uncloak_anthropic_message(msg, entity_map)
        self.assertEqual(result.content[0].text, "Hello John")
        self.assertEqual(
            result.content[1].input["email"],
            "john@example.com",
        )

    def test_uncloak_preserves_structure(self):
        """Test that uncloaking preserves message structure."""
        msg = make_anthropic_msg(
            content="Hello world",
            id="msg_123",
            stop_reason="end_turn",
        )
        result = _uncloak_anthropic_message(msg, {})
        self.assertEqual(result.model, msg.model)
        self.assertEqual(result.role, "assistant")
        self.assertEqual(result.id, "msg_123")
        self.assertEqual(result.stop_reason, "end_turn")

    def test_uncloak_handles_missing_content(self):
        """Test uncloaking handles missing content gracefully."""
        msg = make_anthropic_msg()
        del msg.content
        result = _uncloak_anthropic_message(msg, {"<PERSON_0>": "John"})
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
