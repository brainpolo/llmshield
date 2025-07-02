"""Test Anthropic tool result handling in message validation and processing.

Description:
    This test module validates proper handling of Anthropic tool result
    messages that contain list-formatted content blocks, ensuring validation
    and caching work correctly.

Test Classes:
    - TestAnthropicToolResults: Tests for tool result message handling

Author: LLMShield by brainpolo, 2025
"""

import unittest
from unittest.mock import Mock

from llmshield import LLMShield
from llmshield.error_handling import validate_prompt_input
from llmshield.utils import conversation_hash


class TestAnthropicToolResults(unittest.TestCase):
    """Test Anthropic tool result message handling."""

    def test_validate_messages_with_list_content(self):
        """Test validation accepts list content for tool results."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": "15°C, Partly cloudy",
                    }
                ],
            },
        ]

        # Should not raise ValidationError
        validate_prompt_input(messages=messages)

    def test_ask_with_tool_result_messages(self):
        """Test ask method handles tool result messages with list content."""
        # Mock LLM function
        mock_llm = Mock()
        mock_llm.return_value = "The weather is 15°C and partly cloudy."

        shield = LLMShield(llm_func=mock_llm)

        messages = [
            {"role": "user", "content": "What's the weather in London?"},
            {"role": "assistant", "content": "I'll check the weather."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": "15°C, Partly cloudy in London",
                    }
                ],
            },
        ]

        shield.ask(messages=messages)

        # Verify the mock was called
        self.assertTrue(mock_llm.called)

        # Check that messages were properly processed
        call_args = mock_llm.call_args[1]
        self.assertIn("messages", call_args)
        processed_messages = call_args["messages"]

        # Tool result message should be preserved
        self.assertEqual(len(processed_messages), 3)
        self.assertEqual(
            processed_messages[2]["content"], messages[2]["content"]
        )

    def test_conversation_hash_with_list_content(self):
        """Test conversation_hash handles list content correctly."""
        # Single message with list content
        msg_with_list = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_123",
                    "content": "Result data",
                }
            ],
        }

        hash1 = conversation_hash(msg_with_list)
        self.assertIsInstance(hash1, int)

        # Same content should produce same hash
        msg_with_list_copy = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_123",
                    "content": "Result data",
                }
            ],
        }
        hash2 = conversation_hash(msg_with_list_copy)
        self.assertEqual(hash1, hash2)

        # Different content should produce different hash
        msg_different = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_456",
                    "content": "Different data",
                }
            ],
        }
        hash3 = conversation_hash(msg_different)
        self.assertNotEqual(hash1, hash3)

    def test_conversation_hash_list_of_messages_with_list_content(self):
        """Test conversation_hash with list messages having list content."""
        messages = [
            {"role": "user", "content": "Test message"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "123",
                        "content": "data",
                    }
                ],
            },
            {"role": "assistant", "content": "Response"},
        ]

        hash_value = conversation_hash(messages)
        self.assertIsInstance(hash_value, int)

    def test_cloaking_skips_list_content(self):
        """Test that cloaking skips messages with list content."""
        shield = LLMShield()

        # Message with list content should not be cloaked
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "123",
                    "content": "data",
                }
            ],
        }

        cloaked_msg = shield._cloak_message(msg, {})

        # Content should remain unchanged
        self.assertEqual(cloaked_msg["content"], msg["content"])
        self.assertEqual(cloaked_msg["role"], msg["role"])

    def test_ask_caching_with_tool_results(self):
        """Test that caching works correctly with tool result messages."""
        mock_llm = Mock()
        mock_llm.return_value = "Response"

        shield = LLMShield(llm_func=mock_llm)

        # First conversation with tool results
        messages1 = [
            {"role": "user", "content": "Check weather in London"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "123",
                        "content": "15°C",
                    }
                ],
            },
            {"role": "user", "content": "What about New York?"},
        ]

        shield.ask(messages=messages1)

        # Second conversation with same prefix should use cache
        messages2 = [
            {"role": "user", "content": "Check weather in London"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "123",
                        "content": "15°C",
                    }
                ],
            },
            {"role": "user", "content": "What about Paris?"},  # Different
        ]

        shield.ask(messages=messages2)

        # Both calls should have been made
        self.assertEqual(mock_llm.call_count, 2)


if __name__ == "__main__":
    unittest.main()
