"""Test Anthropic provider integration and parameter handling.

Description:
    This test module validates the AnthropicProvider class that handles
    Anthropic-specific API integration, including parameter conversion,
    tool calling support, and message format handling.

Test Classes:
    - TestAnthropicProvider: Tests Anthropic provider functionality

Author: LLMShield by brainpolo, 2025
"""

# Standard library imports
import unittest
from unittest.mock import Mock

# Local Imports
from llmshield.providers.anthropic_provider import AnthropicProvider


class TestAnthropicProvider(unittest.TestCase):
    """Test Anthropic provider functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock standard Anthropic function
        self.mock_anthropic_func = Mock()
        self.mock_anthropic_func.__name__ = "create"
        self.mock_anthropic_func.__qualname__ = "client.messages.create"
        self.mock_anthropic_func.__module__ = "anthropic.client"

    def test_init(self):
        """Test initialization."""
        provider = AnthropicProvider(self.mock_anthropic_func)

        self.assertEqual(provider.llm_func, self.mock_anthropic_func)

    def test_prepare_single_message_params(self):
        """Test preparing single message parameters."""
        provider = AnthropicProvider(self.mock_anthropic_func)

        cloaked_text = "Hello <PERSON_0>"
        input_param = "prompt"
        stream = True
        kwargs = {
            "prompt": "Original text",
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 1024,
        }

        result = provider.prepare_single_message_params(
            cloaked_text, input_param, stream, **kwargs
        )
        prepared_params, actual_stream = result

        # Check that original parameter is removed
        self.assertNotIn("prompt", prepared_params)

        # Check messages format
        self.assertIn("messages", prepared_params)
        self.assertEqual(len(prepared_params["messages"]), 1)
        self.assertEqual(prepared_params["messages"][0]["role"], "user")
        self.assertEqual(
            prepared_params["messages"][0]["content"], cloaked_text
        )

        # Check other parameters are preserved
        self.assertEqual(prepared_params["model"], "claude-3-5-haiku-20241022")
        self.assertEqual(prepared_params["max_tokens"], 1024)

        # Check streaming is enabled
        self.assertTrue(prepared_params["stream"])
        self.assertTrue(actual_stream)

    def test_prepare_multi_message_params(self):
        """Test preparing multi-message parameters."""
        provider = AnthropicProvider(self.mock_anthropic_func)

        cloaked_messages = [
            {"role": "user", "content": "Hello <PERSON_0>"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Email me at <EMAIL_0>"},
        ]
        stream = False
        kwargs = {"model": "claude-3-5-haiku-20241022", "max_tokens": 1024}

        prepared_params, actual_stream = provider.prepare_multi_message_params(
            cloaked_messages, stream, **kwargs
        )

        # Check messages are preserved and converted
        self.assertEqual(len(prepared_params["messages"]), 3)

        # Check other parameters are preserved
        self.assertEqual(prepared_params["model"], "claude-3-5-haiku-20241022")
        self.assertEqual(prepared_params["max_tokens"], 1024)

        # Check streaming is disabled
        self.assertFalse(prepared_params["stream"])
        self.assertFalse(actual_stream)

    def test_convert_messages_with_tool_calls(self):
        """Test converting messages with tool calls to Anthropic format."""
        provider = AnthropicProvider(self.mock_anthropic_func)

        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "London"}',
                        },
                    }
                ],
            },
        ]

        converted = provider._convert_messages_to_anthropic_format(messages)

        self.assertEqual(len(converted), 2)

        # Check user message is unchanged
        self.assertEqual(converted[0]["role"], "user")
        self.assertEqual(converted[0]["content"], "What's the weather?")

        # Check assistant message with tool calls
        self.assertEqual(converted[1]["role"], "assistant")
        self.assertIsInstance(converted[1]["content"], list)

        content_blocks = converted[1]["content"]
        self.assertEqual(len(content_blocks), 1)

        tool_block = content_blocks[0]
        self.assertEqual(tool_block["type"], "tool_use")
        self.assertEqual(tool_block["id"], "call_123")
        self.assertEqual(tool_block["name"], "get_weather")
        self.assertEqual(tool_block["input"], {"location": "London"})

    def test_convert_messages_with_tool_calls_and_text(self):
        """Test converting messages with both text content and tool calls."""
        provider = AnthropicProvider(self.mock_anthropic_func)

        messages = [
            {
                "role": "assistant",
                "content": "I'll check the weather for you.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "London"}',
                        },
                    }
                ],
            },
        ]

        converted = provider._convert_messages_to_anthropic_format(messages)

        self.assertEqual(len(converted), 1)

        # Check assistant message with both text and tool calls
        self.assertEqual(converted[0]["role"], "assistant")
        self.assertIsInstance(converted[0]["content"], list)

        content_blocks = converted[0]["content"]
        self.assertEqual(len(content_blocks), 2)

        # First block should be text
        text_block = content_blocks[0]
        self.assertEqual(text_block["type"], "text")
        self.assertEqual(text_block["text"], "I'll check the weather for you.")

        # Second block should be tool use
        tool_block = content_blocks[1]
        self.assertEqual(tool_block["type"], "tool_use")
        self.assertEqual(tool_block["id"], "call_123")
        self.assertEqual(tool_block["name"], "get_weather")
        self.assertEqual(tool_block["input"], {"location": "London"})

    def test_convert_messages_with_tool_results(self):
        """Test converting tool result messages."""
        provider = AnthropicProvider(self.mock_anthropic_func)

        messages = [
            {
                "role": "user",
                "content": "15°C, Partly cloudy",
                "tool_call_id": "call_123",
            }
        ]

        converted = provider._convert_messages_to_anthropic_format(messages)

        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0]["role"], "user")
        self.assertIsInstance(converted[0]["content"], list)

        content_blocks = converted[0]["content"]
        self.assertEqual(len(content_blocks), 1)

        tool_result = content_blocks[0]
        self.assertEqual(tool_result["type"], "tool_result")
        self.assertEqual(tool_result["tool_use_id"], "call_123")
        self.assertEqual(tool_result["content"], "15°C, Partly cloudy")

    def test_parse_tool_arguments(self):
        """Test parsing tool arguments."""
        provider = AnthropicProvider(self.mock_anthropic_func)

        # Valid JSON
        args = provider._parse_tool_arguments('{"location": "London"}')
        self.assertEqual(args, {"location": "London"})

        # Empty string
        args = provider._parse_tool_arguments("")
        self.assertEqual(args, {})

        # Invalid JSON
        args = provider._parse_tool_arguments("invalid json")
        self.assertEqual(args, {})

    def test_can_handle_anthropic_functions(self):
        """Test can_handle method for Anthropic functions."""
        # Test with anthropic in module
        mock_func = Mock()
        mock_func.__name__ = "create"
        mock_func.__qualname__ = "client.messages.create"
        mock_func.__module__ = "anthropic.client"
        self.assertTrue(AnthropicProvider.can_handle(mock_func))

        # Test with claude in module
        mock_func = Mock()
        mock_func.__name__ = "create"
        mock_func.__qualname__ = "client.messages.create"
        mock_func.__module__ = "claude.api"
        self.assertTrue(AnthropicProvider.can_handle(mock_func))

        # Test with messages.create in qualname
        mock_func = Mock()
        mock_func.__name__ = "create"
        mock_func.__qualname__ = "client.messages.create"
        mock_func.__module__ = "some.module"
        self.assertTrue(AnthropicProvider.can_handle(mock_func))

    def test_can_handle_non_anthropic_functions(self):
        """Test can_handle method for non-Anthropic functions."""
        mock_func = Mock()
        mock_func.__name__ = "unknown_function"
        mock_func.__qualname__ = "some.module.unknown_function"
        mock_func.__module__ = "openai.client"
        self.assertFalse(AnthropicProvider.can_handle(mock_func))

    def test_can_handle_missing_attributes(self):
        """Test can_handle method with missing function attributes."""
        # Function with no __module__ attribute
        mock_func = Mock()
        mock_func.__name__ = "create"
        mock_func.__qualname__ = "messages.create"
        delattr(mock_func, "__module__")
        self.assertTrue(AnthropicProvider.can_handle(mock_func))

        # Function with no attributes
        mock_func = Mock()
        delattr(mock_func, "__name__")
        delattr(mock_func, "__qualname__")
        delattr(mock_func, "__module__")
        self.assertFalse(AnthropicProvider.can_handle(mock_func))


if __name__ == "__main__":
    unittest.main()
