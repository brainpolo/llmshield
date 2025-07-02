"""Test OpenAI provider integration and parameter handling.

Description:
    This test module validates the OpenAIProvider class that handles
    OpenAI-specific API integration, including parameter conversion,
    beta APIs, and streaming support.

Test Classes:
    - TestOpenAIProvider: Tests OpenAI provider functionality

Author: LLMShield by brainpolo, 2025
"""

# Standard library imports
import unittest
from unittest.mock import Mock, patch

# Local Imports
from llmshield.providers.openai_provider import OpenAIProvider


class TestOpenAIProvider(unittest.TestCase):
    """Test OpenAI provider functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock standard OpenAI function
        self.mock_standard_func = Mock()
        self.mock_standard_func.__name__ = "create"
        self.mock_standard_func.__qualname__ = "client.chat.completions.create"
        self.mock_standard_func.__module__ = "openai.client"

        # Mock beta OpenAI function
        self.mock_beta_func = Mock()
        self.mock_beta_func.__name__ = "parse"
        self.mock_beta_func.__qualname__ = "client.beta.chat.completions.parse"
        self.mock_beta_func.__module__ = "openai.beta.client"

    def test_init_standard_api(self):
        """Test initialization with standard API."""
        provider = OpenAIProvider(self.mock_standard_func)

        self.assertEqual(provider.llm_func, self.mock_standard_func)
        self.assertFalse(provider.is_beta_api)

    def test_init_beta_api(self):
        """Test initialization with beta API."""
        provider = OpenAIProvider(self.mock_beta_func)

        self.assertEqual(provider.llm_func, self.mock_beta_func)
        self.assertTrue(provider.is_beta_api)

    def test_detect_beta_api_by_qualname(self):
        """Test beta API detection by qualname."""
        mock_func = Mock()
        mock_func.__name__ = "create"
        mock_func.__qualname__ = "client.beta.chat.completions.create"
        mock_func.__module__ = "openai.client"

        provider = OpenAIProvider(mock_func)
        self.assertTrue(provider.is_beta_api)

    def test_detect_beta_api_by_module(self):
        """Test beta API detection by module."""
        mock_func = Mock()
        mock_func.__name__ = "create"
        mock_func.__qualname__ = "client.chat.completions.create"
        mock_func.__module__ = "openai.beta.client"

        provider = OpenAIProvider(mock_func)
        self.assertTrue(provider.is_beta_api)

    def test_detect_beta_api_by_name(self):
        """Test beta API detection by function name."""
        mock_func = Mock()
        mock_func.__name__ = "parse"
        mock_func.__qualname__ = "client.chat.completions.parse"
        mock_func.__module__ = "openai.client"

        provider = OpenAIProvider(mock_func)
        self.assertTrue(provider.is_beta_api)

    def test_prepare_single_message_params_standard_api(self):
        """Test preparing single message parameters for standard API."""
        provider = OpenAIProvider(self.mock_standard_func)

        cloaked_text = "Hello <PERSON_0>"
        input_param = "prompt"
        stream = True
        kwargs = {
            "prompt": "Original text",
            "model": "gpt-4",
            "temperature": 0.7,
        }

        prepared_params, actual_stream = (
            provider.prepare_single_message_params(
                cloaked_text, input_param, stream, **kwargs
            )
        )

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
        self.assertEqual(prepared_params["model"], "gpt-4")
        self.assertEqual(prepared_params["temperature"], 0.7)

        # Check streaming is enabled
        self.assertTrue(prepared_params["stream"])
        self.assertTrue(actual_stream)

    @patch("builtins.print")
    def test_prepare_single_message_params_beta_api_with_stream_warning(
        self, mock_print
    ):
        """Test preparing single message parameters for beta API.

        Validates stream warning behavior when using beta API.
        """
        provider = OpenAIProvider(self.mock_beta_func)

        cloaked_text = "Hello <PERSON_0>"
        input_param = "message"
        stream = True
        kwargs = {
            "message": "Original text",
            "model": "gpt-4",
            "response_format": {"type": "json_object"},
        }

        prepared_params, actual_stream = (
            provider.prepare_single_message_params(
                cloaked_text, input_param, stream, **kwargs
            )
        )

        # Check warning was printed
        mock_print.assert_called_once()
        self.assertIn("Warning: Beta API detected", mock_print.call_args[0][0])
        self.assertIn("streaming not supported", mock_print.call_args[0][0])

        # Check that original parameter is removed
        self.assertNotIn("message", prepared_params)

        # Check messages format
        self.assertIn("messages", prepared_params)
        self.assertEqual(
            prepared_params["messages"][0]["content"], cloaked_text
        )

        # Check stream parameter is removed and streaming is disabled
        self.assertNotIn("stream", prepared_params)
        self.assertFalse(actual_stream)

    def test_prepare_single_message_params_beta_api_no_stream(self):
        """Test preparing single message parameters for beta API.

        Validates behavior when streaming is disabled for beta API.
        """
        provider = OpenAIProvider(self.mock_beta_func)

        cloaked_text = "Hello <PERSON_0>"
        input_param = "prompt"
        stream = False
        kwargs = {"prompt": "Original text", "model": "gpt-4"}

        prepared_params, actual_stream = (
            provider.prepare_single_message_params(
                cloaked_text, input_param, stream, **kwargs
            )
        )

        # Check messages format
        self.assertEqual(
            prepared_params["messages"][0]["content"], cloaked_text
        )

        # Check stream parameter is removed
        self.assertNotIn("stream", prepared_params)
        self.assertFalse(actual_stream)

    def test_prepare_multi_message_params_standard_api(self):
        """Test preparing multi-message parameters for standard API."""
        provider = OpenAIProvider(self.mock_standard_func)

        cloaked_messages = [
            {"role": "user", "content": "Hello <PERSON_0>"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Email me at <EMAIL_0>"},
        ]
        stream = True
        kwargs = {"model": "gpt-4", "temperature": 0.5}

        prepared_params, actual_stream = provider.prepare_multi_message_params(
            cloaked_messages, stream, **kwargs
        )

        # Check messages are preserved
        self.assertEqual(prepared_params["messages"], cloaked_messages)

        # Check other parameters are preserved
        self.assertEqual(prepared_params["model"], "gpt-4")
        self.assertEqual(prepared_params["temperature"], 0.5)

        # Check streaming is enabled
        self.assertTrue(prepared_params["stream"])
        self.assertTrue(actual_stream)

    @patch("builtins.print")
    def test_prepare_multi_message_params_beta_api_with_stream_warning(
        self, mock_print
    ):
        """Test preparing multi-message parameters for beta API.

        Validates stream warning behavior for multi-message conversations.
        """
        provider = OpenAIProvider(self.mock_beta_func)

        cloaked_messages = [{"role": "user", "content": "Hello <PERSON_0>"}]
        stream = True
        kwargs = {"model": "gpt-4", "response_format": {"type": "json_object"}}

        prepared_params, actual_stream = provider.prepare_multi_message_params(
            cloaked_messages, stream, **kwargs
        )

        # Check warning was printed
        mock_print.assert_called_once()
        self.assertIn("Warning: Beta API detected", mock_print.call_args[0][0])

        # Check messages are preserved
        self.assertEqual(prepared_params["messages"], cloaked_messages)

        # Check stream parameter is removed and streaming is disabled
        self.assertNotIn("stream", prepared_params)
        self.assertFalse(actual_stream)

    def test_prepare_multi_message_params_beta_api_no_stream(self):
        """Test preparing multi-message parameters for beta API.

        Validates behavior when streaming is disabled for multi-message calls.
        """
        provider = OpenAIProvider(self.mock_beta_func)

        cloaked_messages = [{"role": "user", "content": "Hello <PERSON_0>"}]
        stream = False
        kwargs = {"model": "gpt-4"}

        prepared_params, actual_stream = provider.prepare_multi_message_params(
            cloaked_messages, stream, **kwargs
        )

        # Check messages are preserved
        self.assertEqual(prepared_params["messages"], cloaked_messages)

        # Check stream parameter is removed
        self.assertNotIn("stream", prepared_params)
        self.assertFalse(actual_stream)

    def test_can_handle_openai_standard_functions(self):
        """Test can_handle method for standard OpenAI functions."""
        # Test with chat.completions.create in qualname
        mock_func = Mock()
        mock_func.__name__ = "create"
        mock_func.__qualname__ = "client.chat.completions.create"
        self.assertTrue(OpenAIProvider.can_handle(mock_func))

        # Test with create in name
        mock_func = Mock()
        mock_func.__name__ = "create"
        mock_func.__qualname__ = "some.function"
        self.assertTrue(OpenAIProvider.can_handle(mock_func))

    def test_can_handle_openai_beta_functions(self):
        """Test can_handle method for beta OpenAI functions."""
        # Test with chat.completions.parse in qualname
        mock_func = Mock()
        mock_func.__name__ = "parse"
        mock_func.__qualname__ = "client.chat.completions.parse"
        self.assertTrue(OpenAIProvider.can_handle(mock_func))

        # Test with parse in name
        mock_func = Mock()
        mock_func.__name__ = "parse"
        mock_func.__qualname__ = "some.function"
        self.assertTrue(OpenAIProvider.can_handle(mock_func))

    def test_can_handle_non_openai_functions(self):
        """Test can_handle method for non-OpenAI functions."""
        mock_func = Mock()
        mock_func.__name__ = "unknown_function"
        mock_func.__qualname__ = "some.module.unknown_function"
        self.assertFalse(OpenAIProvider.can_handle(mock_func))

    def test_can_handle_missing_attributes(self):
        """Test can_handle method with missing function attributes."""
        # Function with no __name__ attribute
        mock_func = Mock()
        delattr(mock_func, "__name__")
        mock_func.__qualname__ = "some.function"
        self.assertFalse(OpenAIProvider.can_handle(mock_func))

        # Function with no __qualname__ attribute
        mock_func = Mock()
        mock_func.__name__ = "create"
        delattr(mock_func, "__qualname__")
        self.assertTrue(OpenAIProvider.can_handle(mock_func))

    def test_prepare_params_with_tool_calls(self):
        """Test parameter preparation preserves tool call message structure."""
        # Messages with tool calls (None content)
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
            {
                "role": "tool",
                "content": "15°C, Partly cloudy",
                "tool_call_id": "call_123",
            },
        ]

        provider = OpenAIProvider(self.mock_standard_func)

        # Test that prepare_multi_message_params preserves structure
        params, stream = provider.prepare_multi_message_params(
            cloaked_messages=messages,
            stream=False,
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        # Verify message structure is preserved
        self.assertEqual(len(params["messages"]), 3)
        self.assertIsNone(params["messages"][1]["content"])
        self.assertIn("tool_calls", params["messages"][1])
        self.assertEqual(params["messages"][2]["role"], "tool")


if __name__ == "__main__":
    unittest.main()
