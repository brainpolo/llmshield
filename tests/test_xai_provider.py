"""Tests for xAI provider, response detection, and uncloaking.

Description:
    This test module validates the XAIProvider class, xAI response
    detection functions, and xAI response uncloaking.

Author: LLMShield by brainpolo, 2025-2026
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import llmshield.providers.xai_provider as xai_mod
from llmshield.detection_utils import (
    extract_response_content,
    extract_xai_content,
    is_xai_response_like,
)
from llmshield.providers.xai_provider import XAIProvider
from llmshield.uncloak_response import _uncloak_response


# Helper to create mock xAI responses without auto-attributes
class MockXAIResponse:
    """Mock xAI response with only xAI-specific attributes."""

    def __init__(
        self,
        content="",
        tool_calls=None,
        usage=None,
    ):
        """Initialise mock xAI response."""
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage = usage or MockUsage()


class MockUsage:
    """Mock token usage."""

    def __init__(
        self,
        prompt_tokens=10,
        completion_tokens=5,
    ):
        """Initialise mock usage."""
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class MockToolCall:
    """Mock xAI tool call."""

    def __init__(self, tc_id, name, arguments):
        """Initialise mock tool call."""
        self.id = tc_id
        self.function = MockFunction(name, arguments)


class MockFunction:
    """Mock tool call function."""

    def __init__(self, name, arguments):
        """Initialise mock function."""
        self.name = name
        self.arguments = arguments


def _make_xai_func():
    """Create a mock function that looks like xai_sdk."""
    func = Mock()
    func.__name__ = "create"
    func.__qualname__ = "Chat.create"
    func.__module__ = "xai_sdk.chat"
    return func


class TestXAIProviderCanHandle(unittest.TestCase):
    """Test XAIProvider.can_handle detection."""

    def test_xai_module(self):
        """Test detection via xai_sdk in module."""
        func = _make_xai_func()
        with patch.object(xai_mod, "_HAS_XAI_SDK", True):
            self.assertTrue(XAIProvider.can_handle(func))

    def test_xai_qualname(self):
        """Test detection via xai_sdk in qualname."""
        func = Mock()
        func.__module__ = "some.module"
        func.__qualname__ = "xai_sdk.Client.chat.create"
        with patch.object(xai_mod, "_HAS_XAI_SDK", True):
            self.assertTrue(XAIProvider.can_handle(func))

    def test_non_xai_function(self):
        """Test non-xAI function is rejected."""
        func = Mock()
        func.__module__ = "openai.chat"
        func.__qualname__ = "Chat.create"
        with patch.object(xai_mod, "_HAS_XAI_SDK", True):
            self.assertFalse(XAIProvider.can_handle(func))

    def test_without_sdk_installed(self):
        """Test returns False when xai_sdk not installed."""
        func = _make_xai_func()
        with patch.object(xai_mod, "_HAS_XAI_SDK", False):
            self.assertFalse(XAIProvider.can_handle(func))


class TestXAIProviderExecute(unittest.TestCase):
    """Test XAIProvider execute methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_response = MockXAIResponse(content="Hello back")
        self.mock_chat = MagicMock()
        self.mock_chat.sample.return_value = self.mock_response

        self.mock_llm_func = Mock(return_value=self.mock_chat)
        self.mock_llm_func.__name__ = "create"
        self.mock_llm_func.__qualname__ = "Chat.create"
        self.mock_llm_func.__module__ = "xai_sdk.chat"

        self.mock_xai_chat = MagicMock()

    def _make_provider(self):
        """Create provider with mocked xai_chat."""
        provider = XAIProvider(self.mock_llm_func)
        return provider

    def test_execute_single_message(self):
        """Test single message execution."""
        provider = self._make_provider()

        with patch.object(xai_mod, "_xai_chat", self.mock_xai_chat):
            response, stream = provider.execute_single_message(
                "Hello <PERSON_0>",
                "prompt",
                False,
                model="grok-4",
            )

        self.assertEqual(response, self.mock_response)
        self.assertFalse(stream)
        self.mock_llm_func.assert_called_once_with(model="grok-4")
        self.mock_chat.sample.assert_called_once()
        self.mock_xai_chat.user.assert_called_once_with("Hello <PERSON_0>")

    def test_execute_multi_message(self):
        """Test multi-message execution."""
        provider = self._make_provider()
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
        ]

        with patch.object(xai_mod, "_xai_chat", self.mock_xai_chat):
            response, stream = provider.execute_multi_message(
                messages,
                False,
                model="grok-4",
                temperature=0.7,
            )

        self.assertEqual(response, self.mock_response)
        self.assertFalse(stream)
        self.mock_llm_func.assert_called_once_with(
            model="grok-4", temperature=0.7
        )
        self.mock_xai_chat.system.assert_called_once()
        self.assertEqual(self.mock_xai_chat.user.call_count, 2)
        self.mock_xai_chat.assistant.assert_called_once()

    def test_execute_raw(self):
        """Test raw execution without cloaking."""
        provider = self._make_provider()

        with patch.object(xai_mod, "_xai_chat", self.mock_xai_chat):
            response = provider.execute_raw(
                prompt="Hello",
                model="grok-4",
            )

        self.assertEqual(response, self.mock_response)
        self.mock_llm_func.assert_called_once_with(model="grok-4")

    def test_max_tokens_removed(self):
        """Test token limits are removed before chat.create."""
        provider = self._make_provider()

        with patch.object(xai_mod, "_xai_chat", self.mock_xai_chat):
            provider.execute_single_message(
                "Hello",
                "prompt",
                False,
                model="grok-4",
                max_tokens=1000,
            )

        call_kwargs = self.mock_llm_func.call_args[1]
        self.assertNotIn("max_tokens", call_kwargs)
        self.assertNotIn("max_completion_tokens", call_kwargs)

    def test_kwargs_passthrough(self):
        """Test extra kwargs pass through to chat.create."""
        provider = self._make_provider()

        with patch.object(xai_mod, "_xai_chat", self.mock_xai_chat):
            provider.execute_single_message(
                "Hello",
                "prompt",
                False,
                model="grok-4",
                temperature=0.5,
                tools=["web_search"],
                tool_choice="auto",
            )

        call_kwargs = self.mock_llm_func.call_args[1]
        self.assertEqual(call_kwargs["model"], "grok-4")
        self.assertEqual(call_kwargs["temperature"], 0.5)
        self.assertEqual(call_kwargs["tools"], ["web_search"])
        self.assertEqual(call_kwargs["tool_choice"], "auto")

    def test_tool_role_messages(self):
        """Test tool result messages are handled."""
        provider = self._make_provider()
        messages = [
            {"role": "user", "content": "Weather?"},
            {"role": "tool", "content": '{"temp": 20}'},
        ]

        with patch.object(xai_mod, "_xai_chat", self.mock_xai_chat):
            provider.execute_multi_message(messages, False, model="grok-4")

        self.mock_xai_chat.tool_result.assert_called_once()

    def test_none_content_handled(self):
        """Test None content defaults to empty string."""
        provider = self._make_provider()
        messages = [
            {"role": "user", "content": None},
        ]

        with patch.object(xai_mod, "_xai_chat", self.mock_xai_chat):
            provider.execute_multi_message(messages, False, model="grok-4")

        self.mock_xai_chat.user.assert_called_once_with("")

    def test_tools_conversion(self):
        """Test OpenAI-style tools are converted to xAI format."""
        provider = self._make_provider()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {"type": "object"},
                },
            },
            {"custom": True},
        ]

        with patch.object(xai_mod, "_xai_chat", self.mock_xai_chat):
            provider.execute_single_message(
                "Hello",
                "prompt",
                False,
                model="grok-4",
                tools=tools,
            )

        call_kwargs = self.mock_llm_func.call_args[1]
        converted = call_kwargs["tools"]
        self.assertEqual(len(converted), 2)
        self.assertEqual(converted[0]["function"]["name"], "search")
        self.assertIsInstance(converted[0]["function"]["parameters"], str)

    def test_list_content_converted(self):
        """Test list content is converted to string."""
        provider = self._make_provider()
        messages = [
            {
                "role": "user",
                "content": ["part1", "part2"],
            },
        ]

        with patch.object(xai_mod, "_xai_chat", self.mock_xai_chat):
            provider.execute_multi_message(messages, False, model="grok-4")

        self.mock_xai_chat.user.assert_called_once_with("['part1', 'part2']")

    def test_assistant_tool_call_with_content(self):
        """Test assistant message with content and tool_calls."""
        provider = self._make_provider()
        mock_pb2 = MagicMock()

        messages = [
            {
                "role": "assistant",
                "content": "Calling tool",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "search",
                            "arguments": {"q": "test"},
                        },
                    }
                ],
            },
        ]

        with (
            patch.object(xai_mod, "_xai_chat", self.mock_xai_chat),
            patch.object(xai_mod, "_xai_pb2", mock_pb2),
        ):
            provider.execute_multi_message(messages, False, model="grok-4")

        # Content part should be created
        mock_pb2.Content.assert_called_once_with(text="Calling tool")
        # Dict args should be JSON-serialized
        mock_pb2.FunctionCall.assert_called_once()
        fc_kwargs = mock_pb2.FunctionCall.call_args[1]
        self.assertEqual(fc_kwargs["arguments"], '{"q": "test"}')

    def test_streaming_returns_generator(self):
        """Test streaming returns a generator."""
        mock_chunk = Mock()
        mock_chunk.content = "Hello"
        self.mock_chat.stream.return_value = [
            (self.mock_response, mock_chunk),
        ]

        provider = self._make_provider()

        with patch.object(xai_mod, "_xai_chat", self.mock_xai_chat):
            response, stream = provider.execute_single_message(
                "Hello", "prompt", True, model="grok-4"
            )

        self.assertTrue(stream)
        chunks = list(response)
        self.assertEqual(chunks, ["Hello"])


class TestXAIResponseDetection(unittest.TestCase):
    """Test xAI response detection functions."""

    def test_xai_response_detected(self):
        """Test xAI response is detected correctly."""
        resp = MockXAIResponse(content="Hello")
        self.assertTrue(is_xai_response_like(resp))

    def test_chatcompletion_not_detected_as_xai(self):
        """Test ChatCompletion is not detected as xAI."""
        resp = Mock()
        resp.choices = []
        resp.model = "gpt-4"
        resp.content = "Hello"
        resp.usage = Mock()
        self.assertFalse(is_xai_response_like(resp))

    def test_anthropic_not_detected_as_xai(self):
        """Test Anthropic response is not detected as xAI."""
        resp = Mock()
        resp.content = "Hello"
        resp.model = "claude-3"
        resp.role = "assistant"
        resp.usage = Mock()
        self.assertFalse(is_xai_response_like(resp))

    def test_string_not_detected_as_xai(self):
        """Test plain string is not detected as xAI."""
        self.assertFalse(is_xai_response_like("hello"))

    def test_extract_xai_content(self):
        """Test content extraction from xAI response."""
        resp = MockXAIResponse(content="Hello world")
        self.assertEqual(extract_xai_content(resp), "Hello world")

    def test_extract_xai_content_non_xai(self):
        """Test extraction returns None for non-xAI."""
        self.assertIsNone(extract_xai_content("hello"))

    def test_extract_response_content_xai(self):
        """Test universal extractor handles xAI."""
        resp = MockXAIResponse(content="Hello world")
        self.assertEqual(extract_response_content(resp), "Hello world")


class TestXAIResponseUncloaking(unittest.TestCase):
    """Test xAI response uncloaking."""

    def test_uncloak_content(self):
        """Test content is uncloaked."""
        resp = MockXAIResponse(content="Hello <PERSON_0>")
        entity_map = {"<PERSON_0>": "John"}

        result = _uncloak_response(resp, entity_map)
        self.assertEqual(result.content, "Hello John")

    def test_uncloak_tool_calls(self):
        """Test tool call arguments are uncloaked."""
        tc = MockToolCall(
            "call_1",
            "send_email",
            '{"to": "<EMAIL_0>"}',
        )
        resp = MockXAIResponse(
            content="Sending email",
            tool_calls=[tc],
        )
        entity_map = {"<EMAIL_0>": "john@example.com"}

        result = _uncloak_response(resp, entity_map)
        self.assertEqual(
            result.tool_calls[0].function.arguments,
            '{"to": "john@example.com"}',
        )

    def test_uncloak_preserves_original(self):
        """Test original response is not modified."""
        resp = MockXAIResponse(content="Hello <PERSON_0>")
        entity_map = {"<PERSON_0>": "John"}

        _uncloak_response(resp, entity_map)
        self.assertEqual(resp.content, "Hello <PERSON_0>")

    def test_uncloak_none_content(self):
        """Test None content is handled gracefully."""
        resp = MockXAIResponse(content=None)
        entity_map = {"<PERSON_0>": "John"}

        result = _uncloak_response(resp, entity_map)
        self.assertIsNone(result.content)

    def test_uncloak_empty_tool_calls(self):
        """Test empty tool calls list is handled."""
        resp = MockXAIResponse(
            content="Hello <PERSON_0>",
            tool_calls=[],
        )
        entity_map = {"<PERSON_0>": "John"}

        result = _uncloak_response(resp, entity_map)
        self.assertEqual(result.content, "Hello John")
        self.assertEqual(result.tool_calls, [])


if __name__ == "__main__":
    unittest.main()
