"""Tests for Cohere provider, response detection, and uncloaking.

Description:
    This test module validates the CohereProvider class, Cohere
    response detection functions, and Cohere response uncloaking.

Author: LLMShield by brainpolo, 2025-2026
"""

import unittest
from unittest.mock import Mock, patch

import llmshield.providers.cohere_provider as cohere_mod
from llmshield.detection_utils import (
    extract_cohere_content,
    extract_response_content,
    is_cohere_response_like,
)
from llmshield.providers.cohere_provider import CohereProvider
from llmshield.uncloak_response import _uncloak_response

# Helper mock classes (plain classes to avoid Mock auto-attributes)


class MockCohereContentBlock:
    """Mock Cohere content block."""

    def __init__(self, text=None, type="text"):
        """Initialise mock content block."""
        self.type = type
        self.text = text


class MockCohereFunction:
    """Mock Cohere function call."""

    def __init__(self, name, arguments):
        """Initialise mock function."""
        self.name = name
        self.arguments = arguments


class MockCohereToolCall:
    """Mock Cohere tool call."""

    def __init__(self, id, type, function):
        """Initialise mock tool call."""
        self.id = id
        self.type = type
        self.function = function


class MockCohereMessage:
    """Mock Cohere assistant message."""

    def __init__(
        self,
        role="assistant",
        content=None,
        tool_calls=None,
    ):
        """Initialise mock message."""
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class MockCohereTokens:
    """Mock Cohere token counts."""

    def __init__(
        self,
        input_tokens=10,
        output_tokens=5,
    ):
        """Initialise mock tokens."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockCohereUsage:
    """Mock Cohere usage object."""

    def __init__(self, tokens=None):
        """Initialise mock usage."""
        self.tokens = tokens or MockCohereTokens()


class MockCohereResponse:
    """Mock Cohere chat response."""

    def __init__(
        self,
        id="resp-123",
        finish_reason="COMPLETE",
        message=None,
        usage=None,
    ):
        """Initialise mock Cohere response."""
        self.id = id
        self.finish_reason = finish_reason
        self.message = message or MockCohereMessage()
        self.usage = usage or MockCohereUsage()


def _make_cohere_func():
    """Create a mock function that looks like cohere chat."""
    func = Mock()
    func.__name__ = "chat"
    func.__qualname__ = "ClientV2.chat"
    func.__module__ = "cohere.v2.client"
    return func


def _make_simple_response(text="Hello back"):
    """Create a simple mock Cohere response."""
    return MockCohereResponse(
        message=MockCohereMessage(content=[MockCohereContentBlock(text=text)])
    )


class TestCohereProviderCanHandle(unittest.TestCase):
    """Test CohereProvider.can_handle detection."""

    def test_cohere_module(self):
        """Test detection via cohere in module."""
        func = _make_cohere_func()
        with patch.object(cohere_mod, "_HAS_COHERE", True):
            self.assertTrue(CohereProvider.can_handle(func))

    def test_cohere_submodule(self):
        """Test detection via cohere submodule."""
        func = Mock()
        func.__module__ = "cohere.v2.client"
        func.__qualname__ = "ClientV2.chat"
        with patch.object(cohere_mod, "_HAS_COHERE", True):
            self.assertTrue(CohereProvider.can_handle(func))

    def test_non_cohere_function(self):
        """Test non-Cohere function is rejected."""
        func = Mock()
        func.__module__ = "openai.chat"
        func.__qualname__ = "Chat.create"
        with patch.object(cohere_mod, "_HAS_COHERE", True):
            self.assertFalse(CohereProvider.can_handle(func))

    def test_without_sdk_installed(self):
        """Test returns False when cohere not installed."""
        func = _make_cohere_func()
        with patch.object(cohere_mod, "_HAS_COHERE", False):
            self.assertFalse(CohereProvider.can_handle(func))

    def test_coherent_not_matched(self):
        """Test 'coherent' in module does not match."""
        func = Mock()
        func.__module__ = "myapp.coherent_utils"
        func.__qualname__ = "enforce_coherence"
        with patch.object(cohere_mod, "_HAS_COHERE", True):
            self.assertFalse(CohereProvider.can_handle(func))


class TestCohereProviderExecute(unittest.TestCase):
    """Test CohereProvider execute methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_response = _make_simple_response()
        self.mock_llm_func = Mock(return_value=self.mock_response)
        self.mock_llm_func.__name__ = "chat"
        self.mock_llm_func.__qualname__ = "ClientV2.chat"
        self.mock_llm_func.__module__ = "cohere.v2.client"

    def test_execute_single_message(self):
        """Test single message execution."""
        provider = CohereProvider(self.mock_llm_func)
        response, stream = provider.execute_single_message(
            "Hello <PERSON_0>",
            "prompt",
            False,
            model="command-a-03-2025",
        )
        self.assertEqual(response, self.mock_response)
        self.assertFalse(stream)
        self.mock_llm_func.assert_called_once()
        call_kwargs = self.mock_llm_func.call_args[1]
        self.assertEqual(call_kwargs["model"], "command-a-03-2025")
        self.assertEqual(
            call_kwargs["messages"],
            [{"role": "user", "content": "Hello <PERSON_0>"}],
        )

    def test_execute_multi_message(self):
        """Test multi-message execution."""
        provider = CohereProvider(self.mock_llm_func)
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
        ]
        response, stream = provider.execute_multi_message(
            messages,
            False,
            model="command-a-03-2025",
        )
        self.assertEqual(response, self.mock_response)
        self.assertFalse(stream)
        call_kwargs = self.mock_llm_func.call_args[1]
        self.assertEqual(call_kwargs["messages"], messages)

    def test_execute_raw(self):
        """Test raw execution without cloaking."""
        provider = CohereProvider(self.mock_llm_func)
        response = provider.execute_raw(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "Hello"}],
        )
        self.assertEqual(response, self.mock_response)
        self.mock_llm_func.assert_called_once()

    def test_execute_raw_strips_stream_kwarg(self):
        """Test execute_raw removes stream before calling API."""
        provider = CohereProvider(self.mock_llm_func)
        provider.execute_raw(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )
        call_kwargs = self.mock_llm_func.call_args[1]
        self.assertNotIn("stream", call_kwargs)
        self.assertNotIn("prompt", call_kwargs)

    def test_execute_raw_builds_messages_from_prompt(self):
        """Test execute_raw converts prompt kwarg to messages."""
        provider = CohereProvider(self.mock_llm_func)
        provider.execute_raw(
            model="command-a-03-2025",
            prompt="Hello world",
        )
        call_kwargs = self.mock_llm_func.call_args[1]
        self.assertEqual(
            call_kwargs["messages"],
            [{"role": "user", "content": "Hello world"}],
        )
        self.assertNotIn("prompt", call_kwargs)

    def test_streaming_returns_generator(self):
        """Test streaming returns a generator."""
        mock_event = Mock()
        mock_event.type = "content-delta"
        mock_event.delta.message.content.text = "Hello"
        mock_stream_func = Mock(return_value=[mock_event])

        mock_client = Mock()
        mock_client.chat_stream = mock_stream_func
        self.mock_llm_func.__self__ = mock_client

        provider = CohereProvider(self.mock_llm_func)
        response, stream = provider.execute_single_message(
            "Hello",
            "prompt",
            True,
            model="command-a-03-2025",
        )
        self.assertTrue(stream)
        chunks = list(response)
        self.assertEqual(chunks, ["Hello"])

    def test_streaming_fallback_no_stream_func(self):
        """Test streaming falls back when no stream func."""
        provider = CohereProvider(self.mock_llm_func)
        response, stream = provider.execute_single_message(
            "Hello",
            "prompt",
            True,
            model="command-a-03-2025",
        )
        self.assertFalse(stream)
        self.assertEqual(response, self.mock_response)

    def test_kwargs_passthrough(self):
        """Test extra kwargs pass through to API call."""
        provider = CohereProvider(self.mock_llm_func)
        provider.execute_single_message(
            "Hello",
            "prompt",
            False,
            model="command-a-03-2025",
            temperature=0.5,
            max_tokens=100,
        )
        call_kwargs = self.mock_llm_func.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.5)
        self.assertEqual(call_kwargs["max_tokens"], 100)

    def test_stream_param_not_passed_to_api(self):
        """Test stream param is removed before API call."""
        provider = CohereProvider(self.mock_llm_func)
        provider.execute_single_message(
            "Hello",
            "prompt",
            False,
            model="command-a-03-2025",
        )
        call_kwargs = self.mock_llm_func.call_args[1]
        self.assertNotIn("stream", call_kwargs)

    def test_streaming_skips_non_content_events(self):
        """Test streaming skips non-content-delta events."""
        events = [
            Mock(type="message-start"),
            Mock(
                type="content-delta",
                delta=Mock(message=Mock(content=Mock(text="Hello"))),
            ),
            Mock(type="message-end"),
        ]
        mock_stream_func = Mock(return_value=events)
        mock_client = Mock()
        mock_client.chat_stream = mock_stream_func
        self.mock_llm_func.__self__ = mock_client

        provider = CohereProvider(self.mock_llm_func)
        response, stream = provider.execute_multi_message(
            [{"role": "user", "content": "Hi"}],
            True,
            model="command-a-03-2025",
        )
        self.assertTrue(stream)
        chunks = list(response)
        self.assertEqual(chunks, ["Hello"])

    def test_stream_func_not_callable(self):
        """Test _get_stream_func returns None for non-callable."""
        mock_client = Mock()
        mock_client.chat_stream = "not_callable"
        self.mock_llm_func.__self__ = mock_client

        provider = CohereProvider(self.mock_llm_func)
        self.assertIsNone(provider._get_stream_func())

    def test_closure_discovery_finds_stream_func(self):
        """Test closure-walk discovery when __self__ is absent."""
        mock_stream = Mock(return_value=[])

        class FakeClient:
            """Fake client with chat_stream as a real method."""

            def chat(self, **kwargs):
                """Fake chat method."""
                return kwargs

            def chat_stream(self, **kwargs):
                """Fake chat_stream method."""
                return mock_stream(**kwargs)

        client = FakeClient()
        # Grab the bound method (has __self__)
        bound_chat = client.chat

        # Simulate SDK decorator: outer closes over inner,
        # inner closes over the bound method.
        def inner(**kwargs):
            return bound_chat(**kwargs)

        inner.__module__ = "cohere.v2.client"

        def outer(**kwargs):
            return inner(**kwargs)

        outer.__module__ = "cohere.client"

        provider = CohereProvider(outer)
        self.assertIsNotNone(provider._get_stream_func())

    def test_closure_walk_returns_none_when_no_client(self):
        """Test closure walk returns None for plain functions."""

        def plain_func(**kwargs):
            return kwargs

        plain_func.__module__ = "cohere.v2.client"
        provider = CohereProvider(plain_func)
        self.assertIsNone(provider._get_stream_func())

    def test_prepare_does_not_set_stream_param(self):
        """Test prepare methods do not set stream in output."""
        provider = CohereProvider(self.mock_llm_func)
        params, _ = provider.prepare_single_message_params(
            "Hello", "prompt", True, model="cmd"
        )
        self.assertNotIn("stream", params)
        params, _ = provider.prepare_multi_message_params(
            [{"role": "user", "content": "Hi"}],
            True,
            model="cmd",
        )
        self.assertNotIn("stream", params)


class TestCohereResponseDetection(unittest.TestCase):
    """Test Cohere response detection functions."""

    def test_cohere_response_detected(self):
        """Test Cohere response is detected correctly."""
        resp = _make_simple_response()
        self.assertTrue(is_cohere_response_like(resp))

    def test_chatcompletion_not_detected(self):
        """Test ChatCompletion is not detected as Cohere."""
        resp = type(
            "ChatCompletion",
            (),
            {
                "choices": [],
                "model": "gpt-4",
                "message": None,
                "finish_reason": "stop",
            },
        )()
        self.assertFalse(is_cohere_response_like(resp))

    def test_anthropic_not_detected(self):
        """Test Anthropic response not detected as Cohere."""
        resp = type(
            "AnthropicMsg",
            (),
            {
                "content": "Hello",
                "model": "claude-3",
                "role": "assistant",
            },
        )()
        self.assertFalse(is_cohere_response_like(resp))

    def test_google_not_detected(self):
        """Test Google response not detected as Cohere."""
        resp = type(
            "GoogleResp",
            (),
            {
                "candidates": [],
                "usage_metadata": None,
                "message": None,
                "finish_reason": "STOP",
            },
        )()
        self.assertFalse(is_cohere_response_like(resp))

    def test_xai_not_detected(self):
        """Test xAI response not detected as Cohere."""
        resp = type(
            "XAIResp",
            (),
            {"content": "Hello", "usage": None},
        )()
        self.assertFalse(is_cohere_response_like(resp))

    def test_string_not_detected(self):
        """Test plain string not detected as Cohere."""
        self.assertFalse(is_cohere_response_like("hello"))

    def test_extract_cohere_content(self):
        """Test content extraction from Cohere response."""
        resp = _make_simple_response("Hello world")
        self.assertEqual(extract_cohere_content(resp), "Hello world")

    def test_extract_cohere_content_multi_block(self):
        """Test extraction joins all text blocks."""
        blocks = [
            MockCohereContentBlock(text="Hello"),
            MockCohereContentBlock(text="world"),
        ]
        msg = MockCohereMessage(content=blocks)
        resp = MockCohereResponse(message=msg)
        self.assertEqual(extract_cohere_content(resp), "Hello world")

    def test_extract_cohere_content_skips_non_string(self):
        """Test extraction skips non-string text values."""
        block = MockCohereContentBlock(text=123)
        msg = MockCohereMessage(content=[block])
        resp = MockCohereResponse(message=msg)
        self.assertIsNone(extract_cohere_content(resp))

    def test_extract_cohere_content_non_cohere(self):
        """Test extraction returns None for non-Cohere."""
        self.assertIsNone(extract_cohere_content("hello"))

    def test_extract_cohere_content_none_message(self):
        """Test extraction with None message."""
        resp = MockCohereResponse(message=None)
        resp.message = None
        self.assertIsNone(extract_cohere_content(resp))

    def test_extract_response_content_cohere(self):
        """Test universal extractor handles Cohere."""
        resp = _make_simple_response("Hello world")
        self.assertEqual(extract_response_content(resp), "Hello world")


class TestCohereResponseUncloaking(unittest.TestCase):
    """Test Cohere response uncloaking."""

    def test_uncloak_text_content(self):
        """Test text content is uncloaked."""
        resp = _make_simple_response("Hello <PERSON_0>")
        entity_map = {"<PERSON_0>": "John"}

        result = _uncloak_response(resp, entity_map)
        block = result.message.content[0]
        self.assertEqual(block.text, "Hello John")

    def test_uncloak_string_content(self):
        """Test string content (not list) is uncloaked."""
        msg = MockCohereMessage(content="Hello <PERSON_0>")
        resp = MockCohereResponse(message=msg)
        entity_map = {"<PERSON_0>": "John"}

        result = _uncloak_response(resp, entity_map)
        self.assertEqual(result.message.content, "Hello John")

    def test_uncloak_tool_call_args(self):
        """Test tool call arguments are uncloaked."""
        tc = MockCohereToolCall(
            id="call_1",
            type="function",
            function=MockCohereFunction(
                name="send_email",
                arguments='{"to": "<EMAIL_0>"}',
            ),
        )
        msg = MockCohereMessage(
            content=[MockCohereContentBlock(text=None)],
            tool_calls=[tc],
        )
        resp = MockCohereResponse(message=msg)
        entity_map = {"<EMAIL_0>": "john@example.com"}

        result = _uncloak_response(resp, entity_map)
        args = result.message.tool_calls[0].function.arguments
        self.assertIn("john@example.com", args)

    def test_uncloak_preserves_original(self):
        """Test original response is not modified."""
        resp = _make_simple_response("Hello <PERSON_0>")
        entity_map = {"<PERSON_0>": "John"}

        _uncloak_response(resp, entity_map)
        original = resp.message.content[0].text
        self.assertEqual(original, "Hello <PERSON_0>")

    def test_uncloak_none_content(self):
        """Test None content is handled gracefully."""
        msg = MockCohereMessage(content=None)
        resp = MockCohereResponse(message=msg)
        entity_map = {"<PERSON_0>": "John"}

        result = _uncloak_response(resp, entity_map)
        self.assertIsNone(result.message.content)

    def test_uncloak_empty_content_list(self):
        """Test empty content list is handled."""
        msg = MockCohereMessage(content=[])
        resp = MockCohereResponse(message=msg)
        entity_map = {"<PERSON_0>": "John"}

        result = _uncloak_response(resp, entity_map)
        self.assertEqual(result.message.content, [])

    def test_uncloak_none_message(self):
        """Test None message is handled gracefully."""
        resp = MockCohereResponse()
        resp.message = None
        entity_map = {"<PERSON_0>": "John"}

        result = _uncloak_response(resp, entity_map)
        self.assertIsNone(result.message)

    def test_uncloak_empty_tool_calls(self):
        """Test empty tool_calls list is handled."""
        msg = MockCohereMessage(
            content=[MockCohereContentBlock(text="Hello")],
            tool_calls=[],
        )
        resp = MockCohereResponse(message=msg)
        entity_map = {"<PERSON_0>": "John"}

        result = _uncloak_response(resp, entity_map)
        self.assertEqual(result.message.tool_calls, [])

    def test_uncloak_none_tool_calls(self):
        """Test None tool_calls is handled."""
        msg = MockCohereMessage(
            content=[MockCohereContentBlock(text="Hello")],
            tool_calls=None,
        )
        resp = MockCohereResponse(message=msg)
        entity_map = {"<PERSON_0>": "John"}

        result = _uncloak_response(resp, entity_map)
        self.assertIsNone(result.message.tool_calls)

    def test_uncloak_multiple_content_blocks(self):
        """Test multiple content blocks are all uncloaked."""
        blocks = [
            MockCohereContentBlock(text="Hello <PERSON_0>"),
            MockCohereContentBlock(text="From <PLACE_0>"),
        ]
        msg = MockCohereMessage(content=blocks)
        resp = MockCohereResponse(message=msg)
        entity_map = {
            "<PERSON_0>": "John",
            "<PLACE_0>": "London",
        }

        result = _uncloak_response(resp, entity_map)
        parts = result.message.content
        self.assertEqual(parts[0].text, "Hello John")
        self.assertEqual(parts[1].text, "From London")


if __name__ == "__main__":
    unittest.main()
