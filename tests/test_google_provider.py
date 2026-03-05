"""Tests for Google provider, response detection, and uncloaking.

Description:
    This test module validates the GoogleProvider class, Google
    response detection functions, and Google response uncloaking.

Author: LLMShield by brainpolo, 2025-2026
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import llmshield.providers.google_provider as google_mod
from llmshield.detection_utils import (
    extract_google_content,
    extract_response_content,
    is_google_response_like,
)
from llmshield.providers.google_provider import GoogleProvider
from llmshield.uncloak_response import _uncloak_response

# Helper mock classes (plain classes to avoid Mock auto-attributes)


class MockPart:
    """Mock Google Part object."""

    def __init__(
        self,
        text=None,
        thought=False,
        function_call=None,
    ):
        """Initialise mock part."""
        self.text = text
        self.thought = thought
        self.function_call = function_call


class MockFunctionCall:
    """Mock Google function call."""

    def __init__(self, name, args):
        """Initialise mock function call."""
        self.name = name
        self.args = args


class MockContent:
    """Mock Google Content object."""

    def __init__(self, role="model", parts=None):
        """Initialise mock content."""
        self.role = role
        self.parts = parts or []


class MockCandidate:
    """Mock Google Candidate object."""

    def __init__(
        self,
        content=None,
        finish_reason="STOP",
    ):
        """Initialise mock candidate."""
        self.content = content
        self.finish_reason = finish_reason


class MockUsageMetadata:
    """Mock Google usage metadata."""

    def __init__(
        self,
        prompt_token_count=10,
        candidates_token_count=5,
    ):
        """Initialise mock usage metadata."""
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count


class MockGoogleResponse:
    """Mock Google GenerateContentResponse."""

    def __init__(
        self,
        candidates=None,
        usage_metadata=None,
    ):
        """Initialise mock Google response."""
        self.candidates = candidates or []
        self.usage_metadata = usage_metadata or MockUsageMetadata()

    @property
    def text(self):
        """Return text from first candidate's parts."""
        if not self.candidates:
            return None
        parts = self.candidates[0].content.parts
        for part in parts:
            if part.text and not part.thought:
                return part.text
        return None


def _make_google_func():
    """Create a mock function that looks like google.genai."""
    func = Mock()
    func.__name__ = "generate_content"
    func.__qualname__ = "Models.generate_content"
    func.__module__ = "google.genai.models"
    return func


def _make_simple_response(text="Hello back"):
    """Create a simple mock Google response."""
    return MockGoogleResponse(
        candidates=[
            MockCandidate(
                content=MockContent(
                    role="model",
                    parts=[MockPart(text=text)],
                )
            )
        ]
    )


class TestGoogleProviderCanHandle(unittest.TestCase):
    """Test GoogleProvider.can_handle detection."""

    def test_google_genai_module(self):
        """Test detection via google.genai in module."""
        func = _make_google_func()
        with patch.object(google_mod, "_HAS_GOOGLE_GENAI", True):
            self.assertTrue(GoogleProvider.can_handle(func))

    def test_google_genai_submodule(self):
        """Test detection via google.genai submodule."""
        func = Mock()
        func.__module__ = "google.genai._api_client"
        func.__qualname__ = "Models.generate_content"
        with patch.object(google_mod, "_HAS_GOOGLE_GENAI", True):
            self.assertTrue(GoogleProvider.can_handle(func))

    def test_non_google_function(self):
        """Test non-Google function is rejected."""
        func = Mock()
        func.__module__ = "openai.chat"
        func.__qualname__ = "Chat.create"
        with patch.object(google_mod, "_HAS_GOOGLE_GENAI", True):
            self.assertFalse(GoogleProvider.can_handle(func))

    def test_without_sdk_installed(self):
        """Test returns False when google-genai not installed."""
        func = _make_google_func()
        with patch.object(google_mod, "_HAS_GOOGLE_GENAI", False):
            self.assertFalse(GoogleProvider.can_handle(func))


class TestGoogleProviderExecute(unittest.TestCase):
    """Test GoogleProvider execute methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_response = _make_simple_response()
        self.mock_llm_func = Mock(return_value=self.mock_response)
        self.mock_llm_func.__name__ = "generate_content"
        self.mock_llm_func.__qualname__ = "Models.generate_content"
        self.mock_llm_func.__module__ = "google.genai.models"

        self.mock_types = MagicMock()

    def _make_provider(self):
        """Create provider with mocked google_types."""
        return GoogleProvider(self.mock_llm_func)

    def test_execute_single_message(self):
        """Test single message execution."""
        provider = self._make_provider()

        with patch.object(
            google_mod,
            "_google_types",
            self.mock_types,
        ):
            response, stream = provider.execute_single_message(
                "Hello <PERSON_0>",
                "prompt",
                False,
                model="gemini-2.5-flash",
            )

        self.assertEqual(response, self.mock_response)
        self.assertFalse(stream)
        self.mock_llm_func.assert_called_once()
        call_kwargs = self.mock_llm_func.call_args[1]
        self.assertEqual(call_kwargs["model"], "gemini-2.5-flash")
        self.assertIn("contents", call_kwargs)
        self.mock_types.Part.from_text.assert_called_once_with(
            text="Hello <PERSON_0>"
        )

    def test_execute_multi_message(self):
        """Test multi-message execution."""
        provider = self._make_provider()
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
        ]

        with patch.object(
            google_mod,
            "_google_types",
            self.mock_types,
        ):
            response, stream = provider.execute_multi_message(
                messages,
                False,
                model="gemini-2.5-flash",
                temperature=0.7,
            )

        self.assertEqual(response, self.mock_response)
        self.assertFalse(stream)
        # System goes to config, not Content
        # user + model + user = 3 Content calls
        self.assertEqual(self.mock_types.Content.call_count, 3)
        # Config should have system_instruction + temp
        self.mock_types.GenerateContentConfig.assert_called_once()
        config_kw = self.mock_types.GenerateContentConfig.call_args[1]
        self.assertEqual(config_kw["system_instruction"], "Be helpful")
        self.assertEqual(config_kw["temperature"], 0.7)

    def test_execute_raw(self):
        """Test raw execution without cloaking."""
        provider = self._make_provider()

        with patch.object(
            google_mod,
            "_google_types",
            self.mock_types,
        ):
            response = provider.execute_raw(
                prompt="Hello",
                model="gemini-2.5-flash",
            )

        self.assertEqual(response, self.mock_response)
        self.mock_llm_func.assert_called_once()

    def test_max_tokens_translation(self):
        """Test max_tokens translates to max_output_tokens."""
        provider = self._make_provider()

        with patch.object(
            google_mod,
            "_google_types",
            self.mock_types,
        ):
            provider.execute_single_message(
                "Hello",
                "prompt",
                False,
                model="gemini-2.5-flash",
                max_tokens=1000,
            )

        config_kw = self.mock_types.GenerateContentConfig.call_args[1]
        self.assertNotIn("max_tokens", config_kw)
        self.assertEqual(config_kw["max_output_tokens"], 1000)

    def test_config_passthrough(self):
        """Test user-provided config is used as-is."""
        provider = self._make_provider()
        custom_config = MagicMock()

        with patch.object(
            google_mod,
            "_google_types",
            self.mock_types,
        ):
            provider.execute_single_message(
                "Hello",
                "prompt",
                False,
                model="gemini-2.5-flash",
                config=custom_config,
            )

        call_kwargs = self.mock_llm_func.call_args[1]
        self.assertEqual(call_kwargs["config"], custom_config)
        # GenerateContentConfig should NOT be called
        self.mock_types.GenerateContentConfig.assert_not_called()

    def test_assistant_role_mapped_to_model(self):
        """Test assistant role maps to 'model'."""
        provider = self._make_provider()
        messages = [
            {"role": "assistant", "content": "Hi"},
        ]

        with patch.object(
            google_mod,
            "_google_types",
            self.mock_types,
        ):
            provider.execute_multi_message(
                messages,
                False,
                model="gemini-2.5-flash",
            )

        content_kw = self.mock_types.Content.call_args[1]
        self.assertEqual(content_kw["role"], "model")

    def test_none_content_handled(self):
        """Test None content defaults to empty string."""
        provider = self._make_provider()
        messages = [
            {"role": "user", "content": None},
        ]

        with patch.object(
            google_mod,
            "_google_types",
            self.mock_types,
        ):
            provider.execute_multi_message(
                messages,
                False,
                model="gemini-2.5-flash",
            )

        self.mock_types.Part.from_text.assert_called_once_with(text="")

    def test_streaming_returns_generator(self):
        """Test streaming returns a generator."""
        mock_chunk = Mock()
        mock_chunk.text = "Hello"
        mock_stream_func = Mock(return_value=[mock_chunk])

        mock_models = Mock()
        mock_models.generate_content_stream = mock_stream_func
        self.mock_llm_func.__self__ = mock_models

        provider = self._make_provider()

        with patch.object(
            google_mod,
            "_google_types",
            self.mock_types,
        ):
            response, stream = provider.execute_single_message(
                "Hello",
                "prompt",
                True,
                model="gemini-2.5-flash",
            )

        self.assertTrue(stream)
        chunks = list(response)
        self.assertEqual(chunks, ["Hello"])

    def test_streaming_fallback_no_stream_func(self):
        """Test streaming falls back when no stream func."""
        provider = self._make_provider()

        with patch.object(
            google_mod,
            "_google_types",
            self.mock_types,
        ):
            response, stream = provider.execute_single_message(
                "Hello",
                "prompt",
                True,
                model="gemini-2.5-flash",
            )

        # Falls back to non-streaming
        self.assertFalse(stream)
        self.assertEqual(response, self.mock_response)

    def test_kwargs_passthrough_to_config(self):
        """Test extra kwargs pass through to config."""
        provider = self._make_provider()

        with patch.object(
            google_mod,
            "_google_types",
            self.mock_types,
        ):
            provider.execute_single_message(
                "Hello",
                "prompt",
                False,
                model="gemini-2.5-flash",
                temperature=0.5,
                top_p=0.9,
                top_k=40,
            )

        config_kw = self.mock_types.GenerateContentConfig.call_args[1]
        self.assertEqual(config_kw["temperature"], 0.5)
        self.assertEqual(config_kw["top_p"], 0.9)
        self.assertEqual(config_kw["top_k"], 40)

    def test_no_config_when_no_params(self):
        """Test no config built when no config params."""
        provider = self._make_provider()
        messages = [
            {"role": "user", "content": "Hello"},
        ]

        with patch.object(
            google_mod,
            "_google_types",
            self.mock_types,
        ):
            provider.execute_multi_message(
                messages,
                False,
                model="gemini-2.5-flash",
            )

        # No config params, so GenerateContentConfig
        # should not be called
        self.mock_types.GenerateContentConfig.assert_not_called()
        call_kwargs = self.mock_llm_func.call_args[1]
        self.assertNotIn("config", call_kwargs)

    def test_list_content_converted(self):
        """Test list content is converted to string."""
        provider = self._make_provider()
        messages = [
            {
                "role": "user",
                "content": ["part1", "part2"],
            },
        ]

        with patch.object(
            google_mod,
            "_google_types",
            self.mock_types,
        ):
            provider.execute_multi_message(
                messages,
                False,
                model="gemini-2.5-flash",
            )

        self.mock_types.Part.from_text.assert_called_once_with(
            text="['part1', 'part2']"
        )

    def test_tools_conversion(self):
        """Test OpenAI-style tools are converted to Google format."""
        provider = self._make_provider()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object"},
                },
            },
            {"custom_tool": True},
        ]

        with patch.object(
            google_mod,
            "_google_types",
            self.mock_types,
        ):
            provider.execute_single_message(
                "Hello",
                "prompt",
                False,
                model="gemini-2.5-flash",
                tools=tools,
            )

        config_kw = self.mock_types.GenerateContentConfig.call_args[1]
        converted = config_kw["tools"]
        self.assertEqual(len(converted), 2)
        self.assertEqual(
            converted[1]["function_declarations"][0]["name"],
            "search",
        )

    def test_assistant_tool_call_with_content(self):
        """Test assistant message with both content and tool_calls."""
        provider = self._make_provider()
        messages = [
            {
                "role": "assistant",
                "content": "Calling tool",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "search",
                            "arguments": '{"q": "test"}',
                        },
                    }
                ],
            },
        ]

        with patch.object(
            google_mod,
            "_google_types",
            self.mock_types,
        ):
            provider.execute_multi_message(
                messages,
                False,
                model="gemini-2.5-flash",
            )

        # Content text AND function response should be parts
        self.mock_types.Part.from_text.assert_called_once_with(
            text="Calling tool"
        )

    def test_tool_call_bad_json_args(self):
        """Test tool call with invalid JSON arguments."""
        provider = self._make_provider()
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "search",
                            "arguments": "not valid json{",
                        },
                    }
                ],
            },
        ]

        with patch.object(
            google_mod,
            "_google_types",
            self.mock_types,
        ):
            provider.execute_multi_message(
                messages,
                False,
                model="gemini-2.5-flash",
            )

        # Bad JSON should be caught, args become {}
        self.mock_types.Part.from_function_response.assert_called_once()
        call_kwargs = self.mock_types.Part.from_function_response.call_args[1]
        self.assertEqual(call_kwargs["response"], {"call": {}})

    def test_tool_result_message(self):
        """Test tool result message conversion."""
        provider = self._make_provider()
        messages = [
            {
                "role": "tool",
                "content": "result data",
                "tool_call_id": "call_1",
                "name": "search",
            },
        ]

        with patch.object(
            google_mod,
            "_google_types",
            self.mock_types,
        ):
            provider.execute_multi_message(
                messages,
                False,
                model="gemini-2.5-flash",
            )

        self.mock_types.Part.from_function_response.assert_called_once()
        call_kwargs = self.mock_types.Part.from_function_response.call_args[1]
        self.assertEqual(call_kwargs["name"], "search")


class TestGoogleResponseDetection(unittest.TestCase):
    """Test Google response detection functions."""

    def test_google_response_detected(self):
        """Test Google response is detected correctly."""
        resp = _make_simple_response()
        self.assertTrue(is_google_response_like(resp))

    def test_chatcompletion_not_detected(self):
        """Test ChatCompletion is not detected as Google."""
        # Plain class to avoid Mock auto-attributes
        resp = type(
            "ChatCompletion",
            (),
            {
                "choices": [],
                "model": "gpt-4",
                "candidates": [],
                "usage_metadata": None,
            },
        )()
        self.assertFalse(is_google_response_like(resp))

    def test_anthropic_not_detected(self):
        """Test Anthropic response not detected as Google."""
        resp = type(
            "AnthropicMsg",
            (),
            {
                "content": "Hello",
                "model": "claude-3",
                "role": "assistant",
            },
        )()
        self.assertFalse(is_google_response_like(resp))

    def test_xai_not_detected(self):
        """Test xAI response not detected as Google."""
        resp = type(
            "XAIResp",
            (),
            {"content": "Hello", "usage": None},
        )()
        self.assertFalse(is_google_response_like(resp))

    def test_string_not_detected(self):
        """Test plain string not detected as Google."""
        self.assertFalse(is_google_response_like("hello"))

    def test_extract_google_content(self):
        """Test content extraction from Google response."""
        resp = _make_simple_response("Hello world")
        self.assertEqual(extract_google_content(resp), "Hello world")

    def test_extract_google_content_non_google(self):
        """Test extraction returns None for non-Google."""
        self.assertIsNone(extract_google_content("hello"))

    def test_extract_response_content_google(self):
        """Test universal extractor handles Google."""
        resp = _make_simple_response("Hello world")
        self.assertEqual(extract_response_content(resp), "Hello world")


class TestGoogleResponseUncloaking(unittest.TestCase):
    """Test Google response uncloaking."""

    def test_uncloak_text_content(self):
        """Test text content is uncloaked."""
        resp = _make_simple_response("Hello <PERSON_0>")
        entity_map = {"<PERSON_0>": "John"}

        result = _uncloak_response(resp, entity_map)
        part = result.candidates[0].content.parts[0]
        self.assertEqual(part.text, "Hello John")

    def test_uncloak_function_call_args(self):
        """Test function call arguments are uncloaked."""
        fc = MockFunctionCall("send_email", {"to": "<EMAIL_0>"})
        part = MockPart(function_call=fc)
        candidate = MockCandidate(
            content=MockContent(role="model", parts=[part])
        )
        resp = MockGoogleResponse(candidates=[candidate])
        entity_map = {"<EMAIL_0>": "john@example.com"}

        result = _uncloak_response(resp, entity_map)
        result_fc = result.candidates[0].content.parts[0].function_call
        self.assertEqual(result_fc.args, {"to": "john@example.com"})

    def test_uncloak_preserves_original(self):
        """Test original response is not modified."""
        resp = _make_simple_response("Hello <PERSON_0>")
        entity_map = {"<PERSON_0>": "John"}

        _uncloak_response(resp, entity_map)
        original_text = resp.candidates[0].content.parts[0].text
        self.assertEqual(original_text, "Hello <PERSON_0>")

    def test_uncloak_none_text(self):
        """Test None text is handled gracefully."""
        part = MockPart(text=None)
        candidate = MockCandidate(
            content=MockContent(role="model", parts=[part])
        )
        resp = MockGoogleResponse(candidates=[candidate])
        entity_map = {"<PERSON_0>": "John"}

        result = _uncloak_response(resp, entity_map)
        result_text = result.candidates[0].content.parts[0].text
        self.assertIsNone(result_text)

    def test_uncloak_empty_candidates(self):
        """Test empty candidates list is handled."""
        resp = MockGoogleResponse(candidates=[])
        entity_map = {"<PERSON_0>": "John"}

        result = _uncloak_response(resp, entity_map)
        self.assertEqual(result.candidates, [])

    def test_uncloak_multiple_parts(self):
        """Test multiple parts are all uncloaked."""
        parts = [
            MockPart(text="Hello <PERSON_0>"),
            MockPart(text="From <PLACE_0>", thought=True),
        ]
        candidate = MockCandidate(
            content=MockContent(role="model", parts=parts)
        )
        resp = MockGoogleResponse(candidates=[candidate])
        entity_map = {
            "<PERSON_0>": "John",
            "<PLACE_0>": "London",
        }

        result = _uncloak_response(resp, entity_map)
        result_parts = result.candidates[0].content.parts
        self.assertEqual(result_parts[0].text, "Hello John")
        self.assertEqual(result_parts[1].text, "From London")


if __name__ == "__main__":
    unittest.main()
