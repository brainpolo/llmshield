"""Test default provider for generic LLM functions.

Description:
    This test module validates the DefaultProvider class that handles
    generic LLM functions when no specific provider is detected, ensuring
    compatibility with any LLM API.

Test Classes:
    - TestDefaultProvider: Tests generic provider functionality

Author: LLMShield by brainpolo, 2025-2026
"""

import unittest
from unittest.mock import Mock

from parameterized import parameterized

from llmshield.providers.base import BaseLLMProvider
from llmshield.providers.default_provider import DefaultProvider


class TestDefaultProvider(unittest.TestCase):
    """Test default provider functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock generic function
        self.mock_func = Mock()
        self.mock_func.__name__ = "generic_llm_function"
        self.mock_func.__qualname__ = "some.module.generic_llm_function"
        self.mock_func.__module__ = "some.module"

    def test_init(self):
        """Test initialization."""
        provider = DefaultProvider(self.mock_func)
        self.assertEqual(provider.llm_func, self.mock_func)

    def test_prepare_single_message_params_default(self):
        """Test preparing single message parameters with default behavior."""
        provider = DefaultProvider(self.mock_func)

        cloaked_text = "Hello <PERSON_0>"
        input_param = "original_param"
        stream = True
        kwargs = {
            "original_param": "Original text",
            "model": "test-model",
            "temperature": 0.8,
        }

        prepared_params, actual_stream = (
            provider.prepare_single_message_params(
                cloaked_text, input_param, stream, **kwargs
            )
        )

        # Check that original parameter is removed
        self.assertNotIn("original_param", prepared_params)

        # Check that prompt is used as default parameter name
        self.assertEqual(prepared_params["prompt"], cloaked_text)

        # Check other parameters are preserved
        self.assertEqual(prepared_params["model"], "test-model")
        self.assertEqual(prepared_params["temperature"], 0.8)

        # Check streaming is enabled
        self.assertTrue(prepared_params["stream"])
        self.assertTrue(actual_stream)

    def test_prepare_single_message_params_with_message_preference(self):
        """Test preparing single message parameters with 'message' preference.

        Validates parameter preparation when function prefers 'message'.
        """
        # Create mock function with 'message' in parameters
        mock_func_with_message = Mock()
        mock_func_with_message.__code__ = Mock()
        mock_func_with_message.__code__.co_varnames = (
            "self",
            "message",
            "model",
            "stream",
        )

        provider = DefaultProvider(mock_func_with_message)

        cloaked_text = "Hello <PERSON_0>"
        input_param = "text"
        stream = False
        kwargs = {"text": "Original text", "model": "test-model"}

        prepared_params, actual_stream = (
            provider.prepare_single_message_params(
                cloaked_text, input_param, stream, **kwargs
            )
        )

        # Check that 'message' is used as parameter name
        self.assertEqual(prepared_params["message"], cloaked_text)
        self.assertNotIn("text", prepared_params)
        self.assertFalse(prepared_params["stream"])
        self.assertFalse(actual_stream)

    def test_prepare_single_message_params_with_prompt_preference(self):
        """Test preparing single message parameters with 'prompt' preference.

        Validates parameter preparation when function prefers 'prompt'.
        """
        # Create mock function with 'prompt' but not 'message' in parameters
        mock_func_with_prompt = Mock()
        mock_func_with_prompt.__code__ = Mock()
        mock_func_with_prompt.__code__.co_varnames = (
            "self",
            "prompt",
            "model",
            "stream",
        )

        provider = DefaultProvider(mock_func_with_prompt)

        cloaked_text = "Hello <PERSON_0>"
        input_param = "text"
        stream = True
        kwargs = {"text": "Original text", "model": "test-model"}

        prepared_params, actual_stream = (
            provider.prepare_single_message_params(
                cloaked_text, input_param, stream, **kwargs
            )
        )

        # Check that 'prompt' is used as parameter name
        self.assertEqual(prepared_params["prompt"], cloaked_text)
        self.assertNotIn("text", prepared_params)
        self.assertTrue(actual_stream)

    def test_prepare_single_message_params_no_code_attribute(self):
        """Test preparing single message parameters without __code__ attribute.

        Validates fallback behavior when function has no __code__ attribute.
        """
        # Create mock function without __code__ attribute
        mock_func_no_code = Mock()
        delattr(mock_func_no_code, "__code__")

        provider = DefaultProvider(mock_func_no_code)

        cloaked_text = "Hello <PERSON_0>"
        input_param = "text"
        stream = True
        kwargs = {"text": "Original text"}

        prepared_params, actual_stream = (
            provider.prepare_single_message_params(
                cloaked_text, input_param, stream, **kwargs
            )
        )

        # Should fall back to 'prompt' as default
        self.assertEqual(prepared_params["prompt"], cloaked_text)
        self.assertTrue(actual_stream)

    def test_prepare_single_message_params_code_access_error(self):
        """Test preparing single message parameters with code access error.

        Validates error handling when code access raises an exception.
        """
        # Create mock function that raises TypeError when accessing __code__
        mock_func_error = Mock()
        mock_func_error.__code__ = Mock()
        mock_func_error.__code__.co_varnames = Mock(
            side_effect=TypeError("Access error")
        )

        provider = DefaultProvider(mock_func_error)

        cloaked_text = "Hello <PERSON_0>"
        input_param = "text"
        stream = True
        kwargs = {"text": "Original text"}

        prepared_params, actual_stream = (
            provider.prepare_single_message_params(
                cloaked_text, input_param, stream, **kwargs
            )
        )
        self.assertTrue(actual_stream)

        # Should fall back to 'prompt' as default when error occurs
        self.assertEqual(prepared_params["prompt"], cloaked_text)

    def test_prepare_multi_message_params(self):
        """Test preparing multi-message parameters."""
        provider = DefaultProvider(self.mock_func)

        cloaked_messages = [
            {"role": "user", "content": "Hello <PERSON_0>"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        stream = True
        kwargs = {"model": "test-model", "temperature": 0.7}

        prepared_params, actual_stream = provider.prepare_multi_message_params(
            cloaked_messages, stream, **kwargs
        )

        # Check messages are preserved
        self.assertEqual(prepared_params["messages"], cloaked_messages)

        # Check other parameters are preserved
        self.assertEqual(prepared_params["model"], "test-model")
        self.assertEqual(prepared_params["temperature"], 0.7)

        # Check streaming is enabled
        self.assertTrue(prepared_params["stream"])
        self.assertTrue(actual_stream)

    def test_prepare_multi_message_params_no_stream(self):
        """Test preparing multi-message parameters without streaming."""
        provider = DefaultProvider(self.mock_func)

        cloaked_messages = [{"role": "user", "content": "Hello"}]
        stream = False
        kwargs = {"model": "test-model"}

        prepared_params, actual_stream = provider.prepare_multi_message_params(
            cloaked_messages, stream, **kwargs
        )

        # Check streaming is disabled
        self.assertFalse(prepared_params["stream"])
        self.assertFalse(actual_stream)

    @parameterized.expand(
        [
            # (description, function_params, expected_preferred_param)
            (
                "both_message_and_prompt",
                ["self", "message", "prompt", "model"],
                "message",
            ),
            ("prompt_only", ["self", "prompt", "model"], "prompt"),
            ("message_only", ["self", "message", "model"], "message"),
            ("neither_param", ["self", "text", "model"], "prompt"),
            ("empty_params", [], "prompt"),
            (
                "prompt_first",
                ["self", "prompt", "message", "model"],
                "message",
            ),  # message has priority
            (
                "with_stream",
                ["self", "message", "stream", "model"],
                "message",
            ),
            (
                "many_params",
                ["self", "a", "b", "message", "c", "prompt", "d"],
                "message",
            ),
        ]
    )
    def test_preferred_param_name_variations(
        self, description, function_params, expected
    ):
        """Test preferred parameter name selection.

        Parameterized test for different function signature scenarios.
        """
        mock_func = Mock()
        mock_func.__code__ = Mock()
        mock_func.__code__.co_varnames = tuple(function_params)

        provider = DefaultProvider(mock_func)
        result = provider._get_preferred_param_name()
        self.assertEqual(result, expected)

    def test_get_preferred_param_name_attribute_error(self):
        """Test fallback when function has no __code__ attribute."""
        mock_func_error = Mock()
        del mock_func_error.__code__
        provider = DefaultProvider(mock_func_error)
        self.assertEqual(provider._get_preferred_param_name(), "prompt")

    @parameterized.expand(
        [
            # (description, input_param, stream, kwargs, expected_stream)
            (
                "enable_streaming",
                "text",
                True,
                {"text": "hello", "model": "gpt-4"},
                True,
            ),
            (
                "disable_streaming",
                "prompt",
                False,
                {"prompt": "hello", "model": "gpt-4"},
                False,
            ),
            (
                "with_temperature",
                "message",
                True,
                {"message": "hello", "temperature": 0.8},
                True,
            ),
            (
                "with_max_tokens",
                "input",
                False,
                {"input": "hello", "max_tokens": 100},
                False,
            ),
            ("minimal_params", "text", True, {"text": "hello"}, True),
            (
                "many_params",
                "prompt",
                False,
                {
                    "prompt": "hello",
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "max_tokens": 150,
                    "top_p": 0.9,
                },
                False,
            ),
        ]
    )
    def test_single_message_param_preparation_variations(
        self, description, input_param, stream, kwargs, expected_stream
    ):
        """Test single message parameter preparation.

        Parameterized test for different configuration scenarios.
        """
        provider = DefaultProvider(self.mock_func)

        prepared_params, actual_stream = (
            provider.prepare_single_message_params(
                "cloaked text", input_param, stream, **kwargs
            )
        )

        # Original parameter should be removed unless it's the preferred
        # parameter
        preferred_param = provider._get_preferred_param_name()
        if input_param != preferred_param:
            self.assertNotIn(input_param, prepared_params)
        # Stream setting should match
        self.assertEqual(actual_stream, expected_stream)
        self.assertEqual(prepared_params["stream"], expected_stream)
        # Cloaked text should be under preferred param name
        preferred_param = provider._get_preferred_param_name()
        self.assertEqual(prepared_params[preferred_param], "cloaked text")

    @parameterized.expand(
        [
            # (description, stream, additional_kwargs, expected_keys)
            (
                "basic_streaming",
                True,
                {"model": "gpt-4"},
                ["messages", "stream", "model"],
            ),
            (
                "no_streaming",
                False,
                {"model": "gpt-4"},
                ["messages", "stream", "model"],
            ),
            (
                "with_temperature",
                True,
                {"model": "gpt-4", "temperature": 0.8},
                ["messages", "stream", "model", "temperature"],
            ),
            (
                "many_params",
                False,
                {
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "max_tokens": 100,
                    "top_p": 0.9,
                    "frequency_penalty": 0.1,
                },
                [
                    "messages",
                    "stream",
                    "model",
                    "temperature",
                    "max_tokens",
                    "top_p",
                    "frequency_penalty",
                ],
            ),
            ("minimal", True, {}, ["messages", "stream"]),
        ]
    )
    def test_multi_message_param_preparation_variations(
        self, description, stream, additional_kwargs, expected_keys
    ):
        """Test multi-message parameter preparation.

        Parameterized test for different configuration scenarios.
        """
        provider = DefaultProvider(self.mock_func)

        messages = [{"role": "user", "content": "hello"}]
        prepared_params, actual_stream = provider.prepare_multi_message_params(
            messages, stream, **additional_kwargs
        )

        # Check all expected keys are present
        for key in expected_keys:
            self.assertIn(key, prepared_params)

        # Check messages are preserved
        self.assertEqual(prepared_params["messages"], messages)
        # Check stream setting
        self.assertEqual(actual_stream, stream)
        self.assertEqual(prepared_params["stream"], stream)

    @parameterized.expand(
        [
            # (description, function_attributes, should_handle)
            (
                "normal_function",
                {"__name__": "func", "__qualname__": "module.func"},
                True,
            ),
            (
                "lambda_function",
                {"__name__": "<lambda>", "__qualname__": "<lambda>"},
                True,
            ),
            (
                "builtin_function",
                {"__name__": "print", "__qualname__": "builtins.print"},
                True,
            ),
            (
                "mock_object",
                {"__name__": "mock", "__qualname__": "mock"},
                True,
            ),
            ("none_object", {}, True),  # DefaultProvider handles anything
            ("string_object", {}, True),  # Even non-functions
        ]
    )
    def test_can_handle_various_objects(
        self, description, function_attributes, should_handle
    ):
        """Test can_handle method with various object types."""
        if function_attributes:
            mock_obj = Mock()
            for attr, value in function_attributes.items():
                setattr(mock_obj, attr, value)
        else:
            mock_obj = "not a function"  # Test with non-function object

        result = DefaultProvider.can_handle(mock_obj)
        self.assertEqual(result, should_handle)

    @parameterized.expand(
        [
            # (description, error_type, fallback_expected)
            ("attribute_error", AttributeError, "prompt"),
            ("type_error", TypeError, "prompt"),
            ("value_error", ValueError, "prompt"),
            ("runtime_error", RuntimeError, "prompt"),
        ]
    )
    def test_error_handling_in_param_detection(
        self, description, error_type, fallback_expected
    ):
        """Test error handling during parameter name detection.

        Parameterized test for different error scenarios.
        """
        mock_func = Mock()
        mock_func.__code__ = Mock()
        mock_func.__code__.co_varnames = Mock(
            side_effect=error_type("Test error")
        )

        provider = DefaultProvider(mock_func)
        result = provider._get_preferred_param_name()
        self.assertEqual(result, fallback_expected)

    @parameterized.expand(
        [
            # (description, cloaked_text, original_param, extra_params)
            ("simple_text", "Hello world", "prompt", {}),
            ("with_entities", "Hello <PERSON_0>", "message", {}),
            ("unicode_text", "Hola <PERSON_0> 🌍", "text", {}),
            ("long_text", "A" * 1000, "input", {}),
            ("empty_text", "", "prompt", {}),
            ("with_newlines", "Line 1\nLine 2\nLine 3", "content", {}),
            ("with_special_chars", "Text with @#$%^&*() chars", "prompt", {}),
            ("json_like", '{"message": "Hello <PERSON_0>"}', "data", {}),
        ]
    )
    def test_text_handling_variations(
        self, description, cloaked_text, original_param, extra_params
    ):
        """Test handling of various text types and formats."""
        provider = DefaultProvider(self.mock_func)

        kwargs = {original_param: "original", **extra_params}
        prepared_params, _ = provider.prepare_single_message_params(
            cloaked_text, original_param, True, **kwargs
        )

        # Check that cloaked text is properly set
        preferred_param = provider._get_preferred_param_name()
        self.assertEqual(prepared_params[preferred_param], cloaked_text)
        # Original parameter should be removed unless it's the preferred
        # parameter
        preferred_param = provider._get_preferred_param_name()
        if original_param != preferred_param:
            self.assertNotIn(original_param, prepared_params)


class TestBaseLLMProviderDefaults(unittest.TestCase):
    """Test BaseLLMProvider default prepare method implementations."""

    def setUp(self):
        """Set up a minimal subclass that uses base defaults."""
        self.mock_func = Mock(return_value="response")
        self.mock_func.__name__ = "test_func"
        self.mock_func.__qualname__ = "test_func"
        self.mock_func.__module__ = "test"

        class MinimalProvider(BaseLLMProvider):
            @classmethod
            def can_handle(cls, llm_func):
                return True

        self.provider = MinimalProvider(self.mock_func)

    def test_prepare_single_message_default(self):
        """Test base prepare_single_message_params creates messages."""
        params, stream = self.provider.prepare_single_message_params(
            "Hello", "prompt", True, prompt="original", model="t"
        )
        self.assertEqual(
            params["messages"],
            [{"role": "user", "content": "Hello"}],
        )
        self.assertTrue(params["stream"])
        self.assertNotIn("prompt", params)
        self.assertEqual(params["model"], "t")
        self.assertTrue(stream)

    def test_prepare_multi_message_default(self):
        """Test base prepare_multi_message_params passes through."""
        msgs = [{"role": "user", "content": "Hi"}]
        params, stream = self.provider.prepare_multi_message_params(
            msgs, False, model="t"
        )
        self.assertEqual(params["messages"], msgs)
        self.assertFalse(params["stream"])
        self.assertEqual(params["model"], "t")
        self.assertFalse(stream)


class TestBaseProviderExecuteMethods(unittest.TestCase):
    """Test execute methods inherited from BaseLLMProvider."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_func = Mock(return_value="llm response")
        self.mock_func.__name__ = "generic_llm_function"
        self.mock_func.__qualname__ = "module.generic_llm_function"
        self.mock_func.__module__ = "some.module"
        self.provider = DefaultProvider(self.mock_func)

    def test_execute_single_message(self):
        """Test execute_single_message prepares and calls."""
        response, actual_stream = self.provider.execute_single_message(
            "Hello <PERSON_0>",
            "prompt",
            True,
            prompt="original",
            model="test",
        )

        self.assertEqual(response, "llm response")
        self.assertTrue(actual_stream)
        self.mock_func.assert_called_once()

    def test_execute_multi_message(self):
        """Test execute_multi_message prepares and calls."""
        messages = [
            {"role": "user", "content": "Hello <PERSON_0>"},
        ]
        response, actual_stream = self.provider.execute_multi_message(
            messages,
            False,
            model="test",
        )

        self.assertEqual(response, "llm response")
        self.assertFalse(actual_stream)
        self.mock_func.assert_called_once()
        call_kwargs = self.mock_func.call_args[1]
        self.assertEqual(call_kwargs["messages"], messages)
        self.assertFalse(call_kwargs["stream"])

    def test_execute_raw(self):
        """Test execute_raw passes kwargs directly."""
        response = self.provider.execute_raw(
            prompt="Hello",
            model="test",
        )

        self.assertEqual(response, "llm response")
        self.mock_func.assert_called_once_with(
            prompt="Hello",
            model="test",
        )


if __name__ == "__main__":
    unittest.main()
