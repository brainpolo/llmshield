"""Tests for default provider functionality."""

import unittest
from unittest.mock import Mock

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

        prepared_params, actual_stream = provider.prepare_single_message_params(
            cloaked_text, input_param, stream, **kwargs
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
        """Test preparing single message parameters when function prefers 'message'."""
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

        prepared_params, actual_stream = provider.prepare_single_message_params(
            cloaked_text, input_param, stream, **kwargs
        )

        # Check that 'message' is used as parameter name
        self.assertEqual(prepared_params["message"], cloaked_text)
        self.assertNotIn("text", prepared_params)
        self.assertFalse(prepared_params["stream"])
        self.assertFalse(actual_stream)

    def test_prepare_single_message_params_with_prompt_preference(self):
        """Test preparing single message parameters when function prefers 'prompt'."""
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

        prepared_params, actual_stream = provider.prepare_single_message_params(
            cloaked_text, input_param, stream, **kwargs
        )

        # Check that 'prompt' is used as parameter name
        self.assertEqual(prepared_params["prompt"], cloaked_text)
        self.assertNotIn("text", prepared_params)

    def test_prepare_single_message_params_no_code_attribute(self):
        """Test preparing single message parameters when function has no __code__ attribute."""
        # Create mock function without __code__ attribute
        mock_func_no_code = Mock()
        delattr(mock_func_no_code, "__code__")

        provider = DefaultProvider(mock_func_no_code)

        cloaked_text = "Hello <PERSON_0>"
        input_param = "text"
        stream = True
        kwargs = {"text": "Original text"}

        prepared_params, actual_stream = provider.prepare_single_message_params(
            cloaked_text, input_param, stream, **kwargs
        )

        # Should fall back to 'prompt' as default
        self.assertEqual(prepared_params["prompt"], cloaked_text)

    def test_prepare_single_message_params_code_access_error(self):
        """Test preparing single message parameters when code access raises an error."""
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

        prepared_params, actual_stream = provider.prepare_single_message_params(
            cloaked_text, input_param, stream, **kwargs
        )

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

    def test_get_preferred_param_name_message_priority(self):
        """Test that 'message' is preferred over 'prompt' when both are available."""
        # Create mock function with both 'message' and 'prompt' in parameters
        mock_func_both = Mock()
        mock_func_both.__code__ = Mock()
        mock_func_both.__code__.co_varnames = ("self", "prompt", "message", "model")

        provider = DefaultProvider(mock_func_both)

        # Should prefer 'message' over 'prompt'
        result = provider._get_preferred_param_name()
        self.assertEqual(result, "message")

    def test_get_preferred_param_name_prompt_only(self):
        """Test that 'prompt' is used when only 'prompt' is available."""
        # Create mock function with only 'prompt' in parameters
        mock_func_prompt = Mock()
        mock_func_prompt.__code__ = Mock()
        mock_func_prompt.__code__.co_varnames = ("self", "prompt", "model")

        provider = DefaultProvider(mock_func_prompt)

        result = provider._get_preferred_param_name()
        self.assertEqual(result, "prompt")

    def test_get_preferred_param_name_neither_available(self):
        """Test fallback to 'prompt' when neither 'message' nor 'prompt' are available."""
        # Create mock function with neither 'message' nor 'prompt' in parameters
        mock_func_neither = Mock()
        mock_func_neither.__code__ = Mock()
        mock_func_neither.__code__.co_varnames = ("self", "text", "model")

        provider = DefaultProvider(mock_func_neither)

        result = provider._get_preferred_param_name()
        self.assertEqual(result, "prompt")

    def test_get_preferred_param_name_attribute_error(self):
        """Test fallback behavior when AttributeError is raised."""
        # Create mock function that raises AttributeError
        mock_func_error = Mock()
        del mock_func_error.__code__  # Remove __code__ attribute entirely

        provider = DefaultProvider(mock_func_error)

        result = provider._get_preferred_param_name()
        self.assertEqual(result, "prompt")

    def test_can_handle_any_function(self):
        """Test that DefaultProvider can handle any function."""
        # Test with various function types
        self.assertTrue(DefaultProvider.can_handle(self.mock_func))
        self.assertTrue(DefaultProvider.can_handle(lambda x: x))
        self.assertTrue(DefaultProvider.can_handle(print))
        self.assertTrue(DefaultProvider.can_handle(Mock()))

    def test_can_handle_returns_true_always(self):
        """Test that can_handle always returns True as it's the fallback provider."""
        # Even with None or unusual objects
        self.assertTrue(DefaultProvider.can_handle(None))
        self.assertTrue(DefaultProvider.can_handle("not a function"))
        self.assertTrue(DefaultProvider.can_handle(42))


if __name__ == "__main__":
    unittest.main()
