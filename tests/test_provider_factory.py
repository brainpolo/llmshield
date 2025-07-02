"""Test provider factory and registration system.

Description:
    This test module validates the provider factory functionality that
    automatically detects and registers LLM providers, enabling seamless
    integration with various LLM services.

Test Classes:
    - TestProviderFactory: Tests provider registration and retrieval

Author: LLMShield by brainpolo, 2025
"""

# Standard library imports
import unittest
from unittest.mock import Mock

from llmshield.providers.base import BaseLLMProvider
from llmshield.providers.default_provider import DefaultProvider
from llmshield.providers.openai_provider import OpenAIProvider

# Local Imports
from llmshield.providers.provider_factory import (
    PROVIDER_REGISTRY,
    get_provider,
    register_provider,
)


class TestProviderFactory(unittest.TestCase):
    """Test provider factory functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Store original registry to restore after tests
        self.original_registry = PROVIDER_REGISTRY.copy()

    def tearDown(self):
        """Clean up after tests."""
        # Restore original registry
        PROVIDER_REGISTRY.clear()
        PROVIDER_REGISTRY.extend(self.original_registry)

    def test_get_provider_openai(self):
        """Test getting OpenAI provider for OpenAI functions."""
        # Mock OpenAI function
        mock_openai_func = Mock()
        mock_openai_func.__name__ = "create"
        mock_openai_func.__qualname__ = "client.chat.completions.create"
        mock_openai_func.__module__ = "openai.client"

        provider = get_provider(mock_openai_func)
        self.assertIsInstance(provider, OpenAIProvider)

    def test_get_provider_openai_beta(self):
        """Test getting OpenAI provider for beta API functions."""
        # Mock OpenAI beta function
        mock_beta_func = Mock()
        mock_beta_func.__name__ = "parse"
        mock_beta_func.__qualname__ = "client.beta.chat.completions.parse"
        mock_beta_func.__module__ = "openai.beta.client"

        provider = get_provider(mock_beta_func)
        self.assertIsInstance(provider, OpenAIProvider)

    def test_get_provider_default_fallback(self):
        """Test getting default provider for unknown functions."""
        # Mock unknown function
        mock_unknown_func = Mock()
        mock_unknown_func.__name__ = "unknown_function"
        mock_unknown_func.__qualname__ = "some.module.unknown_function"
        mock_unknown_func.__module__ = "some.module"

        provider = get_provider(mock_unknown_func)
        self.assertIsInstance(provider, DefaultProvider)

    def test_get_provider_no_provider_found_error(self):
        """Test RuntimeError when no provider can handle function."""
        # Clear registry to force error
        PROVIDER_REGISTRY.clear()

        mock_func = Mock()
        mock_func.__name__ = "test_func"
        mock_func.__qualname__ = "test.module.test_func"
        mock_func.__module__ = "test.module"

        with self.assertRaises(RuntimeError) as context:
            get_provider(mock_func)

        self.assertIn("No provider found for function", str(context.exception))

    def test_register_provider_at_priority(self):
        """Test registering a provider at specific priority."""

        # Create mock provider class
        class TestProvider(BaseLLMProvider):
            @classmethod
            def can_handle(cls, llm_func):
                return True

            def prepare_single_message_params(
                self, cloaked_text, input_param, stream, **kwargs
            ):
                return kwargs, stream

            def prepare_multi_message_params(
                self, cloaked_messages, stream, **kwargs
            ):
                return kwargs, stream

        # Register at priority 0 (highest)
        original_length = len(PROVIDER_REGISTRY)
        register_provider(TestProvider, priority=0)

        self.assertEqual(len(PROVIDER_REGISTRY), original_length + 1)
        self.assertEqual(PROVIDER_REGISTRY[0], TestProvider)

    def test_register_provider_before_default(self):
        """Test registering a provider before DefaultProvider."""

        # Create mock provider class
        class TestProvider(BaseLLMProvider):
            @classmethod
            def can_handle(cls, llm_func):
                return True

            def prepare_single_message_params(
                self, cloaked_text, input_param, stream, **kwargs
            ):
                return kwargs, stream

            def prepare_multi_message_params(
                self, cloaked_messages, stream, **kwargs
            ):
                return kwargs, stream

        # Register before DefaultProvider (priority -1)
        original_length = len(PROVIDER_REGISTRY)
        register_provider(TestProvider, priority=-1)

        self.assertEqual(len(PROVIDER_REGISTRY), original_length + 1)
        # Should be second to last (before DefaultProvider)
        self.assertEqual(PROVIDER_REGISTRY[-2], TestProvider)
        self.assertEqual(PROVIDER_REGISTRY[-1], DefaultProvider)

    def test_register_provider_custom_priority(self):
        """Test registering a provider at custom priority position."""

        # Create mock provider class
        class TestProvider(BaseLLMProvider):
            @classmethod
            def can_handle(cls, llm_func):
                return True

            def prepare_single_message_params(
                self, cloaked_text, input_param, stream, **kwargs
            ):
                return kwargs, stream

            def prepare_multi_message_params(
                self, cloaked_messages, stream, **kwargs
            ):
                return kwargs, stream

        # Register at position 1
        original_length = len(PROVIDER_REGISTRY)
        register_provider(TestProvider, priority=1)

        self.assertEqual(len(PROVIDER_REGISTRY), original_length + 1)
        self.assertEqual(PROVIDER_REGISTRY[1], TestProvider)

    def test_provider_registry_order(self):
        """Test that provider registry maintains proper order."""
        # Verify DefaultProvider is last
        self.assertEqual(PROVIDER_REGISTRY[-1], DefaultProvider)

        # Verify OpenAIProvider comes before DefaultProvider
        openai_index = PROVIDER_REGISTRY.index(OpenAIProvider)
        default_index = PROVIDER_REGISTRY.index(DefaultProvider)
        self.assertLess(openai_index, default_index)


if __name__ == "__main__":
    unittest.main()
