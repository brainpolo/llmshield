"""Test provider factory and registration system.

Description:
    This test module validates the provider factory functionality that
    automatically detects and registers LLM providers, enabling seamless
    integration with various LLM services.

Test Classes:
    - TestProviderFactory: Tests provider registration and retrieval

Author: LLMShield by brainpolo, 2025-2026
"""

import unittest

from llmshield import LLMShield
from llmshield.providers.base import BaseLLMProvider
from llmshield.providers.cohere_provider import CohereProvider
from llmshield.providers.default_provider import DefaultProvider
from llmshield.providers.openai_provider import OpenAIProvider
from llmshield.providers.provider_factory import (
    PROVIDER_REGISTRY,
    get_provider,
    register_provider,
)
from tests.helpers import make_mock_func


class _StubProvider(BaseLLMProvider):
    """Minimal provider stub for registration tests."""

    @classmethod
    def can_handle(cls, llm_func):
        """Accept all functions."""
        return True

    def prepare_single_message_params(
        self, cloaked_text, input_param, stream, **kwargs
    ):
        """Return kwargs and stream unchanged."""
        return kwargs, stream

    def prepare_multi_message_params(self, cloaked_messages, stream, **kwargs):
        """Return kwargs and stream unchanged."""
        return kwargs, stream


class TestProviderFactory(unittest.TestCase):
    """Test provider factory functionality."""

    def setUp(self):
        """Save original registry to restore after tests."""
        self.original_registry = PROVIDER_REGISTRY.copy()

    def tearDown(self):
        """Restore original registry."""
        PROVIDER_REGISTRY.clear()
        PROVIDER_REGISTRY.extend(self.original_registry)

    def test_get_provider_openai(self):
        """Test getting OpenAI provider for OpenAI functions."""
        func = make_mock_func()
        provider = get_provider(func)
        self.assertIsInstance(provider, OpenAIProvider)

    def test_get_provider_openai_beta(self):
        """Test getting OpenAI provider for beta API functions."""
        func = make_mock_func(
            name="parse",
            qualname="client.beta.chat.completions.parse",
            module="openai.beta.client",
        )
        provider = get_provider(func)
        self.assertIsInstance(provider, OpenAIProvider)

    def test_get_provider_default_fallback(self):
        """Test getting default provider for unknown functions."""
        func = make_mock_func(
            name="unknown_function",
            qualname="some.module.unknown_function",
            module="some.module",
        )
        provider = get_provider(func)
        self.assertIsInstance(provider, DefaultProvider)

    def test_get_provider_no_provider_found_error(self):
        """Test RuntimeError when no provider can handle function."""
        PROVIDER_REGISTRY.clear()
        func = make_mock_func(
            name="test_func",
            qualname="test.module.test_func",
            module="test.module",
        )
        with self.assertRaises(RuntimeError) as ctx:
            get_provider(func)
        self.assertIn(
            "No provider found for function",
            str(ctx.exception),
        )

    def test_register_provider_at_priority_0(self):
        """Test registering a provider at highest priority."""
        original_length = len(PROVIDER_REGISTRY)
        register_provider(_StubProvider, priority=0)

        self.assertEqual(len(PROVIDER_REGISTRY), original_length + 1)
        self.assertEqual(PROVIDER_REGISTRY[0], _StubProvider)

    def test_register_provider_before_default(self):
        """Test registering a provider before DefaultProvider."""
        original_length = len(PROVIDER_REGISTRY)
        register_provider(_StubProvider, priority=-1)

        self.assertEqual(len(PROVIDER_REGISTRY), original_length + 1)
        self.assertEqual(PROVIDER_REGISTRY[-2], _StubProvider)
        self.assertEqual(PROVIDER_REGISTRY[-1], DefaultProvider)

    def test_register_provider_custom_priority(self):
        """Test registering a provider at custom position."""
        original_length = len(PROVIDER_REGISTRY)
        register_provider(_StubProvider, priority=1)

        self.assertEqual(len(PROVIDER_REGISTRY), original_length + 1)
        self.assertEqual(PROVIDER_REGISTRY[1], _StubProvider)

    def test_provider_registry_order(self):
        """Test that provider registry maintains proper order."""
        self.assertEqual(PROVIDER_REGISTRY[-1], DefaultProvider)
        openai_idx = PROVIDER_REGISTRY.index(OpenAIProvider)
        default_idx = PROVIDER_REGISTRY.index(DefaultProvider)
        self.assertLess(openai_idx, default_idx)

        cohere_idx = PROVIDER_REGISTRY.index(CohereProvider)
        self.assertLess(cohere_idx, openai_idx)

    def test_get_provider_cohere(self):
        """Test getting Cohere provider for Cohere functions."""
        func = make_mock_func(
            name="chat",
            qualname="ClientV2.chat",
            module="cohere.v2.client",
        )
        provider = get_provider(func)
        self.assertIsInstance(provider, CohereProvider)


class TestShieldProviderProperty(unittest.TestCase):
    """Test LLMShield.provider readonly property."""

    def test_provider_openai(self):
        """Test OpenAI function exposes OpenAIProvider."""
        func = make_mock_func(module="openai._legacy_response")
        shield = LLMShield(llm_func=func)
        self.assertIsInstance(shield.provider, OpenAIProvider)

    def test_provider_none_without_llm_func(self):
        """Test provider is None when no llm_func provided."""
        shield = LLMShield()
        self.assertIsNone(shield.provider)

    def test_provider_default_for_unknown(self):
        """Test unknown function gets DefaultProvider."""
        func = make_mock_func(
            name="my_func",
            qualname="my_func",
            module="my_app",
        )
        shield = LLMShield(llm_func=func)
        self.assertIsInstance(shield.provider, DefaultProvider)

    def test_provider_is_readonly(self):
        """Test provider property cannot be set."""
        shield = LLMShield()
        with self.assertRaises(AttributeError):
            shield.provider = "something"


if __name__ == "__main__":
    unittest.main()
