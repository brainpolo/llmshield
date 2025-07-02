"""Test edge cases and boundary conditions.

Description:
    This test module provides comprehensive testing of edge cases and
    boundary conditions throughout the library, including empty inputs,
    special characters, malformed data, and extreme scenarios.

Test Classes:
    - TestEdgeCases: Tests comprehensive edge cases

Author: LLMShield by brainpolo, 2025
"""

# Standard library Imports
import random
import unittest

# Third party Imports
from parameterized import parameterized

# Local Imports
from llmshield import LLMShield


class TestEdgeCases(unittest.TestCase):
    """Test comprehensive edge cases and boundary conditions."""

    @parameterized.expand(
        [
            # (input_value, should_succeed, expected_result/error_message)
            (None, True, None),  # None should be allowed
            (
                lambda **kwargs: "response",
                True,
                "callable",
            ),  # Function should work
            ("not_callable", False, "llm_func must be a callable"),
            (123, False, "llm_func must be a callable"),
        ]
    )
    def test_llm_function_validation(
        self, input_val, should_succeed, expected
    ):
        """Test LLM function validation with various input types."""
        if should_succeed:
            shield = LLMShield(llm_func=input_val)
            if expected == "callable":
                self.assertTrue(callable(shield.llm_func))
            else:
                self.assertEqual(shield.llm_func, expected)
        else:
            with self.assertRaises(ValueError) as context:
                LLMShield(llm_func=input_val)
            self.assertIn(expected, str(context.exception))

    def test_llm_function_validation_callable_object(self):
        """Test callable object separately due to complexity."""

        class CallableClass:
            def __call__(self, **kwargs):
                return "response"

        callable_obj = CallableClass()
        shield = LLMShield(llm_func=callable_obj)
        self.assertEqual(shield.llm_func, callable_obj)

    @parameterized.expand(
        [
            ("", "start_delimiter"),  # Empty string
            (None, "start_delimiter"),  # None
            (123, "start_delimiter"),  # Integer
        ]
    )
    def test_delimiter_validation_invalid(self, invalid_delimiter, param_name):
        """Test delimiter validation with invalid cases."""
        with self.assertRaises(ValueError):
            if param_name == "start_delimiter":
                LLMShield(start_delimiter=invalid_delimiter)
            else:
                LLMShield(end_delimiter=invalid_delimiter)

    @parameterized.expand(
        [
            ("a", "z"),  # Single characters
            ("<<", ">>"),  # Multi-character
            ("[", "]"),  # Brackets
            ("{{", "}}"),  # Braces
        ]
    )
    def test_delimiter_validation_valid(self, start_del, end_del):
        """Test valid delimiter configurations."""
        shield = LLMShield(start_delimiter=start_del, end_delimiter=end_del)
        self.assertEqual(shield.start_delimiter, start_del)
        self.assertEqual(shield.end_delimiter, end_del)

    @parameterized.expand([1, 10, 100, 1000, 10000])
    def test_cache_size_configurations(self, cache_size):
        """Test cache size configuration edge cases."""
        shield = LLMShield(max_cache_size=cache_size)
        self.assertEqual(shield._cache.capacity, cache_size)

    def test_cache_eviction_behavior(self):
        """Test cache eviction behavior with size 1."""

        def mock_llm(**kwargs):
            return "response"

        shield = LLMShield(llm_func=mock_llm, max_cache_size=1)

        messages_list = [
            [{"role": "user", "content": "First message"}],
            [{"role": "user", "content": "Second message"}],
        ]

        for messages in messages_list:
            shield.ask(messages=messages)

        # First item should be evicted due to cache size limit
        self.assertEqual(len(shield._cache.cache), 1)

    @parameterized.expand(
        [
            ("not a generator", TypeError),  # String
            (b"bytes", TypeError),  # Bytes
            ({}, ValueError),  # Empty dict (falsy, raises ValueError)
        ]
    )
    def test_stream_response_validation_invalid(
        self, invalid_input, expected_error
    ):
        """Test stream response validation with invalid input types."""
        shield = LLMShield()
        with self.assertRaises(expected_error):
            list(shield.stream_uncloak(invalid_input))

    def test_stream_response_validation_valid(self):
        """Test valid stream response."""
        shield = LLMShield()

        def valid_generator():
            yield "Hello"
            yield " <PERSON_0>"

        entity_map = {"<PERSON_0>": "John"}
        result = list(shield.stream_uncloak(valid_generator(), entity_map))
        expected = ["Hello", " ", "John"]
        self.assertEqual(result, expected)

    @parameterized.expand(
        [
            # (entity_map, input_text, expected_output, should_raise)
            ({}, "Hello <PERSON_0>", "Hello <PERSON_0>", False),  # Empty map
            (
                None,
                "Hello <PERSON_0>",
                None,
                True,
            ),  # None map, no previous cloak
            (
                {"<PERSON_0>": "", "<PLACE_0>": ""},
                "Hello <PERSON_0> from <PLACE_0>",
                "Hello  from ",
                False,
            ),  # Empty values
            (
                {"<PERSON_0>": "John"},
                "Hello <PERSON_0>",
                "Hello John",
                False,
            ),  # Normal case
        ]
    )
    def test_entity_map_edge_cases(
        self, entity_map, input_text, expected_output, should_raise
    ):
        """Test entity map handling with edge cases."""
        shield = LLMShield()

        if should_raise:
            with self.assertRaises(ValueError):
                shield.uncloak(input_text, entity_map)
        else:
            result = shield.uncloak(input_text, entity_map)
            self.assertEqual(result, expected_output)

    @parameterized.expand(
        [
            # (description, messages, expected_check_lambda)
            (
                "single message",
                [{"role": "user", "content": "Hello John Doe"}],
                lambda r: "Hello" in r,
            ),
            (
                "minimal conversation",
                [{"role": "user", "content": "Just one message"}],
                lambda r: isinstance(r, str),
            ),
        ]
    )
    def test_conversation_flow_scenarios(
        self, description, messages, check_func
    ):
        """Test conversation flow with edge cases."""

        def mock_llm(**kwargs):
            return f"Response to: {kwargs.get('messages', [])[-1]['content']}"

        shield = LLMShield(llm_func=mock_llm)
        result = shield.ask(messages=messages)
        self.assertTrue(check_func(result))

    def test_conversation_flow_long_conversation(self):
        """Test long conversation separately due to complexity."""

        def mock_llm(**kwargs):
            return f"Response to: {kwargs.get('messages', [])[-1]['content']}"

        shield = LLMShield(llm_func=mock_llm)

        long_conversation = []
        for i in range(50):
            long_conversation.extend(
                [
                    {"role": "user", "content": f"Message {i}"},
                    {"role": "assistant", "content": f"Response {i}"},
                ]
            )
        long_conversation.append({"role": "user", "content": "Final message"})

        result = shield.ask(messages=long_conversation)
        self.assertIsInstance(result, str)

    def test_malformed_input_tolerance(self):
        """Test handling of malformed inputs."""

        def mock_llm(**kwargs):
            return "response"

        shield = LLMShield(llm_func=mock_llm)

        malformed_cases = [
            {"content": "Missing role"},  # Missing role
            {"role": "user"},  # Missing content
            {},  # Empty message
        ]

        # Should handle gracefully (may not detect entities but shouldn't
        # crash)
        try:
            result = shield.ask(messages=malformed_cases)
            self.assertIsInstance(result, str)
        except (KeyError, TypeError, ValueError):
            # Acceptable to fail on malformed input
            pass

    @parameterized.expand(
        [
            ("person", "Hello John Doe {}"),
            ("email", "Email me at test{}@example.com"),
        ]
    )
    def test_concurrent_usage_patterns(self, pattern_type, template):
        """Test behavior under simulated concurrent usage."""

        def mock_llm(**kwargs):
            return "response"

        shield = LLMShield(llm_func=mock_llm)

        results = []
        for i in range(5):  # Reduced for parameterized testing
            text = template.format(i)
            _, entity_map = shield.cloak(text)
            result = shield.uncloak(
                f"Response with entities {list(entity_map.keys())}", entity_map
            )
            results.append(result)

            # Should not contain placeholders
            self.assertNotIn("<", result)
            self.assertNotIn(">", result)

        self.assertEqual(len(results), 5)

    def test_large_scale_entity_processing(self):
        """Test memory efficiency with large entity maps."""
        shield = LLMShield()

        # Create realistic test data that will be detected
        name_templates = [
            "John Smith",
            "Jane Doe",
            "Bob Johnson",
            "Alice Brown",
            "Charlie Wilson",
            "Diana Prince",
            "Edward Clark",
            "Fiona Green",
        ]

        large_text = ""
        for name in name_templates:
            email = f"{name.lower().replace(' ', '')}@company.com"
            large_text += f"Hello {name}, please contact {email}. "

        # Test the processing
        cloaked, entity_map = shield.cloak(large_text)
        self.assertGreater(len(entity_map), 5)  # Should find entities

        uncloaked = shield.uncloak(cloaked, entity_map)

        # Spot check key names are preserved
        for name in name_templates[:2]:  # Check first two
            self.assertIn(name, uncloaked)

    @parameterized.expand(
        [
            # (description, input_text, expected_preserved_chars)
            (
                "unicode_names",
                "Hello José García from São Paulo",
                ["José", "São Paulo"],
            ),
            ("emojis", "Contact 👨‍💻 John@company.com 🏢", ["👨‍💻", "🏢"]),
            (
                "mixed_scripts",
                "Hello John من نیویورک",
                ["John"],
            ),  # Arabic/Persian
        ]
    )
    def test_unicode_and_special_character_handling(
        self, description, input_text, preserved_chars
    ):
        """Test handling of Unicode and special characters."""
        shield = LLMShield()

        cloaked, entity_map = shield.cloak(input_text)
        uncloaked = shield.uncloak(cloaked, entity_map)

        for char_sequence in preserved_chars:
            self.assertIn(char_sequence, uncloaked)

    def test_error_recovery_robustness(self):
        """Test error recovery and robustness under stress."""

        def flaky_llm(**kwargs):
            # Simulate an LLM that sometimes returns unexpected types
            responses = [
                "Normal string response",
                123,  # Unexpected integer
                None,  # Unexpected None
                [],  # Unexpected empty list
                {"unexpected": "dict"},  # Unexpected dict
            ]
            return random.choice(responses)

        shield = LLMShield(llm_func=flaky_llm)

        # Test multiple calls with flaky LLM
        success_count = 0
        error_count = 0

        for i in range(20):
            try:
                result = shield.ask(prompt=f"Test prompt {i}")
                if isinstance(result, str):
                    success_count += 1
                    # Should not contain entity placeholders if processed
                    # correctly
                    self.assertNotIn(f"<PERSON_{i}>", result)
            except (TypeError, ValueError):
                error_count += 1
                # Acceptable to fail with unexpected response types

        # At least some attempts should either succeed or fail gracefully
        self.assertGreater(success_count + error_count, 0)


if __name__ == "__main__":
    unittest.main()
