"""Test response uncloaking for various LLM response formats.

Description:
    This test module provides comprehensive testing for uncloaking responses
    from different LLM providers, handling various response formats including
    chat completions, tool calls, and edge cases.

Test Classes:
    - MockChatCompletion: Mock OpenAI chat completion response
    - MockChoice: Mock response choice object
    - MockMessage: Mock message object
    - MockFunctionCall: Mock function call object
    - MockToolCall: Mock tool call object
    - MockFunction: Mock function object
    - TestUncloakResponse: Tests response uncloaking logic

Author: LLMShield by brainpolo, 2025
"""

import unittest

from parameterized import parameterized

from llmshield.uncloak_response import _uncloak_response


class MockChatCompletion:
    """Mock ChatCompletion object similar to OpenAI's response."""

    def __init__(self, choices=None, model="gpt-4"):
        """Initialise mock completion with choices and model."""
        self.choices = choices or []
        self.model = model
        self.id = "test-completion-id"


class MockChoice:
    """Mock Choice object from ChatCompletion."""

    def __init__(self, message=None, delta=None):
        """Initialise mock choice with message or delta."""
        self.message = message
        self.delta = delta
        self.index = 0


class MockMessage:
    """Mock Message object from Choice."""

    def __init__(self, content=""):
        """Initialise mock message with content."""
        self.content = content
        self.role = "assistant"


class MockDelta:
    """Mock Delta object from streaming Choice."""

    def __init__(self, content=""):
        """Initialise mock delta with content."""
        self.content = content


class TestUnclokResponse(unittest.TestCase):
    """Test uncloak response functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.entity_map = {
            "<PERSON_0>": "John Doe",
            "<EMAIL_0>": "john@example.com",
            "<PLACE_0>": "New York",
        }

    def test_uncloak_empty_entity_map(self):
        """Test uncloaking with empty entity map returns original response."""
        response = "Hello <PERSON_0>, how are you?"
        result = _uncloak_response(response, {})
        self.assertEqual(result, response)

    def test_uncloak_string_response(self):
        """Test uncloaking string response."""
        response = "Hello <PERSON_0>, email me at <EMAIL_0>"
        result = _uncloak_response(response, self.entity_map)
        expected = "Hello John Doe, email me at john@example.com"
        self.assertEqual(result, expected)

    def test_uncloak_list_response(self):
        """Test uncloaking list response."""
        response = [
            "Hello <PERSON_0>",
            {"message": "Contact <EMAIL_0>"},
            ["Nested <PLACE_0>"],
        ]
        result = _uncloak_response(response, self.entity_map)
        expected = [
            "Hello John Doe",
            {"message": "Contact john@example.com"},
            ["Nested New York"],
        ]
        self.assertEqual(result, expected)

    def test_uncloak_dict_response(self):
        """Test uncloaking dictionary response."""
        response = {
            "greeting": "Hello <PERSON_0>",
            "contact": {"email": "<EMAIL_0>", "location": "<PLACE_0>"},
            "count": 42,  # Non-string value should remain unchanged
        }
        result = _uncloak_response(response, self.entity_map)
        expected = {
            "greeting": "Hello John Doe",
            "contact": {"email": "john@example.com", "location": "New York"},
            "count": 42,
        }
        self.assertEqual(result, expected)

    def test_uncloak_pydantic_like_object(self):
        """Test uncloaking Pydantic-like object."""

        # Create a simple object that has model_dump method but not
        # choices/model
        class MockPydantic:
            @staticmethod
            def model_dump() -> dict:
                return {"name": "<PERSON_0>", "email": "<EMAIL_0>"}

            @classmethod
            def model_validate(cls, data):
                return cls()

        mock_pydantic = MockPydantic()

        result = _uncloak_response(mock_pydantic, self.entity_map)
        expected = {"name": "John Doe", "email": "john@example.com"}
        self.assertEqual(result, expected)

    def test_uncloak_chatcompletion_with_message_content(self):
        """Test uncloaking ChatCompletion object with message content."""
        # Create mock ChatCompletion with message content
        message = MockMessage(content="Hello <PERSON_0>, visit <PLACE_0>")
        choice = MockChoice(message=message)
        chatcompletion = MockChatCompletion(choices=[choice])

        result = _uncloak_response(chatcompletion, self.entity_map)

        # Verify it's a different object (deep copy)
        self.assertIsNot(result, chatcompletion)

        # Verify content was uncloaked
        self.assertEqual(
            result.choices[0].message.content, "Hello John Doe, visit New York"
        )

        # Verify original object is unchanged
        self.assertEqual(
            chatcompletion.choices[0].message.content,
            "Hello <PERSON_0>, visit <PLACE_0>",
        )

    def test_uncloak_chatcompletion_with_delta_content(self):
        """Test uncloaking ChatCompletion object with delta content.

        Validates streaming response handling for delta content.
        """
        # Create mock ChatCompletion with delta content
        delta = MockDelta(content="Hello <PERSON_0>")
        choice = MockChoice(delta=delta)
        chatcompletion = MockChatCompletion(choices=[choice])

        result = _uncloak_response(chatcompletion, self.entity_map)

        # Verify it's a different object (deep copy)
        self.assertIsNot(result, chatcompletion)

        # Verify delta content was uncloaked
        self.assertEqual(result.choices[0].delta.content, "Hello John Doe")

    def test_uncloak_chatcompletion_with_none_content(self):
        """Test uncloaking ChatCompletion object with None content."""
        # Create mock ChatCompletion with None content
        message = MockMessage(content=None)
        choice = MockChoice(message=message)
        chatcompletion = MockChatCompletion(choices=[choice])

        result = _uncloak_response(chatcompletion, self.entity_map)

        # Verify None content is preserved
        self.assertIsNone(result.choices[0].message.content)

    def test_uncloak_chatcompletion_with_tool_calls_none_content(self):
        """Test uncloaking ChatCompletion with tool calls and None content.

        This simulates the scenario where an assistant message has tool calls
        but no text content, which results in content being None.
        """
        # Create mock ChatCompletion with None content (simulating tool calls)
        message = MockMessage(content=None)
        choice = MockChoice(message=message)
        chatcompletion = MockChatCompletion(choices=[choice])

        # This should not raise an error
        result = _uncloak_response(chatcompletion, self.entity_map)

        # Verify None content is preserved and no error occurred
        self.assertIsNone(result.choices[0].message.content)
        # Verify it's a different object (deep copy)
        self.assertIsNot(result, chatcompletion)

    def test_uncloak_chatcompletion_with_empty_choices(self):
        """Test uncloaking ChatCompletion object with empty choices."""
        chatcompletion = MockChatCompletion(choices=[])

        result = _uncloak_response(chatcompletion, self.entity_map)

        # Verify it's a different object (deep copy)
        self.assertIsNot(result, chatcompletion)

        # Verify empty choices are preserved
        self.assertEqual(len(result.choices), 0)

    def test_uncloak_chatcompletion_multiple_choices(self):
        """Test uncloaking ChatCompletion object with multiple choices."""
        # Create multiple choices
        message1 = MockMessage(content="Hello <PERSON_0>")
        message2 = MockMessage(content="Visit <PLACE_0>")
        choice1 = MockChoice(message=message1)
        choice2 = MockChoice(message=message2)
        chatcompletion = MockChatCompletion(choices=[choice1, choice2])

        result = _uncloak_response(chatcompletion, self.entity_map)

        # Verify both choices were processed
        self.assertEqual(result.choices[0].message.content, "Hello John Doe")
        self.assertEqual(result.choices[1].message.content, "Visit New York")

    def test_uncloak_chatcompletion_without_message_attribute(self):
        """Test uncloaking ChatCompletion choice without message attribute."""

        # Create choice without message attribute
        class MockChoiceNoMessage:
            def __init__(self):
                self.index = 0
                # Deliberately don't set message attribute

        choice = MockChoiceNoMessage()
        chatcompletion = MockChatCompletion(choices=[choice])

        result = _uncloak_response(chatcompletion, self.entity_map)

        # Should not raise error, just skip processing
        self.assertIsNot(result, chatcompletion)

    def test_uncloak_chatcompletion_without_content_attribute(self):
        """Test uncloaking ChatCompletion message without content attribute."""

        # Create message without content attribute
        class MockMessageNoContent:
            def __init__(self):
                self.role = "assistant"
                # Deliberately don't set content attribute

        message = MockMessageNoContent()
        choice = MockChoice(message=message)
        chatcompletion = MockChatCompletion(choices=[choice])

        result = _uncloak_response(chatcompletion, self.entity_map)

        # Should not raise error, just skip processing
        self.assertIsNot(result, chatcompletion)

    def test_uncloak_non_chatcompletion_object_with_choices(self):
        """Test uncloaking object that has choices but not model attribute."""

        # Create object with choices but no model attribute
        class MockObjectWithChoices:
            def __init__(self):
                self.choices = []
                # Deliberately don't set model attribute

        mock_obj = MockObjectWithChoices()

        result = _uncloak_response(mock_obj, self.entity_map)

        # Should return original object unchanged
        self.assertEqual(result, mock_obj)

    def test_uncloak_non_supported_type(self):
        """Test uncloaking non-supported data type."""
        # Test with integer
        result = _uncloak_response(42, self.entity_map)
        self.assertEqual(result, 42)

        # Test with float
        result = _uncloak_response(3.14, self.entity_map)
        self.assertEqual(result, 3.14)

        # Test with boolean
        result = _uncloak_response(True, self.entity_map)
        self.assertEqual(result, True)

    def test_uncloak_complex_nested_structure(self):
        """Test uncloaking complex nested data structure."""
        response = {
            "users": [
                {
                    "name": "<PERSON_0>",
                    "contacts": {
                        "email": "<EMAIL_0>",
                        "location": "<PLACE_0>",
                    },
                    "messages": [
                        "Hello from <PERSON_0>",
                        "Living in <PLACE_0>",
                    ],
                }
            ],
            "metadata": {"total": 1, "processed": True},
        }

        result = _uncloak_response(response, self.entity_map)

        expected = {
            "users": [
                {
                    "name": "John Doe",
                    "contacts": {
                        "email": "john@example.com",
                        "location": "New York",
                    },
                    "messages": ["Hello from John Doe", "Living in New York"],
                }
            ],
            "metadata": {"total": 1, "processed": True},
        }

        self.assertEqual(result, expected)

    def test_uncloak_preserves_object_structure(self):
        """Test that uncloaking preserves the original object structure."""
        # Create a complex ChatCompletion-like object
        message = MockMessage(content="Hello <PERSON_0>")
        choice = MockChoice(message=message)
        chatcompletion = MockChatCompletion(choices=[choice], model="gpt-4")
        chatcompletion.additional_field = "test_value"

        result = _uncloak_response(chatcompletion, self.entity_map)

        # Verify structure is preserved
        self.assertEqual(result.model, "gpt-4")
        self.assertEqual(result.id, "test-completion-id")
        self.assertEqual(result.additional_field, "test_value")
        self.assertEqual(len(result.choices), 1)
        self.assertEqual(result.choices[0].index, 0)

    @parameterized.expand(
        [
            # (description, response, entity_map, expected)
            (
                "simple_string",
                "Hello <PERSON_0>",
                {"<PERSON_0>": "Alice"},
                "Hello Alice",
            ),
            (
                "multiple_entities",
                "Contact <PERSON_0> at <EMAIL_0>",
                {"<PERSON_0>": "Bob", "<EMAIL_0>": "bob@test.com"},
                "Contact Bob at bob@test.com",
            ),
            (
                "no_entities",
                "Plain text",
                {"<PERSON_0>": "Alice"},
                "Plain text",
            ),
            ("empty_string", "", {"<PERSON_0>": "Alice"}, ""),
            (
                "repeated_entity",
                "<PERSON_0> and <PERSON_0>",
                {"<PERSON_0>": "Charlie"},
                "Charlie and Charlie",
            ),
            (
                "mixed_delimiters",
                "Hi <PERSON_0> and [EMAIL_0]",
                {"<PERSON_0>": "Dave"},
                "Hi Dave and [EMAIL_0]",
            ),
            (
                "unicode_entities",
                "Greetings <PERSON_0> 🎉",
                {"<PERSON_0>": "José"},
                "Greetings José 🎉",
            ),
            (
                "special_chars_in_replacement",
                "Hello <PERSON_0>",
                {"<PERSON_0>": "O'Connor & Smith"},
                "Hello O'Connor & Smith",
            ),
        ]
    )
    def test_uncloak_string_variations(
        self, description, response, entity_map, expected
    ):
        """Test uncloaking various string scenarios - parameterized."""
        result = _uncloak_response(response, entity_map)
        self.assertEqual(result, expected)

    @parameterized.expand(
        [
            # (description, response, entity_map, expected)
            (
                "simple_list",
                ["Hello <PERSON_0>"],
                {"<PERSON_0>": "Alice"},
                ["Hello Alice"],
            ),
            (
                "nested_list",
                [["<PERSON_0>", "<EMAIL_0>"]],
                {"<PERSON_0>": "Bob", "<EMAIL_0>": "bob@test.com"},
                [["Bob", "bob@test.com"]],
            ),
            (
                "mixed_types_list",
                ["<PERSON_0>", 42, True, None],
                {"<PERSON_0>": "Charlie"},
                ["Charlie", 42, True, None],
            ),
            ("empty_list", [], {"<PERSON_0>": "Alice"}, []),
            (
                "list_with_dicts",
                [{"name": "<PERSON_0>"}],
                {"<PERSON_0>": "Dave"},
                [{"name": "Dave"}],
            ),
            (
                "deeply_nested",
                [[["<PERSON_0>"]]],
                {"<PERSON_0>": "Eve"},
                [[["Eve"]]],
            ),
        ]
    )
    def test_uncloak_list_variations(
        self, description, response, entity_map, expected
    ):
        """Test uncloaking various list scenarios - parameterized."""
        result = _uncloak_response(response, entity_map)
        self.assertEqual(result, expected)

    @parameterized.expand(
        [
            # (description, response, entity_map, expected)
            (
                "simple_dict",
                {"name": "<PERSON_0>"},
                {"<PERSON_0>": "Alice"},
                {"name": "Alice"},
            ),
            (
                "nested_dict",
                {"user": {"name": "<PERSON_0>", "email": "<EMAIL_0>"}},
                {"<PERSON_0>": "Bob", "<EMAIL_0>": "bob@test.com"},
                {"user": {"name": "Bob", "email": "bob@test.com"}},
            ),
            (
                "dict_with_non_string_values",
                {"name": "<PERSON_0>", "age": 30, "active": True},
                {"<PERSON_0>": "Charlie"},
                {"name": "Charlie", "age": 30, "active": True},
            ),
            ("empty_dict", {}, {"<PERSON_0>": "Alice"}, {}),
            (
                "dict_with_list_values",
                {"names": ["<PERSON_0>", "<PERSON_1>"]},
                {"<PERSON_0>": "Dave", "<PERSON_1>": "Eve"},
                {"names": ["Dave", "Eve"]},
            ),
            (
                "keys_not_uncloaked",
                {"<PERSON_0>": "value"},
                {"<PERSON_0>": "Alice"},
                {"<PERSON_0>": "value"},
            ),
        ]
    )
    def test_uncloak_dict_variations(
        self, description, response, entity_map, expected
    ):
        """Test uncloaking various dictionary scenarios - parameterized."""
        result = _uncloak_response(response, entity_map)
        self.assertEqual(result, expected)

    @parameterized.expand(
        [
            # (description, response_type, content, expected_content)
            (
                "string_content",
                "message",
                "Hello <PERSON_0>",
                "Hello John Doe",
            ),
            ("none_content", "message", None, None),
            ("empty_content", "message", "", ""),
            ("string_delta", "delta", "Hello <PERSON_0>", "Hello John Doe"),
            ("none_delta", "delta", None, None),
            ("empty_delta", "delta", "", ""),
        ]
    )
    def test_uncloak_chatcompletion_variations(
        self, description, response_type, content, expected_content
    ):
        """Test uncloaking ChatCompletion content types - parameterized."""
        if response_type == "message":
            message = MockMessage(content=content)
            choice = MockChoice(message=message)
        else:  # delta
            delta = MockDelta(content=content)
            choice = MockChoice(delta=delta)

        chatcompletion = MockChatCompletion(choices=[choice])
        result = _uncloak_response(chatcompletion, self.entity_map)

        if response_type == "message":
            self.assertEqual(
                result.choices[0].message.content, expected_content
            )
        else:
            self.assertEqual(result.choices[0].delta.content, expected_content)

    @parameterized.expand(
        [
            # (description, response, expected_unchanged)
            ("integer", 42, True),
            ("float", 3.14159, True),
            ("boolean_true", True, True),
            ("boolean_false", False, True),
            ("none_value", None, True),
            ("bytes", b"binary_data", True),
            ("complex_number", 1 + 2j, True),
            ("set", {1, 2, 3}, True),
            ("tuple", (1, 2, 3), True),
        ]
    )
    def test_uncloak_non_supported_types(
        self, description, response, expected_unchanged
    ):
        """Test uncloaking non-supported data types - parameterized."""
        result = _uncloak_response(response, self.entity_map)
        if expected_unchanged:
            self.assertEqual(result, response)

    @parameterized.expand(
        [
            # (description, entity_map, should_return_original)
            ("empty_map", {}, True),
            # This would fail validation before reaching _uncloak_response
            ("none_map", None, False),
            ("single_entity", {"<PERSON_0>": "Alice"}, False),
            (
                "multiple_entities",
                {"<PERSON_0>": "Alice", "<EMAIL_0>": "alice@test.com"},
                False,
            ),
            (
                "unicode_entities",
                {"<PERSON_0>": "José", "<PLACE_0>": "São Paulo"},
                False,
            ),
            ("special_chars", {"<PERSON_0>": "O'Connor & Smith"}, False),
        ]
    )
    def test_uncloak_entity_map_variations(
        self, description, entity_map, should_return_original
    ):
        """Test uncloaking with entity map variations - parameterized."""
        response = "Hello <PERSON_0> from <PLACE_0>"
        if entity_map is not None:
            result = _uncloak_response(response, entity_map)
            if should_return_original:
                self.assertEqual(result, response)
            else:
                self.assertNotEqual(result, response)

    @parameterized.expand(
        [
            # (description, complex_structure, expected_keys_preserved)
            (
                "api_response",
                {
                    "status": "success",
                    "data": {
                        "user": {"name": "<PERSON_0>", "email": "<EMAIL_0>"},
                        "messages": [
                            "Hello <PERSON_0>",
                            "Welcome to <PLACE_0>",
                        ],
                        "metadata": {"count": 2, "processed": True},
                    },
                },
                ["status", "data", "user", "messages", "metadata"],
            ),
            (
                "chatbot_conversation",
                [
                    {"role": "user", "content": "Hi, I'm <PERSON_0>"},
                    {
                        "role": "assistant",
                        "content": "Hello <PERSON_0>! How can I help?",
                    },
                    {"role": "user", "content": "I live in <PLACE_0>"},
                ],
                ["role", "content"],
            ),
            (
                "mixed_data_types",
                {
                    "strings": ["<PERSON_0>", "<EMAIL_0>"],
                    "numbers": [1, 2.5, 3],
                    "booleans": [True, False],
                    "nested": {"person": "<PERSON_0>", "active": True},
                },
                ["strings", "numbers", "booleans", "nested"],
            ),
        ]
    )
    def test_uncloak_complex_structures(
        self, description, complex_structure, expected_keys_preserved
    ):
        """Test uncloaking complex nested structures - parameterized."""
        result = _uncloak_response(complex_structure, self.entity_map)

        # Verify structure is preserved
        if isinstance(complex_structure, dict):
            for key in expected_keys_preserved:
                if key in complex_structure:
                    self.assertIn(key, result)
        elif isinstance(complex_structure, list):
            self.assertEqual(len(result), len(complex_structure))
            for item in result:
                if isinstance(item, dict):
                    for key in expected_keys_preserved:
                        if key in item:
                            self.assertIn(key, item)


if __name__ == "__main__":
    unittest.main()
