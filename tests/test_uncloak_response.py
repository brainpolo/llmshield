"""Tests for uncloak response functionality."""

import unittest

from llmshield.uncloak_response import _uncloak_response


class MockChatCompletion:
    """Mock ChatCompletion object similar to OpenAI's response."""

    def __init__(self, choices=None, model="gpt-4"):
        self.choices = choices or []
        self.model = model
        self.id = "test-completion-id"


class MockChoice:
    """Mock Choice object from ChatCompletion."""

    def __init__(self, message=None, delta=None):
        self.message = message
        self.delta = delta
        self.index = 0


class MockMessage:
    """Mock Message object from Choice."""

    def __init__(self, content=""):
        self.content = content
        self.role = "assistant"


class MockDelta:
    """Mock Delta object from streaming Choice."""

    def __init__(self, content=""):
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

        # Create a simple object that has model_dump method but not choices/model
        class MockPydantic:
            def model_dump(self):
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
        """Test uncloaking ChatCompletion object with delta content (streaming)."""
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
                    "contacts": {"email": "<EMAIL_0>", "location": "<PLACE_0>"},
                    "messages": ["Hello from <PERSON_0>", "Living in <PLACE_0>"],
                }
            ],
            "metadata": {"total": 1, "processed": True},
        }

        result = _uncloak_response(response, self.entity_map)

        expected = {
            "users": [
                {
                    "name": "John Doe",
                    "contacts": {"email": "john@example.com", "location": "New York"},
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


if __name__ == "__main__":
    unittest.main()
