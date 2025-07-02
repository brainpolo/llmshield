"""Comprehensive utility tests covering all edge cases.

Description:
    This test module provides exhaustive testing for all utility functions,
    covering edge cases, error conditions, and complex scenarios not covered
    in the basic utility tests.

Test Classes:
    - TestConversationHashComprehensive: Advanced conversation hashing tests
    - TestWrapEntityComprehensive: Advanced entity wrapping tests
    - TestNormaliseSpaces: Tests space normalisation
    - TestSplitFragments: Tests text fragment splitting
    - TestIsValidDelimiter: Tests delimiter validation
    - TestIsValidStreamResponse: Tests streaming response validation
    - TestShouldCloakInputAdvanced: Advanced input cloaking tests
    - TestProcessInput: Tests input processing logic
    - TestPydanticLike: Tests Pydantic-like model handling
    - TestAskHelper: Tests ask helper functionality

Author:
    LLMShield by brainpolo, 2025
"""

import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

from llmshield.entity_detector import EntityType
from llmshield.utils import (
    PydanticLike,
    _should_cloak_input,
    ask_helper,
    conversation_hash,
    is_valid_delimiter,
    is_valid_stream_response,
    normalise_spaces,
    split_fragments,
    wrap_entity,
)


class MockPydantic:
    """Mock Pydantic-like class for testing."""

    def model_dump(self) -> dict:
        """Return test data as dict."""
        return {"test": "data"}

    @classmethod
    def model_validate(cls, data: dict):
        """Create instance from dict."""
        return cls()


class TestUtilsComprehensive(unittest.TestCase):
    """Comprehensive tests for utility functions."""

    def test_split_fragments_empty_string(self):
        """Test split_fragments with empty string."""
        result = split_fragments("")
        self.assertEqual(result, [])

    def test_split_fragments_no_punctuation(self):
        """Test split_fragments with no sentence boundaries."""
        result = split_fragments("This is a test")
        self.assertEqual(result, ["This is a test"])

    def test_split_fragments_multiple_punctuation(self):
        """Test split_fragments with multiple punctuation marks."""
        text = "First sentence!! Second sentence??? Third sentence..."
        result = split_fragments(text)
        # The last sentence retains the ... since they're at the end
        self.assertEqual(
            result, ["First sentence", "Second sentence", "Third sentence..."]
        )

    def test_split_fragments_newlines(self):
        """Test split_fragments with newlines."""
        text = "First line\n\nSecond line\nThird line"
        result = split_fragments(text)
        self.assertEqual(result, ["First line", "Second line", "Third line"])

    def test_split_fragments_mixed_boundaries(self):
        """Test split_fragments with mixed sentence boundaries."""
        text = "First sentence! Second line\nThird sentence? Fourth."
        result = split_fragments(text)
        # The last fragment retains the period since it's at the end
        self.assertEqual(
            result,
            ["First sentence", "Second line", "Third sentence", "Fourth."],
        )

    def test_is_valid_delimiter_valid(self):
        """Test is_valid_delimiter with valid delimiters."""
        self.assertTrue(is_valid_delimiter("<"))
        self.assertTrue(is_valid_delimiter(">>"))
        self.assertTrue(is_valid_delimiter("|||"))
        self.assertTrue(is_valid_delimiter(" "))

    def test_is_valid_delimiter_invalid(self):
        """Test is_valid_delimiter with invalid delimiters."""
        self.assertFalse(is_valid_delimiter(""))
        self.assertFalse(is_valid_delimiter(None))
        self.assertFalse(is_valid_delimiter(123))
        self.assertFalse(is_valid_delimiter([]))

    def test_wrap_entity_basic(self):
        """Test wrap_entity with basic inputs."""
        result = wrap_entity(EntityType.PERSON, 0, "<", ">")
        self.assertEqual(result, "<PERSON_0>")

        result = wrap_entity(EntityType.EMAIL, 5, "[", "]")
        self.assertEqual(result, "[EMAIL_5]")

    def test_wrap_entity_custom_delimiters(self):
        """Test wrap_entity with custom delimiters."""
        result = wrap_entity(EntityType.ORGANISATION, 1, "{{", "}}")
        self.assertEqual(result, "{{ORGANISATION_1}}")

        result = wrap_entity(EntityType.PLACE, 10, "***", "***")
        self.assertEqual(result, "***PLACE_10***")

    def test_normalise_spaces_multiple_spaces(self):
        """Test normalise_spaces with multiple spaces."""
        result = normalise_spaces("Hello    world   test")
        self.assertEqual(result, "Hello world test")

    def test_normalise_spaces_tabs_and_newlines(self):
        """Test normalise_spaces with tabs and newlines."""
        result = normalise_spaces("Hello\t\tworld\n\ntest")
        self.assertEqual(result, "Hello world test")

    def test_normalise_spaces_leading_trailing(self):
        """Test normalise_spaces with leading and trailing whitespace."""
        result = normalise_spaces("   Hello world   ")
        self.assertEqual(result, "Hello world")

    def test_normalise_spaces_empty_string(self):
        """Test normalise_spaces with empty string."""
        result = normalise_spaces("")
        self.assertEqual(result, "")

    def test_normalise_spaces_only_whitespace(self):
        """Test normalise_spaces with only whitespace."""
        result = normalise_spaces("   \t\n   ")
        self.assertEqual(result, "")

    def test_is_valid_stream_response_valid_iterables(self):
        """Test is_valid_stream_response with valid iterables."""
        self.assertTrue(is_valid_stream_response([1, 2, 3]))
        self.assertTrue(is_valid_stream_response((1, 2, 3)))
        self.assertTrue(is_valid_stream_response({1, 2, 3}))
        self.assertTrue(is_valid_stream_response(range(5)))
        self.assertTrue(is_valid_stream_response(iter([1, 2, 3])))

    def test_is_valid_stream_response_excluded_types(self):
        """Test is_valid_stream_response with excluded types."""
        self.assertFalse(is_valid_stream_response("string"))
        self.assertFalse(is_valid_stream_response(b"bytes"))
        self.assertFalse(is_valid_stream_response(bytearray(b"bytearray")))
        self.assertFalse(is_valid_stream_response({"key": "value"}))
        self.assertFalse(is_valid_stream_response(123))
        self.assertFalse(is_valid_stream_response(None))

    def test_pydantic_like_protocol(self):
        """Test PydanticLike protocol."""
        mock_pydantic = MockPydantic()

        # Should satisfy the protocol
        self.assertIsInstance(mock_pydantic, PydanticLike)
        self.assertEqual(mock_pydantic.model_dump(), {"test": "data"})
        self.assertIsInstance(MockPydantic.model_validate({}), MockPydantic)

    def test_should_cloak_input_string(self):
        """Test _should_cloak_input with string input."""
        self.assertTrue(_should_cloak_input("test string"))
        self.assertTrue(_should_cloak_input(""))

    def test_should_cloak_input_list(self):
        """Test _should_cloak_input with list input."""
        self.assertTrue(_should_cloak_input(["test", "list"]))
        self.assertTrue(_should_cloak_input([]))
        self.assertTrue(_should_cloak_input([1, 2, 3]))  # Any list type

    def test_should_cloak_input_other_types(self):
        """Test _should_cloak_input with other types."""
        self.assertFalse(_should_cloak_input({"key": "value"}))
        self.assertFalse(_should_cloak_input(Path("/test/path")))
        self.assertFalse(_should_cloak_input(BytesIO()))
        self.assertFalse(_should_cloak_input(b"bytes"))
        self.assertFalse(_should_cloak_input(("tuple", "data")))
        self.assertFalse(_should_cloak_input(MockPydantic()))
        self.assertFalse(_should_cloak_input(123))
        self.assertFalse(_should_cloak_input(None))

    def test_conversation_hash_single_message(self):
        """Test conversation_hash with single message."""
        message = {"role": "user", "content": "Hello world"}
        hash1 = conversation_hash(message)

        # Same message should produce same hash
        hash2 = conversation_hash({"role": "user", "content": "Hello world"})
        self.assertEqual(hash1, hash2)

        # Different message should produce different hash
        hash3 = conversation_hash(
            {"role": "user", "content": "Different content"}
        )
        self.assertNotEqual(hash1, hash3)

    def test_conversation_hash_missing_keys(self):
        """Test conversation_hash with missing role/content keys."""
        message = {"role": "user"}  # Missing content
        hash1 = conversation_hash(message)

        message2 = {"content": "Hello"}  # Missing role
        hash2 = conversation_hash(message2)

        message3 = {}  # Missing both
        hash3 = conversation_hash(message3)

        # Should handle missing keys gracefully
        self.assertIsInstance(hash1, int)
        self.assertIsInstance(hash2, int)
        self.assertIsInstance(hash3, int)

    def test_conversation_hash_message_list(self):
        """Test conversation_hash with list of messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        hash1 = conversation_hash(messages)

        # Same messages in different order should produce same hash (frozenset)
        messages2 = [
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Hello"},
        ]
        hash2 = conversation_hash(messages2)
        self.assertEqual(hash1, hash2)

        # Different messages should produce different hash
        messages3 = [
            {"role": "user", "content": "Different"},
            {"role": "assistant", "content": "Hi there"},
        ]
        hash3 = conversation_hash(messages3)
        self.assertNotEqual(hash1, hash3)

    def test_conversation_hash_empty_list(self):
        """Test conversation_hash with empty list."""
        hash_result = conversation_hash([])
        self.assertIsInstance(hash_result, int)

    def test_ask_helper_no_cloaking_needed(self):
        """Test ask_helper when no cloaking is needed."""
        mock_shield = Mock()
        mock_llm_func = Mock(return_value="Direct response")
        mock_shield.llm_func = mock_llm_func

        # Non-string input shouldn't be cloaked
        kwargs = {"prompt": {"key": "value"}, "model": "test"}

        result = ask_helper(mock_shield, stream=False, **kwargs)

        # Should call LLM directly without cloaking
        mock_llm_func.assert_called_once_with(**kwargs)
        self.assertEqual(result, "Direct response")
        mock_shield.cloak.assert_not_called()

    def test_ask_helper_message_param(self):
        """Test ask_helper with 'message' parameter instead of 'prompt'."""
        mock_shield = Mock()
        mock_shield.cloak.return_value = ("cloaked", {"<PERSON_0>": "John"})
        mock_shield.uncloak.return_value = "uncloaked response"
        mock_shield.llm_func = Mock(return_value="llm response")

        # Mock provider
        mock_provider = Mock()
        mock_provider.prepare_single_message_params.return_value = (
            {"message": "cloaked", "model": "test"},
            False,
        )

        with unittest.mock.patch(
            "llmshield.utils.get_provider", return_value=mock_provider
        ):
            kwargs = {"message": "Hello John", "model": "test"}
            result = ask_helper(mock_shield, stream=False, **kwargs)

        # Should detect 'message' parameter
        mock_shield.cloak.assert_called_once_with("Hello John")
        self.assertEqual(result, "uncloaked response")

    def test_ask_helper_streaming_invalid_response(self):
        """Test ask_helper when LLM returns invalid stream response."""
        mock_shield = Mock()
        mock_shield.cloak.return_value = ("cloaked", {"<PERSON_0>": "John"})
        mock_shield.uncloak.return_value = "uncloaked response"
        mock_shield.llm_func = Mock(
            return_value="not_a_stream"
        )  # String, not iterable

        # Mock provider to indicate streaming
        mock_provider = Mock()
        mock_provider.prepare_single_message_params.return_value = (
            {"prompt": "cloaked"},
            True,  # actual_stream=True
        )

        with unittest.mock.patch(
            "llmshield.utils.get_provider", return_value=mock_provider
        ):
            result = ask_helper(mock_shield, stream=True, prompt="Hello John")

        # Should convert to iterator when LLM doesn't return valid stream
        self.assertIsInstance(result, type(iter([])))
        result_list = list(result)
        self.assertEqual(result_list, ["uncloaked response"])

    def test_ask_helper_streaming_valid_response(self):
        """Test ask_helper with valid streaming response."""
        mock_shield = Mock()
        mock_shield.cloak.return_value = ("cloaked", {"<PERSON_0>": "John"})
        mock_shield.stream_uncloak.return_value = iter(["chunk1", "chunk2"])
        mock_shield.llm_func = Mock(return_value=["stream", "chunks"])

        # Mock provider
        mock_provider = Mock()
        mock_provider.prepare_single_message_params.return_value = (
            {"prompt": "cloaked"},
            True,
        )

        with unittest.mock.patch(
            "llmshield.utils.get_provider", return_value=mock_provider
        ):
            result = ask_helper(mock_shield, stream=True, prompt="Hello John")

        # Should call stream_uncloak
        mock_shield.stream_uncloak.assert_called_once()
        self.assertIsInstance(result, type(iter([])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
