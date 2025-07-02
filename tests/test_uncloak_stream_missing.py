"""Test streaming uncloaking edge cases and buffer handling.

Description:
    This test module provides testing for edge cases in streaming response
    uncloaking, specifically focusing on buffer handling and incomplete
    placeholder scenarios to ensure full code coverage.

Test Classes:
    - TestUncloakStreamMissingLine: Tests edge cases in streaming uncloaking

Author:
    LLMShield by brainpolo, 2025
"""

import unittest
from unittest.mock import Mock

from llmshield.uncloak_stream_response import uncloak_stream_response


class TestUncloakStreamMissingLine(unittest.TestCase):
    """Test to hit missing line 72 in uncloak_stream_response.py."""

    def test_remaining_buffer_content_yield(self):
        """Test that remaining buffer content is yielded - line 72."""

        # Create a stream that ends with incomplete content (no placeholders)
        def mock_stream():
            yield "Hello "
            yield "world"
            # This should leave content in buffer that needs to be
            # yielded at end

        entity_map = {"<PERSON_0>": "John"}

        result = list(
            uncloak_stream_response(mock_stream(), entity_map=entity_map)
        )

        # Should yield content as it comes (streaming behavior)
        self.assertEqual(result, ["Hello ", "world"])

    def test_incomplete_placeholder_then_text(self):
        """Test incomplete placeholder followed by non-placeholder text."""

        def mock_stream():
            yield "<PERSON"  # Incomplete placeholder
            yield "_incomplete and then regular text"

        entity_map = {"<PERSON_0>": "John"}

        result = list(
            uncloak_stream_response(mock_stream(), entity_map=entity_map)
        )

        # Should yield the incomplete placeholder + text as-is
        self.assertEqual(result, ["<PERSON_incomplete and then regular text"])

    def test_openai_chunk_with_none_content(self):
        """Test OpenAI chunk structure with None content."""
        # Mock OpenAI ChatCompletionChunk with None content
        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta = Mock()
        mock_chunk.choices[0].delta.content = None  # This triggers line 38

        def mock_stream():
            yield mock_chunk
            yield "regular text"

        entity_map = {}

        result = list(
            uncloak_stream_response(mock_stream(), entity_map=entity_map)
        )

        # Should handle None content gracefully and yield remaining text
        self.assertEqual(result, ["regular text"])

    def test_buffer_with_whitespace_only(self):
        """Test buffer with only whitespace content."""

        def mock_stream():
            yield "   "  # Only whitespace
            yield "\t\n"  # More whitespace
            yield "actual content"

        entity_map = {}

        result = list(
            uncloak_stream_response(mock_stream(), entity_map=entity_map)
        )

        # Should yield all content including whitespace
        self.assertEqual(result, ["   \t\nactual content"])

    def test_empty_chunks_in_stream(self):
        """Test stream with empty chunks."""

        def mock_stream():
            yield ""
            yield "Hello"
            yield ""
            yield " World"
            yield ""

        entity_map = {}

        result = list(
            uncloak_stream_response(mock_stream(), entity_map=entity_map)
        )

        # Should handle empty chunks and yield content as it streams
        self.assertEqual(result, ["Hello", " World"])

    def test_placeholder_at_very_end(self):
        """Test placeholder that completes at the very end of stream."""

        def mock_stream():
            yield "Hello <PERSON_"
            yield "0>"  # Placeholder completes at end

        entity_map = {"<PERSON_0>": "John"}

        result = list(
            uncloak_stream_response(mock_stream(), entity_map=entity_map)
        )

        # Should properly uncloak the placeholder
        self.assertEqual(result, ["Hello ", "John"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
