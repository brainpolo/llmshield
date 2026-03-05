"""Test streaming response uncloaking and buffer handling.

Description:
    Tests for streaming response handling including chunk processing,
    buffer management, partial placeholders, and edge cases.

Author:
    LLMShield by brainpolo, 2025-2026
"""

import unittest

from parameterized import parameterized

from llmshield.uncloak_stream_response import uncloak_stream_response


class MockChatCompletionChunk:
    """Mock OpenAI ChatCompletionChunk for testing."""

    def __init__(self, content: str | None):
        """Initialise mock chunk with content."""
        self.choices = [
            type(
                "MockChoiceDelta",
                (),
                {"delta": type("MockDelta", (), {"content": content})()},
            )()
        ]


class TestStreamUncloaking(unittest.TestCase):
    """Test streaming response uncloaking."""

    def setUp(self):
        """Set up test fixtures."""
        self.entity_map = {"<PERSON_0>": "John"}

    def test_openai_chunk_content_extraction(self):
        """Test OpenAI ChatCompletionChunk content extraction."""

        def chunk_stream():
            yield MockChatCompletionChunk("Hello")
            yield MockChatCompletionChunk(None)
            yield MockChatCompletionChunk("<PERSON_0>")

        result = list(uncloak_stream_response(chunk_stream(), self.entity_map))
        self.assertEqual(result, ["Hello", "John"])

    def test_final_buffer_yield(self):
        """Test remaining buffer content is yielded at end of stream."""

        def chunk_stream():
            yield "Remaining text"

        result = list(uncloak_stream_response(chunk_stream(), self.entity_map))
        self.assertEqual(result, ["Remaining text"])

    def test_partial_placeholder_completes(self):
        """Test partial placeholder that completes across chunks."""

        def chunk_stream():
            yield "<PERSON"
            yield "_0>"
            yield " additional content at end"

        result = list(uncloak_stream_response(chunk_stream(), self.entity_map))
        self.assertEqual(result, ["John", " additional content at end"])

    def test_incomplete_placeholder_then_text(self):
        """Test incomplete placeholder followed by non-placeholder text."""

        def mock_stream():
            yield "<PERSON"
            yield "_incomplete and then regular text"

        result = list(uncloak_stream_response(mock_stream(), self.entity_map))
        self.assertEqual(result, ["<PERSON_incomplete and then regular text"])

    def test_placeholder_at_end_of_stream(self):
        """Test placeholder that completes at the very end."""

        def mock_stream():
            yield "Hello <PERSON_"
            yield "0>"

        result = list(uncloak_stream_response(mock_stream(), self.entity_map))
        self.assertEqual(result, ["Hello ", "John"])

    @parameterized.expand(
        [
            (
                "openai_chunk_with_none",
                [MockChatCompletionChunk(None)],
                [],
            ),
            (
                "openai_chunk_with_content",
                [MockChatCompletionChunk("Hello")],
                ["Hello"],
            ),
            (
                "mixed_chunk_types",
                [MockChatCompletionChunk("Hi "), "<PERSON_0>"],
                ["Hi ", "John"],
            ),
            (
                "openai_none_then_text",
                [MockChatCompletionChunk(None), "regular text"],
                ["regular text"],
            ),
        ]
    )
    def test_chunk_processing(self, _name, chunks, expected):
        """Test various chunk processing scenarios."""

        def chunk_stream():
            yield from chunks

        result = list(uncloak_stream_response(chunk_stream(), self.entity_map))
        self.assertEqual(result, expected)

    def test_empty_chunks_in_stream(self):
        """Test stream with empty chunks interspersed."""

        def mock_stream():
            yield ""
            yield "Hello"
            yield ""
            yield " World"
            yield ""

        result = list(uncloak_stream_response(mock_stream(), entity_map={}))
        self.assertEqual(result, ["Hello", " World"])

    def test_buffer_with_whitespace_only(self):
        """Test buffer with only whitespace content."""

        def mock_stream():
            yield "   "
            yield "\t\n"
            yield "actual content"

        result = list(uncloak_stream_response(mock_stream(), entity_map={}))
        self.assertEqual(result, ["   \t\nactual content"])


if __name__ == "__main__":
    unittest.main()
