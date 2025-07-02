"""Test streaming functionality and chunk processing.

Description:
    This test module provides testing for streaming response handling,
    including chunk processing, buffer management, and proper uncloaking
    of streamed content from LLM providers.

Test Classes:
    - MockChatCompletionChunk: Mock OpenAI streaming chunk
    - TestStreamingCoverage: Tests streaming response handling

Author: LLMShield by brainpolo, 2025
"""

# Standard library Imports
import unittest

# Third party Imports
from parameterized import parameterized

# Local Imports
from llmshield.uncloak_stream_response import uncloak_stream_response


class MockChatCompletionChunk:
    """Mock OpenAI ChatCompletionChunk for testing line 32."""

    def __init__(self, content: str | None):
        """Initialise mock chunk with content."""
        self.choices = [
            type(
                "MockChoiceDelta",
                (),
                {"delta": type("MockDelta", (), {"content": content})()},
            )()
        ]


class TestStreamingCoverage(unittest.TestCase):
    """Targeted tests for missing lines in uncloak_stream_response.py."""

    def setUp(self):
        """Set up test fixtures."""
        self.entity_map = {"<PERSON_0>": "John"}

    def test_openai_chunk_content_extraction_line_32(self):
        """Test OpenAI ChatCompletionChunk content extraction (line 32)."""

        def chunk_stream():
            # This should trigger: text = chunk.choices[0].delta.content or ""
            yield MockChatCompletionChunk("Hello")
            yield MockChatCompletionChunk(None)  # Tests the "or ''" part
            yield MockChatCompletionChunk("<PERSON_0>")

        result = list(uncloak_stream_response(chunk_stream(), self.entity_map))
        expected = ["Hello", "John"]
        self.assertEqual(result, expected)

    def test_final_buffer_yield_line_65(self):
        """Test final buffer yield when stream ends (line 65)."""

        def chunk_stream():
            # Send incomplete content that stays in buffer
            yield "Remaining text"

        result = list(uncloak_stream_response(chunk_stream(), self.entity_map))
        # This should trigger: if buffer: yield buffer
        self.assertEqual(result, ["Remaining text"])

    def test_final_buffer_yield_line_65_guaranteed(self):
        """Test final buffer yield when stream ends with remaining content.

        Covers line 65 for buffer handling validation.
        """

        def chunk_stream():
            # Send content that will remain in buffer at end without any
            # placeholders
            yield "This content stays in buffer"
            # No more chunks - this should trigger the final:
            # if buffer: yield buffer

        result = list(uncloak_stream_response(chunk_stream(), self.entity_map))

        # This should trigger line 65: if buffer: yield buffer
        expected = ["This content stays in buffer"]
        self.assertEqual(result, expected)

    @parameterized.expand(
        [
            ("openai_chunk_with_none", [MockChatCompletionChunk(None)], []),
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
        ]
    )
    def test_chunk_processing_comprehensive(
        self, description, chunks, expected
    ):
        """Test comprehensive chunk processing - parameterized."""

        def chunk_stream():
            yield from chunks

        result = list(uncloak_stream_response(chunk_stream(), self.entity_map))
        self.assertEqual(result, expected)

    def test_buffer_accumulation_and_final_yield_line_65(self):
        """Test buffer accumulation with final yield.

        Absolutely guaranteed to hit line 65 for comprehensive coverage.
        """

        def partial_placeholder_stream():
            # Start a placeholder but don't complete it, leaving content in
            # buffer
            yield "<PERSON"
            yield "_0>"
            yield " additional content at end"  # This will be in buffer at end

        result = list(
            uncloak_stream_response(
                partial_placeholder_stream(), self.entity_map
            )
        )

        # Should uncloak the complete placeholder and yield remaining buffer
        # This MUST hit line 65: if buffer: yield buffer
        # Note: self.entity_map maps "<PERSON_0>" to "John" (not "John Doe")
        expected = ["John", " additional content at end"]
        self.assertEqual(result, expected)

    def test_empty_buffer_edge_case_line_65(self):
        """Test empty buffer handling to potentially hit line 65."""

        def empty_chunk_stream():
            yield ""  # Empty chunk
            yield "Final content"  # This should remain in buffer

        result = list(
            uncloak_stream_response(empty_chunk_stream(), self.entity_map)
        )

        # Final content should be yielded via line 65
        expected = ["Final content"]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
