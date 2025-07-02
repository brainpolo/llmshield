"""Test handling of OpenAI tool calls with None content.

Author:
    LLMShield by brainpolo, 2025
"""

import unittest

from llmshield import LLMShield


class TestToolCalls(unittest.TestCase):
    """Test handling of tool call messages."""

    def setUp(self):
        """Set up test fixtures."""
        self.shield = LLMShield()

    def test_validate_tool_call_messages(self):
        """Test validation of messages with tool calls."""
        # Messages with tool calls (None content)
        messages = [
            {"role": "user", "content": "What's 5+3?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": (
                                '{"operation": "add", "a": 5, "b": 3}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": "8",
                "tool_call_id": "call_123",
            },
        ]

        # Should not raise ValidationError
        try:
            # Mock LLM function
            def mock_llm(**kwargs):
                return "The answer is 8"

            self.shield.llm_func = mock_llm
            result = self.shield.ask(messages=messages)
            self.assertIsInstance(result, str)
        except Exception as e:
            self.fail(f"Tool call messages should be handled: {e}")

    def test_cloak_with_tool_calls(self):
        """Test cloaking preserves tool call structure."""
        messages = [
            {"role": "user", "content": "Call John at +1-555-1234"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_456",
                        "type": "function",
                        "function": {
                            "name": "make_call",
                            "arguments": '{"number": "+1-555-1234"}',
                        },
                    }
                ],
            },
        ]

        captured_messages = []

        def capture_messages(**kwargs):
            # Capture the cloaked messages
            nonlocal captured_messages
            captured_messages = kwargs.get("messages", [])
            return "Response after capturing messages"

        self.shield.llm_func = capture_messages
        self.shield.ask(messages=messages)

        # Check structure is preserved in captured messages
        self.assertEqual(len(captured_messages), 2)
        self.assertEqual(captured_messages[1]["role"], "assistant")
        self.assertIsNone(captured_messages[1]["content"])
        self.assertIn("tool_calls", captured_messages[1])
        self.assertEqual(
            captured_messages[1]["tool_calls"][0]["id"], "call_456"
        )

    def test_mixed_content_and_tool_calls(self):
        """Test handling of mixed content types."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {
                "role": "user",
                "content": "Calculate something for me",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_789", "type": "function"}],
            },
            {
                "role": "tool",
                "content": "Result: 42",
                "tool_call_id": "call_789",
            },
            {
                "role": "assistant",
                "content": "The calculation result is 42",
            },
        ]

        def mock_llm(**kwargs):
            return "Response processed"

        self.shield.llm_func = mock_llm
        # Should handle mixed content without errors
        result = self.shield.ask(messages=messages)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
