"""Test tool call handling in core module.

Author:
    LLMShield by brainpolo, 2025
"""

import json
import unittest
from unittest.mock import Mock

from llmshield import LLMShield


class TestCoreToolCalls(unittest.TestCase):
    """Test tool call cloaking in core module."""

    def test_cloak_dict_tool_calls(self):
        """Test cloaking of dict-based tool calls."""
        shield = LLMShield()

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "send_email",
                            "arguments": json.dumps(
                                {
                                    "to": "test@example.com",
                                    "body": "Call 555-123-4567",
                                }
                            ),
                        },
                    }
                ],
            }
        ]

        captured = []

        def capture(**kwargs):
            captured.append(kwargs.get("messages", []))
            return "OK"

        shield.llm_func = capture
        shield.ask(messages=messages)

        # Verify tool calls were processed
        sent_msg = captured[0][0]
        self.assertIn("tool_calls", sent_msg)
        args = json.loads(sent_msg["tool_calls"][0]["function"]["arguments"])
        self.assertIn("EMAIL", args["to"])
        self.assertIn("PHONE", args["body"])

    def test_cloak_mock_tool_calls(self):
        """Test cloaking of Mock-based tool calls."""
        shield = LLMShield()

        # Create Mock tool call
        tool_call = Mock()
        tool_call.id = "call_456"
        tool_call.type = "function"
        tool_call.function = Mock()
        tool_call.function.name = "lookup"
        tool_call.function.arguments = '{"email": "alice@test.com"}'

        messages = [
            {"role": "assistant", "content": None, "tool_calls": [tool_call]}
        ]

        captured = []

        def capture(**kwargs):
            captured.append(kwargs.get("messages", []))
            return "OK"

        shield.llm_func = capture
        shield.ask(messages=messages)

        # Verify Mock was converted to dict
        sent_msg = captured[0][0]
        self.assertIn("tool_calls", sent_msg)
        tc = sent_msg["tool_calls"][0]
        self.assertEqual(tc["id"], "call_456")
        self.assertEqual(tc["type"], "function")
        self.assertEqual(tc["function"]["name"], "lookup")

        # Verify arguments were cloaked
        args = json.loads(tc["function"]["arguments"])
        self.assertIn("EMAIL", args["email"])

    def test_preserve_tool_call_fields(self):
        """Test that all tool call fields are preserved."""
        shield = LLMShield()

        messages = [
            {
                "role": "assistant",
                "content": "I'll help with that",
                "tool_calls": [
                    {
                        "id": "unique_id_789",
                        "type": "function",
                        "function": {
                            "name": "complex_function",
                            "arguments": '{"data": "no PII here"}',
                        },
                        "extra_field": "preserved",
                    }
                ],
                "other_field": "also preserved",
            }
        ]

        captured = []

        def capture(**kwargs):
            captured.append(kwargs.get("messages", []))
            return "OK"

        shield.llm_func = capture
        shield.ask(messages=messages)

        sent_msg = captured[0][0]
        # Check all fields preserved
        self.assertEqual(sent_msg["other_field"], "also preserved")
        self.assertEqual(sent_msg["tool_calls"][0]["id"], "unique_id_789")
        self.assertEqual(sent_msg["tool_calls"][0]["extra_field"], "preserved")


if __name__ == "__main__":
    unittest.main()
