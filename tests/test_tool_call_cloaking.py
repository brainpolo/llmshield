"""Test cloaking of tool call arguments.

Author:
    LLMShield by brainpolo, 2025
"""

import json
import unittest

from llmshield import LLMShield


class TestToolCallCloaking(unittest.TestCase):
    """Test that tool call arguments are properly cloaked."""

    def test_cloak_tool_call_arguments_in_messages(self):
        """Test that PII in tool call arguments gets cloaked."""
        shield = LLMShield()

        # Message with tool call containing PII
        messages = [
            {
                "role": "user",
                "content": "Send an email to john.doe@example.com",
            },
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
                                    "to": "john.doe@example.com",
                                    "subject": "Meeting with Sarah Johnson",
                                    "body": "Contact me at 555-123-4567",
                                }
                            ),
                        },
                    }
                ],
            },
        ]

        # Capture what gets sent to LLM
        captured_messages = []

        def capture_llm(**kwargs):
            captured_messages.extend(kwargs.get("messages", []))
            return "Done"

        shield.llm_func = capture_llm
        shield.ask(messages=messages)

        # Check if tool call arguments were cloaked
        tool_call = captured_messages[1]["tool_calls"][0]
        args = json.loads(tool_call["function"]["arguments"])

        # Debug info (commented out for clean tests)
        # print("Original email: john.doe@example.com")
        # print(f"Cloaked email: {args['to']}")
        # print("Original name: Sarah Johnson")
        # print(f"Cloaked subject: {args['subject']}")
        # print("Original phone: 555-123-4567")
        # print(f"Cloaked body: {args['body']}")

        # Verify PII was cloaked
        self.assertNotEqual(args["to"], "john.doe@example.com")
        self.assertIn("EMAIL", args["to"])
        # Note: Entity detection might split names differently
        self.assertTrue(
            "PERSON" in args["subject"]
            or args["subject"] != "Meeting with Sarah Johnson"
        )
        self.assertNotIn("555-123-4567", args["body"])


if __name__ == "__main__":
    unittest.main()
