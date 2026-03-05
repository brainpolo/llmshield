"""Test tool call handling: cloaking, uncloaking, and integration.

Description:
    Consolidated tests for tool call support including argument cloaking,
    structure preservation, end-to-end flows, and multi-turn conversations.

Author:
    LLMShield by brainpolo, 2025-2026
"""

import json
import re
import unittest
from unittest.mock import Mock

from llmshield import LLMShield
from tests.helpers import make_capture_llm


class TestToolCallCloaking(unittest.TestCase):
    """Test that tool call arguments are properly cloaked."""

    def test_cloak_dict_tool_calls(self):
        """Test cloaking of dict-based tool calls with PII."""
        capture, captured = make_capture_llm()
        shield = LLMShield(llm_func=capture)

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
                                    "to": "john.doe@example.com",
                                    "subject": "Meeting with Sarah Johnson",
                                    "body": "Contact me at 555-123-4567",
                                }
                            ),
                        },
                    }
                ],
            }
        ]

        shield.ask(messages=messages)

        sent_msg = captured[0][0]
        self.assertIn("tool_calls", sent_msg)
        args = json.loads(sent_msg["tool_calls"][0]["function"]["arguments"])
        self.assertIn("EMAIL", args["to"])
        self.assertNotIn("Sarah Johnson", args["subject"])
        self.assertNotIn("555-123-4567", args["body"])

    def test_cloak_mock_tool_calls(self):
        """Test cloaking of Mock-based tool calls (SDK objects)."""
        capture, captured = make_capture_llm()
        shield = LLMShield(llm_func=capture)

        tool_call = Mock()
        tool_call.id = "call_456"
        tool_call.type = "function"
        tool_call.function = Mock()
        tool_call.function.name = "lookup"
        tool_call.function.arguments = '{"email": "alice@test.com"}'

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call],
            }
        ]

        shield.ask(messages=messages)

        sent_msg = captured[0][0]
        tc = sent_msg["tool_calls"][0]
        self.assertEqual(tc["id"], "call_456")
        self.assertEqual(tc["function"]["name"], "lookup")
        args = json.loads(tc["function"]["arguments"])
        self.assertIn("EMAIL", args["email"])

    def test_preserve_tool_call_fields(self):
        """Test that all tool call fields are preserved after cloaking."""
        capture, captured = make_capture_llm()
        shield = LLMShield(llm_func=capture)

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

        shield.ask(messages=messages)

        sent_msg = captured[0][0]
        self.assertEqual(sent_msg["other_field"], "also preserved")
        self.assertEqual(sent_msg["tool_calls"][0]["id"], "unique_id_789")
        self.assertEqual(sent_msg["tool_calls"][0]["extra_field"], "preserved")


class TestToolCallValidation(unittest.TestCase):
    """Test that tool call messages are handled without errors."""

    def test_tool_call_with_tool_response(self):
        """Test messages with tool calls and tool responses."""

        def mock_llm(**kwargs):
            return "The answer is 8"

        shield = LLMShield(llm_func=mock_llm)

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

        result = shield.ask(messages=messages)
        self.assertIsInstance(result, str)


class TestToolCallIntegration(unittest.TestCase):
    """Test end-to-end tool call flows with cloaking and uncloaking."""

    def test_complete_tool_call_flow(self):
        """Test end-to-end tool call with PII cloaking and uncloaking."""
        llm_received = []

        def mock_llm(**kwargs):
            messages = kwargs.get("messages", [])
            llm_received.append(messages)

            cloaked_content = messages[0]["content"] if messages else ""
            email_match = re.search(r"<EMAIL_\d+>", cloaked_content)
            phone_match = re.search(r"<PHONE_\d+>", cloaked_content)

            self.assertIsNotNone(email_match, "Email should be cloaked")
            self.assertIsNotNone(phone_match, "Phone should be cloaked")

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = None

            tool_call = Mock()
            tool_call.id = "call_456"
            tool_call.type = "function"
            tool_call.function = Mock()
            tool_call.function.name = "send_email"
            tool_call.function.arguments = json.dumps(
                {
                    "to": email_match.group(),
                    "subject": "Meeting with Bob Smith",
                    "body": f"Call me at {phone_match.group()}",
                }
            )

            mock_response.choices[0].message.tool_calls = [tool_call]
            mock_response.model = "gpt-4"
            mock_response.id = "chatcmpl-123"
            mock_response.object = "chat.completion"
            mock_response.created = 1234567890
            mock_response.usage = Mock()
            return mock_response

        shield = LLMShield(llm_func=mock_llm)

        messages = [
            {
                "role": "user",
                "content": (
                    "Send an email to alice@example.com about meeting "
                    "Bob Smith. My number is 555-987-6543."
                ),
            }
        ]

        response = shield.ask(messages=messages)

        # Verify PII was cloaked in what LLM received
        sent_content = llm_received[0][0]["content"]
        self.assertNotIn("alice@example.com", sent_content)
        self.assertNotIn("555-987-6543", sent_content)

        # Verify response tool calls were uncloaked
        args = json.loads(
            response.choices[0].message.tool_calls[0].function.arguments
        )
        self.assertEqual(args["to"], "alice@example.com")
        self.assertIn("555-987-6543", args["body"])
        self.assertEqual(response.model, "gpt-4")

    def test_multi_turn_with_tool_responses(self):
        """Test multi-turn conversation with tool responses."""
        call_count = 0

        def mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.model = "gpt-4"
            mock_response.id = f"chatcmpl-{call_count}"
            mock_response.object = "chat.completion"
            mock_response.created = 1234567890
            mock_response.usage = Mock()

            if call_count == 1:
                mock_response.choices[0].message.content = None
                tool_call = Mock()
                tool_call.id = "call_789"
                tool_call.type = "function"
                tool_call.function = Mock()
                tool_call.function.name = "get_user_info"
                tool_call.function.arguments = '{"user_id": "<EMAIL_0>"}'
                mock_response.choices[0].message.tool_calls = [tool_call]
            else:
                mock_response.choices[
                    0
                ].message.content = "Found info for <PERSON_1> at <EMAIL_0>"

            return mock_response

        shield = LLMShield(llm_func=mock_llm)

        messages = [
            {
                "role": "user",
                "content": "Get info for john@example.com",
            }
        ]
        response1 = shield.ask(messages=messages)

        args = json.loads(
            response1.choices[0].message.tool_calls[0].function.arguments
        )
        self.assertEqual(args["user_id"], "john@example.com")

        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_789",
                            "type": "function",
                            "function": {
                                "name": "get_user_info",
                                "arguments": json.dumps(
                                    {"user_id": "john@example.com"}
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": ("User: John Doe, Email: john@example.com"),
                    "tool_call_id": "call_789",
                },
            ]
        )

        response2 = shield.ask(messages=messages)
        final_content = response2.choices[0].message.content
        self.assertIn("john@example.com", final_content)


if __name__ == "__main__":
    unittest.main()
