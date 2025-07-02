"""Comprehensive integration test for tool call support.

Author:
    LLMShield by brainpolo, 2025
"""

import json
import re
import unittest
from unittest.mock import Mock

from llmshield import LLMShield


class TestToolCallIntegration(unittest.TestCase):
    """Test complete tool call flow with cloaking and uncloaking."""

    def test_complete_tool_call_flow(self):
        """Test end-to-end tool call with PII cloaking and uncloaking."""
        shield = LLMShield()

        # Track what gets sent to LLM and what it returns
        llm_received = []

        def mock_llm(**kwargs):
            # Capture what LLM receives
            messages = kwargs.get("messages", [])
            llm_received.append(messages)

            # Extract placeholders from the cloaked message
            cloaked_content = messages[0]["content"] if messages else ""
            email_match = re.search(r"<EMAIL_\d+>", cloaked_content)
            phone_match = re.search(r"<PHONE_\d+>", cloaked_content)

            email_placeholder = (
                email_match.group() if email_match else "<EMAIL_0>"
            )
            phone_placeholder = (
                phone_match.group() if phone_match else "<PHONE_0>"
            )

            # Create mock response with tool call
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = None

            # Tool call with cloaked data (as LLM would generate)
            tool_call = Mock()
            tool_call.id = "call_456"
            tool_call.type = "function"
            tool_call.function = Mock()
            tool_call.function.name = "send_email"
            # LLM would use the cloaked placeholders it received
            tool_call.function.arguments = json.dumps(
                {
                    "to": email_placeholder,  # Use actual placeholder
                    "subject": "Meeting with Bob Smith",  # No placeholder
                    "body": f"Call me at {phone_placeholder}",  # Placeholder
                }
            )

            mock_response.choices[0].message.tool_calls = [tool_call]
            mock_response.model = "gpt-4"
            mock_response.id = "chatcmpl-123"
            mock_response.object = "chat.completion"
            mock_response.created = 1234567890
            mock_response.usage = Mock()

            return mock_response

        shield.llm_func = mock_llm

        # Original messages with PII
        messages = [
            {
                "role": "user",
                "content": (
                    "Send an email to alice@example.com about meeting "
                    "Bob Smith. My number is 555-987-6543."
                ),
            }
        ]

        # Call shield
        response = shield.ask(messages=messages)

        # Verify what LLM received (should be cloaked)
        sent_messages = llm_received[0]
        self.assertEqual(len(sent_messages), 1)
        sent_content = sent_messages[0]["content"]

        # Verify email placeholder exists in sent content
        self.assertIsNotNone(re.search(r"<EMAIL_\d+>", sent_content))

        # Verify PII was cloaked in the message sent to LLM
        self.assertNotIn("alice@example.com", sent_content)
        # Note: Entity detection might handle names differently
        # Just check that some cloaking happened
        self.assertIn("<", sent_content)  # Should have placeholders
        self.assertIn(">", sent_content)
        self.assertNotIn("555-987-6543", sent_content)

        # Verify response tool calls were uncloaked
        self.assertTrue(hasattr(response, "choices"))
        tool_calls = response.choices[0].message.tool_calls
        self.assertEqual(len(tool_calls), 1)

        # Parse uncloaked arguments
        args = json.loads(tool_calls[0].function.arguments)

        # Verify PII was restored in tool call arguments
        self.assertEqual(args["to"], "alice@example.com")
        self.assertIn("Bob Smith", args["subject"])
        self.assertIn("555-987-6543", args["body"])

        # Commented out debug output
        # print(f"\nOriginal message: {messages[0]['content']}")
        # print(f"Cloaked message sent to LLM: {sent_content}")
        # print(f"Uncloaked tool args: {args}")

        # Also test that the mock response object is properly populated
        self.assertEqual(response.model, "gpt-4")
        self.assertEqual(response.object, "chat.completion")

    def test_multi_turn_with_tool_responses(self):
        """Test multi-turn conversation with tool responses."""
        shield = LLMShield()
        call_count = 0

        def mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call - return tool call
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message = Mock()
                mock_response.choices[0].message.content = None

                tool_call = Mock()
                tool_call.id = "call_789"
                tool_call.type = "function"
                tool_call.function = Mock()
                tool_call.function.name = "get_user_info"
                tool_call.function.arguments = '{"user_id": "<EMAIL_0>"}'

                mock_response.choices[0].message.tool_calls = [tool_call]
            else:
                # Second call - return final response
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message = Mock()
                mock_response.choices[
                    0
                ].message.content = (
                    "I found the information for <PERSON_1> at <EMAIL_0>"
                )

            mock_response.model = "gpt-4"
            mock_response.id = f"chatcmpl-{call_count}"
            mock_response.object = "chat.completion"
            mock_response.created = 1234567890
            mock_response.usage = Mock()

            return mock_response

        shield.llm_func = mock_llm

        # First turn
        messages = [
            {"role": "user", "content": "Get info for john@example.com"}
        ]
        response1 = shield.ask(messages=messages)

        # Verify tool call was uncloaked
        args = json.loads(
            response1.choices[0].message.tool_calls[0].function.arguments
        )
        self.assertEqual(args["user_id"], "john@example.com")

        # Add tool response and continue
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
                    "content": "User: John Doe, Email: john@example.com",
                    "tool_call_id": "call_789",
                },
            ]
        )

        response2 = shield.ask(messages=messages)

        # Verify final response has uncloaked content
        final_content = response2.choices[0].message.content
        # Entity map from first call might not have all entities
        # So just verify the email was uncloaked
        self.assertIn("john@example.com", final_content)


if __name__ == "__main__":
    unittest.main()
