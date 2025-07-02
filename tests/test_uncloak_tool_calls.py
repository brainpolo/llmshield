"""Test uncloaking of tool call arguments.

Author:
    LLMShield by brainpolo, 2025
"""

import json
import unittest
from unittest.mock import Mock

from llmshield.uncloak_response import _uncloak_chatcompletion


class TestUncloakToolCalls(unittest.TestCase):
    """Test uncloaking of tool call arguments in ChatCompletion responses."""

    def test_uncloak_tool_call_arguments(self):
        """Test that tool call arguments are properly uncloaked."""
        # Create mock ChatCompletion with tool calls containing cloaked PII
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = None

        # Create tool call with cloaked email
        tool_call = Mock()
        tool_call.id = "call_123"
        tool_call.type = "function"
        tool_call.function = Mock()
        tool_call.function.name = "send_email"
        tool_call.function.arguments = json.dumps(
            {
                "to": "<EMAIL_0>",
                "subject": "Meeting with <PERSON_0>",
                "body": "Please meet <PERSON_0> at <PLACE_0>",
            }
        )

        mock_response.choices[0].message.tool_calls = [tool_call]

        # Entity map
        entity_map = {
            "<EMAIL_0>": "john.doe@example.com",
            "<PERSON_0>": "Sarah Johnson",
            "<PLACE_0>": "Conference Room A",
        }

        # Uncloak the response
        uncloaked = _uncloak_chatcompletion(mock_response, entity_map)

        # Verify tool calls were uncloaked
        args = json.loads(
            uncloaked.choices[0].message.tool_calls[0].function.arguments
        )
        self.assertEqual(args["to"], "john.doe@example.com")
        self.assertEqual(args["subject"], "Meeting with Sarah Johnson")
        self.assertEqual(
            args["body"], "Please meet Sarah Johnson at Conference Room A"
        )

    def test_uncloak_multiple_tool_calls(self):
        """Test uncloaking multiple tool calls."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[
            0
        ].message.content = "I'll help you with both tasks"

        # Create multiple tool calls
        tool_call1 = Mock()
        tool_call1.function = Mock()
        tool_call1.function.name = "send_email"
        tool_call1.function.arguments = (
            '{"to": "<EMAIL_0>", "subject": "Test"}'
        )

        tool_call2 = Mock()
        tool_call2.function = Mock()
        tool_call2.function.name = "schedule_meeting"
        tool_call2.function.arguments = (
            '{"with": "<PERSON_0>", "at": "<PLACE_0>"}'
        )

        mock_response.choices[0].message.tool_calls = [tool_call1, tool_call2]

        entity_map = {
            "<EMAIL_0>": "alice@example.com",
            "<PERSON_0>": "Bob Smith",
            "<PLACE_0>": "Room 123",
        }

        # Uncloak
        uncloaked = _uncloak_chatcompletion(mock_response, entity_map)

        # Verify both tool calls were uncloaked
        args1 = json.loads(
            uncloaked.choices[0].message.tool_calls[0].function.arguments
        )
        self.assertEqual(args1["to"], "alice@example.com")

        args2 = json.loads(
            uncloaked.choices[0].message.tool_calls[1].function.arguments
        )
        self.assertEqual(args2["with"], "Bob Smith")
        self.assertEqual(args2["at"], "Room 123")

    def test_uncloak_no_tool_calls(self):
        """Test that responses without tool calls work normally."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Hello <PERSON_0>"
        mock_response.choices[0].message.tool_calls = None

        entity_map = {"<PERSON_0>": "John"}

        # Uncloak
        uncloaked = _uncloak_chatcompletion(mock_response, entity_map)

        # Verify content was uncloaked but no errors from missing tool_calls
        self.assertEqual(uncloaked.choices[0].message.content, "Hello John")


if __name__ == "__main__":
    unittest.main()
