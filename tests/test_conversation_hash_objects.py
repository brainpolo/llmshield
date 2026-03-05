"""Test conversation_hash with various object types.

Description:
    This test module validates the conversation_hash function's ability
    to handle both dict and object-style messages, particularly for
    Anthropic Message objects.

Test Classes:
    - TestConversationHashObjects: Tests conversation hash with objects

Author: LLMShield by brainpolo, 2025-2026
"""

import unittest
from unittest.mock import Mock

from parameterized import parameterized

from llmshield.utils import conversation_hash


def _msg(role="user", content="Hello"):
    """Create a Mock message object with role and content."""
    obj = Mock()
    obj.role = role
    obj.content = content
    return obj


class TestConversationHashObjects(unittest.TestCase):
    """Test conversation_hash with object types."""

    @parameterized.expand(
        [
            (
                "single_object",
                _msg("user", "Hello world"),
            ),
            (
                "none_content",
                _msg("assistant", None),
            ),
            (
                "list_of_objects",
                [_msg("user", "Hello"), _msg("assistant", "Hi there")],
            ),
            (
                "list_with_none_content",
                [
                    _msg("user", "What's the weather?"),
                    _msg("assistant", None),
                    _msg("tool", "15°C, sunny"),
                ],
            ),
            (
                "mixed_dict_and_objects",
                [
                    {"role": "user", "content": "Hello"},
                    _msg("assistant", "Hi there"),
                ],
            ),
        ]
    )
    def test_hash_returns_int(self, _name, input_data):
        """Test conversation_hash returns int for various inputs."""
        self.assertIsInstance(conversation_hash(input_data), int)

    def test_consistency_dict_vs_object(self):
        """Test equivalent dict and object messages produce same hash."""
        msg_dict = {"role": "user", "content": "Hello world"}
        msg_obj = _msg("user", "Hello world")
        self.assertEqual(
            conversation_hash(msg_dict),
            conversation_hash(msg_obj),
        )

    def test_consistency_list_dict_vs_objects(self):
        """Test equivalent dict and object lists produce same hash."""
        dict_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        obj_messages = [
            _msg("user", "Hello"),
            _msg("assistant", "Hi there"),
        ]
        self.assertEqual(
            conversation_hash(dict_messages),
            conversation_hash(obj_messages),
        )


if __name__ == "__main__":
    unittest.main()
