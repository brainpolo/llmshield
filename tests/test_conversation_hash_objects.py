"""Test conversation_hash with various object types.

Description:
    This test module validates the conversation_hash function's ability
    to handle both dict and object-style messages, particularly for
    Anthropic Message objects.

Test Classes:
    - TestConversationHashObjects: Tests conversation hash with objects

Author: LLMShield by brainpolo, 2025
"""

import unittest
from unittest.mock import Mock

from llmshield.utils import conversation_hash


class TestConversationHashObjects(unittest.TestCase):
    """Test conversation_hash with object types."""

    def test_conversation_hash_single_message_object(self):
        """Test conversation_hash with single message object."""
        # Create mock message object
        msg_obj = Mock()
        msg_obj.role = "user"
        msg_obj.content = "Hello world"

        hash_value = conversation_hash(msg_obj)
        self.assertIsInstance(hash_value, int)

    def test_conversation_hash_single_message_object_none_content(self):
        """Test hash with single message object having None content."""
        # Create mock message object with None content
        msg_obj = Mock()
        msg_obj.role = "assistant"
        msg_obj.content = None

        hash_value = conversation_hash(msg_obj)
        self.assertIsInstance(hash_value, int)

    def test_conversation_hash_list_with_message_objects(self):
        """Test conversation_hash with list of message objects."""
        # Create mock message objects
        msg1 = Mock()
        msg1.role = "user"
        msg1.content = "Hello"

        msg2 = Mock()
        msg2.role = "assistant"
        msg2.content = "Hi there"

        messages = [msg1, msg2]
        hash_value = conversation_hash(messages)
        self.assertIsInstance(hash_value, int)

    def test_conversation_hash_list_with_message_objects_none_content(self):
        """Test hash with list of message objects having None content."""
        # Create mock message objects, some with None content
        msg1 = Mock()
        msg1.role = "user"
        msg1.content = "What's the weather?"

        msg2 = Mock()
        msg2.role = "assistant"
        msg2.content = None  # Tool call response

        msg3 = Mock()
        msg3.role = "tool"
        msg3.content = "15°C, sunny"

        messages = [msg1, msg2, msg3]
        hash_value = conversation_hash(messages)
        self.assertIsInstance(hash_value, int)

    def test_conversation_hash_mixed_dict_and_objects(self):
        """Test conversation_hash with mixed dict and object messages."""
        # Create a mix of dict and object messages
        msg_dict = {"role": "user", "content": "Hello"}

        msg_obj = Mock()
        msg_obj.role = "assistant"
        msg_obj.content = "Hi there"

        messages = [msg_dict, msg_obj]
        hash_value = conversation_hash(messages)
        self.assertIsInstance(hash_value, int)

    def test_conversation_hash_consistency_dict_vs_object(self):
        """Test that equivalent dict and object messages produce same hash."""
        # Create dict message
        msg_dict = {"role": "user", "content": "Hello world"}
        hash_dict = conversation_hash(msg_dict)

        # Create equivalent object message
        msg_obj = Mock()
        msg_obj.role = "user"
        msg_obj.content = "Hello world"
        hash_obj = conversation_hash(msg_obj)

        # They should produce the same hash
        self.assertEqual(hash_dict, hash_obj)

    def test_conversation_hash_list_consistency_dict_vs_object(self):
        """Test that equivalent dict and object lists produce same hash."""
        # Create dict messages
        dict_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        hash_dict = conversation_hash(dict_messages)

        # Create equivalent object messages
        obj1 = Mock()
        obj1.role = "user"
        obj1.content = "Hello"

        obj2 = Mock()
        obj2.role = "assistant"
        obj2.content = "Hi there"

        obj_messages = [obj1, obj2]
        hash_obj = conversation_hash(obj_messages)

        # They should produce the same hash
        self.assertEqual(hash_dict, hash_obj)


if __name__ == "__main__":
    unittest.main()
