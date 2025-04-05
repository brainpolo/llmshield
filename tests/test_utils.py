"""
Tests for utility functions in llmshield.utils

! Module is intended for internal use only.
"""


import unittest

from llmshield.utils import is_valid_delimiter, wrap_entity


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.start_delimiter = '['
        self.end_delimiter = ']'

    def test_is_valid_delimiter(self):
        """Test delimiter validation function."""
        # Valid cases
        self.assertTrue(is_valid_delimiter('['))
        self.assertTrue(is_valid_delimiter(']]'))
        self.assertTrue(is_valid_delimiter('<<<'))
        self.assertTrue(is_valid_delimiter('#'))

        # Invalid cases
        self.assertFalse(is_valid_delimiter(''))
        self.assertFalse(is_valid_delimiter(None))
        self.assertFalse(is_valid_delimiter(123))
        self.assertFalse(is_valid_delimiter(['[']))

    def test_wrap_entity(self):
        """Test entity wrapping function."""
        from llmshield.entity_detector import EntityType

        # Test with different entity types
        self.assertEqual(
            wrap_entity(EntityType.PERSON, 0, '[', ']'),
            '[PERSON_0]'
        )
        self.assertEqual(
            wrap_entity(EntityType.EMAIL, 1, '<', '>'),
            '<EMAIL_1>'
        )

        # Test with multi-character delimiters
        self.assertEqual(
            wrap_entity(EntityType.PHONE_NUMBER, 2, '[[', ']]'),
            '[[PHONE_NUMBER_2]]'
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
