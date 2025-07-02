"""Test missing coverage lines in entity detector.

Description:
    This test module specifically targets lines that were not covered by
    other tests in the entity detector module, ensuring 100% test coverage
    for critical detection functionality.

Test Classes:
    - TestEntityDetectorMissingLines: Tests previously uncovered code paths

Author:
    LLMShield by brainpolo, 2025
"""

import unittest

from llmshield.entity_detector import (
    Entity,
    EntityConfig,
    EntityDetector,
    EntityGroup,
    EntityType,
)


class TestEntityDetectorMissingLines(unittest.TestCase):
    """Test cases to hit missing lines in entity_detector.py."""

    def test_entity_type_locators_method(self):
        """Test EntityType.locators() class method - line 63."""
        locators = EntityType.locators()
        expected = frozenset(
            [EntityType.EMAIL, EntityType.IP_ADDRESS, EntityType.URL]
        )
        self.assertEqual(locators, expected)

    def test_entity_type_numbers_method(self):
        """Test EntityType.numbers() class method - line 68."""
        numbers = EntityType.numbers()
        expected = frozenset([EntityType.PHONE, EntityType.CREDIT_CARD])
        self.assertEqual(numbers, expected)

    def test_entity_type_proper_nouns_method(self):
        """Test EntityType.proper_nouns() class method - line 73."""
        proper_nouns = EntityType.proper_nouns()
        expected = frozenset(
            [
                EntityType.PERSON,
                EntityType.PLACE,
                EntityType.ORGANISATION,
                EntityType.CONCEPT,
            ]
        )
        self.assertEqual(proper_nouns, expected)

    def test_process_fragment_skip_next_logic(self):
        """Test skip_next logic in _process_fragment - lines 300-301."""
        detector = EntityDetector()

        # Test contraction followed by proper noun triggers skip_next
        fragment = "I'm John going home"
        result = detector._process_fragment(fragment)

        # Should capture "John" properly
        self.assertIn("John", result)

    def test_process_fragment_contraction_lookahead(self):
        """Test contraction lookahead logic - lines 317-319."""
        detector = EntityDetector()

        # Test I'm followed by capitalized word
        fragment = "I'm Alice in wonderland"
        result = detector._process_fragment(fragment)

        # Should properly handle the lookahead
        self.assertIn("Alice", result)

    def test_handle_contraction_lookahead_edge_cases(self):
        """Test _handle_contraction_lookahead edge cases - lines 348-350."""
        # Test with I've at end of list (no next word)
        result = EntityDetector._handle_contraction_lookahead(
            "I've", 0, ["I've"]
        )
        self.assertFalse(result)

        # Test with I'm followed by lowercase
        result = EntityDetector._handle_contraction_lookahead(
            "I'm", 0, ["I'm", "going"]
        )
        self.assertFalse(result)

        # Test with I'll followed by uppercase
        result = EntityDetector._handle_contraction_lookahead(
            "I'll", 0, ["I'll", "Alice"]
        )
        self.assertTrue(result)

    def test_clean_person_name_empty_words(self):
        """Test _clean_person_name with empty words list - line 381."""
        detector = EntityDetector()

        # Empty string should return as-is
        result = detector._clean_person_name("")
        self.assertEqual(result, "")

    def test_classify_proper_noun_empty_input(self):
        """Test _classify_proper_noun with empty input - line 396."""
        detector = EntityDetector()

        result = detector._classify_proper_noun("")
        self.assertIsNone(result)

    def test_is_person_empty_words_after_cleanup(self):
        """Test _is_person with empty words after cleanup.

        This tests lines 477, 481, 485.
        """
        detector = EntityDetector()

        # Test only honorifics (should return False after cleanup)
        result = detector._is_person("Dr. Ms.")
        self.assertFalse(result)

        # Test empty words list
        result = detector._is_person("")
        self.assertFalse(result)

        # Test edge case where clean_word becomes empty after stripping
        # Mock a scenario that triggers the empty word condition
        result = detector._is_person("  ")  # Only whitespace
        self.assertFalse(result)

    def test_is_person_hyphenated_name_edge_cases(self):
        """Test _is_person with hyphenated names - lines 493, 499."""
        detector = EntityDetector()

        # Test hyphenated name with non-capitalized parts
        result = detector._is_person("Mary-jane")  # lowercase 'j'
        self.assertFalse(result)

        # Test hyphenated name with empty parts
        result = detector._is_person("Mary-")
        self.assertFalse(result)

        # Test valid hyphenated name
        result = detector._is_person("Mary-Jane")
        self.assertTrue(result)

    def test_entity_config_factory_methods_coverage(self):
        """Test EntityConfig factory methods for full coverage."""
        # Test disable_locations
        config = EntityConfig.disable_locations()
        self.assertFalse(config.is_enabled(EntityType.PLACE))
        self.assertFalse(config.is_enabled(EntityType.IP_ADDRESS))
        self.assertFalse(config.is_enabled(EntityType.URL))
        self.assertTrue(config.is_enabled(EntityType.PERSON))

        # Test disable_persons
        config = EntityConfig.disable_persons()
        self.assertFalse(config.is_enabled(EntityType.PERSON))
        self.assertTrue(config.is_enabled(EntityType.EMAIL))

        # Test disable_contacts
        config = EntityConfig.disable_contacts()
        self.assertFalse(config.is_enabled(EntityType.EMAIL))
        self.assertFalse(config.is_enabled(EntityType.PHONE))
        self.assertTrue(config.is_enabled(EntityType.PERSON))

        # Test only_financial
        config = EntityConfig.only_financial()
        self.assertTrue(config.is_enabled(EntityType.CREDIT_CARD))
        self.assertFalse(config.is_enabled(EntityType.PERSON))
        self.assertFalse(config.is_enabled(EntityType.EMAIL))

    def test_entity_group_property(self):
        """Test Entity.group property."""
        # Test person entity
        entity = Entity(type=EntityType.PERSON, value="John Doe")
        self.assertEqual(entity.group, EntityGroup.PNOUN)

        # Test email entity
        entity = Entity(type=EntityType.EMAIL, value="test@example.com")
        self.assertEqual(entity.group, EntityGroup.LOCATOR)

        # Test phone entity
        entity = Entity(type=EntityType.PHONE, value="555-1234")
        self.assertEqual(entity.group, EntityGroup.NUMBER)

    def test_entity_group_unknown_type(self):
        """Test Entity.group with unknown entity type."""
        # Create a mock entity with invalid type
        entity = Entity(type="INVALID_TYPE", value="test")

        with self.assertRaises(ValueError) as context:
            _ = entity.group

        self.assertIn("Unknown entity type", str(context.exception))

    def test_entity_detector_selective_group_filtering(self):
        """Test selective detection skips entire groups."""
        # Config that disables all LOCATOR types
        config = EntityConfig().with_disabled(
            EntityType.EMAIL, EntityType.URL, EntityType.IP_ADDRESS
        )
        detector = EntityDetector(config)

        # Text with mixed entity types
        text = "Contact john@example.com at https://example.com or 192.168.1.1"
        entities = detector.detect_entities(text)

        # Should not detect any locator entities
        locator_entities = [
            e for e in entities if e.type in EntityType.locators()
        ]
        self.assertEqual(len(locator_entities), 0)

    def test_entity_detector_partial_group_filtering(self):
        """Test selective detection with partial group filtering."""
        # Config that disables only EMAIL from LOCATOR group
        config = EntityConfig().with_disabled(EntityType.EMAIL)
        detector = EntityDetector(config)

        # Text with mixed locator types
        text = "Email john@example.com or visit https://example.com"
        entities = detector.detect_entities(text)

        # Should detect URL but not EMAIL
        email_entities = [e for e in entities if e.type == EntityType.EMAIL]
        url_entities = [e for e in entities if e.type == EntityType.URL]

        self.assertEqual(len(email_entities), 0)
        self.assertEqual(len(url_entities), 1)

    def test_detect_proper_nouns_no_classification(self):
        """Test proper noun detection when classification returns None."""
        detector = EntityDetector()

        # Use text that creates proper nouns but won't classify
        # (lowercase words that get split)
        text = "this is lowercase text"
        entities, reduced_text = detector._detect_proper_nouns(text)

        # Should return empty entities set
        self.assertEqual(len(entities), 0)

    def test_detect_entities_with_working_text_modification(self):
        """Test working_text modification between detection phases."""
        detector = EntityDetector()

        # Text with entities from different groups
        text = (
            "Email john@example.com and visit https://example.com for John Doe"
        )
        entities = detector.detect_entities(text)

        # Should detect entities from all phases
        email_entities = [e for e in entities if e.type == EntityType.EMAIL]
        url_entities = [e for e in entities if e.type == EntityType.URL]
        person_entities = [e for e in entities if e.type == EntityType.PERSON]

        self.assertGreater(len(email_entities), 0)
        self.assertGreater(len(url_entities), 0)
        self.assertGreater(len(person_entities), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
