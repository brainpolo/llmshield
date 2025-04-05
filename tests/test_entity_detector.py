"""Tests for entity detection and classification."""


import unittest
from llmshield.entity_detector import EntityDetector, EntityType, EntityGroup


class TestEntityDetector(unittest.TestCase):
    """Test suite for EntityDetector class."""
    # pylint: disable=protected-access  # Testing internal methods requires access to protected members

    def setUp(self):
        """Initialize detector for each test."""
        self.detector = EntityDetector()

    def test_entity_group_types(self):
        """Test EntityGroup.get_types() method."""
        # Test all group mappings
        self.assertEqual(
            EntityGroup.PNOUN.get_types(),
            {EntityType.PERSON, EntityType.ORGANISATION, EntityType.PLACE, EntityType.CONCEPT}
        )
        self.assertEqual(
            EntityGroup.NUMBER.get_types(),
            {EntityType.PHONE_NUMBER, EntityType.CREDIT_CARD}
        )
        self.assertEqual(
            EntityGroup.LOCATOR.get_types(),
            {EntityType.EMAIL, EntityType.URL, EntityType.IP_ADDRESS}
        )

    def test_detect_proper_nouns_empty(self):
        """Test proper noun detection with empty input."""
        entities, text = self.detector._detect_proper_nouns("")
        self.assertEqual(len(entities), 0)
        self.assertEqual(text, "")

    def test_collect_proper_nouns_contractions(self):
        """Test contraction handling in proper noun collection."""
        # Test when word is a contraction
        text = "I'll be there. I'm John. I've got Smith."
        proper_nouns = self.detector._collect_proper_nouns(text)

        # Verify contractions are handled and proper nouns are collected
        self.assertIn("John", proper_nouns)
        # Handle potential punctuation
        self.assertTrue(any(noun.rstrip(".,!?;:") == "Smith" for noun in proper_nouns))

        # Test skipping next word after contraction
        text = "I'm John"
        proper_nouns = self.detector._collect_proper_nouns(text)
        self.assertIn("John", proper_nouns)

    def test_collect_proper_nouns_honorifics(self):
        """Test honorific handling in collection."""
        # Test honorific-specific handling
        text = "Dr. Jane and Mr. Smith"
        proper_nouns = self.detector._collect_proper_nouns(text)

        # Test what's actually being collected - honorifics and names as separate tokens
        self.assertIn("Jane", proper_nouns)
        self.assertIn("Smith", proper_nouns)
        self.assertTrue(any(noun.startswith("Dr") for noun in proper_nouns))
        self.assertTrue(any(noun.startswith("Mr") for noun in proper_nouns))

    def test_honorific_handling(self):
        """Test honorific handling in person detection."""
        # Test with individual names to verify proper classification
        test_names = [
            "Dr. Jane",
            "Prof. Robert",
            "Mr. William"
        ]

        for name in test_names:
            result = self.detector._classify_proper_noun(name)
            self.assertIsNotNone(result, f"Failed to classify {name}")
            cleaned_value, entity_type = result

            # Check that honorific is removed and entity type is PERSON
            self.assertEqual(entity_type, EntityType.PERSON)
            self.assertNotIn("Dr.", cleaned_value)
            self.assertNotIn("Prof.", cleaned_value)
            self.assertNotIn("Mr.", cleaned_value)

    def test_classify_proper_noun_empty(self):
        """Test classification with empty inputs."""
        # Test empty string
        result = self.detector._classify_proper_noun("")
        self.assertIsNone(result)

        # Test None input
        result = self.detector._classify_proper_noun(None)
        self.assertIsNone(result)

    def test_organization_detection(self):
        """Test organization detection with various formats."""
        # Test numeric organizations
        self.assertTrue(self.detector._is_organization("3M"))
        self.assertTrue(self.detector._is_organization("7-Eleven"))

        # Test multi-word organizations
        self.assertTrue(self.detector._is_organization("New York Times"))
        self.assertTrue(self.detector._is_organization("International Business Machines Corporation"))

    def test_place_detection(self):
        """Test place detection."""
        self.assertTrue(self.detector._is_place("New York"))
        self.assertTrue(self.detector._is_place("London"))

    def test_place_edge_cases(self):
        """Test place detection edge cases."""
        # Line 301 - place component in word
        custom_place = "Washington Street"
        self.assertTrue(self.detector._is_place(custom_place))

        # Ensure non-places aren't detected
        self.assertFalse(self.detector._is_place("Not A Place"))

    def test_person_detection_edge_cases(self):
        """Test person detection edge cases."""
        # Empty input
        self.assertFalse(self.detector._is_person(""))

        # Just honorific - adjust to match implementation
        honorific_only = "Mr."
        cleaned = self.detector._clean_person_name(honorific_only)
        self.assertEqual(cleaned, honorific_only)  # Honorific remains if alone

        # Hyphenated names
        self.assertTrue(self.detector._is_person("John-Paul"))
        self.assertFalse(self.detector._is_person("not-Capitalized"))

        # Names with possessives
        self.assertTrue(self.detector._is_person("John's"))

    def test_detect_numbers_empty(self):
        """Test number detection with empty input."""
        # Test empty string
        entities, text = self.detector._detect_numbers("")
        self.assertEqual(len(entities), 0)
        self.assertEqual(text, "")

    def test_detect_invalid_credit_card(self):
        """Test credit card validation."""
        # Line 368 - invalid credit card (fails Luhn check)
        # Using a card number that appears valid but fails Luhn validation
        text = "1234567890123456"  # Invalid credit card format
        entities, _ = self.detector._detect_numbers(text)
        self.assertEqual(len([e for e in entities if e.type == EntityType.CREDIT_CARD]), 0)

    def test_phone_number_detection(self):
        """Test phone number detection."""
        text = "Call me at +1 (555) 123-4567"
        entities, _ = self.detector._detect_numbers(text)
        self.assertEqual(len([e for e in entities if e.type == EntityType.PHONE_NUMBER]), 1)
        try:
            phone_number = next(e.value for e in entities if e.type == EntityType.PHONE_NUMBER)
            self.assertEqual(phone_number, "+1 (555) 123-4567")
        except StopIteration:
            self.fail("No phone number entity found")

    def test_detect_locators_empty(self):
        """Test locator detection with empty input."""
        # Test empty string
        entities, text = self.detector._detect_locators("")
        self.assertEqual(len(entities), 0)
        self.assertEqual(text, "")


if __name__ == '__main__':
    unittest.main()

