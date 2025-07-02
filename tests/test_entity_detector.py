"""Test entity detection and classification functionality.

Description:
    This test module provides comprehensive testing for the EntityDetector
    class, covering all entity types including emails, phone numbers, IP
    addresses, credit cards, SSNs, URLs, and various proper nouns.

Test Classes:
    - TestEntityDetector: Main test suite for entity detection
    - TestEntityDetectorEdgeCases: Tests edge cases and special scenarios
    - TestEntityDetectorRegexCompilation: Tests regex pattern compilation

Author: LLMShield by brainpolo, 2025
"""

# Standard library Imports
import unittest

# Third party Imports
from parameterized import parameterized

# Local Imports
from llmshield.entity_detector import EntityDetector, EntityGroup, EntityType


class TestEntityDetector(unittest.TestCase):
    """Test suite for EntityDetector class."""

    # pylint: disable=protected-access  # Testing internal methods requires access to protected members

    def setUp(self):
        """Initialise detector for each test."""
        self.detector = EntityDetector()

    def test_entity_group_types(self):
        """Test EntityGroup.get_types() method."""
        # Test all group mappings
        self.assertEqual(
            EntityGroup.PNOUN.get_types(),
            {
                EntityType.PERSON,
                EntityType.ORGANISATION,
                EntityType.PLACE,
                EntityType.CONCEPT,
            },
        )
        self.assertEqual(
            EntityGroup.NUMBER.get_types(),
            {EntityType.PHONE, EntityType.CREDIT_CARD},
        )
        self.assertEqual(
            EntityGroup.LOCATOR.get_types(),
            {EntityType.EMAIL, EntityType.URL, EntityType.IP_ADDRESS},
        )

    def test_detect_proper_nouns_empty(self):
        """Test proper noun detection with empty input."""
        entities, text = self.detector._detect_proper_nouns("")
        self.assertEqual(len(entities), 0)
        self.assertEqual(text, "")

    @parameterized.expand(
        [
            # Basic contraction tests
            ("contraction_im", "I'm John", ["John"]),
            ("contraction_ive", "I've met Alice", ["Alice"]),
            ("contraction_ill", "I'll see Bob", ["Bob"]),
            # Advanced contraction scenarios (consolidating missing lines
            # tests)
            (
                "pending_noun_contraction",
                "Dr. Smith I'm going to see Johnson",
                ["Dr", "Smith", "Johnson"],
            ),
            ("lookahead_contraction", "I'm Alice going home", ["Alice"]),
            (
                "skip_next_logic",
                "Hello I've seen Mary before",
                ["Hello", "Mary"],
            ),
            # Empty word handling scenarios
            ("empty_word_spaces", "Dr.   Smith", ["Dr", "Smith"]),
            (
                "pending_noun_reset",
                "Dr. Smith went to the store quickly",
                ["Dr", "Smith"],
            ),
        ]
    )
    def test_proper_noun_collection_comprehensive(
        self, description, text, expected_names
    ):
        """Comprehensive test for proper noun collection.

        Includes contractions and edge cases for thorough validation.
        """
        result = self.detector._collect_proper_nouns(text)
        for name in expected_names:
            self.assertIn(
                name,
                result,
                f"Expected '{name}' in result for case: {description}",
            )

    def test_collect_proper_nouns_honorifics(self):
        """Test honorific handling in proper noun collection."""
        text = "Dr. Smith and Ms. Johnson"
        proper_nouns = self.detector._collect_proper_nouns(text)

        # Should collect honorifics and names separately
        self.assertIn("Dr", proper_nouns)
        self.assertIn("Smith", proper_nouns)
        self.assertIn("Ms", proper_nouns)
        self.assertIn("Johnson", proper_nouns)

    @parameterized.expand(
        [
            ("no_honorific", "Jane Doe", "Jane Doe"),
            ("empty_string", "", ""),
        ]
    )
    def test_clean_person_name_comprehensive(
        self, description, input_name, expected
    ):
        """Test person name cleaning with various edge cases.

        Parameterized test for comprehensive name cleaning validation.
        """
        result = self.detector._clean_person_name(input_name)
        self.assertEqual(result, expected)

    @parameterized.expand(
        [
            # Basic organisation tests
            ("known_org", "Microsoft", True),
            ("org_with_component", "Google Inc", True),
            ("regular_name", "John Smith", False),
            # Regex pattern tests (consolidating lines 339, 343)
            ("org_with_numbers_dash", "3M-2024", True),  # ^\d+[A-Z].* pattern
            (
                "org_complex_pattern",
                "IBM-Solutions-2024",
                True,
            ),  # .*-.*\d+.* pattern
            ("org_number_prefix", "2024Tech", True),  # ^\d+[A-Z].* pattern
            # Multi-word organisation tests
            ("multi_word_times", "New York Times", True),
            ("multi_word_corporation", "Microsoft Corporation", True),
        ]
    )
    def test_organisation_detection_comprehensive(
        self, description, text, expected
    ):
        """Comprehensive test for organisation detection.

        Includes regex patterns for thorough organisation validation.
        """
        result = self.detector._is_organisation(text)
        self.assertEqual(result, expected, f"Failed for {description}: {text}")

    @parameterized.expand(
        [
            # Basic place tests
            ("major_city", "New York", True),
            ("world_city", "London", True),
            ("non_place", "Not A Place", False),
            # Place component tests (consolidating line 361)
            ("street_component", "Main Street", True),
            ("avenue_component", "Oak Avenue", True),
            ("road_component", "Park Road", True),
            ("custom_place", "Washington Street", True),
        ]
    )
    def test_place_detection_comprehensive(self, description, text, expected):
        """Comprehensive test for place detection including components."""
        result = self.detector._is_place(text)
        self.assertEqual(result, expected, f"Failed for {description}: {text}")

    @parameterized.expand(
        [
            ("simple_name", "John", True),
            ("possessive_name", "John's", True),
            ("hyphenated_name", "Mary-Jane", True),
            ("name_with_digits", "John2", False),
        ]
    )
    def test_person_detection_edge_cases(self, description, text, expected):
        """Test person detection edge cases - parameterized."""
        result = self.detector._is_person(text)
        self.assertEqual(result, expected)

    @parameterized.expand(
        [
            # Consolidating all concept detection tests (lines 287)
            ("uppercase_single_word", "API", True),
            ("rest_concept", "REST", True),
            ("http_concept", "HTTP", True),
            ("lowercase_single", "api", False),
            ("multi_word", "API KEY", False),
            ("with_punctuation", "API!", False),
        ]
    )
    def test_concept_detection_comprehensive(
        self, description, text, expected
    ):
        """Comprehensive test for concept detection hitting line 287."""
        result = self.detector._is_concept(text)
        self.assertEqual(result, expected, f"Failed for {description}: {text}")

    def test_detect_numbers_empty(self):
        """Test number detection with empty input."""
        entities, text = self.detector._detect_numbers("")
        self.assertEqual(len(entities), 0)
        self.assertEqual(text, "")

    def test_detect_invalid_credit_card(self):
        """Test credit card validation."""
        text = "1234567890123456"  # Invalid credit card format
        entities, _ = self.detector._detect_numbers(text)
        self.assertEqual(
            len([e for e in entities if e.type == EntityType.CREDIT_CARD]), 0
        )

    def test_phone_number_detection(self):
        """Test phone number detection."""
        text = "Call me at +1 (555) 123-4567"
        entities, _ = self.detector._detect_numbers(text)
        self.assertEqual(
            len([e for e in entities if e.type == EntityType.PHONE]), 1
        )
        try:
            phone_number = next(
                e.value for e in entities if e.type == EntityType.PHONE
            )
            self.assertEqual(phone_number, "+1 (555) 123-4567")
        except StopIteration:
            self.fail("No phone number entity found")

    def test_detect_locators_empty(self):
        """Test locator detection with empty input."""
        entities, text = self.detector._detect_locators("")
        self.assertEqual(len(entities), 0)
        self.assertEqual(text, "")

    def test_email_detection_in_numbers_method(self):
        """Test email detection in _detect_numbers method."""
        text = "Contact john@example.com for details"
        entities, reduced_text = self.detector._detect_numbers(text)

        # Should find email entity
        email_entities = [e for e in entities if e.type == EntityType.EMAIL]
        self.assertEqual(len(email_entities), 1)
        self.assertEqual(email_entities[0].value, "john@example.com")
        self.assertNotIn("john@example.com", reduced_text)


if __name__ == "__main__":
    unittest.main()
