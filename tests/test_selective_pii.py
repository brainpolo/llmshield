"""Test selective PII detection and filtering functionality.

Description:
    This test module validates the selective entity detection capabilities
    introduced in LLMShield v2.0+, including EntityConfig, factory methods,
    and integration with core functionality for fine-grained PII control.

Test Classes:
    - TestSelectivePIIDetection: Tests selective detection features
    - TestEntityConfigFactory: Tests EntityConfig factory methods
    - TestSelectivePIIIntegration: Tests integration with LLMShield

Author:
    LLMShield by brainpolo, 2025
"""

import unittest

from parameterized import parameterized

from llmshield import LLMShield
from llmshield.entity_detector import EntityConfig, EntityDetector, EntityType


class TestSelectivePIIDetection(unittest.TestCase):
    """Test selective PII detection and filtering functionality."""

    def setUp(self):
        """Set up test fixtures with sample text containing PII types."""
        self.sample_text = (
            "Contact Dr. John Doe at john.doe@company.com or call "
            "555-123-4567. "
            "Visit our office in New York or check "
            "https://company.com. Our server at 192.168.1.1 processes "
            "payments including card 4111111111111111. "
            "Microsoft Corporation is our partner."
        )

        # Expected entities by type for the sample text
        self.expected_entities = {
            EntityType.PERSON: ["John Doe"],  # Detected because of Dr. prefix
            EntityType.EMAIL: ["john.doe@company.com"],
            EntityType.PHONE: ["555-123-4567"],  # Format that actually matches
            EntityType.PLACE: ["New York"],
            EntityType.URL: ["https://company.com"],
            EntityType.IP_ADDRESS: ["192.168.1.1"],
            EntityType.CREDIT_CARD: ["4111111111111111"],  # Valid Luhn number
            EntityType.ORGANISATION: ["Microsoft Corporation"],
        }

    def test_factory_method_disable_locations(self):
        """Test disable_locations factory method."""
        shield = LLMShield.disable_locations()
        config = shield.entity_config

        # Should disable location-based entities
        self.assertFalse(config.is_enabled(EntityType.PLACE))
        self.assertFalse(config.is_enabled(EntityType.IP_ADDRESS))
        self.assertFalse(config.is_enabled(EntityType.URL))

        # Should keep other entities enabled
        self.assertTrue(config.is_enabled(EntityType.PERSON))
        self.assertTrue(config.is_enabled(EntityType.EMAIL))
        self.assertTrue(config.is_enabled(EntityType.PHONE))
        self.assertTrue(config.is_enabled(EntityType.CREDIT_CARD))
        self.assertTrue(config.is_enabled(EntityType.ORGANISATION))

    def test_factory_method_disable_persons(self):
        """Test disable_persons factory method."""
        shield = LLMShield.disable_persons()
        config = shield.entity_config

        # Should disable person entities
        self.assertFalse(config.is_enabled(EntityType.PERSON))

        # Should keep other entities enabled
        self.assertTrue(config.is_enabled(EntityType.EMAIL))
        self.assertTrue(config.is_enabled(EntityType.PHONE))
        self.assertTrue(config.is_enabled(EntityType.PLACE))
        self.assertTrue(config.is_enabled(EntityType.URL))
        self.assertTrue(config.is_enabled(EntityType.IP_ADDRESS))
        self.assertTrue(config.is_enabled(EntityType.CREDIT_CARD))
        self.assertTrue(config.is_enabled(EntityType.ORGANISATION))

    def test_factory_method_disable_contacts(self):
        """Test disable_contacts factory method."""
        shield = LLMShield.disable_contacts()
        config = shield.entity_config

        # Should disable contact information
        self.assertFalse(config.is_enabled(EntityType.EMAIL))
        self.assertFalse(config.is_enabled(EntityType.PHONE))

        # Should keep other entities enabled
        self.assertTrue(config.is_enabled(EntityType.PERSON))
        self.assertTrue(config.is_enabled(EntityType.PLACE))
        self.assertTrue(config.is_enabled(EntityType.URL))
        self.assertTrue(config.is_enabled(EntityType.IP_ADDRESS))
        self.assertTrue(config.is_enabled(EntityType.CREDIT_CARD))
        self.assertTrue(config.is_enabled(EntityType.ORGANISATION))

    def test_factory_method_only_financial(self):
        """Test only_financial factory method."""
        shield = LLMShield.only_financial()
        config = shield.entity_config

        # Should only enable financial entities
        self.assertTrue(config.is_enabled(EntityType.CREDIT_CARD))

        # Should disable all other entities
        self.assertFalse(config.is_enabled(EntityType.PERSON))
        self.assertFalse(config.is_enabled(EntityType.EMAIL))
        self.assertFalse(config.is_enabled(EntityType.PHONE))
        self.assertFalse(config.is_enabled(EntityType.PLACE))
        self.assertFalse(config.is_enabled(EntityType.URL))
        self.assertFalse(config.is_enabled(EntityType.IP_ADDRESS))
        self.assertFalse(config.is_enabled(EntityType.ORGANISATION))

    def test_entity_config_with_disabled(self):
        """Test EntityConfig with_disabled method."""
        config = EntityConfig().with_disabled(
            EntityType.EMAIL, EntityType.PHONE, EntityType.URL
        )

        # Should disable specified entities
        self.assertFalse(config.is_enabled(EntityType.EMAIL))
        self.assertFalse(config.is_enabled(EntityType.PHONE))
        self.assertFalse(config.is_enabled(EntityType.URL))

        # Should keep other entities enabled
        self.assertTrue(config.is_enabled(EntityType.PERSON))
        self.assertTrue(config.is_enabled(EntityType.PLACE))
        self.assertTrue(config.is_enabled(EntityType.IP_ADDRESS))
        self.assertTrue(config.is_enabled(EntityType.CREDIT_CARD))
        self.assertTrue(config.is_enabled(EntityType.ORGANISATION))

    def test_entity_config_with_enabled(self):
        """Test EntityConfig with_enabled method."""
        config = EntityConfig().with_enabled(
            EntityType.PERSON, EntityType.CREDIT_CARD
        )

        # Should only enable specified entities
        self.assertTrue(config.is_enabled(EntityType.PERSON))
        self.assertTrue(config.is_enabled(EntityType.CREDIT_CARD))

        # Should disable all other entities
        self.assertFalse(config.is_enabled(EntityType.EMAIL))
        self.assertFalse(config.is_enabled(EntityType.PHONE))
        self.assertFalse(config.is_enabled(EntityType.PLACE))
        self.assertFalse(config.is_enabled(EntityType.URL))
        self.assertFalse(config.is_enabled(EntityType.IP_ADDRESS))
        self.assertFalse(config.is_enabled(EntityType.ORGANISATION))

    def test_entity_config_default_all_enabled(self):
        """Test EntityConfig defaults to all entities enabled."""
        config = EntityConfig()

        # All entity types should be enabled by default
        for entity_type in EntityType:
            self.assertTrue(config.is_enabled(entity_type))

    @parameterized.expand(
        [
            # (description, disabled_types, expected_detected_types)
            (
                "disable_email_phone",
                [EntityType.EMAIL, EntityType.PHONE],
                [
                    EntityType.PERSON,
                    EntityType.PLACE,
                    EntityType.URL,
                    EntityType.IP_ADDRESS,
                    EntityType.CREDIT_CARD,
                    EntityType.ORGANISATION,
                ],
            ),
            (
                "disable_locations",
                [EntityType.PLACE, EntityType.IP_ADDRESS, EntityType.URL],
                [
                    EntityType.PERSON,
                    EntityType.EMAIL,
                    EntityType.PHONE,
                    EntityType.CREDIT_CARD,
                    EntityType.ORGANISATION,
                ],
            ),
            (
                "disable_persons_organisations",
                [EntityType.PERSON, EntityType.ORGANISATION],
                [
                    EntityType.EMAIL,
                    EntityType.PHONE,
                    EntityType.PLACE,
                    EntityType.URL,
                    EntityType.IP_ADDRESS,
                    EntityType.CREDIT_CARD,
                ],
            ),
            (
                "disable_all_except_credit_card",
                [
                    EntityType.PERSON,
                    EntityType.EMAIL,
                    EntityType.PHONE,
                    EntityType.PLACE,
                    EntityType.URL,
                    EntityType.IP_ADDRESS,
                    EntityType.ORGANISATION,
                ],
                [EntityType.CREDIT_CARD],
            ),
        ]
    )
    def test_selective_detection_with_disabled_types(
        self, description, disabled_types, expected_detected_types
    ):
        """Test selective detection with specific types disabled."""
        config = EntityConfig().with_disabled(*disabled_types)
        detector = EntityDetector(config)

        detected_entities = detector.detect_entities(self.sample_text)
        detected_types = {entity.type for entity in detected_entities}

        # Verify disabled types are not detected
        for disabled_type in disabled_types:
            self.assertNotIn(
                disabled_type,
                detected_types,
                f"Disabled type {disabled_type} was detected",
            )

        # Verify expected types are detected
        for expected_type in expected_detected_types:
            if expected_type in self.expected_entities:
                # Only check if we expect this type in our sample text
                self.assertIn(
                    expected_type,
                    detected_types,
                    f"Expected type {expected_type} was not detected",
                )

    @parameterized.expand(
        [
            # (description, enabled_types, should_detect_types)
            (
                "only_person_email",
                [EntityType.PERSON, EntityType.EMAIL],
                [EntityType.PERSON, EntityType.EMAIL],
            ),
            (
                "only_financial",
                [EntityType.CREDIT_CARD],
                [EntityType.CREDIT_CARD],
            ),
            (
                "only_contact_info",
                [EntityType.EMAIL, EntityType.PHONE],
                [EntityType.EMAIL, EntityType.PHONE],
            ),
            (
                "only_locations",
                [EntityType.PLACE, EntityType.IP_ADDRESS, EntityType.URL],
                [EntityType.PLACE, EntityType.IP_ADDRESS, EntityType.URL],
            ),
        ]
    )
    def test_selective_detection_with_enabled_types(
        self, description, enabled_types, should_detect_types
    ):
        """Test selective detection with only specific types enabled."""
        config = EntityConfig().with_enabled(*enabled_types)
        detector = EntityDetector(config)

        detected_entities = detector.detect_entities(self.sample_text)
        detected_types = {entity.type for entity in detected_entities}

        # Verify only enabled types are detected
        for entity_type in EntityType:
            if entity_type in enabled_types:
                if entity_type in self.expected_entities:
                    # Should detect this type if it exists in sample text
                    self.assertIn(
                        entity_type,
                        detected_types,
                        f"Enabled type {entity_type} was not detected",
                    )
            else:
                # Should not detect this type
                self.assertNotIn(
                    entity_type,
                    detected_types,
                    f"Disabled type {entity_type} was detected",
                )

    def test_selective_cloaking_integration(self):
        """Test selective filtering integration with cloak method."""
        # Create shield that only detects persons and emails
        config = EntityConfig().with_enabled(
            EntityType.PERSON, EntityType.EMAIL
        )
        shield = LLMShield(entity_config=config)

        cloaked_text, entity_map = shield.cloak(self.sample_text)

        # Should have person and email placeholders (flexible numbering)
        self.assertIn("<PERSON_", cloaked_text)
        self.assertIn("<EMAIL_", cloaked_text)

        # Should not have other types of placeholders
        self.assertNotIn("<PHONE_", cloaked_text)
        self.assertNotIn("<PLACE_", cloaked_text)
        self.assertNotIn("<URL_", cloaked_text)
        self.assertNotIn("<IP_ADDRESS_", cloaked_text)
        self.assertNotIn("<CREDIT_CARD_", cloaked_text)
        self.assertNotIn("<ORGANISATION_", cloaked_text)

        # Entity map should only contain person and email entities
        detected_types = set()
        for placeholder in entity_map:
            if placeholder.startswith("<PERSON_"):
                detected_types.add("PERSON")
            elif placeholder.startswith("<EMAIL_"):
                detected_types.add("EMAIL")
            else:
                self.fail(f"Unexpected placeholder type: {placeholder}")

        self.assertEqual(detected_types, {"PERSON", "EMAIL"})

    def test_selective_ask_method_integration(self):
        """Test selective filtering with ask method using mock LLM."""

        def mock_llm(prompt, **kwargs):
            # Simple mock that echoes the input
            return f"Response: {prompt}"

        # Create shield that disables contact information
        shield = LLMShield.disable_contacts(llm_func=mock_llm)

        response = shield.ask(
            prompt="Contact Dr. John Doe at john@company.com or call "
            "555-123-4567"
        )

        # Response should have person protected but not email/phone
        self.assertIn("John Doe", response)  # Person should be uncloaked
        self.assertIn("john@company.com", response)  # Email not protected
        self.assertIn("555-123-4567", response)  # Phone not protected

    @parameterized.expand(
        [
            # (description, factory_method, text, should_cloak,
            #  should_not_cloak)
            (
                "disable_locations_test",
                "disable_locations",
                "Visit Dr. John at https://example.com in New York",
                ["John"],  # Should cloak person (Dr. prefix helps detection)
                [
                    "https://example.com",
                    "New York",
                ],  # Should not cloak locations
            ),
            (
                "disable_persons_test",
                "disable_persons",
                "Contact Dr. John Doe at john@example.com",
                ["john@example.com"],  # Should cloak email
                ["John Doe"],  # Should not cloak person
            ),
            (
                "disable_contacts_test",
                "disable_contacts",
                "Call Dr. John at john@example.com or 555-123-4567",
                ["John"],  # Should cloak person
                [
                    "john@example.com",
                    "555-123-4567",
                ],  # Should not cloak contacts
            ),
            (
                "only_financial_test",
                "only_financial",
                "Dr. John pays with card 4111111111111111 at john@example.com",
                ["4111111111111111"],  # Should cloak credit card (valid Luhn)
                ["John", "john@example.com"],  # Should not cloak others
            ),
        ]
    )
    def test_factory_methods_integration(
        self, description, factory_method, text, should_cloak, should_not_cloak
    ):
        """Test factory methods integration with cloaking."""
        # Get the factory method from LLMShield class
        shield = getattr(LLMShield, factory_method)()

        cloaked_text, entity_map = shield.cloak(text)

        # Verify entities that should be cloaked are cloaked
        for entity_value in should_cloak:
            self.assertNotIn(
                entity_value,
                cloaked_text,
                f"Entity '{entity_value}' should be cloaked but wasn't",
            )
            # Should be in entity map
            self.assertIn(
                entity_value,
                entity_map.values(),
                f"Entity '{entity_value}' not found in entity map",
            )

        # Verify entities that should not be cloaked remain in text
        for entity_value in should_not_cloak:
            self.assertIn(
                entity_value,
                cloaked_text,
                f"Entity '{entity_value}' should not be cloaked but was",
            )
            # Should not be in entity map
            self.assertNotIn(
                entity_value,
                entity_map.values(),
                f"Entity '{entity_value}' found in entity map "
                "but shouldn't be",
            )

    def test_performance_with_selective_detection(self):
        """Test that selective detection improves performance."""
        import time  # noqa: PLC0415

        # Test text with many different entity types
        large_text = " ".join([self.sample_text] * 100)  # Repeat 100 times

        # Measure time with all entities enabled
        start_time = time.time()
        detector_all = EntityDetector()
        entities_all = detector_all.detect_entities(large_text)
        time_all = time.time() - start_time

        # Measure time with only one entity type enabled
        start_time = time.time()
        config_limited = EntityConfig().with_enabled(EntityType.CREDIT_CARD)
        detector_limited = EntityDetector(config_limited)
        entities_limited = detector_limited.detect_entities(large_text)
        time_limited = time.time() - start_time

        # Selective detection should be faster
        self.assertLess(
            time_limited,
            time_all,
            "Selective detection should be faster than full detection",
        )

        # Selective detection should find fewer entities
        self.assertLess(
            len(entities_limited),
            len(entities_all),
            "Selective detection should find fewer entities",
        )

    def test_edge_case_no_entities_enabled(self):
        """Test edge case where no entity types are enabled."""
        config = EntityConfig().with_enabled()  # Enable nothing
        detector = EntityDetector(config)

        detected_entities = detector.detect_entities(self.sample_text)

        # Should detect no entities
        self.assertEqual(
            len(detected_entities),
            0,
            "No entities should be detected when none are enabled",
        )

    def test_edge_case_disable_nonexistent_entities(self):
        """Test disabling entity types that don't exist in text."""
        # Disable entities that aren't in our sample text
        config = EntityConfig().with_disabled(EntityType.CONCEPT)
        detector = EntityDetector(config)

        detected_entities = detector.detect_entities(
            "Simple text with no concepts like API or SQL"
        )

        # Should work normally and not crash
        detected_types = {entity.type for entity in detected_entities}
        self.assertNotIn(EntityType.CONCEPT, detected_types)

    def test_entity_config_immutability(self):
        """Test that EntityConfig methods return new instances."""
        original_config = EntityConfig()

        # with_disabled should return new instance
        disabled_config = original_config.with_disabled(EntityType.EMAIL)
        self.assertIsNot(original_config, disabled_config)
        self.assertTrue(original_config.is_enabled(EntityType.EMAIL))
        self.assertFalse(disabled_config.is_enabled(EntityType.EMAIL))

        # with_enabled should return new instance
        enabled_config = original_config.with_enabled(EntityType.PERSON)
        self.assertIsNot(original_config, enabled_config)
        self.assertTrue(original_config.is_enabled(EntityType.EMAIL))
        self.assertFalse(enabled_config.is_enabled(EntityType.EMAIL))

    def test_chaining_entity_config_methods(self):
        """Test chaining EntityConfig methods."""
        config = (
            EntityConfig()
            .with_disabled(EntityType.EMAIL, EntityType.PHONE)
            .with_disabled(EntityType.URL)
        )

        # All specified types should be disabled
        self.assertFalse(config.is_enabled(EntityType.EMAIL))
        self.assertFalse(config.is_enabled(EntityType.PHONE))
        self.assertFalse(config.is_enabled(EntityType.URL))

        # Other types should still be enabled
        self.assertTrue(config.is_enabled(EntityType.PERSON))
        self.assertTrue(config.is_enabled(EntityType.PLACE))

    @parameterized.expand(
        [
            # (description, config_method, args, expected_enabled,
            #  expected_disabled)
            (
                "disable_locations_config",
                "disable_locations",
                [],
                [
                    EntityType.PERSON,
                    EntityType.EMAIL,
                    EntityType.PHONE,
                    EntityType.CREDIT_CARD,
                    EntityType.ORGANISATION,
                ],
                [EntityType.PLACE, EntityType.IP_ADDRESS, EntityType.URL],
            ),
            (
                "disable_persons_config",
                "disable_persons",
                [],
                [
                    EntityType.EMAIL,
                    EntityType.PHONE,
                    EntityType.PLACE,
                    EntityType.URL,
                    EntityType.IP_ADDRESS,
                    EntityType.CREDIT_CARD,
                    EntityType.ORGANISATION,
                ],
                [EntityType.PERSON],
            ),
            (
                "disable_contacts_config",
                "disable_contacts",
                [],
                [
                    EntityType.PERSON,
                    EntityType.PLACE,
                    EntityType.URL,
                    EntityType.IP_ADDRESS,
                    EntityType.CREDIT_CARD,
                    EntityType.ORGANISATION,
                ],
                [EntityType.EMAIL, EntityType.PHONE],
            ),
            (
                "only_financial_config",
                "only_financial",
                [],
                [EntityType.CREDIT_CARD],
                [
                    EntityType.PERSON,
                    EntityType.EMAIL,
                    EntityType.PHONE,
                    EntityType.PLACE,
                    EntityType.URL,
                    EntityType.IP_ADDRESS,
                    EntityType.ORGANISATION,
                ],
            ),
        ]
    )
    def test_factory_config_methods(
        self,
        description,
        config_method,
        args,
        expected_enabled,
        expected_disabled,
    ):
        """Test EntityConfig factory methods."""
        config = getattr(EntityConfig, config_method)(*args)

        # Verify expected enabled types
        for entity_type in expected_enabled:
            self.assertTrue(
                config.is_enabled(entity_type),
                f"{entity_type} should be enabled in {config_method}",
            )

        # Verify expected disabled types
        for entity_type in expected_disabled:
            self.assertFalse(
                config.is_enabled(entity_type),
                f"{entity_type} should be disabled in {config_method}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
