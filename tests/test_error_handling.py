"""Test error handling and validation functionality.

Description:
    This test module provides comprehensive testing for the error handling
    framework, including custom exceptions, validation functions, and safe
    operations.

Test Classes:
    - TestExceptions: Tests custom exception hierarchy
    - TestValidation: Tests validation functions
    - TestErrorHandling: Tests safe operations and error handling

Author:
    LLMShield by brainpolo, 2025
"""

import unittest
from unittest.mock import Mock, patch

from llmshield.error_handling import (
    safe_resource_load,
    validate_delimiters,
    validate_entity_map,
    validate_prompt_input,
)
from llmshield.exceptions import (
    CloakingError,
    EntityDetectionError,
    LLMShieldError,
    ProviderError,
    ResourceLoadError,
    UncloakingError,
    ValidationError,
)


class TestExceptions(unittest.TestCase):
    """Test custom exception hierarchy."""

    def test_exception_inheritance(self):
        """Test that all exceptions inherit from LLMShieldError."""
        exception_classes = [
            EntityDetectionError,
            ResourceLoadError,
            ValidationError,
            CloakingError,
            UncloakingError,
            ProviderError,
        ]

        for exc_class in exception_classes:
            with self.subTest(exception=exc_class.__name__):
                exc = exc_class("Test error")
                self.assertIsInstance(exc, LLMShieldError)
                self.assertIsInstance(exc, Exception)

    def test_validation_error_dual_inheritance(self):
        """Test ValidationError dual inheritance."""
        exc = ValidationError("Test validation error")
        self.assertIsInstance(exc, LLMShieldError)
        self.assertIsInstance(exc, ValueError)
        self.assertEqual(str(exc), "Test validation error")

    def test_exception_messages(self):
        """Test exception messages are preserved correctly."""
        test_message = "This is a test error message"

        exceptions = [
            (EntityDetectionError, test_message),
            (ResourceLoadError, test_message),
            (ValidationError, test_message),
            (CloakingError, test_message),
            (UncloakingError, test_message),
            (ProviderError, test_message),
        ]

        for exc_class, message in exceptions:
            with self.subTest(exception=exc_class.__name__):
                exc = exc_class(message)
                self.assertEqual(str(exc), message)


class TestValidation(unittest.TestCase):
    """Test validation functions."""

    def test_validate_prompt_input_valid(self):
        """Test validate_prompt_input with valid inputs."""
        # Valid prompt
        validate_prompt_input(prompt="Test prompt")

        # Valid message
        validate_prompt_input(message="Test message")

        # Valid messages
        validate_prompt_input(messages=[{"content": "Test"}])

    def test_validate_prompt_input_invalid(self):
        """Test validate_prompt_input with invalid inputs."""
        # No input provided
        with self.assertRaises(ValidationError) as context:
            validate_prompt_input()
        self.assertIn("Must provide either", str(context.exception))

        # Multiple inputs provided
        with self.assertRaises(ValidationError) as context:
            validate_prompt_input(prompt="test", message="test")
        self.assertIn("Do not provide both", str(context.exception))

        # Invalid type for prompt
        with self.assertRaises(ValidationError) as context:
            validate_prompt_input(prompt=123)
        self.assertIn("Prompt must be string", str(context.exception))

        # Invalid type for messages
        with self.assertRaises(ValidationError) as context:
            validate_prompt_input(messages="not a list")
        self.assertIn("Messages must be list", str(context.exception))

        # Empty messages list
        with self.assertRaises(ValidationError) as context:
            validate_prompt_input(messages=[])
        self.assertIn("Messages list cannot be empty", str(context.exception))

        # Test all three parameters provided
        with self.assertRaises(ValidationError) as context:
            validate_prompt_input(
                prompt="test", message="test", messages=[{"content": "test"}]
            )
        self.assertIn("Do not provide both", str(context.exception))

        # Test extremely long prompt (line 100)
        long_prompt = "x" * 1_000_001  # Exceeds MAX_PROMPT_LENGTH
        with self.assertRaises(ValidationError) as context:
            validate_prompt_input(prompt=long_prompt)
        self.assertIn("too long", str(context.exception))

        # Test non-dict message in messages list (line 116)
        with self.assertRaises(ValidationError) as context:
            validate_prompt_input(messages=["not a dict"])
        self.assertIn("must be dict", str(context.exception))

        # Test non-string content in message (line 122)
        with self.assertRaises(ValidationError) as context:
            validate_prompt_input(messages=[{"content": 123}])
        self.assertIn("content must be string", str(context.exception))

        # Test non-string delimiter in validate_delimiters (line 148)
        with self.assertRaises(ValidationError) as context:
            validate_delimiters("start", 123)
        self.assertIn("End delimiter must be string", str(context.exception))

    def test_validate_delimiters_valid(self):
        """Test validate_delimiters with valid inputs."""
        validate_delimiters("<", ">")
        validate_delimiters("[[", "]]")
        validate_delimiters("START", "END")

    def test_validate_delimiters_invalid(self):
        """Test validate_delimiters with invalid inputs."""
        # Empty delimiters
        with self.assertRaises(ValidationError) as context:
            validate_delimiters("", ">")
        self.assertIn(
            "Start delimiter cannot be empty", str(context.exception)
        )

        # Identical delimiters
        with self.assertRaises(ValidationError) as context:
            validate_delimiters("<", "<")
        self.assertIn("cannot be identical", str(context.exception))

        # Too long delimiters
        with self.assertRaises(ValidationError) as context:
            validate_delimiters("VERYLONGSTART", ">")
        self.assertIn("should be short", str(context.exception))

    def test_validate_entity_map_valid(self):
        """Test validate_entity_map with valid inputs."""
        # Valid entity map provided
        entity_map = {"[PERSON_0]": "John", "[EMAIL_0]": "john@example.com"}
        result = validate_entity_map(entity_map, None)
        self.assertEqual(result, entity_map)

        # Using last entity map
        last_map = {"[PLACE_0]": "London"}
        result = validate_entity_map(None, last_map)
        self.assertEqual(result, last_map)

    def test_validate_entity_map_invalid(self):
        """Test validate_entity_map with invalid inputs."""
        # No maps provided
        with self.assertRaises(ValidationError) as context:
            validate_entity_map(None, None)
        self.assertIn("No entity mapping provided", str(context.exception))

        # Invalid type
        with self.assertRaises(ValidationError) as context:
            validate_entity_map("not a dict", None)
        self.assertIn("Entity map must be dict", str(context.exception))

        # Non-string keys
        with self.assertRaises(ValidationError) as context:
            validate_entity_map({123: "value"}, None)
        self.assertIn("Entity map key must be string", str(context.exception))

        # Non-string values
        with self.assertRaises(ValidationError) as context:
            validate_entity_map({"key": 456}, None)
        self.assertIn(
            "Entity map value must be string", str(context.exception)
        )


class TestErrorHandling(unittest.TestCase):
    """Test safe operations and error handling utilities."""

    @patch("llmshield.error_handling.resources")
    def test_safe_resource_load_valid(self, mock_resources):
        """Test safe_resource_load with valid resource."""
        # Mock successful file load
        mock_file = Mock()
        mock_file.__enter__ = Mock(
            return_value=Mock(
                __iter__=Mock(
                    return_value=iter(["line1\n", "# comment\n", "line2\n"])
                )
            )
        )
        mock_file.__exit__ = Mock(return_value=False)

        (
            mock_resources.files.return_value.joinpath.return_value.open.return_value
        ) = mock_file

        result = safe_resource_load("test.package", "test.txt")
        self.assertEqual(result, ["line1", "line2"])

    @patch("llmshield.error_handling.resources")
    def test_safe_resource_load_file_not_found(self, mock_resources):
        """Test safe_resource_load with missing file."""
        (
            mock_resources.files.return_value.joinpath.return_value.open.side_effect
        ) = FileNotFoundError()

        with self.assertRaises(ResourceLoadError) as context:
            safe_resource_load("test.package", "missing.txt")
        self.assertIn("Resource not found", str(context.exception))

    @patch("llmshield.error_handling.resources")
    def test_safe_resource_load_generic_error(self, mock_resources):
        """Test safe_resource_load with generic error."""
        (
            mock_resources.files.return_value.joinpath.return_value.open.side_effect
        ) = Exception("Generic error")

        with self.assertRaises(ResourceLoadError) as context:
            safe_resource_load("test.package", "error.txt")
        self.assertIn("Generic error", str(context.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
