"""Custom exception hierarchy for LLMShield.

Description:
    This module defines custom exceptions for better error handling and
    debugging throughout the LLMShield library. It provides a structured
    exception hierarchy for different types of errors that can occur during
    entity detection, cloaking, and uncloaking operations.

Classes:
    LLMShieldError: Base exception for all LLMShield operations
    EntityDetectionError: Raised when entity detection fails
    ResourceLoadError: Raised when resource files cannot be loaded
    ValidationError: Raised when input validation fails
    CloakingError: Raised when cloaking operations fail
    UncloakingError: Raised when uncloaking operations fail
    ProviderError: Raised when LLM provider operations fail

Author:
    LLMShield by brainpolo, 2025
"""


class LLMShieldError(Exception):
    """Base exception for all LLMShield operations.

    This is the base class for all custom exceptions in LLMShield.
    It provides a common interface for error handling and allows
    catching all LLMShield-specific errors with a single except clause.
    """

    pass


class EntityDetectionError(LLMShieldError):
    """Raised when entity detection fails.

    This exception is raised when the entity detection process
    encounters an error, such as invalid regex patterns or
    processing failures.
    """

    pass


class ResourceLoadError(LLMShieldError):
    """Raised when resource files cannot be loaded.

    This exception is raised when dictionary files or other
    resources cannot be loaded due to file not found, permission
    errors, or encoding issues.
    """

    pass


class ValidationError(LLMShieldError, ValueError):
    """Raised when input validation fails.

    This exception is raised when input parameters fail validation,
    such as None values where strings are expected, invalid types,
    or values exceeding allowed limits.

    Inherits from both LLMShieldError and ValueError for backward
    compatibility with existing error handling.
    """

    pass


class CloakingError(LLMShieldError):
    """Raised when cloaking operations fail.

    This exception is raised when the cloaking process encounters
    an error, such as delimiter conflicts or processing failures.
    """

    pass


class UncloakingError(LLMShieldError):
    """Raised when uncloaking operations fail.

    This exception is raised when the uncloaking process encounters
    an error, such as missing entity mappings or invalid placeholders.
    """

    pass


class ProviderError(LLMShieldError):
    """Raised when LLM provider operations fail.

    This exception is raised when provider-specific operations fail,
    such as parameter conversion errors or API incompatibilities.
    """

    pass
