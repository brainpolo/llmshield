"""Provider system for optimal support across different LLM APIs.

Description:
    This subpackage provides consistent behaviour and parameter handling for
    various LLM providers. It automatically detects and adapts to different
    LLM APIs while maintaining a uniform interface.

Functions:
    get_provider: Factory function to get appropriate provider for an LLM
        function

Author:
    LLMShield by brainpolo, 2025
"""

from .provider_factory import get_provider

__all__ = ["get_provider"]
