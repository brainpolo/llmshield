"""Entity caching system.

Description:
    This subpackage provides efficient caching mechanisms for entity
    dictionaries used in detection. It implements a singleton pattern to
    ensure resource efficiency and consistent performance across multiple
    LLMShield instances.

Classes:
    EntityDictionaryCache: Singleton cache for entity dictionaries

Author:
    LLMShield by brainpolo, 2025
"""

from .entity_cache import EntityDictionaryCache

__all__ = ["EntityDictionaryCache"]
