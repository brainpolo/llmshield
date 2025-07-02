"""Thread-safe singleton cache for entity dictionaries.

Description:
    This module implements a thread-safe singleton cache for storing entity
    dictionaries (cities, countries, organisations) used in entity detection.
    The singleton pattern ensures efficient memory usage and consistent
    performance across multiple EntityDetector instances.

Classes:
    EntityDictionaryCache: Singleton cache with lazy loading for entity
        dictionaries

Functions:
    get_entity_cache: Factory function to get the singleton instance

Author:
    LLMShield by brainpolo, 2025
"""

import threading

from ..error_handling import safe_resource_load


class EntityDictionaryCache:
    """Thread-safe singleton cache for entity dictionaries.

    This cache loads entity dictionaries once and provides O(1) lookups
    with support for selective entity type filtering.
    """

    _instance: "EntityDictionaryCache | None" = None
    _lock = threading.RLock()

    def __new__(cls) -> "EntityDictionaryCache":
        """Create or return singleton instance with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize cache on first instantiation only."""
        if getattr(self, "_initialized", False):
            return

        with self._lock:
            if self._initialized:
                return

            self._cities: frozenset[str] | None = None
            self._countries: frozenset[str] | None = None
            self._organisations: frozenset[str] | None = None
            self._english_corpus: frozenset[str] | None = None
            self._initialized = True

    @property
    def cities(self) -> frozenset[str]:
        """Get cities dictionary with lazy loading."""
        if self._cities is None:
            with self._lock:
                if self._cities is None:
                    self._cities = self._load_cities()
        return self._cities

    @property
    def countries(self) -> frozenset[str]:
        """Get countries dictionary with lazy loading."""
        if self._countries is None:
            with self._lock:
                if self._countries is None:
                    self._countries = self._load_countries()
        return self._countries

    @property
    def organisations(self) -> frozenset[str]:
        """Get organisations dictionary with lazy loading."""
        if self._organisations is None:
            with self._lock:
                if self._organisations is None:
                    self._organisations = self._load_organisations()
        return self._organisations

    @property
    def english_corpus(self) -> frozenset[str]:
        """Get English corpus with lazy loading."""
        if self._english_corpus is None:
            with self._lock:
                if self._english_corpus is None:
                    self._english_corpus = self._load_english_corpus()
        return self._english_corpus

    def get_all_places(self) -> frozenset[str]:
        """Get combined cities and countries set."""
        return self.cities | self.countries

    def is_place(self, text_lower: str) -> bool:
        """O(1) lookup for place entities."""
        return text_lower in self.cities or text_lower in self.countries

    def is_organisation(self, text_lower: str) -> bool:
        """O(1) lookup for organisation entities."""
        return text_lower in self.organisations

    def is_english_word(self, text_lower: str) -> bool:
        """O(1) lookup for English words."""
        return text_lower in self.english_corpus

    def preload_all(self) -> None:
        """Preload all dictionaries for optimal performance."""
        with self._lock:
            _ = self.cities
            _ = self.countries
            _ = self.organisations
            _ = self.english_corpus

    def get_memory_stats(self) -> dict[str, int]:
        """Get memory usage statistics for loaded dictionaries."""
        stats = {}
        if self._cities is not None:
            stats["cities"] = len(self._cities)
        if self._countries is not None:
            stats["countries"] = len(self._countries)
        if self._organisations is not None:
            stats["organisations"] = len(self._organisations)
        if self._english_corpus is not None:
            stats["english_corpus"] = len(self._english_corpus)
        return stats

    def _load_cities(self) -> frozenset[str]:
        """Load cities dictionary from resource file."""
        return self._load_dict_file("cities.txt")

    def _load_countries(self) -> frozenset[str]:
        """Load countries dictionary from resource file."""
        return self._load_dict_file("countries.txt")

    def _load_organisations(self) -> frozenset[str]:
        """Load organisations dictionary from resource file."""
        return self._load_dict_file("organisations.txt")

    def _load_english_corpus(self) -> frozenset[str]:
        """Load English corpus from resource file."""
        return self._load_dict_file("corpus/english.txt")

    def _load_dict_file(self, filename: str) -> frozenset[str]:
        """Load and process dictionary files.

        Args:
            filename: Name of the dictionary file to load

        Returns:
            Frozenset of lowercased, cleaned entries

        Raises:
            ResourceLoadError: If resource file cannot be loaded

        """
        entries = safe_resource_load(
            "llmshield.matchers.dicts",
            filename,
            f"Loading entity dictionary {filename}",
        )
        # Convert to lowercase and return as frozenset
        return frozenset(entry.lower() for entry in entries)


# Global cache instance for easy access
_cache_instance = EntityDictionaryCache()


def get_entity_cache() -> EntityDictionaryCache:
    """Get the global entity cache instance."""
    return _cache_instance
