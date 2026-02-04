"""Test entity cache singleton implementation.

Description:
    This test module provides comprehensive testing for the
    EntityDictionaryCache singleton, including lazy loading, thread safety,
    cache invalidation, and dictionary file loading behaviour.

Test Classes:
    - TestEntityDictionaryCache: Tests singleton cache functionality
    - TestGetEntityCache: Tests cache retrieval function

Author:
    LLMShield by brainpolo, 2025-2026
"""

# Standard Library Imports
import threading
import time
import unittest
from unittest.mock import mock_open, patch

# Local Imports
from llmshield.cache.entity_cache import (
    EntityDictionaryCache,
    get_entity_cache,
)
from llmshield.exceptions import ResourceLoadError


class TestEntityDictionaryCache(unittest.TestCase):
    """Test suite for EntityDictionaryCache singleton."""

    def setUp(self):
        """Reset singleton before each test."""
        # Reset singleton instance
        EntityDictionaryCache._instance = None

    def tearDown(self):
        """Clean up after each test."""
        # Reset singleton instance
        EntityDictionaryCache._instance = None

    def test_singleton_pattern(self):
        """Test that EntityDictionaryCache follows singleton pattern."""
        cache1 = EntityDictionaryCache()
        cache2 = EntityDictionaryCache()

        # Should be the same instance
        self.assertIs(cache1, cache2)

    def test_thread_safety_singleton(self):
        """Test singleton thread safety."""
        instances = []

        def create_instance():
            """Create and append an instance of EntityDictionaryCache."""
            instances.append(EntityDictionaryCache())

        # Create multiple instances from different threads
        threads = [threading.Thread(target=create_instance) for _ in range(10)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All instances should be the same
        for instance in instances:
            self.assertIs(instance, instances[0])

    def test_lazy_initialization(self):
        """Test that initialization only happens once."""
        cache = EntityDictionaryCache()

        # Should be initialized
        self.assertTrue(cache._initialized)

        # Creating another instance should not re-initialize
        cache2 = EntityDictionaryCache()
        self.assertIs(cache, cache2)
        self.assertTrue(cache2._initialized)

    def test_double_checked_locking_init(self):
        """Test the double-checked locking pattern in __init__."""
        cache = EntityDictionaryCache()
        cache._initialized = False
        cache.__init__()
        self.assertTrue(cache._initialized)

    def test_double_checked_locking_race_condition(self):
        """Test race condition in double-checked locking pattern."""
        cache = EntityDictionaryCache()
        init_count = 0

        def counting_init(*args, **kwargs):
            """Initialise the function and count calls."""
            nonlocal init_count
            if hasattr(cache, "_initialized") and cache._initialized:
                return
            time.sleep(0.001)
            with cache._lock:
                if cache._initialized:
                    return
                init_count += 1
                cache._initialized = True

        cache.__init__ = counting_init
        cache._initialized = False

        threads = [threading.Thread(target=cache.__init__) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(init_count, 1)
        self.assertTrue(cache._initialized)

    def test_init_race_condition_real(self):
        """Test real race condition in __init__ to hit line 51."""
        EntityDictionaryCache._instance = None
        barrier = threading.Barrier(5)
        instances = []

        def create_with_delay():
            """Create instance after a barrier synchronization wait."""
            barrier.wait()
            instances.append(EntityDictionaryCache())

        threads = [
            threading.Thread(target=create_with_delay) for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should be same instance
        self.assertTrue(all(inst is instances[0] for inst in instances))

        # Test line 51 race condition
        cache = instances[0]
        cache._initialized = False

        def set_initialised():
            """Set initialised state with a small delay."""
            time.sleep(0.001)
            with cache._lock:
                cache._initialized = True

        setter = threading.Thread(target=set_initialised)
        setter.start()
        cache.__init__()
        setter.join()

        self.assertTrue(cache._initialized)

    # skipcq: PYL-R0201
    def _mock_resource_file(self, mock_resources, content):
        """Mock resource file loading."""
        mock_file = mock_open(read_data=content)
        mock_open_return = mock_file.return_value
        (
            mock_resources.files.return_value.joinpath.return_value.open.return_value
        ) = mock_open_return

    @patch("llmshield.error_handling.resources")
    def test_cities_property_lazy_loading(self, mock_resources):
        """Test cities property with lazy loading."""
        self._mock_resource_file(mock_resources, "london\nparis\nnew york\n")
        cache = EntityDictionaryCache()

        cities = cache.cities
        self.assertIsInstance(cities, frozenset)
        self.assertEqual(cities, frozenset(["london", "paris", "new york"]))
        self.assertIs(cities, cache.cities)  # Cached

    @patch("llmshield.error_handling.resources")
    def test_countries_property_lazy_loading(self, mock_resources):
        """Test countries property with lazy loading."""
        self._mock_resource_file(
            mock_resources, "united kingdom\nfrance\ncanada\n"
        )
        cache = EntityDictionaryCache()

        countries = cache.countries
        self.assertIsInstance(countries, frozenset)
        expected = frozenset(["united kingdom", "france", "canada"])
        self.assertEqual(countries, expected)

    @patch("llmshield.error_handling.resources")
    def test_organisations_property_lazy_loading(self, mock_resources):
        """Test organisations property with lazy loading."""
        self._mock_resource_file(mock_resources, "microsoft\ngoogle\namazon\n")
        cache = EntityDictionaryCache()

        organisations = cache.organisations
        self.assertIsInstance(organisations, frozenset)
        expected = frozenset(["microsoft", "google", "amazon"])
        self.assertEqual(organisations, expected)

    @patch("llmshield.error_handling.resources")
    def test_english_corpus_property_lazy_loading(self, mock_resources):
        """Test english_corpus property with lazy loading."""
        self._mock_resource_file(mock_resources, "the\nand\nof\nto\na\n")
        cache = EntityDictionaryCache()

        corpus = cache.english_corpus
        self.assertIsInstance(corpus, frozenset)
        self.assertTrue({"the", "and", "of"}.issubset(corpus))

    @patch("llmshield.error_handling.resources")
    def test_get_all_places(self, mock_resources):
        """Test get_all_places method."""
        self._mock_resource_file(mock_resources, "london\nparis\nuk\nfrance\n")
        cache = EntityDictionaryCache()

        all_places = cache.get_all_places()
        expected = frozenset(["london", "paris", "uk", "france"])
        self.assertEqual(all_places, expected)

    @patch("llmshield.error_handling.resources")
    def test_is_place_method(self, mock_resources):
        """Test is_place method."""
        self._mock_resource_file(mock_resources, "london\nparis\nuk\nfrance\n")
        cache = EntityDictionaryCache()

        # Test places
        for place in ["london", "paris", "uk", "france"]:
            self.assertTrue(cache.is_place(place))
        self.assertFalse(cache.is_place("notaplace"))

    @patch("llmshield.error_handling.resources")
    def test_is_organisation_method(self, mock_resources):
        """Test is_organisation method."""
        self._mock_resource_file(mock_resources, "microsoft\ngoogle\n")
        cache = EntityDictionaryCache()

        self.assertTrue(cache.is_organisation("microsoft"))
        self.assertTrue(cache.is_organisation("google"))
        self.assertFalse(cache.is_organisation("notanorg"))

    @patch("llmshield.error_handling.resources")
    def test_is_english_word_method(self, mock_resources):
        """Test is_english_word method."""
        self._mock_resource_file(mock_resources, "the\nand\nof\n")
        cache = EntityDictionaryCache()

        for word in ["the", "and"]:
            self.assertTrue(cache.is_english_word(word))
        self.assertFalse(cache.is_english_word("notaword"))

    @patch("llmshield.error_handling.resources")
    def test_preload_all_method(self, mock_resources):
        """Test preload_all method."""
        self._mock_resource_file(mock_resources, "test\n")
        cache = EntityDictionaryCache()

        # Verify all are None initially
        attrs = ["_cities", "_countries", "_organisations", "_english_corpus"]
        self.assertTrue(all(getattr(cache, attr) is None for attr in attrs))

        cache.preload_all()

        # All should be loaded now
        loaded = all(getattr(cache, attr) is not None for attr in attrs)
        self.assertTrue(loaded)

    @patch("llmshield.error_handling.resources")
    def test_get_memory_stats_empty(self, mock_resources):
        """Test get_memory_stats when nothing is loaded."""
        cache = EntityDictionaryCache()
        self.assertEqual(cache.get_memory_stats(), {})

    @patch("llmshield.error_handling.resources")
    def test_get_memory_stats_partial(self, mock_resources):
        """Test get_memory_stats with partial loading."""
        self._mock_resource_file(mock_resources, "item1\nitem2\nitem3\n")
        cache = EntityDictionaryCache()

        _ = cache.cities  # Load only cities
        stats = cache.get_memory_stats()

        self.assertEqual(stats, {"cities": 3})

    @patch("llmshield.error_handling.resources")
    def test_get_memory_stats_full(self, mock_resources):
        """Test get_memory_stats with full loading."""
        self._mock_resource_file(mock_resources, "item1\nitem2\n")
        cache = EntityDictionaryCache()

        cache.preload_all()
        stats = cache.get_memory_stats()

        # Should have all dictionaries loaded
        expected_keys = {
            "cities",
            "countries",
            "organisations",
            "english_corpus",
        }
        self.assertEqual(set(stats.keys()), expected_keys)
        self.assertEqual(len(stats), 4)

    @patch("llmshield.error_handling.resources")
    def test_load_dict_file_with_comments(self, mock_resources):
        """Test _load_dict_file handles comments and empty lines."""
        content = "# This is a comment\nitem1\n\n# Another comment\nitem2\n\n"
        self._mock_resource_file(mock_resources, content)

        cache = EntityDictionaryCache()
        result = cache._load_dict_file("test.txt")

        self.assertEqual(result, frozenset(["item1", "item2"]))

    @patch("llmshield.error_handling.resources")
    def test_load_dict_file_file_not_found(self, mock_resources):
        """Test _load_dict_file handles FileNotFoundError."""
        (
            mock_resources.files.return_value.joinpath.return_value.open.side_effect
        ) = FileNotFoundError("File not found")

        cache = EntityDictionaryCache()

        with self.assertRaises(ResourceLoadError) as context:
            cache._load_dict_file("missing.txt")

        self.assertIn(
            "Resource not found",
            str(context.exception),
        )

    @patch("llmshield.error_handling.resources")
    def test_load_dict_file_unicode_error(self, mock_resources):
        """Test _load_dict_file handles UnicodeDecodeError."""
        unicode_error = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid byte")
        (
            mock_resources.files.return_value.joinpath.return_value.open.side_effect
        ) = unicode_error

        cache = EntityDictionaryCache()

        with self.assertRaises(ResourceLoadError) as context:
            cache._load_dict_file("bad_encoding.txt")

        self.assertIn("bad_encoding.txt", str(context.exception))

    def test_get_entity_cache_function(self):
        """Test get_entity_cache function returns singleton."""
        cache1 = get_entity_cache()
        cache2 = get_entity_cache()

        # Should be the same instance
        self.assertIs(cache1, cache2)

        # Should be EntityDictionaryCache instance
        self.assertIsInstance(cache1, EntityDictionaryCache)

    @patch("llmshield.error_handling.resources")
    def test_thread_safety_lazy_loading(self, mock_resources):
        """Test thread safety during lazy loading."""
        # Mock to add delay and test race conditions
        original_open = mock_open(read_data="test\n")

        def delayed_open(*args, **kwargs):
            """Mock open with a small delay to test race conditions."""
            time.sleep(0.1)  # Small delay to increase chance of race condition
            return original_open.return_value

        (
            mock_resources.files.return_value.joinpath.return_value.open.side_effect
        ) = delayed_open

        cache = EntityDictionaryCache()
        results = []

        def load_cities():
            """Load cities from cache into the results list."""
            results.append(cache.cities)

        # Start multiple threads trying to load cities simultaneously
        threads = [threading.Thread(target=load_cities) for _ in range(5)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All threads should get the same frozenset instance
        for result in results:
            self.assertIs(result, results[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
