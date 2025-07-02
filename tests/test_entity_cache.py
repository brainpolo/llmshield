"""Test entity cache singleton implementation.

Description:
    This test module provides comprehensive testing for the
    EntityDictionaryCache singleton, including lazy loading, thread safety,
    cache invalidation, and dictionary file loading behaviour.

Test Classes:
    - TestEntityDictionaryCache: Tests singleton cache functionality
    - TestGetEntityCache: Tests cache retrieval function

Author:
    LLMShield by brainpolo, 2025
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

        # Manually set _initialized to False to test the inner check
        cache._initialized = False

        # Call __init__ again - should set _initialized back to True
        cache.__init__()

        self.assertTrue(cache._initialized)

    @patch("llmshield.error_handling.resources")
    def test_cities_property_lazy_loading(self, mock_resources):
        """Test cities property with lazy loading."""
        # Mock file content
        mock_file = mock_open(read_data="london\nparis\nnew york\n")
        (
            mock_resources.files.return_value.joinpath.return_value.open.return_value
        ) = mock_file.return_value

        cache = EntityDictionaryCache()

        # First access should load
        cities = cache.cities
        self.assertIsInstance(cities, frozenset)
        self.assertIn("london", cities)
        self.assertIn("paris", cities)
        self.assertIn("new york", cities)

        # Second access should use cached version
        cities2 = cache.cities
        self.assertIs(cities, cities2)

    @patch("llmshield.error_handling.resources")
    def test_countries_property_lazy_loading(self, mock_resources):
        """Test countries property with lazy loading."""
        mock_file = mock_open(read_data="united kingdom\nfrance\ncanada\n")
        (
            mock_resources.files.return_value.joinpath.return_value.open.return_value
        ) = mock_file.return_value

        cache = EntityDictionaryCache()

        countries = cache.countries
        self.assertIsInstance(countries, frozenset)
        self.assertIn("united kingdom", countries)
        self.assertIn("france", countries)
        self.assertIn("canada", countries)

    @patch("llmshield.error_handling.resources")
    def test_organisations_property_lazy_loading(self, mock_resources):
        """Test organisations property with lazy loading."""
        mock_file = mock_open(read_data="microsoft\ngoogle\namazon\n")
        (
            mock_resources.files.return_value.joinpath.return_value.open.return_value
        ) = mock_file.return_value

        cache = EntityDictionaryCache()

        organisations = cache.organisations
        self.assertIsInstance(organisations, frozenset)
        self.assertIn("microsoft", organisations)
        self.assertIn("google", organisations)
        self.assertIn("amazon", organisations)

    @patch("llmshield.error_handling.resources")
    def test_english_corpus_property_lazy_loading(self, mock_resources):
        """Test english_corpus property with lazy loading."""
        mock_file = mock_open(read_data="the\nand\nof\nto\na\n")
        (
            mock_resources.files.return_value.joinpath.return_value.open.return_value
        ) = mock_file.return_value

        cache = EntityDictionaryCache()

        corpus = cache.english_corpus
        self.assertIsInstance(corpus, frozenset)
        self.assertIn("the", corpus)
        self.assertIn("and", corpus)
        self.assertIn("of", corpus)

    @patch("llmshield.error_handling.resources")
    def test_get_all_places(self, mock_resources):
        """Test get_all_places method."""
        mock_file = mock_open(read_data="london\nparis\nuk\nfrance\n")
        (
            mock_resources.files.return_value.joinpath.return_value.open.return_value
        ) = mock_file.return_value

        cache = EntityDictionaryCache()

        all_places = cache.get_all_places()

        # Should contain both cities and countries
        self.assertIn("london", all_places)
        self.assertIn("paris", all_places)
        self.assertIn("uk", all_places)
        self.assertIn("france", all_places)

    @patch("llmshield.error_handling.resources")
    def test_is_place_method(self, mock_resources):
        """Test is_place method."""
        mock_file = mock_open(read_data="london\nparis\nuk\nfrance\n")
        (
            mock_resources.files.return_value.joinpath.return_value.open.return_value
        ) = mock_file.return_value

        cache = EntityDictionaryCache()

        # Test city lookup
        self.assertTrue(cache.is_place("london"))
        self.assertTrue(cache.is_place("paris"))

        # Test country lookup
        self.assertTrue(cache.is_place("uk"))
        self.assertTrue(cache.is_place("france"))

        # Test non-place
        self.assertFalse(cache.is_place("notaplace"))

    @patch("llmshield.error_handling.resources")
    def test_is_organisation_method(self, mock_resources):
        """Test is_organisation method."""
        mock_file = mock_open(read_data="microsoft\ngoogle\n")
        (
            mock_resources.files.return_value.joinpath.return_value.open.return_value
        ) = mock_file.return_value

        cache = EntityDictionaryCache()

        self.assertTrue(cache.is_organisation("microsoft"))
        self.assertTrue(cache.is_organisation("google"))
        self.assertFalse(cache.is_organisation("notanorg"))

    @patch("llmshield.error_handling.resources")
    def test_is_english_word_method(self, mock_resources):
        """Test is_english_word method."""
        mock_file = mock_open(read_data="the\nand\nof\n")
        (
            mock_resources.files.return_value.joinpath.return_value.open.return_value
        ) = mock_file.return_value

        cache = EntityDictionaryCache()

        self.assertTrue(cache.is_english_word("the"))
        self.assertTrue(cache.is_english_word("and"))
        self.assertFalse(cache.is_english_word("notaword"))

    @patch("llmshield.error_handling.resources")
    def test_preload_all_method(self, mock_resources):
        """Test preload_all method."""
        mock_file = mock_open(read_data="test\n")
        (
            mock_resources.files.return_value.joinpath.return_value.open.return_value
        ) = mock_file.return_value

        cache = EntityDictionaryCache()

        # All should be None initially
        self.assertIsNone(cache._cities)
        self.assertIsNone(cache._countries)
        self.assertIsNone(cache._organisations)
        self.assertIsNone(cache._english_corpus)

        # Preload all
        cache.preload_all()

        # All should be loaded now
        self.assertIsNotNone(cache._cities)
        self.assertIsNotNone(cache._countries)
        self.assertIsNotNone(cache._organisations)
        self.assertIsNotNone(cache._english_corpus)

    @patch("llmshield.error_handling.resources")
    def test_get_memory_stats_empty(self, mock_resources):
        """Test get_memory_stats when nothing is loaded."""
        cache = EntityDictionaryCache()

        stats = cache.get_memory_stats()

        # Should be empty since nothing is loaded
        self.assertEqual(stats, {})

    @patch("llmshield.error_handling.resources")
    def test_get_memory_stats_partial(self, mock_resources):
        """Test get_memory_stats with partial loading."""
        mock_file = mock_open(read_data="item1\nitem2\nitem3\n")
        (
            mock_resources.files.return_value.joinpath.return_value.open.return_value
        ) = mock_file.return_value

        cache = EntityDictionaryCache()

        # Load only cities
        _ = cache.cities

        stats = cache.get_memory_stats()

        # Should only have cities
        self.assertIn("cities", stats)
        self.assertEqual(stats["cities"], 3)
        self.assertNotIn("countries", stats)
        self.assertNotIn("organisations", stats)
        self.assertNotIn("english_corpus", stats)

    @patch("llmshield.error_handling.resources")
    def test_get_memory_stats_full(self, mock_resources):
        """Test get_memory_stats with full loading."""
        mock_file = mock_open(read_data="item1\nitem2\n")
        (
            mock_resources.files.return_value.joinpath.return_value.open.return_value
        ) = mock_file.return_value

        cache = EntityDictionaryCache()

        # Load all
        cache.preload_all()

        stats = cache.get_memory_stats()

        # Should have all dictionaries
        self.assertIn("cities", stats)
        self.assertIn("countries", stats)
        self.assertIn("organisations", stats)
        self.assertIn("english_corpus", stats)

        # Should have all dictionaries loaded
        self.assertEqual(len(stats), 4)

    @patch("llmshield.error_handling.resources")
    def test_load_dict_file_with_comments(self, mock_resources):
        """Test _load_dict_file handles comments and empty lines."""
        mock_file = mock_open(
            read_data="# This is a comment\nitem1\n\n"
            "# Another comment\nitem2\n\n"
        )
        (
            mock_resources.files.return_value.joinpath.return_value.open.return_value
        ) = mock_file.return_value

        cache = EntityDictionaryCache()

        result = cache._load_dict_file("test.txt")

        # Should only contain actual items, not comments or empty lines
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
            time.sleep(0.1)  # Small delay to increase chance of race condition
            return original_open.return_value

        (
            mock_resources.files.return_value.joinpath.return_value.open.side_effect
        ) = delayed_open

        cache = EntityDictionaryCache()
        results = []

        def load_cities():
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
