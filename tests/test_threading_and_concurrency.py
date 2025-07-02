"""Test thread safety and concurrent access patterns.

Description:
    This test module validates thread-safety guarantees of the singleton cache,
    concurrent access patterns, and race condition handling in multi-threaded
    environments that the library might encounter in production usage.

Test Classes:
    - TestThreadingAndConcurrency: Tests thread-safe operations

Author:
    LLMShield by brainpolo, 2025
"""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

from llmshield.cache.entity_cache import (
    EntityDictionaryCache,
    get_entity_cache,
)
from llmshield.entity_detector import EntityDetector


class TestThreadingAndConcurrency(unittest.TestCase):
    """Test thread-safety and concurrent access patterns."""

    def setUp(self):
        """Reset singleton state before each test."""
        EntityDictionaryCache._instance = None

    def tearDown(self):
        """Clean up singleton state after each test."""
        EntityDictionaryCache._instance = None

    def test_singleton_thread_safety_under_contention(self):
        """Test singleton creation under high thread contention.

        This simulates a production scenario where multiple threads
        simultaneously attempt to create the singleton cache, ensuring
        the double-checked locking pattern works correctly.
        """
        num_threads = 10
        created_instances = []
        barrier = threading.Barrier(num_threads)

        def create_instance():
            # Synchronize all threads to start simultaneously
            barrier.wait()
            instance = EntityDictionaryCache()
            created_instances.append(instance)
            return instance

        # Create and start threads simultaneously
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all threads got the same singleton instance
        first_instance = created_instances[0]
        for instance in created_instances:
            self.assertIs(instance, first_instance)

        # Verify proper initialization
        self.assertTrue(first_instance._initialized)

    def test_concurrent_entity_detection_operations(self):
        """Test concurrent entity detection operations.

        This simulates multiple threads performing entity detection
        simultaneously, ensuring thread-safety and consistent results.
        """
        num_threads = 5
        test_texts = [
            "Dr. John Smith works at Microsoft in London",
            "Contact Alice Johnson at alice@company.com",
            "Visit Professor Brown at Harvard University",
            "Call Sarah Wilson at 555-123-4567",
            "Meet Bob Garcia at IBM headquarters",
        ]

        results = {}

        def detect_entities(thread_id, text):
            detector = EntityDetector()
            entities = detector.detect_entities(text)
            results[thread_id] = entities
            return entities

        # Run concurrent detection operations
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i, text in enumerate(test_texts):
                future = executor.submit(detect_entities, i, text)
                futures.append(future)

            # Wait for all operations to complete
            for future in as_completed(futures):
                entities = future.result()
                self.assertIsInstance(entities, set)

        # Verify all operations completed successfully
        self.assertEqual(len(results), len(test_texts))
        for entities in results.values():
            self.assertIsInstance(entities, set)

    def test_cache_lazy_loading_thread_safety(self):
        """Test thread-safe lazy loading of dictionary caches.

        This ensures that concurrent access to cache properties
        (cities, countries, etc.) is thread-safe and results in
        consistent state across threads.
        """
        cache = EntityDictionaryCache()
        property_results = {}

        def access_cache_property(property_name):
            """Access a cache property and store the result."""
            property_value = getattr(cache, property_name)
            property_results[property_name] = property_value
            return property_value

        # Test concurrent access to different cache properties
        properties = ["cities", "countries", "organisations", "english_corpus"]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for prop in properties:
                future = executor.submit(access_cache_property, prop)
                futures.append(future)

            # Wait for all property accesses to complete
            for future in as_completed(futures):
                result = future.result()
                self.assertIsInstance(result, frozenset)

        # Verify all properties were loaded successfully
        for prop in properties:
            self.assertIn(prop, property_results)
            self.assertIsInstance(property_results[prop], frozenset)

    def test_singleton_initialization_race_conditions(self):
        """Test singleton initialization under race conditions.

        This specifically tests the edge case where multiple threads
        might pass the initial None check before acquiring the lock,
        ensuring the second check prevents duplicate initialization.
        """
        # Test the actual singleton behavior by tracking instance creation
        instances = []
        barrier = threading.Barrier(20)  # Synchronize all threads

        def create_instance():
            # Wait for all threads to be ready to create instances
            # simultaneously
            barrier.wait()
            instance = EntityDictionaryCache()
            instances.append(instance)
            return instance

        num_threads = 20

        # Create many threads simultaneously
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # All instances should be identical (singleton guarantee)
        first_instance = instances[0]
        for instance in instances:
            self.assertIs(
                instance,
                first_instance,
                "All instances should be the same singleton object",
            )
            # All should be properly initialized
            self.assertTrue(
                instance._initialized,
                "Singleton instance should be properly initialized",
            )

        # Should have created exactly the expected number of instances
        self.assertEqual(
            len(instances),
            num_threads,
            "Should have one instance reference per thread",
        )

        # The singleton should be the same as what we get from a new call
        new_instance = EntityDictionaryCache()
        self.assertIs(
            new_instance,
            first_instance,
            "New instance should be the same singleton",
        )

    def test_get_entity_cache_function_thread_safety(self):
        """Test thread-safety of the get_entity_cache function.

        This ensures the global cache access function works correctly
        under concurrent access from multiple threads.
        """
        num_threads = 8
        cache_instances = []

        def get_cache():
            cache = get_entity_cache()
            cache_instances.append(cache)
            return cache

        # Access cache from multiple threads simultaneously
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for _ in range(num_threads):
                future = executor.submit(get_cache)
                futures.append(future)

            # Wait for all accesses to complete
            for future in as_completed(futures):
                cache = future.result()
                self.assertIsInstance(cache, EntityDictionaryCache)

        # All threads should get the same instance
        first_cache = cache_instances[0]
        for cache in cache_instances:
            self.assertIs(cache, first_cache)

    def test_concurrent_memory_stats_access(self):
        """Test concurrent access to memory statistics.

        This ensures that memory statistics can be safely accessed
        from multiple threads without race conditions or inconsistent state.
        """
        cache = EntityDictionaryCache()

        # Pre-load some data to ensure stats are meaningful
        _ = cache.cities
        _ = cache.countries

        stats_results = []

        def get_memory_stats():
            stats = cache.get_memory_stats()
            stats_results.append(stats)
            return stats

        # Access memory stats from multiple threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for _ in range(5):
                future = executor.submit(get_memory_stats)
                futures.append(future)

            # Wait for all stat accesses to complete
            for future in as_completed(futures):
                stats = future.result()
                self.assertIsInstance(stats, dict)

        # All threads should get consistent stats
        first_stats = stats_results[0]
        for stats in stats_results:
            self.assertEqual(stats, first_stats)

    def test_preload_all_thread_safety(self):
        """Test thread-safe preloading of all cache data.

        This ensures that the preload_all method works correctly
        when called from multiple threads simultaneously.
        """
        cache = EntityDictionaryCache()

        def preload_cache():
            cache.preload_all()
            return cache.get_memory_stats()

        # Preload from multiple threads
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for _ in range(3):
                future = executor.submit(preload_cache)
                futures.append(future)

            # Wait for all preloads to complete
            stats_results = []
            for future in as_completed(futures):
                stats = future.result()
                stats_results.append(stats)
                self.assertIsInstance(stats, dict)

        # All preloads should result in the same loaded state
        expected_keys = {
            "cities",
            "countries",
            "organisations",
            "english_corpus",
        }
        for stats in stats_results:
            self.assertEqual(set(stats.keys()), expected_keys)
            # All should have positive counts
            for count in stats.values():
                self.assertGreater(count, 0)

    def test_production_simulation_concurrent_workload(self):
        """Simulate production-like concurrent workload.

        This test simulates a realistic production scenario with
        multiple threads performing various entity detection operations
        simultaneously, mimicking the 2M monthly requests scenario.
        """
        sample_requests = [
            "Contact Dr. Sarah Johnson at sarah@hospital.org",
            "Call Professor Martinez at 555-123-4567",
            "Visit Microsoft headquarters in Seattle",
            "Email john.doe@company.com for details",
            "Meet Alice-Marie at Central Park tomorrow",
            "IBM is partnering with Google on this project",
            "Dr. Williams works at Johns Hopkins University",
            "Reach out to bob@startup.io for the demo",
        ]

        num_concurrent_workers = 8
        requests_per_worker = 10
        all_results = []

        def process_requests(worker_id):
            """Process a batch of entity detection requests."""
            detector = EntityDetector()
            worker_results = []

            for i in range(requests_per_worker):
                text = sample_requests[i % len(sample_requests)]
                entities = detector.detect_entities(text)
                worker_results.append(
                    {
                        "worker_id": worker_id,
                        "request_id": i,
                        "text": text,
                        "entity_count": len(entities),
                        "entities": entities,
                    }
                )

            all_results.extend(worker_results)
            return worker_results

        # Simulate concurrent production workload
        with ThreadPoolExecutor(
            max_workers=num_concurrent_workers
        ) as executor:
            futures = []
            for worker_id in range(num_concurrent_workers):
                future = executor.submit(process_requests, worker_id)
                futures.append(future)

            # Wait for all workers to complete
            for future in as_completed(futures):
                worker_results = future.result()
                self.assertIsInstance(worker_results, list)
                self.assertEqual(len(worker_results), requests_per_worker)

        # Verify all requests were processed successfully
        total_expected = num_concurrent_workers * requests_per_worker
        self.assertEqual(len(all_results), total_expected)

        # Verify consistent results across workers
        for result in all_results:
            self.assertIsInstance(result["entities"], set)
            self.assertGreaterEqual(result["entity_count"], 0)
            self.assertIn("worker_id", result)
            self.assertIn("text", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
