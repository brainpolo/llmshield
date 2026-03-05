"""Test thread safety and concurrent access patterns.

Description:
    This test module validates thread-safety guarantees of the singleton
    cache, concurrent access patterns, and race condition handling.

Test Classes:
    - TestThreadingAndConcurrency: Tests thread-safe operations

Author:
    LLMShield by brainpolo, 2025-2026
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

    def _assert_all_same_singleton(self, instances):
        """Assert all instances are the same singleton object."""
        first = instances[0]
        for inst in instances:
            self.assertIs(inst, first)
        self.assertTrue(first._initialized)

    def test_singleton_thread_safety(self):
        """Test singleton creation under high thread contention."""
        num_threads = 20
        instances = []
        barrier = threading.Barrier(num_threads)

        def create_instance():
            barrier.wait()
            instances.append(EntityDictionaryCache())

        threads = [
            threading.Thread(target=create_instance)
            for _ in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self._assert_all_same_singleton(instances)
        self.assertEqual(len(instances), num_threads)

        # New call still returns same singleton
        self.assertIs(EntityDictionaryCache(), instances[0])

    def test_concurrent_entity_detection(self):
        """Test concurrent entity detection produces consistent results."""
        texts = [
            "Dr. John Smith works at Microsoft in London",
            "Contact Alice Johnson at alice@company.com",
            "Visit Professor Brown at Harvard University",
            "Call Sarah Wilson at 555-123-4567",
            "Meet Bob Garcia at IBM headquarters",
        ]

        results = {}

        def detect(thread_id, text):
            entities = EntityDetector().detect_entities(text)
            results[thread_id] = entities

        with ThreadPoolExecutor(max_workers=len(texts)) as pool:
            futures = [pool.submit(detect, i, t) for i, t in enumerate(texts)]
            for f in as_completed(futures):
                f.result()

        self.assertEqual(len(results), len(texts))
        for entities in results.values():
            self.assertIsInstance(entities, set)

    def test_cache_lazy_loading_thread_safety(self):
        """Test concurrent access to cache properties."""
        cache = EntityDictionaryCache()
        properties = [
            "cities",
            "countries",
            "organisations",
            "english_corpus",
        ]
        loaded = {}

        def load_prop(name):
            loaded[name] = getattr(cache, name)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(load_prop, p) for p in properties]
            for f in as_completed(futures):
                f.result()

        for prop in properties:
            self.assertIsInstance(loaded[prop], frozenset)

    def test_get_entity_cache_thread_safety(self):
        """Test get_entity_cache returns same singleton from threads."""
        instances = []

        def get_cache():
            instances.append(get_entity_cache())

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(get_cache) for _ in range(8)]
            for f in as_completed(futures):
                f.result()

        self._assert_all_same_singleton(instances)

    def test_concurrent_memory_stats(self):
        """Test concurrent access to memory statistics."""
        cache = EntityDictionaryCache()
        _ = cache.cities
        _ = cache.countries

        stats_results = []

        def get_stats():
            stats_results.append(cache.get_memory_stats())

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(get_stats) for _ in range(5)]
            for f in as_completed(futures):
                f.result()

        # All threads get consistent stats
        for stats in stats_results:
            self.assertEqual(stats, stats_results[0])

    def test_preload_all_thread_safety(self):
        """Test concurrent preload_all calls."""
        cache = EntityDictionaryCache()
        expected_keys = {
            "cities",
            "countries",
            "organisations",
            "english_corpus",
        }

        def preload():
            cache.preload_all()
            return cache.get_memory_stats()

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(preload) for _ in range(3)]
            for f in as_completed(futures):
                stats = f.result()
                self.assertEqual(set(stats.keys()), expected_keys)
                for count in stats.values():
                    self.assertGreater(count, 0)

    def test_production_concurrent_workload(self):
        """Simulate production-like concurrent entity detection."""
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

        workers = 8
        requests_per_worker = 10
        all_results = []

        def process(worker_id):
            detector = EntityDetector()
            batch = []
            for i in range(requests_per_worker):
                text = sample_requests[i % len(sample_requests)]
                entities = detector.detect_entities(text)
                batch.append((worker_id, i, len(entities), entities))
            all_results.extend(batch)
            return batch

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process, w) for w in range(workers)]
            for f in as_completed(futures):
                batch = f.result()
                self.assertEqual(len(batch), requests_per_worker)

        self.assertEqual(len(all_results), workers * requests_per_worker)
        for _, _, count, entities in all_results:
            self.assertIsInstance(entities, set)
            self.assertGreaterEqual(count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
