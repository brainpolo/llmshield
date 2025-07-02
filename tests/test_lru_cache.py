"""Test LRU cache implementation and eviction policies.

Description:
    This test module provides comprehensive testing for the Least Recently
    Used (LRU) cache implementation, including capacity management,
    eviction policies, and cache hit/miss behaviour.

Test Classes:
    - TestLRUCache: Tests LRU cache operations and eviction

Author: LLMShield by brainpolo, 2025
"""

from unittest import TestCase, main

from llmshield.lru_cache import LRUCache


class TestLRUCache(TestCase):
    """Tests for the LRUCache implementation."""

    def test_init(self):
        """Test cache initialization."""
        cache: LRUCache[str, int] = LRUCache(capacity=3)
        self.assertEqual(cache.capacity, 3)
        self.assertEqual(len(cache.cache), 0)

    def test_put_and_get(self):
        """Test basic put and get operations."""
        cache: LRUCache[str, int] = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        self.assertEqual(cache.get("a"), 1)
        self.assertEqual(cache.get("b"), 2)
        self.assertEqual(len(cache.cache), 2)

    def test_get_nonexistent_key(self):
        """Test getting a non-existent key returns None."""
        cache: LRUCache[str, int] = LRUCache(capacity=2)
        cache.put("a", 1)
        self.assertIsNone(cache.get("b"))

    def test_lru_eviction(self):
        """Test that the least recently used item is evicted."""
        cache: LRUCache[str, int] = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # This should evict "a"
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), 2)
        self.assertEqual(cache.get("c"), 3)
        self.assertEqual(len(cache.cache), 2)

    def test_get_updates_recency(self):
        """Test that getting an item makes it the most recently used."""
        cache: LRUCache[str, int] = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # "a" is now the most recently used
        cache.put("c", 3)  # This should evict "b"
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), 1)
        self.assertEqual(cache.get("c"), 3)

    def test_put_updates_existing_key(self):
        """Test that putting an existing key updates the value and recency."""
        cache: LRUCache[str, int] = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("a", 100)  # Update "a"
        cache.put("c", 3)  # This should evict "b"
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), 100)
        self.assertEqual(cache.get("c"), 3)

    def test_capacity_one(self):
        """Test edge case with capacity of 1."""
        cache: LRUCache[str, int] = LRUCache(capacity=1)
        cache.put("a", 1)
        self.assertEqual(cache.get("a"), 1)
        cache.put("b", 2)
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), 2)


if __name__ == "__main__":
    main()
