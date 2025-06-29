"""A simple LRU cache implementation using OrderedDict."""

# Standard Library Imports
from collections import OrderedDict
from typing import TypeVar

# Type Variables
K = TypeVar("K")
V = TypeVar("V")


class LRUCache[K, V]:
    """A simple LRU cache implementation using OrderedDict.

    Taken directly from: https://llego.dev/posts/implement-lru-cache-python/
    """

    def __init__(self, capacity: int) -> None:
        """Initialise the LRUCache with a specific capacity.

        Args:
            capacity: The maximum number of items cache can hold before
                eviction.

        """
        self.cache: OrderedDict[K, V] = OrderedDict()
        self.capacity = capacity

    def get(self, key: K) -> V | None:
        """Retrieve an item from the cache.

        Args:
            key (K): The key of the item to retrieve.

        Returns:
            V | None: The value associated with the key, or None if not found.

        """
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: K, value: V) -> None:
        """Add an item to the cache.

        Args:
            key (K): The key of the item to add.
            value (V): The value of the item to add.

        """
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
