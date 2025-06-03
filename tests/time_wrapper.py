"""Time wrapper function that allows for a simple measure of time using the context manager."""

import time
from collections.abc import Iterator
from contextlib import contextmanager

@contextmanager
def time_it() -> Iterator[None]:
    """
    This module provides a context manager (for self cleanup) to measure the time 
    taken for a block of code to execute.
    It can be used to wrap any code block where you want to measure the execution time.
    It prints the time taken in milliseconds.
    
    Example usage:
        >>> with time_it():
        ...     # Your code block here
        ...     time.sleep(1)
        # Prints the time taken in milliseconds.
    Args:
        None
    Yields:
        None
    Returns:
        None
    Raises:
        None
    """
    start: float = time.perf_counter()
    try:
        yield
    finally:
        end: float = time.perf_counter()
        print(f"Computation time = {1000*(end - start):.3f}ms")
