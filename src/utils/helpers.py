"""
Miscellaneous helper functions for system operations.
"""
import time
from functools import wraps

def timer(func):
    """Decorator to measure the execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        print(f"[TIMER] {func.__name__} took {duration:.2f} seconds.")
        return result
    return wrapper
