"""
Miscellaneous helper functions for system operations.
"""

import time
from functools import wraps


def timer(func):
    """
    Decorator to measure the execution time of a function.

    Parameters
    ----------
    func : callable
        The function to be wrapped.

    Returns
    -------
    wrapper : callable
        A wrapper function that measures the execution time of the input function.

    Examples
    --------
    @timer
    def my_func():
        # Code to be timed
        pass
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Record the start time
        start_time = time.time()
        # Execute the input function
        result = func(*args, **kwargs)
        # Calculate the duration
        duration = time.time() - start_time
        # Print the duration
        print(f"[TIMER] {func.__name__} took {duration:.2f} seconds.")
        # Return the result
        return result

    return wrapper
