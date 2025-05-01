import time
import functools
from typing import Callable, Type, Union, Tuple, Optional


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable:
    """
    A decorator that retries a function call when it raises specified exceptions.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier applied to delay after each retry
        exceptions: Exception(s) to catch and retry on
        on_retry: Optional callback function that gets called on each retry
                 with the exception and attempt number as arguments
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        raise

                    if on_retry:
                        on_retry(e, attempt)

                    time.sleep(current_delay)
                    current_delay *= backoff

            raise last_exception  # This should never be reached

        return wrapper

    return decorator
