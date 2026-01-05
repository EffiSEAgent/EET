# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import random
import time
import traceback
from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar("T")

# Import OpenAI BadRequestError if available
try:
    import openai

    BadRequestError = openai.BadRequestError
except (ImportError, AttributeError):
    # Fallback if openai is not available or BadRequestError doesn't exist
    BadRequestError = type(None)


def retry_with(
    func: Callable[..., T],
    provider_name: str = "OpenAI",
    max_retries: int = 3,
) -> Callable[..., T]:
    """
    Decorator that adds retry logic with randomized backoff.
    
    BadRequestError (400 errors) are not retried as they indicate client-side
    errors that won't be fixed by retrying.

    Args:
        func: The function to decorate
        provider_name: The name of the model provider being called
        max_retries: Maximum number of retry attempts

    Returns:
        Decorated function with retry logic
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                # Don't retry BadRequestError (400 errors) - these are client-side errors
                # that indicate malformed requests and won't be fixed by retrying
                if BadRequestError and isinstance(e, BadRequestError):
                    raise

                if attempt == max_retries:
                    # Last attempt, re-raise the exception
                    raise

                sleep_time = random.randint(3, 30)
                this_error_message = str(e)
                print(
                    f"{provider_name} API call failed: {this_error_message}. Will sleep for {sleep_time} seconds and will retry.\n{traceback.format_exc()}"
                )
                # Randomly sleep for 3-30 seconds
                time.sleep(sleep_time)

        # This should never be reached, but just in case
        raise last_exception or Exception("Retry failed for unknown reason")

    return wrapper
