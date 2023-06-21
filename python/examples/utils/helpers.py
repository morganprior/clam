"""Helper functions for the examples."""

import functools
import logging
import time
import typing


class TimeIt:
    """A class to provide a decorator for timing the execution of a function."""

    def __init__(
        self,
        logger: logging.Logger,
        template: str = "completed {:s} in {:.3f} seconds",
    ) -> None:
        """Initialize the timer."""
        self.template: str = template
        self.logger = logger

    def __call__(self, function: typing.Callable):  # noqa: ANN204
        """Time the execution of a function."""

        @functools.wraps(function)
        def wrapper(*args, **kwargs):  # noqa: ANN002,ANN003,ANN202
            start = time.perf_counter()
            result = function(*args, **kwargs)
            end = time.perf_counter()

            self.logger.info(self.template.format(function.__name__, end - start))
            return result

        return wrapper
