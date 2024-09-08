import time
import functools


def time_it(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(self, *args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        self._display_solve_time(duration)
        return result
    return wrapper
