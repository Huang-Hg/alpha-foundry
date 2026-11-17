"""Ordered, bounded host work over borrowed immutable arrays."""
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from typing import TypeVar

Input = TypeVar('Input')
Output = TypeVar('Output')


class HostWorkers:
    """At most one outstanding task per worker; results retain input order."""

    def __init__(self, count: int, initializer: Callable[[], None] | None = None):
        self.count = count
        self.pool = ThreadPoolExecutor(count, initializer=initializer, thread_name_prefix='foundry-cpu')

    def map(self, function: Callable[[Input], Output], items: Sequence[Input]) -> Iterator[Output]:
        for start in range(0, len(items), self.count):
            pending = [self.pool.submit(function, item) for item in items[start:start + self.count]]
            try:
                for future in pending:
                    yield future.result()
            finally:
                wait(pending)

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
