"""Layer 03: capped ``multiprocessing`` worker pool.

``WorkerPool.map`` runs a top-level function over a list of tasks. Small jobs
(or a single worker) run inline, so tests and tiny scans do not pay process
start-up cost; results are identical either way and keep input order.

The pool size always goes through :func:`simulator.config.resolve_worker_count`
so it stays inside the 50-60% CPU band.
"""

from __future__ import annotations

import multiprocessing
from typing import Any, Callable, Iterable, List, Optional, Sequence, TypeVar

from .config import resolve_worker_count

T = TypeVar("T")
R = TypeVar("R")


def chunked(items: Sequence[T], parts: int) -> List[List[T]]:
    """Split ``items`` into at most ``parts`` contiguous, near-equal chunks."""
    items = list(items)
    if not items:
        return []
    parts = max(1, min(parts, len(items)))
    size, extra = divmod(len(items), parts)
    out: List[List[T]] = []
    start = 0
    for i in range(parts):
        end = start + size + (1 if i < extra else 0)
        out.append(items[start:end])
        start = end
    return out


class WorkerPool:
    """Lazily started process pool; use as a context manager to reuse it."""

    def __init__(self, workers: Optional[int] = None, start_method: str = "spawn",
                 parallel_threshold: int = 4) -> None:
        self.workers = resolve_worker_count(workers)
        self.start_method = start_method
        self.parallel_threshold = parallel_threshold
        self._pool: Any = None

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _ensure(self) -> Any:
        if self._pool is None:
            ctx = multiprocessing.get_context(self.start_method)
            self._pool = ctx.Pool(processes=self.workers)
        return self._pool

    def will_parallelize(self, n_tasks: int) -> bool:
        return self.workers > 1 and n_tasks >= self.parallel_threshold

    def map(self, fn: Callable[[T], R], tasks: Iterable[T]) -> List[R]:
        tasks = list(tasks)
        if not self.will_parallelize(len(tasks)):
            return [fn(t) for t in tasks]
        return self._ensure().map(fn, tasks, chunksize=1)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


__all__ = ["WorkerPool", "chunked"]
