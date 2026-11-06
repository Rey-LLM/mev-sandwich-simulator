"""Layers 03 / 09: governed ``multiprocessing`` worker pool and resource telemetry.

``WorkerPool.map`` runs a top-level function over a list of tasks. Small jobs
(or a single worker) run inline, so tests and tiny scans do not pay process
start-up cost; results are identical either way and keep input order.

The pool size always goes through :func:`simulator.config.resolve_worker_count`
under a :class:`~simulator.config.CpuPolicy`: the capped policy keeps it
inside the 50-60% CPU band (Moderate Scan, Brutal Aggressive Simulation), the
full policy lets it scale to every usable logical CPU (MAX Brutal
Simulation). This module is the only place in the simulator that starts a
process pool; the Layer 10 validator flags any other pool construction.

Dynamic scaling
---------------
Processes start lazily, on the first ``map`` that parallelises, and only as
many as that map has tasks (never more than the policy ceiling). A later map
with more tasks grows the pool up to the ceiling. Worker processes are
stateless, so growing (which restarts the pool) never changes results.

Several threads may share one pool. Maps check the pool out and back in;
growing waits until no map is running on the old pool (and blocks new
check-outs meanwhile, so it cannot starve), and :meth:`WorkerPool.close`
waits for running maps to finish.

Telemetry
---------
With ``instrument=True`` every task is wrapped so the worker reports its own
CPU time (``time.process_time``), wall time and PID. :meth:`WorkerPool.telemetry`
turns that into per-process task counts, total CPU seconds and a utilisation
figure (CPU seconds over ``busy wall time * active processes``), which is how
the controller shows that a MAX run really keeps every core busy.
:func:`saturation_probe` and :func:`rendezvous_probe` load every worker at
once with a CPU-bound kernel, or hold them all at a barrier, to verify full
concurrency on a given machine.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

from .config import CpuPolicy, detect_cpu_count, resolve_worker_count

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


def _instrumented_call(packed: Tuple[Callable[[Any], Any], Any]) -> Tuple[Any, float, float, int]:
    """Worker-side wrapper: run ``fn(task)`` and report CPU seconds, wall seconds and PID."""
    fn, task = packed
    cpu0 = time.process_time()
    wall0 = time.perf_counter()
    result = fn(task)
    return result, time.process_time() - cpu0, time.perf_counter() - wall0, os.getpid()


@dataclass
class PoolStats:
    """Counters a :class:`WorkerPool` accumulates over its lifetime."""

    maps: int = 0
    parallel_maps: int = 0
    tasks: int = 0
    parallel_tasks: int = 0
    inline_tasks: int = 0
    pool_starts: int = 0
    peak_processes: int = 0
    busy_wall_s: float = 0.0  # wall time spent inside parallel maps (parent clock)
    cpu_s: float = 0.0  # task CPU seconds reported by the workers (instrumented maps only)
    inline_cpu_s: float = 0.0
    task_wall_s: float = 0.0
    tasks_by_pid: Dict[int, int] = field(default_factory=dict)
    cpu_by_pid: Dict[int, float] = field(default_factory=dict)


class WorkerPool:
    """Lazily started, policy-governed process pool; use as a context manager to reuse it."""

    def __init__(self, workers: Optional[int] = None, start_method: str = "spawn",
                 parallel_threshold: int = 4, policy: Any = CpuPolicy.CAPPED,
                 cpus: Optional[int] = None, instrument: bool = False) -> None:
        self.policy = CpuPolicy.parse(policy)
        self.cpus = detect_cpu_count() if cpus is None else int(cpus)
        self.workers = resolve_worker_count(workers, self.cpus, self.policy)
        self.start_method = start_method
        self.parallel_threshold = parallel_threshold
        self.instrument = instrument
        self.stats = PoolStats()
        self._pool: Any = None
        self._size = 0
        self._lock = threading.RLock()
        self._idle = threading.Condition(self._lock)
        self._inflight = 0  # maps currently running on self._pool
        self._growing = False

    @classmethod
    def for_config(cls, config: Any, instrument: bool = False, cpus: Optional[int] = None) -> "WorkerPool":
        """Pool governed by a SimulationConfig: its workers, start method, threshold and CPU policy."""
        return cls(config.workers, config.start_method, config.parallel_threshold,
                   policy=getattr(config, "cpu_policy", CpuPolicy.CAPPED), cpus=cpus, instrument=instrument)

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def active_workers(self) -> int:
        """Processes in the live pool (0 before the first parallel map and after close)."""
        return self._size if self._pool is not None else 0

    def _restart(self, want: int) -> None:
        """(Re)start the pool with ``want`` processes; the caller holds the lock and no map is running."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        ctx = multiprocessing.get_context(self.start_method)
        self._pool = ctx.Pool(processes=want)
        self._size = want
        self.stats.pool_starts += 1
        self.stats.peak_processes = max(self.stats.peak_processes, want)

    def _checkout(self, n_tasks: Optional[int] = None) -> Any:
        """The live pool, started or grown to ``min(workers, n_tasks)`` processes; pair with :meth:`_checkin`."""
        want = self.workers if n_tasks is None else max(1, min(self.workers, int(n_tasks)))
        with self._lock:
            while self._growing:
                self._idle.wait()
            if self._pool is None or self._size < want:
                self._growing = True
                try:
                    while self._inflight:
                        self._idle.wait()
                    if self._pool is None or self._size < want:
                        self._restart(want)
                finally:
                    self._growing = False
                    self._idle.notify_all()
            self._inflight += 1
            return self._pool

    def _checkin(self) -> None:
        with self._lock:
            self._inflight -= 1
            if not self._inflight:
                self._idle.notify_all()

    def _ensure(self, n_tasks: Optional[int] = None) -> Any:
        """Start (or grow) the pool to ``min(workers, n_tasks)`` processes and return it."""
        pool = self._checkout(n_tasks)
        self._checkin()
        return pool

    def will_parallelize(self, n_tasks: int) -> bool:
        return self.workers > 1 and n_tasks >= self.parallel_threshold

    def map(self, fn: Callable[[T], R], tasks: Iterable[T], chunksize: int = 1) -> List[R]:
        """``[fn(t) for t in tasks]``, on worker processes when the batch is large enough."""
        tasks = list(tasks)
        if not self.will_parallelize(len(tasks)):
            with self._lock:
                self.stats.maps += 1
                self.stats.tasks += len(tasks)
            cpu0 = time.process_time()
            out = [fn(t) for t in tasks]
            with self._lock:
                self.stats.inline_tasks += len(tasks)
                self.stats.inline_cpu_s += time.process_time() - cpu0
            return out
        return self.map_parallel(fn, tasks, chunksize)

    def map_parallel(self, fn: Callable[[T], R], tasks: Iterable[T], chunksize: int = 1) -> List[R]:
        """Like :meth:`map` but always on worker processes (unless the pool has a single worker)."""
        tasks = list(tasks)
        if self.workers < 2 or not tasks:
            with self._lock:
                self.stats.maps += 1
                self.stats.tasks += len(tasks)
                self.stats.inline_tasks += len(tasks)
            return [fn(t) for t in tasks]
        with self._lock:
            self.stats.maps += 1
            self.stats.tasks += len(tasks)
        pool = self._checkout(len(tasks))
        try:
            started = time.perf_counter()
            if self.instrument:
                raw: Sequence[Tuple[Any, float, float, int]] = pool.map(
                    _instrumented_call, [(fn, t) for t in tasks], chunksize=max(1, chunksize))
                out = [r[0] for r in raw]
            else:
                raw = ()
                out = pool.map(fn, tasks, chunksize=max(1, chunksize))
            wall = time.perf_counter() - started
        finally:
            self._checkin()
        self._account_parallel(len(tasks), wall, raw)
        return out

    def _account_parallel(self, n: int, wall: float, raw: Sequence[Tuple[Any, float, float, int]]) -> None:
        with self._lock:
            s = self.stats
            s.parallel_maps += 1
            s.parallel_tasks += n
            s.busy_wall_s += wall
            for _, cpu, task_wall, pid in raw:
                s.cpu_s += cpu
                s.task_wall_s += task_wall
                s.tasks_by_pid[pid] = s.tasks_by_pid.get(pid, 0) + 1
                s.cpu_by_pid[pid] = s.cpu_by_pid.get(pid, 0.0) + cpu

    def telemetry(self) -> Dict[str, Any]:
        """Resource usage so far: processes, tasks, CPU seconds and utilisation of the busy cores."""
        with self._lock:
            s = self.stats
            procs = max(1, s.peak_processes)
            capacity = s.busy_wall_s * procs
            utilization = min(1.0, s.cpu_s / capacity) if self.instrument and capacity > 0 else None
            return {
                "cpu_policy": self.policy.value,
                "usable_cpus": self.cpus,
                "workers": self.workers,
                "worker_share_of_cpus": round(self.workers / self.cpus, 4) if self.cpus else None,
                "peak_processes": s.peak_processes,
                "pool_starts": s.pool_starts,
                "maps": s.maps,
                "parallel_maps": s.parallel_maps,
                "tasks": s.tasks,
                "parallel_tasks": s.parallel_tasks,
                "inline_tasks": s.inline_tasks,
                "instrumented": self.instrument,
                "busy_wall_s": round(s.busy_wall_s, 4),
                "worker_cpu_s": round(s.cpu_s, 4),
                "inline_cpu_s": round(s.inline_cpu_s, 4),
                "distinct_worker_pids": len(s.tasks_by_pid),
                "tasks_per_worker": sorted(s.tasks_by_pid.values(), reverse=True),
                "utilization": None if utilization is None else round(utilization, 4),
            }

    def close(self) -> None:
        """Stop the worker processes (after any map still running on another thread finishes)."""
        with self._lock:
            while self._inflight or self._growing:
                self._idle.wait()
            if self._pool is not None:
                self._pool.close()
                self._pool.join()
                self._pool = None
                self._size = 0


# ---------------------------------------------------------------------------
# Load probes: verify that every worker can run at once
# ---------------------------------------------------------------------------


def cpu_burn(task: Tuple[float, int]) -> Dict[str, Any]:
    """CPU-bound kernel: integer mixing for ``seconds`` of wall time (at least ``min_rounds`` rounds)."""
    seconds, min_rounds = task
    cpu0 = time.process_time()
    wall0 = time.perf_counter()
    deadline = wall0 + max(0.0, float(seconds))
    x = 0x9E3779B97F4A7C15
    rounds = 0
    while rounds < min_rounds or time.perf_counter() < deadline:
        for _ in range(2_000):
            x = (x * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        rounds += 1
    return {"pid": os.getpid(), "cpu_s": time.process_time() - cpu0, "wall_s": time.perf_counter() - wall0,
            "rounds": rounds, "checksum": x & 0xFFFF}


def saturation_probe(pool: WorkerPool, seconds: float = 0.25, tasks_per_worker: int = 1,
                     min_rounds: int = 1) -> Dict[str, Any]:
    """Run ``workers * tasks_per_worker`` CPU-bound tasks on ``pool`` and report how busy the cores were.

    ``utilization`` is total task CPU time over ``wall * workers``; on an idle
    machine with every worker busy it approaches 1.0. A single-threaded
    process cannot use more than one core, so each task's ``cpu_s`` is at
    most its ``wall_s`` (plus clock granularity).
    """
    n = max(1, pool.workers * max(1, tasks_per_worker))
    tasks = [(seconds, min_rounds)] * n
    started = time.perf_counter()
    rows = pool.map_parallel(cpu_burn, tasks)
    wall = time.perf_counter() - started
    cpu = sum(r["cpu_s"] for r in rows)
    pids = sorted({r["pid"] for r in rows})
    return {
        "workers": pool.workers,
        "cpu_policy": pool.policy.value,
        "usable_cpus": pool.cpus,
        "tasks": n,
        "distinct_pids": len(pids),
        "wall_s": round(wall, 4),
        "cpu_s": round(cpu, 4),
        "utilization": round(min(1.0, cpu / (wall * pool.workers)), 4) if wall > 0 else 0.0,
        "max_task_cpu_over_wall": round(max(r["cpu_s"] / r["wall_s"] for r in rows if r["wall_s"] > 0), 4)
        if any(r["wall_s"] > 0 for r in rows) else 0.0,
        "rows": rows,
    }


def _rendezvous_task(task: Tuple[Any, float]) -> Tuple[int, bool]:
    barrier, timeout = task
    try:
        barrier.wait(timeout)
        return os.getpid(), True
    except Exception:  # BrokenBarrierError travels as a proxy-side exception
        return os.getpid(), False


def rendezvous_probe(pool: WorkerPool, timeout: float = 30.0) -> Dict[str, Any]:
    """Hold ``pool.workers`` tasks at one barrier: it only opens if every worker runs at the same time.

    Each task blocks its process until all ``workers`` parties arrive, so a
    worker can never pick up a second task while the barrier is closed. The
    barrier therefore opens only when ``workers`` distinct processes are
    executing concurrently, which proves the pool really scaled to that size.
    """
    n = pool.workers
    if n < 2:
        return {"workers": n, "released": True, "distinct_pids": 1, "pids": [os.getpid()]}
    ctx = multiprocessing.get_context(pool.start_method)
    with ctx.Manager() as manager:
        barrier = manager.Barrier(n)
        rows = pool.map_parallel(_rendezvous_task, [(barrier, timeout)] * n)
    pids = sorted({pid for pid, _ in rows})
    return {"workers": n, "released": all(ok for _, ok in rows), "distinct_pids": len(pids), "pids": pids}


__all__ = [
    "PoolStats",
    "WorkerPool",
    "chunked",
    "cpu_burn",
    "rendezvous_probe",
    "saturation_probe",
]
