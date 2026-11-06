"""Layer 03: AMM invariant verification (Halmos-style properties).

Halmos proves Solidity properties symbolically with an SMT solver. This
module checks the same kind of properties against the simulator's integer
AMM math, in two ways and without a solver:

* ``bounded_exhaustive_v2`` enumerates *every* input in a small bounded domain
  (reserves, amounts, fee) and checks the v2 properties for all of them. It
  is a complete check inside the bounds, like a bounded symbolic run, but it
  proves nothing outside them.
* ``fuzz_pool`` / ``parallel_fuzz`` run deterministic random and boundary
  inputs (log-uniform sizes, 1 wei, near-reserve sizes) through stateful swap
  sequences on large realistic pools.

Properties checked after every swap:

v2  - ``k`` never decreases (x*y=k with fee only grows), reserves stay
      positive, output is below the output reserve and never above the
      fee-free constant-product output, and a round trip (A->B->A) never
      returns more than was put in.
v3/v4 - liquidity stays non-negative, sqrt price stays inside
      [MIN_SQRT_RATIO, MAX_SQRT_RATIO] and moves in the swap direction, the
      quote matches the applied swap, input used never exceeds the amount
      offered, and a round trip never returns more than was put in.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .slippage_engine import (
    MAX_SQRT_RATIO,
    MIN_SQRT_RATIO,
    Q96,
    ConcentratedPool,
    MASEError,
    Pool,
    V2Pool,
    v2_get_amount_out,
)


@dataclass(frozen=True)
class InvariantViolation:
    name: str
    pool_id: str
    detail: str
    case: Tuple[Any, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "pool_id": self.pool_id, "detail": self.detail, "case": list(self.case)}


@dataclass
class InvariantReport:
    checks: int = 0
    cases: int = 0
    skipped: int = 0
    violations: List[InvariantViolation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def merge(self, other: "InvariantReport") -> "InvariantReport":
        self.checks += other.checks
        self.cases += other.cases
        self.skipped += other.skipped
        self.violations.extend(other.violations)
        return self

    def to_dict(self, max_violations: int = 20) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": self.checks,
            "cases": self.cases,
            "skipped": self.skipped,
            "violations": [v.to_dict() for v in self.violations[:max_violations]],
            "violation_count": len(self.violations),
        }


class _Checker:
    def __init__(self, report: InvariantReport, pool_id: str, case: Tuple[Any, ...]) -> None:
        self.report, self.pool_id, self.case = report, pool_id, case

    def __call__(self, name: str, condition: bool, detail: str = "") -> None:
        self.report.checks += 1
        if not condition:
            self.report.violations.append(InvariantViolation(name, self.pool_id, detail, self.case))


# ---------------------------------------------------------------------------
# Single-swap property checks (mutate ``pool`` in place on success)
# ---------------------------------------------------------------------------


def check_v2_swap(pool: V2Pool, amount_in: int, zero_for_one: bool, report: InvariantReport) -> bool:
    """Apply one swap to ``pool`` and check the v2 properties. Returns False if skipped."""
    r_in, r_out = (pool.reserve0, pool.reserve1) if zero_for_one else (pool.reserve1, pool.reserve0)
    try:
        res = pool.quote_exact_in(amount_in, zero_for_one)
    except MASEError:
        report.skipped += 1
        return False
    report.cases += 1
    check = _Checker(report, pool.pool_id, ("v2", r_in, r_out, amount_in, zero_for_one, pool.fee_bps))
    ideal = r_out * amount_in // (r_in + amount_in)
    k_before = pool.k
    check("v2.out_below_reserve", res.amount_out < r_out, f"out={res.amount_out} reserve={r_out}")
    check("v2.out_le_fee_free", res.amount_out <= ideal, f"out={res.amount_out} ideal={ideal}")
    if res.amount_out == 0:
        report.skipped += 1
        return False
    pool.apply_swap(res, zero_for_one)
    check("v2.k_non_decreasing", pool.k >= k_before, f"k {k_before} -> {pool.k}")
    check("v2.reserves_positive", pool.reserve0 > 0 and pool.reserve1 > 0, "")
    # Round trip on a scratch copy: selling the output back cannot return more than amount_in.
    back = copy.deepcopy(pool)
    try:
        rt = back.quote_exact_in(res.amount_out, not zero_for_one)
        check("v2.no_free_round_trip", rt.amount_out <= amount_in, f"in={amount_in} back={rt.amount_out}")
    except MASEError:
        pass
    return True


def check_concentrated_swap(pool: ConcentratedPool, amount_in: int, zero_for_one: bool,
                            report: InvariantReport) -> bool:
    sqrt_before = pool.sqrt_price_x96
    try:
        quote = pool.quote_exact_in(amount_in, zero_for_one)
    except MASEError:
        report.skipped += 1
        return False
    report.cases += 1
    check = _Checker(report, pool.pool_id, (pool.version, sqrt_before, pool.liquidity, amount_in, zero_for_one))
    check("cl.input_le_offered", quote.amount_in <= amount_in, f"used={quote.amount_in} offered={amount_in}")
    applied = pool.apply_swap(amount_in, zero_for_one)
    check("cl.quote_matches_apply", applied == quote, "quote and apply diverged")
    check("cl.liquidity_non_negative", pool.liquidity >= 0, f"L={pool.liquidity}")
    check("cl.sqrt_price_in_range", MIN_SQRT_RATIO <= pool.sqrt_price_x96 <= MAX_SQRT_RATIO,
          f"sqrtP={pool.sqrt_price_x96}")
    moved_right_way = pool.sqrt_price_x96 <= sqrt_before if zero_for_one else pool.sqrt_price_x96 >= sqrt_before
    check("cl.price_moves_with_swap", moved_right_way, f"{sqrt_before} -> {pool.sqrt_price_x96}")
    if quote.amount_out > 0:
        back = copy.deepcopy(pool)
        try:
            rt = back.quote_exact_in(quote.amount_out, not zero_for_one)
            check("cl.no_free_round_trip", rt.amount_out <= quote.amount_in,
                  f"in={quote.amount_in} back={rt.amount_out}")
        except MASEError:
            pass
    return True


def check_swap(pool: Pool, amount_in: int, zero_for_one: bool, report: InvariantReport) -> bool:
    if isinstance(pool, V2Pool):
        return check_v2_swap(pool, amount_in, zero_for_one, report)
    return check_concentrated_swap(pool, amount_in, zero_for_one, report)


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------


def input_reserve_scale(pool: Pool, zero_for_one: bool) -> int:
    """Rough size of the input-side reserve, used to scale fuzz amounts."""
    if isinstance(pool, V2Pool):
        return pool.reserve0 if zero_for_one else pool.reserve1
    # Virtual reserves at the current price: x = L / sqrtP, y = L * sqrtP.
    L = max(pool.liquidity, 1)
    return max(1, L * Q96 // pool.sqrt_price_x96 if zero_for_one else L * pool.sqrt_price_x96 // Q96)


def fuzz_amount(rng: random.Random, scale: int) -> int:
    """Log-uniform from 1 wei to ~3x the input reserve, with boundary values mixed in."""
    roll = rng.random()
    if roll < 0.05:
        return 1
    if roll < 0.10:
        return max(1, scale - 1)
    if roll < 0.15:
        return scale * 3
    top = math.log10(max(scale, 10) * 3)
    return max(1, int(10 ** rng.uniform(0, top)))


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def fuzz_pool(pool: Pool, iterations: int, seed: int = 0, reset_every: int = 16) -> InvariantReport:
    """Stateful fuzz: random swap sequences on a copy of ``pool``, reset every ``reset_every`` swaps."""
    rng = random.Random(seed)
    report = InvariantReport()
    work = copy.deepcopy(pool)
    for i in range(iterations):
        if reset_every and i % reset_every == 0:
            work = copy.deepcopy(pool)
        zero_for_one = rng.random() < 0.5
        amount = fuzz_amount(rng, input_reserve_scale(work, zero_for_one))
        check_swap(work, amount, zero_for_one, report)
    return report


def _fuzz_task(task: Tuple[Pool, int, int]) -> InvariantReport:
    pool, iterations, seed = task
    return fuzz_pool(pool, iterations, seed)


def fuzz_tasks(pools: Sequence[Pool], iterations: int, seed: int, shards: int) -> List[Tuple[Pool, int, int]]:
    """Split ``iterations`` per pool into ``shards`` tasks with derived seeds (deterministic)."""
    tasks: List[Tuple[Pool, int, int]] = []
    shards = max(1, shards)
    for p_idx, pool in enumerate(pools):
        base, extra = divmod(iterations, shards)
        for s in range(shards):
            n = base + (1 if s < extra else 0)
            if n:
                tasks.append((pool, n, seed * 1_000_003 + p_idx * 1_009 + s))
    return tasks


def parallel_fuzz(pools: Iterable[Pool], iterations: int, seed: int = 0, workers: Optional[int] = None,
                  shards: int = 8, start_method: str = "spawn", parallel_threshold: int = 4,
                  executor: Optional[Any] = None) -> InvariantReport:
    """Fuzz every pool with ``iterations`` swaps, sharded across a capped worker pool.

    Pass ``executor`` (a :class:`simulator.parallel.WorkerPool`) to reuse an
    existing pool instead of starting one. The shard layout depends only on
    ``shards``, never on the worker count, so results are identical however
    many workers run them.
    """
    tasks = fuzz_tasks(list(pools), iterations, seed, shards)
    if executor is not None:
        parts = executor.map(_fuzz_task, tasks)
    elif workers is None:
        parts = [_fuzz_task(t) for t in tasks]
    else:
        from .parallel import WorkerPool

        with WorkerPool(workers, start_method, parallel_threshold) as wp:
            parts = wp.map(_fuzz_task, tasks)
    report = InvariantReport()
    for part in parts:
        report.merge(part)
    return report


def bounded_exhaustive_v2(max_reserve: int = 24, max_amount: int = 24,
                          fees_bps: Sequence[int] = (0, 30, 100)) -> InvariantReport:
    """Check v2 properties for every (reserve0, reserve1, amount, fee) in the bounds.

    Also checks that ``v2_get_amount_out`` is monotonic in ``amount_in`` for
    each reserve pair, which the per-swap checks do not cover.
    """
    report = InvariantReport()
    for fee in fees_bps:
        for r0 in range(1, max_reserve + 1):
            for r1 in range(1, max_reserve + 1):
                prev = -1
                for amount in range(1, max_amount + 1):
                    pool = V2Pool("X", "Y", r0, r1, fee_bps=fee, pool_id=f"bounded:{fee}")
                    check_v2_swap(pool, amount, True, report)
                    out = v2_get_amount_out(amount, r0, r1, fee)
                    report.checks += 1
                    if out < prev:
                        report.violations.append(InvariantViolation(
                            "v2.monotonic_out", f"bounded:{fee}", f"out {prev} -> {out}", (r0, r1, amount, fee)))
                    prev = out
    return report


__all__ = [
    "InvariantViolation",
    "InvariantReport",
    "check_v2_swap",
    "check_concentrated_swap",
    "check_swap",
    "fuzz_amount",
    "input_reserve_scale",
    "fuzz_pool",
    "fuzz_tasks",
    "parallel_fuzz",
    "bounded_exhaustive_v2",
]
