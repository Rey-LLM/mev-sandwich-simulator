"""Layer 10: core integrity validator (pre-flight static audit).

:class:`IntegrityValidator` runs before any simulation loop starts (the
controller calls it first and refuses to run when it finds a blocking
problem). It never mutates the pools it is given and needs no network. Four
rule families are checked:

``state.*`` - pool state invariants
    Unique pool ids, distinct tokens, v2 reserves positive and inside uint112,
    v3/v4 sqrt price inside ``[MIN_SQRT_RATIO, MAX_SQRT_RATIO)``, the cached
    tick consistent with the price, liquidity inside uint128, initialized
    ticks aligned to the spacing and in range, the sorted tick array in sync
    with the tick map, net liquidity summing to zero with no negative prefix,
    active liquidity equal to the sum of nets at or below the current tick,
    fee ranges, quotes that do not mutate state, and a StateFork
    snapshot / revert / fork round trip that restores the exact state root.

``tick.*`` - tick arithmetic and rounding
    Every TickMath magic constant is recomputed from ``1.0001`` with
    100-digit decimals and must be the nearest integer; ``sqrt(1.0001^t)``
    is checked against the same reference, for anchors and a seeded tick
    sample; tick <-> sqrt price round trips and the floor property; strict
    monotonicity; amount deltas bracket the exact rational value with a gap
    of at most one wei; SwapMath steps never round in the trader's favour and
    never spend more than offered; ``price_to_sqrt_price_x96`` floors; and a
    full sweep of every pool's tick array crosses exactly the initialized
    ticks and ends at the liquidity the array predicts.

``signal.*`` - signal boundary limits
    Basis-point fields within ``[0, 10000]``, gas fields positive and inside
    uint256 with a block limit in ``[21000, 30M]`` (and room for a sandwich),
    counts non-negative, the CPU policy and worker count against this
    machine, the multiprocessing start method available on this platform,
    shard density, path-fuzzer and AI agent factory sizes, the MEV pattern store
    settings (Layer 12), the RL regime, state-bin edges and
    their coverage of the regime's signal ranges, the action grid, softmax
    stability on extreme preferences, reward clipping on extreme rewards,
    learning-rate / exploration schedules, and the trace destination.

``code.*`` - logic flaws in the simulator's own source (AST scan)
    Syntax errors, bare or silent ``except``, mutable default arguments,
    lambdas or closures handed to worker pools (they cannot be pickled under
    ``spawn``), the global unseeded ``random`` API or time-seeded RNGs
    (non-deterministic runs), process pools created outside
    ``simulator.parallel`` (they bypass the CPU policy), ``__all__`` names
    that do not exist, duplicate dict keys, identity comparison with
    literals, division by a literal zero, ``int()`` of a true division
    (precision loss above 2**53 wei), float and ``None`` equality,
    unreachable statements, self-assignment, and text-mode ``open`` without
    an explicit encoding.

Every rule has a default :class:`Severity`. A report is *blocking* when any
finding is at or above ``fail_on`` (``error`` by default), and the controller
raises :class:`IntegrityError` in that case.
"""

from __future__ import annotations

import ast
import copy
import math
import multiprocessing
import os
import random
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from enum import IntEnum
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .config import (
    AUDIT_LEVELS,
    MAX_AGENT_BOTS,
    MAX_AGENT_GENERATIONS,
    MAX_SHARD_DENSITY,
    RL_REGIMES,
    ConfigError,
    CpuPolicy,
    RunMode,
    detect_cpu_count,
    worker_ceiling,
)
from .slippage_engine import (
    FEE_PIPS_DENOMINATOR,
    MAX_SQRT_RATIO,
    MAX_TICK,
    MIN_SQRT_RATIO,
    MIN_TICK,
    Q96,
    V4_MAX_LP_FEE,
    V4_MAX_PROTOCOL_FEE,
    ConcentratedPool,
    MASEError,
    Pool,
    V2Pool,
    V4Pool,
    _TICK_MAGIC,
    compute_swap_step,
    get_amount0_delta,
    get_amount1_delta,
    get_sqrt_ratio_at_tick,
    get_tick_at_sqrt_ratio,
    price_to_sqrt_price_x96,
)

FAMILIES: Tuple[str, ...] = ("state", "tick", "signal", "code")
UINT112 = 1 << 112
UINT128 = 1 << 128
INT128 = 1 << 127
UINT256 = 1 << 256
TICK_PRECISION_REL = Decimal("1e-18")  # observed worst case is ~4.5e-20
MAGIC_TOLERANCE = Decimal("0.5000001")  # constants are round-to-nearest
SANDWICH_MIN_GAS = 3 * 81_000  # front-run + victim + back-run, cheapest v2 hop each
GOVERNED_POOL_MODULE = "parallel.py"
_DEC_PREC = 100


class Severity(IntEnum):
    INFO = 10
    WARNING = 20
    ERROR = 30

    @property
    def label(self) -> str:
        return self.name.lower()

    @classmethod
    def parse(cls, value: Any) -> "Severity":
        if isinstance(value, cls):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return cls(value)
        key = str(value).strip().lower()
        for member in cls:
            if member.label == key:
                return member
        raise ConfigError(f"unknown severity {value!r} (expected {', '.join(AUDIT_LEVELS)})")


@dataclass(frozen=True)
class Rule:
    rule_id: str
    severity: Severity
    summary: str

    @property
    def family(self) -> str:
        return self.rule_id.split(".", 1)[0]

    def to_dict(self) -> Dict[str, Any]:
        return {"rule": self.rule_id, "family": self.family, "severity": self.severity.label,
                "summary": self.summary}


RULES: Dict[str, Rule] = {}


def _rule(rule_id: str, severity: Severity, summary: str) -> str:
    if rule_id in RULES:
        raise ValueError(f"duplicate rule {rule_id}")
    if rule_id.split(".", 1)[0] not in FAMILIES:
        raise ValueError(f"rule {rule_id} has no known family")
    RULES[rule_id] = Rule(rule_id, severity, summary)
    return rule_id


E, W, I = Severity.ERROR, Severity.WARNING, Severity.INFO

# -- state --------------------------------------------------------------------
R_DUP_POOL = _rule("state.duplicate_pool_id", E, "two pools share one pool id")
R_SAME_TOKENS = _rule("state.identical_tokens", E, "a pool trades a token against itself")
R_V2_RESERVES = _rule("state.v2_reserves", E, "v2 reserves must be positive")
R_V2_BOUNDS = _rule("state.v2_reserve_bounds", E, "v2 reserves must fit in uint112")
R_V2_FEE = _rule("state.v2_fee", E, "v2 fee_bps must be within [0, 10000)")
R_CL_PRICE = _rule("state.cl_sqrt_price_range", E, "sqrt price must be within [MIN_SQRT_RATIO, MAX_SQRT_RATIO)")
R_CL_TICK_CACHE = _rule("state.cl_tick_cache", E, "cached tick must bracket the sqrt price")
R_CL_LIQ_BOUNDS = _rule("state.cl_liquidity_bounds", E, "liquidity must fit uint128 and nets int128")
R_CL_ALIGN = _rule("state.cl_tick_alignment", E, "initialized ticks must be multiples of tick_spacing")
R_CL_TICK_RANGE = _rule("state.cl_tick_range", E, "initialized ticks must be within [MIN_TICK, MAX_TICK]")
R_CL_ARRAY_SYNC = _rule("state.cl_tick_array_sync", E, "sorted tick array must match the tick map")
R_CL_NET_SUM = _rule("state.cl_net_liquidity_sum", E, "liquidity nets must sum to zero")
R_CL_NET_PREFIX = _rule("state.cl_net_liquidity_prefix", E, "no tick range may hold negative liquidity")
R_CL_ACTIVE = _rule("state.cl_active_liquidity", E, "active liquidity must equal the nets at or below the tick")
R_CL_FEE = _rule("state.cl_fee", E, "v3 fee_pips must be within [0, 1000000)")
R_V4_FEE = _rule("state.v4_fee", E, "v4 hook, protocol and effective fees must be in range")
R_NATIVE_MISSING = _rule("state.native_token_missing", W, "no pool trades the native token")
R_QUOTE_PURITY = _rule("state.quote_purity", E, "quoting a swap must not mutate the pool")
R_FORK_ROUNDTRIP = _rule("state.fork_roundtrip", E, "snapshot / revert / fork must restore the state root")
R_NO_POOLS = _rule("state.no_pools", W, "there are no pools to simulate")

# -- tick ---------------------------------------------------------------------
R_MAGIC = _rule("tick.magic_constants", E, "TickMath magic constants must round 2^128/1.0001^(m/2)")
R_ANCHOR = _rule("tick.anchor", E, "MIN_TICK, 0 and MAX_TICK must map to their exact sqrt ratios")
R_ROUND_TRIP = _rule("tick.round_trip", E, "tick -> sqrt price -> tick must round trip and floor")
R_MONOTONIC = _rule("tick.monotonic", E, "sqrt ratio must strictly increase with the tick")
R_PRECISION = _rule("tick.precision", E, "sqrt ratio must match sqrt(1.0001^t)*2^96 within 1 + 1e-18 rel")
R_DELTA = _rule("tick.delta_rounding", E, "amount deltas must bracket the exact value within one wei")
R_STEP = _rule("tick.step_conservation", E, "swap steps must round in the pool's favour and conserve input")
R_PRICE_FLOOR = _rule("tick.price_floor", E, "price_to_sqrt_price_x96 must floor the square root")
R_CROSSING = _rule("tick.crossing_liquidity", E, "a full sweep must cross every initialized tick exactly once")
R_SPACING = _rule("tick.nonstandard_spacing", I, "tick spacing is not one of 1, 10, 60, 200")

# -- signal -------------------------------------------------------------------
R_BPS = _rule("signal.bps_range", E, "basis-point fields must be within [0, 10000]")
R_GAS = _rule("signal.gas_bounds", E, "gas fields must be positive, fit uint256 and respect 30M")
R_GAS_CAP = _rule("signal.gas_capacity", W, "the block gas limit cannot fit a three-transaction sandwich")
R_COUNTS = _rule("signal.counts", E, "block, tx, grid, fuzz and RL counts must be in range")
R_WORKERS_LOW = _rule("signal.workers_range", E, "workers must be >= 1")
R_WORKERS_CLAMP = _rule("signal.workers_clamped", W, "requested workers exceed this machine's policy ceiling")
R_CPU_POLICY = _rule("signal.cpu_policy", E, "cpu_policy must be capped or full")
R_MODE_POLICY = _rule("signal.mode_policy", I, "the mode runs under a non-default CPU policy")
R_START = _rule("signal.start_method", E, "the multiprocessing start method must exist on this platform")
R_THRESHOLD = _rule("signal.parallel_threshold", E, "parallel_threshold must be >= 1")
R_DENSITY = _rule("signal.shard_density", E, "shard_density must be within [1, 64]")
R_HEADROOM = _rule("signal.parallel_headroom", W, "fewer parallel tasks per stage than workers")
R_PG = _rule("signal.pg_paths", E, "path fuzzer sizes must be valid")
R_PG_IDLE = _rule("signal.pg_paths_ignored", I, "pg_paths is set but the mode does not run the path fuzzer")
R_AGENTS = _rule("signal.agent_factory", E, "AI agent factory sizes must be valid")
R_AGENTS_IDLE = _rule("signal.agent_factory_ignored", I, "agent_bots is set but the mode does not run the factory")
R_PATTERNS = _rule("signal.pattern_store", E, "MEV pattern settings must be valid and the store must exist")
R_PATTERNS_IDLE = _rule("signal.pattern_seed_ignored", I, "pattern_seed_bps is set but no AI factory runs")
R_REGIME = _rule("signal.rl_regime", E, "rl_regime must be calm, volatile or extreme")
R_BINS = _rule("signal.rl_bins", E, "RL state-bin edges must be strictly increasing")
R_BIN_COVER = _rule("signal.rl_bin_coverage", W, "an RL signal's bin edges never split the regime's range")
R_ACTIONS = _rule("signal.action_grid", E, "the RL action grid must be unique, bounded and include skip")
R_SOFTMAX = _rule("signal.softmax_stability", E, "softmax must stay a finite distribution on extreme input")
R_REWARD_CLIP = _rule("signal.reward_clip", E, "reward normalisation must stay inside [-clip, clip]")
R_SCHEDULE = _rule("signal.learning_schedule", E, "learning rates and exploration must stay in (0, 1]")
R_TRACE = _rule("signal.trace_path", E, "the trace destination must be a writable file path")
R_AUDIT = _rule("signal.audit_settings", E, "audit settings must be valid")
R_NATIVE = _rule("signal.native_token", E, "native_token must be a non-empty string")

# -- code ---------------------------------------------------------------------
R_SYNTAX = _rule("code.syntax", E, "the module must parse")
R_BARE_EXCEPT = _rule("code.bare_except", E, "bare except also catches SystemExit and KeyboardInterrupt")
R_SILENT_EXCEPT = _rule("code.silent_except", W, "an exception handler silently discards the error")
R_MUTABLE_DEFAULT = _rule("code.mutable_default", E, "mutable default argument is shared between calls")
R_UNPICKLABLE = _rule("code.unpicklable_task", E, "lambda or closure passed to a worker pool cannot be pickled")
R_GLOBAL_RANDOM = _rule("code.global_random", E, "global unseeded random API makes runs non-deterministic")
R_TIME_SEED = _rule("code.time_seeded_rng", E, "RNG seeded from the clock makes runs non-deterministic")
R_UNGOVERNED = _rule("code.ungoverned_pool", E, "process pool created outside simulator.parallel")
R_UNGOVERNED_PROC = _rule("code.ungoverned_process", W, "process started outside simulator.parallel")
R_ALL = _rule("code.all_exports", E, "__all__ names a symbol the module does not define")
R_ALL_DUP = _rule("code.all_duplicates", W, "__all__ lists a name twice")
R_DICT_KEY = _rule("code.duplicate_dict_key", E, "duplicate key in a dict literal overwrites a value")
R_LIT_IDENTITY = _rule("code.literal_identity", E, "'is' comparison with a literal")
R_DIV_ZERO = _rule("code.division_by_zero", E, "division or modulo by a literal zero")
R_FLOAT_TRUNC = _rule("code.float_truncation", W, "int() of a true division loses precision above 2**53")
R_FLOAT_EQ = _rule("code.float_equality", W, "exact equality against a float literal")
R_NONE_EQ = _rule("code.none_equality", W, "== None instead of 'is None'")
R_UNREACHABLE = _rule("code.unreachable", W, "statement after return, raise, break or continue")
R_SELF_ASSIGN = _rule("code.self_assign", W, "a name is assigned to itself")
R_OPEN_ENCODING = _rule("code.open_encoding", W, "text-mode open() without an explicit encoding")
R_BAD_PRAGMA = _rule("code.bad_pragma", W, "an '# audit: allow' comment names a rule that does not exist")

del E, W, I


# ---------------------------------------------------------------------------
# Findings and reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: Severity
    subject: str
    message: str

    @property
    def family(self) -> str:
        return self.rule.split(".", 1)[0]

    def to_dict(self) -> Dict[str, Any]:
        return {"rule": self.rule, "severity": self.severity.label, "subject": self.subject,
                "message": self.message}


class Collector:
    """Accumulates findings and per-family check counts."""

    def __init__(self) -> None:
        self.findings: List[Finding] = []
        self.checks: Counter = Counter()
        self.notes: List[str] = []  # context for the reader (e.g. the workload estimate); never blocks a run
        self.suppressed = 0  # code findings silenced by an '# audit: allow <rule>' comment

    def check(self, rule: str, ok: bool, subject: str, message: str = "",
              severity: Optional[Severity] = None) -> bool:
        self.checks[rule.split(".", 1)[0]] += 1
        if not ok:
            self.emit(rule, subject, message, severity)
        return ok

    def emit(self, rule: str, subject: str, message: str = "", severity: Optional[Severity] = None) -> None:
        r = RULES[rule]
        self.findings.append(Finding(rule, r.severity if severity is None else severity, subject,
                                     message or r.summary))

    def extend(self, findings: Iterable[Finding], checks: Mapping[str, int], suppressed: int = 0) -> None:
        self.findings.extend(findings)
        self.checks.update(checks)
        self.suppressed += suppressed

    def note(self, text: str) -> None:
        self.notes.append(text)


@dataclass
class AuditReport:
    findings: List[Finding]
    checks_by_family: Dict[str, int]
    elapsed_s: float
    fail_on: Severity = Severity.ERROR
    families: Tuple[str, ...] = FAMILIES
    files_scanned: int = 0
    pools_checked: int = 0
    notes: List[str] = field(default_factory=list)
    suppressed: int = 0

    @property
    def checks(self) -> int:
        return sum(self.checks_by_family.values())

    def blocking(self) -> List[Finding]:
        return [f for f in self.findings if f.severity >= self.fail_on]

    @property
    def ok(self) -> bool:
        return not self.blocking()

    def counts(self) -> Dict[str, int]:
        c = Counter(f.severity.label for f in self.findings)
        return {s.label: c.get(s.label, 0) for s in sorted(Severity, reverse=True)}

    def by_rule(self) -> Dict[str, int]:
        return dict(sorted(Counter(f.rule for f in self.findings).items()))

    def rules_triggered(self) -> Set[str]:
        return {f.rule for f in self.findings}

    def to_dict(self, max_findings: int = 50) -> Dict[str, Any]:
        ordered = sorted(self.findings, key=lambda f: (-int(f.severity), f.rule, f.subject))
        return {
            "ok": self.ok,
            "fail_on": self.fail_on.label,
            "families": list(self.families),
            "checks": self.checks,
            "checks_by_family": dict(sorted(self.checks_by_family.items())),
            "counts": self.counts(),
            "blocking": len(self.blocking()),
            "by_rule": self.by_rule(),
            "findings": [f.to_dict() for f in ordered[:max_findings]],
            "finding_count": len(self.findings),
            "files_scanned": self.files_scanned,
            "pools_checked": self.pools_checked,
            "suppressed": self.suppressed,
            "notes": list(self.notes),
            "elapsed_s": round(self.elapsed_s, 4),
        }


class IntegrityError(RuntimeError):
    """Raised by the controller when the pre-flight audit finds a blocking problem."""

    def __init__(self, report: AuditReport) -> None:
        self.report = report
        top = report.blocking()[:3]
        detail = "; ".join(f"{f.rule} ({f.subject}): {f.message}" for f in top)
        super().__init__(f"integrity audit failed with {len(report.blocking())} blocking finding(s): {detail}")


# ---------------------------------------------------------------------------
# state.*: pool invariants
# ---------------------------------------------------------------------------


def _fingerprint(pool: Pool) -> Tuple[Any, ...]:
    if isinstance(pool, V2Pool):
        return ("v2", pool.reserve0, pool.reserve1, pool.fee_bps)
    return (pool.version, pool.sqrt_price_x96, pool.tick, pool.liquidity, tuple(sorted(pool.ticks.items())),
            tuple(pool.tick_array), pool.fee_pips)


def _probe_amount(pool: Pool, zero_for_one: bool) -> int:
    if isinstance(pool, V2Pool):
        r_in = pool.reserve0 if zero_for_one else pool.reserve1
        return max(1, r_in // 1_000_000)
    L = max(pool.liquidity, 1)
    scale = L * Q96 // pool.sqrt_price_x96 if zero_for_one else L * pool.sqrt_price_x96 // Q96
    return max(1, scale // 1_000_000)


def _cl_state_ok(pool: ConcentratedPool) -> bool:
    return MIN_SQRT_RATIO <= pool.sqrt_price_x96 < MAX_SQRT_RATIO and MIN_TICK <= pool.tick <= MAX_TICK


def check_pool(pool: Pool, col: Collector) -> None:
    """Every state.* rule that concerns a single pool."""
    pid = pool.pool_id
    t0, t1 = pool.tokens()
    col.check(R_SAME_TOKENS, t0 != t1, pid, f"token0 == token1 == {t0!r}")
    if isinstance(pool, V2Pool):
        positive = col.check(R_V2_RESERVES, pool.reserve0 > 0 and pool.reserve1 > 0, pid,
                             f"reserves {pool.reserve0}/{pool.reserve1}")
        col.check(R_V2_BOUNDS, pool.reserve0 < UINT112 and pool.reserve1 < UINT112, pid,
                  "a reserve does not fit in uint112")
        col.check(R_V2_FEE, 0 <= pool.fee_bps < 10_000, pid, f"fee_bps={pool.fee_bps}")
        if positive:
            _check_quote_purity(pool, col)
        return
    if not isinstance(pool, ConcentratedPool):
        col.emit(R_V2_RESERVES, pid, f"unknown pool type {type(pool).__name__}")
        return
    in_range = col.check(R_CL_PRICE, MIN_SQRT_RATIO <= pool.sqrt_price_x96 < MAX_SQRT_RATIO, pid,
                         f"sqrtPriceX96={pool.sqrt_price_x96}")
    col.check(R_CL_LIQ_BOUNDS, 0 <= pool.liquidity < UINT128, pid, f"liquidity={pool.liquidity}")
    col.check(R_CL_FEE, 0 <= pool.fee_pips < FEE_PIPS_DENOMINATOR, pid, f"fee_pips={pool.fee_pips}")
    spacing = pool.tick_spacing
    col.check(R_CL_ALIGN, spacing > 0, pid, f"tick_spacing={spacing}")
    for t, net in sorted(pool.ticks.items()):
        if spacing > 0:
            col.check(R_CL_ALIGN, t % spacing == 0, pid, f"tick {t} not a multiple of {spacing}")
        col.check(R_CL_TICK_RANGE, MIN_TICK <= t <= MAX_TICK, pid, f"tick {t} out of range")
        col.check(R_CL_LIQ_BOUNDS, -INT128 <= net < INT128, pid, f"liquidityNet {net} at tick {t} exceeds int128")
    col.check(R_CL_ARRAY_SYNC, list(pool.tick_array) == sorted(pool.ticks), pid,
              f"tick array {list(pool.tick_array)[:6]}... differs from the tick map")
    nets = [pool.ticks[t] for t in sorted(pool.ticks)]
    col.check(R_CL_NET_SUM, sum(nets) == 0, pid, f"sum of liquidityNet is {sum(nets)}")
    running = 0
    lowest = None
    for t in sorted(pool.ticks):
        running += pool.ticks[t]
        if running < 0 and lowest is None:
            lowest = (t, running)
    col.check(R_CL_NET_PREFIX, lowest is None, pid,
              "" if lowest is None else f"liquidity {lowest[1]} above tick {lowest[0]}")
    if in_range:
        _check_tick_cache(pool, col)
        if MIN_TICK <= pool.tick <= MAX_TICK:
            active = sum(n for t, n in pool.ticks.items() if t <= pool.tick)
            col.check(R_CL_ACTIVE, active == pool.liquidity, pid,
                      f"liquidity {pool.liquidity} but nets at or below tick {pool.tick} sum to {active}")
    if isinstance(pool, V4Pool):
        ok = (pool.hook_fee_pips is None or 0 <= pool.hook_fee_pips <= V4_MAX_LP_FEE) \
            and 0 <= pool.protocol_fee_pips <= V4_MAX_PROTOCOL_FEE
        try:
            eff = pool.effective_fee()
            ok = ok and 0 <= eff <= FEE_PIPS_DENOMINATOR
        except MASEError as exc:
            ok, eff = False, str(exc)
        col.check(R_V4_FEE, ok, pid, f"hook={pool.hook_fee_pips} protocol={pool.protocol_fee_pips} effective={eff}")
    if in_range and _cl_state_ok(pool):
        _check_quote_purity(pool, col)


def _check_tick_cache(pool: ConcentratedPool, col: Collector) -> None:
    """``ratio(tick) <= sqrtP <= ratio(tick + 1)``; equality on the right is the state after a downward cross."""
    pid = pool.pool_id
    if not MIN_TICK <= pool.tick <= MAX_TICK:
        col.emit(R_CL_TICK_CACHE, pid, f"tick {pool.tick} out of range")
        return
    lo = get_sqrt_ratio_at_tick(pool.tick)
    hi = get_sqrt_ratio_at_tick(pool.tick + 1) if pool.tick < MAX_TICK else MAX_SQRT_RATIO
    col.check(R_CL_TICK_CACHE, lo <= pool.sqrt_price_x96 <= hi, pid,
              f"tick {pool.tick} does not bracket sqrtPriceX96 {pool.sqrt_price_x96}")


def _check_quote_purity(pool: Pool, col: Collector) -> None:
    before = _fingerprint(pool)
    for zero_for_one in (True, False):
        try:
            pool.quote_exact_in(_probe_amount(pool, zero_for_one), zero_for_one)
        except MASEError:
            pass
    col.check(R_QUOTE_PURITY, _fingerprint(pool) == before, pool.pool_id, "quote_exact_in changed pool state")


def check_pools(pools: Sequence[Pool], native_token: Optional[str], col: Collector) -> None:
    """state.* rules over the whole market, including the StateFork round trip."""
    if not col.check(R_NO_POOLS, bool(pools), "market", "no pools given"):
        return
    ids = Counter(p.pool_id for p in pools)
    for pid, n in sorted(ids.items()):
        col.check(R_DUP_POOL, n == 1, pid, f"{n} pools use id {pid!r}")
    for pool in pools:
        check_pool(pool, col)
    if native_token:
        col.check(R_NATIVE_MISSING, any(native_token in p.tokens() for p in pools), native_token,
                  f"no pool trades {native_token}; native-denominated profit and gas cannot be priced")
    _check_fork_roundtrip(pools, col)


def _check_fork_roundtrip(pools: Sequence[Pool], col: Collector) -> None:
    from .state_fork import StateFork, SwapReverted

    if len({p.pool_id for p in pools}) != len(pools):
        return
    try:
        state = StateFork(copy.deepcopy(list(pools)))
        root = state.state_root()
        sid = state.snapshot()
        for pid, pool in list(state.pools.items()):
            if isinstance(pool, ConcentratedPool) and not _cl_state_ok(pool):
                continue
            token = pool.tokens()[0]
            try:
                state.swap("validator", pid, token, _probe_amount(pool, True))
            except (SwapReverted, MASEError):
                pass
        state.mine()
        state.revert(sid)
        ok = state.state_root() == root and state.fork().state_root() == root
        col.check(R_FORK_ROUNDTRIP, ok, "state_fork", "revert or fork did not restore the state root")
    except Exception as exc:  # any crash here is itself a finding
        col.emit(R_FORK_ROUNDTRIP, "state_fork", f"round trip raised {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# tick.*: TickMath, SqrtPriceMath and SwapMath rounding
# ---------------------------------------------------------------------------

_SQRT_BASE_CACHE: Dict[int, Decimal] = {}


def _sqrt_base() -> Decimal:
    with localcontext() as ctx:
        ctx.prec = _DEC_PREC
        if _DEC_PREC not in _SQRT_BASE_CACHE:
            _SQRT_BASE_CACHE[_DEC_PREC] = Decimal("1.0001").sqrt()
        return _SQRT_BASE_CACHE[_DEC_PREC]


def exact_sqrt_ratio(tick: int) -> Decimal:
    """``sqrt(1.0001 ** tick) * 2**96`` to 100 significant digits."""
    with localcontext() as ctx:
        ctx.prec = _DEC_PREC
        return _sqrt_base() ** tick * (Decimal(2) ** 96)


def exact_magic(mask: int) -> Decimal:
    """``2**128 / sqrt(1.0001) ** mask``: the value TickMath's constant for ``mask`` rounds."""
    with localcontext() as ctx:
        ctx.prec = _DEC_PREC
        return (Decimal(2) ** 128) / (_sqrt_base() ** mask)


def tick_magic_table() -> List[Tuple[int, int]]:
    """``(mask, constant)`` pairs used by ``get_sqrt_ratio_at_tick``, bit 0 included when it can be found."""
    raw = getattr(get_sqrt_ratio_at_tick, "__wrapped__", get_sqrt_ratio_at_tick)
    consts = [c for c in getattr(raw, "__code__").co_consts
              if isinstance(c, int) and not isinstance(c, bool) and (1 << 127) < c < (1 << 128)]
    table = [(0x1, consts[0])] if consts else []
    return table + list(_TICK_MAGIC)


def check_magic_constants(col: Collector, table: Optional[Sequence[Tuple[int, int]]] = None) -> None:
    for mask, magic in (tick_magic_table() if table is None else table):
        diff = abs(Decimal(magic) - exact_magic(mask))
        col.check(R_MAGIC, diff <= MAGIC_TOLERANCE, f"mask {mask:#x}",
                  f"constant {magic:#x} is {diff:.3g} from 2^128/1.0001^({mask}/2)")


def sample_ticks(samples: int, seed: int = 0) -> List[int]:
    """Anchors, the neighbourhood of 0 and both ends, plus ``samples`` seeded random ticks."""
    rng = random.Random(f"validator-ticks:{seed}")
    fixed = {MIN_TICK, MAX_TICK, 0}
    fixed.update(range(-16, 17))
    fixed.update(range(MIN_TICK, MIN_TICK + 5))
    fixed.update(range(MAX_TICK - 4, MAX_TICK + 1))
    fixed.update(rng.randint(MIN_TICK, MAX_TICK) for _ in range(max(0, samples)))
    return sorted(fixed)


def check_tick_math(col: Collector, samples: int = 256, seed: int = 0,
                    ratio_fn: Callable[[int], int] = get_sqrt_ratio_at_tick,
                    tick_fn: Callable[[int], int] = get_tick_at_sqrt_ratio) -> None:
    """Global TickMath / SqrtPriceMath / SwapMath checks (pool independent).

    An exception from the math under test on a valid input is reported as a
    finding of the rule being checked, never raised.
    """
    check_magic_constants(col)
    for subject, fn, arg, expected in (("MIN_TICK", ratio_fn, MIN_TICK, MIN_SQRT_RATIO),
                                       ("MAX_TICK", ratio_fn, MAX_TICK, MAX_SQRT_RATIO),
                                       ("tick 0", ratio_fn, 0, Q96),
                                       ("MIN_SQRT_RATIO", tick_fn, MIN_SQRT_RATIO, MIN_TICK),
                                       ("MAX_SQRT_RATIO-1", tick_fn, MAX_SQRT_RATIO - 1, MAX_TICK - 1)):
        _guard(col, R_ANCHOR, subject, _check_anchor, col, subject, fn, arg, expected)
    ticks = sample_ticks(samples, seed)
    for t in ticks:
        _guard(col, R_ROUND_TRIP, f"tick {t}", _check_tick, col, t, ratio_fn, tick_fn)
    rng = random.Random(f"validator-math:{seed}")
    n = max(8, min(samples, 512))
    for _ in range(n):
        ta, tb = sorted(rng.sample(ticks, 2)) if len(ticks) > 1 else (MIN_TICK, MAX_TICK)
        if ta == tb:
            continue
        liquidity = rng.randint(1, UINT128 - 1)
        _guard(col, R_DELTA, f"ticks {ta}..{tb}", lambda a=ta, b=tb, liq=liquidity:
               _check_deltas(col, ratio_fn(a), ratio_fn(b), liq))
    for _ in range(n):
        _guard(col, R_STEP, "random step", _check_random_step, col, rng, ticks, ratio_fn)
    for _ in range(max(4, n // 4)):
        num = rng.randint(1, 1 << 120)
        den = rng.randint(1, 1 << 120)
        p = Fraction(num, den)
        s = price_to_sqrt_price_x96(p)
        target = Fraction(p.numerator << 192, p.denominator)
        col.check(R_PRICE_FLOOR, s * s <= target < (s + 1) * (s + 1), f"price {num}/{den}",
                  f"sqrt {s} is not the floor of sqrt(p * 2^192)")


def _guard(col: Collector, rule: str, subject: str, check: Callable[..., Any], *args: Any) -> None:
    """Run ``check(*args)``; an exception from the code under test becomes a finding of ``rule``."""
    try:
        check(*args)
    except Exception as exc:  # a crash of the math under test on valid input is itself a finding
        col.emit(rule, subject, f"raised {type(exc).__name__}: {exc}")


def _check_anchor(col: Collector, subject: str, fn: Callable[[int], int], arg: int, expected: int) -> None:
    got = fn(arg)
    col.check(R_ANCHOR, got == expected, subject, f"got {got}, expected {expected}")


def _check_tick(col: Collector, t: int, ratio_fn: Callable[[int], int], tick_fn: Callable[[int], int]) -> None:
    r = ratio_fn(t)
    if t < MAX_TICK:
        back = tick_fn(r)
        col.check(R_ROUND_TRIP, back == t, f"tick {t}", f"ratio {r} maps back to {back}")
        nxt = ratio_fn(t + 1)
        col.check(R_MONOTONIC, r < nxt, f"tick {t}", f"ratio({t})={r} >= ratio({t + 1})={nxt}")
    if t > MIN_TICK:
        col.check(R_ROUND_TRIP, tick_fn(r - 1) == t - 1, f"tick {t}", "ratio - 1 does not floor to t - 1")
    exact = exact_sqrt_ratio(t)
    err = abs(Decimal(r) - exact)
    col.check(R_PRECISION, err <= 1 + exact * TICK_PRECISION_REL, f"tick {t}",
              f"|ratio - exact| = {err:.4g} (rel {err / exact:.3g})")


def _check_deltas(col: Collector, sa: int, sb: int, liquidity: int) -> None:
    e0 = Fraction(liquidity * Q96 * (sb - sa), sb * sa)
    up, dn = get_amount0_delta(sa, sb, liquidity, True), get_amount0_delta(sa, sb, liquidity, False)
    col.check(R_DELTA, dn <= e0 <= up and 0 <= up - dn <= 1, f"amount0 L={liquidity}",
              f"down={dn} up={up} exact~{float(e0):.6g}")
    e1 = Fraction(liquidity * (sb - sa), Q96)
    up, dn = get_amount1_delta(sa, sb, liquidity, True), get_amount1_delta(sa, sb, liquidity, False)
    col.check(R_DELTA, dn <= e1 <= up and 0 <= up - dn <= 1, f"amount1 L={liquidity}",
              f"down={dn} up={up} exact~{float(e1):.6g}")


def _exact_amounts(sqrt_current: int, sqrt_next: int, liquidity: int, zero_for_one: bool) -> Tuple[Fraction,
                                                                                                    Fraction]:
    """Exact (input, output) for moving the price from ``sqrt_current`` to ``sqrt_next``."""
    lo, hi = sorted((sqrt_current, sqrt_next))
    a0 = Fraction(liquidity * Q96 * (hi - lo), hi * lo)
    a1 = Fraction(liquidity * (hi - lo), Q96)
    return (a0, a1) if zero_for_one else (a1, a0)


def check_swap_step(col: Collector, sqrt_current: int, sqrt_target: int, liquidity: int, amount_remaining: int,
                    fee_pips: int) -> None:
    subject = f"step L={liquidity} amt={amount_remaining} fee={fee_pips}"
    try:
        sn, a_in, a_out, fee = compute_swap_step(sqrt_current, sqrt_target, liquidity, amount_remaining, fee_pips)
    except MASEError as exc:
        col.emit(R_STEP, subject, f"compute_swap_step raised {exc}")
        return
    zero_for_one = sqrt_current >= sqrt_target
    lo, hi = sorted((sqrt_current, sqrt_target))
    ok = a_in >= 0 and a_out >= 0 and fee >= 0 and lo <= sn <= hi
    exact_in, exact_out = _exact_amounts(sqrt_current, sn, liquidity, zero_for_one)
    ok = ok and a_in >= exact_in and a_out <= exact_out
    if amount_remaining > 0:
        ok = ok and a_in + fee <= amount_remaining
        if sn != sqrt_target:
            ok = ok and a_in + fee == amount_remaining
    else:
        ok = ok and a_out <= -amount_remaining
    col.check(R_STEP, ok, subject, f"next={sn} in={a_in} out={a_out} fee={fee} "
                                   f"exact_in~{float(exact_in):.6g} exact_out~{float(exact_out):.6g}")


def _check_random_step(col: Collector, rng: random.Random, ticks: Sequence[int],
                       ratio_fn: Callable[[int], int]) -> None:
    ta, tb = rng.sample(list(ticks), 2) if len(ticks) > 1 else (0, 1)
    current = ratio_fn(ta)
    if ta < MAX_TICK:
        span = ratio_fn(ta + 1) - current
        current += rng.randrange(max(1, span))
    target = ratio_fn(tb)
    if current == target:
        return
    liquidity = rng.randint(1, 1 << rng.choice((64, 96, 110, 127)))
    magnitude = rng.randint(1, 1 << rng.choice((16, 64, 96, 128)))
    amount = magnitude if rng.random() < 0.6 else -magnitude
    fee = rng.choice((0, 100, 500, 3_000, 10_000, 100_000))
    check_swap_step(col, current, target, liquidity, amount, fee)


def check_pool_ticks(pool: ConcentratedPool, col: Collector) -> None:
    """Sweep the pool's whole tick array in both directions on scratch copies."""
    pid = pool.pool_id
    col.check(R_SPACING, pool.tick_spacing in (1, 10, 60, 200), pid, f"tick_spacing={pool.tick_spacing}")
    if not _cl_state_ok(pool):
        return
    ticks = sorted(pool.ticks)
    for zero_for_one in (True, False):
        limit = MIN_SQRT_RATIO + 1 if zero_for_one else MAX_SQRT_RATIO - 1
        if (zero_for_one and pool.sqrt_price_x96 <= limit) or (not zero_for_one and pool.sqrt_price_x96 >= limit):
            continue
        if zero_for_one:
            crossed = [t for t in ticks if t <= pool.tick and get_sqrt_ratio_at_tick(t) > limit]
            expected_liq = pool.liquidity - sum(pool.ticks[t] for t in crossed)
        else:
            crossed = [t for t in ticks if t > pool.tick and get_sqrt_ratio_at_tick(t) < limit]
            expected_liq = pool.liquidity + sum(pool.ticks[t] for t in crossed)
        work = copy.deepcopy(pool)
        direction = "down" if zero_for_one else "up"
        try:
            res, sqrt_p, _tick, liquidity = work._swap(1 << 255, zero_for_one)
        except MASEError as exc:
            col.emit(R_CROSSING, pid, f"sweep {direction} raised {exc}")
            continue
        ok = res.ticks_crossed == len(crossed) and liquidity == expected_liq and sqrt_p == limit
        col.check(R_CROSSING, ok, pid, f"sweep {direction}: crossed {res.ticks_crossed} of {len(crossed)} ticks, "
                                       f"ended at L={liquidity} (expected {expected_liq}), sqrtP={sqrt_p}")


# ---------------------------------------------------------------------------
# signal.*: configuration and learning-signal bounds
# ---------------------------------------------------------------------------

BPS_FIELDS: Tuple[Tuple[str, int, int], ...] = (
    ("max_frontrun_bps", 1, 10_000),
    ("default_tolerance_bps", 0, 10_000),
    ("builder_margin_bps", 0, 10_000),
    ("bribe_bps", 0, 10_000),
    ("private_flow_bps", 0, 10_000),
    ("mev_share_refund_bps", 0, 10_000),
)
COUNT_FIELDS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("blocks", 1, None),
    ("txs_per_block", 0, None),
    ("frontrun_grid", 0, None),
    ("refine_iterations", 0, None),
    ("fuzz_iterations", 0, None),
    ("layout_fuzz_iterations", 0, None),
    ("router_max_hops", 1, 6),
    ("rl_rounds", 0, None),
    ("rl_steps_per_shard", 0, None),
    ("rl_eval_steps", 0, None),
)


def _int_in(value: Any, lo: int, hi: Optional[int]) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= lo and (hi is None or value <= hi)


def check_signals(cfg: Any, col: Collector, cpus: Optional[int] = None) -> None:
    """signal.* rules over a :class:`SimulationConfig` (or any object with the same attributes)."""
    cpus = detect_cpu_count() if cpus is None else cpus
    for name, lo, hi in BPS_FIELDS:
        value = getattr(cfg, name, None)
        col.check(R_BPS, _int_in(value, lo, hi), name, f"{name}={value!r} outside [{lo}, {hi}]")
    for name, lo, hi in COUNT_FIELDS:
        value = getattr(cfg, name, None)
        col.check(R_COUNTS, _int_in(value, lo, hi), name,
                  f"{name}={value!r} outside [{lo}, {'inf' if hi is None else hi}]")
    base, prio, limit = (getattr(cfg, n, None) for n in ("base_fee_wei", "priority_fee_wei", "block_gas_limit"))
    col.check(R_GAS, _int_in(base, 1, UINT256 - 1), "base_fee_wei", f"base_fee_wei={base!r}")
    col.check(R_GAS, _int_in(prio, 0, UINT256 - 1), "priority_fee_wei", f"priority_fee_wei={prio!r}")
    if _int_in(base, 1, None) and _int_in(prio, 0, None):
        col.check(R_GAS, base + prio < UINT256, "gas_price_wei", "base + priority fee overflows uint256")
    if col.check(R_GAS, _int_in(limit, 21_000, 30_000_000), "block_gas_limit", f"block_gas_limit={limit!r}"):
        mode = getattr(cfg, "mode", None)
        if isinstance(mode, RunMode) and mode.aggressive:
            col.check(R_GAS_CAP, limit >= SANDWICH_MIN_GAS, "block_gas_limit",
                      f"{limit} gas cannot fit front-run + victim + back-run (~{SANDWICH_MIN_GAS})")
    native = getattr(cfg, "native_token", None)
    col.check(R_NATIVE, isinstance(native, str) and bool(native.strip()), "native_token", f"{native!r}")
    _check_parallel_signals(cfg, col, cpus)
    _check_path_signals(cfg, col)
    _check_agent_signals(cfg, col)
    _check_pattern_signals(cfg, col)
    regime = getattr(cfg, "rl_regime", None)
    if col.check(R_REGIME, regime in RL_REGIMES, "rl_regime", f"rl_regime={regime!r}"):
        check_rl_signals(col, regime)
    else:
        check_rl_signals(col, None)
    _check_trace(cfg, col)
    _check_audit_settings(cfg, col)
    _workload_note(cfg, col)


def _check_parallel_signals(cfg: Any, col: Collector, cpus: int) -> None:
    workers = getattr(cfg, "workers", None)
    try:
        policy: Optional[CpuPolicy] = CpuPolicy.parse(getattr(cfg, "cpu_policy", "capped"))
        col.check(R_CPU_POLICY, True, "cpu_policy")
    except ConfigError as exc:
        policy = None
        col.emit(R_CPU_POLICY, "cpu_policy", str(exc))
    if col.check(R_WORKERS_LOW, _int_in(workers, 1, None), "workers", f"workers={workers!r}") and policy:
        ceiling = worker_ceiling(cpus, policy)
        col.check(R_WORKERS_CLAMP, workers <= ceiling, "workers",
                  f"{workers} workers requested; the {policy.value} policy allows {ceiling} on {cpus} CPUs "
                  f"and the pool will clamp to that")
    mode = getattr(cfg, "mode", None)
    if isinstance(mode, RunMode) and policy is not None:
        col.check(R_MODE_POLICY, policy is mode.default_cpu_policy, "cpu_policy",
                  f"{mode.label} runs under the {policy.value} policy ({policy.label})")
    method = getattr(cfg, "start_method", None)
    available = multiprocessing.get_all_start_methods()
    col.check(R_START, method in available, "start_method",
              f"{method!r} is not available here (available: {', '.join(available)})")
    col.check(R_THRESHOLD, _int_in(getattr(cfg, "parallel_threshold", None), 1, None), "parallel_threshold",
              f"parallel_threshold={getattr(cfg, 'parallel_threshold', None)!r}")
    density = getattr(cfg, "shard_density", None)
    if col.check(R_DENSITY, _int_in(density, 1, MAX_SHARD_DENSITY), "shard_density",
                 f"shard_density={density!r}") and _int_in(workers, 1, None) and policy is not None:
        effective = min(workers, worker_ceiling(cpus, policy))
        if isinstance(mode, RunMode) and mode.aggressive:
            tasks = 16 * density  # TASK_CHUNKS / FUZZ_SHARDS / RELAY_CHUNKS / ROUTER_CHUNKS base
            col.check(R_HEADROOM, tasks >= effective, "shard_density",
                      f"stages split into {tasks} tasks but {effective} workers run; raise shard_density "
                      f"to keep every core busy")


def _check_path_signals(cfg: Any, col: Collector) -> None:
    paths, steps, batch = (getattr(cfg, n, None) for n in ("pg_paths", "pg_path_steps", "pg_batch"))
    ok = _int_in(paths, 0, None) and _int_in(steps, 1, None) and _int_in(batch, 1, None)
    col.check(R_PG, ok, "pg_paths", f"pg_paths={paths!r} pg_path_steps={steps!r} pg_batch={batch!r}")
    mode = getattr(cfg, "mode", None)
    if ok and paths and isinstance(mode, RunMode):
        col.check(R_PG_IDLE, mode.aggressive, "pg_paths", f"{mode.label} does not run the path fuzzer")


def _check_agent_signals(cfg: Any, col: Collector) -> None:
    bots, gens, size, slots = (getattr(cfg, n, None) for n in ("agent_bots", "agent_generations",
                                                                 "agent_heat_size", "agent_heat_slots"))
    ok = _int_in(bots, 0, MAX_AGENT_BOTS) and _int_in(gens, 1, MAX_AGENT_GENERATIONS) and _int_in(size, 1, None) \
        and _int_in(slots, 1, None)
    col.check(R_AGENTS, ok, "agent_bots", f"agent_bots={bots!r} agent_generations={gens!r} "
                                          f"agent_heat_size={size!r} agent_heat_slots={slots!r}")
    mode = getattr(cfg, "mode", None)
    if ok and bots and isinstance(mode, RunMode):
        col.check(R_AGENTS_IDLE, mode.aggressive, "agent_bots", f"{mode.label} does not run the AI agent factory")


def _check_pattern_signals(cfg: Any, col: Collector) -> None:
    store, cap, seed, limit, synthetic = (getattr(cfg, n, None) for n in (
        "pattern_store", "pattern_max_storage_gb", "pattern_seed_bps", "pattern_limit", "pattern_synthetic"))
    if seed is None:
        return
    ok = isinstance(store, str) and _int_in(cap, 1, None) and _int_in(seed, 0, 10_000) and _int_in(limit, 0, None) \
        and _int_in(synthetic, 0, None)
    detail = f"pattern_store={store!r} max_storage_gb={cap!r} seed_bps={seed!r} limit={limit!r}"
    if ok and store and seed:
        root = Path(store)
        ok = (root / "index.sqlite").is_file()
        detail = f"no pattern store at {store!r} (create one with --ingest-patterns or --build-synthetic-patterns)"
    col.check(R_PATTERNS, ok, "pattern_store", detail)
    mode = getattr(cfg, "mode", None)
    if ok and seed and isinstance(mode, RunMode):
        runs = mode.aggressive and bool(getattr(cfg, "agent_bots", 0))
        col.check(R_PATTERNS_IDLE, runs, "pattern_seed_bps",
                  f"{mode.label} with agent_bots={getattr(cfg, 'agent_bots', 0)} does not run the AI factory")


def check_rl_signals(col: Collector, regime: Optional[str] = None, bins: Any = None,
                     actions: Optional[Sequence[Any]] = None) -> None:
    """RL state-bin, action-grid, softmax, reward-clip and schedule checks."""
    from .reinforcement_engine import (
        REGIMES,
        LearningSchedule,
        PolicyGradientAgent,
        RewardScaler,
        StateBins,
        action_grid,
    )

    if bins is None:
        try:
            bins = StateBins()
        except ConfigError as exc:
            col.emit(R_BINS, "StateBins()", str(exc))
            bins = None
    if bins is not None:
        for name in ("reserve_bps", "gas_bps", "delay_ms", "victim_bps", "rivals"):
            edges = list(getattr(bins, name))
            col.check(R_BINS, all(a < b for a, b in zip(edges, edges[1:])), name, f"edges {edges}")
        if regime in REGIMES:
            reg = REGIMES[regime]
            lo_d, hi_d = max(0, reg.delay_mean_ms - 3 * reg.delay_jitter_ms), reg.delay_mean_ms + 3 * reg.delay_jitter_ms
            for name, lo, hi in (("victim_bps", reg.victim_min_bps, reg.victim_max_bps), ("delay_ms", lo_d, hi_d)):
                edges = list(getattr(bins, name))
                col.check(R_BIN_COVER, any(lo < e <= hi for e in edges), name,
                          f"no edge of {edges} falls inside ({lo}, {hi}] for regime {regime}")
    try:
        grid = list(action_grid() if actions is None else actions)
    except ConfigError as exc:
        col.emit(R_ACTIONS, "action_grid", str(exc))
        grid = []
    if grid:
        keys = [(a.size_bps, a.bid_bps) for a in grid]
        col.check(R_ACTIONS, len(set(keys)) == len(keys), "action_grid", "duplicate actions")
        col.check(R_ACTIONS, all(0 <= s <= 20_000 and 0 <= b <= 10_000 for s, b in keys), "action_grid",
                  "an action is out of bounds")
        col.check(R_ACTIONS, any(s == 0 for s, _ in keys), "action_grid", "no skip action")
        agent = PolicyGradientAgent(actions=grid, temperature=0.25)
        n = len(grid)
        rows = [[1e6 if i == 0 else -1e6 for i in range(n)], [1e300] * n, [-1e300 if i % 2 else 0.0 for i in range(n)],
                [float(i) * 1e12 for i in range(n)]]
        for idx, row in enumerate(rows):
            agent.table.values[(idx,)] = list(row)
            agent.table.visits[(idx,)] = [0] * n
            p = agent.probs((idx,))
            ok = all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in p) and abs(math.fsum(p) - 1.0) <= 1e-9
            col.check(R_SOFTMAX, ok, f"row {idx}", f"probabilities {p[:4]}... sum {math.fsum(p)!r}")
    scaler = RewardScaler()
    for r in (10**40, -10**40, 1, -1, 0, 10**18):
        scaler.observe(r)
        v = scaler.normalize(r)
        col.check(R_REWARD_CLIP, math.isfinite(v) and abs(v) <= scaler.clip, f"reward {r}", f"normalized={v!r}")
    sched = LearningSchedule()
    for v in (0, 1, 10, 1_000, 10**6):
        lr = sched.lr(v)
        col.check(R_SCHEDULE, 0 < lr <= 1, f"lr(visits={v})", f"lr={lr!r}")
    for s in (0, 1, 100, 10**6):
        eps = sched.epsilon(s)
        col.check(R_SCHEDULE, 0 <= eps <= 1, f"epsilon(step={s})", f"epsilon={eps!r}")


def _check_trace(cfg: Any, col: Collector) -> None:
    path = getattr(cfg, "trace_path", "")
    if not isinstance(path, str):
        col.emit(R_TRACE, "trace_path", f"trace_path={path!r} is not a string")
        return
    if not getattr(cfg, "trace", False) or not path:
        col.check(R_TRACE, True, "trace_path")
        return
    p = Path(path)
    parent = p.parent if str(p.parent) else Path(".")
    ok = not p.is_dir() and parent.is_dir() and os.access(parent, os.W_OK) and (not p.exists() or os.access(p, os.W_OK))
    col.check(R_TRACE, ok, "trace_path", f"cannot write the trace to {path!r}")


def _check_audit_settings(cfg: Any, col: Collector) -> None:
    fail_on = getattr(cfg, "audit_fail_on", "error")
    samples = getattr(cfg, "audit_tick_samples", 0)
    col.check(R_AUDIT, fail_on in AUDIT_LEVELS, "audit_fail_on", f"audit_fail_on={fail_on!r}")
    col.check(R_AUDIT, _int_in(samples, 0, 100_000), "audit_tick_samples", f"audit_tick_samples={samples!r}")


def _workload_note(cfg: Any, col: Collector) -> None:
    mode = getattr(cfg, "mode", None)
    if not isinstance(mode, RunMode):
        return
    try:
        if mode.aggressive:
            pg = cfg.pg_paths * cfg.pg_path_steps
            rl = cfg.rl_rounds * 8 * cfg.shard_density * cfg.rl_steps_per_shard
            heats = -(-cfg.agent_bots // max(1, cfg.agent_heat_size)) * cfg.agent_generations
            note = (f"{mode.label}: {cfg.blocks} blocks x {cfg.txs_per_block} txs, grid {cfg.frontrun_grid}, "
                    f"{cfg.fuzz_iterations} fuzz swaps, {cfg.layout_fuzz_iterations} layouts/slot, "
                    f"{rl} RL steps, {pg} policy-gradient path steps, {cfg.agent_bots} AI bots x "
                    f"{cfg.agent_generations} generations ({heats} arena heats) on {cfg.workers} worker(s)")
        else:
            note = f"{mode.label}: {cfg.txs_per_block} txs quoted, {cfg.fuzz_iterations} fuzz swaps"
    except (AttributeError, TypeError):
        return
    col.note(note)


# ---------------------------------------------------------------------------
# code.*: AST scan of the simulator's own source
# ---------------------------------------------------------------------------

RANDOM_GLOBAL_FUNCS = frozenset({
    "random", "randint", "randrange", "choice", "choices", "shuffle", "sample", "uniform", "gauss",
    "normalvariate", "lognormvariate", "expovariate", "betavariate", "gammavariate", "triangular",
    "vonmisesvariate", "paretovariate", "weibullvariate", "getrandbits", "seed", "randbytes",
    "binomialvariate",
})
POOL_METHODS = frozenset({"map", "map_parallel", "imap", "imap_unordered", "starmap", "map_async",
                          "starmap_async", "apply_async", "submit"})
POOL_CONSTRUCTORS = frozenset({"Pool", "ProcessPoolExecutor"})
PROCESS_CONSTRUCTORS = frozenset({"Process"})
CLOCK_FUNCS = frozenset({"time", "time_ns", "perf_counter", "perf_counter_ns", "monotonic", "monotonic_ns"})
MUTABLE_CALLS = frozenset({"list", "dict", "set", "defaultdict", "bytearray", "deque", "OrderedDict", "Counter"})


def _name_of(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _contains_true_division(node: ast.AST) -> bool:
    return any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) for n in ast.walk(node))


def _is_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and not (node.value is None or isinstance(node.value, bool)
                                                   or node.value is Ellipsis)


def _module_names(tree: ast.Module) -> Tuple[Set[str], bool]:
    """Names bound at module level (descending into if / try / with bodies), and whether ``import *`` occurs."""
    names: Set[str] = set()
    star = False

    def targets(t: ast.AST) -> None:
        if isinstance(t, ast.Name):
            names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                targets(e)
        elif isinstance(t, ast.Starred):
            targets(t.value)

    def visit(stmts: Sequence[ast.stmt]) -> None:
        nonlocal star
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(s.name)
            elif isinstance(s, ast.Assign):
                for t in s.targets:
                    targets(t)
            elif isinstance(s, (ast.AnnAssign, ast.AugAssign)):
                targets(s.target)
            elif isinstance(s, ast.Import):
                for a in s.names:
                    names.add(a.asname or a.name.split(".")[0])
            elif isinstance(s, ast.ImportFrom):
                for a in s.names:
                    if a.name == "*":
                        star = True
                    else:
                        names.add(a.asname or a.name)
            elif isinstance(s, (ast.For, ast.AsyncFor)):
                targets(s.target)
                visit(s.body)
                visit(s.orelse)
            elif isinstance(s, (ast.With, ast.AsyncWith)):
                for item in s.items:
                    if item.optional_vars is not None:
                        targets(item.optional_vars)
                visit(s.body)
            elif isinstance(s, ast.If):
                visit(s.body)
                visit(s.orelse)
            elif isinstance(s, ast.Try) or type(s).__name__ == "TryStar":
                visit(s.body)  # type: ignore[attr-defined]
                for h in s.handlers:  # type: ignore[attr-defined]
                    if h.name:
                        names.add(h.name)
                    visit(h.body)
                visit(s.orelse)  # type: ignore[attr-defined]
                visit(s.finalbody)  # type: ignore[attr-defined]
    visit(tree.body)
    return names, star


_PRAGMA = re.compile(r"#\s*audit:\s*allow\s+([A-Za-z0-9_.,\s-]+)")


def _pragmas(source: str) -> Dict[int, Set[str]]:
    """``# audit: allow <rule>[, <rule>]`` comments by the 1-based line they cover.

    A trailing comment covers its own line; a comment on a line of its own
    covers the next line. Words after the rule ids (a reason) are ignored.
    """
    out: Dict[int, Set[str]] = {}
    for n, line in enumerate(source.splitlines(), start=1):
        if "audit:" not in line:
            continue
        m = _PRAGMA.search(line)
        if m:
            rules = {r for r in re.split(r"[,\s]+", m.group(1).strip()) if "." in r}
            target = n + 1 if line.lstrip().startswith("#") else n
            out.setdefault(target, set()).update(rules)
    return out


class _CodeScanner(ast.NodeVisitor):
    def __init__(self, filename: str, governed: bool, col: Collector,
                 allowed: Optional[Mapping[int, Set[str]]] = None) -> None:
        self.filename = filename
        self.governed = governed
        self.col = col
        self.allowed = dict(allowed or {})
        self.nested_funcs: List[Set[str]] = []
        self.random_aliases: Set[str] = set()
        self.random_funcs: Set[str] = set()

    def where(self, node: ast.AST) -> str:
        return f"{self.filename}:{getattr(node, 'lineno', 0)}"

    def flag(self, rule: str, node: ast.AST, message: str = "") -> None:
        line = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", None) or line
        for n in range(line, end + 1):
            if rule in self.allowed.get(n, ()):
                self.col.suppressed += 1
                return
        self.col.emit(rule, self.where(node), message)

    # -- imports -------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            if a.name == "random":
                self.random_aliases.add(a.asname or "random")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "random":
            for a in node.names:
                if a.name in RANDOM_GLOBAL_FUNCS:
                    self.random_funcs.add(a.asname or a.name)
                    self.flag(R_GLOBAL_RANDOM, node, f"imports random.{a.name} (global RNG state)")
        self.generic_visit(node)

    # -- functions -----------------------------------------------------------

    def _visit_function(self, node: Any) -> None:
        args = node.args
        for d in list(args.defaults) + [d for d in args.kw_defaults if d is not None]:
            mutable = isinstance(d, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)) or (
                isinstance(d, ast.Call) and _name_of(d.func) in MUTABLE_CALLS)
            if mutable:
                self.flag(R_MUTABLE_DEFAULT, d, f"default of {node.name}() is a mutable {type(d).__name__}")
        nested = {n.name for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n is not node}
        self.nested_funcs.append(nested)
        self._check_block(node.body)
        self.generic_visit(node)
        self.nested_funcs.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    # -- statements ----------------------------------------------------------

    def _check_block(self, stmts: Sequence[ast.stmt]) -> None:
        for i, s in enumerate(stmts[:-1]):
            if isinstance(s, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
                self.flag(R_UNREACHABLE, stmts[i + 1], f"unreachable after {type(s).__name__.lower()}")
                break

    def generic_visit(self, node: ast.AST) -> None:
        for name in ("body", "orelse", "finalbody"):
            block = getattr(node, name, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt) \
                    and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._check_block(block)
        super().generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.flag(R_BARE_EXCEPT, node, "bare except")
        broad = node.type is None or _name_of(node.type) in ("Exception", "BaseException")
        if broad and all(isinstance(s, (ast.Pass, ast.Continue)) or (
                isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and s.value.value is Ellipsis)
                for s in node.body):
            self.flag(R_SILENT_EXCEPT, node, "broad exception handler discards the error")
        self._check_block(node.body)
        super().generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Name) \
                and node.targets[0].id == node.value.id:
            self.flag(R_SELF_ASSIGN, node, f"{node.value.id} = {node.value.id}")
        self.generic_visit(node)

    # -- expressions ---------------------------------------------------------

    def visit_Dict(self, node: ast.Dict) -> None:
        seen: Set[Tuple[type, Any]] = set()
        for k in node.keys:
            if isinstance(k, ast.Constant):
                key = (type(k.value), k.value)
                if key in seen:
                    self.flag(R_DICT_KEY, k, f"duplicate key {k.value!r}")
                seen.add(key)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Is, ast.IsNot)) and (_is_literal(left) or _is_literal(right)):
                self.flag(R_LIT_IDENTITY, node, "identity comparison with a literal")
            if isinstance(op, (ast.Eq, ast.NotEq)):
                if any(isinstance(x, ast.Constant) and isinstance(x.value, float) for x in (left, right)):
                    self.flag(R_FLOAT_EQ, node, "exact comparison with a float literal")
                if any(isinstance(x, ast.Constant) and x.value is None for x in (left, right)):
                    self.flag(R_NONE_EQ, node, "compare with None using 'is'")
            left = right
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and isinstance(node.right, ast.Constant) \
                and not isinstance(node.right.value, bool) and node.right.value == 0 \
                and isinstance(node.right.value, (int, float)):
            self.flag(R_DIV_ZERO, node, "division by literal zero")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = _name_of(func)
        # int(a / b): truncation and float precision loss on wei amounts.
        if isinstance(func, ast.Name) and func.id == "int" and len(node.args) == 1 \
                and _contains_true_division(node.args[0]):
            self.flag(R_FLOAT_TRUNC, node, "int() of a true division; use integer // on wei amounts")
        # Global RNG.
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
                and func.value.id in self.random_aliases and func.attr in RANDOM_GLOBAL_FUNCS:
            self.flag(R_GLOBAL_RANDOM, node, f"random.{func.attr}() uses the global unseeded RNG")
        if isinstance(func, ast.Name) and func.id in self.random_funcs:
            self.flag(R_GLOBAL_RANDOM, node, f"{func.id}() uses the global unseeded RNG")
        # Clock-seeded RNG.
        if name in ("Random", "seed") and any(
                isinstance(n, ast.Call) and _name_of(n.func) in CLOCK_FUNCS
                for arg in list(node.args) + [k.value for k in node.keywords] for n in ast.walk(arg)):
            self.flag(R_TIME_SEED, node, "RNG seeded from the clock")
        # Worker-pool tasks must be picklable top-level functions.
        if isinstance(func, ast.Attribute) and func.attr in POOL_METHODS and node.args:
            fn = node.args[0]
            nested = self.nested_funcs[-1] if self.nested_funcs else set()
            if isinstance(fn, ast.Lambda):
                self.flag(R_UNPICKLABLE, node, f".{func.attr}() gets a lambda")
            elif isinstance(fn, ast.Name) and fn.id in nested:
                self.flag(R_UNPICKLABLE, node, f".{func.attr}() gets the nested function {fn.id}")
        # Process pools outside the governed module.
        if not self.governed:
            if name in POOL_CONSTRUCTORS:
                self.flag(R_UNGOVERNED, node, f"{name}() bypasses the CPU policy; use simulator.parallel.WorkerPool")
            elif name in PROCESS_CONSTRUCTORS and isinstance(func, ast.Attribute):
                self.flag(R_UNGOVERNED_PROC, node, f"{name}() starts a process outside simulator.parallel")
        # open() without encoding in text mode.
        if isinstance(func, ast.Name) and func.id == "open":
            mode: Any = "r"
            if len(node.args) >= 2:
                mode = node.args[1].value if isinstance(node.args[1], ast.Constant) else None
            for k in node.keywords:
                if k.arg == "mode":
                    mode = k.value.value if isinstance(k.value, ast.Constant) else None
            has_encoding = any(k.arg == "encoding" for k in node.keywords) or len(node.args) >= 4
            if isinstance(mode, str) and "b" not in mode and not has_encoding:
                self.flag(R_OPEN_ENCODING, node, "pass encoding= (the platform default differs on Windows)")
        self.generic_visit(node)


def _check_all_exports(tree: ast.Module, filename: str, col: Collector) -> None:
    declared: Optional[List[Tuple[str, ast.AST]]] = None
    for s in tree.body:
        if isinstance(s, (ast.Assign, ast.AnnAssign)):
            targets = s.targets if isinstance(s, ast.Assign) else [s.target]
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets) and \
                    isinstance(s.value, (ast.List, ast.Tuple)):
                declared = [(e.value, e) for e in s.value.elts if isinstance(e, ast.Constant)
                            and isinstance(e.value, str)]
    if declared is None:
        return
    names, star = _module_names(tree)
    counts = Counter(n for n, _ in declared)
    for name, node in declared:
        if not star:
            col.check(R_ALL, name in names, f"{filename}:{getattr(node, 'lineno', 0)}",
                      f"__all__ exports {name!r} but the module does not define it")
    for name, n in sorted(counts.items()):
        col.check(R_ALL_DUP, n == 1, filename, f"__all__ lists {name!r} {n} times")


def scan_source(source: str, filename: str = "<memory>", governed: Optional[bool] = None,
                col: Optional[Collector] = None) -> List[Finding]:
    """Scan one module's source; returns the findings (also added to ``col`` when given)."""
    own = Collector()
    try:
        tree = ast.parse(source, filename=filename)
        own.check(R_SYNTAX, True, filename)
    except SyntaxError as exc:
        own.emit(R_SYNTAX, f"{filename}:{exc.lineno or 0}", f"syntax error: {exc.msg}")
        tree = None
    if tree is not None:
        if governed is None:
            governed = Path(filename).name == GOVERNED_POOL_MODULE
        allowed = _pragmas(source)
        for line, rules in sorted(allowed.items()):
            for rule in sorted(rules):
                own.check(R_BAD_PRAGMA, rule in RULES and rule.startswith("code."), f"{filename}:{line}",
                          f"'# audit: allow {rule}' names no code.* rule")
        scanner = _CodeScanner(filename, governed, own, allowed)
        scanner.visit(tree)  # generic_visit checks the module body for unreachable statements too
        _check_all_exports(tree, filename, own)
        own.checks["code"] += sum(1 for _ in ast.walk(tree))
    if col is not None:
        col.extend(own.findings, own.checks, own.suppressed)
    return own.findings


_SCAN_CACHE: Dict[Tuple[str, int, int], Tuple[List[Finding], Dict[str, int], int]] = {}
_SCAN_LOCK = threading.Lock()


def package_sources() -> List[Path]:
    """The simulator package's own modules."""
    here = Path(__file__).resolve().parent
    return sorted(p for p in here.glob("*.py") if p.is_file())


def scan_paths(paths: Optional[Iterable[Path]], col: Collector) -> int:
    """Scan files (default: the simulator package); cached per (path, mtime, size). Returns files scanned."""
    files = package_sources() if paths is None else [Path(p) for p in paths]
    scanned = 0
    for path in files:
        try:
            st = path.stat()
        except OSError as exc:
            col.emit(R_SYNTAX, str(path), f"cannot read source: {exc}", severity=Severity.INFO)
            continue
        key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
        with _SCAN_LOCK:
            cached = _SCAN_CACHE.get(key)
        if cached is None:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                col.emit(R_SYNTAX, str(path), f"cannot read source: {exc}")
                continue
            local = Collector()
            scan_source(text, path.name, col=local)
            cached = (list(local.findings), dict(local.checks), local.suppressed)
            with _SCAN_LOCK:
                _SCAN_CACHE[key] = cached
        col.extend(cached[0], cached[1], cached[2])
        scanned += 1
    return scanned


def clear_scan_cache() -> None:
    with _SCAN_LOCK:
        _SCAN_CACHE.clear()


# ---------------------------------------------------------------------------
# The validator
# ---------------------------------------------------------------------------


class IntegrityValidator:
    """Pre-flight audit of a configuration, a market and the simulator source."""

    def __init__(self, config: Any = None, pools: Optional[Sequence[Pool]] = None, *, fail_on: Any = "error",
                 code_scan: bool = True, tick_samples: int = 256, source_paths: Optional[Iterable[Path]] = None,
                 cpus: Optional[int] = None, families: Sequence[str] = FAMILIES, seed: int = 0) -> None:
        unknown = set(families) - set(FAMILIES)
        if unknown:
            raise ConfigError(f"unknown audit families {sorted(unknown)}")
        if tick_samples < 0:
            raise ConfigError("tick_samples must be non-negative")
        self.config = config
        self.pools = list(pools) if pools is not None else None
        self.fail_on = Severity.parse(fail_on)
        self.code_scan = code_scan
        self.tick_samples = tick_samples
        self.source_paths = None if source_paths is None else [Path(p) for p in source_paths]
        self.cpus = cpus
        self.families = tuple(f for f in FAMILIES if f in families)
        self.seed = seed

    @classmethod
    def for_config(cls, config: Any, pools: Optional[Sequence[Pool]] = None, **kwargs: Any) -> "IntegrityValidator":
        """Validator driven by a config's ``audit_*`` fields (keyword arguments override them)."""
        params: Dict[str, Any] = dict(fail_on=getattr(config, "audit_fail_on", "error"),
                                      code_scan=getattr(config, "audit_code_scan", True),
                                      tick_samples=getattr(config, "audit_tick_samples", 256),
                                      seed=getattr(config, "seed", 0))
        params.update(kwargs)
        return cls(config, pools, **params)

    def run(self) -> AuditReport:
        started = time.perf_counter()
        col = Collector()
        files = 0
        pools = self.pools
        if pools is None and self.config is not None and "state" in self.families:
            from .controller import demo_market

            pools = demo_market()
        if "state" in self.families and pools is not None:
            check_pools(pools, getattr(self.config, "native_token", None), col)
        if "tick" in self.families:
            check_tick_math(col, self.tick_samples, self.seed)
            for pool in pools or ():
                if isinstance(pool, ConcentratedPool):
                    check_pool_ticks(pool, col)
        if "signal" in self.families and self.config is not None:
            check_signals(self.config, col, self.cpus)
        if "code" in self.families and self.code_scan:
            files = scan_paths(self.source_paths, col)
        return AuditReport(col.findings, dict(col.checks), time.perf_counter() - started, self.fail_on,
                           self.families, files, len(pools or ()), list(col.notes), col.suppressed)


def audit(config: Any = None, pools: Optional[Sequence[Pool]] = None, **kwargs: Any) -> AuditReport:
    """Shorthand for ``IntegrityValidator(config, pools, **kwargs).run()``."""
    return IntegrityValidator(config, pools, **kwargs).run()


def rule_catalog() -> List[Dict[str, Any]]:
    """Every rule with its family, default severity and summary (for the CLI and GUI)."""
    return [RULES[k].to_dict() for k in sorted(RULES)]


__all__ = [
    "FAMILIES",
    "RULES",
    "AuditReport",
    "Collector",
    "Finding",
    "IntegrityError",
    "IntegrityValidator",
    "Rule",
    "Severity",
    "audit",
    "check_magic_constants",
    "check_pool",
    "check_pool_ticks",
    "check_pools",
    "check_rl_signals",
    "check_signals",
    "check_swap_step",
    "check_tick_math",
    "clear_scan_cache",
    "exact_magic",
    "exact_sqrt_ratio",
    "package_sources",
    "rule_catalog",
    "sample_ticks",
    "scan_paths",
    "scan_source",
    "tick_magic_table",
]
