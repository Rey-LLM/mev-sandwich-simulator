"""Layers 03 / 09 / 10 / 11 / 12: run modes, simulation presets and the CPU worker policy.

Three hardcoded execution modes (run matrices) drive the simulator:

* ``RunMode.MODERATE_SCAN`` - read-only telemetry. Every pending swap is
  quoted against the current pool state (slippage vector, revert check), its
  gas limit is audited against a per-hop estimate, token movements are traced
  into a passive dataflow graph and the pools get a short invariant fuzz.
  Pool state is never mutated.
* ``RunMode.BRUTAL_AGGRESSIVE`` - multi-block sandwich search. For each
  pending swap a dense grid of front-run sizes is brute-forced (then refined
  by integer ternary search), the most profitable sandwiches are executed on
  a forkable in-memory state across several blocks, and every block is
  followed by a long invariant fuzz. Runs on the capped worker pool.
* ``RunMode.MAX_BRUTAL`` - the brutal pipeline with the CPU cap removed and
  every workload scaled up. Worker processes scale to 100% of the logical
  CPUs this process may run on, every fixed shard / chunk count is multiplied
  by ``shard_density`` (so machines with many cores have enough independent
  tasks to stay busy), and thousands of policy-gradient paths are fuzzed
  concurrently (Layer 09, :mod:`simulator.path_fuzzer`), and the embedded AI
  agent factory deploys hundreds of machine-generated competitor bots per
  generation across every core (Layer 11, :mod:`simulator.ai_agent_factory`).

Brutal and MAX modes also run the Layer 04 PBS simulation (``block_builder``):
per slot, competing builders (vanilla, greedy, and a layout fuzzer on the
worker pool) build blocks from the same transactions and sandwich bundles,
relays re-execute the bids and the proposer takes the best one.

Layer 05 adds the private relay (``private_relay``): ``private_flow_bps`` of
user swaps bypass the public pool and go to a Flashbots Protect / MEV-Share
style relay, whose backrun auction and hint exposure matrix run on the same
worker pool. Layer 06 (``multi_hop_router``) scans every arbitrage cycle up to
``router_max_hops`` hops in every mode. Layer 08 (``reinforcement_engine``)
trains an online bidding agent against competing searcher bots for
``rl_rounds`` rounds of parallel policy iteration.

Layer 11 (``ai_agent_factory``) runs when ``agent_bots > 0``: a NumPy neural
generator conditioned on the RL results synthesises ``agent_bots`` competitor
bot profiles per generation, deploys them in heats of ``agent_heat_size`` bots
for ``agent_heat_slots`` blocks on the worker pool, and retrains itself on the
winners for ``agent_generations`` generations.

Layer 12 (``pattern_ingestor``) feeds that factory with MEV attack patterns
from a local store (``pattern_store``; segments + SQLite index, capped at
``pattern_max_storage_gb``). When the factory runs and ``pattern_seed_bps > 0``
the store is scanned in parallel shards, the ``pattern_limit`` heaviest
patterns pre-train the generator, and ``pattern_seed_bps`` of generation 0
starts from them. Without a store, ``pattern_synthetic`` generated reference
patterns (clearly marked ``source = "synthetic"``) are used instead.

Layer 14 (``custom_builder``) runs when ``custom_bots > 0``: the AI factory's
generator designs ``custom_bots`` complete bot blueprints per generation
(trigger trees, action scripts, execution pipelines), the custom builder
compiles them into runtime classes and scores them on seeded scenarios on the
worker pool, for ``custom_generations`` generations.

Layer 10 (:mod:`simulator.validator`) audits the configuration, the pools and
the simulator's own source before any simulation loop starts; the ``audit_*``
fields control it. Settings can also come from a ``sandwich.toml`` file
(:mod:`simulator.config_sync`), whose ``[modes.*]`` tables mirror
:data:`PRESETS`.

Everything runs offline on the simulator's own pool objects; no RPC or chain
access is involved.

CPU worker policy
-----------------
Parallel stages use ``multiprocessing`` pools sized by
:func:`resolve_worker_count` under a :class:`CpuPolicy`:

* ``CpuPolicy.CAPPED`` (Moderate Scan and Brutal Aggressive Simulation):
  between 50% and 60% of the CPUs this process may run on, never more than
  ``floor(0.6 * cpus)`` and never fewer than one. An explicit request
  (argument or ``MEV_SIM_WORKERS``) is clamped into that band, so a run
  cannot take the whole machine.
* ``CpuPolicy.FULL`` (MAX Brutal Simulation): up to 100% of the usable
  logical CPUs. The default is one worker process per logical CPU; an
  explicit request is clamped into ``[1, cpus]`` (oversubscribing cores does
  not speed up CPU-bound Python, it only adds context switches).

Worker processes rather than threads carry the load because the simulator is
pure-Python, CPU-bound code and threads would serialise on the GIL.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from .i18n import LANGUAGES

CPU_FLOOR_FRACTION = 0.5
CPU_CEILING_FRACTION = 0.6
FULL_CPU_FRACTION = 1.0
WORKERS_ENV_VAR = "MEV_SIM_WORKERS"
MAX_SHARD_DENSITY = 64
MAX_PG_PATHS = 1_000_000
MAX_AGENT_BOTS = 100_000
MAX_PATTERN_LIMIT = 10_000_000
MAX_PATTERN_SYNTHETIC = 1_000_000
MAX_PATTERN_STORAGE_GB = 1_000_000
MAX_AGENT_GENERATIONS = 1_000
MAX_CUSTOM_BOTS = 100_000
MAX_CUSTOM_GENERATIONS = 1_000
AUDIT_LEVELS = ("error", "warning", "info")
RL_REGIMES = ("calm", "volatile", "extreme")
START_METHODS = ("spawn", "fork", "forkserver")


class ConfigError(ValueError):
    """Invalid run mode or configuration value."""


class CpuPolicy(str, Enum):
    """How much of the machine a run's worker pool may take."""

    CAPPED = "capped"
    FULL = "full"

    @property
    def label(self) -> str:
        return "50-60% of usable CPUs" if self is CpuPolicy.CAPPED else "100% of usable CPUs"

    @property
    def ceiling_fraction(self) -> float:
        return CPU_CEILING_FRACTION if self is CpuPolicy.CAPPED else FULL_CPU_FRACTION

    @classmethod
    def parse(cls, value: Any) -> "CpuPolicy":
        """Accept enum members, values and a few spellings (``cap``, ``max``, ``100%``)."""
        if isinstance(value, cls):
            return value
        key = str(value).strip().lower().replace("-", " ").replace("_", " ")
        aliases = {
            "capped": cls.CAPPED,
            "cap": cls.CAPPED,
            "throttled": cls.CAPPED,
            "50 60%": cls.CAPPED,
            "full": cls.FULL,
            "max": cls.FULL,
            "unrestricted": cls.FULL,
            "100%": cls.FULL,
        }
        try:
            return aliases[key]
        except KeyError:
            raise ConfigError(f"unknown cpu policy {value!r} (expected capped or full)") from None


class RunMode(str, Enum):
    MODERATE_SCAN = "moderate_scan"
    BRUTAL_AGGRESSIVE = "brutal_aggressive"
    MAX_BRUTAL = "max_brutal"

    @property
    def label(self) -> str:
        return _MODE_LABELS[self]

    @property
    def short(self) -> str:
        """Name used on the command line and in ``sandwich.toml`` (``moderate``, ``brutal``, ``max``)."""
        return _MODE_SHORT[self]

    @property
    def aggressive(self) -> bool:
        """True for the modes that execute sandwiches (brutal and MAX)."""
        return self is not RunMode.MODERATE_SCAN

    @property
    def default_cpu_policy(self) -> CpuPolicy:
        return CpuPolicy.FULL if self is RunMode.MAX_BRUTAL else CpuPolicy.CAPPED

    @classmethod
    def parse(cls, value: Any) -> "RunMode":
        """Accept enum members, values, labels and short names (``moderate``, ``brutal``, ``max``)."""
        if isinstance(value, cls):
            return value
        key = str(value).strip().lower().replace("-", " ").replace("_", " ")
        aliases = {
            "moderate": cls.MODERATE_SCAN,
            "moderate scan": cls.MODERATE_SCAN,
            "scan": cls.MODERATE_SCAN,
            "brutal": cls.BRUTAL_AGGRESSIVE,
            "brutal aggressive": cls.BRUTAL_AGGRESSIVE,
            "brutal aggressive simulation": cls.BRUTAL_AGGRESSIVE,
            "aggressive": cls.BRUTAL_AGGRESSIVE,
            "max": cls.MAX_BRUTAL,
            "max brutal": cls.MAX_BRUTAL,
            "max brutal simulation": cls.MAX_BRUTAL,
            "max mode": cls.MAX_BRUTAL,
            "maximum": cls.MAX_BRUTAL,
        }
        try:
            return aliases[key]
        except KeyError:
            raise ConfigError(f"unknown run mode {value!r} (expected moderate, brutal or max)") from None


_MODE_LABELS = {
    RunMode.MODERATE_SCAN: "Moderate Scan",
    RunMode.BRUTAL_AGGRESSIVE: "Brutal Aggressive Simulation",
    RunMode.MAX_BRUTAL: "MAX Brutal Simulation",
}
_MODE_SHORT = {
    RunMode.MODERATE_SCAN: "moderate",
    RunMode.BRUTAL_AGGRESSIVE: "brutal",
    RunMode.MAX_BRUTAL: "max",
}


# ---------------------------------------------------------------------------
# Hardware detection and the worker policy
# ---------------------------------------------------------------------------


def detect_cpu_count() -> int:
    """CPUs this process may actually use (affinity-aware where supported)."""
    counter = getattr(os, "process_cpu_count", None)  # Python 3.13+
    if counter is not None:
        n = counter()
        if n:
            return n
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def worker_ceiling(cpus: int, policy: Any = CpuPolicy.CAPPED) -> int:
    """Hard cap: ``floor(60% of cpus)`` (capped) or every CPU (full), at least one."""
    if cpus < 1:
        raise ConfigError("cpu count must be positive")
    if CpuPolicy.parse(policy) is CpuPolicy.FULL:
        return cpus
    return max(1, math.floor(cpus * CPU_CEILING_FRACTION))


def worker_floor(cpus: int, policy: Any = CpuPolicy.CAPPED) -> int:
    """Lower edge of the band: ``ceil(50% of cpus)`` (capped) or one (full), never above the ceiling."""
    if cpus < 1:
        raise ConfigError("cpu count must be positive")
    if CpuPolicy.parse(policy) is CpuPolicy.FULL:
        return 1
    return min(worker_ceiling(cpus), max(1, math.ceil(cpus * CPU_FLOOR_FRACTION)))


def env_worker_request() -> Optional[int]:
    """``$MEV_SIM_WORKERS`` as an integer, or None when unset or blank."""
    env = os.environ.get(WORKERS_ENV_VAR, "").strip()
    if not env:
        return None
    try:
        return int(env)
    except ValueError:
        raise ConfigError(f"{WORKERS_ENV_VAR} must be an integer, got {env!r}") from None


def resolve_worker_count(requested: Optional[int] = None, cpus: Optional[int] = None,
                         policy: Any = CpuPolicy.CAPPED) -> int:
    """Worker processes to use under ``policy``.

    ``requested`` falls back to ``$MEV_SIM_WORKERS``; with neither set the
    policy's ceiling is used (60% of CPUs capped, every CPU full). An explicit
    request is clamped into ``[worker_floor, worker_ceiling]``.
    """
    policy = CpuPolicy.parse(policy)
    cpus = detect_cpu_count() if cpus is None else cpus
    ceiling = worker_ceiling(cpus, policy)
    floor = worker_floor(cpus, policy)
    if requested is None:
        requested = env_worker_request()
    if requested is None:
        return ceiling
    return max(floor, min(int(requested), ceiling))


@dataclass(frozen=True)
class SystemSpec:
    """Snapshot of the host taken when a run starts."""

    logical_cpus: int
    usable_cpus: int
    worker_floor: int
    worker_ceiling: int
    load_avg_1m: Optional[float]
    full_ceiling: int = 0  # worker ceiling under CpuPolicy.FULL (every usable CPU)

    @classmethod
    def detect(cls) -> "SystemSpec":
        usable = detect_cpu_count()
        try:
            load: Optional[float] = os.getloadavg()[0]
        except (AttributeError, OSError):
            load = None
        return cls(
            logical_cpus=os.cpu_count() or usable,
            usable_cpus=usable,
            worker_floor=worker_floor(usable),
            worker_ceiling=worker_ceiling(usable),
            load_avg_1m=load,
            full_ceiling=worker_ceiling(usable, CpuPolicy.FULL),
        )

    def ceiling_for(self, policy: Any) -> int:
        return self.full_ceiling if CpuPolicy.parse(policy) is CpuPolicy.FULL else self.worker_ceiling

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logical_cpus": self.logical_cpus,
            "usable_cpus": self.usable_cpus,
            "worker_floor": self.worker_floor,
            "worker_ceiling": self.worker_ceiling,
            "full_ceiling": self.full_ceiling,
            "load_avg_1m": self.load_avg_1m,
        }


# ---------------------------------------------------------------------------
# Simulation presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationConfig:
    mode: RunMode
    workers: int
    seed: int = 7
    blocks: int = 1
    txs_per_block: int = 32
    # Sandwich search (brutal and MAX modes).
    frontrun_grid: int = 0
    refine_iterations: int = 0
    max_frontrun_bps: int = 2_000  # largest front-run as bps of the pool's input-side reserve
    # Invariant fuzzing.
    fuzz_iterations: int = 128
    # Gas pricing for the audit and for sandwich net profit.
    base_fee_wei: int = 20 * 10**9
    priority_fee_wei: int = 2 * 10**9
    native_token: str = "WETH"
    default_tolerance_bps: int = 50
    # Layer 04: PBS block building.
    block_gas_limit: int = 30_000_000  # hard cap; see block_builder.BLOCK_GAS_LIMIT
    layout_fuzz_iterations: int = 0  # block layouts the fuzzer builder evaluates per slot
    builder_margin_bps: int = 500  # share of block value a builder keeps (rest is bid to the proposer)
    bribe_bps: int = 8_000  # share of a sandwich's net profit paid to the builder as coinbase transfer
    # Layer 05: private order flow (Flashbots Protect / MEV-Share relay).
    private_flow_bps: int = 0  # share of user swaps sent to the private relay instead of the public pool
    mev_share_refund_bps: int = 9_000  # share of a winning backrun payment refunded to the user
    # Layer 06: multi-hop router.
    router_max_hops: int = 3  # longest route / cycle the router enumerates
    # Layer 08: online reinforcement learning (0 rounds = off).
    rl_rounds: int = 0  # parallel policy-iteration rounds
    rl_steps_per_shard: int = 48  # environment steps each shard trains per round
    rl_eval_steps: int = 96  # greedy evaluation steps after training
    rl_regime: str = "volatile"  # calm | volatile | extreme
    # Multiprocessing.
    start_method: str = "spawn"
    parallel_threshold: int = 4  # fewer tasks than this run inline
    # Layer 09: CPU policy, parallel density and policy-gradient path fuzzing.
    cpu_policy: str = "capped"  # capped (50-60% of CPUs) | full (100%)
    shard_density: int = 1  # multiplies every fixed shard / chunk count (1 = the Layer 03-08 layout)
    pg_paths: int = 0  # policy-gradient paths fuzzed concurrently (0 = off)
    pg_path_steps: int = 16  # environment steps per path
    pg_batch: int = 32  # paths per worker task (fixed, so results never depend on the worker count)
    # Layer 11: embedded AI agent factory (0 bots = off).
    agent_bots: int = 0  # machine-generated competitor bots per generation
    agent_generations: int = 2  # synthesise -> deploy -> retrain rounds
    agent_heat_size: int = 8  # bots per arena heat (one worker task; fixed, so results never depend on workers)
    agent_heat_slots: int = 12  # blocks each heat simulates
    # Layer 12: MEV pattern ingestion (0 seed bps = off).
    pattern_store: str = ""  # pattern store directory; empty = synthetic reference patterns only
    pattern_max_storage_gb: int = 500  # storage cap of the pattern store (segments + index)
    pattern_segment_mb: int = 64  # segment file rotation size
    pattern_cache_mb: int = 256  # decoded-record LRU cache budget
    pattern_seed_bps: int = 0  # share of generation-0 AI bots seeded from stored patterns
    pattern_limit: int = 4_096  # heaviest patterns loaded into the factory
    pattern_synthetic: int = 0  # generated reference patterns used when no store is configured
    # Layer 14: custom bot builder (0 bots = off).
    custom_bots: int = 0  # AI-designed custom bot blueprints per generation
    custom_generations: int = 2  # design -> compile -> score -> retrain rounds
    # Layer 10: pre-flight integrity audit and trace output.
    audit: bool = True  # run simulator.validator before any simulation loop
    audit_fail_on: str = "error"  # lowest finding severity that blocks a run: error | warning | info
    audit_code_scan: bool = True  # include the AST scan of the simulator's own source
    audit_tick_samples: int = 256  # ticks sampled by the tick-arithmetic rounding checks
    trace: bool = False  # log every sealed block's tx trace
    trace_path: str = ""  # trace destination file; empty = stderr
    # Layer 15: language of human-facing output (sandwich.toml comments, GUI, HTML charts, logs).
    language: str = "ru"  # ru | en
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.blocks < 1 or self.txs_per_block < 0:
            raise ConfigError("blocks must be >= 1 and txs_per_block >= 0")
        if self.frontrun_grid < 0 or self.refine_iterations < 0 or self.fuzz_iterations < 0:
            raise ConfigError("grid, refine and fuzz counts must be non-negative")
        if not 0 < self.max_frontrun_bps <= 10_000:
            raise ConfigError("max_frontrun_bps must be within (0, 10000]")
        if not 0 <= self.default_tolerance_bps <= 10_000:
            raise ConfigError("default_tolerance_bps must be within [0, 10000]")
        if self.workers < 1:
            raise ConfigError("workers must be >= 1")
        if not 21_000 <= self.block_gas_limit <= 30_000_000:
            raise ConfigError("block_gas_limit must be within [21000, 30000000]")
        if self.layout_fuzz_iterations < 0:
            raise ConfigError("layout_fuzz_iterations must be non-negative")
        if not 0 <= self.builder_margin_bps <= 10_000 or not 0 <= self.bribe_bps <= 10_000:
            raise ConfigError("builder_margin_bps and bribe_bps must be within [0, 10000]")
        if not 0 <= self.private_flow_bps <= 10_000 or not 0 <= self.mev_share_refund_bps <= 10_000:
            raise ConfigError("private_flow_bps and mev_share_refund_bps must be within [0, 10000]")
        if not 1 <= self.router_max_hops <= 6:
            raise ConfigError("router_max_hops must be within [1, 6]")
        if self.rl_rounds < 0 or self.rl_steps_per_shard < 0 or self.rl_eval_steps < 0:
            raise ConfigError("rl_rounds, rl_steps_per_shard and rl_eval_steps must be non-negative")
        if self.rl_regime not in RL_REGIMES:
            raise ConfigError("rl_regime must be calm, volatile or extreme")
        if self.base_fee_wei <= 0 or self.priority_fee_wei < 0:
            raise ConfigError("base_fee_wei must be positive and priority_fee_wei non-negative")
        if self.parallel_threshold < 1:
            raise ConfigError("parallel_threshold must be >= 1")
        if self.start_method not in START_METHODS:
            raise ConfigError(f"start_method must be one of {', '.join(START_METHODS)}")
        CpuPolicy.parse(self.cpu_policy)
        if not 1 <= self.shard_density <= MAX_SHARD_DENSITY:
            raise ConfigError(f"shard_density must be within [1, {MAX_SHARD_DENSITY}]")
        if not 0 <= self.pg_paths <= MAX_PG_PATHS:
            raise ConfigError(f"pg_paths must be within [0, {MAX_PG_PATHS}]")
        if self.pg_path_steps < 1 or self.pg_batch < 1:
            raise ConfigError("pg_path_steps and pg_batch must be >= 1")
        if not 0 <= self.agent_bots <= MAX_AGENT_BOTS:
            raise ConfigError(f"agent_bots must be within [0, {MAX_AGENT_BOTS}]")
        if not 1 <= self.agent_generations <= MAX_AGENT_GENERATIONS:
            raise ConfigError(f"agent_generations must be within [1, {MAX_AGENT_GENERATIONS}]")
        if self.agent_heat_size < 1 or self.agent_heat_slots < 1:
            raise ConfigError("agent_heat_size and agent_heat_slots must be >= 1")
        if not isinstance(self.pattern_store, str):
            raise ConfigError("pattern_store must be a string")
        if not 1 <= self.pattern_max_storage_gb <= MAX_PATTERN_STORAGE_GB:
            raise ConfigError(f"pattern_max_storage_gb must be within [1, {MAX_PATTERN_STORAGE_GB}]")
        if not 1 <= self.pattern_segment_mb <= 4_096 or not 0 <= self.pattern_cache_mb <= 1_048_576:
            raise ConfigError("pattern_segment_mb must be within [1, 4096] and pattern_cache_mb within [0, 1048576]")
        if not 0 <= self.pattern_seed_bps <= 10_000:
            raise ConfigError("pattern_seed_bps must be within [0, 10000]")
        if not 0 <= self.pattern_limit <= MAX_PATTERN_LIMIT:
            raise ConfigError(f"pattern_limit must be within [0, {MAX_PATTERN_LIMIT}]")
        if not 0 <= self.pattern_synthetic <= MAX_PATTERN_SYNTHETIC:
            raise ConfigError(f"pattern_synthetic must be within [0, {MAX_PATTERN_SYNTHETIC}]")
        if not 0 <= self.custom_bots <= MAX_CUSTOM_BOTS:
            raise ConfigError(f"custom_bots must be within [0, {MAX_CUSTOM_BOTS}]")
        if not 1 <= self.custom_generations <= MAX_CUSTOM_GENERATIONS:
            raise ConfigError(f"custom_generations must be within [1, {MAX_CUSTOM_GENERATIONS}]")
        if self.audit_fail_on not in AUDIT_LEVELS:
            raise ConfigError(f"audit_fail_on must be one of {', '.join(AUDIT_LEVELS)}")
        if not 0 <= self.audit_tick_samples <= 100_000:
            raise ConfigError("audit_tick_samples must be within [0, 100000]")
        if not isinstance(self.native_token, str) or not self.native_token:
            raise ConfigError("native_token must be a non-empty string")
        if not isinstance(self.trace_path, str):
            raise ConfigError("trace_path must be a string")
        if self.language not in LANGUAGES:
            raise ConfigError(f"language must be one of {', '.join(LANGUAGES)}")

    @property
    def gas_price_wei(self) -> int:
        return self.base_fee_wei + self.priority_fee_wei

    @property
    def policy(self) -> CpuPolicy:
        return CpuPolicy.parse(self.cpu_policy)

    def shards(self, base: int) -> int:
        """A module's fixed shard / chunk count scaled by ``shard_density``."""
        return max(1, int(base)) * self.shard_density

    @classmethod
    def for_mode(cls, mode: Any, workers: Optional[int] = None, cpus: Optional[int] = None,
                 **overrides: Any) -> "SimulationConfig":
        """Preset for ``mode`` with ``overrides`` applied; workers resolved under the run's CPU policy."""
        mode = RunMode.parse(mode)
        params: Dict[str, Any] = dict(PRESETS[mode])
        params.update(overrides)
        policy = CpuPolicy.parse(params.get("cpu_policy", mode.default_cpu_policy))
        params["cpu_policy"] = policy.value
        return cls(mode=mode, workers=resolve_worker_count(workers, cpus, policy), **params)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "mode_label": self.mode.label,
            "workers": self.workers,
            "cpu_policy": self.cpu_policy,
            "start_method": self.start_method,
            "parallel_threshold": self.parallel_threshold,
            "seed": self.seed,
            "native_token": self.native_token,
            "default_tolerance_bps": self.default_tolerance_bps,
            "blocks": self.blocks,
            "txs_per_block": self.txs_per_block,
            "frontrun_grid": self.frontrun_grid,
            "refine_iterations": self.refine_iterations,
            "max_frontrun_bps": self.max_frontrun_bps,
            "fuzz_iterations": self.fuzz_iterations,
            "base_fee_wei": self.base_fee_wei,
            "priority_fee_wei": self.priority_fee_wei,
            "gas_price_wei": self.gas_price_wei,
            "block_gas_limit": self.block_gas_limit,
            "layout_fuzz_iterations": self.layout_fuzz_iterations,
            "builder_margin_bps": self.builder_margin_bps,
            "bribe_bps": self.bribe_bps,
            "private_flow_bps": self.private_flow_bps,
            "mev_share_refund_bps": self.mev_share_refund_bps,
            "router_max_hops": self.router_max_hops,
            "rl_rounds": self.rl_rounds,
            "rl_steps_per_shard": self.rl_steps_per_shard,
            "rl_eval_steps": self.rl_eval_steps,
            "rl_regime": self.rl_regime,
            "shard_density": self.shard_density,
            "pg_paths": self.pg_paths,
            "pg_path_steps": self.pg_path_steps,
            "pg_batch": self.pg_batch,
            "agent_bots": self.agent_bots,
            "agent_generations": self.agent_generations,
            "agent_heat_size": self.agent_heat_size,
            "agent_heat_slots": self.agent_heat_slots,
            "pattern_store": self.pattern_store,
            "pattern_max_storage_gb": self.pattern_max_storage_gb,
            "pattern_segment_mb": self.pattern_segment_mb,
            "pattern_cache_mb": self.pattern_cache_mb,
            "pattern_seed_bps": self.pattern_seed_bps,
            "pattern_limit": self.pattern_limit,
            "pattern_synthetic": self.pattern_synthetic,
            "custom_bots": self.custom_bots,
            "custom_generations": self.custom_generations,
            "audit": self.audit,
            "audit_fail_on": self.audit_fail_on,
            "audit_code_scan": self.audit_code_scan,
            "audit_tick_samples": self.audit_tick_samples,
            "trace": self.trace,
            "trace_path": self.trace_path,
            "language": self.language,
        }


# The three hardcoded run matrices. Every preset sets the same keys, so the
# matrix is rectangular and maps one-to-one onto the [modes.*] tables of
# sandwich.toml (see simulator.config_sync).
PRESETS: Dict[RunMode, Dict[str, Any]] = {
    RunMode.MODERATE_SCAN: dict(
        blocks=1,
        txs_per_block=32,
        frontrun_grid=0,
        refine_iterations=0,
        fuzz_iterations=128,
        layout_fuzz_iterations=0,
        private_flow_bps=0,
        router_max_hops=3,
        rl_rounds=0,
        rl_steps_per_shard=48,
        rl_eval_steps=96,
        cpu_policy="capped",
        shard_density=1,
        pg_paths=0,
        pg_path_steps=16,
        pg_batch=32,
        agent_bots=0,
        agent_generations=2,
        agent_heat_size=8,
        agent_heat_slots=12,
        pattern_seed_bps=0,
        pattern_limit=4_096,
        pattern_synthetic=0,
        custom_bots=0,
        custom_generations=2,
    ),
    RunMode.BRUTAL_AGGRESSIVE: dict(
        blocks=6,
        txs_per_block=48,
        frontrun_grid=48,
        refine_iterations=24,
        fuzz_iterations=2_000,
        layout_fuzz_iterations=2_048,
        private_flow_bps=3_000,
        router_max_hops=4,
        rl_rounds=3,
        rl_steps_per_shard=64,
        rl_eval_steps=128,
        cpu_policy="capped",
        shard_density=1,
        pg_paths=0,
        pg_path_steps=16,
        pg_batch=32,
        agent_bots=0,
        agent_generations=2,
        agent_heat_size=8,
        agent_heat_slots=12,
        pattern_seed_bps=0,
        pattern_limit=4_096,
        pattern_synthetic=0,
        custom_bots=0,
        custom_generations=2,
    ),
    RunMode.MAX_BRUTAL: dict(
        blocks=8,
        txs_per_block=64,
        frontrun_grid=64,
        refine_iterations=32,
        fuzz_iterations=8_192,
        layout_fuzz_iterations=4_096,
        private_flow_bps=3_000,
        router_max_hops=5,
        rl_rounds=4,
        rl_steps_per_shard=64,
        rl_eval_steps=256,
        cpu_policy="full",
        shard_density=16,
        pg_paths=4_096,
        pg_path_steps=16,
        pg_batch=32,
        agent_bots=512,
        agent_generations=6,
        agent_heat_size=8,
        agent_heat_slots=16,
        pattern_seed_bps=5_000,
        pattern_limit=4_096,
        pattern_synthetic=3_072,
        custom_bots=128,
        custom_generations=4,
    ),
}

# Keys of every run matrix, in display order (rows of the [modes.*] tables).
MODE_KEYS: Tuple[str, ...] = tuple(PRESETS[RunMode.MODERATE_SCAN])


__all__ = [
    "AUDIT_LEVELS",
    "CPU_FLOOR_FRACTION",
    "CPU_CEILING_FRACTION",
    "FULL_CPU_FRACTION",
    "MAX_AGENT_BOTS",
    "MAX_AGENT_GENERATIONS",
    "MAX_CUSTOM_BOTS",
    "MAX_CUSTOM_GENERATIONS",
    "MAX_PATTERN_LIMIT",
    "MAX_PATTERN_STORAGE_GB",
    "MAX_PATTERN_SYNTHETIC",
    "MAX_PG_PATHS",
    "MAX_SHARD_DENSITY",
    "MODE_KEYS",
    "RL_REGIMES",
    "START_METHODS",
    "WORKERS_ENV_VAR",
    "ConfigError",
    "CpuPolicy",
    "RunMode",
    "SystemSpec",
    "SimulationConfig",
    "PRESETS",
    "detect_cpu_count",
    "env_worker_request",
    "worker_ceiling",
    "worker_floor",
    "resolve_worker_count",
]
