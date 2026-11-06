"""Layer 03: run modes, simulation presets and the CPU worker ceiling.

Two hardcoded execution modes drive the simulator:

* ``RunMode.MODERATE_SCAN`` - read-only telemetry. Every pending swap is
  quoted against the current pool state (slippage vector, revert check), its
  gas limit is audited against a per-hop estimate, token movements are traced
  into a passive dataflow graph and the pools get a short invariant fuzz.
  Pool state is never mutated.
* ``RunMode.BRUTAL_AGGRESSIVE`` - multi-block sandwich search. For each
  pending swap a dense grid of front-run sizes is brute-forced (then refined
  by integer ternary search), the most profitable sandwiches are executed on
  a forkable in-memory state across several blocks, and every block is
  followed by a long invariant fuzz.

Brutal mode also runs the Layer 04 PBS simulation (``block_builder``): per
slot, competing builders (vanilla, greedy, and a layout fuzzer on the worker
pool) build blocks from the same transactions and sandwich bundles, relays
re-execute the bids and the proposer takes the best one.

Layer 05 adds the private relay (``private_relay``): in brutal mode
``private_flow_bps`` of user swaps bypass the public pool and go to a
Flashbots Protect / MEV-Share style relay, whose backrun auction and hint
exposure matrix run on the same worker pool. Layer 06 (``multi_hop_router``)
scans every arbitrage cycle up to ``router_max_hops`` hops in both modes.
Layer 08 (``reinforcement_engine``) trains an online bidding agent against
competing searcher bots for ``rl_rounds`` rounds of parallel policy
iteration in brutal mode, sharded over the same capped worker pool.

Everything runs offline on the simulator's own pool objects; no RPC or chain
access is involved.

Worker ceiling
--------------
Parallel stages use ``multiprocessing`` pools sized by
:func:`resolve_worker_count`: between 50% and 60% of the CPUs this process
may run on, never more than ``floor(0.6 * cpus)`` and never fewer than one.
An explicit request (argument or ``MEV_SIM_WORKERS``) is clamped into that
band, so a run cannot take the whole machine.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Optional

CPU_FLOOR_FRACTION = 0.5
CPU_CEILING_FRACTION = 0.6
WORKERS_ENV_VAR = "MEV_SIM_WORKERS"


class ConfigError(ValueError):
    """Invalid run mode or configuration value."""


class RunMode(str, Enum):
    MODERATE_SCAN = "moderate_scan"
    BRUTAL_AGGRESSIVE = "brutal_aggressive"

    @property
    def label(self) -> str:
        return "Moderate Scan" if self is RunMode.MODERATE_SCAN else "Brutal Aggressive Simulation"

    @classmethod
    def parse(cls, value: Any) -> "RunMode":
        """Accept enum members, values, labels and short names (``moderate``, ``brutal``)."""
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
        }
        try:
            return aliases[key]
        except KeyError:
            raise ConfigError(f"unknown run mode {value!r}") from None


# ---------------------------------------------------------------------------
# Hardware detection and the worker ceiling
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


def worker_ceiling(cpus: int) -> int:
    """Hard cap: ``floor(60% of cpus)``, at least one."""
    if cpus < 1:
        raise ConfigError("cpu count must be positive")
    return max(1, math.floor(cpus * CPU_CEILING_FRACTION))


def worker_floor(cpus: int) -> int:
    """Lower edge of the band: ``ceil(50% of cpus)``, never above the ceiling."""
    if cpus < 1:
        raise ConfigError("cpu count must be positive")
    return min(worker_ceiling(cpus), max(1, math.ceil(cpus * CPU_FLOOR_FRACTION)))


def resolve_worker_count(requested: Optional[int] = None, cpus: Optional[int] = None) -> int:
    """Worker processes to use, clamped into the 50-60% CPU band.

    ``requested`` falls back to ``$MEV_SIM_WORKERS``; with neither set the
    ceiling is used.
    """
    cpus = detect_cpu_count() if cpus is None else cpus
    ceiling = worker_ceiling(cpus)
    floor = worker_floor(cpus)
    if requested is None:
        env = os.environ.get(WORKERS_ENV_VAR, "").strip()
        if env:
            try:
                requested = int(env)
            except ValueError:
                raise ConfigError(f"{WORKERS_ENV_VAR} must be an integer, got {env!r}") from None
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
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logical_cpus": self.logical_cpus,
            "usable_cpus": self.usable_cpus,
            "worker_floor": self.worker_floor,
            "worker_ceiling": self.worker_ceiling,
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
    # Sandwich search (brutal mode only).
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
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.blocks < 1 or self.txs_per_block < 0:
            raise ConfigError("blocks must be >= 1 and txs_per_block >= 0")
        if self.frontrun_grid < 0 or self.refine_iterations < 0 or self.fuzz_iterations < 0:
            raise ConfigError("grid, refine and fuzz counts must be non-negative")
        if not 0 < self.max_frontrun_bps <= 10_000:
            raise ConfigError("max_frontrun_bps must be within (0, 10000]")
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
        if self.rl_regime not in ("calm", "volatile", "extreme"):
            raise ConfigError("rl_regime must be calm, volatile or extreme")

    @property
    def gas_price_wei(self) -> int:
        return self.base_fee_wei + self.priority_fee_wei

    @classmethod
    def for_mode(cls, mode: Any, workers: Optional[int] = None, cpus: Optional[int] = None,
                 **overrides: Any) -> "SimulationConfig":
        mode = RunMode.parse(mode)
        preset = PRESETS[mode]
        cfg = cls(mode=mode, workers=resolve_worker_count(workers, cpus), **preset)
        return replace(cfg, **overrides) if overrides else cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "mode_label": self.mode.label,
            "workers": self.workers,
            "seed": self.seed,
            "blocks": self.blocks,
            "txs_per_block": self.txs_per_block,
            "frontrun_grid": self.frontrun_grid,
            "refine_iterations": self.refine_iterations,
            "max_frontrun_bps": self.max_frontrun_bps,
            "fuzz_iterations": self.fuzz_iterations,
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
        }


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
    ),
}


__all__ = [
    "CPU_FLOOR_FRACTION",
    "CPU_CEILING_FRACTION",
    "WORKERS_ENV_VAR",
    "ConfigError",
    "RunMode",
    "SystemSpec",
    "SimulationConfig",
    "PRESETS",
    "detect_cpu_count",
    "worker_ceiling",
    "worker_floor",
    "resolve_worker_count",
]
