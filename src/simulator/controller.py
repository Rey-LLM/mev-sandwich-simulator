"""Layers 03-11: simulation controller.

``SimulationController`` is the single entry point that dispatches a run to
the mode chosen in :class:`SimulationConfig`, owns the worker pool for the
whole run, and returns a JSON-serialisable report.

Pre-flight (Layer 10): before any simulation loop runs, the controller audits
the configuration, the market and the simulator's own source with
:class:`simulator.validator.IntegrityValidator` (unless ``config.audit`` is
off) and raises :class:`~simulator.validator.IntegrityError` when a finding
reaches ``config.audit_fail_on``. Nothing is simulated after a failed audit.

Run matrices:

* Moderate Scan: read-only telemetry over the pending flow (quotes, gas-limit
  audit, dataflow and invariant fuzz), plus the multi-hop cycle scan.
* Brutal Aggressive Simulation: multi-block sandwich search, the PBS auction
  (``pbs``), the private relay (``private_relay``, when
  ``private_flow_bps > 0``) and RL policy iteration (``reinforcement``, when
  ``rl_rounds > 0``) on the capped pool (50-60% of CPUs).
* MAX Brutal Simulation (Layer 09): the same loops on a pool sized to every
  logical CPU, every fixed shard count multiplied by ``shard_density``, and a
  ``policy_paths`` section from
  :class:`simulator.path_fuzzer.PolicyPathFuzzer` (``pg_paths`` concurrent
  policy-gradient paths) and, when ``agent_bots > 0``, an ``agent_factory``
  section from :class:`simulator.ai_agent_factory.AgentFactory` (Layer 11):
  the factory ingests the RL report, its trained table (which becomes the
  incumbent policy) and the policy-path summary, then deploys
  ``agent_bots`` machine-generated competitor bots per generation on the
  same full-core worker pool. The factory needs NumPy and is imported only
  when it runs, so the other modes stay standard-library only.

Every report carries ``audit`` (the pre-flight result) and ``resources``
(worker-pool telemetry: processes, tasks, CPU seconds and utilization).
When ``config.trace`` is on, every sealed block's transaction trace is
written to ``config.trace_path`` (or stderr) for the duration of the run.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from fractions import Fraction
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from .block_builder import PBSSimulation
from .config import RunMode, SimulationConfig, SystemSpec
from .mempool_emulator import MempoolEmulator, PendingTx
from .multi_hop_router import ROUTER_CHUNKS, MultiHopRouter
from .parallel import WorkerPool
from .path_fuzzer import PolicyPathFuzzer
from .private_relay import PrivateOrderflowSimulation
from .reinforcement_engine import RL_SHARDS, ReinforcementSimulation
from .slippage_engine import ConcentratedPool, Pool, Position, V2Pool, V4Pool, price_to_sqrt_price_x96
from .validator import AuditReport, IntegrityError, IntegrityValidator

E18 = 10**18
E6 = 10**6
TRACE_LOGGER = "simulator.block_builder"


def demo_market() -> List[Pool]:
    """Three pools with realistic raw-unit sizes (18-decimal WETH/DAI, 6-decimal USDC)."""
    # token0=USDC, token1=WETH: price is WETH wei per USDC unit at $3000/WETH.
    usdc_weth = price_to_sqrt_price_x96(Fraction(E18, 3_000 * E6))
    weth_dai = price_to_sqrt_price_x96(3_000)
    return [
        V2Pool("WETH", "USDC", 4_000 * E18, 12_000_000 * E6),
        ConcentratedPool.from_positions(
            "USDC", "WETH", usdc_weth,
            [Position(-887220, 887220, 2 * 10**17), Position(193_200, 199_200, 2 * 10**18)],
            fee_pips=3000, tick_spacing=60,
        ),
        V4Pool.from_positions(
            "WETH", "DAI", weth_dai, [Position(-887270, 887270, 10**23)],
            fee_pips=500, tick_spacing=10, dynamic_fee=True, hook_fee_pips=800,
        ),
    ]


@contextmanager
def trace_logging(enabled: bool, path: str = "") -> Iterator[Optional[logging.Handler]]:
    """Send sealed-block traces (DEBUG on ``simulator.block_builder``) to ``path`` or stderr.

    The logger's level, propagation and handlers are restored on exit, so a
    traced run leaves no logging configuration behind.
    """
    if not enabled:
        yield None
        return
    logger = logging.getLogger(TRACE_LOGGER)
    handler: logging.Handler
    if path:
        handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(logging.DEBUG)
    level, propagate = logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate
        handler.close()


def invariants_ok(report: Mapping[str, Any]) -> bool:
    """True when every invariant section of a controller report passed."""
    result = report["result"]
    return bool(
        result["invariants"]["ok"]
        and result.get("pbs", {}).get("invariants_ok", True)
        and result.get("private_relay", {}).get("invariants_ok", True)
        and result.get("policy_paths", {}).get("invariants", {}).get("ok", True)
        and result.get("agent_factory", {}).get("invariants", {}).get("ok", True)
    )


class SimulationController:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.last_audit: Optional[AuditReport] = None

    def preflight(self, pools: Sequence[Pool]) -> Optional[AuditReport]:
        """Layer 10 audit of config, market and source; raises IntegrityError on a blocking finding."""
        cfg = self.config
        self.last_audit = None
        if not cfg.audit:
            return None
        report = IntegrityValidator.for_config(cfg, pools).run()
        self.last_audit = report
        if not report.ok:
            raise IntegrityError(report)
        return report

    def run(self, pools: Optional[Iterable[Pool]] = None,
            txs: Optional[Sequence[PendingTx]] = None) -> Dict[str, Any]:
        cfg = self.config
        pools = list(demo_market() if pools is None else pools)
        spec = SystemSpec.detect()
        audit = self.preflight(pools)  # before any simulation loop
        emulator = MempoolEmulator(pools, cfg)
        started = time.perf_counter()
        with trace_logging(cfg.trace, cfg.trace_path), WorkerPool.for_config(cfg, instrument=True) as wp:
            if cfg.mode is RunMode.MODERATE_SCAN:
                body = emulator.scan(txs, executor=wp).summary()
            else:
                body = emulator.run_blocks(executor=wp).summary()
                body["pbs"] = PBSSimulation(pools, cfg).run(executor=wp).summary()
                if cfg.private_flow_bps > 0:
                    body["private_relay"] = PrivateOrderflowSimulation(pools, cfg).run(executor=wp).summary()
                rl: Optional[ReinforcementSimulation] = None
                if cfg.rl_rounds > 0:
                    rl = ReinforcementSimulation(
                        seed=cfg.seed, rounds=cfg.rl_rounds, steps_per_shard=cfg.rl_steps_per_shard,
                        eval_steps=cfg.rl_eval_steps, regime=cfg.rl_regime, shards=cfg.shards(RL_SHARDS),
                    )
                    body["reinforcement"] = rl.run(executor=wp)
                if cfg.pg_paths > 0:
                    body["policy_paths"] = PolicyPathFuzzer(
                        paths=cfg.pg_paths, steps=cfg.pg_path_steps, batch=cfg.pg_batch, seed=cfg.seed,
                    ).run(executor=wp).summary()
                if cfg.agent_bots > 0:
                    body["agent_factory"] = run_agent_factory(cfg, wp, body, rl)
            router = MultiHopRouter(pools, cfg.native_token, cfg.router_max_hops)
            body["router"] = router.scan_cycles(executor=wp, gas_price_wei=cfg.gas_price_wei,
                                                chunks=cfg.shards(ROUTER_CHUNKS)).summary()
            resources = wp.telemetry()
        return {
            "mode": cfg.mode.value,
            "mode_label": cfg.mode.label,
            "config": cfg.to_dict(),
            "system": spec.to_dict(),
            "audit": {"enabled": True, **audit.to_dict()} if audit is not None else {"enabled": False},
            "resources": resources,
            "elapsed_s": round(time.perf_counter() - started, 4),
            "result": body,
        }


def run_agent_factory(cfg: SimulationConfig, executor: Optional[WorkerPool], body: Mapping[str, Any],
                      rl: Optional[ReinforcementSimulation] = None) -> Dict[str, Any]:
    """Layer 11: feed the RL results into the AI agent factory and deploy its bots on ``executor``."""
    from .ai_agent_factory import AgentFactory  # NumPy is only needed when the factory runs

    factory = AgentFactory.for_config(cfg)
    table = rl.last_train.table if rl is not None and rl.last_train is not None else None
    factory.ingest(reinforcement=body.get("reinforcement"), table=table, policy_paths=body.get("policy_paths"))
    return factory.run(executor=executor).summary()


def run(mode: Any = RunMode.MODERATE_SCAN, **overrides: Any) -> Dict[str, Any]:
    workers = overrides.pop("workers", None)
    return SimulationController(SimulationConfig.for_mode(mode, workers=workers, **overrides)).run()


__all__ = ["TRACE_LOGGER", "SimulationController", "demo_market", "invariants_ok", "run", "run_agent_factory",
           "trace_logging"]
