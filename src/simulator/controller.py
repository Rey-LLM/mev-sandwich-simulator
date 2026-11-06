"""Layers 03-06: simulation controller.

``SimulationController`` is the single entry point that dispatches a run to
the mode chosen in :class:`SimulationConfig`, owns the capped worker pool
for the whole run, and returns a JSON-serialisable report. Every run adds a
``router`` section (multi-hop cycle scan from
:class:`simulator.multi_hop_router.MultiHopRouter`); brutal runs add a ``pbs``
section from :class:`simulator.block_builder.PBSSimulation` and, when
``private_flow_bps > 0``, a ``private_relay`` section from
:class:`simulator.private_relay.PrivateOrderflowSimulation`.
"""

from __future__ import annotations

import time
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .block_builder import PBSSimulation
from .config import RunMode, SimulationConfig, SystemSpec
from .mempool_emulator import MempoolEmulator, PendingTx
from .multi_hop_router import MultiHopRouter
from .parallel import WorkerPool
from .private_relay import PrivateOrderflowSimulation
from .slippage_engine import ConcentratedPool, Pool, Position, V2Pool, V4Pool, price_to_sqrt_price_x96

E18 = 10**18
E6 = 10**6


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


class SimulationController:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def run(self, pools: Optional[Iterable[Pool]] = None,
            txs: Optional[Sequence[PendingTx]] = None) -> Dict[str, Any]:
        cfg = self.config
        pools = list(demo_market() if pools is None else pools)
        spec = SystemSpec.detect()
        emulator = MempoolEmulator(pools, cfg)
        started = time.perf_counter()
        with WorkerPool(cfg.workers, cfg.start_method, cfg.parallel_threshold) as wp:
            if cfg.mode is RunMode.MODERATE_SCAN:
                body = emulator.scan(txs, executor=wp).summary()
            else:
                body = emulator.run_blocks(executor=wp).summary()
                body["pbs"] = PBSSimulation(pools, cfg).run(executor=wp).summary()
                if cfg.private_flow_bps > 0:
                    body["private_relay"] = PrivateOrderflowSimulation(pools, cfg).run(executor=wp).summary()
            router = MultiHopRouter(pools, cfg.native_token, cfg.router_max_hops)
            body["router"] = router.scan_cycles(executor=wp, gas_price_wei=cfg.gas_price_wei).summary()
        return {
            "mode": cfg.mode.value,
            "mode_label": cfg.mode.label,
            "config": cfg.to_dict(),
            "system": spec.to_dict(),
            "elapsed_s": round(time.perf_counter() - started, 4),
            "result": body,
        }


def run(mode: Any = RunMode.MODERATE_SCAN, **overrides: Any) -> Dict[str, Any]:
    workers = overrides.pop("workers", None)
    return SimulationController(SimulationConfig.for_mode(mode, workers=workers, **overrides)).run()


__all__ = ["SimulationController", "demo_market", "run"]
