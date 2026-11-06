"""Layer 03: mempool emulator with parallel scan and sandwich search.

The emulator generates deterministic pending swaps against the simulator's
pools and processes them in one of two ways:

* :meth:`MempoolEmulator.scan` (Moderate Scan) quotes every pending swap
  against the current state in parallel, audits its gas limit and projects its
  token flows into a passive :class:`TokenFlowGraph`. Nothing is mutated.
* :meth:`MempoolEmulator.run_blocks` (Brutal Aggressive Simulation) searches
  the best front-run size for every pending swap in parallel, then builds
  blocks on a :class:`StateFork`: each sandwich is re-simulated inside a
  checkpoint and kept only if it still pays, otherwise rolled back. State
  carries over block to block, and every block ends with an invariant fuzz.

Parallel stages use one capped :class:`WorkerPool` per run. Task layout never
depends on the worker count, so parallel and serial runs give identical
results.

Gas figures are heuristic per-hop estimates for auditing, not EVM-measured.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import SimulationConfig
from .dataflow import TokenFlowGraph
from .invariants import InvariantReport, InvariantViolation, input_reserve_scale, parallel_fuzz
from .parallel import WorkerPool, chunked
from .slippage_engine import (
    ConcentratedPool,
    MASEEngine,
    MASEError,
    Pool,
    SwapResult,
    V2Pool,
    _direction,
    evaluate_trade,
    min_amount_out,
)
from .state_fork import StateFork, SwapReverted

GAS_TX_BASE = 21_000
GAS_PER_HOP = {"v2": 60_000, "v3": 80_000, "v4": 70_000}
GAS_PER_TICK_CROSS = 20_000
SEARCHER = "searcher"
TASK_CHUNKS = 16  # fixed so results never depend on worker count


# ---------------------------------------------------------------------------
# Pending transactions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PendingTx:
    tx_hash: str
    sender: str
    payload: Dict[str, Any]
    gas_limit: int
    priority_fee_wei: int
    nonce: int = 0

    @property
    def route(self) -> Tuple[str, ...]:
        r = self.payload.get("route") or [self.payload["pool_id"]]
        return tuple([r] if isinstance(r, str) else r)

    @property
    def token_in(self) -> str:
        return str(self.payload["token_in"])

    @property
    def amount_in(self) -> int:
        return int(self.payload["amount_in"])

    @property
    def amount_out_min(self) -> int:
        return int(self.payload.get("amount_out_min") or 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tx_hash": self.tx_hash,
            "sender": self.sender,
            "payload": dict(self.payload),
            "gas_limit": self.gas_limit,
            "priority_fee_wei": self.priority_fee_wei,
            "nonce": self.nonce,
        }


def pool_version(pool: Pool) -> str:
    return "v2" if isinstance(pool, V2Pool) else pool.version


def estimate_swap_gas(pools: Sequence[Pool], ticks_crossed: int = 0) -> int:
    return GAS_TX_BASE + sum(GAS_PER_HOP.get(pool_version(p), 80_000) for p in pools) \
        + GAS_PER_TICK_CROSS * ticks_crossed


def block_order(txs: Iterable[PendingTx]) -> List[PendingTx]:
    """Highest priority fee first; hash breaks ties deterministically."""
    return sorted(txs, key=lambda t: (-t.priority_fee_wei, t.tx_hash))


def generate_mempool(pools: Iterable[Pool], count: int, seed: int, block: int = 0) -> List[PendingTx]:
    """Deterministic single-hop swaps sized 0.01%-3% of the input-side reserve.

    ``amount_out_min`` is set from a quote on the given state at a random
    slippage tolerance (10-1000 bps), the way a wallet would. Gas limits are
    drawn around the estimate, so some are deliberately under-provisioned.
    """
    pools = sorted(pools, key=lambda p: p.pool_id)
    if not pools:
        return []
    rng = random.Random(f"mempool:{seed}:{block}")
    txs: List[PendingTx] = []
    for i in range(count):
        pool = pools[rng.randrange(len(pools))]
        token_in = pool.tokens()[rng.randrange(2)]
        zero_for_one, _ = _direction(pool, token_in)
        size_bps = 10 ** rng.uniform(0, math.log10(300))
        amount = max(1, int(input_reserve_scale(pool, zero_for_one) * size_bps / 10_000))
        tolerance = rng.choice((10, 30, 50, 100, 300, 500, 1000))
        try:
            quote = evaluate_trade(pool, token_in, amount, tolerance)
        except MASEError:
            continue
        gas_est = estimate_swap_gas([pool], quote.ticks_crossed)
        gas_limit = int(gas_est * rng.choice((0.8, 1.1, 1.25, 1.5)))
        txs.append(PendingTx(
            tx_hash=f"0x{block:06x}{i:06x}{rng.getrandbits(96):024x}",
            sender=f"trader{rng.randrange(1_000):03d}",
            payload={
                "route": [pool.pool_id],
                "token_in": token_in,
                "amount_in": amount,
                "amount_out_min": min_amount_out(quote.amount_out, tolerance),
                "tolerance_bps": tolerance,
            },
            gas_limit=gas_limit,
            priority_fee_wei=rng.randrange(1, 6) * 10**9,
            nonce=i,
        ))
    return txs


# ---------------------------------------------------------------------------
# Moderate Scan: telemetry + gas audit + passive dataflow
# ---------------------------------------------------------------------------


def _scan_chunk(task: Tuple[List[Pool], int, List[PendingTx]]) -> List[Dict[str, Any]]:
    pools, tolerance, txs = task
    engine = MASEEngine(pools, tolerance)
    rows: List[Dict[str, Any]] = []
    for tx in txs:
        row: Dict[str, Any] = {"tx_hash": tx.tx_hash, "sender": tx.sender, "gas_limit": tx.gas_limit}
        try:
            ev = engine.evaluate_payload(tx.payload)
        except MASEError as exc:
            row.update(error=str(exc), gas_ok=None)
            rows.append(row)
            continue
        route_pools = [engine.get_pool(pid) for pid in tx.route]
        gas_est = estimate_swap_gas(route_pools, ev["ticks_crossed"])
        row.update(ev)
        row.update(
            gas_estimate=gas_est,
            gas_ok=tx.gas_limit >= gas_est,
            gas_headroom_bps=(tx.gas_limit - gas_est) * 10_000 // gas_est,
        )
        rows.append(row)
    return rows


@dataclass
class ScanReport:
    rows: List[Dict[str, Any]]
    flow: TokenFlowGraph
    invariants: InvariantReport

    def summary(self) -> Dict[str, Any]:
        ok_rows = [r for r in self.rows if "error" not in r]
        impacts = [r["price_impact_bps"] for r in ok_rows]
        return {
            "txs": len(self.rows),
            "quoted": len(ok_rows),
            "errors": len(self.rows) - len(ok_rows),
            "would_revert": sum(1 for r in ok_rows if r["would_revert"]),
            "gas_underprovisioned": sum(1 for r in ok_rows if r["gas_ok"] is False),
            "mean_price_impact_bps": sum(impacts) / len(impacts) if impacts else 0.0,
            "max_price_impact_bps": max(impacts) if impacts else 0.0,
            "total_slack_out": sum(r["slack_out"] or 0 for r in ok_rows),
            "flow_transfers": len(self.flow),
            "invariants": self.invariants.to_dict(),
        }


# ---------------------------------------------------------------------------
# Brutal Aggressive Simulation: sandwich search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SandwichPlan:
    tx_hash: str
    pool_id: str
    token_in: str
    token_out: str
    frontrun_in: int
    frontrun_out: int
    victim_out: int
    backrun_out: int
    candidates: int

    @property
    def profit(self) -> int:
        """Gross profit in ``token_in`` (before gas)."""
        return self.backrun_out - self.frontrun_in

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tx_hash": self.tx_hash,
            "pool_id": self.pool_id,
            "token_in": self.token_in,
            "token_out": self.token_out,
            "frontrun_in": self.frontrun_in,
            "frontrun_out": self.frontrun_out,
            "victim_out": self.victim_out,
            "backrun_out": self.backrun_out,
            "profit": self.profit,
            "candidates": self.candidates,
        }


def _apply_full(pool: Pool, amount_in: int, zero_for_one: bool) -> Optional[SwapResult]:
    """Exact-input swap on ``pool``; None (pool unchanged) unless fully filled with output."""
    try:
        res = pool.quote_exact_in(amount_in, zero_for_one)
    except MASEError:
        return None
    if not res.fully_filled or res.amount_out <= 0:
        return None
    if isinstance(pool, V2Pool):
        pool.apply_swap(res, zero_for_one)
    else:
        assert isinstance(pool, ConcentratedPool)
        pool.apply_swap(amount_in, zero_for_one)
    return res


def simulate_sandwich(pool: Pool, victim: PendingTx, frontrun_in: int) -> Optional[Tuple[int, int, int]]:
    """(front_out, victim_out, back_out) on a scratch copy, or None if any leg fails."""
    work = copy.deepcopy(pool)
    zero_for_one, _ = _direction(work, victim.token_in)
    front = _apply_full(work, frontrun_in, zero_for_one)
    if front is None:
        return None
    victim_res = _apply_full(work, victim.amount_in, zero_for_one)
    if victim_res is None or victim_res.amount_out < victim.amount_out_min:
        return None
    back = _apply_full(work, front.amount_out, not zero_for_one)
    if back is None:
        return None
    return front.amount_out, victim_res.amount_out, back.amount_out


def geometric_grid(lo: int, hi: int, points: int) -> List[int]:
    if points <= 0 or hi < lo:
        return []
    if points == 1 or hi == lo:
        return [hi]
    ratio = (hi / lo) ** (1 / (points - 1))
    return sorted({min(hi, max(lo, int(round(lo * ratio ** i)))) for i in range(points)})


def search_sandwich(pool: Pool, victim: PendingTx, grid: int, refine: int,
                    max_frontrun_bps: int) -> Optional[SandwichPlan]:
    """Brute-force the front-run size, then refine around the best grid point.

    Profit rises with front-run size until the victim's ``amount_out_min``
    makes the victim revert, so the grid finds the region and an integer
    ternary search between the best point's neighbours tightens it.
    """
    if len(victim.route) != 1 or grid <= 0:
        return None
    try:
        zero_for_one, token_out = _direction(pool, victim.token_in)
    except MASEError:
        return None
    hi = input_reserve_scale(pool, zero_for_one) * max_frontrun_bps // 10_000
    lo = max(1, victim.amount_in // 1_000)
    sizes = geometric_grid(lo, hi, grid)
    if not sizes:
        return None

    cache: Dict[int, Optional[Tuple[int, int, int]]] = {}

    def profit(a: int) -> Optional[int]:
        if a not in cache:
            cache[a] = simulate_sandwich(pool, victim, a)
        r = cache[a]
        return None if r is None else r[2] - a

    scores = [profit(a) for a in sizes]
    best_i = max(range(len(sizes)), key=lambda i: (scores[i] is not None, scores[i] or 0, -i))
    if scores[best_i] is None:
        return None
    left = sizes[best_i - 1] if best_i > 0 else lo
    right = sizes[best_i + 1] if best_i + 1 < len(sizes) else hi
    for _ in range(refine):
        if right - left < 3:
            break
        m1 = left + (right - left) // 3
        m2 = right - (right - left) // 3
        p1, p2 = profit(m1), profit(m2)
        if (p1 if p1 is not None else -math.inf) < (p2 if p2 is not None else -math.inf):
            left = m1
        else:
            right = m2
    feasible = [(r[2] - a, -a) for a, r in cache.items() if r is not None]
    best_profit, neg_a = max(feasible)
    if best_profit <= 0:
        return None
    best_a = -neg_a
    front_out, victim_out, back_out = cache[best_a]  # type: ignore[misc]
    return SandwichPlan(victim.tx_hash, pool.pool_id, victim.token_in, token_out, best_a,
                        front_out, victim_out, back_out, len(cache))


def _search_chunk(task: Tuple[Dict[str, Pool], List[PendingTx], int, int, int]) -> List[Optional[SandwichPlan]]:
    pools, txs, grid, refine, max_bps = task
    out: List[Optional[SandwichPlan]] = []
    for tx in txs:
        pool = pools.get(tx.route[0])
        out.append(None if pool is None else search_sandwich(pool, tx, grid, refine, max_bps))
    return out


@dataclass
class BlockReport:
    block_number: int
    txs: int
    sandwiches: List[Dict[str, Any]] = field(default_factory=list)
    victims_reverted: int = 0
    plans_found: int = 0
    plans_rolled_back: int = 0
    gas_spent_wei: int = 0
    state_root: str = ""

    def to_dict(self) -> Dict[str, Any]:
        extracted: Dict[str, int] = {}
        for s in self.sandwiches:
            extracted[s["token_in"]] = extracted.get(s["token_in"], 0) + s["profit"]
        return {
            "block_number": self.block_number,
            "txs": self.txs,
            "plans_found": self.plans_found,
            "sandwiches_executed": len(self.sandwiches),
            "plans_rolled_back": self.plans_rolled_back,
            "victims_reverted": self.victims_reverted,
            "extracted": extracted,
            "gas_spent_wei": self.gas_spent_wei,
            "state_root": self.state_root,
        }


@dataclass
class AggressiveReport:
    blocks: List[BlockReport]
    flow: TokenFlowGraph
    invariants: InvariantReport
    state: StateFork

    def summary(self) -> Dict[str, Any]:
        extracted: Dict[str, int] = {}
        for b in self.blocks:
            for tok, amt in b.to_dict()["extracted"].items():
                extracted[tok] = extracted.get(tok, 0) + amt
        searcher_flow = self.flow.extraction(SEARCHER)
        searcher_state = {t: v for t, v in self.state.balances.get(SEARCHER, {}).items() if v}
        return {
            "blocks": [b.to_dict() for b in self.blocks],
            "sandwiches_executed": sum(len(b.sandwiches) for b in self.blocks),
            "extracted": extracted,
            "gas_spent_wei": sum(b.gas_spent_wei for b in self.blocks),
            "searcher_balance": searcher_state,
            "dataflow_consistent": searcher_flow == searcher_state,
            "flow_transfers": len(self.flow),
            "invariants": self.invariants.to_dict(),
            "final_state_root": self.state.state_root(),
        }


# ---------------------------------------------------------------------------
# Emulator
# ---------------------------------------------------------------------------


class MempoolEmulator:
    def __init__(self, pools: Iterable[Pool], config: SimulationConfig) -> None:
        self.pools: List[Pool] = list(pools)
        self.config = config

    def _executor(self) -> WorkerPool:
        cfg = self.config
        return WorkerPool(cfg.workers, cfg.start_method, cfg.parallel_threshold)

    def generate(self, count: Optional[int] = None, block: int = 0,
                 pools: Optional[Iterable[Pool]] = None) -> List[PendingTx]:
        n = self.config.txs_per_block if count is None else count
        return generate_mempool(self.pools if pools is None else pools, n, self.config.seed, block)

    # -- Moderate Scan -------------------------------------------------------

    def scan(self, txs: Optional[Sequence[PendingTx]] = None,
             executor: Optional[WorkerPool] = None) -> ScanReport:
        cfg = self.config
        txs = list(self.generate() if txs is None else txs)
        own = executor is None
        wp = self._executor() if own else executor
        try:
            tasks = [(self.pools, cfg.default_tolerance_bps, chunk) for chunk in chunked(txs, TASK_CHUNKS)]
            rows = [row for part in wp.map(_scan_chunk, tasks) for row in part]
            fuzz = parallel_fuzz(self.pools, cfg.fuzz_iterations, cfg.seed, executor=wp)
        finally:
            if own:
                wp.close()
        flow = TokenFlowGraph()
        for row in rows:
            if "error" in row or row["would_revert"]:
                continue
            route = "->".join(row["pool_ids"])
            flow.record_swap(row["tx_hash"], row["sender"], route, row["token_in"], row["amount_in"],
                             row["token_out"], row["amount_out"], label="projected")
        return ScanReport(rows, flow, fuzz)

    # -- Brutal Aggressive Simulation ------------------------------------------

    def search(self, pools: Dict[str, Pool], txs: Sequence[PendingTx],
               executor: WorkerPool) -> Dict[str, SandwichPlan]:
        cfg = self.config
        tasks = [(pools, chunk, cfg.frontrun_grid, cfg.refine_iterations, cfg.max_frontrun_bps)
                 for chunk in chunked(list(txs), TASK_CHUNKS)]
        plans: Dict[str, SandwichPlan] = {}
        for part in executor.map(_search_chunk, tasks):
            for plan in part:
                if plan is not None:
                    plans[plan.tx_hash] = plan
        return plans

    def run_blocks(self, executor: Optional[WorkerPool] = None) -> AggressiveReport:
        cfg = self.config
        state = StateFork(copy.deepcopy(self.pools))  # never mutate the caller's pools
        flow = TokenFlowGraph()
        invariants = InvariantReport()
        blocks: List[BlockReport] = []
        own = executor is None
        wp = self._executor() if own else executor
        try:
            for _ in range(cfg.blocks):
                txs = generate_mempool(state.pools.values(), cfg.txs_per_block, cfg.seed, state.block_number)
                plans = self.search(state.pools, txs, wp)
                report = BlockReport(state.block_number, len(txs), plans_found=len(plans))
                k_before = {pid: p.k for pid, p in state.pools.items() if isinstance(p, V2Pool)}
                for tx in block_order(txs):
                    plan = plans.get(tx.tx_hash)
                    if plan is not None and self._execute_sandwich(state, tx, plan, flow, report):
                        continue
                    try:
                        state.swap(tx.sender, tx.route[0], tx.token_in, tx.amount_in, tx.amount_out_min,
                                   tx=tx.tx_hash, flow=flow, label="victim")
                    except SwapReverted:
                        report.victims_reverted += 1
                for pid, k in k_before.items():
                    invariants.checks += 1
                    pool = state.pools[pid]
                    assert isinstance(pool, V2Pool)
                    if pool.k < k:
                        invariants.violations.append(InvariantViolation(
                            "block.v2_k_non_decreasing", pid, f"k {k} -> {pool.k}", (state.block_number,)))
                per_block = cfg.fuzz_iterations // cfg.blocks
                invariants.merge(parallel_fuzz(state.pools.values(), per_block,
                                               cfg.seed + state.block_number, executor=wp))
                report.state_root = state.state_root()
                blocks.append(report)
                state.mine()
        finally:
            if own:
                wp.close()
        return AggressiveReport(blocks, flow, invariants, state)

    def _execute_sandwich(self, state: StateFork, tx: PendingTx, plan: SandwichPlan,
                          flow: TokenFlowGraph, report: BlockReport) -> bool:
        """Front-run, victim, back-run inside a checkpoint; keep only if still profitable."""
        cfg = self.config
        legs = TokenFlowGraph()
        with state.checkpoint():
            try:
                front = state.swap(SEARCHER, plan.pool_id, plan.token_in, plan.frontrun_in,
                                   tx=f"{tx.tx_hash}:front", flow=legs, label="frontrun")
                state.swap(tx.sender, plan.pool_id, tx.token_in, tx.amount_in, tx.amount_out_min,
                           tx=tx.tx_hash, flow=legs, label="victim")
                back = state.swap(SEARCHER, plan.pool_id, plan.token_out, front.amount_out,
                                  tx=f"{tx.tx_hash}:back", flow=legs, label="backrun")
            except SwapReverted:
                report.plans_rolled_back += 1
                return False
            profit = back.amount_out - plan.frontrun_in
            if profit <= 0:
                report.plans_rolled_back += 1
                return False
            state.commit_checkpoint()
        pool = state.pools[plan.pool_id]
        gas = 2 * estimate_swap_gas([pool]) * cfg.gas_price_wei
        report.gas_spent_wei += gas
        trace = legs.paths(tx.sender, SEARCHER)
        record = plan.to_dict()
        record.update(
            frontrun_out=front.amount_out,
            backrun_out=back.amount_out,
            profit=profit,
            gas_cost_wei=gas,
            net_profit_native=profit - gas if plan.token_in == cfg.native_token else None,
            victim_to_searcher_path=[f"{t.src}->{t.dst}:{t.token}" for t in trace[0]] if trace else [],
        )
        report.sandwiches.append(record)
        flow.extend(legs)
        return True


__all__ = [
    "GAS_TX_BASE",
    "GAS_PER_HOP",
    "GAS_PER_TICK_CROSS",
    "SEARCHER",
    "PendingTx",
    "ScanReport",
    "SandwichPlan",
    "BlockReport",
    "AggressiveReport",
    "MempoolEmulator",
    "estimate_swap_gas",
    "block_order",
    "generate_mempool",
    "simulate_sandwich",
    "search_sandwich",
    "geometric_grid",
]
