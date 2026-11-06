"""Layer 06: graph-based multi-hop cross-pool router.

The router treats every pool as a pair of directed edges in a token graph and
executes routes hop by hop on integer-exact pool math from
:mod:`simulator.slippage_engine`:

* Uniswap v2 hops move reserves along ``x * y = k`` (0.3% input fee).
* Uniswap v3 / v4 hops walk the tick array and update ``sqrtPriceX96``,
  ``tick`` and active liquidity (v4 adds hook / protocol fees).

A route such as ``WETH -(v2)-> USDC -(v3)-> DAI -(v4)-> WETH`` runs in one
sequential pass: the exact output of each hop is the input of the next, and
every hop sees the state left by the hops before it. A route may visit the
same pool twice; the second visit then trades against the shifted state
(``execute_path`` clones the pools it touches, so callers' pools only change
with ``mutate=True``).

Slippage degradation
--------------------
For an exact-input route with per-hop spot prices ``s_i`` (output per input,
fee-free) and per-hop execution prices ``e_i = out_i / in_i``:

* per-hop impact ``d_i = 1 - e_i / s_i`` (fee plus curve movement),
* combined impact ``D = 1 - out / (in * prod(s_i)) = 1 - prod(1 - d_i)``
  (exact, because ``in_{i+1} = out_i``),
* the additive estimate ``sum(d_i)`` always overstates ``D``; the gap is the
  compounding interaction term,
* fee drag ``1 - prod(1 - f_i)`` and curve drag ``1 - prod(1 - c_i)`` with
  ``c_i`` the fee-free impact of hop ``i``.

:class:`SlippageDegradation` carries all of these as exact ``Fraction`` values
plus a cumulative per-hop vector.

Arbitrage
---------
* :meth:`TokenGraph.enumerate_cycles` lists closed routes
  ``A -> ... -> A`` up to ``max_hops`` without reusing a pool.
* :meth:`TokenGraph.negative_cycles` runs Bellman-Ford on
  ``-log(spot * (1 - fee))`` edge weights as an independent cross-check: a
  negative cycle is a loop whose marginal rate product exceeds 1.
* :func:`optimize_cycle_input` searches the profit-maximising input with a
  geometric grid and an integer ternary refine (profit is unimodal in the
  input size for AMM chains).
* :meth:`MultiHopRouter.scan_cycles` evaluates every cycle in a fixed number
  of chunks on the capped :class:`~simulator.parallel.WorkerPool`, so results
  do not depend on the worker count.

Order splitting
---------------
:func:`split_route` water-fills an order across several paths in equal
increments, always sending the next increment where it buys the most output
given the state the earlier increments left behind (paths that share a pool
see each other's impact). It never returns less than the best single path.

Limits: spot prices for ranking come from the current marginal price and fee;
gas is the per-hop estimate from :mod:`simulator.mempool_emulator`, not
EVM-metered.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .block_builder import GWEI, BuilderTx, TxKind, clone_pool, native_value
from .invariants import input_reserve_scale
from .mempool_emulator import estimate_swap_gas, geometric_grid, pool_version
from .parallel import WorkerPool, chunked
from .slippage_engine import (
    BPS_DENOMINATOR,
    FEE_PIPS_DENOMINATOR,
    ConcentratedPool,
    MASEError,
    Pool,
    SwapResult,
    V2Pool,
    _direction,
    get_tick_at_sqrt_ratio,
    MAX_SQRT_RATIO,
    MIN_SQRT_RATIO,
)

ROUTER_CHUNKS = 16  # fixed so scan results never depend on worker count
DEFAULT_MAX_HOPS = 3
MAX_SUPPORTED_HOPS = 6


class RouterError(MASEError):
    """Route cannot be built or executed."""


# ---------------------------------------------------------------------------
# Pool helpers
# ---------------------------------------------------------------------------


def v2_implied_sqrt_price_x96(pool: V2Pool) -> int:
    """sqrt(reserve1 / reserve0) in Q64.96, so v2 hops report the same price field as v3/v4."""
    return isqrt((pool.reserve1 << 192) // pool.reserve0)


def pool_sqrt_price_x96(pool: Pool) -> int:
    if isinstance(pool, V2Pool):
        return v2_implied_sqrt_price_x96(pool)
    return pool.sqrt_price_x96


def pool_tick(pool: Pool) -> Optional[int]:
    """Current tick (v3/v4) or the tick implied by v2 reserves when it is in range."""
    if isinstance(pool, ConcentratedPool):
        return pool.tick
    sp = v2_implied_sqrt_price_x96(pool)
    if MIN_SQRT_RATIO <= sp < MAX_SQRT_RATIO:
        return get_tick_at_sqrt_ratio(sp)
    return None


def pool_liquidity(pool: Pool) -> int:
    """Active liquidity (v3/v4) or sqrt(k) for v2."""
    if isinstance(pool, V2Pool):
        return isqrt(pool.k)
    return pool.liquidity


def pool_fee_fraction(pool: Pool) -> Fraction:
    if isinstance(pool, V2Pool):
        return Fraction(pool.fee_bps, BPS_DENOMINATOR)
    return Fraction(pool.effective_fee(), FEE_PIPS_DENOMINATOR)


def _reserves(pool: Pool) -> Optional[Tuple[int, int]]:
    return (pool.reserve0, pool.reserve1) if isinstance(pool, V2Pool) else None


def _quote(pool: Pool, amount_in: int, zero_for_one: bool) -> SwapResult:
    return pool.quote_exact_in(amount_in, zero_for_one)


def _apply(pool: Pool, res: SwapResult, amount_in: int, zero_for_one: bool) -> None:
    if isinstance(pool, V2Pool):
        pool.apply_swap(res, zero_for_one)
    else:
        pool.apply_swap(amount_in, zero_for_one)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    pool_id: str
    token_in: str
    token_out: str
    version: str
    zero_for_one: bool
    spot: Fraction  # marginal output per input, fee-free
    fee: Fraction

    @property
    def rate(self) -> Fraction:
        """Marginal output per input after the fee."""
        return self.spot * (1 - self.fee)

    @property
    def log_weight(self) -> float:
        r = self.rate
        if r <= 0:
            return math.inf
        return -(math.log(r.numerator) - math.log(r.denominator))


@dataclass(frozen=True)
class RoutePath:
    """Token sequence and the pool used for each hop (``len(tokens) == hops + 1``)."""

    tokens: Tuple[str, ...]
    pool_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.pool_ids) < 1:
            raise RouterError("route needs at least one hop")
        if len(self.tokens) != len(self.pool_ids) + 1:
            raise RouterError("route needs exactly one more token than pools")
        for a, b in zip(self.tokens, self.tokens[1:]):
            if a == b:
                raise RouterError(f"hop swaps {a} for itself")

    @property
    def hops(self) -> int:
        return len(self.pool_ids)

    @property
    def token_in(self) -> str:
        return self.tokens[0]

    @property
    def token_out(self) -> str:
        return self.tokens[-1]

    @property
    def is_cycle(self) -> bool:
        return self.tokens[0] == self.tokens[-1]

    @property
    def reuses_pool(self) -> bool:
        return len(set(self.pool_ids)) != len(self.pool_ids)

    @property
    def key(self) -> str:
        parts = [self.tokens[0]]
        for pid, tok in zip(self.pool_ids, self.tokens[1:]):
            parts.append(f"-[{pid}]->{tok}")
        return "".join(parts)

    def reversed(self) -> "RoutePath":
        return RoutePath(tuple(reversed(self.tokens)), tuple(reversed(self.pool_ids)))

    def rotate(self, start: int) -> "RoutePath":
        """Same cycle starting at hop ``start`` (profit is then taken in a different token)."""
        if not self.is_cycle:
            raise RouterError("only cycles can be rotated")
        n = self.hops
        start %= n
        pools = self.pool_ids[start:] + self.pool_ids[:start]
        toks = self.tokens[start:-1] + self.tokens[:start] + (self.tokens[start],)
        return RoutePath(toks, pools)

    def canonical_cycle_key(self) -> Tuple[str, ...]:
        """Rotation-independent identity of a cycle (direction still matters)."""
        if not self.is_cycle:
            return self.pool_ids
        n = self.hops
        seqs = [tuple(zip(self.pool_ids[i:] + self.pool_ids[:i], self.tokens[i:-1] + self.tokens[:i]))
                for i in range(n)]
        best = min(seqs)
        return tuple(f"{p}|{t}" for p, t in best)

    def to_dict(self) -> Dict[str, Any]:
        return {"tokens": list(self.tokens), "pool_ids": list(self.pool_ids), "hops": self.hops,
                "is_cycle": self.is_cycle, "key": self.key}

    @classmethod
    def from_route(cls, pools: Mapping[str, Pool], token_in: str, pool_ids: Sequence[str]) -> "RoutePath":
        """Build the token sequence for a pool-id route starting at ``token_in``."""
        tokens = [token_in]
        for pid in pool_ids:
            pool = pools.get(pid)
            if pool is None:
                raise RouterError(f"unknown pool {pid!r}")
            _, out = _direction(pool, tokens[-1])
            tokens.append(out)
        return cls(tuple(tokens), tuple(pool_ids))


def validate_path(pools: Mapping[str, Pool], path: RoutePath) -> None:
    for pid, a, b in zip(path.pool_ids, path.tokens, path.tokens[1:]):
        pool = pools.get(pid)
        if pool is None:
            raise RouterError(f"unknown pool {pid!r}")
        if set(pool.tokens()) != {a, b}:
            raise RouterError(f"pool {pid} does not trade {a}->{b}")


class TokenGraph:
    """Directed token graph: one edge per pool and direction, sorted for determinism."""

    def __init__(self, pools: Iterable[Pool]) -> None:
        self.pools: Dict[str, Pool] = {}
        for p in pools:
            if p.pool_id in self.pools:
                raise RouterError(f"duplicate pool id {p.pool_id}")
            self.pools[p.pool_id] = p
        self.adjacency: Dict[str, List[Edge]] = {}
        self.refresh()

    def refresh(self) -> None:
        """Recompute edge spot prices and fees from the pools' current state."""
        adj: Dict[str, List[Edge]] = {}
        for pid in sorted(self.pools):
            pool = self.pools[pid]
            fee = pool_fee_fraction(pool)
            t0, t1 = pool.tokens()
            for tin, tout, zfo in ((t0, t1, True), (t1, t0, False)):
                adj.setdefault(tin, []).append(Edge(pid, tin, tout, pool_version(pool), zfo, pool.spot_price(zfo), fee))
        for tok in adj:
            adj[tok].sort(key=lambda e: (e.token_out, e.pool_id))
        self.adjacency = adj

    @property
    def tokens(self) -> List[str]:
        toks: Set[str] = set()
        for p in self.pools.values():
            toks.update(p.tokens())
        return sorted(toks)

    def edges_from(self, token: str) -> List[Edge]:
        return list(self.adjacency.get(token, []))

    def edge(self, pool_id: str, token_in: str) -> Edge:
        for e in self.adjacency.get(token_in, []):
            if e.pool_id == pool_id:
                return e
        raise RouterError(f"no edge {token_in} via {pool_id}")

    def pools_between(self, a: str, b: str) -> List[str]:
        return [e.pool_id for e in self.adjacency.get(a, []) if e.token_out == b]

    def _dfs(self, start: str, target: str, max_hops: int, allow_reuse: bool, cycles: bool,
             limit: Optional[int]) -> List[RoutePath]:
        out: List[RoutePath] = []
        toks: List[str] = [start]
        pids: List[str] = []

        def walk() -> bool:
            here = toks[-1]
            for e in self.adjacency.get(here, []):
                if not allow_reuse and e.pool_id in pids:
                    continue
                nxt = e.token_out
                if nxt == target and (cycles or nxt != start):
                    if not cycles or len(pids) + 1 >= 2:
                        out.append(RoutePath(tuple(toks + [nxt]), tuple(pids + [e.pool_id])))
                        if limit is not None and len(out) >= limit:
                            return True
                    continue  # a path ends at the target
                if nxt in toks:  # simple paths only (cycles close at start)
                    continue
                if len(pids) + 1 >= max_hops:
                    continue
                toks.append(nxt)
                pids.append(e.pool_id)
                stop = walk()
                toks.pop()
                pids.pop()
                if stop:
                    return True
            return False

        if max_hops >= 1:
            walk()
        return out

    def enumerate_paths(self, token_in: str, token_out: str, max_hops: int = DEFAULT_MAX_HOPS,
                        allow_pool_reuse: bool = False, limit: Optional[int] = None) -> List[RoutePath]:
        """Simple token paths from ``token_in`` to ``token_out`` with at most ``max_hops`` hops."""
        if token_in == token_out:
            raise RouterError("use enumerate_cycles for closed routes")
        _check_hops(max_hops)
        return self._dfs(token_in, token_out, max_hops, allow_pool_reuse, False, limit)

    def enumerate_cycles(self, max_hops: int = DEFAULT_MAX_HOPS, start_tokens: Optional[Iterable[str]] = None,
                         dedupe_rotations: bool = True) -> List[RoutePath]:
        """Closed routes ``A -> ... -> A`` (2..max_hops hops, no pool reused).

        With ``dedupe_rotations`` each loop appears once, started from the
        first token in ``start_tokens`` order (alphabetical by default) it visits.
        """
        _check_hops(max_hops)
        starts = list(start_tokens) if start_tokens is not None else self.tokens
        seen: Set[Tuple[str, ...]] = set()
        out: List[RoutePath] = []
        for s in starts:
            for c in self._dfs(s, s, max_hops, False, True, None):
                if dedupe_rotations:
                    key = c.canonical_cycle_key()
                    if key in seen:
                        continue
                    seen.add(key)
                out.append(c)
        return out

    def cycles_through(self, pool_ids: Iterable[str], max_hops: int = DEFAULT_MAX_HOPS,
                       start_tokens: Optional[Iterable[str]] = None) -> List[RoutePath]:
        wanted = set(pool_ids)
        return [c for c in self.enumerate_cycles(max_hops, start_tokens) if wanted & set(c.pool_ids)]

    def negative_cycles(self) -> List[RoutePath]:
        """Bellman-Ford on ``-log(rate)``: loops whose marginal rate product exceeds 1.

        Runs from a virtual source connected to every token; each detected loop
        is returned once (rotation-deduplicated). This is a marginal-price test:
        whether a loop is profitable after price impact is decided by
        :func:`optimize_cycle_input`.
        """
        toks = self.tokens
        edges = [e for t in toks for e in self.adjacency.get(t, [])]
        dist = {t: 0.0 for t in toks}
        pred: Dict[str, Optional[Edge]] = {t: None for t in toks}
        last: Optional[str] = None
        for _ in range(len(toks)):
            last = None
            for e in edges:
                w = e.log_weight
                if dist[e.token_in] + w < dist[e.token_out] - 1e-15:
                    dist[e.token_out] = dist[e.token_in] + w
                    pred[e.token_out] = e
                    last = e.token_out
            if last is None:
                return []
        found: List[RoutePath] = []
        seen: Set[Tuple[str, ...]] = set()
        for t in toks:
            # Walk back len(toks) steps to land inside a cycle, then extract it.
            x: Optional[str] = t
            for _ in range(len(toks)):
                e = pred.get(x) if x is not None else None
                x = e.token_in if e is not None else None
            if x is None:
                continue
            cyc_edges: List[Edge] = []
            cur = x
            for _ in range(len(toks) + 1):
                e = pred[cur]
                if e is None:
                    break
                cyc_edges.append(e)
                cur = e.token_in
                if cur == x:
                    break
            if cur != x or not cyc_edges:
                continue
            cyc_edges.reverse()
            path = RoutePath(tuple([cyc_edges[0].token_in] + [e.token_out for e in cyc_edges]),
                             tuple(e.pool_id for e in cyc_edges))
            if path.reuses_pool:
                continue
            key = path.canonical_cycle_key()
            if key not in seen and cycle_rate(self.pools, path) > 1:
                seen.add(key)
                found.append(path)
        return found


def _check_hops(max_hops: int) -> None:
    if not 1 <= max_hops <= MAX_SUPPORTED_HOPS:
        raise RouterError(f"max_hops must be within [1, {MAX_SUPPORTED_HOPS}]")


def cycle_rate(pools: Mapping[str, Pool], path: RoutePath) -> Fraction:
    """Product of marginal after-fee rates along ``path`` (> 1 means a marginal arbitrage)."""
    r = Fraction(1)
    for pid, tin in zip(path.pool_ids, path.tokens):
        pool = pools[pid]
        zfo, _ = _direction(pool, tin)
        r *= pool.spot_price(zfo) * (1 - pool_fee_fraction(pool))
    return r


# ---------------------------------------------------------------------------
# Sequential cross-protocol execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HopExecution:
    index: int
    pool_id: str
    version: str
    token_in: str
    token_out: str
    zero_for_one: bool
    amount_in: int
    amount_out: int
    fee_paid: int
    fee: Fraction
    spot_before: Fraction  # output per input, fee-free, before this hop
    spot_after: Fraction
    sqrt_price_before_x96: int  # v2: implied from reserves
    sqrt_price_after_x96: int
    tick_before: Optional[int]
    tick_after: Optional[int]
    ticks_crossed: int
    liquidity_before: int
    liquidity_after: int
    reserves_before: Optional[Tuple[int, int]]
    reserves_after: Optional[Tuple[int, int]]
    revisit: bool  # pool already traded earlier on this route

    @property
    def execution_price(self) -> Fraction:
        return Fraction(self.amount_out, self.amount_in)

    @property
    def impact(self) -> Fraction:
        """1 - execution / spot (fee plus curve)."""
        return 1 - self.execution_price / self.spot_before

    @property
    def curve_impact(self) -> Fraction:
        """Impact from curve movement alone (fee removed)."""
        return 1 - self.execution_price / (self.spot_before * (1 - self.fee))

    @property
    def price_shift(self) -> Fraction:
        """Relative move of this direction's marginal price (negative: worse for the next buyer)."""
        return self.spot_after / self.spot_before - 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "pool_id": self.pool_id,
            "version": self.version,
            "token_in": self.token_in,
            "token_out": self.token_out,
            "zero_for_one": self.zero_for_one,
            "amount_in": self.amount_in,
            "amount_out": self.amount_out,
            "fee_paid": self.fee_paid,
            "fee": float(self.fee),
            "spot_before": float(self.spot_before),
            "spot_after": float(self.spot_after),
            "execution_price": float(self.execution_price),
            "impact_bps": float(self.impact * BPS_DENOMINATOR),
            "curve_impact_bps": float(self.curve_impact * BPS_DENOMINATOR),
            "price_shift_bps": float(self.price_shift * BPS_DENOMINATOR),
            "sqrt_price_before_x96": self.sqrt_price_before_x96,
            "sqrt_price_after_x96": self.sqrt_price_after_x96,
            "tick_before": self.tick_before,
            "tick_after": self.tick_after,
            "ticks_crossed": self.ticks_crossed,
            "liquidity_before": self.liquidity_before,
            "liquidity_after": self.liquidity_after,
            "reserves_before": list(self.reserves_before) if self.reserves_before else None,
            "reserves_after": list(self.reserves_after) if self.reserves_after else None,
            "revisit": self.revisit,
        }


@dataclass(frozen=True)
class SlippageDegradation:
    """Exact combined slippage of a multi-hop route (see module docstring)."""

    per_hop_impact: Tuple[Fraction, ...]
    per_hop_fee: Tuple[Fraction, ...]
    per_hop_curve: Tuple[Fraction, ...]
    spot_product: Fraction
    amount_in: int
    amount_out: int

    @property
    def combined_impact(self) -> Fraction:
        return 1 - Fraction(self.amount_out, self.amount_in) / self.spot_product

    @property
    def compounded_impact(self) -> Fraction:
        acc = Fraction(1)
        for d in self.per_hop_impact:
            acc *= 1 - d
        return 1 - acc

    @property
    def additive_impact(self) -> Fraction:
        return sum(self.per_hop_impact, Fraction(0))

    @property
    def interaction(self) -> Fraction:
        """Additive estimate minus the true combined impact (>= 0 when every hop impact is in [0, 1])."""
        return self.additive_impact - self.combined_impact

    @property
    def fee_drag(self) -> Fraction:
        acc = Fraction(1)
        for f in self.per_hop_fee:
            acc *= 1 - f
        return 1 - acc

    @property
    def curve_drag(self) -> Fraction:
        acc = Fraction(1)
        for c in self.per_hop_curve:
            acc *= 1 - c
        return 1 - acc

    @property
    def ideal_amount_out(self) -> int:
        """Output at marginal prices with fees but no curve movement."""
        return int(self.amount_in * self.spot_product * (1 - self.fee_drag))

    @property
    def spot_amount_out(self) -> int:
        """Output at marginal prices with no fees and no curve movement."""
        return int(self.amount_in * self.spot_product)

    @property
    def curve_loss(self) -> int:
        return self.ideal_amount_out - self.amount_out

    def cumulative_vector(self) -> Tuple[Fraction, ...]:
        """Combined impact after each hop: non-decreasing when hop impacts are non-negative."""
        out: List[Fraction] = []
        acc = Fraction(1)
        for d in self.per_hop_impact:
            acc *= 1 - d
            out.append(1 - acc)
        return tuple(out)

    def to_dict(self) -> Dict[str, Any]:
        bps = BPS_DENOMINATOR
        return {
            "per_hop_impact_bps": [float(d * bps) for d in self.per_hop_impact],
            "per_hop_fee_bps": [float(f * bps) for f in self.per_hop_fee],
            "per_hop_curve_bps": [float(c * bps) for c in self.per_hop_curve],
            "cumulative_impact_bps": [float(c * bps) for c in self.cumulative_vector()],
            "combined_impact_bps": float(self.combined_impact * bps),
            "additive_impact_bps": float(self.additive_impact * bps),
            "interaction_bps": float(self.interaction * bps),
            "fee_drag_bps": float(self.fee_drag * bps),
            "curve_drag_bps": float(self.curve_drag * bps),
            "spot_amount_out": self.spot_amount_out,
            "ideal_amount_out": self.ideal_amount_out,
            "amount_out": self.amount_out,
            "curve_loss": self.curve_loss,
        }


@dataclass(frozen=True)
class PathExecution:
    path: RoutePath
    amount_in: int
    amount_out: int
    hops: Tuple[HopExecution, ...]
    degradation: SlippageDegradation
    gas_estimate: int

    @property
    def profit(self) -> Optional[int]:
        return self.amount_out - self.amount_in if self.path.is_cycle else None

    @property
    def ticks_crossed(self) -> int:
        return sum(h.ticks_crossed for h in self.hops)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path.to_dict(),
            "amount_in": self.amount_in,
            "amount_out": self.amount_out,
            "profit": self.profit,
            "gas_estimate": self.gas_estimate,
            "ticks_crossed": self.ticks_crossed,
            "hops": [h.to_dict() for h in self.hops],
            "degradation": self.degradation.to_dict(),
        }


def execute_path(pools: Mapping[str, Pool], path: RoutePath, amount_in: int, *, mutate: bool = False,
                 min_amount_out: Optional[int] = None) -> PathExecution:
    """Run ``path`` hop by hop, each hop on the state the previous hops left.

    Pools are cloned; with ``mutate=True`` the final states are copied back
    into the caller's pool objects only if every hop fills and the output
    meets ``min_amount_out`` (the route is atomic).
    """
    if amount_in <= 0:
        raise RouterError("amount_in must be positive")
    validate_path(pools, path)
    work: Dict[str, Pool] = {pid: clone_pool(pools[pid]) for pid in set(path.pool_ids)}
    hops: List[HopExecution] = []
    impacts: List[Fraction] = []
    fees: List[Fraction] = []
    curves: List[Fraction] = []
    spot_product = Fraction(1)
    seen: Set[str] = set()
    amount = amount_in
    for i, (pid, tin, tout) in enumerate(zip(path.pool_ids, path.tokens, path.tokens[1:])):
        pool = work[pid]
        zfo, _ = _direction(pool, tin)
        fee = pool_fee_fraction(pool)
        spot_before = pool.spot_price(zfo)
        sp_before, tick_before, liq_before, res_before = (pool_sqrt_price_x96(pool), pool_tick(pool),
                                                          pool_liquidity(pool), _reserves(pool))
        try:
            res = _quote(pool, amount, zfo)
        except MASEError as exc:
            raise RouterError(f"hop {i} ({pid}): {exc}") from exc
        if not res.fully_filled:
            raise RouterError(f"hop {i} ({pid}): insufficient liquidity (partial fill)")
        if res.amount_out <= 0:
            raise RouterError(f"hop {i} ({pid}): zero output")
        _apply(pool, res, amount, zfo)
        hop = HopExecution(
            index=i, pool_id=pid, version=pool_version(pool), token_in=tin, token_out=tout, zero_for_one=zfo,
            amount_in=amount, amount_out=res.amount_out, fee_paid=res.fee_paid, fee=fee,
            spot_before=spot_before, spot_after=pool.spot_price(zfo),
            sqrt_price_before_x96=sp_before, sqrt_price_after_x96=pool_sqrt_price_x96(pool),
            tick_before=tick_before, tick_after=pool_tick(pool), ticks_crossed=res.ticks_crossed,
            liquidity_before=liq_before, liquidity_after=pool_liquidity(pool),
            reserves_before=res_before, reserves_after=_reserves(pool), revisit=pid in seen,
        )
        seen.add(pid)
        hops.append(hop)
        impacts.append(hop.impact)
        fees.append(fee)
        curves.append(hop.curve_impact)
        spot_product *= spot_before
        amount = res.amount_out
    if min_amount_out is not None and amount < min_amount_out:
        raise RouterError(f"slippage: amount_out {amount} below min {min_amount_out}")
    if mutate:
        for pid, twin in work.items():
            vars(pools[pid]).update(vars(twin))
    deg = SlippageDegradation(tuple(impacts), tuple(fees), tuple(curves), spot_product, amount_in, amount)
    gas = estimate_swap_gas([work[p] for p in path.pool_ids], sum(h.ticks_crossed for h in hops))
    return PathExecution(path, amount_in, amount, tuple(hops), deg, gas)


def quote_path_output(pools: Mapping[str, Pool], path: RoutePath, amount_in: int) -> Optional[int]:
    """Output of ``path`` for ``amount_in`` or None if any hop cannot fill.

    Routes that never revisit a pool are quoted with pure quotes (no cloning);
    routes that revisit a pool fall back to sequential execution on clones.
    """
    if amount_in <= 0:
        return None
    if path.reuses_pool:
        try:
            return execute_path(pools, path, amount_in).amount_out
        except (RouterError, MASEError):
            return None
    amount = amount_in
    for pid, tin in zip(path.pool_ids, path.tokens):
        pool = pools.get(pid)
        if pool is None:
            return None
        try:
            zfo, _ = _direction(pool, tin)
            res = _quote(pool, amount, zfo)
        except MASEError:
            return None
        if not res.fully_filled or res.amount_out <= 0:
            return None
        amount = res.amount_out
    return amount


# ---------------------------------------------------------------------------
# Cycle optimisation and arbitrage
# ---------------------------------------------------------------------------


def _profit_fn(pools: Mapping[str, Pool], path: RoutePath
               ) -> Tuple[Callable[[int], Optional[int]], Dict[int, Optional[int]]]:
    cache: Dict[int, Optional[int]] = {}

    def profit(a: int) -> Optional[int]:
        if a not in cache:
            out = quote_path_output(pools, path, a)
            cache[a] = None if out is None else out - a
        return cache[a]

    return profit, cache


def optimize_cycle_input(pools: Mapping[str, Pool], path: RoutePath, lo: Optional[int] = None,
                         hi: Optional[int] = None, grid: int = 24, refine: int = 48,
                         max_input_bps: int = 2_000) -> Optional[Tuple[int, int]]:
    """Profit-maximising input for a cycle: ``(amount_in, profit)`` with profit > 0, or None.

    Geometric grid over ``[lo, hi]`` (default ``hi``: ``max_input_bps`` of the
    first pool's input-side reserve), then an integer ternary search between
    the best grid point's neighbours. Every evaluated size is a candidate, so
    the result is never worse than the best grid point.
    """
    if not path.is_cycle:
        raise RouterError("optimize_cycle_input needs a cycle")
    validate_path(pools, path)
    first = pools[path.pool_ids[0]]
    zfo, _ = _direction(first, path.token_in)
    if hi is None:
        hi = max(1, input_reserve_scale(first, zfo) * max_input_bps // BPS_DENOMINATOR)
    if lo is None:
        lo = max(1, hi // 10**9)
    if hi < lo or lo < 1:
        return None
    profit, cache = _profit_fn(pools, path)
    floor = -(1 << 512)

    def score(a: int) -> Tuple[int, int]:
        p = profit(a)
        return (floor, 0) if p is None else (p, -a)

    sizes = geometric_grid(lo, hi, grid)
    if not sizes:
        return None
    scores = [score(a) for a in sizes]
    i = max(range(len(sizes)), key=lambda k: scores[k])
    if scores[i][0] == floor:
        return None
    left = sizes[i - 1] if i > 0 else lo
    right = sizes[i + 1] if i + 1 < len(sizes) else hi
    for _ in range(refine):
        if right - left < 3:
            break
        m1, m2 = left + (right - left) // 3, right - (right - left) // 3
        if score(m1) < score(m2):
            left = m1
        else:
            right = m2
    for a in range(left, right + 1) if right - left < 3 else (left, right):
        score(a)
    best = max(((p, -a) for a, p in cache.items() if p is not None), default=None)
    if best is None or best[0] <= 0:
        return None
    return -best[1], best[0]


@dataclass(frozen=True)
class ArbitrageOpportunity:
    path: RoutePath
    amount_in: int
    amount_out: int
    profit: int  # in path.token_in units
    profit_native: int
    gas_estimate: int
    gas_cost_native: int
    marginal_rate: Fraction

    @property
    def net_profit_native(self) -> int:
        return self.profit_native - self.gas_cost_native

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path.to_dict(),
            "amount_in": self.amount_in,
            "amount_out": self.amount_out,
            "profit": self.profit,
            "profit_token": self.path.token_in,
            "profit_native": self.profit_native,
            "gas_estimate": self.gas_estimate,
            "gas_cost_native": self.gas_cost_native,
            "net_profit_native": self.net_profit_native,
            "marginal_rate": float(self.marginal_rate),
        }


def evaluate_cycle(pools: Mapping[str, Pool], path: RoutePath, native: str = "WETH", gas_price_wei: int = 0,
                   grid: int = 24, refine: int = 48, max_input_bps: int = 2_000) -> Optional[ArbitrageOpportunity]:
    """Best execution of one cycle, or None when no input size is profitable."""
    rate = cycle_rate(pools, path)
    if rate <= 1:
        return None  # marginal rate <= 1: price impact only makes it worse
    best = optimize_cycle_input(pools, path, grid=grid, refine=refine, max_input_bps=max_input_bps)
    if best is None:
        return None
    a, p = best
    ex = execute_path(pools, path, a)
    gas = ex.gas_estimate
    return ArbitrageOpportunity(path, a, ex.amount_out, p, native_value(pools, path.token_in, p, native), gas,
                                gas * gas_price_wei, rate)


def _cycle_chunk(task: Tuple[List[Pool], List[RoutePath], str, int, int, int, int]) -> List[ArbitrageOpportunity]:
    pool_list, cycles, native, gas_price, grid, refine, max_bps = task
    pools = {p.pool_id: p for p in pool_list}
    out: List[ArbitrageOpportunity] = []
    for c in cycles:
        try:
            opp = evaluate_cycle(pools, c, native, gas_price, grid, refine, max_bps)
        except (RouterError, MASEError):
            continue
        if opp is not None:
            out.append(opp)
    return out


def _route_chunk(task: Tuple[List[Pool], List[RoutePath], int]) -> List[Tuple[str, Optional[int]]]:
    pool_list, paths, amount = task
    pools = {p.pool_id: p for p in pool_list}
    return [(p.key, quote_path_output(pools, p, amount)) for p in paths]


def _run(executor: Optional[WorkerPool], fn: Callable[[Any], Any], tasks: List[Any]) -> List[Any]:
    if executor is None:
        return [fn(t) for t in tasks]
    return executor.map(fn, tasks)


# ---------------------------------------------------------------------------
# Order splitting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitResult:
    paths: Tuple[RoutePath, ...]
    allocations: Tuple[int, ...]
    outputs: Tuple[int, ...]
    amount_in: int
    amount_out: int
    best_single_path: Optional[RoutePath]
    best_single_out: int

    @property
    def improvement(self) -> int:
        return self.amount_out - self.best_single_out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "amount_in": self.amount_in,
            "amount_out": self.amount_out,
            "best_single_out": self.best_single_out,
            "best_single_path": self.best_single_path.key if self.best_single_path else None,
            "improvement": self.improvement,
            "legs": [{"path": p.key, "amount_in": a, "amount_out": o}
                     for p, a, o in zip(self.paths, self.allocations, self.outputs) if a],
        }


def split_route(pools: Mapping[str, Pool], paths: Sequence[RoutePath], amount_in: int, parts: int = 16) -> SplitResult:
    """Water-fill ``amount_in`` across ``paths`` in ``parts`` increments.

    Increments are applied to shared cloned state, so paths that share pools
    see each other's price impact. If the split is worse than the best single
    path (possible with rounding or shared pools), the single path is returned.
    """
    if amount_in <= 0:
        raise RouterError("amount_in must be positive")
    if not paths:
        raise RouterError("split_route needs at least one path")
    if parts < 1:
        raise RouterError("parts must be >= 1")
    t_in, t_out = paths[0].token_in, paths[0].token_out
    for p in paths:
        validate_path(pools, p)
        if p.token_in != t_in or p.token_out != t_out:
            raise RouterError("all paths must share token_in and token_out")
    singles = [(quote_path_output(pools, p, amount_in) or 0, -i) for i, p in enumerate(paths)]
    best_out, neg_i = max(singles)
    best_path = paths[-neg_i] if best_out > 0 else None

    involved = {pid for p in paths for pid in p.pool_ids}
    work: Dict[str, Pool] = {pid: clone_pool(pools[pid]) for pid in involved}
    alloc = [0] * len(paths)
    outs = [0] * len(paths)
    step, extra = divmod(amount_in, parts)
    for k in range(parts):
        inc = step + (extra if k == parts - 1 else 0)
        if inc <= 0:
            continue
        best_k: Optional[Tuple[int, int]] = None
        for i, p in enumerate(paths):
            o = quote_path_output(work, p, inc)
            if o is not None and (best_k is None or o > best_k[0]):
                best_k = (o, i)
        if best_k is None:
            break
        o, i = best_k
        ex = execute_path(work, paths[i], inc, mutate=True)
        alloc[i] += inc
        outs[i] += ex.amount_out
    total = sum(outs)
    if sum(alloc) != amount_in or total < best_out:
        alloc = [amount_in if (best_path is not None and p is best_path) else 0 for p in paths]
        outs = [best_out if a else 0 for a in alloc]
        total = best_out
    return SplitResult(tuple(paths), tuple(alloc), tuple(outs), amount_in, total, best_path, best_out)


# ---------------------------------------------------------------------------
# Router facade
# ---------------------------------------------------------------------------


@dataclass
class RouteQuote:
    token_in: str
    token_out: str
    amount_in: int
    best: Optional[PathExecution]
    ranked: List[Tuple[str, Optional[int]]]  # (path key, output) best first; None = unfillable
    paths_evaluated: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_in": self.token_in,
            "token_out": self.token_out,
            "amount_in": self.amount_in,
            "paths_evaluated": self.paths_evaluated,
            "best": self.best.to_dict() if self.best else None,
            "ranked": [{"path": k, "amount_out": o} for k, o in self.ranked],
        }


@dataclass
class CycleScanReport:
    opportunities: List[ArbitrageOpportunity]
    cycles_evaluated: int
    negative_cycles: List[RoutePath]
    elapsed_s: float
    workers: int
    max_hops: int
    pools: int
    tokens: int

    @property
    def best(self) -> Optional[ArbitrageOpportunity]:
        return self.opportunities[0] if self.opportunities else None

    def summary(self) -> Dict[str, Any]:
        return {
            "pools": self.pools,
            "tokens": self.tokens,
            "max_hops": self.max_hops,
            "cycles_evaluated": self.cycles_evaluated,
            "profitable_cycles": len(self.opportunities),
            "negative_cycles_bellman_ford": [c.key for c in self.negative_cycles],
            "best": self.best.to_dict() if self.best else None,
            "top": [o.to_dict() for o in self.opportunities[:5]],
            "total_net_profit_native": sum(max(0, o.net_profit_native) for o in self.opportunities),
            "elapsed_s": round(self.elapsed_s, 4),
            "cycles_per_s": round(self.cycles_evaluated / self.elapsed_s, 1) if self.elapsed_s > 0 else 0.0,
            "workers": self.workers,
        }


class MultiHopRouter:
    """Path finding, sequential execution, order splitting and cycle arbitrage over a pool set."""

    def __init__(self, pools: Iterable[Pool], native: str = "WETH", max_hops: int = DEFAULT_MAX_HOPS) -> None:
        _check_hops(max_hops)
        self.graph = TokenGraph(pools)
        self.native = native
        self.max_hops = max_hops
        self._cycle_cache: Dict[Tuple[int, Optional[Tuple[str, ...]]], List[RoutePath]] = {}

    @property
    def pools(self) -> Dict[str, Pool]:
        return self.graph.pools

    def refresh(self) -> None:
        self.graph.refresh()

    def paths(self, token_in: str, token_out: str, max_hops: Optional[int] = None) -> List[RoutePath]:
        return self.graph.enumerate_paths(token_in, token_out, max_hops or self.max_hops)

    def best_route(self, token_in: str, token_out: str, amount_in: int, max_hops: Optional[int] = None,
                   executor: Optional[WorkerPool] = None) -> RouteQuote:
        paths = self.paths(token_in, token_out, max_hops)
        pool_list = list(self.pools.values())
        tasks = [(pool_list, chunk, amount_in) for chunk in chunked(paths, ROUTER_CHUNKS)]
        rows = [r for part in _run(executor, _route_chunk, tasks) for r in part]
        order = {p.key: i for i, p in enumerate(paths)}
        rows.sort(key=lambda r: (r[1] is None, -(r[1] or 0), order[r[0]]))
        best = None
        if rows and rows[0][1] is not None:
            best_path = next(p for p in paths if p.key == rows[0][0])
            best = execute_path(self.pools, best_path, amount_in)
        return RouteQuote(token_in, token_out, amount_in, best, rows, len(paths))

    def split(self, token_in: str, token_out: str, amount_in: int, parts: int = 16,
              max_paths: int = 4, max_hops: Optional[int] = None) -> SplitResult:
        quote = self.best_route(token_in, token_out, amount_in, max_hops)
        keys = [k for k, o in quote.ranked if o is not None][:max_paths]
        if not keys:
            raise RouterError(f"no fillable path {token_in} -> {token_out}")
        by_key = {p.key: p for p in self.paths(token_in, token_out, max_hops)}
        return split_route(self.pools, [by_key[k] for k in keys], amount_in, parts)

    def execute(self, path: RoutePath, amount_in: int, min_amount_out: Optional[int] = None) -> PathExecution:
        """Atomic, state-mutating execution; refreshes the graph afterwards."""
        ex = execute_path(self.pools, path, amount_in, mutate=True, min_amount_out=min_amount_out)
        self.refresh()
        return ex

    def cycles(self, max_hops: Optional[int] = None, start_tokens: Optional[Iterable[str]] = None) -> List[RoutePath]:
        """Cycles of the pool graph (cached: the graph's shape never changes, only prices do)."""
        hops = max_hops or self.max_hops
        starts = None if start_tokens is None else tuple(start_tokens)
        key = (hops, starts)
        if key not in self._cycle_cache:
            self._cycle_cache[key] = self.graph.enumerate_cycles(hops, starts)
        return list(self._cycle_cache[key])

    def scan_cycles(self, max_hops: Optional[int] = None, executor: Optional[WorkerPool] = None,
                    gas_price_wei: int = 0, min_net_profit_native: int = 0,
                    start_tokens: Optional[Iterable[str]] = None, grid: int = 24, refine: int = 48,
                    max_input_bps: int = 2_000, through_pools: Optional[Iterable[str]] = None,
                    chunks: int = ROUTER_CHUNKS) -> CycleScanReport:
        """Evaluate every cycle (optionally only those through ``through_pools``) in parallel chunks."""
        hops = max_hops or self.max_hops
        started = time.perf_counter()
        cycles = self.cycles(hops, start_tokens)
        if through_pools is not None:
            wanted = set(through_pools)
            cycles = [c for c in cycles if wanted & set(c.pool_ids)]
        pool_list = list(self.pools.values())
        tasks = [(pool_list, chunk, self.native, gas_price_wei, grid, refine, max_input_bps)
                 for chunk in chunked(cycles, max(1, chunks))]
        opps = [o for part in _run(executor, _cycle_chunk, tasks) for o in part]
        opps = [o for o in opps if o.net_profit_native >= min_net_profit_native]
        opps.sort(key=lambda o: (-o.net_profit_native, -o.profit_native, o.path.key))
        neg = self.graph.negative_cycles()
        return CycleScanReport(opps, len(cycles), neg, time.perf_counter() - started,
                               executor.workers if executor is not None else 1, hops, len(self.pools),
                               len(self.graph.tokens))

    def best_backrun(self, touched_pools: Iterable[str], pools: Optional[Mapping[str, Pool]] = None,
                     max_hops: Optional[int] = None, gas_price_wei: int = 0, grid: int = 16,
                     refine: int = 32) -> Optional[ArbitrageOpportunity]:
        """Most profitable cycle through any of ``touched_pools`` on ``pools`` (default: router state)."""
        state = dict(pools) if pools is not None else self.pools
        wanted = set(touched_pools)
        best: Optional[ArbitrageOpportunity] = None
        for c in self.cycles(max_hops, start_tokens=self._start_tokens()):
            if not wanted & set(c.pool_ids):
                continue
            try:
                opp = evaluate_cycle(state, c, self.native, gas_price_wei, grid, refine)
            except (RouterError, MASEError):
                continue
            if opp is not None and (best is None or (opp.net_profit_native, opp.path.key) >
                                    (best.net_profit_native, best.path.key)):
                best = opp
        return best

    def _start_tokens(self) -> List[str]:
        """Native token first so cycles are preferably denominated in it."""
        toks = self.graph.tokens
        return ([self.native] if self.native in toks else []) + [t for t in toks if t != self.native]

    def build_tx(self, path: RoutePath, amount_in: int, sender: str, nonce: int, base_fee: int,
                 tip_wei: int = GWEI, min_amount_out: Optional[int] = None, tx_hash: Optional[str] = None,
                 gas_headroom_pct: int = 50) -> BuilderTx:
        """Builder transaction for ``path`` (block builder routes may not repeat a pool)."""
        if path.reuses_pool:
            raise RouterError("block builder transactions cannot revisit a pool")
        validate_path(self.pools, path)
        ex = execute_path(self.pools, path, amount_in)
        gas = ex.gas_estimate * (100 + gas_headroom_pct) // 100 + 40_000
        min_out = ex.amount_out if min_amount_out is None else min_amount_out
        return BuilderTx(tx_hash or f"router:{sender}:{nonce}", sender, nonce, gas, base_fee * 2 + tip_wei, tip_wei,
                         TxKind.SWAP, {"route": list(path.pool_ids), "token_in": path.token_in,
                                       "amount_in": amount_in, "amount_out_min": min_out})


__all__ = [
    "DEFAULT_MAX_HOPS",
    "MAX_SUPPORTED_HOPS",
    "ROUTER_CHUNKS",
    "ArbitrageOpportunity",
    "CycleScanReport",
    "Edge",
    "HopExecution",
    "MultiHopRouter",
    "PathExecution",
    "RouteQuote",
    "RoutePath",
    "RouterError",
    "SlippageDegradation",
    "SplitResult",
    "TokenGraph",
    "cycle_rate",
    "evaluate_cycle",
    "execute_path",
    "optimize_cycle_input",
    "pool_fee_fraction",
    "pool_liquidity",
    "pool_sqrt_price_x96",
    "pool_tick",
    "quote_path_output",
    "split_route",
    "v2_implied_sqrt_price_x96",
    "validate_path",
]
