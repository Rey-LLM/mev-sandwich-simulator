"""Layer 16: pre-trade sandwich check.

Before sending a swap, answer two questions with the verified AMM math from
:mod:`simulator.slippage_engine`:

1. **Can this swap be sandwiched at the slippage I am about to sign?** The
   attacker's best front-run is searched exactly: the largest front-run the
   victim's ``amountOutMin`` still tolerates is found by bisection, and the
   profit curve below it is scanned on a geometric grid and refined by a
   ternary search. The swap is *sandwichable* when the best front-run plus
   back-run nets more than the attacker's gas.
2. **What slippage is safe?** Attacker profit can only grow with the slippage
   tolerance (a looser ``amountOutMin`` admits every front-run a tighter one
   did, and more), so the largest tolerance with no profitable sandwich is
   found by bisection over basis points. When even that is not enough, the
   largest swap size that is safe at the requested tolerance is reported, so
   the trade can be split.

Scope: one pool, the state given (reserves for v2; price, active liquidity and
optionally the initialized ticks for v3). A v3 pool given without ticks is a
single-range view: the active liquidity is assumed to continue past the
current range, which is accurate for swaps that do not cross a tick. Gas is
counted only when it can be expressed in the input token (input is WETH, or a
price is supplied); otherwise it is ignored, which makes the answer
conservative (a swap is called sandwichable more often, never less).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from . import i18n
from .invariants import input_reserve_scale
from .mempool_emulator import estimate_swap_gas, geometric_grid
from .sandwich_detector import V3_TICK_SPACING, WETH
from .slippage_engine import BPS_DENOMINATOR, ConcentratedPool, MASEError, V2Pool, min_amount_out

__all__ = ["AttackOutcome", "GuardReport", "assess_swap", "best_attack", "pool_from_spec", "simulate_attack"]

Pool = Union[V2Pool, ConcentratedPool]

MAX_SEARCH_SLIPPAGE_BPS = 5_000
GRID_POINTS = 40
REFINE_STEPS = 60
WEI_PER_ETH = 10 ** 18


@dataclass(frozen=True)
class AttackOutcome:
    front_in: int
    front_out: int
    victim_out: int
    back_out: int
    gas_cost: int  # in the victim's input token

    @property
    def gross_profit(self) -> int:
        return self.back_out - self.front_in

    @property
    def net_profit(self) -> int:
        return self.gross_profit - self.gas_cost

    def to_dict(self) -> Dict[str, int]:
        return {"front_in": self.front_in, "front_out": self.front_out, "victim_out": self.victim_out,
                "back_out": self.back_out, "gross_profit": self.gross_profit, "gas_cost": self.gas_cost,
                "net_profit": self.net_profit}


def _swap(pool: Pool, amount_in: int, zero_for_one: bool) -> Optional[int]:
    """Exact-input swap applied to ``pool`` in place; None if it cannot fill."""
    try:
        res = pool.quote_exact_in(amount_in, zero_for_one)
    except MASEError:
        return None
    if not res.fully_filled or res.amount_out <= 0:
        return None
    if isinstance(pool, V2Pool):
        pool.apply_swap(res, zero_for_one)
    else:
        pool.apply_swap(amount_in, zero_for_one)
    return res.amount_out


def simulate_attack(pool: Pool, zero_for_one: bool, front_in: int, victim_in: int) -> Optional[Tuple[int, int, int]]:
    """(front_out, victim_out, back_out) of a sandwich on a copy of ``pool``.

    A shallow copy is enough: ``apply_swap`` only rebinds the reserve / price
    fields and never mutates the shared tick map.
    """
    work = copy.copy(pool)
    front_out = _swap(work, front_in, zero_for_one)
    if front_out is None:
        return None
    victim_out = _swap(work, victim_in, zero_for_one)
    if victim_out is None:
        return None
    back_out = _swap(work, front_out, not zero_for_one)
    if back_out is None:
        return None
    return front_out, victim_out, back_out


def best_attack(pool: Pool, zero_for_one: bool, victim_in: int, min_out: int, gas_cost: int = 0) -> Optional[AttackOutcome]:
    """The attacker's most profitable sandwich that still lets the victim clear ``min_out``.

    None when no front-run (of at least a millionth of the victim's input) leaves
    the victim above its limit.
    """
    cache: Dict[int, Optional[Tuple[int, int, int]]] = {}

    def run(a: int) -> Optional[Tuple[int, int, int]]:
        if a not in cache:
            cache[a] = simulate_attack(pool, zero_for_one, a, victim_in)
        return cache[a]

    def ok(a: int) -> bool:
        r = run(a)
        return r is not None and r[1] >= min_out

    # Dust front-runs can round to zero output, so the search starts at a
    # millionth of the victim's input (never below 1 unit).
    floor = max(1, victim_in // 1_000_000)
    if not ok(floor):
        return None
    hi = max(floor + 1, input_reserve_scale(pool, zero_for_one) * 4)
    if ok(hi):
        f_max = hi
    else:
        lo = floor
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ok(mid):
                lo = mid
            else:
                hi = mid
        f_max = lo

    def profit(a: int) -> int:
        r = run(a)
        return -(1 << 255) if r is None or r[1] < min_out else r[2] - a

    sizes = geometric_grid(floor, f_max, GRID_POINTS) or [f_max]
    if f_max not in sizes:
        sizes.append(f_max)
    scores = [profit(a) for a in sizes]
    best_i = max(range(len(sizes)), key=lambda i: (scores[i], -i))
    left = sizes[best_i - 1] if best_i > 0 else floor
    right = sizes[best_i + 1] if best_i + 1 < len(sizes) else f_max
    for _ in range(REFINE_STEPS):
        if right - left < 3:
            break
        m1 = left + (right - left) // 3
        m2 = right - (right - left) // 3
        if profit(m1) < profit(m2):
            left = m1
        else:
            right = m2
    for a in range(left, min(right, left + 3) + 1):
        profit(a)
    feasible = [(r[2] - a, -a) for a, r in cache.items() if r is not None and r[1] >= min_out]
    _, neg_a = max(feasible)
    a = -neg_a
    front_out, victim_out, back_out = cache[a]  # type: ignore[misc]
    return AttackOutcome(a, front_out, victim_out, back_out, gas_cost)


@dataclass
class GuardReport:
    pool: str
    version: str
    token_in: str
    token_out: str
    amount_in: int
    quoted_out: int
    slippage_bps: int
    min_out: int
    gas_cost: int
    gas_counted: bool
    attack: Optional[AttackOutcome]
    sandwichable: bool
    victim_loss: int
    safe_slippage_bps: Optional[int]
    max_safe_amount_in: Optional[int]
    advice: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pool": self.pool, "version": self.version, "token_in": self.token_in, "token_out": self.token_out,
            "amount_in": self.amount_in, "quoted_out": self.quoted_out, "slippage_bps": self.slippage_bps,
            "min_out": self.min_out, "sandwichable": self.sandwichable,
            "attack": None if self.attack is None else self.attack.to_dict(),
            "victim_loss_at_best_attack": self.victim_loss, "gas_cost_in_token_in": self.gas_cost,
            "gas_counted": self.gas_counted, "safe_slippage_bps": self.safe_slippage_bps,
            "max_safe_amount_in": self.max_safe_amount_in, "advice": list(self.advice), "notes": list(self.notes),
        }


def _direction(pool: Pool, token_in: str) -> Tuple[bool, str]:
    t0, t1 = pool.tokens()
    if token_in.lower() == t0.lower():
        return True, t1
    if token_in.lower() == t1.lower():
        return False, t0
    raise MASEError(f"token {token_in} is not in pool {pool.pool_id}")


def _profitable(pool: Pool, zfo: bool, amount_in: int, slippage_bps: int, gas_cost: int) -> Tuple[bool, Optional[AttackOutcome], int]:
    quoted = pool.quote_exact_in(amount_in, zfo).amount_out
    min_out = min_amount_out(quoted, slippage_bps)
    attack = best_attack(pool, zfo, amount_in, min_out, gas_cost)
    return (attack is not None and attack.net_profit > 0), attack, quoted


def assess_swap(pool: Pool, token_in: str, amount_in: int, slippage_bps: int = 50, gas_price_wei: int = 0,
                attack_gas: Optional[int] = None, token_in_per_eth: Optional[Fraction] = None,
                weth: str = WETH, lang: Optional[str] = None) -> GuardReport:
    """Is this swap sandwichable at ``slippage_bps``, and what tolerance is safe?"""
    if amount_in <= 0:
        raise MASEError("amount_in must be positive")
    if not 0 <= slippage_bps < BPS_DENOMINATOR:
        raise MASEError("slippage_bps must be in [0, 10000)")
    zfo, token_out = _direction(pool, token_in)
    gas_units = attack_gas if attack_gas is not None else 2 * estimate_swap_gas([pool])
    gas_wei = gas_units * gas_price_wei
    notes: List[str] = []
    if token_in.lower() == weth:
        gas_cost, counted = gas_wei, True
    elif token_in_per_eth is not None:
        gas_cost, counted = int(Fraction(gas_wei) * Fraction(token_in_per_eth) // WEI_PER_ETH), True
    else:
        gas_cost, counted = 0, gas_wei == 0
        if gas_wei:
            notes.append(i18n.t("guard.note.gas_ignored", lang))
    if isinstance(pool, ConcentratedPool) and not pool.ticks:
        notes.append(i18n.t("guard.note.single_range", lang))

    sandwichable, attack, quoted = _profitable(pool, zfo, amount_in, slippage_bps, gas_cost)
    min_out = min_amount_out(quoted, slippage_bps)
    victim_loss = quoted - attack.victim_out if attack is not None else 0

    # Largest tolerance with no profitable sandwich (profit is monotone in it).
    if _profitable(pool, zfo, amount_in, 0, gas_cost)[0]:
        safe: Optional[int] = None
    elif not _profitable(pool, zfo, amount_in, MAX_SEARCH_SLIPPAGE_BPS, gas_cost)[0]:
        safe = MAX_SEARCH_SLIPPAGE_BPS
    else:
        lo, hi = 0, MAX_SEARCH_SLIPPAGE_BPS
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if _profitable(pool, zfo, amount_in, mid, gas_cost)[0]:
                hi = mid
            else:
                lo = mid
        safe = lo

    # Largest size that is safe at the requested tolerance (only when this one is not).
    max_safe: Optional[int] = None
    if sandwichable:
        lo, hi = 0, amount_in
        while hi - lo > 1:
            mid = (lo + hi) // 2
            try:
                bad = _profitable(pool, zfo, mid, slippage_bps, gas_cost)[0]
            except MASEError:
                bad = False
            if bad:
                hi = mid
            else:
                lo = mid
        max_safe = lo or None

    advice: List[str] = []
    if not sandwichable:
        advice.append(i18n.t("guard.advice.safe", lang, slippage=slippage_bps))
    else:
        profit = attack.net_profit if attack is not None else 0
        advice.append(i18n.t("guard.advice.unsafe", lang, profit=profit, loss=victim_loss))
        if safe is not None and safe > 0:
            advice.append(i18n.t("guard.advice.lower_slippage", lang, safe=safe))
        if max_safe:
            parts = -(-amount_in // max_safe)
            advice.append(i18n.t("guard.advice.split", lang, size=max_safe, parts=parts))
        advice.append(i18n.t("guard.advice.private_rpc", lang))

    return GuardReport(pool=pool.pool_id, version="v2" if isinstance(pool, V2Pool) else pool.version,
                       token_in=token_in.lower(), token_out=token_out.lower(), amount_in=amount_in,
                       quoted_out=quoted, slippage_bps=slippage_bps, min_out=min_out, gas_cost=gas_cost,
                       gas_counted=counted, attack=attack, sandwichable=sandwichable, victim_loss=victim_loss,
                       safe_slippage_bps=safe, max_safe_amount_in=max_safe, advice=advice, notes=notes)


def pool_from_spec(spec: Mapping[str, Any]) -> Pool:
    """Build a pool from a JSON-style spec (also what the RPC pool fetchers return)."""
    version = str(spec.get("version", "v2")).lower()
    t0, t1 = str(spec["token0"]).lower(), str(spec["token1"]).lower()
    pool_id = str(spec.get("address") or spec.get("pool_id") or "")
    if version == "v2":
        return V2Pool(token0=t0, token1=t1, reserve0=int(spec["reserve0"]), reserve1=int(spec["reserve1"]),
                      fee_bps=int(spec.get("fee_bps", 30)), pool_id=pool_id)
    if version == "v3":
        fee = int(spec.get("fee", 3000))
        ticks = {int(k): int(v) for k, v in (spec.get("ticks") or {}).items()}
        return ConcentratedPool(token0=t0, token1=t1, sqrt_price_x96=int(spec["sqrt_price_x96"]),
                                liquidity=int(spec["liquidity"]), fee_pips=fee,
                                tick_spacing=int(spec.get("tick_spacing", V3_TICK_SPACING.get(fee, 60))),
                                ticks=ticks, pool_id=pool_id)
    raise MASEError(f"unsupported pool version {version!r} (expected v2 or v3)")
