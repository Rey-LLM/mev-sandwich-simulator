"""Layer 13: MEV attack vectors beyond the plain sandwich.

Every attack here is a sequence of primitive actions executed on a private
copy of a :class:`Scenario` by :class:`AttackState`, with the same exact
integer AMM math the rest of the simulator uses (``slippage_engine``, which
is bit-exact against the Uniswap v2 and v3 test vectors). Nothing connects to
a chain; all state is generated from a seed.

Primitive actions
-----------------

``swap``       attacker exact-input swap (reverts when not fully filled)
``victim``     a pending user swap; reverts below its ``amount_out_min``
``mint``       add concentrated liquidity (v3 / v4), Uniswap v3 mint rounding
``burn``       remove it and collect the fees it earned
``liquidate``  repay part of an unhealthy loan in a lending market (model)
``deposit`` / ``borrow``  lending-market collateral and debt (model)
``settle``     burn open positions and swap every non-base balance to base

Attack types (each returns an :class:`AttackResult` or ``None``)
----------------------------------------------------------------

``multi_victim_sandwich``  one front-run, every same-direction victim on the
                           pool that still clears its limit, one back-run
``jit_liquidity``          mint a narrow position just before a large v3/v4
                           swap and burn it right after, taking the fee
``backrun_arbitrage``      after a victim moves a pool, close the price gap
                           through a 2- or 3-pool cycle
``liquidation_backrun``    a victim moves an oracle pool, a loan becomes
                           unhealthy, the attacker liquidates it and sells
                           the seized collateral
``oracle_manipulation``    push a lending market's spot-price oracle, borrow
                           against inflated collateral and walk away; run
                           against a spot oracle and a TWAP oracle to show
                           why the TWAP defends against it

What is exact and what is a model
---------------------------------

* Token amounts of swaps, mints and burns are exact (v3 ``mint`` rounds up,
  ``burn`` rounds down, as in ``UniswapV3Pool._modifyPosition``).
* JIT fees are exact: every swap on a pool with an open attacker position is
  split at the position's tick boundaries (both are initialized ticks, so the
  contract ends a swap step there too and the split does not change any
  rounding; this is asserted on every swap). The fee growth of the in-range
  segments is accumulated in X128 and converted once at burn with
  ``mulDiv(feeGrowthInside, liquidity, Q128)`` like ``Position.update``.
  Pools with a v4 protocol fee are refused (the protocol share is not split
  out of fee growth here).
* The lending market is a **model** (Aave-style: LTV, liquidation threshold,
  close factor, bonus); its oracle is a pool's spot price or a TWAP of
  per-block spot observations.
* Gas is an **estimate** (per-action constants, like the rest of the
  simulator), and so is any value of a non-base token, which is converted at
  the scenario's reference prices (spot prices before the attack).
* A result is ``flat`` when every token except the base token nets to zero;
  then ``net_profit_base`` is exact up to the gas estimate.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .invariants import input_reserve_scale
from .mempool_emulator import (
    GAS_PER_HOP,
    GAS_PER_TICK_CROSS,
    GAS_TX_BASE,
    PendingTx,
    block_order,
    generate_mempool,
    geometric_grid,
    pool_version,
)
from .slippage_engine import (
    BPS_DENOMINATOR,
    MAX_SQRT_RATIO,
    MAX_TICK,
    MIN_SQRT_RATIO,
    MIN_TICK,
    Q128,
    UINT256_MAX,
    ConcentratedPool,
    MASEError,
    Pool,
    Position,
    SwapResult,
    V2Pool,
    V4Pool,
    _direction,
    get_amount0_delta,
    get_amount1_delta,
    get_sqrt_ratio_at_tick,
    mul_div,
    price_to_sqrt_price_x96,
)

__all__ = [
    "ATTACK_TYPES",
    "ActionReverted",
    "AttackResult",
    "AttackState",
    "JitPosition",
    "LendingMarket",
    "Loan",
    "Scenario",
    "backrun_arbitrage",
    "build_scenario",
    "jit_liquidity",
    "liquidation_backrun",
    "maximize_int",
    "multi_victim_sandwich",
    "oracle_manipulation",
    "reference_prices",
    "run_attack_catalog",
]

E18 = 10**18
E6 = 10**6

# Gas estimates for non-swap actions (not EVM-metered, same spirit as GAS_PER_HOP).
GAS_MINT = 180_000
GAS_BURN = 120_000  # burn + collect
GAS_LIQUIDATE = 250_000
GAS_DEPOSIT = 120_000
GAS_BORROW = 150_000
DEFAULT_GAS_PRICE_WEI = 11 * 10**9  # 10 gwei base fee + 1 gwei tip

ATTACK_TYPES = (
    "sandwich",
    "multi_victim_sandwich",
    "jit_liquidity",
    "backrun_arbitrage",
    "liquidation_backrun",
    "oracle_manipulation",
)


class ActionReverted(MASEError):
    """A primitive action reverted; the attack that issued it is invalid."""


# ---------------------------------------------------------------------------
# Lending market (model)
# ---------------------------------------------------------------------------


@dataclass
class Loan:
    collateral: int  # raw units of the collateral token
    debt: int  # raw units of the debt token


@dataclass
class LendingMarket:
    """Aave-style single-pair market. Rates are basis points.

    ``oracle_mode`` is ``"spot"`` (the oracle pool's current marginal price,
    the vulnerable design) or ``"twap"`` (the mean of the last
    ``twap_window`` per-block spot observations; observations are recorded
    at block boundaries, so trades inside the current block do not move it).
    Prices are debt-token units per collateral-token unit.
    """

    market_id: str
    collateral_token: str
    debt_token: str
    oracle_pool_id: str
    oracle_mode: str = "spot"
    twap_window: int = 30
    ltv_bps: int = 8_000
    liquidation_threshold_bps: int = 8_250
    liquidation_bonus_bps: int = 500
    close_factor_bps: int = 5_000
    available_debt: int = 0  # lendable liquidity
    loans: Dict[str, Loan] = field(default_factory=dict)
    observations: List[Fraction] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.oracle_mode not in ("spot", "twap"):
            raise MASEError(f"unknown oracle mode {self.oracle_mode!r}")
        if not 0 < self.ltv_bps <= self.liquidation_threshold_bps < BPS_DENOMINATOR:
            raise MASEError("lending: need 0 < ltv <= liquidation threshold < 100%")
        if self.twap_window <= 0:
            raise MASEError("lending: twap_window must be positive")

    def spot(self, pools: Mapping[str, Pool]) -> Fraction:
        pool = pools[self.oracle_pool_id]
        zero_for_one, _ = _direction(pool, self.collateral_token)
        return pool.spot_price(zero_for_one)

    def price(self, pools: Mapping[str, Pool]) -> Fraction:
        if self.oracle_mode == "spot" or not self.observations:
            return self.spot(pools)
        window = self.observations[-self.twap_window:]
        return sum(window, Fraction(0)) / len(window)

    def record_observation(self, pools: Mapping[str, Pool]) -> None:
        self.observations.append(self.spot(pools))

    def health(self, loan: Loan, price: Fraction) -> Fraction:
        if loan.debt <= 0:
            return Fraction(10**9)
        return loan.collateral * price * self.liquidation_threshold_bps / (loan.debt * BPS_DENOMINATOR)

    def max_repay(self, loan: Loan) -> int:
        return loan.debt * self.close_factor_bps // BPS_DENOMINATOR

    def seize_for(self, repay: int, price: Fraction) -> int:
        """Collateral given for ``repay`` debt: repay * (1 + bonus) / price, floored."""
        return math.floor(Fraction(repay * (BPS_DENOMINATOR + self.liquidation_bonus_bps), BPS_DENOMINATOR) / price)


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    pools: List[Pool]
    victims: List[PendingTx]
    markets: List[LendingMarket] = field(default_factory=list)
    base: str = "WETH"
    gas_price_wei: int = DEFAULT_GAS_PRICE_WEI
    capital_base: int = 2_000 * E18  # attacker capital bound, in base units
    seed: int = 0

    def __post_init__(self) -> None:
        self.pools = sorted(self.pools, key=lambda p: p.pool_id)
        self.victims = block_order(self.victims)
        self.prices = reference_prices(self.pools, self.base)

    def pool(self, pool_id: str) -> Pool:
        for p in self.pools:
            if p.pool_id == pool_id:
                return p
        raise MASEError(f"unknown pool {pool_id!r}")

    def market(self, market_id: str) -> LendingMarket:
        for m in self.markets:
            if m.market_id == market_id:
                return m
        raise MASEError(f"unknown market {market_id!r}")

    def value(self, token: str, amount: int) -> Fraction:
        """Value of ``amount`` of ``token`` in base units at the reference prices."""
        if token == self.base:
            return Fraction(amount)
        try:
            return amount * self.prices[token]
        except KeyError:
            raise MASEError(f"no reference price for {token!r}") from None

    def describe(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "base": self.base,
            "pools": [p.pool_id for p in self.pools],
            "victims": len(self.victims),
            "markets": [m.market_id for m in self.markets],
            "gas_price_wei": self.gas_price_wei,
            "capital_base": self.capital_base,
        }


def reference_prices(pools: Iterable[Pool], base: str) -> Dict[str, Fraction]:
    """Spot price of each token in ``base``, from the deepest direct pool against ``base``."""
    best: Dict[str, Tuple[int, Fraction]] = {}
    for pool in pools:
        t0, t1 = pool.tokens()
        if base not in (t0, t1):
            continue
        token = t1 if t0 == base else t0
        zero_for_one, _ = _direction(pool, token)
        depth = input_reserve_scale(pool, not zero_for_one)  # the base side
        if token not in best or depth > best[token][0]:
            best[token] = (depth, pool.spot_price(zero_for_one))
    out = {t: p for t, (_, p) in best.items()}
    out[base] = Fraction(1)
    return out


def _jitter(rng: random.Random, value: int, pct: int) -> int:
    return value * rng.randrange(100 - pct, 101 + pct) // 100


def build_scenario(seed: int = 0, victims: int = 8, oracle_mode: str = "spot",
                   gas_price_wei: int = DEFAULT_GAS_PRICE_WEI) -> Scenario:
    """A seeded market: two WETH/USDC venues (v2 + v3), WETH/DAI (v4), DAI/USDC (v2),
    one WETH-collateral / USDC-debt lending market whose oracle is the v2 pool,
    and ``victims`` pending swaps (``generate_mempool``) plus loans that sit just
    above their liquidation threshold.

    Reserves are jittered by the seed, so the same layout gives a different
    state per seed while pool indices stay stable (the discovery engine
    relies on that).
    """
    rng = random.Random(f"scenario:{seed}")
    usd_per_eth = rng.randrange(2_700, 3_301)
    weth_v2 = _jitter(rng, 3_000, 20) * E18
    usdc_v2 = weth_v2 * usd_per_eth * E6 // E18
    # v3 USDC/WETH: token0 = USDC; price = WETH wei per USDC unit; off by up to +-0.4%.
    skew = Fraction(rng.randrange(996, 1_005), 1_000)
    v3_price = Fraction(E18, usd_per_eth * E6) * skew
    v3_sqrt = price_to_sqrt_price_x96(v3_price)
    v3_tick = ConcentratedPool(token0="USDC", token1="WETH", sqrt_price_x96=v3_sqrt, liquidity=0).tick
    centre = v3_tick // 60 * 60
    v3 = ConcentratedPool.from_positions(
        "USDC", "WETH", v3_sqrt,
        [Position(-887220, 887220, _jitter(rng, 2 * 10**17, 30)),
         Position(centre - 3_000, centre + 3_000, _jitter(rng, 2 * 10**18, 30))],
        fee_pips=3_000, tick_spacing=60,
    )
    dai_price = Fraction(usd_per_eth * rng.randrange(997, 1_004), 1_000)  # DAI wei per WETH wei
    v4 = V4Pool.from_positions(
        "WETH", "DAI", price_to_sqrt_price_x96(dai_price),
        [Position(-887270, 887270, _jitter(rng, 10**23, 30))],
        fee_pips=500, tick_spacing=10, dynamic_fee=True, hook_fee_pips=800,
    )
    dai_usdc_dai = _jitter(rng, 8_000_000, 20) * E18
    dai_usdc = V2Pool("DAI", "USDC", dai_usdc_dai, dai_usdc_dai * rng.randrange(998, 1_003) // 1_000 // 10**12)
    v2 = V2Pool("WETH", "USDC", weth_v2, usdc_v2)
    pools: List[Pool] = [v2, v3, v4, dai_usdc]

    market = LendingMarket(
        market_id="lend:WETH/USDC",
        collateral_token="WETH",
        debt_token="USDC",
        oracle_pool_id=v2.pool_id,
        oracle_mode=oracle_mode,
        available_debt=_jitter(rng, 30_000_000, 30) * E6,
    )
    spot = market.spot({p.pool_id: p for p in pools})
    market.observations = [spot] * market.twap_window
    for i in range(6):
        collateral = _jitter(rng, 200, 80) * E18
        # health factor between 1.002 and 1.04 at the current price
        hf = Fraction(rng.randrange(1_002, 1_041), 1_000)
        debt = math.floor(collateral * spot * market.liquidation_threshold_bps / (BPS_DENOMINATOR * hf))
        market.loans[f"borrower{i}"] = Loan(collateral, debt)

    txs = generate_mempool(pools, victims, seed=seed, block=1)
    return Scenario(pools=pools, victims=txs, markets=[market], gas_price_wei=gas_price_wei, seed=seed)


# ---------------------------------------------------------------------------
# Concentrated liquidity mint / burn (UniswapV3Pool._modifyPosition)
# ---------------------------------------------------------------------------


@dataclass
class JitPosition:
    pos_id: int
    pool_id: str
    tick_lower: int
    tick_upper: int
    liquidity: int
    paid0: int
    paid1: int
    # fee growth inside the range accumulated while the position was open (X128, mod 2**256)
    fee_growth_inside0_x128: int = 0
    fee_growth_inside1_x128: int = 0


def mint_amounts(pool: ConcentratedPool, lower: int, upper: int, liquidity: int,
                 round_up: bool) -> Tuple[int, int]:
    """Token amounts for a liquidity change at the pool's current price."""
    sa, sb = get_sqrt_ratio_at_tick(lower), get_sqrt_ratio_at_tick(upper)
    if pool.tick < lower:
        return get_amount0_delta(sa, sb, liquidity, round_up), 0
    if pool.tick < upper:
        return (get_amount0_delta(pool.sqrt_price_x96, sb, liquidity, round_up),
                get_amount1_delta(sa, pool.sqrt_price_x96, liquidity, round_up))
    return 0, get_amount1_delta(sa, sb, liquidity, round_up)


def _update_tick(pool: ConcentratedPool, tick: int, net_delta: int, gross_delta: int) -> None:
    gross = pool.tick_gross.get(tick, 0) + gross_delta
    if gross < 0:
        raise ActionReverted("tick liquidity underflow")
    if gross == 0:
        pool.ticks.pop(tick, None)
        pool.tick_gross.pop(tick, None)
    else:
        pool.ticks[tick] = pool.ticks.get(tick, 0) + net_delta
        pool.tick_gross[tick] = gross
    pool._tick_array = sorted(pool.ticks)


def add_liquidity(pool: ConcentratedPool, lower: int, upper: int, liquidity: int) -> Tuple[int, int]:
    if not isinstance(pool, ConcentratedPool):
        raise ActionReverted("mint needs a concentrated-liquidity pool")
    if liquidity <= 0:
        raise ActionReverted("liquidity must be positive")
    if not (MIN_TICK <= lower < upper <= MAX_TICK):
        raise ActionReverted("invalid tick range")
    if lower % pool.tick_spacing or upper % pool.tick_spacing:
        raise ActionReverted("ticks must align with tick spacing")
    amount0, amount1 = mint_amounts(pool, lower, upper, liquidity, round_up=True)
    _update_tick(pool, lower, liquidity, liquidity)
    _update_tick(pool, upper, -liquidity, liquidity)
    if lower <= pool.tick < upper:
        pool.liquidity += liquidity
    return amount0, amount1


def remove_liquidity(pool: ConcentratedPool, lower: int, upper: int, liquidity: int) -> Tuple[int, int]:
    amount0, amount1 = mint_amounts(pool, lower, upper, liquidity, round_up=False)
    _update_tick(pool, lower, -liquidity, -liquidity)
    _update_tick(pool, upper, liquidity, -liquidity)
    if lower <= pool.tick < upper:
        pool.liquidity -= liquidity
        if pool.liquidity < 0:
            raise ActionReverted("active liquidity underflow")
    return amount0, amount1


# ---------------------------------------------------------------------------
# Execution state
# ---------------------------------------------------------------------------


class AttackState:
    """A private, mutable copy of a scenario plus the attacker's ledger.

    The ledger is signed: the attacker is assumed to have (flash) capital, and
    ``capital_used`` records the largest base-valued outlay so callers can
    enforce ``Scenario.capital_base``.
    """

    ACTOR = "attacker"

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.pools: Dict[str, Pool] = {p.pool_id: copy.deepcopy(p) for p in scenario.pools}
        self.markets: Dict[str, LendingMarket] = {m.market_id: copy.deepcopy(m) for m in scenario.markets}
        self.ledger: Dict[str, int] = {}
        self.gas_used = 0
        self.trace: List[Dict[str, Any]] = []
        self.positions: Dict[int, JitPosition] = {}
        self._next_pos = 0
        self._last_actor = ""
        self.attacker_txs = 0
        self.victims_executed: List[str] = []
        self.victim_outputs: Dict[str, int] = {}
        self.capital_used = Fraction(0)

    # -- bookkeeping ---------------------------------------------------------

    def _credit(self, token: str, amount: int) -> None:
        self.ledger[token] = self.ledger.get(token, 0) + amount
        outlay = -sum((self.scenario.value(t, a) for t, a in self.ledger.items() if a < 0), Fraction(0))
        if outlay > self.capital_used:
            self.capital_used = outlay

    def _attacker_action(self, gas: int) -> None:
        if self._last_actor != self.ACTOR:
            self.attacker_txs += 1
            self.gas_used += GAS_TX_BASE
        self._last_actor = self.ACTOR
        self.gas_used += gas

    def _pool(self, pool_id: str) -> Pool:
        try:
            return self.pools[pool_id]
        except KeyError:
            raise ActionReverted(f"unknown pool {pool_id!r}") from None

    def _market(self, market_id: str) -> LendingMarket:
        try:
            return self.markets[market_id]
        except KeyError:
            raise ActionReverted(f"unknown market {market_id!r}") from None

    # -- swaps ----------------------------------------------------------------

    def _open_positions(self, pool_id: str) -> List[JitPosition]:
        return [p for p in self.positions.values() if p.pool_id == pool_id]

    def _execute_swap(self, pool_id: str, token_in: str, amount_in: int, min_out: int) -> Tuple[SwapResult, str]:
        """Exact-input swap; the pool is unchanged when it reverts."""
        if amount_in <= 0:
            raise ActionReverted("amount_in must be positive")
        pool = self._pool(pool_id)
        try:
            zero_for_one, token_out = _direction(pool, token_in)
            quote = pool.quote_exact_in(amount_in, zero_for_one)
        except MASEError as exc:
            raise ActionReverted(str(exc)) from exc
        if not quote.fully_filled or quote.amount_out <= 0:
            raise ActionReverted("insufficient liquidity for a full fill")
        if quote.amount_out < min_out:
            raise ActionReverted(f"amount_out {quote.amount_out} below min {min_out}")
        if isinstance(pool, V2Pool):
            pool.apply_swap(quote, zero_for_one)
            return quote, token_out
        assert isinstance(pool, ConcentratedPool)
        positions = self._open_positions(pool_id)
        if positions:
            self._accrue_fees(pool, positions, amount_in, zero_for_one)
        res = pool.apply_swap(amount_in, zero_for_one)
        if res.amount_out != quote.amount_out:
            raise MASEError("internal: applied swap differs from its quote")
        return res, token_out

    def _accrue_fees(self, pool: ConcentratedPool, positions: Sequence[JitPosition], amount_in: int,
                     zero_for_one: bool) -> None:
        """Split the swap at the open positions' boundaries and credit in-range fee growth."""
        cuts = set()
        for p in positions:
            cuts.add(get_sqrt_ratio_at_tick(p.tick_lower))
            cuts.add(get_sqrt_ratio_at_tick(p.tick_upper))
        start = pool.sqrt_price_x96
        if zero_for_one:
            limits = sorted((c for c in cuts if MIN_SQRT_RATIO < c < start), reverse=True)
        else:
            limits = sorted(c for c in cuts if start < c < MAX_SQRT_RATIO)
        work = copy.deepcopy(pool)
        reference = copy.deepcopy(pool).apply_swap(amount_in, zero_for_one)
        remaining = amount_in
        total_out = 0
        growth: Dict[int, int] = {}
        for limit in [*limits, None]:
            if remaining == 0:
                break
            seg_start = work.sqrt_price_x96
            res = work.apply_swap(remaining, zero_for_one, sqrt_price_limit_x96=limit)
            remaining -= res.amount_in
            total_out += res.amount_out
            lo, hi = sorted((seg_start, work.sqrt_price_x96))
            if lo == hi:
                continue
            for p in positions:
                if get_sqrt_ratio_at_tick(p.tick_lower) <= lo and hi <= get_sqrt_ratio_at_tick(p.tick_upper):
                    growth[p.pos_id] = (growth.get(p.pos_id, 0) + (res.fee_growth_global_x128_delta or 0)) \
                        & UINT256_MAX
        if (total_out != reference.amount_out or work.sqrt_price_x96 != reference.sqrt_price_after_x96
                or work.liquidity != reference.liquidity_after):
            raise MASEError("internal: boundary-split swap differs from the single swap")
        for p in positions:
            g = growth.get(p.pos_id, 0)
            if zero_for_one:
                p.fee_growth_inside0_x128 = (p.fee_growth_inside0_x128 + g) & UINT256_MAX
            else:
                p.fee_growth_inside1_x128 = (p.fee_growth_inside1_x128 + g) & UINT256_MAX

    def swap(self, pool_id: str, token_in: str, amount_in: int, label: str = "swap",
             settle: bool = False) -> SwapResult:
        res, token_out = self._execute_swap(pool_id, token_in, amount_in, 0)
        pool = self.pools[pool_id]
        self._attacker_action(GAS_PER_HOP.get(pool_version(pool), 80_000) + GAS_PER_TICK_CROSS * res.ticks_crossed)
        self._credit(token_in, -amount_in)
        self._credit(token_out, res.amount_out)
        self.trace.append({"op": "swap", "label": label, "pool_id": pool_id, "token_in": token_in,
                           "token_out": token_out, "amount_in": amount_in, "amount_out": res.amount_out,
                           "settle": settle})
        return res

    def swap_exact_out(self, pool_id: str, token_out: str, amount_out: int, label: str = "buy",
                       settle: bool = False) -> SwapResult:
        """Buy at least ``amount_out``: quote the exact-output input, then swap it exactly in."""
        pool = self._pool(pool_id)
        t0, t1 = pool.tokens()
        if token_out not in (t0, t1):
            raise ActionReverted(f"token {token_out!r} not in pool {pool_id}")
        token_in = t1 if token_out == t0 else t0
        zero_for_one, _ = _direction(pool, token_in)
        try:
            q = pool.quote_exact_out(amount_out, zero_for_one)
        except MASEError as exc:
            raise ActionReverted(str(exc)) from exc
        if not q.fully_filled:
            raise ActionReverted("insufficient liquidity for the exact-output amount")
        return self.swap(pool_id, token_in, q.amount_in, label=label, settle=settle)

    def victim(self, tx: PendingTx) -> SwapResult:
        if len(tx.route) != 1:
            raise ActionReverted("only single-hop victims are modelled")
        if tx.tx_hash in self.victim_outputs:
            raise ActionReverted("victim already executed")
        res, token_out = self._execute_swap(tx.route[0], tx.token_in, tx.amount_in, tx.amount_out_min)
        self._last_actor = tx.sender
        self.victims_executed.append(tx.tx_hash)
        self.victim_outputs[tx.tx_hash] = res.amount_out
        self.trace.append({"op": "victim", "tx_hash": tx.tx_hash, "pool_id": tx.route[0], "token_in": tx.token_in,
                           "token_out": token_out, "amount_in": tx.amount_in, "amount_out": res.amount_out})
        return res

    # -- liquidity ------------------------------------------------------------

    def mint(self, pool_id: str, tick_lower: int, tick_upper: int, liquidity: int) -> JitPosition:
        pool = self._pool(pool_id)
        if not isinstance(pool, ConcentratedPool):
            raise ActionReverted("mint needs a concentrated-liquidity pool")
        if isinstance(pool, V4Pool) and pool.protocol_fee_pips:
            raise ActionReverted("JIT on a pool with a v4 protocol fee is not modelled")
        amount0, amount1 = add_liquidity(pool, tick_lower, tick_upper, liquidity)
        self._attacker_action(GAS_MINT)
        self._credit(pool.token0, -amount0)
        self._credit(pool.token1, -amount1)
        pos = JitPosition(self._next_pos, pool_id, tick_lower, tick_upper, liquidity, amount0, amount1)
        self.positions[pos.pos_id] = pos
        self._next_pos += 1
        self.trace.append({"op": "mint", "pool_id": pool_id, "pos_id": pos.pos_id, "tick_lower": tick_lower,
                           "tick_upper": tick_upper, "liquidity": liquidity, "amount0": amount0,
                           "amount1": amount1})
        return pos

    def burn(self, pos_id: Optional[int] = None, settle: bool = False) -> Dict[str, int]:
        if not self.positions:
            raise ActionReverted("no open position to burn")
        if pos_id is None:
            pos_id = min(self.positions)
        try:
            pos = self.positions.pop(pos_id)
        except KeyError:
            raise ActionReverted(f"unknown position {pos_id}") from None
        pool = self.pools[pos.pool_id]
        assert isinstance(pool, ConcentratedPool)
        amount0, amount1 = remove_liquidity(pool, pos.tick_lower, pos.tick_upper, pos.liquidity)
        fee0 = mul_div(pos.fee_growth_inside0_x128, pos.liquidity, Q128)
        fee1 = mul_div(pos.fee_growth_inside1_x128, pos.liquidity, Q128)
        self._attacker_action(GAS_BURN)
        self._credit(pool.token0, amount0 + fee0)
        self._credit(pool.token1, amount1 + fee1)
        out = {"amount0": amount0, "amount1": amount1, "fee0": fee0, "fee1": fee1}
        self.trace.append({"op": "burn", "pool_id": pos.pool_id, "pos_id": pos_id, "settle": settle, **out})
        return out

    # -- lending (model) ------------------------------------------------------

    def liquidate(self, market_id: str, borrower: str, repay: int) -> int:
        market = self._market(market_id)
        loan = market.loans.get(borrower)
        if loan is None:
            raise ActionReverted(f"unknown borrower {borrower!r}")
        price = market.price(self.pools)
        if market.health(loan, price) >= 1:
            raise ActionReverted("loan is healthy")
        if not 0 < repay <= market.max_repay(loan):
            raise ActionReverted("repay outside (0, close factor * debt]")
        seized = min(loan.collateral, market.seize_for(repay, price))
        if seized <= 0:
            raise ActionReverted("nothing to seize")
        loan.debt -= repay
        loan.collateral -= seized
        self._attacker_action(GAS_LIQUIDATE)
        self._credit(market.debt_token, -repay)
        self._credit(market.collateral_token, seized)
        self.trace.append({"op": "liquidate", "market_id": market_id, "borrower": borrower, "repay": repay,
                           "seized": seized, "oracle_price": str(price)})
        return seized

    def deposit(self, market_id: str, amount: int) -> None:
        market = self._market(market_id)
        if amount <= 0:
            raise ActionReverted("deposit must be positive")
        loan = market.loans.setdefault(self.ACTOR, Loan(0, 0))
        loan.collateral += amount
        self._attacker_action(GAS_DEPOSIT)
        self._credit(market.collateral_token, -amount)
        self.trace.append({"op": "deposit", "market_id": market_id, "amount": amount})

    def borrow(self, market_id: str, amount: int) -> None:
        market = self._market(market_id)
        loan = market.loans.get(self.ACTOR)
        if loan is None or amount <= 0:
            raise ActionReverted("borrow needs collateral and a positive amount")
        price = market.price(self.pools)
        limit = math.floor(loan.collateral * price * market.ltv_bps / BPS_DENOMINATOR)
        if loan.debt + amount > limit:
            raise ActionReverted("borrow exceeds the loan-to-value limit")
        if amount > market.available_debt:
            raise ActionReverted("not enough lendable liquidity")
        market.available_debt -= amount
        loan.debt += amount
        self._attacker_action(GAS_BORROW)
        self._credit(market.debt_token, amount)
        self.trace.append({"op": "borrow", "market_id": market_id, "amount": amount, "oracle_price": str(price)})

    def max_borrow(self, market_id: str) -> int:
        market = self._market(market_id)
        loan = market.loans.get(self.ACTOR)
        if loan is None:
            return 0
        limit = math.floor(loan.collateral * market.price(self.pools) * market.ltv_bps / BPS_DENOMINATOR)
        return max(0, min(limit - loan.debt, market.available_debt))

    # -- settlement -----------------------------------------------------------

    def _base_pools(self, token: str) -> List[Pool]:
        base = self.scenario.base
        return [p for p in self.pools.values() if set(p.tokens()) == {token, base}]

    def settle(self) -> None:
        """Burn open positions, then swap every non-base balance to base through
        the direct pool that gives the best price. Shortfalls are bought with
        an exact-output quote. A balance that cannot be settled stays in the
        ledger and is valued at reference prices."""
        for pos_id in sorted(self.positions):
            self.burn(pos_id, settle=True)
        base = self.scenario.base
        for token in sorted(self.ledger):
            amount = self.ledger[token]
            if token == base or amount == 0:
                continue
            pools = self._base_pools(token)
            if amount > 0:
                quotes = []
                for p in pools:
                    zfo, _ = _direction(p, token)
                    try:
                        q = p.quote_exact_in(amount, zfo)
                    except MASEError:
                        continue
                    if q.fully_filled and q.amount_out > 0:
                        quotes.append((q.amount_out, p.pool_id))
                if quotes:
                    self.swap(max(quotes)[1], token, amount, label="settle", settle=True)
            else:
                costs = []
                for p in pools:
                    zfo, _ = _direction(p, base)
                    try:
                        q = p.quote_exact_out(-amount, zfo)
                    except MASEError:
                        continue
                    if q.fully_filled:
                        costs.append((q.amount_in, p.pool_id))
                if costs:
                    self.swap_exact_out(min(costs)[1], token, -amount, label="settle", settle=True)

    # -- results --------------------------------------------------------------

    def gas_cost_base(self) -> int:
        cost_native = self.gas_used * self.scenario.gas_price_wei
        return math.ceil(self.scenario.value("WETH", cost_native)) if "WETH" in self.scenario.prices else 0

    def result(self, kind: str, label: str, status: str = "exact", **extra: Any) -> "AttackResult":
        base = self.scenario.base
        base_delta = self.ledger.get(base, 0)
        residual = {t: a for t, a in self.ledger.items() if t != base and a != 0}
        residual_value = math.floor(sum((self.scenario.value(t, a) for t, a in residual.items()), Fraction(0)))
        gas_cost = self.gas_cost_base()
        return AttackResult(
            kind=kind,
            label=label,
            status=status,
            steps=list(self.trace),
            ledger={t: a for t, a in sorted(self.ledger.items()) if a != 0},
            base_delta=base_delta,
            residual_value_base=residual_value,
            gas_used=self.gas_used,
            gas_cost_base=gas_cost,
            net_profit_base=base_delta + residual_value - gas_cost,
            flat=not residual,
            capital_used_base=math.ceil(self.capital_used),
            victims=list(self.victims_executed),
            extra=extra,
        )


@dataclass
class AttackResult:
    kind: str
    label: str
    status: str  # "exact" (AMM math) or "model" (uses the lending model)
    steps: List[Dict[str, Any]]
    ledger: Dict[str, int]
    base_delta: int
    residual_value_base: int
    gas_used: int
    gas_cost_base: int
    net_profit_base: int
    flat: bool
    capital_used_base: int
    victims: List[str]
    victim_loss: Dict[str, int] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "net_profit_base": self.net_profit_base,
            "base_delta": self.base_delta,
            "residual_value_base": self.residual_value_base,
            "flat": self.flat,
            "gas_used": self.gas_used,
            "gas_cost_base": self.gas_cost_base,
            "capital_used_base": self.capital_used_base,
            "ledger": dict(self.ledger),
            "victims": list(self.victims),
            "victim_loss": dict(self.victim_loss),
            "steps": [dict(s) for s in self.steps],
            "extra": dict(self.extra),
        }


# ---------------------------------------------------------------------------
# Helpers shared by the attack searches
# ---------------------------------------------------------------------------


def maximize_int(f: Callable[[int], Optional[int]], lo: int, hi: int, grid: int = 24,
                 refine: int = 40) -> Optional[Tuple[int, int]]:
    """Best ``(x, f(x))`` over a geometric grid of [lo, hi], then an integer
    ternary search between the best point's neighbours. ``f`` returns None
    for infeasible points. Returns None when nothing is feasible."""
    sizes = geometric_grid(max(1, lo), max(1, hi), grid)
    if not sizes:
        return None
    cache: Dict[int, Optional[int]] = {}

    def g(x: int) -> Optional[int]:
        if x not in cache:
            cache[x] = f(x)
        return cache[x]

    scores = [g(x) for x in sizes]
    feasible = [i for i, s in enumerate(scores) if s is not None]
    if not feasible:
        return None
    best_i = max(feasible, key=lambda i: (scores[i], -i))
    left = sizes[best_i - 1] if best_i > 0 else sizes[0]
    right = sizes[best_i + 1] if best_i + 1 < len(sizes) else sizes[-1]
    for _ in range(refine):
        if right - left < 3:
            break
        m1 = left + (right - left) // 3
        m2 = right - (right - left) // 3
        p1, p2 = g(m1), g(m2)
        if (p1 if p1 is not None else -math.inf) < (p2 if p2 is not None else -math.inf):
            left = m1
        else:
            right = m2
    x, s = max(((x, s) for x, s in cache.items() if s is not None), key=lambda kv: (kv[1], -kv[0]))
    return x, s


def victim_baseline(scenario: Scenario, victims: Sequence[PendingTx]) -> Dict[str, int]:
    """Each victim's output when the victims run in the same order with no attacker."""
    state = AttackState(scenario)
    out: Dict[str, int] = {}
    for tx in victims:
        try:
            out[tx.tx_hash] = state.victim(tx).amount_out
        except ActionReverted:
            continue
    return out


def _with_victim_loss(result: AttackResult, scenario: Scenario, victims: Sequence[PendingTx]) -> AttackResult:
    executed = [v for v in victims if v.tx_hash in result.victims]
    base = victim_baseline(scenario, executed)
    outs = {s["tx_hash"]: s["amount_out"] for s in result.steps if s["op"] == "victim"}
    result.victim_loss = {h: base[h] - outs[h] for h in outs if h in base}
    return result


def _single_hop(tx: PendingTx) -> bool:
    return len(tx.route) == 1


def _capital_ok(state: AttackState) -> bool:
    return state.capital_used <= state.scenario.capital_base


# ---------------------------------------------------------------------------
# 1. Multi-victim sandwich
# ---------------------------------------------------------------------------


def _run_multi_sandwich(scenario: Scenario, pool_id: str, token_in: str, victims: Sequence[PendingTx],
                        front: int) -> Optional[AttackState]:
    state = AttackState(scenario)
    try:
        res = state.swap(pool_id, token_in, front, label="front-run")
    except ActionReverted:
        return None
    for tx in victims:
        try:
            state.victim(tx)
        except ActionReverted:
            continue  # this victim would revert at this size; it is left out of the bundle
    if not state.victims_executed:
        return None
    try:
        state.swap(pool_id, state.trace[0]["token_out"], res.amount_out, label="back-run")
        state.settle()
    except ActionReverted:
        return None
    return state


def multi_victim_sandwich(scenario: Scenario, pool_id: str, token_in: str, min_victims: int = 2,
                          grid: int = 24, refine: int = 40) -> Optional[AttackResult]:
    """One front-run ahead of every pending same-direction swap on the pool,
    one back-run after the last. Victims that would revert at a given size
    are dropped from the bundle; the size is optimised for net profit."""
    victims = [v for v in scenario.victims if _single_hop(v) and v.route[0] == pool_id and v.token_in == token_in]
    if len(victims) < min_victims:
        return None
    pool = scenario.pool(pool_id)
    zero_for_one, _ = _direction(pool, token_in)
    hi = input_reserve_scale(pool, zero_for_one) // 10

    def score(a: int) -> Optional[int]:
        st = _run_multi_sandwich(scenario, pool_id, token_in, victims, a)
        if st is None or len(st.victims_executed) < min_victims or not _capital_ok(st):
            return None
        return st.result("multi_victim_sandwich", "").net_profit_base

    best = maximize_int(score, max(1, min(v.amount_in for v in victims) // 100), hi, grid, refine)
    if best is None or best[1] <= 0:
        return None
    state = _run_multi_sandwich(scenario, pool_id, token_in, victims, best[0])
    assert state is not None
    kind = "multi_victim_sandwich" if len(state.victims_executed) >= 2 else "sandwich"
    res = state.result(kind, f"{kind} on {pool_id} ({len(state.victims_executed)} victims)",
                       pending_same_direction=len(victims))
    return _with_victim_loss(res, scenario, victims)


# ---------------------------------------------------------------------------
# 2. JIT liquidity
# ---------------------------------------------------------------------------


def _jit_range(pool: ConcentratedPool, tx: PendingTx) -> Optional[Tuple[int, int]]:
    zero_for_one, _ = _direction(pool, tx.token_in)
    try:
        q = pool.quote_exact_in(tx.amount_in, zero_for_one)
    except MASEError:
        return None
    lo_tick = min(pool.tick, q.tick_after if q.tick_after is not None else pool.tick)
    hi_tick = max(pool.tick, q.tick_after if q.tick_after is not None else pool.tick)
    s = pool.tick_spacing
    lower = (lo_tick // s) * s - s
    upper = (hi_tick // s) * s + 2 * s
    lower = max(lower, -(MAX_TICK // s) * s)
    upper = min(upper, (MAX_TICK // s) * s)
    return lower, upper


def _run_jit(scenario: Scenario, tx: PendingTx, lower: int, upper: int, liquidity: int,
             hedge: bool) -> Optional[AttackState]:
    state = AttackState(scenario)
    pool_id = tx.route[0]
    try:
        state.mint(pool_id, lower, upper, liquidity)
        state.victim(tx)
        state.burn()
        if hedge:
            state.settle()
    except ActionReverted:
        return None
    return state


def jit_liquidity(scenario: Scenario, tx: PendingTx, hedge: bool = False, grid: int = 20,
                  refine: int = 30) -> Optional[AttackResult]:
    """Mint a position that brackets the victim's price path, let the victim
    trade through it, burn and collect. With ``hedge`` the resulting inventory
    is swapped back to base (that turns it into JIT + back-run); without it
    the inventory change is valued at reference prices."""
    if not _single_hop(tx):
        return None
    pool = scenario.pool(tx.route[0])
    if not isinstance(pool, ConcentratedPool) or pool.liquidity <= 0:
        return None
    rng = _jit_range(pool, tx)
    if rng is None:
        return None
    lower, upper = rng

    def score(L: int) -> Optional[int]:
        st = _run_jit(scenario, tx, lower, upper, L, hedge)
        if st is None or not _capital_ok(st):
            return None
        return st.result("jit_liquidity", "").net_profit_base

    best = maximize_int(score, max(1, pool.liquidity // 100), pool.liquidity * 200, grid, refine)
    if best is None or best[1] <= 0:
        return None
    state = _run_jit(scenario, tx, lower, upper, best[0], hedge)
    assert state is not None
    burn = next(s for s in state.trace if s["op"] == "burn")
    res = state.result("jit_liquidity", f"JIT liquidity around {tx.tx_hash[:10]} on {pool.pool_id}",
                       tick_lower=lower, tick_upper=upper, liquidity=best[0],
                       fees={"fee0": burn["fee0"], "fee1": burn["fee1"]}, hedged=hedge,
                       share_of_active_liquidity=str(Fraction(best[0], best[0] + pool.liquidity)))
    return _with_victim_loss(res, scenario, [tx])


# ---------------------------------------------------------------------------
# 3. Back-run arbitrage (2- and 3-pool cycles through the victim's pool)
# ---------------------------------------------------------------------------


def _cycles_through(scenario: Scenario, pool_id: str) -> List[Tuple[Tuple[str, str], ...]]:
    """Cycles as ((pool_id, token_in), ...) that use ``pool_id`` and start at the base
    token when the cycle contains it."""
    pools = scenario.pools
    out: List[Tuple[Tuple[str, str], ...]] = []
    seen = set()

    def other(p: Pool, t: str) -> str:
        t0, t1 = p.tokens()
        return t1 if t == t0 else t0

    for start_pool in pools:
        for t in start_pool.tokens():
            # length 2
            u = other(start_pool, t)
            for q in pools:
                if q is start_pool or set(q.tokens()) != set(start_pool.tokens()):
                    continue
                cyc = ((start_pool.pool_id, t), (q.pool_id, u))
                out.append(cyc)
            # length 3
            for q in pools:
                if q is start_pool or u not in q.tokens():
                    continue
                w = other(q, u)
                if w == t:
                    continue
                for r in pools:
                    if r is start_pool or r is q or set(r.tokens()) != {w, t}:
                        continue
                    out.append(((start_pool.pool_id, t), (q.pool_id, u), (r.pool_id, w)))
    base = scenario.base
    result = []
    for cyc in out:
        if pool_id not in (c[0] for c in cyc):
            continue
        tokens = [c[1] for c in cyc]
        if base in tokens:
            k = tokens.index(base)
            cyc = cyc[k:] + cyc[:k]
        key = cyc
        if key in seen:
            continue
        seen.add(key)
        result.append(cyc)
    return result


def _run_cycle(state: AttackState, cycle: Sequence[Tuple[str, str]], amount: int, label: str) -> None:
    for i, (pid, token_in) in enumerate(cycle):
        res = state.swap(pid, token_in, amount, label=f"{label}{i}")
        amount = res.amount_out


def _run_backrun(scenario: Scenario, tx: PendingTx, cycle: Sequence[Tuple[str, str]],
                 amount: int) -> Optional[AttackState]:
    state = AttackState(scenario)
    try:
        state.victim(tx)
        _run_cycle(state, cycle, amount, "arb")
    except ActionReverted:
        return None
    return state


def backrun_arbitrage(scenario: Scenario, tx: PendingTx, grid: int = 20, refine: int = 40) -> Optional[AttackResult]:
    if not _single_hop(tx):
        return None
    best: Optional[Tuple[int, Tuple[Tuple[str, str], ...], int]] = None
    for cycle in _cycles_through(scenario, tx.route[0]):
        pool = scenario.pool(cycle[0][0])
        zfo, _ = _direction(pool, cycle[0][1])
        hi = input_reserve_scale(pool, zfo) // 20

        def score(a: int, cycle: Tuple[Tuple[str, str], ...] = cycle) -> Optional[int]:
            st = _run_backrun(scenario, tx, cycle, a)
            if st is None or not _capital_ok(st):
                return None
            return st.result("backrun_arbitrage", "").net_profit_base

        found = maximize_int(score, max(1, hi // 10**6), hi, grid, refine)
        if found is not None and (best is None or found[1] > best[0]):
            best = (found[1], cycle, found[0])
    if best is None or best[0] <= 0:
        return None
    _, cycle, amount = best
    state = _run_backrun(scenario, tx, cycle, amount)
    assert state is not None
    hops = " -> ".join(f"{t}@{p}" for p, t in cycle)
    res = state.result("backrun_arbitrage", f"back-run arbitrage after {tx.tx_hash[:10]}: {hops}",
                       cycle=[list(c) for c in cycle], amount_in=amount)
    return _with_victim_loss(res, scenario, [tx])


# ---------------------------------------------------------------------------
# 4. Liquidation back-run (model)
# ---------------------------------------------------------------------------


def _run_liquidation(scenario: Scenario, tx: Optional[PendingTx], market_id: str, borrower: str,
                     repay: int) -> Optional[AttackState]:
    state = AttackState(scenario)
    try:
        if tx is not None:
            state.victim(tx)
        state.liquidate(market_id, borrower, repay)
        state.settle()
    except ActionReverted:
        return None
    return state


def liquidation_backrun(scenario: Scenario, tx: PendingTx, grid: int = 16,
                        refine: int = 30) -> Optional[AttackResult]:
    """Loans healthy before the victim and unhealthy after it are liquidated
    right behind it; the seized collateral is sold for base."""
    if not _single_hop(tx):
        return None
    best: Optional[Tuple[int, str, str, int]] = None
    for market in scenario.markets:
        if market.oracle_pool_id != tx.route[0]:
            continue
        before = AttackState(scenario)
        price_before = market.price(before.pools)
        after = AttackState(scenario)
        try:
            after.victim(tx)
        except ActionReverted:
            return None
        price_after = after.markets[market.market_id].price(after.pools)
        for borrower, loan in sorted(market.loans.items()):
            if market.health(loan, price_before) < 1 or market.health(loan, price_after) >= 1:
                continue

            def score(r: int, borrower: str = borrower, mid: str = market.market_id) -> Optional[int]:
                st = _run_liquidation(scenario, tx, mid, borrower, r)
                if st is None or not _capital_ok(st):
                    return None
                return st.result("liquidation_backrun", "").net_profit_base

            found = maximize_int(score, 1, market.max_repay(loan), grid, refine)
            if found is not None and (best is None or found[1] > best[0]):
                best = (found[1], market.market_id, borrower, found[0])
    if best is None or best[0] <= 0:
        return None
    _, mid, borrower, repay = best
    state = _run_liquidation(scenario, tx, mid, borrower, repay)
    assert state is not None
    res = state.result("liquidation_backrun", f"liquidation of {borrower} behind {tx.tx_hash[:10]}",
                       status="model", market_id=mid, borrower=borrower, repay=repay)
    return _with_victim_loss(res, scenario, [tx])


# ---------------------------------------------------------------------------
# 5. Oracle manipulation (model): spot oracle vs TWAP
# ---------------------------------------------------------------------------


def _run_oracle(scenario: Scenario, market_id: str, push: int) -> Optional[AttackState]:
    state = AttackState(scenario)
    market = state.markets[market_id]
    try:
        res = state.swap(market.oracle_pool_id, market.debt_token, push, label="push-oracle")
        state.deposit(market_id, res.amount_out)
        amount = state.max_borrow(market_id)
        if amount <= 0:
            return None
        state.borrow(market_id, amount)
        state.settle()
    except ActionReverted:
        return None
    return state


def _best_oracle(scenario: Scenario, market_id: str, grid: int, refine: int) -> Optional[Tuple[int, int]]:
    market = scenario.market(market_id)
    pool = scenario.pool(market.oracle_pool_id)
    zfo, _ = _direction(pool, market.debt_token)

    def score(a: int) -> Optional[int]:
        st = _run_oracle(scenario, market_id, a)
        if st is None or not _capital_ok(st):
            return None
        return st.result("oracle_manipulation", "").net_profit_base

    return maximize_int(score, max(1, input_reserve_scale(pool, zfo) // 10_000), input_reserve_scale(pool, zfo) * 3,
                        grid, refine)


def oracle_manipulation(scenario: Scenario, market_id: str, grid: int = 24,
                        refine: int = 40) -> Optional[AttackResult]:
    """Buy collateral on the oracle pool (pushing its price up), deposit it,
    borrow the maximum against the inflated price and keep the loan. The same
    search is run against a TWAP-oracle copy of the market; ``extra`` holds
    both, so the report shows whether the TWAP removes the profit."""
    market = scenario.market(market_id)
    twap_scn = copy.deepcopy(scenario)
    twap_scn.market(market_id).oracle_mode = "twap"
    spot_scn = copy.deepcopy(scenario)
    spot_scn.market(market_id).oracle_mode = "spot"
    twap_best = _best_oracle(twap_scn, market_id, grid, refine)
    spot_best = _best_oracle(spot_scn, market_id, grid, refine)
    best = spot_best if market.oracle_mode == "spot" else twap_best
    if best is None or best[1] <= 0:
        return None
    state = _run_oracle(scenario, market_id, best[0])
    assert state is not None
    return state.result(
        "oracle_manipulation",
        f"{market.oracle_mode} oracle manipulation of {market_id} via {market.oracle_pool_id}",
        status="model",
        oracle_mode=market.oracle_mode,
        push_amount=best[0],
        spot_oracle_best_profit=spot_best[1] if spot_best else None,
        twap_oracle_best_profit=twap_best[1] if twap_best else None,
        twap_defends=twap_best is None or twap_best[1] <= 0,
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def run_attack_catalog(scenario: Scenario, max_victims: Optional[int] = None) -> Dict[str, Any]:
    """Run every attack type on the scenario and summarise the results."""
    victims = [v for v in scenario.victims if _single_hop(v)]
    if max_victims is not None:
        victims = victims[:max_victims]
    results: Dict[str, List[AttackResult]] = {k: [] for k in ATTACK_TYPES}
    for tx in victims:
        single = multi_victim_sandwich(
            Scenario(scenario.pools, [tx], scenario.markets, scenario.base, scenario.gas_price_wei,
                     scenario.capital_base, scenario.seed),
            tx.route[0], tx.token_in, min_victims=1)
        if single is not None:
            single.kind = "sandwich"
            results["sandwich"].append(single)
        for fn, key in ((jit_liquidity, "jit_liquidity"), (backrun_arbitrage, "backrun_arbitrage"),
                        (liquidation_backrun, "liquidation_backrun")):
            r = fn(scenario, tx)
            if r is not None:
                results[key].append(r)
    groups = sorted({(v.route[0], v.token_in) for v in victims})
    for pool_id, token_in in groups:
        r = multi_victim_sandwich(scenario, pool_id, token_in, min_victims=2)
        if r is not None and r.kind == "multi_victim_sandwich":
            results["multi_victim_sandwich"].append(r)
    for market in scenario.markets:
        r = oracle_manipulation(scenario, market.market_id)
        if r is not None:
            results["oracle_manipulation"].append(r)

    summary = {}
    for kind in ATTACK_TYPES:
        rs = results[kind]
        summary[kind] = {
            "found": len(rs),
            "best_net_profit_base": max((r.net_profit_base for r in rs), default=0),
            "total_net_profit_base": sum(r.net_profit_base for r in rs),
            "total_victim_loss": _sum_losses(rs),
        }
    return {
        "scenario": scenario.describe(),
        "summary": summary,
        "results": {k: [r.to_dict() for r in results[k]] for k in ATTACK_TYPES},
    }


def _sum_losses(results: Sequence[AttackResult]) -> Dict[str, int]:
    """Victims' lost output per output token (raw units; negative = victims gained, e.g. under JIT)."""
    out: Dict[str, int] = {}
    for r in results:
        token_of = {s["tx_hash"]: s["token_out"] for s in r.steps if s["op"] == "victim"}
        for h, loss in r.victim_loss.items():
            out[token_of[h]] = out.get(token_of[h], 0) + loss
    return dict(sorted(out.items()))
