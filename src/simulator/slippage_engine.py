"""Layer 02: Multi-Asset Slippage Extraction (MASE) engine.

Integer-exact AMM math for slippage analysis inside the offline simulator:

* Uniswap v2 constant-product pools (x * y = k with the 0.3% input fee).
* Uniswap v3 / v4 concentrated liquidity: TickMath, SqrtPriceMath and
  SwapMath ported to Python integers (Q64.96 ``sqrtPriceX96``), plus a tick
  array swap loop that crosses initialized ticks.
* ``evaluate_trade`` / ``evaluate_route``: given a trade volume, return the
  price impact and the slippage vector (token deltas, minimum output at a
  tolerance, and the gap between quoted and minimum output).
* ``MASEEngine.evaluate_payload``: entry point for execution payloads coming
  from the mempool layer.
* Layer 03: ``MASEEngine.evaluate_batch(workers=...)`` and
  ``sweep_trade_sizes`` spread batches and size sweeps over a capped
  ``multiprocessing`` pool (see ``simulator.parallel``).

All quoting functions are pure: pool objects are never mutated unless
``apply_swap`` is called explicitly.

Python integers are unbounded, so uint256 overflow branches from the Solidity
sources are not reproduced. The v3/v4 swap loop walks a sorted tick array
instead of the on-chain tick bitmap, so a swap that the contracts would split
at an uninitialized word boundary is computed as one step here; results can
differ from chain by rounding dust (a few wei) in that case.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Q96 = 1 << 96
Q128 = 1 << 128
Q192 = 1 << 192
UINT256_MAX = (1 << 256) - 1

MIN_TICK = -887272
MAX_TICK = 887272
MIN_SQRT_RATIO = 4295128739
MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342

FEE_PIPS_DENOMINATOR = 1_000_000  # v3 / v4 fees are in hundredths of a bip
BPS_DENOMINATOR = 10_000
V2_FEE_BPS = 30  # Uniswap v2: 0.3%, i.e. the 997/1000 factor
V4_MAX_LP_FEE = 1_000_000


class MASEError(ValueError):
    """Raised for invalid pool state or trade parameters."""


# ---------------------------------------------------------------------------
# Full-precision helpers (FullMath / UnsafeMath equivalents)
# ---------------------------------------------------------------------------


def mul_div(a: int, b: int, denominator: int) -> int:
    if denominator <= 0:
        raise MASEError("mul_div: denominator must be positive")
    return (a * b) // denominator


def mul_div_rounding_up(a: int, b: int, denominator: int) -> int:
    if denominator <= 0:
        raise MASEError("mul_div_rounding_up: denominator must be positive")
    return -((-(a * b)) // denominator)


def div_rounding_up(a: int, b: int) -> int:
    if b <= 0:
        raise MASEError("div_rounding_up: divisor must be positive")
    return -((-a) // b)


# ---------------------------------------------------------------------------
# Uniswap v2: constant product
# ---------------------------------------------------------------------------


def v2_get_amount_out(amount_in: int, reserve_in: int, reserve_out: int, fee_bps: int = V2_FEE_BPS) -> int:
    """UniswapV2Library.getAmountOut, generalised to any fee in bps."""
    if amount_in <= 0:
        raise MASEError("v2: amount_in must be positive")
    if reserve_in <= 0 or reserve_out <= 0:
        raise MASEError("v2: insufficient liquidity")
    amount_in_with_fee = amount_in * (BPS_DENOMINATOR - fee_bps)
    numerator = amount_in_with_fee * reserve_out
    denominator = reserve_in * BPS_DENOMINATOR + amount_in_with_fee
    return numerator // denominator


def v2_get_amount_in(amount_out: int, reserve_in: int, reserve_out: int, fee_bps: int = V2_FEE_BPS) -> int:
    """UniswapV2Library.getAmountIn, generalised to any fee in bps."""
    if amount_out <= 0:
        raise MASEError("v2: amount_out must be positive")
    if reserve_in <= 0 or reserve_out <= 0 or amount_out >= reserve_out:
        raise MASEError("v2: insufficient liquidity")
    numerator = reserve_in * amount_out * BPS_DENOMINATOR
    denominator = (reserve_out - amount_out) * (BPS_DENOMINATOR - fee_bps)
    return numerator // denominator + 1


@dataclass
class V2Pool:
    """Uniswap v2 pair. ``reserve0``/``reserve1`` are raw token units."""

    token0: str
    token1: str
    reserve0: int
    reserve1: int
    fee_bps: int = V2_FEE_BPS
    pool_id: str = ""

    def __post_init__(self) -> None:
        if self.reserve0 <= 0 or self.reserve1 <= 0:
            raise MASEError("v2: reserves must be positive")
        if not 0 <= self.fee_bps < BPS_DENOMINATOR:
            raise MASEError("v2: fee_bps out of range")
        if not self.pool_id:
            self.pool_id = f"v2:{self.token0}/{self.token1}"

    @property
    def k(self) -> int:
        return self.reserve0 * self.reserve1

    def tokens(self) -> Tuple[str, str]:
        return self.token0, self.token1

    def _reserves(self, zero_for_one: bool) -> Tuple[int, int]:
        return (self.reserve0, self.reserve1) if zero_for_one else (self.reserve1, self.reserve0)

    def spot_price(self, zero_for_one: bool) -> Fraction:
        """Marginal price (output per input) before fees."""
        r_in, r_out = self._reserves(zero_for_one)
        return Fraction(r_out, r_in)

    def quote_exact_in(self, amount_in: int, zero_for_one: bool) -> "SwapResult":
        r_in, r_out = self._reserves(zero_for_one)
        amount_out = v2_get_amount_out(amount_in, r_in, r_out, self.fee_bps)
        new_in, new_out = r_in + amount_in, r_out - amount_out
        fee = div_rounding_up(amount_in * self.fee_bps, BPS_DENOMINATOR)
        return SwapResult(
            amount_in=amount_in,
            amount_out=amount_out,
            fee_paid=fee,
            price_before=Fraction(r_out, r_in),
            price_after=Fraction(new_out, new_in),
            fully_filled=True,
            ticks_crossed=0,
        )

    def quote_exact_out(self, amount_out: int, zero_for_one: bool) -> "SwapResult":
        r_in, r_out = self._reserves(zero_for_one)
        amount_in = v2_get_amount_in(amount_out, r_in, r_out, self.fee_bps)
        return SwapResult(
            amount_in=amount_in,
            amount_out=amount_out,
            fee_paid=div_rounding_up(amount_in * self.fee_bps, BPS_DENOMINATOR),
            price_before=Fraction(r_out, r_in),
            price_after=Fraction(r_out - amount_out, r_in + amount_in),
            fully_filled=True,
            ticks_crossed=0,
        )

    def apply_swap(self, result: "SwapResult", zero_for_one: bool) -> None:
        if zero_for_one:
            self.reserve0 += result.amount_in
            self.reserve1 -= result.amount_out
        else:
            self.reserve1 += result.amount_in
            self.reserve0 -= result.amount_out


# ---------------------------------------------------------------------------
# Uniswap v3 / v4: TickMath
# ---------------------------------------------------------------------------

_TICK_MAGIC = (
    (0x2, 0xFFF97272373D413259A46990580E213A),
    (0x4, 0xFFF2E50F5F656932EF12357CF3C7FDCC),
    (0x8, 0xFFE5CACA7E10E4E61C3624EAA0941CD0),
    (0x10, 0xFFCB9843D60F6159C9DB58835C926644),
    (0x20, 0xFF973B41FA98C081472E6896DFB254C0),
    (0x40, 0xFF2EA16466C96A3843EC78B326B52861),
    (0x80, 0xFE5DEE046A99A2A811C461F1969C3053),
    (0x100, 0xFCBE86C7900A88AEDCFFC83B479AA3A4),
    (0x200, 0xF987A7253AC413176F2B074CF7815E54),
    (0x400, 0xF3392B0822B70005940C7A398E4B70F3),
    (0x800, 0xE7159475A2C29B7443B29C7FA6E889D9),
    (0x1000, 0xD097F3BDFD2022B8845AD8F792AA5825),
    (0x2000, 0xA9F746462D870FDF8A65DC1F90E061E5),
    (0x4000, 0x70D869A156D2A1B890BB3DF62BAF32F7),
    (0x8000, 0x31BE135F97D08FD981231505542FCFA6),
    (0x10000, 0x9AA508B5B7A84E1C677DE54F3E99BC9),
    (0x20000, 0x5D6AF8DEDB81196699C329225EE604),
    (0x40000, 0x2216E584F5FA1EA926041BEDFE98),
    (0x80000, 0x48A170391F7DC42444E8FA2),
)


def get_sqrt_ratio_at_tick(tick: int) -> int:
    """TickMath.getSqrtRatioAtTick: sqrt(1.0001^tick) as a Q64.96."""
    abs_tick = -tick if tick < 0 else tick
    if abs_tick > MAX_TICK:
        raise MASEError("tick out of range")
    ratio = 0xFFFCB933BD6FAD37AA2D162D1A594001 if abs_tick & 0x1 else Q128
    for mask, magic in _TICK_MAGIC:
        if abs_tick & mask:
            ratio = (ratio * magic) >> 128
    if tick > 0:
        ratio = UINT256_MAX // ratio
    return (ratio >> 32) + (0 if ratio % (1 << 32) == 0 else 1)


def get_tick_at_sqrt_ratio(sqrt_price_x96: int) -> int:
    """TickMath.getTickAtSqrtRatio: greatest tick with ratio(tick) <= price."""
    if not MIN_SQRT_RATIO <= sqrt_price_x96 < MAX_SQRT_RATIO:
        raise MASEError("sqrt price out of range")
    lo, hi = MIN_TICK, MAX_TICK
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        if get_sqrt_ratio_at_tick(mid) <= sqrt_price_x96:
            lo = mid
        else:
            hi = mid - 1
    return lo


def sqrt_price_x96_to_price(sqrt_price_x96: int) -> Fraction:
    """token1 per token0 (raw units) for a Q64.96 square-root price."""
    return Fraction(sqrt_price_x96 * sqrt_price_x96, Q192)


def price_to_sqrt_price_x96(price: Union[Fraction, int, float]) -> int:
    """Floor of sqrt(price) * 2^96 for a token1/token0 price in raw units."""
    from math import isqrt

    p = Fraction(price)
    if p <= 0:
        raise MASEError("price must be positive")
    return isqrt(p.numerator * Q192 // p.denominator)


# ---------------------------------------------------------------------------
# Uniswap v3 / v4: SqrtPriceMath
# ---------------------------------------------------------------------------


def get_next_sqrt_price_from_amount0_rounding_up(sqrt_p: int, liquidity: int, amount: int, add: bool) -> int:
    if amount == 0:
        return sqrt_p
    numerator1 = liquidity << 96
    product = amount * sqrt_p
    if add:
        return mul_div_rounding_up(numerator1, sqrt_p, numerator1 + product)
    if numerator1 <= product:
        raise MASEError("amount0 output exceeds virtual reserves")
    return mul_div_rounding_up(numerator1, sqrt_p, numerator1 - product)


def get_next_sqrt_price_from_amount1_rounding_down(sqrt_p: int, liquidity: int, amount: int, add: bool) -> int:
    if add:
        return sqrt_p + (amount << 96) // liquidity
    quotient = div_rounding_up(amount << 96, liquidity)
    if sqrt_p <= quotient:
        raise MASEError("amount1 output exceeds virtual reserves")
    return sqrt_p - quotient


def get_next_sqrt_price_from_input(sqrt_p: int, liquidity: int, amount_in: int, zero_for_one: bool) -> int:
    if sqrt_p <= 0 or liquidity <= 0:
        raise MASEError("invalid price or liquidity")
    if zero_for_one:
        return get_next_sqrt_price_from_amount0_rounding_up(sqrt_p, liquidity, amount_in, True)
    return get_next_sqrt_price_from_amount1_rounding_down(sqrt_p, liquidity, amount_in, True)


def get_next_sqrt_price_from_output(sqrt_p: int, liquidity: int, amount_out: int, zero_for_one: bool) -> int:
    if sqrt_p <= 0 or liquidity <= 0:
        raise MASEError("invalid price or liquidity")
    if zero_for_one:
        return get_next_sqrt_price_from_amount1_rounding_down(sqrt_p, liquidity, amount_out, False)
    return get_next_sqrt_price_from_amount0_rounding_up(sqrt_p, liquidity, amount_out, False)


def get_amount0_delta(sqrt_a: int, sqrt_b: int, liquidity: int, round_up: bool) -> int:
    if sqrt_a > sqrt_b:
        sqrt_a, sqrt_b = sqrt_b, sqrt_a
    if sqrt_a <= 0:
        raise MASEError("sqrt price must be positive")
    numerator1 = liquidity << 96
    numerator2 = sqrt_b - sqrt_a
    if round_up:
        return div_rounding_up(mul_div_rounding_up(numerator1, numerator2, sqrt_b), sqrt_a)
    return mul_div(numerator1, numerator2, sqrt_b) // sqrt_a


def get_amount1_delta(sqrt_a: int, sqrt_b: int, liquidity: int, round_up: bool) -> int:
    if sqrt_a > sqrt_b:
        sqrt_a, sqrt_b = sqrt_b, sqrt_a
    if round_up:
        return mul_div_rounding_up(liquidity, sqrt_b - sqrt_a, Q96)
    return mul_div(liquidity, sqrt_b - sqrt_a, Q96)


# ---------------------------------------------------------------------------
# Uniswap v3 / v4: SwapMath
# ---------------------------------------------------------------------------


def compute_swap_step(
    sqrt_current: int,
    sqrt_target: int,
    liquidity: int,
    amount_remaining: int,
    fee_pips: int,
) -> Tuple[int, int, int, int]:
    """SwapMath.computeSwapStep.

    ``amount_remaining`` > 0 means exact input, < 0 exact output (v3 sign
    convention). Returns ``(sqrt_next, amount_in, amount_out, fee_amount)``.
    """
    zero_for_one = sqrt_current >= sqrt_target
    exact_in = amount_remaining >= 0
    amount_in = amount_out = 0

    if exact_in:
        remaining_less_fee = mul_div(amount_remaining, FEE_PIPS_DENOMINATOR - fee_pips, FEE_PIPS_DENOMINATOR)
        amount_in = (
            get_amount0_delta(sqrt_target, sqrt_current, liquidity, True)
            if zero_for_one
            else get_amount1_delta(sqrt_current, sqrt_target, liquidity, True)
        )
        if remaining_less_fee >= amount_in:
            sqrt_next = sqrt_target
        else:
            sqrt_next = get_next_sqrt_price_from_input(sqrt_current, liquidity, remaining_less_fee, zero_for_one)
    else:
        amount_out = (
            get_amount1_delta(sqrt_target, sqrt_current, liquidity, False)
            if zero_for_one
            else get_amount0_delta(sqrt_current, sqrt_target, liquidity, False)
        )
        if -amount_remaining >= amount_out:
            sqrt_next = sqrt_target
        else:
            sqrt_next = get_next_sqrt_price_from_output(sqrt_current, liquidity, -amount_remaining, zero_for_one)

    reached = sqrt_target == sqrt_next
    if zero_for_one:
        if not (reached and exact_in):
            amount_in = get_amount0_delta(sqrt_next, sqrt_current, liquidity, True)
        if not (reached and not exact_in):
            amount_out = get_amount1_delta(sqrt_next, sqrt_current, liquidity, False)
    else:
        if not (reached and exact_in):
            amount_in = get_amount1_delta(sqrt_current, sqrt_next, liquidity, True)
        if not (reached and not exact_in):
            amount_out = get_amount0_delta(sqrt_current, sqrt_next, liquidity, False)

    if not exact_in and amount_out > -amount_remaining:
        amount_out = -amount_remaining

    if exact_in and sqrt_next != sqrt_target:
        fee_amount = amount_remaining - amount_in
    else:
        fee_amount = mul_div_rounding_up(amount_in, fee_pips, FEE_PIPS_DENOMINATOR - fee_pips)

    return sqrt_next, amount_in, amount_out, fee_amount


# ---------------------------------------------------------------------------
# Concentrated liquidity pool (v3, and v4 with dynamic fees)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Position:
    tick_lower: int
    tick_upper: int
    liquidity: int


@dataclass
class ConcentratedPool:
    """Uniswap v3 pool state reduced to what a swap needs.

    ``ticks`` maps initialized tick -> liquidityNet; it is kept in sync with a
    sorted tick array used to find the next initialized tick.
    """

    token0: str
    token1: str
    sqrt_price_x96: int
    liquidity: int
    fee_pips: int = 3000
    tick_spacing: int = 60
    ticks: Dict[int, int] = field(default_factory=dict)
    pool_id: str = ""
    version: str = "v3"

    def __post_init__(self) -> None:
        if not MIN_SQRT_RATIO <= self.sqrt_price_x96 < MAX_SQRT_RATIO:
            raise MASEError("sqrt price out of range")
        if self.liquidity < 0:
            raise MASEError("liquidity must be non-negative")
        if not 0 <= self.fee_pips < FEE_PIPS_DENOMINATOR:
            raise MASEError("fee out of range")
        if self.tick_spacing <= 0:
            raise MASEError("tick spacing must be positive")
        self.ticks = {t: n for t, n in self.ticks.items() if n != 0}
        self._tick_array: List[int] = sorted(self.ticks)
        self.tick = get_tick_at_sqrt_ratio(self.sqrt_price_x96)
        if not self.pool_id:
            self.pool_id = f"{self.version}:{self.token0}/{self.token1}/{self.fee_pips}"

    # -- construction -------------------------------------------------------

    @classmethod
    def from_positions(
        cls,
        token0: str,
        token1: str,
        sqrt_price_x96: int,
        positions: Iterable[Position],
        fee_pips: int = 3000,
        tick_spacing: int = 60,
        **kwargs: Any,
    ) -> "ConcentratedPool":
        ticks: Dict[int, int] = {}
        current_tick = get_tick_at_sqrt_ratio(sqrt_price_x96)
        active = 0
        for pos in positions:
            if pos.tick_lower >= pos.tick_upper:
                raise MASEError("tick_lower must be below tick_upper")
            if pos.tick_lower < MIN_TICK or pos.tick_upper > MAX_TICK:
                raise MASEError("position ticks out of range")
            if pos.tick_lower % tick_spacing or pos.tick_upper % tick_spacing:
                raise MASEError("position ticks must align with tick spacing")
            if pos.liquidity <= 0:
                raise MASEError("position liquidity must be positive")
            ticks[pos.tick_lower] = ticks.get(pos.tick_lower, 0) + pos.liquidity
            ticks[pos.tick_upper] = ticks.get(pos.tick_upper, 0) - pos.liquidity
            if pos.tick_lower <= current_tick < pos.tick_upper:
                active += pos.liquidity
        return cls(
            token0=token0,
            token1=token1,
            sqrt_price_x96=sqrt_price_x96,
            liquidity=active,
            fee_pips=fee_pips,
            tick_spacing=tick_spacing,
            ticks=ticks,
            **kwargs,
        )

    # -- views --------------------------------------------------------------

    def tokens(self) -> Tuple[str, str]:
        return self.token0, self.token1

    @property
    def tick_array(self) -> Tuple[int, ...]:
        return tuple(self._tick_array)

    def effective_fee(self, payload: Optional[Mapping[str, Any]] = None) -> int:
        return self.fee_pips

    def spot_price(self, zero_for_one: bool) -> Fraction:
        p = sqrt_price_x96_to_price(self.sqrt_price_x96)
        return p if zero_for_one else 1 / p

    def _next_initialized_tick(self, tick: int, zero_for_one: bool) -> Tuple[int, bool]:
        arr = self._tick_array
        if zero_for_one:
            i = bisect_right(arr, tick) - 1
            return (arr[i], True) if i >= 0 else (MIN_TICK, False)
        i = bisect_right(arr, tick)
        return (arr[i], True) if i < len(arr) else (MAX_TICK, False)

    # -- swap ---------------------------------------------------------------

    def _swap(
        self,
        amount_specified: int,
        zero_for_one: bool,
        sqrt_price_limit_x96: Optional[int] = None,
        fee_pips: Optional[int] = None,
    ) -> Tuple["SwapResult", int, int, int]:
        if amount_specified == 0:
            raise MASEError("amount must be non-zero")
        fee = self.fee_pips if fee_pips is None else fee_pips
        if sqrt_price_limit_x96 is None:
            sqrt_price_limit_x96 = MIN_SQRT_RATIO + 1 if zero_for_one else MAX_SQRT_RATIO - 1
        if zero_for_one:
            if not (MIN_SQRT_RATIO < sqrt_price_limit_x96 < self.sqrt_price_x96):
                raise MASEError("invalid sqrt price limit")
        elif not (self.sqrt_price_x96 < sqrt_price_limit_x96 < MAX_SQRT_RATIO):
            raise MASEError("invalid sqrt price limit")

        exact_in = amount_specified > 0
        remaining = amount_specified
        calculated = 0
        sqrt_p = self.sqrt_price_x96
        tick = self.tick
        liquidity = self.liquidity
        total_in = total_out = total_fee = 0
        crossed = 0
        price_before = self.spot_price(zero_for_one)

        while remaining != 0 and sqrt_p != sqrt_price_limit_x96:
            sqrt_start = sqrt_p
            tick_next, initialized = self._next_initialized_tick(tick, zero_for_one)
            tick_next = max(MIN_TICK, min(MAX_TICK, tick_next))
            sqrt_next_tick = get_sqrt_ratio_at_tick(tick_next)
            if zero_for_one:
                target = sqrt_price_limit_x96 if sqrt_next_tick < sqrt_price_limit_x96 else sqrt_next_tick
            else:
                target = sqrt_price_limit_x96 if sqrt_next_tick > sqrt_price_limit_x96 else sqrt_next_tick

            sqrt_p, step_in, step_out, step_fee = compute_swap_step(sqrt_p, target, liquidity, remaining, fee)
            if exact_in:
                remaining -= step_in + step_fee
                calculated -= step_out
            else:
                remaining += step_out
                calculated += step_in + step_fee
            total_in += step_in + step_fee
            total_out += step_out
            total_fee += step_fee

            if sqrt_p == sqrt_next_tick:
                if initialized:
                    net = self.ticks[tick_next]
                    liquidity += -net if zero_for_one else net
                    if liquidity < 0:
                        raise MASEError("tick array produced negative liquidity")
                    crossed += 1
                tick = tick_next - 1 if zero_for_one else tick_next
            elif sqrt_p != sqrt_start:
                tick = get_tick_at_sqrt_ratio(sqrt_p)

        p_after = sqrt_price_x96_to_price(sqrt_p)
        result = SwapResult(
            amount_in=total_in,
            amount_out=total_out,
            fee_paid=total_fee,
            price_before=price_before,
            price_after=p_after if zero_for_one else 1 / p_after,
            fully_filled=remaining == 0,
            ticks_crossed=crossed,
            sqrt_price_after_x96=sqrt_p,
            tick_after=tick,
        )
        return result, sqrt_p, tick, liquidity

    def quote_exact_in(self, amount_in: int, zero_for_one: bool, sqrt_price_limit_x96: Optional[int] = None,
                       fee_pips: Optional[int] = None) -> "SwapResult":
        if amount_in <= 0:
            raise MASEError("amount_in must be positive")
        return self._swap(amount_in, zero_for_one, sqrt_price_limit_x96, fee_pips)[0]

    def quote_exact_out(self, amount_out: int, zero_for_one: bool, sqrt_price_limit_x96: Optional[int] = None,
                        fee_pips: Optional[int] = None) -> "SwapResult":
        if amount_out <= 0:
            raise MASEError("amount_out must be positive")
        return self._swap(-amount_out, zero_for_one, sqrt_price_limit_x96, fee_pips)[0]

    def apply_swap(self, amount_in: int, zero_for_one: bool, sqrt_price_limit_x96: Optional[int] = None,
                   fee_pips: Optional[int] = None) -> "SwapResult":
        """Execute an exact-input swap and mutate pool state."""
        result, sqrt_p, tick, liquidity = self._swap(amount_in, zero_for_one, sqrt_price_limit_x96, fee_pips)
        self.sqrt_price_x96, self.tick, self.liquidity = sqrt_p, tick, liquidity
        return result


@dataclass
class V4Pool(ConcentratedPool):
    """Uniswap v4 pool: v3 swap math plus hook-controlled dynamic LP fees.

    ``dynamic_fee`` mirrors the 0x800000 flag; when set, ``hook_fee_pips``
    (or a ``fee_override_pips`` field on the payload) replaces the static fee.
    ``protocol_fee_pips`` is applied before the LP fee as in v4's
    ProtocolFeeLibrary.calculateSwapFee.
    """

    dynamic_fee: bool = False
    hook_fee_pips: Optional[int] = None
    protocol_fee_pips: int = 0
    version: str = "v4"

    def __post_init__(self) -> None:
        super().__post_init__()
        for value in (self.hook_fee_pips, self.protocol_fee_pips):
            if value is not None and not 0 <= value <= V4_MAX_LP_FEE:
                raise MASEError("v4 fee out of range")

    def effective_fee(self, payload: Optional[Mapping[str, Any]] = None) -> int:
        lp_fee = self.fee_pips
        if self.dynamic_fee:
            override = (payload or {}).get("fee_override_pips")
            if override is not None:
                lp_fee = int(override)
            elif self.hook_fee_pips is not None:
                lp_fee = self.hook_fee_pips
        if not 0 <= lp_fee <= V4_MAX_LP_FEE:
            raise MASEError("v4 lp fee out of range")
        pf = self.protocol_fee_pips
        # protocolFee + lpFee - protocolFee * lpFee / 1e6
        total = pf + lp_fee - (pf * lp_fee) // FEE_PIPS_DENOMINATOR
        return min(total, FEE_PIPS_DENOMINATOR - 1)

    def quote_exact_in(self, amount_in: int, zero_for_one: bool, sqrt_price_limit_x96: Optional[int] = None,
                       fee_pips: Optional[int] = None) -> "SwapResult":
        fee = self.effective_fee() if fee_pips is None else fee_pips
        return super().quote_exact_in(amount_in, zero_for_one, sqrt_price_limit_x96, fee)

    def quote_exact_out(self, amount_out: int, zero_for_one: bool, sqrt_price_limit_x96: Optional[int] = None,
                        fee_pips: Optional[int] = None) -> "SwapResult":
        fee = self.effective_fee() if fee_pips is None else fee_pips
        return super().quote_exact_out(amount_out, zero_for_one, sqrt_price_limit_x96, fee)

    def apply_swap(self, amount_in: int, zero_for_one: bool, sqrt_price_limit_x96: Optional[int] = None,
                   fee_pips: Optional[int] = None) -> "SwapResult":
        fee = self.effective_fee() if fee_pips is None else fee_pips
        return super().apply_swap(amount_in, zero_for_one, sqrt_price_limit_x96, fee)


Pool = Union[V2Pool, ConcentratedPool]


# ---------------------------------------------------------------------------
# Results and slippage reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwapResult:
    amount_in: int  # gross input, including fee
    amount_out: int
    fee_paid: int
    price_before: Fraction  # marginal output-per-input before the swap
    price_after: Fraction  # marginal output-per-input after the swap
    fully_filled: bool
    ticks_crossed: int
    sqrt_price_after_x96: Optional[int] = None
    tick_after: Optional[int] = None

    @property
    def execution_price(self) -> Fraction:
        if self.amount_in == 0:
            return Fraction(0)
        return Fraction(self.amount_out, self.amount_in)


@dataclass(frozen=True)
class SlippageVector:
    """Per-trade token movement and the slippage bound around it.

    ``deltas`` is from the trader's side: negative = paid in, positive =
    received. ``max_slippage_out`` is how many output tokens the trade can
    lose before hitting ``min_amount_out`` (quoted minus minimum).
    """

    token_in: str
    token_out: str
    deltas: Tuple[Tuple[str, int], ...]
    amount_out_quoted: int
    min_amount_out: int
    max_slippage_out: int
    tolerance_bps: int

    def as_dict(self) -> Dict[str, int]:
        return dict(self.deltas)


@dataclass(frozen=True)
class SlippageReport:
    pool_ids: Tuple[str, ...]
    token_in: str
    token_out: str
    amount_in: int
    amount_out: int
    spot_price: Fraction  # combined marginal price before the trade (fee-free)
    execution_price: Fraction
    price_after: Fraction
    price_impact: Fraction  # 1 - execution_price / spot_price (includes fee)
    price_impact_ex_fee: Fraction  # impact from curve movement alone
    fees_paid: Tuple[int, ...]
    vector: SlippageVector
    fully_filled: bool
    ticks_crossed: int

    @property
    def price_impact_bps(self) -> float:
        return float(self.price_impact * BPS_DENOMINATOR)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pool_ids": list(self.pool_ids),
            "token_in": self.token_in,
            "token_out": self.token_out,
            "amount_in": self.amount_in,
            "amount_out": self.amount_out,
            "spot_price": float(self.spot_price),
            "execution_price": float(self.execution_price),
            "price_after": float(self.price_after),
            "price_impact": float(self.price_impact),
            "price_impact_ex_fee": float(self.price_impact_ex_fee),
            "price_impact_bps": self.price_impact_bps,
            "fees_paid": list(self.fees_paid),
            "min_amount_out": self.vector.min_amount_out,
            "max_slippage_out": self.vector.max_slippage_out,
            "tolerance_bps": self.vector.tolerance_bps,
            "deltas": self.vector.as_dict(),
            "fully_filled": self.fully_filled,
            "ticks_crossed": self.ticks_crossed,
        }


def min_amount_out(amount_out: int, tolerance_bps: int) -> int:
    if not 0 <= tolerance_bps <= BPS_DENOMINATOR:
        raise MASEError("tolerance_bps must be within [0, 10000]")
    return amount_out * (BPS_DENOMINATOR - tolerance_bps) // BPS_DENOMINATOR


def _direction(pool: Pool, token_in: str) -> Tuple[bool, str]:
    t0, t1 = pool.tokens()
    if token_in == t0:
        return True, t1
    if token_in == t1:
        return False, t0
    raise MASEError(f"token {token_in!r} not in pool {pool.pool_id}")


def _pool_fee_fraction(pool: Pool, fee_pips: Optional[int]) -> Fraction:
    if isinstance(pool, V2Pool):
        return Fraction(pool.fee_bps, BPS_DENOMINATOR)
    fee = pool.effective_fee() if fee_pips is None else fee_pips
    return Fraction(fee, FEE_PIPS_DENOMINATOR)


def evaluate_route(
    pools: Sequence[Pool],
    token_in: str,
    amount_in: int,
    tolerance_bps: int = 50,
    fee_overrides: Optional[Sequence[Optional[int]]] = None,
) -> SlippageReport:
    """Quote an exact-input multi-hop trade and report its slippage."""
    if not pools:
        raise MASEError("route needs at least one pool")
    if amount_in <= 0:
        raise MASEError("amount_in must be positive")
    overrides = list(fee_overrides or [None] * len(pools))
    if len(overrides) != len(pools):
        raise MASEError("fee_overrides length must match pools")

    current_token = token_in
    current_amount = amount_in
    spot = Fraction(1)
    after = Fraction(1)
    fee_factor = Fraction(1)
    fees: List[int] = []
    filled = True
    crossed = 0

    for pool, override in zip(pools, overrides):
        zero_for_one, token_out = _direction(pool, current_token)
        if isinstance(pool, V2Pool):
            res = pool.quote_exact_in(current_amount, zero_for_one)
        else:
            res = pool.quote_exact_in(current_amount, zero_for_one, fee_pips=override)
        spot *= res.price_before
        after *= res.price_after
        fee_factor *= 1 - _pool_fee_fraction(pool, override)
        fees.append(res.fee_paid)
        filled = filled and res.fully_filled
        crossed += res.ticks_crossed
        if res.amount_out <= 0:
            raise MASEError(f"route produced zero output at {pool.pool_id}")
        if not res.fully_filled:
            current_amount = res.amount_out
            current_token = token_out
            break
        current_amount = res.amount_out
        current_token = token_out

    amount_out = current_amount
    execution = Fraction(amount_out, amount_in)
    impact = 1 - execution / spot
    impact_ex_fee = 1 - execution / (spot * fee_factor)
    min_out = min_amount_out(amount_out, tolerance_bps)
    vector = SlippageVector(
        token_in=token_in,
        token_out=current_token,
        deltas=((token_in, -amount_in), (current_token, amount_out)),
        amount_out_quoted=amount_out,
        min_amount_out=min_out,
        max_slippage_out=amount_out - min_out,
        tolerance_bps=tolerance_bps,
    )
    return SlippageReport(
        pool_ids=tuple(p.pool_id for p in pools),
        token_in=token_in,
        token_out=current_token,
        amount_in=amount_in,
        amount_out=amount_out,
        spot_price=spot,
        execution_price=execution,
        price_after=after,
        price_impact=impact,
        price_impact_ex_fee=impact_ex_fee,
        fees_paid=tuple(fees),
        vector=vector,
        fully_filled=filled,
        ticks_crossed=crossed,
    )


def evaluate_trade(pool: Pool, token_in: str, amount_in: int, tolerance_bps: int = 50,
                   fee_pips: Optional[int] = None) -> SlippageReport:
    """Single-pool form of :func:`evaluate_route`."""
    return evaluate_route([pool], token_in, amount_in, tolerance_bps, [fee_pips])


# ---------------------------------------------------------------------------
# Engine: pool registry + mempool payload entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwapPayload:
    """Normalised swap intent extracted from a pending transaction."""

    tx_hash: str
    route: Tuple[str, ...]  # pool ids, in hop order
    token_in: str
    amount_in: int
    amount_out_min: Optional[int] = None
    tolerance_bps: Optional[int] = None
    fee_override_pips: Optional[int] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SwapPayload":
        """Accept a dict payload from the mempool layer.

        Required keys: ``token_in``, ``amount_in`` and ``route`` (list of pool
        ids) or ``pool_id``. Optional: ``tx_hash``/``hash``,
        ``amount_out_min``, ``tolerance_bps``/``slippage_bps``,
        ``fee_override_pips``. Amounts may be ints, decimal strings or hex.
        """
        route = data.get("route")
        if route is None:
            pool_id = data.get("pool_id")
            if pool_id is None:
                raise MASEError("payload needs 'route' or 'pool_id'")
            route = [pool_id]
        if isinstance(route, str):
            route = [route]
        if "token_in" not in data or "amount_in" not in data:
            raise MASEError("payload needs 'token_in' and 'amount_in'")
        tol = data.get("tolerance_bps", data.get("slippage_bps"))
        min_out = data.get("amount_out_min")
        fee_override = data.get("fee_override_pips")
        return cls(
            tx_hash=str(data.get("tx_hash", data.get("hash", ""))),
            route=tuple(str(r) for r in route),
            token_in=str(data["token_in"]),
            amount_in=_to_int(data["amount_in"]),
            amount_out_min=None if min_out is None else _to_int(min_out),
            tolerance_bps=None if tol is None else int(tol),
            fee_override_pips=None if fee_override is None else int(fee_override),
        )


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        raise MASEError("boolean is not an amount")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 16) if value.lower().startswith("0x") else int(value)
    raise MASEError(f"cannot read amount from {type(value).__name__}")


@dataclass(frozen=True)
class PayloadEvaluation:
    payload: SwapPayload
    report: SlippageReport
    # Effective tolerance implied by the victim's amount_out_min, if given.
    implied_tolerance: Optional[Fraction]
    # Quoted output minus amount_out_min (the slack the trade leaves open).
    slack_out: Optional[int]
    would_revert: bool


class MASEEngine:
    """Pool registry that evaluates swap payloads from the mempool layer."""

    def __init__(self, pools: Iterable[Pool] = (), default_tolerance_bps: int = 50) -> None:
        self.pools: Dict[str, Pool] = {}
        self.default_tolerance_bps = default_tolerance_bps
        for pool in pools:
            self.register_pool(pool)

    def register_pool(self, pool: Pool) -> None:
        self.pools[pool.pool_id] = pool

    def get_pool(self, pool_id: str) -> Pool:
        try:
            return self.pools[pool_id]
        except KeyError:
            raise MASEError(f"unknown pool {pool_id!r}") from None

    def evaluate(self, payload: Union[SwapPayload, Mapping[str, Any]]) -> PayloadEvaluation:
        if not isinstance(payload, SwapPayload):
            payload = SwapPayload.from_mapping(payload)
        pools = [self.get_pool(pid) for pid in payload.route]
        overrides: List[Optional[int]] = []
        for pool in pools:
            if isinstance(pool, ConcentratedPool):
                ctx = {"fee_override_pips": payload.fee_override_pips}
                overrides.append(pool.effective_fee(ctx))
            else:
                overrides.append(None)
        tol = payload.tolerance_bps if payload.tolerance_bps is not None else self.default_tolerance_bps
        report = evaluate_route(pools, payload.token_in, payload.amount_in, tol, overrides)
        implied = slack = None
        would_revert = not report.fully_filled
        if payload.amount_out_min is not None:
            slack = report.amount_out - payload.amount_out_min
            implied = 1 - Fraction(payload.amount_out_min, report.amount_out)
            would_revert = would_revert or slack < 0
        return PayloadEvaluation(payload, report, implied, slack, would_revert)

    def evaluate_payload(self, payload: Union[SwapPayload, Mapping[str, Any]]) -> Dict[str, Any]:
        """Dict-in / dict-out entry point for the mempool emulator."""
        ev = self.evaluate(payload)
        out = ev.report.to_dict()
        out.update(
            tx_hash=ev.payload.tx_hash,
            amount_out_min=ev.payload.amount_out_min,
            implied_tolerance=None if ev.implied_tolerance is None else float(ev.implied_tolerance),
            slack_out=ev.slack_out,
            would_revert=ev.would_revert,
        )
        return out

    def evaluate_batch(self, payloads: Iterable[Union[SwapPayload, Mapping[str, Any]]],
                       workers: Optional[int] = None, parallel_threshold: int = 4,
                       start_method: str = "spawn") -> List[Dict[str, Any]]:
        """Evaluate payloads against the current pool state, in input order.

        With ``workers`` set, payload chunks are spread over a capped
        ``multiprocessing`` pool (see :mod:`simulator.parallel`). Each worker
        gets its own copy of the pools, so results match the serial path.
        """
        payloads = list(payloads)
        if workers is None:
            return [self.evaluate_payload(p) for p in payloads]
        from .parallel import WorkerPool, chunked

        with WorkerPool(workers, start_method, parallel_threshold) as wp:
            if not wp.will_parallelize(len(payloads)):
                return [self.evaluate_payload(p) for p in payloads]
            pools = list(self.pools.values())
            tasks = [(pools, self.default_tolerance_bps, chunk)
                     for chunk in chunked(payloads, wp.workers * 4)]
            return [row for part in wp.map(_evaluate_chunk, tasks) for row in part]


def _evaluate_chunk(task: Tuple[List[Pool], int, List[Any]]) -> List[Dict[str, Any]]:
    pools, tolerance, payloads = task
    engine = MASEEngine(pools, tolerance)
    return [engine.evaluate_payload(p) for p in payloads]


# ---------------------------------------------------------------------------
# Parallel trade-size sweeps
# ---------------------------------------------------------------------------


def _sweep_chunk(task: Tuple[Pool, str, List[int], int]) -> List[Dict[str, Any]]:
    pool, token_in, amounts, tolerance = task
    rows: List[Dict[str, Any]] = []
    for amount in amounts:
        try:
            rows.append(evaluate_trade(pool, token_in, amount, tolerance).to_dict())
        except MASEError as exc:
            rows.append({"amount_in": amount, "error": str(exc)})
    return rows


def sweep_trade_sizes(pool: Pool, token_in: str, amounts: Sequence[int], tolerance_bps: int = 50,
                      workers: Optional[int] = None, parallel_threshold: int = 4,
                      start_method: str = "spawn") -> List[Dict[str, Any]]:
    """Price-impact curve: quote every size in ``amounts`` against ``pool``.

    One swap walks the tick array sequentially, so the parallelism is across
    sizes: chunks of the size array go to separate worker processes. Rows come
    back in input order; a size that cannot be quoted yields
    ``{"amount_in": n, "error": ...}``.
    """
    amounts = [int(a) for a in amounts]
    if workers is None:
        return _sweep_chunk((pool, token_in, amounts, tolerance_bps))
    from .parallel import WorkerPool, chunked

    with WorkerPool(workers, start_method, parallel_threshold) as wp:
        if not wp.will_parallelize(len(amounts)):
            return _sweep_chunk((pool, token_in, amounts, tolerance_bps))
        tasks = [(pool, token_in, chunk, tolerance_bps) for chunk in chunked(amounts, wp.workers * 4)]
        return [row for part in wp.map(_sweep_chunk, tasks) for row in part]


__all__ = [
    "Q96",
    "MIN_TICK",
    "MAX_TICK",
    "MIN_SQRT_RATIO",
    "MAX_SQRT_RATIO",
    "MASEError",
    "V2Pool",
    "ConcentratedPool",
    "V4Pool",
    "Position",
    "SwapResult",
    "SlippageVector",
    "SlippageReport",
    "SwapPayload",
    "PayloadEvaluation",
    "MASEEngine",
    "v2_get_amount_out",
    "v2_get_amount_in",
    "get_sqrt_ratio_at_tick",
    "get_tick_at_sqrt_ratio",
    "sqrt_price_x96_to_price",
    "price_to_sqrt_price_x96",
    "get_amount0_delta",
    "get_amount1_delta",
    "get_next_sqrt_price_from_input",
    "get_next_sqrt_price_from_output",
    "compute_swap_step",
    "evaluate_trade",
    "evaluate_route",
    "min_amount_out",
    "sweep_trade_sizes",
]
