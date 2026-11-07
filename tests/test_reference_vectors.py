"""Bit-exact checks of the AMM math against the official Uniswap test vectors.

Every expected number in this file is copied from the Uniswap repositories,
not computed by this project:

* v2: ``Uniswap/v2-core`` ``test/UniswapV2Pair.spec.ts`` (``swapTestCases``,
  ``optimisticTestCases``) and ``Uniswap/v2-periphery``
  ``test/UniswapV2Router02.spec.ts`` (``getAmountOut`` / ``getAmountIn``).
* v3: ``Uniswap/v3-core`` ``test/TickMath.spec.ts`` and its snapshot,
  ``test/SqrtPriceMath.spec.ts``, ``test/SwapMath.spec.ts``, and the full-pool
  swap snapshot ``test/__snapshots__/UniswapV3Pool.swaps.spec.ts.snap``
  (converted to ``tests/vectors/uniswap_v3_pool_swaps.json`` by
  ``tools/convert_v3_swap_snapshot.py``).

Each pool-swap vector is a real Hardhat run of the v3-core contracts: 15
pools x 16 swaps = 240 scenarios. We require exact equality of both token
deltas, the final tick, ``feeGrowthGlobal{0,1}X128`` and the reverts.
"""

from __future__ import annotations

import json
import math
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal, localcontext
from pathlib import Path

import pytest

from simulator.slippage_engine import (
    MAX_SQRT_RATIO,
    MAX_TICK,
    MIN_SQRT_RATIO,
    MIN_TICK,
    ConcentratedPool,
    MASEError,
    Position,
    V2Pool,
    compute_swap_step,
    get_amount0_delta,
    get_amount1_delta,
    get_next_sqrt_price_from_input,
    get_next_sqrt_price_from_output,
    get_sqrt_ratio_at_tick,
    get_tick_at_sqrt_ratio,
    v2_get_amount_in,
    v2_get_amount_out,
    v2_swap_satisfies_k,
)

E18 = 10**18
UINT256_MAX = 2**256 - 1
VECTORS = Path(__file__).parent / "vectors" / "uniswap_v3_pool_swaps.json"


def encode_price_sqrt(reserve1: int, reserve0: int) -> int:
    """v3-core test/shared/utilities.ts encodePriceSqrt.

    bignumber.js is configured with DECIMAL_PLACES=40 and ROUND_HALF_UP for
    both the division and the square root, then the result times 2**96 is
    floored. Reproducing that rounding matters: 1/2**127 rounds to 5.9e-39.
    """
    with localcontext() as ctx:
        ctx.prec = 200
        q = Decimal(1).scaleb(-40)
        ratio = (Decimal(reserve1) / Decimal(reserve0)).quantize(q, rounding=ROUND_HALF_UP)
        root = ratio.sqrt().quantize(q, rounding=ROUND_HALF_UP)
        return int((root * (Decimal(2) ** 96)).to_integral_value(rounding=ROUND_FLOOR))


# ---------------------------------------------------------------------------
# Uniswap v2
# ---------------------------------------------------------------------------

V2_SWAP_CASES = [  # (swapAmount, token0Amount, token1Amount, expectedOutputAmount), UniswapV2Pair.spec.ts
    (1 * E18, 5 * E18, 10 * E18, 1662497915624478906),
    (1 * E18, 10 * E18, 5 * E18, 453305446940074565),
    (2 * E18, 5 * E18, 10 * E18, 2851015155847869602),
    (2 * E18, 10 * E18, 5 * E18, 831248957812239453),
    (1 * E18, 10 * E18, 10 * E18, 906610893880149131),
    (1 * E18, 100 * E18, 100 * E18, 987158034397061298),
    (1 * E18, 1000 * E18, 1000 * E18, 996006981039903216),
]


@pytest.mark.parametrize("amount_in,r0,r1,expected", V2_SWAP_CASES)
def test_v2_get_amount_out_matches_pair_spec(amount_in, r0, r1, expected):
    assert v2_get_amount_out(amount_in, r0, r1) == expected
    # The pair accepts exactly this output and rejects one wei more ("UniswapV2: K").
    assert v2_swap_satisfies_k(r0, r1, amount_in, 0, 0, expected)
    assert not v2_swap_satisfies_k(r0, r1, amount_in, 0, 0, expected + 1)


@pytest.mark.parametrize("amount_in,r0,r1,expected", V2_SWAP_CASES)
def test_v2_pool_quote_and_apply_match_pair_spec(amount_in, r0, r1, expected):
    pool = V2Pool("T0", "T1", r0, r1)
    res = pool.quote_exact_in(amount_in, True)
    assert res.amount_out == expected
    pool.apply_swap(res, True)
    assert (pool.reserve0, pool.reserve1) == (r0 + amount_in, r1 - expected)


@pytest.mark.parametrize("amount_out,r0,r1,amount_in", [  # optimisticTestCases
    (997000000000000000, 5 * E18, 10 * E18, 1 * E18),
    (997000000000000000, 10 * E18, 5 * E18, 1 * E18),
    (997000000000000000, 5 * E18, 5 * E18, 1 * E18),
    (1 * E18, 5 * E18, 5 * E18, 1003009027081243732),
])
def test_v2_optimistic_same_token_repayment(amount_out, r0, r1, amount_in):
    # Flash swap: take token0 out, repay token0. The pair accepts amount_out and rejects amount_out + 1.
    assert v2_swap_satisfies_k(r0, r1, amount_in, 0, amount_out, 0)
    assert not v2_swap_satisfies_k(r0, r1, amount_in, 0, amount_out + 1, 0)


def test_v2_router_get_amount_out_and_in():
    assert v2_get_amount_out(2, 100, 100) == 1
    assert v2_get_amount_in(1, 100, 100) == 2
    for bad in ((0, 100, 100), (2, 0, 100), (2, 100, 0)):
        with pytest.raises(MASEError):
            v2_get_amount_out(*bad)
        with pytest.raises(MASEError):
            v2_get_amount_in(*bad)


def test_v2_get_amount_in_is_minimal():
    # getAmountIn rounds up (+1), so its result always buys the output and is at most 1 wei above the minimum.
    for r0, r1, out in [(5 * E18, 10 * E18, 10**17), (10**6, 3 * 10**20, 12345678901234), (997, 1000, 1)]:
        a = v2_get_amount_in(out, r0, r1)
        assert v2_get_amount_out(a, r0, r1) >= out
        # Minimal input is ceil(N/D); getAmountIn returns floor(N/D) + 1, so a - 2 never suffices.
        assert a - 2 <= 0 or v2_get_amount_out(a - 2, r0, r1) < out


# ---------------------------------------------------------------------------
# Uniswap v3: TickMath
# ---------------------------------------------------------------------------

SQRT_RATIO_AT_TICK = {  # TickMath.spec.ts.snap "getSqrtRatioAtTick tick N result"
    -50: 79030349367926598376800521322,
    -100: 78833030112140176575862854579,
    -250: 78244023372248365697264290337,
    -500: 77272108795590369356373805297,
    -1000: 75364347830767020784054125655,
    -2500: 69919044979842180277688105136,
    -3000: 68192822843687888778582228483,
    -4000: 64867181785621769311890333195,
    -5000: 61703726247759831737814779831,
    -50000: 6504256538020985011912221507,
    -150000: 43836292794701720435367485,
    -250000: 295440463448801648376846,
    -500000: 1101692437043807371,
    -738203: 7409801140451,
    50: 79426470787362580746886972461,
    100: 79625275426524748796330556128,
    250: 80224679980005306637834519095,
    500: 81233731461783161732293370115,
    1000: 83290069058676223003182343270,
    2500: 89776708723587163891445672585,
    3000: 92049301871182272007977902845,
    4000: 96768528593268422080558758223,
    5000: 101729702841318637793976746270,
    50000: 965075977353221155028623082916,
    150000: 143194173941309278083010301478497,
    250000: 21246587762933397357449903968194344,
    500000: 5697689776495288729098254600827762987878,
    738203: 847134979253254120489401328389043031315994541,
    MIN_TICK: 4295128739,
    MIN_TICK + 1: 4295343490,
    MAX_TICK - 1: 1461373636630004318706518188784493106690254656249,
    MAX_TICK: 1461446703485210103287273052203988822378723970342,
    0: 2**96,
}

TICK_AT_SQRT_RATIO = {  # TickMath.spec.ts.snap "getTickAtSqrtRatio ratio N result" + explicit cases
    4295128739: -887272,
    79228162514264337593543: -276325,
    79228162514264337593543950: -138163,
    9903520314283042199192993792: -41591,
    28011385487393069959365969113: -20796,
    56022770974786139918731938227: -6932,
    79228162514264337593543950336: 0,
    112045541949572279837463876454: 6931,
    224091083899144559674927752909: 20795,
    633825300114114700748351602688: 41590,
    79228162514264337593543950336000: 138162,
    79228162514264337593543950336000000: 276324,
    1461446703485210103287273052203988822378723970341: 887271,
    4295343490: MIN_TICK + 1,
    1461373636630004318706518188784493106690254656249: MAX_TICK - 1,
}


@pytest.mark.parametrize("tick,expected", sorted(SQRT_RATIO_AT_TICK.items()))
def test_get_sqrt_ratio_at_tick(tick, expected):
    assert get_sqrt_ratio_at_tick(tick) == expected


@pytest.mark.parametrize("ratio,expected", sorted(TICK_AT_SQRT_RATIO.items()))
def test_get_tick_at_sqrt_ratio(ratio, expected):
    assert get_tick_at_sqrt_ratio(ratio) == expected
    assert get_sqrt_ratio_at_tick(expected) <= ratio < get_sqrt_ratio_at_tick(expected + 1)


def test_tick_math_bounds():
    assert get_sqrt_ratio_at_tick(MIN_TICK) == MIN_SQRT_RATIO
    assert get_sqrt_ratio_at_tick(MAX_TICK) == MAX_SQRT_RATIO
    for bad in (MIN_TICK - 1, MAX_TICK + 1):
        with pytest.raises(MASEError):
            get_sqrt_ratio_at_tick(bad)
    for bad in (MIN_SQRT_RATIO - 1, MAX_SQRT_RATIO):
        with pytest.raises(MASEError):
            get_tick_at_sqrt_ratio(bad)
    assert get_sqrt_ratio_at_tick(MIN_TICK) < encode_price_sqrt(1, 2**127)
    assert get_sqrt_ratio_at_tick(MAX_TICK) > encode_price_sqrt(2**127, 1)


# ---------------------------------------------------------------------------
# Uniswap v3: SqrtPriceMath
# ---------------------------------------------------------------------------

P1 = encode_price_sqrt(1, 1)


def test_next_sqrt_price_from_input_vectors():
    assert get_next_sqrt_price_from_input(P1, E18, E18 // 10, False) == 87150978765690771352898345369
    assert get_next_sqrt_price_from_input(P1, E18, E18 // 10, True) == 72025602285694852357767227579
    assert get_next_sqrt_price_from_input(P1, 10 * E18, 2**100, True) == 624999999995069620
    assert get_next_sqrt_price_from_input(P1, 1, UINT256_MAX // 2, True) == 1
    assert get_next_sqrt_price_from_input(1, 1, 2**255, True) == 1
    assert get_next_sqrt_price_from_input(P1, E18 // 10, 0, True) == P1
    assert get_next_sqrt_price_from_input(P1, E18 // 10, 0, False) == P1


def test_next_sqrt_price_from_input_max_inputs_use_overflow_branch():
    # "returns the minimum price for max inputs": amount * sqrtP overflows uint256,
    # so the contract takes the divRoundingUp fallback.
    sqrt_p = 2**160 - 1
    liquidity = 2**128 - 1
    max_amount_no_overflow = UINT256_MAX - (liquidity << 96) // sqrt_p
    assert get_next_sqrt_price_from_input(sqrt_p, liquidity, max_amount_no_overflow, True) == 1


@pytest.mark.parametrize("args", [
    (0, 0, E18 // 10, False),  # price is zero
    (1, 0, E18 // 10, True),  # liquidity is zero
    (2**160 - 1, 1024, 1024, False),  # input overflows the price
])
def test_next_sqrt_price_from_input_reverts(args):
    with pytest.raises(MASEError):
        get_next_sqrt_price_from_input(*args)


def test_next_sqrt_price_from_output_vectors():
    price = 20282409603651670423947251286016
    assert get_next_sqrt_price_from_output(price, 1024, 262143, True) == 77371252455336267181195264
    assert get_next_sqrt_price_from_output(P1, E18, E18 // 10, False) == 88031291682515930659493278152
    assert get_next_sqrt_price_from_output(P1, E18, E18 // 10, True) == 71305346262837903834189555302
    assert get_next_sqrt_price_from_output(P1, E18 // 10, 0, True) == P1
    assert get_next_sqrt_price_from_output(P1, E18 // 10, 0, False) == P1


@pytest.mark.parametrize("args", [
    (0, 0, E18 // 10, False),
    (1, 0, E18 // 10, True),
    (20282409603651670423947251286016, 1024, 4, False),  # exactly the virtual reserves of token0
    (20282409603651670423947251286016, 1024, 5, False),
    (20282409603651670423947251286016, 1024, 262145, True),
    (20282409603651670423947251286016, 1024, 262144, True),  # exactly the virtual reserves of token1
    (P1, 1, UINT256_MAX, True),
    (P1, 1, UINT256_MAX, False),
])
def test_next_sqrt_price_from_output_reverts(args):
    with pytest.raises(MASEError):
        get_next_sqrt_price_from_output(*args)


def test_amount_deltas():
    p121 = encode_price_sqrt(121, 100)
    assert get_amount0_delta(P1, encode_price_sqrt(2, 1), 0, True) == 0
    assert get_amount0_delta(P1, P1, 0, True) == 0
    assert get_amount0_delta(P1, p121, E18, True) == 90909090909090910
    assert get_amount0_delta(P1, p121, E18, False) == 90909090909090909
    assert get_amount1_delta(P1, p121, E18, True) == 100000000000000000
    assert get_amount1_delta(P1, p121, E18, False) == 99999999999999999
    up = get_amount0_delta(encode_price_sqrt(2**90, 1), encode_price_sqrt(2**96, 1), E18, True)
    down = get_amount0_delta(encode_price_sqrt(2**90, 1), encode_price_sqrt(2**96, 1), E18, False)
    assert up == down + 1


# ---------------------------------------------------------------------------
# Uniswap v3: SwapMath.computeSwapStep (SwapMath.spec.ts)
# ---------------------------------------------------------------------------

SWAP_STEP_CASES = [
    # (sqrtCurrent, sqrtTarget, liquidity, amountRemaining, feePips) -> (sqrtNext, amountIn, amountOut, fee)
    ((P1, encode_price_sqrt(101, 100), 2 * E18, E18, 600),
     (encode_price_sqrt(101, 100), 9975124224178055, 9925619580021728, 5988667735148)),
    ((P1, encode_price_sqrt(101, 100), 2 * E18, -E18, 600),
     (encode_price_sqrt(101, 100), 9975124224178055, 9925619580021728, 5988667735148)),
    ((P1, encode_price_sqrt(1000, 100), 2 * E18, E18, 600),
     (get_next_sqrt_price_from_input(P1, 2 * E18, E18 - 600000000000000, False),
      999400000000000000, 666399946655997866, 600000000000000)),
    ((P1, encode_price_sqrt(10000, 100), 2 * E18, -E18, 600),
     (get_next_sqrt_price_from_output(P1, 2 * E18, E18, False),
      2000000000000000000, E18, 1200720432259356)),
    ((417332158212080721273783715441582, 1452870262520218020823638996, 159344665391607089467575320103, -1, 1),
     (417332158212080721273783715441581, 1, 1, 1)),
    ((2, 1, 1, 3915081100057732413702495386755767, 1),
     (1, 39614081257132168796771975168, 0, 39614120871253040049813)),
    ((2413, 79887613182836312, 1985041575832132834610021537970, 10, 1872),
     (2413, 0, 0, 10)),
    ((20282409603651670423947251286016, 20282409603651670423947251286016 * 11 // 10, 1024, -4, 3000),
     (20282409603651670423947251286016 * 11 // 10, 26215, 0, 79)),
    ((20282409603651670423947251286016, 20282409603651670423947251286016 * 9 // 10, 1024, -263000, 3000),
     (20282409603651670423947251286016 * 9 // 10, 1, 26214, 1)),
]


@pytest.mark.parametrize("args,expected", SWAP_STEP_CASES)
def test_compute_swap_step(args, expected):
    assert compute_swap_step(*args) == expected


# ---------------------------------------------------------------------------
# Uniswap v3: full pool swaps (UniswapV3Pool.swaps.spec.ts snapshot)
# ---------------------------------------------------------------------------


def _load_pool_cases():
    data = json.loads(VECTORS.read_text())
    cases = []
    for pool in data["pools"]:
        for swap in pool["swaps"]:
            cases.append(pytest.param(pool, swap, id=f"{pool['description']} | {swap['description']}"))
    return cases


POOL_CASES = _load_pool_cases()


def _build_pool(case) -> ConcentratedPool:
    return ConcentratedPool.from_positions(
        "token0", "token1", int(case["sqrt_price_x96"]),
        [Position(lo, hi, int(liq)) for lo, hi, liq in case["positions"]],
        fee_pips=case["fee"], tick_spacing=case["tick_spacing"])


def _mint_amounts(case) -> tuple:
    """Token amounts the pool holds after minting, as NonfungiblePositionManager-free Pool.mint computes them."""
    sqrt_p = int(case["sqrt_price_x96"])
    tick = get_tick_at_sqrt_ratio(sqrt_p)
    a0 = a1 = 0
    for lo, hi, liq in case["positions"]:
        liq = int(liq)
        sa, sb = get_sqrt_ratio_at_tick(lo), get_sqrt_ratio_at_tick(hi)
        if tick < lo:
            a0 += get_amount0_delta(sa, sb, liq, True)
        elif tick < hi:
            a0 += get_amount0_delta(sqrt_p, sb, liq, True)
            a1 += get_amount1_delta(sa, sqrt_p, liq, True)
        else:
            a1 += get_amount1_delta(sa, sb, liq, True)
    return a0, a1


def test_vector_file_is_complete():
    data = json.loads(VECTORS.read_text())
    assert len(data["pools"]) == 15
    assert sum(len(p["swaps"]) for p in data["pools"]) == 240


@pytest.mark.parametrize("case,swap", POOL_CASES)
def test_uniswap_v3_pool_swap_snapshot(case, swap):
    exp = swap["expected"]
    pool = _build_pool(case)
    assert pool.tick == exp["tickBefore"]
    balance0, balance1 = _mint_amounts(case)
    assert str(balance0) == exp.get("amount0Before", exp.get("poolBalance0"))
    assert str(balance1) == exp.get("amount1Before", exp.get("poolBalance1"))

    zero_for_one = swap["zero_for_one"]
    amount = int(swap["amount_specified"])
    limit = None if swap["sqrt_price_limit_x96"] is None else int(swap["sqrt_price_limit_x96"])
    if "swapError" in exp:
        assert exp["swapError"].endswith("revert SPL")
        with pytest.raises(MASEError):
            if amount > 0:
                pool.quote_exact_in(amount, zero_for_one, limit)
            else:
                pool.quote_exact_out(-amount, zero_for_one, limit)
        return
    if amount > 0:
        res = pool.quote_exact_in(amount, zero_for_one, limit)
    else:
        res = pool.quote_exact_out(-amount, zero_for_one, limit)
    amount0 = res.amount_in if zero_for_one else -res.amount_out
    amount1 = -res.amount_out if zero_for_one else res.amount_in
    assert str(amount0) == exp["amount0Delta"]
    assert str(amount1) == exp["amount1Delta"]
    assert res.tick_after == exp["tickAfter"]
    fee_growth = (res.fee_growth_global_x128_delta, 0) if zero_for_one else (0, res.fee_growth_global_x128_delta)
    assert (str(fee_growth[0]), str(fee_growth[1])) == (exp["feeGrowthGlobal0X128Delta"],
                                                      exp["feeGrowthGlobal1X128Delta"])
    if amount > 0:
        # apply_swap must land on the same state the quote reports.
        applied = pool.apply_swap(amount, zero_for_one, limit)
        assert applied == res
        assert pool.tick == exp["tickAfter"]
        assert pool.sqrt_price_x96 == res.sqrt_price_after_x96


def test_word_boundary_split_changes_rounding():
    """Regression: stepping by tick-bitmap word changes the result by a few wei vs. one big step.

    ``low fee, 1:1 price`` / ``swap token0 for exactly 1.0000 token1`` needs 2001000500250125077
    token0 on chain; computing it as a single step (the previous behaviour) gave ...063.
    """
    pool = ConcentratedPool.from_positions(
        "token0", "token1", 2**96, [Position(math.ceil(-887272 / 10) * 10, 887272 // 10 * 10, 2 * E18)],
        fee_pips=500, tick_spacing=10)
    assert pool.quote_exact_out(E18, True).amount_in == 2001000500250125077
