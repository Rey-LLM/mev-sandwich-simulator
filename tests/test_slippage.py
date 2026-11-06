"""Verification suite for the Layer 02 MASE slippage engine."""

from __future__ import annotations

import gc
import math
import random
import time
from fractions import Fraction

import pytest

from simulator.slippage_engine import (
    MAX_SQRT_RATIO,
    MAX_TICK,
    MIN_SQRT_RATIO,
    MIN_TICK,
    Q96,
    ConcentratedPool,
    MASEEngine,
    MASEError,
    Position,
    SwapPayload,
    V2Pool,
    V4Pool,
    compute_swap_step,
    evaluate_route,
    evaluate_trade,
    get_amount0_delta,
    get_amount1_delta,
    get_sqrt_ratio_at_tick,
    get_tick_at_sqrt_ratio,
    min_amount_out,
    price_to_sqrt_price_x96,
    sqrt_price_x96_to_price,
    v2_get_amount_in,
    v2_get_amount_out,
)

E18 = 10**18
E6 = 10**6
LATENCY_BUDGET_S = 0.005
SEED = 20261106


def _reference_amount_out(amount_in: int, r_in: int, r_out: int, fee_bps: int = 30) -> int:
    """Independent closed form: floor(x*(1-f)*Y / (X + x*(1-f)))."""
    x_eff = Fraction(amount_in) * Fraction(10_000 - fee_bps, 10_000)
    return math.floor(x_eff * r_out / (r_in + x_eff))


def _time_each(fn, cases):
    """Run fn over cases with GC paused; return per-call durations in seconds."""
    for case in cases[:5]:
        fn(*case)  # warm-up
    durations = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for case in cases:
            start = time.perf_counter()
            fn(*case)
            durations.append(time.perf_counter() - start)
    finally:
        if gc_was_enabled:
            gc.enable()
    return durations


# ---------------------------------------------------------------------------
# Uniswap v2: constant product against known reserve states
# ---------------------------------------------------------------------------

KNOWN_V2_STATES = [
    # (amount_in, reserve_in, reserve_out, expected_out)
    (1 * E18, 100 * E18, 200 * E18, 1_974_316_068_794_122_597),
    (10 * E18, 1_000 * E18, 1_000 * E18, 9_871_580_343_970_612_988),
    (1 * E18, 10**24, 10**24, 996_999_005_991_991_025),
    (1_000, 1_000_000, 1_000_000, 996),
    (1, 2, 2, 0),
]


@pytest.mark.parametrize("amount_in,r_in,r_out,expected", KNOWN_V2_STATES)
def test_v2_known_reserve_states(amount_in, r_in, r_out, expected):
    assert v2_get_amount_out(amount_in, r_in, r_out) == expected
    assert _reference_amount_out(amount_in, r_in, r_out) == expected


def test_v2_matches_997_over_1000_formula():
    rng = random.Random(SEED)
    for _ in range(2_000):
        r_in = rng.randint(10**6, 10**30)
        r_out = rng.randint(10**6, 10**30)
        x = rng.randint(1, r_in)
        solidity = (x * 997 * r_out) // (r_in * 1000 + x * 997)
        assert v2_get_amount_out(x, r_in, r_out) == solidity


def test_v2_random_states_match_reference_and_preserve_k():
    rng = random.Random(SEED + 1)
    for _ in range(2_000):
        r_in = rng.randint(10**9, 10**28)
        r_out = rng.randint(10**9, 10**28)
        fee = rng.choice([0, 5, 30, 100])
        x = rng.randint(1, r_in * 3)
        out = v2_get_amount_out(x, r_in, r_out, fee)
        assert out == _reference_amount_out(x, r_in, r_out, fee)
        assert 0 <= out < r_out
        assert (r_in + x) * (r_out - out) >= r_in * r_out  # k never decreases


def test_v2_amount_in_is_minimal_inverse():
    rng = random.Random(SEED + 2)
    for _ in range(1_000):
        r_in = rng.randint(10**12, 10**27)
        r_out = rng.randint(10**12, 10**27)
        want = rng.randint(1, r_out - 1)
        need = v2_get_amount_in(want, r_in, r_out)
        assert v2_get_amount_out(need, r_in, r_out) >= want
        if need > 1:
            # The on-chain formula adds 1 wei of safety; one less than that
            # must not overshoot by more than that wei.
            assert v2_get_amount_out(need - 2, r_in, r_out) < want or need - 2 == 0


def test_v2_price_impact_monotonic_in_size():
    pool = V2Pool("WETH", "USDC", 5_000 * E18, 15_000_000 * E6)
    last = Fraction(-1)
    for size in [E18 // 100, E18, 10 * E18, 100 * E18, 1_000 * E18]:
        rep = evaluate_trade(pool, "WETH", size)
        assert rep.price_impact > last
        assert rep.price_impact_ex_fee >= 0
        last = rep.price_impact


def test_v2_price_impact_closed_form():
    # Without fees, impact of x into reserve X is exactly x / (X + x).
    pool = V2Pool("A", "B", 1_000 * E18, 1_000 * E18, fee_bps=0)
    rep = evaluate_trade(pool, "A", 100 * E18)
    exact = Fraction(100, 1_100)
    assert abs(rep.price_impact - exact) < Fraction(1, 10**15)


def test_v2_quote_does_not_mutate_and_apply_does():
    pool = V2Pool("A", "B", 100 * E18, 200 * E18)
    res = pool.quote_exact_in(E18, True)
    assert (pool.reserve0, pool.reserve1) == (100 * E18, 200 * E18)
    pool.apply_swap(res, True)
    assert pool.reserve0 == 101 * E18
    assert pool.reserve1 == 200 * E18 - res.amount_out


def test_v2_rejects_bad_input():
    with pytest.raises(MASEError):
        v2_get_amount_out(0, 1, 1)
    with pytest.raises(MASEError):
        v2_get_amount_in(10, 100, 10)
    with pytest.raises(MASEError):
        V2Pool("A", "B", 0, 1)


# ---------------------------------------------------------------------------
# TickMath / SqrtPriceX96
# ---------------------------------------------------------------------------


def test_tick_math_boundaries():
    assert get_sqrt_ratio_at_tick(0) == Q96
    assert get_sqrt_ratio_at_tick(MIN_TICK) == MIN_SQRT_RATIO
    assert get_sqrt_ratio_at_tick(MAX_TICK) == MAX_SQRT_RATIO
    assert get_tick_at_sqrt_ratio(MIN_SQRT_RATIO) == MIN_TICK
    assert get_tick_at_sqrt_ratio(MAX_SQRT_RATIO - 1) == MAX_TICK - 1
    with pytest.raises(MASEError):
        get_sqrt_ratio_at_tick(MAX_TICK + 1)
    with pytest.raises(MASEError):
        get_tick_at_sqrt_ratio(MAX_SQRT_RATIO)


def test_tick_math_against_float_reference():
    rng = random.Random(SEED + 3)
    for _ in range(500):
        tick = rng.randint(-400_000, 400_000)
        exact = get_sqrt_ratio_at_tick(tick)
        approx = (1.0001 ** (tick / 2)) * Q96
        assert abs(exact - approx) / approx < 1e-9


def test_tick_math_monotonic_and_round_trip():
    rng = random.Random(SEED + 4)
    for _ in range(300):
        tick = rng.randint(MIN_TICK, MAX_TICK - 1)
        s0 = get_sqrt_ratio_at_tick(tick)
        s1 = get_sqrt_ratio_at_tick(tick + 1)
        assert s0 < s1
        assert get_tick_at_sqrt_ratio(s0) == tick
        assert get_tick_at_sqrt_ratio(s1 - 1) == tick


def test_price_sqrt_round_trip():
    # ETH/USDC-style raw price: 3000 USDC (6 dp) per ETH (18 dp).
    price = Fraction(3000 * E6, E18)
    sp = price_to_sqrt_price_x96(price)
    back = sqrt_price_x96_to_price(sp)
    assert abs(back - price) / price < Fraction(1, 10**20)


def test_amount_deltas_rounding():
    a, b = get_sqrt_ratio_at_tick(-60), get_sqrt_ratio_at_tick(60)
    L = 10**20
    assert get_amount0_delta(a, b, L, True) - get_amount0_delta(a, b, L, False) in (0, 1)
    assert get_amount1_delta(a, b, L, True) - get_amount1_delta(a, b, L, False) in (0, 1)
    assert get_amount0_delta(a, b, L, True) == get_amount0_delta(b, a, L, True)


def test_swap_step_fee_accounting():
    rng = random.Random(SEED + 5)
    for _ in range(500):
        cur = get_sqrt_ratio_at_tick(rng.randint(-1000, 1000))
        tgt = get_sqrt_ratio_at_tick(rng.randint(-2000, 2000))
        if tgt == cur:
            continue
        L = rng.randint(10**15, 10**25)
        amt = rng.randint(1, 10**22)
        fee = rng.choice([100, 500, 3000, 10000])
        nxt, a_in, a_out, f = compute_swap_step(cur, tgt, L, amt, fee)
        assert a_in + f <= amt
        if nxt != tgt:
            assert a_in + f == amt
        assert a_out >= 0 and f >= 0
        assert min(cur, tgt) <= nxt <= max(cur, tgt)


# ---------------------------------------------------------------------------
# Concentrated liquidity: v3 / v4
# ---------------------------------------------------------------------------


def _full_range_pool(L=10**24, sqrt_p=Q96, fee=3000):
    return ConcentratedPool.from_positions("A", "B", sqrt_p, [Position(-887220, 887220, L)], fee, 60)


def test_full_range_v3_matches_v2_virtual_reserves():
    # Inside a range a v3 pool is x*y=k on virtual reserves x=L/sqrtP, y=L*sqrtP.
    rng = random.Random(SEED + 6)
    for _ in range(200):
        L = rng.randint(10**20, 10**26)
        pool = _full_range_pool(L)
        v2 = V2Pool("A", "B", L, L)
        x = rng.randint(10**6, L // 100)
        zero_for_one = rng.random() < 0.5
        token = "A" if zero_for_one else "B"
        out3 = evaluate_trade(pool, token, x).amount_out
        out2 = evaluate_trade(v2, token, x).amount_out
        assert abs(out3 - out2) <= 2


def test_v3_tick_crossing_changes_liquidity():
    positions = [Position(-600, 600, 10**21), Position(-60, 60, 5 * 10**21)]
    pool = ConcentratedPool.from_positions("A", "B", Q96, positions, 500, 60)
    assert pool.liquidity == 6 * 10**21
    assert pool.tick_array == (-600, -60, 60, 600)
    # Push the price down past tick -60 into the thinner range.
    res = pool.quote_exact_in(3 * 10**19, True)
    assert res.fully_filled
    assert res.ticks_crossed == 1
    assert -600 <= res.tick_after < -60
    snapshot = (pool.sqrt_price_x96, pool.liquidity, pool.tick)
    applied = pool.apply_swap(3 * 10**19, True)
    assert applied == res
    assert pool.liquidity == 10**21
    assert pool.sqrt_price_x96 == res.sqrt_price_after_x96 != snapshot[0]


def test_v3_partial_fill_when_liquidity_runs_out():
    pool = ConcentratedPool.from_positions("A", "B", Q96, [Position(-60, 60, 10**18)], 3000, 60)
    res = pool.quote_exact_in(10**24, True)
    assert not res.fully_filled
    assert res.amount_out < 10**18


def test_v3_exact_out_inverts_exact_in():
    pool = _full_range_pool(10**24)
    fwd = pool.quote_exact_in(10**20, True)
    back = pool.quote_exact_out(fwd.amount_out, True)
    assert back.amount_out == fwd.amount_out
    assert 0 <= back.amount_in - fwd.amount_in <= 2


def test_v3_price_limit_respected():
    pool = _full_range_pool(10**22)
    limit = get_sqrt_ratio_at_tick(-100)
    res = pool.quote_exact_in(10**24, True, sqrt_price_limit_x96=limit)
    assert res.sqrt_price_after_x96 == limit
    assert not res.fully_filled


def test_v4_dynamic_fee_and_protocol_fee():
    base = dict(token0="A", token1="B", sqrt_price_x96=Q96, positions=[Position(-887220, 887220, 10**24)],
                fee_pips=3000, tick_spacing=60)
    v4_static = V4Pool.from_positions(**base)
    v4_dyn = V4Pool.from_positions(**base, dynamic_fee=True, hook_fee_pips=100)
    v4_proto = V4Pool.from_positions(**base, protocol_fee_pips=1000)
    v3 = ConcentratedPool.from_positions(**base)
    x = 10**20
    assert v4_static.quote_exact_in(x, True) == v3.quote_exact_in(x, True)
    assert v4_dyn.effective_fee() == 100
    assert v4_dyn.effective_fee({"fee_override_pips": 50}) == 50
    assert v4_dyn.quote_exact_in(x, True).amount_out > v3.quote_exact_in(x, True).amount_out
    assert v4_proto.effective_fee() == 1000 + 3000 - 3
    assert v4_proto.pool_id.startswith("v4:")


# ---------------------------------------------------------------------------
# Slippage reports and multi-hop routes
# ---------------------------------------------------------------------------


def test_slippage_vector_fields():
    pool = V2Pool("WETH", "USDC", 5_000 * E18, 15_000_000 * E6)
    rep = evaluate_trade(pool, "WETH", 50 * E18, tolerance_bps=100)
    v = rep.vector
    assert v.as_dict() == {"WETH": -50 * E18, "USDC": rep.amount_out}
    assert v.min_amount_out == min_amount_out(rep.amount_out, 100)
    assert v.max_slippage_out == rep.amount_out - v.min_amount_out
    assert rep.spot_price == Fraction(15_000_000 * E6, 5_000 * E18)
    assert rep.price_after < rep.spot_price
    assert 0 < rep.price_impact < 1


def test_multi_hop_route_composes_hops():
    p1 = V2Pool("WETH", "USDC", 5_000 * E18, 15_000_000 * E6)
    p2 = ConcentratedPool.from_positions(
        "DAI", "USDC", price_to_sqrt_price_x96(Fraction(E6, E18)),
        [Position(-887220, 887220, 10**22)], 100, 1,
    )
    rep = evaluate_route([p1, p2], "WETH", E18)
    hop1 = p1.quote_exact_in(E18, True).amount_out
    hop2 = p2.quote_exact_in(hop1, False).amount_out
    assert rep.amount_out == hop2
    assert rep.token_out == "DAI"
    assert len(rep.fees_paid) == 2
    assert rep.spot_price == p1.spot_price(True) * p2.spot_price(False)


def test_route_rejects_disconnected_pools():
    p1 = V2Pool("A", "B", E18, E18)
    p2 = V2Pool("C", "D", E18, E18)
    with pytest.raises(MASEError):
        evaluate_route([p1, p2], "A", 10**15)


# ---------------------------------------------------------------------------
# Mempool payload entry point
# ---------------------------------------------------------------------------


def _engine():
    v2 = V2Pool("WETH", "USDC", 5_000 * E18, 15_000_000 * E6, pool_id="uni-v2-weth-usdc")
    v3 = ConcentratedPool.from_positions(
        "WETH", "USDC", price_to_sqrt_price_x96(Fraction(3000 * E6, E18)),
        [Position(-887220, 887220, 10**18)], 500, 10, pool_id="uni-v3-weth-usdc-500",
    )
    return MASEEngine([v2, v3])


def test_engine_evaluates_dict_payload():
    engine = _engine()
    out = engine.evaluate_payload({
        "hash": "0xabc",
        "pool_id": "uni-v2-weth-usdc",
        "token_in": "WETH",
        "amount_in": hex(10 * E18),
        "amount_out_min": str(29_000 * E6),
        "slippage_bps": 75,
    })
    assert out["tx_hash"] == "0xabc"
    assert out["amount_in"] == 10 * E18
    assert out["tolerance_bps"] == 75
    assert out["slack_out"] == out["amount_out"] - 29_000 * E6
    assert out["would_revert"] is False
    assert 0 < out["implied_tolerance"] < 1


def test_engine_flags_reverting_payload():
    engine = _engine()
    ev = engine.evaluate(SwapPayload(
        tx_hash="0x1", route=("uni-v2-weth-usdc",), token_in="WETH",
        amount_in=10 * E18, amount_out_min=40_000 * E6,
    ))
    assert ev.would_revert
    assert ev.slack_out < 0


def test_engine_rejects_unknown_pool_and_bad_payload():
    engine = _engine()
    with pytest.raises(MASEError):
        engine.evaluate_payload({"pool_id": "nope", "token_in": "WETH", "amount_in": 1})
    with pytest.raises(MASEError):
        engine.evaluate_payload({"token_in": "WETH", "amount_in": 1})


def test_engine_batch():
    engine = _engine()
    payloads = [{"pool_id": pid, "token_in": "WETH", "amount_in": E18}
                for pid in ("uni-v2-weth-usdc", "uni-v3-weth-usdc-500")]
    results = engine.evaluate_batch(payloads)
    assert [r["pool_ids"][0] for r in results] == ["uni-v2-weth-usdc", "uni-v3-weth-usdc-500"]
    assert all(r["amount_out"] > 0 for r in results)


# ---------------------------------------------------------------------------
# Latency budget: every evaluation strictly under 5 ms
# ---------------------------------------------------------------------------


def test_latency_v2_evaluation_under_5ms():
    rng = random.Random(SEED + 7)
    cases = []
    for _ in range(500):
        r0, r1 = rng.randint(10**18, 10**27), rng.randint(10**18, 10**27)
        cases.append((V2Pool("A", "B", r0, r1), "A", rng.randint(1, r0)))
    durations = _time_each(evaluate_trade, cases)
    assert max(durations) < LATENCY_BUDGET_S, f"slowest v2 evaluation {max(durations) * 1e3:.3f} ms"


def test_latency_v3_tick_crossing_evaluation_under_5ms():
    # 40 stacked ranges so large swaps cross many initialized ticks.
    positions = [Position(-60 * i, 60 * i, 10**20) for i in range(1, 41)]
    pool = ConcentratedPool.from_positions("A", "B", Q96, positions, 3000, 60)
    rng = random.Random(SEED + 8)
    cases = [(pool, rng.choice("AB"), rng.randint(10**15, 10**22)) for _ in range(300)]
    durations = _time_each(evaluate_trade, cases)
    assert max(durations) < LATENCY_BUDGET_S, f"slowest v3 evaluation {max(durations) * 1e3:.3f} ms"
    assert evaluate_trade(pool, "A", 10**22).ticks_crossed > 10


def test_latency_payload_evaluation_under_5ms():
    engine = _engine()
    rng = random.Random(SEED + 9)
    payloads = [({"pool_id": rng.choice(["uni-v2-weth-usdc", "uni-v3-weth-usdc-500"]),
                  "token_in": rng.choice(["WETH", "USDC"]),
                  "amount_in": rng.randint(10**6, 10**20)},) for _ in range(300)]
    durations = _time_each(engine.evaluate_payload, payloads)
    assert max(durations) < LATENCY_BUDGET_S, f"slowest payload evaluation {max(durations) * 1e3:.3f} ms"
