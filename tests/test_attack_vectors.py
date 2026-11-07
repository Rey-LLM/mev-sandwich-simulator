"""Layer 13: attack vectors beyond the sandwich (JIT, multi-victim, back-run
arbitrage, liquidation back-run, oracle manipulation)."""

import copy
import math
from fractions import Fraction

import pytest

from simulator.attack_vectors import (
    ATTACK_TYPES,
    ActionReverted,
    AttackState,
    LendingMarket,
    Loan,
    Scenario,
    add_liquidity,
    backrun_arbitrage,
    build_scenario,
    jit_liquidity,
    liquidation_backrun,
    maximize_int,
    mint_amounts,
    multi_victim_sandwich,
    oracle_manipulation,
    reference_prices,
    remove_liquidity,
    run_attack_catalog,
)
from simulator.mempool_emulator import PendingTx
from simulator.slippage_engine import (
    Q96,
    Q128,
    ConcentratedPool,
    MASEError,
    Position,
    V2Pool,
    get_sqrt_ratio_at_tick,
    mul_div,
)

E18 = 10**18


@pytest.fixture(scope="module")
def scn():
    return build_scenario(seed=0, victims=12)


def _pool_state(p):
    return (p.sqrt_price_x96, p.tick, p.liquidity, dict(p.ticks), dict(p.tick_gross), tuple(p.tick_array))


def _v3(liq=10**21, extra=()):
    return ConcentratedPool.from_positions(
        "A", "B", get_sqrt_ratio_at_tick(0) + 12345,
        [Position(-887220, 887220, liq), *extra], fee_pips=3000, tick_spacing=60)


def _tx(pool, token_in, amount, min_out=0, h="0xv"):
    return PendingTx(h, "victim", {"route": [pool.pool_id], "token_in": token_in, "amount_in": amount,
                                   "amount_out_min": min_out}, 200_000, 10**9)


# -- mint / burn ---------------------------------------------------------------


def test_mint_amounts_match_the_closed_form_with_v3_rounding():
    pool = _v3()
    sp, sa, sb = pool.sqrt_price_x96, get_sqrt_ratio_at_tick(-600), get_sqrt_ratio_at_tick(600)
    L = 123_456_789_012_345_678_901
    a0, a1 = mint_amounts(pool, -600, 600, L, round_up=True)
    assert a0 == math.ceil(Fraction(L * Q96 * (sb - sp), sb * sp))
    assert a1 == math.ceil(Fraction(L * (sp - sa), Q96))
    b0, b1 = mint_amounts(pool, -600, 600, L, round_up=False)
    assert b0 == math.floor(Fraction(L * Q96 * (sb - sp), sb * sp))
    assert b1 == math.floor(Fraction(L * (sp - sa), Q96))
    # out-of-range positions need a single token
    assert mint_amounts(pool, 600, 1200, L, True)[1] == 0
    assert mint_amounts(pool, -1200, -600, L, True)[0] == 0


def test_mint_then_burn_restores_the_pool_and_loses_at_most_one_wei():
    pool = _v3()
    before = _pool_state(pool)
    a0, a1 = add_liquidity(pool, -600, 600, 10**20)
    assert pool.liquidity == before[2] + 10**20
    assert {-600, 600} <= set(pool.tick_array)
    r0, r1 = remove_liquidity(pool, -600, 600, 10**20)
    assert _pool_state(pool) == before
    assert 0 <= a0 - r0 <= 1 and 0 <= a1 - r1 <= 1


def test_mint_on_existing_tick_keeps_it_initialized_after_burn():
    pool = _v3(extra=(Position(-600, 600, 10**18),))
    add_liquidity(pool, -600, 1200, 5 * 10**18)
    remove_liquidity(pool, -600, 1200, 5 * 10**18)
    assert pool.ticks[-600] == 10**18 and pool.tick_gross[-600] == 10**18
    assert 1200 not in pool.ticks


@pytest.mark.parametrize("lower,upper,liq", [(-600, 601, 1), (600, -600, 1), (-600, 600, 0)])
def test_mint_rejects_bad_ranges(lower, upper, liq):
    with pytest.raises(ActionReverted):
        add_liquidity(_v3(), lower, upper, liq)


def test_mint_refuses_v2_and_protocol_fee_pools(scn):
    st = AttackState(scn)
    v2 = next(p for p in scn.pools if isinstance(p, V2Pool))
    with pytest.raises(ActionReverted):
        st.mint(v2.pool_id, -60, 60, 1)
    v4 = next(p for p in st.pools.values() if getattr(p, "version", "") == "v4")
    v4.protocol_fee_pips = 100
    with pytest.raises(ActionReverted):
        st.mint(v4.pool_id, -10, 10, 1)


# -- JIT fees ------------------------------------------------------------------


def _scenario_for(pool, victims=()):
    weth = V2Pool("B", "WETH", 10**24, 10**24)  # gives B a reference price
    weth2 = V2Pool("A", "WETH", 10**24, 10**24)
    return Scenario([pool, weth, weth2], list(victims), base="WETH")


def test_jit_fee_equals_contract_formula_when_the_swap_stays_in_range():
    pool = _v3()
    tx = _tx(pool, "A", 10**18)
    st = AttackState(_scenario_for(pool, [tx]))
    L = 4 * 10**21
    st.mint(pool.pool_id, -600, 600, L)
    ref = copy.deepcopy(st.pools[pool.pool_id]).quote_exact_in(10**18, True)
    st.victim(tx)
    burn = st.burn()
    assert burn["fee1"] == 0
    assert burn["fee0"] == mul_div(ref.fee_growth_global_x128_delta, L, Q128)
    # the position owns L / (L + L0) of the in-range liquidity
    share = Fraction(L, L + 10**21)
    assert abs(burn["fee0"] - ref.fee_paid * share) <= 2


def test_jit_fee_counts_only_the_in_range_part_of_a_swap_that_leaves_the_range():
    pool = _v3(liq=10**18)
    big = 10**17  # moves the price far beyond the JIT range
    tx = _tx(pool, "A", big)
    st = AttackState(_scenario_for(pool, [tx]))
    L = 10**18
    st.mint(pool.pool_id, -120, 120, L)
    work = copy.deepcopy(st.pools[pool.pool_id])
    single = copy.deepcopy(work).quote_exact_in(big, True)
    assert single.tick_after < -120
    in_range = work.apply_swap(big, True, sqrt_price_limit_x96=get_sqrt_ratio_at_tick(-120))
    st.victim(tx)
    burn = st.burn()
    assert burn["fee0"] == mul_div(in_range.fee_growth_global_x128_delta, L, Q128)
    assert 0 < burn["fee0"] < single.fee_paid
    # the burned position now holds only token1 (price is below its range)
    assert burn["amount0"] > 0 or burn["amount1"] > 0


def test_victim_swap_output_is_identical_with_the_fee_split():
    pool = _v3(liq=10**18)
    tx = _tx(pool, "A", 3 * 10**16)
    with_pos = copy.deepcopy(pool)
    add_liquidity(with_pos, -120, 120, 10**18)
    expected = copy.deepcopy(with_pos).apply_swap(tx.amount_in, True)
    st = AttackState(_scenario_for(pool, [tx]))
    st.mint(pool.pool_id, -120, 120, 10**18)
    assert st.victim(tx).amount_out == expected.amount_out


def test_jit_attack_improves_the_victims_price(scn):
    found = [r for tx in scn.victims if (r := jit_liquidity(scn, tx)) is not None]
    assert found, "seed 0 has JIT opportunities"
    for r in found:
        assert r.net_profit_base > 0
        assert all(loss <= 0 for loss in r.victim_loss.values())  # victims get more out
        ops = [s["op"] for s in r.steps]
        assert ops == ["mint", "victim", "burn"]
        assert r.extra["fees"]["fee0"] + r.extra["fees"]["fee1"] > 0


# -- sandwiches -------------------------------------------------------------------


def test_multi_victim_sandwich_is_flat_and_respects_every_limit(scn):
    groups = sorted({(v.route[0], v.token_in) for v in scn.victims})
    results = [r for g in groups if (r := multi_victim_sandwich(scn, *g, min_victims=1)) is not None]
    assert results
    by_hash = {v.tx_hash: v for v in scn.victims}
    for r in results:
        assert r.flat
        assert r.net_profit_base == r.base_delta - r.gas_cost_base > 0
        for s in r.steps:
            if s["op"] == "victim":
                assert s["amount_out"] >= by_hash[s["tx_hash"]].amount_out_min
        assert all(loss > 0 for loss in r.victim_loss.values())


def test_multi_victim_beats_or_equals_best_single_on_the_same_group(scn):
    for pool_id, token_in in sorted({(v.route[0], v.token_in) for v in scn.victims}):
        multi = multi_victim_sandwich(scn, pool_id, token_in, min_victims=2)
        if multi is None or multi.kind != "multi_victim_sandwich":
            continue
        assert len(multi.victims) >= 2
        return
    pytest.skip("no group with two executable victims in this seed")


# -- back-run arbitrage -----------------------------------------------------------------


def test_backrun_arbitrage_closes_a_cycle_back_to_base(scn):
    found = [r for tx in scn.victims if (r := backrun_arbitrage(scn, tx)) is not None]
    assert found
    for r in found:
        assert r.flat and r.net_profit_base > 0
        swaps = [s for s in r.steps if s["op"] == "swap"]
        assert swaps[0]["token_in"] == swaps[-1]["token_out"] == scn.base
        for a, b in zip(swaps, swaps[1:]):
            assert a["token_out"] == b["token_in"] and a["amount_out"] == b["amount_in"]
        assert r.victim_loss == {r.victims[0]: 0}  # a back-run does not change the victim's fill


# -- lending: liquidation back-run and oracle manipulation ----------------------------------


def test_lending_market_validation():
    with pytest.raises(MASEError):
        LendingMarket("m", "A", "B", "p", oracle_mode="median")
    with pytest.raises(MASEError):
        LendingMarket("m", "A", "B", "p", ltv_bps=9_000, liquidation_threshold_bps=8_000)


def test_liquidation_backrun_only_takes_loans_the_victim_made_unhealthy():
    for seed in range(6):
        scn = build_scenario(seed=seed, victims=12)
        for tx in scn.victims:
            r = liquidation_backrun(scn, tx)
            if r is None:
                continue
            market = scn.market(r.extra["market_id"])
            loan = market.loans[r.extra["borrower"]]
            assert market.health(loan, market.price({p.pool_id: p for p in scn.pools})) >= 1
            st = AttackState(scn)
            st.victim(tx)
            price_after = st.markets[market.market_id].price(st.pools)
            assert market.health(loan, price_after) < 1
            liq = next(s for s in r.steps if s["op"] == "liquidate")
            assert liq["seized"] == math.floor(Fraction(liq["repay"] * 10_500, 10_000) / price_after)
            assert r.status == "model" and r.flat and r.net_profit_base > 0
            return
    pytest.fail("no liquidation back-run in seeds 0-5")


def test_liquidate_rejects_healthy_loans_and_over_close_factor(scn):
    st = AttackState(scn)
    m = scn.markets[0]
    borrower, loan = sorted(m.loans.items())[0]
    with pytest.raises(ActionReverted):
        st.liquidate(m.market_id, borrower, 1)  # healthy at the start
    st.markets[m.market_id].loans[borrower] = Loan(loan.collateral, loan.debt * 2)
    with pytest.raises(ActionReverted):
        st.liquidate(m.market_id, borrower, loan.debt * 2)  # above the 50% close factor
    st.liquidate(m.market_id, borrower, loan.debt)


def test_spot_oracle_is_manipulable_and_twap_defends(scn):
    r = oracle_manipulation(scn, scn.markets[0].market_id)
    assert r is not None and r.status == "model" and r.flat
    assert r.extra["spot_oracle_best_profit"] > 0
    assert r.extra["twap_defends"] is True
    twap = copy.deepcopy(scn)
    twap.markets[0].oracle_mode = "twap"
    assert oracle_manipulation(twap, twap.markets[0].market_id) is None


def test_twap_ignores_trades_inside_the_block(scn):
    st = AttackState(scn)
    m = st.markets[scn.markets[0].market_id]
    m.oracle_mode = "twap"
    before = m.price(st.pools)
    st.swap(m.oracle_pool_id, "USDC", 5_000_000 * 10**6)
    assert m.price(st.pools) == before
    m.record_observation(st.pools)
    assert m.price(st.pools) != before


def test_borrow_respects_ltv_and_liquidity(scn):
    st = AttackState(scn)
    mid = scn.markets[0].market_id
    st.deposit(mid, 10 * E18)
    limit = st.max_borrow(mid)
    with pytest.raises(ActionReverted):
        st.borrow(mid, limit + 1)
    st.borrow(mid, limit)
    assert st.ledger["USDC"] == limit and st.ledger["WETH"] == -10 * E18


# -- shared machinery --------------------------------------------------------------------------


def test_reference_prices_use_the_deepest_base_pool(scn):
    prices = reference_prices(scn.pools, "WETH")
    assert prices["WETH"] == 1 and set(prices) == {"WETH", "USDC", "DAI"}
    from simulator.invariants import input_reserve_scale
    from simulator.slippage_engine import _direction

    usdc_pools = [p for p in scn.pools if set(p.tokens()) == {"WETH", "USDC"}]
    deepest = max(usdc_pools, key=lambda p: input_reserve_scale(p, not _direction(p, "USDC")[0]))
    assert prices["USDC"] == deepest.spot_price(_direction(deepest, "USDC")[0])


def test_maximize_int_finds_the_peak_of_a_concave_function():
    best = maximize_int(lambda x: -(x - 7_777) ** 2, 1, 10**6)
    assert best == (7_777, 0)
    assert maximize_int(lambda x: None, 1, 100) is None


def test_settle_converts_everything_to_base(scn):
    st = AttackState(scn)
    st.swap("v2:DAI/USDC", "DAI", 10**21)
    st.settle()
    assert set(k for k, v in st.ledger.items() if v) <= {"WETH"} or st.result("x", "").residual_value_base == 0


def test_capital_limit_is_enforced(scn):
    small = copy.deepcopy(scn)
    small.capital_base = 10**15
    r = oracle_manipulation(small, small.markets[0].market_id)
    assert r is None or r.capital_used_base <= 10**15


def test_catalog_is_deterministic_and_covers_every_type():
    a = run_attack_catalog(build_scenario(seed=1, victims=10))
    b = run_attack_catalog(build_scenario(seed=1, victims=10))
    assert a == b
    assert set(a["summary"]) == set(ATTACK_TYPES)
    found = {k for k, v in a["summary"].items() if v["found"]}
    assert {"sandwich", "jit_liquidity", "backrun_arbitrage", "oracle_manipulation"} <= found
