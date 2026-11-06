"""Verification suite for Layer 06: multi-hop cross-pool router.

Covers the token graph, path and cycle enumeration, the Bellman-Ford
cross-check, sequential v2 / v3 / v4 execution with state carried between
hops (including routes that revisit a pool), exact combined slippage
degradation vectors, cycle optimisation and arbitrage scans on the capped
worker pool, order splitting, and integration with the Layer 04 builder.
"""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from simulator.__main__ import main as cli_main
from simulator.block_builder import BuilderState, Status, execute_tx
from simulator.config import ConfigError, SimulationConfig
from simulator.controller import SimulationController, demo_market
from simulator.multi_hop_router import (
    MultiHopRouter,
    RoutePath,
    RouterError,
    TokenGraph,
    cycle_rate,
    evaluate_cycle,
    execute_path,
    optimize_cycle_input,
    pool_fee_fraction,
    pool_tick,
    quote_path_output,
    split_route,
    v2_implied_sqrt_price_x96,
    validate_path,
)
from simulator.parallel import WorkerPool
from simulator.slippage_engine import (
    ConcentratedPool,
    Position,
    V2Pool,
    V4Pool,
    evaluate_route,
    evaluate_trade,
    get_tick_at_sqrt_ratio,
    price_to_sqrt_price_x96,
)

E18 = 10**18
E6 = 10**6
V2_WU = "v2:WETH/USDC"
V3_UW = "v3:USDC/WETH/500"
V2_UD = "v2:USDC/DAI"
V4_WD = "v4:WETH/DAI/500"
V2_DL = "v2:DAI/LINK"
V2_LW = "v2:LINK/WETH"


def market(skew: float = 1.0, link: bool = False, v4_dynamic: bool = False):
    """WETH/USDC (v2 + v3), USDC/DAI (v2), WETH/DAI (v4); ``skew`` reprices the v3 pool."""
    v3_price = Fraction(E18, int(3000 * skew * E6))
    pools = [
        V2Pool("WETH", "USDC", 1_000 * E18, 3_000_000 * E6),
        ConcentratedPool.from_positions("USDC", "WETH", price_to_sqrt_price_x96(v3_price),
                                        [Position(-887220, 887220, 3 * 10**17)], fee_pips=500, tick_spacing=60),
        V2Pool("USDC", "DAI", 5_000_000 * E6, 5_000_000 * E18),
        V4Pool.from_positions("WETH", "DAI", price_to_sqrt_price_x96(3000), [Position(-887270, 887270, 10**23)],
                              fee_pips=500, tick_spacing=10, dynamic_fee=v4_dynamic, hook_fee_pips=1_200),
    ]
    if link:
        pools += [V2Pool("DAI", "LINK", 1_500_000 * E18, 100_000 * E18),
                  V2Pool("LINK", "WETH", 100_000 * E18, 500 * E18)]
    return pools


def pools_map(pools):
    return {p.pool_id: p for p in pools}


def state_of(pools):
    return [(p.pool_id, p.reserve0, p.reserve1) if isinstance(p, V2Pool)
            else (p.pool_id, p.sqrt_price_x96, p.tick, p.liquidity, tuple(sorted(p.ticks.items()))) for p in pools]


def narrow_cl():
    """A v3 pool with stacked narrow ranges so large swaps cross ticks."""
    sp = price_to_sqrt_price_x96(Fraction(E18, 3000 * E6))
    t = get_tick_at_sqrt_ratio(sp) // 60 * 60
    pos = [Position(t - 600, t + 600, 2 * 10**17), Position(t - 120, t + 120, 6 * 10**17),
           Position(-887220, 887220, 10**16)]
    return ConcentratedPool.from_positions("USDC", "WETH", sp, pos, fee_pips=3000, tick_spacing=60)


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------


def test_graph_has_two_directed_edges_per_pool_and_sorted_tokens():
    g = TokenGraph(market())
    assert g.tokens == ["DAI", "USDC", "WETH"]
    assert sum(len(v) for v in g.adjacency.values()) == 2 * 4
    weth = g.edges_from("WETH")
    assert [(e.token_out, e.pool_id) for e in weth] == sorted((e.token_out, e.pool_id) for e in weth)
    assert set(g.pools_between("WETH", "USDC")) == {V2_WU, V3_UW}


def test_graph_rejects_duplicate_pool_ids():
    p = V2Pool("A", "B", 10**6, 10**6)
    with pytest.raises(RouterError):
        TokenGraph([p, V2Pool("A", "B", 10**6, 10**6)])


def test_edge_rate_is_spot_times_fee_complement_and_log_weight_matches():
    g = TokenGraph(market())
    e = g.edge(V2_WU, "WETH")
    assert e.rate == e.spot * (1 - Fraction(30, 10_000))
    assert e.log_weight == pytest.approx(-__import__("math").log(float(e.rate)))
    with pytest.raises(RouterError):
        g.edge(V2_UD, "WETH")


@pytest.mark.parametrize("tokens,pools", [
    (("A",), ()),  # zero hops
    (("A", "B"), ("p1", "p2")),  # length mismatch
    (("A", "A"), ("p1",)),  # self swap
])
def test_route_path_rejects_malformed_shapes(tokens, pools):
    with pytest.raises(RouterError):
        RoutePath(tokens, pools)


def test_route_path_from_route_and_validation():
    pm = pools_map(market())
    p = RoutePath.from_route(pm, "WETH", [V2_WU, V2_UD])
    assert p.tokens == ("WETH", "USDC", "DAI") and p.hops == 2 and not p.is_cycle
    validate_path(pm, p)
    with pytest.raises(RouterError):
        RoutePath.from_route(pm, "WETH", ["v2:NOPE"])
    with pytest.raises(RouterError):
        validate_path(pm, RoutePath(("WETH", "DAI"), (V2_WU,)))


def test_cycle_rotation_reversal_and_canonical_key():
    pm = pools_map(market())
    c = RoutePath.from_route(pm, "WETH", [V2_WU, V2_UD, V4_WD])
    assert c.is_cycle
    r1 = c.rotate(1)
    assert r1.tokens == ("USDC", "DAI", "WETH", "USDC") and r1.pool_ids == (V2_UD, V4_WD, V2_WU)
    validate_path(pm, r1)
    assert c.canonical_cycle_key() == r1.canonical_cycle_key() == c.rotate(2).canonical_cycle_key()
    assert c.reversed().canonical_cycle_key() != c.canonical_cycle_key()
    assert c.rotate(3) == c
    with pytest.raises(RouterError):
        RoutePath.from_route(pm, "WETH", [V2_WU]).rotate(1)


# ---------------------------------------------------------------------------
# Path and cycle enumeration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_hops,expected", [(1, 1), (2, 3), (3, 3)])
def test_enumerate_paths_respects_hop_limit(max_hops, expected):
    g = TokenGraph(market())
    paths = g.enumerate_paths("WETH", "DAI", max_hops)
    assert len(paths) == expected
    for p in paths:
        assert p.hops <= max_hops and p.token_in == "WETH" and p.token_out == "DAI"
        assert len(set(p.tokens)) == len(p.tokens) and not p.reuses_pool


def test_enumerate_paths_with_four_tokens_and_limit():
    g = TokenGraph(market(link=True))
    all3 = g.enumerate_paths("WETH", "LINK", 3)
    assert any(p.hops == 3 for p in all3) and all(p.hops <= 3 for p in all3)
    assert len(g.enumerate_paths("WETH", "LINK", 3, limit=2)) == 2
    keys = [p.key for p in all3]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("bad", [0, 7])
def test_enumeration_rejects_bad_hop_counts(bad):
    g = TokenGraph(market())
    with pytest.raises(RouterError):
        g.enumerate_paths("WETH", "DAI", bad)
    with pytest.raises(RouterError):
        g.enumerate_cycles(bad)


def test_enumerate_paths_rejects_same_token():
    with pytest.raises(RouterError):
        TokenGraph(market()).enumerate_paths("WETH", "WETH")


def test_cycles_are_closed_unique_and_never_reuse_pools():
    g = TokenGraph(market())
    cycles = g.enumerate_cycles(3)
    raw = g.enumerate_cycles(3, dedupe_rotations=False)
    assert len(raw) > len(cycles) > 0
    keys = [c.canonical_cycle_key() for c in cycles]
    assert len(keys) == len(set(keys))
    for c in cycles:
        assert c.is_cycle and 2 <= c.hops <= 3 and not c.reuses_pool
    two_hop = [c for c in cycles if c.hops == 2]
    assert all(set(c.pool_ids) == {V2_WU, V3_UW} for c in two_hop) and len(two_hop) == 2


def test_four_hop_cycles_appear_only_with_max_hops_four():
    g = TokenGraph(market(link=True))
    assert not any(c.hops == 4 for c in g.enumerate_cycles(3))
    assert any(c.hops == 4 for c in g.enumerate_cycles(4))


def test_cycles_through_filters_by_pool():
    g = TokenGraph(market())
    assert all(V2_UD in c.pool_ids for c in g.cycles_through([V2_UD]))
    assert g.cycles_through([V2_UD]) and len(g.cycles_through([V2_UD])) < len(g.enumerate_cycles())


# ---------------------------------------------------------------------------
# Marginal rates and Bellman-Ford
# ---------------------------------------------------------------------------


def test_balanced_market_has_no_marginal_arbitrage():
    pm = pools_map(market())
    g = TokenGraph(pm.values())
    assert all(cycle_rate(pm, c) < 1 for c in g.enumerate_cycles(3))
    assert g.negative_cycles() == []


@pytest.mark.parametrize("skew", [0.95, 1.05])
def test_skewed_market_produces_negative_cycles(skew):
    pm = pools_map(market(skew))
    g = TokenGraph(pm.values())
    neg = g.negative_cycles()
    assert neg
    for c in neg:
        assert cycle_rate(pm, c) > 1 and c.is_cycle and not c.reuses_pool


def test_bellman_ford_cycles_are_found_by_the_profit_scan():
    r = MultiHopRouter(market(1.05))
    scan = r.scan_cycles()
    profitable = {o.path.canonical_cycle_key() for o in scan.opportunities}
    assert scan.negative_cycles
    assert all(c.canonical_cycle_key() in profitable for c in scan.negative_cycles)


# ---------------------------------------------------------------------------
# Sequential cross-protocol execution
# ---------------------------------------------------------------------------


def test_hops_chain_outputs_into_inputs_across_v2_v3_v4():
    pm = pools_map(market())
    path = RoutePath.from_route(pm, "USDC", [V3_UW, V4_WD, V2_UD])
    ex = execute_path(pm, path, 50_000 * E6)
    assert [h.version for h in ex.hops] == ["v3", "v4", "v2"]
    for a, b in zip(ex.hops, ex.hops[1:]):
        assert a.amount_out == b.amount_in and a.token_out == b.token_in
    assert ex.amount_out == ex.hops[-1].amount_out == quote_path_output(pm, path, 50_000 * E6)


def test_execute_path_is_pure_unless_mutate():
    pools = market()
    pm = pools_map(pools)
    before = state_of(pools)
    path = RoutePath.from_route(pm, "WETH", [V2_WU, V2_UD])
    execute_path(pm, path, 10 * E18)
    assert state_of(pools) == before
    ex = execute_path(pm, path, 10 * E18, mutate=True)
    assert state_of(pools) != before
    assert pm[V2_WU].reserve0 == ex.hops[0].reserves_after[0]
    assert pm[V2_UD].reserve1 == ex.hops[1].reserves_after[1]


def test_mutating_execution_moves_prices_the_right_way():
    pools = market()
    pm = pools_map(pools)
    k_before = pm[V2_WU].k
    sp_before = pm[V3_UW].sqrt_price_x96
    path = RoutePath.from_route(pm, "WETH", [V2_WU, V3_UW])  # WETH -> USDC (v2) -> WETH (v3: USDC in, zfo)
    ex = execute_path(pm, path, 5 * E18, mutate=True)
    assert pm[V2_WU].k >= k_before
    assert ex.hops[1].zero_for_one and pm[V3_UW].sqrt_price_x96 < sp_before
    assert ex.hops[1].sqrt_price_after_x96 == pm[V3_UW].sqrt_price_x96
    assert ex.hops[1].tick_after == pm[V3_UW].tick


def test_min_amount_out_failure_is_atomic():
    pools = market()
    pm = pools_map(pools)
    before = state_of(pools)
    path = RoutePath.from_route(pm, "WETH", [V2_WU, V2_UD])
    out = quote_path_output(pm, path, 3 * E18)
    with pytest.raises(RouterError, match="slippage"):
        execute_path(pm, path, 3 * E18, mutate=True, min_amount_out=out + 1)
    assert state_of(pools) == before
    execute_path(pm, path, 3 * E18, mutate=True, min_amount_out=out)
    assert state_of(pools) != before


def test_v2_hops_report_implied_sqrt_price_and_tick():
    pm = pools_map(market())
    v2 = pm[V2_WU]
    assert v2_implied_sqrt_price_x96(v2) ** 2 <= (v2.reserve1 << 192) // v2.reserve0
    ex = execute_path(pm, RoutePath.from_route(pm, "WETH", [V2_WU]), E18)
    h = ex.hops[0]
    assert h.sqrt_price_after_x96 < h.sqrt_price_before_x96  # WETH in: USDC per WETH falls
    v3_equiv = ConcentratedPool.from_positions("WETH", "USDC", v2_implied_sqrt_price_x96(v2),
                                               [Position(-887220, 887220, 10**18)])
    assert abs(pool_tick(v2) - v3_equiv.tick) <= 1


def test_large_swap_crosses_initialized_ticks_and_changes_liquidity():
    cl = narrow_cl()
    pm = {cl.pool_id: cl}
    path = RoutePath.from_route(pm, "WETH", [cl.pool_id])
    ex = execute_path(pm, path, 300 * E18)
    h = ex.hops[0]
    assert h.ticks_crossed >= 2 and h.liquidity_after != h.liquidity_before
    assert h.tick_after > h.tick_before  # WETH (token1) in: price of token0 in token1 rises
    small = execute_path(pm, path, E18 // 100).hops[0]
    assert small.ticks_crossed == 0 and small.liquidity_after == small.liquidity_before


def test_route_revisiting_a_pool_sees_shifted_state():
    pm = pools_map(market())
    path = RoutePath(("WETH", "USDC", "WETH", "USDC"), (V2_WU, V3_UW, V2_WU))
    assert path.reuses_pool
    ex = execute_path(pm, path, 20 * E18)
    assert [h.revisit for h in ex.hops] == [False, False, True]
    third = ex.hops[2]
    assert third.reserves_before == ex.hops[0].reserves_after  # state carried, not the original reserves
    fresh = pm[V2_WU].quote_exact_in(third.amount_in, True).amount_out
    assert third.amount_out < fresh
    assert quote_path_output(pm, path, 20 * E18) == ex.amount_out


def test_unfillable_and_dust_routes_raise_or_quote_none():
    thin = ConcentratedPool.from_positions("USDC", "WETH", price_to_sqrt_price_x96(Fraction(E18, 3000 * E6)),
                                           [Position(-600, 600, 10**9)], fee_pips=3000, tick_spacing=60)
    pm = {thin.pool_id: thin}
    path = RoutePath.from_route(pm, "WETH", [thin.pool_id])
    with pytest.raises(RouterError, match="partial fill"):
        execute_path(pm, path, 10**30)
    assert quote_path_output(pm, path, 10**30) is None
    v2 = pools_map(market())
    with pytest.raises(RouterError):
        execute_path(v2, RoutePath.from_route(v2, "WETH", [V2_WU]), 1)  # 1 wei of WETH rounds to 0 USDC
    with pytest.raises(RouterError):
        execute_path(v2, RoutePath.from_route(v2, "USDC", [V2_WU]), 0)


def test_dynamic_v4_fee_is_used_for_the_hop():
    pm = pools_map(market(v4_dynamic=True))
    ex = execute_path(pm, RoutePath.from_route(pm, "WETH", [V4_WD]), E18)
    assert ex.hops[0].fee == Fraction(1_200, 10**6) == pool_fee_fraction(pm[V4_WD])


# ---------------------------------------------------------------------------
# Slippage degradation vectors
# ---------------------------------------------------------------------------


def _three_hop():
    pm = pools_map(market())
    return pm, execute_path(pm, RoutePath.from_route(pm, "WETH", [V2_WU, V2_UD, V4_WD]), 40 * E18)


def test_combined_impact_equals_compounded_hop_impacts_exactly():
    _, ex = _three_hop()
    d = ex.degradation
    assert d.combined_impact == d.compounded_impact
    assert isinstance(d.combined_impact, Fraction)


def test_additive_impact_overstates_and_interaction_is_non_negative():
    _, ex = _three_hop()
    d = ex.degradation
    assert all(0 <= x < 1 for x in d.per_hop_impact)
    assert d.additive_impact >= d.combined_impact and d.interaction >= 0
    assert d.interaction > 0  # three non-trivial hops always compound


def test_cumulative_vector_is_monotone_and_ends_at_combined():
    _, ex = _three_hop()
    v = ex.degradation.cumulative_vector()
    assert len(v) == 3 and list(v) == sorted(v) and v[-1] == ex.degradation.combined_impact
    assert v[0] == ex.degradation.per_hop_impact[0]


def test_fee_and_curve_drag_decompose_the_loss():
    pm, ex = _three_hop()
    d = ex.degradation
    fees = [Fraction(30, 10_000), Fraction(30, 10_000), Fraction(500, 10**6)]
    expected = 1 - (1 - fees[0]) * (1 - fees[1]) * (1 - fees[2])
    assert d.fee_drag == expected
    assert 0 < d.curve_drag < d.combined_impact
    assert (1 - d.fee_drag) * (1 - d.curve_drag) == 1 - d.combined_impact
    assert d.spot_amount_out >= d.ideal_amount_out >= ex.amount_out and d.curve_loss >= 0


def test_impact_grows_with_trade_size():
    pm = pools_map(market())
    path = RoutePath.from_route(pm, "WETH", [V3_UW, V2_UD])
    impacts = [execute_path(pm, path, a).degradation.combined_impact for a in (E18 // 10, 5 * E18, 50 * E18)]
    assert impacts == sorted(impacts) and impacts[0] < impacts[-1]


def test_single_hop_degradation_matches_layer02_report():
    pm = pools_map(market())
    for pid, tok, amt in ((V2_WU, "WETH", 7 * E18), (V3_UW, "USDC", 25_000 * E6), (V4_WD, "DAI", 9_000 * E18)):
        ex = execute_path(pm, RoutePath.from_route(pm, tok, [pid]), amt)
        rep = evaluate_trade(pm[pid], tok, amt)
        assert ex.amount_out == rep.amount_out
        assert ex.degradation.combined_impact == rep.price_impact
        assert ex.hops[0].curve_impact == rep.price_impact_ex_fee


def test_multi_hop_output_matches_layer02_evaluate_route():
    pm = pools_map(market())
    route = [V3_UW, V4_WD, V2_UD]
    ex = execute_path(pm, RoutePath.from_route(pm, "USDC", route), 80_000 * E6)
    rep = evaluate_route([pm[p] for p in route], "USDC", 80_000 * E6)
    assert ex.amount_out == rep.amount_out and ex.degradation.combined_impact == rep.price_impact
    assert json.dumps(ex.to_dict())  # serialisable


# ---------------------------------------------------------------------------
# Cycle optimisation and arbitrage scans
# ---------------------------------------------------------------------------


def test_optimize_cycle_input_none_without_arbitrage():
    pm = pools_map(market())
    for c in TokenGraph(pm.values()).enumerate_cycles(3):
        assert optimize_cycle_input(pm, c) is None
        assert evaluate_cycle(pm, c) is None


def test_optimum_beats_its_neighbours():
    pm = pools_map(market(1.05))
    best = MultiHopRouter(pm.values()).scan_cycles().best
    a, p = optimize_cycle_input(pm, best.path)
    assert p > 0 and p == best.profit
    for f in (Fraction(9, 10), Fraction(11, 10), Fraction(1, 2), Fraction(2)):
        other = quote_path_output(pm, best.path, int(a * f))
        assert other is None or other - int(a * f) <= p


def test_optimize_cycle_input_requires_cycle_and_handles_empty_window():
    pm = pools_map(market(1.05))
    with pytest.raises(RouterError):
        optimize_cycle_input(pm, RoutePath.from_route(pm, "WETH", [V2_WU]))
    c = TokenGraph(pm.values()).negative_cycles()[0]
    assert optimize_cycle_input(pm, c, lo=10, hi=5) is None


def test_net_profit_subtracts_gas_and_min_filter_applies():
    r = MultiHopRouter(market(1.05))
    free = r.scan_cycles(gas_price_wei=0)
    paid = r.scan_cycles(gas_price_wei=50 * 10**9)
    for o in paid.opportunities:
        assert o.net_profit_native == o.profit_native - o.gas_estimate * 50 * 10**9
    huge = max(o.profit_native for o in free.opportunities) + 1
    assert r.scan_cycles(min_net_profit_native=huge).opportunities == []
    nets = [o.net_profit_native for o in paid.opportunities]
    assert nets == sorted(nets, reverse=True)


def test_scan_through_pools_restricts_cycles():
    r = MultiHopRouter(market(1.05))
    rep = r.scan_cycles(through_pools=[V2_UD])
    assert rep.cycles_evaluated < r.scan_cycles().cycles_evaluated
    assert all(V2_UD in o.path.pool_ids for o in rep.opportunities)


def test_executing_best_cycle_realises_reported_profit_and_shrinks_the_gap():
    r = MultiHopRouter(market(1.05))
    best = r.scan_cycles().best
    rate_before = cycle_rate(r.pools, best.path)
    ex = r.execute(best.path, best.amount_in)
    assert ex.profit == best.profit
    assert cycle_rate(r.pools, best.path) < rate_before
    after = evaluate_cycle(r.pools, best.path)
    assert after is None or after.profit < best.profit


def test_repeated_arbitrage_converges():
    r = MultiHopRouter(market(1.08))
    profits = []
    for _ in range(6):
        best = r.scan_cycles().best
        if best is None:
            break
        profits.append(best.profit_native)
        r.execute(best.path, best.amount_in)
    assert profits and profits[-1] <= profits[0]
    final = r.scan_cycles().best
    assert final is None or final.profit_native < profits[0]


def test_parallel_cycle_scan_matches_serial():
    r = MultiHopRouter(market(1.05, link=True), max_hops=4)
    serial = r.scan_cycles(gas_price_wei=10**9)
    with WorkerPool(2, parallel_threshold=1) as wp:
        par = r.scan_cycles(gas_price_wei=10**9, executor=wp)
    assert [o.to_dict() for o in serial.opportunities] == [o.to_dict() for o in par.opportunities]
    assert serial.cycles_evaluated == par.cycles_evaluated > 0
    assert json.dumps(par.summary())


# ---------------------------------------------------------------------------
# Routing and order splitting
# ---------------------------------------------------------------------------


def test_best_route_ranks_by_output():
    r = MultiHopRouter(market(1.05))
    q = r.best_route("WETH", "DAI", 10 * E18)
    outs = [o for _, o in q.ranked if o is not None]
    assert outs == sorted(outs, reverse=True)
    assert q.best is not None and q.best.amount_out == outs[0] and q.paths_evaluated == len(q.ranked)


def test_best_route_parallel_matches_serial():
    r = MultiHopRouter(market(1.02, link=True), max_hops=3)
    serial = r.best_route("USDC", "LINK", 20_000 * E6)
    with WorkerPool(2, parallel_threshold=1) as wp:
        par = r.best_route("USDC", "LINK", 20_000 * E6, executor=wp)
    assert serial.ranked == par.ranked


def test_split_never_worse_than_best_single_path():
    pm = pools_map(market())
    g = TokenGraph(pm.values())
    paths = g.enumerate_paths("WETH", "DAI", 2)
    for amount in (E18, 100 * E18, 600 * E18):
        s = split_route(pm, paths, amount, parts=20)
        assert s.amount_out >= s.best_single_out and sum(s.allocations) == amount
        assert s.amount_out == sum(s.outputs)


def test_large_order_actually_splits():
    s = MultiHopRouter(market()).split("WETH", "DAI", 500 * E18, parts=16)
    assert sum(1 for a in s.allocations if a) >= 2 and s.improvement > 0
    assert json.dumps(s.to_dict())


def test_split_paths_sharing_a_pool_account_for_each_other():
    pm = pools_map(market())
    # Both paths end on the v2 USDC/DAI pool, so each leg moves the other's last hop.
    a = RoutePath.from_route(pm, "WETH", [V2_WU, V2_UD])
    b = RoutePath.from_route(pm, "WETH", [V3_UW, V2_UD])
    s = split_route(pm, [a, b], 300 * E18, parts=12)
    assert all(s.allocations)
    independent = quote_path_output(pm, a, s.allocations[0]) + quote_path_output(pm, b, s.allocations[1])
    assert s.amount_out < independent  # shared state: the legs' impacts compound
    assert s.amount_out >= s.best_single_out


def test_split_validation_errors():
    pm = pools_map(market())
    a = RoutePath.from_route(pm, "WETH", [V2_WU])
    b = RoutePath.from_route(pm, "WETH", [V4_WD])
    with pytest.raises(RouterError):
        split_route(pm, [a, b], E18)  # different token_out
    with pytest.raises(RouterError):
        split_route(pm, [a], E18, parts=0)
    with pytest.raises(RouterError):
        split_route(pm, [], E18)
    with pytest.raises(RouterError):
        split_route(pm, [a], 0)


def test_router_execute_refreshes_graph_prices():
    r = MultiHopRouter(market())
    before = r.graph.edge(V2_WU, "WETH").spot
    r.execute(RoutePath.from_route(r.pools, "WETH", [V2_WU]), 25 * E18)
    assert r.graph.edge(V2_WU, "WETH").spot < before


def test_router_tx_executes_in_the_block_builder():
    pools = market()
    r = MultiHopRouter(pools)
    path = RoutePath.from_route(r.pools, "WETH", [V2_WU, V2_UD, V4_WD])
    with pytest.raises(RouterError):
        r.build_tx(RoutePath(("WETH", "USDC", "WETH", "USDC"), (V2_WU, V3_UW, V2_WU)), E18, "s", 0, 10**9)
    tx = r.build_tx(path, 2 * E18, "alice", 0, 10 * 10**9)
    assert tx.route == path.pool_ids and tx.amount_out_min == quote_path_output(r.pools, path, 2 * E18)
    st = BuilderState.from_pools(pools)
    t = execute_tx(st, tx, 10 * 10**9, 30_000_000)
    assert t.status is Status.SUCCESS
    assert st.fork.balances["alice"]["WETH"] == tx.amount_out_min - 2 * E18


def test_best_backrun_after_victim_trade():
    pools = market()
    r = MultiHopRouter(pools)
    assert r.best_backrun([V2_WU]) is None
    work = {p.pool_id: p for p in market()}
    execute_path(work, RoutePath.from_route(work, "WETH", [V2_WU]), 60 * E18, mutate=True)
    opp = r.best_backrun([V2_WU], pools=work)
    assert opp is not None and V2_WU in opp.path.pool_ids and opp.profit > 0
    assert opp.path.token_in == "WETH"  # native token preferred as the cycle's start


# ---------------------------------------------------------------------------
# Config, controller and CLI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field,value", [("router_max_hops", 0), ("router_max_hops", 7),
                                         ("private_flow_bps", 10_001), ("mev_share_refund_bps", -1)])
def test_config_rejects_out_of_range_layer05_06_fields(field, value):
    with pytest.raises(ConfigError):
        SimulationConfig.for_mode("moderate", cpus=4, **{field: value})


def test_controller_moderate_report_includes_router_scan():
    cfg = SimulationConfig.for_mode("moderate", cpus=4, txs_per_block=4, fuzz_iterations=8, parallel_threshold=10**9)
    out = SimulationController(cfg).run()
    router = out["result"]["router"]
    assert router["pools"] == 3 and router["max_hops"] == 3 and "cycles_evaluated" in router
    assert "private_relay" not in out["result"]


def test_cli_max_hops_flag(capsys):
    rc = cli_main(["--mode", "moderate", "--txs", "4", "--fuzz", "8", "--max-hops", "2"])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0 and data["config"]["router_max_hops"] == 2 and data["result"]["router"]["max_hops"] == 2


def test_demo_market_router_scan_is_clean():
    rep = MultiHopRouter(demo_market()).scan_cycles()
    assert rep.tokens == 3 and rep.cycles_evaluated >= 1
    assert rep.best is None  # the demo pools are priced consistently
