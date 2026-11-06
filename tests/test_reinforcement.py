"""Layer 08: online RL engine, competitor bots and the bundle eviction model."""

from __future__ import annotations

import dataclasses
import math
import random
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import pytest

from simulator.block_builder import (
    GWEI,
    BlockBuilder,
    BuilderState,
    BuilderTx,
    Bundle,
    CompetitorBot,
    CompetitorStrategy,
    FeeStructure,
    Strategy,
    TxKind,
    _quote_hops,
    apply_eviction,
    best_sandwich,
    bundle_payer,
    bundles_conflict,
    default_competitors,
    make_sandwich_bundle,
    outbid_value,
    quote_eviction,
    run_competition,
    sandwich_economics,
    score_bundle,
    simulate_route_sandwich,
)
from simulator.config import ConfigError, SimulationConfig, resolve_worker_count, worker_ceiling, worker_floor
from simulator.parallel import WorkerPool
from simulator.reinforcement_engine import (
    AGENT,
    REGIMES,
    RL_SHARDS,
    Action,
    ActionTable,
    EnvConfig,
    FixedPolicy,
    LearningSchedule,
    MarketRegime,
    NoisyBanditEnvironment,
    Observation,
    PolicyGradientAgent,
    QLearningAgent,
    ReinforcementSimulation,
    RewardScaler,
    SandwichEnvironment,
    StateBins,
    action_grid,
    benchmark_strategies,
    bucket,
    make_agent,
    parallel_policy_iteration,
    run_agent,
    train_bandit,
)
from simulator.slippage_engine import V2Pool, min_amount_out

E18 = 10**18
E6 = 10**6
BASE = 20 * GWEI
PID = "v2:WETH/USDC"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def market(extra_pool: bool = False) -> BuilderState:
    pools = [V2Pool("WETH", "USDC", 4_000 * E18, 12_000_000 * E6)]
    if extra_pool:
        pools.append(V2Pool("WETH", "DAI", 4_000 * E18, 12_000_000 * E18))
    return BuilderState.from_pools(pools)


def victim(state: BuilderState, amount: int = 40 * E18, token: str = "WETH", tol: int = 100,
           pool_id: str = PID, tx_hash: str = "0xvictim", sender: str = "user1") -> BuilderTx:
    probe = BuilderTx("p", sender, 0, 10**6, BASE * 2 + GWEI, GWEI, TxKind.SWAP,
                      {"route": [pool_id], "token_in": token, "amount_in": amount})
    hops, _ = _quote_hops(state, probe)
    return BuilderTx(tx_hash, sender, state.nonce(sender), 150_000, BASE * 2 + GWEI, GWEI, TxKind.SWAP,
                     {"route": [pool_id], "token_in": token, "amount_in": amount,
                      "amount_out_min": min_amount_out(hops[-1][3].amount_out, tol)})


def sandwich_for(state: BuilderState, v: BuilderTx, searcher: str = AGENT, bribe: int = 0, tag: str = AGENT,
                 size_bps: int = 10_000) -> Bundle:
    o = best_sandwich(state.pools, v, 12, 6)
    assert o is not None
    q = sandwich_economics(state, v, o.hop, o.frontrun_in * size_bps // 10_000, BASE)
    assert q is not None
    return make_sandwich_bundle(state, v, q, BASE, searcher, bribe, GWEI, tag=tag)


def rivals_for(state: BuilderState, v: BuilderTx, bots=None):
    o = best_sandwich(state.pools, v, 12, 6)
    bots = default_competitors() if bots is None else bots
    return [b for b in (bot.build(state, v, o, BASE) for bot in bots) if b is not None]


@pytest.fixture(scope="module")
def pool2():
    with WorkerPool(2, parallel_threshold=2) as wp:
        yield wp


SMALL_ENV = dict(episode_steps=16, grid=6, refine=3)


# ---------------------------------------------------------------------------
# Eviction math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("score,gas,wins_ties,expected", [
    (Fraction(7, 2), 10, False, 36),  # 35 exactly: must strictly exceed
    (Fraction(7, 2), 10, True, 35),  # tie goes to us
    (Fraction(7, 3), 10, False, 24),  # 23.33 -> 24
    (Fraction(7, 3), 10, True, 24),
    (Fraction(0), 5, False, 1),
    (Fraction(10**12), 243_000, False, 243 * 10**15 + 1),
])
def test_outbid_value_exact(score, gas, wins_ties, expected):
    assert outbid_value(score, gas, wins_ties) == expected


def test_outbid_value_rejects_zero_gas():
    with pytest.raises(ConfigError):
        outbid_value(Fraction(1), 0, False)


def test_outbid_value_is_minimal():
    for num in range(1, 40):
        s = Fraction(num, 7)
        v = outbid_value(s, 13, False)
        assert Fraction(v, 13) > s and Fraction(v - 1, 13) <= s


def test_bundles_conflict_on_shared_victim():
    st = market()
    v = victim(st)
    a = sandwich_for(st, v, "s1", tag="a")
    b = sandwich_for(st, v, "s2", tag="b")
    assert bundles_conflict(a, b)


def test_bundles_conflict_on_shared_pool_only():
    st = market()
    a = sandwich_for(st, victim(st, tx_hash="0xa"), "s1", tag="a")
    b = sandwich_for(st, victim(st, tx_hash="0xb", sender="user2"), "s2", tag="b")
    assert set(a.tx_hashes).isdisjoint(b.tx_hashes)
    assert bundles_conflict(a, b)


def test_bundles_do_not_conflict_on_disjoint_pools():
    st = market(extra_pool=True)
    a = sandwich_for(st, victim(st, tx_hash="0xa"), "s1", tag="a")
    b = sandwich_for(st, victim(st, tx_hash="0xb", sender="user2", pool_id="v2:WETH/DAI"), "s2", tag="b")
    assert not bundles_conflict(a, b)


def test_score_bundle_reports_value_and_gas():
    st = market()
    v = victim(st)
    b = sandwich_for(st, v, bribe=10**15)
    s = score_bundle(st, b, BASE)
    assert s.included and s.gas_used > 0
    assert s.searcher_gas_used < s.gas_used  # victim's gas is not the searcher's
    assert s.value_wei == 10**15 + s.searcher_gas_used * GWEI + (s.gas_used - s.searcher_gas_used) * GWEI
    assert s.score == Fraction(s.value_wei, s.gas_used)


def test_score_bundle_failing_bundle():
    st = market()
    v = victim(st)
    b = sandwich_for(st, v)
    broken = dataclasses.replace(b, txs=(b.txs[0], v, dataclasses.replace(
        b.txs[2], payload={**b.txs[2].payload, "amount_out_min": 10**40})))
    s = score_bundle(st, broken, BASE)
    assert not s.included and s.reason.startswith("tx_reverted")


def test_bundle_payer_defaults_to_last_sender():
    st = market()
    b = dataclasses.replace(sandwich_for(st, victim(st)), searcher="")
    assert bundle_payer(b) == AGENT


def test_eviction_minimum_coinbase_wins_and_one_wei_less_loses():
    st = market()
    v = victim(st)
    rivals = rivals_for(st, v)
    mine = sandwich_for(st, v)
    q = quote_eviction(st, mine, rivals, BASE, prefer=FeeStructure.COINBASE)
    assert q.structure is FeeStructure.COINBASE and q.extra_value_wei > 0
    won = apply_eviction(mine, q)
    assert run_competition(st, [won] + rivals, BASE, [v]).won(mine.bundle_id)
    short = dataclasses.replace(won, coinbase_transfer_wei=won.coinbase_transfer_wei - 1)
    assert not run_competition(st, [short] + rivals, BASE, [v]).won(mine.bundle_id)


def test_eviction_priority_tip_structure_wins_and_one_gwei_less_loses():
    st = market()
    v = victim(st)
    rivals = rivals_for(st, v)
    mine = sandwich_for(st, v)
    q = quote_eviction(st, mine, rivals, BASE, prefer=FeeStructure.PRIORITY_TIP)
    assert q.structure is FeeStructure.PRIORITY_TIP
    won = apply_eviction(mine, q)
    assert all(t.max_priority_fee_per_gas == q.new_priority_fee_per_gas for t in won.txs if t.sender == AGENT)
    assert run_competition(st, [won] + rivals, BASE, [v]).won(mine.bundle_id)
    lower = dataclasses.replace(q, new_priority_fee_per_gas=q.new_priority_fee_per_gas - 1)
    assert not run_competition(st, [apply_eviction(mine, lower)] + rivals, BASE, [v]).won(mine.bundle_id)


def test_eviction_picks_cheaper_structure_and_rounding_overhead_bounded():
    st = market()
    v = victim(st)
    q = quote_eviction(st, sandwich_for(st, v), rivals_for(st, v), BASE)
    assert q.coinbase_cost_wei == q.extra_value_wei
    assert 0 <= q.rounding_overhead_wei < q.searcher_gas_used
    assert q.extra_cost_wei == min(q.coinbase_cost_wei, q.tip_cost_wei)


def test_eviction_none_when_already_top():
    st = market()
    v = victim(st)
    rivals = rivals_for(st, v)
    mine = sandwich_for(st, v, bribe=10**21)
    q = quote_eviction(st, mine, rivals, BASE)
    assert q.structure is FeeStructure.NONE and q.extra_cost_wei == 0
    assert apply_eviction(mine, q) is mine


def test_eviction_no_rivals():
    st = market()
    v = victim(st)
    q = quote_eviction(st, sandwich_for(st, v), [], BASE)
    assert q.feasible and q.structure is FeeStructure.NONE and q.live_rivals == ()


def test_eviction_ignores_non_conflicting_rivals():
    st = market(extra_pool=True)
    other = sandwich_for(st, victim(st, tx_hash="0xb", sender="u2", pool_id="v2:WETH/DAI"), "s2", 10**21, "b")
    q = quote_eviction(st, sandwich_for(st, victim(st)), [other], BASE)
    assert q.rivals == () and q.structure is FeeStructure.NONE


def test_eviction_budget_marks_infeasible():
    st = market()
    v = victim(st)
    q = quote_eviction(st, sandwich_for(st, v), rivals_for(st, v), BASE, budget_wei=1)
    assert not q.feasible and q.reason.startswith("over_budget")


def test_eviction_fee_cap_forces_coinbase():
    st = market()
    v = victim(st)
    q = quote_eviction(st, sandwich_for(st, v), rivals_for(st, v), BASE, max_fee_per_gas=BASE + 2 * GWEI,
                       prefer=FeeStructure.PRIORITY_TIP)
    assert q.structure is FeeStructure.COINBASE


def test_eviction_own_failing_bundle():
    st = market()
    v = victim(st)
    b = sandwich_for(st, v)
    bad = dataclasses.replace(b, txs=(dataclasses.replace(b.txs[0], nonce=7),) + b.txs[1:])
    q = quote_eviction(st, bad, rivals_for(st, v), BASE)
    assert not q.own_included and not q.feasible and q.reason.startswith("own_bundle_fails")


def test_eviction_does_not_change_gas_or_burn():
    st = market()
    v = victim(st)
    mine = sandwich_for(st, v)
    rivals = rivals_for(st, v)
    for pref in (FeeStructure.COINBASE, FeeStructure.PRIORITY_TIP):
        q = quote_eviction(st, mine, rivals, BASE, prefer=pref)
        after = score_bundle(st, apply_eviction(mine, q), BASE)
        assert after.gas_used == q.gas_used
        assert q.gas_burn_wei == BASE * q.gas_used
        assert after.value_wei >= q.required_value_wei


@pytest.mark.parametrize("bot_index", range(4))
def test_eviction_against_each_single_rival(bot_index):
    st = market()
    v = victim(st)
    bot = default_competitors()[bot_index]
    rivals = rivals_for(st, v, [bot])
    assert rivals, bot.name
    mine = sandwich_for(st, v)
    q = quote_eviction(st, mine, rivals, BASE)
    out = run_competition(st, [apply_eviction(mine, q)] + rivals, BASE, [v])
    assert out.won(mine.bundle_id)
    assert out.evicted[rivals[0].bundle_id].startswith("conflict_already_included")


@pytest.mark.parametrize("seed", range(10))
def test_multi_bot_displacement_random_victims(seed):
    rng = random.Random(seed)
    st = market()
    token = rng.choice(["WETH", "USDC"])
    reserve = 4_000 * E18 if token == "WETH" else 12_000_000 * E6
    amount = int(reserve * rng.uniform(20, 150) / 10_000)
    v = victim(st, amount, token, rng.choice([100, 300, 500]), tx_hash=f"0x{seed:04x}")
    rivals = rivals_for(st, v)
    o = best_sandwich(st.pools, v, 12, 6)
    if o is None or not rivals:
        pytest.skip("no opportunity for this draw")
    mine = sandwich_for(st, v)
    q = quote_eviction(st, mine, rivals, BASE)
    out = run_competition(st, [apply_eviction(mine, q)] + rivals, BASE, [v], rank=True)
    assert out.winners[v.tx_hash] == mine.bundle_id
    assert len([b for b in out.block.bundles_included()]) == 1
    scores = [s for _, s in out.ranking]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Competitor bots and the arena
# ---------------------------------------------------------------------------


def test_static_tip_bot_pays_no_coinbase():
    st = market()
    v = victim(st)
    bot = CompetitorBot("s", CompetitorStrategy.STATIC_TIP, tip_wei=3 * GWEI)
    b = bot.build(st, v, best_sandwich(st.pools, v, 12, 6), BASE)
    assert b is not None and b.coinbase_transfer_wei == 0
    assert all(t.max_priority_fee_per_gas == 3 * GWEI for t in b.txs if t.sender == bot.searcher)


def test_fixed_share_bot_pays_share_of_net():
    st = market()
    v = victim(st)
    o = best_sandwich(st.pools, v, 12, 6)
    bot = CompetitorBot("f", CompetitorStrategy.FIXED_SHARE, share_bps=4_000)
    b = bot.build(st, v, o, BASE)
    q = sandwich_economics(st, v, o.hop, o.frontrun_in, BASE, bot.tip_wei)
    assert b.coinbase_transfer_wei == q.net_wei * 4_000 // 10_000


def test_router_bot_sizes_from_victim_amount():
    st = market()
    v = victim(st)
    bot = CompetitorBot("r", CompetitorStrategy.STATIC_ROUTER, size_bps=5_000)
    b = bot.build(st, v, None, BASE)
    assert b is not None and b.txs[0].amount_in == v.amount_in // 2


def test_router_bot_overshoot_returns_none():
    st = market()
    v = victim(st, tol=30)
    bot = CompetitorBot("r", CompetitorStrategy.STATIC_ROUTER, size_bps=20_000)
    assert bot.build(st, v, None, BASE) is None  # victim would revert


def test_bots_need_oracle_except_router():
    st = market()
    v = victim(st)
    for s in (CompetitorStrategy.STATIC_TIP, CompetitorStrategy.FIXED_SHARE, CompetitorStrategy.ADAPTIVE):
        assert CompetitorBot("x", s).build(st, v, None, BASE) is None


def test_adaptive_bot_moves_share():
    bot = CompetitorBot("a", CompetitorStrategy.ADAPTIVE, share_bps=5_000, step_bps=400)
    bot.observe(False)
    assert bot.share_bps == 5_400
    bot.observe(True)
    assert bot.share_bps == 5_200
    for _ in range(100):
        bot.observe(False)
    assert bot.share_bps == bot.max_share_bps
    assert bot.losses == 101 and bot.wins == 1


def test_fixed_bot_share_never_moves():
    bot = CompetitorBot("f", CompetitorStrategy.FIXED_SHARE, share_bps=5_000)
    for won in (True, False, False):
        bot.observe(won)
    assert bot.share_bps == 5_000


@pytest.mark.parametrize("kwargs", [dict(share_bps=-1), dict(share_bps=10_001), dict(size_bps=0),
                                    dict(min_share_bps=6_000, max_share_bps=5_000)])
def test_bot_rejects_bad_params(kwargs):
    with pytest.raises(ConfigError):
        CompetitorBot("x", **kwargs)


def test_bot_delay_non_negative():
    rng = random.Random(1)
    bot = CompetitorBot("x", latency_ms=5, jitter_ms=500)
    assert all(bot.sample_delay(rng) >= 0 for _ in range(500))


def test_adaptive_beats_fixed_share_with_same_size():
    st = market()
    v = victim(st)
    o = best_sandwich(st.pools, v, 12, 6)
    a = CompetitorBot("a", CompetitorStrategy.ADAPTIVE, share_bps=6_000).build(st, v, o, BASE)
    f = CompetitorBot("f", CompetitorStrategy.FIXED_SHARE, share_bps=5_000).build(st, v, o, BASE)
    s = CompetitorBot("s", CompetitorStrategy.STATIC_TIP).build(st, v, o, BASE)
    out = run_competition(st, [s, f, a], BASE, [v])
    assert out.winners[v.tx_hash] == a.bundle_id
    assert set(out.evicted) == {s.bundle_id, f.bundle_id}


def test_competition_two_pools_both_land():
    st = market(extra_pool=True)
    a = sandwich_for(st, victim(st, tx_hash="0xa"), "s1", tag="a")
    b = sandwich_for(st, victim(st, tx_hash="0xb", sender="u2", pool_id="v2:WETH/DAI"), "s2", tag="b")
    out = run_competition(st, [a, b], BASE)
    assert out.won(a.bundle_id) and out.won(b.bundle_id) and not out.evicted


def test_competition_same_pool_different_victims_second_reverts():
    st = market()
    a = sandwich_for(st, victim(st, tx_hash="0xa"), "s1", 10**18, "a")
    b = sandwich_for(st, victim(st, tx_hash="0xb", sender="u2"), "s2", 0, "b")
    out = run_competition(st, [a, b], BASE)
    assert out.won(a.bundle_id) and not out.won(b.bundle_id)
    assert out.evicted[b.bundle_id].startswith("tx_reverted")


def test_competition_victim_lands_without_bundles():
    st = market()
    v = victim(st)
    out = run_competition(st, [], BASE, [v])
    assert f"tx:{v.tx_hash}" in out.block.included and out.winners == {}


def test_competition_summary_serialisable():
    st = market()
    v = victim(st)
    out = run_competition(st, rivals_for(st, v), BASE, [v], rank=True)
    s = out.summary()
    assert set(s) == {"winners", "evicted", "ranking", "gas_used", "value_wei"}
    assert all(isinstance(x, float) for _, x in s["ranking"])


def test_competition_uses_custom_builder_policy():
    from simulator.block_builder import GuardAction, GuardPolicy
    st = market()
    v = victim(st)
    strict = BlockBuilder("strict", Strategy.GREEDY, policy=GuardPolicy(sandwich=GuardAction.REJECT))
    out = run_competition(st, rivals_for(st, v), BASE, [v], builder=strict)
    assert out.winners == {} and all(r.startswith("guard_rejected") for r in out.evicted.values())


def test_sandwich_economics_matches_simulation():
    st = market()
    v = victim(st)
    o = best_sandwich(st.pools, v, 12, 6)
    q = sandwich_economics(st, v, 0, o.frontrun_in, BASE)
    fo, _, bo = simulate_route_sandwich(st.pools, v, 0, o.frontrun_in)
    assert (q.frontrun_out, q.backrun_out, q.profit) == (fo, bo, bo - o.frontrun_in)
    assert q.net_wei == q.profit_native_wei - 2 * q.leg_gas * (BASE + GWEI)


@pytest.mark.parametrize("hop,size", [(1, 10**18), (-1, 10**18), (0, 0)])
def test_sandwich_economics_invalid(hop, size):
    st = market()
    assert sandwich_economics(st, victim(st), hop, size, BASE) is None


def test_make_sandwich_bundle_slack():
    st = market()
    v = victim(st)
    o = best_sandwich(st.pools, v, 12, 6)
    q = sandwich_economics(st, v, 0, o.frontrun_in, BASE)
    b = make_sandwich_bundle(st, v, q, BASE, "s", min_out_slack_bps=100)
    assert b.txs[0].amount_out_min == q.frontrun_out * 9_900 // 10_000
    assert b.txs[2].amount_out_min == q.frontrun_in + 1


# ---------------------------------------------------------------------------
# State / action spaces and tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [(-100, 0), (-20, 1), (0, 1), (19, 1), (20, 2), (10**9, 2)])
def test_bucket(value, expected):
    assert bucket(value, (-20, 20)) == expected


def test_state_bins_encode_and_size():
    bins = StateBins()
    key = bins.encode(Observation(0, 0, 100, 50, 4))
    assert len(key) == 5
    assert all(0 <= k < n for k, n in zip(key, bins.shape))
    assert bins.size == math.prod(bins.shape)


def test_state_bins_reject_unsorted():
    with pytest.raises(ConfigError):
        StateBins(reserve_bps=(5, 1))


@pytest.mark.parametrize("size,bid", [(-1, 0), (20_001, 0), (100, -1), (100, 10_001)])
def test_action_validation(size, bid):
    with pytest.raises(ConfigError):
        Action(size, bid)


def test_action_grid():
    g = action_grid((5_000, 10_000), (1_000, 2_000))
    assert g[0].skip and len(g) == 5
    assert len(action_grid((5_000,), (1_000,), include_skip=False)) == 1
    with pytest.raises(ConfigError):
        action_grid((), (), include_skip=False)


def test_action_table_lazy_rows_and_ties():
    t = ActionTable(3, init=0.5)
    assert len(t) == 0 and t.best((1,)) == 0 and (1,) not in t
    t.row((1,))[2] = 0.9
    assert t.best((1,)) == 2 and (1,) in t


def test_action_table_copy_is_independent():
    t = ActionTable(2)
    t.row((0,))[0] = 1.0
    c = t.copy()
    c.row((0,))[0] = 5.0
    assert t.values[(0,)][0] == 1.0


def test_action_table_payload_roundtrip():
    t = ActionTable(3, 0.25)
    t.row((1, 2))[1] = 0.7
    t.counts((1, 2))[1] = 4
    u = ActionTable.from_payload(t.to_payload())
    assert u.values == t.values and u.visits == t.visits and u.init == 0.25


def test_action_table_payload_width_check():
    p = ActionTable(3).to_payload()
    p["rows"] = [([0], [0.0, 0.0], [0, 0])]
    with pytest.raises(ConfigError):
        ActionTable.from_payload(p)


def test_action_table_merge_visit_weighted():
    base = ActionTable(2)
    base.row((0,))[0] = 1.0
    base.counts((0,))[0] = 2
    a, b = base.copy(), base.copy()
    a.values[(0,)][0], a.visits[(0,)][0] = 4.0, 3  # +1 visit
    b.values[(0,)][0], b.visits[(0,)][0] = 7.0, 5  # +3 visits
    m = ActionTable.merge(base, [a, b])
    assert m.values[(0,)][0] == pytest.approx((1.0 * 2 + 4.0 * 1 + 7.0 * 3) / 6)
    assert m.visits[(0,)][0] == 6


def test_action_table_merge_untouched_keeps_base():
    base = ActionTable(2)
    base.row((0,))[1] = 3.0
    m = ActionTable.merge(base, [base.copy(), base.copy()])
    assert m.values == base.values


def test_action_table_merge_new_state_from_shard():
    base = ActionTable(2)
    a = base.copy()
    a.row((9,))[1] = 2.0
    a.counts((9,))[1] = 1
    m = ActionTable.merge(base, [a])
    assert m.values[(9,)] == [0.0, 2.0]


def test_action_table_merge_width_mismatch():
    with pytest.raises(ConfigError):
        ActionTable.merge(ActionTable(2), [ActionTable(3)])


# ---------------------------------------------------------------------------
# Schedules, reward scaling, agents
# ---------------------------------------------------------------------------


def test_schedule_lr_and_epsilon():
    s = LearningSchedule()
    lrs = [s.lr(n) for n in range(0, 10_000, 50)]
    assert lrs == sorted(lrs, reverse=True) and lrs[-1] == s.alpha_min
    assert s.epsilon(0) == s.epsilon0 and s.epsilon(10**6) == s.epsilon_min


@pytest.mark.parametrize("kwargs", [dict(alpha0=0), dict(alpha_min=0.9, alpha0=0.5), dict(gamma=1.0),
                                    dict(epsilon_decay=0), dict(td_clip=-1), dict(alpha_power=0)])
def test_schedule_validation(kwargs):
    with pytest.raises(ConfigError):
        LearningSchedule(**kwargs)


def test_reward_scaler_normalise_and_clip():
    s = RewardScaler(floor_wei=10, clip=3.0)
    s.observe(100)
    assert s.scale == 100
    assert s.normalize(50) == 0.5
    assert s.normalize(10**9) == 3.0 and s.normalize(-10**9) == -3.0


def test_reward_scaler_winsorised_under_outliers():
    s = RewardScaler(beta=0.1, floor_wei=1, clip=5.0)
    s.observe(1_000)
    for _ in range(3):
        s.observe(10**30)
    assert s.scale == pytest.approx(1_000 * (0.9 + 0.1 * 5.0) ** 3)  # each outlier grows scale by at most 1.4x


def test_reward_scaler_floor_and_zero():
    s = RewardScaler(floor_wei=10**12)
    s.observe(0)
    s.observe(5)
    assert s.scale == 10**12 and s.observed == 1


def test_reward_scaler_validation():
    with pytest.raises(ConfigError):
        RewardScaler(beta=0)


def test_q_update_exact():
    agent = QLearningAgent(actions=action_grid()[:3], schedule=LearningSchedule(gamma=0.0, td_clip=0),
                           optimistic_init=0.0, scaler=RewardScaler(floor_wei=100))
    td = agent.learn((0,), 1, 50)
    assert td == 0.5
    assert agent.table.values[(0,)][1] == pytest.approx(0.6 * 0.5)
    assert agent.table.visits[(0,)][1] == 1


def test_q_update_bootstraps_next_state():
    agent = QLearningAgent(actions=action_grid()[:2], schedule=LearningSchedule(gamma=0.5, td_clip=0),
                           optimistic_init=0.0, scaler=RewardScaler(floor_wei=100))
    agent.table.row((1,))[0] = 2.0
    agent.learn((0,), 0, 0, next_obs=(1,))
    assert agent.table.values[(0,)][0] == pytest.approx(0.6 * 0.5 * 2.0)
    agent.learn((2,), 0, 0, next_obs=(1,), done=True)
    assert agent.table.values[(2,)][0] == 0.0  # terminal: no bootstrap


def test_q_td_clip_bounds_step():
    agent = QLearningAgent(actions=action_grid()[:2], schedule=LearningSchedule(gamma=0.0, td_clip=0.25),
                           optimistic_init=0.0, scaler=RewardScaler(floor_wei=1, clip=5.0))
    agent.scaler.observe(1)
    agent.learn((0,), 0, 10**9)
    assert agent.table.values[(0,)][0] == pytest.approx(0.6 * 0.25)


def test_q_optimistic_init_tries_every_action():
    acts = action_grid()[:5]
    agent = QLearningAgent(actions=acts, schedule=LearningSchedule(epsilon0=0.0, epsilon_min=0.0), seed=1)
    chosen = set()
    for _ in range(len(acts)):
        a = agent.act((0,))
        chosen.add(a)
        agent.learn((0,), a, 0, done=True)
    assert chosen == set(range(len(acts)))


def test_q_cumulative_reward_is_exact_integer():
    agent = QLearningAgent(actions=action_grid()[:2])
    rewards = [10**18 + 1, -3, 7, 2**70 + 5]
    for r in rewards:
        agent.learn((0,), 0, r, done=True)
    assert agent.cumulative_reward_wei == sum(rewards)


def test_q_rejects_bad_action():
    with pytest.raises(ConfigError):
        QLearningAgent(actions=action_grid()[:2]).learn((0,), 5, 1)


def test_q_greedy_act_does_not_count_steps():
    agent = QLearningAgent(actions=action_grid()[:3])
    for _ in range(10):
        agent.act((0,), explore=False)
    assert agent.steps == 0


def test_agent_table_width_must_match():
    with pytest.raises(ConfigError):
        QLearningAgent(actions=action_grid()[:3], table=ActionTable(4))


def test_pg_probs_sum_to_one():
    agent = PolicyGradientAgent(actions=action_grid()[:4])
    agent.table.row((0,))[2] = 3.0
    p = agent.probs((0,))
    assert sum(p) == pytest.approx(1.0) and max(p) == p[2]


def test_pg_positive_advantage_raises_probability():
    agent = PolicyGradientAgent(actions=action_grid()[:4], scaler=RewardScaler(floor_wei=10))
    before = agent.probs((0,))[1]
    agent.learn((0,), 1, 10)
    assert agent.probs((0,))[1] > before


def test_pg_negative_advantage_lowers_probability():
    agent = PolicyGradientAgent(actions=action_grid()[:4], scaler=RewardScaler(floor_wei=10))
    before = agent.probs((0,))[3]
    agent.learn((0,), 3, -10)
    assert agent.probs((0,))[3] < before
    assert agent.baselines[(0,)] < 0


def test_pg_validation():
    with pytest.raises(ConfigError):
        PolicyGradientAgent(lr=0)


def test_make_agent_kinds():
    assert isinstance(make_agent("q_learning"), QLearningAgent)
    assert isinstance(make_agent("policy_gradient"), PolicyGradientAgent)
    with pytest.raises(ConfigError):
        make_agent("sarsa")


# ---------------------------------------------------------------------------
# Convergence under noise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_q_learning_converges_moderate_noise(seed):
    env = NoisyBanditEnvironment(noise_wei=2 * 10**15, outlier_prob=0.02, outlier_wei=10**17, seed=seed)
    agent = QLearningAgent(actions=action_grid()[:6], seed=seed)
    assert train_bandit(agent, env, 6_000)["optimal_contexts"] >= 3


@pytest.mark.parametrize("seed", range(6))
def test_policy_gradient_converges_moderate_noise(seed):
    env = NoisyBanditEnvironment(noise_wei=2 * 10**15, outlier_prob=0.02, outlier_wei=10**17, seed=seed)
    agent = PolicyGradientAgent(actions=action_grid()[:6], seed=seed)
    assert train_bandit(agent, env, 6_000)["optimal_contexts"] >= 3


@pytest.mark.parametrize("seed", range(4))
def test_q_values_stay_bounded_under_extreme_noise(seed):
    env = NoisyBanditEnvironment(noise_wei=5 * 10**16, outlier_prob=0.1, outlier_wei=10**21, seed=seed)
    agent = QLearningAgent(actions=action_grid()[:6], seed=seed)
    train_bandit(agent, env, 3_000)
    bound = max(agent.scaler.clip, 1.0) / (1 - agent.schedule.gamma) + 1.0
    vals = [v for row in agent.table.values.values() for v in row]
    assert all(math.isfinite(v) and abs(v) <= bound for v in vals)


def test_extreme_noise_still_beats_uniform_on_average():
    total = 0
    for seed in range(6):
        env = NoisyBanditEnvironment(noise_wei=5 * 10**15, outlier_prob=0.05, outlier_wei=10**18, seed=seed)
        total += train_bandit(QLearningAgent(actions=action_grid()[:6], seed=seed), env, 6_000)["optimal_contexts"]
    assert total / 6 > 4 / 6 * 1.5  # uniform guessing finds 4/6 contexts on average


def test_converged_policy_is_stable():
    env = NoisyBanditEnvironment(noise_wei=10**15, seed=11)
    agent = QLearningAgent(actions=action_grid()[:6], seed=11)
    train_bandit(agent, env, 4_000)
    before = {c: agent.greedy((c,)) for c in range(4)}
    train_bandit(agent, env, 2_000)
    assert {c: agent.greedy((c,)) for c in range(4)} == before


def test_bandit_validation_and_expectations():
    with pytest.raises(ConfigError):
        NoisyBanditEnvironment(n_actions=1)
    env = NoisyBanditEnvironment(gap_wei=5, base_wei=1)
    assert env.expected(0, env.optimal(0)) == 6 and env.expected(0, (env.optimal(0) + 1) % 6) == 1


# ---------------------------------------------------------------------------
# Sandwich environment
# ---------------------------------------------------------------------------


def _rollout(seed, steps=40, action=Action(10_000, 6_500), **kw):
    env = SandwichEnvironment(EnvConfig(seed=seed, **{**SMALL_ENV, **kw}))
    out = []
    for _ in range(steps):
        r = env.step(action)
        out.append((r.reward_wei, r.info.get("agent"), r.info.get("winner")))
    return out


def test_env_deterministic_for_seed():
    assert _rollout(3) == _rollout(3)


def test_env_different_seeds_differ():
    assert _rollout(3) != _rollout(4)


def test_env_observation_fields():
    env = SandwichEnvironment(EnvConfig(seed=1, **SMALL_ENV))
    obs = env.observe()
    assert obs.rivals == 4 and obs.relay_delay_ms >= 0 and obs.victim_size_bps >= 1
    assert set(obs.to_dict()) == {"reserve_delta_bps", "gas_delta_bps", "relay_delay_ms", "victim_size_bps",
                                  "rivals"}


def test_env_skip_earns_zero():
    for reward, status, _ in _rollout(5, 30, Action(0, 0)):
        assert reward == 0 and status == "skipped"


def test_env_reward_zero_unless_won():
    for reward, status, _ in _rollout(6, 60):
        if status != "won":
            assert reward == 0


def test_env_reward_matches_balance_delta_exactly():
    env = SandwichEnvironment(EnvConfig(seed=8, **SMALL_ENV))
    checked = 0
    for _ in range(120):
        if env.victim.token_in != "WETH" or env.oracle is None:
            env.step(Action(0, 0))
            continue
        mine, _ = env.agent_bundle(Action(10_000, 9_700))
        res = env.step(Action(10_000, 9_700))
        if res.info.get("agent") != "won":
            continue
        profit = int(mine.meta["profit"])  # WETH (native) units, exact
        expected = profit - res.info["agent_fees_wei"] - mine.coinbase_transfer_wei
        assert res.reward_wei == expected
        assert res.info["agent_gas_used"] > 0
        checked += 1
    assert checked >= 1


def test_env_won_bundle_is_the_unique_winner():
    for reward, status, winner in _rollout(9, 60):
        if status == "won":
            assert winner.startswith(f"{AGENT}:")


def test_env_overbid_never_negative():
    env = SandwichEnvironment(EnvConfig(seed=2, **SMALL_ENV))
    seen = 0
    for _ in range(80):
        r = env.step(Action(10_000, 9_000))
        if "overbid_wei" in r.info:
            assert r.info["overbid_wei"] >= 0
            assert r.info["min_bribe_wei"] >= 0
            seen += 1
    assert seen >= 1


def test_env_episode_reset():
    env = SandwichEnvironment(EnvConfig(seed=1, episode_steps=5, grid=4, refine=2))
    dones = [env.step(Action(0, 0)).done for _ in range(10)]
    assert dones == [False] * 4 + [True] + [False] * 4 + [True]
    assert env.pool.reserve0 == EnvConfig().reserve_weth  # fresh market after reset


@pytest.mark.parametrize("regime", sorted(REGIMES))
def test_env_regimes_keep_base_fee_in_bounds(regime):
    cfg = EnvConfig(seed=4, regime=regime, **SMALL_ENV)
    env = SandwichEnvironment(cfg)
    for _ in range(40):
        env.step(Action(10_000, 5_000))
        assert cfg.min_base_fee_wei <= env.base_fee <= cfg.max_base_fee_wei


def test_env_noise_is_mean_reverting():
    env = SandwichEnvironment(EnvConfig(seed=2, regime="extreme", episode_steps=10**6, grid=4, refine=2))
    for _ in range(400):
        env._noise_trade()
    price = env.pool.reserve1 / env.pool.reserve0
    assert 0.7 * 3_000e-12 < price < 1.3 * 3_000e-12


def test_env_late_bundles_never_win():
    env = SandwichEnvironment(EnvConfig(seed=3, cutoff_ms=(0, 0), **SMALL_ENV))
    for _ in range(30):
        r = env.step(Action(10_000, 9_700))
        if r.info["relay_delay_ms"] > 0:
            assert r.info.get("agent") != "won" and r.reward_wei == 0
        if r.info.get("agent") == "late":
            assert any(x.startswith(f"{AGENT}:") for x in r.info["late"])


def test_env_custom_competitors():
    cfg = EnvConfig(seed=1, competitors=({"name": "solo", "strategy": "fixed_share", "share_bps": 100},),
                    **SMALL_ENV)
    env = SandwichEnvironment(cfg)
    assert [b.name for b in env.bots] == ["solo"] and env.observe().rivals == 1


@pytest.mark.parametrize("kwargs", [dict(regime="nope"), dict(episode_steps=0), dict(cutoff_ms=(5, 1)),
                                    dict(base_fee_wei=0)])
def test_env_config_validation(kwargs):
    with pytest.raises(ConfigError):
        EnvConfig(**kwargs)


def test_market_regime_validation():
    with pytest.raises(ConfigError):
        MarketRegime("x", -1, 0, 0, 0, 1, 2)
    with pytest.raises(ConfigError):
        MarketRegime("x", 1, 0, 0, 0, 5, 2)


def test_run_agent_report_accounting():
    agent = QLearningAgent(seed=1)
    rep = run_agent(agent, SandwichEnvironment(EnvConfig(seed=1, **SMALL_ENV)), 60)
    assert rep.wins + rep.late + rep.evicted == rep.submitted
    assert rep.submitted + rep.skipped == rep.steps == 60
    assert rep.cumulative_reward_wei == agent.cumulative_reward_wei == sum(rep.rewards_wei)
    assert len(rep.moving_average(8)) == 60
    s = rep.summary()
    assert s["first_half_reward_wei"] + s["second_half_reward_wei"] == s["cumulative_reward_wei"]


def test_run_agent_eval_does_not_learn():
    agent = QLearningAgent(seed=1)
    run_agent(agent, SandwichEnvironment(EnvConfig(seed=1, **SMALL_ENV)), 30, learn=False)
    assert agent.updates == 0 and len(agent.table) == 0


def test_run_agent_rejects_negative_steps():
    with pytest.raises(ConfigError):
        run_agent(QLearningAgent(), SandwichEnvironment(EnvConfig(**SMALL_ENV)), -1)


def test_fixed_policy_baseline():
    pol = FixedPolicy(Action(10_000, 5_000))
    rep = run_agent(pol, SandwichEnvironment(EnvConfig(seed=2, **SMALL_ENV)), 20)
    assert pol.cumulative_reward_wei == rep.cumulative_reward_wei


def test_agent_learns_low_bids_without_rivals():
    # No rivals and no latency risk: any bid wins, so keeping more of the profit is strictly better.
    kw = dict(competitors=(), cutoff_ms=(5_000, 5_000), regime="calm", **SMALL_ENV)
    agent = QLearningAgent(seed=5)
    run_agent(agent, SandwichEnvironment(EnvConfig(seed=5, **kw)), 600)
    eval_cfg = EnvConfig(seed=99, **kw)
    learned = run_agent(agent, SandwichEnvironment(eval_cfg), 120, learn=False).cumulative_reward_wei
    fixed = run_agent(FixedPolicy(Action(10_000, 9_000)), SandwichEnvironment(eval_cfg), 120,
                      learn=False).cumulative_reward_wei
    assert learned > fixed > 0


def test_agent_never_wins_against_full_share_rival():
    cfg = EnvConfig(seed=5, competitors=({"name": "max", "strategy": "fixed_share", "share_bps": 10_000,
                                          "latency_ms": 0, "jitter_ms": 0},), cutoff_ms=(1_000, 1_000),
                    regime="calm", **SMALL_ENV)
    rep = run_agent(QLearningAgent(seed=5), SandwichEnvironment(cfg), 120)
    assert rep.wins == 0 and rep.cumulative_reward_wei == 0
    assert rep.bot_outcomes["max"].get("won", 0) > 0


def test_benchmark_strategies_ranks_all():
    out = benchmark_strategies(EnvConfig(seed=1, **SMALL_ENV), train_steps=60, eval_steps=30)
    assert set(out["ranking"]) == {"q_learning", "fixed_bid_30pct", "fixed_bid_90pct", "half_size_bid_50pct"}
    rewards = [out["eval"][k]["cumulative_reward_wei"] for k in out["ranking"]]
    assert rewards == sorted(rewards, reverse=True)


# ---------------------------------------------------------------------------
# Parallel policy iteration and concurrency
# ---------------------------------------------------------------------------


def test_parallel_matches_inline(pool2):
    cfg = EnvConfig(seed=3, **SMALL_ENV)
    inline = parallel_policy_iteration(cfg, rounds=2, steps_per_shard=12, workers=1, seed=3)
    par = parallel_policy_iteration(cfg, rounds=2, steps_per_shard=12, executor=pool2, seed=3)
    assert par.table.values == inline.table.values and par.table.visits == inline.table.visits
    assert par.round_rewards_wei == inline.round_rewards_wei
    assert par.parallel == pool2.will_parallelize(RL_SHARDS)


def test_parallel_visits_add_up(pool2):
    rep = parallel_policy_iteration(EnvConfig(seed=1, **SMALL_ENV), rounds=3, steps_per_shard=10,
                                    executor=pool2, seed=1)
    assert rep.table.total_visits() == 3 * RL_SHARDS * 10
    assert rep.summary()["total_steps"] == 3 * RL_SHARDS * 10


def test_parallel_policy_gradient(pool2):
    rep = parallel_policy_iteration(EnvConfig(seed=2, **SMALL_ENV), rounds=1, steps_per_shard=10,
                                    kind="policy_gradient", executor=pool2, seed=2)
    assert isinstance(rep.agent(), PolicyGradientAgent) and rep.table.total_visits() == RL_SHARDS * 10


def test_parallel_worker_count_in_cpu_band():
    rep = parallel_policy_iteration(EnvConfig(seed=1, **SMALL_ENV), rounds=1, steps_per_shard=2, seed=1,
                                    workers=10**6)
    from simulator.config import detect_cpu_count
    cpus = detect_cpu_count()
    assert worker_floor(cpus) <= rep.workers <= worker_ceiling(cpus)
    assert rep.workers == resolve_worker_count(10**6)


def test_parallel_continues_from_table(pool2):
    cfg = EnvConfig(seed=4, **SMALL_ENV)
    first = parallel_policy_iteration(cfg, rounds=1, steps_per_shard=8, executor=pool2, seed=4)
    second = parallel_policy_iteration(cfg, rounds=1, steps_per_shard=8, executor=pool2, seed=5,
                                       table=first.table)
    assert second.table.total_visits() == 2 * RL_SHARDS * 8
    assert first.table.total_visits() == RL_SHARDS * 8  # input table not mutated


def test_parallel_validation():
    with pytest.raises(ConfigError):
        parallel_policy_iteration(rounds=0)
    with pytest.raises(ConfigError):
        parallel_policy_iteration(shards=0)


def test_concurrent_threads_are_isolated():
    def job(seed):
        agent = QLearningAgent(seed=seed)
        return run_agent(agent, SandwichEnvironment(EnvConfig(seed=seed, **SMALL_ENV)), 25).rewards_wei
    serial = [job(s) for s in range(4)]
    with ThreadPoolExecutor(max_workers=4) as ex:
        threaded = list(ex.map(job, range(4)))
    assert threaded == serial


def test_concurrent_merge_order_independent(pool2):
    base = ActionTable(3)
    shards = []
    for s in range(4):
        t = base.copy()
        t.row((s % 2,))[s % 3] = float(s)
        t.counts((s % 2,))[s % 3] = s + 1
        shards.append(t)
    a = ActionTable.merge(base, shards)
    b = ActionTable.merge(base, list(reversed(shards)))
    for k in a.values:
        assert a.values[k] == pytest.approx(b.values[k]) and a.visits[k] == b.visits[k]


# ---------------------------------------------------------------------------
# Config and controller wiring
# ---------------------------------------------------------------------------


def test_brutal_preset_enables_rl():
    cfg = SimulationConfig.for_mode("brutal", cpus=4)
    assert cfg.rl_rounds > 0 and cfg.to_dict()["rl_rounds"] == cfg.rl_rounds
    assert SimulationConfig.for_mode("moderate", cpus=4).rl_rounds == 0


@pytest.mark.parametrize("kwargs", [dict(rl_rounds=-1), dict(rl_steps_per_shard=-1), dict(rl_eval_steps=-1),
                                    dict(rl_regime="wild")])
def test_rl_config_validation(kwargs):
    with pytest.raises(ConfigError):
        SimulationConfig.for_mode("brutal", cpus=4, **kwargs)


def test_reinforcement_simulation_summary(pool2):
    out = ReinforcementSimulation(seed=1, rounds=1, steps_per_shard=6, eval_steps=10).run(executor=pool2)
    assert set(out) == {"train", "eval", "policy_states", "regime"}
    assert out["train"]["total_steps"] == RL_SHARDS * 6 and out["eval"]["steps"] == 10


def test_cli_accepts_rl_flags(monkeypatch, capsys):
    import json
    from simulator.__main__ import main
    rc = main(["--mode", "moderate", "--txs", "4", "--fuzz", "4", "--rl-rounds", "0", "--rl-regime", "calm",
               "--workers", "1"])
    report = json.loads(capsys.readouterr().out)
    assert rc == 0 and report["config"]["rl_regime"] == "calm" and "reinforcement" not in report["result"]
