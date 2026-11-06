"""Verification suite for Layer 05: private RPC relay (Flashbots Protect / MEV-Share).

Covers calldata encoding and partial disclosure, the hint leakage matrix,
searcher-side trade inference at every knowledge level, predatory exposure
analysis on the worker pool, signatures and encrypted builder envelopes, the
relay's admission rules (backrun-only privacy, windows, rate limits,
replacement), the order-flow auction with refunds, settlement, multi-block
races, and the end-to-end private order-flow simulation.
"""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction

import pytest

from simulator.__main__ import main as cli_main
from simulator.block_builder import (
    ETH,
    GWEI,
    BlockBuilder,
    Bundle,
    BuilderState,
    BuilderTx,
    MevBoostAuction,
    Strategy,
    TxKind,
    best_sandwich,
    seal_block,
)
from simulator.config import SimulationConfig
from simulator.controller import SimulationController, demo_market
from simulator.multi_hop_router import RoutePath, quote_path_output
from simulator.parallel import WorkerPool
from simulator.private_relay import (
    FULL_DISCLOSURE,
    HINT_PROFILES,
    KIND_BY_SELECTOR,
    MEV_SHARE_DEFAULT,
    ROUTER_ADDRESS,
    SELECTORS,
    V2_SWAP_TOPIC,
    V2_SYNC_TOPIC,
    V3_SWAP_TOPIC,
    V4_SWAP_TOPIC,
    BackrunSearcher,
    BodyItem,
    FrontrunSearcher,
    Hint,
    Knowledge,
    MevShareBundle,
    PrivacyPreferences,
    PrivateFlowReceiver,
    PrivateOrderflowSimulation,
    PrivateRelay,
    RelayError,
    TxStatus,
    VictimHypothesis,
    Word,
    address_of,
    build_hint_view,
    bundle_from_wire,
    bundle_to_wire,
    encode_calldata,
    hint_exposure_matrix,
    hint_names,
    infer_victim,
    measure_exposure,
    open_envelope,
    parse_hints,
    pool_word_range,
    reveal_ranges,
    revealed_selector,
    revealed_words,
    seal_envelope,
    select_hypotheses,
    sign_payload,
    simulate_private_tx,
    tx_to_address,
    verify_signature,
    word_range,
)
from simulator.slippage_engine import min_amount_out

from test_multi_hop import V2_UD, V2_WU, V3_UW, V4_WD, E18, market

BASE_FEE = 20 * GWEI


def fresh_state(skew: float = 1.0) -> BuilderState:
    return BuilderState.from_pools(market(skew))


def swap_tx(state: BuilderState, route=(V2_WU,), token_in="WETH", amount=60 * E18, sender="alice", nonce=0,
            tolerance_bps=100, tx_hash="0xvictim", tip=2 * GWEI, fee_cap=200 * GWEI) -> BuilderTx:
    out = quote_path_output(state.pools, RoutePath.from_route(state.pools, token_in, list(route)), amount)
    return BuilderTx(tx_hash, sender, nonce, 600_000, fee_cap, tip, TxKind.SWAP,
                     {"route": list(route), "token_in": token_in, "amount_in": amount,
                      "amount_out_min": min_amount_out(out, tolerance_bps)})


def relay_with(builders=("b1", "b2"), searchers=("s1", "s2"), **kw) -> PrivateRelay:
    r = PrivateRelay(**kw)
    for b in builders:
        r.register_builder(b)
    for s in searchers:
        r.register_searcher(s)
    return r


def dummy_backrun(searcher: str, n: int = 0, nonce: int = 0) -> BuilderTx:
    return BuilderTx(f"{searcher}:br{n}", searcher, nonce, 400_000, 200 * GWEI, GWEI, TxKind.SWAP,
                     {"route": [V3_UW], "token_in": "WETH", "amount_in": 10**15, "amount_out_min": 1})


def bundle(relay: PrivateRelay, searcher: str, ref, after=(), before=(), block=0, max_block=None,
           payment=10**15, bid="b0", key=None) -> MevShareBundle:
    body = tuple(BodyItem(tx=t) for t in before)
    body += tuple(BodyItem(hash=h) for h in ([ref] if isinstance(ref, str) else list(ref)))
    body += tuple(BodyItem(tx=t) for t in after)
    b = MevShareBundle(bid, searcher, body, block, block + 2 if max_block is None else max_block, payment)
    return b.signed(key or relay.searchers[searcher].key)


# ---------------------------------------------------------------------------
# Encoding and partial disclosure
# ---------------------------------------------------------------------------


def test_addresses_are_deterministic_and_distinct():
    a, b = address_of(V2_WU), address_of(V3_UW)
    assert a == address_of(V2_WU) and a != b and len(a) == 42 and a.startswith("0x")


def test_selectors_are_distinct_and_reversible():
    assert len(set(SELECTORS.values())) == 4
    assert all(KIND_BY_SELECTOR[s] is k for k, s in SELECTORS.items())


def test_swap_calldata_layout_round_trips_through_words():
    st = fresh_state()
    tx = swap_tx(st, route=(V2_WU, V2_UD))
    cd = encode_calldata(tx)
    assert len(cd) == 4 + 32 * (Word.POOLS + 2) and cd[:4] == SELECTORS[TxKind.SWAP]
    w = revealed_words([(0, cd)])
    assert w[Word.AMOUNT_IN] == tx.amount_in and w[Word.AMOUNT_OUT_MIN] == tx.amount_out_min
    assert w[Word.HOP_COUNT] == 2
    assert "0x" + w[Word.TOKEN_IN].to_bytes(20, "big").hex() == address_of("WETH")
    assert "0x" + w[Word.POOLS + 1].to_bytes(20, "big").hex() == address_of(V2_UD)


def test_lp_calldata_encodes_negative_ticks_as_twos_complement():
    tx = BuilderTx("0xm", "lp", 0, 200_000, 10**11, 10**9, TxKind.MINT,
                   {"pool_id": V3_UW, "tick_lower": -600, "tick_upper": 600, "liquidity": 5})
    w = revealed_words([(0, encode_calldata(tx))])
    assert w[1] == (1 << 256) - 600 and w[2] == 600 and w[3] == 5
    assert encode_calldata(tx)[:4] == SELECTORS[TxKind.MINT]


def test_reveal_ranges_clip_and_merge():
    data = bytes(range(100))
    frags = reveal_ranges(data, [(10, 20), (15, 30), (90, 200), (50, 50), (-5, 2)])
    assert frags == ((0, data[0:2]), (10, data[10:30]), (90, data[90:100]))


def test_only_fully_revealed_words_decode():
    st = fresh_state()
    cd = encode_calldata(swap_tx(st))
    s, e = word_range(Word.AMOUNT_IN)
    assert set(revealed_words(reveal_ranges(cd, [(s, e)]))) == {Word.AMOUNT_IN}
    assert revealed_words(reveal_ranges(cd, [(s, e - 1)])) == {}
    assert revealed_selector(reveal_ranges(cd, [(0, 4)])) == cd[:4]
    assert revealed_selector(reveal_ranges(cd, [(1, 4)])) is None


def test_tx_to_address_reveals_pool_only_for_single_hop():
    st = fresh_state()
    assert tx_to_address(swap_tx(st)) == address_of(V2_WU)
    assert tx_to_address(swap_tx(st, route=(V2_WU, V2_UD))) == ROUTER_ADDRESS


# ---------------------------------------------------------------------------
# Preferences and hints
# ---------------------------------------------------------------------------


def test_parse_hints_and_names():
    h = parse_hints(["calldata", "LOGS", "tx_hash"])
    assert h == Hint.HASH | Hint.CALLDATA | Hint.LOGS
    assert set(hint_names(h)) == {"hash", "calldata", "logs"}
    assert parse_hints([]) == Hint.HASH
    with pytest.raises(RelayError):
        parse_hints(["mempool_everything"])


@pytest.mark.parametrize("kwargs", [
    {"refund_bps": 10_001},
    {"refund_bps": -1},
    {"calldata_offsets": ((4, 36),)},  # offsets without the calldata_offsets hint
    {"hints": Hint.HASH | Hint.CALLDATA_OFFSETS, "calldata_offsets": ((40, 36),)},
])
def test_privacy_preferences_validation(kwargs):
    with pytest.raises(RelayError):
        PrivacyPreferences(**kwargs)


def test_protect_mode_hides_everything_but_hash():
    assert HINT_PROFILES["mev_share_default"].hints == MEV_SHARE_DEFAULT
    assert Hint.CALLDATA not in MEV_SHARE_DEFAULT and Hint.LOGS not in MEV_SHARE_DEFAULT
    p = HINT_PROFILES["protect"]
    assert p.effective_hints == Hint.HASH and not p.share_with_searchers
    v = build_hint_view(swap_tx(fresh_state()), PrivacyPreferences(hints=FULL_DISCLOSURE, share_with_searchers=False),
                        fresh_state(), 0)
    assert v.to is None and v.selector is None and not v.calldata and not v.logs and v.balance_deltas is None


def test_contract_selector_view_has_no_amounts():
    st = fresh_state()
    v = build_hint_view(swap_tx(st), HINT_PROFILES["contract_selector"], st, 3)
    assert v.to == address_of(V2_WU) and v.selector == SELECTORS[TxKind.SWAP]
    assert not v.calldata and not v.logs and v.balance_deltas is None
    assert v.block == 3 and v.max_block == 3 + 25


def test_default_logs_expose_topics_only():
    st = fresh_state()
    v = build_hint_view(swap_tx(st, route=(V2_WU, V3_UW, V4_WD)), HINT_PROFILES["default_logs"], st, 0)
    assert [lg.topics for lg in v.logs] == [(V2_SWAP_TOPIC,), (V3_SWAP_TOPIC,), (V4_SWAP_TOPIC,)]
    assert all(lg.data == () for lg in v.logs)


def test_full_logs_match_execution_for_every_protocol():
    st = fresh_state()
    tx = swap_tx(st, route=(V2_WU, V3_UW, V4_WD), amount=5 * E18)
    sim = simulate_private_tx(st, tx)
    assert sim.ok
    sync, v2swap, v3swap, v4swap = sim.logs
    assert sync.topics == (V2_SYNC_TOPIC,) and v2swap.topics[0] == V2_SWAP_TOPIC
    assert v2swap.field("amount0In") == 5 * E18 and v2swap.field("amount1Out") > 0
    usdc = v2swap.field("amount1Out")
    assert v3swap.field("amount0") == usdc and v3swap.field("amount1") < 0  # USDC in, WETH out
    weth = -v3swap.field("amount1")
    assert v4swap.field("amount0") == weth and v4swap.field("fee") == 500
    dai = -v4swap.field("amount1")
    assert dict(sim.balance_deltas) == {"WETH": -5 * E18, "DAI": dai}  # intermediate USDC nets to zero
    post = st.copy()
    from simulator.block_builder import execute_tx
    execute_tx(post, tx, 0, 30_000_000)
    assert v3swap.field("sqrtPriceX96") == post.pools[V3_UW].sqrt_price_x96
    assert sync.field("reserve0") == post.pools[V2_WU].reserve0


def test_balance_deltas_are_exact_trade_amounts():
    st = fresh_state()
    tx = swap_tx(st, amount=7 * E18)
    v = build_hint_view(tx, HINT_PROFILES["balance_deltas"], st, 0)
    out = quote_path_output(st.pools, RoutePath.from_route(st.pools, "WETH", [V2_WU]), 7 * E18)
    assert dict(v.balance_deltas) == {"WETH": -7 * E18, "USDC": out}


def test_failed_simulation_leaks_no_logs_but_queued_nonce_still_simulates():
    st = fresh_state()
    bad = replace(swap_tx(st), payload={**swap_tx(st).payload, "amount_out_min": 10**40})
    sim = simulate_private_tx(st, bad)
    assert not sim.ok and sim.logs == () and "below min" in sim.error
    v = build_hint_view(bad, HINT_PROFILES["full"], st, 0)
    assert v.logs == () and v.balance_deltas is None and v.calldata  # calldata still shared
    queued = swap_tx(st, nonce=3)
    assert simulate_private_tx(st, queued).ok


def test_leakage_score_orders_profiles():
    st = fresh_state()
    tx = swap_tx(st)
    score = {n: build_hint_view(tx, p, st, 0).leakage_score for n, p in HINT_PROFILES.items()}
    assert score["protect"] == score["hash_only"] == 0
    assert 0 < score["contract_selector"] < score["mev_share_default"] < score["full"]
    assert score["amount_offset"] == 32 and score["full_calldata"] == len(encode_calldata(tx))
    assert json.dumps(build_hint_view(tx, HINT_PROFILES["full"], st, 0).to_dict())


# ---------------------------------------------------------------------------
# Searcher inference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile,route,level", [
    ("protect", (V2_WU,), Knowledge.BLIND),
    ("hash_only", (V2_WU,), Knowledge.BLIND),
    ("amount_offset", (V2_WU,), Knowledge.BLIND),
    ("contract_selector", (V2_WU,), Knowledge.POOL),
    ("contract_selector", (V2_WU, V2_UD), Knowledge.KIND),
    ("mev_share_default", (V2_WU,), Knowledge.POOL),
    ("default_logs", (V2_WU,), Knowledge.POOL),
    ("default_logs", (V2_WU, V2_UD), Knowledge.DIRECTION),  # only one direction chains through both pools
    ("balance_deltas", (V2_WU,), Knowledge.EXACT),
    ("balance_deltas", (V2_WU, V2_UD), Knowledge.EXACT),
    ("amount_route_offsets", (V2_WU, V2_UD), Knowledge.EXACT),
    ("full_logs", (V3_UW, V4_WD), Knowledge.EXACT),
    ("full_calldata", (V4_WD,), Knowledge.EXACT),
    ("full", (V2_WU, V2_UD), Knowledge.EXACT),
])
def test_inference_level_per_hint_profile(profile, route, level):
    st = fresh_state()
    token_in = "USDC" if route[0] == V3_UW else "WETH"
    amount = 30_000 * 10**6 if token_in == "USDC" else 8 * E18
    tx = swap_tx(st, route=route, token_in=token_in, amount=amount)
    inf = infer_victim(build_hint_view(tx, HINT_PROFILES[profile], st, 0), st.pools)
    assert inf.knowledge is level
    if level is Knowledge.EXACT:
        (h,) = inf.hypotheses
        assert h.route == route and h.token_in == token_in and h.amount_in == amount


def test_only_calldata_reveals_the_slippage_bound():
    st = fresh_state()
    tx = swap_tx(st, tolerance_bps=300)
    for profile, expect in (("full_calldata", tx.amount_out_min), ("full_logs", 0), ("balance_deltas", 0)):
        (h,) = infer_victim(build_hint_view(tx, HINT_PROFILES[profile], st, 0), st.pools).hypotheses
        assert h.amount_out_min == expect


def test_pool_level_inference_spans_both_directions_and_a_size_grid():
    st = fresh_state()
    inf = infer_victim(build_hint_view(swap_tx(st), HINT_PROFILES["contract_selector"], st, 0), st.pools,
                       size_grid=5)
    assert {h.token_in for h in inf.hypotheses} == {"WETH", "USDC"} and len(inf.hypotheses) == 10
    assert sum(h.weight for h in inf.hypotheses) == 1
    sizes = sorted(h.amount_in for h in inf.hypotheses if h.token_in == "WETH")
    assert sizes[0] < sizes[-1]


def test_direction_level_with_route_and_token_offsets():
    st = fresh_state()
    prefs = PrivacyPreferences(hints=Hint.HASH | Hint.CALLDATA_OFFSETS,
                               calldata_offsets=(word_range(Word.TOKEN_IN), word_range(Word.HOP_COUNT),
                                                 pool_word_range(0)))
    inf = infer_victim(build_hint_view(swap_tx(st), prefs, st, 0), st.pools)
    assert inf.knowledge is Knowledge.DIRECTION and inf.token_in == "WETH"
    assert all(h.token_in == "WETH" and h.route == (V2_WU,) for h in inf.hypotheses)


def test_ambiguous_deltas_keep_several_routes():
    st = fresh_state()
    tx = swap_tx(st, route=(V2_WU,), amount=E18)
    view = build_hint_view(tx, HINT_PROFILES["balance_deltas"], st, 0)
    # A searcher whose pool snapshot is stale cannot match the delta exactly.
    stale = fresh_state().pools
    stale[V2_WU].reserve1 += 10**9
    inf = infer_victim(view, stale)
    assert inf.knowledge is Knowledge.AMOUNT_NO_ROUTE and len(inf.hypotheses) >= 2
    assert all(h.amount_in == E18 for h in inf.hypotheses)


def test_transfer_selector_gives_kind_only():
    st = fresh_state()
    tx = BuilderTx("0xt", "bob", 0, 21_000, 10**11, 10**9, TxKind.TRANSFER, {"to": "carol", "value": 10**18})
    inf = infer_victim(build_hint_view(tx, HINT_PROFILES["full"], st, 0), st.pools)
    assert inf.knowledge is Knowledge.KIND and inf.kind is TxKind.TRANSFER and not inf.hypotheses


def test_select_hypotheses_spreads_equal_weights_and_prefers_heavy_ones():
    hs = [VictimHypothesis((V2_WU,), "WETH", a, 0, Fraction(1, 10), "g") for a in range(1, 11)]
    picked = select_hypotheses(hs, 4)
    assert [h.amount_in for h in picked] == [1, 4, 7, 10]
    heavy = VictimHypothesis((V2_WU,), "WETH", 99, 0, Fraction(1, 2), "x")
    assert select_hypotheses(hs + [heavy], 1) == [heavy]
    assert select_hypotheses(hs, 0) == [] and len(select_hypotheses(hs[:2], 5)) == 2


# ---------------------------------------------------------------------------
# Predatory exposure
# ---------------------------------------------------------------------------


def test_full_calldata_leak_lets_a_rogue_builder_take_the_perfect_sandwich():
    st = fresh_state()
    tx = swap_tx(st, amount=40 * E18, tolerance_bps=500)
    r = measure_exposure(st, tx, HINT_PROFILES["full_calldata"], "full_calldata")
    assert r.perfect_profit_native > 0 and r.knowledge is Knowledge.EXACT
    assert r.realized_profit_native == r.perfect_profit_native and r.leak_efficiency == 1
    assert not r.victim_reverted


@pytest.mark.parametrize("profile", ["protect", "hash_only", "amount_offset"])
def test_blind_profiles_leak_nothing(profile):
    st = fresh_state()
    r = measure_exposure(st, swap_tx(st, amount=40 * E18, tolerance_bps=500), HINT_PROFILES[profile], profile)
    assert r.knowledge is Knowledge.BLIND and not r.attempted
    assert r.realized_profit_native == 0 and r.leak_efficiency == 0


def test_missing_slippage_bound_makes_the_leaked_sandwich_overshoot():
    st = fresh_state()
    tx = swap_tx(st, amount=40 * E18, tolerance_bps=30)  # tight bound
    logs = measure_exposure(st, tx, HINT_PROFILES["full_logs"], "full_logs")
    cd = measure_exposure(st, tx, HINT_PROFILES["full_calldata"], "full_calldata")
    assert cd.leak_efficiency == 1
    assert logs.victim_reverted and logs.leak_efficiency < 1


def test_exposure_matrix_thresholds_and_parallel_equivalence():
    st = fresh_state()
    txs = [swap_tx(st, amount=a * E18, tolerance_bps=500, tx_hash=f"0x{a}") for a in (20, 45)]
    serial = hint_exposure_matrix(st, txs)
    with WorkerPool(2, parallel_threshold=1) as wp:
        par = hint_exposure_matrix(st, txs, executor=wp)
    assert [r.to_dict() for r in serial.results] == [r.to_dict() for r in par.results]
    rows = serial.rows()
    assert set(rows) == set(HINT_PROFILES) and rows["protect"]["mean_leak_efficiency"] == 0
    above = serial.profiles_above(0.99)
    assert "full_calldata" in above and "protect" not in above and "hash_only" not in above
    assert serial.minimal_leak_profile(0.99) == above[0]
    assert serial.minimal_leak_profile(1.5) is None
    assert json.dumps(serial.summary())


# ---------------------------------------------------------------------------
# Signatures, envelopes and wire format
# ---------------------------------------------------------------------------


def test_signatures_detect_tampering():
    key = b"k" * 32
    sig = sign_payload(key, b"bundle")
    assert verify_signature(key, b"bundle", sig)
    assert not verify_signature(key, b"bundle!", sig) and not verify_signature(b"x" * 32, b"bundle", sig)


def test_envelope_round_trip_and_failures():
    key, other = b"a" * 32, b"b" * 32
    env = seal_envelope(key, "b1", b"secret bundle", b"n" * 16)
    assert env.ciphertext != b"secret bundle"
    assert open_envelope(key, env, "b1") == b"secret bundle"
    with pytest.raises(RelayError, match="decryption_failed"):
        open_envelope(other, env)
    with pytest.raises(RelayError, match="another builder"):
        open_envelope(key, env, "b2")
    flipped = replace(env, ciphertext=bytes([env.ciphertext[0] ^ 1]) + env.ciphertext[1:])
    with pytest.raises(RelayError):
        open_envelope(key, flipped)
    renamed = replace(env, builder="b2")
    with pytest.raises(RelayError):
        open_envelope(key, renamed)
    assert seal_envelope(key, "b1", b"secret bundle", b"m" * 16).ciphertext != env.ciphertext


def test_bundle_wire_round_trip():
    st = fresh_state()
    b = Bundle("x", (swap_tx(st), dummy_backrun("s1")), 123, "s1", frozenset({"s1:br0"}), "0xvictim",
               {"private": True, "refund_wei": 5})
    back = bundle_from_wire(bundle_to_wire(b))
    assert back.bundle_id == "x" and back.coinbase_transfer_wei == 123 and back.reverting_tx_hashes == {"s1:br0"}
    assert [t.to_dict() for t in back.txs] == [t.to_dict() for t in b.txs] and dict(back.meta) == dict(b.meta)


def test_body_item_needs_exactly_one_field():
    with pytest.raises(RelayError):
        BodyItem()
    with pytest.raises(RelayError):
        BodyItem(hash="0x", tx=dummy_backrun("s"))


# ---------------------------------------------------------------------------
# Private transaction admission
# ---------------------------------------------------------------------------


def test_private_tx_publishes_hint_but_protect_does_not():
    st = fresh_state()
    r = relay_with()
    r.send_private_transaction(swap_tx(st), st, 0, HINT_PROFILES["mev_share_default"])
    r.send_private_transaction(swap_tx(st, sender="bob", tx_hash="0xbob"), st, 0, HINT_PROFILES["protect"])
    assert [v.tx_hash for v in r.stream()] == ["0xvictim"] and r.stream(1) == []
    assert r.is_private("0xbob") and r.status("0xbob") is TxStatus.PENDING
    assert [rec.tx.tx_hash for rec in r.pending(0)] == ["0xvictim", "0xbob"]


@pytest.mark.parametrize("mutate,prefs,reason", [
    ("dup", None, "already_known"),
    ("tip", None, "tip_above_fee_cap"),
    ("nonce", None, "nonce_too_low"),
    (None, PrivacyPreferences(builders=("ghost",)), "unknown_builders"),
    (None, PrivacyPreferences(max_block_number=4), "max_block_in_past"),
])
def test_private_tx_rejections(mutate, prefs, reason):
    st = fresh_state()
    st.nonces["alice"] = 1 if mutate == "nonce" else 0
    r = relay_with()
    tx = swap_tx(st, nonce=0)
    if mutate == "dup":
        r.send_private_transaction(tx, st, 5)
    if mutate == "tip":
        tx = swap_tx(st, tip=10 * GWEI, fee_cap=5 * GWEI)
    with pytest.raises(RelayError, match=reason):
        r.send_private_transaction(tx, st, 5, prefs)


def test_replacement_needs_a_fee_bump():
    st = fresh_state()
    r = relay_with()
    r.send_private_transaction(swap_tx(st), st, 0)
    with pytest.raises(RelayError, match="replacement_underpriced"):
        r.send_private_transaction(swap_tx(st, tx_hash="0xv2", tip=2 * GWEI + 1), st, 0)
    r.send_private_transaction(swap_tx(st, tx_hash="0xv3", tip=3 * GWEI, fee_cap=300 * GWEI), st, 0)
    assert r.status("0xvictim") is TxStatus.CANCELLED and r.records["0xvictim"].reason == "replaced_by:0xv3"
    assert r.status("0xv3") is TxStatus.PENDING


def test_cancel_and_bundles_against_cancelled_tx():
    st = fresh_state()
    r = relay_with()
    r.send_private_transaction(swap_tx(st), st, 0)
    assert r.cancel_private_transaction("0xvictim") and not r.cancel_private_transaction("0xvictim")
    assert not r.cancel_private_transaction("0xnothing")
    rc = r.send_bundle(bundle(r, "s1", "0xvictim", [dummy_backrun("s1")]), 0)
    assert not rc.accepted and rc.reason == "target_cancelled"


# ---------------------------------------------------------------------------
# Bundle admission rules
# ---------------------------------------------------------------------------


def _admission_case(case):
    st = fresh_state()
    r = relay_with(max_bundles_per_searcher_per_block=2, max_future_blocks=5)
    r.send_private_transaction(swap_tx(st), st, 0, PrivacyPreferences(max_block_number=10))
    r.send_private_transaction(swap_tx(st, sender="bob", tx_hash="0xbob"), st, 0, HINT_PROFILES["protect"])
    r.send_private_transaction(swap_tx(st, sender="carl", tx_hash="0xcarl"), st, 0)
    br = [dummy_backrun("s1")]
    if case == "unknown_searcher":
        return r, MevShareBundle("x", "nobody", (BodyItem(hash="0xvictim"),), 0, 1, 1).signed(b"z"), 0
    if case == "bad_signature":
        return r, bundle(r, "s1", "0xvictim", br, key=r.searchers["s2"].key), 0
    if case == "duplicate_bundle":
        assert r.send_bundle(bundle(r, "s1", "0xvictim", br), 0).accepted
        return r, bundle(r, "s1", "0xvictim", br), 0
    if case == "empty_body":
        return r, MevShareBundle("e", "s1", (), 0, 1, 1).signed(r.searchers["s1"].key), 0
    if case == "bad_block_range":
        return r, bundle(r, "s1", "0xvictim", br, block=3, max_block=2), 0
    if case == "block_range_expired":
        return r, bundle(r, "s1", "0xvictim", br, block=0, max_block=1), 2
    if case == "too_far_in_future":
        return r, bundle(r, "s1", "0xvictim", br, block=9, max_block=9), 0
    if case == "negative_payment":
        return r, bundle(r, "s1", "0xvictim", br, payment=-1), 0
    if case == "need_exactly_one_hash_ref:none":
        return r, MevShareBundle("n", "s1", (BodyItem(tx=br[0]),), 0, 1, 1).signed(r.searchers["s1"].key), 0
    if case == "need_exactly_one_hash_ref:two":
        return r, bundle(r, "s1", ["0xvictim", "0xcarl"], br), 0
    if case == "unknown_hash:missing":
        return r, bundle(r, "s1", "0xnope", br), 0
    if case == "unknown_hash:protect":
        return r, bundle(r, "s1", "0xbob", br), 0
    if case == "privacy_violation":
        return r, bundle(r, "s1", "0xvictim", br, before=[dummy_backrun("s1", 9)]), 0
    if case == "no_backrun":
        return r, bundle(r, "s1", "0xvictim", []), 0
    if case == "foreign_sender":
        return r, bundle(r, "s1", "0xvictim", [dummy_backrun("s2")]), 0
    if case == "private_tx_in_body":
        return r, bundle(r, "s1", "0xvictim", [replace(dummy_backrun("s1"), tx_hash="0xcarl")]), 0
    if case == "beyond_tx_window":
        return r, bundle(r, "s1", "0xvictim", br, block=11, max_block=12), 0
    if case == "rate_limited":
        for i in range(2):
            assert r.send_bundle(bundle(r, "s1", "0xvictim", br, bid=f"r{i}"), 0).accepted
        return r, bundle(r, "s1", "0xvictim", br, bid="r9"), 0
    raise AssertionError(case)


@pytest.mark.parametrize("case", [
    "unknown_searcher", "bad_signature", "duplicate_bundle", "empty_body", "bad_block_range",
    "block_range_expired", "too_far_in_future", "negative_payment", "need_exactly_one_hash_ref:none",
    "need_exactly_one_hash_ref:two", "unknown_hash:missing", "unknown_hash:protect", "privacy_violation",
    "no_backrun", "foreign_sender", "private_tx_in_body", "beyond_tx_window", "rate_limited",
])
def test_bundle_admission_rejections(case):
    r, b, block = _admission_case(case)
    if case == "beyond_tx_window":
        r.max_future_blocks = 25
    rc = r.send_bundle(b, block)
    assert not rc.accepted
    assert rc.reason.startswith(case.split(":")[0])
    assert b.bundle_id not in r.bundles or case == "duplicate_bundle"


def test_rate_limit_resets_per_block_and_sub_accounts_are_allowed():
    st = fresh_state()
    r = relay_with(max_bundles_per_searcher_per_block=1)
    r.send_private_transaction(swap_tx(st), st, 0)
    assert r.send_bundle(bundle(r, "s1", "0xvictim", [dummy_backrun("s1")], bid="a"), 0).accepted
    assert not r.send_bundle(bundle(r, "s1", "0xvictim", [dummy_backrun("s1")], bid="b"), 0).accepted
    sub = replace(dummy_backrun("s1"), sender="s1#7")
    assert r.send_bundle(bundle(r, "s1", "0xvictim", [sub], bid="c", block=1), 1).accepted
    assert not r.send_bundle(bundle(r, "s1", "0xvictim", [replace(sub, sender="s1x")], bid="d", block=2), 2).accepted


def test_frontrun_searcher_is_always_blocked():
    st = fresh_state()
    r = relay_with(searchers=("pred",))
    r.send_private_transaction(swap_tx(st, amount=40 * E18, tolerance_bps=500), st, 0, HINT_PROFILES["full"])
    fr = FrontrunSearcher("pred", r.searchers["pred"].key)
    bundles = fr.on_event(r.stream()[0], st, 0, BASE_FEE)
    assert bundles
    for b in bundles:
        rc = r.send_bundle(b, 0)
        assert rc.reason == "privacy_violation:frontrun"
    assert r.stats["bundle_rejected:privacy_violation"] == len(bundles)


# ---------------------------------------------------------------------------
# Order-flow auction, delivery and settlement
# ---------------------------------------------------------------------------


def _backrun_bundles(r, st, names=("s1", "s2"), bids=(9_000, 6_000), profile="full", block=0):
    out = []
    view = r.stream()[-1]
    for name, bid in zip(names, bids):
        s = BackrunSearcher(name, r.searchers[name].key, bid_bps=bid)
        for b in s.on_event(view, st, block, BASE_FEE):
            out.append((b, r.send_bundle(b, block)))
    return out


def test_backrun_auction_picks_highest_payment_and_splits_refund():
    st = fresh_state()
    r = relay_with()
    r.send_private_transaction(swap_tx(st), st, 0, replace(HINT_PROFILES["full"], refund_bps=8_000))
    subs = _backrun_bundles(r, st)
    assert subs and all(rc.accepted for _, rc in subs)
    auc = r.build_auction(st, 0, BASE_FEE)
    e = auc.entries["0xvictim"]
    best = max(subs, key=lambda t: t[0].coinbase_transfer_wei)[0]
    assert e.winner == best.bundle_id and best.searcher == "s1"
    assert e.refund_wei == best.coinbase_transfer_wei * 8_000 // 10_000
    win = next(b for b in auc.bundles if b.bundle_id == f"mevshare:{best.bundle_id}")
    assert win.coinbase_transfer_wei == best.coinbase_transfer_wei - e.refund_wei
    assert win.txs[0].tx_hash == "0xvictim" and win.meta["private"]
    assert any(b.bundle_id == "protect:0xvictim" for b in auc.bundles)
    assert json.dumps(auc.summary())


def test_auction_ties_go_to_the_earlier_bundle():
    st = fresh_state()
    r = relay_with()
    r.send_private_transaction(swap_tx(st), st, 0, HINT_PROFILES["full"])
    (first, _), = [(b, rc) for b, rc in _backrun_bundles(r, st, ("s1",), (9_000,))][:1]
    clone = MevShareBundle("tie", "s2", tuple(BodyItem(hash=i.hash) if i.hash else
                                              BodyItem(tx=replace(i.tx, sender="s2", tx_hash="s2:tie"))
                                              for i in first.body),
                           first.block, first.max_block, first.coinbase_transfer_wei).signed(r.searchers["s2"].key)
    assert r.send_bundle(clone, 0).accepted
    assert r.build_auction(st, 0, BASE_FEE).entries["0xvictim"].winner == first.bundle_id


def test_reverting_backrun_loses_and_protect_fallback_remains():
    st = fresh_state()
    r = relay_with()
    r.send_private_transaction(swap_tx(st), st, 0, HINT_PROFILES["full"])
    greedy = replace(dummy_backrun("s1"), payload={**dummy_backrun("s1").payload, "amount_out_min": 10**40})
    assert r.send_bundle(bundle(r, "s1", "0xvictim", [greedy], payment=10**20), 0).accepted
    auc = r.build_auction(st, 0, BASE_FEE)
    e = auc.entries["0xvictim"]
    assert e.winner is None and not e.candidates[0]["valid"] and "reverted" in e.candidates[0]["reason"]
    assert [b.bundle_id for b in auc.bundles] == ["protect:0xvictim"]
    assert r.searchers["s1"].failed_simulations == 1


def test_can_revert_backrun_still_wins_when_allowed_to_revert():
    st = fresh_state()
    r = relay_with()
    r.send_private_transaction(swap_tx(st), st, 0, HINT_PROFILES["full"])
    bad = replace(dummy_backrun("s1"), payload={**dummy_backrun("s1").payload, "amount_out_min": 10**40})
    body = (BodyItem(hash="0xvictim"), BodyItem(tx=bad, can_revert=True))
    b = MevShareBundle("cr", "s1", body, 0, 1, 5 * 10**15).signed(r.searchers["s1"].key)
    assert r.send_bundle(b, 0).accepted
    auc = r.build_auction(st, 0, BASE_FEE)
    assert auc.entries["0xvictim"].winner == "cr"
    assert next(x for x in auc.bundles if x.bundle_id == "mevshare:cr").reverting_tx_hashes == {"s1:br0"}


def test_reputation_drops_after_repeated_failed_simulations():
    st = fresh_state()
    r = relay_with(max_bundles_per_searcher_per_block=100)
    r.send_private_transaction(swap_tx(st), st, 0, HINT_PROFILES["full"])
    bad = replace(dummy_backrun("s1"), payload={**dummy_backrun("s1").payload, "amount_out_min": 10**40})
    for i in range(9):
        r.send_bundle(bundle(r, "s1", "0xvictim", [replace(bad, tx_hash=f"s1:{i}")], bid=f"f{i}"), 0)
    assert r.searchers["s1"].high_priority
    r.build_auction(st, 0, BASE_FEE)
    assert not r.searchers["s1"].high_priority and r.summary()["searchers"]["s1"]["failed_simulations"] == 9


def test_delivery_respects_builder_preferences_and_fast_mode():
    st = fresh_state()
    r = relay_with(builders=("b1", "b2", "b3"))
    r.send_private_transaction(swap_tx(st), st, 0, PrivacyPreferences(builders=("b2",)))
    r.send_private_transaction(swap_tx(st, sender="z", tx_hash="0xz"), st, 0,
                               PrivacyPreferences(builders=("b2",), fast=True))
    env = r.deliver(r.build_auction(st, 0, BASE_FEE))
    assert len(env["b1"]) == len(env["b3"]) == 1 and len(env["b2"]) == 2
    rcv1 = PrivateFlowReceiver("b1", r.builders["b1"])
    rcv2 = PrivateFlowReceiver("b2", r.builders["b2"])
    assert [b.bundle_id for b in rcv2.open(env["b2"])] == ["protect:0xvictim", "protect:0xz"]
    assert rcv1.open(env["b2"]) == [] and len(rcv1.failures) == 2


def _run_slot(r, st, block=0, builders=None):
    builders = builders or [BlockBuilder("b1", Strategy.GREEDY, margin_bps=500)]
    auc = r.build_auction(st, block, BASE_FEE)
    env = r.deliver(auc)
    extra = {b.name: PrivateFlowReceiver(b.name, r.builders[b.name]).open(env.get(b.name, ()))
             for b in builders if b.name in r.builders}
    res = MevBoostAuction(builders).run_slot(st, [], [], BASE_FEE, block, extra_bundles=extra)
    eth_before = sum(t.get(ETH, 0) for t in res.post_state.fork.balances.values())
    settle = r.on_block(block, res.block, res.post_state)
    eth_after = sum(t.get(ETH, 0) for t in res.post_state.fork.balances.values())
    assert eth_before == eth_after
    return auc, res, settle


def test_end_to_end_private_backrun_lands_and_refunds_user():
    st = fresh_state()
    r = relay_with(builders=("b1",))
    r.send_private_transaction(swap_tx(st), st, 0, HINT_PROFILES["full"])
    _backrun_bundles(r, st)
    auc, res, settle = _run_slot(r, st)
    winner = auc.entries["0xvictim"].winner
    assert f"bundle:mevshare:{winner}" in res.block.included
    assert settle.included == ["0xvictim"] and settle.refunds["0xvictim"] == auc.entries["0xvictim"].refund_wei > 0
    assert res.post_state.fork.balances["alice"][ETH] > st.fork.balances["alice"][ETH] - 600_000 * 200 * GWEI
    rec = r.records["0xvictim"]
    assert rec.status is TxStatus.INCLUDED and rec.included_block == 0 and rec.winning_bundle == winner
    assert r.searchers["s1"].wins == 1 and r.bundle_status[winner] == "included"
    assert all(s == "target_resolved" for bid, s in r.bundle_status.items() if bid != winner)
    assert settle.via_public == []


def test_unregistered_builder_never_sees_private_flow():
    st = fresh_state()
    r = relay_with(builders=("b1",))
    r.send_private_transaction(swap_tx(st), st, 0, HINT_PROFILES["protect"])
    builders = [BlockBuilder("public", Strategy.GREEDY)]
    auc, res, settle = _run_slot(r, st, builders=builders)
    assert "0xvictim" not in {t.tx_hash for t in res.block.txs}
    assert r.status("0xvictim") is TxStatus.PENDING and settle.included == []


def test_protect_only_tx_lands_without_searchers():
    st = fresh_state()
    r = relay_with(builders=("b1",))
    r.send_private_transaction(swap_tx(st), st, 0, HINT_PROFILES["protect"])
    auc, res, settle = _run_slot(r, st)
    assert "bundle:protect:0xvictim" in res.block.included and settle.refunds_wei == 0
    assert r.status("0xvictim") is TxStatus.INCLUDED


def test_bundles_for_already_included_tx_are_rejected_next_block():
    st = fresh_state()
    r = relay_with(builders=("b1",))
    r.send_private_transaction(swap_tx(st), st, 0, HINT_PROFILES["full"])
    _run_slot(r, st)
    rc = r.send_bundle(bundle(r, "s2", "0xvictim", [dummy_backrun("s2")], block=1, bid="late"), 1)
    assert not rc.accepted and rc.reason == "target_included"


def test_public_nonce_race_fails_the_private_tx():
    st = fresh_state()
    r = relay_with(builders=("b1",))
    r.send_private_transaction(swap_tx(st), st, 0, HINT_PROFILES["full"])
    _backrun_bundles(r, st)
    public = swap_tx(st, amount=E18, tx_hash="0xpublic")  # same sender and nonce, sent publicly
    block, post = seal_block(st, [public], BASE_FEE)
    settle = r.on_block(0, block, post)
    assert settle.failed == {"0xvictim": "nonce_consumed"} and r.status("0xvictim") is TxStatus.FAILED
    assert all(s == "target_resolved" for s in r.bundle_status.values())


def test_multi_block_window_expiry_and_bundle_expiry():
    st = fresh_state()
    r = relay_with(builders=("b1",))
    r.send_private_transaction(swap_tx(st), st, 0, PrivacyPreferences(hints=FULL_DISCLOSURE, max_block_number=2))
    assert r.send_bundle(bundle(r, "s1", "0xvictim", [dummy_backrun("s1")], block=0, max_block=1), 0).accepted
    empty, post = seal_block(st, [], BASE_FEE)
    assert r.on_block(0, empty, post).expired == [] and r.status("0xvictim") is TxStatus.PENDING
    r.on_block(1, empty, post)
    assert r.bundle_status["b0"] == "expired"
    assert r.on_block(2, empty, post).expired == ["0xvictim"] and r.status("0xvictim") is TxStatus.EXPIRED
    assert r.build_auction(st, 3, BASE_FEE).bundles == []


def test_future_bundle_is_not_auctioned_early():
    st = fresh_state()
    r = relay_with()
    r.send_private_transaction(swap_tx(st), st, 0, HINT_PROFILES["full"])
    subs = _backrun_bundles(r, st)
    b0 = subs[0][0]
    later = MevShareBundle("later", "s2", b0.body[:1] + (BodyItem(tx=replace(b0.body[1].tx, sender="s2",
                                                                               tx_hash="s2:later")),),
                           3, 4, b0.coinbase_transfer_wei * 10).signed(r.searchers["s2"].key)
    assert r.send_bundle(later, 0).accepted
    assert r.build_auction(st, 0, BASE_FEE).entries["0xvictim"].winner != "later"
    assert r.build_auction(st, 3, BASE_FEE).entries["0xvictim"].winner in ("later", None)


def test_stale_backrun_fails_after_state_moves_between_blocks():
    st = fresh_state()
    r = relay_with(builders=("b1",))
    r.send_private_transaction(swap_tx(st), st, 0, HINT_PROFILES["full"])
    subs = _backrun_bundles(r, st)
    assert r.build_auction(st, 0, BASE_FEE).entries["0xvictim"].winner is not None
    # Another trader moves the v3 pool the other way before the next block; the backrun's
    # profit guard (amount_out_min = amount_in + 1) now trips.
    mover = swap_tx(st, route=(V3_UW,), token_in="WETH", amount=300 * E18, sender="whale", tx_hash="0xwhale")
    _, moved = seal_block(st, [mover], BASE_FEE)
    e = r.build_auction(moved, 1, BASE_FEE).entries["0xvictim"]
    assert e.winner is None and all(not c["valid"] for c in e.candidates) and len(e.candidates) == len(subs)


def test_two_private_txs_backrun_from_separate_sub_accounts_both_land():
    st = fresh_state()
    r = relay_with(builders=("b1",))
    r.send_private_transaction(swap_tx(st), st, 0, HINT_PROFILES["full"])
    _backrun_bundles(r, st, ("s1",), (9_000,))
    usdc_victim = swap_tx(st, route=(V2_UD,), token_in="DAI", amount=400_000 * E18, sender="dave", tx_hash="0xdave")
    r.send_private_transaction(usdc_victim, st, 0, HINT_PROFILES["full"])
    _backrun_bundles(r, st, ("s1",), (9_000,))
    auc, res, settle = _run_slot(r, st)
    assert set(settle.included) == {"0xvictim", "0xdave"}
    senders = {t.sender for t in res.block.txs if t.sender.startswith("s1#")}
    assert len(senders) == sum(1 for e in auc.entries.values() if e.winner and
                               f"bundle:mevshare:{e.winner}" in res.block.included)


# ---------------------------------------------------------------------------
# End-to-end simulation and controller wiring
# ---------------------------------------------------------------------------


def _sim_config(**kw):
    base = dict(blocks=2, txs_per_block=16, frontrun_grid=8, refine_iterations=4, layout_fuzz_iterations=0,
                private_flow_bps=6_000, parallel_threshold=10**9)
    base.update(kw)
    return SimulationConfig.for_mode("brutal", cpus=4, **base)


def test_private_flow_simulation_keeps_privacy_invariants():
    rep = PrivateOrderflowSimulation(market(), _sim_config()).run()
    s = rep.summary()
    assert s["invariants_ok"], s["invariant_failures"]
    assert s["private_txs"] > 0 and s["private_sandwiched"] == 0
    assert all(slot["settlement"]["via_public"] == [] for slot in s["slots"])
    assert s["privacy_violations_blocked"] >= 0 and json.dumps(s)
    assert set(s["exposure"]["profiles"]) <= set(HINT_PROFILES)


def test_private_flow_simulation_is_deterministic():
    a = PrivateOrderflowSimulation(market(), _sim_config(seed=11)).run(exposure=False).summary()
    b = PrivateOrderflowSimulation(market(), _sim_config(seed=11)).run(exposure=False).summary()
    a.pop("elapsed_s"), b.pop("elapsed_s")
    a.pop("exposure"), b.pop("exposure")
    assert a == b


def test_private_flow_parallel_exposure_matches_serial():
    serial = PrivateOrderflowSimulation(market(), _sim_config(seed=5, blocks=1)).run().exposure
    with WorkerPool(2, parallel_threshold=1) as wp:
        par = PrivateOrderflowSimulation(market(), _sim_config(seed=5, blocks=1)).run(executor=wp).exposure
    assert [r.to_dict() for r in serial.results] == [r.to_dict() for r in par.results]


def test_private_share_zero_routes_everything_public():
    s = PrivateOrderflowSimulation(market(), _sim_config(private_flow_bps=0, blocks=1)).run().summary()
    assert s["private_txs"] == 0 and s["relay"]["private_txs"] == 0 and s["invariants_ok"]


def test_controller_brutal_includes_private_relay_and_cli_flag(capsys):
    cfg = _sim_config(blocks=1, txs_per_block=8, fuzz_iterations=16, frontrun_grid=6, refine_iterations=3)
    out = SimulationController(cfg).run()
    assert out["result"]["private_relay"]["invariants_ok"] and "router" in out["result"]
    assert out["config"]["private_flow_bps"] == 6_000
    rc = cli_main(["--mode", "brutal", "--blocks", "1", "--txs", "8", "--fuzz", "16", "--layouts", "16",
                   "--private-share", "5000", "--refund", "7000", "--max-hops", "3"])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0 and data["config"]["mev_share_refund_bps"] == 7_000
    assert data["result"]["private_relay"]["invariants_ok"]
    no_private = SimulationController(_sim_config(blocks=1, txs_per_block=6, fuzz_iterations=8,
                                                  private_flow_bps=0)).run()
    assert "private_relay" not in no_private["result"]


def test_demo_market_exposure_matrix_runs():
    st = BuilderState.from_pools(demo_market())
    tx = swap_tx(st, route=("v2:WETH/USDC",), amount=80 * E18, tolerance_bps=500)
    m = hint_exposure_matrix(st, [tx], {"full": HINT_PROFILES["full"], "protect": HINT_PROFILES["protect"]})
    rows = m.rows()
    assert rows["protect"]["realized_profit_native"] == 0
    assert rows["full"]["exact_knowledge_share"] == 1.0
    assert best_sandwich(st.pools, tx) is not None
