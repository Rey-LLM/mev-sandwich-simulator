"""Verification suite for Layer 04: PBS block builder, transaction ordering,
bundle atomicity, sandwich / JIT guards, gas-capped sealing with traces, the
transaction pool's eviction rules, the block layout fuzzer and the MEV-Boost
auction."""

from __future__ import annotations

import json
import logging
import random

import pytest

from simulator.__main__ import main as cli_main
from simulator.block_builder import (
    BLOCK_GAS_LIMIT,
    BURN,
    ETH,
    FUZZ_SHARDS,
    GWEI,
    BlockBuilder,
    Bundle,
    BuilderBid,
    BuilderState,
    BuilderTx,
    GuardAction,
    GuardPolicy,
    MevBoostAuction,
    PBSSimulation,
    Proposer,
    Relay,
    Status,
    Strategy,
    TxKind,
    TxPool,
    best_sandwich,
    build_jit_bundle,
    build_sandwich_bundles,
    clamp_gas_limit,
    classify_bundle,
    execute_bundle,
    execute_tx,
    fuzz_block_layout,
    generate_builder_mempool,
    item_id,
    modify_liquidity,
    mutate_layout,
    next_base_fee,
    price_and_nonce_order,
    repair_nonce_order,
    seal_block,
)
from simulator.config import PRESETS, ConfigError, RunMode, SimulationConfig
from simulator.controller import SimulationController, demo_market
from simulator.parallel import WorkerPool
from simulator.slippage_engine import evaluate_trade, min_amount_out
from simulator.state_fork import SwapReverted

E18 = 10**18
E6 = 10**6
BASE = 20 * GWEI
V2 = "v2:WETH/USDC"
V3 = "v3:USDC/WETH/3000"
V4 = "v4:WETH/DAI/500"
SERIAL = 10**9


def state() -> BuilderState:
    return BuilderState.from_pools(demo_market())


def swap(h, sender="alice", nonce=0, route=(V2,), token_in="WETH", amount=E18, min_out=0,
         gas=300_000, cap=100 * GWEI, tip=2 * GWEI) -> BuilderTx:
    return BuilderTx(h, sender, nonce, gas, cap, tip, TxKind.SWAP,
                     {"route": list(route), "token_in": token_in, "amount_in": amount, "amount_out_min": min_out})


def transfer(h, sender="bob", nonce=0, gas=21_000, cap=100 * GWEI, tip=GWEI, value=E18) -> BuilderTx:
    return BuilderTx(h, sender, nonce, gas, cap, tip, TxKind.TRANSFER, {"to": "carol", "value": value})


def victim(st: BuilderState, h="0xvictim", amount=40 * E18, tolerance=1_000, sender="alice") -> BuilderTx:
    quote = evaluate_trade(st.pools[V2], "WETH", amount)
    return swap(h, sender, st.nonce(sender), amount=amount, min_out=min_amount_out(quote.amount_out, tolerance))


def small_cfg(**kw) -> SimulationConfig:
    base = dict(blocks=2, txs_per_block=16, frontrun_grid=10, refine_iterations=4, fuzz_iterations=32,
                layout_fuzz_iterations=64, parallel_threshold=SERIAL)
    base.update(kw)
    return SimulationConfig.for_mode("brutal", cpus=4, **base)


# ---------------------------------------------------------------------------
# Gas limit, base fee, config
# ---------------------------------------------------------------------------


def test_gas_limit_is_clamped_to_30m():
    assert clamp_gas_limit(None) == BLOCK_GAS_LIMIT == 30_000_000
    assert clamp_gas_limit(60_000_000) == 30_000_000
    assert clamp_gas_limit(5) == 21_000


@pytest.mark.parametrize("used,expected", [(15_000_000, BASE), (30_000_000, BASE + BASE // 8), (0, BASE - BASE // 8)])
def test_eip1559_base_fee_update(used, expected):
    assert next_base_fee(BASE, used, 30_000_000) == expected


def test_config_rejects_gas_limit_over_cap_and_presets_wire_fuzzer():
    with pytest.raises(ConfigError):
        SimulationConfig.for_mode("brutal", cpus=4, block_gas_limit=30_000_001)
    assert PRESETS[RunMode.BRUTAL_AGGRESSIVE]["layout_fuzz_iterations"] > 0
    assert PRESETS[RunMode.MODERATE_SCAN]["layout_fuzz_iterations"] == 0
    assert SimulationConfig.for_mode("brutal", cpus=4).to_dict()["block_gas_limit"] == 30_000_000


# ---------------------------------------------------------------------------
# Single transaction execution
# ---------------------------------------------------------------------------


def test_effective_tip_follows_eip1559():
    tx = swap("0x1", cap=25 * GWEI, tip=10 * GWEI)
    assert tx.effective_tip(BASE) == 5 * GWEI
    assert tx.effective_tip(10 * GWEI) == 10 * GWEI
    assert tx.effective_tip(26 * GWEI) is None


def test_fee_cap_below_base_fee_is_invalid_and_untouched():
    st = state()
    root = st.state_root()
    t = execute_tx(st, swap("0x1", cap=BASE - 1, tip=1), BASE, BLOCK_GAS_LIMIT)
    assert t.status is Status.INVALID and t.error == "fee_cap_below_base_fee"
    assert st.state_root() == root and st.nonce("alice") == 0


@pytest.mark.parametrize("nonce,err", [(1, "nonce_gap"), (0, "nonce_too_low")])
def test_nonce_rules(nonce, err):
    st = state()
    st.nonces["alice"] = 0 if err == "nonce_gap" else 1
    t = execute_tx(st, swap("0x1", nonce=nonce), BASE, BLOCK_GAS_LIMIT)
    assert t.status is Status.INVALID and t.error == err


def test_successful_swap_pays_fees_and_traces_state_changes():
    st = state()
    t = execute_tx(st, swap("0x1"), BASE, BLOCK_GAS_LIMIT, coinbase="cb", trace=True)
    assert t.status is Status.SUCCESS and st.nonce("alice") == 1
    assert t.priority_fee_wei == t.gas_used * 2 * GWEI and t.base_fee_burned_wei == t.gas_used * BASE
    assert st.fork.balance("cb", ETH) == t.priority_fee_wei and st.fork.balance(BURN, ETH) == t.base_fee_burned_wei
    assert st.fork.balance("alice", "WETH") == -E18 and st.fork.balance("alice", "USDC") > 0
    assert any(c.startswith(f"pool {V2} reserve0:") for c in t.state_changes)
    assert any(c.startswith("balance alice WETH: 0 -> -") for c in t.state_changes)


def test_out_of_gas_consumes_limit_and_discards_body():
    st = state()
    pool_before = (st.pools[V2].reserve0, st.pools[V2].reserve1)
    t = execute_tx(st, swap("0x1", gas=50_000), BASE, BLOCK_GAS_LIMIT)
    assert t.status is Status.OUT_OF_GAS and t.gas_used == 50_000
    assert (st.pools[V2].reserve0, st.pools[V2].reserve1) == pool_before
    assert st.nonce("alice") == 1 and st.fork.balance("alice", ETH) == -50_000 * (BASE + 2 * GWEI)


def test_min_out_revert_charges_gas_but_keeps_pool():
    st = state()
    k = st.pools[V2].k
    t = execute_tx(st, swap("0x1", min_out=10**30), BASE, BLOCK_GAS_LIMIT)
    assert t.status is Status.REVERTED and "below min" in t.error and t.gas_used > 0
    assert st.pools[V2].k == k and st.fork.balance("alice", "WETH") == 0


def test_multi_hop_swap_nets_intermediate_token_to_zero():
    st = state()
    t = execute_tx(st, swap("0x1", route=(V2, V4), token_in="USDC", amount=3_000 * E6), BASE, BLOCK_GAS_LIMIT)
    assert t.status is Status.SUCCESS
    assert st.fork.balance("alice", "WETH") == 0
    assert st.fork.balance("alice", "USDC") == -3_000 * E6 and st.fork.balance("alice", "DAI") > 0


def test_route_repeating_a_pool_reverts():
    t = execute_tx(state(), swap("0x1", route=(V2, V2)), BASE, BLOCK_GAS_LIMIT)
    assert t.status is Status.REVERTED and "once" in t.error


def test_transfer_costs_intrinsic_gas_and_moves_eth():
    st = state()
    t = execute_tx(st, transfer("0xt"), BASE, BLOCK_GAS_LIMIT)
    assert t.status is Status.SUCCESS and t.gas_used == 21_000
    assert st.fork.balance("carol", ETH) == E18


def test_mint_burn_roundtrip_restores_liquidity_and_ticks():
    st = state()
    pool = st.pools[V3]
    liq, ticks = pool.liquidity, dict(pool.ticks)
    s = pool.tick_spacing
    lo, hi = (pool.tick // s - 2) * s, (pool.tick // s + 2) * s
    body = {"pool_id": V3, "tick_lower": lo, "tick_upper": hi, "liquidity": 10**18}
    mint = BuilderTx("0xm", "lp", 0, 200_000, 100 * GWEI, GWEI, TxKind.MINT, body)
    burn = BuilderTx("0xb", "lp", 1, 200_000, 100 * GWEI, GWEI, TxKind.BURN, body)
    assert execute_tx(st, mint, BASE, BLOCK_GAS_LIMIT).status is Status.SUCCESS
    assert pool.liquidity == liq + 10**18 and st.positions[("lp", V3, lo, hi)] == 10**18
    assert execute_tx(st, burn, BASE, BLOCK_GAS_LIMIT).status is Status.SUCCESS
    assert pool.liquidity == liq and pool.ticks == ticks and not st.positions
    over = BuilderTx("0xb2", "lp", 2, 200_000, 100 * GWEI, GWEI, TxKind.BURN, body)
    t = execute_tx(st, over, BASE, BLOCK_GAS_LIMIT)
    assert t.status is Status.REVERTED and "cannot burn" in t.error and pool.liquidity == liq


def test_modify_liquidity_rejects_v2_and_misaligned_ticks():
    st = state()
    with pytest.raises(SwapReverted):
        modify_liquidity(st.pools[V2], -60, 60, 1)
    with pytest.raises(SwapReverted):
        modify_liquidity(st.pools[V3], -61, 60, 1)


# ---------------------------------------------------------------------------
# Bundles and guards
# ---------------------------------------------------------------------------


def test_bundle_is_atomic():
    st = state()
    root = st.state_root()
    b = Bundle("b", (swap("0x1", "s", 0), swap("0x2", "s", 1, min_out=10**30), swap("0x3", "s", 2)))
    r = execute_bundle(st, b, BASE, BLOCK_GAS_LIMIT, set())
    assert not r.included and r.reason.startswith("tx_reverted:0x2")
    assert st.state_root() == root and st.nonce("s") == 0


def test_bundle_reverting_tx_hashes_allow_listed_revert():
    st = state()
    b = Bundle("b", (swap("0x1", "s", 0), swap("0x2", "s", 1, min_out=10**30)), reverting_tx_hashes=frozenset({"0x2"}))
    r = execute_bundle(st, b, BASE, BLOCK_GAS_LIMIT, set())
    assert r.included and [t.status for t in r.traces] == [Status.SUCCESS, Status.REVERTED]


def test_bundle_conflicting_with_included_tx_is_dropped():
    st = state()
    r = execute_bundle(st, Bundle("b", (swap("0x1"),)), BASE, BLOCK_GAS_LIMIT, {"0x1"})
    assert not r.included and r.reason == "conflict_already_included:0x1"


def test_coinbase_transfer_reaches_builder():
    st = state()
    b = Bundle("b", (swap("0x1", "s", 0),), coinbase_transfer_wei=7 * E18 // 100, searcher="s")
    block, post = seal_block(st, [b], BASE, builder="bx", coinbase="cb")
    assert block.coinbase_transfers_wei == 7 * E18 // 100
    assert post.fork.balance("cb", ETH) == block.value_wei
    assert block.value_wei == block.priority_fees_wei + 7 * E18 // 100


def test_classify_bundle_labels():
    st = state()
    v = victim(st)
    sandwich = Bundle("s", (swap("f", "m", 0), v, swap("b", "m", 1, token_in="USDC")))
    multi = Bundle("m", (swap("f", "m", 0), swap("v", "alice", route=(V2, V4), token_in="USDC"),
                         swap("b", "m", 1, token_in="USDC")))
    jit = build_jit_bundle(st, swap("v3v", route=(V3,), token_in="USDC", amount=10**9), "lp", 10**18, BASE)
    assert classify_bundle(sandwich) == ["sandwich"]
    assert classify_bundle(multi) == ["multi_hop_sandwich"]
    assert jit is not None and classify_bundle(jit) == ["jit_liquidity"]
    assert classify_bundle(Bundle("p", (swap("a", "x", 0), swap("b", "y", 0)))) == []


def test_guard_policy_reject_and_flag():
    st = state()
    v = victim(st)
    bundles = build_sandwich_bundles(st, [v], BASE, grid=12, refine=6)
    assert bundles
    reject = GuardPolicy(sandwich=GuardAction.REJECT)
    block, _ = seal_block(st, bundles, BASE, policy=reject)
    assert not block.included and block.skipped[item_id(bundles[0])] == "guard_rejected:sandwich"
    block, _ = seal_block(st, bundles, BASE)
    assert block.included == [item_id(bundles[0])] and block.flags[item_id(bundles[0])] == ["sandwich"]


def test_jit_bundle_executes_and_returns_liquidity():
    st = state()
    v = swap("0xv3", route=(V3,), token_in="USDC", amount=50_000 * E6)
    liq = st.pools[V3].liquidity
    jit = build_jit_bundle(st, v, "lp", liq * 4, BASE)
    block, post = seal_block(st, [jit], BASE, policy=GuardPolicy(jit_liquidity=GuardAction.FLAG))
    assert block.included == [item_id(jit)] and post.pools[V3].liquidity == liq and not post.positions
    rejected, _ = seal_block(st, [jit], BASE, policy=GuardPolicy(jit_liquidity=GuardAction.REJECT))
    assert rejected.skipped[item_id(jit)] == "guard_rejected:jit_liquidity"


def test_stale_state_makes_sandwich_bundle_drop_instead_of_mispricing():
    st = state()
    v = victim(st)
    bundles = build_sandwich_bundles(st, [v], BASE, grid=12, refine=6)
    mover = swap("0xmove", "whale", 0, amount=20 * E18)
    block, _ = seal_block(st, [mover] + bundles, BASE)
    assert block.skipped[item_id(bundles[0])].startswith("tx_reverted:")
    assert block.coinbase_transfers_wei == 0


# ---------------------------------------------------------------------------
# Sealing: gas cap, overflow rejection loop, traces
# ---------------------------------------------------------------------------


def test_gas_overflow_rejection_loop_keeps_filling_with_smaller_items():
    st = state()
    # 0xa reserves 80k but uses 21k, leaving 79k: 0xb (85k limit) cannot fit, 0xc still can.
    txs = [transfer("0xa", "a", gas=80_000), transfer("0xb", "b", gas=85_000), transfer("0xc", "c")]
    block, _ = seal_block(st, txs, BASE, gas_limit=100_000)
    assert block.included == ["tx:0xa", "tx:0xc"]
    assert block.skipped["tx:0xb"] == "gas_limit_exceeds_remaining" and block.gas_rejections == 1
    assert block.gas_used <= 100_000


def test_block_gas_never_exceeds_30m():
    st = state()
    txs = [transfer(f"0x{i:05x}", f"u{i}", gas=21_000) for i in range(1_500)]
    block, _ = seal_block(st, txs, BASE, gas_limit=90_000_000, trace=False)
    assert block.gas_limit == 30_000_000
    assert block.gas_used == 30_000_000 // 21_000 * 21_000 <= 30_000_000
    assert sum(1 for r in block.skipped.values() if r == "block_full") == 1_500 - 30_000_000 // 21_000


def test_cumulative_gas_and_log_lines():
    st = state()
    txs = [swap("0x1"), transfer("0x2"), swap("0x3", "carl", route=(V2, V4), token_in="USDC", amount=1_000 * E6)]
    block, _ = seal_block(st, txs, BASE, builder="b1")
    cums = [t.cumulative_gas for t in block.txs]
    assert cums == sorted(cums) and cums[-1] == block.gas_used
    assert [t.index for t in block.txs] == [0, 1, 2]
    lines = block.log_lines()
    assert lines[0].startswith("block 0 builder=b1") and f"/{BLOCK_GAS_LIMIT}" in lines[0]
    assert sum(1 for line in lines if line.lstrip().startswith("pool ")) >= 3
    assert all(any(t.tx_hash in line for line in lines) for t in block.txs)


def test_seal_never_mutates_parent_and_skips_duplicates():
    st = state()
    root = st.state_root()
    tx = swap("0x1")
    block, post = seal_block(st, [tx, tx], BASE)
    assert st.state_root() == root and post.state_root() != root
    assert block.skipped == {"tx:0x1": "duplicate"}


def test_unrealized_mev_is_measured_for_unsandwiched_swaps():
    st = state()
    block, _ = seal_block(st, [victim(st)], BASE, measure_unrealized=True)
    assert block.unrealized_mev_wei > 0 and block.txs[0].unrealized_mev_wei == block.unrealized_mev_wei


def test_trace_is_logged_at_debug(caplog):
    st = state()
    with caplog.at_level(logging.DEBUG, logger="simulator.block_builder"):
        seal_block(st, [swap("0x1")], BASE, builder="logged")
    assert any("builder=logged" in r.message for r in caplog.records)
    assert any("reserve0" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Transaction pool and ordering
# ---------------------------------------------------------------------------


def test_replacement_needs_ten_percent_bump():
    pool = TxPool()
    assert pool.add(swap("0x1", cap=100 * GWEI, tip=10 * GWEI), BASE).status == "added"
    assert pool.add(swap("0x2", cap=109 * GWEI, tip=11 * GWEI), BASE).status == "rejected_replacement_underpriced"
    res = pool.add(swap("0x3", cap=110 * GWEI, tip=11 * GWEI), BASE)
    assert res.status == "replaced" and res.evicted == ("0x1",) and "0x3" in pool and "0x1" not in pool


def test_full_pool_evicts_cheapest_with_its_later_nonces():
    pool = TxPool(capacity=3)
    pool.add(swap("0xa0", "a", 0, tip=GWEI), BASE)
    pool.add(swap("0xa1", "a", 1, tip=9 * GWEI), BASE)
    pool.add(swap("0xb0", "b", 0, tip=5 * GWEI), BASE)
    assert pool.add(swap("0xc0", "c", 0, tip=GWEI // 2), BASE).status == "rejected_pool_full"
    res = pool.add(swap("0xd0", "d", 0, tip=3 * GWEI), BASE)
    assert res.status == "added" and set(res.evicted) == {"0xa0", "0xa1"} and len(pool) == 2


def test_pending_queued_and_prune():
    pool = TxPool()
    for tx in (swap("0x0", nonce=0), swap("0x1", nonce=1), swap("0x3", nonce=3)):
        pool.add(tx, BASE)
    assert [t.tx_hash for t in pool.pending({})] == ["0x0", "0x1"]
    assert [t.tx_hash for t in pool.queued({})] == ["0x3"]
    assert pool.prune({"alice": 2}) == ["0x0", "0x1"] and len(pool) == 1


@pytest.mark.parametrize("tx,status", [
    (swap("0x1", gas=30_000_001), "rejected_gas_limit"),
    (swap("0x1", gas=20_000), "rejected_intrinsic_gas"),
    (swap("0x1", cap=GWEI, tip=2 * GWEI), "rejected_tip_above_cap"),
])
def test_pool_admission_rules(tx, status):
    assert TxPool().add(tx, BASE).status == status


def test_pool_rejects_stale_nonce():
    assert TxPool().add(swap("0x1", nonce=0), BASE, account_nonce=1).status == "rejected_nonce_too_low"


def test_price_and_nonce_order_keeps_nonces_and_drops_underpriced():
    txs = [swap("a0", "a", 0, tip=GWEI), swap("a1", "a", 1, tip=10 * GWEI), swap("b0", "b", 0, tip=5 * GWEI),
           swap("c0", "c", 0, cap=BASE - 1, tip=1)]
    order = [t.tx_hash for t in price_and_nonce_order(txs, BASE, {})]
    assert order == ["b0", "a0", "a1"]


def test_greedy_ordering_is_stable_under_input_permutation():
    st = state()
    txs = generate_builder_mempool(st, 24, seed=5, base_fee=BASE)
    bundles = build_sandwich_bundles(st, txs, BASE, grid=8, refine=4)
    builder = BlockBuilder("g", Strategy.GREEDY)
    layout_a, _ = builder.order(st, bundles, txs, BASE, BLOCK_GAS_LIMIT)
    rng = random.Random(1)
    shuffled_b, shuffled_t = bundles[:], txs[:]
    rng.shuffle(shuffled_b)
    rng.shuffle(shuffled_t)
    layout_b, _ = builder.order(st, shuffled_b, shuffled_t, BASE, BLOCK_GAS_LIMIT)
    assert [item_id(i) for i in layout_a] == [item_id(i) for i in layout_b]
    bundle_part = [i for i in layout_a if isinstance(i, Bundle)]
    assert layout_a[:len(bundle_part)] == bundle_part


# ---------------------------------------------------------------------------
# Sandwich search
# ---------------------------------------------------------------------------


def test_best_sandwich_respects_victim_tolerance():
    st = state()
    loose = best_sandwich(st.pools, victim(st, tolerance=1_000), grid=16, refine=8)
    assert loose is not None and loose.profit > 0 and not loose.multi_hop
    tight = victim(st, amount=E18 // 10, tolerance=1)
    assert best_sandwich(st.pools, tight, grid=16, refine=8) is None


def test_multi_hop_victim_gets_multi_hop_sandwich():
    st = state()
    tx = swap("0xmh", route=(V2, V4), token_in="USDC", amount=300_000 * E6)
    quote = execute_tx(st.copy(), tx, BASE, BLOCK_GAS_LIMIT)
    assert quote.status is Status.SUCCESS
    loose = swap("0xmh", route=(V2, V4), token_in="USDC", amount=300_000 * E6, min_out=1)
    cand = best_sandwich(st.pools, loose, grid=16, refine=8)
    assert cand is not None and cand.multi_hop
    bundles = build_sandwich_bundles(st, [loose], BASE, grid=16, refine=8)
    assert bundles and classify_bundle(bundles[0]) == ["multi_hop_sandwich"]


def test_sandwich_bundle_pays_bribe_and_searcher_profits():
    st = state()
    bundles = build_sandwich_bundles(st, [victim(st)], BASE, grid=16, refine=8, bribe_bps=8_000)
    block, post = seal_block(st, bundles, BASE, coinbase="cb")
    b = bundles[0]
    assert block.included == [item_id(b)] and block.coinbase_transfers_wei == b.coinbase_transfer_wei > 0
    assert post.fork.balance(b.searcher, "WETH") > 0


# ---------------------------------------------------------------------------
# Layout fuzzer
# ---------------------------------------------------------------------------


def test_repair_nonce_order_sorts_each_sender_in_place():
    items = [swap("a1", "a", 1), swap("b0", "b", 0), swap("a0", "a", 0)]
    assert repair_nonce_order([0, 1, 2], items) == [2, 1, 0]


def test_mutations_preserve_the_item_set():
    st = state()
    items = generate_builder_mempool(st, 20, seed=2, base_fee=BASE)
    rng = random.Random(3)
    layout, dropped = list(range(len(items))), []
    for _ in range(300):
        layout, dropped, _ = mutate_layout(layout, dropped, items, rng)
        assert sorted(layout + dropped) == list(range(len(items)))


def _conflicted_layout():
    st = state()
    v = victim(st)
    bundles = build_sandwich_bundles(st, [v], BASE, grid=12, refine=6)
    filler = [transfer(f"0xf{i}", f"f{i}") for i in range(4)]
    items = [v] + filler + bundles  # public victim first: the bundle conflicts and is lost
    return st, items


def test_fuzzer_recovers_mev_lost_to_a_bad_layout():
    st, items = _conflicted_layout()
    fz = fuzz_block_layout(st, items, base_fee=BASE, iterations=256, seed=1)
    assert fz.value_wei >= fz.baseline_value_wei
    assert fz.gain_wei > 0 and fz.evaluated == 256 + FUZZ_SHARDS
    block, _ = seal_block(st, [items[i] for i in fz.layout], BASE, trace=False)
    assert block.value_wei == fz.value_wei and block.coinbase_transfers_wei > 0


def test_fuzzer_parallel_matches_serial():
    st, items = _conflicted_layout()
    serial = fuzz_block_layout(st, items, base_fee=BASE, iterations=128, seed=4)
    with WorkerPool(2, "spawn", parallel_threshold=1) as wp:
        par = fuzz_block_layout(st, items, base_fee=BASE, iterations=128, seed=4, executor=wp)
    assert (par.layout, par.value_wei, par.evaluated, par.ops) == (serial.layout, serial.value_wei,
                                                                    serial.evaluated, serial.ops)


def test_fuzz_builder_never_bids_below_greedy():
    st = state()
    txs = generate_builder_mempool(st, 24, seed=11, base_fee=BASE)
    bundles = build_sandwich_bundles(st, txs, BASE, grid=8, refine=4)
    greedy, _, _ = BlockBuilder("g", Strategy.GREEDY).build(st, bundles, txs, BASE)
    fuzzed, _, info = BlockBuilder("f", Strategy.FUZZ, fuzz_iterations=96).build(st, bundles, txs, BASE, seed=3)
    assert fuzzed.value_wei >= greedy.value_wei and info["fuzz"]["layouts_evaluated"] == 96 + FUZZ_SHARDS


# ---------------------------------------------------------------------------
# Relays, proposer, auction
# ---------------------------------------------------------------------------


def _built():
    st = state()
    txs = generate_builder_mempool(st, 16, seed=8, base_fee=BASE)
    bundles = build_sandwich_bundles(st, txs, BASE, grid=8, refine=4)
    builder = BlockBuilder("g", Strategy.GREEDY, margin_bps=1_000)
    block, _, _ = builder.build(st, bundles, txs, BASE)
    return st, builder, block


def test_builder_margin_sets_bid():
    _, builder, block = _built()
    bid = builder.bid_for(block, 0)
    assert bid.value_wei == block.value_wei * 9_000 // 10_000 and bid.block_value_wei == block.value_wei


@pytest.mark.parametrize("tamper,reason", [
    (dict(value_wei=10**30), "bid_exceeds_block_value"),
    (dict(gas_used=1), "gas_used_mismatch"),
    (dict(slot=9), "wrong_slot"),
    (dict(block_hash="0xforged"), "block_hash_mismatch"),
    (dict(gas_limit=40_000_000), "gas_limit_above_cap"),
])
def test_relay_rejects_tampered_bids(tamper, reason):
    st, builder, block = _built()
    good = builder.bid_for(block, 0)
    relay = Relay()
    assert relay.submit(good, block, st, 0, BASE) is None
    fields = {**good.to_dict(), **tamper}
    assert relay.submit(BuilderBid(**fields), block, st, 0, BASE) == reason


def test_proposer_takes_highest_bid_and_falls_back_to_local():
    st = state()
    txs = generate_builder_mempool(st, 16, seed=8, base_fee=BASE)
    bundles = build_sandwich_bundles(st, txs, BASE, grid=8, refine=4)
    builders = [BlockBuilder("vanilla", Strategy.FEE_ONLY, 0), BlockBuilder("greedy", Strategy.GREEDY, 500)]
    res = MevBoostAuction(builders).run_slot(st, bundles, txs, BASE, slot=0)
    assert res.winner == "greedy" and res.proposer_payment_wei == max(b.value_wei for b in res.bids)
    assert res.post_state.fork.balance("proposer:validator", ETH) == res.proposer_payment_wei
    picky = MevBoostAuction(builders, proposer=Proposer(min_bid_wei=10**40)).run_slot(st, bundles, txs, BASE, 0)
    assert picky.winner == "local" and picky.proposer_payment_wei == picky.local_value_wei


def test_pbs_simulation_multi_slot_invariants_and_determinism():
    cfg = small_cfg(blocks=3)
    a = PBSSimulation(demo_market(), cfg).run().summary()
    b = PBSSimulation(demo_market(), cfg).run().summary()
    assert a["invariants_ok"], a["invariant_failures"]
    assert a["final_state_root"] == b["final_state_root"]
    assert len(a["slots"]) == 3 and a["max_gas_used"] <= 30_000_000
    assert len(a["base_fees"]) == 3 and a["proposer_revenue_wei"] > 0
    assert {"fuzzer", "greedy", "vanilla"} == set(a["slots"][0]["bids"])
    json.dumps(a)


def test_pbs_simulation_parallel_matches_serial():
    cfg = small_cfg(blocks=2)
    serial = PBSSimulation(demo_market(), cfg).run().summary()
    with WorkerPool(2, "spawn", parallel_threshold=1) as wp:
        par = PBSSimulation(demo_market(), cfg).run(executor=wp).summary()
    assert serial["final_state_root"] == par["final_state_root"]
    assert [s["winner"] for s in serial["slots"]] == [s["winner"] for s in par["slots"]]


def test_controller_brutal_includes_pbs_and_cli_clamps_gas(capsys):
    out = SimulationController(small_cfg(blocks=1)).run()
    assert out["result"]["pbs"]["invariants_ok"] and out["config"]["layout_fuzz_iterations"] == 64
    rc = cli_main(["--mode", "brutal", "--blocks", "1", "--txs", "8", "--fuzz", "16", "--layouts", "32",
                   "--gas-limit", "99000000", "--workers", "1"])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0 and data["config"]["block_gas_limit"] == 30_000_000
    assert data["result"]["pbs"]["gas_limit"] == 30_000_000


def test_builder_state_payload_roundtrip():
    st = state()
    st, _ = seal_block(st, [swap("0x1"), transfer("0x2")], BASE)[1], None
    assert BuilderState.from_payload(st.to_payload()).state_root() == st.state_root()
