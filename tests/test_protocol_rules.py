"""Protocol rules that the simulator must follow exactly.

* EIP-1559 base fee update (vectors from geth ``consensus/misc/eip1559`` tests
  and the EIP text) and effective tip.
* Nonces: per-sender consecutive nonces that persist across blocks, and
  price-and-nonce ordering in the simplified block model.
* Out-of-gas transactions use their nonce but change nothing.
* Sandwich accounting: an executed sandwich is net-positive after gas.
* Funding rules in ``strict_balances`` mode.
* Transaction replacement threshold with geth's integer rounding.
* Uniswap v4 fee bounds and tick liquidityGross bookkeeping.
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from simulator.block_builder import (
    ETH,
    GWEI,
    BuilderState,
    BuilderTx,
    Bundle,
    Status,
    TxKind,
    TxPool,
    execute_bundle,
    execute_tx,
    modify_liquidity,
    next_base_fee,
)
from simulator.config import SimulationConfig
from simulator.controller import demo_market
from simulator.mempool_emulator import (
    MempoolEmulator,
    PendingTx,
    _has_enough_gas,
    block_order,
    estimate_swap_gas,
    generate_mempool,
    native_to_token,
)
from simulator.slippage_engine import (
    ConcentratedPool,
    MASEError,
    Position,
    V2Pool,
    V4Pool,
    compute_swap_step,
)
from simulator.state_fork import StateFork

E18 = 10**18
SERIAL = 10**9
V2 = "v2:WETH/USDC"


# ---------------------------------------------------------------------------
# EIP-1559
# ---------------------------------------------------------------------------

INITIAL_BASE_FEE = 1_000_000_000  # params.InitialBaseFee


@pytest.mark.parametrize("gas_limit,gas_used,expected", [
    # geth consensus/misc/eip1559/eip1559_test.go TestCalcBaseFee
    (20_000_000, 10_000_000, 1_000_000_000),  # usage == target
    (20_000_000, 9_000_000, 987_500_000),  # usage below target
    (20_000_000, 11_000_000, 1_012_500_000),  # usage above target
])
def test_next_base_fee_geth_vectors(gas_limit, gas_used, expected):
    assert next_base_fee(INITIAL_BASE_FEE, gas_used, gas_limit) == expected


def test_next_base_fee_rounding_and_minimum_increase():
    # Increase is at least 1 wei; decrease is floored and may be 0.
    assert next_base_fee(7, 15_000_001, 30_000_000) == 8
    assert next_base_fee(7, 14_999_999, 30_000_000) == 7
    # Full block: +12.5%, empty block: -12.5% (floored).
    assert next_base_fee(10**9 + 7, 30_000_000, 30_000_000) == 10**9 + 7 + (10**9 + 7) // 8
    assert next_base_fee(10**9 + 7, 0, 30_000_000) == 10**9 + 7 - (10**9 + 7) // 8


def _swap_tx(h, sender="alice", nonce=0, amount=E18, gas=300_000, cap=100 * GWEI, tip=2 * GWEI, token_in="WETH",
             min_out=0):
    return BuilderTx(h, sender, nonce, gas, cap, tip, TxKind.SWAP,
                     {"route": [V2], "token_in": token_in, "amount_in": amount, "amount_out_min": min_out})


def test_fees_are_charged_on_gas_used_at_effective_price():
    st = BuilderState.from_pools(demo_market())
    base = 30 * GWEI
    tx = _swap_tx("0x1", cap=31 * GWEI, tip=5 * GWEI)  # effective tip = min(5, 31 - 30) = 1 gwei
    t = execute_tx(st, tx, base, 30_000_000)
    assert t.status is Status.SUCCESS
    assert t.tip_per_gas == GWEI
    assert t.priority_fee_wei == t.gas_used * GWEI
    assert t.base_fee_burned_wei == t.gas_used * base
    assert st.fork.balance("alice", ETH) == -t.gas_used * (base + GWEI)
    assert t.gas_used < tx.gas_limit  # unused gas is not charged


# ---------------------------------------------------------------------------
# Nonces and ordering in the simplified (mempool emulator) block model
# ---------------------------------------------------------------------------


def test_generated_nonces_are_consecutive_per_sender():
    txs = generate_mempool(demo_market(), 400, seed=11)
    by_sender = defaultdict(list)
    for tx in txs:
        by_sender[tx.sender].append(tx.nonce)
    assert any(len(v) > 1 for v in by_sender.values())
    for nonces in by_sender.values():
        assert nonces == list(range(len(nonces)))


def test_generated_nonces_start_at_account_nonce():
    first = generate_mempool(demo_market(), 50, seed=3)
    start = {tx.sender: 7 for tx in first}
    again = generate_mempool(demo_market(), 50, seed=3, nonces=start)
    assert [t.nonce for t in again] == [t.nonce + 7 for t in first]


def _ptx(h, sender, nonce, fee):
    return PendingTx(h, sender, {"route": [V2], "token_in": "WETH", "amount_in": 1}, 100_000, fee, nonce)


def test_block_order_keeps_sender_nonce_order_under_price_priority():
    txs = [_ptx("0xa0", "a", 0, 1 * GWEI), _ptx("0xa1", "a", 1, 9 * GWEI), _ptx("0xb0", "b", 0, 5 * GWEI),
           _ptx("0xc1", "c", 1, 8 * GWEI), _ptx("0xc0", "c", 0, 2 * GWEI)]
    order = [t.tx_hash for t in block_order(txs)]
    # b (5) beats a's head (1) and c's head (2); c0 then unlocks c1 (8); a0 unlocks a1 (9).
    assert order == ["0xb0", "0xc0", "0xc1", "0xa0", "0xa1"]


def test_block_order_drops_transactions_after_a_nonce_gap():
    txs = [_ptx("0xa0", "a", 0, GWEI), _ptx("0xa2", "a", 2, 9 * GWEI)]
    assert [t.tx_hash for t in block_order(txs)] == ["0xa0"]


def _cfg(**kw):
    base = dict(blocks=3, txs_per_block=40, frontrun_grid=8, refine_iterations=4, fuzz_iterations=24,
                parallel_threshold=SERIAL)
    base.update(kw)
    return SimulationConfig.for_mode("brutal", cpus=4, **base)


def test_run_blocks_counts_out_of_gas_and_nets_sandwiches_after_gas():
    report = MempoolEmulator(demo_market(), _cfg()).run_blocks()
    summary = report.summary()
    assert sum(b["out_of_gas"] for b in summary["blocks"]) > 0  # 0.8x gas limits are generated
    executed = [s for b in report.blocks for s in b.sandwiches]
    assert executed
    for s in executed:
        assert s["profit"] > 0
        if s["net_profit"] is not None:
            assert s["net_profit"] == s["profit"] - s["gas_cost_in_token_in"]
            assert s["net_profit"] > 0
        if s["token_in"] == "WETH":
            assert s["net_profit"] == s["net_profit_native"] == s["profit"] - s["gas_cost_wei"]


def test_gas_limit_check_uses_the_swap_estimate():
    pool = V2Pool("WETH", "USDC", 4_000 * E18, 12_000_000 * 10**6)
    state = StateFork([pool])
    payload = {"route": [pool.pool_id], "token_in": "WETH", "amount_in": E18}
    need = estimate_swap_gas([pool])
    assert not _has_enough_gas(state, PendingTx("0x1", "alice", payload, need - 1, GWEI, 0))
    assert _has_enough_gas(state, PendingTx("0x2", "alice", payload, need, GWEI, 0))


def test_native_to_token_rounds_up_at_spot():
    pools = {p.pool_id: p for p in [V2Pool("WETH", "USDC", 4 * E18, 12_000 * 10**6)]}
    # 1 WETH = 3000 USDC (6 decimals): 1 wei of WETH costs 3000e6/1e18 USDC units -> rounds up to 1.
    assert native_to_token(pools, "USDC", 1, "WETH") == 1
    assert native_to_token(pools, "USDC", E18, "WETH") == 3_000 * 10**6
    assert native_to_token(pools, "WETH", 5, "WETH") == 5
    assert native_to_token(pools, "DAI", 5, "WETH") is None


# ---------------------------------------------------------------------------
# Funding rules (strict_balances)
# ---------------------------------------------------------------------------


def _strict(balances):
    return BuilderState.from_pools(demo_market(), balances=balances, strict_balances=True)


def test_strict_insufficient_funds_for_gas_is_invalid():
    tx = _swap_tx("0x1", gas=300_000, cap=100 * GWEI)
    need = 300_000 * 100 * GWEI
    st = _strict({"alice": {ETH: need - 1, "WETH": E18}})
    t = execute_tx(st, tx, 20 * GWEI, 30_000_000)
    assert t.status is Status.INVALID and t.error == "insufficient_funds"
    assert st.nonce("alice") == 0
    st = _strict({"alice": {ETH: need, "WETH": E18}})
    assert execute_tx(st, tx, 20 * GWEI, 30_000_000).status is Status.SUCCESS


def test_strict_swap_above_token_balance_reverts_but_pays_gas():
    st = _strict({"alice": {ETH: 10 * E18, "WETH": E18 - 1}})
    t = execute_tx(st, _swap_tx("0x1"), 20 * GWEI, 30_000_000)
    assert t.status is Status.REVERTED and "exceeds balance" in t.error
    assert st.nonce("alice") == 1
    assert st.fork.balance("alice", "WETH") == E18 - 1
    assert st.fork.balance("alice", ETH) == 10 * E18 - t.gas_used * (20 * GWEI + 2 * GWEI)


def test_strict_bundle_with_unpayable_coinbase_transfer_is_dropped():
    st = _strict({"s": {ETH: E18, "WETH": 2 * E18}})
    root = st.state_root()
    bundle = Bundle("b", (_swap_tx("0x1", sender="s"),), coinbase_transfer_wei=10 * E18, searcher="s")
    r = execute_bundle(st, bundle, 20 * GWEI, 30_000_000, set())
    assert not r.included and r.reason == "coinbase_transfer_exceeds_balance"
    assert st.state_root() == root


def test_default_mode_keeps_ledger_semantics():
    st = BuilderState.from_pools(demo_market())
    assert not st.strict_balances
    assert execute_tx(st, _swap_tx("0x1"), 20 * GWEI, 30_000_000).status is Status.SUCCESS
    assert st.fork.balance("alice", "WETH") == -E18
    assert st.copy().strict_balances is False
    assert BuilderState.from_payload(_strict({}).to_payload()).strict_balances is True


# ---------------------------------------------------------------------------
# Transaction replacement (geth legacypool priceBump = 10)
# ---------------------------------------------------------------------------


def test_replacement_threshold_uses_integer_division_like_geth():
    pool = TxPool()
    old = BuilderTx("0xold", "a", 0, 21_000, 15, 15, TxKind.TRANSFER, {"to": "b", "value": 0})
    assert pool.add(old, 1).accepted
    # threshold = 15 * 110 / 100 = 16 (integer division), so 16 replaces and 15 does not.
    same = BuilderTx("0xsame", "a", 0, 21_000, 15, 15, TxKind.TRANSFER, {"to": "b", "value": 0})
    assert pool.add(same, 1).status == "rejected_replacement_underpriced"
    bumped = BuilderTx("0xnew", "a", 0, 21_000, 16, 16, TxKind.TRANSFER, {"to": "b", "value": 0})
    assert pool.add(bumped, 1).status == "replaced"


# ---------------------------------------------------------------------------
# Uniswap v4 fees and tick bookkeeping
# ---------------------------------------------------------------------------


def _v4(**kw):
    return V4Pool.from_positions("A", "B", 2**96, [Position(-600, 600, 10**21)], fee_pips=3000,
                                 tick_spacing=60, **kw)


def test_v4_protocol_fee_is_capped_at_1000_pips():
    assert _v4(protocol_fee_pips=1000).effective_fee() == 1000 + 3000 - 3
    with pytest.raises(MASEError):
        _v4(protocol_fee_pips=1001)


def test_v4_full_fee_takes_whole_input_and_rejects_exact_output():
    assert compute_swap_step(2**96, 2**96 // 2, 10**18, 1000, 1_000_000) == (2**96, 0, 0, 1000)
    with pytest.raises(MASEError):
        compute_swap_step(2**96, 2**96 // 2, 10**18, -1000, 1_000_000)
    pool = _v4(dynamic_fee=True, hook_fee_pips=1_000_000)
    res = pool.quote_exact_in(10**6, True)
    assert res.amount_out == 0 and res.fee_paid == 10**6


def test_tick_with_zero_net_but_positive_gross_stays_initialized():
    # Positions [-120, 0) and [0, 120) meet at tick 0: net 0, gross 2L. The contracts stop a swap step there.
    pool = ConcentratedPool.from_positions("A", "B", 2**96 + 10**20, [Position(-120, 0, 10**20),
                                                                      Position(0, 120, 10**20)])
    assert pool.ticks[0] == 0 and pool.tick_gross[0] == 2 * 10**20
    assert 0 in pool.tick_array
    # Removing one position keeps tick 0 initialized (gross L), removing the other clears it.
    modify_liquidity(pool, 0, 120, -10**20)
    assert pool.ticks[0] == -10**20 and pool.tick_gross[0] == 10**20  # upper tick of [-120, 0)
    modify_liquidity(pool, -120, 0, -10**20)
    assert 0 not in pool.ticks and 0 not in pool.tick_array


def test_cli_help_renders(capsys):
    """Regression: a literal '%' in a setting's help text used to crash ``--help``."""
    from simulator.__main__ import main
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out
