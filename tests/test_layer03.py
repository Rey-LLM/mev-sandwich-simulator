"""Verification suite for Layer 03: run modes, worker ceiling, state forking,
token dataflow, invariant checks, mempool emulator and the controller."""

from __future__ import annotations

import copy
import json

import pytest

from simulator.__main__ import main as cli_main
from simulator.config import (
    PRESETS,
    ConfigError,
    RunMode,
    SimulationConfig,
    SystemSpec,
    WORKERS_ENV_VAR,
    resolve_worker_count,
    worker_ceiling,
    worker_floor,
)
from simulator.controller import SimulationController, demo_market
from simulator.dataflow import TokenFlowGraph, merge_graphs
from simulator.invariants import (
    InvariantReport,
    bounded_exhaustive_v2,
    check_v2_swap,
    fuzz_pool,
    fuzz_tasks,
    parallel_fuzz,
)
from simulator.mempool_emulator import (
    SEARCHER,
    MempoolEmulator,
    PendingTx,
    block_order,
    estimate_swap_gas,
    generate_mempool,
    geometric_grid,
    search_sandwich,
    simulate_sandwich,
)
from simulator.parallel import WorkerPool, chunked
from simulator.slippage_engine import (
    MASEEngine,
    SwapResult,
    V2Pool,
    evaluate_trade,
    min_amount_out,
    sweep_trade_sizes,
)
from simulator.state_fork import StateFork, SwapReverted

E18 = 10**18
E6 = 10**6
SERIAL = 10**9  # parallel_threshold that forces the inline path


def small_config(mode: str = "brutal", threshold: int = SERIAL, **kw) -> SimulationConfig:
    base = dict(blocks=2, txs_per_block=12, frontrun_grid=16, refine_iterations=8,
                fuzz_iterations=96, parallel_threshold=threshold)
    base.update(kw)
    return SimulationConfig.for_mode(mode, cpus=4, **base)


def v2() -> V2Pool:
    return V2Pool("WETH", "USDC", 4_000 * E18, 12_000_000 * E6)


def victim(pool: V2Pool, amount: int, tolerance_bps: int, token_in: str = "WETH") -> PendingTx:
    quote = evaluate_trade(pool, token_in, amount)
    return PendingTx("0xv", "alice", {
        "route": [pool.pool_id], "token_in": token_in, "amount_in": amount,
        "amount_out_min": min_amount_out(quote.amount_out, tolerance_bps),
    }, gas_limit=200_000, priority_fee_wei=10**9)


# ---------------------------------------------------------------------------
# Config: run modes and the CPU worker ceiling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    ("Moderate Scan", RunMode.MODERATE_SCAN),
    ("moderate", RunMode.MODERATE_SCAN),
    ("moderate_scan", RunMode.MODERATE_SCAN),
    ("Brutal Aggressive Simulation", RunMode.BRUTAL_AGGRESSIVE),
    ("brutal", RunMode.BRUTAL_AGGRESSIVE),
    ("BRUTAL-AGGRESSIVE", RunMode.BRUTAL_AGGRESSIVE),
    (RunMode.BRUTAL_AGGRESSIVE, RunMode.BRUTAL_AGGRESSIVE),
])
def test_run_mode_parse(value, expected):
    assert RunMode.parse(value) is expected


def test_run_mode_rejects_unknown():
    with pytest.raises(ConfigError):
        RunMode.parse("gentle")


@pytest.mark.parametrize("cpus", range(1, 65))
def test_worker_ceiling_never_exceeds_sixty_percent(cpus):
    ceiling = worker_ceiling(cpus)
    assert ceiling >= 1
    if cpus >= 2:
        assert ceiling <= 0.6 * cpus
    assert worker_floor(cpus) <= ceiling
    default = resolve_worker_count(None, cpus)
    assert default == ceiling


@pytest.mark.parametrize("cpus", [2, 4, 5, 8, 10, 12, 16, 20, 32, 64])
def test_default_workers_inside_band(cpus):
    w = resolve_worker_count(None, cpus)
    assert 0.5 * cpus <= w <= 0.6 * cpus


def test_requested_workers_are_clamped(monkeypatch):
    monkeypatch.delenv(WORKERS_ENV_VAR, raising=False)
    assert resolve_worker_count(100, 16) == 9
    assert resolve_worker_count(1, 16) == 8
    assert resolve_worker_count(9, 16) == 9
    monkeypatch.setenv(WORKERS_ENV_VAR, "64")
    assert resolve_worker_count(None, 10) == 6
    monkeypatch.setenv(WORKERS_ENV_VAR, "lots")
    with pytest.raises(ConfigError):
        resolve_worker_count(None, 10)


def test_system_spec_detect():
    spec = SystemSpec.detect()
    assert spec.usable_cpus >= 1
    assert 1 <= spec.worker_floor <= spec.worker_ceiling <= max(1, spec.usable_cpus)
    assert set(spec.to_dict()) >= {"usable_cpus", "worker_ceiling"}


def test_presets_and_overrides():
    mod = SimulationConfig.for_mode("moderate", cpus=8)
    brutal = SimulationConfig.for_mode("brutal", cpus=8)
    assert mod.frontrun_grid == 0 and brutal.frontrun_grid > 0
    assert brutal.blocks > mod.blocks
    assert brutal.fuzz_iterations > mod.fuzz_iterations
    assert mod.workers == brutal.workers == 4
    assert set(PRESETS) == set(RunMode)
    tuned = SimulationConfig.for_mode("brutal", cpus=8, blocks=2)
    assert tuned.blocks == 2 and tuned.to_dict()["mode_label"] == "Brutal Aggressive Simulation"
    with pytest.raises(ConfigError):
        SimulationConfig.for_mode("brutal", cpus=8, max_frontrun_bps=0)


# ---------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------


def _square(x: int) -> int:
    return x * x


def test_chunked_keeps_order_and_balance():
    parts = chunked(list(range(10)), 3)
    assert [len(p) for p in parts] == [4, 3, 3]
    assert [x for p in parts for x in p] == list(range(10))
    assert chunked([], 4) == []
    assert len(chunked([1, 2], 8)) == 2


def test_worker_pool_parallel_matches_inline():
    data = list(range(40))
    with WorkerPool(2, parallel_threshold=1) as wp:
        assert wp.will_parallelize(len(data)) == (wp.workers > 1)
        assert wp.map(_square, data) == [x * x for x in data]
    with WorkerPool(2, parallel_threshold=SERIAL) as wp:
        assert not wp.will_parallelize(len(data))
        assert wp.map(_square, data) == [x * x for x in data]


def test_worker_pool_respects_ceiling():
    wp = WorkerPool(10_000)
    assert wp.workers == resolve_worker_count(None)


# ---------------------------------------------------------------------------
# Slippage engine: parallel batch and sweeps
# ---------------------------------------------------------------------------


def test_parallel_evaluate_batch_matches_serial():
    pools = demo_market()
    engine = MASEEngine(pools)
    txs = generate_mempool(pools, 24, seed=3)
    payloads = [t.payload for t in txs]
    serial = engine.evaluate_batch(payloads)
    parallel = engine.evaluate_batch(payloads, workers=2, parallel_threshold=1)
    assert parallel == serial


def test_parallel_sweep_matches_serial_and_is_monotonic():
    pool = demo_market()[1]
    amounts = [10**k * E6 for k in range(0, 8)] + [3 * 10**k * E6 for k in range(0, 7)]
    amounts.sort()
    serial = sweep_trade_sizes(pool, "USDC", amounts)
    parallel = sweep_trade_sizes(pool, "USDC", amounts, workers=2, parallel_threshold=1)
    assert parallel == serial
    impacts = [r["price_impact"] for r in serial if "error" not in r]
    assert impacts == sorted(impacts)


def test_sweep_reports_errors_inline():
    rows = sweep_trade_sizes(v2(), "WETH", [0, E18])
    assert "error" in rows[0] and rows[1]["amount_out"] > 0


# ---------------------------------------------------------------------------
# State forking (Anvil-style)
# ---------------------------------------------------------------------------


def test_snapshot_revert_restores_state():
    st = StateFork([v2()])
    root0 = st.state_root()
    sid = st.snapshot()
    st.swap("bob", "v2:WETH/USDC", "WETH", 10 * E18)
    st.mine()
    assert st.state_root() != root0
    st.revert(sid)
    assert st.state_root() == root0 and st.block_number == 0
    with pytest.raises(KeyError):
        st.revert(sid)


def test_revert_drops_later_snapshots():
    st = StateFork([v2()])
    a = st.snapshot()
    st.swap("bob", "v2:WETH/USDC", "WETH", E18)
    b = st.snapshot()
    st.revert(a)
    assert b not in st.snapshot_ids and st.snapshot_ids == ()


def test_checkpoint_rolls_back_by_default_and_commits_on_request():
    st = StateFork([v2()])
    root0 = st.state_root()
    with st.checkpoint():
        st.swap("bob", "v2:WETH/USDC", "WETH", E18)
    assert st.state_root() == root0
    with st.checkpoint():
        st.swap("bob", "v2:WETH/USDC", "WETH", E18)
        st.commit_checkpoint()
    assert st.state_root() != root0
    root1 = st.state_root()
    with pytest.raises(RuntimeError):
        with st.checkpoint(commit=True):
            st.swap("bob", "v2:WETH/USDC", "WETH", E18)
            raise RuntimeError("boom")
    assert st.state_root() == root1
    assert st.snapshot_ids == ()


def test_nested_checkpoints():
    st = StateFork([v2()])
    with st.checkpoint(commit=True):
        st.swap("a", "v2:WETH/USDC", "WETH", E18)
        inner_root = st.state_root()
        with st.checkpoint():
            st.swap("a", "v2:WETH/USDC", "WETH", E18)
        assert st.state_root() == inner_root
    assert st.state_root() == inner_root


def test_fork_is_independent():
    st = StateFork([v2()])
    child = st.fork()
    child.swap("bob", "v2:WETH/USDC", "WETH", 50 * E18)
    assert st.pools["v2:WETH/USDC"].reserve0 == 4_000 * E18
    assert child.pools["v2:WETH/USDC"].reserve0 == 4_050 * E18


def test_swap_revert_leaves_state_untouched():
    st = StateFork([v2()])
    root = st.state_root()
    with pytest.raises(SwapReverted):
        st.swap("bob", "v2:WETH/USDC", "WETH", E18, min_amount_out=10**30)
    with pytest.raises(SwapReverted):
        st.swap("bob", "missing", "WETH", E18)
    with pytest.raises(SwapReverted):
        st.swap("bob", "v2:WETH/USDC", "DAI", E18)
    assert st.state_root() == root


def test_swap_updates_balances_and_flow():
    st = StateFork([v2()])
    flow = TokenFlowGraph()
    res = st.swap("bob", "v2:WETH/USDC", "WETH", E18, tx="0x1", flow=flow)
    assert st.balance("bob", "WETH") == -E18
    assert st.balance("bob", "USDC") == res.amount_out
    assert flow.extraction("bob") == {"WETH": -E18, "USDC": res.amount_out}


def test_concentrated_swap_through_state_fork():
    st = StateFork(demo_market())
    pid = "v3:USDC/WETH/3000"
    sqrt0 = st.pools[pid].sqrt_price_x96
    with st.checkpoint():
        st.swap("bob", pid, "USDC", 50_000 * E6)
        assert st.pools[pid].sqrt_price_x96 < sqrt0  # USDC is token0: zero_for_one lowers sqrtP
    assert st.pools[pid].sqrt_price_x96 == sqrt0


# ---------------------------------------------------------------------------
# Token dataflow (Slither-style)
# ---------------------------------------------------------------------------


def sandwich_graph() -> TokenFlowGraph:
    g = TokenFlowGraph()
    g.record_swap("f", "mev", "pool", "WETH", 10, "USDC", 100, "frontrun")
    g.record_swap("v", "alice", "pool", "WETH", 5, "USDC", 40, "victim")
    g.record_swap("b", "mev", "pool", "USDC", 100, "WETH", 12, "backrun")
    return g


def test_net_flows_and_extraction():
    g = sandwich_graph()
    assert g.extraction("mev") == {"WETH": 2}
    assert g.net_flows()["alice"] == {"WETH": -5, "USDC": 40}
    pool = g.net_flows()["pool"]
    assert pool == {"WETH": 3, "USDC": -40}
    assert sum(sum(v for t, v in d.items() if t == "WETH") for d in g.net_flows().values()) == 0


def test_taint_respects_time_order():
    g = sandwich_graph()
    taint = g.taint_from("alice")
    assert taint["pool"] == 2  # alice's transfer is step 2
    assert taint["mev"] == 5  # reached only via the later back-run
    g2 = TokenFlowGraph()
    g2.record("t1", "X", "c", "d", 1)
    g2.record("t2", "X", "a", "c", 1)
    assert "d" not in g2.taint_from("a")  # c -> d happened before a -> c


def test_paths_and_dot():
    g = sandwich_graph()
    paths = g.paths("alice", "mev")
    assert paths and all(p[0].src == "alice" and p[-1].dst == "mev" for p in paths)
    assert all(p[i].step < p[i + 1].step for p in paths for i in range(len(p) - 1))
    assert g.paths("mev", "nobody") == []
    dot = g.to_dot()
    assert dot.startswith("digraph") and '"alice" -> "pool"' in dot
    merged = merge_graphs([g, g])
    assert len(merged) == 2 * len(g)
    with pytest.raises(ValueError):
        g.record("x", "T", "a", "b", -1)


# ---------------------------------------------------------------------------
# Invariants (Halmos-style)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_fuzz_market_pools_hold_invariants(idx):
    pool = demo_market()[idx]
    report = fuzz_pool(pool, 400, seed=idx)
    assert report.ok, report.to_dict()
    assert report.cases > 300 and report.checks > report.cases


def test_bounded_exhaustive_v2_is_clean():
    report = bounded_exhaustive_v2(max_reserve=16, max_amount=16)
    assert report.ok, report.to_dict()
    assert report.cases > 5_000


class _LeakyPool(V2Pool):
    """Deliberately broken pool: pays 10% more than the curve allows."""

    def quote_exact_in(self, amount_in: int, zero_for_one: bool) -> SwapResult:
        res = super().quote_exact_in(amount_in, zero_for_one)
        r_out = self.reserve1 if zero_for_one else self.reserve0
        out = min(r_out - 1, res.amount_out * 11 // 10)
        return SwapResult(res.amount_in, out, res.fee_paid, res.price_before, res.price_after, True, 0)


def test_invariant_checks_catch_a_broken_curve():
    report = InvariantReport()
    pool = _LeakyPool("A", "B", 1_000 * E18, 1_000 * E18)
    check_v2_swap(pool, 100 * E18, True, report)
    names = {v.name for v in report.violations}
    assert "v2.out_le_fee_free" in names and "v2.k_non_decreasing" in names


def test_parallel_fuzz_matches_serial_and_is_worker_independent():
    pools = demo_market()
    serial = parallel_fuzz(pools, 200, seed=5)
    par = parallel_fuzz(pools, 200, seed=5, workers=2, parallel_threshold=1)
    assert serial.to_dict() == par.to_dict()
    assert serial.ok
    tasks = fuzz_tasks(pools, 10, 1, 4)
    assert sum(t[1] for t in tasks) == 30


def test_fuzz_does_not_mutate_input_pool():
    pool = v2()
    before = (pool.reserve0, pool.reserve1)
    fuzz_pool(pool, 50, seed=1)
    assert (pool.reserve0, pool.reserve1) == before


# ---------------------------------------------------------------------------
# Mempool emulator
# ---------------------------------------------------------------------------


def test_generate_mempool_is_deterministic():
    pools = demo_market()
    a = generate_mempool(pools, 20, seed=1, block=3)
    b = generate_mempool(pools, 20, seed=1, block=3)
    c = generate_mempool(pools, 20, seed=2, block=3)
    assert [t.to_dict() for t in a] == [t.to_dict() for t in b]
    assert [t.tx_hash for t in a] != [t.tx_hash for t in c]
    assert len({t.tx_hash for t in a}) == 20
    assert generate_mempool([], 5, seed=1) == []


def test_block_order_by_priority_fee():
    txs = generate_mempool(demo_market(), 30, seed=4)
    ordered = block_order(txs)
    fees = [t.priority_fee_wei for t in ordered]
    assert fees == sorted(fees, reverse=True)


def test_gas_estimate():
    pools = demo_market()
    assert estimate_swap_gas([pools[0]]) == 81_000
    assert estimate_swap_gas([pools[1]], ticks_crossed=2) == 21_000 + 80_000 + 40_000
    assert estimate_swap_gas(pools) == 21_000 + 60_000 + 80_000 + 70_000


def test_scan_is_read_only_and_audits_gas():
    pools = demo_market()
    roots = StateFork(copy.deepcopy(pools)).state_root()
    emu = MempoolEmulator(pools, small_config("moderate"))
    report = emu.scan()
    assert StateFork(pools).state_root() == roots
    summary = report.summary()
    assert summary["txs"] == 12 and summary["errors"] == 0
    assert summary["invariants"]["ok"]
    for row in report.rows:
        assert row["gas_ok"] == (row["gas_limit"] >= row["gas_estimate"])
        assert row["slack_out"] >= 0 and not row["would_revert"]
    assert summary["flow_transfers"] == 2 * summary["quoted"]


def test_scan_parallel_matches_serial():
    pools = demo_market()
    serial = MempoolEmulator(pools, small_config("moderate")).scan()
    par = MempoolEmulator(pools, small_config("moderate", threshold=1)).scan()
    assert par.rows == serial.rows
    assert par.summary() == serial.summary()


def test_scan_reports_bad_payload():
    pools = demo_market()
    bad = PendingTx("0xbad", "eve", {"route": ["nope"], "token_in": "WETH", "amount_in": 1}, 1, 1)
    report = MempoolEmulator(pools, small_config("moderate")).scan([bad])
    assert "error" in report.rows[0] and report.summary()["errors"] == 1


# ---------------------------------------------------------------------------
# Sandwich search
# ---------------------------------------------------------------------------


def test_geometric_grid():
    g = geometric_grid(10, 10_000, 8)
    assert g[0] == 10 and g[-1] == 10_000 and g == sorted(set(g))
    assert geometric_grid(5, 1, 4) == [] and geometric_grid(1, 9, 1) == [9]


def test_simulate_sandwich_does_not_mutate_pool():
    pool = v2()
    tx = victim(pool, 50 * E18, 300)
    before = (pool.reserve0, pool.reserve1)
    assert simulate_sandwich(pool, tx, 20 * E18) is not None
    assert (pool.reserve0, pool.reserve1) == before


def test_loose_victim_gets_sandwiched_within_their_min_out():
    pool = v2()
    tx = victim(pool, 50 * E18, 300)
    plan = search_sandwich(pool, tx, grid=32, refine=16, max_frontrun_bps=2_000)
    assert plan is not None and plan.profit > 0
    assert plan.victim_out >= tx.amount_out_min
    unsandwiched = evaluate_trade(pool, "WETH", tx.amount_in).amount_out
    assert plan.victim_out < unsandwiched
    # Refinement never does worse than the plain grid.
    coarse = search_sandwich(pool, tx, grid=32, refine=0, max_frontrun_bps=2_000)
    assert plan.profit >= coarse.profit


def test_tight_victim_is_not_profitable():
    pool = v2()
    tx = victim(pool, E18, 1)
    assert search_sandwich(pool, tx, grid=32, refine=16, max_frontrun_bps=2_000) is None


def test_multihop_victims_are_skipped():
    pool = v2()
    tx = PendingTx("0xm", "a", {"route": ["x", "y"], "token_in": "WETH", "amount_in": 1}, 1, 1)
    assert search_sandwich(pool, tx, 8, 4, 2_000) is None


# ---------------------------------------------------------------------------
# Multi-block aggressive run
# ---------------------------------------------------------------------------


def test_run_blocks_extracts_value_and_stays_consistent():
    report = MempoolEmulator(demo_market(), small_config()).run_blocks()
    summary = report.summary()
    assert len(summary["blocks"]) == 2
    assert summary["sandwiches_executed"] > 0
    assert all(v > 0 for v in summary["extracted"].values())
    assert summary["dataflow_consistent"]
    assert summary["invariants"]["ok"], summary["invariants"]
    assert report.state.block_number == 2
    for block in report.blocks:
        for s in block.sandwiches:
            assert s["victim_to_searcher_path"], s
            assert s["victim_to_searcher_path"][-1].endswith(f"->{SEARCHER}:{s['token_in']}")


def test_run_blocks_parallel_matches_serial():
    serial = MempoolEmulator(demo_market(), small_config()).run_blocks().summary()
    par = MempoolEmulator(demo_market(), small_config(threshold=1)).run_blocks().summary()
    assert par == serial


def test_run_blocks_does_not_mutate_input_pools():
    pools = demo_market()
    root = StateFork(pools).state_root()
    MempoolEmulator(pools, small_config(blocks=1)).run_blocks()
    assert StateFork(pools).state_root() == root


# ---------------------------------------------------------------------------
# Controller and CLI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["moderate", "brutal"])
def test_controller_runs_each_mode(mode):
    out = SimulationController(small_config(mode)).run()
    assert out["mode"] == RunMode.parse(mode).value
    assert out["result"]["invariants"]["ok"]
    assert out["config"]["workers"] <= out["system"]["worker_ceiling"] or out["system"]["usable_cpus"] != 4
    json.dumps(out)


def test_cli(capsys):
    rc = cli_main(["--mode", "moderate", "--txs", "6", "--fuzz", "32", "--seed", "9"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["mode_label"] == "Moderate Scan" and data["result"]["txs"] == 6
