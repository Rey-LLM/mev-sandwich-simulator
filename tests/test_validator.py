"""Layer 10: the core integrity validator.

Every registered rule is triggered by a crafted scenario (and a meta test
fails if a rule is added without one), the clean baseline stays clean, and
the validator is exercised concurrently and as the controller's pre-flight.
"""

import json
import random
import threading
import time
from types import SimpleNamespace

import pytest

from simulator import validator as V
from simulator.config import ConfigError, RunMode, SimulationConfig, detect_cpu_count
from simulator.controller import SimulationController, demo_market
from simulator.reinforcement_engine import Action
from simulator.slippage_engine import (
    MAX_SQRT_RATIO,
    MAX_TICK,
    ConcentratedPool,
    Position,
    V2Pool,
    get_sqrt_ratio_at_tick,
    get_tick_at_sqrt_ratio,
    price_to_sqrt_price_x96,
)
from simulator.validator import (
    FAMILIES,
    RULES,
    AuditReport,
    Collector,
    Finding,
    IntegrityError,
    IntegrityValidator,
    Severity,
    audit,
    check_magic_constants,
    check_pool,
    check_pool_ticks,
    check_pools,
    check_rl_signals,
    check_signals,
    check_swap_step,
    check_tick_math,
    clear_scan_cache,
    exact_sqrt_ratio,
    package_sources,
    rule_catalog,
    sample_ticks,
    scan_paths,
    scan_source,
    tick_magic_table,
)

E18, E6 = 10**18, 10**6
CPUS = detect_cpu_count()


def rules_of(findings):
    return {f.rule for f in findings}


def v2():
    return V2Pool("WETH", "USDC", 4_000 * E18, 12_000_000 * E6)


def cl():
    return demo_market()[1]


def v4():
    return demo_market()[2]


def state(pools, native="WETH"):
    col = Collector()
    check_pools(pools, native, col)
    return col.findings


def signals(cfg, cpus=4):
    col = Collector()
    check_signals(cfg, col, cpus)
    return col.findings


def corrupt(cfg, **changes):
    for k, v in changes.items():
        object.__setattr__(cfg, k, v)
    return cfg


def brutal(**kw):
    return SimulationConfig.for_mode("brutal", cpus=4, **kw)


def ticks(**kw):
    col = Collector()
    check_tick_math(col, samples=kw.pop("samples", 8), **kw)
    return col.findings


def code(src):
    return scan_source(src, "snippet.py")


def mutated_cl(fn):
    p = cl()
    fn(p)
    return state([p])


class LeakyV2(V2Pool):
    def quote_exact_in(self, amount_in, zero_for_one):
        self.reserve0 += 1  # quoting must never touch state
        return super().quote_exact_in(amount_in, zero_for_one)


def _first_tick(p):
    return sorted(p.ticks)[0]


def _swap_step_off_by_one(monkeypatch):
    real = V.compute_swap_step

    def broken(*args):
        sn, a_in, a_out, fee = real(*args)
        return sn, a_in - 1, a_out, fee  # rounds the input in the trader's favour

    monkeypatch.setattr(V, "compute_swap_step", broken)
    col = Collector()
    check_swap_step(col, get_sqrt_ratio_at_tick(100), get_sqrt_ratio_at_tick(-100), 10**20, 10**18, 3000)
    return col.findings


def _delta_too_wide(monkeypatch):
    real = V.get_amount0_delta
    monkeypatch.setattr(V, "get_amount0_delta", lambda a, b, liq, up: real(a, b, liq, up) + (2 if up else 0))
    return ticks()


def _price_not_floored(monkeypatch):
    real = V.price_to_sqrt_price_x96
    monkeypatch.setattr(V, "price_to_sqrt_price_x96", lambda p: real(p) + 1)
    return ticks()


def _fork_revert_broken(monkeypatch):
    from simulator import state_fork

    monkeypatch.setattr(state_fork.StateFork, "revert", lambda self, sid: None)
    return state(demo_market())


def _crossing_desync(p):
    p._tick_array.remove(sorted(p.ticks)[-1])  # the swap walks the array, the map still has the tick
    col = Collector()
    check_pool_ticks(p, col)
    return col.findings


def _nonstandard_spacing():
    p = ConcentratedPool.from_positions("AAA", "BBB", price_to_sqrt_price_x96(1), [Position(-70, 70, 10**18)],
                                        fee_pips=3000, tick_spacing=7)
    col = Collector()
    check_pool_ticks(p, col)
    return col.findings


def _rl(monkeypatch, target, attr, replacement, **kw):
    from simulator import reinforcement_engine as rl

    monkeypatch.setattr(getattr(rl, target), attr, replacement)
    col = Collector()
    check_rl_signals(col, "volatile", **kw)
    return col.findings


def _bins(**edges):
    base = dict(reserve_bps=(-20, 20), gas_bps=(1, 2), delay_ms=(100, 200, 300), victim_bps=(10, 50),
                rivals=(1, 2))
    base.update(edges)
    return SimpleNamespace(**base)


def _rl_plain(**kw):
    col = Collector()
    check_rl_signals(col, kw.pop("regime", None), **kw)
    return col.findings


TRIGGERS = {
    # state.*
    "state.duplicate_pool_id": lambda mp, tmp: state([v2(), v2()]),
    "state.identical_tokens": lambda mp, tmp: state([corrupt_pool(v2(), token1="WETH")]),
    "state.v2_reserves": lambda mp, tmp: state([corrupt_pool(v2(), reserve0=0)]),
    "state.v2_reserve_bounds": lambda mp, tmp: state([corrupt_pool(v2(), reserve1=1 << 112)]),
    "state.v2_fee": lambda mp, tmp: state([corrupt_pool(v2(), fee_bps=10_000)]),
    "state.cl_sqrt_price_range": lambda mp, tmp: state([corrupt_pool(cl(), sqrt_price_x96=MAX_SQRT_RATIO)]),
    "state.cl_tick_cache": lambda mp, tmp: mutated_cl(lambda p: setattr(p, "tick", p.tick + 120)),
    "state.cl_liquidity_bounds": lambda mp, tmp: state([corrupt_pool(cl(), liquidity=1 << 128)]),
    "state.cl_tick_alignment": lambda mp, tmp: mutated_cl(lambda p: p.ticks.update({7: 0})),
    "state.cl_tick_range": lambda mp, tmp: mutated_cl(lambda p: p.ticks.update({MAX_TICK + 60: 0})),
    "state.cl_tick_array_sync": lambda mp, tmp: mutated_cl(lambda p: p._tick_array.pop()),
    "state.cl_net_liquidity_sum": lambda mp, tmp: mutated_cl(
        lambda p: p.ticks.update({_first_tick(p): p.ticks[_first_tick(p)] + 1})),
    "state.cl_net_liquidity_prefix": lambda mp, tmp: mutated_cl(
        lambda p: p.ticks.update({t: -n for t, n in p.ticks.items()})),
    "state.cl_active_liquidity": lambda mp, tmp: mutated_cl(lambda p: setattr(p, "liquidity", p.liquidity + 1)),
    "state.cl_fee": lambda mp, tmp: state([corrupt_pool(cl(), fee_pips=1_000_000)]),
    "state.v4_fee": lambda mp, tmp: state([corrupt_pool(v4(), hook_fee_pips=2_000_000)]),
    "state.native_token_missing": lambda mp, tmp: state(demo_market(), native="XYZ"),
    "state.quote_purity": lambda mp, tmp: state([LeakyV2("WETH", "USDC", 4_000 * E18, 12_000_000 * E6)]),
    "state.fork_roundtrip": lambda mp, tmp: _fork_revert_broken(mp),
    "state.no_pools": lambda mp, tmp: state([]),
    # tick.*
    "tick.magic_constants": lambda mp, tmp: _magic(),
    "tick.anchor": lambda mp, tmp: ticks(ratio_fn=lambda t: get_sqrt_ratio_at_tick(t) + (t == 0)),
    "tick.round_trip": lambda mp, tmp: ticks(tick_fn=lambda r: get_tick_at_sqrt_ratio(r) + 1),
    "tick.monotonic": lambda mp, tmp: ticks(
        ratio_fn=lambda t: get_sqrt_ratio_at_tick(t - 1) if t == 5 else get_sqrt_ratio_at_tick(t)),
    "tick.precision": lambda mp, tmp: ticks(
        ratio_fn=lambda t: get_sqrt_ratio_at_tick(t) + (10**6 if t < 0 else 0)),
    "tick.delta_rounding": lambda mp, tmp: _delta_too_wide(mp),
    "tick.step_conservation": lambda mp, tmp: _swap_step_off_by_one(mp),
    "tick.price_floor": lambda mp, tmp: _price_not_floored(mp),
    "tick.crossing_liquidity": lambda mp, tmp: _crossing_desync(cl()),
    "tick.nonstandard_spacing": lambda mp, tmp: _nonstandard_spacing(),
    # signal.*
    "signal.bps_range": lambda mp, tmp: signals(corrupt(brutal(), bribe_bps=20_000)),
    "signal.gas_bounds": lambda mp, tmp: signals(corrupt(brutal(), base_fee_wei=0)),
    "signal.gas_capacity": lambda mp, tmp: signals(brutal(block_gas_limit=100_000)),
    "signal.counts": lambda mp, tmp: signals(corrupt(brutal(), blocks=0)),
    "signal.workers_range": lambda mp, tmp: signals(corrupt(brutal(), workers=0)),
    "signal.workers_clamped": lambda mp, tmp: signals(corrupt(brutal(), workers=64)),
    "signal.cpu_policy": lambda mp, tmp: signals(corrupt(brutal(), cpu_policy="turbo")),
    "signal.mode_policy": lambda mp, tmp: signals(SimulationConfig.for_mode("max", cpus=4, cpu_policy="capped")),
    "signal.start_method": lambda mp, tmp: signals(corrupt(brutal(), start_method="thread")),
    "signal.parallel_threshold": lambda mp, tmp: signals(corrupt(brutal(), parallel_threshold=0)),
    "signal.shard_density": lambda mp, tmp: signals(corrupt(brutal(), shard_density=0)),
    "signal.parallel_headroom": lambda mp, tmp: signals(SimulationConfig.for_mode("brutal", cpus=64), cpus=64),
    "signal.pg_paths": lambda mp, tmp: signals(corrupt(brutal(), pg_batch=0)),
    "signal.pg_paths_ignored": lambda mp, tmp: signals(SimulationConfig.for_mode("moderate", cpus=4, pg_paths=8)),
    "signal.rl_regime": lambda mp, tmp: signals(corrupt(brutal(), rl_regime="chaos")),
    "signal.rl_bins": lambda mp, tmp: _rl_plain(bins=_bins(reserve_bps=(5, 1))),
    "signal.rl_bin_coverage": lambda mp, tmp: _rl_plain(regime="calm", bins=_bins(victim_bps=(10**9,))),
    "signal.action_grid": lambda mp, tmp: _rl_plain(actions=[Action(0, 0), Action(0, 0)]),
    "signal.softmax_stability": lambda mp, tmp: _rl(mp, "PolicyGradientAgent", "probs",
                                                    lambda self, key: [float("nan")] * len(self.actions)),
    "signal.reward_clip": lambda mp, tmp: _rl(mp, "RewardScaler", "normalize", lambda self, r: 1e9),
    "signal.learning_schedule": lambda mp, tmp: _rl(mp, "LearningSchedule", "lr", lambda self, v: 2.0),
    "signal.trace_path": lambda mp, tmp: signals(corrupt(brutal(), trace=True, trace_path=str(tmp))),
    "signal.audit_settings": lambda mp, tmp: signals(corrupt(brutal(), audit_fail_on="fatal")),
    "signal.native_token": lambda mp, tmp: signals(corrupt(brutal(), native_token="  ")),
    # code.*
    "code.syntax": lambda mp, tmp: code("def broken(:\n    pass\n"),
    "code.bare_except": lambda mp, tmp: code("try:\n    x = 1\nexcept:\n    raise\n"),
    "code.silent_except": lambda mp, tmp: code("try:\n    x = 1\nexcept Exception:\n    pass\n"),
    "code.mutable_default": lambda mp, tmp: code("def f(a=[]):\n    return a\n"),
    "code.unpicklable_task": lambda mp, tmp: code("def run(pool):\n    return pool.map(lambda x: x, [1])\n"),
    "code.global_random": lambda mp, tmp: code("import random\nx = random.random()\n"),
    "code.time_seeded_rng": lambda mp, tmp: code("import random\nimport time\nrng = random.Random(time.time())\n"),
    "code.ungoverned_pool": lambda mp, tmp: code("import multiprocessing\np = multiprocessing.Pool(2)\n"),
    "code.ungoverned_process": lambda mp, tmp: code(
        "import multiprocessing\nmultiprocessing.Process(target=print).start()\n"),
    "code.all_exports": lambda mp, tmp: code("__all__ = ['missing']\n"),
    "code.all_duplicates": lambda mp, tmp: code("def a():\n    pass\n__all__ = ['a', 'a']\n"),
    "code.duplicate_dict_key": lambda mp, tmp: code("d = {'a': 1, 'a': 2}\n"),
    "code.literal_identity": lambda mp, tmp: code("x = 3\nif x is 1:\n    pass\n"),
    "code.division_by_zero": lambda mp, tmp: code("x = 1 / 0\n"),
    "code.float_truncation": lambda mp, tmp: code("x = int(a / b)\n"),
    "code.float_equality": lambda mp, tmp: code("if x == 0.5:\n    pass\n"),
    "code.none_equality": lambda mp, tmp: code("if x == None:\n    pass\n"),
    "code.unreachable": lambda mp, tmp: code("def f():\n    return 1\n    x = 2\n"),
    "code.self_assign": lambda mp, tmp: code("x = 1\nx = x\n"),
    "code.open_encoding": lambda mp, tmp: code("f = open('a.txt')\n"),
    "code.bad_pragma": lambda mp, tmp: code("x = 1  # audit: allow code.nonexistent_rule\n"),
}


def corrupt_pool(pool, **changes):
    for k, v in changes.items():
        setattr(pool, k, v)
    return pool


def _magic():
    mask, magic = tick_magic_table()[3]
    col = Collector()
    check_magic_constants(col, table=[(mask, magic + 2)])
    return col.findings


# ---------------------------------------------------------------------------
# Every rule has a trigger
# ---------------------------------------------------------------------------


def test_every_registered_rule_has_a_trigger_scenario():
    assert set(TRIGGERS) == set(RULES)


@pytest.mark.parametrize("rule", sorted(TRIGGERS))
def test_rule_is_triggered(rule, monkeypatch, tmp_path):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # "is" with a literal
        findings = TRIGGERS[rule](monkeypatch, tmp_path)
    hits = [f for f in findings if f.rule == rule]
    assert hits, f"{rule} not triggered; got {sorted(rules_of(findings))}"
    assert all(f.severity is RULES[rule].severity for f in hits)
    assert all(f.message and f.subject for f in hits)


# ---------------------------------------------------------------------------
# The clean baseline
# ---------------------------------------------------------------------------


def test_demo_market_state_is_clean():
    assert state(demo_market()) == []


@pytest.mark.parametrize("mode", [m.short for m in RunMode])
def test_default_configs_have_no_signal_findings(mode):
    cfg = SimulationConfig.for_mode(mode)
    assert signals(cfg, cpus=CPUS) == []


def test_package_source_scan_is_clean_with_documented_suppressions():
    clear_scan_cache()
    col = Collector()
    files = scan_paths(None, col)
    assert files == len(package_sources()) >= 18
    assert col.findings == []
    assert col.suppressed == 4  # the four random size draws, each annotated with its reason


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_real_tick_math_is_clean(seed):
    assert ticks(samples=400, seed=seed) == []


def test_crashing_tick_math_is_reported_not_raised():
    def explode(t):
        if t == 7:
            raise OverflowError("boom")
        return get_sqrt_ratio_at_tick(t)

    found = ticks(ratio_fn=explode)
    crash = [f for f in found if "OverflowError" in f.message]
    assert crash and {f.rule for f in crash} <= {"tick.round_trip", "tick.delta_rounding", "tick.step_conservation"}


def test_magic_constants_are_round_to_nearest():
    col = Collector()
    check_magic_constants(col)
    assert col.findings == [] and col.checks["tick"] == len(tick_magic_table()) >= 19


def test_tick_samples_include_the_anchors():
    s = sample_ticks(10, seed=3)
    assert s == sorted(set(s)) and {-887272, 0, 887272} <= set(s) and len(s) >= 10
    assert sample_ticks(10, seed=3) == s and sample_ticks(10, seed=4) != s


def test_exact_ratio_matches_the_known_anchors():
    assert abs(exact_sqrt_ratio(0) - 2**96) < 1e-60
    assert abs(exact_sqrt_ratio(-887272) - 4295128739) < 1


def test_pool_sweeps_are_clean_on_the_demo_market():
    for pool in demo_market():
        if isinstance(pool, ConcentratedPool):
            col = Collector()
            check_pool_ticks(pool, col)
            assert col.findings == [], pool.pool_id


def test_invariants_hold_after_random_real_swaps():
    rng = random.Random(5)
    pools = demo_market()
    for _ in range(60):
        pool = rng.choice(pools[1:])
        zfo = rng.random() < 0.5
        amount = rng.randint(1, 10**21) if zfo else rng.randint(1, 10**10)
        try:
            pool.apply_swap(amount, zfo)
        except Exception:
            continue
    assert state(pools) == []
    for pool in pools[1:]:
        col = Collector()
        check_pool_ticks(pool, col)
        assert col.findings == []


def test_boundary_state_after_a_downward_cross_is_accepted():
    p = cl()
    boundary = sorted(p.ticks)[1]
    p.sqrt_price_x96 = get_sqrt_ratio_at_tick(boundary)
    p.tick = boundary - 1  # what a downward crossing leaves behind
    p.liquidity = sum(n for t, n in p.ticks.items() if t <= p.tick)
    assert "state.cl_tick_cache" not in rules_of(state([p]))


# ---------------------------------------------------------------------------
# Reports, severities and the validator object
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [("error", Severity.ERROR), ("WARNING", Severity.WARNING),
                                           (" info ", Severity.INFO), (30, Severity.ERROR),
                                           (Severity.INFO, Severity.INFO)])
def test_severity_parse(text, expected):
    assert Severity.parse(text) is expected


def test_severity_parse_rejects_unknown():
    with pytest.raises(ConfigError):
        Severity.parse("fatal")


def _report(*sevs, fail_on="error"):
    findings = [Finding(f"code.r{i}", s, f"s{i}", "m") for i, s in enumerate(sevs)]
    return AuditReport(findings, {"code": 3}, 0.1, Severity.parse(fail_on))


@pytest.mark.parametrize("fail_on,blocking", [("error", 1), ("warning", 2), ("info", 3)])
def test_fail_on_controls_what_blocks(fail_on, blocking):
    rep = _report(Severity.ERROR, Severity.WARNING, Severity.INFO, fail_on=fail_on)
    assert len(rep.blocking()) == blocking and rep.ok is False


def test_report_without_blocking_findings_is_ok():
    rep = _report(Severity.WARNING, Severity.INFO)
    assert rep.ok and rep.counts() == {"error": 0, "warning": 1, "info": 1}


def test_report_to_dict_orders_and_truncates():
    rep = _report(Severity.INFO, Severity.ERROR, Severity.WARNING)
    d = rep.to_dict(max_findings=2)
    assert [f["severity"] for f in d["findings"]] == ["error", "warning"]
    assert d["finding_count"] == 3 and d["blocking"] == 1 and d["checks"] == 3
    json.dumps(d)


def test_integrity_error_names_the_blocking_rules():
    rep = _report(Severity.ERROR, Severity.ERROR)
    err = IntegrityError(rep)
    assert err.report is rep and "code.r0" in str(err) and "2 blocking" in str(err)


def test_workload_note_never_blocks_even_at_info_level():
    rep = IntegrityValidator.for_config(SimulationConfig.for_mode("max", audit_fail_on="info")).run()
    assert rep.ok, rep.to_dict()
    assert any("MAX Brutal Simulation" in n for n in rep.notes)


def test_validator_family_selection():
    rep = IntegrityValidator(SimulationConfig.for_mode("brutal"), families=("signal",)).run()
    assert rep.families == ("signal",) and set(rep.checks_by_family) == {"signal"}
    assert rep.pools_checked == 0 and rep.files_scanned == 0
    with pytest.raises(ConfigError):
        IntegrityValidator(families=("state", "vibes"))
    with pytest.raises(ConfigError):
        IntegrityValidator(tick_samples=-1)


def test_for_config_follows_the_audit_settings():
    cfg = SimulationConfig.for_mode("moderate", audit_code_scan=False, audit_tick_samples=16,
                                    audit_fail_on="warning")
    v = IntegrityValidator.for_config(cfg)
    assert (v.code_scan, v.tick_samples, v.fail_on) == (False, 16, Severity.WARNING)
    rep = v.run()
    assert rep.files_scanned == 0 and rep.pools_checked == 3 and rep.ok


def test_audit_shorthand_and_full_run_check_every_family():
    rep = audit(SimulationConfig.for_mode("brutal"))
    assert rep.ok and set(rep.checks_by_family) == set(FAMILIES)
    assert rep.checks > 10_000 and rep.files_scanned >= 18


def test_rule_catalog_lists_every_rule_once():
    cat = rule_catalog()
    assert [r["rule"] for r in cat] == sorted(RULES)
    assert {r["family"] for r in cat} == set(FAMILIES)
    assert {r["severity"] for r in cat} <= {"error", "warning", "info"}


# ---------------------------------------------------------------------------
# Code scan details: suppressions, single reports, caching
# ---------------------------------------------------------------------------


def test_trailing_pragma_suppresses_only_the_named_rule():
    col = Collector()
    scan_source("x = int(a / b)  # audit: allow code.float_truncation (sampled size)\n", "m.py", col=col)
    assert col.findings == [] and col.suppressed == 1
    other = scan_source("x = int(a / b)  # audit: allow code.float_equality\n", "m.py")
    assert rules_of(other) == {"code.float_truncation"}


def test_standalone_pragma_covers_the_next_line_only():
    src = ("# audit: allow code.none_equality, code.float_equality\n"
           "ok = x == None or y == 0.5\n"
           "bad = x == None\n")
    found = scan_source(src, "m.py")
    assert [(f.rule, f.subject) for f in found] == [("code.none_equality", "m.py:3")]


def test_unreachable_code_is_reported_once_at_every_level():
    src = ("raise SystemExit\nx = 1\n"
           "def f():\n    return 1\n    y = 2\n"
           "for i in range(3):\n    break\n    z = 3\n")
    found = [f for f in scan_source(src, "m.py") if f.rule == "code.unreachable"]
    assert sorted(f.subject for f in found) == ["m.py:2", "m.py:5", "m.py:8"]


def test_governed_module_may_create_pools():
    src = "import multiprocessing\np = multiprocessing.get_context('spawn').Pool(2)\n"
    assert "code.ungoverned_pool" in rules_of(scan_source(src, "other.py"))
    assert scan_source(src, "parallel.py") == []


def test_open_in_binary_mode_or_with_encoding_is_fine():
    assert scan_source("a = open('x', 'rb')\nb = open('y', encoding='utf-8')\n", "m.py") == []


def test_scan_cache_tracks_file_changes(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text("x = 1\n", encoding="utf-8")
    clear_scan_cache()
    col = Collector()
    assert scan_paths([path], col) == 1 and col.findings == []
    time.sleep(0.01)
    path.write_text("x = 1\nd = {'k': 1, 'k': 2}\n", encoding="utf-8")
    col = Collector()
    scan_paths([path], col)
    assert rules_of(col.findings) == {"code.duplicate_dict_key"}


def test_missing_source_file_is_reported_without_blocking(tmp_path):
    col = Collector()
    assert scan_paths([tmp_path / "gone.py"], col) == 0
    assert [(f.rule, f.severity) for f in col.findings] == [("code.syntax", Severity.INFO)]


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_validators_agree():
    clear_scan_cache()
    cfg = SimulationConfig.for_mode("max")
    reports, errors = [], []

    def worker():
        try:
            reports.append(IntegrityValidator.for_config(cfg, tick_samples=64).run().to_dict())
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert not errors and len(reports) == 8
    first = {k: v for k, v in reports[0].items() if k != "elapsed_s"}
    for rep in reports[1:]:
        assert {k: v for k, v in rep.items() if k != "elapsed_s"} == first
    assert first["ok"]


def test_concurrent_scans_of_corrupted_sources_find_the_same_flaws(tmp_path):
    paths = []
    for i in range(6):
        p = tmp_path / f"m{i}.py"
        p.write_text(f"def f(a={{}}):\n    return a\nx{i} = x{i}\n", encoding="utf-8")
        paths.append(p)
    clear_scan_cache()
    results = []

    def worker():
        col = Collector()
        scan_paths(paths, col)
        results.append(sorted((f.rule, f.subject) for f in col.findings))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert len(results) == 8 and all(r == results[0] for r in results)
    assert len(results[0]) == 12


# ---------------------------------------------------------------------------
# Pre-flight in the controller and the CLI
# ---------------------------------------------------------------------------


def test_controller_refuses_to_simulate_after_a_blocking_audit(monkeypatch):
    import simulator.controller as ctl

    def no_simulation(*a, **k):
        raise AssertionError("a simulation loop started after a failed audit")

    monkeypatch.setattr(ctl, "MempoolEmulator", no_simulation)
    cfg = SimulationConfig.for_mode("brutal", block_gas_limit=100_000, audit_fail_on="warning")
    controller = SimulationController(cfg)
    with pytest.raises(IntegrityError) as exc:
        controller.run()
    assert "signal.gas_capacity" in str(exc.value)
    assert controller.last_audit is exc.value.report and not controller.last_audit.ok


def test_controller_rejects_a_corrupted_market():
    broken = corrupt_pool(cl(), liquidity=cl().liquidity + 5)
    with pytest.raises(IntegrityError) as exc:
        SimulationController(SimulationConfig.for_mode("moderate", txs_per_block=4, fuzz_iterations=4)).run(
            pools=[v2(), broken])
    assert "state.cl_active_liquidity" in exc.value.report.rules_triggered()


def test_controller_skips_the_audit_when_disabled(monkeypatch):
    monkeypatch.setattr(IntegrityValidator, "run", lambda self: (_ for _ in ()).throw(AssertionError("audited")))
    cfg = SimulationConfig.for_mode("moderate", txs_per_block=4, fuzz_iterations=4, audit=False)
    out = SimulationController(cfg).run()
    assert out["audit"] == {"enabled": False}


def test_cli_audit_only(capsys):
    from simulator.__main__ import main

    assert main(["--audit-only", "--mode", "max"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["mode_label"] == "MAX Brutal Simulation" and data["audit"]["ok"]
    rc = main(["--audit-only", "--mode", "brutal", "--gas-limit", "100000", "--audit-fail-on", "warning"])
    data = json.loads(capsys.readouterr().out)
    assert rc == 2 and not data["audit"]["ok"] and "signal.gas_capacity" in data["audit"]["by_rule"]


def test_cli_run_blocked_by_the_audit_exits_2(capsys):
    from simulator.__main__ import main

    rc = main(["--mode", "brutal", "--blocks", "1", "--txs", "4", "--gas-limit", "100000",
               "--audit-fail-on", "warning"])
    captured = capsys.readouterr()
    assert rc == 2 and "integrity audit failed" in captured.err
    assert json.loads(captured.out)["audit"]["blocking"] >= 1


def test_cli_audit_rules(capsys):
    from simulator.__main__ import main

    assert main(["--audit-rules"]) == 0
    assert [r["rule"] for r in json.loads(capsys.readouterr().out)] == sorted(RULES)
