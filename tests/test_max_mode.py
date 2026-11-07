"""Layer 09: run matrices, the full-core CPU policy, pool telemetry under load and the path fuzzer."""

import json
import math
import threading

import pytest

from simulator.config import (
    CPU_CEILING_FRACTION,
    MAX_PG_PATHS,
    MAX_SHARD_DENSITY,
    MODE_KEYS,
    PRESETS,
    ConfigError,
    CpuPolicy,
    RunMode,
    SimulationConfig,
    SystemSpec,
    detect_cpu_count,
    resolve_worker_count,
    worker_ceiling,
    worker_floor,
)
from simulator.controller import SimulationController, invariants_ok, trace_logging
from simulator.parallel import WorkerPool, cpu_burn, rendezvous_probe, saturation_probe
from simulator.path_fuzzer import (
    PATH_REGIMES,
    PATH_RULES,
    PolicyPathFuzzer,
    make_path_spec,
    make_path_specs,
    merge_partials,
    run_path,
    softmax,
)
from simulator.reinforcement_engine import ActionTable, NoisyBanditEnvironment

CPUS = detect_cpu_count()
needs_two_cpus = pytest.mark.skipif(CPUS < 2, reason="needs at least two usable CPUs")
TIMING_KEYS = {"elapsed_s", "cpu_s", "paths_per_s", "steps_per_s", "parallel_efficiency", "workers", "parallel",
               "distinct_worker_pids", "wall_s"}


def strip_timing(obj):
    if isinstance(obj, dict):
        return {k: strip_timing(v) for k, v in obj.items() if k not in TIMING_KEYS and not k.endswith("_per_s")}
    if isinstance(obj, list):
        return [strip_timing(v) for v in obj]
    return obj


def max_cfg(**kw):
    base = dict(blocks=1, txs_per_block=8, frontrun_grid=8, refine_iterations=4, fuzz_iterations=32,
                layout_fuzz_iterations=16, rl_rounds=1, rl_steps_per_shard=6, rl_eval_steps=8, pg_paths=48,
                pg_path_steps=4, pg_batch=8, shard_density=2, parallel_threshold=1, agent_bots=12,
                agent_generations=2, agent_heat_size=4, agent_heat_slots=2, pattern_synthetic=96, pattern_limit=64)
    base.update(kw)
    return SimulationConfig.for_mode("max", **base)


@pytest.fixture(scope="module")
def full_pool():
    with WorkerPool(policy="full", parallel_threshold=1, instrument=True) as wp:
        yield wp


# ---------------------------------------------------------------------------
# Run matrices
# ---------------------------------------------------------------------------


def test_three_run_matrices_with_exact_labels():
    assert [m.label for m in RunMode] == ["Moderate Scan", "Brutal Aggressive Simulation", "MAX Brutal Simulation"]
    assert [m.short for m in RunMode] == ["moderate", "brutal", "max"]
    assert [m.aggressive for m in RunMode] == [False, True, True]


@pytest.mark.parametrize("text", ["max", "MAX", "max brutal", "MAX Brutal Simulation", "max_brutal", "max-brutal",
                                  "maximum", "max mode", RunMode.MAX_BRUTAL])
def test_max_mode_aliases(text):
    assert RunMode.parse(text) is RunMode.MAX_BRUTAL


def test_presets_are_rectangular_and_cover_every_mode():
    assert set(PRESETS) == set(RunMode)
    for mode, preset in PRESETS.items():
        assert tuple(preset) == MODE_KEYS, mode


def test_max_preset_is_the_heaviest_matrix():
    brutal, mx = PRESETS[RunMode.BRUTAL_AGGRESSIVE], PRESETS[RunMode.MAX_BRUTAL]
    for key in ("blocks", "txs_per_block", "frontrun_grid", "refine_iterations", "fuzz_iterations",
                "layout_fuzz_iterations", "router_max_hops", "rl_rounds", "rl_eval_steps", "shard_density",
                "pg_paths"):
        assert mx[key] >= brutal[key], key
    assert mx["cpu_policy"] == "full" and brutal["cpu_policy"] == PRESETS[RunMode.MODERATE_SCAN]["cpu_policy"] == "capped"
    assert mx["pg_paths"] >= 1_000, "MAX fuzzes thousands of policy-gradient paths"
    assert mx["shard_density"] > 1


def test_moderate_scan_is_passive():
    m = PRESETS[RunMode.MODERATE_SCAN]
    assert m["frontrun_grid"] == m["refine_iterations"] == m["layout_fuzz_iterations"] == 0
    assert m["rl_rounds"] == m["pg_paths"] == m["private_flow_bps"] == 0


# ---------------------------------------------------------------------------
# CPU policy and worker resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [("capped", CpuPolicy.CAPPED), ("CAP", CpuPolicy.CAPPED),
                                           ("throttled", CpuPolicy.CAPPED), ("full", CpuPolicy.FULL),
                                           ("max", CpuPolicy.FULL), ("100%", CpuPolicy.FULL),
                                           ("unrestricted", CpuPolicy.FULL), (CpuPolicy.FULL, CpuPolicy.FULL)])
def test_cpu_policy_parse(text, expected):
    assert CpuPolicy.parse(text) is expected


def test_cpu_policy_rejects_unknown():
    with pytest.raises(ConfigError):
        CpuPolicy.parse("turbo")


@pytest.mark.parametrize("cpus", [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128])
def test_full_policy_scales_to_every_logical_cpu(cpus):
    assert worker_ceiling(cpus, "full") == cpus
    assert worker_floor(cpus, "full") == 1
    assert resolve_worker_count(None, cpus, "full") == cpus
    assert worker_ceiling(cpus, "capped") == max(1, math.floor(cpus * CPU_CEILING_FRACTION))


def test_capped_band_holds_for_every_cpu_count():
    for cpus in range(1, 257):
        lo, hi = worker_floor(cpus), worker_ceiling(cpus)
        assert 1 <= lo <= hi <= cpus
        if cpus >= 2:
            assert hi <= 0.6 * cpus
        full = worker_ceiling(cpus, CpuPolicy.FULL)
        assert full == cpus and full >= hi


@pytest.mark.parametrize("requested,cpus,policy,expected", [
    (10_000, 8, "full", 8), (10_000, 8, "capped", 4), (1, 8, "full", 1), (1, 8, "capped", 4),
    (5, 8, "full", 5), (0, 16, "full", 1), (-3, 4, "full", 1), (3, 3, "full", 3), (None, 3, "capped", 1),
])
def test_resolve_worker_count_clamps_requests(requested, cpus, policy, expected):
    assert resolve_worker_count(requested, cpus, policy) == expected


def test_env_worker_request_obeys_the_policy(monkeypatch):
    monkeypatch.setenv("MEV_SIM_WORKERS", "6")
    assert resolve_worker_count(None, 8, "full") == 6
    assert resolve_worker_count(None, 8, "capped") == 4
    assert resolve_worker_count(2, 8, "full") == 2  # an explicit request beats the environment
    monkeypatch.setenv("MEV_SIM_WORKERS", "lots")
    with pytest.raises(ConfigError):
        resolve_worker_count(None, 8, "full")


def test_for_mode_resolves_workers_under_each_policy():
    assert SimulationConfig.for_mode("max", cpus=8).workers == 8
    assert SimulationConfig.for_mode("brutal", cpus=8).workers == 4
    assert SimulationConfig.for_mode("moderate", cpus=8).workers == 4
    capped_max = SimulationConfig.for_mode("max", cpus=8, cpu_policy="capped")
    assert capped_max.workers == 4 and capped_max.policy is CpuPolicy.CAPPED
    full_brutal = SimulationConfig.for_mode("brutal", cpus=8, cpu_policy="full")
    assert full_brutal.workers == 8 and full_brutal.cpu_policy == "full"


@pytest.mark.parametrize("field_name,value", [
    ("shard_density", 0), ("shard_density", MAX_SHARD_DENSITY + 1), ("pg_paths", -1), ("pg_paths", MAX_PG_PATHS + 1),
    ("pg_batch", 0), ("pg_path_steps", 0), ("cpu_policy", "turbo"), ("audit_fail_on", "fatal"),
    ("audit_tick_samples", -1), ("trace_path", 5), ("start_method", "thread"), ("parallel_threshold", 0),
    ("base_fee_wei", 0), ("priority_fee_wei", -1), ("default_tolerance_bps", 10_001), ("native_token", ""),
])
def test_config_rejects_out_of_range_layer09_fields(field_name, value):
    with pytest.raises(ConfigError):
        SimulationConfig.for_mode("max", cpus=4, **{field_name: value})


def test_shards_scale_with_density():
    cfg = SimulationConfig.for_mode("max", cpus=4)
    assert cfg.shards(16) == 16 * cfg.shard_density == 256
    assert cfg.shards(0) == cfg.shard_density
    assert SimulationConfig.for_mode("brutal", cpus=4).shards(8) == 8


def test_config_to_dict_carries_every_new_setting():
    d = SimulationConfig.for_mode("max", cpus=4).to_dict()
    for key in ("cpu_policy", "shard_density", "pg_paths", "pg_path_steps", "pg_batch", "audit", "audit_fail_on",
                "audit_code_scan", "audit_tick_samples", "trace", "trace_path", "start_method",
                "parallel_threshold", "base_fee_wei", "priority_fee_wei", "native_token"):
        assert key in d, key
    assert d["mode_label"] == "MAX Brutal Simulation" and d["workers"] == 4
    json.dumps(d)


def test_system_spec_reports_the_full_ceiling():
    spec = SystemSpec.detect()
    assert spec.full_ceiling == spec.usable_cpus == CPUS
    assert spec.ceiling_for("full") == CPUS and spec.ceiling_for("capped") == spec.worker_ceiling
    assert spec.to_dict()["full_ceiling"] == CPUS


# ---------------------------------------------------------------------------
# The worker pool under 100% CPU load
# ---------------------------------------------------------------------------


def test_pool_for_config_follows_the_policy():
    pool = WorkerPool.for_config(SimulationConfig.for_mode("max", cpus=CPUS))
    assert pool.policy is CpuPolicy.FULL and pool.workers == CPUS
    capped = WorkerPool.for_config(SimulationConfig.for_mode("brutal", cpus=CPUS))
    assert capped.policy is CpuPolicy.CAPPED and capped.workers == worker_ceiling(CPUS)
    assert pool.active_workers == capped.active_workers == 0  # lazy: nothing started yet


@needs_two_cpus
@pytest.mark.skipif(CPUS > 64, reason="spawning one process per CPU is too slow beyond 64 CPUs")
def test_rendezvous_proves_every_logical_cpu_runs_a_worker_at_once(full_pool):
    probe = rendezvous_probe(full_pool, timeout=60)
    assert probe["released"], probe
    assert probe["workers"] == CPUS and probe["distinct_pids"] == CPUS
    assert full_pool.telemetry()["peak_processes"] == CPUS


@needs_two_cpus
def test_saturation_probe_keeps_every_core_busy(full_pool):
    rendezvous_probe(full_pool, timeout=60)  # every worker is up before timing starts
    probe = saturation_probe(full_pool, seconds=0.3, tasks_per_worker=2)
    assert probe["tasks"] == 2 * CPUS and probe["cpu_policy"] == "full"
    assert probe["utilization"] >= 0.5, probe
    # a single-threaded worker cannot exceed one core
    assert 0.3 <= probe["max_task_cpu_over_wall"] <= 1.1
    assert all(r["rounds"] >= 1 for r in probe["rows"])
    tel = full_pool.telemetry()
    assert tel["instrumented"] and tel["utilization"] is not None and 0 < tel["utilization"] <= 1
    assert tel["worker_share_of_cpus"] == 1.0
    assert sum(tel["tasks_per_worker"]) == tel["parallel_tasks"]


def test_cpu_burn_reports_its_own_cpu_time():
    row = cpu_burn((0.05, 1))
    assert row["wall_s"] >= 0.05 and row["rounds"] >= 1
    assert 0 < row["cpu_s"] <= row["wall_s"] + 0.05


@needs_two_cpus
def test_pool_grows_dynamically_up_to_the_ceiling():
    with WorkerPool(policy="full", parallel_threshold=1) as wp:
        wp.map_parallel(cpu_burn, [(0.0, 1)] * 2)
        assert wp.active_workers == 2
        wp.map_parallel(cpu_burn, [(0.0, 1)] * (CPUS * 3))
        assert wp.active_workers == CPUS
        wp.map_parallel(cpu_burn, [(0.0, 1)] * 2)  # never shrinks
        tel = wp.telemetry()
        assert wp.active_workers == CPUS and tel["peak_processes"] == CPUS
        assert tel["pool_starts"] == (1 if CPUS == 2 else 2)
    assert wp.active_workers == 0


def test_small_batches_run_inline_without_processes():
    with WorkerPool(policy="full", parallel_threshold=10, instrument=True) as wp:
        out = wp.map(cpu_burn, [(0.0, 1)] * 3)
        tel = wp.telemetry()
    assert len(out) == 3 and tel["inline_tasks"] == 3 and tel["parallel_tasks"] == 0 and tel["pool_starts"] == 0


@needs_two_cpus
def test_instrumented_results_match_plain_results():
    tasks = [(0.0, n) for n in range(1, 9)]
    with WorkerPool(policy="full", parallel_threshold=1) as plain, \
            WorkerPool(policy="full", parallel_threshold=1, instrument=True) as inst:
        a = [r["checksum"] for r in plain.map(cpu_burn, tasks)]
        b = [r["checksum"] for r in inst.map(cpu_burn, tasks)]
        assert plain.telemetry()["utilization"] is None
    assert a == b


@needs_two_cpus
def test_threads_can_share_one_pool_while_it_grows():
    errors, results = [], {}
    with WorkerPool(policy="full", parallel_threshold=1, instrument=True) as wp:
        def worker(i):
            try:
                n = 2 + (i % 3) * CPUS  # mixed sizes force growth while other maps run
                rows = wp.map_parallel(cpu_burn, [(0.01, i + 1)] * n)
                results[i] = len(rows)
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
        tel = wp.telemetry()
    assert not errors and len(results) == 6
    assert tel["parallel_tasks"] == sum(results.values()) and tel["peak_processes"] == CPUS


# ---------------------------------------------------------------------------
# Policy-gradient path fuzzer
# ---------------------------------------------------------------------------


def test_path_specs_are_deterministic_and_regimes_round_robin():
    specs = make_path_specs(30, steps=5, seed=4)
    assert specs == make_path_specs(30, steps=5, seed=4)
    assert specs[:10] == make_path_specs(10, steps=5, seed=4)  # adding paths never changes earlier ones
    assert [s.regime for s in specs[:6]] == list(PATH_REGIMES) * 2
    assert make_path_spec(3, 5, seed=5) != specs[3]
    for s in specs:
        assert 0.25 <= s.temperature <= 4.0 and 0.02 <= s.lr <= 0.5 and s.cutoff_ms[0] < s.cutoff_ms[1]


@pytest.mark.parametrize("kwargs", [dict(paths=-1), dict(steps=0), dict(batch=0), dict(env_kind="casino"),
                                    dict(regimes=("calm", "bogus"))])
def test_path_fuzzer_rejects_bad_arguments(kwargs):
    with pytest.raises(ConfigError):
        PolicyPathFuzzer(**kwargs).run()


def test_softmax_is_a_distribution_for_extreme_preferences():
    for prefs in ([1e300, -1e300, 0.0], [5.0] * 7, [-1e9, 1e9]):
        p = softmax(prefs, temperature=0.25)
        assert all(0 <= x <= 1 for x in p) and abs(math.fsum(p) - 1) < 1e-12


def test_single_path_checks_every_invariant():
    table = ActionTable(len(PolicyPathFuzzer().actions))
    spec = make_path_spec(0, steps=12, seed=1)
    agent, outcome, violations, checks = run_path(spec, table)
    assert violations == [] and outcome.violations == 0
    assert checks == 5 * 12 + 2 and table.total_visits() == 12
    assert agent.cumulative_reward_wei == outcome.reward_wei and isinstance(outcome.reward_wei, int)


@needs_two_cpus
def test_path_fuzzer_is_identical_serial_and_on_every_core(full_pool):
    fuzzer = PolicyPathFuzzer(paths=40, steps=4, batch=6, seed=3)
    serial = fuzzer.run()
    parallel = fuzzer.run(executor=full_pool)
    assert serial.ok and parallel.ok
    assert strip_timing(serial.summary()) == strip_timing(parallel.summary())
    assert serial.table.values == parallel.table.values and serial.table.visits == parallel.table.visits
    assert parallel.summary()["parallel"] and parallel.batches == 7


@needs_two_cpus
def test_path_fuzzer_does_not_depend_on_the_worker_count():
    fuzzer = PolicyPathFuzzer(paths=24, steps=3, batch=4, seed=11)
    with WorkerPool(2, policy="full", parallel_threshold=1) as two:
        a = fuzzer.run(executor=two)
    with WorkerPool(policy="full", parallel_threshold=1) as every:
        b = fuzzer.run(executor=every)
    assert strip_timing(a.summary()) == strip_timing(b.summary())
    assert a.table.values == b.table.values


def test_path_fuzzer_summary_shape():
    s = PolicyPathFuzzer(paths=9, steps=3, batch=4, seed=2).run().summary()
    assert s["paths"] == 9 and s["batches"] == 3 and s["total_steps"] == 27
    assert s["invariants"]["ok"] and s["invariants"]["rules"] == list(PATH_RULES)
    assert set(s["regimes"]) == set(PATH_REGIMES) and sum(r["paths"] for r in s["regimes"].values()) == 9
    r = s["reward_wei"]
    assert r["min"] <= r["p05"] <= r["p50"] <= r["p95"] <= r["max"]
    assert s["policy"]["visits"] == 27 and 0 <= s["policy"]["mean_entropy_bits"] <= s["policy"]["max_entropy_bits"]
    json.dumps(s)


def test_bandit_paths_learn_the_optimal_arm():
    fuzzer = PolicyPathFuzzer(paths=1500, steps=8, batch=50, seed=0, env_kind="bandit",
                              env_overrides={"n_contexts": 4, "n_actions": 6, "noise_wei": 10**15})
    report = fuzzer.run()
    assert report.ok
    env = NoisyBanditEnvironment(n_contexts=4, n_actions=6)
    learned = [report.table.best((c,)) == env.optimal(c) for c in range(4)]
    assert sum(learned) >= 3, learned


def test_merge_partials_equals_the_visit_weighted_table_merge():
    from simulator.path_fuzzer import _accumulate

    base = ActionTable(6)
    base.values[(0,)] = [0.1, 0.2, 0.0, 0.0, 0.0, 0.0]
    base.visits[(0,)] = [2, 1, 0, 0, 0, 0]
    shards, partials = [], []
    for i in range(5):
        t = base.copy()
        run_path(make_path_spec(i, 6, seed=9), t, env_kind="bandit",
                 env_overrides={"n_contexts": 2, "n_actions": 6})
        shards.append(t)
        cells = {}
        _accumulate(cells, base, t)
        partials.append(cells)
    merged = merge_partials(base, partials)
    expected = ActionTable.merge(base, shards)
    assert merged.visits == expected.visits
    for key in expected.values:
        assert merged.values[key] == pytest.approx(expected.values[key], abs=1e-12)


def test_merge_partials_rejects_wrong_width():
    with pytest.raises(ConfigError):
        merge_partials(ActionTable(3), [{(0,): [[1.0, 2.0], [1, 1]]}])


def test_path_invariant_violations_are_detected(monkeypatch):
    from simulator import path_fuzzer

    real_learn = path_fuzzer.PolicyGradientAgent.learn

    def leaky_learn(self, obs, a, reward, next_obs, done):
        adv = real_learn(self, obs, a, reward, next_obs, done)
        self.table.values[self.key(obs)][a] += 1.0  # breaks the zero-sum REINFORCE update
        return adv

    monkeypatch.setattr(path_fuzzer.PolicyGradientAgent, "learn", leaky_learn)
    report = PolicyPathFuzzer(paths=3, steps=4, batch=2, seed=1, env_kind="bandit").run()
    assert not report.ok and report.violation_count == 12
    assert {v["rule"] for v in report.violations} == {"pg.zero_sum_update"}
    assert report.summary()["invariants"]["violation_count"] == 12


def test_violation_list_is_capped_but_every_violation_is_counted(monkeypatch):
    from simulator import path_fuzzer

    monkeypatch.setattr(path_fuzzer, "_simplex_ok", lambda p: False)
    report = PolicyPathFuzzer(paths=4, steps=5, batch=4, seed=1, env_kind="bandit", violation_limit=3).run()
    assert report.violation_count == 20 and len(report.violations) == 3


# ---------------------------------------------------------------------------
# The controller in MAX Brutal Simulation
# ---------------------------------------------------------------------------


def test_max_controller_run_uses_every_core_and_fuzzes_paths():
    cfg = max_cfg()
    out = SimulationController(cfg).run()
    res = out["resources"]
    assert out["mode_label"] == "MAX Brutal Simulation" and invariants_ok(out)
    assert res["cpu_policy"] == "full" and res["workers"] == CPUS == cfg.workers
    if CPUS > 1:
        assert res["peak_processes"] == CPUS and res["parallel_tasks"] > 0
        assert res["distinct_worker_pids"] >= 2
    pp = out["result"]["policy_paths"]
    assert pp["paths"] == 48 and pp["invariants"]["ok"] and pp["total_steps"] == 48 * 4
    assert out["audit"]["enabled"] and out["audit"]["ok"]
    assert out["result"]["reinforcement"]["train"]["total_steps"] == 8 * cfg.shard_density * 6
    json.dumps(out)


def test_max_results_do_not_depend_on_the_worker_count():
    one = SimulationController(max_cfg(workers=1)).run()
    every = SimulationController(max_cfg()).run()
    assert one["config"]["workers"] == 1 and every["config"]["workers"] == CPUS
    assert strip_timing(one["result"]) == strip_timing(every["result"])


def test_moderate_scan_runs_no_predatory_loops():
    cfg = SimulationConfig.for_mode("moderate", txs_per_block=6, fuzz_iterations=16, pg_paths=32)
    out = SimulationController(cfg).run()
    for key in ("pbs", "private_relay", "reinforcement", "policy_paths"):
        assert key not in out["result"]
    assert out["resources"]["cpu_policy"] == "capped"
    assert "signal.pg_paths_ignored" in out["audit"]["by_rule"]  # informational: the scan ignores pg_paths


def test_trace_is_written_to_the_configured_file(tmp_path):
    import logging

    logger = logging.getLogger("simulator.block_builder")
    before = (logger.level, logger.propagate, list(logger.handlers))
    path = tmp_path / "trace.log"
    cfg = SimulationConfig.for_mode("brutal", blocks=1, txs_per_block=6, fuzz_iterations=8, frontrun_grid=4,
                                    refine_iterations=2, layout_fuzz_iterations=4, rl_rounds=0,
                                    private_flow_bps=0, trace=True, trace_path=str(path))
    out = SimulationController(cfg).run()
    text = path.read_text(encoding="utf-8")
    assert out["config"]["trace"] and "block" in text.lower() and len(text.splitlines()) > 3
    assert (logger.level, logger.propagate, list(logger.handlers)) == before


def test_trace_logging_is_a_no_op_when_disabled():
    import logging

    logger = logging.getLogger("simulator.block_builder")
    before = list(logger.handlers)
    with trace_logging(False, "") as handler:
        assert handler is None and logger.handlers == before


@needs_two_cpus
def test_path_fuzz_stays_exact_while_another_pool_saturates_every_core(full_pool):
    fuzzer = PolicyPathFuzzer(paths=30, steps=3, batch=5, seed=21)
    expected = strip_timing(fuzzer.run().summary())
    load = {}

    def burn():
        load["probe"] = saturation_probe(full_pool, seconds=0.6, tasks_per_worker=2)

    t = threading.Thread(target=burn)
    t.start()
    with WorkerPool(policy="full", parallel_threshold=1) as other:
        got = strip_timing(fuzzer.run(executor=other).summary())
    t.join(120)
    assert got == expected
    assert load["probe"]["tasks"] == 2 * CPUS
