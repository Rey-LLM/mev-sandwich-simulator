"""Layer 11: embedded AI agent factory.

Matrix weight initialisation, forward / backward / AdamW, adaptive mutation
stability, RL metric ingestion, gene <-> profile decoding, runtime bot
classes, the arena's block-building queue, multi-agent resource allocation,
and the factory end to end - inline, on worker processes, and with several
threads sharing (or saturating) one pool.
"""

import json
import math
import pickle
import random
import threading

import numpy as np
import pytest

from simulator.ai_agent_factory import (
    ACTIVATIONS,
    ARENA_POOLS,
    ARENA_ROUTES,
    FACTORY_RULES,
    FEATURE_NAMES,
    GENE_NAMES,
    HOP_POLICIES,
    INIT_SCHEMES,
    MAX_BUNDLE_SLOTS,
    N_FEATURES,
    N_GENES,
    STRATEGY_VALUES,
    AdamOptimizer,
    AdaptiveMutation,
    AgentFactory,
    ArenaConfig,
    BotArena,
    BotProfile,
    HopOracles,
    IncumbentPolicy,
    NeuralMatrixEngine,
    RLMetrics,
    SynthesizedBot,
    activate,
    activation_grad,
    allocate_capital,
    allocate_offspring,
    apportion,
    arena_market,
    bot_class,
    class_name,
    decode_profile,
    encode_profile,
    fitness_key,
    hop_candidate,
    init_matrix,
    init_std,
    lineage,
    plan_heats,
    profile_from_competitor,
    profile_issues,
    rank_order,
    reflect_unit,
    seeded_generator,
    sigmoid,
    signed_log_scale,
    spawn_bot,
)
from simulator.block_builder import (
    GWEI,
    BuilderState,
    BuilderTx,
    CompetitorBot,
    TxKind,
    _quote_hops,
    _route_hops,
    best_sandwich,
    default_competitors,
    native_value,
)
from simulator.config import ConfigError, SimulationConfig, detect_cpu_count
from simulator.controller import SimulationController, invariants_ok
from simulator.parallel import WorkerPool, saturation_probe
from simulator.reinforcement_engine import (
    Action,
    ActionTable,
    EnvConfig,
    FixedPolicy,
    Observation,
    ReinforcementSimulation,
    SandwichEnvironment,
    StateBins,
    action_grid,
    run_agent,
)
from simulator.slippage_engine import _direction, min_amount_out
from simulator.validator import Collector, check_signals, scan_source

E18 = 10**18
CPUS = detect_cpu_count()
needs_two_cpus = pytest.mark.skipif(CPUS < 2, reason="needs at least two usable CPUs")
TIMING_KEYS = {"elapsed_s", "cpu_s", "workers", "parallel", "distinct_worker_pids", "parallel_efficiency"}


def strip_timing(obj):
    if isinstance(obj, dict):
        return {k: strip_timing(v) for k, v in obj.items() if k not in TIMING_KEYS and not k.endswith("_per_s")}
    if isinstance(obj, list):
        return [strip_timing(v) for v in obj]
    return obj


def small_factory(**kw):
    params = dict(bots=12, generations=2, heat_size=4, heat_slots=3, seed=5, hidden=(16,), latent_dim=4, epochs=15)
    params.update(kw)
    return AgentFactory(**params)


def make_profile(**kw):
    base = dict(name="t", strategy="fixed_share", share_bps=5_000, step_bps=400, min_share_bps=1_000,
                max_share_bps=9_000, tip_min_wei=GWEI, tip_max_wei=10 * GWEI, tip_step_bps=1_000, size_bps=10_000,
                max_impact_bps=2_000, slippage_floor_bps=0, max_hops=2, hop_policy="oracle",
                target_pools=ARENA_POOLS, bundle_slots=1, front_slack_bps=0, latency_ms=100, jitter_ms=10,
                capital_wei=500 * E18, generation=0)
    base.update(kw)
    return BotProfile(**base)


def make_victim(state, token_in, route, bps=80, tolerance=500, tag="a", base_fee=20 * GWEI):
    pool = state.pools[route[0]]
    zfo, _ = _direction(pool, token_in)
    amount = (pool.reserve0 if zfo else pool.reserve1) * bps // 10_000
    probe = BuilderTx("probe", "p", 0, 10**6, base_fee * 3, GWEI, TxKind.SWAP,
                      {"route": list(route), "token_in": token_in, "amount_in": amount})
    hops, _ = _quote_hops(state, probe)
    out = hops[-1][3].amount_out
    return BuilderTx(f"0x{tag}", f"user:{tag}", 0, 400_000, base_fee * 3 + GWEI, GWEI, TxKind.SWAP,
                     {"route": list(route), "token_in": token_in, "amount_in": amount,
                      "amount_out_min": min_amount_out(out, tolerance), "tolerance_bps": tolerance})


@pytest.fixture(scope="module")
def market():
    return BuilderState.from_pools(arena_market())


@pytest.fixture(scope="module")
def two_workers():
    with WorkerPool(workers=2, policy="full", parallel_threshold=1, instrument=True) as wp:
        yield wp


@pytest.fixture(scope="module")
def baseline_report():
    return small_factory().run().summary()


# ---------------------------------------------------------------------------
# Matrix weight initialisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", INIT_SCHEMES)
@pytest.mark.parametrize("shape", [(1, 1), (7, 3), (3, 7), (24, 32)])
def test_init_matrix_shape_finite_and_seeded(scheme, shape):
    a = init_matrix(*shape, scheme, seeded_generator("t", scheme, shape))
    b = init_matrix(*shape, scheme, seeded_generator("t", scheme, shape))
    assert a.shape == shape and a.dtype == np.float64 and np.all(np.isfinite(a))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("scheme", INIT_SCHEMES)
def test_init_matrix_variance_matches_the_scheme(scheme):
    w = init_matrix(300, 200, scheme, seeded_generator("variance", scheme))
    assert abs(float(np.std(w)) / init_std(300, 200, scheme) - 1.0) < 0.03
    assert abs(float(np.mean(w))) < 0.01 * init_std(300, 200, scheme) * 10


@pytest.mark.parametrize("shape", [(5, 5), (40, 9), (9, 40)])
def test_xavier_weights_stay_inside_the_glorot_limit(shape):
    w = init_matrix(*shape, "xavier", seeded_generator("glorot", shape))
    assert np.max(np.abs(w)) <= math.sqrt(6.0 / sum(shape))


@pytest.mark.parametrize("shape", [(12, 5), (5, 12), (8, 8)])
def test_orthogonal_init_is_orthonormal(shape):
    w = init_matrix(*shape, "orthogonal", seeded_generator("orth", shape))
    gram = w.T @ w if shape[0] >= shape[1] else w @ w.T
    assert np.allclose(gram, np.eye(min(shape)), atol=1e-12)


def test_init_rejects_bad_fans_and_schemes():
    with pytest.raises(ConfigError):
        init_std(0, 3)
    with pytest.raises(ConfigError):
        init_matrix(3, 3, "glorot-ish")


def test_engine_initialises_zero_biases_and_distinct_layers():
    e = NeuralMatrixEngine((6, 8, 8, 4), seed=3)
    assert all(np.all(b == 0) for b in e.biases)
    assert not np.array_equal(e.weights[1], e.weights[1].T) and e.weights[0].shape == (6, 8)
    assert e.num_params == 6 * 8 + 8 + 8 * 8 + 8 + 8 * 4 + 4 == e.flat().size


def test_engine_is_a_pure_function_of_its_seed():
    a, b, c = NeuralMatrixEngine((5, 7, 3), seed=11), NeuralMatrixEngine((5, 7, 3), seed=11), \
        NeuralMatrixEngine((5, 7, 3), seed=12)
    assert a.digest() == b.digest() != c.digest()


@pytest.mark.parametrize("sizes,kw", [((4,), {}), ((4, 0, 2), {}), ((4, 2), {"init": "kaiming"}),
                                      ((4, 2), {"hidden_activation": "gelu"})])
def test_engine_rejects_bad_architectures(sizes, kw):
    with pytest.raises(ConfigError):
        NeuralMatrixEngine(sizes, **kw)


def test_seeded_generator_streams_are_reproducible_and_independent():
    a = seeded_generator("x", 1).standard_normal(5)
    assert np.array_equal(a, seeded_generator("x", 1).standard_normal(5))
    assert not np.array_equal(a, seeded_generator("x", 2).standard_normal(5))


# ---------------------------------------------------------------------------
# Forward, backward and the optimiser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_activation_gradient_matches_finite_differences(name):
    z = np.linspace(-3, 3, 13) + 0.05  # avoid the ReLU kink at 0
    eps = 1e-6
    numeric = (activate(name, z + eps) - activate(name, z - eps)) / (2 * eps)
    assert np.allclose(activation_grad(name, z, activate(name, z)), numeric, atol=1e-6)


def test_unknown_activation_is_rejected():
    with pytest.raises(ConfigError):
        activate("swish", np.zeros(2))
    with pytest.raises(ConfigError):
        activation_grad("swish", np.zeros(2), np.zeros(2))


def test_sigmoid_is_stable_on_extreme_inputs():
    out = sigmoid(np.array([-1e6, -800.0, 0.0, 800.0, 1e6]))
    assert np.all(np.isfinite(out)) and np.all((out >= 0) & (out <= 1)) and out[2] == pytest.approx(0.5)


def test_forward_outputs_unit_interval_and_batch_matches_single():
    e = NeuralMatrixEngine((N_FEATURES + 4, 16, N_GENES), seed=1)
    x = seeded_generator("fwd").standard_normal((5, N_FEATURES + 4))
    batch = e.forward(x)
    assert batch.shape == (5, N_GENES) and np.all((batch > 0) & (batch < 1))
    assert np.allclose(e(x[2]), batch[2])
    with pytest.raises(ConfigError):
        e.forward(np.zeros((2, 3)))


@pytest.mark.parametrize("sizes,hidden,output", [((3, 4, 2), "tanh", "sigmoid"), ((5, 6, 6, 3), "tanh", "linear"),
                                                 ((4, 5, 2), "relu", "sigmoid"), ((2, 3), "tanh", "tanh"),
                                                 ((6, 8, 4, 4, 5), "tanh", "sigmoid")])
def test_backprop_matches_numerical_gradient(sizes, hidden, output):
    e = NeuralMatrixEngine(sizes, hidden, output, init="xavier", seed=7)
    rng = seeded_generator("gc", sizes)
    x = rng.standard_normal((6, sizes[0]))
    y = rng.uniform(0, 1, (6, sizes[-1]))
    w = rng.uniform(0.5, 2.0, 6)
    analytic, numeric = e.numerical_gradient(x, y, w, eps=1e-6, max_entries=40)
    assert np.allclose(analytic, numeric, rtol=1e-4, atol=1e-8)


def test_loss_is_invariant_to_weight_scale_and_ignores_zero_weights():
    e = NeuralMatrixEngine((3, 4, 2), seed=2)
    rng = seeded_generator("w")
    x, y = rng.standard_normal((4, 3)), rng.uniform(0, 1, (4, 2))
    w = np.array([1.0, 2.0, 0.0, 3.0])
    assert e.loss(x, y, w) == pytest.approx(e.loss(x, y, 10 * w))
    keep = [0, 1, 3]
    assert e.loss(x, y, w) == pytest.approx(e.loss(x[keep], y[keep], w[keep]))
    _, g_all = e.gradients(x, y, w)
    _, g_keep = e.gradients(x[keep], y[keep], w[keep])
    assert all(np.allclose(a, b) for a, b in zip(g_all, g_keep))


@pytest.mark.parametrize("weights", [[1, -1, 1], [0, 0, 0], [1, float("nan"), 1], [1, 1]])
def test_bad_sample_weights_are_rejected(weights):
    e = NeuralMatrixEngine((2, 2), seed=1)
    with pytest.raises(ConfigError):
        e.loss(np.zeros((3, 2)), np.zeros((3, 2)), weights)


def test_fit_learns_a_smooth_target():
    rng = seeded_generator("fit")
    x = rng.uniform(-1, 1, (64, 3))
    y = sigmoid(np.stack([x[:, 0] - x[:, 1], 2 * x[:, 2]], axis=1))
    e = NeuralMatrixEngine((3, 12, 2), seed=4, optimizer=AdamOptimizer(lr=0.05))
    hist = e.fit(x, y, epochs=300)
    assert hist[-1] < 0.1 * hist[0] and hist[-1] < 1e-3 and e.is_finite()


def test_adam_clips_the_global_gradient_norm():
    p = [np.zeros(3)]
    opt = AdamOptimizer(lr=0.1, clip_norm=1.0, weight_decay=0.0)
    norm = opt.step(p, [np.array([300.0, 400.0, 0.0])])
    assert norm == pytest.approx(500.0) and opt.t == 1
    assert np.all(np.abs(p[0]) <= 0.1 + 1e-12)  # Adam's first step moves each weight by at most lr


def test_adam_skips_non_finite_gradients():
    p = [np.ones(2)]
    opt = AdamOptimizer()
    opt.step(p, [np.array([float("inf"), 1.0])])
    assert np.array_equal(p[0], np.ones(2)) and opt.skipped == 1 and opt.t == 0


def test_adamw_weight_decay_shrinks_idle_weights():
    p = [np.full(4, 2.0)]
    opt = AdamOptimizer(lr=0.1, weight_decay=0.5)
    for _ in range(10):
        opt.step(p, [np.zeros(4)])
    assert np.all(p[0] < 2.0) and np.all(p[0] > 0)


@pytest.mark.parametrize("kw", [{"lr": 0.0}, {"lr": float("inf")}, {"beta1": 1.0}, {"eps": 0.0},
                                {"weight_decay": -1.0}, {"clip_norm": -1.0}])
def test_adam_rejects_bad_settings(kw):
    with pytest.raises(ConfigError):
        AdamOptimizer(**kw)


def test_flat_round_trip_copy_and_payload():
    e = NeuralMatrixEngine((4, 5, 3), seed=9)
    twin = e.copy()
    assert twin.digest() == e.digest()
    twin.weights[0][0, 0] += 1.0
    assert twin.digest() != e.digest()
    restored = NeuralMatrixEngine.from_payload(json.loads(json.dumps(e.to_payload())))
    assert restored.digest() == e.digest()
    with pytest.raises(ConfigError):
        e.set_flat(np.zeros(3))


def test_perturbed_engine_noise_has_the_requested_scale():
    e = NeuralMatrixEngine((30, 40, 20), seed=1)
    assert e.perturbed(0.0, seeded_generator("p")).digest() == e.digest()
    delta = e.perturbed(0.05, seeded_generator("p")).flat() - e.flat()
    assert abs(float(np.std(delta)) / 0.05 - 1) < 0.05
    with pytest.raises(ConfigError):
        e.perturbed(-1.0, seeded_generator("p"))


def test_training_on_extreme_targets_with_a_huge_step_stays_finite():
    rng = seeded_generator("extreme")
    x = rng.standard_normal((32, 5)) * 1e3
    y = (rng.uniform(0, 1, (32, 4)) > 0.5).astype(float)
    e = NeuralMatrixEngine((5, 16, 4), seed=2, optimizer=AdamOptimizer(lr=5.0, clip_norm=1.0))
    hist = e.fit(x, y, epochs=200)
    assert e.is_finite() and all(math.isfinite(h) for h in hist)
    assert np.all(np.isfinite(e.forward(x)))


def test_describe_reports_the_architecture():
    d = NeuralMatrixEngine((4, 6, 2), seed=1).describe()
    assert d["layers"] == [4, 6, 2] and d["params"] == 4 * 6 + 6 + 6 * 2 + 2 and len(d["digest"]) == 64
    json.dumps(d)


# ---------------------------------------------------------------------------
# Adaptive mutation stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [(0.3, 0.3), (0.0, 0.0), (1.0, 1.0), (1.25, 0.75), (-0.25, 0.25),
                                            (2.5, 0.5), (-3.2, 0.8), (7.0, 1.0), (float("nan"), 0.5),
                                            (float("inf"), 0.5)])
def test_reflect_unit(value, expected):
    assert float(reflect_unit(value)) == pytest.approx(expected)


@pytest.mark.parametrize("rate", [0.0, 0.1, 0.2, 0.5, 1.0])
def test_one_fifth_rule_direction(rate):
    m = AdaptiveMutation(sigma=0.1)
    new = m.update(rate)
    if rate > m.target_success:
        assert new > 0.1
    elif rate < m.target_success:
        assert new < 0.1
    else:
        assert new == pytest.approx(0.1)
    assert m.history == [new]


@pytest.mark.parametrize("seed", range(5))
def test_sigma_never_leaves_its_bounds(seed):
    rng = random.Random(seed)
    m = AdaptiveMutation(sigma=0.2, sigma_min=0.02, sigma_max=0.3)
    choices = [float("nan"), float("inf"), -float("inf"), -5.0, 7.0, 0.0, 1.0]
    for _ in range(500):
        m.update(rng.choice(choices) if rng.random() < 0.2 else rng.random())
        assert 0.02 <= m.sigma <= 0.3 and math.isfinite(m.sigma)


def test_non_finite_success_rates_leave_sigma_unchanged():
    m = AdaptiveMutation(sigma=0.1)
    for bad in (float("nan"), float("inf"), -float("inf")):
        assert m.update(bad) == pytest.approx(0.1)


def test_persistent_success_or_failure_saturates_at_the_bounds():
    up, down = AdaptiveMutation(), AdaptiveMutation()
    for _ in range(100):
        up.update(1.0)
        down.update(0.0)
    assert up.sigma == up.sigma_max and down.sigma == down.sigma_min


@pytest.mark.parametrize("sigma", [0.01, 0.2, 0.45])
def test_gene_mutation_stays_in_the_unit_cube(sigma):
    m = AdaptiveMutation(sigma=sigma, sigma_max=0.45)
    genes = seeded_generator("g", sigma).uniform(0, 1, (400, N_GENES))
    genes[0], genes[1] = 0.0, 1.0
    out = m.mutate_genes(genes, seeded_generator("m", sigma))
    assert out.shape == genes.shape and np.all((out >= 0) & (out <= 1))


def test_mutation_is_reproducible_per_stream():
    m = AdaptiveMutation()
    g = np.full((3, N_GENES), 0.5)
    a = m.mutate_genes(g, seeded_generator("s", 1))
    assert np.array_equal(a, m.mutate_genes(g, seeded_generator("s", 1)))
    assert not np.array_equal(a, m.mutate_genes(g, seeded_generator("s", 2)))


def test_small_mutation_steps_have_the_expected_size():
    m = AdaptiveMutation(sigma=0.02)
    g = np.full((2_000, N_GENES), 0.5)
    step = np.abs(m.mutate_genes(g, seeded_generator("size")) - g)
    assert abs(float(np.mean(step)) / (0.02 * math.sqrt(2 / math.pi)) - 1) < 0.03


def test_long_mutation_chains_neither_escape_nor_collapse():
    m = AdaptiveMutation(sigma=0.3)
    g = np.full((256, N_GENES), 0.5)
    for i in range(200):
        g = m.mutate_genes(g, seeded_generator("chain", i))
    assert np.all((g >= 0) & (g <= 1))
    assert 0.2 < float(np.std(g)) < 0.4 and abs(float(np.mean(g)) - 0.5) < 0.02  # ~uniform on [0, 1]


def test_vector_mutation_is_unbounded_and_scaled():
    m = AdaptiveMutation(sigma=0.1)
    v = m.mutate_vector(np.zeros(20_000), seeded_generator("vec"), scale=3.0)
    assert abs(float(np.std(v)) / 0.3 - 1) < 0.03


@pytest.mark.parametrize("kw", [{"sigma": 0.0, "sigma_min": 0.0}, {"sigma": 0.5, "sigma_max": 0.4},
                                {"sigma": 0.01, "sigma_min": 0.05}, {"target_success": 1.0}, {"damping": 0.0},
                                {"sigma": float("nan")}])
def test_mutation_rejects_bad_parameters(kw):
    with pytest.raises(ConfigError):
        AdaptiveMutation(**kw)


# ---------------------------------------------------------------------------
# RL metric ingestion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", ["calm", "volatile", "extreme"])
def test_neutral_features_are_bounded_with_a_regime_one_hot(regime):
    m = RLMetrics.neutral(regime)
    v = m.vector()
    assert v.shape == (N_FEATURES,) and np.all((v >= -1) & (v <= 1))
    one_hot = [m.to_dict()[f"regime_{r}"] for r in ("calm", "volatile", "extreme")]
    assert sum(one_hot) == 1.0 and m.to_dict()[f"regime_{regime}"] == 1.0 and m.sources == ()


@pytest.fixture(scope="module")
def rl_run():
    sim = ReinforcementSimulation(seed=3, rounds=1, steps_per_shard=6, eval_steps=12, shards=2)
    report = sim.run()
    return sim, report


def test_features_from_a_real_rl_run_and_its_table(rl_run):
    sim, report = rl_run
    m = RLMetrics.from_sources(report, sim.last_train.table)
    d = m.to_dict()
    assert set(d) == set(FEATURE_NAMES) and m.sources == ("reinforcement", "table")
    assert all(-1 <= v <= 1 for v in d.values())
    assert d["eval_win_rate"] == pytest.approx(report["eval"]["win_rate"], abs=1e-9)
    assert d["regime_volatile"] == 1.0 and 0 < d["policy_coverage"] <= 1


def test_policy_path_entropy_feature():
    summary = {"policy": {"mean_entropy_bits": 2.0, "max_entropy_bits": 4.0}}
    assert RLMetrics.from_sources(policy_paths=summary).to_dict()["pg_entropy"] == 0.5
    assert RLMetrics.from_sources().to_dict()["pg_entropy"] == 1.0


@pytest.mark.parametrize("bad", [{"train": "oops"}, {"eval": {"win_rate": "high", "submitted": -3}},
                                 {"train": {"round_win_rates": [None], "round_rewards_wei": ["x", 5]}},
                                 {"eval": {"bot_outcomes": {"a": {"won": float("nan")}}}}, {"regime": 42}])
def test_malformed_rl_reports_fall_back_to_neutral_values(bad):
    m = RLMetrics.from_sources(bad)
    assert all(math.isfinite(v) and -1 <= v <= 1 for v in m.values)


def test_unknown_regime_and_table_width_mismatch_are_rejected():
    with pytest.raises(ConfigError):
        RLMetrics.from_sources(regime="chaos")
    with pytest.raises(ConfigError):
        RLMetrics.from_sources(table=ActionTable(3))
    with pytest.raises(ConfigError):
        RLMetrics((0.0,) * 3)


@pytest.mark.parametrize("wei", [0, 1, 10**12, 10**15, 10**18, 10**30, -10**18])
def test_signed_log_scale_is_bounded_and_odd(wei):
    v = signed_log_scale(wei)
    assert -1 <= v <= 1 and signed_log_scale(-wei) == -v
    assert signed_log_scale(abs(wei) * 10 + 1) >= abs(v)


# ---------------------------------------------------------------------------
# Genes <-> bot profiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_random_genes_decode_to_valid_profiles(seed):
    genes = seeded_generator("decode", seed).uniform(-0.5, 1.5, N_GENES)
    p = decode_profile(genes, f"bot{seed}", generation=seed)
    assert profile_issues(p) == []
    assert p.strategy in STRATEGY_VALUES and p.hop_policy in HOP_POLICIES and 1 <= p.bundle_slots <= MAX_BUNDLE_SLOTS
    assert set(p.target_pools) <= set(ARENA_POOLS) and p.tip_min_wei <= p.tip_max_wei <= 200 * GWEI
    assert p.tip_min_wei % 10**7 == 0 and p.capital_wei % 10**15 == 0


@pytest.mark.parametrize("seed", range(12))
def test_encode_then_decode_is_exact_for_decoded_profiles(seed):
    p = decode_profile(seeded_generator("roundtrip", seed).uniform(0, 1, N_GENES), "x", 3)
    genes = encode_profile(p)
    assert genes.shape == (N_GENES,) and np.all((genes >= 0) & (genes <= 1))
    assert decode_profile(genes, "x", 3) == p


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_extreme_gene_vectors_decode(value):
    p = decode_profile(np.full(N_GENES, value), "edge")
    assert profile_issues(p) == []
    if value == 1.0:
        assert p.target_pools == ARENA_POOLS and p.max_hops == 2 and p.bundle_slots == MAX_BUNDLE_SLOTS
    else:
        assert p.target_pools == (ARENA_POOLS[0],) and p.max_hops == 1 and p.bundle_slots == 1


def test_strategy_ties_go_to_the_first_gene_and_wrong_length_is_rejected():
    assert decode_profile(np.full(N_GENES, 0.5), "tie").strategy == STRATEGY_VALUES[0]
    with pytest.raises(ConfigError):
        decode_profile(np.zeros(N_GENES - 1), "short")
    assert len(GENE_NAMES) == N_GENES == 23


@pytest.mark.parametrize("field_,value", [("strategy", "yolo"), ("share_bps", 9_500), ("tip_min_wei", 0),
                                          ("tip_max_wei", 300 * GWEI), ("size_bps", 0), ("max_hops", 3),
                                          ("hop_policy", "random"), ("target_pools", ()),
                                          ("target_pools", ("v2:WETH/DAI", "v2:WETH/USDC")), ("bundle_slots", 9),
                                          ("capital_wei", -1), ("name", ""), ("latency_ms", 1.5)])
def test_invalid_profiles_are_rejected(field_, value):
    with pytest.raises(ConfigError):
        make_profile(**{field_: value})


def test_fingerprint_tracks_behaviour_only():
    p = make_profile()
    assert p.fingerprint() == make_profile(name="other", generation=4, capital_wei=E18).fingerprint()
    assert p.fingerprint() != make_profile(tip_step_bps=1_001).fingerprint()


def test_profile_serialises_describes_and_maps_to_a_layer08_bot():
    p = decode_profile(seeded_generator("ser").uniform(0, 1, N_GENES), "ser")
    d = json.loads(json.dumps(p.to_dict()))
    assert d["fingerprint"] == p.fingerprint() and d["target_pools"] == list(p.target_pools)
    text = p.describe()
    assert p.strategy in text and "gwei" in text and "ETH" in text
    bot = CompetitorBot(**p.competitor_kwargs())
    assert bot.tip_wei == p.tip_min_wei and bot.strategy.value == p.strategy


def test_default_layer08_competitors_convert_to_profiles():
    for bot in default_competitors():
        p = profile_from_competitor(bot)
        assert profile_issues(p) == [] and p.strategy == bot.strategy.value and p.tip_min_wei == p.tip_max_wei


# ---------------------------------------------------------------------------
# Runtime bot classes
# ---------------------------------------------------------------------------


def test_bot_class_is_a_cached_dynamic_subclass():
    p = make_profile(name="cls")
    cls = bot_class(p)
    assert issubclass(cls, SynthesizedBot) and cls.PROFILE is p and bot_class(p) is cls
    assert cls.__name__ == class_name(p) and cls.__name__.startswith("AIBot_FixedShare_Oracle_")
    assert cls.__doc__ == p.describe()
    bot = cls()
    assert bot.name == "cls" and bot.searcher == "ai:cls" and bot.account(2) == "ai:cls#2"


def test_bots_pickle_with_their_state_and_class():
    bot = spawn_bot(make_profile(name="pickled", strategy="adaptive"))
    bot.observe(False)
    bot.observe(False)
    clone = pickle.loads(pickle.dumps(bot))
    assert type(clone).__name__ == type(bot).__name__ and isinstance(clone, SynthesizedBot)
    assert (clone.tip_wei, clone.share_bps, clone.losses) == (bot.tip_wei, bot.share_bps, 2)


def test_a_bare_synthesized_bot_needs_a_profile():
    with pytest.raises(ConfigError):
        SynthesizedBot()


@pytest.mark.parametrize("step", [0, 500, 3_000])
def test_priority_fee_escalation_stays_in_the_tip_range(step):
    p = make_profile(tip_min_wei=GWEI, tip_max_wei=3 * GWEI, tip_step_bps=step)
    bot = spawn_bot(p)
    for _ in range(50):
        bot.observe(False)
        assert p.tip_min_wei <= bot.tip_wei <= p.tip_max_wei
    assert bot.tip_wei == (p.tip_max_wei if step else p.tip_min_wei)
    for _ in range(200):
        bot.observe(True)
        assert p.tip_min_wei <= bot.tip_wei <= p.tip_max_wei
    assert bot.tip_wei == p.tip_min_wei or step == 0


def test_adaptive_share_moves_inside_its_band():
    p = make_profile(strategy="adaptive", share_bps=5_000, min_share_bps=4_000, max_share_bps=6_000, step_bps=700)
    bot = spawn_bot(p)
    for _ in range(10):
        bot.observe(False)
    assert bot.share_bps == 6_000
    for _ in range(30):
        bot.observe(True)
    assert bot.share_bps == 4_000 and bot.wins == 30 and bot.losses == 10


def test_bots_bid_on_profitable_victims(market):
    v = make_victim(market, "WETH", ("v2:WETH/USDC",), tag="bid")
    bot = spawn_bot(make_profile(name="bidder"))
    plan = bot.plan(market, v, HopOracles(market.pools, v), 20 * GWEI)
    assert plan is not None and plan.quote.net_wei > 0 and plan.bribe_wei == plan.quote.net_wei * 5_000 // 10_000
    b = bot.bundle(market, v, plan, 20 * GWEI, slot=1)
    assert b.searcher == "ai:bidder#1" and b.txs[1] is v and b.coinbase_transfer_wei == plan.bribe_wei


@pytest.mark.parametrize("profile_kw,victim_kw", [
    ({"slippage_floor_bps": 600}, {"tolerance": 500}),
    ({"max_hops": 1}, {"route": ("v2:WETH/USDC", "v2:USDC/DAI")}),
    ({"target_pools": ("v2:USDC/DAI",)}, {}),
    ({"strategy": "static_router", "target_pools": ("v2:USDC/DAI",)},
     {"route": ("v2:WETH/USDC", "v2:USDC/DAI")}),
])
def test_bots_pass_on_victims_outside_their_profile(market, profile_kw, victim_kw):
    route = victim_kw.get("route", ("v2:WETH/USDC",))
    v = make_victim(market, "WETH", route, tolerance=victim_kw.get("tolerance", 500), tag="pass")
    bot = spawn_bot(make_profile(name="picky", **profile_kw))
    assert bot.plan(market, v, HopOracles(market.pools, v), 20 * GWEI) is None


def test_capital_and_impact_caps_bound_the_front_run(market):
    v = make_victim(market, "WETH", ("v2:WETH/USDC",), bps=150, tolerance=1_000, tag="caps")
    oracles = HopOracles(market.pools, v)
    free = spawn_bot(make_profile(name="free")).plan(market, v, oracles, 20 * GWEI)
    poor = spawn_bot(make_profile(name="poor", capital_wei=E18)).plan(market, v, oracles, 20 * GWEI)
    tiny = spawn_bot(make_profile(name="tiny", max_impact_bps=5)).plan(market, v, oracles, 20 * GWEI)
    reserve = market.pools["v2:WETH/USDC"].reserve0
    assert free is not None and free.quote.frontrun_in > E18
    assert poor is None or native_value(market.pools, "WETH", poor.quote.frontrun_in) <= E18
    assert tiny is None or tiny.quote.frontrun_in <= reserve * 5 // 10_000


@pytest.mark.parametrize("policy", HOP_POLICIES)
def test_every_hop_policy_targets_a_hop_of_the_route(market, policy):
    v = make_victim(market, "WETH", ("v2:WETH/USDC", "v2:USDC/DAI"), bps=100, tolerance=1_000, tag=policy)
    oracles = HopOracles(market.pools, v)
    plan = spawn_bot(make_profile(name=policy, hop_policy=policy)).plan(market, v, oracles, 20 * GWEI)
    if plan is not None:
        assert plan.hop in (0, 1) and plan.quote.pool_id == v.route[plan.hop]
    if policy == "oracle":
        best = oracles.best()
        assert plan is not None and best is not None and plan.hop == best.hop


def test_hop_oracles_cache_and_agree_with_the_builder_search(market):
    v = make_victim(market, "USDC", ("v2:WETH/USDC",), bps=60, tolerance=300, tag="oracle")
    oracles = HopOracles(market.pools, v)
    cand = oracles(0)
    assert oracles(0) is cand and oracles.evaluations == 1 and len(oracles) == 1
    reference = best_sandwich(market.pools, v, 10, 5, 2_000, "WETH")
    assert cand is not None and reference is not None and cand.hop == reference.hop == 0
    assert cand.profit > 0 and cand.profit >= reference.profit * 0.98
    assert hop_candidate(market.pools, v, 3) is None and oracles(5) is None


def test_synthesized_bots_compete_inside_the_layer08_environment():
    bots = [spawn_bot(make_profile(name=f"env{i}", share_bps=3_000 + 3_000 * i, target_pools=("v2:WETH/USDC",),
                                   latency_ms=50, jitter_ms=5)) for i in range(2)]
    env = SandwichEnvironment(EnvConfig(seed=2, regime="calm"), bots=bots)
    report = run_agent(FixedPolicy(Action(10_000, 1_000)), env, 25, learn=False)
    assert set(report.bot_outcomes) == {"env0", "env1"}
    assert sum(b.wins + b.losses for b in bots) > 0
    assert report.bot_outcomes["env1"].get("won", 0) >= report.bot_outcomes["env0"].get("won", 0)


# ---------------------------------------------------------------------------
# The arena and its block-building queue
# ---------------------------------------------------------------------------


def test_arena_market_resolves_every_victim_route(market):
    assert [p.pool_id for p in arena_market()] == list(ARENA_POOLS)
    for token_in, route in ARENA_ROUTES:
        tx = BuilderTx("r", "u", 0, 1, 1, 1, TxKind.SWAP, {"route": list(route), "token_in": token_in})
        hops = _route_hops(market.pools, tx)
        assert hops is not None and len(hops) == len(route) <= 2
    for token in ("USDC", "DAI"):
        assert native_value(market.pools, token, 3_000 * 10**(6 if token == "USDC" else 18)) == pytest.approx(
            E18, rel=0.01)


def arena_bots(n, seed=0):
    return [spawn_bot(decode_profile(seeded_generator("arena-bots", seed, i).uniform(0, 1, N_GENES), f"a{i}"))
            for i in range(n)]


def run_arena(n=6, seed=1, slots=4, incumbent=True, **cfg):
    arena = BotArena(arena_bots(n, seed), ArenaConfig(seed=seed, slots=slots, **cfg),
                     IncumbentPolicy.default() if incumbent else None)
    return arena, arena.run()


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_arena_heat_keeps_every_queue_invariant(seed):
    arena, out = run_arena(seed=seed, slots=5)
    assert out["violation_count"] == 0 and out["checks"] >= 5 * 3
    q = out["queue"]
    assert q["blocks"] == 5 and q["victims"] == 15 and q["submitted"] == q["arrived"] + q["late"]
    assert q["included"] <= q["victims"] and q["max_gas_used"] <= 30_000_000
    assert q["arrived"] - q["included"] == sum(q["evictions"].values())


def test_arena_ledgers_balance(seed=4):
    _, out = run_arena(n=8, seed=seed, slots=6)
    rows = out["bots"] + [out["incumbent"]]
    for r in rows:
        assert r["submitted"] == r["arrived"] + r["late"] and r["arrived"] == r["won"] + r["evicted"]
        assert type(r["pnl_wei"]) is int and (r["won"] or r["pnl_wei"] == 0)
    assert sum(r["won"] for r in rows) == out["queue"]["included"]


def test_arena_runs_are_deterministic():
    _, a = run_arena(seed=9)
    _, b = run_arena(seed=9)
    _, c = run_arena(seed=10)
    assert a == b and a != c


def test_arena_without_bots_or_incumbent_still_builds_blocks():
    arena, out = run_arena(n=0, incumbent=False, slots=3)
    assert out["bots"] == [] and out["incumbent"] is None and out["queue"]["blocks"] == 3
    assert out["queue"]["submitted"] == 0 and out["violation_count"] == 0


def test_arena_rejects_duplicate_bot_names():
    bot = spawn_bot(make_profile(name="dup"))
    with pytest.raises(ConfigError):
        BotArena([bot, spawn_bot(make_profile(name="dup", share_bps=6_000))])


@pytest.mark.parametrize("kw", [{"regime": "chaos"}, {"slots": 0}, {"victims_per_slot": 0}, {"grid": 0},
                                {"max_frontrun_bps": 0}, {"cutoff_ms": (900, 100)}, {"gas_limit": 10},
                                {"base_fee_wei": 10**15}, {"native": "DAI"}])
def test_arena_config_validation(kw):
    with pytest.raises(ConfigError):
        ArenaConfig(**kw)


def test_late_bundles_never_reach_the_builder():
    slow = [spawn_bot(make_profile(name=f"slow{i}", latency_ms=5_000, jitter_ms=0)) for i in range(3)]
    out = BotArena(slow, ArenaConfig(seed=3, slots=4)).run()
    for r in out["bots"]:
        assert r["arrived"] == 0 and r["late"] == r["submitted"] and r["pnl_wei"] == 0


def test_a_fast_high_bidder_outcompetes_a_slow_cheap_one():
    fast = spawn_bot(make_profile(name="fast", share_bps=9_000, max_share_bps=9_500, latency_ms=20, jitter_ms=1))
    slow = spawn_bot(make_profile(name="slow", strategy="static_tip", share_bps=1_000, latency_ms=20, jitter_ms=1))
    out = BotArena([fast, slow], ArenaConfig(seed=11, slots=8, regime="calm")).run()
    rows = {r["name"]: r for r in out["bots"]}
    assert rows["fast"]["won"] > rows["slow"]["won"]


def test_incumbent_policy_from_a_trained_table():
    grid = action_grid()
    table = ActionTable(len(grid))
    key = StateBins().encode(Observation(0, 0, 100, 20, 3))
    table.row(key)[5] = 1.0
    table.counts(key)[5] = 4
    table.row((9, 9, 9, 9, 9))  # never visited
    inc = IncumbentPolicy.from_table(table)
    assert inc.source == "rl_table" and inc.greedy == ((key, 5),) and inc.fallback == 5
    assert inc.action(Observation(0, 0, 100, 20, 3)) == grid[5]
    assert inc.action(Observation(999, 999, 999, 999, 9)) == grid[5]
    assert IncumbentPolicy.from_table(ActionTable(len(grid))).source == "default"
    with pytest.raises(ConfigError):
        IncumbentPolicy.from_table(ActionTable(3))
    assert pickle.loads(pickle.dumps(inc)).action(Observation(0, 0, 100, 20, 3)) == grid[5]


def test_default_incumbent_plays_full_size_half_share():
    inc = IncumbentPolicy.default()
    assert inc.actions[inc.fallback] == Action(10_000, 5_000) and inc.to_dict()["states"] == 0


# ---------------------------------------------------------------------------
# Multi-agent resource allocation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_apportion_is_exact_and_within_one_of_the_quota(seed):
    rng = random.Random(seed)
    weights = [rng.randrange(0, 1_000) for _ in range(rng.randrange(1, 40))]
    total = rng.randrange(0, 10**6)
    shares = apportion(total, weights)
    assert sum(shares) == total and len(shares) == len(weights)
    W = sum(weights) or len(weights)
    for s, w in zip(shares, weights if sum(weights) else [1] * len(weights)):
        assert total * w // W <= s <= total * w // W + 1


def test_apportion_edge_cases():
    assert apportion(5, [0, 0]) == [3, 2] and apportion(0, []) == [] and apportion(7, [1]) == [7]
    assert apportion(3, [1, 1, 1, 1]) == [1, 1, 1, 0]
    for args in ((-1, [1]), (3, [-1, 2]), (2, []), (1.5, [1])):
        with pytest.raises(ConfigError):
            apportion(*args)


@pytest.mark.parametrize("seed", range(8))
def test_capital_allocation_conserves_the_budget(seed):
    rng = random.Random(100 + seed)
    requests = [rng.randrange(1, 500) * E18 for _ in range(rng.randrange(1, 64))]
    budget = rng.randrange(1, 100) * E18 * len(requests) // 4
    granted = allocate_capital(requests, budget, floor=E18)
    assert sum(granted) == min(budget, sum(requests))
    assert all(0 <= g <= r for g, r in zip(granted, requests))
    floor = min(E18, budget // len(requests))
    assert all(g >= min(r, floor) for g, r in zip(granted, requests))


def test_undersubscribed_capital_is_granted_in_full():
    assert allocate_capital([E18, 2 * E18], 10 * E18) == [E18, 2 * E18]
    with pytest.raises(ConfigError):
        allocate_capital([-1], 5)


def test_offspring_follow_linear_rank_weights():
    keys = [(5, 0, 0, 0), (9, 1, 1, 1), (0, 0, 3, 3), (9, 2, 0, 0), (-4, 0, 0, 1)]
    off = allocate_offspring(keys, 10, 3)
    assert list(off) == [3, 1, 0] and sum(off.values()) == 10
    assert list(off.values()) == sorted(off.values(), reverse=True) == [5, 3, 2]
    assert lineage(off) == [3] * 5 + [1] * 3 + [0] * 2
    with pytest.raises(ConfigError):
        allocate_offspring(keys, 10, 0)
    with pytest.raises(ConfigError):
        allocate_offspring(keys, 10, 6)


@pytest.mark.parametrize("n,elite", [(1, 1), (7, 2), (64, 16), (100, 25), (13, 13)])
def test_offspring_allocation_is_exact_for_any_population(n, elite):
    keys = [(random.Random(n).randrange(-10, 10), i % 3, 0, 0) for i in range(n)]
    off = allocate_offspring(keys, n, elite)
    assert sum(off.values()) == n and len(off) == elite and len(lineage(off)) == n
    ranked = rank_order(keys)
    assert list(off) == ranked[:elite]


def test_rank_order_is_lexicographic_with_index_ties():
    assert rank_order([3, 7, 7, -1]) == [1, 2, 0, 3]
    assert rank_order([(0, 0, 5, 9), (0, 1, 0, 0), (0, 0, 5, 9)]) == [1, 0, 2]
    assert fitness_key({"pnl_wei": 3, "won": 1, "arrived": 2, "submitted": 4}) == (3, 1, 2, 4)


@pytest.mark.parametrize("n,size", [(0, 4), (1, 4), (7, 3), (8, 8), (9, 8), (100, 8), (512, 8), (33, 1)])
def test_heats_cover_every_bot_once_in_balanced_groups(n, size):
    heats = plan_heats(n, size, seed=3, generation=1)
    flat = sorted(i for h in heats for i in h)
    assert flat == list(range(n))
    if n:
        sizes = [len(h) for h in heats]
        assert max(sizes) <= size and max(sizes) - min(sizes) <= 1 and len(heats) == -(-n // size)
    assert plan_heats(n, size, seed=3, generation=1) == heats


def test_heat_layout_changes_with_seed_and_generation():
    a = plan_heats(64, 8, seed=1, generation=0)
    assert a != plan_heats(64, 8, seed=2, generation=0) and a != plan_heats(64, 8, seed=1, generation=1)
    with pytest.raises(ConfigError):
        plan_heats(5, 0)


# ---------------------------------------------------------------------------
# The factory end to end
# ---------------------------------------------------------------------------


def test_factory_report_is_complete_and_clean(baseline_report):
    s = baseline_report
    json.dumps(s)
    assert s["invariants"]["ok"] and s["invariants"]["rules"] == list(FACTORY_RULES)
    assert s["invariants"]["checks"] > 100 and s["total_bots"] == 24 and len(s["history"]) == 2
    assert s["heats_per_generation"] == 3 and s["queue"]["blocks"] == 2 * 3 * 3
    assert sum(s["strategy_mix"].values()) == 12 and sum(s["hop_policy_mix"].values()) == 12
    assert s["engine"]["layers"] == [N_FEATURES + 4, 16, N_GENES] and len(s["top_bots"]) == 5
    assert s["features"] == RLMetrics.neutral().to_dict() and s["incumbent"]["source"] == "default"


def test_every_generation_deploys_distinct_strategies(baseline_report):
    for h in baseline_report["history"]:
        assert h["distinct_strategies"] == h["bots"] == 12 and h["heats"] == 3


def test_factory_runs_are_deterministic(baseline_report):
    again = small_factory().run().summary()
    other = small_factory(seed=6).run().summary()
    assert strip_timing(again) == strip_timing(baseline_report)
    assert strip_timing(other) != strip_timing(baseline_report)


def test_generation_zero_has_no_parents_and_later_ones_descend_from_the_elite():
    f = small_factory(bots=16, generations=3, elite_frac=0.25)
    pops = []
    parents, latents = None, None
    for g in range(3):
        pop = f.synthesize(g, parents, latents)
        pops.append(pop)
        keys = [(i % 5, 0, 0, 0) for i in range(16)]
        off = allocate_offspring(keys, 16, 4)
        parents, latents = lineage(off), pop.latents
        if g == 0:
            assert pop.parents == [None] * 16
        else:
            assert sorted(set(pop.parents)) == sorted(off)
    child, parent = pops[1].latents[0], pops[0].latents[pops[1].parents[0]]
    assert np.linalg.norm(child - parent) < np.linalg.norm(pops[0].latents[0] - pops[0].latents[1]) + 1.0


def test_synthesis_rejects_a_lineage_of_the_wrong_size():
    f = small_factory()
    pop = f.synthesize(0)
    with pytest.raises(ConfigError):
        f.synthesize(1, [0, 1], pop.latents)


def test_training_pulls_the_generator_toward_the_elite(baseline_report):
    for h in baseline_report["history"]:
        assert h["loss_after"] < h["loss_before"]
    f = small_factory(epochs=60)
    pop = f.synthesize(0)
    idx = list(range(4))
    before = float(np.mean((f.engine.forward(pop.inputs[idx]) - pop.genes[idx]) ** 2))
    f.engine.fit(pop.inputs[idx], pop.genes[idx], np.ones(4), 60)
    after = float(np.mean((f.engine.forward(pop.inputs[idx]) - pop.genes[idx]) ** 2))
    assert after < 0.5 * before


def test_sigma_follows_the_success_rate(baseline_report):
    hist = baseline_report["history"]
    assert hist[0]["success_rate"] is None and hist[0]["next_sigma"] == hist[0]["sigma"]
    h = hist[1]
    rate = h["success_rate"]
    assert 0 <= rate <= 1
    if rate > 0.2:
        assert h["next_sigma"] > h["sigma"]
    elif rate < 0.2:
        assert h["next_sigma"] < h["sigma"]


def test_capital_budget_oversubscription_is_apportioned_exactly():
    f = small_factory(capital_per_bot_wei=2 * E18, capital_floor_wei=E18 // 2)
    pop = f.synthesize(0)
    budget = 2 * E18 * 12
    assert pop.requested_capital_wei > budget and pop.granted_capital_wei == budget
    assert all(p.capital_wei <= r for p, r in zip(pop.profiles, pop.requested_capital))
    s = f.run().summary()
    assert s["allocation"]["oversubscribed_generations"] == 2 and s["invariants"]["ok"]
    assert s["allocation"]["capital_granted_wei"] == budget


def test_ingesting_rl_results_conditions_the_generator_and_sets_the_incumbent(rl_run):
    sim, report = rl_run
    f = small_factory()
    m = f.ingest(report, sim.last_train.table, policy_paths={"policy": {"mean_entropy_bits": 1.0,
                                                                        "max_entropy_bits": 4.0}})
    assert m.sources == ("reinforcement", "table", "policy_paths") and f.incumbent.source == "rl_table"
    assert np.array_equal(f.synthesize(0).inputs[:, :N_FEATURES], np.tile(m.vector(), (12, 1)))
    s = f.run().summary()
    assert s["incumbent"]["source"] == "rl_table" and s["features"]["pg_entropy"] == 0.25 and s["invariants"]["ok"]


def test_factory_with_zero_bots_is_a_no_op():
    s = small_factory(bots=0).run().summary()
    assert s["total_bots"] == 0 and s["queue"]["blocks"] == 0 and s["invariants"]["ok"] and s["top_bots"] == []


@pytest.mark.parametrize("kw", [{"bots": -1}, {"generations": 0}, {"heat_size": 0}, {"heat_slots": 0},
                                {"elite_frac": 0.0}, {"elite_frac": 1.5}, {"latent_dim": -1},
                                {"capital_per_bot_wei": -1}, {"regime": "chaos"}, {"init": "zeros"}])
def test_factory_rejects_bad_settings(kw):
    with pytest.raises(ConfigError):
        small_factory(**kw)


def test_for_config_maps_the_simulation_settings():
    cfg = SimulationConfig.for_mode("max", agent_bots=24, agent_generations=3, agent_heat_size=6,
                                    agent_heat_slots=5, seed=17, rl_regime="calm")
    f = AgentFactory.for_config(cfg)
    assert (f.bots, f.generations, f.heat_size, f.heat_slots, f.seed) == (24, 3, 6, 5, 17)
    assert f.arena.regime == "calm" and f.arena.slots == 5 and f.arena.gas_limit == cfg.block_gas_limit


def test_the_feedback_loop_makes_bots_more_competitive():
    s = AgentFactory(bots=48, generations=6, heat_size=8, heat_slots=6, seed=7).run().summary()
    first, last = s["history"][0], s["history"][-1]
    assert s["invariants"]["ok"]
    assert last["arrived"] > first["arrived"] and last["wins"] >= first["wins"]


def test_new_module_passes_the_integrity_code_scan():
    import simulator.ai_agent_factory as mod

    with open(mod.__file__, encoding="utf-8") as fh:
        assert scan_source(fh.read(), "ai_agent_factory.py") == []


# ---------------------------------------------------------------------------
# Concurrency: worker processes, shared pools, saturated cores
# ---------------------------------------------------------------------------


@needs_two_cpus
def test_results_do_not_depend_on_worker_processes(baseline_report, two_workers):
    rep = small_factory().run(executor=two_workers)
    assert rep.parallel and rep.workers == 2 and len(set(rep.worker_pids)) >= 1
    assert strip_timing(rep.summary()) == strip_timing(baseline_report)


@needs_two_cpus
def test_full_core_pool_gives_identical_results(baseline_report):
    with WorkerPool(policy="full", parallel_threshold=1) as wp:
        rep = small_factory().run(executor=wp)
    assert rep.workers == CPUS and strip_timing(rep.summary()) == strip_timing(baseline_report)


@needs_two_cpus
def test_owned_pool_by_worker_count(baseline_report):
    rep = small_factory().run(workers=2)
    assert strip_timing(rep.summary()) == strip_timing(baseline_report)


@needs_two_cpus
def test_concurrent_factories_share_one_pool(two_workers):
    seeds = [21, 22, 23]
    expected = {s: strip_timing(small_factory(seed=s).run().summary()) for s in seeds}
    got = {}
    errors = []

    def worker(seed):
        try:
            got[seed] = strip_timing(small_factory(seed=seed).run(executor=two_workers).summary())
        except Exception as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(s,)) for s in seeds]
    for t in threads:
        t.start()
    for t in threads:
        t.join(180)
    assert not errors and got == expected


@needs_two_cpus
def test_heats_stay_exact_while_another_pool_saturates_every_core(baseline_report):
    load = {}
    with WorkerPool(policy="full", parallel_threshold=1, instrument=True) as burner:
        t = threading.Thread(target=lambda: load.update(probe=saturation_probe(burner, 0.5, 2)))
        t.start()
        with WorkerPool(workers=2, policy="full", parallel_threshold=1) as wp:
            rep = small_factory().run(executor=wp)
        t.join(120)
    assert strip_timing(rep.summary()) == strip_timing(baseline_report)
    assert load["probe"]["tasks"] == 2 * CPUS


@needs_two_cpus
def test_hundreds_of_bots_deploy_in_parallel_heats(two_workers):
    f = AgentFactory(bots=200, generations=1, heat_size=8, heat_slots=2, seed=13, hidden=(16,), epochs=5)
    before = two_workers.telemetry()["parallel_tasks"]
    s = f.run(executor=two_workers).summary()
    assert s["heats_per_generation"] == 25 and two_workers.telemetry()["parallel_tasks"] - before == 25
    assert s["history"][0]["distinct_strategies"] == 200 and s["invariants"]["ok"]
    assert s["queue"]["blocks"] == 50 and s["queue"]["submitted"] > 0


# ---------------------------------------------------------------------------
# Controller, run matrices and the pre-flight audit
# ---------------------------------------------------------------------------


def test_max_preset_deploys_hundreds_of_bots_and_lower_modes_do_not():
    mx = SimulationConfig.for_mode("max")
    assert mx.agent_bots >= 256 and mx.agent_generations >= 2
    assert SimulationConfig.for_mode("moderate").agent_bots == SimulationConfig.for_mode("brutal").agent_bots == 0
    assert mx.to_dict()["agent_bots"] == mx.agent_bots


@pytest.mark.parametrize("kw", [{"agent_bots": -1}, {"agent_generations": 0}, {"agent_heat_size": 0},
                                {"agent_heat_slots": 0}, {"agent_bots": 10**9}])
def test_config_rejects_bad_factory_settings(kw):
    with pytest.raises(ConfigError):
        SimulationConfig.for_mode("max", **kw)


def _signal_rules(cfg):
    col = Collector()
    check_signals(cfg, col, 4)
    return {f.rule for f in col.findings}


def test_validator_flags_factory_settings():
    cfg = SimulationConfig.for_mode("brutal", cpus=4)
    object.__setattr__(cfg, "agent_generations", 0)
    assert "signal.agent_factory" in _signal_rules(cfg)
    assert "signal.agent_factory_ignored" in _signal_rules(SimulationConfig.for_mode("moderate", cpus=4,
                                                                                     agent_bots=4))
    assert not {"signal.agent_factory", "signal.agent_factory_ignored"} & _signal_rules(
        SimulationConfig.for_mode("max", cpus=4))


def _small(mode, **kw):
    base = dict(blocks=1, txs_per_block=6, frontrun_grid=4, refine_iterations=2, fuzz_iterations=8,
                layout_fuzz_iterations=4, rl_rounds=1, rl_steps_per_shard=4, rl_eval_steps=6, pg_paths=8,
                pg_path_steps=2, pg_batch=4, shard_density=1, private_flow_bps=0, agent_bots=8,
                agent_generations=2, agent_heat_size=4, agent_heat_slots=2, pattern_synthetic=64,
                pattern_limit=48)
    base.update(kw)
    return SimulationConfig.for_mode(mode, **base)


def test_controller_runs_the_factory_after_the_rl_stages():
    out = SimulationController(_small("max")).run()
    af = out["result"]["agent_factory"]
    assert invariants_ok(out) and af["invariants"]["ok"] and af["total_bots"] == 16
    assert af["feature_sources"] == ["reinforcement", "table", "policy_paths"]
    assert af["incumbent"]["source"] in ("rl_table", "default")
    json.dumps(out)


def test_brutal_mode_runs_the_factory_when_asked():
    out = SimulationController(_small("brutal", pg_paths=0)).run()
    assert out["result"]["agent_factory"]["feature_sources"] == ["reinforcement", "table"]


def test_moderate_scan_ignores_the_factory():
    cfg = SimulationConfig.for_mode("moderate", txs_per_block=4, fuzz_iterations=8, agent_bots=8)
    out = SimulationController(cfg).run()
    assert "agent_factory" not in out["result"] and "signal.agent_factory_ignored" in out["audit"]["by_rule"]


def test_a_failed_factory_invariant_fails_the_run():
    report = {"result": {"invariants": {"ok": True}, "agent_factory": {"invariants": {"ok": False}}}}
    assert not invariants_ok(report)
