"""Layer 11: local embedded AI agent factory (machine-generated competitor bots).

The factory scripts, mutates and spins up its own competitor searcher bots at
runtime, deploys hundreds of them at once against the Layer 08 RL policy on
the worker pool, and learns from how they did. Nothing leaves the process:
the neural network is a few NumPy matrices, the market is the simulator's
own in-memory pools, and every profit figure is an exact integer of wei.

Neural matrix engine (:class:`NeuralMatrixEngine`)
--------------------------------------------------
A small fully connected network written directly on NumPy matrices (no deep
learning framework): ``tanh`` hidden layers, a ``sigmoid`` output layer,
hand-written backpropagation, weighted mean-squared error, and an AdamW
optimiser (:class:`AdamOptimizer`) with global gradient-norm clipping and
decoupled weight decay. Weights are initialised per layer with one of
:data:`INIT_SCHEMES` (Glorot/Xavier uniform, He normal, LeCun normal or
orthogonal) from a seeded ``numpy.random.Generator``, so an engine is a pure
function of its seed. :meth:`NeuralMatrixEngine.numerical_gradient` checks
the analytic gradient by central differences.

RL feedback (:class:`RLMetrics`)
--------------------------------
The factory ingests what the Layer 08 learner produced: the
:class:`~simulator.reinforcement_engine.ReinforcementSimulation` report (win
rates, reward trend, evictions, late bundles, overbidding, how often the
rival bots won), the trained :class:`~simulator.reinforcement_engine.ActionTable`
(greedy front-run size and builder bid per state) and, in MAX mode, the
Layer 09 policy-path summary (policy entropy). These become
:data:`FEATURE_NAMES`, a fixed vector in ``[-1, 1]``. The trained table also
becomes the :class:`IncumbentPolicy`, which plays its greedy action in every
arena heat, so every generated bot is measured against the RL agent itself.

Bot synthesis (:class:`BotProfile`)
-----------------------------------
The generator maps ``[RL features, latent noise]`` to :data:`GENE_NAMES`
(23 genes in ``[0, 1]``), :class:`AdaptiveMutation` perturbs them, and
:func:`decode_profile` turns genes into a strategy:

* bidding: one of the four :class:`~simulator.block_builder.CompetitorStrategy`
  rules, a builder share with its adaptive band and step,
* gas: a priority-fee range ``[tip_min_wei, tip_max_wei]`` and an escalation
  step (the tip rises after every lost bundle and falls after a win),
* slippage: the tightest victim tolerance the bot attacks
  (``slippage_floor_bps``) and the largest front-run as bps of the pool's
  input reserve (``max_impact_bps``),
* paths: the longest victim route it attacks (``max_hops``) and which hop it
  sandwiches (:data:`HOP_POLICIES`: oracle-best, first, last, deepest or
  shallowest pool),
* bundles: the pools it targets, bundles per block (one EOA each), front-leg
  ``amount_out_min`` slack, relay latency and jitter, and requested capital.

:func:`bot_class` spins up a dedicated Python class per profile (a subclass
of :class:`SynthesizedBot` created with ``type()``); profiles travel to
worker processes as plain data and the classes are recreated there, so bots
pickle under ``spawn``. A synthesized bot is a drop-in competitor for the
Layer 08 :class:`~simulator.reinforcement_engine.SandwichEnvironment` too.

Arena and block-building queue (:class:`BotArena`)
--------------------------------------------------
A heat is ``heat_slots`` blocks on a three-pool v2 market (WETH/USDC,
USDC/DAI, WETH/DAI) with one- and two-hop victim swaps. Each block every
bot plans a bundle for each victim it is willing to attack, keeps its best
``bundle_slots``, and samples one relay delay; the incumbent RL policy does
the same. Bundles that miss the builder's (random) cutoff are late; the rest
go into one block-building queue that the Layer 04 greedy builder orders and
seals under the block gas limit, so at most one sandwich per victim lands
and every other bundle is evicted with a reason. P&L is each bot's exact
balance change (tokens valued in the native token at pre-block spot).

Resource allocation
-------------------
* Compute: :func:`plan_heats` cuts each generation into fixed, balanced heats
  (sizes differ by at most one) from a seeded shuffle. A heat is one worker
  task, and the layout never depends on the worker count, so results are
  identical on one worker or on every core.
* Capital: :func:`allocate_capital` grants requested capital in full when the
  population budget allows it and otherwise apportions the budget exactly
  (a floor for everyone, then largest remainders), never above a request.
* Offspring: :func:`allocate_offspring` splits the next generation's
  ``bots`` slots over the elite by linear rank weights (largest remainder,
  exact). Those counts are the sample weights the generator trains on.

Feedback loop
-------------
Per generation: synthesise -> deploy every heat on the worker pool -> rank
bots by P&L -> train the generator on the elite (``(input, deployed genes)``
pairs weighted by offspring counts) -> adapt the mutation step with the 1/5
success rule, where success means beating the incumbent RL policy in the
same heat. The engine's weights, the genes, the profiles and the
allocations are checked against :data:`FACTORY_RULES` on every generation.

Limits: v2 pools only in the arena, victims of at most two hops, a greedy
builder, sampled latencies, and a generator conditioned on one RL report per
run (its features are constant within a run; diversity comes from the
latent noise and the mutation).
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import threading
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type

import numpy as np

from .block_builder import (
    BLOCK_GAS_LIMIT,
    ETH,
    GWEI,
    BuilderState,
    BuilderTx,
    Bundle,
    CompetitorBot,
    CompetitorStrategy,
    SandwichCandidate,
    SandwichQuote,
    TxKind,
    _quote_hops,
    _route_hops,
    clamp_gas_limit,
    make_sandwich_bundle,
    native_value,
    run_competition,
    sandwich_economics,
    simulate_route_sandwich,
)
from .config import ConfigError
from .invariants import input_reserve_scale
from .mempool_emulator import estimate_swap_gas, geometric_grid
from .parallel import WorkerPool, chunked
from .reinforcement_engine import REGIMES, Action, ActionTable, Observation, StateBins, _seed_int, action_grid
from .slippage_engine import V2Pool, _direction, min_amount_out
from .state_fork import SwapReverted

E18 = 10**18
E6 = 10**6
BPS = 10_000

# ---------------------------------------------------------------------------
# Constants: features, genes, arena
# ---------------------------------------------------------------------------

ACTIVATIONS: Tuple[str, ...] = ("tanh", "relu", "sigmoid", "linear")
INIT_SCHEMES: Tuple[str, ...] = ("xavier", "he", "lecun", "orthogonal")

FEATURE_NAMES: Tuple[str, ...] = (
    "train_win_rate",
    "eval_win_rate",
    "reward_trend",
    "eval_reward_scale",
    "regime_calm",
    "regime_volatile",
    "regime_extreme",
    "greedy_size",
    "greedy_bid",
    "greedy_skip",
    "eval_evicted_share",
    "eval_late_share",
    "overbid_scale",
    "rival_win_share",
    "policy_coverage",
    "pg_entropy",
)
N_FEATURES = len(FEATURE_NAMES)

ARENA_NATIVE = "WETH"
ARENA_POOLS: Tuple[str, ...] = ("v2:WETH/USDC", "v2:USDC/DAI", "v2:WETH/DAI")
# (token_in, route) of every victim swap the arena generates: six single hops and four two-hop routes.
ARENA_ROUTES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("WETH", ("v2:WETH/USDC",)),
    ("USDC", ("v2:WETH/USDC",)),
    ("USDC", ("v2:USDC/DAI",)),
    ("DAI", ("v2:USDC/DAI",)),
    ("WETH", ("v2:WETH/DAI",)),
    ("DAI", ("v2:WETH/DAI",)),
    ("WETH", ("v2:WETH/USDC", "v2:USDC/DAI")),
    ("DAI", ("v2:USDC/DAI", "v2:WETH/USDC")),
    ("USDC", ("v2:WETH/USDC", "v2:WETH/DAI")),
    ("DAI", ("v2:WETH/DAI", "v2:WETH/USDC")),
)
MAX_ARENA_HOPS = 2
VICTIM_TOLERANCES: Tuple[int, ...] = (30, 50, 100, 200, 300, 500, 1_000)

STRATEGIES: Tuple[CompetitorStrategy, ...] = (
    CompetitorStrategy.STATIC_TIP,
    CompetitorStrategy.FIXED_SHARE,
    CompetitorStrategy.STATIC_ROUTER,
    CompetitorStrategy.ADAPTIVE,
)
STRATEGY_VALUES: Tuple[str, ...] = tuple(s.value for s in STRATEGIES)
HOP_POLICIES: Tuple[str, ...] = ("oracle", "first", "last", "deepest", "shallowest")
MAX_BUNDLE_SLOTS = 3

GENE_NAMES: Tuple[str, ...] = (
    "strategy_static_tip",
    "strategy_fixed_share",
    "strategy_static_router",
    "strategy_adaptive",
    "share",
    "share_step",
    "share_band",
    "tip_floor",
    "tip_ceiling_ratio",
    "tip_escalation",
    "frontrun_size",
    "max_impact",
    "slippage_floor",
    "max_hops",
    "hop_policy",
    "pool_weth_usdc",
    "pool_usdc_dai",
    "pool_weth_dai",
    "bundle_slots",
    "front_slack",
    "latency",
    "jitter",
    "capital",
)
N_GENES = len(GENE_NAMES)
G = {name: i for i, name in enumerate(GENE_NAMES)}

# Decoded ranges (inclusive). Log-scaled values are quantised so encode -> decode is exact.
SHARE_RANGE = (500, 9_800)
SHARE_STEP_RANGE = (50, 1_500)
SHARE_BAND_RANGE = (0, 4_000)
TIP_MIN_RANGE = (GWEI // 10, 20 * GWEI)
TIP_RATIO_RANGE = (1.0, 10.0)
TIP_CAP_WEI = 200 * GWEI
TIP_QUANTUM = 10**7  # 0.01 gwei
TIP_STEP_RANGE = (0, 3_000)
SIZE_RANGE = (3_000, 15_000)
IMPACT_RANGE = (5, 2_000)
SLIPPAGE_FLOOR_RANGE = (0, 1_000)
FRONT_SLACK_RANGE = (0, 300)
LATENCY_RANGE = (60, 700)
JITTER_RANGE = (5, 300)
CAPITAL_RANGE = (E18, 500 * E18)
CAPITAL_QUANTUM = 10**15  # 0.001 ETH
CAPITAL_MAX_WEI = CAPITAL_RANGE[1]

INCUMBENT = "rl:incumbent"
BOT_PREFIX = "ai:"
VIOLATION_LIMIT = 64
FACTORY_RULES: Tuple[str, ...] = (
    "ai.finite_weights",
    "ai.gene_bounds",
    "ai.profile_bounds",
    "ai.unique_bots",
    "ai.heat_coverage",
    "ai.allocation_conservation",
    "ai.sigma_bounds",
    "ai.block_gas",
    "ai.single_winner",
    "ai.exact_pnl",
    "ai.idle_balances",
)


# ---------------------------------------------------------------------------
# Neural matrix engine
# ---------------------------------------------------------------------------


def seeded_generator(*parts: Any) -> np.random.Generator:
    """A PCG64 generator whose stream depends only on ``parts``."""
    return np.random.Generator(np.random.PCG64(_seed_int("ai-factory", *parts)))


def sigmoid(z: Any) -> np.ndarray:
    """Logistic function without overflow: ``1 / (1 + e^-z) = (1 + tanh(z / 2)) / 2``."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=np.float64)))


def activate(name: str, z: np.ndarray) -> np.ndarray:
    if name == "tanh":
        return np.tanh(z)
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "sigmoid":
        return sigmoid(z)
    if name == "linear":
        return np.array(z, dtype=np.float64, copy=True)
    raise ConfigError(f"unknown activation {name!r} (expected one of {', '.join(ACTIVATIONS)})")


def activation_grad(name: str, z: np.ndarray, a: np.ndarray) -> np.ndarray:
    """d activation / d z, from the pre-activation ``z`` and the output ``a``."""
    if name == "tanh":
        return 1.0 - a * a
    if name == "relu":
        return (z > 0).astype(np.float64)
    if name == "sigmoid":
        return a * (1.0 - a)
    if name == "linear":
        return np.ones_like(z)
    raise ConfigError(f"unknown activation {name!r}")


def init_std(fan_in: int, fan_out: int, scheme: str = "xavier") -> float:
    """Theoretical standard deviation of one weight under ``scheme``."""
    if fan_in < 1 or fan_out < 1:
        raise ConfigError("fan_in and fan_out must be >= 1")
    if scheme == "xavier":
        return math.sqrt(2.0 / (fan_in + fan_out))
    if scheme == "he":
        return math.sqrt(2.0 / fan_in)
    if scheme == "lecun":
        return math.sqrt(1.0 / fan_in)
    if scheme == "orthogonal":
        return math.sqrt(1.0 / max(fan_in, fan_out))
    raise ConfigError(f"unknown init scheme {scheme!r} (expected one of {', '.join(INIT_SCHEMES)})")


def init_matrix(fan_in: int, fan_out: int, scheme: str = "xavier",
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """A ``fan_in x fan_out`` weight matrix.

    * ``xavier``: Glorot uniform on ``[-a, a]``, ``a = sqrt(6 / (fan_in + fan_out))``.
    * ``he``: normal with std ``sqrt(2 / fan_in)`` (for ReLU layers).
    * ``lecun``: normal with std ``sqrt(1 / fan_in)``.
    * ``orthogonal``: orthonormal columns (or rows when ``fan_in < fan_out``),
      from the QR factorisation of a Gaussian matrix with the sign fixed by
      ``diag(R)`` so the distribution is uniform (Haar).
    """
    std = init_std(fan_in, fan_out, scheme)
    gen = rng if rng is not None else seeded_generator("init", fan_in, fan_out, scheme)
    if scheme == "xavier":
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return gen.uniform(-limit, limit, size=(fan_in, fan_out))
    if scheme in ("he", "lecun"):
        return gen.normal(0.0, std, size=(fan_in, fan_out))
    rows, cols = max(fan_in, fan_out), min(fan_in, fan_out)
    q, r = np.linalg.qr(gen.standard_normal((rows, cols)))
    q = q * np.copysign(1.0, np.diag(r))
    return q if fan_in >= fan_out else q.T


@dataclass
class AdamOptimizer:
    """AdamW: Adam moments, decoupled weight decay and global gradient-norm clipping.

    A step whose gradient norm is not finite is skipped (the weights stay as
    they were and the step counter does not advance).
    """

    lr: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-4
    clip_norm: float = 5.0
    t: int = 0
    skipped: int = 0
    m: List[np.ndarray] = field(default_factory=list, repr=False)
    v: List[np.ndarray] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not (self.lr > 0 and math.isfinite(self.lr)):
            raise ConfigError("lr must be positive and finite")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1):
            raise ConfigError("beta1 and beta2 must be within [0, 1)")
        if not self.eps > 0 or self.weight_decay < 0 or self.clip_norm < 0:
            raise ConfigError("eps must be positive, weight_decay and clip_norm non-negative")

    def reset(self) -> None:
        self.t = 0
        self.m, self.v = [], []

    def step(self, params: Sequence[np.ndarray], grads: Sequence[np.ndarray]) -> float:
        """Update ``params`` in place; returns the pre-clip global gradient norm."""
        if len(params) != len(grads):
            raise ConfigError("params and grads differ in length")
        if not self.m:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads))
        if not math.isfinite(norm):
            self.skipped += 1
            return norm
        scale = self.clip_norm / norm if self.clip_norm > 0 and norm > self.clip_norm else 1.0
        self.t += 1
        c1 = 1.0 - self.beta1 ** self.t
        c2 = 1.0 - self.beta2 ** self.t
        for p, g, m, v in zip(params, grads, self.m, self.v):
            gs = g * scale
            m *= self.beta1
            m += (1.0 - self.beta1) * gs
            v *= self.beta2
            v += (1.0 - self.beta2) * gs * gs
            p -= self.lr * ((m / c1) / (np.sqrt(v / c2) + self.eps) + self.weight_decay * p)
        return norm


class NeuralMatrixEngine:
    """Fully connected network on NumPy matrices with manual backprop and AdamW.

    ``layer_sizes = (n_in, h1, ..., n_out)``; hidden layers use
    ``hidden_activation`` and the last layer ``output_activation``. Inputs are
    ``(n_in,)`` vectors or ``(batch, n_in)`` matrices.
    """

    def __init__(self, layer_sizes: Sequence[int], hidden_activation: str = "tanh",
                 output_activation: str = "sigmoid", init: str = "xavier", seed: int = 0,
                 optimizer: Optional[AdamOptimizer] = None) -> None:
        sizes = tuple(int(s) for s in layer_sizes)
        if len(sizes) < 2 or any(s < 1 for s in sizes):
            raise ConfigError("layer_sizes needs at least an input and an output size, all >= 1")
        for name in (hidden_activation, output_activation):
            if name not in ACTIVATIONS:
                raise ConfigError(f"unknown activation {name!r}")
        if init not in INIT_SCHEMES:
            raise ConfigError(f"unknown init scheme {init!r}")
        self.layer_sizes = sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.init = init
        self.seed = int(seed)
        self.activations: Tuple[str, ...] = tuple(
            [hidden_activation] * (len(sizes) - 2) + [output_activation])
        self.weights: List[np.ndarray] = [
            init_matrix(a, b, init, seeded_generator("weights", self.seed, i, a, b))
            for i, (a, b) in enumerate(zip(sizes, sizes[1:]))]
        self.biases: List[np.ndarray] = [np.zeros(b, dtype=np.float64) for b in sizes[1:]]
        self.optimizer = optimizer if optimizer is not None else AdamOptimizer()

    # -- shape ---------------------------------------------------------------

    @property
    def input_dim(self) -> int:
        return self.layer_sizes[0]

    @property
    def output_dim(self) -> int:
        return self.layer_sizes[-1]

    @property
    def n_layers(self) -> int:
        return len(self.weights)

    @property
    def num_params(self) -> int:
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases))

    def parameters(self) -> List[np.ndarray]:
        """``[W0, b0, W1, b1, ...]`` (the live arrays)."""
        out: List[np.ndarray] = []
        for w, b in zip(self.weights, self.biases):
            out.extend((w, b))
        return out

    # -- forward / backward --------------------------------------------------

    def _as_batch(self, x: Any) -> Tuple[np.ndarray, bool]:
        arr = np.asarray(x, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != self.input_dim:
            raise ConfigError(f"input must have shape (batch, {self.input_dim}), got {np.shape(x)}")
        return arr, single

    def _forward_cache(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        acts = [x]
        pre: List[np.ndarray] = []
        a = x
        for w, b, act in zip(self.weights, self.biases, self.activations):
            z = a @ w + b
            a = activate(act, z)
            pre.append(z)
            acts.append(a)
        return pre, acts

    def forward(self, x: Any) -> np.ndarray:
        batch, single = self._as_batch(x)
        _, acts = self._forward_cache(batch)
        return acts[-1][0] if single else acts[-1]

    __call__ = forward

    @staticmethod
    def _weights(n: int, sample_weights: Any) -> np.ndarray:
        if sample_weights is None:
            return np.ones(n, dtype=np.float64)
        w = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != n:
            raise ConfigError(f"sample_weights has {w.shape[0]} entries for {n} samples")
        if not np.all(np.isfinite(w)) or np.any(w < 0) or not float(np.sum(w)) > 0:
            raise ConfigError("sample_weights must be finite, non-negative and not all zero")
        return w

    def _targets(self, y: Any, n: int) -> np.ndarray:
        arr = np.asarray(y, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape != (n, self.output_dim):
            raise ConfigError(f"targets must have shape ({n}, {self.output_dim}), got {np.shape(y)}")
        return arr

    def loss(self, x: Any, y: Any, sample_weights: Any = None) -> float:
        """Weighted mean-squared error ``sum_i w_i mean_j (f(x_i)_j - y_ij)^2 / sum_i w_i``."""
        batch, _ = self._as_batch(x)
        target = self._targets(y, batch.shape[0])
        w = self._weights(batch.shape[0], sample_weights)
        pred = self._forward_cache(batch)[1][-1]
        per_sample = np.mean((pred - target) ** 2, axis=1)
        return float(np.sum(w * per_sample) / np.sum(w))

    def gradients(self, x: Any, y: Any, sample_weights: Any = None) -> Tuple[float, List[np.ndarray]]:
        """Loss and its gradient for every array of :meth:`parameters` (same order)."""
        batch, _ = self._as_batch(x)
        target = self._targets(y, batch.shape[0])
        w = self._weights(batch.shape[0], sample_weights)
        pre, acts = self._forward_cache(batch)
        pred = acts[-1]
        total = float(np.sum(w))
        diff = pred - target
        loss = float(np.sum(w * np.mean(diff * diff, axis=1)) / total)
        delta = (2.0 / (self.output_dim * total)) * diff * w[:, None]
        grads: List[np.ndarray] = []
        for layer in range(self.n_layers - 1, -1, -1):
            delta = delta * activation_grad(self.activations[layer], pre[layer], acts[layer + 1])
            gw = acts[layer].T @ delta
            gb = np.sum(delta, axis=0)
            grads.extend((gb, gw))
            if layer:
                delta = delta @ self.weights[layer].T
        grads.reverse()
        return loss, grads

    def train_step(self, x: Any, y: Any, sample_weights: Any = None) -> float:
        """One AdamW step on the full batch; returns the loss before the step."""
        loss, grads = self.gradients(x, y, sample_weights)
        self.optimizer.step(self.parameters(), grads)
        return loss

    def fit(self, x: Any, y: Any, sample_weights: Any = None, epochs: int = 40) -> List[float]:
        """``epochs`` full-batch steps; returns the loss history (one entry per step, then the final loss)."""
        if epochs < 0:
            raise ConfigError("epochs must be non-negative")
        history = [self.train_step(x, y, sample_weights) for _ in range(epochs)]
        history.append(self.loss(x, y, sample_weights))
        return history

    def numerical_gradient(self, x: Any, y: Any, sample_weights: Any = None, eps: float = 1e-6,
                           max_entries: int = 64) -> Tuple[List[float], List[float]]:
        """(analytic, central-difference) partial derivatives for up to ``max_entries`` weights."""
        _, grads = self.gradients(x, y, sample_weights)
        analytic: List[float] = []
        numeric: List[float] = []
        params = self.parameters()
        gen = seeded_generator("gradcheck", self.seed, max_entries)
        for _ in range(max_entries):
            k = int(gen.integers(len(params)))
            p, g = params[k], grads[k]
            idx = tuple(int(gen.integers(s)) for s in p.shape)
            old = float(p[idx])
            p[idx] = old + eps
            up = self.loss(x, y, sample_weights)
            p[idx] = old - eps
            down = self.loss(x, y, sample_weights)
            p[idx] = old
            analytic.append(float(g[idx]))
            numeric.append((up - down) / (2.0 * eps))
        return analytic, numeric

    # -- flat parameter vector, copies, mutation -------------------------------

    def flat(self) -> np.ndarray:
        return np.concatenate([p.ravel() for p in self.parameters()])

    def set_flat(self, vector: Any) -> None:
        vec = np.asarray(vector, dtype=np.float64).reshape(-1)
        if vec.shape[0] != self.num_params:
            raise ConfigError(f"expected {self.num_params} parameters, got {vec.shape[0]}")
        start = 0
        for p in self.parameters():
            end = start + p.size
            p[...] = vec[start:end].reshape(p.shape)
            start = end

    def copy(self) -> "NeuralMatrixEngine":
        twin = NeuralMatrixEngine(self.layer_sizes, self.hidden_activation, self.output_activation, self.init,
                                  self.seed, AdamOptimizer(**self._optimizer_settings()))
        twin.set_flat(self.flat())
        return twin

    def _optimizer_settings(self) -> Dict[str, float]:
        o = self.optimizer
        return {"lr": o.lr, "beta1": o.beta1, "beta2": o.beta2, "eps": o.eps, "weight_decay": o.weight_decay,
                "clip_norm": o.clip_norm}

    def perturbed(self, sigma: float, rng: np.random.Generator) -> "NeuralMatrixEngine":
        """A copy whose weights carry ``N(0, sigma^2)`` noise (neuro-evolution step)."""
        if not (sigma >= 0 and math.isfinite(sigma)):
            raise ConfigError("sigma must be finite and non-negative")
        twin = self.copy()
        twin.set_flat(self.flat() + sigma * rng.standard_normal(self.num_params))
        return twin

    def is_finite(self) -> bool:
        return all(bool(np.all(np.isfinite(p))) for p in self.parameters())

    def weight_norm(self) -> float:
        return float(np.linalg.norm(self.flat()))

    def digest(self) -> str:
        """SHA-256 of the architecture and the exact float64 weights."""
        h = hashlib.sha256(repr((self.layer_sizes, self.activations, self.init)).encode())
        for p in self.parameters():
            h.update(np.ascontiguousarray(p, dtype=np.float64).tobytes())
        return h.hexdigest()

    def to_payload(self) -> Dict[str, Any]:
        return {"layer_sizes": list(self.layer_sizes), "hidden_activation": self.hidden_activation,
                "output_activation": self.output_activation, "init": self.init, "seed": self.seed,
                "params": [float(x) for x in self.flat()]}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NeuralMatrixEngine":
        engine = cls(payload["layer_sizes"], payload["hidden_activation"], payload["output_activation"],
                     payload["init"], int(payload["seed"]))
        engine.set_flat(payload["params"])
        return engine

    def describe(self) -> Dict[str, Any]:
        return {"layers": list(self.layer_sizes), "params": self.num_params, "init": self.init,
                "hidden_activation": self.hidden_activation, "output_activation": self.output_activation,
                "optimizer": {"name": "adamw", **self._optimizer_settings(), "steps": self.optimizer.t,
                              "skipped_steps": self.optimizer.skipped},
                "weight_norm": round(self.weight_norm(), 9), "digest": self.digest()}


# ---------------------------------------------------------------------------
# Adaptive mutation
# ---------------------------------------------------------------------------


def reflect_unit(x: Any) -> np.ndarray:
    """Fold values into ``[0, 1]`` by reflecting at both edges (non-finite values become 0.5)."""
    arr = np.asarray(x, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, 0.5)
    y = np.mod(arr, 2.0)
    return np.where(y > 1.0, 2.0 - y, y)


@dataclass
class AdaptiveMutation:
    """Gaussian mutation whose step size follows Rechenberg's 1/5 success rule.

    ``update(rate)`` multiplies ``sigma`` by
    ``exp(damping * (rate - target) / (1 - target))``: above the target success
    rate the search widens, below it it narrows. ``sigma`` always stays in
    ``[sigma_min, sigma_max]``; a non-finite rate leaves it unchanged.
    """

    sigma: float = 0.12
    sigma_min: float = 0.01
    sigma_max: float = 0.45
    target_success: float = 0.2
    damping: float = 0.8
    history: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        values = (self.sigma, self.sigma_min, self.sigma_max, self.target_success, self.damping)
        if not all(math.isfinite(v) for v in values):
            raise ConfigError("mutation parameters must be finite")
        if not 0 < self.sigma_min <= self.sigma <= self.sigma_max:
            raise ConfigError("need 0 < sigma_min <= sigma <= sigma_max")
        if not 0 < self.target_success < 1 or self.damping <= 0:
            raise ConfigError("target_success must be within (0, 1) and damping positive")

    def factor(self, success_rate: float) -> float:
        rate = float(success_rate)
        if not math.isfinite(rate):
            return 1.0
        rate = min(1.0, max(0.0, rate))
        return math.exp(self.damping * (rate - self.target_success) / (1.0 - self.target_success))

    def update(self, success_rate: float) -> float:
        self.sigma = min(self.sigma_max, max(self.sigma_min, self.sigma * self.factor(success_rate)))
        self.history.append(self.sigma)
        return self.sigma

    def mutate_genes(self, genes: Any, rng: np.random.Generator) -> np.ndarray:
        """``genes + sigma * N(0, 1)``, reflected back into ``[0, 1]``."""
        arr = np.asarray(genes, dtype=np.float64)
        return reflect_unit(arr + self.sigma * rng.standard_normal(arr.shape))

    def mutate_vector(self, vector: Any, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        """Unbounded mutation (network weights): ``vector + scale * sigma * N(0, 1)``."""
        arr = np.asarray(vector, dtype=np.float64)
        return arr + scale * self.sigma * rng.standard_normal(arr.shape)

    def to_dict(self) -> Dict[str, Any]:
        return {"sigma": round(self.sigma, 9), "sigma_min": self.sigma_min, "sigma_max": self.sigma_max,
                "target_success": self.target_success, "damping": self.damping,
                "history": [round(s, 9) for s in self.history]}


# ---------------------------------------------------------------------------
# RL metric ingestion
# ---------------------------------------------------------------------------


def _num(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return min(hi, max(lo, x))


def signed_log_scale(wei: Any, unit: int = 10**15, decades: float = 4.0) -> float:
    """``sign(x) * min(1, log10(1 + |x| / unit) / decades)``: wei amounts squashed into ``[-1, 1]``."""
    x = _num(wei)
    if not x:
        return 0.0
    return math.copysign(min(1.0, math.log10(1.0 + abs(x) / unit) / decades), x)


def _dig(data: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(data, Mapping):
            return None
        data = data.get(key)
    return data


@dataclass(frozen=True)
class RLMetrics:
    """The RL feedback the generator is conditioned on (aligned with :data:`FEATURE_NAMES`)."""

    values: Tuple[float, ...]
    regime: str = "volatile"
    sources: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.values) != N_FEATURES:
            raise ConfigError(f"expected {N_FEATURES} features, got {len(self.values)}")
        if not all(isinstance(v, float) and math.isfinite(v) and -1.0 <= v <= 1.0 for v in self.values):
            raise ConfigError("features must be finite floats within [-1, 1]")
        if self.regime not in REGIMES:
            raise ConfigError(f"unknown regime {self.regime!r}")

    def vector(self) -> np.ndarray:
        return np.array(self.values, dtype=np.float64)

    def to_dict(self) -> Dict[str, float]:
        return {name: round(v, 9) for name, v in zip(FEATURE_NAMES, self.values)}

    @classmethod
    def neutral(cls, regime: str = "volatile") -> "RLMetrics":
        return cls.from_sources(regime=regime)

    @classmethod
    def from_sources(cls, reinforcement: Optional[Mapping[str, Any]] = None, table: Optional[ActionTable] = None,
                     actions: Optional[Sequence[Action]] = None, policy_paths: Optional[Mapping[str, Any]] = None,
                     regime: Optional[str] = None) -> "RLMetrics":
        """Features from a ReinforcementSimulation report, its trained table and a policy-path summary.

        Every source is optional; a missing or malformed field falls back to a
        neutral value, so the factory can run without any RL stage.
        """
        r = reinforcement if isinstance(reinforcement, Mapping) else {}
        regime = regime or (r.get("regime") if isinstance(r.get("regime"), str) else None) or "volatile"
        if regime not in REGIMES:
            raise ConfigError(f"unknown regime {regime!r}")
        f: Dict[str, float] = {}
        rates = _dig(r, "train", "round_win_rates")
        rates = rates if isinstance(rates, (list, tuple)) else []
        f["train_win_rate"] = _clip(_num(rates[-1]), 0.0, 1.0) if rates else 0.0
        f["eval_win_rate"] = _clip(_num(_dig(r, "eval", "win_rate")), 0.0, 1.0)
        rewards = _dig(r, "train", "round_rewards_wei")
        rewards = rewards if isinstance(rewards, (list, tuple)) else []
        f["reward_trend"] = signed_log_scale(_num(rewards[-1]) - _num(rewards[0])) if len(rewards) > 1 else 0.0
        steps = _num(_dig(r, "eval", "steps"))
        f["eval_reward_scale"] = signed_log_scale(_num(_dig(r, "eval", "cumulative_reward_wei")) / steps) \
            if steps > 0 else 0.0
        for name in ("calm", "volatile", "extreme"):
            f[f"regime_{name}"] = 1.0 if regime == name else 0.0
        size, bid, skip = 0.5, 0.5, 0.0
        if table is not None:
            grid = tuple(actions) if actions is not None else action_grid()
            if table.n_actions != len(grid):
                raise ConfigError("table width does not match the action grid")
            sizes: List[int] = []
            bids: List[int] = []
            skips = 0
            for key in sorted(table.values):
                if sum(table.visits.get(key, ())) <= 0:
                    continue
                act = grid[table.best(key)]
                if act.skip:
                    skips += 1
                else:
                    sizes.append(act.size_bps)
                    bids.append(act.bid_bps)
            visited = skips + len(sizes)
            if sizes:
                size = sum(sizes) / len(sizes) / 20_000
                bid = sum(bids) / len(bids) / BPS
            skip = skips / visited if visited else 0.0
        f["greedy_size"], f["greedy_bid"], f["greedy_skip"] = _clip(size, 0, 1), _clip(bid, 0, 1), _clip(skip, 0, 1)
        submitted = _num(_dig(r, "eval", "submitted"))
        f["eval_evicted_share"] = _clip(_num(_dig(r, "eval", "evicted")) / submitted, 0, 1) if submitted > 0 else 0.0
        f["eval_late_share"] = _clip(_num(_dig(r, "eval", "late")) / submitted, 0, 1) if submitted > 0 else 0.0
        f["overbid_scale"] = signed_log_scale(_dig(r, "eval", "mean_overbid_wei"))
        outcomes = _dig(r, "eval", "bot_outcomes")
        won = total = 0.0
        if isinstance(outcomes, Mapping):
            for counts in outcomes.values():
                if isinstance(counts, Mapping):
                    won += _num(counts.get("won"))
                    total += sum(_num(v) for v in counts.values())
        f["rival_win_share"] = _clip(won / total, 0, 1) if total > 0 else 0.0
        states = _num(r.get("policy_states"), float(len(table)) if table is not None else 0.0)
        f["policy_coverage"] = _clip(states / StateBins().size, 0, 1)
        mean_h = _num(_dig(policy_paths, "policy", "mean_entropy_bits"), -1.0)
        max_h = _num(_dig(policy_paths, "policy", "max_entropy_bits"), 0.0)
        f["pg_entropy"] = _clip(mean_h / max_h, 0, 1) if max_h > 0 and mean_h >= 0 else 1.0
        sources = tuple(name for name, present in (("reinforcement", bool(r)), ("table", table is not None),
                                                   ("policy_paths", isinstance(policy_paths, Mapping)))
                        if present)
        return cls(tuple(float(f[name]) for name in FEATURE_NAMES), regime, sources)


# ---------------------------------------------------------------------------
# Bot profiles: genes <-> strategy
# ---------------------------------------------------------------------------


def _lin(x: float, lo: int, hi: int) -> int:
    return min(hi, max(lo, lo + round((hi - lo) * float(x))))


def _lin_gene(v: int, lo: int, hi: int) -> float:
    return (v - lo) / (hi - lo) if hi > lo else 0.0


def _logf(x: float, lo: float, hi: float) -> float:
    return math.exp(math.log(lo) + float(x) * (math.log(hi) - math.log(lo)))


def _log_q(x: float, lo: int, hi: int, quantum: int) -> int:
    return min(hi, max(lo, round(_logf(x, lo, hi) / quantum) * quantum))


def _log_gene(v: float, lo: float, hi: float) -> float:
    return (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))


def _argmax(xs: Sequence[float]) -> int:
    """First index of the maximum (ties go to the lowest index)."""
    best = 0
    for i, x in enumerate(xs):
        if x > xs[best]:
            best = i
    return best


@dataclass(frozen=True)
class BotProfile:
    """A machine-generated competitor strategy (see the module docstring for every field)."""

    name: str
    strategy: str
    share_bps: int
    step_bps: int
    min_share_bps: int
    max_share_bps: int
    tip_min_wei: int
    tip_max_wei: int
    tip_step_bps: int
    size_bps: int
    max_impact_bps: int
    slippage_floor_bps: int
    max_hops: int
    hop_policy: str
    target_pools: Tuple[str, ...]
    bundle_slots: int
    front_slack_bps: int
    latency_ms: int
    jitter_ms: int
    capital_wei: int
    generation: int = 0

    def __post_init__(self) -> None:
        issues = profile_issues(self)
        if issues:
            raise ConfigError(f"invalid bot profile {self.name!r}: {'; '.join(issues)}")

    def strategy_key(self) -> Tuple[Any, ...]:
        """Every behavioural field (not the name, generation or granted capital)."""
        return (self.strategy, self.share_bps, self.step_bps, self.min_share_bps, self.max_share_bps,
                self.tip_min_wei, self.tip_max_wei, self.tip_step_bps, self.size_bps, self.max_impact_bps,
                self.slippage_floor_bps, self.max_hops, self.hop_policy, self.target_pools, self.bundle_slots,
                self.front_slack_bps, self.latency_ms, self.jitter_ms)

    def fingerprint(self) -> str:
        return hashlib.sha256(repr(self.strategy_key()).encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "generation": self.generation, "strategy": self.strategy,
                "share_bps": self.share_bps, "step_bps": self.step_bps, "min_share_bps": self.min_share_bps,
                "max_share_bps": self.max_share_bps, "tip_min_wei": self.tip_min_wei, "tip_max_wei": self.tip_max_wei,
                "tip_step_bps": self.tip_step_bps, "size_bps": self.size_bps, "max_impact_bps": self.max_impact_bps,
                "slippage_floor_bps": self.slippage_floor_bps, "max_hops": self.max_hops,
                "hop_policy": self.hop_policy, "target_pools": list(self.target_pools),
                "bundle_slots": self.bundle_slots, "front_slack_bps": self.front_slack_bps,
                "latency_ms": self.latency_ms, "jitter_ms": self.jitter_ms, "capital_wei": self.capital_wei,
                "fingerprint": self.fingerprint()}

    def describe(self) -> str:
        bid = "no builder share" if self.strategy in ("static_tip", "static_router") else (
            f"{self.share_bps / 100:.1f}% builder share [{self.min_share_bps / 100:.1f}-{self.max_share_bps / 100:.1f}%]")
        return (f"{self.strategy}: {bid}; tips {self.tip_min_wei / GWEI:.2f}-{self.tip_max_wei / GWEI:.2f} gwei "
                f"(+{self.tip_step_bps / 100:.1f}% per loss); front-run {self.size_bps / 100:.0f}% of the hop "
                f"optimum, <= {self.max_impact_bps / 100:.2f}% of reserves; victims with >= "
                f"{self.slippage_floor_bps / 100:.2f}% tolerance on <= {self.max_hops} hop(s), {self.hop_policy} hop; "
                f"pools {', '.join(self.target_pools)}; {self.bundle_slots} bundle(s)/block, "
                f"{self.front_slack_bps} bps front slack; {self.latency_ms}+-{self.jitter_ms} ms; "
                f"{self.capital_wei / E18:.3f} ETH")

    def competitor_kwargs(self) -> Dict[str, Any]:
        """Closest Layer 08 :class:`CompetitorBot` arguments (fixed tip at ``tip_min_wei``)."""
        return {"name": self.name, "strategy": self.strategy, "share_bps": self.share_bps,
                "tip_wei": self.tip_min_wei, "size_bps": self.size_bps, "latency_ms": self.latency_ms,
                "jitter_ms": self.jitter_ms, "step_bps": self.step_bps, "min_share_bps": self.min_share_bps,
                "max_share_bps": self.max_share_bps}


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def profile_issues(p: Any) -> List[str]:
    """Every bound a :class:`BotProfile` violates (empty when valid)."""
    issues: List[str] = []
    if not isinstance(p.name, str) or not p.name:
        issues.append("name must be a non-empty string")
    if p.strategy not in STRATEGY_VALUES:
        issues.append(f"strategy {p.strategy!r} is unknown")
    ints = ("share_bps", "step_bps", "min_share_bps", "max_share_bps", "tip_min_wei", "tip_max_wei", "tip_step_bps",
            "size_bps", "max_impact_bps", "slippage_floor_bps", "max_hops", "bundle_slots", "front_slack_bps",
            "latency_ms", "jitter_ms", "capital_wei", "generation")
    bad = [n for n in ints if not _is_int(getattr(p, n))]
    if bad:
        issues.append(f"not integers: {', '.join(bad)}")
        return issues
    if not 0 <= p.min_share_bps <= p.share_bps <= p.max_share_bps <= BPS:
        issues.append("need 0 <= min_share <= share <= max_share <= 10000")
    if not 0 <= p.step_bps <= BPS or not 0 <= p.tip_step_bps <= BPS:
        issues.append("step_bps and tip_step_bps must be within [0, 10000]")
    if not 0 < p.tip_min_wei <= p.tip_max_wei <= TIP_CAP_WEI:
        issues.append("need 0 < tip_min <= tip_max <= 200 gwei")
    if not 0 < p.size_bps <= 2 * BPS:
        issues.append("size_bps must be within (0, 20000]")
    if not 1 <= p.max_impact_bps <= BPS or not 0 <= p.slippage_floor_bps <= BPS or not 0 <= p.front_slack_bps <= BPS:
        issues.append("impact, slippage floor and front slack must be valid bps")
    if not 1 <= p.max_hops <= MAX_ARENA_HOPS:
        issues.append(f"max_hops must be within [1, {MAX_ARENA_HOPS}]")
    if p.hop_policy not in HOP_POLICIES:
        issues.append(f"hop_policy {p.hop_policy!r} is unknown")
    pools = tuple(p.target_pools) if isinstance(p.target_pools, tuple) else None
    if not pools or len(set(pools)) != len(pools) or any(x not in ARENA_POOLS for x in pools) \
            or list(pools) != sorted(pools, key=ARENA_POOLS.index):
        issues.append("target_pools must be a non-empty, ordered tuple of arena pools")
    if not 1 <= p.bundle_slots <= MAX_BUNDLE_SLOTS:
        issues.append(f"bundle_slots must be within [1, {MAX_BUNDLE_SLOTS}]")
    if not 0 <= p.latency_ms <= 10_000 or not 0 <= p.jitter_ms <= 10_000:
        issues.append("latency and jitter must be within [0, 10000] ms")
    if not 0 <= p.capital_wei <= CAPITAL_MAX_WEI:
        issues.append("capital must be within [0, 500 ETH]")
    if p.generation < 0:
        issues.append("generation must be non-negative")
    return issues


def decode_profile(genes: Any, name: str, generation: int = 0) -> BotProfile:
    """Turn 23 genes in ``[0, 1]`` into a :class:`BotProfile` (out-of-range genes are folded back)."""
    g = reflect_unit(np.asarray(genes, dtype=np.float64).reshape(-1))
    if g.shape[0] != N_GENES:
        raise ConfigError(f"expected {N_GENES} genes, got {g.shape[0]}")
    x = [float(v) for v in g]
    strategy = STRATEGIES[_argmax(x[0:4])].value
    share = _lin(x[G["share"]], *SHARE_RANGE)
    band = _lin(x[G["share_band"]], *SHARE_BAND_RANGE)
    tip_min = _log_q(x[G["tip_floor"]], TIP_MIN_RANGE[0], TIP_MIN_RANGE[1], TIP_QUANTUM)
    ratio = _logf(x[G["tip_ceiling_ratio"]], *TIP_RATIO_RANGE)
    tip_max = min(TIP_CAP_WEI, max(tip_min, round(tip_min * ratio / TIP_QUANTUM) * TIP_QUANTUM))
    pool_genes = [x[G["pool_weth_usdc"]], x[G["pool_usdc_dai"]], x[G["pool_weth_dai"]]]
    pools = tuple(pid for pid, v in zip(ARENA_POOLS, pool_genes) if v >= 0.5) or (ARENA_POOLS[_argmax(pool_genes)],)
    return BotProfile(
        name=name,
        strategy=strategy,
        share_bps=share,
        step_bps=_lin(x[G["share_step"]], *SHARE_STEP_RANGE),
        min_share_bps=max(0, share - band),
        max_share_bps=min(BPS, share + band),
        tip_min_wei=tip_min,
        tip_max_wei=tip_max,
        tip_step_bps=_lin(x[G["tip_escalation"]], *TIP_STEP_RANGE),
        size_bps=_lin(x[G["frontrun_size"]], *SIZE_RANGE),
        max_impact_bps=_log_q(x[G["max_impact"]], IMPACT_RANGE[0], IMPACT_RANGE[1], 1),
        slippage_floor_bps=_lin(x[G["slippage_floor"]], *SLIPPAGE_FLOOR_RANGE),
        max_hops=1 if x[G["max_hops"]] < 0.5 else 2,
        hop_policy=HOP_POLICIES[min(len(HOP_POLICIES) - 1, math.floor(x[G["hop_policy"]] * len(HOP_POLICIES)))],
        target_pools=pools,
        bundle_slots=1 + min(MAX_BUNDLE_SLOTS - 1, math.floor(x[G["bundle_slots"]] * MAX_BUNDLE_SLOTS)),
        front_slack_bps=_lin(x[G["front_slack"]], *FRONT_SLACK_RANGE),
        latency_ms=_lin(x[G["latency"]], *LATENCY_RANGE),
        jitter_ms=_lin(x[G["jitter"]], *JITTER_RANGE),
        capital_wei=_log_q(x[G["capital"]], CAPITAL_RANGE[0], CAPITAL_RANGE[1], CAPITAL_QUANTUM),
        generation=int(generation),
    )


def encode_profile(p: BotProfile) -> np.ndarray:
    """Genes that decode back to ``p`` (exactly, for any profile :func:`decode_profile` produced)."""
    x = np.zeros(N_GENES, dtype=np.float64)
    x[STRATEGY_VALUES.index(p.strategy)] = 1.0
    x[G["share"]] = _lin_gene(p.share_bps, *SHARE_RANGE)
    x[G["share_step"]] = _lin_gene(p.step_bps, *SHARE_STEP_RANGE)
    if p.min_share_bps > 0:
        band = p.share_bps - p.min_share_bps
    else:
        band = p.max_share_bps - p.share_bps
    x[G["share_band"]] = _lin_gene(band, *SHARE_BAND_RANGE)
    x[G["tip_floor"]] = _log_gene(p.tip_min_wei, *TIP_MIN_RANGE)
    x[G["tip_ceiling_ratio"]] = _log_gene(p.tip_max_wei / p.tip_min_wei, *TIP_RATIO_RANGE)
    x[G["tip_escalation"]] = _lin_gene(p.tip_step_bps, *TIP_STEP_RANGE)
    x[G["frontrun_size"]] = _lin_gene(p.size_bps, *SIZE_RANGE)
    x[G["max_impact"]] = _log_gene(p.max_impact_bps, *IMPACT_RANGE)
    x[G["slippage_floor"]] = _lin_gene(p.slippage_floor_bps, *SLIPPAGE_FLOOR_RANGE)
    x[G["max_hops"]] = 0.25 if p.max_hops == 1 else 0.75
    x[G["hop_policy"]] = (HOP_POLICIES.index(p.hop_policy) + 0.5) / len(HOP_POLICIES)
    for i, pid in enumerate(ARENA_POOLS):
        x[G["pool_weth_usdc"] + i] = 0.75 if pid in p.target_pools else 0.25
    x[G["bundle_slots"]] = (p.bundle_slots - 0.5) / MAX_BUNDLE_SLOTS
    x[G["front_slack"]] = _lin_gene(p.front_slack_bps, *FRONT_SLACK_RANGE)
    x[G["latency"]] = _lin_gene(p.latency_ms, *LATENCY_RANGE)
    x[G["jitter"]] = _lin_gene(p.jitter_ms, *JITTER_RANGE)
    x[G["capital"]] = _log_gene(max(p.capital_wei, CAPITAL_RANGE[0]), *CAPITAL_RANGE)
    return np.clip(x, 0.0, 1.0)


def profile_from_competitor(bot: CompetitorBot, generation: int = 0, capital_wei: int = 50 * E18) -> BotProfile:
    """A Layer 08 rival as a profile: fixed tip, oracle hop, single-hop victims on every arena pool."""
    tip = min(TIP_CAP_WEI, max(1, int(bot.tip_wei)))
    return BotProfile(
        name=bot.name, strategy=CompetitorStrategy(bot.strategy).value, share_bps=bot.share_bps,
        step_bps=min(BPS, bot.step_bps), min_share_bps=min(bot.min_share_bps, bot.share_bps),
        max_share_bps=max(bot.max_share_bps, bot.share_bps), tip_min_wei=tip, tip_max_wei=tip, tip_step_bps=0,
        size_bps=bot.size_bps, max_impact_bps=2_000, slippage_floor_bps=0, max_hops=1, hop_policy="oracle",
        target_pools=ARENA_POOLS, bundle_slots=1, front_slack_bps=0, latency_ms=min(10_000, bot.latency_ms),
        jitter_ms=min(10_000, bot.jitter_ms), capital_wei=capital_wei, generation=generation)


# ---------------------------------------------------------------------------
# Runtime bots (a class per profile)
# ---------------------------------------------------------------------------


def hop_candidate(pools: Mapping[str, Any], victim: BuilderTx, hop: int, grid: int = 10, refine: int = 5,
                  max_frontrun_bps: int = 2_000) -> Optional[SandwichCandidate]:
    """Best sandwich on one hop of the victim's route (grid + integer ternary refine), or None."""
    if victim.kind is not TxKind.SWAP or grid <= 0:
        return None
    hops = _route_hops(pools, victim)
    if hops is None or not 0 <= hop < len(hops):
        return None
    pid, tin, tout = hops[hop]
    pool = pools[pid]
    zfo, _ = _direction(pool, tin)
    hi = input_reserve_scale(pool, zfo) * max_frontrun_bps // BPS
    if hi < 1:
        return None
    lo = max(1, hi // 100_000)
    cache: Dict[int, Optional[Tuple[int, int, int]]] = {}

    def score(a: int) -> Tuple[int, int]:
        if a not in cache:
            cache[a] = simulate_route_sandwich(pools, victim, hop, a)
        r = cache[a]
        return (0, 0) if r is None else (1, r[2] - a)

    sizes = geometric_grid(lo, hi, grid)
    if not sizes:
        return None
    scores = [score(a) for a in sizes]
    i = max(range(len(sizes)), key=lambda k: (scores[k], -k))
    if scores[i][0] == 0:
        return None
    left = sizes[i - 1] if i > 0 else lo
    right = sizes[i + 1] if i + 1 < len(sizes) else hi
    for _ in range(refine):
        if right - left < 3:
            break
        m1, m2 = left + (right - left) // 3, right - (right - left) // 3
        if score(m1) < score(m2):
            left = m1
        else:
            right = m2
    valid = [(r[2] - a, -a) for a, r in cache.items() if r is not None]
    profit, neg_a = max(valid)
    if profit <= 0:
        return None
    a = -neg_a
    fo, vo, bo = cache[a]  # type: ignore[misc]
    return SandwichCandidate(victim.tx_hash, hop, pid, tin, tout, a, fo, vo, bo, len(hops) > 1)


class HopOracles:
    """Per-victim cache of :func:`hop_candidate` for each hop, shared by every bot in a block."""

    def __init__(self, pools: Mapping[str, Any], victim: BuilderTx, grid: int = 10, refine: int = 5,
                 max_frontrun_bps: int = 2_000, native: str = ARENA_NATIVE,
                 known: Optional[SandwichCandidate] = None) -> None:
        self.pools = pools
        self.victim = victim
        self.grid = grid
        self.refine = refine
        self.max_frontrun_bps = max_frontrun_bps
        self.native = native
        self.hops = _route_hops(pools, victim) or []
        self._cache: Dict[int, Optional[SandwichCandidate]] = {}
        if known is not None and 0 <= known.hop < len(self.hops):
            self._cache[known.hop] = known
        self.evaluations = 0

    def __len__(self) -> int:
        return len(self.hops)

    def __call__(self, hop: int) -> Optional[SandwichCandidate]:
        if hop not in self._cache:
            self.evaluations += 1
            self._cache[hop] = hop_candidate(self.pools, self.victim, hop, self.grid, self.refine,
                                             self.max_frontrun_bps)
        return self._cache[hop]

    def native_profit(self, cand: Optional[SandwichCandidate]) -> int:
        if cand is None:
            return 0
        return native_value(self.pools, cand.token_in, cand.profit, self.native)

    def best(self, hops: Optional[Iterable[int]] = None) -> Optional[SandwichCandidate]:
        """Most profitable candidate (native value) over ``hops`` (default: all); ties go to the lower hop."""
        best: Optional[SandwichCandidate] = None
        value = 0
        for h in (range(len(self.hops)) if hops is None else hops):
            cand = self(h)
            v = self.native_profit(cand)
            if cand is not None and (best is None or v > value):
                best, value = cand, v
        return best


@dataclass(frozen=True)
class BotPlan:
    quote: SandwichQuote
    bribe_wei: int
    hop: int


class SynthesizedBot:
    """A competitor driven by a :class:`BotProfile`; same interface as :class:`CompetitorBot`.

    Classes made by :func:`bot_class` set ``PROFILE`` so they can be
    instantiated without arguments.
    """

    PROFILE: Optional[BotProfile] = None

    def __init__(self, profile: Optional[BotProfile] = None) -> None:
        profile = profile if profile is not None else type(self).PROFILE
        if profile is None:
            raise ConfigError("a SynthesizedBot needs a profile")
        self.profile = profile
        self.name = profile.name
        self.strategy = CompetitorStrategy(profile.strategy)
        self.share_bps = profile.share_bps
        self.tip_wei = profile.tip_min_wei
        self.wins = 0
        self.losses = 0

    def __reduce__(self) -> Tuple[Any, ...]:
        return (spawn_bot, (self.profile,), dict(self.__dict__))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, tip={self.tip_wei}, share={self.share_bps})"

    @property
    def searcher(self) -> str:
        return f"{BOT_PREFIX}{self.name}"

    def account(self, slot: int = 0) -> str:
        """The EOA used for this bot's ``slot``-th bundle in a block (one nonce stream each)."""
        return self.searcher if slot == 0 else f"{self.searcher}#{slot}"

    def sample_delay(self, rng: random.Random) -> int:
        return max(0, int(rng.gauss(self.profile.latency_ms, self.profile.jitter_ms)))

    def choose_hop(self, pools: Mapping[str, Any], hops: Sequence[Tuple[str, str, str]], allowed: Sequence[int],
                   oracles: HopOracles) -> Optional[int]:
        policy = self.profile.hop_policy
        if not allowed:
            return None
        if policy == "first":
            return allowed[0]
        if policy == "last":
            return allowed[-1]
        if policy == "oracle":
            cand = oracles.best(allowed)
            return None if cand is None else cand.hop
        depth = []
        for h in allowed:
            pid, tin, _ = hops[h]
            zfo, _ = _direction(pools[pid], tin)
            depth.append((native_value(pools, tin, input_reserve_scale(pools[pid], zfo), oracles.native), h))
        if policy == "deepest":
            return max(depth, key=lambda d: (d[0], -d[1]))[1]
        return min(depth)[1]

    def plan(self, state: BuilderState, victim: BuilderTx, oracles: HopOracles, base_fee: int,
             native: str = ARENA_NATIVE) -> Optional[BotPlan]:
        """The quote and builder payment this bot would bid on ``victim``, or None when it passes."""
        p = self.profile
        if victim.kind is not TxKind.SWAP or victim.effective_tip(base_fee) is None:
            return None
        tolerance = int(victim.payload.get("tolerance_bps", BPS))
        if tolerance < p.slippage_floor_bps:
            return None
        hops = _route_hops(state.pools, victim)
        if not hops or len(hops) > p.max_hops:
            return None
        allowed = [h for h, (pid, _, _) in enumerate(hops) if pid in p.target_pools]
        if self.strategy is CompetitorStrategy.STATIC_ROUTER:
            if 0 not in allowed:
                return None
            hop = 0
            size = victim.amount_in * p.size_bps // BPS
        else:
            chosen = self.choose_hop(state.pools, hops, allowed, oracles)
            cand = None if chosen is None else oracles(chosen)
            if cand is None:
                return None
            hop = chosen  # type: ignore[assignment]
            size = cand.frontrun_in * p.size_bps // BPS
        pid, tin, _ = hops[hop]
        zfo, _ = _direction(state.pools[pid], tin)
        size = min(size, input_reserve_scale(state.pools[pid], zfo) * p.max_impact_bps // BPS)
        value = native_value(state.pools, tin, size, native)
        if value > p.capital_wei:
            size = size * p.capital_wei // value
        if size <= 0:
            return None
        q = sandwich_economics(state, victim, hop, size, base_fee, self.tip_wei, native)
        if q is None or q.net_wei <= 0:
            return None
        if self.strategy in (CompetitorStrategy.STATIC_TIP, CompetitorStrategy.STATIC_ROUTER):
            bribe = 0
        else:
            bribe = q.net_wei * self.share_bps // BPS
        return BotPlan(q, bribe, hop)

    def bundle(self, state: BuilderState, victim: BuilderTx, plan: BotPlan, base_fee: int, slot: int = 0) -> Bundle:
        tag = self.name if slot == 0 else f"{self.name}#{slot}"
        return make_sandwich_bundle(state, victim, plan.quote, base_fee, self.account(slot), plan.bribe_wei,
                                    self.tip_wei, tag=tag, min_out_slack_bps=self.profile.front_slack_bps)

    def build(self, state: BuilderState, victim: BuilderTx, oracle: Optional[SandwichCandidate], base_fee: int,
              native: str = ARENA_NATIVE, oracles: Optional[HopOracles] = None) -> Optional[Bundle]:
        """Layer 08 interface: the bot's bundle for ``victim``, or None."""
        if oracles is None:
            oracles = HopOracles(state.pools, victim, native=native, known=oracle)
        plan = self.plan(state, victim, oracles, base_fee, native)
        return None if plan is None else self.bundle(state, victim, plan, base_fee)

    def observe(self, won: bool) -> None:
        """Priority-gas escalation inside ``[tip_min, tip_max]`` and the adaptive share rule."""
        p = self.profile
        if won:
            self.wins += 1
            self.tip_wei = max(p.tip_min_wei, self.tip_wei * (BPS - p.tip_step_bps // 2) // BPS)
            if self.strategy is CompetitorStrategy.ADAPTIVE:
                self.share_bps = max(p.min_share_bps, self.share_bps - p.step_bps // 2)
        else:
            self.losses += 1
            self.tip_wei = min(p.tip_max_wei, self.tip_wei * (BPS + p.tip_step_bps) // BPS + (p.tip_step_bps > 0))
            if self.strategy is CompetitorStrategy.ADAPTIVE:
                self.share_bps = min(p.max_share_bps, self.share_bps + p.step_bps)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "class": type(self).__name__, "strategy": self.strategy.value,
                "share_bps": self.share_bps, "tip_wei": self.tip_wei, "wins": self.wins, "losses": self.losses}


_CLASS_CACHE: Dict[BotProfile, Type[SynthesizedBot]] = {}
_CLASS_LOCK = threading.Lock()
CLASS_CACHE_LIMIT = 8_192


def class_name(profile: BotProfile) -> str:
    camel = "".join(part.capitalize() for part in profile.strategy.split("_"))
    return f"AIBot_{camel}_{profile.hop_policy.capitalize()}_{profile.fingerprint()[:8]}"


def bot_class(profile: BotProfile) -> Type[SynthesizedBot]:
    """A dedicated subclass of :class:`SynthesizedBot` for ``profile``, created once per process."""
    with _CLASS_LOCK:
        cls = _CLASS_CACHE.get(profile)
        if cls is None:
            if len(_CLASS_CACHE) >= CLASS_CACHE_LIMIT:
                _CLASS_CACHE.clear()
            cls = type(class_name(profile), (SynthesizedBot,),
                       {"PROFILE": profile, "__module__": __name__, "__doc__": profile.describe()})
            _CLASS_CACHE[profile] = cls
        return cls


def spawn_bot(profile: BotProfile) -> SynthesizedBot:
    """Instantiate the profile's runtime class (also the unpickling hook for bot instances)."""
    return bot_class(profile)()


# ---------------------------------------------------------------------------
# The arena: one heat of bots, the incumbent RL policy and the block-building queue
# ---------------------------------------------------------------------------


def arena_market() -> List[V2Pool]:
    """Three v2 pools at $3000/WETH: WETH/USDC, USDC/DAI and WETH/DAI (18-decimal DAI, 6-decimal USDC)."""
    return [
        V2Pool("WETH", "USDC", 4_000 * E18, 12_000_000 * E6),
        V2Pool("USDC", "DAI", 9_000_000 * E6, 9_000_000 * E18),
        V2Pool("WETH", "DAI", 2_500 * E18, 7_500_000 * E18),
    ]


@dataclass(frozen=True)
class IncumbentPolicy:
    """The trained RL table's greedy action per state (the benchmark every generated bot faces)."""

    actions: Tuple[Action, ...]
    greedy: Tuple[Tuple[Tuple[int, ...], int], ...] = ()
    fallback: int = 0
    tip_wei: int = GWEI
    source: str = "default"

    def __post_init__(self) -> None:
        if not self.actions:
            raise ConfigError("the incumbent needs at least one action")
        if not 0 <= self.fallback < len(self.actions) or any(not 0 <= a < len(self.actions) for _, a in self.greedy):
            raise ConfigError("incumbent action index out of range")
        object.__setattr__(self, "_lookup", {tuple(k): a for k, a in self.greedy})

    @classmethod
    def default(cls, actions: Optional[Sequence[Action]] = None) -> "IncumbentPolicy":
        grid = tuple(actions) if actions is not None else action_grid()
        target = Action(BPS, 5_000)
        fallback = grid.index(target) if target in grid else len(grid) - 1
        return cls(grid, (), fallback, GWEI, "default")

    @classmethod
    def from_table(cls, table: ActionTable, actions: Optional[Sequence[Action]] = None) -> "IncumbentPolicy":
        grid = tuple(actions) if actions is not None else action_grid()
        if table.n_actions != len(grid):
            raise ConfigError("table width does not match the action grid")
        greedy = tuple((tuple(k), table.best(k)) for k in sorted(table.values)
                       if sum(table.visits.get(k, ())) > 0)
        if not greedy:
            return cls.default(grid)
        counts = Counter(a for _, a in greedy)
        fallback = min(counts, key=lambda a: (-counts[a], a))
        return cls(grid, greedy, fallback, GWEI, "rl_table")

    def action(self, obs: Observation, bins: Optional[StateBins] = None) -> Action:
        key = (bins or StateBins()).encode(obs)
        return self.actions[self._lookup.get(key, self.fallback)]  # type: ignore[attr-defined]

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "states": len(self.greedy), "fallback": self.actions[self.fallback].label()}


@dataclass(frozen=True)
class ArenaConfig:
    seed: int = 0
    regime: str = "volatile"
    slots: int = 12
    victims_per_slot: int = 3
    grid: int = 10
    refine: int = 5
    max_frontrun_bps: int = 2_000
    base_fee_wei: int = 20 * GWEI
    min_base_fee_wei: int = GWEI
    max_base_fee_wei: int = 500 * GWEI
    cutoff_ms: Tuple[int, int] = (250, 900)
    gas_limit: int = BLOCK_GAS_LIMIT
    native: str = ARENA_NATIVE

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ConfigError(f"unknown regime {self.regime!r}")
        if self.slots < 1 or self.victims_per_slot < 1 or self.grid < 1 or self.refine < 0:
            raise ConfigError("slots, victims_per_slot and grid must be >= 1, refine >= 0")
        if not 0 < self.max_frontrun_bps <= BPS:
            raise ConfigError("max_frontrun_bps must be within (0, 10000]")
        if not 0 < self.min_base_fee_wei <= self.base_fee_wei <= self.max_base_fee_wei:
            raise ConfigError("need 0 < min_base_fee <= base_fee <= max_base_fee")
        lo, hi = self.cutoff_ms
        if not 0 <= lo <= hi:
            raise ConfigError("cutoff_ms must be an ordered non-negative window")
        if clamp_gas_limit(self.gas_limit) != self.gas_limit:
            raise ConfigError("gas_limit must be within [21000, 30000000]")
        if self.native != ARENA_NATIVE:
            raise ConfigError(f"the arena market is priced in {ARENA_NATIVE}")

    def to_dict(self) -> Dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


@dataclass
class Ledger:
    """One searcher's results over a heat."""

    pnl_wei: int = 0
    submitted: int = 0
    arrived: int = 0
    won: int = 0
    late: int = 0
    evicted: int = 0
    idle_blocks: int = 0
    gas_used: int = 0
    coinbase_wei: int = 0

    def to_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)


@dataclass
class QueueStats:
    """Block-building queue load over a heat (or a whole run)."""

    blocks: int = 0
    victims: int = 0
    submitted: int = 0
    arrived: int = 0
    late: int = 0
    included: int = 0
    peak_depth: int = 0
    max_gas_used: int = 0
    gas_used: int = 0
    value_wei: int = 0
    evictions: Dict[str, int] = field(default_factory=dict)

    def add(self, other: Mapping[str, Any]) -> None:
        for k in ("blocks", "victims", "submitted", "arrived", "late", "included", "gas_used", "value_wei"):
            setattr(self, k, getattr(self, k) + int(other[k]))
        self.peak_depth = max(self.peak_depth, int(other["peak_depth"]))
        self.max_gas_used = max(self.max_gas_used, int(other["max_gas_used"]))
        for reason, n in other["evictions"].items():
            self.evictions[reason] = self.evictions.get(reason, 0) + int(n)

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["evictions"] = dict(sorted(self.evictions.items()))
        return out


def _reason(note: str) -> str:
    return note.split(":", 1)[0] if note else "not_included"


class BotArena:
    """Run one heat: ``config.slots`` blocks of bots + incumbent competing for victims in one queue."""

    def __init__(self, bots: Sequence[SynthesizedBot], config: Optional[ArenaConfig] = None,
                 incumbent: Optional[IncumbentPolicy] = None, violation_limit: int = VIOLATION_LIMIT) -> None:
        names = [b.name for b in bots]
        if len(set(names)) != len(names):
            raise ConfigError("bot names must be unique within a heat")
        self.bots = list(bots)
        self.config = config or ArenaConfig()
        self.incumbent = incumbent
        self.regime = REGIMES[self.config.regime]
        self.rng = random.Random(_seed_int("ai-arena", self.config.seed))
        self.state = BuilderState.from_pools(arena_market())
        self.initial = {pid: (p.reserve0, p.reserve1) for pid, p in self.state.pools.items()}
        self.base_fee = self.config.base_fee_wei
        self.prev_base_fee = self.base_fee
        self.prev_reserve = self.state.pools[ARENA_POOLS[0]].reserve0
        self.slot = 0
        self.ledgers: Dict[str, Ledger] = {b.name: Ledger() for b in self.bots}
        self.incumbent_ledger = Ledger()
        self.queue = QueueStats()
        self.violation_limit = violation_limit
        self.violations: List[Dict[str, Any]] = []
        self.violation_count = 0
        self.checks = 0
        self._victim_bps: Dict[str, int] = {}

    # -- bookkeeping ---------------------------------------------------------

    def _check(self, rule: str, ok: bool, detail: str) -> None:
        self.checks += 1
        if not ok:
            self.violation_count += 1
            if len(self.violations) < self.violation_limit:
                self.violations.append({"rule": rule, "slot": self.slot, "detail": detail})

    def _realized(self, account: str, before: Mapping[str, int], post: BuilderState,
                  pools_before: Mapping[str, Any]) -> int:
        after = post.fork.balances.get(account, {})
        total = 0
        for tok in sorted(set(before) | set(after)):
            delta = after.get(tok, 0) - before.get(tok, 0)
            if delta == 0:
                continue
            if tok in (ETH, self.config.native):
                total += delta
            else:
                v = native_value(pools_before, tok, abs(delta), self.config.native)
                total += v if delta > 0 else -v
        return total

    # -- market --------------------------------------------------------------

    def _victims(self) -> List[BuilderTx]:
        out: List[BuilderTx] = []
        lo, hi = math.log(self.regime.victim_min_bps), math.log(self.regime.victim_max_bps)
        for j in range(self.config.victims_per_slot):
            tin, route = ARENA_ROUTES[self.rng.randrange(len(ARENA_ROUTES))]
            pool = self.state.pools[route[0]]
            zfo, _ = _direction(pool, tin)
            reserve_in = pool.reserve0 if zfo else pool.reserve1
            size_bps = math.exp(self.rng.uniform(lo, hi))
            amount = max(1, int(reserve_in * size_bps) // BPS)
            tolerance = self.rng.choice(VICTIM_TOLERANCES)
            tip = self.rng.randrange(1, 6) * GWEI // 2
            sender = f"user:{self.slot}:{j}"
            tx_hash = f"0x{self.slot:06x}{j:02x}{self.rng.getrandbits(88):022x}"
            probe = BuilderTx("probe", sender, 0, 10**6, self.base_fee * 3 + tip, tip, TxKind.SWAP,
                              {"route": list(route), "token_in": tin, "amount_in": amount})
            try:
                hops, _ = _quote_hops(self.state, probe)
                quoted = hops[-1][3].amount_out
            except SwapReverted:
                quoted = 0
            gas = estimate_swap_gas([self.state.pools[pid] for pid in route]) * 3 // 2
            victim = BuilderTx(tx_hash, sender, self.state.nonce(sender), gas, self.base_fee * 3 + tip, tip,
                               TxKind.SWAP, {"route": list(route), "token_in": tin, "amount_in": amount,
                                             "amount_out_min": min_amount_out(quoted, tolerance),
                                             "tolerance_bps": tolerance})
            self._victim_bps[tx_hash] = round(size_bps)
            out.append(victim)
        return out

    def _noise(self) -> None:
        sigma = self.regime.noise_bps
        if sigma <= 0:
            return
        for pid in ARENA_POOLS:
            pool = self.state.pools[pid]
            r0, r1 = self.initial[pid]
            dev = math.log((pool.reserve1 * r0) / (pool.reserve0 * r1))
            zfo = self.rng.random() < 0.5 + 0.45 * math.tanh(20 * dev)
            reserve_in = pool.reserve0 if zfo else pool.reserve1
            amount = int(reserve_in * abs(self.rng.gauss(0.0, sigma))) // BPS
            if amount > 0:
                pool.apply_swap(pool.quote_exact_in(amount, zfo), zfo)

    def _gas_step(self) -> None:
        cfg = self.config
        self.prev_base_fee = self.base_fee
        move = math.exp(self.rng.gauss(0.0, self.regime.gas_vol))
        self.base_fee = int(min(cfg.max_base_fee_wei, max(cfg.min_base_fee_wei, self.base_fee * move)))

    def observation(self, victim: BuilderTx, delay_ms: int) -> Observation:
        pool = self.state.pools[ARENA_POOLS[0]]
        dr = (pool.reserve0 - self.prev_reserve) * BPS // max(1, self.prev_reserve)
        dg = (self.base_fee - self.prev_base_fee) * BPS // max(1, self.prev_base_fee)
        return Observation(int(dr), int(dg), delay_ms, self._victim_bps.get(victim.tx_hash, 0), len(self.bots))

    # -- one block -----------------------------------------------------------

    def step(self) -> Dict[str, Any]:
        cfg = self.config
        parent = self.state
        base_fee = self.base_fee
        victims = self._victims()
        oracles = [HopOracles(parent.pools, v, cfg.grid, cfg.refine, cfg.max_frontrun_bps, cfg.native)
                   for v in victims]
        cutoff = self.rng.uniform(*cfg.cutoff_ms)
        queue: List[Tuple[Optional[SynthesizedBot], Bundle, int]] = []
        for bot in self.bots:
            plans = []
            for j, (v, o) in enumerate(zip(victims, oracles)):
                plan = bot.plan(parent, v, o, base_fee, cfg.native)
                if plan is not None:
                    plans.append((plan.quote.net_wei, -j, j, plan))
            if not plans:
                self.ledgers[bot.name].idle_blocks += 1
                continue
            plans.sort(key=lambda c: (c[0], c[1]), reverse=True)
            chosen = sorted(plans[: bot.profile.bundle_slots], key=lambda c: c[2])
            delay = bot.sample_delay(self.rng)
            for k, (_, _, j, plan) in enumerate(chosen):
                queue.append((bot, bot.bundle(parent, victims[j], plan, base_fee, slot=k), delay))
        if self.incumbent is not None:
            delay = max(0, int(self.rng.gauss(self.regime.delay_mean_ms, self.regime.delay_jitter_ms)))
            planned = 0
            for j, (v, o) in enumerate(zip(victims, oracles)):
                act = self.incumbent.action(self.observation(v, delay))
                if act.skip:
                    continue
                cand = o.best()
                if cand is None:
                    continue
                size = max(1, cand.frontrun_in * act.size_bps // BPS)
                q = sandwich_economics(parent, v, cand.hop, size, base_fee, self.incumbent.tip_wei, cfg.native)
                if q is None or q.net_wei <= 0:
                    continue
                b = make_sandwich_bundle(parent, v, q, base_fee, f"{INCUMBENT}#{j}", q.net_wei * act.bid_bps // BPS,
                                         self.incumbent.tip_wei, tag=f"incumbent#{j}")
                queue.append((None, b, delay))
                planned += 1
            if not planned:
                self.incumbent_ledger.idle_blocks += 1
        arrived = [b for _, b, d in queue if d <= cutoff]
        accounts = sorted({b.searcher for b in arrived})
        before = {a: dict(parent.fork.balances.get(a, {})) for a in accounts}
        pools_before = dict(parent.pools)
        outcome = run_competition(parent, arrived, base_fee, victims, gas_limit=cfg.gas_limit)
        post = outcome.post_state
        block = outcome.block
        self._check("ai.block_gas", block.gas_used <= cfg.gas_limit,
                    f"block used {block.gas_used} gas over the {cfg.gas_limit} limit")
        landed_per_victim = Counter(b.target_tx for b in arrived if outcome.won(b.bundle_id))
        self._check("ai.single_winner", all(n <= 1 for n in landed_per_victim.values()),
                    f"several sandwiches landed on one victim: {dict(landed_per_victim)}")
        stats = {"submitted": len(queue), "arrived": len(arrived), "late": len(queue) - len(arrived),
                 "included": sum(landed_per_victim.values()), "gas_used": block.gas_used,
                 "value_wei": block.value_wei, "evictions": Counter()}
        for bot, b, d in queue:
            ledger = self.incumbent_ledger if bot is None else self.ledgers[bot.name]
            ledger.submitted += 1
            if d > cutoff:
                ledger.late += 1
                won = False
            elif outcome.won(b.bundle_id):
                won = True
                ledger.arrived += 1
                ledger.won += 1
                pnl = self._realized(b.searcher, before[b.searcher], post, pools_before)
                self._check("ai.exact_pnl", type(pnl) is int, f"{b.searcher} P&L is {type(pnl).__name__}")
                ledger.pnl_wei += pnl
                ledger.coinbase_wei += b.coinbase_transfer_wei
                ledger.gas_used += sum(t.gas_used for t in block.txs if t.sender == b.searcher)
            else:
                won = False
                ledger.arrived += 1
                ledger.evicted += 1
                stats["evictions"][_reason(outcome.evicted.get(b.bundle_id, ""))] += 1
                after = {t: v for t, v in post.fork.balances.get(b.searcher, {}).items() if v}
                was = {t: v for t, v in before[b.searcher].items() if v}
                self._check("ai.idle_balances", after == was, f"evicted {b.bundle_id} still moved balances")
            if bot is not None:
                bot.observe(won)
        self.queue.add({"blocks": 1, "victims": len(victims), "peak_depth": len(arrived),
                        "max_gas_used": block.gas_used, **stats})
        self.state = post
        self.state.fork.mine()
        self.prev_reserve = self.state.pools[ARENA_POOLS[0]].reserve0
        self._noise()
        self._gas_step()
        self.slot += 1
        return {"slot": self.slot - 1, "cutoff_ms": int(cutoff), **{k: v for k, v in stats.items()
                                                                    if k != "evictions"}}

    def run(self) -> Dict[str, Any]:
        for _ in range(self.config.slots):
            self.step()
        return {
            "bots": [{"name": b.name, "class": type(b).__name__, **self.ledgers[b.name].to_dict(),
                      "tip_final_wei": b.tip_wei, "share_final_bps": b.share_bps} for b in self.bots],
            "incumbent": self.incumbent_ledger.to_dict() if self.incumbent is not None else None,
            "queue": self.queue.to_dict(),
            "violations": list(self.violations),
            "violation_count": self.violation_count,
            "checks": self.checks,
        }


def _heat_task(task: Tuple[Any, ...]) -> Dict[str, Any]:
    """Worker entry point: spin up the heat's bot classes and run the arena."""
    heat, generation, profiles, arena_cfg, incumbent, violation_limit = task
    cpu0 = time.process_time()
    bots = [spawn_bot(p) for p in profiles]
    result = BotArena(bots, ArenaConfig(**arena_cfg), incumbent, violation_limit).run()
    result.update(heat=heat, generation=generation, cpu_s=time.process_time() - cpu0, pid=os.getpid())
    return result


# ---------------------------------------------------------------------------
# Resource allocation
# ---------------------------------------------------------------------------


def apportion(total: int, weights: Sequence[int]) -> List[int]:
    """Split ``total`` units over integer ``weights`` exactly (Hamilton / largest remainder).

    Each share is ``floor(total * w / W)`` plus one for the largest
    remainders (ties: lowest index). All-zero weights split evenly.
    """
    if not _is_int(total) or total < 0:
        raise ConfigError("total must be a non-negative integer")
    ws = [int(w) for w in weights]
    if any(w < 0 for w in ws):
        raise ConfigError("weights must be non-negative")
    if not ws:
        if total:
            raise ConfigError("cannot apportion a positive total over no weights")
        return []
    if sum(ws) == 0:
        ws = [1] * len(ws)
    W = sum(ws)
    shares = [total * w // W for w in ws]
    rems = [total * w % W for w in ws]
    left = total - sum(shares)
    for i in sorted(range(len(ws)), key=lambda k: (-rems[k], k))[:left]:
        shares[i] += 1
    return shares


def allocate_capital(requests: Sequence[int], budget: int, floor: int = E18) -> List[int]:
    """Capital each bot gets: its request when the budget covers everyone, else an exact split.

    When oversubscribed every bot first gets ``min(request, floor')`` (with
    ``floor' = min(floor, budget // n)``), then the rest of the budget is
    apportioned over the unmet requests. Grants never exceed requests and
    always sum to ``min(budget, sum(requests))``.
    """
    reqs = [int(r) for r in requests]
    if any(r < 0 for r in reqs) or budget < 0 or floor < 0:
        raise ConfigError("requests, budget and floor must be non-negative")
    if sum(reqs) <= budget:
        return reqs
    f = min(floor, budget // len(reqs))
    base = [min(r, f) for r in reqs]
    extra = apportion(budget - sum(base), [r - b for r, b in zip(reqs, base)])
    return [b + e for b, e in zip(base, extra)]


def fitness_key(row: Mapping[str, Any]) -> Tuple[int, int, int, int]:
    """``(P&L, wins, bundles that made the cutoff, bundles submitted)``, compared lexicographically.

    P&L decides; among bots that never landed a bundle (P&L 0) the ones that
    competed (reached the builder in time, or at least bid) rank higher.
    """
    return (int(row["pnl_wei"]), int(row["won"]), int(row["arrived"]), int(row["submitted"]))


def rank_order(keys: Sequence[Any]) -> List[int]:
    """Indices best-first by key (ints or tuples compared lexicographically); ties go to the lower index."""
    norm = [k if isinstance(k, tuple) else (k,) for k in keys]
    return sorted(range(len(norm)), key=lambda i: (tuple(-v for v in norm[i]), i))


def allocate_offspring(keys: Sequence[Any], children: int, elite: int) -> Dict[int, int]:
    """Next-generation slots per elite index (best first): linear rank weights ``elite, ..., 1``, exact."""
    if elite < 1 or elite > len(keys):
        raise ConfigError("elite must be within [1, len(keys)]")
    order = rank_order(keys)[:elite]
    counts = apportion(children, [elite - r for r in range(elite)])
    return {idx: n for idx, n in zip(order, counts)}


def lineage(offspring: Mapping[int, int]) -> List[int]:
    """Parent index of every child slot, best parent first (``offspring`` in rank order)."""
    return [i for i, c in offspring.items() for _ in range(c)]


def plan_heats(n_bots: int, heat_size: int, seed: int = 0, generation: int = 0) -> List[List[int]]:
    """Seeded, balanced heats of at most ``heat_size`` bots; never depends on the worker count."""
    if n_bots < 0 or heat_size < 1:
        raise ConfigError("n_bots must be >= 0 and heat_size >= 1")
    if n_bots == 0:
        return []
    order = list(range(n_bots))
    random.Random(_seed_int("ai-heats", seed, generation)).shuffle(order)
    return [sorted(h) for h in chunked(order, -(-n_bots // heat_size))]


# ---------------------------------------------------------------------------
# The factory
# ---------------------------------------------------------------------------


@dataclass
class Population:
    """One generation: profiles plus the generator inputs, latents, deployed genes and parents."""

    generation: int
    profiles: List[BotProfile]
    inputs: np.ndarray
    genes: np.ndarray
    requested_capital: List[int]
    parents: List[Optional[int]]
    resampled: int = 0

    @property
    def latents(self) -> np.ndarray:
        return self.inputs[:, N_FEATURES:]

    @property
    def requested_capital_wei(self) -> int:
        return sum(self.requested_capital)

    @property
    def granted_capital_wei(self) -> int:
        return sum(p.capital_wei for p in self.profiles)


def _percentile(sorted_vals: Sequence[int], q: float) -> int:
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[i]


@dataclass
class FactoryReport:
    bots: int
    generations: int
    heat_size: int
    heat_slots: int
    victims_per_slot: int
    history: List[Dict[str, Any]]
    top_bots: List[Dict[str, Any]]
    last_profiles: List[BotProfile]
    queue: Dict[str, Any]
    violations: List[Dict[str, Any]]
    violation_count: int
    checks: int
    features: Dict[str, float]
    feature_sources: List[str]
    incumbent: Dict[str, Any]
    engine: Dict[str, Any]
    mutation: Dict[str, Any]
    allocation: Dict[str, Any]
    elapsed_s: float
    cpu_s: float
    workers: int
    parallel: bool
    worker_pids: List[int]

    @property
    def ok(self) -> bool:
        return self.violation_count == 0

    def summary(self) -> Dict[str, Any]:
        total = self.bots * self.generations
        capacity = self.elapsed_s * max(1, self.workers)
        last = self.history[-1] if self.history else {}
        return {
            "bots_per_generation": self.bots,
            "generations": self.generations,
            "total_bots": total,
            "heat_size": self.heat_size,
            "heats_per_generation": last.get("heats", 0),
            "heat_slots": self.heat_slots,
            "victims_per_slot": self.victims_per_slot,
            "workers": self.workers,
            "parallel": self.parallel,
            "distinct_worker_pids": len(set(self.worker_pids)),
            "elapsed_s": round(self.elapsed_s, 4),
            "cpu_s": round(self.cpu_s, 4),
            "bots_per_s": round(total / self.elapsed_s, 1) if self.elapsed_s > 0 else 0.0,
            "parallel_efficiency": round(min(1.0, self.cpu_s / capacity), 4) if capacity > 0 else 0.0,
            "engine": dict(self.engine),
            "features": dict(self.features),
            "feature_sources": list(self.feature_sources),
            "incumbent": dict(self.incumbent),
            "mutation": dict(self.mutation),
            "history": [dict(h) for h in self.history],
            "queue": dict(self.queue),
            "allocation": dict(self.allocation),
            "strategy_mix": dict(sorted(Counter(p.strategy for p in self.last_profiles).items())),
            "hop_policy_mix": dict(sorted(Counter(p.hop_policy for p in self.last_profiles).items())),
            "top_bots": [dict(b) for b in self.top_bots],
            "invariants": {
                "ok": self.ok,
                "rules": list(FACTORY_RULES),
                "checks": self.checks,
                "violation_count": self.violation_count,
                "violations": self.violations[:20],
            },
        }


class AgentFactory:
    """Generate, deploy, rank and learn from ``bots`` machine-made competitors for ``generations`` rounds."""

    def __init__(self, bots: int = 256, generations: int = 3, heat_size: int = 8, heat_slots: int = 12,
                 seed: int = 0, regime: str = "volatile", victims_per_slot: int = 3,
                 hidden: Sequence[int] = (32, 32), latent_dim: int = 8, elite_frac: float = 0.25, epochs: int = 40,
                 lr: float = 0.01, init: str = "xavier", mutation: Optional[AdaptiveMutation] = None,
                 capital_per_bot_wei: int = 50 * E18, capital_floor_wei: int = E18,
                 base_fee_wei: int = 20 * GWEI, gas_limit: int = BLOCK_GAS_LIMIT, grid: int = 10, refine: int = 5,
                 violation_limit: int = VIOLATION_LIMIT, max_resample: int = 8) -> None:
        if bots < 0 or generations < 1 or heat_size < 1 or heat_slots < 1 or victims_per_slot < 1:
            raise ConfigError("bots must be >= 0; generations, heat_size, heat_slots, victims_per_slot >= 1")
        if latent_dim < 0 or epochs < 0 or not 0 < elite_frac <= 1:
            raise ConfigError("latent_dim and epochs must be >= 0 and elite_frac within (0, 1]")
        if capital_per_bot_wei < 0 or capital_floor_wei < 0 or max_resample < 0:
            raise ConfigError("capital settings and max_resample must be non-negative")
        self.bots = bots
        self.generations = generations
        self.heat_size = heat_size
        self.heat_slots = heat_slots
        self.victims_per_slot = victims_per_slot
        self.seed = seed
        self.latent_dim = latent_dim
        self.elite_frac = elite_frac
        self.epochs = epochs
        self.capital_per_bot_wei = capital_per_bot_wei
        self.capital_floor_wei = capital_floor_wei
        self.violation_limit = violation_limit
        self.max_resample = max_resample
        self.arena = ArenaConfig(seed=seed, regime=regime, slots=heat_slots, victims_per_slot=victims_per_slot,
                                 grid=grid, refine=refine, base_fee_wei=base_fee_wei,
                                 max_base_fee_wei=max(500 * GWEI, base_fee_wei), gas_limit=clamp_gas_limit(gas_limit))
        self.engine = NeuralMatrixEngine((N_FEATURES + latent_dim, *hidden, N_GENES), "tanh", "sigmoid", init,
                                         seed=_seed_int("ai-engine", seed) % (1 << 63),
                                         optimizer=AdamOptimizer(lr=lr))
        self.mutation = mutation if mutation is not None else AdaptiveMutation()
        self.metrics = RLMetrics.neutral(regime)
        self.incumbent = IncumbentPolicy.default()

    @classmethod
    def for_config(cls, config: Any, **overrides: Any) -> "AgentFactory":
        """Factory sized by a SimulationConfig (``agent_*`` fields, seed, RL regime, gas)."""
        params: Dict[str, Any] = dict(
            bots=config.agent_bots, generations=config.agent_generations, heat_size=config.agent_heat_size,
            heat_slots=config.agent_heat_slots, seed=config.seed, regime=config.rl_regime,
            base_fee_wei=config.base_fee_wei, gas_limit=config.block_gas_limit)
        params.update(overrides)
        return cls(**params)

    # -- feedback in ---------------------------------------------------------

    def ingest(self, reinforcement: Optional[Mapping[str, Any]] = None, table: Optional[ActionTable] = None,
               actions: Optional[Sequence[Action]] = None,
               policy_paths: Optional[Mapping[str, Any]] = None) -> RLMetrics:
        """Condition the generator on RL results; a trained table also becomes the incumbent."""
        self.metrics = RLMetrics.from_sources(reinforcement, table, actions, policy_paths,
                                              regime=self.arena.regime)
        if table is not None:
            self.incumbent = IncumbentPolicy.from_table(table, actions)
        return self.metrics

    # -- synthesis -----------------------------------------------------------

    def inputs(self, generation: int, parents: Optional[Sequence[int]] = None,
               parent_latents: Optional[np.ndarray] = None) -> np.ndarray:
        """Generator inputs ``[RL features, latent]``: fresh latents, or each parent's latent plus noise."""
        n = self.bots
        noise = seeded_generator("latent", self.seed, generation).standard_normal((n, self.latent_dim))
        if parents is None:
            z = noise
        else:
            if len(parents) != n or parent_latents is None:
                raise ConfigError(f"need one parent per child ({n}) and the parents' latents")
            z = np.asarray(parent_latents, dtype=np.float64)[list(parents)] + self.mutation.sigma * noise
        return np.hstack([np.tile(self.metrics.vector(), (n, 1)), z])

    def synthesize(self, generation: int, parents: Optional[Sequence[int]] = None,
                   parent_latents: Optional[np.ndarray] = None) -> Population:
        """Generator forward pass, adaptive mutation, decoding, de-duplication and capital allocation.

        Generation 0 samples fresh latents. Later generations pass ``parents``
        (one parent index per child, from :func:`lineage`) and the previous
        generation's latents, so every child starts next to its parent.
        """
        n = self.bots
        x = self.inputs(generation, parents, parent_latents)
        raw = self.engine.forward(x) if n else np.zeros((0, N_GENES))
        genes = self.mutation.mutate_genes(raw, seeded_generator("mutation", self.seed, generation)) if n \
            else np.zeros((0, N_GENES))
        profiles: List[BotProfile] = []
        seen: set = set()
        resampled = 0
        for i in range(n):
            p = decode_profile(genes[i], f"g{generation}-b{i:04d}", generation)
            attempt = 0
            while p.fingerprint() in seen and attempt < self.max_resample:
                attempt += 1
                resampled += 1
                rng = seeded_generator("resample", self.seed, generation, i, attempt)
                genes[i] = reflect_unit(raw[i] + max(self.mutation.sigma, 0.05) * rng.standard_normal(N_GENES))
                p = decode_profile(genes[i], p.name, generation)
            seen.add(p.fingerprint())
            profiles.append(p)
        requested = [p.capital_wei for p in profiles]
        granted = allocate_capital(requested, self.capital_per_bot_wei * n, self.capital_floor_wei) if n else []
        profiles = [replace(p, capital_wei=c) for p, c in zip(profiles, granted)]
        lineage_ = list(parents) if parents is not None else [None] * n
        return Population(generation, profiles, x, genes, requested, lineage_, resampled)

    def tasks(self, population: Population) -> Tuple[List[List[int]], List[Tuple[Any, ...]]]:
        heats = plan_heats(len(population.profiles), self.heat_size, self.seed, population.generation)
        tasks = []
        for h, members in enumerate(heats):
            arena = replace(self.arena, seed=_seed_int("ai-heat", self.seed, population.generation, h) % (1 << 63))
            tasks.append((h, population.generation, [population.profiles[i] for i in members], arena.to_dict(),
                          self.incumbent, self.violation_limit))
        return heats, tasks

    # -- the loop ------------------------------------------------------------

    def deploy(self, population: Population, executor: Optional[WorkerPool] = None
               ) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
        """Run every heat of ``population`` (on the worker pool when given); results in heat order."""
        heats, tasks = self.tasks(population)
        if executor is None or not tasks:
            return heats, [_heat_task(t) for t in tasks]
        return heats, executor.map(_heat_task, tasks)

    def run(self, executor: Optional[WorkerPool] = None, workers: Optional[int] = None) -> FactoryReport:
        """Synthesise -> deploy -> rank -> train -> adapt, for every generation."""
        own = executor is None and workers is not None
        wp = WorkerPool(workers) if own else executor
        started = time.perf_counter()
        history: List[Dict[str, Any]] = []
        everyone: List[Dict[str, Any]] = []
        queue = QueueStats()
        violations: List[Dict[str, Any]] = []
        count = checks = 0
        cpu = 0.0
        pids: List[int] = []
        n_workers, parallel = 1, False
        oversubscribed = 0
        last_profiles: List[BotProfile] = []
        offspring: Dict[int, int] = {}
        budget = requested_total = granted_total = 0
        parents: Optional[List[int]] = None
        parent_latents: Optional[np.ndarray] = None
        parent_keys: Optional[List[Tuple[int, ...]]] = None

        def check(rule: str, ok: bool, detail: str, generation: int) -> None:
            nonlocal count, checks
            checks += 1
            if not ok:
                count += 1
                if len(violations) < self.violation_limit:
                    violations.append({"rule": rule, "generation": generation, "detail": detail})

        try:
            for g in range(self.generations):
                pop = self.synthesize(g, parents, parent_latents)
                n = len(pop.profiles)
                last_profiles = pop.profiles
                # -- synthesis invariants ------------------------------------------
                check("ai.gene_bounds", bool(np.all((pop.genes >= 0.0) & (pop.genes <= 1.0))),
                      "a gene left [0, 1]", g)
                for p in pop.profiles:
                    issues = profile_issues(p)
                    check("ai.profile_bounds", not issues, f"{p.name}: {issues}", g)
                fps = [p.fingerprint() for p in pop.profiles]
                check("ai.unique_bots", len(set(fps)) == n and len({p.name for p in pop.profiles}) == n,
                      f"{n - len(set(fps))} duplicate strategies", g)
                budget = self.capital_per_bot_wei * n
                requested_total, granted_total = pop.requested_capital_wei, pop.granted_capital_wei
                oversubscribed += requested_total > budget
                check("ai.allocation_conservation", granted_total == min(budget, requested_total)
                      and all(p.capital_wei <= q for p, q in zip(pop.profiles, pop.requested_capital)),
                      f"granted {granted_total} of budget {budget} for requests {requested_total}", g)
                # -- deployment ---------------------------------------------------
                heats, parts = self.deploy(pop, wp)
                if wp is not None and heats:
                    n_workers = wp.workers
                    parallel = parallel or wp.will_parallelize(len(heats))
                index = {p.name: i for i, p in enumerate(pop.profiles)}
                results: List[Optional[Dict[str, Any]]] = [None] * n
                heat_of = [0] * n
                incumbent_pnl: List[int] = []
                for part in parts:
                    cpu += part["cpu_s"]
                    pids.append(part["pid"])
                    queue.add(part["queue"])
                    count += part["violation_count"]
                    checks += part["checks"]
                    room = self.violation_limit - len(violations)
                    if room > 0:
                        violations.extend({**v, "generation": g, "heat": part["heat"]}
                                          for v in part["violations"][:room])
                    incumbent_pnl.append(part["incumbent"]["pnl_wei"] if part["incumbent"] is not None else 0)
                    for row in part["bots"]:
                        i = index[row["name"]]
                        check("ai.heat_coverage", results[i] is None, f"{row['name']} ran in two heats", g)
                        results[i] = row
                        heat_of[i] = part["heat"]
                check("ai.heat_coverage", all(r is not None for r in results)
                      and sorted(i for h in heats for i in h) == list(range(n)), "a bot was never deployed", g)
                rows = [r if r is not None else Ledger().to_dict() for r in results]
                # -- ranking, allocation, training --------------------------------
                keys = [fitness_key(r) for r in rows]
                pnl = [k[0] for k in keys]
                wins = [k[1] for k in keys]
                beat = [pnl[i] > incumbent_pnl[heat_of[i]] for i in range(n)]
                improved = [keys[i] > parent_keys[pop.parents[i]] for i in range(n)  # type: ignore[index]
                            if parent_keys is not None and pop.parents[i] is not None]
                success = sum(improved) / len(improved) if improved else None
                elite = max(1, math.ceil(self.elite_frac * n)) if n else 0
                loss_before = loss_after = 0.0
                if n:
                    offspring = allocate_offspring(keys, n, elite)
                    check("ai.allocation_conservation", sum(offspring.values()) == n, "offspring do not sum", g)
                    idx = list(offspring)
                    weights = np.array([offspring[i] for i in idx], dtype=np.float64)
                    if float(np.sum(weights)) > 0:
                        losses = self.engine.fit(pop.inputs[idx], pop.genes[idx], weights, self.epochs)
                        loss_before, loss_after = losses[0], losses[-1]
                check("ai.finite_weights", self.engine.is_finite(), "non-finite generator weight", g)
                sigma_before = self.mutation.sigma
                if success is not None:
                    self.mutation.update(success)
                check("ai.sigma_bounds", self.mutation.sigma_min <= self.mutation.sigma <= self.mutation.sigma_max,
                      f"sigma {self.mutation.sigma}", g)
                parents = lineage(offspring) if n else None
                parent_latents = pop.latents
                parent_keys = keys
                # -- report -------------------------------------------------------
                ranked = sorted(pnl)
                total = sum(pnl)
                order = rank_order(keys)
                for i in order[:5]:
                    everyone.append({"name": pop.profiles[i].name, "generation": g, "pnl_wei": pnl[i],
                                     "wins": wins[i], "arrived": keys[i][2], "submitted": keys[i][3],
                                     "parent": None if pop.parents[i] is None else f"g{g - 1}-b{pop.parents[i]:04d}",
                                     "class": class_name(pop.profiles[i]), "strategy": pop.profiles[i].describe(),
                                     "profile": pop.profiles[i].to_dict()})
                history.append({
                    "generation": g, "bots": n, "heats": len(heats), "distinct_strategies": len(set(fps)),
                    "resampled": pop.resampled, "sigma": round(sigma_before, 9),
                    "next_sigma": round(self.mutation.sigma, 9),
                    "success_rate": None if success is None else round(success, 6),
                    "beat_incumbent": sum(beat), "elite": elite, "parents": len(set(p for p in pop.parents
                                                                                    if p is not None)),
                    "loss_before": round(loss_before, 9), "loss_after": round(loss_after, 9),
                    "pnl_wei": {"total": total, "mean": total // n if n else 0, "min": ranked[0] if n else 0,
                                "p50": _percentile(ranked, 0.5), "p95": _percentile(ranked, 0.95),
                                "max": ranked[-1] if n else 0, "positive_bots": sum(1 for v in pnl if v > 0)},
                    "incumbent_pnl_wei": sum(incumbent_pnl),
                    "wins": sum(wins), "arrived": sum(k[2] for k in keys), "submitted": sum(k[3] for k in keys),
                    "active_bots": sum(1 for k in keys if k[3] > 0),
                    "best_bot": pop.profiles[order[0]].name if n else None,
                })
        finally:
            if own and wp is not None:
                wp.close()
        everyone.sort(key=lambda b: (-b["pnl_wei"], -b["wins"], -b["arrived"], -b["submitted"], b["generation"],
                                     b["name"]))
        return FactoryReport(
            bots=self.bots, generations=self.generations, heat_size=self.heat_size, heat_slots=self.heat_slots,
            victims_per_slot=self.victims_per_slot, history=history, top_bots=everyone[:5],
            last_profiles=last_profiles, queue={**queue.to_dict(), "gas_limit": self.arena.gas_limit},
            violations=violations[: self.violation_limit], violation_count=count, checks=checks,
            features=self.metrics.to_dict(), feature_sources=list(self.metrics.sources),
            incumbent=self.incumbent.to_dict(), engine=self.engine.describe(), mutation=self.mutation.to_dict(),
            allocation={"capital_budget_wei": budget, "capital_requested_wei": requested_total,
                        "capital_granted_wei": granted_total, "oversubscribed_generations": oversubscribed,
                        "elite_offspring": [{"bot": last_profiles[i].name, "children": c}
                                            for i, c in list(offspring.items())[:8]] if last_profiles else []},
            elapsed_s=time.perf_counter() - started, cpu_s=cpu, workers=n_workers, parallel=parallel,
            worker_pids=pids,
        )


__all__ = [
    "ACTIVATIONS",
    "ARENA_POOLS",
    "ARENA_ROUTES",
    "FACTORY_RULES",
    "FEATURE_NAMES",
    "GENE_NAMES",
    "HOP_POLICIES",
    "INIT_SCHEMES",
    "MAX_BUNDLE_SLOTS",
    "N_FEATURES",
    "N_GENES",
    "STRATEGY_VALUES",
    "AdamOptimizer",
    "AdaptiveMutation",
    "AgentFactory",
    "ArenaConfig",
    "BotArena",
    "BotPlan",
    "BotProfile",
    "FactoryReport",
    "HopOracles",
    "IncumbentPolicy",
    "Ledger",
    "NeuralMatrixEngine",
    "Population",
    "QueueStats",
    "RLMetrics",
    "SynthesizedBot",
    "activate",
    "activation_grad",
    "allocate_capital",
    "allocate_offspring",
    "apportion",
    "arena_market",
    "bot_class",
    "class_name",
    "decode_profile",
    "encode_profile",
    "fitness_key",
    "hop_candidate",
    "init_matrix",
    "init_std",
    "lineage",
    "plan_heats",
    "profile_from_competitor",
    "profile_issues",
    "rank_order",
    "reflect_unit",
    "seeded_generator",
    "sigmoid",
    "signed_log_scale",
    "spawn_bot",
]
