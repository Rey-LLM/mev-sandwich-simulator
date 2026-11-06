"""Layer 08: online adaptive reinforcement learning for sandwich bidding.

An agent learns, while it runs, how large a front-run to place and how much
of the expected profit to pay the builder, against competing searcher bots
whose bundles target the same victims. Everything is in memory and uses
only the standard library; the environment executes real bundles through
the Layer 04 block builder, so every reward is an exact integer number of
wei taken from the agent's post-block balances.

Environment (:class:`SandwichEnvironment`)
------------------------------------------
Each step one victim swap arrives on a Uniswap v2 WETH/USDC pool. The agent
observes a discretised :class:`Observation`:

* ``reserve_delta_bps`` - signed change of the pool's WETH reserve since the
  previous step (market noise plus the previous block's trades),
* ``gas_delta_bps`` - change of the base fee since the previous step
  (log-normal random walk, volatility set by the :class:`MarketRegime`),
* ``relay_delay_ms`` - the agent's measured relay latency for this slot,
* ``victim_size_bps`` - the victim's input as bps of the input reserve,
* ``rivals`` - how many competitor bots are active,

then picks an :class:`Action` (front-run size as bps of the oracle-optimal
size, and builder payment as bps of expected net profit, or skip). Every
bot in :func:`simulator.block_builder.default_competitors` builds its own
bundle; bundles whose relay delay exceeds the builder's (random) cutoff miss
the slot; a greedy builder then ranks the rest by value per gas, so at most
one sandwich per victim lands (see the eviction model in ``block_builder``).

Reward
------
``reward_wei`` is the agent's realised balance change for the block: profit
token delta converted to the native token at spot, plus the ETH it paid in
priority fees, burned base fee and coinbase transfers. A bundle that is
evicted or misses the cutoff costs nothing and earns nothing (0 wei).

The learning signal is ``reward_wei`` normalised by :class:`RewardScaler`,
an exponentially weighted estimate of reward magnitude that keeps adapting
as the market regime changes, clipped to ``[-clip, clip]`` so a single
outlier block under extreme noise cannot blow up the value estimates.
The exact integer rewards are accumulated separately and never rounded.

Agents
------
* :class:`QLearningAgent` - tabular one-step Q-learning with epsilon-greedy
  exploration, per-(state, action) Robbins-Monro step sizes
  ``alpha_n = max(alpha_min, alpha0 / (1 + n) ** power)`` and a small
  discount (consecutive victims are almost independent).
* :class:`PolicyGradientAgent` - tabular softmax policy trained with
  REINFORCE and a per-state reward baseline.

Both store their parameters in an :class:`ActionTable` (dict of lists), so
they pickle cheaply and can be merged.

Parallel policy iteration
-------------------------
:func:`parallel_policy_iteration` runs ``rounds`` rounds; in each, a fixed
number of shards (``RL_SHARDS``) each copy the current table, train on an
independently seeded environment and return the table. Shards are merged by
visit-weighted averaging (:meth:`ActionTable.merge`). Shards run on the
capped :class:`~simulator.parallel.WorkerPool`, which is always sized by
:func:`simulator.config.resolve_worker_count` (50-60% of usable CPUs); the
shard count is fixed, so results are identical for any worker count.

Limits: single v2 pool, single-hop victims, rival bots with simple rules;
latency is a sampled number, not a network model; the builder is greedy (no
layout fuzzing) inside the training loop for speed.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .block_builder import (
    ETH,
    GWEI,
    BuilderState,
    BuilderTx,
    Bundle,
    CompetitorBot,
    FeeStructure,
    TxKind,
    _quote_hops,
    best_sandwich,
    default_competitors,
    make_sandwich_bundle,
    native_value,
    quote_eviction,
    run_competition,
    sandwich_economics,
)
from .config import ConfigError
from .mempool_emulator import estimate_swap_gas
from .parallel import WorkerPool
from .slippage_engine import V2Pool, min_amount_out
from .state_fork import SwapReverted

E18 = 10**18
E6 = 10**6
RL_SHARDS = 8  # fixed so parallel training never depends on worker count
AGENT = "agent"

Key = Tuple[int, ...]


# ---------------------------------------------------------------------------
# State and action spaces
# ---------------------------------------------------------------------------


def bucket(value: float, edges: Sequence[float]) -> int:
    """Index of the half-open bucket ``value`` falls in (``len(edges) + 1`` buckets)."""
    return bisect_right(edges, value)


@dataclass(frozen=True)
class Observation:
    reserve_delta_bps: int
    gas_delta_bps: int
    relay_delay_ms: int
    victim_size_bps: int
    rivals: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class StateBins:
    """Bucket edges that turn an :class:`Observation` into a table key."""

    reserve_bps: Tuple[int, ...] = (-20, 20)
    gas_bps: Tuple[int, ...] = (-500, 500)
    delay_ms: Tuple[int, ...] = (250, 600)
    victim_bps: Tuple[int, ...] = (10, 30, 80)
    rivals: Tuple[int, ...] = (2, 4)

    def __post_init__(self) -> None:
        for name in ("reserve_bps", "gas_bps", "delay_ms", "victim_bps", "rivals"):
            edges = getattr(self, name)
            if list(edges) != sorted(edges) or len(set(edges)) != len(edges):
                raise ConfigError(f"{name} edges must be strictly increasing")

    def encode(self, obs: Observation) -> Key:
        return (
            bucket(obs.reserve_delta_bps, self.reserve_bps),
            bucket(obs.gas_delta_bps, self.gas_bps),
            bucket(obs.relay_delay_ms, self.delay_ms),
            bucket(obs.victim_size_bps, self.victim_bps),
            bucket(obs.rivals, self.rivals),
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(len(e) + 1 for e in (self.reserve_bps, self.gas_bps, self.delay_ms, self.victim_bps,
                                          self.rivals))

    @property
    def size(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class Action:
    size_bps: int  # front-run as bps of the oracle-optimal size; 0 = skip
    bid_bps: int  # builder payment as bps of expected net profit

    def __post_init__(self) -> None:
        if not 0 <= self.size_bps <= 20_000 or not 0 <= self.bid_bps <= 10_000:
            raise ConfigError("size_bps must be within [0, 20000] and bid_bps within [0, 10000]")

    @property
    def skip(self) -> bool:
        return self.size_bps == 0

    def label(self) -> str:
        return "skip" if self.skip else f"size={self.size_bps}bps bid={self.bid_bps}bps"


def action_grid(sizes: Sequence[int] = (5_000, 7_500, 10_000),
                bids: Sequence[int] = (1_000, 3_000, 5_000, 6_500, 8_000, 9_000, 9_700),
                include_skip: bool = True) -> Tuple[Action, ...]:
    acts = [Action(0, 0)] if include_skip else []
    acts.extend(Action(s, b) for s in sizes for b in bids)
    if not acts:
        raise ConfigError("action grid is empty")
    return tuple(acts)


# ---------------------------------------------------------------------------
# Tables, schedules, reward scaling
# ---------------------------------------------------------------------------


class ActionTable:
    """Per-state action values (or preferences) plus visit counts, created lazily."""

    def __init__(self, n_actions: int, init: float = 0.0) -> None:
        if n_actions < 1:
            raise ConfigError("n_actions must be >= 1")
        self.n_actions = n_actions
        self.init = float(init)
        self.values: Dict[Key, List[float]] = {}
        self.visits: Dict[Key, List[int]] = {}

    def __len__(self) -> int:
        return len(self.values)

    def __contains__(self, key: Key) -> bool:
        return key in self.values

    def row(self, key: Key) -> List[float]:
        r = self.values.get(key)
        if r is None:
            r = self.values[key] = [self.init] * self.n_actions
            self.visits[key] = [0] * self.n_actions
        return r

    def counts(self, key: Key) -> List[int]:
        self.row(key)
        return self.visits[key]

    def peek(self, key: Key) -> List[float]:
        """Row without creating it."""
        return list(self.values.get(key, [self.init] * self.n_actions))

    def best(self, key: Key) -> int:
        """Greedy action; ties go to the lowest index."""
        r = self.peek(key)
        top = max(r)
        return r.index(top)

    def total_visits(self) -> int:
        return sum(sum(v) for v in self.visits.values())

    def copy(self) -> "ActionTable":
        t = ActionTable(self.n_actions, self.init)
        t.values = {k: list(v) for k, v in self.values.items()}
        t.visits = {k: list(v) for k, v in self.visits.items()}
        return t

    def to_payload(self) -> Dict[str, Any]:
        return {"n_actions": self.n_actions, "init": self.init,
                "rows": [(list(k), list(self.values[k]), list(self.visits[k])) for k in sorted(self.values)]}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ActionTable":
        t = cls(int(payload["n_actions"]), float(payload["init"]))
        for k, vals, vis in payload["rows"]:
            if len(vals) != t.n_actions or len(vis) != t.n_actions:
                raise ConfigError("row width does not match n_actions")
            t.values[tuple(k)] = [float(x) for x in vals]
            t.visits[tuple(k)] = [int(x) for x in vis]
        return t

    @classmethod
    def merge(cls, base: "ActionTable", shards: Sequence["ActionTable"]) -> "ActionTable":
        """Visit-weighted merge of tables that were each trained from a copy of ``base``.

        For every cell, ``base`` keeps weight equal to its own visits and each
        shard gets weight equal to the visits it added; cells no shard touched
        keep the base value. Visits add up.
        """
        if any(s.n_actions != base.n_actions for s in shards):
            raise ConfigError("cannot merge tables with different action counts")
        out = base.copy()
        keys = sorted({k for s in shards for k in s.values} | set(base.values))
        for key in keys:
            b_vals = base.peek(key)
            b_vis = list(base.visits.get(key, [0] * base.n_actions))
            vals, vis = out.row(key), out.visits[key]
            for a in range(base.n_actions):
                num = b_vals[a] * b_vis[a]
                den = b_vis[a]
                for s in shards:
                    added = s.visits.get(key, [0] * s.n_actions)[a] - b_vis[a]
                    if added > 0:
                        num += s.values[key][a] * added
                        den += added
                vals[a] = num / den if den > 0 else b_vals[a]
                vis[a] = den
        return out


@dataclass(frozen=True)
class LearningSchedule:
    alpha0: float = 0.6
    alpha_min: float = 0.02
    alpha_power: float = 0.6
    epsilon0: float = 0.35
    epsilon_min: float = 0.03
    epsilon_decay: float = 0.996
    gamma: float = 0.1
    td_clip: float = 1.0  # Huber-style cap on each TD step (normalised units); 0 disables

    def __post_init__(self) -> None:
        if self.td_clip < 0:
            raise ConfigError("td_clip must be non-negative")
        if not 0 < self.alpha_min <= self.alpha0 <= 1:
            raise ConfigError("need 0 < alpha_min <= alpha0 <= 1")
        if not 0 <= self.epsilon_min <= self.epsilon0 <= 1 or not 0 < self.epsilon_decay <= 1:
            raise ConfigError("need 0 <= epsilon_min <= epsilon0 <= 1 and 0 < epsilon_decay <= 1")
        if not 0 <= self.gamma < 1 or self.alpha_power <= 0:
            raise ConfigError("gamma must be within [0, 1) and alpha_power positive")

    def lr(self, visits: int) -> float:
        return max(self.alpha_min, self.alpha0 / (1 + max(0, visits)) ** self.alpha_power)

    def epsilon(self, step: int) -> float:
        return max(self.epsilon_min, self.epsilon0 * self.epsilon_decay ** max(0, step))


class RewardScaler:
    """Adaptive reward normalisation: winsorised EWMA of |reward| with a floor, then clipping.

    Each observed magnitude is capped at ``clip * scale`` before it enters the
    average, so rare extreme rewards move the scale by a bounded amount.
    """

    def __init__(self, beta: float = 0.02, floor_wei: int = 10**12, clip: float = 5.0) -> None:
        if not 0 < beta <= 1 or floor_wei <= 0 or clip <= 0:
            raise ConfigError("need 0 < beta <= 1, floor_wei > 0, clip > 0")
        self.beta = beta
        self.floor = float(floor_wei)
        self.clip = clip
        self.scale = float(floor_wei)
        self.observed = 0

    def observe(self, reward_wei: int) -> None:
        mag = abs(float(reward_wei))
        if mag == 0:
            return
        self.observed += 1
        if self.observed == 1:
            self.scale = max(self.floor, mag)
        else:
            mag = min(mag, self.clip * self.scale)
            self.scale = max(self.floor, (1 - self.beta) * self.scale + self.beta * mag)

    def normalize(self, reward_wei: int) -> float:
        x = float(reward_wei) / self.scale
        return max(-self.clip, min(self.clip, x))

    def to_dict(self) -> Dict[str, Any]:
        return {"scale_wei": self.scale, "observed": self.observed, "clip": self.clip}


def _seed_int(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class _AgentBase:
    kind = "base"

    def __init__(self, actions: Optional[Sequence[Action]] = None, bins: Optional[StateBins] = None,
                 seed: int = 0, table: Optional[ActionTable] = None,
                 scaler: Optional[RewardScaler] = None) -> None:
        self.actions: Tuple[Action, ...] = tuple(actions) if actions is not None else action_grid()
        if not self.actions:
            raise ConfigError("agent needs at least one action")
        self.bins = bins or StateBins()
        self.rng = random.Random(seed)
        self.table = table if table is not None else ActionTable(len(self.actions))
        if self.table.n_actions != len(self.actions):
            raise ConfigError("table width does not match the action grid")
        self.scaler = scaler or RewardScaler()
        self.steps = 0
        self.updates = 0
        self.explored = 0
        self.cumulative_reward_wei = 0

    def key(self, obs: Union[Observation, Key]) -> Key:
        return obs if isinstance(obs, tuple) else self.bins.encode(obs)

    def greedy(self, obs: Union[Observation, Key]) -> int:
        return self.table.best(self.key(obs))

    def policy(self) -> Dict[Key, Action]:
        """Greedy action for every state seen so far."""
        return {k: self.actions[self.table.best(k)] for k in sorted(self.table.values)}

    def act(self, obs: Union[Observation, Key], explore: bool = True) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def learn(self, obs: Union[Observation, Key], action: int, reward_wei: int,
              next_obs: Optional[Union[Observation, Key]] = None, done: bool = False) -> float:
        raise NotImplementedError  # pragma: no cover - abstract

    def stats(self) -> Dict[str, Any]:
        return {"kind": self.kind, "steps": self.steps, "updates": self.updates, "explored": self.explored,
                "states": len(self.table), "visits": self.table.total_visits(),
                "cumulative_reward_wei": self.cumulative_reward_wei, "reward_scale": self.scaler.to_dict()}


class QLearningAgent(_AgentBase):
    """Tabular online Q-learning with epsilon-greedy exploration.

    Values start at ``optimistic_init`` (normalised units), so every action is
    tried in a new state before the agent settles; skipping earns exactly 0.
    """

    kind = "q_learning"

    def __init__(self, actions: Optional[Sequence[Action]] = None, bins: Optional[StateBins] = None,
                 schedule: Optional[LearningSchedule] = None, seed: int = 0,
                 table: Optional[ActionTable] = None, scaler: Optional[RewardScaler] = None,
                 optimistic_init: float = 1.0) -> None:
        if table is None:
            table = ActionTable(len(tuple(actions) if actions is not None else action_grid()), optimistic_init)
        super().__init__(actions, bins, seed, table, scaler)
        self.schedule = schedule or LearningSchedule()

    @property
    def epsilon(self) -> float:
        return self.schedule.epsilon(self.steps)

    def act(self, obs: Union[Observation, Key], explore: bool = True) -> int:
        key = self.key(obs)
        if explore:
            self.steps += 1
            if self.rng.random() < self.schedule.epsilon(self.steps - 1):
                self.explored += 1
                return self.rng.randrange(len(self.actions))
        return self.table.best(key)

    def learn(self, obs: Union[Observation, Key], action: int, reward_wei: int,
              next_obs: Optional[Union[Observation, Key]] = None, done: bool = False) -> float:
        """One Q-learning update; returns the TD error (normalised units)."""
        if not 0 <= action < len(self.actions):
            raise ConfigError(f"action index {action} out of range")
        reward_wei = int(reward_wei)
        self.cumulative_reward_wei += reward_wei
        self.scaler.observe(reward_wei)
        r = self.scaler.normalize(reward_wei)
        key = self.key(obs)
        row = self.table.row(key)
        counts = self.table.counts(key)
        future = 0.0
        if not done and next_obs is not None and self.schedule.gamma > 0:
            future = max(self.table.peek(self.key(next_obs)))
        target = r + self.schedule.gamma * future
        td = target - row[action]
        step = td
        if self.schedule.td_clip > 0:
            step = max(-self.schedule.td_clip, min(self.schedule.td_clip, td))
        row[action] += self.schedule.lr(counts[action]) * step
        counts[action] += 1
        self.updates += 1
        return td


class PolicyGradientAgent(_AgentBase):
    """Tabular softmax policy trained with REINFORCE and a per-state baseline."""

    kind = "policy_gradient"

    def __init__(self, actions: Optional[Sequence[Action]] = None, bins: Optional[StateBins] = None,
                 lr: float = 0.15, baseline_beta: float = 0.1, temperature: float = 1.0, seed: int = 0,
                 table: Optional[ActionTable] = None, scaler: Optional[RewardScaler] = None) -> None:
        super().__init__(actions, bins, seed, table, scaler)
        if lr <= 0 or not 0 < baseline_beta <= 1 or temperature <= 0:
            raise ConfigError("need lr > 0, 0 < baseline_beta <= 1, temperature > 0")
        self.lr = lr
        self.baseline_beta = baseline_beta
        self.temperature = temperature
        self.baselines: Dict[Key, float] = {}

    def probs(self, obs: Union[Observation, Key]) -> List[float]:
        prefs = self.table.peek(self.key(obs))
        m = max(prefs)
        ex = [math.exp((p - m) / self.temperature) for p in prefs]
        total = sum(ex)
        return [e / total for e in ex]

    def act(self, obs: Union[Observation, Key], explore: bool = True) -> int:
        key = self.key(obs)
        if not explore:
            return self.table.best(key)
        self.steps += 1
        p = self.probs(key)
        x = self.rng.random()
        acc = 0.0
        for i, pi in enumerate(p):
            acc += pi
            if x < acc:
                if i != self.table.best(key):
                    self.explored += 1
                return i
        return len(p) - 1

    def learn(self, obs: Union[Observation, Key], action: int, reward_wei: int,
              next_obs: Optional[Union[Observation, Key]] = None, done: bool = False) -> float:
        """REINFORCE step; returns the advantage (normalised units)."""
        if not 0 <= action < len(self.actions):
            raise ConfigError(f"action index {action} out of range")
        reward_wei = int(reward_wei)
        self.cumulative_reward_wei += reward_wei
        self.scaler.observe(reward_wei)
        r = self.scaler.normalize(reward_wei)
        key = self.key(obs)
        p = self.probs(key)
        prefs = self.table.row(key)
        counts = self.table.counts(key)
        base = self.baselines.get(key, 0.0)
        adv = r - base
        for b in range(len(prefs)):
            grad = (1.0 if b == action else 0.0) - p[b]
            prefs[b] += self.lr * adv * grad
        counts[action] += 1
        self.baselines[key] = base + self.baseline_beta * (r - base)
        self.updates += 1
        return adv


Agent = Union[QLearningAgent, PolicyGradientAgent]


def make_agent(kind: str = "q_learning", seed: int = 0, table: Optional[ActionTable] = None,
               actions: Optional[Sequence[Action]] = None, **kwargs: Any) -> Agent:
    if kind == QLearningAgent.kind:
        return QLearningAgent(actions=actions, seed=seed, table=table, **kwargs)
    if kind == PolicyGradientAgent.kind:
        return PolicyGradientAgent(actions=actions, seed=seed, table=table, **kwargs)
    raise ConfigError(f"unknown agent kind {kind!r}")


# ---------------------------------------------------------------------------
# Synthetic noisy bandit (fast convergence checks)
# ---------------------------------------------------------------------------


class NoisyBanditEnvironment:
    """Contextual bandit with known expected rewards and heavy-tailed noise.

    Each context ``c`` has an optimal arm ``(c * 7 + 3) % n_actions``; its
    expected reward is ``gap_wei`` above every other arm. Noise is Gaussian
    with standard deviation ``noise_wei`` plus, with probability
    ``outlier_prob``, a spike of ``+/- outlier_wei``.
    """

    def __init__(self, n_contexts: int = 4, n_actions: int = 6, gap_wei: int = 10**15,
                 base_wei: int = 10**15, noise_wei: int = 10**15, outlier_prob: float = 0.0,
                 outlier_wei: int = 10**18, seed: int = 0) -> None:
        if n_contexts < 1 or n_actions < 2:
            raise ConfigError("need at least one context and two actions")
        self.n_contexts = n_contexts
        self.n_actions = n_actions
        self.gap = gap_wei
        self.base = base_wei
        self.noise = noise_wei
        self.outlier_prob = outlier_prob
        self.outlier = outlier_wei
        self.rng = random.Random(seed)
        self.context = 0

    def optimal(self, context: int) -> int:
        return (context * 7 + 3) % self.n_actions

    def expected(self, context: int, action: int) -> int:
        return self.base + (self.gap if action == self.optimal(context) else 0)

    def reset(self) -> Key:
        self.context = self.rng.randrange(self.n_contexts)
        return (self.context,)

    def step(self, action: int) -> Tuple[Key, int]:
        r = self.expected(self.context, action) + int(self.rng.gauss(0, self.noise))
        if self.outlier_prob and self.rng.random() < self.outlier_prob:
            r += self.outlier if self.rng.random() < 0.5 else -self.outlier
        return self.reset(), r


def train_bandit(agent: Agent, env: NoisyBanditEnvironment, steps: int) -> Dict[str, Any]:
    key = env.reset()
    total = 0
    for _ in range(steps):
        a = agent.act(key)
        nxt, r = env.step(a)
        agent.learn(key, a, r, nxt, done=True)
        total += r
        key = nxt
    correct = sum(agent.greedy((c,)) == env.optimal(c) for c in range(env.n_contexts))
    return {"steps": steps, "reward_wei": total, "optimal_contexts": correct, "contexts": env.n_contexts}


# ---------------------------------------------------------------------------
# Sandwich environment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketRegime:
    name: str
    noise_bps: float  # std-dev of each step's noise trade, bps of the input reserve
    gas_vol: float  # std-dev of the per-step log change of the base fee
    delay_mean_ms: int  # agent relay latency
    delay_jitter_ms: int
    victim_min_bps: float  # victim input as bps of the input reserve (log-uniform range)
    victim_max_bps: float

    def __post_init__(self) -> None:
        if self.noise_bps < 0 or self.gas_vol < 0 or self.delay_mean_ms < 0 or self.delay_jitter_ms < 0:
            raise ConfigError("regime parameters must be non-negative")
        if not 0 < self.victim_min_bps <= self.victim_max_bps:
            raise ConfigError("need 0 < victim_min_bps <= victim_max_bps")


REGIMES: Dict[str, MarketRegime] = {
    "calm": MarketRegime("calm", 2.0, 0.02, 120, 40, 5.0, 60.0),
    "volatile": MarketRegime("volatile", 15.0, 0.08, 220, 120, 3.0, 120.0),
    "extreme": MarketRegime("extreme", 60.0, 0.25, 350, 300, 1.0, 200.0),
}


@dataclass(frozen=True)
class EnvConfig:
    seed: int = 0
    regime: str = "volatile"
    reserve_weth: int = 4_000 * E18
    reserve_usdc: int = 12_000_000 * E6
    base_fee_wei: int = 20 * GWEI
    min_base_fee_wei: int = GWEI
    max_base_fee_wei: int = 500 * GWEI
    cutoff_ms: Tuple[int, int] = (250, 900)  # builder seals at a uniform time in this window
    episode_steps: int = 64
    grid: int = 10
    refine: int = 5
    max_frontrun_bps: int = 2_000
    tip_wei: int = GWEI
    competitors: Optional[Tuple[Dict[str, Any], ...]] = None  # CompetitorBot kwargs; None = defaults
    measure_overbid: bool = True
    native: str = "WETH"

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ConfigError(f"unknown regime {self.regime!r}")
        if self.episode_steps < 1 or self.grid < 1 or self.refine < 0:
            raise ConfigError("episode_steps and grid must be >= 1, refine >= 0")
        lo, hi = self.cutoff_ms
        if not 0 <= lo <= hi:
            raise ConfigError("cutoff_ms must be an ordered non-negative window")
        if not 0 < self.min_base_fee_wei <= self.base_fee_wei <= self.max_base_fee_wei:
            raise ConfigError("need 0 < min_base_fee <= base_fee <= max_base_fee")

    def bots(self) -> List[CompetitorBot]:
        if self.competitors is None:
            return default_competitors()
        return [CompetitorBot(**dict(c)) for c in self.competitors]


@dataclass
class StepResult:
    observation: Observation
    reward_wei: int
    done: bool
    info: Dict[str, Any]


class SandwichEnvironment:
    """One victim per step, competing bots, a greedy builder and exact wei rewards."""

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self.config = config or EnvConfig()
        self.regime = REGIMES[self.config.regime]
        self.rng = random.Random(_seed_int("env", self.config.seed))
        self.bots = self.config.bots()
        self.total_steps = 0
        self.episode = 0
        self._reset_market()

    # -- market --------------------------------------------------------------

    def _reset_market(self) -> None:
        cfg = self.config
        self.pool_id = f"v2:{cfg.native}/USDC"
        self.state = BuilderState.from_pools([V2Pool(cfg.native, "USDC", cfg.reserve_weth, cfg.reserve_usdc)])
        self.initial_price = (cfg.reserve_usdc, cfg.reserve_weth)
        self.base_fee = cfg.base_fee_wei
        self.prev_base_fee = self.base_fee
        self.prev_reserve = cfg.reserve_weth
        self.episode_step = 0
        self._next_victim()

    @property
    def pool(self) -> V2Pool:
        p = self.state.pools[self.pool_id]
        assert isinstance(p, V2Pool)
        return p

    def _noise_trade(self) -> None:
        """Mean-reverting random trade straight on the pool (other traders between blocks)."""
        pool = self.pool
        sigma = self.regime.noise_bps
        if sigma <= 0:
            return
        r_usdc, r_weth = self.initial_price
        dev = math.log((pool.reserve1 * r_weth) / (pool.reserve0 * r_usdc))
        p_zfo = 0.5 + 0.45 * math.tanh(20 * dev)
        zfo = self.rng.random() < p_zfo
        reserve_in = pool.reserve0 if zfo else pool.reserve1
        # audit: allow code.float_truncation (a random size draw, not exact wei math)
        amount = int(reserve_in * abs(self.rng.gauss(0, sigma)) / 10_000)
        if amount > 0:
            pool.apply_swap(pool.quote_exact_in(amount, zfo), zfo)

    def _gas_step(self) -> None:
        cfg = self.config
        self.prev_base_fee = self.base_fee
        move = math.exp(self.rng.gauss(0, self.regime.gas_vol))
        self.base_fee = int(min(cfg.max_base_fee_wei, max(cfg.min_base_fee_wei, self.base_fee * move)))

    def _next_victim(self) -> None:
        pool = self.pool
        token_in = pool.tokens()[self.rng.randrange(2)]
        zfo = token_in == pool.token0
        reserve_in = pool.reserve0 if zfo else pool.reserve1
        lo, hi = math.log(self.regime.victim_min_bps), math.log(self.regime.victim_max_bps)
        size_bps = math.exp(self.rng.uniform(lo, hi))
        # audit: allow code.float_truncation (a random size draw, not exact wei math)
        amount = max(1, int(reserve_in * size_bps / 10_000))
        tolerance = self.rng.choice((30, 50, 100, 200, 300, 500, 1_000))
        sender = f"user{self.rng.randrange(10_000):04d}"
        tip = self.rng.randrange(1, 6) * GWEI // 2
        n = self.total_steps
        tx_hash = f"0x{n:08x}{self.rng.getrandbits(96):024x}"
        probe = BuilderTx("probe", sender, 0, 10**6, self.base_fee * 3 + tip, tip, TxKind.SWAP,
                          {"route": [self.pool_id], "token_in": token_in, "amount_in": amount})
        try:
            hops, _ = _quote_hops(self.state, probe)
            out = hops[-1][3].amount_out
        except SwapReverted:  # pragma: no cover - v2 quotes never fail for positive amounts
            out = 0
        gas = estimate_swap_gas([pool]) * 3 // 2
        self.victim = BuilderTx(tx_hash, sender, self.state.nonce(sender), gas, self.base_fee * 3 + tip, tip,
                                TxKind.SWAP, {"route": [self.pool_id], "token_in": token_in, "amount_in": amount,
                                              "amount_out_min": min_amount_out(out, tolerance),
                                              "tolerance_bps": tolerance})
        self.victim_size_bps = int(round(size_bps))
        self.agent_delay = max(0, int(self.rng.gauss(self.regime.delay_mean_ms, self.regime.delay_jitter_ms)))
        self._oracle_cache: Optional[Tuple[Any]] = None

    @property
    def oracle(self):
        """Best sandwich on the current victim (grid + ternary search), cached per victim."""
        if self._oracle_cache is None:
            cfg = self.config
            self._oracle_cache = (best_sandwich(self.state.pools, self.victim, cfg.grid, cfg.refine,
                                                cfg.max_frontrun_bps, cfg.native),)
        return self._oracle_cache[0]

    # -- API -----------------------------------------------------------------

    def observe(self) -> Observation:
        pool = self.pool
        dr = (pool.reserve0 - self.prev_reserve) * 10_000 // max(1, self.prev_reserve)
        dg = (self.base_fee - self.prev_base_fee) * 10_000 // max(1, self.prev_base_fee)
        return Observation(int(dr), int(dg), self.agent_delay, self.victim_size_bps, len(self.bots))

    def reset(self) -> Observation:
        self.episode += 1
        self._reset_market()
        return self.observe()

    def _realized(self, before: Mapping[str, int], post: BuilderState, pools_before: Mapping[str, Any]) -> int:
        """Agent's balance change in native wei (ETH and WETH 1:1, other tokens at pre-block spot)."""
        after = post.fork.balances.get(AGENT, {})
        total = 0
        for tok in set(before) | set(after):
            delta = after.get(tok, 0) - before.get(tok, 0)
            if delta == 0:
                continue
            if tok in (ETH, self.config.native):
                total += delta
            else:
                v = native_value(pools_before, tok, abs(delta), self.config.native)
                total += v if delta > 0 else -v
        return total

    def agent_bundle(self, action: Action) -> Tuple[Optional[Bundle], Dict[str, Any]]:
        """The agent's bundle for ``action`` on the current victim, and its expected economics."""
        info: Dict[str, Any] = {"action": action.label()}
        oracle = self.oracle
        if action.skip:
            info["agent"] = "skipped"
            return None, info
        if oracle is None:
            info["agent"] = "no_opportunity"
            return None, info
        size = max(1, oracle.frontrun_in * action.size_bps // 10_000)
        q = sandwich_economics(self.state, self.victim, oracle.hop, size, self.base_fee, self.config.tip_wei,
                               self.config.native)
        if q is None or q.net_wei <= 0:
            info["agent"] = "unprofitable"
            return None, info
        bribe = q.net_wei * action.bid_bps // 10_000
        info.update(expected_net_wei=q.net_wei, bribe_wei=bribe)
        return make_sandwich_bundle(self.state, self.victim, q, self.base_fee, AGENT, bribe,
                                    self.config.tip_wei, tag=AGENT), info

    def step(self, action: Action) -> StepResult:
        cfg = self.config
        parent = self.state
        victim = self.victim
        oracle = self.oracle
        mine, info = self.agent_bundle(action)
        info["oracle_profit_native_wei"] = 0
        if oracle is not None:
            info["oracle_profit_native_wei"] = native_value(parent.pools, oracle.token_in, oracle.profit, cfg.native)
        cutoff = self.rng.uniform(*cfg.cutoff_ms)
        arrived: List[Bundle] = []
        late: List[str] = []
        bot_bundles: Dict[str, Optional[Bundle]] = {}
        for bot in self.bots:
            b = bot.build(parent, victim, oracle, self.base_fee, cfg.native)
            d = bot.sample_delay(self.rng)
            bot_bundles[bot.name] = b
            if b is None:
                continue
            if d <= cutoff:
                arrived.append(b)
            else:
                late.append(b.bundle_id)
        agent_late = False
        if mine is not None:
            if self.agent_delay <= cutoff:
                arrived.append(mine)
            else:
                agent_late = True
                late.append(mine.bundle_id)
        before = dict(parent.fork.balances.get(AGENT, {}))
        pools_before = {pid: p for pid, p in parent.pools.items()}
        outcome = run_competition(parent, arrived, self.base_fee, [victim])
        reward = self._realized(before, outcome.post_state, pools_before) if mine is not None else 0
        won = mine is not None and outcome.won(mine.bundle_id)
        winner = outcome.winners.get(victim.tx_hash, "")
        info.update(won=won, winner=winner, late=late, relay_delay_ms=self.agent_delay, cutoff_ms=int(cutoff),
                    evicted=dict(outcome.evicted), base_fee=self.base_fee)
        own = [t for t in outcome.block.txs if t.sender == AGENT]
        info["agent_gas_used"] = sum(t.gas_used for t in own)
        info["agent_fees_wei"] = sum(t.priority_fee_wei + t.base_fee_burned_wei for t in own)
        if mine is not None:
            info["agent"] = "won" if won else ("late" if agent_late else
                                               f"evicted:{outcome.evicted.get(mine.bundle_id, '')}")
        bots_out: Dict[str, str] = {}
        for bot in self.bots:
            b = bot_bundles[bot.name]
            if b is None:
                bots_out[bot.name] = "no_bundle"
                continue
            bw = outcome.won(b.bundle_id)
            bots_out[bot.name] = "won" if bw else ("late" if b.bundle_id in late else "evicted")
            bot.observe(bw)
        info["bots"] = bots_out
        if won and cfg.measure_overbid and mine is not None:
            rivals = [b for b in arrived if b is not mine]
            zero = make_sandwich_bundle(parent, victim, sandwich_economics(
                parent, victim, int(mine.meta["hop"]), int(mine.txs[0].amount_in), self.base_fee,
                cfg.tip_wei, cfg.native), self.base_fee, AGENT, 0, cfg.tip_wei, tag=AGENT)  # type: ignore[arg-type]
            quote = quote_eviction(parent, zero, rivals, self.base_fee, prefer=FeeStructure.COINBASE)
            info["min_bribe_wei"] = quote.extra_value_wei
            info["overbid_wei"] = mine.coinbase_transfer_wei - quote.extra_value_wei
        # Advance the market.
        self.state = outcome.post_state
        self.state.fork.mine()
        self.prev_reserve = self.pool.reserve0
        self._noise_trade()
        self._gas_step()
        self.total_steps += 1
        self.episode_step += 1
        done = self.episode_step >= cfg.episode_steps
        if done:
            self.episode += 1
            self._reset_market()
        else:
            self._next_victim()
        return StepResult(self.observe(), reward, done, info)


# ---------------------------------------------------------------------------
# Training, evaluation and benchmarking
# ---------------------------------------------------------------------------


@dataclass
class TrainingReport:
    agent: str
    steps: int
    rewards_wei: List[int]
    wins: int
    submitted: int
    late: int
    evicted: int
    skipped: int
    overbid_wei: List[int]
    bot_outcomes: Dict[str, Dict[str, int]]
    elapsed_s: float
    agent_stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def cumulative_reward_wei(self) -> int:
        return sum(self.rewards_wei)

    @property
    def win_rate(self) -> float:
        return self.wins / self.submitted if self.submitted else 0.0

    def moving_average(self, window: int = 32) -> List[float]:
        out: List[float] = []
        acc = 0
        for i, r in enumerate(self.rewards_wei):
            acc += r
            if i >= window:
                acc -= self.rewards_wei[i - window]
            out.append(acc / min(i + 1, window))
        return out

    def summary(self) -> Dict[str, Any]:
        half = len(self.rewards_wei) // 2
        first = sum(self.rewards_wei[:half])
        second = sum(self.rewards_wei[half:])
        return {
            "agent": self.agent,
            "steps": self.steps,
            "cumulative_reward_wei": self.cumulative_reward_wei,
            "first_half_reward_wei": first,
            "second_half_reward_wei": second,
            "wins": self.wins,
            "submitted": self.submitted,
            "win_rate": round(self.win_rate, 4),
            "late": self.late,
            "evicted": self.evicted,
            "skipped": self.skipped,
            "mean_overbid_wei": (sum(self.overbid_wei) // len(self.overbid_wei)) if self.overbid_wei else 0,
            "bot_outcomes": {k: dict(v) for k, v in self.bot_outcomes.items()},
            "elapsed_s": round(self.elapsed_s, 4),
            "steps_per_s": round(self.steps / self.elapsed_s, 1) if self.elapsed_s > 0 else 0.0,
            "agent_stats": dict(self.agent_stats),
        }


def run_agent(agent: Agent, env: SandwichEnvironment, steps: int, learn: bool = True) -> TrainingReport:
    """Run ``steps`` environment steps; learn online after each one unless ``learn`` is False."""
    if steps < 0:
        raise ConfigError("steps must be non-negative")
    started = time.perf_counter()
    obs = env.observe()
    rewards: List[int] = []
    overbid: List[int] = []
    wins = submitted = late = evicted = skipped = 0
    bots: Dict[str, Dict[str, int]] = {}
    for _ in range(steps):
        a = agent.act(obs, explore=learn)
        res = env.step(agent.actions[a])
        if learn:
            agent.learn(obs, a, res.reward_wei, res.observation, res.done)
        rewards.append(res.reward_wei)
        status = res.info.get("agent", "")
        if status in ("skipped", "no_opportunity", "unprofitable"):
            skipped += 1
        else:
            submitted += 1
            if status == "won":
                wins += 1
            elif status == "late":
                late += 1
            else:
                evicted += 1
        if "overbid_wei" in res.info:
            overbid.append(res.info["overbid_wei"])
        for name, st in res.info.get("bots", {}).items():
            d = bots.setdefault(name, {})
            d[st] = d.get(st, 0) + 1
        obs = res.observation
    return TrainingReport(agent.kind, steps, rewards, wins, submitted, late, evicted, skipped, overbid, bots,
                          time.perf_counter() - started, agent.stats())


class FixedPolicy(_AgentBase):
    """Always plays one action; a non-learning baseline."""

    kind = "fixed"

    def __init__(self, action: Action, seed: int = 0) -> None:
        super().__init__((action,), seed=seed)

    def act(self, obs: Union[Observation, Key], explore: bool = True) -> int:
        self.steps += explore
        return 0

    def learn(self, obs: Union[Observation, Key], action: int, reward_wei: int,
              next_obs: Optional[Union[Observation, Key]] = None, done: bool = False) -> float:
        self.cumulative_reward_wei += int(reward_wei)
        return 0.0


def benchmark_strategies(env_config: Optional[EnvConfig] = None, train_steps: int = 400, eval_steps: int = 200,
                         seed: int = 0, baselines: Optional[Mapping[str, Action]] = None) -> Dict[str, Any]:
    """Trained Q-learning agent vs fixed-action baselines, each evaluated on the same seeded market."""
    cfg = env_config or EnvConfig(seed=seed)
    baselines = dict(baselines or {"fixed_bid_30pct": Action(10_000, 3_000),
                                   "fixed_bid_90pct": Action(10_000, 9_000),
                                   "half_size_bid_50pct": Action(5_000, 5_000)})
    agent = QLearningAgent(seed=seed)
    train = run_agent(agent, SandwichEnvironment(_with_seed(cfg, _seed_int(cfg.seed, "train"))), train_steps)
    eval_cfg = _with_seed(cfg, _seed_int(cfg.seed, "eval"))
    out: Dict[str, Any] = {"train": train.summary(), "eval": {}}
    out["eval"]["q_learning"] = run_agent(agent, SandwichEnvironment(eval_cfg), eval_steps, learn=False).summary()
    for name, action in baselines.items():
        out["eval"][name] = run_agent(FixedPolicy(action), SandwichEnvironment(eval_cfg), eval_steps,
                                      learn=False).summary()
    ranking = sorted(out["eval"], key=lambda k: -out["eval"][k]["cumulative_reward_wei"])
    out["ranking"] = ranking
    return out


def _with_seed(cfg: EnvConfig, seed: int) -> EnvConfig:
    return EnvConfig(**{**{f: getattr(cfg, f) for f in cfg.__dataclass_fields__}, "seed": seed})


# ---------------------------------------------------------------------------
# Parallel policy iteration on the capped worker pool
# ---------------------------------------------------------------------------


def _env_config_dict(cfg: EnvConfig) -> Dict[str, Any]:
    return {f: getattr(cfg, f) for f in cfg.__dataclass_fields__}


def _policy_shard(task: Tuple[Any, ...]) -> Dict[str, Any]:
    """Worker entry point: train a copy of the table on one seeded environment."""
    kind, table_payload, env_cfg, seed, steps = task
    table = ActionTable.from_payload(table_payload)
    agent = make_agent(kind, seed=seed, table=table)
    env = SandwichEnvironment(EnvConfig(**{**env_cfg, "seed": seed}))
    report = run_agent(agent, env, steps)
    return {"table": agent.table.to_payload(), "summary": report.summary()}


@dataclass
class ParallelTrainingReport:
    kind: str
    rounds: int
    shards: int
    steps_per_shard: int
    workers: int
    parallel: bool
    table: ActionTable
    round_rewards_wei: List[int]
    round_win_rates: List[float]
    elapsed_s: float

    def agent(self, seed: int = 0) -> Agent:
        return make_agent(self.kind, seed=seed, table=self.table.copy())

    def summary(self) -> Dict[str, Any]:
        steps = self.rounds * self.shards * self.steps_per_shard
        return {
            "agent": self.kind,
            "rounds": self.rounds,
            "shards": self.shards,
            "steps_per_shard": self.steps_per_shard,
            "total_steps": steps,
            "workers": self.workers,
            "parallel": self.parallel,
            "states": len(self.table),
            "visits": self.table.total_visits(),
            "round_rewards_wei": list(self.round_rewards_wei),
            "round_win_rates": [round(x, 4) for x in self.round_win_rates],
            "cumulative_reward_wei": sum(self.round_rewards_wei),
            "elapsed_s": round(self.elapsed_s, 4),
            "steps_per_s": round(steps / self.elapsed_s, 1) if self.elapsed_s > 0 else 0.0,
        }


def parallel_policy_iteration(env_config: Optional[EnvConfig] = None, rounds: int = 2,
                              steps_per_shard: int = 64, shards: int = RL_SHARDS, kind: str = "q_learning",
                              seed: int = 0, executor: Optional[WorkerPool] = None,
                              workers: Optional[int] = None, table: Optional[ActionTable] = None
                              ) -> ParallelTrainingReport:
    """Train ``shards`` independent copies per round on the worker pool and merge them."""
    if rounds < 1 or steps_per_shard < 0 or shards < 1:
        raise ConfigError("rounds and shards must be >= 1, steps_per_shard >= 0")
    cfg = env_config or EnvConfig(seed=seed)
    base = table.copy() if table is not None else make_agent(kind).table
    env_cfg = _env_config_dict(cfg)
    own = executor is None
    wp = WorkerPool(workers) if own else executor
    started = time.perf_counter()
    rewards: List[int] = []
    rates: List[float] = []
    parallel = False
    try:
        for r in range(rounds):
            tasks = [(kind, base.to_payload(), env_cfg, _seed_int(seed, cfg.seed, r, s), steps_per_shard)
                     for s in range(shards)]
            parallel = parallel or wp.will_parallelize(len(tasks))
            results = wp.map(_policy_shard, tasks)
            base = ActionTable.merge(base, [ActionTable.from_payload(x["table"]) for x in results])
            rewards.append(sum(x["summary"]["cumulative_reward_wei"] for x in results))
            sub = sum(x["summary"]["submitted"] for x in results)
            rates.append(sum(x["summary"]["wins"] for x in results) / sub if sub else 0.0)
    finally:
        if own:
            wp.close()
    return ParallelTrainingReport(kind, rounds, shards, steps_per_shard, wp.workers, parallel, base, rewards, rates,
                                  time.perf_counter() - started)


class ReinforcementSimulation:
    """Controller entry: parallel policy iteration, then a greedy evaluation and a static-bot breakdown."""

    def __init__(self, seed: int = 0, rounds: int = 2, steps_per_shard: int = 48, eval_steps: int = 96,
                 regime: str = "volatile", kind: str = "q_learning", shards: int = RL_SHARDS) -> None:
        if shards < 1:
            raise ConfigError("shards must be >= 1")
        self.env_config = EnvConfig(seed=seed, regime=regime)
        self.seed = seed
        self.rounds = rounds
        self.steps_per_shard = steps_per_shard
        self.eval_steps = eval_steps
        self.kind = kind
        self.shards = shards

    def run(self, executor: Optional[WorkerPool] = None) -> Dict[str, Any]:
        train = parallel_policy_iteration(self.env_config, self.rounds, self.steps_per_shard, shards=self.shards,
                                          kind=self.kind, seed=self.seed, executor=executor)
        agent = train.agent(self.seed)
        ev = run_agent(agent, SandwichEnvironment(_with_seed(self.env_config, _seed_int(self.seed, "eval"))),
                       self.eval_steps, learn=False)
        return {"train": train.summary(), "eval": ev.summary(),
                "policy_states": len(agent.policy()), "regime": self.env_config.regime}


__all__ = [
    "AGENT",
    "REGIMES",
    "RL_SHARDS",
    "Action",
    "ActionTable",
    "EnvConfig",
    "FixedPolicy",
    "LearningSchedule",
    "MarketRegime",
    "NoisyBanditEnvironment",
    "Observation",
    "ParallelTrainingReport",
    "PolicyGradientAgent",
    "QLearningAgent",
    "ReinforcementSimulation",
    "RewardScaler",
    "SandwichEnvironment",
    "StateBins",
    "StepResult",
    "TrainingReport",
    "action_grid",
    "benchmark_strategies",
    "bucket",
    "make_agent",
    "parallel_policy_iteration",
    "run_agent",
    "train_bandit",
]
