"""Layer 09: high-density policy-gradient path fuzzer (MAX Brutal Simulation).

Status: experimental. A stress / throughput harness for the RL policy; its results
are about the simulator, not about real searcher behaviour.

A *path* is one REINFORCE trajectory: a
:class:`~simulator.reinforcement_engine.PolicyGradientAgent` starts from a
shared base table and walks a seeded environment for ``steps`` steps,
learning online after every step. The fuzzer generates thousands of paths
whose parameters are randomised per path and runs them concurrently on the
worker pool:

* market regime (round-robin over calm / volatile / extreme, so every regime
  gets the same number of paths),
* environment seed (victims, noise trades, gas walk, rival latencies),
* softmax temperature (log-uniform in ``[0.25, 4]``),
* learning rate (log-uniform in ``[0.02, 0.5]``) and baseline step size,
* the builder's sealing cutoff window (how late bundles may arrive).

Every parameter of path ``i`` is derived from ``(seed, i)`` alone, so adding
paths never changes earlier ones.

Path invariants
---------------
Checked on every step of every path; a violation is reported with the path
index, step and detail:

* ``pg.simplex`` - the policy is a probability distribution: every
  probability finite and in ``[0, 1]``, total ``1`` within ``1e-9``.
* ``pg.exact_reward`` - the environment's reward is an exact ``int`` (wei).
* ``pg.normalized_bounds`` - the normalised learning signal stays inside
  ``[-clip, clip]``.
* ``pg.advantage_bounds`` - the advantage stays inside ``[-2 clip, 2 clip]``
  (the baseline is a convex combination of clipped rewards).
* ``pg.zero_sum_update`` - a REINFORCE step leaves the row's preference sum
  unchanged, because ``sum_b (1[b = a] - p_b) = 0``.
* ``pg.finite_table`` - every preference is finite after the update.
* ``pg.reward_accounting`` - the agent's cumulative reward equals the exact
  sum of the path's step rewards.
* ``pg.visit_accounting`` - the table gained exactly one visit per step.

Parallel layout and merge
-------------------------
Paths are cut into fixed batches of ``batch`` paths (one worker task per
batch). Each batch returns partial sums ``(sum value * added_visits,
sum added_visits)`` for every cell its paths touched; the parent folds the
batches in index order into the base table. The result equals the
visit-weighted merge of :meth:`ActionTable.merge` over all paths (up to float
summation order) and, because the batch layout depends only on ``batch``,
it is bit-for-bit identical for any worker count. Under
:class:`~simulator.config.CpuPolicy.FULL` the batches spread over every
logical CPU.

``env_kind="bandit"`` swaps the sandwich market for the Layer 08
:class:`~simulator.reinforcement_engine.NoisyBanditEnvironment` (known optimal
arm per context), which is how the tests check that thousands of merged
paths really learn the right policy.
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import ConfigError
from .parallel import WorkerPool
from .reinforcement_engine import (
    REGIMES,
    Action,
    ActionTable,
    EnvConfig,
    NoisyBanditEnvironment,
    PolicyGradientAgent,
    SandwichEnvironment,
    _seed_int,
    action_grid,
)

Key = Tuple[int, ...]

PATH_REGIMES: Tuple[str, ...] = tuple(REGIMES)
ENV_KINDS: Tuple[str, ...] = ("sandwich", "bandit")
VIOLATION_LIMIT = 64  # violations each batch reports in full (all are counted)
SIMPLEX_TOL = 1e-9
TEMPERATURE_RANGE = (0.25, 4.0)
LR_RANGE = (0.02, 0.5)
BASELINE_RANGE = (0.05, 0.3)
CUTOFF_LO_MS = (150, 400)
CUTOFF_WIDTH_MS = (200, 700)
PATH_RULES: Tuple[str, ...] = (
    "pg.simplex",
    "pg.exact_reward",
    "pg.normalized_bounds",
    "pg.advantage_bounds",
    "pg.zero_sum_update",
    "pg.finite_table",
    "pg.reward_accounting",
    "pg.visit_accounting",
)


@dataclass(frozen=True)
class PathSpec:
    """Randomised parameters of one policy-gradient path."""

    index: int
    seed: int
    regime: str
    steps: int
    temperature: float
    lr: float
    baseline_beta: float
    cutoff_ms: Tuple[int, int]

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "seed": self.seed, "regime": self.regime, "steps": self.steps,
                "temperature": round(self.temperature, 6), "lr": round(self.lr, 6),
                "baseline_beta": round(self.baseline_beta, 6), "cutoff_ms": list(self.cutoff_ms)}


def _log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def make_path_spec(index: int, steps: int, seed: int = 0, regimes: Sequence[str] = PATH_REGIMES) -> PathSpec:
    """Parameters of path ``index``; depends only on ``(seed, index, steps, regimes)``."""
    if not regimes:
        raise ConfigError("need at least one regime")
    rng = random.Random(_seed_int("pg-path", seed, index))
    lo = rng.randint(*CUTOFF_LO_MS)
    return PathSpec(
        index=index,
        seed=_seed_int("pg-path-env", seed, index),
        regime=regimes[index % len(regimes)],
        steps=steps,
        temperature=_log_uniform(rng, *TEMPERATURE_RANGE),
        lr=_log_uniform(rng, *LR_RANGE),
        baseline_beta=rng.uniform(*BASELINE_RANGE),
        cutoff_ms=(lo, lo + rng.randint(*CUTOFF_WIDTH_MS)),
    )


def make_path_specs(paths: int, steps: int, seed: int = 0,
                    regimes: Sequence[str] = PATH_REGIMES) -> List[PathSpec]:
    if paths < 0 or steps < 1:
        raise ConfigError("paths must be >= 0 and steps >= 1")
    for r in regimes:
        if r not in REGIMES:
            raise ConfigError(f"unknown regime {r!r}")
    return [make_path_spec(i, steps, seed, regimes) for i in range(paths)]


def bandit_actions(n_actions: int) -> Tuple[Action, ...]:
    """Placeholder action labels for the bandit environment (only the index matters there)."""
    if n_actions < 2:
        raise ConfigError("bandit needs at least two actions")
    return tuple(Action(10_000, (i + 1) * 10_000 // (n_actions + 1)) for i in range(n_actions))


def _simplex_ok(p: Sequence[float]) -> bool:
    if not p:
        return False
    for x in p:
        if not math.isfinite(x) or x < -SIMPLEX_TOL or x > 1 + SIMPLEX_TOL:
            return False
    return abs(math.fsum(p) - 1.0) <= SIMPLEX_TOL


def _entropy_bits(p: Sequence[float]) -> float:
    return -sum(x * math.log2(x) for x in p if x > 0)


def softmax(prefs: Sequence[float], temperature: float = 1.0) -> List[float]:
    m = max(prefs)
    ex = [math.exp((x - m) / temperature) for x in prefs]
    total = sum(ex)
    return [e / total for e in ex]


@dataclass
class PathOutcome:
    """Result of one path (the trained table travels separately)."""

    index: int
    regime: str
    steps: int
    reward_wei: int
    wins: int
    submitted: int
    skipped: int
    violations: int
    temperature: float
    lr: float

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "regime": self.regime, "steps": self.steps, "reward_wei": self.reward_wei,
                "wins": self.wins, "submitted": self.submitted, "skipped": self.skipped,
                "violations": self.violations, "temperature": round(self.temperature, 6),
                "lr": round(self.lr, 6)}


def _make_env(spec: PathSpec, env_kind: str, env_overrides: Mapping[str, Any]) -> Any:
    if env_kind == "sandwich":
        params: Dict[str, Any] = {"seed": spec.seed, "regime": spec.regime, "cutoff_ms": spec.cutoff_ms}
        params.update(env_overrides)
        return SandwichEnvironment(EnvConfig(**params))
    if env_kind == "bandit":
        params = {"seed": spec.seed}
        params.update(env_overrides)
        return NoisyBanditEnvironment(**params)
    raise ConfigError(f"unknown env kind {env_kind!r}")


def run_path(spec: PathSpec, table: ActionTable, env_kind: str = "sandwich",
             env_overrides: Optional[Mapping[str, Any]] = None,
             actions: Optional[Sequence[Action]] = None,
             violation_limit: int = VIOLATION_LIMIT) -> Tuple[PolicyGradientAgent, PathOutcome,
                                                              List[Dict[str, Any]], int]:
    """Walk one path, learning online and checking the path invariants on every step.

    ``table`` is trained in place. Returns ``(agent, outcome, violations,
    checks)``; ``violations`` holds at most ``violation_limit`` entries while
    ``outcome.violations`` counts all of them.
    """
    overrides = dict(env_overrides or {})
    env = _make_env(spec, env_kind, overrides)
    if actions is None:
        actions = bandit_actions(env.n_actions) if env_kind == "bandit" else action_grid()
    agent = PolicyGradientAgent(actions=actions, seed=spec.seed, table=table, lr=spec.lr,
                                baseline_beta=spec.baseline_beta, temperature=spec.temperature)
    clip = agent.scaler.clip
    visits_before = table.total_visits()
    found: List[Dict[str, Any]] = []
    count = 0
    checks = 0

    def violate(rule: str, step: int, detail: str) -> None:
        nonlocal count
        count += 1
        if len(found) < violation_limit:
            found.append({"rule": rule, "path": spec.index, "step": step, "detail": detail})

    obs: Any = env.reset() if env_kind == "bandit" else env.observe()
    total = 0
    wins = submitted = skipped = 0
    for step in range(spec.steps):
        key = agent.key(obs)
        p = agent.probs(key)
        checks += 1
        if not _simplex_ok(p):
            violate("pg.simplex", step, f"sum={math.fsum(p)!r}")
        a = agent.act(obs)
        if env_kind == "bandit":
            next_obs, reward = env.step(a)
            done = True
        else:
            res = env.step(agent.actions[a])
            next_obs, reward, done = res.observation, res.reward_wei, res.done
            status = res.info.get("agent", "")
            if status in ("skipped", "no_opportunity", "unprofitable"):
                skipped += 1
            else:
                submitted += 1
                wins += status == "won"
        checks += 1
        if type(reward) is not int:
            violate("pg.exact_reward", step, f"reward type {type(reward).__name__}")
            reward = int(reward)
        before = math.fsum(agent.table.peek(key))
        adv = agent.learn(obs, a, reward, next_obs, done)
        row = agent.table.values[key]
        after = math.fsum(row)
        scale = max([1.0] + [abs(x) for x in row if math.isfinite(x)])
        checks += 3
        if not all(math.isfinite(x) for x in row):
            violate("pg.finite_table", step, "non-finite preference")
        elif abs(after - before) > 1e-9 * scale * len(row):
            violate("pg.zero_sum_update", step, f"row sum {before!r} -> {after!r}")
        if not (math.isfinite(adv) and abs(adv) <= 2 * clip + 1e-9):
            violate("pg.advantage_bounds", step, f"advantage={adv!r}")
        r_norm = agent.scaler.normalize(reward)
        if not abs(r_norm) <= clip:
            violate("pg.normalized_bounds", step, f"normalized={r_norm!r}")
        total += reward
        obs = next_obs
    checks += 2
    if agent.cumulative_reward_wei != total:
        violate("pg.reward_accounting", spec.steps, f"{agent.cumulative_reward_wei} != {total}")
    if agent.table.total_visits() - visits_before != spec.steps:
        violate("pg.visit_accounting", spec.steps,
                f"visits +{agent.table.total_visits() - visits_before} for {spec.steps} steps")
    outcome = PathOutcome(spec.index, spec.regime, spec.steps, total, wins, submitted, skipped, count,
                          spec.temperature, spec.lr)
    return agent, outcome, found, checks


def _accumulate(cells: Dict[Key, List[List[Any]]], base: ActionTable, trained: ActionTable) -> None:
    """Add ``trained``'s visit-weighted deltas over ``base`` to the partial sums in ``cells``."""
    n = base.n_actions
    for key in sorted(trained.values):
        vals = trained.values[key]
        vis = trained.visits[key]
        b_vis = base.visits.get(key, [0] * n)
        cell = None
        for a in range(n):
            added = vis[a] - b_vis[a]
            if added > 0:
                if cell is None:
                    cell = cells.setdefault(key, [[0.0] * n, [0] * n])
                cell[0][a] += vals[a] * added
                cell[1][a] += added


def _path_batch(task: Tuple[Any, ...]) -> Dict[str, Any]:
    """Worker entry point: run a batch of paths from the same base table and return partial sums."""
    base_payload, specs, env_kind, env_overrides, violation_limit = task
    cpu0 = time.process_time()
    base = ActionTable.from_payload(base_payload)
    actions = None
    if env_kind == "bandit":
        actions = bandit_actions(int(dict(env_overrides).get("n_actions", 6)))
    cells: Dict[Key, List[List[Any]]] = {}
    outcomes: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []
    violation_count = checks = steps = 0
    for spec in specs:
        trained = base.copy()  # run_path trains this copy in place
        _, outcome, found, n_checks = run_path(spec, trained, env_kind, env_overrides, actions, violation_limit)
        outcomes.append(outcome.to_dict())
        violation_count += outcome.violations
        room = violation_limit - len(violations)
        if room > 0:
            violations.extend(found[:room])
        checks += n_checks
        steps += outcome.steps
        _accumulate(cells, base, trained)
    return {"cells": cells, "outcomes": outcomes, "violations": violations, "violation_count": violation_count,
            "checks": checks, "steps": steps, "cpu_s": time.process_time() - cpu0, "pid": os.getpid()}


def merge_partials(base: ActionTable, partials: Sequence[Mapping[Key, Sequence[Sequence[Any]]]]) -> ActionTable:
    """Fold batch partial sums (in the given order) into ``base``: the visit-weighted path merge."""
    n = base.n_actions
    acc: Dict[Key, List[List[Any]]] = {}
    for part in partials:
        for key in sorted(part):
            num, den = part[key]
            if len(num) != n or len(den) != n:
                raise ConfigError("partial row width does not match the table")
            cell = acc.setdefault(key, [[0.0] * n, [0] * n])
            for a in range(n):
                cell[0][a] += num[a]
                cell[1][a] += den[a]
    out = base.copy()
    for key in sorted(acc):
        b_vals = base.peek(key)
        b_vis = list(base.visits.get(key, [0] * n))
        vals, vis = out.row(key), out.visits[key]
        for a in range(n):
            den = b_vis[a] + acc[key][1][a]
            vals[a] = (b_vals[a] * b_vis[a] + acc[key][0][a]) / den if den > 0 else b_vals[a]
            vis[a] = den
    return out


def _percentile(sorted_vals: Sequence[int], q: float) -> int:
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


@dataclass
class PathFuzzReport:
    env_kind: str
    paths: int
    steps_per_path: int
    batch: int
    batches: int
    table: ActionTable
    actions: Tuple[Action, ...]
    outcomes: List[Dict[str, Any]]
    violations: List[Dict[str, Any]]
    violation_count: int
    checks: int
    total_steps: int
    elapsed_s: float
    cpu_s: float
    worker_pids: List[int]
    workers: int
    parallel: bool
    specs: List[PathSpec] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.violation_count == 0

    @property
    def rewards_wei(self) -> List[int]:
        return [o["reward_wei"] for o in self.outcomes]

    def policy_stats(self) -> Dict[str, Any]:
        entropies: List[float] = []
        greedy: Dict[str, int] = {}
        for key in sorted(self.table.values):
            p = softmax(self.table.values[key])
            entropies.append(_entropy_bits(p))
            label = self.actions[self.table.best(key)].label() if self.env_kind == "sandwich" \
                else f"arm{self.table.best(key)}"
            greedy[label] = greedy.get(label, 0) + 1
        return {
            "states": len(self.table),
            "visits": self.table.total_visits(),
            "mean_entropy_bits": round(sum(entropies) / len(entropies), 6) if entropies else 0.0,
            "min_entropy_bits": round(min(entropies), 6) if entropies else 0.0,
            "max_entropy_bits": round(math.log2(self.table.n_actions), 6),
            "greedy_actions": dict(sorted(greedy.items(), key=lambda kv: (-kv[1], kv[0]))),
        }

    def summary(self) -> Dict[str, Any]:
        rewards = sorted(self.rewards_wei)
        total = sum(rewards)
        wins = sum(o["wins"] for o in self.outcomes)
        submitted = sum(o["submitted"] for o in self.outcomes)
        regimes: Dict[str, Dict[str, Any]] = {}
        for o in self.outcomes:
            r = regimes.setdefault(o["regime"], {"paths": 0, "reward_wei": 0, "wins": 0, "submitted": 0})
            r["paths"] += 1
            r["reward_wei"] += o["reward_wei"]
            r["wins"] += o["wins"]
            r["submitted"] += o["submitted"]
        for r in regimes.values():
            r["win_rate"] = round(r["wins"] / r["submitted"], 4) if r["submitted"] else 0.0
        best = max(self.outcomes, key=lambda o: (o["reward_wei"], -o["index"])) if self.outcomes else None
        capacity = self.elapsed_s * max(1, self.workers)
        return {
            "env": self.env_kind,
            "paths": self.paths,
            "steps_per_path": self.steps_per_path,
            "total_steps": self.total_steps,
            "batch": self.batch,
            "batches": self.batches,
            "workers": self.workers,
            "parallel": self.parallel,
            "distinct_worker_pids": len(set(self.worker_pids)),
            "elapsed_s": round(self.elapsed_s, 4),
            "cpu_s": round(self.cpu_s, 4),
            "paths_per_s": round(self.paths / self.elapsed_s, 1) if self.elapsed_s > 0 else 0.0,
            "steps_per_s": round(self.total_steps / self.elapsed_s, 1) if self.elapsed_s > 0 else 0.0,
            "parallel_efficiency": round(min(1.0, self.cpu_s / capacity), 4) if capacity > 0 else 0.0,
            "invariants": {
                "ok": self.ok,
                "rules": list(PATH_RULES),
                "checks": self.checks,
                "violation_count": self.violation_count,
                "violations": self.violations[:20],
            },
            "reward_wei": {
                "total": total,
                "mean": total // len(rewards) if rewards else 0,
                "min": rewards[0] if rewards else 0,
                "p05": _percentile(rewards, 0.05),
                "p50": _percentile(rewards, 0.5),
                "p95": _percentile(rewards, 0.95),
                "max": rewards[-1] if rewards else 0,
                "positive_paths": sum(1 for r in rewards if r > 0),
            },
            "wins": wins,
            "submitted": submitted,
            "win_rate": round(wins / submitted, 4) if submitted else 0.0,
            "regimes": {k: regimes[k] for k in sorted(regimes)},
            "policy": self.policy_stats(),
            "best_path": None if best is None else {k: best[k] for k in ("index", "regime", "reward_wei",
                                                                          "temperature", "lr")},
        }


class PolicyPathFuzzer:
    """Run ``paths`` randomised policy-gradient paths concurrently and merge what they learned."""

    def __init__(self, paths: int = 4_096, steps: int = 16, batch: int = 32, seed: int = 0,
                 regimes: Sequence[str] = PATH_REGIMES, env_kind: str = "sandwich",
                 env_overrides: Optional[Mapping[str, Any]] = None,
                 violation_limit: int = VIOLATION_LIMIT) -> None:
        if paths < 0 or steps < 1 or batch < 1:
            raise ConfigError("paths must be >= 0, steps and batch >= 1")
        if env_kind not in ENV_KINDS:
            raise ConfigError(f"env_kind must be one of {', '.join(ENV_KINDS)}")
        self.paths = paths
        self.steps = steps
        self.batch = batch
        self.seed = seed
        self.regimes = tuple(regimes)
        self.env_kind = env_kind
        self.env_overrides = dict(env_overrides or {})
        self.violation_limit = violation_limit
        if env_kind == "bandit":
            self.actions: Tuple[Action, ...] = bandit_actions(int(self.env_overrides.get("n_actions", 6)))
        else:
            self.actions = action_grid()

    def specs(self) -> List[PathSpec]:
        return make_path_specs(self.paths, self.steps, self.seed, self.regimes)

    def tasks(self, base: ActionTable) -> List[Tuple[Any, ...]]:
        payload = base.to_payload()
        specs = self.specs()
        return [(payload, specs[i:i + self.batch], self.env_kind, self.env_overrides, self.violation_limit)
                for i in range(0, len(specs), self.batch)]

    def run(self, executor: Optional[WorkerPool] = None, table: Optional[ActionTable] = None,
            workers: Optional[int] = None) -> PathFuzzReport:
        """Fuzz every path from ``table`` (default: a fresh uniform policy) and merge the results."""
        base = table.copy() if table is not None else ActionTable(len(self.actions))
        if base.n_actions != len(self.actions):
            raise ConfigError("table width does not match the action grid")
        tasks = self.tasks(base)
        own = executor is None and workers is not None
        wp = WorkerPool(workers) if own else executor
        started = time.perf_counter()
        try:
            if wp is None:
                parts = [_path_batch(t) for t in tasks]
                n_workers, parallel = 1, False
            else:
                parts = wp.map(_path_batch, tasks)
                n_workers, parallel = wp.workers, wp.will_parallelize(len(tasks))
        finally:
            if own and wp is not None:
                wp.close()
        elapsed = time.perf_counter() - started
        merged = merge_partials(base, [p["cells"] for p in parts])
        violations: List[Dict[str, Any]] = []
        for p in parts:
            violations.extend(p["violations"])
        return PathFuzzReport(
            env_kind=self.env_kind,
            paths=self.paths,
            steps_per_path=self.steps,
            batch=self.batch,
            batches=len(tasks),
            table=merged,
            actions=self.actions,
            outcomes=[o for p in parts for o in p["outcomes"]],
            violations=violations[: self.violation_limit],
            violation_count=sum(p["violation_count"] for p in parts),
            checks=sum(p["checks"] for p in parts),
            total_steps=sum(p["steps"] for p in parts),
            elapsed_s=elapsed,
            cpu_s=sum(p["cpu_s"] for p in parts),
            worker_pids=[p["pid"] for p in parts],
            workers=n_workers,
            parallel=parallel,
            specs=self.specs(),
        )


__all__ = [
    "ENV_KINDS",
    "PATH_REGIMES",
    "PATH_RULES",
    "PathFuzzReport",
    "PathOutcome",
    "PathSpec",
    "PolicyPathFuzzer",
    "bandit_actions",
    "make_path_spec",
    "make_path_specs",
    "merge_partials",
    "run_path",
    "softmax",
]
