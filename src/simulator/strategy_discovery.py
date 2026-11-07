"""Layer 13: local discovery of new MEV attack compositions.

The engine searches over sequences of primitive actions (``attack_vectors``:
swap, close, victim, mint, burn, liquidate) on seeded offline scenarios,
executes every candidate with the exact AMM math, and keeps the profitable
ones in a MAP-Elites archive keyed by the candidate's *structure signature*,
so the search keeps structurally different strategies instead of converging
on one. Structures that match none of the predefined attack templates
(sandwich, multi-victim sandwich, JIT, back-run arbitrage, liquidation
back-run, plain cyclic arbitrage, plain liquidation) are minimised, re-checked
on other seeds, labelled from their components and written to a local
strategy library on disk (JSON Lines), where each record can be replayed and
re-verified later.

Everything runs offline inside the simulator: scenarios are generated from a
seed, nothing reads a chain or sends a transaction, and a discovered pattern
is a description of a profitable action order *in this model*, not evidence
that it is profitable on mainnet. Gas is an estimate and the lending market
is a model (see ``attack_vectors``).

Genes (all parameters are integers, so genomes are JSON and replayable):

``("swap", pool, side, size_ppm)``    sell ``tokens()[side]``; size is ppm of the
                                      input-side reserve scale at execution time
``("close", pool, side)``             sell the whole positive balance of ``tokens()[side]``
``("victim", pool, side, rank)``      include the ``rank``-th largest pending swap
                                      on ``pool`` that sells ``tokens()[side]``
``("mint", pool, half_width, pct)``   concentrated position of ``pct``% of the active
                                      liquidity, ``half_width`` tick spacings around
                                      the current tick
``("burn",)``                         burn the oldest open position
``("liquidate", market, repay_ppm)``  liquidate the least healthy liquidatable loan

Pools and markets are referred to by index into the scenario's sorted lists,
which ``build_scenario`` keeps stable across seeds; victims are referred to by
role (pool, direction, size rank), so a genome can be re-run on other seeds.
After the last gene the engine settles: open positions are burned and every
non-base balance is swapped to base, so the score is in base-token wei.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from . import __version__
from .attack_vectors import ActionReverted, AttackResult, AttackState, Scenario, build_scenario
from .invariants import input_reserve_scale
from .mempool_emulator import PendingTx
from .slippage_engine import ConcentratedPool, MASEError, _direction

__all__ = [
    "KNOWN_TEMPLATES",
    "DiscoveryEngine",
    "DiscoveryReport",
    "Evaluation",
    "StrategyLibrary",
    "classify",
    "default_library_dir",
    "evaluate_genome",
    "label_for",
    "minimize",
    "random_genome",
    "signature_of",
]

Gene = Tuple[Any, ...]
Genome = Tuple[Gene, ...]

MAX_GENES = 10
SIZE_PPM_RANGE = (10, 200_000)
LIQ_PCT_RANGE = (1, 20_000)
HALF_WIDTH_RANGE = (1, 40)
LIBRARY_FILE = "strategies.jsonl"
LIBRARY_ENV = "MEV_SIM_STRATEGY_DIR"

_ANY = r"[^\]]+"
_TAIL = rf"(?: SETTLE\[{_ANY}\]\+?)*(?: BURN\[{_ANY}\])?(?: SETTLE\[{_ANY}\]\+?)*"
KNOWN_TEMPLATES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = tuple(
    (name, re.compile(rx)) for name, rx in (
        ("sandwich", rf"^SWAP\[V1=\]\+? VICTIM\[V1\] (?:SWAP|SETTLE)\[V1~\]\+?{_TAIL}$"),
        ("multi_victim_sandwich", rf"^SWAP\[V1=\]\+? VICTIM\[V1\]\+ (?:SWAP|SETTLE)\[V1~\]\+?{_TAIL}$"),
        ("backrun_arbitrage", rf"^VICTIM\[V1\]\+?(?: (?:SWAP|SETTLE)\[{_ANY}\]\+?)+$"),
        ("cyclic_arbitrage", rf"^(?:SWAP|SETTLE)\[{_ANY}\]\+?(?: (?:SWAP|SETTLE)\[{_ANY}\]\+?)*$"),
        ("jit_liquidity", rf"^MINT\[V1\] VICTIM\[V1\]\+? BURN\[V1\]{_TAIL}$"),
        ("liquidation_backrun", rf"^VICTIM\[V1\]\+? LIQ\[M1\]\+?{_TAIL}$"),
        ("liquidation", rf"^LIQ\[M1\]\+?{_TAIL}$"),
    )
)


def default_library_dir() -> Path:
    """``$MEV_SIM_STRATEGY_DIR``, else ``~/.mev_sim/strategies`` on this machine."""
    env = os.environ.get(LIBRARY_ENV)
    return Path(env).expanduser() if env else Path.home() / ".mev_sim" / "strategies"


# ---------------------------------------------------------------------------
# Genome execution
# ---------------------------------------------------------------------------


def _victims_by_role(scenario: Scenario) -> Dict[Tuple[int, int], List[PendingTx]]:
    index = {p.pool_id: i for i, p in enumerate(scenario.pools)}
    out: Dict[Tuple[int, int], List[PendingTx]] = {}
    for tx in scenario.victims:
        if len(tx.route) != 1 or tx.route[0] not in index:
            continue
        pi = index[tx.route[0]]
        side = scenario.pools[pi].tokens().index(tx.token_in)
        out.setdefault((pi, side), []).append(tx)
    for lst in out.values():
        lst.sort(key=lambda t: (-t.amount_in, t.tx_hash))
    return out


def _apply_gene(state: AttackState, scenario: Scenario, roles: Dict[Tuple[int, int], List[PendingTx]],
                gene: Gene) -> None:
    op = gene[0]
    pools = scenario.pools
    if op == "swap":
        _, pi, side, ppm = gene
        pool = state.pools[pools[pi].pool_id]
        token_in = pool.tokens()[side]
        zfo, _ = _direction(pool, token_in)
        amount = input_reserve_scale(pool, zfo) * ppm // 1_000_000
        state.swap(pool.pool_id, token_in, amount, label="gene")
    elif op == "close":
        _, pi, side = gene
        pool = state.pools[pools[pi].pool_id]
        token = pool.tokens()[side]
        amount = state.ledger.get(token, 0)
        if amount <= 0:
            raise ActionReverted("nothing to close")
        state.swap(pool.pool_id, token, amount, label="close")
    elif op == "victim":
        _, pi, side, rank = gene
        lst = roles.get((pi, side), [])
        if rank >= len(lst):
            raise ActionReverted("no such victim")
        state.victim(lst[rank])
    elif op == "mint":
        _, pi, hw, pct = gene
        pool = state.pools[pools[pi].pool_id]
        if not isinstance(pool, ConcentratedPool) or pool.liquidity <= 0:
            raise ActionReverted("mint needs an active concentrated pool")
        s = pool.tick_spacing
        centre = pool.tick // s * s
        state.mint(pool.pool_id, centre - hw * s, centre + (hw + 1) * s, pool.liquidity * pct // 100)
    elif op == "burn":
        state.burn()
    elif op == "liquidate":
        _, mi, ppm = gene
        if mi >= len(scenario.markets):
            raise ActionReverted("no such market")
        market = state.markets[scenario.markets[mi].market_id]
        price = market.price(state.pools)
        candidates = sorted((market.health(loan, price), b) for b, loan in market.loans.items()
                            if b != state.ACTOR and loan.debt > 0 and market.health(loan, price) < 1)
        if not candidates:
            raise ActionReverted("no liquidatable loan")
        borrower = candidates[0][1]
        repay = market.max_repay(market.loans[borrower]) * ppm // 1_000_000
        state.liquidate(market.market_id, borrower, repay)
    else:
        raise ActionReverted(f"unknown gene {op!r}")


@dataclass
class Evaluation:
    genome: Genome
    valid: bool
    net_profit_base: int = 0
    signature: str = ""
    result: Optional[AttackResult] = None
    error: str = ""

    @property
    def profitable(self) -> bool:
        return self.valid and self.result is not None and self.result.flat and self.net_profit_base > 0


def evaluate_genome(scenario: Scenario, genome: Sequence[Gene]) -> Evaluation:
    genome = tuple(tuple(g) for g in genome)
    if not genome or len(genome) > MAX_GENES:
        return Evaluation(genome, False, error="empty or too long")
    state = AttackState(scenario)
    roles = _victims_by_role(scenario)
    try:
        for gene in genome:
            _apply_gene(state, scenario, roles, gene)
        state.settle()
    except (ActionReverted, MASEError, ValueError, ZeroDivisionError) as exc:
        return Evaluation(genome, False, error=str(exc))
    if state.capital_used > scenario.capital_base:
        return Evaluation(genome, False, error="exceeds capital")
    res = state.result("discovered", "")
    return Evaluation(genome, True, res.net_profit_base, signature_of(scenario, res), res)


# ---------------------------------------------------------------------------
# Structure signatures and labels
# ---------------------------------------------------------------------------


def signature_of(scenario: Scenario, result: AttackResult) -> str:
    """Canonical structure of an executed strategy, independent of sizes.

    Pools touched by a victim are ``V1, V2, ...`` (order of first victim),
    other lending-oracle pools ``O1, ...``, other pools ``P1, ...``, markets
    ``M1, ...``. A swap on a victim pool is
    ``=`` when it sells what that pool's first victim sells, else ``~``; on
    another pool ``b`` when it sells the base token, else ``x``. Settlement
    swaps are ``SETTLE``. Consecutive repeats collapse to one token with ``+``.
    """
    steps = result.steps
    victim_dir: Dict[str, str] = {}
    roles: Dict[str, str] = {}
    for s in steps:
        if s["op"] == "victim" and s["pool_id"] not in roles:
            roles[s["pool_id"]] = f"V{len(roles) + 1}"
            victim_dir[s["pool_id"]] = s["token_in"]
    oracle_pools = {m.oracle_pool_id for m in scenario.markets}
    others = oracles = 0
    markets: Dict[str, str] = {}
    tokens: List[str] = []
    for s in steps:
        op = s["op"]
        if "pool_id" in s and s["pool_id"] not in roles:
            if s["pool_id"] in oracle_pools:
                oracles += 1
                roles[s["pool_id"]] = f"O{oracles}"
            else:
                others += 1
                roles[s["pool_id"]] = f"P{others}"
        if op == "swap":
            role = roles[s["pool_id"]]
            if role.startswith("V"):
                d = "=" if s["token_in"] == victim_dir[s["pool_id"]] else "~"
            else:
                d = "b" if s["token_in"] == scenario.base else "x"
            tokens.append(f"{'SETTLE' if s.get('settle') else 'SWAP'}[{role}{d}]")
        elif op == "victim":
            tokens.append(f"VICTIM[{roles[s['pool_id']]}]")
        elif op in ("mint", "burn"):
            tokens.append(f"{op.upper()}[{roles[s['pool_id']]}]")
        elif op == "liquidate":
            markets.setdefault(s["market_id"], f"M{len(markets) + 1}")
            tokens.append(f"LIQ[{markets[s['market_id']]}]")
        else:
            tokens.append(op.upper())
    collapsed: List[str] = []
    for t in tokens:
        if collapsed and collapsed[-1].rstrip("+") == t:
            if not collapsed[-1].endswith("+"):
                collapsed[-1] += "+"
        else:
            collapsed.append(t)
    return " ".join(collapsed)


def classify(signature: str) -> Optional[str]:
    """Name of the predefined template the signature matches, or None if it is new."""
    for name, rx in KNOWN_TEMPLATES:
        if rx.match(signature):
            return name
    return None


def label_for(signature: str, core_only: bool = False) -> str:
    """Human-readable label assembled from the signature's components.

    ``core_only`` drops the victim count; that string is the strategy's
    *family*, used to keep one record per kind of composition."""
    toks = signature.split()
    liq_at = [i for i, t in enumerate(toks) if t.startswith("LIQ[")]
    parts: List[str] = []
    victims_seen: set = set()
    victim_count = 0
    for i, t in enumerate(toks):
        m = re.match(r"(\w+)\[(\w+?)([=~bx]?)\](\+?)", t)
        if not m:
            continue
        op, role, d, rep = m.groups()
        if op == "VICTIM":
            victims_seen.add(role)
            victim_count += 2 if rep else 1
            continue
        if op in ("MINT", "BURN"):
            phrase = f"JIT liquidity on {role}"
        elif op == "LIQ":
            phrase = "liquidation" + (" back-run" if victims_seen else "")
        elif role.startswith("V"):
            if role in victims_seen:
                phrase = f"back-run on {role}" if d == "~" else f"push-through on {role}"
            else:
                phrase = f"front-run on {role}" if d == "=" else f"counter-trade on {role}"
        else:
            if op != "SWAP":
                phrase = ""
            elif role.startswith("O") and liq_at and i < liq_at[-1]:
                phrase = f"oracle push via {role}"
            else:
                phrase = f"hop via {role}" if victims_seen else f"pre-position via {role}"
        if phrase and phrase not in parts:
            parts.append(phrase)
    pools = len(victims_seen)
    if core_only:
        return " + ".join(parts) or "no-op"
    tail = f" [{victim_count}{'+' if '+' in signature else ''} victim tx on {pools} pool(s)]" if victim_count else ""
    return (" + ".join(parts) or "no-op") + tail


# ---------------------------------------------------------------------------
# Search operators
# ---------------------------------------------------------------------------


def _log_uniform(rng: random.Random, lo: int, hi: int) -> int:
    return max(lo, min(hi, round(math.exp(rng.uniform(math.log(lo), math.log(hi))))))


def random_gene(rng: random.Random, scenario: Scenario) -> Gene:
    n_pools = len(scenario.pools)
    roll = rng.random()
    if roll < 0.32:
        return ("swap", rng.randrange(n_pools), rng.randrange(2), _log_uniform(rng, *SIZE_PPM_RANGE))
    if roll < 0.50:
        return ("victim", rng.randrange(n_pools), rng.randrange(2), rng.choice((0, 0, 0, 1, 1, 2)))
    if roll < 0.64:
        return ("close", rng.randrange(n_pools), rng.randrange(2))
    if roll < 0.76:
        cl = [i for i, p in enumerate(scenario.pools) if isinstance(p, ConcentratedPool)] or [0]
        return ("mint", rng.choice(cl), _log_uniform(rng, *HALF_WIDTH_RANGE), _log_uniform(rng, *LIQ_PCT_RANGE))
    if roll < 0.88:
        return ("burn",)
    if scenario.markets:
        return ("liquidate", rng.randrange(len(scenario.markets)), rng.randrange(50_000, 1_000_001))
    return ("burn",)


def random_genome(rng: random.Random, scenario: Scenario) -> Genome:
    return tuple(random_gene(rng, scenario) for _ in range(rng.randint(2, 6)))


def _tweak(rng: random.Random, scenario: Scenario, gene: Gene) -> Gene:
    op = gene[0]
    n_pools = len(scenario.pools)
    g = list(gene)
    if op == "swap":
        k = rng.randrange(3)
        if k == 0:
            g[3] = max(SIZE_PPM_RANGE[0], min(SIZE_PPM_RANGE[1], round(g[3] * math.exp(rng.gauss(0, 0.7)))))
        elif k == 1:
            g[2] = 1 - g[2]
        else:
            g[1] = rng.randrange(n_pools)
    elif op == "victim":
        k = rng.randrange(3)
        if k == 0:
            g[3] = max(0, g[3] + rng.choice((-1, 1)))
        elif k == 1:
            g[2] = 1 - g[2]
        else:
            g[1] = rng.randrange(n_pools)
    elif op == "close":
        if rng.random() < 0.5:
            g[1] = rng.randrange(n_pools)
        else:
            g[2] = 1 - g[2]
    elif op == "mint":
        if rng.random() < 0.5:
            g[2] = max(HALF_WIDTH_RANGE[0], min(HALF_WIDTH_RANGE[1], g[2] + rng.choice((-2, -1, 1, 2))))
        else:
            g[3] = max(LIQ_PCT_RANGE[0], min(LIQ_PCT_RANGE[1], round(g[3] * math.exp(rng.gauss(0, 0.7)))))
    elif op == "liquidate":
        g[2] = max(1_000, min(1_000_000, round(g[2] * math.exp(rng.gauss(0, 0.4)))))
    else:
        return random_gene(rng, scenario)
    return tuple(g)


def mutate(rng: random.Random, scenario: Scenario, genome: Genome) -> Genome:
    g = list(genome)
    for _ in range(rng.randint(1, 3)):
        k = rng.random()
        if k < 0.40 and g:
            i = rng.randrange(len(g))
            g[i] = _tweak(rng, scenario, g[i])
        elif k < 0.60 and len(g) < MAX_GENES:
            g.insert(rng.randrange(len(g) + 1), random_gene(rng, scenario))
        elif k < 0.75 and len(g) > 1:
            del g[rng.randrange(len(g))]
        elif k < 0.90 and len(g) > 1:
            i = rng.randrange(len(g) - 1)
            g[i], g[i + 1] = g[i + 1], g[i]
        elif g and len(g) < MAX_GENES:
            i = rng.randrange(len(g))
            g.insert(i, g[i])
    return tuple(g[:MAX_GENES])


def crossover(rng: random.Random, a: Genome, b: Genome) -> Genome:
    i = rng.randint(0, len(a))
    j = rng.randint(0, len(b))
    child = a[:i] + b[j:]
    return child[:MAX_GENES] if child else a


def minimize(scenario: Scenario, ev: Evaluation) -> Evaluation:
    """Drop genes one at a time while the profit does not fall (delta debugging)."""
    best = ev
    changed = True
    while changed and len(best.genome) > 1:
        changed = False
        for i in range(len(best.genome)):
            cand = evaluate_genome(scenario, best.genome[:i] + best.genome[i + 1:])
            if cand.profitable and cand.net_profit_base >= best.net_profit_base:
                best = cand
                changed = True
                break
    return best


# ---------------------------------------------------------------------------
# Template seeds (the predefined attacks, expressed as genomes)
# ---------------------------------------------------------------------------


def template_genomes(scenario: Scenario) -> List[Genome]:
    out: List[Genome] = []
    roles = _victims_by_role(scenario)
    for (pi, side), _lst in sorted(roles.items()):
        pool = scenario.pools[pi]
        out.append((("swap", pi, side, 5_000), ("victim", pi, side, 0), ("close", pi, 1 - side)))
        out.append((("victim", pi, side, 0), ("swap", pi, 1 - side, 500)))
        if isinstance(pool, ConcentratedPool):
            out.append((("mint", pi, 3, 300), ("victim", pi, side, 0), ("burn",)))
        if scenario.markets:
            out.append((("victim", pi, side, 0), ("liquidate", 0, 500_000)))
    return out


# ---------------------------------------------------------------------------
# Parallel evaluation (worker processes rebuild the scenario from its seed)
# ---------------------------------------------------------------------------

_SCENARIO_CACHE: Dict[Tuple[int, int], Scenario] = {}


def _scenario(seed: int, victims: int) -> Scenario:
    key = (seed, victims)
    if key not in _SCENARIO_CACHE:
        if len(_SCENARIO_CACHE) > 32:
            _SCENARIO_CACHE.clear()
        _SCENARIO_CACHE[key] = build_scenario(seed, victims)
    return _SCENARIO_CACHE[key]


def _eval_task(task: Tuple[int, int, Genome]) -> Tuple[Genome, bool, int, str, bool]:
    seed, victims, genome = task
    ev = evaluate_genome(_scenario(seed, victims), genome)
    flat = bool(ev.result is not None and ev.result.flat)
    return ev.genome, ev.valid, ev.net_profit_base, ev.signature, flat


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class Elite:
    genome: Genome
    net_profit_base: int
    signature: str
    seed: int


@dataclass
class DiscoveryReport:
    seed: int
    generations: int
    evaluations: int
    valid: int
    archive: Dict[str, Elite]
    novel: List[Dict[str, Any]] = field(default_factory=list)
    saved: List[str] = field(default_factory=list)
    library: str = ""

    def to_dict(self) -> Dict[str, Any]:
        known: Dict[str, int] = {}
        for sig, e in self.archive.items():
            name = classify(sig)
            if name is not None:
                known[name] = max(known.get(name, 0), e.net_profit_base)
        return {
            "seed": self.seed,
            "generations": self.generations,
            "evaluations": self.evaluations,
            "valid": self.valid,
            "structures_found": len(self.archive),
            "best_known_by_template": known,
            "novel": self.novel,
            "saved": self.saved,
            "library": self.library,
        }


class DiscoveryEngine:
    """MAP-Elites search over action genomes on seeded scenarios.

    ``scenario_seeds`` are the training scenarios (a candidate's score is its
    profit on the scenario it was drawn for); ``holdout`` further seeds are
    used only to report how often a found structure also pays elsewhere.
    """

    def __init__(self, seed: int = 0, generations: int = 30, population: int = 64, victims: int = 10,
                 scenario_seeds: int = 3, holdout: int = 4, min_profit_base: int = 1) -> None:
        if generations < 0 or population <= 0 or scenario_seeds <= 0 or holdout < 0:
            raise ValueError("generations >= 0, population > 0, scenario_seeds > 0, holdout >= 0")
        self.seed = seed
        self.generations = generations
        self.population = population
        self.victims = victims
        self.scenario_seeds = [seed * 1_000 + i for i in range(scenario_seeds)]
        self.holdout_seeds = [seed * 1_000 + 500 + i for i in range(holdout)]
        self.min_profit_base = min_profit_base
        self.rng = random.Random(f"discover:{seed}")

    def _evaluate(self, tasks: List[Tuple[int, int, Genome]], executor: Any) -> List[Tuple[Genome, bool, int, str, bool]]:
        if executor is not None:
            return executor.map(_eval_task, tasks, chunksize=max(1, len(tasks) // 32))
        return [_eval_task(t) for t in tasks]

    def run(self, executor: Any = None, library: Optional["StrategyLibrary"] = None) -> DiscoveryReport:
        archive: Dict[str, Elite] = {}
        evaluations = valid = 0

        def consider(seed: int, outcome: Tuple[Genome, bool, int, str, bool]) -> None:
            nonlocal valid
            genome, ok, profit, sig, flat = outcome
            if not ok:
                return
            valid += 1
            if not flat or profit < self.min_profit_base:
                return
            cur = archive.get(sig)
            if cur is None or profit > cur.net_profit_base:
                archive[sig] = Elite(genome, profit, sig, seed)

        # generation 0: the predefined attacks as genomes plus random genomes
        tasks: List[Tuple[int, int, Genome]] = []
        for s in self.scenario_seeds:
            scn = _scenario(s, self.victims)
            for g in template_genomes(scn):
                tasks.append((s, self.victims, g))
            for _ in range(self.population):
                tasks.append((s, self.victims, random_genome(self.rng, scn)))
        for t, out in zip(tasks, self._evaluate(tasks, executor)):
            evaluations += 1
            consider(t[0], out)

        for _ in range(self.generations):
            tasks = []
            elites = sorted(archive.values(), key=lambda e: e.signature)
            for _ in range(self.population):
                s = self.rng.choice(self.scenario_seeds)
                scn = _scenario(s, self.victims)
                if not elites or self.rng.random() < 0.1:
                    child = random_genome(self.rng, scn)
                else:
                    parent = self.rng.choice(elites).genome
                    if self.rng.random() < 0.3:
                        child = crossover(self.rng, parent, self.rng.choice(elites).genome)
                    else:
                        child = parent
                    child = mutate(self.rng, scn, child)
                tasks.append((s, self.victims, child))
            for t, out in zip(tasks, self._evaluate(tasks, executor)):
                evaluations += 1
                consider(t[0], out)

        report = DiscoveryReport(self.seed, self.generations, evaluations, valid, archive)
        seen: set = set()
        families: Dict[str, Tuple[Evaluation, int]] = {}
        for sig in sorted(archive):
            if classify(sig) is not None:
                continue
            elite = archive[sig]
            scn = _scenario(elite.seed, self.victims)
            ev = minimize(scn, evaluate_genome(scn, elite.genome))
            if not ev.profitable or classify(ev.signature) is not None or ev.signature in seen:
                continue
            seen.add(ev.signature)
            family = label_for(ev.signature, core_only=True)
            cur = families.get(family)
            if cur is None or ev.net_profit_base > cur[0].net_profit_base:
                families[family] = (ev, elite.seed)
        for family in sorted(families):
            ev, seed = families[family]
            record = self._record(ev, seed)
            report.novel.append(record)
            if library is not None and library.add(record):
                report.saved.append(record["id"])
        report.novel.sort(key=lambda r: -r["net_profit_base"])
        if library is not None:
            report.library = str(library.path)
        return report

    def _record(self, ev: Evaluation, seed: int) -> Dict[str, Any]:
        assert ev.result is not None
        hold = []
        for s in self.holdout_seeds:
            h = evaluate_genome(_scenario(s, self.victims), ev.genome)
            hold.append({"seed": s, "profitable": h.profitable, "net_profit_base": h.net_profit_base,
                         "same_structure": h.signature == ev.signature})
        res = ev.result
        return {
            "id": "novel-" + hashlib.sha256(label_for(ev.signature, core_only=True).encode()).hexdigest()[:12],
            "family": label_for(ev.signature, core_only=True),
            "label": label_for(ev.signature),
            "signature": ev.signature,
            "genome": [list(g) for g in ev.genome],
            "scenario": {"seed": seed, "victims": self.victims},
            "net_profit_base": ev.net_profit_base,
            "base_delta": res.base_delta,
            "gas_used": res.gas_used,
            "gas_cost_base": res.gas_cost_base,
            "capital_used_base": res.capital_used_base,
            "victim_tx": list(res.victims),
            "status": "model" if any(s["op"] == "liquidate" for s in res.steps) else "exact",
            "steps": res.steps,
            "holdout": hold,
            "holdout_profitable": sum(1 for h in hold if h["profitable"]),
            "discovery_seed": self.seed,
            "simulator_version": __version__,
        }


# ---------------------------------------------------------------------------
# Local strategy library
# ---------------------------------------------------------------------------


class StrategyLibrary:
    """Discovered strategies on this machine, one JSON object per line.

    The best record per family (see :func:`label_for`) wins; appends are line-atomic (one
    ``write`` of a complete line). ``replay`` rebuilds the scenario from the
    stored seed and checks the stored profit to the wei.
    """

    def __init__(self, directory: Optional[os.PathLike] = None) -> None:
        self.dir = Path(directory) if directory is not None else default_library_dir()
        self.path = self.dir / LIBRARY_FILE

    def records(self) -> Iterator[Dict[str, Any]]:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def best(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for r in self.records():
            key = r.get("family") or r["signature"]
            cur = out.get(key)
            if cur is None or r["net_profit_base"] > cur["net_profit_base"]:
                out[key] = r
        return out

    def add(self, record: Dict[str, Any]) -> bool:
        """Append unless an equal-or-better record of the same family exists."""
        cur = self.best().get(record.get("family") or record["signature"])
        if cur is not None and cur["net_profit_base"] >= record["net_profit_base"]:
            return False
        self.dir.mkdir(parents=True, exist_ok=True)
        stamped = dict(record, saved_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
        line = json.dumps(stamped, sort_keys=True, default=str) + "\n"
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line)
        return True

    def replay(self, record: Dict[str, Any]) -> Dict[str, Any]:
        scn = build_scenario(record["scenario"]["seed"], record["scenario"]["victims"])
        ev = evaluate_genome(scn, [tuple(g) for g in record["genome"]])
        return {
            "id": record["id"],
            "ok": ev.profitable and ev.net_profit_base == record["net_profit_base"]
                  and ev.signature == record["signature"],
            "net_profit_base": ev.net_profit_base,
            "expected": record["net_profit_base"],
            "signature": ev.signature,
        }

    def verify(self) -> Dict[str, Any]:
        results = [self.replay(r) for r in self.best().values()]
        return {"path": str(self.path), "records": len(results), "ok": all(r["ok"] for r in results),
                "results": results}

    def summary(self) -> List[Dict[str, Any]]:
        return [{"id": r["id"], "family": r.get("family"), "label": r["label"], "signature": r["signature"],
                 "net_profit_base": r["net_profit_base"], "holdout_profitable": r.get("holdout_profitable"),
                 "status": r.get("status")}
                for r in sorted(self.best().values(), key=lambda r: -r["net_profit_base"])]
