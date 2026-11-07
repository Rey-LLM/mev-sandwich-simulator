"""Layer 14: custom bot architecture builder and runtime blueprint compiler.

Status: experimental. A *blueprint* is a JSON document that describes a
complete competitor bot: when it fires (a nested trigger tree over victim
metrics, modelled opcode traces and synthetic on-chain events), what it does
(a nested action script) and how its execution pipeline behaves (revert
policy, settlement, gas and capital caps, optimiser budget). The compiler
validates a blueprint against :data:`BLUEPRINT_SCHEMA`, assembles it into a
small bytecode for a stack / jump virtual machine and spins up a dedicated
Python class per blueprint at runtime (``type()``), cached by the
blueprint's digest. Every action leg runs on :class:`~simulator.attack_vectors.AttackState`,
so token amounts use the same exact integer AMM math as the rest of the
simulator. Nothing here reads a chain or sends a transaction.

Blueprint layout (see :func:`normalize_blueprint` for every field)
------------------------------------------------------------------

``trigger``    nested node tree, evaluated per pending single-hop victim:

               ``{"all": [...]}`` / ``{"any": [...]}`` / ``{"not": node}``
               ``{"metric": name, "op": cmp, "value": v}``   (:data:`METRICS`)
               ``{"opcode": name, "op": cmp, "value": n}``  (count in the modelled trace)
               ``{"opcode_seq": [names]}``                  (ordered match; contiguous when
                                                             ``tracing.sequential``)
               ``{"event": name, ...params}``               (:data:`EVENTS` listeners)
               ``{"const": true}``
               Inside scripts, conditions may also test ``{"tload": key, ...}``
               and ``{"balance": token, ...}``.

``tracing``    ``opcodes`` (enable opcode nodes), ``sequential`` (contiguous
               ``opcode_seq`` matching), ``record`` (attach the modelled opcode
               trace of every executed step to the result).

``script``     nested actions (:data:`ACTIONS`): ``frontrun``, ``victim``,
               ``victims`` (more same-direction victims), ``backrun``, ``swap``,
               ``close``, ``mint``, ``burn``, ``liquidate``, ``arb`` (best 2/3-pool
               cycle through a pool), ``tstore`` / ``tswap`` (EIP-1153-style
               transient storage that lives for one bundle and is rolled back
               with it), and the control nodes ``if``, ``repeat`` (unrolled at
               compile time), ``atomic`` (snapshot; ``rollback`` or ``abort`` on
               revert) and ``bridge`` (a delayed cross-chain leg, below).

``pipeline``   execution policy; every key can be overridden per run
               (:meth:`CustomBot.run` ``overrides``) without recompiling.

Sizes are value specs: an integer or ``{"ppm": n}`` (ppm of the input-side
reserve scale), ``{"victim_ppm": n}``, ``{"tload": key}`` (raw wei from
transient storage), ``{"balance": token, "ppm": n}`` or
``{"optimize": [lo, hi]}`` (ppm searched per victim by
:func:`~simulator.attack_vectors.maximize_int`; one per blueprint).

What is exact and what is a model
---------------------------------
* Swap, victim, mint / burn and cycle legs: **exact** (bit-exact AMM math).
* Opcode traces: a **model** (a fixed opcode pattern per pool version plus
  two opcodes per crossed tick), not an EVM trace.
* ``bridge``: a **model**. The remote chain is another seeded scenario; the
  delay is the first ``delay_blocks`` remote pending swaps executing before
  the remote leg; fees are ``fee_bps`` each way and remote gas costs
  ``1 / remote_gas_divisor`` of local gas. Results using it carry
  ``status = "model"``.
* ``liquidate`` uses the lending **model** of ``attack_vectors``.

Compiler guarantees
-------------------
Every jump in the assembled program goes forward, ``repeat`` is unrolled and
the program is bounded by :data:`MAX_INSTRUCTIONS`, so every compiled bot
terminates. Compiled code is made of tuples and :class:`FrozenMap` values,
so a class shared between threads holds no mutable state; every run works on
its own :class:`~simulator.attack_vectors.AttackState` copy and its own
transient storage, so concurrent runs cannot see each other.

AI synthesis (:class:`BlueprintSynthesizer`)
--------------------------------------------
:func:`blueprint_from_genes` maps :data:`N_BLUEPRINT_GENES` genes in ``[0, 1]``
onto a blueprint, including the trigger tree's *structure*. A NumPy network
from :mod:`simulator.ai_agent_factory` maps ``[RL features, latent]`` to
genes; each generation the blueprints are compiled and scored on seeded
scenarios in fixed batches on the worker pool, the network is retrained on
the elite and every trigger structure (:func:`trigger_class`) that is not in
:data:`KNOWN_TRIGGER_CLASSES` and was not seen before is reported as a newly
synthesised trigger class. NumPy is imported only when the synthesizer runs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type

from .attack_vectors import (
    ActionReverted,
    AttackResult,
    AttackState,
    Scenario,
    _cycles_through,
    _run_cycle,
    build_scenario,
    maximize_int,
)
from .config import ConfigError
from .invariants import input_reserve_scale
from .mempool_emulator import PendingTx, pool_version
from .slippage_engine import ConcentratedPool, MASEError, V2Pool, _direction

__all__ = [
    "ACTIONS",
    "BLUEPRINT_SCHEMA",
    "COMPARATORS",
    "EVENTS",
    "KNOWN_TRIGGER_CLASSES",
    "METRICS",
    "N_BLUEPRINT_GENES",
    "OPCODES",
    "SCHEMA_VERSION",
    "TEMPLATES",
    "BlueprintError",
    "BlueprintSynthesizer",
    "BotRun",
    "CompiledProgram",
    "CustomBot",
    "FrozenMap",
    "PlanOutcome",
    "SynthesisReport",
    "VictimContext",
    "assemble",
    "blueprint_digest",
    "blueprint_from_genes",
    "cache_info",
    "clear_class_cache",
    "compile_blueprint",
    "disassemble",
    "evaluate_blueprint",
    "normalize_blueprint",
    "opcode_trace",
    "run_blueprint",
    "scenario_for",
    "spawn_custom_bot",
    "template",
    "trigger_class",
    "validate_blueprint",
]

SCHEMA_VERSION = 1
MAX_DEPTH = 6
MAX_TRIGGER_NODES = 64
MAX_CHILDREN = 16
MAX_SCRIPT_ACTIONS = 64
MAX_INSTRUCTIONS = 1_024
MAX_REPEAT = 8
MAX_OPSEQ = 8
MAX_KEY_LEN = 32
MAX_CYCLES = 8
CLASS_CACHE_LIMIT = 4_096
SCENARIO_CACHE_LIMIT = 64
PPM = 1_000_000
BPS = 10_000
GAS_BRIDGE = 160_000  # per direction (estimate, like GAS_PER_HOP)
GAS_TRANSIENT = 100  # TSTORE / TLOAD warm cost
NAME_RX = re.compile(r"^[A-Za-z0-9_.:\-]{1,64}$")
KEY_RX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,31}$")

COMPARATORS: Tuple[str, ...] = ("<", "<=", ">", ">=", "==", "!=", "between", "in")

METRICS: Dict[str, str] = {
    "victim_priority_fee_wei": "victim priority fee (wei)",
    "victim_gas_limit": "victim gas limit",
    "victim_amount_in": "victim input amount (raw units)",
    "victim_value_base": "victim input value in base-token wei at reference prices",
    "victim_size_ppm": "victim input as ppm of the pool's input-side reserve scale",
    "victim_slippage_bps": "victim tolerance: (quoted out - amount_out_min) / quoted out, bps",
    "victim_impact_bps": "marginal price move caused by the victim swap alone, bps",
    "victim_rank": "size rank among victims on the same pool and side (0 = largest)",
    "same_side_victims": "victims on the same pool and side",
    "block_position": "index of the victim in block order",
    "ticks_crossed": "initialized ticks the victim swap crosses (0 on v2)",
    "pool_index": "index of the victim pool in the scenario's sorted pool list",
    "pool_version": "2, 3 or 4",
    "pool_side": "0 when the victim sells token0, else 1",
    "pool_fee_pips": "pool LP fee in pips (v2 fee bps x 100)",
    "gas_price_wei": "scenario gas price (wei)",
    "cross_venue_spread_bps": "spot spread between pools of the same pair, bps",
    "loan_min_health_bps": "lowest loan health after the victim, bps (oracle on this pool)",
}

OPCODES: Tuple[str, ...] = ("CALL", "SLOAD", "SSTORE", "TLOAD", "TSTORE", "LOG1", "LOG3")

# listener name -> {param: (default, lo, hi)}
EVENTS: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "Swap": {"pool": (-1, -1, 63), "side": (-1, -1, 1)},
    "LargeSwap": {"min_ppm": (1_000, 1, PPM)},
    "PriceDeviation": {"min_bps": (10, 0, BPS)},
    "LoanAtRisk": {"max_health_bps": (10_100, 0, 10**9)},
    "TickCross": {"min_ticks": (1, 0, 10_000)},
}

ACTIONS: Tuple[str, ...] = ("frontrun", "victim", "victims", "backrun", "swap", "close", "mint", "burn",
                            "liquidate", "arb", "tstore", "tswap", "if", "repeat", "atomic", "bridge")
REMOTE_ACTIONS = frozenset({"swap", "close", "arb", "tstore", "tswap", "if", "repeat", "atomic"})
VICTIM_ACTIONS = frozenset({"frontrun", "victim", "victims", "backrun"})

PIPELINE_DEFAULTS: Dict[str, Any] = {
    "on_revert": "abort",
    "settle": True,
    "require_flat": True,
    "max_gas": 2_500_000,
    "min_profit_base": 1,
    "max_attempts": 6,
    "optimize_grid": 8,
    "optimize_refine": 8,
    "remote_gas_divisor": 10,
}
PIPELINE_RANGES: Dict[str, Tuple[int, int]] = {
    "max_gas": (21_000, 30_000_000),
    "min_profit_base": (0, 10**30),
    "max_attempts": (1, 64),
    "optimize_grid": (2, 64),
    "optimize_refine": (0, 64),
    "remote_gas_divisor": (1, 1_000),
}

BLUEPRINT_SCHEMA: Dict[str, Any] = {
    "schema": SCHEMA_VERSION,
    "fields": {
        "schema": "int, must equal SCHEMA_VERSION (default)",
        "name": "1-64 chars of [A-Za-z0-9_.:-]",
        "description": "free text, at most 512 chars",
        "trigger": "trigger node (default {'const': true})",
        "tracing": {"opcodes": "bool", "sequential": "bool", "record": "bool"},
        "script": "non-empty list of actions",
        "pipeline": {k: f"{type(v).__name__}, default {v!r}" for k, v in PIPELINE_DEFAULTS.items()},
    },
    "metrics": dict(METRICS),
    "comparators": list(COMPARATORS),
    "opcodes": list(OPCODES),
    "events": {name: {p: list(r) for p, r in params.items()} for name, params in EVENTS.items()},
    "actions": list(ACTIONS),
    "value_specs": ["int (ppm)", "{'ppm': n}", "{'victim_ppm': n}", "{'tload': key}",
                    "{'balance': token, 'ppm': n}", "{'optimize': [lo, hi]}"],
    "token_refs": ["$in", "$out", "$base", "<symbol>"],
    "pool_refs": ["$victim", "<index>"],
    "side_refs": ["$victim", "$counter", 0, 1],
    "limits": {"depth": MAX_DEPTH, "trigger_nodes": MAX_TRIGGER_NODES, "children": MAX_CHILDREN,
               "script_actions": MAX_SCRIPT_ACTIONS, "instructions": MAX_INSTRUCTIONS, "repeat": MAX_REPEAT},
}


class BlueprintError(ConfigError):
    """A blueprint failed validation; ``issues`` lists ``(path, message)`` pairs."""

    def __init__(self, issues: Sequence[Tuple[str, str]]) -> None:
        self.issues: List[Tuple[str, str]] = list(issues)
        shown = "; ".join(f"{p}: {m}" for p, m in self.issues[:8])
        more = f" (+{len(self.issues) - 8} more)" if len(self.issues) > 8 else ""
        super().__init__(f"invalid blueprint: {shown}{more}")


# ---------------------------------------------------------------------------
# Immutable compiled values
# ---------------------------------------------------------------------------


class FrozenMap(tuple):
    """Immutable mapping stored as sorted ``(key, value)`` pairs (hashable, picklable)."""

    __slots__ = ()

    def get(self, key: str, default: Any = None) -> Any:
        for k, v in self:
            if k == key:
                return v
        return default

    def keys(self) -> List[str]:
        return [k for k, _ in self]

    def to_dict(self) -> Dict[str, Any]:
        return {k: _thaw(v) for k, v in self}


def _freeze(obj: Any) -> Any:
    if isinstance(obj, dict):
        return FrozenMap(sorted((str(k), _freeze(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v) for v in obj)
    return obj


def _thaw(obj: Any) -> Any:
    if isinstance(obj, FrozenMap):
        return obj.to_dict()
    if isinstance(obj, tuple):
        return [_thaw(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Schema validation and normalisation
# ---------------------------------------------------------------------------


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


class _Checker:
    def __init__(self) -> None:
        self.issues: List[Tuple[str, str]] = []
        self.trigger_nodes = 0
        self.actions = 0
        self.optimize_paths: List[str] = []
        self.uses_opcodes = False

    def bad(self, path: str, msg: str) -> None:
        self.issues.append((path, msg))

    def int_in(self, path: str, v: Any, lo: int, hi: int) -> Optional[int]:
        if not _is_int(v):
            self.bad(path, f"must be an integer, got {type(v).__name__}")
            return None
        if not lo <= v <= hi:
            self.bad(path, f"must be within [{lo}, {hi}], got {v}")
            return None
        return v

    def keys(self, path: str, node: Mapping[str, Any], allowed: Iterable[str]) -> None:
        extra = sorted(set(node) - set(allowed))
        if extra:
            self.bad(path, f"unknown key(s) {', '.join(map(repr, extra))}")

    # -- comparison values ---------------------------------------------------

    def compare(self, path: str, node: Mapping[str, Any]) -> Optional[Tuple[str, Any]]:
        op = node.get("op", ">=")
        if op not in COMPARATORS:
            self.bad(f"{path}.op", f"unknown comparator {op!r} (one of {', '.join(COMPARATORS)})")
            return None
        value = node.get("value")
        if op == "between":
            if (not isinstance(value, (list, tuple)) or len(value) != 2 or not all(_is_int(x) for x in value)
                    or value[0] > value[1]):
                self.bad(f"{path}.value", "'between' needs [lo, hi] integers with lo <= hi")
                return None
            return op, [value[0], value[1]]
        if op == "in":
            if (not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 32
                    or not all(_is_int(x) for x in value)):
                self.bad(f"{path}.value", "'in' needs a list of 1-32 integers")
                return None
            return op, sorted(set(value))
        if not _is_int(value):
            self.bad(f"{path}.value", "must be an integer")
            return None
        return op, value

    # -- trigger nodes -------------------------------------------------------

    def node(self, path: str, node: Any, depth: int, script: bool, remote: bool) -> Optional[Dict[str, Any]]:
        self.trigger_nodes += 1
        if self.trigger_nodes > MAX_TRIGGER_NODES:
            if self.trigger_nodes == MAX_TRIGGER_NODES + 1:
                self.bad(path, f"more than {MAX_TRIGGER_NODES} trigger nodes")
            return None
        if depth > MAX_DEPTH:
            self.bad(path, f"nested deeper than {MAX_DEPTH}")
            return None
        if not isinstance(node, dict) or not node:
            self.bad(path, "a trigger node must be a non-empty object")
            return None
        if "all" in node or "any" in node:
            kind = "all" if "all" in node else "any"
            self.keys(path, node, (kind,))
            kids = node[kind]
            if not isinstance(kids, list) or not 1 <= len(kids) <= MAX_CHILDREN:
                self.bad(f"{path}.{kind}", f"needs a list of 1-{MAX_CHILDREN} nodes")
                return None
            out = [self.node(f"{path}.{kind}[{i}]", k, depth + 1, script, remote) for i, k in enumerate(kids)]
            return None if any(o is None for o in out) else {kind: out}
        if "not" in node:
            self.keys(path, node, ("not",))
            inner = self.node(f"{path}.not", node["not"], depth + 1, script, remote)
            return None if inner is None else {"not": inner}
        if "const" in node:
            self.keys(path, node, ("const",))
            if not isinstance(node["const"], bool):
                self.bad(f"{path}.const", "must be true or false")
                return None
            return {"const": node["const"]}
        if "metric" in node:
            self.keys(path, node, ("metric", "op", "value"))
            if remote:
                self.bad(path, "metric conditions need a victim; the remote leg of a bridge has none")
                return None
            if node["metric"] not in METRICS:
                self.bad(f"{path}.metric", f"unknown metric {node['metric']!r}")
                return None
            cmp_ = self.compare(path, node)
            return None if cmp_ is None else {"metric": node["metric"], "op": cmp_[0], "value": cmp_[1]}
        if "opcode" in node:
            self.keys(path, node, ("opcode", "op", "value"))
            self.uses_opcodes = True
            if remote:
                self.bad(path, "opcode conditions need a victim trace")
                return None
            if node["opcode"] not in OPCODES:
                self.bad(f"{path}.opcode", f"unknown opcode {node['opcode']!r} (one of {', '.join(OPCODES)})")
                return None
            cmp_ = self.compare(path, node)
            return None if cmp_ is None else {"opcode": node["opcode"], "op": cmp_[0], "value": cmp_[1]}
        if "opcode_seq" in node:
            self.keys(path, node, ("opcode_seq",))
            self.uses_opcodes = True
            if remote:
                self.bad(path, "opcode conditions need a victim trace")
                return None
            seq = node["opcode_seq"]
            if (not isinstance(seq, list) or not 1 <= len(seq) <= MAX_OPSEQ
                    or not all(s in OPCODES for s in seq)):
                self.bad(f"{path}.opcode_seq", f"needs 1-{MAX_OPSEQ} opcodes from {', '.join(OPCODES)}")
                return None
            return {"opcode_seq": list(seq)}
        if "event" in node:
            name = node["event"]
            if remote:
                self.bad(path, "event listeners need a victim")
                return None
            if name not in EVENTS:
                self.bad(f"{path}.event", f"unknown event {name!r} (one of {', '.join(EVENTS)})")
                return None
            params = EVENTS[name]
            self.keys(path, node, ("event", *params))
            out: Dict[str, Any] = {"event": name}
            for p, (default, lo, hi) in params.items():
                v = self.int_in(f"{path}.{p}", node.get(p, default), lo, hi)
                if v is None:
                    return None
                out[p] = v
            return out
        if "tload" in node or "balance" in node:
            kind = "tload" if "tload" in node else "balance"
            self.keys(path, node, (kind, "op", "value"))
            if not script:
                self.bad(path, f"{kind!r} conditions are only allowed inside script 'if' nodes")
                return None
            ref = node[kind]
            if kind == "tload" and not (isinstance(ref, str) and KEY_RX.match(ref)):
                self.bad(f"{path}.tload", "needs a transient key ([A-Za-z_][A-Za-z0-9_]{0,31})")
                return None
            if kind == "balance" and not self.token_ok(ref):
                self.bad(f"{path}.balance", "needs a token reference ($in, $out, $base or a symbol)")
                return None
            if remote and kind == "balance" and ref in ("$in", "$out"):
                self.bad(f"{path}.balance", "$in / $out need a victim")
                return None
            cmp_ = self.compare(path, node)
            return None if cmp_ is None else {kind: ref, "op": cmp_[0], "value": cmp_[1]}
        self.bad(path, f"unknown trigger node with keys {sorted(node)}")
        return None

    # -- references and value specs -----------------------------------------

    @staticmethod
    def token_ok(ref: Any) -> bool:
        return isinstance(ref, str) and (ref in ("$in", "$out", "$base") or bool(re.match(r"^[A-Za-z0-9]{1,16}$", ref)))

    def pool_ref(self, path: str, v: Any, remote: bool) -> Any:
        if v == "$victim" and not remote:
            return v
        if _is_int(v) and 0 <= v <= 63:
            return v
        self.bad(path, "pool must be an index 0-63" + ("" if remote else " or '$victim'"))
        return None

    def side_ref(self, path: str, v: Any, remote: bool) -> Any:
        if v in ("$victim", "$counter") and not isinstance(v, bool) and not remote:
            return v
        if _is_int(v) and v in (0, 1):
            return v
        self.bad(path, "side must be 0 or 1" + ("" if remote else ", '$victim' or '$counter'"))
        return None

    def size(self, path: str, v: Any, remote: bool) -> Any:
        if _is_int(v):
            v = {"ppm": v}
        if not isinstance(v, dict) or len(v) not in (1, 2):
            self.bad(path, "size must be an integer (ppm) or a value spec object")
            return None
        if "ppm" in v and len(v) == 1:
            n = self.int_in(f"{path}.ppm", v["ppm"], 1, PPM)
            return None if n is None else {"ppm": n}
        if "victim_ppm" in v and len(v) == 1:
            if remote:
                self.bad(path, "victim_ppm needs a victim")
                return None
            n = self.int_in(f"{path}.victim_ppm", v["victim_ppm"], 1, 10 * PPM)
            return None if n is None else {"victim_ppm": n}
        if "tload" in v and len(v) == 1:
            if not (isinstance(v["tload"], str) and KEY_RX.match(v["tload"])):
                self.bad(f"{path}.tload", "needs a transient key")
                return None
            return {"tload": v["tload"]}
        if "balance" in v:
            self.keys(path, v, ("balance", "ppm"))
            if not self.token_ok(v["balance"]) or (remote and v["balance"] in ("$in", "$out")):
                self.bad(f"{path}.balance", "needs a token reference")
                return None
            n = self.int_in(f"{path}.ppm", v.get("ppm", PPM), 1, PPM)
            return None if n is None else {"balance": v["balance"], "ppm": n}
        if "optimize" in v and len(v) == 1:
            r = v["optimize"]
            if (not isinstance(r, (list, tuple)) or len(r) != 2 or not all(_is_int(x) for x in r)
                    or not 1 <= r[0] < r[1] <= PPM):
                self.bad(f"{path}.optimize", f"needs [lo, hi] ppm with 1 <= lo < hi <= {PPM}")
                return None
            self.optimize_paths.append(path)
            return {"optimize": [r[0], r[1]]}
        self.bad(path, f"unknown value spec with keys {sorted(v)}")
        return None

    def tvalue(self, path: str, v: Any, remote: bool) -> Any:
        if _is_int(v):
            v = {"const": v}
        if not isinstance(v, dict) or len(v) not in (1, 2):
            self.bad(path, "tstore value must be an integer or one of {last_out}, {balance}, {victim_amount}, {const}")
            return None
        if "const" in v and len(v) == 1:
            n = self.int_in(f"{path}.const", v["const"], 0, 2**256 - 1)
            return None if n is None else {"const": n}
        if v.get("last_out") is True and len(v) == 1:
            return {"last_out": True}
        if v.get("victim_amount") is True and len(v) == 1:
            if remote:
                self.bad(path, "victim_amount needs a victim")
                return None
            return {"victim_amount": True}
        if "balance" in v:
            self.keys(path, v, ("balance", "ppm"))
            if not self.token_ok(v["balance"]) or (remote and v["balance"] in ("$in", "$out")):
                self.bad(f"{path}.balance", "needs a token reference")
                return None
            n = self.int_in(f"{path}.ppm", v.get("ppm", PPM), 1, PPM)
            return None if n is None else {"balance": v["balance"], "ppm": n}
        self.bad(path, f"unknown tstore value with keys {sorted(v)}")
        return None

    # -- actions -------------------------------------------------------------

    def script(self, path: str, actions: Any, depth: int, remote: bool) -> Optional[List[Dict[str, Any]]]:
        if depth > MAX_DEPTH:
            self.bad(path, f"nested deeper than {MAX_DEPTH}")
            return None
        if not isinstance(actions, list):
            self.bad(path, "must be a list of actions")
            return None
        out = []
        for i, a in enumerate(actions):
            out.append(self.action(f"{path}[{i}]", a, depth, remote))
        return None if any(o is None for o in out) else out

    def action(self, path: str, a: Any, depth: int, remote: bool) -> Optional[Dict[str, Any]]:
        self.actions += 1
        if self.actions > MAX_SCRIPT_ACTIONS:
            if self.actions == MAX_SCRIPT_ACTIONS + 1:
                self.bad(path, f"more than {MAX_SCRIPT_ACTIONS} actions")
            return None
        if not isinstance(a, dict) or "op" not in a:
            self.bad(path, "an action must be an object with an 'op'")
            return None
        op = a["op"]
        if op not in ACTIONS:
            self.bad(f"{path}.op", f"unknown action {op!r}")
            return None
        if remote and op not in REMOTE_ACTIONS:
            self.bad(f"{path}.op", f"{op!r} is not available on the remote leg of a bridge")
            return None
        p = path
        if op in ("victim", "backrun", "burn"):
            self.keys(p, a, ("op",))
            return {"op": op}
        if op == "frontrun":
            self.keys(p, a, ("op", "size"))
            s = self.size(f"{p}.size", a.get("size", 5_000), remote)
            return None if s is None else {"op": op, "size": s}
        if op == "victims":
            self.keys(p, a, ("op", "max"))
            n = self.int_in(f"{p}.max", a.get("max", 2), 1, 8)
            return None if n is None else {"op": op, "max": n}
        if op == "swap":
            self.keys(p, a, ("op", "pool", "side", "size"))
            pool = self.pool_ref(f"{p}.pool", a.get("pool", 0 if remote else "$victim"), remote)
            side = self.side_ref(f"{p}.side", a.get("side", 0 if remote else "$victim"), remote)
            s = self.size(f"{p}.size", a.get("size", 1_000), remote)
            return None if None in (pool, side, s) else {"op": op, "pool": pool, "side": side, "size": s}
        if op == "close":
            self.keys(p, a, ("op", "pool", "side"))
            pool = self.pool_ref(f"{p}.pool", a.get("pool", 0 if remote else "$victim"), remote)
            side = self.side_ref(f"{p}.side", a.get("side", 0 if remote else "$counter"), remote)
            return None if None in (pool, side) else {"op": op, "pool": pool, "side": side}
        if op == "mint":
            self.keys(p, a, ("op", "pool", "half_width", "pct"))
            pool = self.pool_ref(f"{p}.pool", a.get("pool", "$victim"), remote)
            hw = self.int_in(f"{p}.half_width", a.get("half_width", 3), 1, 40)
            pct = self.int_in(f"{p}.pct", a.get("pct", 300), 1, 20_000)
            return None if None in (pool, hw, pct) else {"op": op, "pool": pool, "half_width": hw, "pct": pct}
        if op == "liquidate":
            self.keys(p, a, ("op", "market", "repay_ppm"))
            m = self.int_in(f"{p}.market", a.get("market", 0), 0, 15)
            r = self.int_in(f"{p}.repay_ppm", a.get("repay_ppm", 500_000), 1, PPM)
            return None if None in (m, r) else {"op": op, "market": m, "repay_ppm": r}
        if op == "arb":
            self.keys(p, a, ("op", "pool", "size", "max_hops"))
            pool = self.pool_ref(f"{p}.pool", a.get("pool", 0 if remote else "$victim"), remote)
            s = self.size(f"{p}.size", a.get("size", 2_000), remote)
            if s is not None and ("balance" in s or "tload" in s):
                self.bad(f"{p}.size", "arb sizes are ppm, victim_ppm or optimize specs")
                s = None
            hops = self.int_in(f"{p}.max_hops", a.get("max_hops", 3), 2, 3)
            return None if None in (pool, s, hops) else {"op": op, "pool": pool, "size": s, "max_hops": hops}
        if op == "tstore":
            self.keys(p, a, ("op", "key", "value"))
            key = a.get("key")
            if not (isinstance(key, str) and KEY_RX.match(key)):
                self.bad(f"{p}.key", "needs a transient key ([A-Za-z_][A-Za-z0-9_]{0,31})")
                return None
            v = self.tvalue(f"{p}.value", a.get("value", {"last_out": True}), remote)
            return None if v is None else {"op": op, "key": key, "value": v}
        if op == "tswap":
            self.keys(p, a, ("op", "a", "b"))
            ka, kb = a.get("a"), a.get("b")
            if not all(isinstance(k, str) and KEY_RX.match(k) for k in (ka, kb)) or ka == kb:
                self.bad(p, "tswap needs two different transient keys 'a' and 'b'")
                return None
            return {"op": op, "a": ka, "b": kb}
        if op == "if":
            self.keys(p, a, ("op", "cond", "then", "else"))
            cond = self.node(f"{p}.cond", a.get("cond"), depth + 1, True, remote)
            then = self.script(f"{p}.then", a.get("then", []), depth + 1, remote)
            other = self.script(f"{p}.else", a.get("else", []), depth + 1, remote)
            return None if None in (cond, then, other) else {"op": op, "cond": cond, "then": then, "else": other}
        if op == "repeat":
            self.keys(p, a, ("op", "times", "body"))
            n = self.int_in(f"{p}.times", a.get("times", 1), 1, MAX_REPEAT)
            body = self.script(f"{p}.body", a.get("body"), depth + 1, remote)
            if body is not None and not body:
                self.bad(f"{p}.body", "repeat needs a non-empty body")
                body = None
            return None if None in (n, body) else {"op": op, "times": n, "body": body}
        if op == "atomic":
            self.keys(p, a, ("op", "body", "on_revert"))
            mode = a.get("on_revert", "rollback")
            if mode not in ("rollback", "abort"):
                self.bad(f"{p}.on_revert", "must be 'rollback' or 'abort'")
                mode = None
            body = self.script(f"{p}.body", a.get("body"), depth + 1, remote)
            if body is not None and not body:
                self.bad(f"{p}.body", "atomic needs a non-empty body")
                body = None
            return None if None in (mode, body) else {"op": op, "body": body, "on_revert": mode}
        # bridge
        self.keys(p, a, ("op", "chain", "token", "amount_ppm", "fee_bps", "delay_blocks", "body"))
        chain = a.get("chain", "l2")
        if not (isinstance(chain, str) and KEY_RX.match(chain)):
            self.bad(f"{p}.chain", "needs a chain name ([A-Za-z_][A-Za-z0-9_]{0,31})")
            chain = None
        token = a.get("token", "$base")
        if not self.token_ok(token):
            self.bad(f"{p}.token", "needs a token reference")
            token = None
        amount = self.int_in(f"{p}.amount_ppm", a.get("amount_ppm", PPM), 1, PPM)
        fee = self.int_in(f"{p}.fee_bps", a.get("fee_bps", 5), 0, 1_000)
        delay = self.int_in(f"{p}.delay_blocks", a.get("delay_blocks", 2), 0, 32)
        body = self.script(f"{p}.body", a.get("body"), depth + 1, True)
        if body is not None and not body:
            self.bad(f"{p}.body", "bridge needs a non-empty remote body")
            body = None
        if None in (chain, token, amount, fee, delay, body):
            return None
        return {"op": op, "chain": chain, "token": token, "amount_ppm": amount, "fee_bps": fee,
                "delay_blocks": delay, "body": body}

    # -- pipeline ------------------------------------------------------------

    def pipeline(self, path: str, raw: Any, base: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        out = dict(PIPELINE_DEFAULTS if base is None else base)
        if raw is None:
            return out
        if not isinstance(raw, dict):
            self.bad(path, "pipeline must be an object")
            return out
        self.keys(path, raw, PIPELINE_DEFAULTS)
        for k, v in raw.items():
            if k not in PIPELINE_DEFAULTS:
                continue
            if k == "on_revert":
                if v not in ("abort", "skip"):
                    self.bad(f"{path}.on_revert", "must be 'abort' or 'skip'")
                    continue
                out[k] = v
            elif isinstance(PIPELINE_DEFAULTS[k], bool):
                if not isinstance(v, bool):
                    self.bad(f"{path}.{k}", "must be true or false")
                    continue
                out[k] = v
            else:
                lo, hi = PIPELINE_RANGES[k]
                n = self.int_in(f"{path}.{k}", v, lo, hi)
                if n is not None:
                    out[k] = n
        return out


def _check_blueprint(raw: Any) -> Tuple[Optional[Dict[str, Any]], List[Tuple[str, str]]]:
    ck = _Checker()
    if not isinstance(raw, dict):
        return None, [("$", "a blueprint must be a JSON object")]
    ck.keys("$", raw, ("schema", "name", "description", "trigger", "tracing", "script", "pipeline"))
    schema = raw.get("schema", SCHEMA_VERSION)
    if schema != SCHEMA_VERSION or isinstance(schema, bool):
        ck.bad("$.schema", f"unsupported schema {schema!r} (this version reads {SCHEMA_VERSION})")
    name = raw.get("name")
    if not (isinstance(name, str) and NAME_RX.match(name)):
        ck.bad("$.name", "needs 1-64 chars of [A-Za-z0-9_.:-]")
    desc = raw.get("description", "")
    if not isinstance(desc, str) or len(desc) > 512:
        ck.bad("$.description", "must be a string of at most 512 chars")
        desc = ""
    tracing_raw = raw.get("tracing", {})
    tracing = {"opcodes": False, "sequential": False, "record": False}
    if not isinstance(tracing_raw, dict):
        ck.bad("$.tracing", "must be an object")
    else:
        ck.keys("$.tracing", tracing_raw, tracing)
        for k in tracing:
            v = tracing_raw.get(k, False)
            if not isinstance(v, bool):
                ck.bad(f"$.tracing.{k}", "must be true or false")
            else:
                tracing[k] = v
    trigger = ck.node("$.trigger", raw.get("trigger", {"const": True}), 1, False, False)
    if ck.uses_opcodes and not tracing["opcodes"]:
        ck.bad("$.tracing.opcodes", "opcode trigger nodes need tracing.opcodes = true")
    script = ck.script("$.script", raw.get("script"), 1, False)
    if script is not None and not script:
        ck.bad("$.script", "needs at least one action")
    if len(ck.optimize_paths) > 1:
        ck.bad(ck.optimize_paths[1], "only one 'optimize' size per blueprint")
    pipeline = ck.pipeline("$.pipeline", raw.get("pipeline"))
    if ck.issues:
        return None, ck.issues
    bp = {"schema": SCHEMA_VERSION, "name": name, "description": desc, "trigger": trigger, "tracing": tracing,
          "script": script, "pipeline": pipeline}
    try:
        program = _assemble_normalized(bp)
    except BlueprintError as exc:
        return None, exc.issues
    if len(program.script_code) > MAX_INSTRUCTIONS:
        return None, [("$.script", f"assembles to {len(program.script_code)} instructions "
                                   f"(limit {MAX_INSTRUCTIONS})")]
    return bp, []


def validate_blueprint(raw: Any) -> List[Tuple[str, str]]:
    """Every ``(path, message)`` problem in ``raw`` (empty when it compiles)."""
    return _check_blueprint(raw)[1]


def normalize_blueprint(raw: Any) -> Dict[str, Any]:
    """``raw`` with every default filled in; raises :class:`BlueprintError` listing all problems."""
    bp, issues = _check_blueprint(raw)
    if bp is None:
        raise BlueprintError(issues)
    return bp


def canonical_json(bp: Mapping[str, Any]) -> str:
    return json.dumps(bp, sort_keys=True, separators=(",", ":"))


def blueprint_digest(raw: Any) -> str:
    """SHA-256 of the normalised blueprint's canonical JSON (plus the schema version)."""
    bp = normalize_blueprint(raw)
    return hashlib.sha256(f"custom-bot:v{SCHEMA_VERSION}:{canonical_json(bp)}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Trigger classes
# ---------------------------------------------------------------------------


def trigger_class(node: Mapping[str, Any]) -> str:
    """Structure of a trigger tree without thresholds (children of all / any sorted)."""
    if "all" in node or "any" in node:
        kind = "all" if "all" in node else "any"
        return f"{kind}(" + ",".join(sorted(trigger_class(k) for k in node[kind])) + ")"
    if "not" in node:
        return f"not({trigger_class(node['not'])})"
    if "const" in node:
        return f"const:{str(bool(node['const'])).lower()}"
    if "metric" in node:
        return f"metric:{node['metric']}{node.get('op', '>=')}"
    if "opcode" in node:
        return f"opcode:{node['opcode']}{node.get('op', '>=')}"
    if "opcode_seq" in node:
        return "opseq:" + ">".join(node["opcode_seq"])
    if "event" in node:
        return f"event:{node['event']}"
    if "tload" in node:
        return f"tload{node.get('op', '>=')}"
    if "balance" in node:
        return f"balance{node.get('op', '>=')}"
    raise BlueprintError([("$.trigger", f"unknown node {sorted(node)}")])


# ---------------------------------------------------------------------------
# Assembler: trigger trees and scripts -> bytecode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledProgram:
    trigger_code: Tuple[Any, ...]
    script_code: Tuple[Any, ...]
    optimize: Optional[Tuple[int, int]]
    model: bool  # uses a modelled component (bridge or lending)
    victim_free: bool  # the script never needs a victim

    def code_bytes(self) -> bytes:
        return json.dumps([_thaw(self.trigger_code), _thaw(self.script_code)], sort_keys=True,
                          separators=(",", ":"), default=str).encode()


def _assemble_trigger(node: Mapping[str, Any], out: List[Any]) -> None:
    if "all" in node or "any" in node:
        kind = "all" if "all" in node else "any"
        for k in node[kind]:
            _assemble_trigger(k, out)
        out.append(("AND" if kind == "all" else "OR", len(node[kind])))
    elif "not" in node:
        _assemble_trigger(node["not"], out)
        out.append(("NOT",))
    elif "const" in node:
        out.append(("CONST", bool(node["const"])))
    elif "metric" in node:
        out.append(("METRIC", node["metric"], node["op"], _freeze(node["value"])))
    elif "opcode" in node:
        out.append(("OPCODE", node["opcode"], node["op"], _freeze(node["value"])))
    elif "opcode_seq" in node:
        out.append(("OPSEQ", tuple(node["opcode_seq"])))
    elif "event" in node:
        out.append(("EVENT", node["event"], _freeze({k: v for k, v in node.items() if k != "event"})))
    elif "tload" in node:
        out.append(("TLOAD", node["tload"], node["op"], _freeze(node["value"])))
    elif "balance" in node:
        out.append(("BALANCE", node["balance"], node["op"], _freeze(node["value"])))
    else:
        raise BlueprintError([("$.trigger", f"unknown node {sorted(node)}")])


def _assemble_script(actions: Sequence[Mapping[str, Any]], out: List[Any], flags: Dict[str, Any]) -> None:
    for a in actions:
        if len(out) > MAX_INSTRUCTIONS:
            raise BlueprintError([("$.script", f"assembles to more than {MAX_INSTRUCTIONS} instructions")])
        op = a["op"]
        if op == "if":
            cond: List[Any] = []
            _assemble_trigger(a["cond"], cond)
            at = len(out)
            out.append(None)  # patched: JUMPF
            _assemble_script(a["then"], out, flags)
            if a["else"]:
                jmp = len(out)
                out.append(None)  # patched: JUMP over else
                out[at] = ("JUMPF", tuple(cond), len(out))
                _assemble_script(a["else"], out, flags)
                out[jmp] = ("JUMP", len(out))
            else:
                out[at] = ("JUMPF", tuple(cond), len(out))
        elif op == "repeat":
            for _ in range(a["times"]):
                _assemble_script(a["body"], out, flags)
        elif op == "atomic":
            at = len(out)
            out.append(None)  # patched: ATOMIC with the address after COMMIT
            _assemble_script(a["body"], out, flags)
            out.append(("COMMIT",))
            out[at] = ("ATOMIC", len(out), a["on_revert"])
        elif op == "bridge":
            body: List[Any] = []
            _assemble_script(a["body"], body, flags)
            body.append(("HALT",))
            params = {k: v for k, v in a.items() if k not in ("op", "body")}
            out.append(("BRIDGE", _freeze(params), tuple(body)))
            flags["model"] = True
        else:
            if op == "liquidate":
                flags["model"] = True
            if op in VICTIM_ACTIONS:
                flags["victim"] = True
            size = a.get("size")
            if isinstance(size, dict) and "optimize" in size:
                flags["optimize"] = tuple(size["optimize"])
            out.append(("ACT", op, _freeze({k: v for k, v in a.items() if k != "op"})))


def _assemble_normalized(bp: Mapping[str, Any]) -> CompiledProgram:
    trig: List[Any] = []
    _assemble_trigger(bp["trigger"], trig)
    code: List[Any] = []
    flags: Dict[str, Any] = {"model": False, "victim": False, "optimize": None}
    _assemble_script(bp["script"], code, flags)
    code.append(("HALT",))
    _verify_program(code)
    return CompiledProgram(tuple(trig), tuple(code), flags["optimize"], bool(flags["model"]), not flags["victim"])


def _verify_program(code: Sequence[Any]) -> None:
    """Every jump goes forward and lands inside the program; ATOMIC / COMMIT are balanced."""
    depth = 0
    for pc, ins in enumerate(code):
        if ins is None:
            raise BlueprintError([("$.script", f"unpatched instruction at {pc}")])
        kind = ins[0]
        if kind in ("JUMP", "JUMPF", "ATOMIC"):
            target = ins[1] if kind != "JUMPF" else ins[2]
            if kind == "ATOMIC":
                target = ins[1]
            if not pc < target <= len(code):
                raise BlueprintError([("$.script", f"jump at {pc} to {target} is not forward")])
        if kind == "ATOMIC":
            depth += 1
        elif kind == "COMMIT":
            depth -= 1
            if depth < 0:
                raise BlueprintError([("$.script", f"COMMIT without ATOMIC at {pc}")])
        elif kind == "BRIDGE":
            _verify_program(ins[2])
    if depth:
        raise BlueprintError([("$.script", "unbalanced ATOMIC / COMMIT")])


def assemble(raw: Any) -> CompiledProgram:
    """Validate ``raw`` and return its bytecode."""
    return _assemble_normalized(normalize_blueprint(raw))


def _fmt(v: Any) -> str:
    return json.dumps(_thaw(v), sort_keys=True, separators=(",", ":"), default=str)


def disassemble(code: Sequence[Any], indent: str = "") -> List[str]:
    """Human-readable listing of trigger or script bytecode."""
    lines: List[str] = []
    for pc, ins in enumerate(code):
        kind = ins[0]
        if kind == "ACT":
            lines.append(f"{indent}{pc:04d} ACT     {ins[1]} {_fmt(ins[2])}")
        elif kind == "JUMPF":
            cond = " ".join(i[0] for i in ins[1])
            lines.append(f"{indent}{pc:04d} JUMPF   -> {ins[2]:04d} if not [{cond}]")
        elif kind == "JUMP":
            lines.append(f"{indent}{pc:04d} JUMP    -> {ins[1]:04d}")
        elif kind == "ATOMIC":
            lines.append(f"{indent}{pc:04d} ATOMIC  {ins[2]} (end {ins[1]:04d})")
        elif kind == "BRIDGE":
            lines.append(f"{indent}{pc:04d} BRIDGE  {_fmt(ins[1])}")
            lines.extend(disassemble(ins[2], indent + "    "))
        elif kind in ("METRIC", "OPCODE", "TLOAD", "BALANCE"):
            lines.append(f"{indent}{pc:04d} {kind:<7} {ins[1]} {ins[2]} {_fmt(ins[3])}")
        elif kind == "EVENT":
            lines.append(f"{indent}{pc:04d} EVENT   {ins[1]} {_fmt(ins[2])}")
        elif kind == "OPSEQ":
            lines.append(f"{indent}{pc:04d} OPSEQ   {'>'.join(ins[1])}")
        elif kind in ("AND", "OR"):
            lines.append(f"{indent}{pc:04d} {kind:<7} {ins[1]}")
        elif kind == "CONST":
            lines.append(f"{indent}{pc:04d} CONST   {str(ins[1]).lower()}")
        else:
            lines.append(f"{indent}{pc:04d} {kind}")
    return lines


# ---------------------------------------------------------------------------
# Modelled opcode traces and victim contexts
# ---------------------------------------------------------------------------


def opcode_trace(pool: Any, ticks_crossed: int = 0) -> Tuple[str, ...]:
    """Modelled opcode sequence of one swap on ``pool`` (not an EVM trace)."""
    version = pool_version(pool)
    ticks = max(0, min(int(ticks_crossed), 64))
    if version == "v2":
        return ("CALL", "SLOAD", "SLOAD", "CALL", "SSTORE", "SSTORE", "LOG1", "LOG3")
    seq: List[str] = ["CALL", "TSTORE", "SLOAD", "SLOAD"] if version == "v4" else ["CALL", "SLOAD", "SLOAD"]
    seq.extend(["SLOAD", "SSTORE"] * ticks)
    if version == "v4":
        if getattr(pool, "dynamic_fee", False) or getattr(pool, "hook_fee_pips", 0):
            seq.append("CALL")  # hook call (fee override)
        seq.extend(["SSTORE", "LOG3", "TSTORE", "TLOAD", "CALL", "TSTORE"])
    else:
        seq.extend(["CALL", "CALL", "SSTORE", "LOG3"])
    return tuple(seq)


def _subsequence(needle: Sequence[str], hay: Sequence[str], contiguous: bool) -> bool:
    n = len(needle)
    if contiguous:
        return any(tuple(hay[i:i + n]) == tuple(needle) for i in range(len(hay) - n + 1))
    it = iter(hay)
    return all(any(h == x for h in it) for x in needle)


def _compare(lhs: int, op: str, value: Any) -> bool:
    if op == "<":
        return lhs < value
    if op == "<=":
        return lhs <= value
    if op == ">":
        return lhs > value
    if op == ">=":
        return lhs >= value
    if op == "==":
        return lhs == value
    if op == "!=":
        return lhs != value
    if op == "between":
        return value[0] <= lhs <= value[1]
    if op == "in":
        return lhs in value
    raise BlueprintError([("$", f"unknown comparator {op!r}")])


def _pool_fee_pips(pool: Any) -> int:
    if isinstance(pool, V2Pool):
        return int(pool.fee_bps) * 100
    return int(getattr(pool, "fee_pips", 0))


class VictimContext:
    """Metrics, modelled trace and events of one pending single-hop victim, on the scenario's state."""

    def __init__(self, scenario: Scenario, tx: PendingTx, position: int = 0) -> None:
        if len(tx.route) != 1:
            raise ActionReverted("only single-hop victims are candidates")
        self.scenario = scenario
        self.tx = tx
        self.position = position
        self.pool_index = next((i for i, p in enumerate(scenario.pools) if p.pool_id == tx.route[0]), -1)
        if self.pool_index < 0:
            raise ActionReverted(f"victim pool {tx.route[0]!r} is not in the scenario")
        self.pool = scenario.pools[self.pool_index]
        self.zero_for_one, self.token_out = _direction(self.pool, tx.token_in)
        self.side = 0 if self.zero_for_one else 1
        self.quote = self.pool.quote_exact_in(tx.amount_in, self.zero_for_one)
        self._memo: Dict[str, Any] = {}

    def _same_side(self) -> List[PendingTx]:
        if "same" not in self._memo:
            lst = [t for t in self.scenario.victims
                   if len(t.route) == 1 and t.route[0] == self.pool.pool_id and t.token_in == self.tx.token_in]
            lst.sort(key=lambda t: (-t.amount_in, t.tx_hash))
            self._memo["same"] = lst
        return self._memo["same"]

    def trace(self) -> Tuple[str, ...]:
        if "trace" not in self._memo:
            self._memo["trace"] = opcode_trace(self.pool, self.quote.ticks_crossed)
        return self._memo["trace"]

    def metric(self, name: str) -> int:
        if name in self._memo:
            return self._memo[name]
        tx, q, pool = self.tx, self.quote, self.pool
        if name == "victim_priority_fee_wei":
            v = tx.priority_fee_wei
        elif name == "victim_gas_limit":
            v = tx.gas_limit
        elif name == "victim_amount_in":
            v = tx.amount_in
        elif name == "victim_value_base":
            v = math.floor(self.scenario.value(tx.token_in, tx.amount_in))
        elif name == "victim_size_ppm":
            v = tx.amount_in * PPM // max(1, input_reserve_scale(pool, self.zero_for_one))
        elif name == "victim_slippage_bps":
            v = (q.amount_out - tx.amount_out_min) * BPS // q.amount_out if q.amount_out > 0 else 0
        elif name == "victim_impact_bps":
            before, after = q.price_before, q.price_after
            v = math.floor((before - after) * BPS / before) if before > 0 else 0
        elif name == "victim_rank":
            v = next(i for i, t in enumerate(self._same_side()) if t.tx_hash == tx.tx_hash)
        elif name == "same_side_victims":
            v = len(self._same_side())
        elif name == "block_position":
            v = self.position
        elif name == "ticks_crossed":
            v = q.ticks_crossed
        elif name == "pool_index":
            v = self.pool_index
        elif name == "pool_version":
            v = int(pool_version(pool)[1:])
        elif name == "pool_side":
            v = self.side
        elif name == "pool_fee_pips":
            v = _pool_fee_pips(pool)
        elif name == "gas_price_wei":
            v = self.scenario.gas_price_wei
        elif name == "cross_venue_spread_bps":
            v = self._spread_bps()
        elif name == "loan_min_health_bps":
            v = self._min_health_bps()
        else:
            raise BlueprintError([("$", f"unknown metric {name!r}")])
        self._memo[name] = int(v)
        return int(v)

    def _spread_bps(self) -> int:
        t0 = self.pool.tokens()[0]
        prices: List[Fraction] = []
        for p in self.scenario.pools:
            if set(p.tokens()) != set(self.pool.tokens()):
                continue
            zfo, _ = _direction(p, t0)
            prices.append(p.spot_price(zfo))
        if len(prices) < 2 or min(prices) <= 0:
            return 0
        return math.floor((max(prices) - min(prices)) * BPS / min(prices))

    def _min_health_bps(self) -> int:
        markets = [m for m in self.scenario.markets if m.oracle_pool_id == self.pool.pool_id]
        if not markets:
            return 10**9
        state = AttackState(self.scenario)
        try:
            state.victim(self.tx)
        except (ActionReverted, MASEError):
            pass
        best = 10**9
        for m in markets:
            market = state.markets[m.market_id]
            price = market.price(state.pools)
            for loan in market.loans.values():
                if loan.debt > 0:
                    best = min(best, math.floor(market.health(loan, price) * BPS))
        return best

    def event(self, name: str, params: FrozenMap) -> bool:
        if name == "Swap":
            pool, side = params.get("pool", -1), params.get("side", -1)
            return (pool < 0 or pool == self.pool_index) and (side < 0 or side == self.side)
        if name == "LargeSwap":
            return self.metric("victim_size_ppm") >= params.get("min_ppm", 1_000)
        if name == "PriceDeviation":
            return self.metric("cross_venue_spread_bps") >= params.get("min_bps", 10)
        if name == "LoanAtRisk":
            return self.metric("loan_min_health_bps") < params.get("max_health_bps", 10_100)
        if name == "TickCross":
            return self.metric("ticks_crossed") >= params.get("min_ticks", 1)
        raise BlueprintError([("$", f"unknown event {name!r}")])


# ---------------------------------------------------------------------------
# Virtual machine
# ---------------------------------------------------------------------------


class _Runtime:
    """Per-execution state: the attack state, transient storage and the victim context."""

    __slots__ = ("scenario", "state", "ctx", "tstore", "last_out", "param", "pipeline", "tracing", "model",
                 "remote_legs", "depth")

    def __init__(self, scenario: Scenario, ctx: Optional[VictimContext], param: Optional[int],
                 pipeline: Mapping[str, Any], tracing: Mapping[str, Any], depth: int = 0) -> None:
        self.scenario = scenario
        self.state = AttackState(scenario)
        self.ctx = ctx
        self.tstore: Dict[str, int] = {}
        self.last_out = 0
        self.param = param
        self.pipeline = pipeline
        self.tracing = tracing
        self.model = False
        self.remote_legs: List[Dict[str, Any]] = []
        self.depth = depth

    def snapshot(self) -> Tuple[AttackState, Dict[str, int], int, int]:
        memo = {id(self.scenario): self.scenario}
        return copy.deepcopy(self.state, memo), dict(self.tstore), self.last_out, len(self.remote_legs)

    def restore(self, snap: Tuple[AttackState, Dict[str, int], int, int]) -> None:
        self.state, tstore, self.last_out, legs = snap
        self.tstore = dict(tstore)
        del self.remote_legs[legs:]

    # -- references ----------------------------------------------------------

    def need_ctx(self) -> VictimContext:
        if self.ctx is None:
            raise ActionReverted("this action needs a victim")
        return self.ctx

    def token(self, ref: str) -> str:
        if ref == "$base":
            return self.scenario.base
        if ref == "$in":
            return self.need_ctx().tx.token_in
        if ref == "$out":
            return self.need_ctx().token_out
        if ref not in self.scenario.prices:
            raise ActionReverted(f"unknown token {ref!r}")
        return ref

    def pool(self, ref: Any) -> Any:
        if ref == "$victim":
            return self.state.pools[self.need_ctx().pool.pool_id]
        if not 0 <= ref < len(self.scenario.pools):
            raise ActionReverted(f"no pool {ref}")
        return self.state.pools[self.scenario.pools[ref].pool_id]

    def side_token(self, pool: Any, ref: Any) -> str:
        if ref == "$victim":
            return self.need_ctx().tx.token_in
        if ref == "$counter":
            return self.need_ctx().token_out
        return pool.tokens()[ref]

    def amount(self, spec: Any, pool: Any, token_in: str) -> int:
        if isinstance(spec, FrozenMap):
            if spec.get("ppm") is not None:
                zfo, _ = _direction(pool, token_in)
                return input_reserve_scale(pool, zfo) * spec.get("ppm") // PPM
            if spec.get("optimize") is not None:
                if self.param is None:
                    raise ActionReverted("optimize size without a search value")
                zfo, _ = _direction(pool, token_in)
                return input_reserve_scale(pool, zfo) * self.param // PPM
            if spec.get("victim_ppm") is not None:
                return self.need_ctx().tx.amount_in * spec.get("victim_ppm") // PPM
            if spec.get("tload") is not None:
                return self.tload(spec.get("tload"))
            if spec.get("balance") is not None:
                bal = self.state.ledger.get(self.token(spec.get("balance")), 0)
                return max(0, bal) * spec.get("ppm", PPM) // PPM
        raise ActionReverted(f"bad size spec {spec!r}")

    def tload(self, key: str) -> int:
        self.state.gas_used += GAS_TRANSIENT
        return self.tstore.get(key, 0)

    # -- conditions ----------------------------------------------------------

    def eval_trigger(self, code: Sequence[Any]) -> bool:
        stack: List[bool] = []
        for ins in code:
            kind = ins[0]
            if kind == "CONST":
                stack.append(ins[1])
            elif kind == "METRIC":
                stack.append(_compare(self.need_ctx().metric(ins[1]), ins[2], ins[3]))
            elif kind == "OPCODE":
                stack.append(_compare(self.need_ctx().trace().count(ins[1]), ins[2], ins[3]))
            elif kind == "OPSEQ":
                stack.append(_subsequence(ins[1], self.need_ctx().trace(), bool(self.tracing.get("sequential"))))
            elif kind == "EVENT":
                stack.append(self.need_ctx().event(ins[1], ins[2]))
            elif kind == "TLOAD":
                stack.append(_compare(self.tload(ins[1]), ins[2], ins[3]))
            elif kind == "BALANCE":
                stack.append(_compare(self.state.ledger.get(self.token(ins[1]), 0), ins[2], ins[3]))
            elif kind in ("AND", "OR"):
                n = ins[1]
                vals, stack[-n:] = stack[-n:], []
                stack.append(all(vals) if kind == "AND" else any(vals))
            elif kind == "NOT":
                stack.append(not stack.pop())
            else:
                raise BlueprintError([("$", f"bad trigger instruction {kind!r}")])
        if len(stack) != 1:
            raise BlueprintError([("$", "trigger program left a malformed stack")])
        return stack[0]

    # -- actions -------------------------------------------------------------

    def act(self, op: str, a: FrozenMap) -> None:
        st = self.state
        if op == "frontrun":
            ctx = self.need_ctx()
            pool = self.pool("$victim")
            res = st.swap(pool.pool_id, ctx.tx.token_in, self.amount(a.get("size"), pool, ctx.tx.token_in),
                          label="frontrun")
            self.last_out = res.amount_out
        elif op == "victim":
            res = st.victim(self.need_ctx().tx)
            self.last_out = res.amount_out
        elif op == "victims":
            ctx = self.need_ctx()
            added = 0
            for tx in ctx._same_side():
                if added >= a.get("max"):
                    break
                if tx.tx_hash == ctx.tx.tx_hash or tx.tx_hash in st.victim_outputs:
                    continue
                try:
                    st.victim(tx)
                    added += 1
                except ActionReverted:
                    continue  # a pending swap that would revert is left out of the bundle
        elif op == "backrun":
            ctx = self.need_ctx()
            amount = st.ledger.get(ctx.token_out, 0)
            if amount <= 0:
                raise ActionReverted("nothing to back-run")
            res = st.swap(ctx.pool.pool_id, ctx.token_out, amount, label="backrun")
            self.last_out = res.amount_out
        elif op == "swap":
            pool = self.pool(a.get("pool"))
            token_in = self.side_token(pool, a.get("side"))
            res = st.swap(pool.pool_id, token_in, self.amount(a.get("size"), pool, token_in), label="script")
            self.last_out = res.amount_out
        elif op == "close":
            pool = self.pool(a.get("pool"))
            token = self.side_token(pool, a.get("side"))
            amount = st.ledger.get(token, 0)
            if amount <= 0:
                raise ActionReverted("nothing to close")
            res = st.swap(pool.pool_id, token, amount, label="close")
            self.last_out = res.amount_out
        elif op == "mint":
            pool = self.pool(a.get("pool"))
            if not isinstance(pool, ConcentratedPool) or pool.liquidity <= 0:
                raise ActionReverted("mint needs an active concentrated pool")
            s = pool.tick_spacing
            centre = pool.tick // s * s
            hw = a.get("half_width")
            st.mint(pool.pool_id, centre - hw * s, centre + (hw + 1) * s, pool.liquidity * a.get("pct") // 100)
        elif op == "burn":
            st.burn()
        elif op == "liquidate":
            mi = a.get("market")
            if mi >= len(self.scenario.markets):
                raise ActionReverted("no such market")
            market = st.markets[self.scenario.markets[mi].market_id]
            price = market.price(st.pools)
            cands = sorted((market.health(loan, price), b) for b, loan in market.loans.items()
                           if b != st.ACTOR and loan.debt > 0 and market.health(loan, price) < 1)
            if not cands:
                raise ActionReverted("no liquidatable loan")
            borrower = cands[0][1]
            repay = market.max_repay(market.loans[borrower]) * a.get("repay_ppm") // PPM
            st.liquidate(market.market_id, borrower, repay)
            self.model = True
        elif op == "arb":
            self._arb(a)
        elif op == "tstore":
            self.tstore[a.get("key")] = self._tvalue(a.get("value"))
            st.gas_used += GAS_TRANSIENT
        elif op == "tswap":
            ka, kb = a.get("a"), a.get("b")
            self.tstore[ka], self.tstore[kb] = self.tstore.get(kb, 0), self.tstore.get(ka, 0)
            st.gas_used += 4 * GAS_TRANSIENT
        else:
            raise BlueprintError([("$", f"unknown action {op!r}")])

    def _tvalue(self, v: FrozenMap) -> int:
        if v.get("const") is not None:
            return v.get("const")
        if v.get("last_out"):
            return self.last_out
        if v.get("victim_amount"):
            return self.need_ctx().tx.amount_in
        if v.get("balance") is not None:
            bal = self.state.ledger.get(self.token(v.get("balance")), 0)
            return max(0, bal) * v.get("ppm", PPM) // PPM
        raise ActionReverted(f"bad tstore value {v!r}")

    def _arb(self, a: FrozenMap) -> None:
        pool = self.pool(a.get("pool"))
        cycles = [c for c in _cycles_through(self.scenario, pool.pool_id) if len(c) <= a.get("max_hops")]
        best: Optional[Tuple[int, int, Tuple[Tuple[str, str], ...], int]] = None
        memo = {id(self.scenario): self.scenario}
        for i, cyc in enumerate(cycles[:MAX_CYCLES]):
            first = self.state.pools[cyc[0][0]]
            amount = self.amount(a.get("size"), first, cyc[0][1])
            if amount <= 0:
                continue
            trial = copy.deepcopy(self.state, memo)
            try:
                before = trial.ledger.get(cyc[0][1], 0)
                _run_cycle(trial, cyc, amount, "arb")
                gain = trial.ledger.get(cyc[0][1], 0) - before
            except (ActionReverted, MASEError):
                continue
            value = math.floor(self.scenario.value(cyc[0][1], gain))
            if value > 0 and (best is None or (value, -i) > (best[0], -best[1])):
                best = (value, i, cyc, amount)
        if best is None:
            raise ActionReverted("no profitable cycle")
        _run_cycle(self.state, best[2], best[3], "arb")

    def bridge(self, params: FrozenMap, body: Sequence[Any]) -> None:
        if self.depth > 0:
            raise ActionReverted("bridges do not nest")
        st = self.state
        token = self.token(params.get("token"))
        balance = st.ledger.get(token, 0)
        amount = max(0, balance) * params.get("amount_ppm") // PPM
        if amount <= 0:
            raise ActionReverted(f"no {token} balance to bridge")
        fee_bps = params.get("fee_bps")
        fee_out = -(-amount * fee_bps // BPS)
        remote_scn = scenario_for(_remote_seed(self.scenario.seed, params.get("chain")), len(self.scenario.victims))
        if token not in remote_scn.prices:
            raise ActionReverted(f"{token} does not exist on {params.get('chain')}")
        remote = _Runtime(remote_scn, None, self.param, self.pipeline, self.tracing, depth=self.depth + 1)
        delayed = 0
        for tx in remote_scn.victims:
            if delayed >= params.get("delay_blocks"):
                break
            if len(tx.route) != 1:
                continue
            try:
                remote.state.victim(tx)
            except (ActionReverted, MASEError):
                pass  # a remote user swap that reverts simply does not happen
            delayed += 1
        remote.state._credit(token, amount - fee_out)
        remote.execute(body)
        if remote.state.capital_used > remote_scn.capital_base:
            raise ActionReverted("remote leg exceeds capital")
        back: Dict[str, int] = {}
        for tok, amt in sorted(remote.state.ledger.items()):
            if amt < 0:
                raise ActionReverted(f"remote leg left a {tok} debt")
            if amt > 0:
                back[tok] = amt - (-(-amt * fee_bps // BPS))
        st._attacker_action(GAS_BRIDGE)
        st._credit(token, -amount)
        st.gas_used += GAS_BRIDGE + remote.state.gas_used // self.pipeline["remote_gas_divisor"]
        for tok, amt in back.items():
            if amt > 0:
                st._credit(tok, amt)
        leg = {"op": "bridge", "chain": params.get("chain"), "token": token, "amount": amount, "fee": fee_out,
               "delay_blocks": params.get("delay_blocks"), "remote_seed": remote_scn.seed,
               "remote_steps": len(remote.state.trace), "remote_gas": remote.state.gas_used, "returned": back}
        st.trace.append(dict(leg))
        self.remote_legs.append(leg)
        self.model = True

    # -- program execution ---------------------------------------------------

    def execute(self, code: Sequence[Any]) -> None:
        frames: List[Tuple[Tuple[AttackState, Dict[str, int], int, int], int, str]] = []
        skip = self.pipeline.get("on_revert") == "skip"
        pc = 0
        steps = 0
        while True:
            steps += 1
            if steps > 4 * MAX_INSTRUCTIONS:
                raise BlueprintError([("$", "program did not terminate")])
            ins = code[pc]
            kind = ins[0]
            if kind == "HALT":
                return
            if kind == "JUMP":
                pc = ins[1]
                continue
            if kind == "COMMIT":
                frames.pop()
                pc += 1
                continue
            if kind == "ATOMIC":
                frames.append((self.snapshot(), ins[1], ins[2]))
                pc += 1
                continue
            try:
                if kind == "JUMPF":
                    pc = pc + 1 if self.eval_trigger(ins[1]) else ins[2]
                    continue
                if kind == "ACT":
                    self.act(ins[1], ins[2])
                elif kind == "BRIDGE":
                    self.bridge(ins[1], ins[2])
                else:
                    raise BlueprintError([("$", f"bad instruction {kind!r}")])
                pc += 1
            except (ActionReverted, MASEError, ZeroDivisionError) as exc:
                while frames and frames[-1][2] == "abort":
                    frames.pop()
                if frames:
                    snap, end, _mode = frames.pop()
                    self.restore(snap)
                    pc = end
                    continue
                if skip and kind in ("ACT", "BRIDGE"):
                    pc += 1
                    continue
                if isinstance(exc, ActionReverted):
                    raise
                raise ActionReverted(str(exc)) from exc


def _remote_seed(seed: int, chain: str) -> int:
    return int.from_bytes(hashlib.sha256(f"bridge:{seed}:{chain}".encode()).digest()[:4], "big")


_SCENARIO_LOCK = threading.Lock()
_SCENARIOS: Dict[Tuple[int, int], Scenario] = {}


def scenario_for(seed: int, victims: int) -> Scenario:
    """``build_scenario(seed, victims)``, cached per process (scenarios are never mutated)."""
    key = (int(seed), int(victims))
    with _SCENARIO_LOCK:
        scn = _SCENARIOS.get(key)
    if scn is not None:
        return scn
    scn = build_scenario(key[0], key[1])
    with _SCENARIO_LOCK:
        if len(_SCENARIOS) >= SCENARIO_CACHE_LIMIT:
            _SCENARIOS.clear()
        return _SCENARIOS.setdefault(key, scn)


# ---------------------------------------------------------------------------
# Compiled bot classes
# ---------------------------------------------------------------------------


@dataclass
class PlanOutcome:
    victim: str
    valid: bool
    net_profit_base: int = 0
    param: Optional[int] = None
    result: Optional[AttackResult] = None
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"victim": self.victim, "valid": self.valid, "net_profit_base": self.net_profit_base,
                "param": self.param, "error": self.error,
                "result": self.result.to_dict() if self.result is not None else None}


@dataclass
class BotRun:
    name: str
    digest: str
    trigger_class: str
    scenario_seed: int
    candidates: int
    fired: int
    attempted: int
    plans: List[PlanOutcome] = field(default_factory=list)
    errors: Dict[str, int] = field(default_factory=dict)

    @property
    def valid(self) -> List[PlanOutcome]:
        return [p for p in self.plans if p.valid]

    @property
    def best(self) -> Optional[PlanOutcome]:
        valid = self.valid
        if not valid:
            return None
        return max(valid, key=lambda p: (p.net_profit_base, -self.plans.index(p)))

    @property
    def net_profit_base(self) -> int:
        b = self.best
        return b.net_profit_base if b is not None else 0

    def summary(self) -> Dict[str, Any]:
        best = self.best
        return {
            "name": self.name,
            "digest": self.digest,
            "trigger_class": self.trigger_class,
            "scenario_seed": self.scenario_seed,
            "candidates": self.candidates,
            "fired": self.fired,
            "attempted": self.attempted,
            "valid_plans": len(self.valid),
            "net_profit_base": self.net_profit_base,
            "best_victim": best.victim if best else None,
            "best_param": best.param if best else None,
            "status": best.result.status if best and best.result else None,
            "errors": dict(self.errors),
        }


class CustomBot:
    """Base class of every compiled blueprint; subclasses are created by :func:`compile_blueprint`."""

    BLUEPRINT: str = "{}"
    DIGEST: str = ""
    TRIGGER_CODE: Tuple[Any, ...] = ()
    SCRIPT_CODE: Tuple[Any, ...] = ()
    TRIGGER_CLASS: str = ""
    OPTIMIZE: Optional[Tuple[int, int]] = None
    MODEL: bool = False
    PIPELINE: FrozenMap = FrozenMap()
    TRACING: FrozenMap = FrozenMap()
    NAME: str = ""

    def __init__(self, overrides: Optional[Mapping[str, Any]] = None) -> None:
        self.pipeline = self.resolve_pipeline(overrides)

    @classmethod
    def blueprint(cls) -> Dict[str, Any]:
        return json.loads(cls.BLUEPRINT)

    @classmethod
    def resolve_pipeline(cls, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """The class pipeline with ``overrides`` applied (validated like the blueprint's own)."""
        ck = _Checker()
        out = ck.pipeline("overrides", None if overrides is None else dict(overrides), cls.PIPELINE.to_dict())
        if ck.issues:
            raise BlueprintError(ck.issues)
        return out

    @classmethod
    def disassemble(cls) -> Dict[str, List[str]]:
        return {"trigger": disassemble(cls.TRIGGER_CODE), "script": disassemble(cls.SCRIPT_CODE)}

    def contexts(self, scenario: Scenario) -> List[VictimContext]:
        out = []
        for i, tx in enumerate(scenario.victims):
            if len(tx.route) != 1:
                continue
            try:
                out.append(VictimContext(scenario, tx, i))
            except (ActionReverted, MASEError):
                continue
        return out

    def fires(self, ctx: VictimContext) -> bool:
        rt = _Runtime(ctx.scenario, ctx, None, self.pipeline, self.TRACING.to_dict())
        return rt.eval_trigger(self.TRIGGER_CODE)

    def execute(self, scenario: Scenario, ctx: Optional[VictimContext], param: Optional[int] = None) -> PlanOutcome:
        """Run the script once (with ``param`` for an optimize size) and score it."""
        name = ctx.tx.tx_hash if ctx is not None else ""
        pipe = self.pipeline
        rt = _Runtime(scenario, ctx, param, pipe, self.TRACING.to_dict())
        try:
            rt.execute(self.SCRIPT_CODE)
            if pipe["settle"]:
                rt.state.settle()
        except (ActionReverted, MASEError, ValueError, ZeroDivisionError) as exc:
            if isinstance(exc, BlueprintError):
                raise
            return PlanOutcome(name, False, param=param, error=_error_key(exc))
        st = rt.state
        extra: Dict[str, Any] = {"blueprint": self.NAME, "digest": self.DIGEST}
        if rt.remote_legs:
            extra["bridges"] = [dict(leg) for leg in rt.remote_legs]
        if self.TRACING.get("record"):
            extra["opcode_trace"] = [list(_step_trace(st, s)) for s in st.trace]
        result = st.result("custom", self.NAME, status="model" if rt.model else "exact", **extra)
        if not st.trace:
            return PlanOutcome(name, False, param=param, result=result, error="empty plan")
        if result.gas_used > pipe["max_gas"]:
            return PlanOutcome(name, False, param=param, result=result, error="gas cap")
        if st.capital_used > scenario.capital_base:
            return PlanOutcome(name, False, param=param, result=result, error="exceeds capital")
        if pipe["require_flat"] and not result.flat:
            return PlanOutcome(name, False, param=param, result=result, error="not flat")
        if result.net_profit_base < pipe["min_profit_base"]:
            return PlanOutcome(name, False, result.net_profit_base, param, result, "below min profit")
        return PlanOutcome(name, True, result.net_profit_base, param, result)

    def plan(self, scenario: Scenario, ctx: Optional[VictimContext]) -> PlanOutcome:
        """Best execution for one victim; an optimize size is searched with ``maximize_int``."""
        if self.OPTIMIZE is None:
            return self.execute(scenario, ctx)
        lo, hi = self.OPTIMIZE
        cache: Dict[int, PlanOutcome] = {}

        def score(x: int) -> Optional[int]:
            out = self.execute(scenario, ctx, x)
            cache[x] = out
            return out.net_profit_base if out.valid else None

        found = maximize_int(score, lo, hi, grid=self.pipeline["optimize_grid"],
                             refine=self.pipeline["optimize_refine"])
        if found is None:
            first = cache[min(cache)] if cache else self.execute(scenario, ctx, lo)
            return PlanOutcome(first.victim, False, param=None, error=first.error or "no feasible size")
        return cache[found[0]]

    def run(self, scenario: Scenario) -> BotRun:
        """Evaluate the trigger on every candidate victim and plan the first ``max_attempts`` that fire."""
        contexts = self.contexts(scenario)
        fired = [c for c in contexts if self.fires(c)]
        attempts = fired[: self.pipeline["max_attempts"]]
        run = BotRun(self.NAME, self.DIGEST, self.TRIGGER_CLASS, scenario.seed, len(contexts), len(fired),
                     len(attempts))
        errors: Counter = Counter()
        for ctx in attempts:
            out = self.plan(scenario, ctx)
            run.plans.append(out)
            if not out.valid:
                errors[out.error] += 1
        run.errors = dict(sorted(errors.items()))
        return run


def _error_key(exc: BaseException) -> str:
    text = str(exc)
    text = re.sub(r"\d+", "#", text)
    return text[:80] or type(exc).__name__


def _step_trace(state: AttackState, step: Mapping[str, Any]) -> Tuple[str, ...]:
    pid = step.get("pool_id")
    if step.get("op") in ("swap", "victim") and pid in state.pools:
        return opcode_trace(state.pools[pid], 0)
    if step.get("op") == "bridge":
        return ("CALL", "SSTORE", "LOG3")
    return ("CALL", "SSTORE")


_CLASS_LOCK = threading.Lock()
_CLASSES: Dict[str, Type[CustomBot]] = {}
_STATS = {"hits": 0, "misses": 0}


def _class_name(name: str, digest: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]", "_", name)[:40]
    return f"Custom_{slug}_{digest[:12]}"


def compile_blueprint(raw: Any) -> Type[CustomBot]:
    """Validate, assemble and return the bot class for ``raw`` (cached by digest, thread-safe)."""
    bp = normalize_blueprint(raw)
    text = canonical_json(bp)
    digest = hashlib.sha256(f"custom-bot:v{SCHEMA_VERSION}:{text}".encode()).hexdigest()
    with _CLASS_LOCK:
        cls = _CLASSES.get(digest)
        if cls is not None:
            _STATS["hits"] += 1
            return cls
    program = _assemble_normalized(bp)
    attrs = {
        "BLUEPRINT": text,
        "DIGEST": digest,
        "NAME": bp["name"],
        "TRIGGER_CODE": program.trigger_code,
        "SCRIPT_CODE": program.script_code,
        "TRIGGER_CLASS": trigger_class(bp["trigger"]),
        "OPTIMIZE": program.optimize,
        "MODEL": program.model,
        "PIPELINE": _freeze(bp["pipeline"]),
        "TRACING": _freeze(bp["tracing"]),
        "__module__": __name__,
        "__doc__": f"Compiled custom bot {bp['name']!r} ({digest[:12]}).",
    }
    new_cls = type(_class_name(bp["name"], digest), (CustomBot,), attrs)
    with _CLASS_LOCK:
        cls = _CLASSES.get(digest)
        if cls is not None:
            _STATS["hits"] += 1
            return cls
        if len(_CLASSES) >= CLASS_CACHE_LIMIT:
            _CLASSES.clear()
        _CLASSES[digest] = new_cls
        _STATS["misses"] += 1
        return new_cls


def spawn_custom_bot(raw: Any, overrides: Optional[Mapping[str, Any]] = None) -> CustomBot:
    return compile_blueprint(raw)(overrides)


def cache_info() -> Dict[str, int]:
    with _CLASS_LOCK:
        return {"classes": len(_CLASSES), "hits": _STATS["hits"], "misses": _STATS["misses"]}


def clear_class_cache() -> None:
    with _CLASS_LOCK:
        _CLASSES.clear()
        _STATS["hits"] = 0
        _STATS["misses"] = 0


def run_blueprint(raw: Any, seed: int = 0, victims: int = 8,
                  overrides: Optional[Mapping[str, Any]] = None) -> BotRun:
    """Compile ``raw`` and run it on the seeded scenario."""
    return spawn_custom_bot(raw, overrides).run(scenario_for(seed, victims))


def evaluate_blueprint(raw: Any, seeds: Sequence[int], victims: int = 8,
                       overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Score ``raw`` on several seeds; invalid blueprints score zero with ``error`` set."""
    try:
        bot = spawn_custom_bot(raw, overrides)
    except BlueprintError as exc:
        return {"name": raw.get("name") if isinstance(raw, dict) else None, "valid": False, "error": str(exc),
                "net_profit_base": 0, "fired": 0, "valid_plans": 0, "attempted": 0, "trigger_class": "",
                "digest": "", "runs": []}
    runs = [bot.run(scenario_for(s, victims)) for s in seeds]
    return {
        "name": bot.NAME,
        "valid": True,
        "error": "",
        "digest": bot.DIGEST,
        "trigger_class": bot.TRIGGER_CLASS,
        "net_profit_base": sum(r.net_profit_base for r in runs),
        "fired": sum(r.fired for r in runs),
        "attempted": sum(r.attempted for r in runs),
        "valid_plans": sum(len(r.valid) for r in runs),
        "model": bot.MODEL,
        "runs": [r.summary() for r in runs],
    }


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

TEMPLATES: Dict[str, Dict[str, Any]] = {
    "classic_sandwich": {
        "name": "classic_sandwich",
        "description": "front-run, victim, back-run on the victim pool",
        "trigger": {"all": [{"metric": "victim_slippage_bps", "op": ">=", "value": 30},
                            {"metric": "victim_size_ppm", "op": ">=", "value": 200}]},
        "script": [{"op": "frontrun", "size": {"optimize": [50, 200_000]}}, {"op": "victim"}, {"op": "backrun"}],
    },
    "multi_victim_sandwich": {
        "name": "multi_victim_sandwich",
        "description": "one front-run, the trigger victim and up to three more same-direction victims",
        "trigger": {"event": "Swap"},
        "script": [{"op": "frontrun", "size": {"optimize": [50, 200_000]}}, {"op": "victim"},
                   {"op": "victims", "max": 3}, {"op": "backrun"}],
    },
    "jit_liquidity": {
        "name": "jit_liquidity",
        "description": "narrow mint before a large concentrated-liquidity swap, burn after",
        "tracing": {"opcodes": True},
        "trigger": {"all": [{"event": "LargeSwap", "min_ppm": 500},
                            {"metric": "pool_version", "op": "in", "value": [3, 4]}]},
        "script": [{"op": "mint", "half_width": 2, "pct": 400}, {"op": "victim"}, {"op": "burn"}],
    },
    "backrun_arbitrage": {
        "name": "backrun_arbitrage",
        "description": "let the victim move the pool, then close the gap through a cycle",
        "trigger": {"event": "PriceDeviation", "min_bps": 0},
        "script": [{"op": "victim"}, {"op": "arb", "size": {"optimize": [10, 50_000]}}],
    },
    "transient_sandwich": {
        "name": "transient_sandwich",
        "description": "front-run output kept in transient storage and sold back by exact amount",
        "trigger": {"metric": "victim_slippage_bps", "op": ">=", "value": 20},
        "script": [{"op": "frontrun", "size": {"optimize": [50, 200_000]}},
                   {"op": "tstore", "key": "front_out", "value": {"last_out": True}},
                   {"op": "victim"},
                   {"op": "swap", "pool": "$victim", "side": "$counter", "size": {"tload": "front_out"}}],
    },
    "composite_bridge_bundle": {
        "name": "composite_bridge_bundle",
        "description": "multi-pool sandwich + guarded cycle arb + transient state swap + delayed cross-chain "
                       "arbitrage, in one simulated bundle",
        "tracing": {"opcodes": True, "sequential": False, "record": True},
        "trigger": {"all": [{"metric": "victim_slippage_bps", "op": ">=", "value": 10},
                            {"any": [{"event": "LargeSwap", "min_ppm": 300},
                                     {"event": "PriceDeviation", "min_bps": 5}]},
                            {"not": {"opcode": "SSTORE", "op": ">", "value": 60}}]},
        "script": [
            {"op": "frontrun", "size": {"optimize": [50, 200_000]}},
            {"op": "tstore", "key": "front_out", "value": {"last_out": True}},
            {"op": "victim"},
            {"op": "victims", "max": 2},
            {"op": "swap", "pool": "$victim", "side": "$counter", "size": {"tload": "front_out"}},
            {"op": "atomic", "on_revert": "rollback", "body": [{"op": "arb", "size": 2_000}]},
            {"op": "tswap", "a": "front_out", "b": "spare"},
            {"op": "if", "cond": {"balance": "$base", "op": ">", "value": 0}, "then": [
                {"op": "atomic", "on_revert": "rollback", "body": [
                    {"op": "bridge", "chain": "l2", "token": "$base", "amount_ppm": 250_000, "fee_bps": 5,
                     "delay_blocks": 2, "body": [{"op": "arb", "pool": 0, "size": 1_500}]}]}]},
        ],
    },
}

KNOWN_TRIGGER_CLASSES: frozenset = frozenset(trigger_class(normalize_blueprint(t)["trigger"])
                                             for t in TEMPLATES.values())


def template(name: str) -> Dict[str, Any]:
    """A deep copy of a built-in template (edit freely)."""
    try:
        return copy.deepcopy(TEMPLATES[name])
    except KeyError:
        raise BlueprintError([("$", f"unknown template {name!r} (one of {', '.join(TEMPLATES)})")]) from None


# ---------------------------------------------------------------------------
# Genes -> blueprint (the AI's design space)
# ---------------------------------------------------------------------------

N_BLUEPRINT_GENES = 24
EVENT_CHOICES: Tuple[Optional[str], ...] = (None, "LargeSwap", "PriceDeviation", "LoanAtRisk", "TickCross")
CORE_KINDS: Tuple[str, ...] = ("sandwich", "multi_victim", "jit", "backrun_arb", "transient_sandwich")


def _pick(g: float, n: int) -> int:
    return min(n - 1, max(0, math.floor(g * n)))


def _lin(g: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, round(lo + g * (hi - lo))))


def _logq(g: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, round(math.exp(math.log(lo) + g * (math.log(hi) - math.log(lo))))))


def blueprint_from_genes(genes: Sequence[float], name: str = "ai-custom") -> Dict[str, Any]:
    """Decode :data:`N_BLUEPRINT_GENES` genes in ``[0, 1]`` into a valid blueprint."""
    g = [min(1.0, max(0.0, float(v))) if math.isfinite(float(v)) else 0.5 for v in genes]
    if len(g) != N_BLUEPRINT_GENES:
        raise BlueprintError([("$genes", f"need {N_BLUEPRINT_GENES} genes, got {len(g)}")])
    nodes: List[Dict[str, Any]] = []
    if g[1] >= 0.5:
        nodes.append({"metric": "victim_priority_fee_wei", "op": ">=", "value": _logq(g[2], 10**7, 5 * 10**9)})
    if g[3] >= 0.5:
        nodes.append({"metric": "victim_slippage_bps", "op": ">=", "value": _lin(g[4], 0, 800)})
    if g[5] >= 0.5:
        nodes.append({"metric": "victim_size_ppm", "op": ">=", "value": _logq(g[6], 10, 30_000)})
    event = EVENT_CHOICES[_pick(g[7], len(EVENT_CHOICES))]
    if event == "LargeSwap":
        nodes.append({"event": event, "min_ppm": _logq(g[8], 10, 30_000)})
    elif event == "PriceDeviation":
        nodes.append({"event": event, "min_bps": _lin(g[8], 0, 200)})
    elif event == "LoanAtRisk":
        nodes.append({"event": event, "max_health_bps": _lin(g[8], 9_500, 11_000)})
    elif event == "TickCross":
        nodes.append({"event": event, "min_ticks": _lin(g[8], 0, 4)})
    op_choice = _pick(g[9], 5)
    opnode: Optional[Dict[str, Any]] = None
    if op_choice == 1:
        opnode = {"opcode": "SSTORE", "op": ">=", "value": _lin(g[10], 2, 12)}
    elif op_choice == 2:
        opnode = {"opcode": "TSTORE", "op": ">=", "value": _lin(g[10], 1, 4)}
    elif op_choice == 3:
        opnode = {"opcode_seq": ["CALL", "SSTORE"]}
    elif op_choice == 4:
        opnode = {"opcode_seq": ["TSTORE", "CALL"]}
    if opnode is not None:
        nodes.append({"not": opnode} if g[11] >= 0.5 else opnode)
    if not nodes:
        trigger: Dict[str, Any] = {"const": True}
    elif len(nodes) == 1:
        trigger = nodes[0]
    else:
        trigger = {"all" if g[0] < 0.5 else "any": nodes}

    kind = CORE_KINDS[_pick(g[12], len(CORE_KINDS))]
    lo = _logq(g[13], 50, 20_000)
    size: Any = {"optimize": [lo, min(PPM, lo * 16)]} if g[14] < 0.5 else lo
    if kind == "sandwich":
        script: List[Dict[str, Any]] = [{"op": "frontrun", "size": size}, {"op": "victim"}, {"op": "backrun"}]
    elif kind == "multi_victim":
        script = [{"op": "frontrun", "size": size}, {"op": "victim"}, {"op": "victims", "max": _lin(g[15], 1, 4)},
                  {"op": "backrun"}]
    elif kind == "jit":
        script = [{"op": "mint", "half_width": _lin(g[15], 1, 8), "pct": _logq(g[13], 20, 5_000)},
                  {"op": "victim"}, {"op": "burn"}]
    elif kind == "backrun_arb":
        script = [{"op": "victim"}, {"op": "arb", "size": size}]
    else:
        script = [{"op": "frontrun", "size": size}, {"op": "tstore", "key": "front_out", "value": {"last_out": True}},
                  {"op": "victim"},
                  {"op": "swap", "pool": "$victim", "side": "$counter", "size": {"tload": "front_out"}}]
    addons: List[Dict[str, Any]] = []
    if g[16] >= 0.6 and kind != "backrun_arb":
        addons.append({"op": "arb", "size": _logq(g[17], 100, 20_000)})
    if g[18] >= 0.6:
        addons.append({"op": "bridge", "chain": "l2", "token": "$base", "amount_ppm": _lin(g[17], 50_000, PPM),
                       "fee_bps": _lin(g[20], 0, 50), "delay_blocks": _lin(g[19], 0, 8),
                       "body": [{"op": "arb", "pool": _pick(g[15], 4), "size": _logq(g[17], 100, 20_000)}]})
    if addons:
        if g[21] >= 0.5:
            script.append({"op": "atomic", "on_revert": "rollback", "body": addons})
        else:
            script.extend(addons)
    pipeline = {"on_revert": "skip" if g[22] >= 0.7 else "abort", "max_attempts": _lin(g[23], 2, 6),
                "optimize_grid": 6, "optimize_refine": 6}
    tracing = {"opcodes": opnode is not None, "sequential": g[11] < 0.25 and opnode is not None, "record": False}
    return {"schema": SCHEMA_VERSION, "name": name, "description": f"synthesised {kind}", "trigger": trigger,
            "tracing": tracing, "script": script, "pipeline": pipeline}


# ---------------------------------------------------------------------------
# AI synthesis on the worker pool
# ---------------------------------------------------------------------------


def _evaluate_task(task: Tuple[Tuple[str, ...], Tuple[int, ...], int]) -> List[Dict[str, Any]]:
    """Worker task: compile and score a fixed batch of blueprints (canonical JSON strings)."""
    texts, seeds, victims = task
    out = []
    for text in texts:
        row = evaluate_blueprint(json.loads(text), seeds, victims)
        row.pop("runs", None)
        out.append(row)
    return out


def fitness(row: Mapping[str, Any]) -> Tuple[int, int, int]:
    return (int(row.get("net_profit_base", 0)), int(row.get("valid_plans", 0)), int(row.get("fired", 0)))


@dataclass
class SynthesisReport:
    seed: int
    bots: int
    generations: List[Dict[str, Any]]
    best: Optional[Dict[str, Any]]
    novel_classes: List[Dict[str, Any]]
    invariants: Dict[str, Any]
    engine: Dict[str, Any]
    tasks: int
    features: List[float] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "status": "experimental",
            "seed": self.seed,
            "bots_per_generation": self.bots,
            "generations": self.generations,
            "tasks": self.tasks,
            "best": self.best,
            "novel_trigger_classes": self.novel_classes,
            "novel_trigger_class_count": len(self.novel_classes),
            "known_trigger_classes": sorted(KNOWN_TRIGGER_CLASSES),
            "engine": self.engine,
            "features": self.features,
            "invariants": self.invariants,
        }


class BlueprintSynthesizer:
    """NumPy generator of custom bot blueprints, trained on its own best designs.

    Each generation: ``x = [features, latent]`` -> network -> mutated genes ->
    :func:`blueprint_from_genes` -> compiled and scored on ``scenario_seeds``
    seeded scenarios in batches of ``batch`` (one worker task each, so results
    never depend on the worker count) -> the top ``elite_frac`` are the
    training targets for the next generation.
    """

    def __init__(self, bots: int = 64, generations: int = 3, seed: int = 0, features: Any = None,
                 latent_dim: int = 8, hidden: Sequence[int] = (32,), victims: int = 8, scenario_seeds: int = 2,
                 batch: int = 4, elite_frac: float = 0.25, epochs: int = 30, lr: float = 0.02) -> None:
        if bots < 1 or generations < 1 or batch < 1 or scenario_seeds < 1 or victims < 1:
            raise ConfigError("bots, generations, batch, scenario_seeds and victims must be >= 1")
        if latent_dim < 1 or epochs < 0 or not 0 < elite_frac <= 1:
            raise ConfigError("latent_dim >= 1, epochs >= 0 and elite_frac within (0, 1] are required")
        from .ai_agent_factory import AdamOptimizer, AdaptiveMutation, NeuralMatrixEngine, _seed_int
        import numpy as np

        self.np = np
        self.bots = bots
        self.generations = generations
        self.seed = seed
        self.victims = victims
        self.batch = batch
        self.elite_frac = elite_frac
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.features = np.zeros(0) if features is None else np.asarray(features, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(self.features)):
            raise ConfigError("features must be finite")
        self.scenario_seeds = tuple(seed * 1_000 + 700 + i for i in range(scenario_seeds))
        self.engine = NeuralMatrixEngine((len(self.features) + latent_dim, *hidden, N_BLUEPRINT_GENES),
                                         "tanh", "sigmoid", "xavier",
                                         seed=_seed_int("custom-engine", seed) % (1 << 63),
                                         optimizer=AdamOptimizer(lr=lr))
        self.mutation = AdaptiveMutation()

    def _generator(self, *parts: Any) -> Any:
        from .ai_agent_factory import seeded_generator

        return seeded_generator("custom-synth", self.seed, *parts)

    def design(self, generation: int) -> Tuple[Any, Any, List[Dict[str, Any]]]:
        """Inputs, genes and blueprints of one generation (deterministic in seed and generation)."""
        np = self.np
        rng = self._generator(generation)
        latents = rng.standard_normal((self.bots, self.latent_dim))
        x = np.hstack([np.tile(self.features, (self.bots, 1)), latents])
        genes = self.mutation.mutate_genes(self.engine.forward(x), rng)
        bps = [blueprint_from_genes(genes[i], f"ai-custom-g{generation}-{i}") for i in range(self.bots)]
        return x, genes, bps

    def evaluate(self, blueprints: Sequence[Mapping[str, Any]], executor: Any = None) -> Tuple[List[Dict[str, Any]], int]:
        texts = [canonical_json(normalize_blueprint(b)) for b in blueprints]
        tasks = [(tuple(texts[i:i + self.batch]), self.scenario_seeds, self.victims)
                 for i in range(0, len(texts), self.batch)]
        if executor is not None:
            parts = executor.map(_evaluate_task, tasks, chunksize=1)
        else:
            parts = [_evaluate_task(t) for t in tasks]
        return [row for part in parts for row in part], len(tasks)

    def run(self, executor: Any = None) -> SynthesisReport:
        np = self.np
        seen = set(KNOWN_TRIGGER_CLASSES)
        novel: List[Dict[str, Any]] = []
        gens: List[Dict[str, Any]] = []
        best: Optional[Tuple[Tuple[int, int, int], Dict[str, Any], Dict[str, Any]]] = None
        tasks = 0
        genes_ok = True
        for gen in range(self.generations):
            x, genes, bps = self.design(gen)
            genes_ok = genes_ok and bool(np.all((genes >= 0) & (genes <= 1)))
            rows, n_tasks = self.evaluate(bps, executor)
            tasks += n_tasks
            keys = [fitness(r) for r in rows]
            order = sorted(range(len(rows)), key=lambda i: (keys[i], -i), reverse=True)
            n_elite = max(1, math.ceil(self.elite_frac * len(rows)))
            elite = order[:n_elite]
            losses = self.engine.fit(x[elite], genes[elite], epochs=self.epochs) if self.epochs else [0.0]
            profitable = sum(1 for r in rows if r["net_profit_base"] > 0)
            self.mutation.update(profitable / len(rows))
            new_here = []
            classes = Counter(r["trigger_class"] for r in rows if r["valid"])
            for i in range(len(rows)):
                cls = rows[i]["trigger_class"]
                if rows[i]["valid"] and cls not in seen:
                    seen.add(cls)
                    entry = {"trigger_class": cls, "generation": gen, "first_bot": rows[i]["name"],
                             "net_profit_base": rows[i]["net_profit_base"]}
                    novel.append(entry)
                    new_here.append(cls)
            for e in novel:
                for r in rows:
                    if r["valid"] and r["trigger_class"] == e["trigger_class"]:
                        e["net_profit_base"] = max(e["net_profit_base"], r["net_profit_base"])
            top = order[0]
            if best is None or keys[top] > best[0]:
                best = (keys[top], rows[top], bps[top])
            profits = sorted(r["net_profit_base"] for r in rows)
            gens.append({
                "generation": gen,
                "bots": len(rows),
                "valid": sum(1 for r in rows if r["valid"]),
                "profitable": profitable,
                "fired": sum(r["fired"] for r in rows),
                "best_net_profit_base": profits[-1],
                "median_net_profit_base": profits[len(profits) // 2],
                "trigger_classes": len(classes),
                "new_trigger_classes": new_here,
                "loss_before": round(float(losses[0]), 9),
                "loss_after": round(float(losses[-1]), 9),
                "sigma": round(self.mutation.sigma, 9),
            })
        assert best is not None
        replay = evaluate_blueprint(best[2], self.scenario_seeds, self.victims)
        invariants = {
            "genes_in_unit_interval": genes_ok,
            "engine_finite": bool(self.engine.is_finite()),
            "best_replays_to_the_wei": replay["net_profit_base"] == best[1]["net_profit_base"],
            "best_digest_stable": replay["digest"] == best[1]["digest"],
        }
        invariants["ok"] = all(invariants.values())
        best_out = {k: v for k, v in best[1].items() if k != "runs"}
        best_out["blueprint"] = normalize_blueprint(best[2])
        return SynthesisReport(self.seed, self.bots, gens, best_out, novel, invariants, self.engine.describe(),
                               tasks, [round(float(f), 9) for f in self.features])
