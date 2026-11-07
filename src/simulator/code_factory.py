"""Layer 15: autonomous on-disk code factory and offline self-optimisation loop.

Status: experimental. Every bot this module writes runs only inside the
simulator: on the arena's in-memory Uniswap v2 pools, through the Layer 04
greedy block builder, with exact integer P&L. The generated files never
import anything that touches a network, keys or a real chain, and the
loader refuses any file that tries to.

What it does
------------
The Layer 11 generator (:class:`~simulator.ai_agent_factory.NumPy MLP
<simulator.ai_agent_factory.NeuralMatrixEngine>`, conditioned on the RL
features) produces gene vectors, which decode into
:class:`~simulator.ai_agent_factory.BotProfile` strategies. The code factory
turns each profile into **source code on disk**:

* ``bot_NNNN.py`` - an independent Python module: a docstring, one import
  (``BotProfile`` and ``SynthesizedBot`` from :mod:`simulator.ai_agent_factory`),
  literal constants (genes, fingerprint, profile fields, fitness), one
  ``SynthesizedBot`` subclass and a ``build()`` function;
* ``bot_NNNN.sol`` - a Solidity ``library`` holding the same strategy as
  ``uint256`` / ``bytes8`` constants. It is a **parameter record** with no
  functions, so it carries no executable logic; the factory checks it with
  its own structural validator (no ``solc`` is available offline) and parses
  it back to prove it matches the Python file to the last integer.

Every write is atomic (temporary file in the same directory, ``fsync``,
``os.replace``) and recorded with its SHA-256 in ``manifest.json``.

Then the loop deploys **from disk**: each arena heat is a worker task that
receives only file paths and digests, re-reads the files, re-checks the
digest and the AST allow-list, compiles them, instantiates the generated
classes and runs :class:`~simulator.ai_agent_factory.BotArena` (the Layer 04
builder decides who lands). Profits come back as exact wei; the factory
ranks the bots, trains the generator on the elite (weighted by offspring),
adapts the mutation step, and **rewrites the bot files** with the next
generation. The best bot of every generation is also kept in ``champions/``.

Code safety (the loader's allow-list)
-------------------------------------
:func:`python_source_issues` accepts a module only when it has exactly the
generated shape: a docstring, ``from simulator.ai_agent_factory import
BotProfile, SynthesizedBot``, assignments of literal values to the known
constant names, one class whose only base is ``SynthesizedBot`` and whose
only statement is ``PROFILE = BotProfile(**PROFILE_FIELDS)``, and a
``build()`` that returns an instance of that class. No other import, call,
attribute access, loop, decorator or statement is allowed, so a tampered
file cannot run code when it is loaded. The digest check runs first; the
source that was checked is the source that is compiled (it is never read
twice).

Block builder integration
-------------------------
:func:`load_manifest_bots` returns verified bot instances from a factory
directory; :func:`simulator.block_builder.generated_competitors` and
:func:`simulator.block_builder.run_generated_competition` use it to put the
generated bots straight into a builder competition.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import threading
import time
import types
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import i18n
from .ai_agent_factory import (
    ARENA_POOLS,
    HOP_POLICIES,
    N_GENES,
    STRATEGY_VALUES,
    AgentFactory,
    ArenaConfig,
    BotArena,
    BotProfile,
    Ledger,
    Population,
    SynthesizedBot,
    allocate_offspring,
    class_name,
    decode_profile,
    fitness_key,
    lineage,
    profile_issues,
    rank_order,
)
from .config import ConfigError
from .parallel import WorkerPool

CODEGEN_VERSION = 1
BOT_DIR_ENV = "MEV_SIM_BOT_DIR"
DEFAULT_BOT_DIR = Path(__file__).resolve().parent / "generated_bots"
MANIFEST_NAME = "manifest.json"
CHAMPIONS_DIR = "champions"
IMPORT_MODULE = "simulator.ai_agent_factory"
IMPORT_NAMES: Tuple[str, ...] = ("BotProfile", "SynthesizedBot")
PY_CONSTANTS: Tuple[str, ...] = ("CODEGEN_VERSION", "GENERATION", "FINGERPRINT", "GENES", "FITNESS",
                                 "PROFILE_FIELDS")
PROFILE_FIELD_NAMES: Tuple[str, ...] = tuple(f.name for f in fields(BotProfile))
SOL_PRAGMA = "pragma solidity ^0.8.20;"
SOL_SPDX = "// SPDX-License-Identifier: MIT"
UINT256_MAX = (1 << 256) - 1
BOT_FILE = re.compile(r"^bot_(\d{4})\.(py|sol)$")
GENERATED_PACKAGE = "simulator_generated"
MAX_SOURCE_BYTES = 256 * 1024

FACTORY_CODE_RULES: Tuple[str, ...] = (
    "factory.py_allowlist",
    "factory.py_compiles",
    "factory.py_roundtrip",
    "factory.sol_allowlist",
    "factory.sol_roundtrip",
    "factory.digest",
    "factory.disk_origin",
    "factory.heat_coverage",
    "factory.finite_weights",
    "factory.offspring",
)

# Integer profile fields stored as Solidity constants (strategy, hop policy and pools are encoded separately).
SOL_INT_FIELDS: Tuple[str, ...] = ("share_bps", "step_bps", "min_share_bps", "max_share_bps", "tip_min_wei",
                                   "tip_max_wei", "tip_step_bps", "size_bps", "max_impact_bps", "slippage_floor_bps",
                                   "max_hops", "bundle_slots", "front_slack_bps", "latency_ms", "jitter_ms",
                                   "capital_wei")


class CodeRejected(ConfigError):
    """A generated file failed the allow-list, the digest check or a round-trip check."""


def resolve_bot_dir(path: Optional[Any] = None) -> Path:
    """``path``, else ``$MEV_SIM_BOT_DIR``, else ``src/simulator/generated_bots``."""
    if path:
        return Path(path)
    env = os.environ.get(BOT_DIR_ENV)
    return Path(env) if env else DEFAULT_BOT_DIR


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def profile_fields(profile: BotProfile) -> Dict[str, Any]:
    """The profile as constructor keyword arguments (``target_pools`` stays a tuple)."""
    return {name: getattr(profile, name) for name in PROFILE_FIELD_NAMES}


def _literal(value: Any) -> str:
    """Python literal text that ``ast.literal_eval`` turns back into exactly ``value``."""
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CodeRejected(f"cannot render non-finite float {value!r}")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, tuple):
        inner = ", ".join(_literal(v) for v in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if isinstance(value, list):
        return "[" + ", ".join(_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_literal(k)}: {_literal(v)}" for k, v in value.items()) + "}"
    raise CodeRejected(f"cannot render {type(value).__name__} as a literal")


def _docstring(text: str) -> str:
    safe = text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    return f'"""{safe}"""'


def render_python(profile: BotProfile, genes: Sequence[float], fitness: Optional[Mapping[str, int]] = None,
                  lang: Optional[str] = None) -> str:
    """Source of an independent bot module for ``profile`` (deterministic: same inputs, same bytes)."""
    if profile_issues(profile):
        raise CodeRejected(f"invalid profile {profile.name!r}: {profile_issues(profile)}")
    g = [float(x) for x in genes]
    if len(g) != N_GENES:
        raise CodeRejected(f"expected {N_GENES} genes, got {len(g)}")
    cls = class_name(profile)
    header = i18n.t("factory.py_header", lang, name=profile.name, generation=profile.generation)
    fit = {k: int(v) for k, v in fitness.items()} if fitness else None
    lines = [
        "# -*- coding: utf-8 -*-",
        _docstring(f"{header}\n\n{profile.describe()}\n"),
        "",
        f"from {IMPORT_MODULE} import {', '.join(IMPORT_NAMES)}",
        "",
        f"CODEGEN_VERSION = {CODEGEN_VERSION}",
        f"GENERATION = {int(profile.generation)}",
        f"FINGERPRINT = {_literal(profile.fingerprint())}",
        f"GENES = {_literal(tuple(g))}",
        f"FITNESS = {_literal(fit)}",
        "PROFILE_FIELDS = {",
    ]
    for name, value in profile_fields(profile).items():
        lines.append(f"    {_literal(name)}: {_literal(value)},")
    lines += [
        "}",
        "",
        "",
        f"class {cls}(SynthesizedBot):",
        f"    {_docstring(profile.describe())}",
        "",
        "    PROFILE = BotProfile(**PROFILE_FIELDS)",
        "",
        "",
        "def build():",
        f"    return {cls}()",
        "",
    ]
    return "\n".join(lines)


def _sol_name(field_name: str) -> str:
    return field_name.upper()


def solidity_constants(profile: BotProfile) -> Dict[str, int]:
    """Every constant of the Solidity parameter record (all non-negative integers)."""
    out: Dict[str, int] = {"CODEGEN_VERSION": CODEGEN_VERSION, "GENERATION": int(profile.generation),
                           "STRATEGY": STRATEGY_VALUES.index(profile.strategy),
                           "HOP_POLICY": HOP_POLICIES.index(profile.hop_policy),
                           "TARGET_POOLS_MASK": sum(1 << ARENA_POOLS.index(p) for p in profile.target_pools)}
    for name in SOL_INT_FIELDS:
        out[_sol_name(name)] = int(getattr(profile, name))
    return out


def render_solidity(profile: BotProfile, fitness: Optional[Mapping[str, int]] = None, lang: Optional[str] = None) -> str:
    """A Solidity ``library`` of constants mirroring ``profile`` (no functions, no external calls)."""
    if profile_issues(profile):
        raise CodeRejected(f"invalid profile {profile.name!r}: {profile_issues(profile)}")
    fp = profile.fingerprint()
    lines = [SOL_SPDX,
             "// " + i18n.t("factory.sol_header", lang, name=profile.name, generation=profile.generation)]
    if fitness:
        lines.append("// " + i18n.t("factory.fitness", lang, pnl=int(fitness.get("pnl_wei", 0)),
                                    wins=int(fitness.get("won", 0))))
    lines += [SOL_PRAGMA, "", f"library SimBot_{fp} {{"]
    lines.append(f"    bytes8 internal constant FINGERPRINT = 0x{fp};")
    for name, value in solidity_constants(profile).items():
        lines.append(f"    uint256 internal constant {name} = {value};")
    lines += ["}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def _profile_assign_ok(node: ast.stmt) -> bool:
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "PROFILE"):
        return False
    call = node.value
    return (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "BotProfile"
            and not call.args and len(call.keywords) == 1 and call.keywords[0].arg is None
            and isinstance(call.keywords[0].value, ast.Name) and call.keywords[0].value.id == "PROFILE_FIELDS")


def python_source_issues(source: str, filename: str = "<generated>") -> List[str]:
    """Every way ``source`` departs from the generated-bot shape (empty = safe to compile and load)."""
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return [f"source larger than {MAX_SOURCE_BYTES} bytes"]
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [f"syntax error at line {exc.lineno}: {exc.msg}"]
    issues: List[str] = []
    constants: Dict[str, Any] = {}
    imports = classes = builders = 0
    cls_name: Optional[str] = None
    body = list(tree.body)
    if body and _is_docstring(body[0]):
        body = body[1:]
    for node in body:
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.ImportFrom):
            imports += 1
            names = [a.name for a in node.names]
            if node.module != IMPORT_MODULE or node.level != 0:
                issues.append(f"line {line}: import from {node.module!r} is not allowed")
            if any(a.asname for a in node.names) or not set(names) <= set(IMPORT_NAMES) or len(set(names)) != len(names):
                issues.append(f"line {line}: only {', '.join(IMPORT_NAMES)} may be imported, without aliases")
        elif isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                issues.append(f"line {line}: only simple NAME = literal assignments are allowed")
                continue
            name = node.targets[0].id
            if name not in PY_CONSTANTS:
                issues.append(f"line {line}: unknown constant {name!r}")
                continue
            if name in constants:
                issues.append(f"line {line}: {name} assigned twice")
                continue
            try:
                constants[name] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                issues.append(f"line {line}: {name} is not a literal")
        elif isinstance(node, ast.ClassDef):
            classes += 1
            cls_name = node.name
            if node.decorator_list or node.keywords or len(node.bases) != 1 \
                    or not isinstance(node.bases[0], ast.Name) or node.bases[0].id != "SynthesizedBot":
                issues.append(f"line {line}: the class must subclass SynthesizedBot only, without decorators")
            stmts = list(node.body)
            if stmts and _is_docstring(stmts[0]):
                stmts = stmts[1:]
            if len(stmts) != 1 or not _profile_assign_ok(stmts[0]):
                issues.append(f"line {line}: the class body must be PROFILE = BotProfile(**PROFILE_FIELDS)")
        elif isinstance(node, ast.FunctionDef):
            builders += 1
            a = node.args
            ret = node.body[0] if len(node.body) == 1 else None
            ok = (node.name == "build" and not node.decorator_list and node.returns is None
                  and not (a.args or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg or a.defaults)
                  and isinstance(ret, ast.Return) and isinstance(ret.value, ast.Call)
                  and isinstance(ret.value.func, ast.Name) and not ret.value.args and not ret.value.keywords)
            if not ok:
                issues.append(f"line {line}: only 'def build(): return <BotClass>()' is allowed")
            elif cls_name is None or ret.value.func.id != cls_name:  # type: ignore[union-attr]
                issues.append(f"line {line}: build() must return the bot class defined above it")
        else:
            issues.append(f"line {line}: {type(node).__name__} statements are not allowed")
    if imports != 1:
        issues.append(f"expected exactly one import, found {imports}")
    if classes != 1:
        issues.append(f"expected exactly one bot class, found {classes}")
    if builders != 1:
        issues.append(f"expected exactly one build() function, found {builders}")
    missing = [c for c in PY_CONSTANTS if c not in constants]
    if missing:
        issues.append(f"missing constants: {', '.join(missing)}")
    if issues:
        return issues
    if constants["CODEGEN_VERSION"] != CODEGEN_VERSION:
        issues.append(f"codegen version {constants['CODEGEN_VERSION']!r} != {CODEGEN_VERSION}")
    pf = constants["PROFILE_FIELDS"]
    if not isinstance(pf, dict) or set(pf) != set(PROFILE_FIELD_NAMES):
        issues.append("PROFILE_FIELDS must name every BotProfile field exactly once")
        return issues
    try:
        profile = BotProfile(**pf)
    except (ConfigError, TypeError) as exc:
        return issues + [f"PROFILE_FIELDS is not a valid profile: {exc}"]
    if profile.fingerprint() != constants["FINGERPRINT"]:
        issues.append("FINGERPRINT does not match PROFILE_FIELDS")
    if profile.generation != constants["GENERATION"]:
        issues.append("GENERATION does not match PROFILE_FIELDS")
    genes = constants["GENES"]
    if not isinstance(genes, tuple) or len(genes) != N_GENES or not all(
            isinstance(x, float) and math.isfinite(x) for x in genes):
        issues.append(f"GENES must be a tuple of {N_GENES} finite floats")
    fit = constants["FITNESS"]
    if fit is not None and (not isinstance(fit, dict) or not all(isinstance(k, str) and type(v) is int
                                                                   for k, v in fit.items())):
        issues.append("FITNESS must be None or a dict of integers")
    if cls_name != class_name(profile):
        issues.append(f"class name {cls_name!r} != {class_name(profile)!r}")
    return issues


_SOL_LINE = re.compile(r"^    uint256 internal constant ([A-Z][A-Z0-9_]*) = ([0-9]{1,78});$")
_SOL_FP = re.compile(r"^    bytes8 internal constant FINGERPRINT = 0x([0-9a-f]{16});$")
_SOL_LIB = re.compile(r"^library SimBot_([0-9a-f]{16}) \{$")
_SOL_FORBIDDEN = ("function", "call", "delegatecall", "assembly", "selfdestruct", "address", "external", "public",
                  "payable", "import", "contract", "modifier", "fallback", "receive", "storage")


def solidity_source_issues(source: str) -> Tuple[List[str], Dict[str, int]]:
    """Check the parameter-record shape line by line; returns ``(issues, constants)``."""
    issues: List[str] = []
    consts: Dict[str, int] = {}
    lines = source.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines or lines[0] != SOL_SPDX:
        issues.append("first line must be the SPDX licence identifier")
    state = "head"
    lib_fp = fp = None
    for n, line in enumerate(lines[1:], start=2):
        if state == "head":
            if line.startswith("//"):
                continue
            if line == SOL_PRAGMA:
                state = "pragma"
                continue
            issues.append(f"line {n}: expected comments then {SOL_PRAGMA!r}")
            break
        if state == "pragma":
            if line == "":
                continue
            m = _SOL_LIB.match(line)
            if not m:
                issues.append(f"line {n}: expected 'library SimBot_<fingerprint> {{'")
                break
            lib_fp, state = m.group(1), "body"
            continue
        if state == "body":
            if line == "}":
                state = "closed"
                continue
            m = _SOL_FP.match(line)
            if m:
                if fp is not None:
                    issues.append(f"line {n}: FINGERPRINT declared twice")
                fp = m.group(1)
                continue
            m = _SOL_LINE.match(line)
            if not m:
                lowered = line.lower()
                bad = [w for w in _SOL_FORBIDDEN if re.search(rf"\b{w}\b", lowered)]
                issues.append(f"line {n}: not a constant declaration" + (f" (uses {', '.join(bad)})" if bad else ""))
                continue
            name, value = m.group(1), int(m.group(2))
            if name in consts:
                issues.append(f"line {n}: {name} declared twice")
            if value > UINT256_MAX:
                issues.append(f"line {n}: {name} overflows uint256")
            consts[name] = value
            continue
        if line.strip():
            issues.append(f"line {n}: text after the closing brace")
    if state != "closed" and not issues:
        issues.append("the library is never closed")
    if lib_fp is not None and fp is not None and lib_fp != fp:
        issues.append("library name and FINGERPRINT disagree")
    if fp is None and not issues:
        issues.append("FINGERPRINT is missing")
    if fp is not None:
        consts["__fingerprint__"] = int(fp, 16)
    return issues, consts


def profile_from_solidity(consts: Mapping[str, int], name: str) -> BotProfile:
    """Rebuild a profile from a parameter record (raises CodeRejected when a constant is missing or invalid)."""
    need = ["GENERATION", "STRATEGY", "HOP_POLICY", "TARGET_POOLS_MASK"] + [_sol_name(f) for f in SOL_INT_FIELDS]
    missing = [k for k in need if k not in consts]
    if missing:
        raise CodeRejected(f"Solidity record lacks {', '.join(missing)}")
    if not consts["STRATEGY"] < len(STRATEGY_VALUES) or not consts["HOP_POLICY"] < len(HOP_POLICIES):
        raise CodeRejected("strategy or hop policy index out of range")
    mask = consts["TARGET_POOLS_MASK"]
    if not 0 < mask < (1 << len(ARENA_POOLS)):
        raise CodeRejected("target pool mask out of range")
    kwargs: Dict[str, Any] = {_f: consts[_sol_name(_f)] for _f in SOL_INT_FIELDS}
    try:
        return BotProfile(name=name, strategy=STRATEGY_VALUES[consts["STRATEGY"]],
                          hop_policy=HOP_POLICIES[consts["HOP_POLICY"]],
                          target_pools=tuple(p for i, p in enumerate(ARENA_POOLS) if mask >> i & 1),
                          generation=consts["GENERATION"], **kwargs)
    except ConfigError as exc:
        raise CodeRejected(f"Solidity record is not a valid profile: {exc}") from exc


# ---------------------------------------------------------------------------
# Disk I/O and loading
# ---------------------------------------------------------------------------


def write_atomic(path: Path, text: str) -> str:
    """Write ``text`` (UTF-8) to ``path`` via temp file + fsync + replace; returns its SHA-256."""
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class LoadedBot:
    path: str
    digest: str
    module_name: str
    cls: type
    profile: BotProfile
    genes: Tuple[float, ...]
    fitness: Optional[Dict[str, int]]

    def spawn(self) -> SynthesizedBot:
        return self.cls()


def load_bot(path: Any, expected_digest: Optional[str] = None) -> LoadedBot:
    """Read, digest-check, allow-list-check, compile and execute one generated module (never imported twice).

    The bytes that are hashed are the bytes that are checked and compiled.
    """
    p = Path(path)
    raw = p.read_bytes()
    if len(raw) > MAX_SOURCE_BYTES:
        raise CodeRejected(f"{p.name}: larger than {MAX_SOURCE_BYTES} bytes")
    digest = hashlib.sha256(raw).hexdigest()
    if expected_digest is not None and digest != expected_digest:
        raise CodeRejected(f"{p.name}: digest {digest[:12]} does not match the manifest ({expected_digest[:12]})")
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CodeRejected(f"{p.name}: not UTF-8 ({exc})") from exc
    issues = python_source_issues(source, str(p))
    if issues:
        raise CodeRejected(f"{p.name}: " + "; ".join(issues))
    code = compile(source, str(p), "exec", dont_inherit=True)
    module_name = f"{GENERATED_PACKAGE}.{p.stem}_{digest[:12]}"
    module = types.ModuleType(module_name)
    module.__file__ = str(p)
    exec(code, module.__dict__)  # the allow-list above admits only literals, one import, one class, build()
    cls = None
    for value in module.__dict__.values():
        if isinstance(value, type) and issubclass(value, SynthesizedBot) and value is not SynthesizedBot:
            cls = value
    if cls is None:
        raise CodeRejected(f"{p.name}: no bot class after loading")
    cls.__module__ = module_name
    bot = module.build()
    if type(bot) is not cls or bot.profile != cls.PROFILE:
        raise CodeRejected(f"{p.name}: build() did not return the bot class")
    return LoadedBot(str(p), digest, module_name, cls, cls.PROFILE, tuple(module.GENES),
                     dict(module.FITNESS) if module.FITNESS else None)


def check_solidity_file(path: Any, profile: BotProfile, expected_digest: Optional[str] = None) -> List[str]:
    """Issues of a ``.sol`` file against the Python profile it mirrors (empty = consistent)."""
    raw = Path(path).read_bytes()
    if expected_digest is not None and hashlib.sha256(raw).hexdigest() != expected_digest:
        return ["digest does not match the manifest"]
    issues, consts = solidity_source_issues(raw.decode("utf-8", errors="replace"))
    if issues:
        return issues
    try:
        back = profile_from_solidity(consts, profile.name)
    except CodeRejected as exc:
        return [str(exc)]
    if back != profile:
        return ["the Solidity record does not round-trip to the Python profile"]
    if consts["__fingerprint__"] != int(profile.fingerprint(), 16):
        return ["Solidity FINGERPRINT differs from the profile"]
    return []


@dataclass(frozen=True)
class GeneratedFile:
    slot: int
    name: str
    py_path: str
    py_digest: str
    py_bytes: int
    sol_path: str
    sol_digest: str
    sol_bytes: int
    fingerprint: str
    class_name: str
    generation: int

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def read_manifest(directory: Any = None) -> Dict[str, Any]:
    path = resolve_bot_dir(directory) / MANIFEST_NAME
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or data.get("codegen_version") != CODEGEN_VERSION or not isinstance(
            data.get("bots"), list):
        raise CodeRejected(f"{path}: not a code factory manifest")
    return data


def load_manifest_bots(directory: Any = None, limit: Optional[int] = None,
                       check_solidity: bool = True) -> List[SynthesizedBot]:
    """Verified bot instances for every entry of a factory directory's manifest (digests enforced)."""
    root = resolve_bot_dir(directory)
    data = read_manifest(root)
    out: List[SynthesizedBot] = []
    for entry in data["bots"][: limit if limit is not None else None]:
        loaded = load_bot(root / Path(entry["py_path"]).name, entry["py_digest"])
        if check_solidity and entry.get("sol_path"):
            problems = check_solidity_file(root / Path(entry["sol_path"]).name, loaded.profile, entry["sol_digest"])
            if problems:
                raise CodeRejected(f"{entry['sol_path']}: " + "; ".join(problems))
        out.append(loaded.spawn())
    return out


def _disk_heat_task(task: Tuple[Any, ...]) -> Dict[str, Any]:
    """Worker entry point: load the heat's bots from disk (digest-checked) and run the arena."""
    heat, generation, entries, arena_cfg, incumbent, violation_limit = task
    cpu0 = time.process_time()
    loaded = [load_bot(path, digest) for path, digest in entries]
    bots = [lb.spawn() for lb in loaded]
    result = BotArena(bots, ArenaConfig(**arena_cfg), incumbent, violation_limit).run()
    result.update(heat=heat, generation=generation, cpu_s=time.process_time() - cpu0, pid=os.getpid(),
                  modules=[type(b).__module__ for b in bots], digests=[lb.digest for lb in loaded])
    return result


# ---------------------------------------------------------------------------
# The factory
# ---------------------------------------------------------------------------


Listener = Callable[[Dict[str, Any]], None]


@dataclass
class CodeFactoryReport:
    directory: str
    bots: int
    generations: int
    history: List[Dict[str, Any]]
    champions: List[Dict[str, Any]]
    files_written: int
    bytes_written: int
    write_s: float
    elapsed_s: float
    cpu_s: float
    workers: int
    violations: List[Dict[str, Any]]
    violation_count: int
    checks: int
    engine: Dict[str, Any]
    language: str
    manifest_digest: str = ""
    pids: List[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.violation_count == 0

    def summary(self) -> Dict[str, Any]:
        return {
            "directory": self.directory,
            "bots_per_generation": self.bots,
            "generations": self.generations,
            "files_written": self.files_written,
            "bytes_written": self.bytes_written,
            "write_s": round(self.write_s, 6),
            "files_per_s": round(self.files_written / self.write_s, 1) if self.write_s > 0 else 0.0,
            "elapsed_s": round(self.elapsed_s, 4),
            "cpu_s": round(self.cpu_s, 4),
            "workers": self.workers,
            "distinct_worker_pids": len(set(self.pids)),
            "language": self.language,
            "manifest_digest": self.manifest_digest,
            "engine": dict(self.engine),
            "history": [dict(h) for h in self.history],
            "champions": [dict(c) for c in self.champions],
            "invariants": {"ok": self.ok, "rules": list(FACTORY_CODE_RULES), "checks": self.checks,
                           "violation_count": self.violation_count, "violations": self.violations[:20]},
        }


class CodeFactory:
    """Write bots to disk, deploy them from disk, rank, retrain, rewrite - for ``generations`` rounds."""

    def __init__(self, directory: Any = None, bots: int = 16, generations: int = 3, heat_size: int = 8,
                 heat_slots: int = 6, seed: int = 0, regime: str = "volatile", lang: Optional[str] = None,
                 listener: Optional[Listener] = None, write_solidity: bool = True, keep_champions: bool = True,
                 epochs: int = 40, violation_limit: int = 64, **factory_kwargs: Any) -> None:
        if bots < 1:
            raise ConfigError("the code factory needs at least one bot per generation")
        self.directory = resolve_bot_dir(directory)
        self.agent = AgentFactory(bots=bots, generations=generations, heat_size=heat_size, heat_slots=heat_slots,
                                  seed=seed, regime=regime, epochs=epochs, **factory_kwargs)
        self.lang = i18n.resolve_language(lang)
        self.listener = listener
        self.write_solidity = write_solidity
        self.keep_champions = keep_champions
        self.violation_limit = violation_limit
        self._lock = threading.Lock()
        self._violations: List[Dict[str, Any]] = []
        self._count = 0
        self._checks = 0

    @classmethod
    def for_config(cls, config: Any, **overrides: Any) -> "CodeFactory":
        params: Dict[str, Any] = dict(seed=config.seed, regime=config.rl_regime,
                                      lang=getattr(config, "language", None), base_fee_wei=config.base_fee_wei,
                                      gas_limit=config.block_gas_limit)
        params.update(overrides)
        return cls(**params)

    # -- bookkeeping ---------------------------------------------------------

    def _emit(self, event: Dict[str, Any]) -> None:
        if self.listener is not None:
            self.listener(event)

    def _check(self, rule: str, ok: bool, detail: str, generation: int) -> None:
        with self._lock:
            self._checks += 1
            if not ok:
                self._count += 1
                if len(self._violations) < self.violation_limit:
                    self._violations.append({"rule": rule, "generation": generation, "detail": detail})

    # -- writing -------------------------------------------------------------

    def write_population(self, pop: Population, fitness: Optional[Sequence[Optional[Mapping[str, int]]]] = None
                         ) -> Tuple[List[GeneratedFile], float]:
        """Render and atomically write every bot of ``pop``; returns the files and the seconds spent."""
        started = time.perf_counter()
        files: List[GeneratedFile] = []
        self.directory.mkdir(parents=True, exist_ok=True)
        for i, (profile, genes) in enumerate(zip(pop.profiles, pop.genes)):
            fit = fitness[i] if fitness is not None else None
            py_src = render_python(profile, [float(x) for x in genes], fit, self.lang)
            issues = python_source_issues(py_src)
            self._check("factory.py_allowlist", not issues, f"{profile.name}: {issues}", pop.generation)
            try:
                compile(py_src, f"bot_{i:04d}.py", "exec", dont_inherit=True)
                compiled = True
            except SyntaxError as exc:
                compiled = False
                self._emit({"event": "rejected", "path": f"bot_{i:04d}.py", "reason": str(exc)})
            self._check("factory.py_compiles", compiled, f"{profile.name} does not compile", pop.generation)
            back = decode_profile(np.asarray(genes, dtype=np.float64), profile.name, profile.generation)
            self._check("factory.py_roundtrip", replace(back, capital_wei=profile.capital_wei) == profile,
                        f"{profile.name}: genes do not decode to the written profile", pop.generation)
            py_path = self.directory / f"bot_{i:04d}.py"
            py_digest = write_atomic(py_path, py_src)
            sol_path, sol_digest, sol_bytes = "", "", 0
            if self.write_solidity:
                sol_src = render_solidity(profile, fit, self.lang)
                sol_issues, consts = solidity_source_issues(sol_src)
                self._check("factory.sol_allowlist", not sol_issues, f"{profile.name}: {sol_issues}", pop.generation)
                same = not sol_issues and profile_from_solidity(consts, profile.name) == profile
                self._check("factory.sol_roundtrip", same, f"{profile.name}: record differs", pop.generation)
                sp = self.directory / f"bot_{i:04d}.sol"
                sol_digest = write_atomic(sp, sol_src)
                sol_path, sol_bytes = str(sp), len(sol_src.encode("utf-8"))
            f = GeneratedFile(i, profile.name, str(py_path), py_digest, len(py_src.encode("utf-8")), sol_path,
                              sol_digest, sol_bytes, profile.fingerprint(), class_name(profile), pop.generation)
            files.append(f)
            self._emit({"event": "file_written", "generation": pop.generation, "slot": i, "path": f.py_path,
                        "bytes": f.py_bytes + f.sol_bytes, "t": time.perf_counter() - started})
        self._prune(len(files))
        return files, time.perf_counter() - started

    def _prune(self, keep: int) -> None:
        """Remove bot files from earlier, larger generations (only ``bot_NNNN.py/.sol`` in this directory)."""
        for p in self.directory.iterdir():
            m = BOT_FILE.match(p.name)
            if m and int(m.group(1)) >= keep and p.is_file():
                p.unlink()

    def write_manifest(self, files: Sequence[GeneratedFile], history: Sequence[Mapping[str, Any]]) -> str:
        data = {"codegen_version": CODEGEN_VERSION, "language": self.lang, "generation": files[0].generation
                if files else 0, "bots": [f.to_dict() for f in files], "history": list(history)}
        return write_atomic(self.directory / MANIFEST_NAME, json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    def write_champion(self, profile: BotProfile, genes: Sequence[float], fit: Mapping[str, int]) -> Dict[str, Any]:
        root = self.directory / CHAMPIONS_DIR
        stem = f"gen_{profile.generation:03d}_{profile.fingerprint()}"
        py = write_atomic(root / f"{stem}.py", render_python(profile, genes, fit, self.lang))
        out = {"name": profile.name, "generation": profile.generation, "py_path": str(root / f"{stem}.py"),
               "py_digest": py, **{k: int(v) for k, v in fit.items()}}
        if self.write_solidity:
            out["sol_digest"] = write_atomic(root / f"{stem}.sol", render_solidity(profile, fit, self.lang))
        return out

    # -- deployment ----------------------------------------------------------

    def tasks(self, pop: Population, files: Sequence[GeneratedFile]) -> Tuple[List[List[int]], List[Tuple[Any, ...]]]:
        heats, base = self.agent.tasks(pop)
        tasks = [(h, pop.generation, [(files[i].py_path, files[i].py_digest) for i in members], t[3], t[4], t[5])
                 for (h, members), t in zip(enumerate(heats), base)]
        return heats, tasks

    def deploy(self, pop: Population, files: Sequence[GeneratedFile], executor: Optional[WorkerPool] = None
               ) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
        heats, tasks = self.tasks(pop, files)
        if executor is None or not tasks:
            return heats, [_disk_heat_task(t) for t in tasks]
        return heats, executor.map(_disk_heat_task, tasks)

    # -- the loop ------------------------------------------------------------

    def run(self, executor: Optional[WorkerPool] = None, workers: Optional[int] = None,
            stop: Optional[threading.Event] = None) -> CodeFactoryReport:
        own = executor is None and workers is not None
        wp = WorkerPool(workers) if own else executor
        agent = self.agent
        started = time.perf_counter()
        history: List[Dict[str, Any]] = []
        champions: List[Dict[str, Any]] = []
        files_written = bytes_written = 0
        write_s = cpu = 0.0
        pids: List[int] = []
        n_workers = 1
        parents: Optional[List[int]] = None
        parent_latents: Optional[np.ndarray] = None
        parent_keys: Optional[List[Tuple[int, ...]]] = None
        manifest_digest = ""
        try:
            for g in range(agent.generations):
                if stop is not None and stop.is_set():
                    break
                pop = agent.synthesize(g, parents, parent_latents)
                n = len(pop.profiles)
                files, spent = self.write_population(pop)
                write_s += spent
                files_written += len(files) * (2 if self.write_solidity else 1)
                bytes_written += sum(f.py_bytes + f.sol_bytes for f in files)
                manifest_digest = self.write_manifest(files, history)
                heats, parts = self.deploy(pop, files, wp)
                if wp is not None and heats:
                    n_workers = wp.workers
                self._emit({"event": "deployed", "generation": g, "bots": n, "heats": len(heats),
                            "message": i18n.t("factory.log.deployed", self.lang, bots=n, heats=len(heats))})
                index = {p.name: i for i, p in enumerate(pop.profiles)}
                rows: List[Optional[Dict[str, Any]]] = [None] * n
                expected = {f.py_digest for f in files}
                for part in parts:
                    cpu += part["cpu_s"]
                    pids.append(part["pid"])
                    self._check("factory.digest", set(part["digests"]) <= expected,
                                f"heat {part['heat']} ran code that is not in the manifest", g)
                    self._check("factory.disk_origin", all(m.startswith(GENERATED_PACKAGE + ".")
                                                           for m in part["modules"]),
                                f"heat {part['heat']} ran in-memory classes", g)
                    with self._lock:
                        self._count += part["violation_count"]
                        self._checks += part["checks"]
                    for row in part["bots"]:
                        i = index[row["name"]]
                        self._check("factory.heat_coverage", rows[i] is None, f"{row['name']} ran twice", g)
                        rows[i] = row
                self._check("factory.heat_coverage", all(r is not None for r in rows), "a bot never ran", g)
                ledger_rows = [r if r is not None else {**Ledger().to_dict(), "name": pop.profiles[i].name}
                               for i, r in enumerate(rows)]
                keys = [fitness_key(r) for r in ledger_rows]
                pnl = [k[0] for k in keys]
                improved = [keys[i] > parent_keys[pop.parents[i]] for i in range(n)  # type: ignore[index]
                            if parent_keys is not None and pop.parents[i] is not None]
                elite = max(1, math.ceil(agent.elite_frac * n))
                offspring = allocate_offspring(keys, n, elite)
                self._check("factory.offspring", sum(offspring.values()) == n, "offspring do not sum to n", g)
                idx = list(offspring)
                weights = np.array([offspring[i] for i in idx], dtype=np.float64)
                losses = agent.engine.fit(pop.inputs[idx], pop.genes[idx], weights, agent.epochs) \
                    if float(np.sum(weights)) > 0 else [0.0]
                self._check("factory.finite_weights", agent.engine.is_finite(), "non-finite generator weight", g)
                if improved:
                    agent.mutation.update(sum(improved) / len(improved))
                order = rank_order(keys)
                best = order[0]
                fit = {"pnl_wei": pnl[best], "won": keys[best][1], "arrived": keys[best][2],
                       "submitted": keys[best][3]}
                if self.keep_champions:
                    champions.append(self.write_champion(pop.profiles[best], [float(x) for x in pop.genes[best]],
                                                         fit))
                entry = {"generation": g, "bots": n, "heats": len(heats), "files": len(files),
                         "bytes": sum(f.py_bytes + f.sol_bytes for f in files), "write_s": round(spent, 6),
                         "files_per_s": round(len(files) / spent, 1) if spent > 0 else 0.0,
                         "loss_before": round(float(losses[0]), 9), "loss_after": round(float(losses[-1]), 9),
                         "best_bot": pop.profiles[best].name, "best_pnl_wei": pnl[best],
                         "total_pnl_wei": sum(pnl), "positive_bots": sum(1 for v in pnl if v > 0),
                         "sigma": round(agent.mutation.sigma, 9)}
                history.append(entry)
                manifest_digest = self.write_manifest(files, history)
                self._emit({"event": "generation", **entry,
                            "message": i18n.t("factory.log.generation", self.lang, generation=g, files=len(files),
                                              bytes=entry["bytes"], seconds=entry["write_s"], best=pnl[best],
                                              loss_before=entry["loss_before"], loss_after=entry["loss_after"])})
                parents = lineage(offspring)
                parent_latents = pop.latents
                parent_keys = keys
        finally:
            if own and wp is not None:
                wp.close()
        return CodeFactoryReport(
            directory=str(self.directory), bots=agent.bots, generations=len(history), history=history,
            champions=champions, files_written=files_written, bytes_written=bytes_written, write_s=write_s,
            elapsed_s=time.perf_counter() - started, cpu_s=cpu, workers=n_workers,
            violations=list(self._violations), violation_count=self._count, checks=self._checks,
            engine=agent.engine.describe(), language=self.lang, manifest_digest=manifest_digest, pids=pids)


def decode_check(loaded: LoadedBot) -> bool:
    """True when the file's genes decode to its profile (ignoring the capital grant, name and generation)."""
    back = decode_profile(np.asarray(loaded.genes), loaded.profile.name, loaded.profile.generation)
    return replace(back, capital_wei=loaded.profile.capital_wei) == loaded.profile


__all__ = [
    "BOT_DIR_ENV",
    "CHAMPIONS_DIR",
    "CODEGEN_VERSION",
    "DEFAULT_BOT_DIR",
    "FACTORY_CODE_RULES",
    "GENERATED_PACKAGE",
    "IMPORT_MODULE",
    "MANIFEST_NAME",
    "PROFILE_FIELD_NAMES",
    "CodeFactory",
    "CodeFactoryReport",
    "CodeRejected",
    "GeneratedFile",
    "LoadedBot",
    "check_solidity_file",
    "decode_check",
    "load_bot",
    "load_manifest_bots",
    "profile_fields",
    "profile_from_solidity",
    "python_source_issues",
    "read_manifest",
    "render_python",
    "render_solidity",
    "resolve_bot_dir",
    "sha256_text",
    "solidity_constants",
    "solidity_source_issues",
    "write_atomic",
]
