"""Layers 09 / 10 / 11: ``sandwich.toml`` settings sync engine.

The central settings file
-------------------------
``sandwich.toml`` (at the repository root) holds every setting of the
simulator in one place::

    [meta]            schema_version
    [run]             mode, seed, native_token, default_tolerance_bps
    [parallel]        start_method, parallel_threshold
    [gas]             base_fee_wei, priority_fee_wei, block_gas_limit
    [search]          max_frontrun_bps
    [pbs]             builder_margin_bps, bribe_bps
    [private_relay]   mev_share_refund_bps
    [rl]              regime
    [trace]           enabled, path
    [audit]           enabled, fail_on, code_scan, tick_samples
    [modes.moderate]  \
    [modes.brutal]     > the three run matrices (config.PRESETS) plus workers
    [modes.max]       /  (0 = automatic under the mode's CPU policy) and cpu_policy

Precedence, lowest first: the hardcoded preset of the mode
(:data:`simulator.config.PRESETS`) < ``sandwich.toml`` < ``$MEV_SIM_WORKERS``
(worker count only) < ``--set TABLE.KEY=VALUE`` < command-line flags.

One schema, three views
-----------------------
Every setting is declared once in :data:`SETTINGS` (TOML location,
:class:`~simulator.config.SimulationConfig` field, type, bounds, CLI flag and
GUI widget). The CLI parser (:func:`add_setting_arguments`), the GUI form
(:func:`form_schema`, served by :mod:`simulator.settings_gui`) and the TOML
layout (:func:`render_layout`) are all generated from it, and :func:`resolve_config` turns any mix of file
values and CLI values into a ``SimulationConfig`` through the same
validation, so the three views cannot drift apart.

Sync
----
:meth:`SettingsStore.sync` adds keys that newer versions introduce (with their
default value and a comment) without touching the values, comments or layout
already in the file, and reports keys it does not know. Updates
(:meth:`SettingsStore.update`) rewrite only the value text of the keys they
change, so hand-written comments, ordering, CRLF line endings and a UTF-8 BOM
all survive.

Safe writes
-----------
Every write goes through :class:`SettingsStore`: an exclusive lock file
(stale locks from dead or long-gone writers are broken), an
optimistic-concurrency check against the digest the caller loaded,
validation of the new text before anything touches the disk, a temporary
file in the same directory that is written, fsynced and read back, a
``.bak`` copy of the previous version, the original permission bits, and an
atomic ``os.replace`` (retried on transient Windows sharing violations).
Symlinked settings files are written through to their target. Any failure
leaves the original file byte-for-byte intact and removes the temporary
file. File operations go through :class:`FileOps`, so failures can be
injected in tests.

Parsing uses :mod:`tomllib` (Python 3.11+) and falls back to a built-in
parser for the TOML subset this layout uses on Python 3.10.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import re
import secrets
import socket
import stat as stat_mod
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .config import (
    AUDIT_LEVELS,
    MAX_AGENT_BOTS,
    MAX_AGENT_GENERATIONS,
    MAX_PG_PATHS,
    MAX_SHARD_DENSITY,
    MODE_KEYS,
    PRESETS,
    RL_REGIMES,
    START_METHODS,
    ConfigError,
    CpuPolicy,
    RunMode,
    SimulationConfig,
    env_worker_request,
)

SCHEMA_VERSION = 1
CONFIG_FILENAME = "sandwich.toml"
CONFIG_ENV_VAR = "MEV_SIM_CONFIG"
MODE_NAMES: Tuple[str, ...] = tuple(m.short for m in RunMode)
Location = Tuple[str, str]  # (table, key), e.g. ("gas", "base_fee_wei") or ("modes.max", "workers")
UINT256_MAX = (1 << 256) - 1


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SettingsIssue:
    location: str
    message: str
    line: Optional[int] = None
    severity: str = "error"

    def __str__(self) -> str:
        where = f"{self.location} (line {self.line})" if self.line else self.location
        return f"{where}: {self.message}"


class SettingsError(ConfigError):
    """The settings content is invalid; carries every issue found."""

    def __init__(self, issues: Sequence[SettingsIssue], path: Optional[Path] = None) -> None:
        self.issues = list(issues)
        self.path = path
        head = f"{path}: " if path else ""
        super().__init__(head + "; ".join(str(i) for i in self.issues))


class SettingsFileError(RuntimeError):
    """A settings file could not be locked, read or written safely."""


class SettingsLockError(SettingsFileError):
    """Another writer holds the settings lock."""


class SettingsConflictError(SettingsFileError):
    """The file changed after the caller loaded it."""


class SettingsWriteError(SettingsFileError):
    """Writing failed; the original file is unchanged."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Setting:
    attr: str  # SimulationConfig field, or "mode"
    table: str  # TOML table for common settings; "modes" for the per-mode matrix
    key: str
    kind: type
    flags: Tuple[str, ...]
    help: str
    choices: Tuple[str, ...] = ()
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    zero_means: str = ""
    normalize: Optional[Callable[[Any], Any]] = None
    lenient_choice: bool = False  # parse through normalize instead of a fixed choice list on the CLI

    @property
    def per_mode(self) -> bool:
        return self.table == "modes"

    @property
    def widget(self) -> str:
        if self.kind is bool:
            return "toggle"
        if self.choices:
            return "select"
        if self.kind is int:
            return "number"
        return "path" if self.key == "path" else "text"

    @property
    def dest(self) -> str:
        return f"cfg_{self.attr}"

    def location(self, mode: Optional[str] = None) -> Location:
        if self.per_mode:
            if mode not in MODE_NAMES:
                raise ConfigError(f"per-mode setting {self.key} needs a mode")
            return f"modes.{mode}", self.key
        return self.table, self.key

    def dotted(self, mode: Optional[str] = None) -> str:
        table, key = self.location(mode if self.per_mode else None)
        return f"{table}.{key}"

    def coerce(self, value: Any, where: str) -> Any:
        """Type-check, normalise and range-check one value; raises SettingsError."""
        if self.kind is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise SettingsError([SettingsIssue(where, f"expected an integer, got {_type_name(value)}")])
        elif self.kind is bool:
            if not isinstance(value, bool):
                raise SettingsError([SettingsIssue(where, f"expected true or false, got {_type_name(value)}")])
        elif self.kind is str:
            if not isinstance(value, str):
                raise SettingsError([SettingsIssue(where, f"expected a string, got {_type_name(value)}")])
        if self.normalize is not None:
            try:
                value = self.normalize(value)
            except (ConfigError, ValueError, TypeError) as exc:
                raise SettingsError([SettingsIssue(where, str(exc))]) from None
        if self.choices and value not in self.choices:
            hint = _suggest(str(value), self.choices)
            raise SettingsError([SettingsIssue(where, f"{value!r} is not one of {', '.join(self.choices)}{hint}")])
        if self.kind is int:
            if self.minimum is not None and value < self.minimum:
                raise SettingsError([SettingsIssue(where, f"{value} is below the minimum {self.minimum}")])
            if self.maximum is not None and value > self.maximum:
                raise SettingsError([SettingsIssue(where, f"{value} is above the maximum {self.maximum}")])
        return value

    def parse_text(self, text: str, where: str) -> Any:
        """Parse a command-line / ``--set`` string into this setting's type, then coerce."""
        raw = text.strip()
        if self.kind is int:
            try:
                value: Any = int(raw.replace("_", ""), 0)
            except ValueError:
                raise SettingsError([SettingsIssue(where, f"{text!r} is not an integer")]) from None
        elif self.kind is bool:
            low = raw.lower()
            if low in ("1", "true", "yes", "on"):
                value = True
            elif low in ("0", "false", "no", "off"):
                value = False
            else:
                raise SettingsError([SettingsIssue(where, f"{text!r} is not a boolean")])
        else:
            value = raw[1:-1] if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'" else raw
        return self.coerce(value, where)

    def to_form(self, default: Any) -> Dict[str, Any]:
        return {
            "key": self.key,
            "attr": self.attr,
            "toml": f"{self.table}.<mode>.{self.key}" if self.per_mode else f"{self.table}.{self.key}",
            "type": self.kind.__name__,
            "widget": self.widget,
            "default": default,
            "min": self.minimum,
            "max": self.maximum,
            "choices": list(self.choices),
            "zero_means": self.zero_means or None,
            "cli": list(self.flags),
            "help": self.help,
        }


def _type_name(value: Any) -> str:
    return "a boolean" if isinstance(value, bool) else f"{type(value).__name__} {value!r}"


def _suggest(word: str, options: Iterable[str]) -> str:
    close = difflib.get_close_matches(word, list(options), n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _norm_mode(value: Any) -> str:
    return RunMode.parse(value).short


def _norm_policy(value: Any) -> str:
    return CpuPolicy.parse(value).value


def _norm_gas_limit(value: int) -> int:
    from .block_builder import clamp_gas_limit

    return clamp_gas_limit(value)


def _s(attr: str, table: str, key: str, kind: type, flags: Sequence[str], help: str, **kw: Any) -> Setting:
    return Setting(attr, table, key, kind, tuple(flags), help, **kw)


BPS = dict(minimum=0, maximum=10_000)
SETTINGS: Tuple[Setting, ...] = (
    # [run]
    _s("mode", "run", "mode", str, ["--mode"], "run mode: moderate | brutal | max", choices=MODE_NAMES,
       normalize=_norm_mode, lenient_choice=True),
    _s("seed", "run", "seed", int, ["--seed"], "seed for every deterministic stream"),
    _s("native_token", "run", "native_token", str, ["--native-token"], "token profits and gas are valued in"),
    _s("default_tolerance_bps", "run", "default_tolerance_bps", int, ["--tolerance"],
       "slippage tolerance (bps) for payloads without one", **BPS),
    # [parallel]
    _s("start_method", "parallel", "start_method", str, ["--start-method"], "multiprocessing start method",
       choices=START_METHODS),
    _s("parallel_threshold", "parallel", "parallel_threshold", int, ["--parallel-threshold"],
       "batches smaller than this run inline", minimum=1),
    # [gas]
    _s("base_fee_wei", "gas", "base_fee_wei", int, ["--base-fee"], "starting base fee (wei)", minimum=1,
       maximum=UINT256_MAX),
    _s("priority_fee_wei", "gas", "priority_fee_wei", int, ["--priority-fee"], "searcher priority fee (wei)",
       minimum=0, maximum=UINT256_MAX),
    _s("block_gas_limit", "gas", "block_gas_limit", int, ["--gas-limit"],
       "block gas limit, clamped into [21000, 30000000]", minimum=21_000, maximum=30_000_000,
       normalize=_norm_gas_limit),
    # [search]
    _s("max_frontrun_bps", "search", "max_frontrun_bps", int, ["--max-frontrun"],
       "largest front-run as bps of the input reserve", minimum=1, maximum=10_000),
    # [pbs]
    _s("builder_margin_bps", "pbs", "builder_margin_bps", int, ["--builder-margin"],
       "share of block value a builder keeps (bps)", **BPS),
    _s("bribe_bps", "pbs", "bribe_bps", int, ["--bribe"], "share of sandwich profit paid to the builder (bps)",
       **BPS),
    # [private_relay]
    _s("mev_share_refund_bps", "private_relay", "mev_share_refund_bps", int, ["--refund"],
       "MEV-Share refund to users (bps)", **BPS),
    # [rl]
    _s("rl_regime", "rl", "regime", str, ["--rl-regime"], "RL market regime: calm | volatile | extreme",
       choices=RL_REGIMES),
    # [trace]
    _s("trace", "trace", "enabled", bool, ["--trace"], "log every sealed block's transaction trace"),
    _s("trace_path", "trace", "path", str, ["--trace-file"], "trace destination file (empty = stderr)"),
    # [audit]
    _s("audit", "audit", "enabled", bool, ["--audit"], "run the integrity validator before simulating"),
    _s("audit_fail_on", "audit", "fail_on", str, ["--audit-fail-on"],
       "lowest finding severity that blocks a run: error | warning | info", choices=AUDIT_LEVELS),
    _s("audit_code_scan", "audit", "code_scan", bool, ["--code-scan"], "include the source code scan"),
    _s("audit_tick_samples", "audit", "tick_samples", int, ["--audit-tick-samples"],
       "ticks sampled by the tick-arithmetic checks", minimum=0, maximum=100_000),
    # [modes.<mode>]: the run matrices
    _s("workers", "modes", "workers", int, ["--workers"], "worker processes (0 = automatic for the CPU policy)",
       minimum=0, zero_means="auto"),
    _s("cpu_policy", "modes", "cpu_policy", str, ["--cpu-policy"],
       "capped = 50-60% of CPUs, full = 100% of logical CPUs", choices=("capped", "full"),
       normalize=_norm_policy, lenient_choice=True),
    _s("blocks", "modes", "blocks", int, ["--blocks"], "blocks (slots) to simulate", minimum=1),
    _s("txs_per_block", "modes", "txs_per_block", int, ["--txs"], "pending transactions per block", minimum=0),
    _s("frontrun_grid", "modes", "frontrun_grid", int, ["--frontrun-grid"], "front-run sizes brute-forced per victim",
       minimum=0),
    _s("refine_iterations", "modes", "refine_iterations", int, ["--refine"], "ternary-search refinement steps",
       minimum=0),
    _s("fuzz_iterations", "modes", "fuzz_iterations", int, ["--fuzz"], "invariant fuzz swaps", minimum=0),
    _s("layout_fuzz_iterations", "modes", "layout_fuzz_iterations", int, ["--layouts"],
       "block layouts fuzzed per slot", minimum=0),
    _s("private_flow_bps", "modes", "private_flow_bps", int, ["--private-share"],
       "share of user swaps sent to the private relay (bps)", **BPS),
    _s("router_max_hops", "modes", "router_max_hops", int, ["--max-hops"], "longest route / cycle (1-6)",
       minimum=1, maximum=6),
    _s("rl_rounds", "modes", "rl_rounds", int, ["--rl-rounds"], "parallel policy-iteration rounds", minimum=0),
    _s("rl_steps_per_shard", "modes", "rl_steps_per_shard", int, ["--rl-steps"], "RL steps per shard and round",
       minimum=0),
    _s("rl_eval_steps", "modes", "rl_eval_steps", int, ["--rl-eval-steps"], "greedy RL evaluation steps",
       minimum=0),
    _s("shard_density", "modes", "shard_density", int, ["--shard-density"],
       "multiplier on every fixed shard / chunk count", minimum=1, maximum=MAX_SHARD_DENSITY),
    _s("pg_paths", "modes", "pg_paths", int, ["--pg-paths"], "policy-gradient paths fuzzed concurrently",
       minimum=0, maximum=MAX_PG_PATHS),
    _s("pg_path_steps", "modes", "pg_path_steps", int, ["--pg-steps"], "steps per policy-gradient path", minimum=1),
    _s("pg_batch", "modes", "pg_batch", int, ["--pg-batch"], "paths per worker task", minimum=1),
    _s("agent_bots", "modes", "agent_bots", int, ["--agent-bots"],
       "AI-generated competitor bots per generation (0 = off)", minimum=0, maximum=MAX_AGENT_BOTS),
    _s("agent_generations", "modes", "agent_generations", int, ["--agent-generations"],
       "AI factory generations (synthesise, deploy, retrain)", minimum=1, maximum=MAX_AGENT_GENERATIONS),
    _s("agent_heat_size", "modes", "agent_heat_size", int, ["--agent-heat-size"],
       "AI bots per arena heat (one worker task)", minimum=1),
    _s("agent_heat_slots", "modes", "agent_heat_slots", int, ["--agent-heat-slots"],
       "blocks each AI arena heat simulates", minimum=1),
)

COMMON_TABLES: Tuple[str, ...] = ("run", "parallel", "gas", "search", "pbs", "private_relay", "rl", "trace", "audit")
TABLE_TITLES: Dict[str, str] = {
    "meta": "Meta", "run": "Run", "parallel": "Parallel", "gas": "Gas", "search": "Sandwich search",
    "pbs": "PBS builders", "private_relay": "Private relay", "rl": "Reinforcement learning", "trace": "Trace output",
    "audit": "Integrity audit", "modes": "Run matrices",
}
BY_ATTR: Dict[str, Setting] = {s.attr: s for s in SETTINGS}
COMMON_BY_LOCATION: Dict[Location, Setting] = {(s.table, s.key): s for s in SETTINGS if not s.per_mode}
PER_MODE_BY_KEY: Dict[str, Setting] = {s.key: s for s in SETTINGS if s.per_mode}
TABLE_KEYS: Dict[str, List[str]] = {t: [s.key for s in SETTINGS if s.table == t] for t in COMMON_TABLES}


def lookup(table: str, key: str) -> Setting:
    """The setting stored at ``[table] key``; raises SettingsError for unknown locations."""
    if table.startswith("modes."):
        mode = table.split(".", 1)[1]
        if mode not in MODE_NAMES:
            raise SettingsError([SettingsIssue(table, f"unknown mode table{_suggest(mode, MODE_NAMES)}")])
        if key in PER_MODE_BY_KEY:
            return PER_MODE_BY_KEY[key]
        raise SettingsError([SettingsIssue(f"{table}.{key}", _unknown_key_message(key, list(PER_MODE_BY_KEY)))])
    if (table, key) in COMMON_BY_LOCATION:
        return COMMON_BY_LOCATION[(table, key)]
    if table not in TABLE_KEYS:
        raise SettingsError([SettingsIssue(table, f"unknown table{_suggest(table, list(TABLE_KEYS) + ['modes'])}")])
    raise SettingsError([SettingsIssue(f"{table}.{key}", _unknown_key_message(key, TABLE_KEYS[table]))])


def _unknown_key_message(key: str, known: Sequence[str]) -> str:
    if key in PER_MODE_BY_KEY:
        return f"{key} is set per mode; put it under [modes.moderate], [modes.brutal] or [modes.max]"
    for (table, k), s in COMMON_BY_LOCATION.items():
        if k == key:
            return f"{key} belongs under [{table}]"
    return f"unknown key{_suggest(key, known)}"


def _dataclass_defaults() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for f in fields(SimulationConfig):
        if f.default is not getattr(f, "default_factory", None) and f.name not in ("mode", "workers", "extra"):
            out[f.name] = f.default
    return out


def default_value(setting: Setting, mode: Optional[str] = None) -> Any:
    """Built-in default of a setting (mode preset for per-mode keys)."""
    if setting.attr == "mode":
        return RunMode.MODERATE_SCAN.short
    if setting.per_mode:
        if setting.attr == "workers":
            return 0
        return PRESETS[RunMode.parse(mode or "moderate")][setting.attr]
    return _dataclass_defaults()[setting.attr]


def default_locations() -> Dict[Location, Any]:
    """Every TOML location with its built-in default."""
    out: Dict[Location, Any] = {("meta", "schema_version"): SCHEMA_VERSION}
    for s in SETTINGS:
        if s.per_mode:
            for m in MODE_NAMES:
                out[s.location(m)] = default_value(s, m)
        else:
            out[s.location()] = default_value(s)
    return out


# ---------------------------------------------------------------------------
# TOML: parse (tomllib or the built-in subset parser) and render
# ---------------------------------------------------------------------------


class TomlSyntaxError(ValueError):
    def __init__(self, message: str, line: int) -> None:
        self.line = line
        super().__init__(f"line {line}: {message}")


_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
_ESCAPES = {"b": "\b", "t": "\t", "n": "\n", "f": "\f", "r": "\r", '"': '"', "\\": "\\"}
_NUMBER = re.compile(r"[+-]?(?:0x[0-9A-Fa-f_]+|0o[0-7_]+|0b[01_]+|inf|nan|[0-9_]+(?:\.[0-9_]+)?(?:[eE][+-]?[0-9_]+)?)")


class MiniToml:
    """TOML subset parser: tables, dotted keys, strings, numbers, booleans, arrays and inline tables.

    Dates, times and arrays of tables are rejected. Used when :mod:`tomllib`
    is unavailable (Python 3.10).
    """

    def __init__(self, text: str) -> None:
        self.s = text
        self.i = 0
        self.line = 1

    def error(self, message: str) -> TomlSyntaxError:
        return TomlSyntaxError(message, self.line)

    def peek(self, n: int = 1) -> str:
        return self.s[self.i:self.i + n]

    def advance(self, n: int = 1) -> None:
        chunk = self.s[self.i:self.i + n]
        self.line += chunk.count("\n")
        self.i += n

    def skip_ws(self) -> None:
        while self.peek() in (" ", "\t"):
            self.advance()

    def skip_comment(self) -> None:
        if self.peek() == "#":
            while self.i < len(self.s) and self.peek() != "\n":
                if self.peek() not in ("\t",) and ord(self.peek()) < 0x20 and self.peek() != "\r":
                    raise self.error("control character in comment")
                self.advance()

    def skip_blank(self) -> None:
        """Whitespace, newlines and comments."""
        while self.i < len(self.s):
            c = self.peek()
            if c in (" ", "\t", "\n"):
                self.advance()
            elif c == "\r" and self.peek(2) == "\r\n":
                self.advance(2)
            elif c == "#":
                self.skip_comment()
            else:
                break

    def expect_eol(self) -> None:
        self.skip_ws()
        self.skip_comment()
        if self.i >= len(self.s):
            return
        if self.peek() == "\n":
            self.advance()
        elif self.peek(2) == "\r\n":
            self.advance(2)
        else:
            raise self.error(f"expected end of line, found {self.peek()!r}")

    def parse(self) -> Dict[str, Any]:
        root: Dict[str, Any] = {}
        current = root
        defined: set = set()
        while True:
            self.skip_blank()
            if self.i >= len(self.s):
                return root
            if self.peek() == "[":
                if self.peek(2) == "[[":
                    raise self.error("arrays of tables are not supported")
                self.advance()
                self.skip_ws()
                path = self.key_path()
                self.skip_ws()
                if self.peek() != "]":
                    raise self.error("expected ']' after table name")
                self.advance()
                self.expect_eol()
                key = tuple(path)
                if key in defined:
                    raise self.error(f"table [{'.'.join(path)}] defined twice")
                defined.add(key)
                current = root
                for part in path:
                    nxt = current.setdefault(part, {})
                    if not isinstance(nxt, dict):
                        raise self.error(f"key {part!r} is not a table")
                    current = nxt
            else:
                path = self.key_path()
                self.skip_ws()
                if self.peek() != "=":
                    raise self.error("expected '=' after key")
                self.advance()
                self.skip_ws()
                value = self.value()
                target = current
                for part in path[:-1]:
                    nxt = target.setdefault(part, {})
                    if not isinstance(nxt, dict):
                        raise self.error(f"key {part!r} is not a table")
                    target = nxt
                if path[-1] in target:
                    raise self.error(f"duplicate key {'.'.join(path)!r}")
                target[path[-1]] = value
                self.expect_eol()

    def key_path(self) -> List[str]:
        parts = [self.simple_key()]
        while True:
            self.skip_ws()
            if self.peek() != ".":
                return parts
            self.advance()
            self.skip_ws()
            parts.append(self.simple_key())

    def simple_key(self) -> str:
        c = self.peek()
        if c == '"':
            return self.basic_string()
        if c == "'":
            return self.literal_string()
        m = _BARE_KEY.match(self.s, self.i)
        if not m:
            raise self.error(f"invalid key starting with {c!r}")
        self.advance(m.end() - self.i)
        return m.group(0)

    def value(self) -> Any:
        c = self.peek()
        if self.peek(3) in ('"""', "'''"):
            return self.multiline_string()
        if c == '"':
            return self.basic_string()
        if c == "'":
            return self.literal_string()
        if c == "[":
            return self.array()
        if c == "{":
            return self.inline_table()
        if self.s.startswith("true", self.i):
            self.advance(4)
            return True
        if self.s.startswith("false", self.i):
            self.advance(5)
            return False
        return self.number()

    def number(self) -> Any:
        m = _NUMBER.match(self.s, self.i)
        if not m or not m.group(0):
            raise self.error(f"invalid value starting with {self.peek()!r}")
        tok = m.group(0)
        end = m.end()
        if end < len(self.s) and (self.s[end].isalnum() or self.s[end] in ":-+._"):
            raise self.error("invalid number (dates, times and unquoted strings are not supported)")
        self.advance(end - self.i)
        sign = -1 if tok[0] == "-" else 1
        body = tok[1:] if tok[0] in "+-" else tok
        if body in ("inf", "nan"):
            return sign * float("inf") if body == "inf" else float("nan")
        prefix = body[:2] if body[:2] in ("0x", "0o", "0b") else ""
        digits = body[len(prefix):]
        digit_class = "0-9A-Fa-f" if prefix == "0x" else "0-9"
        if re.search(rf"(?:^|[^{digit_class}])_|_(?:$|[^{digit_class}])", digits):
            raise self.error(f"underscores must sit between digits in {tok!r}")
        clean = digits.replace("_", "")
        if prefix:
            if tok[0] in "+-":
                raise self.error("signed non-decimal integers are not allowed")
            if not clean:
                raise self.error(f"missing digits in {tok!r}")
            return int(clean, {"0x": 16, "0o": 8, "0b": 2}[prefix])
        whole = re.split(r"[.eE]", clean, maxsplit=1)[0]
        if len(whole) > 1 and whole.startswith("0"):
            raise self.error("leading zeros are not allowed")
        if any(ch in clean for ch in ".eE"):
            return sign * float(clean)
        return sign * int(clean)

    def basic_string(self) -> str:
        self.advance()  # opening quote
        out: List[str] = []
        while True:
            if self.i >= len(self.s):
                raise self.error("unterminated string")
            c = self.peek()
            if c == '"':
                self.advance()
                return "".join(out)
            if c == "\n":
                raise self.error("newline in basic string")
            if c == "\\":
                out.append(self.escape())
                continue
            if ord(c) < 0x20 and c != "\t":
                raise self.error("control character in string")
            out.append(c)
            self.advance()

    def escape(self) -> str:
        self.advance()  # backslash
        c = self.peek()
        if c in _ESCAPES:
            self.advance()
            return _ESCAPES[c]
        if c in ("u", "U"):
            n = 4 if c == "u" else 8
            digits = self.s[self.i + 1:self.i + 1 + n]
            if len(digits) != n or any(ch not in "0123456789abcdefABCDEF" for ch in digits):
                raise self.error("invalid unicode escape")
            self.advance(n + 1)
            return chr(int(digits, 16))
        raise self.error(f"invalid escape \\{c}")

    def literal_string(self) -> str:
        self.advance()
        end = self.s.find("'", self.i)
        nl = self.s.find("\n", self.i)
        if end < 0 or (0 <= nl < end):
            raise self.error("unterminated literal string")
        out = self.s[self.i:end]
        self.advance(end - self.i + 1)
        return out

    def multiline_string(self) -> str:
        quote = self.peek(3)
        self.advance(3)
        if self.peek() == "\n":
            self.advance()
        elif self.peek(2) == "\r\n":
            self.advance(2)
        out: List[str] = []
        while True:
            if self.i >= len(self.s):
                raise self.error("unterminated multi-line string")
            if self.peek(3) == quote:
                # the delimiter is the last three of a run of quotes; up to two more belong to the content
                run = 3
                while self.s[self.i + run:self.i + run + 1] == quote[0]:
                    run += 1
                if run > 5:
                    raise self.error("too many quotes closing a multi-line string")
                out.append(quote[0] * (run - 3))
                self.advance(run)
                return "".join(out)
            c = self.peek()
            if quote == '"""' and c == "\\":
                nxt = self.s[self.i + 1:self.i + 2]
                if nxt in (" ", "\t", "\n", "\r"):
                    # line-ending backslash: must be the last non-blank character on its line
                    self.advance()
                    self.skip_ws()
                    if self.peek() == "\n":
                        self.advance()
                    elif self.peek(2) == "\r\n":
                        self.advance(2)
                    else:
                        raise self.error("a line-ending backslash must end its line")
                    while self.peek() in (" ", "\t", "\n") or self.peek(2) == "\r\n":
                        self.advance(2 if self.peek(2) == "\r\n" else 1)
                    continue
                out.append(self.escape())
                continue
            if c == "\r":
                if self.peek(2) != "\r\n":
                    raise self.error("bare carriage return in string")
                out.append("\n")
                self.advance(2)
                continue
            if (ord(c) < 0x20 and c not in ("\t", "\n")) or c == "\x7f":
                raise self.error("control character in string")
            out.append(c)
            self.advance()

    def array(self) -> List[Any]:
        self.advance()
        out: List[Any] = []
        while True:
            self.skip_blank()
            if self.peek() == "]":
                self.advance()
                return out
            out.append(self.value())
            self.skip_blank()
            if self.peek() == ",":
                self.advance()
            elif self.peek() == "]":
                self.advance()
                return out
            else:
                raise self.error("expected ',' or ']' in array")

    def inline_table(self) -> Dict[str, Any]:
        self.advance()
        out: Dict[str, Any] = {}
        self.skip_ws()
        if self.peek() == "}":
            self.advance()
            return out
        while True:
            self.skip_ws()
            path = self.key_path()
            self.skip_ws()
            if self.peek() != "=":
                raise self.error("expected '=' in inline table")
            self.advance()
            self.skip_ws()
            value = self.value()
            target = out
            for part in path[:-1]:
                target = target.setdefault(part, {})
                if not isinstance(target, dict):
                    raise self.error(f"key {part!r} is not a table")
            if path[-1] in target:
                raise self.error(f"duplicate key {'.'.join(path)!r}")
            target[path[-1]] = value
            self.skip_ws()
            if self.peek() == ",":
                self.advance()
                continue
            if self.peek() == "}":
                self.advance()
                return out
            raise self.error("expected ',' or '}' in inline table")


def parse_toml(text: str, prefer_builtin: bool = False) -> Dict[str, Any]:
    """Parse TOML text (a leading UTF-8 BOM is ignored); raises TomlSyntaxError."""
    if text.startswith("\ufeff"):
        text = text[1:]
    if not prefer_builtin:
        try:
            import tomllib
        except ImportError:  # Python 3.10
            pass
        else:
            try:
                return tomllib.loads(text)
            except tomllib.TOMLDecodeError as exc:
                m = re.search(r"line (\d+)", str(exc))
                raise TomlSyntaxError(str(exc), int(m.group(1)) if m else 0) from None
    return MiniToml(text).parse()


def render_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        sign = "-" if value < 0 else ""
        digits = str(abs(value))
        if len(digits) > 4:
            groups = []
            while digits:
                groups.append(digits[-3:])
                digits = digits[:-3]
            digits = "_".join(reversed(groups))
        return sign + digits
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        out = ['"']
        for ch in value:
            if ch == '"':
                out.append('\\"')
            elif ch == "\\":
                out.append("\\\\")
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\t":
                out.append("\\t")
            elif ch == "\r":
                out.append("\\r")
            elif ord(ch) < 0x20 or ord(ch) == 0x7F:
                out.append(f"\\u{ord(ch):04x}")
            else:
                out.append(ch)
        out.append('"')
        return "".join(out)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(render_value(v) for v in value) + "]"
    raise ConfigError(f"cannot render {type(value).__name__} as TOML")


HEADER = (
    "# sandwich.toml: central settings for the MEV sandwich simulator.\n"
    "#\n"
    "# Precedence, lowest first: the built-in preset of the mode < this file\n"
    "# < $MEV_SIM_WORKERS (worker count only) < --set TABLE.KEY=VALUE < flags.\n"
    "#\n"
    "#   python -m simulator --show-config      print the effective settings\n"
    "#   python -m simulator --sync-config      add keys that newer versions introduce\n"
    "#   python -m simulator --save-config ...  write command-line values back here\n"
    "#   python -m simulator --gui              edit this file in the browser\n"
)
MODE_NOTES = {
    "moderate": "Moderate Scan: read-only telemetry, gas-limit audit, passive dataflow (50-60% of CPUs).",
    "brutal": "Brutal Aggressive Simulation: multi-block sandwich search on the capped pool (50-60% of CPUs).",
    "max": "MAX Brutal Simulation: no CPU cap (100% of logical CPUs), dense shards, policy-gradient path fuzz, "
           "AI agent factory.",
}


def _key_line(key: str, value: Any, comment: str = "") -> str:
    line = f"{key} = {render_value(value)}"
    return f"{line:<40} # {comment}" if comment else line


def render_layout(values: Optional[Mapping[Location, Any]] = None, comments: bool = True) -> str:
    """The canonical ``sandwich.toml`` layout; ``values`` override the defaults location by location."""
    vals = default_locations()
    if values:
        vals.update(values)
    out: List[str] = [HEADER] if comments else []
    out.append("[meta]")
    out.append(_key_line("schema_version", vals[("meta", "schema_version")],
                         "layout version (written by the simulator)" if comments else ""))
    for table in COMMON_TABLES:
        out.append("")
        out.append(f"[{table}]")
        for s in SETTINGS:
            if s.table == table:
                out.append(_key_line(s.key, vals[s.location()], s.help if comments else ""))
    for mode in MODE_NAMES:
        out.append("")
        if comments:
            out.append(f"# {MODE_NOTES[mode]}")
        out.append(f"[modes.{mode}]")
        for s in SETTINGS:
            if s.per_mode:
                out.append(_key_line(s.key, vals[s.location(mode)], s.help if comments else ""))
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _line_index(text: str) -> Dict[Location, int]:
    """1-based line of every ``key = value`` line, by (table, key)."""
    out: Dict[Location, int] = {}
    table = ""
    for n, raw in enumerate(text.splitlines(), start=1):
        h = _HEADER_LINE.match(raw)
        if h:
            table = _norm_table(h.group(1))
            continue
        k = _KEY_LINE.match(raw)
        if k:
            out.setdefault((table, _unquote_key(k.group(2))), n)
    return out


def validate_data(data: Mapping[str, Any], text: str = "", allow_unknown: bool = False
                  ) -> Tuple[Dict[Location, Any], List[SettingsIssue]]:
    """Check parsed TOML against the schema.

    Returns ``(values, warnings)`` where ``values`` maps every location the
    file sets to its coerced value. Raises :class:`SettingsError` listing
    every problem (unknown keys only become warnings with ``allow_unknown``).
    """
    lines = _line_index(text) if text else {}
    values: Dict[Location, Any] = {}
    errors: List[SettingsIssue] = []
    warnings: List[SettingsIssue] = []

    def problem(where: Location, message: str, unknown: bool = False) -> None:
        issue = SettingsIssue(".".join(p for p in where if p), message, lines.get(where),
                              "warning" if unknown and allow_unknown else "error")
        (warnings if issue.severity == "warning" else errors).append(issue)

    for table, body in data.items():
        if table == "meta":
            if not isinstance(body, dict):
                problem(("meta", ""), "must be a table")
                continue
            for key, value in body.items():
                if key != "schema_version":
                    problem(("meta", key), f"unknown key{_suggest(key, ['schema_version'])}", unknown=True)
                elif isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    problem(("meta", key), "schema_version must be a positive integer")
                elif value > SCHEMA_VERSION:
                    problem(("meta", key), f"written by a newer simulator (schema {value} > {SCHEMA_VERSION})")
                else:
                    values[("meta", key)] = value
            continue
        if table == "modes":
            if not isinstance(body, dict):
                problem(("modes", ""), "must be a table of [modes.<mode>] tables")
                continue
            for mode, mbody in body.items():
                if mode not in MODE_NAMES:
                    problem((f"modes.{mode}", ""), f"unknown mode{_suggest(mode, MODE_NAMES)}", unknown=True)
                    continue
                if not isinstance(mbody, dict):
                    problem((f"modes.{mode}", ""), "must be a table")
                    continue
                for key, value in mbody.items():
                    _validate_one(f"modes.{mode}", key, value, values, problem)
            continue
        if table not in TABLE_KEYS:
            if table in PER_MODE_BY_KEY or (any(k == table for _, k in COMMON_BY_LOCATION)):
                problem((table, ""), f"top-level key; {_unknown_key_message(table, [])}")
            else:
                problem((table, ""), f"unknown table{_suggest(table, list(TABLE_KEYS) + ['modes', 'meta'])}",
                        unknown=True)
            continue
        if not isinstance(body, dict):
            problem((table, ""), "must be a table")
            continue
        for key, value in body.items():
            _validate_one(table, key, value, values, problem)
    if errors:
        raise SettingsError(errors)
    return values, warnings


def _validate_one(table: str, key: str, value: Any, values: Dict[Location, Any],
                  problem: Callable[..., None]) -> None:
    try:
        setting = lookup(table, key)
    except SettingsError as exc:
        problem((table, key), exc.issues[0].message, unknown=key not in PER_MODE_BY_KEY and not any(
            k == key for _, k in COMMON_BY_LOCATION))
        return
    try:
        values[(table, key)] = setting.coerce(value, f"{table}.{key}")
    except SettingsError as exc:
        problem((table, key), exc.issues[0].message)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def digest_bytes(data: Optional[bytes]) -> str:
    return "" if data is None else hashlib.sha256(data).hexdigest()


@dataclass
class SettingsDocument:
    """A loaded (or absent) settings file."""

    path: Optional[Path]
    text: str = ""
    digest: str = ""  # sha256 of the raw bytes; "" when the file does not exist
    exists: bool = False
    values: Dict[Location, Any] = field(default_factory=dict)
    warnings: List[SettingsIssue] = field(default_factory=list)
    bom: bool = False
    newline: str = "\n"

    @classmethod
    def from_bytes(cls, raw: Optional[bytes], path: Optional[Path] = None, allow_unknown: bool = False
                   ) -> "SettingsDocument":
        if raw is None:
            return cls(path)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SettingsError([SettingsIssue(str(path or "settings"), f"not valid UTF-8 ({exc.reason} at byte "
                                                                         f"{exc.start})")], path) from None
        bom = text.startswith("\ufeff")
        if bom:
            text = text[1:]
        try:
            data = parse_toml(text)
        except TomlSyntaxError as exc:
            raise SettingsError([SettingsIssue(str(path or "settings"), f"TOML syntax error: {exc}",
                                               exc.line or None)], path) from None
        try:
            values, warnings = validate_data(data, text, allow_unknown)
        except SettingsError as exc:
            raise SettingsError(exc.issues, path) from None
        return cls(path, text, digest_bytes(raw), True, values, warnings, bom, "\r\n" if "\r\n" in text else "\n")

    @classmethod
    def from_text(cls, text: str, path: Optional[Path] = None) -> "SettingsDocument":
        return cls.from_bytes(text.encode("utf-8"), path)

    @property
    def mode(self) -> Optional[str]:
        return self.values.get(("run", "mode"))

    def get(self, setting: Setting, mode: Optional[str] = None, default: Any = None) -> Any:
        return self.values.get(setting.location(mode if setting.per_mode else None), default)

    def mode_table(self, mode: str) -> Dict[str, Any]:
        return {key: v for (table, key), v in self.values.items() if table == f"modes.{mode}"}


# ---------------------------------------------------------------------------
# Format-preserving updates
# ---------------------------------------------------------------------------

_HEADER_LINE = re.compile(r"^\s*\[\s*([A-Za-z0-9_\-]+(?:\s*\.\s*[A-Za-z0-9_\-]+)*)\s*\]\s*(?:#.*)?$")
_KEY_LINE = re.compile(r'^(\s*)([A-Za-z0-9_\-]+|"[^"\\]*")(\s*=\s*)(.*)$')


def _norm_table(name: str) -> str:
    return ".".join(p.strip() for p in name.split("."))


def _unquote_key(key: str) -> str:
    return key[1:-1] if key.startswith('"') else key


def _value_end(rest: str) -> Optional[int]:
    """Length of the single-line value at the start of ``rest`` (None if it continues past the line)."""
    if rest.startswith('"""') or rest.startswith("'''"):
        return None
    if rest.startswith('"'):
        i = 1
        while i < len(rest):
            if rest[i] == "\\":
                i += 2
                continue
            if rest[i] == '"':
                return i + 1
            i += 1
        return None
    if rest.startswith("'"):
        j = rest.find("'", 1)
        return j + 1 if j >= 0 else None
    if rest.startswith("[") or rest.startswith("{"):
        depth = 0
        i = 0
        quote = ""
        while i < len(rest):
            c = rest[i]
            if quote:
                if c == "\\" and quote == '"':
                    i += 2
                    continue
                if c == quote:
                    quote = ""
            elif c in "\"'":
                quote = c
            elif c in "[{":
                depth += 1
            elif c in "]}":
                depth -= 1
                if depth == 0:
                    return i + 1
            elif c == "#":
                return None
            i += 1
        return None
    j = rest.find("#")
    v = rest if j < 0 else rest[:j]
    return len(v.rstrip())


@dataclass
class _Layout:
    lines: List[str]  # without line endings
    tables: Dict[str, Tuple[int, int]]  # table -> (header line index or -1 for root, last content line index)
    keys: Dict[Location, int]


def _scan_layout(text: str) -> _Layout:
    lines = text.splitlines()
    tables: Dict[str, Tuple[int, int]] = {"": (-1, -1)}
    keys: Dict[Location, int] = {}
    table = ""
    for n, raw in enumerate(lines):
        h = _HEADER_LINE.match(raw)
        if h:
            table = _norm_table(h.group(1))
            if table not in tables:
                tables[table] = (n, n)
            continue
        k = _KEY_LINE.match(raw)
        if k:
            keys.setdefault((table, _unquote_key(k.group(2))), n)
            header, _ = tables.get(table, (n, n))
            tables[table] = (header, n)
    return _Layout(lines, tables, keys)


def apply_patch(text: str, patch: Mapping[Location, Any], comments: bool = True) -> str:
    """Set ``patch`` values in ``text``, editing only the value text of existing keys.

    Missing keys are inserted after the last key of their table; missing
    tables are appended in canonical order. Line endings, a trailing newline,
    comments and key order are preserved.
    """
    newline = "\r\n" if "\r\n" in text else "\n"
    layout = _scan_layout(text)
    lines = list(layout.lines)
    inserts: Dict[int, List[str]] = {}
    appended: Dict[str, List[str]] = {}
    for (table, key), value in patch.items():
        setting_help = ""
        if table != "meta":
            setting_help = lookup(table, key).help if comments else ""
        idx = layout.keys.get((table, key))
        if idx is not None:
            m = _KEY_LINE.match(lines[idx])
            assert m is not None
            indent, k, eq, rest = m.groups()
            end = _value_end(rest)
            if end is None:
                raise SettingsError([SettingsIssue(f"{table}.{key}", "cannot update a multi-line value in place",
                                                   idx + 1)])
            head = f"{indent}{k}{eq}{render_value(value)}"
            tail = rest[end:]
            stripped = tail.lstrip(" \t")
            if stripped.startswith("#") and tail != stripped:
                # keep a trailing comment in its column when the new value fits in front of it
                column = len(indent) + len(k) + len(eq) + end + (len(tail) - len(stripped))
                tail = " " * max(1, column - len(head)) + stripped
            lines[idx] = head + tail
            continue
        new_line = _key_line(key, value, setting_help)
        if table in layout.tables and table != "":
            _, last = layout.tables[table]
            inserts.setdefault(last, []).append(new_line)
        else:
            appended.setdefault(table, []).append(new_line)
    out: List[str] = []
    for n, line in enumerate(lines):
        out.append(line)
        out.extend(inserts.get(n, []))
    order = ["meta"] + list(COMMON_TABLES) + [f"modes.{m}" for m in MODE_NAMES]
    for table in sorted(appended, key=lambda t: (order.index(t) if t in order else len(order), t)):
        while out and out[-1].strip() == "":
            out.pop()
        if out:
            out.append("")
        if comments and table.startswith("modes.") and table.split(".", 1)[1] in MODE_NOTES:
            out.append(f"# {MODE_NOTES[table.split('.', 1)[1]]}")
        out.append(f"[{table}]")
        out.extend(appended[table])
    result = newline.join(out)
    if out and (text.endswith("\n") or not text):
        result += newline
    return result


# ---------------------------------------------------------------------------
# File operations (injectable for fault tests)
# ---------------------------------------------------------------------------


class FileOps:
    """Thin wrapper over the OS calls :class:`SettingsStore` makes."""

    def read_bytes(self, path: Path) -> Optional[bytes]:
        try:
            with open(path, "rb") as fh:
                return fh.read()
        except FileNotFoundError:
            return None

    def write_new(self, path: Path, data: bytes) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
        try:
            view = memoryview(data)
            while view:
                n = os.write(fd, view)
                view = view[n:]
            os.fsync(fd)
        finally:
            os.close(fd)

    def replace(self, src: Path, dst: Path) -> None:
        os.replace(src, dst)

    def remove(self, path: Path) -> None:
        os.remove(path)

    def stat(self, path: Path) -> Optional[os.stat_result]:
        try:
            return os.stat(path)
        except FileNotFoundError:
            return None

    def chmod(self, path: Path, mode: int) -> None:
        os.chmod(path, mode)

    def fsync_dir(self, path: Path) -> None:
        if os.name != "posix":
            return
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def create_lock(self, path: Path, payload: bytes) -> bool:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return False
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        return True

    def read_lock(self, path: Path) -> Optional[Tuple[str, float]]:
        """Lock payload and age in seconds, or None if it vanished."""
        try:
            with open(path, "rb") as fh:
                payload = fh.read(256).decode("utf-8", "replace")
            age = time.time() - os.stat(path).st_mtime
        except FileNotFoundError:
            return None
        return payload, age

    def pid_alive(self, pid: int) -> Optional[bool]:
        if os.name != "posix" or pid <= 0:
            return None  # os.kill(pid, 0) would terminate the process on Windows
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return None
        return True

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def now(self) -> float:
        return time.monotonic()


def _is_sharing_violation(exc: OSError) -> bool:
    return isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in (5, 32, 33)


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


@dataclass
class SyncReport:
    path: str
    created: bool
    added: List[str]
    unknown: List[str]
    written: bool
    digest: str

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "created": self.created, "added": list(self.added),
                "unknown": list(self.unknown), "written": self.written, "digest": self.digest}


class SettingsStore:
    """Locked, conflict-checked, atomic reads and writes of one settings file."""

    def __init__(self, path: Any, *, lock_timeout: float = 10.0, stale_lock_s: float = 120.0, backup: bool = True,
                 ops: Optional[FileOps] = None, replace_retries: int = 5) -> None:
        given = Path(path)
        self.link = given
        self.path = Path(os.path.realpath(given)) if given.is_symlink() else given
        self.lock_timeout = lock_timeout
        self.stale_lock_s = stale_lock_s
        self.backup = backup
        self.ops = ops or FileOps()
        self.replace_retries = max(1, replace_retries)
        self._mutex = threading.RLock()
        self._depth = 0

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    @property
    def backup_path(self) -> Path:
        return self.path.with_name(self.path.name + ".bak")

    # -- locking -----------------------------------------------------------------

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Exclusive writer lock (re-entrant within one store instance)."""
        with self._mutex:
            if self._depth:
                self._depth += 1
                try:
                    yield
                finally:
                    self._depth -= 1
                return
            self._acquire()
            self._depth = 1
            try:
                yield
            finally:
                self._depth = 0
                try:
                    self.ops.remove(self.lock_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass

    def _acquire(self) -> None:
        ops = self.ops
        deadline = ops.now() + max(0.0, self.lock_timeout)
        delay = 0.002
        payload = f"{os.getpid()} {socket.gethostname()} {time.time():.3f}\n".encode()
        while True:
            try:
                if ops.create_lock(self.lock_path, payload):
                    return
            except FileNotFoundError as exc:
                raise SettingsLockError(f"directory {self.path.parent} does not exist") from exc
            except OSError as exc:
                raise SettingsLockError(f"cannot create lock {self.lock_path}: {exc}") from exc
            if self._stale():
                try:
                    ops.remove(self.lock_path)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    raise SettingsLockError(f"cannot break stale lock {self.lock_path}: {exc}") from exc
                continue
            if ops.now() >= deadline:
                raise SettingsLockError(f"{self.lock_path} is held by another writer "
                                        f"(waited {self.lock_timeout:.1f}s)")
            ops.sleep(delay)
            delay = min(0.05, delay * 2)

    def _stale(self) -> bool:
        info = self.ops.read_lock(self.lock_path)
        if info is None:
            return False
        payload, age = info
        if age > self.stale_lock_s:
            return True
        parts = payload.split()
        if len(parts) >= 2 and parts[1] == socket.gethostname():
            try:
                pid = int(parts[0])
            except ValueError:
                return age > 1.0  # unreadable payload: treat a lock that is not brand new as abandoned
            if pid != os.getpid():
                return self.ops.pid_alive(pid) is False
        return False

    # -- reading -----------------------------------------------------------------

    def read_bytes(self) -> Optional[bytes]:
        try:
            return self.ops.read_bytes(self.path)
        except IsADirectoryError:
            raise SettingsFileError(f"{self.path} is a directory") from None
        except OSError as exc:
            raise SettingsFileError(f"cannot read {self.path}: {exc}") from exc

    def load(self, allow_unknown: bool = False) -> SettingsDocument:
        return SettingsDocument.from_bytes(self.read_bytes(), self.path, allow_unknown)

    # -- writing -----------------------------------------------------------------

    def _check_conflict(self, current: Optional[bytes], expected_digest: Optional[str], force: bool) -> None:
        if expected_digest is None or force:
            return
        now = digest_bytes(current)
        if now != expected_digest:
            state = "was deleted" if current is None else "changed on disk"
            raise SettingsConflictError(f"{self.path} {state} since it was loaded "
                                        f"(expected {expected_digest[:12] or 'no file'}, found {now[:12] or 'no file'})"
                                        "; reload it or pass force=True")

    def _tmp_path(self, target: Path) -> Path:
        return target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")

    def _atomic_write(self, target: Path, data: bytes, mode: Optional[int]) -> None:
        ops = self.ops
        tmp = self._tmp_path(target)
        try:
            ops.write_new(tmp, data)
            if ops.read_bytes(tmp) != data:
                raise SettingsWriteError(f"short or corrupted write to {tmp}")
            if mode is not None:
                ops.chmod(tmp, mode)
            for attempt in range(self.replace_retries):
                try:
                    ops.replace(tmp, target)
                    break
                except OSError as exc:
                    if attempt + 1 >= self.replace_retries or not _is_sharing_violation(exc):
                        raise
                    ops.sleep(0.01 * (2 ** attempt))
        except SettingsFileError:
            self._discard(tmp)
            raise
        except OSError as exc:
            self._discard(tmp)
            raise SettingsWriteError(f"could not write {target}: {exc}; the previous file is unchanged") from exc
        try:
            ops.fsync_dir(target.parent)  # durability of the rename; the new file is already in place
        except OSError:
            pass

    def _discard(self, tmp: Path) -> None:
        try:
            if self.ops.stat(tmp) is not None:
                self.ops.remove(tmp)
        except OSError:
            pass

    def _commit(self, current: Optional[bytes], new_text: str, bom: bool, allow_unknown: bool = False) -> bytes:
        """Validate ``new_text`` and atomically replace the file with it; returns the bytes written.

        Unknown keys are tolerated only with ``allow_unknown`` (sync and
        update keep keys they do not own); invalid values never are.
        """
        # raises SettingsError before anything touches the disk
        SettingsDocument.from_bytes(new_text.encode("utf-8"), self.path, allow_unknown)
        data = (("\ufeff" if bom else "") + new_text).encode("utf-8")
        if current == data:
            return data
        st = self.ops.stat(self.path)
        if st is not None and stat_mod.S_ISDIR(st.st_mode):
            raise SettingsWriteError(f"{self.path} is a directory")
        mode = stat_mod.S_IMODE(st.st_mode) if st is not None else None
        if current is not None and self.backup:
            self._atomic_write(self.backup_path, current, mode)
        self._atomic_write(self.path, data, mode)
        return data

    def save_text(self, text: str, expected_digest: Optional[str] = None, force: bool = False) -> SettingsDocument:
        """Replace the whole file with ``text`` (validated first)."""
        with self.lock():
            current = self.read_bytes()
            self._check_conflict(current, expected_digest, force)
            bom = current is not None and current.startswith(b"\xef\xbb\xbf")
            data = self._commit(current, text, bom)
            return SettingsDocument.from_bytes(data, self.path)

    def update(self, patch: Mapping[Location, Any], expected_digest: Optional[str] = None, force: bool = False,
               create: bool = True) -> SettingsDocument:
        """Set ``patch`` values (format-preserving) under the lock and write the file atomically."""
        clean: Dict[Location, Any] = {}
        for (table, key), value in patch.items():
            if table == "meta":
                clean[(table, key)] = value
                continue
            setting = lookup(table, key)
            clean[(table, key)] = setting.coerce(value, f"{table}.{key}")
        with self.lock():
            current = self.read_bytes()
            self._check_conflict(current, expected_digest, force)
            if current is None:
                if not create:
                    raise SettingsFileError(f"{self.path} does not exist")
                base, bom = render_layout(), False
            else:
                doc = SettingsDocument.from_bytes(current, self.path, allow_unknown=True)
                base, bom = doc.text, doc.bom
            data = self._commit(current, apply_patch(base, clean), bom, allow_unknown=True)
            return SettingsDocument.from_bytes(data, self.path, allow_unknown=True)

    def init(self, force: bool = False, values: Optional[Mapping[Location, Any]] = None) -> SettingsDocument:
        """Write the full default layout; refuses to overwrite an existing file unless ``force``."""
        with self.lock():
            current = self.read_bytes()
            if current is not None and not force:
                raise SettingsFileError(f"{self.path} already exists (use force to overwrite)")
            data = self._commit(current, render_layout(values), False)
            return SettingsDocument.from_bytes(data, self.path)

    def sync(self, write: bool = True) -> SyncReport:
        """Add every key the schema knows but the file lacks; report keys the schema does not know."""
        with self.lock():
            current = self.read_bytes()
            if current is None:
                text = render_layout()
                digest = ""
                if write:
                    digest = digest_bytes(self._commit(None, text, False))
                return SyncReport(str(self.path), True, sorted(f"{t}.{k}" for t, k in default_locations()), [],
                                  write, digest)
            doc = SettingsDocument.from_bytes(current, self.path, allow_unknown=True)
            missing = {loc: v for loc, v in default_locations().items() if loc not in doc.values}
            unknown = sorted(i.location for i in doc.warnings)
            written = False
            digest = doc.digest
            if missing and write:
                digest = digest_bytes(self._commit(current, apply_patch(doc.text, missing), doc.bom,
                                                   allow_unknown=True))
                written = True
            return SyncReport(str(self.path), False, sorted(f"{t}.{k}" for t, k in missing), unknown, written,
                              digest)


# ---------------------------------------------------------------------------
# Discovery and resolution
# ---------------------------------------------------------------------------


def discover_config(start: Optional[Any] = None) -> Optional[Path]:
    """``$MEV_SIM_CONFIG`` (``none`` disables), else the nearest sandwich.toml up to the repository root."""
    env = os.environ.get(CONFIG_ENV_VAR)
    if env is not None:
        env = env.strip()
        if env.lower() in ("", "none", "off", "0", "false"):
            return None
        return Path(env)
    here = Path(start if start is not None else os.getcwd()).resolve()
    for d in (here, *here.parents):
        candidate = d / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
        if (d / ".git").exists():
            break
    return None


def patch_from_attrs(values: Mapping[str, Any], mode: str) -> Dict[Location, Any]:
    """Map ``attr -> value`` (e.g. from CLI flags) to TOML locations for ``mode``."""
    out: Dict[Location, Any] = {}
    for attr, value in values.items():
        setting = BY_ATTR.get(attr)
        if setting is None:
            raise SettingsError([SettingsIssue(attr, "unknown setting")])
        out[setting.location(mode)] = setting.coerce(value, setting.dotted(mode))
    return out


def parse_assignment(text: str) -> Tuple[Location, Any]:
    """``table.key=value`` or ``modes.<mode>.key=value`` (``--set``) -> (location, value)."""
    if "=" not in text:
        raise SettingsError([SettingsIssue(text, "expected KEY=VALUE")])
    dotted, raw = text.split("=", 1)
    parts = [p.strip() for p in dotted.strip().split(".")]
    if len(parts) == 3 and parts[0] == "modes":
        table, key = f"modes.{parts[1]}", parts[2]
    elif len(parts) == 2:
        table, key = parts
    else:
        raise SettingsError([SettingsIssue(dotted, "use table.key or modes.<mode>.key")])
    setting = lookup(table, key)
    return (table, key), setting.parse_text(raw, f"{table}.{key}")


def effective_mode(file_values: Mapping[Location, Any], cli_mode: Optional[str] = None) -> str:
    if cli_mode is not None:
        return _norm_mode(cli_mode)
    return str(file_values.get(("run", "mode"), RunMode.MODERATE_SCAN.short))


def resolve_config(file_values: Optional[Mapping[Location, Any]] = None,
                   cli_values: Optional[Mapping[Location, Any]] = None,
                   cpus: Optional[int] = None) -> SimulationConfig:
    """Build a SimulationConfig: preset < file < $MEV_SIM_WORKERS (workers only) < CLI."""
    file_values = dict(file_values or {})
    cli_values = dict(cli_values or {})
    merged = dict(file_values)
    merged.update(cli_values)
    mode = effective_mode(merged)
    table = f"modes.{mode}"
    overrides: Dict[str, Any] = {}
    for (tbl, key), value in merged.items():
        if tbl == "meta":
            continue
        setting = lookup(tbl, key)
        if setting.attr == "mode" or (setting.per_mode and tbl != table):
            continue
        overrides[setting.attr] = value
    file_workers = int(file_values.get((table, "workers"), 0) or 0)
    cli_workers = cli_values.get((table, "workers"))
    overrides.pop("workers", None)
    if cli_workers:
        requested: Optional[int] = int(cli_workers)
    elif env_worker_request() is not None:
        requested = None
    elif file_workers > 0:
        requested = file_workers
    else:
        requested = None
    return SimulationConfig.for_mode(mode, workers=requested, cpus=cpus, **overrides)


def config_locations(cfg: SimulationConfig) -> Dict[Location, Any]:
    """The TOML locations a resolved config corresponds to (its mode's table only)."""
    mode = cfg.mode.short
    out: Dict[Location, Any] = {("run", "mode"): mode}
    for s in SETTINGS:
        if s.attr == "mode":
            continue
        out[s.location(mode)] = getattr(cfg, s.attr)
    return out


def render_effective(cfg: SimulationConfig, file_values: Optional[Mapping[Location, Any]] = None) -> str:
    """``--show-config`` output: the layout with the effective values of the active mode."""
    values: Dict[Location, Any] = dict(file_values or {})
    values.update(config_locations(cfg))
    head = (f"# Effective settings: {cfg.mode.label}, {cfg.workers} worker(s) under the "
            f"{cfg.cpu_policy} CPU policy.\n")
    return head + render_layout(values)


# ---------------------------------------------------------------------------
# CLI and GUI views
# ---------------------------------------------------------------------------


def add_setting_arguments(parser: argparse.ArgumentParser) -> None:
    """One flag per setting (defaults None so that only flags actually given count)."""
    groups: Dict[str, Any] = {}
    for s in SETTINGS:
        title = TABLE_TITLES["modes" if s.per_mode else s.table]
        group = groups.setdefault(title, parser.add_argument_group(title))
        kwargs: Dict[str, Any] = {"dest": s.dest, "default": None, "help": s.help}
        if s.kind is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = _cli_type(s)
            kwargs["metavar"] = s.key.upper() if not s.choices else "{" + ",".join(s.choices) + "}"
            if s.choices and not s.lenient_choice:
                kwargs["choices"] = s.choices
                kwargs.pop("metavar")
        group.add_argument(*s.flags, **kwargs)


def _cli_type(s: Setting) -> Callable[[str], Any]:
    def convert(text: str) -> Any:
        try:
            return s.parse_text(text, s.flags[0])
        except SettingsError as exc:
            raise argparse.ArgumentTypeError(exc.issues[0].message) from None
    convert.__name__ = s.kind.__name__
    return convert


def values_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """``attr -> value`` for every setting flag the user actually passed."""
    out: Dict[str, Any] = {}
    for s in SETTINGS:
        value = getattr(args, s.dest, None)
        if value is not None:
            out[s.attr] = value
    return out


def form_schema() -> Dict[str, Any]:
    """Description of every setting for a GUI form: common sections plus the per-mode matrix."""
    sections = []
    for table in COMMON_TABLES:
        sections.append({
            "table": table,
            "title": TABLE_TITLES[table],
            "fields": [s.to_form(default_value(s)) for s in SETTINGS if s.table == table],
        })
    per_mode = [s for s in SETTINGS if s.per_mode]
    return {
        "schema_version": SCHEMA_VERSION,
        "file": CONFIG_FILENAME,
        "precedence": ["mode preset", CONFIG_FILENAME, "$MEV_SIM_WORKERS (workers only)", "command line"],
        "sections": sections,
        "modes": {
            "title": TABLE_TITLES["modes"],
            "columns": [{"mode": m, "label": RunMode.parse(m).label, "note": MODE_NOTES[m]} for m in MODE_NAMES],
            "fields": [s.to_form(None) for s in per_mode],
            "presets": {m: {s.key: default_value(s, m) for s in per_mode} for m in MODE_NAMES},
        },
    }


__all__ = [
    "CONFIG_ENV_VAR",
    "CONFIG_FILENAME",
    "MODE_NAMES",
    "SCHEMA_VERSION",
    "SETTINGS",
    "FileOps",
    "MiniToml",
    "Setting",
    "SettingsConflictError",
    "SettingsDocument",
    "SettingsError",
    "SettingsFileError",
    "SettingsIssue",
    "SettingsLockError",
    "SettingsStore",
    "SettingsWriteError",
    "SyncReport",
    "TomlSyntaxError",
    "add_setting_arguments",
    "apply_patch",
    "config_locations",
    "default_locations",
    "default_value",
    "digest_bytes",
    "discover_config",
    "effective_mode",
    "form_schema",
    "lookup",
    "parse_assignment",
    "parse_toml",
    "patch_from_attrs",
    "render_effective",
    "render_layout",
    "render_value",
    "resolve_config",
    "validate_data",
    "values_from_args",
]
