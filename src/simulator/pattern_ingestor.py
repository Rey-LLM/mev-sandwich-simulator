"""Layer 12: MEV pattern ingestion (local pattern store, normalisers, slippage evaluation).

Status: experimental. The normalisers follow the documented field names of the
source formats but have only been exercised on the built-in synthetic corpus
(records marked ``source="synthetic"``); no real export ships with the repo.

The ingestor turns MEV attack records from public data exports into one
normalised, content-addressed :class:`PatternRecord` format and keeps them in
a local on-disk store that the Layer 11 AI agent factory reads from. Nothing
is fetched from the network: records come from files you download yourself
(or from the deterministic synthetic reference corpus).

Sources (:func:`normalize`)
---------------------------
* ``flashbots_blocks``: block objects as the Flashbots blocks API returns them
  (``block_number``, ``miner_reward``, ``transactions`` with ``bundle_index``,
  ``eoa_address``, ``coinbase_transfer``, ``gas_price`` ...). A bundle of three
  or more transactions whose first and last transaction come from the same EOA
  and wrap a different sender is read as a sandwich; its coinbase transfers
  become the bribe. The blocks API carries no swap data, so hops, pools and
  slippage are left at their neutral values.
* ``mev_inspect``: rows of the mev-inspect-py ``sandwiches`` table
  (``frontrun_swap_transaction_hash``, ``sandwiched_swaps``, ``profit_amount``),
  optionally with the swap objects (``frontrun_swap``, ``backrun_swap``) that
  carry pool addresses and amounts.
* ``advisory``: security-advisory entries (``advisory_id``, ``title``,
  ``chain_id`` / ``dst_chain_id``, and an ``attack`` object with normalised
  fields).
* ``normalized``: rows that already use the :class:`PatternRecord` field names
  (JSON, JSONL or CSV), e.g. a previous export of this store.
* ``synthetic``: :func:`synthetic_corpus`, a seeded reference corpus. It is
  generated, not observed, and every record says so in its ``source``.

Categories (:data:`CATEGORIES`)
-------------------------------
Every record lands in exactly one of four matrices: ``multi_hop_frontrun``,
``jit_liquidity``, ``multi_pool_sandwich`` and ``cross_chain_drift``. An
explicit category wins; otherwise :func:`classify` decides, in this order:
a destination chain that differs from the source chain or a non-zero state
drift -> cross-chain; JIT liquidity added -> JIT; two or more hops -> multi-hop
front-run; anything else (a sandwich on one or more pools) -> multi-pool.

Store (:class:`PatternStore`)
-----------------------------
A directory with an SQLite index (``index.sqlite``, WAL mode) and append-only
segment files (``segments/seg-NNNNNN.bin``). Each record is one frame:
``b"MEVP"``, payload length and CRC-32 (big-endian ``uint32`` each), then the
zlib-compressed canonical JSON. The index row holds the record's SHA-256
(unique, so re-ingesting a file adds nothing), every numeric field the
category matrices need, and the frame's segment, offset and length, so
matrix queries never touch the segments. Segments rotate at
``segment_bytes``; the store refuses any write that would push segments plus
index above ``max_storage_bytes`` (500 GiB by default) and raises
:class:`StorageFullError` before writing anything. Writes run inside an
``IMMEDIATE`` SQLite transaction, so concurrent writers (threads or
processes) serialise on the database lock.

Reads go through a byte-budgeted LRU cache (:class:`ByteLRU`) of decoded
records and positional reads (``os.pread`` where available). Bulk reads sort
frames by (segment, offset) and stream each segment in ``chunk_bytes`` chunks
(:meth:`PatternStore.iter_segment`), so a scan is sequential IO.
:func:`parallel_scan` splits the id range into fixed shards (a function of
``shards`` only, never of the worker count, so results are identical on one
worker or on every core) and runs one shard per worker task on a
:class:`~simulator.parallel.WorkerPool`; each task opens the store read-only
in its own process.

Slippage evaluation (:func:`evaluate_pattern`)
----------------------------------------------
Each record is replayed on a constant-product pool (reserves from the record,
or a nominal 1,000 x 1,000 pool scaled to it) with the exact integer v2
formula: victim quote without the front-run, victim output after it, the
back-run, and a binary search for the largest front-run the victim's minimum
output still tolerates. The evaluation reports realised slippage, whether the
victim's limit held, extraction in bps of the victim's input (zero when the
victim's limit would have reverted the swap), how much of the tolerated front-run the attacker used, and a training weight that the
factory uses (:func:`pattern_weight`).

Limits: classification is heuristic; the blocks API and many advisories do
not carry swap amounts, so those records are evaluated on the nominal pool;
the store has one writer lock and no delete/compaction.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import sqlite3
import struct
import threading
import time
import zlib
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from .config import ConfigError
from .parallel import WorkerPool, chunked
from .slippage_engine import v2_get_amount_out

E18 = 10**18
GWEI = 10**9
BPS = 10_000
KIB = 1 << 10
MIB = 1 << 20
GIB = 1 << 30

CATEGORIES: Tuple[str, ...] = ("multi_hop_frontrun", "jit_liquidity", "multi_pool_sandwich", "cross_chain_drift")
SOURCES: Tuple[str, ...] = ("flashbots_blocks", "mev_inspect", "advisory", "normalized", "synthetic")
MATRIX_FIELDS: Tuple[str, ...] = (
    "hops",
    "pools",
    "victim_tolerance_bps",
    "victim_slippage_bps",
    "frontrun_impact_bps",
    "extraction_bps",
    "bribe_bps",
    "priority_fee_gwei",
    "latency_ms",
    "capital_milli_eth",
    "drift_blocks",
    "jit_share_bps",
)

SCHEMA_VERSION = 1
MAGIC = b"MEVP"
HEADER = struct.Struct(">4sII")  # magic, payload length, crc32
INDEX_FILE = "index.sqlite"
SEGMENT_DIR = "segments"
DEFAULT_MAX_STORAGE_BYTES = 500 * GIB
DEFAULT_SEGMENT_BYTES = 64 * MIB
DEFAULT_CACHE_BYTES = 256 * MIB
DEFAULT_CHUNK_BYTES = 1 * MIB
DEFAULT_BATCH = 512
DEFAULT_SCAN_SHARDS = 32
MAX_HOPS = 8
MAX_POOLS = 16
MAX_LATENCY_MS = 60_000
MAX_WEI = (1 << 256) - 1
NOMINAL_RESERVE_WEI = 1_000 * E18


class PatternError(ConfigError):
    """A pattern record or a pattern store operation is invalid."""


class StorageFullError(PatternError):
    """A write would push the store above its storage cap."""


class CorruptFrameError(PatternError):
    """A segment frame failed its magic, length or CRC check."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


@dataclass(frozen=True)
class PatternRecord:
    """One normalised MEV attack pattern (integers are wei, bps, ms or block counts)."""

    category: str
    source: str
    chain_id: int = 1
    block_number: int = 0
    hops: int = 1
    pools: int = 1
    victim_tolerance_bps: int = 50
    victim_slippage_bps: int = 0
    frontrun_impact_bps: int = 0
    victim_size_bps: int = 0
    reserve_in_wei: int = 0
    extracted_wei: int = 0
    bribe_bps: int = 0
    priority_fee_wei: int = 0
    latency_ms: int = 0
    capital_wei: int = 0
    dst_chain_id: int = 0
    drift_blocks: int = 0
    jit_liquidity_wei: int = 0
    tx_hashes: Tuple[str, ...] = ()
    reference: str = ""

    def __post_init__(self) -> None:
        issues = record_issues(self)
        if issues:
            raise PatternError(f"invalid pattern record: {'; '.join(issues)}")

    @property
    def jit_share_bps(self) -> int:
        """JIT liquidity as bps of the pool's input reserve (0 without either)."""
        if self.jit_liquidity_wei <= 0:
            return 0
        reserve = self.reserve_in_wei or NOMINAL_RESERVE_WEI
        return min(BPS, self.jit_liquidity_wei * BPS // reserve)

    def canonical(self) -> Dict[str, Any]:
        """Every field, big integers as decimal strings, keys sorted (the hashed form)."""
        out: Dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "tx_hashes":
                v = list(v)
            elif _is_int(v) and abs(v) >= 1 << 53:
                v = str(v)
            out[f.name] = v
        return dict(sorted(out.items()))

    def to_bytes(self) -> bytes:
        return json.dumps(self.canonical(), separators=(",", ":"), sort_keys=True).encode()

    def digest(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_canonical(cls, data: Mapping[str, Any]) -> "PatternRecord":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise PatternError(f"unknown pattern fields: {', '.join(unknown)}")
        kw: Dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            v = data[f.name]
            if f.name == "tx_hashes":
                v = tuple(str(x) for x in v)
            elif f.name not in ("category", "source", "reference"):
                v = _as_int(v, f.name)
            kw[f.name] = v
        return cls(**kw)

    @classmethod
    def from_bytes(cls, payload: bytes) -> "PatternRecord":
        return cls.from_canonical(json.loads(payload.decode()))

    def matrix_row(self) -> Tuple[int, ...]:
        """The record's :data:`MATRIX_FIELDS` (extraction from :func:`evaluate_pattern`)."""
        ev = evaluate_pattern(self)
        return (self.hops, self.pools, self.victim_tolerance_bps, self.victim_slippage_bps,
                self.frontrun_impact_bps, ev.extraction_bps, self.bribe_bps, self.priority_fee_wei // GWEI,
                self.latency_ms, self.capital_wei // 10**15, self.drift_blocks, self.jit_share_bps)


def _as_int(v: Any, name: str = "value") -> int:
    if _is_int(v):
        return v
    if isinstance(v, float):
        if not math.isfinite(v) or v != math.floor(v):
            raise PatternError(f"{name}: {v!r} is not a whole number")
        return int(v)
    if isinstance(v, str):
        s = v.strip().replace("_", "")
        if s.lower().startswith("0x"):
            try:
                return int(s, 16)
            except ValueError as exc:
                raise PatternError(f"{name}: {v!r} is not an integer") from exc
        if s.lstrip("+-").isdigit():
            return int(s)
        try:
            f = float(s)
        except ValueError as exc:
            raise PatternError(f"{name}: {v!r} is not an integer") from exc
        return _as_int(f, name)
    raise PatternError(f"{name}: {v!r} is not an integer")


def record_issues(r: Any) -> List[str]:
    """Every bound a :class:`PatternRecord` violates (empty when valid)."""
    issues: List[str] = []
    if r.category not in CATEGORIES:
        issues.append(f"category {r.category!r} is unknown")
    if r.source not in SOURCES:
        issues.append(f"source {r.source!r} is unknown")
    ints = ("chain_id", "block_number", "hops", "pools", "victim_tolerance_bps", "victim_slippage_bps",
            "frontrun_impact_bps", "victim_size_bps", "reserve_in_wei", "extracted_wei", "bribe_bps",
            "priority_fee_wei", "latency_ms", "capital_wei", "dst_chain_id", "drift_blocks", "jit_liquidity_wei")
    bad = [n for n in ints if not _is_int(getattr(r, n))]
    if bad:
        issues.append(f"not integers: {', '.join(bad)}")
        return issues
    if r.chain_id < 1 or r.dst_chain_id < 0 or r.block_number < 0:
        issues.append("chain_id must be >= 1, dst_chain_id and block_number >= 0")
    if not 1 <= r.hops <= MAX_HOPS:
        issues.append(f"hops must be within [1, {MAX_HOPS}]")
    if not 1 <= r.pools <= MAX_POOLS:
        issues.append(f"pools must be within [1, {MAX_POOLS}]")
    for n in ("victim_tolerance_bps", "victim_slippage_bps", "frontrun_impact_bps", "victim_size_bps", "bribe_bps"):
        if not 0 <= getattr(r, n) <= BPS:
            issues.append(f"{n} must be within [0, 10000]")
    for n in ("reserve_in_wei", "extracted_wei", "priority_fee_wei", "capital_wei", "jit_liquidity_wei"):
        if not 0 <= getattr(r, n) <= MAX_WEI:
            issues.append(f"{n} must be a uint256")
    if not 0 <= r.latency_ms <= MAX_LATENCY_MS:
        issues.append(f"latency_ms must be within [0, {MAX_LATENCY_MS}]")
    if not 0 <= r.drift_blocks <= 10_000:
        issues.append("drift_blocks must be within [0, 10000]")
    if not isinstance(r.tx_hashes, tuple) or not all(isinstance(h, str) for h in r.tx_hashes):
        issues.append("tx_hashes must be a tuple of strings")
    if not isinstance(r.reference, str):
        issues.append("reference must be a string")
    return issues


def classify(hops: int, pools: int, chain_id: int = 1, dst_chain_id: int = 0, drift_blocks: int = 0,
             jit_liquidity_wei: int = 0) -> str:
    """The category of an unlabelled record (see the module docstring for the order)."""
    if (dst_chain_id and dst_chain_id != chain_id) or drift_blocks > 0:
        return "cross_chain_drift"
    if jit_liquidity_wei > 0:
        return "jit_liquidity"
    if hops >= 2:
        return "multi_hop_frontrun"
    return "multi_pool_sandwich"


# ---------------------------------------------------------------------------
# Normalisers
# ---------------------------------------------------------------------------


def detect_source(raw: Mapping[str, Any]) -> str:
    """Which :data:`SOURCES` parser a raw object needs."""
    if not isinstance(raw, Mapping):
        raise PatternError(f"expected an object, got {type(raw).__name__}")
    if "transactions" in raw and ("miner_reward" in raw or "block_number" in raw):
        return "flashbots_blocks"
    if "frontrun_swap_transaction_hash" in raw or "sandwiched_swaps" in raw:
        return "mev_inspect"
    if "advisory_id" in raw:
        return "advisory"
    if raw.get("source") == "synthetic":
        return "synthetic"
    if "category" in raw or "hops" in raw:
        return "normalized"
    raise PatternError(f"cannot tell the source of an object with keys {sorted(raw)[:8]}")


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _get_int(raw: Mapping[str, Any], key: str, default: int = 0) -> int:
    v = raw.get(key)
    if v is None or v == "":
        return default
    return _as_int(v, key)


def _finish(category: Optional[str], source: str, **kw: Any) -> PatternRecord:
    kw["hops"] = _clamp(kw.get("hops", 1), 1, MAX_HOPS)
    kw["pools"] = _clamp(kw.get("pools", 1), 1, MAX_POOLS)
    for n in ("victim_tolerance_bps", "victim_slippage_bps", "frontrun_impact_bps", "victim_size_bps", "bribe_bps"):
        if n in kw:
            kw[n] = _clamp(kw[n], 0, BPS)
    if "latency_ms" in kw:
        kw["latency_ms"] = _clamp(kw["latency_ms"], 0, MAX_LATENCY_MS)
    cat = category or classify(kw["hops"], kw["pools"], kw.get("chain_id", 1), kw.get("dst_chain_id", 0),
                               kw.get("drift_blocks", 0), kw.get("jit_liquidity_wei", 0))
    return PatternRecord(category=cat, source=source, **kw)


def parse_flashbots_block(block: Mapping[str, Any]) -> List[PatternRecord]:
    """Sandwich-shaped bundles of one Flashbots blocks-API block (see the module docstring)."""
    number = _get_int(block, "block_number")
    bundles: Dict[int, List[Mapping[str, Any]]] = {}
    for tx in block.get("transactions") or []:
        if tx.get("bundle_index") is None:
            continue
        bundles.setdefault(_as_int(tx["bundle_index"], "bundle_index"), []).append(tx)
    out: List[PatternRecord] = []
    for idx in sorted(bundles):
        txs = sorted(bundles[idx], key=lambda t: _get_int(t, "tx_index"))
        if len(txs) < 3:
            continue
        first, last = str(txs[0].get("eoa_address", "")).lower(), str(txs[-1].get("eoa_address", "")).lower()
        middle = {str(t.get("eoa_address", "")).lower() for t in txs[1:-1]}
        if not first or first != last or first in middle:
            continue
        bribe = sum(_get_int(t, "coinbase_transfer") for t in txs)
        reward = sum(_get_int(t, "total_miner_reward") for t in txs) or bribe
        gas_fees = max(0, reward - bribe)
        tip = _get_int(txs[0], "gas_price")
        out.append(_finish(None, "flashbots_blocks", chain_id=1, block_number=number, hops=1,
                           pools=max(1, len(txs) - 2), extracted_wei=reward,
                           bribe_bps=bribe * BPS // reward if reward else 0,
                           priority_fee_wei=tip if tip > 0 else (gas_fees // max(1, len(txs))),
                           tx_hashes=tuple(str(t.get("transaction_hash", "")) for t in txs),
                           reference=f"flashbots:block:{number}:bundle:{idx}"))
    return out


def parse_mev_inspect(row: Mapping[str, Any]) -> PatternRecord:
    """One mev-inspect-py ``sandwiches`` row (optionally with its swap objects)."""
    victims = row.get("sandwiched_swaps") or []
    front = row.get("frontrun_swap") or {}
    back = row.get("backrun_swap") or {}
    pools = {str(s.get("contract_address", "")).lower() for s in (front, *victims) if s.get("contract_address")}
    hops = max([1] + [len(s.get("path") or s.get("route") or ()) or 1 for s in victims])
    victim_in = sum(_get_int(s, "token_in_amount") for s in victims)
    front_in = _get_int(front, "token_in_amount")
    reserve = _get_int(row, "reserve_in")
    hashes = [str(row.get("frontrun_swap_transaction_hash", ""))]
    hashes += [str(s.get("transaction_hash", "")) for s in victims]
    hashes.append(str(row.get("backrun_swap_transaction_hash", "")))
    return _finish(None, "mev_inspect", chain_id=_get_int(row, "chain_id", 1),
                   block_number=_get_int(row, "block_number"), hops=hops, pools=max(1, len(pools)),
                   victim_tolerance_bps=_get_int(row, "victim_tolerance_bps", 50),
                   frontrun_impact_bps=front_in * BPS // reserve if reserve and front_in else 0,
                   victim_size_bps=victim_in * BPS // reserve if reserve and victim_in else 0,
                   reserve_in_wei=reserve, extracted_wei=max(0, _get_int(row, "profit_amount")),
                   bribe_bps=_get_int(row, "bribe_bps"), priority_fee_wei=_get_int(back, "gas_price"),
                   capital_wei=front_in, tx_hashes=tuple(h for h in hashes if h),
                   reference=f"mev-inspect:{row.get('sandwicher_address', '')}:{row.get('block_number', '')}")


def parse_advisory(entry: Mapping[str, Any]) -> PatternRecord:
    """A security-advisory entry; its ``attack`` object uses :class:`PatternRecord` field names."""
    attack = dict(entry.get("attack") or {})
    kw = {k: _get_int(attack, k) for k in attack if k not in ("category", "tx_hashes", "reference", "source")}
    kw.setdefault("chain_id", _get_int(entry, "chain_id", 1))
    if "dst_chain_id" in entry:
        kw["dst_chain_id"] = _get_int(entry, "dst_chain_id")
    kw["tx_hashes"] = tuple(str(h) for h in attack.get("tx_hashes", ()))
    kw["reference"] = f"advisory:{entry['advisory_id']}"
    return _finish(entry.get("category") or attack.get("category"), "advisory", **_known(kw))


def _known(kw: Mapping[str, Any]) -> Dict[str, Any]:
    names = {f.name for f in fields(PatternRecord)} - {"category", "source"}
    unknown = sorted(set(kw) - names)
    if unknown:
        raise PatternError(f"unknown pattern fields: {', '.join(unknown)}")
    return dict(kw)


def parse_normalized(row: Mapping[str, Any], source: str = "normalized") -> PatternRecord:
    """A row in :class:`PatternRecord` field names (CSV cells may be strings; empty cells are defaults)."""
    kw: Dict[str, Any] = {}
    for k, v in row.items():
        if k in ("category", "source") or v is None or v == "":
            continue
        if k == "tx_hashes":
            kw[k] = tuple(v.split("|")) if isinstance(v, str) else tuple(str(x) for x in v)
        elif k == "reference":
            kw[k] = str(v)
        else:
            kw[k] = _as_int(v, k)
    cat = row.get("category") or None
    return _finish(cat, source if row.get("source") in (None, "", source) else str(row["source"]), **_known(kw))


def normalize(raw: Mapping[str, Any], source: Optional[str] = None) -> List[PatternRecord]:
    """Records from one raw object (a Flashbots block may hold several sandwiches, or none)."""
    src = source or detect_source(raw)
    if src == "flashbots_blocks":
        return parse_flashbots_block(raw)
    if src == "mev_inspect":
        return [parse_mev_inspect(raw)]
    if src == "advisory":
        return [parse_advisory(raw)]
    if src in ("normalized", "synthetic"):
        return [parse_normalized(raw, src)]
    raise PatternError(f"unknown source {src!r}")


# ---------------------------------------------------------------------------
# Chunked file readers
# ---------------------------------------------------------------------------


def iter_lines(path: Union[str, Path], chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> Iterator[str]:
    """Lines of a UTF-8 file read in ``chunk_bytes`` chunks (memory stays O(chunk + longest line))."""
    if chunk_bytes < 1:
        raise PatternError("chunk_bytes must be >= 1")
    tail = b""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            parts = (tail + chunk).split(b"\n")
            tail = parts.pop()
            for p in parts:
                yield p.decode("utf-8").rstrip("\r")
    if tail:
        yield tail.decode("utf-8").rstrip("\r")


def _sniff(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    with open(path, "rb") as fh:
        head = fh.read(4096).lstrip()
    if head.startswith(b"["):
        return "json"
    if head.startswith(b"{"):
        first = head.split(b"\n", 1)[0].strip()
        try:
            json.loads(first)
        except ValueError:
            return "json"
        return "jsonl"
    return "csv"


def iter_raw(path: Union[str, Path], chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> Iterator[Tuple[int, Any]]:
    """``(line_or_index, raw object)`` from a JSON array / object, JSONL or CSV file."""
    p = Path(path)
    kind = _sniff(p)
    if kind == "jsonl":
        for n, line in enumerate(iter_lines(p, chunk_bytes), 1):
            if line.strip():
                yield n, json.loads(line)
    elif kind == "csv":
        lines = iter_lines(p, chunk_bytes)
        for n, row in enumerate(csv.DictReader(lines), 2):
            yield n, row
    else:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        items = data if isinstance(data, list) else data.get("blocks") or data.get("patterns") or [data]
        for n, item in enumerate(items):
            yield n, item


@dataclass
class IngestReport:
    added: int = 0
    duplicates: int = 0
    rejected: int = 0
    raw_objects: int = 0
    bytes_written: int = 0
    by_category: Dict[str, int] = field(default_factory=dict)
    by_source: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def merge(self, other: "IngestReport", error_limit: int = 50) -> None:
        self.added += other.added
        self.duplicates += other.duplicates
        self.rejected += other.rejected
        self.raw_objects += other.raw_objects
        self.bytes_written += other.bytes_written
        for k, v in other.by_category.items():
            self.by_category[k] = self.by_category.get(k, 0) + v
        for k, v in other.by_source.items():
            self.by_source[k] = self.by_source.get(k, 0) + v
        self.errors.extend(other.errors[: max(0, error_limit - len(self.errors))])
        self.elapsed_s += other.elapsed_s

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["by_category"] = dict(sorted(self.by_category.items()))
        d["by_source"] = dict(sorted(self.by_source.items()))
        d["elapsed_s"] = round(self.elapsed_s, 4)
        return d


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class ByteLRU:
    """Thread-safe LRU cache bounded by the summed size of its values (in bytes)."""

    def __init__(self, capacity_bytes: int = DEFAULT_CACHE_BYTES) -> None:
        if capacity_bytes < 0:
            raise PatternError("cache capacity must be non-negative")
        self.capacity = capacity_bytes
        self.used = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._data: "OrderedDict[Any, Tuple[Any, int]]" = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: Any) -> bool:
        with self._lock:
            return key in self._data

    def get(self, key: Any) -> Any:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return item[0]

    def put(self, key: Any, value: Any, size: int) -> bool:
        """Insert (or refresh) ``key``; values larger than the whole cache are not kept."""
        if size < 0:
            raise PatternError("size must be non-negative")
        with self._lock:
            old = self._data.pop(key, None)
            if old is not None:
                self.used -= old[1]
            if size > self.capacity:
                return False
            self._data[key] = (value, size)
            self.used += size
            while self.used > self.capacity:
                _, (_, s) = self._data.popitem(last=False)
                self.used -= s
                self.evictions += 1
            return True

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.used = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {"entries": len(self._data), "used_bytes": self.used, "capacity_bytes": self.capacity,
                    "hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                    "hit_rate": round(self.hits / total, 6) if total else 0.0}


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def encode_frame(record: PatternRecord) -> bytes:
    payload = zlib.compress(record.to_bytes(), 6)
    return HEADER.pack(MAGIC, len(payload), zlib.crc32(payload)) + payload


def decode_frame(frame: bytes) -> PatternRecord:
    if len(frame) < HEADER.size:
        raise CorruptFrameError("frame shorter than its header")
    magic, length, crc = HEADER.unpack_from(frame)
    if magic != MAGIC:
        raise CorruptFrameError("bad frame magic")
    payload = frame[HEADER.size:HEADER.size + length]
    if len(payload) != length:
        raise CorruptFrameError("truncated frame")
    if zlib.crc32(payload) != crc:
        raise CorruptFrameError("frame CRC mismatch")
    return PatternRecord.from_bytes(zlib.decompress(payload))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

_INDEX_COLUMNS = ("hash", "category", "source", "chain_id", "block_number") + MATRIX_FIELDS + (
    "segment", "offset", "length")

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS patterns (
    id INTEGER PRIMARY KEY,
    hash TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    source TEXT NOT NULL,
    chain_id INTEGER NOT NULL,
    block_number INTEGER NOT NULL,
    {", ".join(f"{c} INTEGER NOT NULL" for c in MATRIX_FIELDS)},
    segment INTEGER NOT NULL,
    offset INTEGER NOT NULL,
    length INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS patterns_category ON patterns (category, extraction_bps);
CREATE INDEX IF NOT EXISTS patterns_source ON patterns (source);
CREATE INDEX IF NOT EXISTS patterns_block ON patterns (block_number);
CREATE INDEX IF NOT EXISTS patterns_location ON patterns (segment, offset);
"""


@dataclass(frozen=True)
class PatternRow:
    """One index row: the record's hash, category, matrix fields and frame location."""

    id: int
    hash: str
    category: str
    source: str
    chain_id: int
    block_number: int
    values: Tuple[int, ...]
    segment: int
    offset: int
    length: int

    def field(self, name: str) -> int:
        return self.values[MATRIX_FIELDS.index(name)]


def _row(r: Sequence[Any]) -> PatternRow:
    n = len(MATRIX_FIELDS)
    return PatternRow(r[0], r[1], r[2], r[3], r[4], r[5], tuple(r[6:6 + n]), r[6 + n], r[7 + n], r[8 + n])


class PatternStore:
    """Local MEV pattern store: SQLite index + append-only CRC-framed segments (see the module docstring)."""

    def __init__(self, root: Union[str, Path], max_storage_bytes: int = DEFAULT_MAX_STORAGE_BYTES,
                 segment_bytes: int = DEFAULT_SEGMENT_BYTES, cache_bytes: int = DEFAULT_CACHE_BYTES,
                 chunk_bytes: int = DEFAULT_CHUNK_BYTES, readonly: bool = False) -> None:
        if max_storage_bytes < 1 or segment_bytes < HEADER.size + 1 or chunk_bytes < 1:
            raise PatternError("max_storage_bytes, segment_bytes and chunk_bytes must be positive")
        self.root = Path(root)
        self.max_storage_bytes = max_storage_bytes
        self.segment_bytes = segment_bytes
        self.chunk_bytes = chunk_bytes
        self.readonly = readonly
        self.cache = ByteLRU(cache_bytes)
        self._lock = threading.RLock()
        self._local = threading.local()
        self._closed = False
        self.segment_dir = self.root / SEGMENT_DIR
        if readonly:
            if not (self.root / INDEX_FILE).is_file():
                raise PatternError(f"no pattern store at {self.root}")
        else:
            self.segment_dir.mkdir(parents=True, exist_ok=True)
            with self._lock:
                db = self._db()
                db.executescript(_SCHEMA)
                db.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
                db.commit()
        version = self._meta("schema_version")
        if version is not None and int(version) != SCHEMA_VERSION:
            raise PatternError(f"pattern store schema {version} is not {SCHEMA_VERSION}")

    # -- connections ---------------------------------------------------------

    def _db(self) -> sqlite3.Connection:
        """One connection per thread (SQLite connections must not cross threads)."""
        if self._closed:
            raise PatternError("pattern store is closed")
        conn = getattr(self._local, "conn", None)
        if conn is None:
            path = self.root / INDEX_FILE
            if self.readonly:
                conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60, check_same_thread=False)
            else:
                conn = sqlite3.connect(str(path), timeout=60, isolation_level=None, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
            with self._lock:
                self._conns = getattr(self, "_conns", []) + [conn]
        return conn

    def close(self) -> None:
        with self._lock:
            for conn in getattr(self, "_conns", []):
                conn.close()
            self._conns = []
            self._closed = True
        self._local = threading.local()

    def __enter__(self) -> "PatternStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _meta(self, key: str) -> Optional[str]:
        row = self._db().execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    # -- segments ------------------------------------------------------------

    def segment_path(self, segment: int) -> Path:
        return self.segment_dir / f"seg-{segment:06d}.bin"

    def segments(self) -> List[int]:
        if not self.segment_dir.is_dir():
            return []
        out = []
        for p in self.segment_dir.glob("seg-*.bin"):
            try:
                out.append(int(p.stem[4:]))
            except ValueError:
                continue
        return sorted(out)

    def segment_sizes(self) -> Dict[int, int]:
        return {s: self.segment_path(s).stat().st_size for s in self.segments()}

    def bytes_used(self) -> int:
        """Segments plus the index (and its WAL), i.e. what counts against the cap."""
        total = sum(self.segment_sizes().values())
        for name in (INDEX_FILE, INDEX_FILE + "-wal", INDEX_FILE + "-shm"):
            p = self.root / name
            if p.exists():
                total += p.stat().st_size
        return total

    def _active_segment(self, incoming: int) -> Tuple[int, int]:
        segs = self.segments()
        if not segs:
            return 0, 0
        last = segs[-1]
        size = self.segment_path(last).stat().st_size
        if size > 0 and size + incoming > self.segment_bytes:
            return last + 1, 0
        return last, size

    # -- writes --------------------------------------------------------------

    def add(self, records: Iterable[PatternRecord], batch: int = DEFAULT_BATCH) -> IngestReport:
        """Store new records (duplicates by hash are skipped), ``batch`` per transaction."""
        if self.readonly:
            raise PatternError("pattern store is read-only")
        if batch < 1:
            raise PatternError("batch must be >= 1")
        report = IngestReport()
        started = time.perf_counter()
        buf: List[PatternRecord] = []
        for r in records:
            buf.append(r)
            if len(buf) >= batch:
                self._write_batch(buf, report)
                buf = []
        if buf:
            self._write_batch(buf, report)
        report.elapsed_s = time.perf_counter() - started
        return report

    def _write_batch(self, records: Sequence[PatternRecord], report: IngestReport) -> None:
        with self._lock:
            db = self._db()
            db.execute("BEGIN IMMEDIATE")
            try:
                fresh: Dict[str, PatternRecord] = {}
                for r in records:
                    h = r.digest()
                    if h in fresh:
                        report.duplicates += 1
                        continue
                    fresh[h] = r
                hashes = list(fresh)
                known: set = set()
                for part in chunked(hashes, max(1, -(-len(hashes) // 500))):
                    marks = ",".join("?" * len(part))
                    known.update(x[0] for x in db.execute(f"SELECT hash FROM patterns WHERE hash IN ({marks})",
                                                          part))
                report.duplicates += len(known)
                todo = [(h, fresh[h]) for h in hashes if h not in known]
                frames = [(h, r, encode_frame(r)) for h, r in todo]
                incoming = sum(len(f) for _, _, f in frames)
                used = self.bytes_used()
                if used + incoming > self.max_storage_bytes:
                    raise StorageFullError(f"writing {incoming} bytes would exceed the {self.max_storage_bytes}"
                                           f"-byte cap ({used} bytes used)")
                rows = []
                handle = None
                seg = -1
                pos = 0
                try:
                    for h, r, frame in frames:
                        if handle is None or (pos > 0 and pos + len(frame) > self.segment_bytes):
                            if handle is not None:
                                handle.flush()
                                os.fsync(handle.fileno())
                                handle.close()
                                seg, pos = seg + 1, 0
                            else:
                                seg, pos = self._active_segment(len(frame))
                            handle = open(self.segment_path(seg), "ab")
                        handle.write(frame)
                        rows.append((h, r.category, r.source, r.chain_id, r.block_number, *r.matrix_row(), seg,
                                     pos, len(frame)))
                        pos += len(frame)
                        report.added += 1
                        report.bytes_written += len(frame)
                        report.by_category[r.category] = report.by_category.get(r.category, 0) + 1
                        report.by_source[r.source] = report.by_source.get(r.source, 0) + 1
                finally:
                    if handle is not None:
                        handle.flush()
                        os.fsync(handle.fileno())
                        handle.close()
                cols = ",".join(_INDEX_COLUMNS)
                db.executemany(f"INSERT INTO patterns ({cols}) VALUES ({','.join('?' * len(_INDEX_COLUMNS))})",
                               rows)
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def ingest_file(self, path: Union[str, Path], source: Optional[str] = None,
                    batch: int = DEFAULT_BATCH, error_limit: int = 50) -> IngestReport:
        """Stream a JSON / JSONL / CSV export through :func:`normalize` into the store."""
        report = IngestReport()
        rejected = IngestReport()
        started = time.perf_counter()

        def records() -> Iterator[PatternRecord]:
            for where, raw in iter_raw(path, self.chunk_bytes):
                rejected.raw_objects += 1
                try:
                    yield from normalize(raw, source)
                except (PatternError, KeyError, TypeError, ValueError) as exc:
                    rejected.rejected += 1
                    if len(rejected.errors) < error_limit:
                        rejected.errors.append(f"{path}:{where}: {exc}")

        report.merge(self.add(records(), batch))
        report.merge(rejected, error_limit)
        report.elapsed_s = time.perf_counter() - started
        return report

    # -- reads ---------------------------------------------------------------

    def __len__(self) -> int:
        return self.count()

    def count(self, category: Optional[str] = None) -> int:
        if category is None:
            return self._db().execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
        return self._db().execute("SELECT COUNT(*) FROM patterns WHERE category = ?", (category,)).fetchone()[0]

    def id_range(self) -> Tuple[int, int]:
        lo, hi = self._db().execute("SELECT MIN(id), MAX(id) FROM patterns").fetchone()
        return (0, -1) if lo is None else (lo, hi)

    def rows(self, category: Optional[str] = None, source: Optional[str] = None,
             min_extraction_bps: Optional[int] = None, id_lo: Optional[int] = None, id_hi: Optional[int] = None,
             order: str = "id", limit: Optional[int] = None) -> List[PatternRow]:
        """Index rows matching every filter given; ``order`` is ``id`` or ``extraction`` (best first)."""
        where: List[str] = []
        args: List[Any] = []
        if category is not None:
            if category not in CATEGORIES:
                raise PatternError(f"unknown category {category!r}")
            where.append("category = ?")
            args.append(category)
        if source is not None:
            where.append("source = ?")
            args.append(source)
        if min_extraction_bps is not None:
            where.append("extraction_bps >= ?")
            args.append(min_extraction_bps)
        if id_lo is not None:
            where.append("id >= ?")
            args.append(id_lo)
        if id_hi is not None:
            where.append("id <= ?")
            args.append(id_hi)
        if order not in ("id", "extraction"):
            raise PatternError("order must be 'id' or 'extraction'")
        sql = "SELECT id, " + ", ".join(_INDEX_COLUMNS) + " FROM patterns"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id" if order == "id" else " ORDER BY extraction_bps DESC, id"
        if limit is not None:
            if limit < 0:
                raise PatternError("limit must be non-negative")
            sql += f" LIMIT {int(limit)}"
        return [_row(r) for r in self._db().execute(sql, args)]

    def row(self, digest: str) -> Optional[PatternRow]:
        r = self._db().execute("SELECT id, " + ", ".join(_INDEX_COLUMNS) + " FROM patterns WHERE hash = ?",
                               (digest,)).fetchone()
        return None if r is None else _row(r)

    def _pread(self, segment: int, offset: int, length: int) -> bytes:
        path = self.segment_path(segment)
        if hasattr(os, "pread"):
            fd = os.open(path, os.O_RDONLY)
            try:
                data = os.pread(fd, length, offset)
            finally:
                os.close(fd)
        else:  # pragma: no cover - Windows
            with self._lock, open(path, "rb") as fh:
                fh.seek(offset)
                data = fh.read(length)
        return data

    def read(self, row: PatternRow) -> PatternRecord:
        cached = self.cache.get(row.hash)
        if cached is not None:
            return cached
        record = decode_frame(self._pread(row.segment, row.offset, row.length))
        if record.digest() != row.hash:
            raise CorruptFrameError(f"record at segment {row.segment} offset {row.offset} does not match its hash")
        self.cache.put(row.hash, record, row.length * 4)
        return record

    def get(self, digest: str) -> Optional[PatternRecord]:
        row = self.row(digest)
        return None if row is None else self.read(row)

    def read_many(self, rows: Sequence[PatternRow]) -> List[PatternRecord]:
        """Records for ``rows`` (in the given order), read segment by segment in offset order."""
        out: List[Optional[PatternRecord]] = [None] * len(rows)
        pending: Dict[int, List[Tuple[int, PatternRow]]] = {}
        for i, r in enumerate(rows):
            hit = self.cache.get(r.hash)
            if hit is not None:
                out[i] = hit
            else:
                pending.setdefault(r.segment, []).append((i, r))
        for seg in sorted(pending):
            items = sorted(pending[seg], key=lambda t: t[1].offset)
            with open(self.segment_path(seg), "rb") as fh:
                window_start, window = -1, b""
                for i, r in items:
                    end = r.offset + r.length
                    if not window_start <= r.offset or end > window_start + len(window):
                        fh.seek(r.offset)
                        window_start = r.offset
                        window = fh.read(max(self.chunk_bytes, r.length))
                    frame = window[r.offset - window_start:end - window_start]
                    record = decode_frame(frame)
                    if record.digest() != r.hash:
                        raise CorruptFrameError(f"record {r.id} does not match its hash")
                    self.cache.put(r.hash, record, r.length * 4)
                    out[i] = record
        return [r for r in out if r is not None]

    def records(self, category: Optional[str] = None, limit: Optional[int] = None,
                order: str = "id") -> List[PatternRecord]:
        return self.read_many(self.rows(category=category, order=order, limit=limit))

    def iter_segment(self, segment: int, chunk_bytes: Optional[int] = None
                     ) -> Iterator[Tuple[int, int, PatternRecord]]:
        """``(offset, length, record)`` for every frame of a segment, streamed in chunks."""
        size = chunk_bytes or self.chunk_bytes
        buf = bytearray()
        base = 0
        with open(self.segment_path(segment), "rb") as fh:
            eof = False
            while True:
                while len(buf) < HEADER.size and not eof:
                    chunk = fh.read(size)
                    eof = not chunk
                    buf += chunk
                if len(buf) < HEADER.size:
                    if buf:
                        raise CorruptFrameError(f"segment {segment}: {len(buf)} trailing bytes at {base}")
                    return
                magic, length, _ = HEADER.unpack_from(buf)
                if magic != MAGIC:
                    raise CorruptFrameError(f"segment {segment}: bad magic at offset {base}")
                need = HEADER.size + length
                while len(buf) < need and not eof:
                    chunk = fh.read(max(size, need - len(buf)))
                    eof = not chunk
                    buf += chunk
                if len(buf) < need:
                    raise CorruptFrameError(f"segment {segment}: truncated frame at offset {base}")
                yield base, need, decode_frame(bytes(buf[:need]))
                del buf[:need]
                base += need

    # -- analysis ------------------------------------------------------------

    def matrix(self, category: str, limit: Optional[int] = None, order: str = "id") -> Any:
        """The category's :data:`MATRIX_FIELDS` as an ``(n, 12)`` int64 NumPy matrix (index only, no IO)."""
        import numpy as np

        rows = self.rows(category=category, order=order, limit=limit)
        if not rows:
            return np.zeros((0, len(MATRIX_FIELDS)), dtype=np.int64)
        return np.array([r.values for r in rows], dtype=np.int64)

    def category_matrices(self, limit: Optional[int] = None) -> Dict[str, Any]:
        return {c: self.matrix(c, limit) for c in CATEGORIES}

    def stats(self) -> Dict[str, Any]:
        db = self._db()
        by_cat = dict(db.execute("SELECT category, COUNT(*) FROM patterns GROUP BY category"))
        by_src = dict(db.execute("SELECT source, COUNT(*) FROM patterns GROUP BY source"))
        sizes = self.segment_sizes()
        used = self.bytes_used()
        means = {}
        for c in CATEGORIES:
            r = db.execute("SELECT AVG(extraction_bps), AVG(victim_slippage_bps), AVG(hops) FROM patterns "
                           "WHERE category = ?", (c,)).fetchone()
            means[c] = {"extraction_bps": round(r[0] or 0.0, 3), "victim_slippage_bps": round(r[1] or 0.0, 3),
                        "hops": round(r[2] or 0.0, 3)}
        return {
            "root": str(self.root),
            "patterns": sum(by_cat.values()),
            "by_category": {c: by_cat.get(c, 0) for c in CATEGORIES},
            "by_source": dict(sorted(by_src.items())),
            "category_means": means,
            "segments": len(sizes),
            "segment_bytes": sum(sizes.values()),
            "bytes_used": used,
            "max_storage_bytes": self.max_storage_bytes,
            "capacity_used": round(used / self.max_storage_bytes, 9),
            "cache": self.cache.stats(),
        }

    def verify(self) -> Dict[str, Any]:
        """Stream every segment, check every frame and that the index points at exactly those frames."""
        frames: Dict[Tuple[int, int], Tuple[int, str]] = {}
        errors: List[str] = []
        for seg in self.segments():
            try:
                for off, length, rec in self.iter_segment(seg):
                    frames[(seg, off)] = (length, rec.digest())
            except CorruptFrameError as exc:
                errors.append(str(exc))
        indexed = 0
        for r in self.rows():
            indexed += 1
            got = frames.get((r.segment, r.offset))
            if got is None:
                errors.append(f"row {r.id}: no frame at segment {r.segment} offset {r.offset}")
            elif got != (r.length, r.hash):
                errors.append(f"row {r.id}: frame does not match the index")
        orphans = len(frames) - indexed if len(frames) > indexed else 0
        return {"ok": not errors and not orphans, "frames": len(frames), "indexed": indexed,
                "orphan_frames": orphans, "errors": errors[:50]}


# ---------------------------------------------------------------------------
# Slippage evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatternEvaluation:
    reserve_in_wei: int
    victim_in_wei: int
    frontrun_in_wei: int
    victim_expected_out: int
    victim_actual_out: int
    victim_min_out: int
    realized_slippage_bps: int
    victim_limit_held: bool
    profit_wei: int
    extraction_bps: int
    max_frontrun_in_wei: int
    utilization_bps: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sandwich(r_in: int, r_out: int, front: int, victim: int, fee_bps: int) -> Tuple[int, int, int]:
    """(victim output after ``front``, back-run output, attacker profit in the input token)."""
    f_out = v2_get_amount_out(front, r_in, r_out, fee_bps) if front else 0
    r_in1, r_out1 = r_in + front, r_out - f_out
    v_out = v2_get_amount_out(victim, r_in1, r_out1, fee_bps)
    r_in2, r_out2 = r_in1 + victim, r_out1 - v_out
    back = v2_get_amount_out(f_out, r_out2, r_in2, fee_bps) if f_out else 0
    return v_out, back, back - front


def evaluate_pattern(record: PatternRecord, fee_bps: int = 30) -> PatternEvaluation:
    """Replay ``record`` as a sandwich on a constant-product pool (see the module docstring)."""
    reserve = record.reserve_in_wei or NOMINAL_RESERVE_WEI
    impact = record.frontrun_impact_bps
    size = record.victim_size_bps or max(1, impact // 2 or 10)
    victim = max(1, reserve * size // BPS)
    front = reserve * impact // BPS
    r_in = r_out = reserve
    expected = v2_get_amount_out(victim, r_in, r_out, fee_bps)
    min_out = expected * (BPS - record.victim_tolerance_bps) // BPS
    v_out, _, profit = _sandwich(r_in, r_out, front, victim, fee_bps)
    slip = (expected - v_out) * BPS // expected if expected else 0
    lo, hi = 0, reserve * 2
    if _sandwich(r_in, r_out, hi, victim, fee_bps)[0] >= min_out:
        lo = hi
    else:
        while hi - lo > 1:  # exact: about log2(2 * reserve) steps
            mid = (lo + hi) // 2
            if _sandwich(r_in, r_out, mid, victim, fee_bps)[0] >= min_out:
                lo = mid
            else:
                hi = mid
    held = v_out >= min_out
    return PatternEvaluation(
        reserve_in_wei=reserve, victim_in_wei=victim, frontrun_in_wei=front, victim_expected_out=expected,
        victim_actual_out=v_out, victim_min_out=min_out, realized_slippage_bps=max(0, slip),
        victim_limit_held=held, profit_wei=profit,
        extraction_bps=min(BPS, profit * BPS // victim) if profit > 0 and held else 0,
        max_frontrun_in_wei=lo, utilization_bps=min(BPS, front * BPS // lo) if lo else 0,
    )


def pattern_weight(record: PatternRecord, evaluation: Optional[PatternEvaluation] = None) -> float:
    """Training weight in ``[0.05, 1]``: feasible, extracting patterns that use their slippage room weigh most."""
    ev = evaluation or evaluate_pattern(record)
    if not ev.victim_limit_held:
        return 0.05
    extraction = min(1.0, ev.extraction_bps / 100.0)
    utilization = ev.utilization_bps / BPS
    return round(0.1 + 0.6 * extraction + 0.3 * utilization, 9)


# ---------------------------------------------------------------------------
# Parallel scan
# ---------------------------------------------------------------------------


def scan_shards(lo: int, hi: int, shards: int) -> List[Tuple[int, int]]:
    """Fixed, contiguous ``[lo, hi]`` id ranges (depends on ``shards`` only, never on workers)."""
    if shards < 1:
        raise PatternError("shards must be >= 1")
    if hi < lo:
        return []
    ids = hi - lo + 1
    n = min(shards, ids)
    size, extra = divmod(ids, n)
    out = []
    start = lo
    for i in range(n):
        end = start + size + (1 if i < extra else 0) - 1
        out.append((start, end))
        start = end + 1
    return out


def _scan_task(task: Tuple[Any, ...]) -> Dict[str, Any]:
    """One shard: read its records (sequential IO), evaluate them, return digests + aggregates."""
    root, lo, hi, category, min_extraction, cache_bytes, chunk_bytes = task
    started = time.perf_counter()
    cpu0 = time.process_time()
    store = PatternStore(root, cache_bytes=cache_bytes, chunk_bytes=chunk_bytes, readonly=True)
    try:
        rows = store.rows(category=category, min_extraction_bps=min_extraction, id_lo=lo, id_hi=hi)
        records = store.read_many(sorted(rows, key=lambda r: (r.segment, r.offset)))
    finally:
        store.close()
    by_hash = {r.digest(): r for r in records}
    out_rows = []
    cats: Counter = Counter()
    feasible = 0
    for row in rows:
        rec = by_hash[row.hash]
        ev = evaluate_pattern(rec)
        feasible += ev.victim_limit_held
        cats[rec.category] += 1
        out_rows.append({"id": row.id, "hash": row.hash, "category": rec.category,
                         "weight": pattern_weight(rec, ev), "extraction_bps": ev.extraction_bps,
                         "record": rec.canonical()})
    return {"lo": lo, "hi": hi, "rows": out_rows, "by_category": dict(cats), "feasible": feasible,
            "bytes_read": sum(r.length for r in rows), "pid": os.getpid(),
            "elapsed_s": time.perf_counter() - started, "cpu_s": time.process_time() - cpu0}


@dataclass
class ScanResult:
    records: List[PatternRecord]
    weights: List[float]
    hashes: List[str]
    by_category: Dict[str, int]
    feasible: int
    shards: int
    bytes_read: int
    elapsed_s: float
    cpu_s: float
    worker_pids: List[int]
    parallel: bool

    def summary(self) -> Dict[str, Any]:
        return {"patterns": len(self.records), "by_category": {c: self.by_category.get(c, 0) for c in CATEGORIES},
                "feasible": self.feasible, "shards": self.shards, "bytes_read": self.bytes_read,
                "mean_weight": round(sum(self.weights) / len(self.weights), 6) if self.weights else 0.0,
                "elapsed_s": round(self.elapsed_s, 4), "cpu_s": round(self.cpu_s, 4),
                "distinct_worker_pids": len(set(self.worker_pids)), "parallel": self.parallel}


def parallel_scan(root: Union[str, Path], executor: Optional[WorkerPool] = None, shards: int = DEFAULT_SCAN_SHARDS,
                  category: Optional[str] = None, min_extraction_bps: Optional[int] = None,
                  limit: Optional[int] = None, cache_bytes: int = 32 * MIB,
                  chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ScanResult:
    """Read and evaluate the store's patterns, one fixed id shard per worker task.

    Results come back in id order; ``limit`` keeps the ``limit`` heaviest
    patterns (ties by id), so the selection is identical for any worker count.
    """
    started = time.perf_counter()
    with PatternStore(root, readonly=True) as store:
        lo, hi = store.id_range()
    tasks = [(str(root), a, b, category, min_extraction_bps, cache_bytes, chunk_bytes)
             for a, b in scan_shards(lo, hi, shards)]
    if executor is None:
        parts = [_scan_task(t) for t in tasks]
        parallel = False
    else:
        parts = executor.map(_scan_task, tasks)
        parallel = executor.will_parallelize(len(tasks))
    rows = [r for p in parts for r in p["rows"]]
    if limit is not None:
        keep = sorted(rows, key=lambda r: (-r["weight"], r["id"]))[: max(0, limit)]
        rows = sorted(keep, key=lambda r: r["id"])
    cats: Counter = Counter(r["category"] for r in rows)
    return ScanResult(
        records=[PatternRecord.from_canonical(r["record"]) for r in rows],
        weights=[r["weight"] for r in rows], hashes=[r["hash"] for r in rows], by_category=dict(cats),
        feasible=sum(p["feasible"] for p in parts) if limit is None else sum(
            1 for r in rows if r["weight"] > 0.05),
        shards=len(tasks), bytes_read=sum(p["bytes_read"] for p in parts),
        elapsed_s=time.perf_counter() - started, cpu_s=sum(p["cpu_s"] for p in parts),
        worker_pids=[p["pid"] for p in parts], parallel=parallel)


# ---------------------------------------------------------------------------
# Synthetic reference corpus
# ---------------------------------------------------------------------------

_CHAINS = (1, 10, 56, 137, 8453, 42161)


def synthetic_record(category: str, rng: random.Random, index: int = 0) -> PatternRecord:
    """One generated (not observed) record of ``category`` with plausible parameter ranges."""
    if category not in CATEGORIES:
        raise PatternError(f"unknown category {category!r}")
    tol = rng.choice((10, 30, 50, 100, 200, 300, 500, 1_000))
    reserve = rng.choice((200, 1_000, 5_000, 20_000)) * E18
    size = rng.randint(5, 400)
    probe = PatternRecord(category=category, source="synthetic", victim_tolerance_bps=tol, victim_size_bps=size,
                          reserve_in_wei=reserve)
    room = evaluate_pattern(probe).max_frontrun_in_wei
    # Most observed sandwiches use 30-100% of the victim's slippage room; one in ten overshoots it.
    use = rng.uniform(1.0, 1.5) if rng.random() < 0.1 else rng.uniform(0.3, 1.0)
    impact = max(1, min(BPS, math.floor(room * use * BPS / reserve)))
    kw: Dict[str, Any] = dict(
        chain_id=1, block_number=17_000_000 + rng.randint(0, 3_000_000), hops=1, pools=1,
        victim_tolerance_bps=tol, frontrun_impact_bps=impact, victim_size_bps=size,
        reserve_in_wei=reserve, bribe_bps=rng.randint(3_000, 9_900),
        priority_fee_wei=rng.randint(1, 4_000) * GWEI // 100, latency_ms=rng.randint(40, 900),
        capital_wei=rng.randint(1, 400_000) * 10**15, reference=f"synthetic:{category}:{index}")
    if category == "multi_hop_frontrun":
        kw["hops"] = rng.randint(2, 4)
        kw["pools"] = kw["hops"]
    elif category == "multi_pool_sandwich":
        kw["pools"] = rng.randint(1, 4)
    elif category == "jit_liquidity":
        kw["jit_liquidity_wei"] = reserve * rng.randint(50, 5_000) // BPS
        kw["latency_ms"] = rng.randint(20, 250)
    else:
        kw["chain_id"] = rng.choice(_CHAINS)
        kw["dst_chain_id"] = rng.choice([c for c in _CHAINS if c != kw["chain_id"]])
        kw["drift_blocks"] = rng.randint(1, 64)
        kw["latency_ms"] = rng.randint(300, 6_000)
    rec = PatternRecord(category=category, source="synthetic", **kw)
    ev = evaluate_pattern(rec)
    return PatternRecord(category=category, source="synthetic", victim_slippage_bps=ev.realized_slippage_bps,
                         extracted_wei=max(0, ev.profit_wei), **kw)


def synthetic_corpus(n: int, seed: int = 0) -> List[PatternRecord]:
    """``n`` generated records, round-robin over the four categories, from a seeded stream."""
    if n < 0:
        raise PatternError("n must be non-negative")
    rng = random.Random(f"mev-patterns:{seed}")
    return [synthetic_record(CATEGORIES[i % len(CATEGORIES)], rng, i) for i in range(n)]


def write_jsonl(records: Iterable[PatternRecord], path: Union[str, Path]) -> int:
    """Export records as normalised JSONL (re-ingestable with source ``normalized``)."""
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            d = r.canonical()
            d["tx_hashes"] = list(r.tx_hashes)
            fh.write(json.dumps(d, sort_keys=True) + "\n")
            n += 1
    return n


def open_store(config: Any, readonly: bool = False) -> PatternStore:
    """The store a SimulationConfig points at (``pattern_store`` and its size settings)."""
    if not config.pattern_store:
        raise PatternError("no pattern store configured ([patterns] store is empty)")
    return PatternStore(config.pattern_store, max_storage_bytes=config.pattern_max_storage_gb * GIB,
                        segment_bytes=config.pattern_segment_mb * MIB, cache_bytes=config.pattern_cache_mb * MIB,
                        readonly=readonly)


def render_text(stats: Mapping[str, Any]) -> str:
    buf = io.StringIO()
    buf.write(f"{stats['patterns']} patterns in {stats['segments']} segment(s), {stats['bytes_used']} bytes "
              f"({stats['capacity_used'] * 100:.6f}% of the cap)\n")
    for c, n in stats["by_category"].items():
        buf.write(f"  {c:<22} {n}\n")
    return buf.getvalue()


__all__ = [
    "CATEGORIES",
    "DEFAULT_MAX_STORAGE_BYTES",
    "MATRIX_FIELDS",
    "SOURCES",
    "ByteLRU",
    "CorruptFrameError",
    "IngestReport",
    "PatternError",
    "PatternEvaluation",
    "PatternRecord",
    "PatternRow",
    "PatternStore",
    "ScanResult",
    "StorageFullError",
    "classify",
    "decode_frame",
    "detect_source",
    "encode_frame",
    "evaluate_pattern",
    "iter_lines",
    "iter_raw",
    "normalize",
    "open_store",
    "parallel_scan",
    "parse_advisory",
    "parse_flashbots_block",
    "parse_mev_inspect",
    "parse_normalized",
    "pattern_weight",
    "record_issues",
    "render_text",
    "scan_shards",
    "synthetic_corpus",
    "synthetic_record",
    "write_jsonl",
]
