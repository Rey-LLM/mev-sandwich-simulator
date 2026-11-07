"""Layer 05: private RPC relay (Flashbots Protect / MEV-Share style) emulator.

Status: model. Flashbots Protect / MEV-Share are emulated from their public
documentation (hint kinds, backrun-only bundles, refunds); hint inference is
this project's heuristic, and the crypto is simulation-only (see below).

Everything here runs offline on the simulator's pools. The module models the
path a private transaction takes and how much a searcher can learn from it:

1. A user sends a transaction to :meth:`PrivateRelay.send_private_transaction`
   instead of the public :class:`~simulator.block_builder.TxPool`. It never
   enters the public mempool; :class:`PrivacyPreferences` choose which hints
   are shared with searchers, which builders may receive it, the inclusion
   window and the refund share.
2. The relay simulates the transaction on the latest state and publishes a
   :class:`HintView` on its event stream, exposing only the selected parts:
   contract address, function selector, full calldata or chosen calldata
   byte ranges, full swap logs or topic-only "default" logs, and token
   balance deltas.
3. Searchers read hints, infer the hidden trade (:func:`infer_victim`) and
   submit signed :class:`MevShareBundle` objects that reference the private
   transaction by hash. The relay only accepts backruns: any searcher
   transaction placed before the referenced hash is rejected as a privacy
   violation.
4. :meth:`PrivateRelay.build_auction` simulates every candidate bundle on top
   of the user's transaction and keeps the highest-paying valid one per
   transaction (the order-flow auction). The user receives
   ``refund_bps`` of the winning payment; the rest goes to the builder as a
   coinbase transfer. A "protect" bundle holding the user's transaction alone
   is always delivered as a fallback, so the transaction still lands when
   every backrun fails.
5. :meth:`PrivateRelay.deliver` encrypts each bundle for every builder the
   user allowed (:func:`seal_envelope`); a :class:`PrivateFlowReceiver`
   decrypts with its own key and rejects anything tampered with or addressed
   to another builder.
6. After a block is sealed, :meth:`PrivateRelay.on_block` settles refunds
   and moves transactions to included, failed (nonce consumed by another
   transaction), expired or cancelled.

Leakage analysis
----------------
:func:`measure_exposure` asks what a predatory searcher could take if the
hints leaked to a builder that does not enforce backrun-only bundles: it
builds the best sandwich against the *inferred* trade, executes it against
the *real* trade, and compares the realized profit with the profit a searcher
with perfect information would make (``leak_efficiency``).
:func:`hint_exposure_matrix` runs that over transactions and hint profiles on
the capped :class:`~simulator.parallel.WorkerPool`.

Crypto disclaimer
-----------------
Hashes use SHA3-256 from the standard library (not Ethereum's keccak-256),
signatures are HMAC-SHA256 (standing in for ``X-Flashbots-Signature``), and
envelopes use a SHA-256 counter-mode keystream with encrypt-then-MAC. They
make tampering and wrong-key decryption detectable inside the simulation and
are not meant to protect real data.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import Enum, Flag, IntEnum
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .block_builder import (
    BLOCK_GAS_LIMIT,
    ETH,
    GWEI,
    REPLACEMENT_BUMP_PCT,
    BlockBuilder,
    Bundle,
    BuilderState,
    BuilderTx,
    MevBoostAuction,
    Proposer,
    SealedBlock,
    Status,
    Strategy,
    TxKind,
    TxPool,
    best_sandwich,
    build_sandwich_bundles,
    clamp_gas_limit,
    clone_pool,
    execute_tx,
    generate_builder_mempool,
    native_value,
    next_base_fee,
    seal_block,
)
from .config import ConfigError, SimulationConfig
from .invariants import input_reserve_scale
from .mempool_emulator import _apply_full, geometric_grid
from .multi_hop_router import MultiHopRouter, RoutePath, RouterError, execute_path, quote_path_output
from .parallel import WorkerPool, chunked
from .slippage_engine import MASEError, Pool, V2Pool, V4Pool, _direction

RELAY_CHUNKS = 16  # fixed so exposure results never depend on worker count
DEFAULT_INCLUSION_WINDOW = 25
DEFAULT_REFUND_BPS = 9_000
WORD = 32
SELECTOR_SIZE = 4
UINT256 = 1 << 256


class RelayError(ValueError):
    """Request rejected by the relay."""


# ---------------------------------------------------------------------------
# Hashing, addresses, selectors and topics
# ---------------------------------------------------------------------------


def sha3(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()


def address_of(label: str) -> str:
    """Deterministic 20-byte address for a pool id, token or account label."""
    return "0x" + sha3(b"address:" + label.encode()).hex()[:40]


def selector_of(signature: str) -> bytes:
    return sha3(signature.encode())[:SELECTOR_SIZE]


def topic_of(signature: str) -> str:
    return "0x" + sha3(signature.encode()).hex()


SWAP_SIGNATURE = "exactInputRoute(uint256,uint256,address,address,uint256,address[])"
TRANSFER_SIGNATURE = "transfer(address,uint256)"
MINT_SIGNATURE = "mint(address,int24,int24,uint128)"
BURN_SIGNATURE = "burn(address,int24,int24,uint128)"
SELECTORS: Dict[TxKind, bytes] = {
    TxKind.SWAP: selector_of(SWAP_SIGNATURE),
    TxKind.TRANSFER: selector_of(TRANSFER_SIGNATURE),
    TxKind.MINT: selector_of(MINT_SIGNATURE),
    TxKind.BURN: selector_of(BURN_SIGNATURE),
}
KIND_BY_SELECTOR: Dict[bytes, TxKind] = {v: k for k, v in SELECTORS.items()}
ROUTER_ADDRESS = address_of("router:universal")
V2_SWAP_TOPIC = topic_of("Swap(address,uint256,uint256,uint256,uint256,address)")
V2_SYNC_TOPIC = topic_of("Sync(uint112,uint112)")
V3_SWAP_TOPIC = topic_of("Swap(address,address,int256,int256,uint160,uint128,int24)")
V4_SWAP_TOPIC = topic_of("Swap(bytes32,address,int128,int128,uint160,uint128,int24,uint24)")
SWAP_TOPICS = frozenset({V2_SWAP_TOPIC, V3_SWAP_TOPIC, V4_SWAP_TOPIC})


# ---------------------------------------------------------------------------
# Calldata encoding (selector + 32-byte words)
# ---------------------------------------------------------------------------


class Word(IntEnum):
    """Word positions in swap calldata; pool ``i`` sits at word ``POOLS + i``."""

    AMOUNT_IN = 0
    AMOUNT_OUT_MIN = 1
    TOKEN_IN = 2
    RECIPIENT = 3
    DEADLINE = 4
    HOP_COUNT = 5
    POOLS = 6


def word_range(index: int) -> Tuple[int, int]:
    """Byte range ``[start, end)`` of calldata word ``index``."""
    if index < 0:
        raise RelayError("word index must be non-negative")
    start = SELECTOR_SIZE + WORD * index
    return start, start + WORD


SELECTOR_RANGE = (0, SELECTOR_SIZE)


def pool_word_range(hop: int) -> Tuple[int, int]:
    return word_range(Word.POOLS + hop)


def _u256(value: int) -> bytes:
    return (value % UINT256).to_bytes(WORD, "big")


def _addr_word(address: str) -> bytes:
    return bytes(12) + bytes.fromhex(address[2:])


def encode_calldata(tx: BuilderTx) -> bytes:
    """ABI-shaped calldata for a builder transaction (selector + static words)."""
    sel = SELECTORS[tx.kind]
    p = tx.payload
    if tx.kind is TxKind.SWAP:
        words = [_u256(tx.amount_in), _u256(tx.amount_out_min), _addr_word(address_of(tx.token_in)),
                 _addr_word(address_of(tx.sender)), _u256(int(p.get("deadline", 0))), _u256(len(tx.route))]
        words += [_addr_word(address_of(pid)) for pid in tx.route]
    elif tx.kind is TxKind.TRANSFER:
        words = [_addr_word(address_of(str(p.get("to", "")))), _u256(int(p.get("value", 0)))]
    else:
        words = [_addr_word(address_of(str(p.get("pool_id", "")))), _u256(int(p.get("tick_lower", 0))),
                 _u256(int(p.get("tick_upper", 0))), _u256(int(p.get("liquidity", 0)))]
    return sel + b"".join(words)


def tx_to_address(tx: BuilderTx) -> str:
    """``to`` field: the pool for single-hop swaps and LP calls, the router for multi-hop swaps."""
    if tx.kind is TxKind.SWAP:
        return address_of(tx.route[0]) if len(tx.route) == 1 else ROUTER_ADDRESS
    if tx.kind is TxKind.TRANSFER:
        return address_of(str(tx.payload.get("to", "")))
    return address_of(str(tx.payload.get("pool_id", "")))


Fragment = Tuple[int, bytes]  # (byte offset, revealed bytes)


def reveal_ranges(calldata: bytes, ranges: Iterable[Tuple[int, int]]) -> Tuple[Fragment, ...]:
    """Calldata fragments for ``ranges``, clipped to the calldata and merged when they overlap."""
    spans = sorted((max(0, s), min(len(calldata), e)) for s, e in ranges)
    merged: List[List[int]] = []
    for s, e in spans:
        if e <= s:
            continue
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return tuple((s, calldata[s:e]) for s, e in merged)


def _byte_map(fragments: Iterable[Fragment]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for off, data in fragments:
        for i, b in enumerate(data):
            out[off + i] = b
    return out


def revealed_words(fragments: Iterable[Fragment]) -> Dict[int, int]:
    """Calldata words whose 32 bytes are all revealed, as unsigned ints by word index."""
    bm = _byte_map(fragments)
    if not bm:
        return {}
    out: Dict[int, int] = {}
    last = max(bm)
    i = 0
    while True:
        s, e = word_range(i)
        if s > last:
            break
        if all(k in bm for k in range(s, e)):
            out[i] = int.from_bytes(bytes(bm[k] for k in range(s, e)), "big")
        i += 1
    return out


def revealed_selector(fragments: Iterable[Fragment]) -> Optional[bytes]:
    bm = _byte_map(fragments)
    if all(k in bm for k in range(SELECTOR_SIZE)):
        return bytes(bm[k] for k in range(SELECTOR_SIZE))
    return None


def _word_address(value: int) -> str:
    return "0x" + (value % (1 << 160)).to_bytes(20, "big").hex()


# ---------------------------------------------------------------------------
# Hints and privacy preferences
# ---------------------------------------------------------------------------


class Hint(Flag):
    NONE = 0
    HASH = 1
    CALLDATA = 2
    CONTRACT_ADDRESS = 4
    FUNCTION_SELECTOR = 8
    LOGS = 16
    DEFAULT_LOGS = 32  # swap logs with address and event topic only, no amounts
    BALANCE_DELTAS = 64  # sender's net token deltas from the relay's simulation
    CALLDATA_OFFSETS = 128  # only the byte ranges in PrivacyPreferences.calldata_offsets


HINT_NAMES: Dict[str, Hint] = {
    "hash": Hint.HASH,
    "tx_hash": Hint.HASH,
    "calldata": Hint.CALLDATA,
    "contract_address": Hint.CONTRACT_ADDRESS,
    "function_selector": Hint.FUNCTION_SELECTOR,
    "logs": Hint.LOGS,
    "default_logs": Hint.DEFAULT_LOGS,
    "balance_deltas": Hint.BALANCE_DELTAS,
    "calldata_offsets": Hint.CALLDATA_OFFSETS,
}
MEV_SHARE_DEFAULT = Hint.HASH | Hint.CONTRACT_ADDRESS | Hint.FUNCTION_SELECTOR | Hint.DEFAULT_LOGS
FULL_DISCLOSURE = (Hint.HASH | Hint.CALLDATA | Hint.CONTRACT_ADDRESS | Hint.FUNCTION_SELECTOR | Hint.LOGS
                   | Hint.BALANCE_DELTAS)


def parse_hints(names: Iterable[str]) -> Hint:
    """MEV-Share style hint names (``["calldata", "logs"]``) to a :class:`Hint` set; hash is always on."""
    out = Hint.HASH
    for n in names:
        key = str(n).strip().lower()
        if key not in HINT_NAMES:
            raise RelayError(f"unknown hint {n!r}")
        out |= HINT_NAMES[key]
    return out


def hint_names(hints: Hint) -> List[str]:
    seen: List[str] = []
    for name, flag in HINT_NAMES.items():
        if name != "tx_hash" and flag in hints and flag is not Hint.NONE:
            seen.append(name)
    return seen


@dataclass(frozen=True)
class PrivacyPreferences:
    hints: Hint = MEV_SHARE_DEFAULT
    share_with_searchers: bool = True  # False: Flashbots Protect without MEV-Share
    builders: Tuple[str, ...] = ()  # empty: every registered builder
    refund_bps: int = DEFAULT_REFUND_BPS
    max_block_number: Optional[int] = None  # None: received block + inclusion window
    calldata_offsets: Tuple[Tuple[int, int], ...] = ()
    fast: bool = False  # share with every registered builder regardless of ``builders``

    def __post_init__(self) -> None:
        if not 0 <= self.refund_bps <= 10_000:
            raise RelayError("refund_bps must be within [0, 10000]")
        for s, e in self.calldata_offsets:
            if s < 0 or e <= s:
                raise RelayError(f"bad calldata range [{s}, {e})")
        if self.calldata_offsets and Hint.CALLDATA_OFFSETS not in self.hints:
            raise RelayError("calldata_offsets need the calldata_offsets hint")

    @property
    def effective_hints(self) -> Hint:
        return (self.hints | Hint.HASH) if self.share_with_searchers else Hint.HASH

    def to_dict(self) -> Dict[str, Any]:
        return {"hints": hint_names(self.effective_hints), "share": self.share_with_searchers,
                "builders": list(self.builders), "refund_bps": self.refund_bps,
                "max_block_number": self.max_block_number,
                "calldata_offsets": [list(r) for r in self.calldata_offsets], "fast": self.fast}


def _pool_offsets(hops: int) -> Tuple[Tuple[int, int], ...]:
    return tuple(pool_word_range(i) for i in range(hops))


HINT_PROFILES: Dict[str, PrivacyPreferences] = {
    "protect": PrivacyPreferences(hints=Hint.HASH, share_with_searchers=False),
    "hash_only": PrivacyPreferences(hints=Hint.HASH),
    "mev_share_default": PrivacyPreferences(hints=MEV_SHARE_DEFAULT),
    "contract_selector": PrivacyPreferences(hints=Hint.HASH | Hint.CONTRACT_ADDRESS | Hint.FUNCTION_SELECTOR),
    "default_logs": PrivacyPreferences(hints=Hint.HASH | Hint.DEFAULT_LOGS),
    "balance_deltas": PrivacyPreferences(hints=Hint.HASH | Hint.BALANCE_DELTAS),
    "amount_offset": PrivacyPreferences(hints=Hint.HASH | Hint.CALLDATA_OFFSETS,
                                        calldata_offsets=(word_range(Word.AMOUNT_IN),)),
    "amount_route_offsets": PrivacyPreferences(
        hints=Hint.HASH | Hint.CALLDATA_OFFSETS,
        calldata_offsets=(word_range(Word.AMOUNT_IN), word_range(Word.TOKEN_IN), word_range(Word.HOP_COUNT))
        + _pool_offsets(4)),
    "full_logs": PrivacyPreferences(hints=Hint.HASH | Hint.LOGS),
    "full_calldata": PrivacyPreferences(hints=Hint.HASH | Hint.CALLDATA),
    "full": PrivacyPreferences(hints=FULL_DISCLOSURE),
}


# ---------------------------------------------------------------------------
# Relay-side simulation: logs and balance deltas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Log:
    address: str
    topics: Tuple[str, ...]
    data: Tuple[Tuple[str, int], ...] = ()

    def field(self, name: str) -> Optional[int]:
        for k, v in self.data:
            if k == name:
                return v
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"address": self.address, "topics": list(self.topics), "data": dict(self.data)}


@dataclass(frozen=True)
class SimOutcome:
    ok: bool
    logs: Tuple[Log, ...] = ()
    balance_deltas: Tuple[Tuple[str, int], ...] = ()
    gas_used: int = 0
    error: str = ""


def _swap_logs(pools: Mapping[str, Pool], tx: BuilderTx) -> Tuple[Log, ...]:
    path = RoutePath.from_route(pools, tx.token_in, tx.route)
    ex = execute_path(pools, path, tx.amount_in)
    sender, recipient = address_of(tx.sender), address_of(tx.sender)
    logs: List[Log] = []
    for h in ex.hops:
        addr = address_of(h.pool_id)
        pool = pools[h.pool_id]
        if isinstance(pool, V2Pool):
            a0_in, a1_in = (h.amount_in, 0) if h.zero_for_one else (0, h.amount_in)
            a0_out, a1_out = (0, h.amount_out) if h.zero_for_one else (h.amount_out, 0)
            assert h.reserves_after is not None
            logs.append(Log(addr, (V2_SYNC_TOPIC,), (("reserve0", h.reserves_after[0]),
                                                      ("reserve1", h.reserves_after[1]))))
            logs.append(Log(addr, (V2_SWAP_TOPIC, sender, recipient),
                            (("amount0In", a0_in), ("amount1In", a1_in), ("amount0Out", a0_out),
                             ("amount1Out", a1_out))))
        else:
            amount0 = h.amount_in if h.zero_for_one else -h.amount_out
            amount1 = -h.amount_out if h.zero_for_one else h.amount_in
            data: List[Tuple[str, int]] = [("amount0", amount0), ("amount1", amount1),
                                           ("sqrtPriceX96", h.sqrt_price_after_x96),
                                           ("liquidity", h.liquidity_after), ("tick", int(h.tick_after or 0))]
            topic = V3_SWAP_TOPIC
            if isinstance(pool, V4Pool):
                topic = V4_SWAP_TOPIC
                data.append(("fee", pool.effective_fee()))
            logs.append(Log(addr, (topic, sender, recipient), tuple(data)))
    return tuple(logs)


def simulate_private_tx(state: BuilderState, tx: BuilderTx) -> SimOutcome:
    """Simulate ``tx`` on a copy of ``state`` (base fee 0) for hint generation."""
    st = state.copy()
    # Simulate as if the sender's earlier pending nonces had landed, like a relay's
    # "latest + pending" simulation, so queued transactions still produce hints.
    if tx.nonce > st.nonce(tx.sender):
        st.nonces[tx.sender] = tx.nonce
    before = dict(st.fork.balances.get(tx.sender, {}))
    t = execute_tx(st, tx, 0, BLOCK_GAS_LIMIT, coinbase="relay-sim", trace=False)
    if t.status is not Status.SUCCESS:
        return SimOutcome(False, gas_used=t.gas_used, error=t.error or t.status.value)
    after = st.fork.balances.get(tx.sender, {})
    gas_paid = t.priority_fee_wei + t.base_fee_burned_wei
    deltas: Dict[str, int] = {}
    for tok in sorted(set(before) | set(after)):
        d = after.get(tok, 0) - before.get(tok, 0)
        if tok == ETH:
            d += gas_paid
        if d:
            deltas[tok] = d
    logs: Tuple[Log, ...] = ()
    if tx.kind is TxKind.SWAP:
        try:
            logs = _swap_logs(state.pools, tx)
        except (RouterError, MASEError):  # pragma: no cover - the sim above already succeeded
            logs = ()
    return SimOutcome(True, logs, tuple(sorted(deltas.items())), t.gas_used)


# ---------------------------------------------------------------------------
# Hint views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HintView:
    """What a searcher sees on the relay's event stream for one private transaction."""

    tx_hash: str
    hints: Hint
    block: int
    max_block: int
    to: Optional[str] = None
    selector: Optional[bytes] = None
    calldata: Tuple[Fragment, ...] = ()
    logs: Tuple[Log, ...] = ()
    balance_deltas: Optional[Tuple[Tuple[str, int], ...]] = None

    @property
    def revealed_calldata_bytes(self) -> int:
        return len(_byte_map(self.calldata))

    @property
    def leakage_score(self) -> int:
        """Rough count of revealed bytes (selector 4, address 20, calldata bytes, 32 per log/delta value)."""
        score = self.revealed_calldata_bytes
        if self.selector is not None and revealed_selector(self.calldata) is None:
            score += SELECTOR_SIZE
        if self.to is not None:
            score += 20
        for lg in self.logs:
            score += 20 + 32 * len(lg.topics) + 32 * len(lg.data)
        score += 32 * len(self.balance_deltas or ())
        return score

    def to_dict(self) -> Dict[str, Any]:
        """MEV-Share SSE event shape: hash, logs and a txs entry with to / selector / calldata."""
        tx: Dict[str, Any] = {}
        if self.to is not None:
            tx["to"] = self.to
        if self.selector is not None:
            tx["functionSelector"] = "0x" + self.selector.hex()
        if self.calldata:
            tx["callData"] = [{"offset": o, "data": "0x" + d.hex()} for o, d in self.calldata]
        out: Dict[str, Any] = {"hash": self.tx_hash, "logs": [lg.to_dict() for lg in self.logs],
                               "txs": [tx] if tx else [], "block": self.block, "maxBlock": self.max_block}
        if self.balance_deltas is not None:
            out["balanceDeltas"] = dict(self.balance_deltas)
        return out


def build_hint_view(tx: BuilderTx, prefs: PrivacyPreferences, state: BuilderState, block: int,
                    max_block: Optional[int] = None, sim: Optional[SimOutcome] = None) -> HintView:
    hints = prefs.effective_hints
    max_block = block + DEFAULT_INCLUSION_WINDOW if max_block is None else max_block
    calldata = encode_calldata(tx)
    ranges: List[Tuple[int, int]] = []
    if Hint.CALLDATA in hints:
        ranges.append((0, len(calldata)))
    if Hint.CALLDATA_OFFSETS in hints:
        ranges.extend(prefs.calldata_offsets)
    fragments = reveal_ranges(calldata, ranges)
    selector = calldata[:SELECTOR_SIZE] if Hint.FUNCTION_SELECTOR in hints else None
    to = tx_to_address(tx) if Hint.CONTRACT_ADDRESS in hints else None
    need_sim = bool(hints & (Hint.LOGS | Hint.DEFAULT_LOGS | Hint.BALANCE_DELTAS))
    if need_sim and sim is None:
        sim = simulate_private_tx(state, tx)
    logs: Tuple[Log, ...] = ()
    deltas: Optional[Tuple[Tuple[str, int], ...]] = None
    if sim is not None and sim.ok:
        if Hint.LOGS in hints:
            logs = sim.logs
        elif Hint.DEFAULT_LOGS in hints:
            logs = tuple(Log(lg.address, lg.topics[:1]) for lg in sim.logs if lg.topics[0] in SWAP_TOPICS)
        if Hint.BALANCE_DELTAS in hints:
            deltas = sim.balance_deltas
    return HintView(tx.tx_hash, hints, block, max_block, to, selector, fragments, logs, deltas)


# ---------------------------------------------------------------------------
# Searcher-side inference
# ---------------------------------------------------------------------------


class Knowledge(IntEnum):
    BLIND = 0  # hash only
    KIND = 1  # transaction type known (selector) but nothing about the trade
    POOL = 2  # pools known, direction and size unknown
    DIRECTION = 3  # pools and direction known, size unknown
    AMOUNT_NO_ROUTE = 4  # tokens and size known, several candidate routes
    EXACT = 5  # route, direction and size known


@dataclass(frozen=True)
class VictimHypothesis:
    route: Tuple[str, ...]
    token_in: str
    amount_in: int
    amount_out_min: int
    weight: Fraction
    source: str

    def as_tx(self, tx_hash: str, sender: str = "victim?", nonce: int = 0) -> BuilderTx:
        return BuilderTx(tx_hash, sender, nonce, BLOCK_GAS_LIMIT, 10**30, 0, TxKind.SWAP,
                         {"route": list(self.route), "token_in": self.token_in, "amount_in": self.amount_in,
                          "amount_out_min": self.amount_out_min})


@dataclass(frozen=True)
class Inference:
    tx_hash: str
    knowledge: Knowledge
    hypotheses: Tuple[VictimHypothesis, ...]
    kind: Optional[TxKind] = None
    route: Optional[Tuple[str, ...]] = None
    token_in: Optional[str] = None
    amount_in: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"tx_hash": self.tx_hash, "knowledge": self.knowledge.name, "kind": self.kind.value if self.kind else None,
                "route": list(self.route) if self.route else None, "token_in": self.token_in,
                "amount_in": self.amount_in, "hypotheses": len(self.hypotheses)}


class PoolDirectory:
    """Public address book a searcher builds from the pool set it can see on chain."""

    def __init__(self, pools: Mapping[str, Pool]) -> None:
        self.pool_by_address = {address_of(pid): pid for pid in pools}
        toks: Set[str] = set()
        for p in pools.values():
            toks.update(p.tokens())
        self.token_by_address = {address_of(t): t for t in sorted(toks)}

    def pool(self, address: Optional[str]) -> Optional[str]:
        return self.pool_by_address.get(address) if address else None

    def token(self, address: Optional[str]) -> Optional[str]:
        return self.token_by_address.get(address) if address else None


def _valid_route(pools: Mapping[str, Pool], token_in: str, route: Sequence[str]) -> bool:
    try:
        RoutePath.from_route(pools, token_in, route)
    except (RouterError, MASEError):
        return False
    return len(set(route)) == len(route)


def _size_grid(pools: Mapping[str, Pool], route: Sequence[str], token_in: str, points: int) -> List[int]:
    pool = pools[route[0]]
    zfo, _ = _direction(pool, token_in)
    scale = input_reserve_scale(pool, zfo)
    return geometric_grid(max(1, scale // 10_000), max(1, scale * 200 // 10_000), points)


def _log_direction(pools: Mapping[str, Pool], lg: Log, pid: str) -> Optional[Tuple[str, int]]:
    """(token_in, amount_in) for a full swap log."""
    pool = pools[pid]
    t0, t1 = pool.tokens()
    if lg.topics and lg.topics[0] == V2_SWAP_TOPIC:
        a0, a1 = lg.field("amount0In"), lg.field("amount1In")
        if a0:
            return t0, a0
        if a1:
            return t1, a1
        return None
    a0, a1 = lg.field("amount0"), lg.field("amount1")
    if a0 is None or a1 is None:
        return None
    if a0 > 0:
        return t0, a0
    if a1 > 0:
        return t1, a1
    return None


def infer_victim(view: HintView, pools: Mapping[str, Pool], size_grid: int = 6, max_route_hops: int = 2,
                 directory: Optional[PoolDirectory] = None) -> Inference:
    """Reconstruct the hidden trade from a hint view and public pool state.

    Sources, strongest first: calldata words (amount, token, route), full
    swap logs (route, direction and gross input), balance deltas (tokens and
    input; the route is then recovered by quoting every candidate path and
    keeping the ones that reproduce the output delta exactly), topic-only
    logs or the ``to`` address (route only). Unknown sizes become a
    geometric grid of hypotheses; unknown directions become one set per
    direction.
    """
    d = directory or PoolDirectory(pools)
    words = revealed_words(view.calldata)
    sel = view.selector if view.selector is not None else revealed_selector(view.calldata)
    kind = KIND_BY_SELECTOR.get(sel) if sel is not None else None
    if kind is not None and kind is not TxKind.SWAP:
        return Inference(view.tx_hash, Knowledge.KIND, (), kind)

    route: Optional[Tuple[str, ...]] = None
    if Word.HOP_COUNT in words:
        n = words[Word.HOP_COUNT]
        if 0 < n <= 8 and all(Word.POOLS + i in words for i in range(n)):
            pids = [d.pool(_word_address(words[Word.POOLS + i])) for i in range(n)]
            if all(pids):
                route = tuple(pids)  # type: ignore[arg-type]
    swap_logs = [lg for lg in view.logs if lg.topics and lg.topics[0] in SWAP_TOPICS]
    if route is None and swap_logs:
        pids = [d.pool(lg.address) for lg in swap_logs]
        if all(pids):
            route = tuple(pids)  # type: ignore[arg-type]
    if route is None and view.to is not None and d.pool(view.to) is not None:
        route = (d.pool(view.to),)  # type: ignore[assignment]

    token_in: Optional[str] = None
    amount: Optional[int] = None
    if Word.TOKEN_IN in words:
        token_in = d.token(_word_address(words[Word.TOKEN_IN]))
    if Word.AMOUNT_IN in words:
        amount = words[Word.AMOUNT_IN]
    if swap_logs and swap_logs[0].data and route is not None:
        got = _log_direction(pools, swap_logs[0], route[0])
        if got is not None:
            token_in = token_in or got[0]
            amount = amount if amount is not None else got[1]
    token_out: Optional[str] = None
    if view.balance_deltas is not None:
        deltas = dict(view.balance_deltas)
        neg = [t for t, v in deltas.items() if v < 0 and t != ETH]
        pos = [t for t, v in deltas.items() if v > 0 and t != ETH]
        if len(neg) == 1:
            token_in = token_in or neg[0]
            amount = amount if amount is not None else -deltas[neg[0]]
        if len(pos) == 1:
            token_out = pos[0]
    min_out = words.get(Word.AMOUNT_OUT_MIN, 0)
    if kind is None and (route is not None or token_in is not None):
        kind = TxKind.SWAP

    def hyp(r: Sequence[str], t: str, a: int, w: Fraction, src: str) -> VictimHypothesis:
        return VictimHypothesis(tuple(r), t, a, min_out, w, src)

    if route is not None and token_in is not None and amount is not None and amount > 0:
        if _valid_route(pools, token_in, route):
            return Inference(view.tx_hash, Knowledge.EXACT, (hyp(route, token_in, amount, Fraction(1), "exact"),),
                             kind, route, token_in, amount)
    if route is None and token_in is not None and amount is not None and amount > 0:
        candidates: List[Tuple[str, ...]] = []
        if token_out is not None and token_out != token_in:
            router = MultiHopRouter(pools.values(), max_hops=max_route_hops)
            candidates = [p.pool_ids for p in router.paths(token_in, token_out, max_route_hops)]
        else:
            candidates = [(pid,) for pid in sorted(pools) if token_in in pools[pid].tokens()]
        if token_out is not None and view.balance_deltas is not None:
            want = dict(view.balance_deltas).get(token_out)
            exact = [r for r in candidates
                     if quote_path_output(pools, RoutePath.from_route(pools, token_in, r), amount) == want]
            if len(exact) == 1:
                r = exact[0]
                return Inference(view.tx_hash, Knowledge.EXACT, (hyp(r, token_in, amount, Fraction(1), "deltas+quote"),),
                                 kind, r, token_in, amount)
            if exact:
                candidates = exact
        if candidates:
            w = Fraction(1, len(candidates))
            return Inference(view.tx_hash, Knowledge.AMOUNT_NO_ROUTE,
                             tuple(hyp(r, token_in, amount, w, "amount") for r in candidates),
                             kind, None, token_in, amount)
    if route is not None:
        directions = [token_in] if token_in is not None else list(pools[route[0]].tokens())
        directions = [t for t in directions if t is not None and _valid_route(pools, t, route)]
        hyps: List[VictimHypothesis] = []
        for t in directions:
            sizes = [amount] if amount is not None and amount > 0 else _size_grid(pools, route, t, size_grid)
            hyps.extend(hyp(route, t, a, Fraction(1), "grid") for a in sizes)
        if hyps:
            w = Fraction(1, len(hyps))
            hyps = [replace(h, weight=w) for h in hyps]
            level = Knowledge.DIRECTION if len(directions) == 1 else Knowledge.POOL
            return Inference(view.tx_hash, level, tuple(hyps), kind, route,
                             token_in if len(directions) == 1 else None, amount)
    return Inference(view.tx_hash, Knowledge.KIND if kind is not None else Knowledge.BLIND, (), kind)


def select_hypotheses(hyps: Sequence[VictimHypothesis], k: int) -> List[VictimHypothesis]:
    """Top-``k`` by weight; equal-weight sets are sampled evenly so a size grid keeps its spread."""
    if k <= 0 or not hyps:
        return []
    ordered = sorted(enumerate(hyps), key=lambda t: (-t[1].weight, t[0]))
    if len(ordered) <= k:
        return [h for _, h in ordered]
    top_w = ordered[0][1].weight
    same = [h for _, h in ordered if h.weight == top_w]
    if len(same) >= k:
        step = (len(same) - 1) / max(1, k - 1)
        return [same[round(i * step)] for i in range(k)]
    return [h for _, h in ordered[:k]]


# ---------------------------------------------------------------------------
# Predatory exposure (what leaks if a builder ignored backrun-only rules)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExposureResult:
    tx_hash: str
    profile: str
    knowledge: Knowledge
    hypotheses: int
    leakage_score: int
    perfect_profit_native: int
    realized_profit_native: int
    victim_reverted: bool
    attempted: bool

    @property
    def leak_efficiency(self) -> Fraction:
        if self.perfect_profit_native <= 0:
            return Fraction(0)
        return Fraction(max(0, min(self.realized_profit_native, self.perfect_profit_native)),
                        self.perfect_profit_native)

    def to_dict(self) -> Dict[str, Any]:
        return {"tx_hash": self.tx_hash, "profile": self.profile, "knowledge": self.knowledge.name,
                "hypotheses": self.hypotheses, "leakage_score": self.leakage_score,
                "perfect_profit_native": self.perfect_profit_native,
                "realized_profit_native": self.realized_profit_native,
                "leak_efficiency": float(self.leak_efficiency), "victim_reverted": self.victim_reverted,
                "attempted": self.attempted}


def realize_sandwich(pools: Mapping[str, Pool], victim: BuilderTx, pool_id: str, token_in: str,
                     frontrun_in: int) -> Tuple[Optional[int], bool]:
    """Run front-run, the *real* victim and back-run on clones.

    Returns ``(back_out - frontrun_in, victim_reverted)``; profit is None when
    the front-run or back-run cannot fill.
    """
    work = {pid: clone_pool(p) for pid, p in pools.items()}
    target = work.get(pool_id)
    if target is None:
        return None, False
    try:
        zfo, _ = _direction(target, token_in)
    except MASEError:
        return None, False
    front = _apply_full(target, frontrun_in, zfo)
    if front is None:
        return None, False
    reverted = False
    try:
        path = RoutePath.from_route(work, victim.token_in, victim.route)
        out = quote_path_output(work, path, victim.amount_in)
        if out is None or out < victim.amount_out_min:
            reverted = True
        else:
            execute_path(work, path, victim.amount_in, mutate=True)
    except (RouterError, MASEError):
        reverted = True
    back = _apply_full(target, front.amount_out, not zfo)
    if back is None:
        return None, reverted
    return back.amount_out - frontrun_in, reverted


def measure_exposure(state: BuilderState, tx: BuilderTx, prefs: PrivacyPreferences, profile: str = "",
                     native: str = "WETH", grid: int = 12, refine: int = 6, max_hypotheses: int = 4,
                     block: Optional[int] = None) -> ExposureResult:
    block = state.block_number if block is None else block
    view = build_hint_view(tx, prefs, state, block)
    pools = state.pools
    perfect = best_sandwich(pools, tx, grid, refine, native=native) if tx.kind is TxKind.SWAP else None
    perfect_native = max(0, native_value(pools, perfect.token_in, perfect.profit, native)) if perfect else 0
    inf = infer_victim(view, pools)
    chosen = None
    best_expected = Fraction(0)
    for i, h in enumerate(select_hypotheses(inf.hypotheses, max_hypotheses)):
        cand = best_sandwich(pools, h.as_tx(f"{tx.tx_hash}:h{i}"), grid, refine, native=native)
        if cand is None:
            continue
        expected = h.weight * native_value(pools, cand.token_in, cand.profit, native)
        if chosen is None or expected > best_expected:
            chosen, best_expected = cand, expected
    realized_native, reverted = 0, False
    if chosen is not None:
        profit, reverted = realize_sandwich(pools, tx, chosen.pool_id, chosen.token_in, chosen.frontrun_in)
        if profit is not None:
            realized_native = native_value(pools, chosen.token_in, profit, native)
    return ExposureResult(tx.tx_hash, profile, inf.knowledge, len(inf.hypotheses), view.leakage_score,
                          perfect_native, realized_native, reverted, chosen is not None)


def _exposure_chunk(task: Tuple[Any, ...]) -> List[ExposureResult]:
    payload, items, native, grid, refine, max_hyp, block = task
    state = BuilderState.from_payload(payload)
    return [measure_exposure(state, tx, prefs, name, native, grid, refine, max_hyp, block)
            for tx, prefs, name in items]


@dataclass
class ExposureMatrix:
    results: List[ExposureResult]
    elapsed_s: float = 0.0

    def by_profile(self) -> Dict[str, List[ExposureResult]]:
        out: Dict[str, List[ExposureResult]] = defaultdict(list)
        for r in self.results:
            out[r.profile].append(r)
        return dict(out)

    def rows(self) -> Dict[str, Dict[str, Any]]:
        rows: Dict[str, Dict[str, Any]] = {}
        for name, rs in sorted(self.by_profile().items()):
            effs = [r.leak_efficiency for r in rs]
            rows[name] = {
                "txs": len(rs),
                "mean_leak_efficiency": float(sum(effs, Fraction(0)) / len(effs)) if effs else 0.0,
                "max_leak_efficiency": float(max(effs)) if effs else 0.0,
                "exact_knowledge_share": sum(1 for r in rs if r.knowledge is Knowledge.EXACT) / len(rs),
                "mean_leakage_score": sum(r.leakage_score for r in rs) / len(rs),
                "perfect_profit_native": sum(r.perfect_profit_native for r in rs),
                "realized_profit_native": sum(max(0, r.realized_profit_native) for r in rs),
                "victim_reverts": sum(1 for r in rs if r.victim_reverted),
            }
        return rows

    def profiles_above(self, threshold: float) -> List[str]:
        """Profiles whose mean leak efficiency reaches ``threshold``, least revealing first."""
        rows = self.rows()
        hits = [n for n, r in rows.items() if r["mean_leak_efficiency"] >= threshold]
        return sorted(hits, key=lambda n: (rows[n]["mean_leakage_score"], n))

    def minimal_leak_profile(self, threshold: float) -> Optional[str]:
        hits = self.profiles_above(threshold)
        return hits[0] if hits else None

    def summary(self) -> Dict[str, Any]:
        return {"profiles": self.rows(), "evaluations": len(self.results), "elapsed_s": round(self.elapsed_s, 4)}


def hint_exposure_matrix(state: BuilderState, txs: Sequence[BuilderTx],
                         profiles: Optional[Mapping[str, PrivacyPreferences]] = None,
                         executor: Optional[WorkerPool] = None, native: str = "WETH", grid: int = 12,
                         refine: int = 6, max_hypotheses: int = 4, chunks: int = RELAY_CHUNKS) -> ExposureMatrix:
    """Exposure of every swap in ``txs`` under every hint profile, in fixed parallel chunks."""
    profiles = HINT_PROFILES if profiles is None else profiles
    items = [(tx, prefs, name) for tx in txs if tx.kind is TxKind.SWAP for name, prefs in profiles.items()]
    return exposure_for_items(state, items, executor, native, grid, refine, max_hypotheses, chunks)


def exposure_for_items(state: BuilderState, items: Sequence[Tuple[BuilderTx, PrivacyPreferences, str]],
                       executor: Optional[WorkerPool] = None, native: str = "WETH", grid: int = 12,
                       refine: int = 6, max_hypotheses: int = 4, chunks: int = RELAY_CHUNKS) -> ExposureMatrix:
    """Exposure of explicit ``(tx, prefs, profile_name)`` items (swaps only), in fixed parallel chunks."""
    started = time.perf_counter()
    items = [it for it in items if it[0].kind is TxKind.SWAP]
    payload = state.to_payload()
    tasks = [(payload, chunk, native, grid, refine, max_hypotheses, state.block_number)
             for chunk in chunked(items, max(1, chunks))]
    if executor is None:
        parts = [_exposure_chunk(t) for t in tasks]
    else:
        parts = executor.map(_exposure_chunk, tasks)
    return ExposureMatrix([r for p in parts for r in p], time.perf_counter() - started)


# ---------------------------------------------------------------------------
# Signatures and encrypted envelopes
# ---------------------------------------------------------------------------


def sign_payload(key: bytes, payload: bytes) -> str:
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def verify_signature(key: bytes, payload: bytes, signature: str) -> bool:
    return hmac.compare_digest(sign_payload(key, payload), signature)


@dataclass(frozen=True)
class Envelope:
    builder: str
    nonce: bytes
    ciphertext: bytes
    tag: bytes


def _subkey(key: bytes, label: bytes) -> bytes:
    return hashlib.sha256(key + b"|" + label).digest()


def _keystream(key: bytes, nonce: bytes, n: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < n:
        out += hashlib.sha256(key + nonce + counter.to_bytes(8, "big")).digest()
        counter += 1
    return bytes(out[:n])


def seal_envelope(key: bytes, builder: str, plaintext: bytes, nonce: bytes) -> Envelope:
    """Encrypt-then-MAC (SHA-256 keystream + HMAC); simulation-grade, see module docstring."""
    ks = _keystream(_subkey(key, b"enc"), nonce, len(plaintext))
    ct = bytes(a ^ b for a, b in zip(plaintext, ks))
    tag = hmac.new(_subkey(key, b"mac"), builder.encode() + nonce + ct, hashlib.sha256).digest()
    return Envelope(builder, nonce, ct, tag)


def open_envelope(key: bytes, env: Envelope, builder: Optional[str] = None) -> bytes:
    if builder is not None and env.builder != builder:
        raise RelayError("envelope addressed to another builder")
    tag = hmac.new(_subkey(key, b"mac"), env.builder.encode() + env.nonce + env.ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, env.tag):
        raise RelayError("decryption_failed")
    ks = _keystream(_subkey(key, b"enc"), env.nonce, len(env.ciphertext))
    return bytes(a ^ b for a, b in zip(env.ciphertext, ks))


def tx_from_dict(d: Mapping[str, Any]) -> BuilderTx:
    return BuilderTx(str(d["tx_hash"]), str(d["sender"]), int(d["nonce"]), int(d["gas_limit"]),
                     int(d["max_fee_per_gas"]), int(d["max_priority_fee_per_gas"]), TxKind(d["kind"]),
                     dict(d.get("payload", {})))


def bundle_to_wire(b: Bundle) -> bytes:
    body = {"bundle_id": b.bundle_id, "searcher": b.searcher, "coinbase_transfer_wei": b.coinbase_transfer_wei,
            "reverting_tx_hashes": sorted(b.reverting_tx_hashes), "target_tx": b.target_tx,
            "meta": dict(b.meta), "txs": [t.to_dict() for t in b.txs]}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def bundle_from_wire(data: bytes) -> Bundle:
    d = json.loads(data.decode())
    return Bundle(d["bundle_id"], tuple(tx_from_dict(t) for t in d["txs"]), int(d["coinbase_transfer_wei"]),
                  d.get("searcher", ""), frozenset(d.get("reverting_tx_hashes", ())), d.get("target_tx", ""),
                  dict(d.get("meta", {})))


# ---------------------------------------------------------------------------
# MEV-Share bundles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BodyItem:
    """One bundle body entry: a reference to a private tx by hash, or a signed searcher tx."""

    hash: Optional[str] = None
    tx: Optional[BuilderTx] = None
    can_revert: bool = False

    def __post_init__(self) -> None:
        if (self.hash is None) == (self.tx is None):
            raise RelayError("body item needs exactly one of hash or tx")

    def to_dict(self) -> Dict[str, Any]:
        if self.hash is not None:
            return {"hash": self.hash}
        assert self.tx is not None
        return {"tx": self.tx.to_dict(), "canRevert": self.can_revert}


@dataclass(frozen=True, eq=False)
class MevShareBundle:
    bundle_id: str
    searcher: str
    body: Tuple[BodyItem, ...]
    block: int
    max_block: int
    coinbase_transfer_wei: int
    signature: str = ""
    version: str = "v0.1"

    def payload_bytes(self) -> bytes:
        doc = {"version": self.version, "id": self.bundle_id, "searcher": self.searcher,
               "inclusion": {"block": self.block, "maxBlock": self.max_block},
               "body": [b.to_dict() for b in self.body], "payment": self.coinbase_transfer_wei}
        return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()

    def signed(self, key: bytes) -> "MevShareBundle":
        return replace(self, signature=sign_payload(key, self.payload_bytes()))

    @property
    def ref_hashes(self) -> Tuple[str, ...]:
        return tuple(b.hash for b in self.body if b.hash is not None)

    @property
    def searcher_txs(self) -> Tuple[BuilderTx, ...]:
        return tuple(b.tx for b in self.body if b.tx is not None)

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(self.payload_bytes().decode()) | {"signature": self.signature}


# ---------------------------------------------------------------------------
# Relay
# ---------------------------------------------------------------------------


class TxStatus(str, Enum):
    PENDING = "pending"
    INCLUDED = "included"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class PrivateTxRecord:
    tx: BuilderTx
    prefs: PrivacyPreferences
    received_block: int
    max_block: int
    view: HintView
    seq: int
    status: TxStatus = TxStatus.PENDING
    included_block: Optional[int] = None
    reason: str = ""
    refund_wei: int = 0
    winning_bundle: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"tx_hash": self.tx.tx_hash, "sender": self.tx.sender, "status": self.status.value,
                "received_block": self.received_block, "max_block": self.max_block,
                "included_block": self.included_block, "reason": self.reason, "refund_wei": self.refund_wei,
                "winning_bundle": self.winning_bundle, "prefs": self.prefs.to_dict()}


@dataclass
class SearcherAccount:
    name: str
    key: bytes
    submitted: int = 0
    accepted: int = 0
    rejected: int = 0
    failed_simulations: int = 0
    wins: int = 0

    @property
    def high_priority(self) -> bool:
        """Searchers whose bundles keep failing simulation fall to the low-priority queue."""
        return self.failed_simulations <= 3 * self.wins + 8


@dataclass(frozen=True)
class BundleReceipt:
    bundle_id: str
    accepted: bool
    reason: str = ""


@dataclass
class AuctionEntry:
    tx_hash: str
    winner: Optional[str]
    payment_wei: int
    refund_wei: int
    candidates: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {"tx_hash": self.tx_hash, "winner": self.winner, "payment_wei": self.payment_wei,
                "refund_wei": self.refund_wei, "candidates": list(self.candidates)}


@dataclass
class RelayAuction:
    block: int
    bundles: List[Bundle]
    entries: Dict[str, AuctionEntry]
    allowed_builders: Dict[str, Tuple[str, ...]]  # builder bundle id -> builders allowed to receive it

    def summary(self) -> Dict[str, Any]:
        return {"block": self.block, "bundles": len(self.bundles),
                "auctions": {h: e.to_dict() for h, e in self.entries.items()},
                "backruns_won": sum(1 for e in self.entries.values() if e.winner),
                "payments_wei": sum(e.payment_wei for e in self.entries.values()),
                "refunds_wei": sum(e.refund_wei for e in self.entries.values())}


@dataclass
class Settlement:
    block: int
    included: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)
    expired: List[str] = field(default_factory=list)
    refunds_wei: int = 0
    refunds: Dict[str, int] = field(default_factory=dict)
    via_public: List[str] = field(default_factory=list)  # should stay empty: private txs only land via bundles

    def to_dict(self) -> Dict[str, Any]:
        return {"block": self.block, "included": list(self.included), "failed": dict(self.failed),
                "expired": list(self.expired), "refunds_wei": self.refunds_wei, "refunds": dict(self.refunds),
                "via_public": list(self.via_public)}


def _owned_by(sender: str, searcher: str) -> bool:
    """A searcher may send from its own address or its ``<searcher>#<suffix>`` sub-accounts."""
    return sender == searcher or sender.startswith(searcher + "#")


def _derive_key(label: str) -> bytes:
    return hashlib.sha256(f"key:{label}".encode()).digest()


class PrivateRelay:
    """Flashbots Protect / MEV-Share relay: private pool, hint stream, bundle auction and delivery."""

    def __init__(self, name: str = "flashbots", inclusion_window: int = DEFAULT_INCLUSION_WINDOW,
                 max_bundles_per_searcher_per_block: int = 16, max_future_blocks: int = 25) -> None:
        if inclusion_window < 0 or max_bundles_per_searcher_per_block < 1 or max_future_blocks < 0:
            raise ConfigError("bad relay limits")
        self.name = name
        self.inclusion_window = inclusion_window
        self.max_bundles_per_searcher_per_block = max_bundles_per_searcher_per_block
        self.max_future_blocks = max_future_blocks
        self.builders: Dict[str, bytes] = {}
        self.searchers: Dict[str, SearcherAccount] = {}
        self.records: Dict[str, PrivateTxRecord] = {}
        self.bundles: Dict[str, MevShareBundle] = {}
        self.bundle_status: Dict[str, str] = {}
        self.events: List[HintView] = []
        self.stats: Dict[str, int] = defaultdict(int)
        self._submissions: Dict[Tuple[str, int], int] = defaultdict(int)
        self._bundle_seq: Dict[str, int] = {}
        self._seq = 0

    # -- registration -------------------------------------------------------

    def register_builder(self, name: str, key: Optional[bytes] = None) -> bytes:
        key = key or _derive_key(f"builder:{self.name}:{name}")
        self.builders[name] = key
        return key

    def register_searcher(self, name: str, key: Optional[bytes] = None) -> SearcherAccount:
        acct = SearcherAccount(name, key or _derive_key(f"searcher:{self.name}:{name}"))
        self.searchers[name] = acct
        return acct

    # -- private transactions -------------------------------------------------

    def send_private_transaction(self, tx: BuilderTx, state: BuilderState, block: int,
                                 prefs: Optional[PrivacyPreferences] = None) -> str:
        """eth_sendPrivateTransaction: store privately and publish the hint view."""
        prefs = prefs or PrivacyPreferences()
        if tx.tx_hash in self.records:
            raise RelayError("already_known")
        if tx.max_priority_fee_per_gas > tx.max_fee_per_gas:
            raise RelayError("tip_above_fee_cap")
        if tx.nonce < state.nonce(tx.sender):
            raise RelayError("nonce_too_low")
        unknown = [b for b in prefs.builders if b not in self.builders]
        if unknown:
            raise RelayError(f"unknown_builders:{','.join(unknown)}")
        max_block = prefs.max_block_number if prefs.max_block_number is not None else block + self.inclusion_window
        if max_block < block:
            raise RelayError("max_block_in_past")
        for rec in self.records.values():
            if rec.status is TxStatus.PENDING and rec.tx.sender == tx.sender and rec.tx.nonce == tx.nonce:
                old, new = rec.tx, tx
                bump = 100 + REPLACEMENT_BUMP_PCT
                if new.max_priority_fee_per_gas * 100 < old.max_priority_fee_per_gas * bump \
                        or new.max_fee_per_gas * 100 < old.max_fee_per_gas * bump:
                    raise RelayError("replacement_underpriced")
                rec.status, rec.reason = TxStatus.CANCELLED, f"replaced_by:{tx.tx_hash}"
                self.stats["replaced"] += 1
        view = build_hint_view(tx, prefs, state, block, max_block)
        self._seq += 1
        self.records[tx.tx_hash] = PrivateTxRecord(tx, prefs, block, max_block, view, self._seq)
        if prefs.share_with_searchers:
            self.events.append(view)
        self.stats["private_txs"] += 1
        return tx.tx_hash

    def cancel_private_transaction(self, tx_hash: str) -> bool:
        rec = self.records.get(tx_hash)
        if rec is None or rec.status is not TxStatus.PENDING:
            return False
        rec.status, rec.reason = TxStatus.CANCELLED, "cancelled_by_user"
        self.stats["cancelled"] += 1
        return True

    def status(self, tx_hash: str) -> Optional[TxStatus]:
        rec = self.records.get(tx_hash)
        return rec.status if rec else None

    def stream(self, since: int = 0) -> List[HintView]:
        return list(self.events[since:])

    def pending(self, block: int) -> List[PrivateTxRecord]:
        return sorted((r for r in self.records.values() if r.status is TxStatus.PENDING
                       and r.received_block <= block <= r.max_block), key=lambda r: r.seq)

    def is_private(self, tx_hash: str) -> bool:
        return tx_hash in self.records

    # -- bundles --------------------------------------------------------------

    def _reject(self, acct: Optional[SearcherAccount], bundle_id: str, reason: str) -> BundleReceipt:
        self.stats[f"bundle_rejected:{reason.split(':')[0]}"] += 1
        if acct is not None:
            acct.rejected += 1
        return BundleReceipt(bundle_id, False, reason)

    def send_bundle(self, bundle: MevShareBundle, block: int) -> BundleReceipt:
        """mev_sendBundle: authenticate, enforce backrun-only privacy and inclusion windows."""
        acct = self.searchers.get(bundle.searcher)
        if acct is None:
            return self._reject(None, bundle.bundle_id, "unknown_searcher")
        acct.submitted += 1
        if not verify_signature(acct.key, bundle.payload_bytes(), bundle.signature):
            return self._reject(acct, bundle.bundle_id, "bad_signature")
        if bundle.bundle_id in self.bundles:
            return self._reject(acct, bundle.bundle_id, "duplicate_bundle")
        if not bundle.body:
            return self._reject(acct, bundle.bundle_id, "empty_body")
        if bundle.max_block < bundle.block:
            return self._reject(acct, bundle.bundle_id, "bad_block_range")
        if bundle.max_block < block:
            return self._reject(acct, bundle.bundle_id, "block_range_expired")
        if bundle.block > block + self.max_future_blocks:
            return self._reject(acct, bundle.bundle_id, "too_far_in_future")
        if bundle.coinbase_transfer_wei < 0:
            return self._reject(acct, bundle.bundle_id, "negative_payment")
        refs = bundle.ref_hashes
        if len(refs) != 1:
            return self._reject(acct, bundle.bundle_id, "need_exactly_one_hash_ref")
        rec = self.records.get(refs[0])
        if rec is None or not rec.prefs.share_with_searchers:
            return self._reject(acct, bundle.bundle_id, "unknown_hash")
        if rec.status is not TxStatus.PENDING:
            return self._reject(acct, bundle.bundle_id, f"target_{rec.status.value}")
        ref_pos = next(i for i, b in enumerate(bundle.body) if b.hash is not None)
        if any(b.tx is not None for b in bundle.body[:ref_pos]):
            return self._reject(acct, bundle.bundle_id, "privacy_violation:frontrun")
        if not bundle.searcher_txs:
            return self._reject(acct, bundle.bundle_id, "no_backrun")
        if any(not _owned_by(t.sender, bundle.searcher) for t in bundle.searcher_txs):
            return self._reject(acct, bundle.bundle_id, "foreign_sender")
        if any(t.tx_hash in self.records for t in bundle.searcher_txs):
            return self._reject(acct, bundle.bundle_id, "private_tx_in_body")
        if bundle.block > rec.max_block:
            return self._reject(acct, bundle.bundle_id, "beyond_tx_window")
        key = (bundle.searcher, block)
        if self._submissions[key] >= self.max_bundles_per_searcher_per_block:
            return self._reject(acct, bundle.bundle_id, "rate_limited")
        self._submissions[key] += 1
        self.bundles[bundle.bundle_id] = bundle
        self.bundle_status[bundle.bundle_id] = "pending"
        self._bundle_seq[bundle.bundle_id] = len(self._bundle_seq)
        acct.accepted += 1
        self.stats["bundles_accepted"] += 1
        return BundleReceipt(bundle.bundle_id, True)

    def _builder_bundle(self, b: MevShareBundle, rec: PrivateTxRecord) -> Bundle:
        refund = b.coinbase_transfer_wei * rec.prefs.refund_bps // 10_000
        txs: List[BuilderTx] = []
        reverting: Set[str] = set()
        for item in b.body:
            if item.hash is not None:
                txs.append(rec.tx)
            else:
                assert item.tx is not None
                txs.append(item.tx)
                if item.can_revert:
                    reverting.add(item.tx.tx_hash)
        return Bundle(f"mevshare:{b.bundle_id}", tuple(txs), b.coinbase_transfer_wei - refund, b.searcher,
                      frozenset(reverting), rec.tx.tx_hash,
                      {"private": True, "relay": self.name, "mev_share_bundle": b.bundle_id, "refund_wei": refund,
                       "refund_to": rec.tx.sender, "payer": b.searcher, "payment_wei": b.coinbase_transfer_wei})

    def _protect_bundle(self, rec: PrivateTxRecord) -> Bundle:
        return Bundle(f"protect:{rec.tx.tx_hash}", (rec.tx,), 0, rec.tx.sender, frozenset(), rec.tx.tx_hash,
                      {"private": True, "relay": self.name, "protect": True})

    def _allowed_builders(self, rec: PrivateTxRecord) -> Tuple[str, ...]:
        if rec.prefs.fast or not rec.prefs.builders:
            return tuple(sorted(self.builders))
        return tuple(b for b in rec.prefs.builders if b in self.builders)

    def build_auction(self, state: BuilderState, block: int, base_fee: int,
                      gas_limit: int = BLOCK_GAS_LIMIT) -> RelayAuction:
        """Order-flow auction: per private tx, the highest-paying backrun that simulates cleanly wins."""
        gas_limit = clamp_gas_limit(gas_limit)
        for rec in self.records.values():
            if rec.status is TxStatus.PENDING and rec.max_block < block:
                rec.status, rec.reason = TxStatus.EXPIRED, "window_passed"
        out: List[Bundle] = []
        entries: Dict[str, AuctionEntry] = {}
        allowed: Dict[str, Tuple[str, ...]] = {}
        for rec in self.pending(block):
            h = rec.tx.tx_hash
            cands = [b for bid, b in self.bundles.items()
                     if self.bundle_status.get(bid) == "pending" and b.ref_hashes == (h,)
                     and b.block <= block <= b.max_block]
            # High-priority searchers are simulated first; ties resolve by submission order.
            cands.sort(key=lambda b: (not self.searchers[b.searcher].high_priority, self._bundle_seq[b.bundle_id]))
            rows: List[Dict[str, Any]] = []
            valid: List[Tuple[int, int, MevShareBundle, Bundle]] = []
            for b in cands:
                bb = self._builder_bundle(b, rec)
                sim, _ = seal_block(state, [bb], base_fee, gas_limit, builder="relay-sim", trace=False,
                                    compute_root=False)
                if f"bundle:{bb.bundle_id}" in sim.included:
                    valid.append((b.coinbase_transfer_wei, -self._bundle_seq[b.bundle_id], b, bb))
                    rows.append({"bundle": b.bundle_id, "searcher": b.searcher,
                                 "payment_wei": b.coinbase_transfer_wei, "valid": True})
                else:
                    self.searchers[b.searcher].failed_simulations += 1
                    reason = sim.skipped.get(f"bundle:{bb.bundle_id}", "not_included")
                    rows.append({"bundle": b.bundle_id, "searcher": b.searcher,
                                 "payment_wei": b.coinbase_transfer_wei, "valid": False, "reason": reason})
                    self.stats["bundle_sim_failed"] += 1
            winner = max(valid, key=lambda v: (v[0], v[1])) if valid else None
            builders = self._allowed_builders(rec)
            if winner is not None:
                _, _, b, bb = winner
                out.append(bb)
                allowed[bb.bundle_id] = builders
                entries[h] = AuctionEntry(h, b.bundle_id, b.coinbase_transfer_wei, int(bb.meta["refund_wei"]), rows)
            else:
                entries[h] = AuctionEntry(h, None, 0, 0, rows)
            protect = self._protect_bundle(rec)
            out.append(protect)
            allowed[protect.bundle_id] = builders
        return RelayAuction(block, out, entries, allowed)

    def _nonce(self, bundle_id: str, builder: str, block: int) -> bytes:
        return hashlib.sha256(f"{self.name}|{bundle_id}|{builder}|{block}".encode()).digest()[:16]

    def deliver(self, auction: RelayAuction) -> Dict[str, List[Envelope]]:
        """Encrypt every auction bundle for each builder the user allowed."""
        out: Dict[str, List[Envelope]] = {name: [] for name in sorted(self.builders)}
        for b in auction.bundles:
            wire = bundle_to_wire(b)
            for name in auction.allowed_builders.get(b.bundle_id, ()):
                out[name].append(seal_envelope(self.builders[name], name, wire,
                                               self._nonce(b.bundle_id, name, auction.block)))
                self.stats["envelopes"] += 1
        return out

    def on_block(self, block_number: int, sealed: SealedBlock, post_state: BuilderState) -> Settlement:
        """Settle refunds and update private tx / bundle status after a block is sealed."""
        st = Settlement(block_number)
        tx_item = {t.tx_hash: t.item for t in sealed.txs}
        included_items = set(sealed.included)
        bal = post_state.fork.balances
        for rec in sorted(self.records.values(), key=lambda r: r.seq):
            if rec.status is not TxStatus.PENDING:
                continue
            h = rec.tx.tx_hash
            if h in tx_item:
                rec.status, rec.included_block = TxStatus.INCLUDED, block_number
                st.included.append(h)
                item = tx_item[h]
                if item.startswith("tx:"):
                    st.via_public.append(h)
                if item.startswith("bundle:mevshare:") and item in included_items:
                    bid = item[len("bundle:mevshare:"):]
                    b = self.bundles[bid]
                    refund = b.coinbase_transfer_wei * rec.prefs.refund_bps // 10_000
                    bal[b.searcher][ETH] -= refund
                    bal[rec.tx.sender][ETH] += refund
                    rec.refund_wei, rec.winning_bundle = refund, bid
                    st.refunds_wei += refund
                    st.refunds[h] = refund
                    self.bundle_status[bid] = "included"
                    self.searchers[b.searcher].wins += 1
                continue
            if post_state.nonce(rec.tx.sender) > rec.tx.nonce:
                rec.status, rec.reason = TxStatus.FAILED, "nonce_consumed"
                st.failed[h] = rec.reason
            elif block_number >= rec.max_block:
                rec.status, rec.reason = TxStatus.EXPIRED, "window_passed"
                st.expired.append(h)
        for bid, b in self.bundles.items():
            if self.bundle_status.get(bid) != "pending":
                continue
            target = self.records.get(b.ref_hashes[0])
            if target is None or target.status is not TxStatus.PENDING:
                self.bundle_status[bid] = "target_resolved"
            elif b.max_block <= block_number:
                self.bundle_status[bid] = "expired"
        self.stats["refunds_wei"] += st.refunds_wei
        return st

    def summary(self) -> Dict[str, Any]:
        statuses: Dict[str, int] = defaultdict(int)
        for r in self.records.values():
            statuses[r.status.value] += 1
        bstat: Dict[str, int] = defaultdict(int)
        for s in self.bundle_status.values():
            bstat[s] += 1
        return {"relay": self.name, "private_txs": len(self.records), "tx_status": dict(statuses),
                "bundle_status": dict(bstat), "stats": dict(self.stats),
                "searchers": {n: {"submitted": a.submitted, "accepted": a.accepted, "rejected": a.rejected,
                                  "failed_simulations": a.failed_simulations, "wins": a.wins,
                                  "high_priority": a.high_priority} for n, a in sorted(self.searchers.items())}}


class PrivateFlowReceiver:
    """Builder-side endpoint: decrypts envelopes with the builder's own key."""

    def __init__(self, name: str, key: bytes) -> None:
        self.name = name
        self.key = key
        self.failures: List[str] = []

    def open(self, envelopes: Iterable[Envelope]) -> List[Bundle]:
        out: List[Bundle] = []
        for env in envelopes:
            try:
                out.append(bundle_from_wire(open_envelope(self.key, env, self.name)))
            except (RelayError, ValueError, KeyError) as exc:
                self.failures.append(str(exc))
        return out


# ---------------------------------------------------------------------------
# Searchers
# ---------------------------------------------------------------------------


@dataclass
class BackrunSearcher:
    """Reads the hint stream, infers each trade and submits router-found backrun arbitrage."""

    name: str
    key: bytes
    native: str = "WETH"
    bid_bps: int = 9_000  # share of expected net profit offered as payment
    max_hypotheses: int = 4
    max_hops: int = 3
    window: int = 2  # bundles stay valid for block .. block + window
    tip_wei: int = GWEI
    min_net_native: int = 0
    _router: Optional[MultiHopRouter] = field(default=None, repr=False)

    def _get_router(self, pools: Mapping[str, Pool]) -> MultiHopRouter:
        if self._router is None or set(self._router.pools) != set(pools):
            self._router = MultiHopRouter([clone_pool(p) for p in pools.values()], self.native, self.max_hops)
        return self._router

    def on_event(self, view: HintView, state: BuilderState, block: int, base_fee: int) -> List[MevShareBundle]:
        pools = state.pools
        inf = infer_victim(view, pools)
        router = self._get_router(pools)
        out: List[MevShareBundle] = []
        seen: Set[Tuple[Tuple[str, ...], str, int]] = set()
        # One sub-account per target, so backruns on different private txs never collide on a nonce.
        sender = f"{self.name}#{view.tx_hash[:18]}"
        nonce = state.nonce(sender)
        for k, h in enumerate(select_hypotheses(inf.hypotheses, self.max_hypotheses)):
            work = {pid: clone_pool(p) for pid, p in pools.items()}
            try:
                path = RoutePath.from_route(work, h.token_in, h.route)
                execute_path(work, path, h.amount_in, mutate=True, min_amount_out=h.amount_out_min or None)
            except (RouterError, MASEError):
                continue
            opp = router.best_backrun(h.route, pools=work, gas_price_wei=base_fee + self.tip_wei)
            if opp is None or opp.net_profit_native <= self.min_net_native:
                continue
            sig = (opp.path.pool_ids, opp.path.token_in, opp.amount_in)
            if sig in seen:
                continue
            seen.add(sig)
            payment = opp.net_profit_native * self.bid_bps // 10_000
            gas = opp.gas_estimate * 3 // 2 + 40_000
            tx = BuilderTx(f"{self.name}:{view.tx_hash}:{k}", sender, nonce, gas, base_fee * 2 + self.tip_wei,
                           self.tip_wei, TxKind.SWAP,
                           {"route": list(opp.path.pool_ids), "token_in": opp.path.token_in,
                            "amount_in": opp.amount_in, "amount_out_min": opp.amount_in + 1})
            b = MevShareBundle(f"{self.name}:{view.tx_hash}:{k}", self.name, (BodyItem(hash=view.tx_hash), BodyItem(tx=tx)),
                               block, block + self.window, payment)
            out.append(b.signed(self.key))
        return out


@dataclass
class FrontrunSearcher:
    """Tries to sandwich private flow through the relay; the relay must reject every attempt."""

    name: str
    key: bytes
    native: str = "WETH"

    def on_event(self, view: HintView, state: BuilderState, block: int, base_fee: int) -> List[MevShareBundle]:
        inf = infer_victim(view, state.pools)
        if not inf.hypotheses:
            return []
        h = select_hypotheses(inf.hypotheses, 1)[0]
        cand = best_sandwich(state.pools, h.as_tx(view.tx_hash), 8, 4, native=self.native)
        if cand is None:
            return []
        n = state.nonce(self.name)
        gas = 400_000
        front = BuilderTx(f"{self.name}:{view.tx_hash}:front", self.name, n, gas, base_fee * 2 + GWEI, GWEI,
                          TxKind.SWAP, {"route": [cand.pool_id], "token_in": cand.token_in,
                                        "amount_in": cand.frontrun_in, "amount_out_min": cand.frontrun_out})
        back = BuilderTx(f"{self.name}:{view.tx_hash}:back", self.name, n + 1, gas, base_fee * 2 + GWEI, GWEI,
                         TxKind.SWAP, {"route": [cand.pool_id], "token_in": cand.token_out,
                                       "amount_in": cand.frontrun_out, "amount_out_min": cand.frontrun_in + 1})
        b = MevShareBundle(f"{self.name}:{view.tx_hash}:sandwich", self.name,
                           (BodyItem(tx=front), BodyItem(hash=view.tx_hash), BodyItem(tx=back)), block, block + 1, 1)
        return [b.signed(self.key)]


# ---------------------------------------------------------------------------
# Multi-slot private order-flow simulation (wired into brutal mode)
# ---------------------------------------------------------------------------


def private_flow_builders(config: SimulationConfig) -> List[BlockBuilder]:
    return [BlockBuilder("vanilla", Strategy.FEE_ONLY, margin_bps=0),
            BlockBuilder("greedy", Strategy.GREEDY, margin_bps=config.builder_margin_bps),
            BlockBuilder("public-greedy", Strategy.GREEDY, margin_bps=config.builder_margin_bps)]


@dataclass
class PrivateFlowReport:
    slots: List[Dict[str, Any]]
    relay: Dict[str, Any]
    exposure: ExposureMatrix
    invariant_failures: List[str]
    final_state_root: str
    elapsed_s: float

    def summary(self) -> Dict[str, Any]:
        return {
            "slots": self.slots,
            "relay": self.relay,
            "exposure": self.exposure.summary(),
            "private_txs": sum(s["private_txs"] for s in self.slots),
            "private_included": sum(len(s["settlement"]["included"]) for s in self.slots),
            "refunds_wei": sum(s["settlement"]["refunds_wei"] for s in self.slots),
            "backruns_won": sum(s["auction"]["backruns_won"] for s in self.slots),
            "privacy_violations_blocked": self.relay["stats"].get("bundle_rejected:privacy_violation", 0),
            "private_sandwiched": sum(s["private_sandwiched"] for s in self.slots),
            "public_sandwiches": sum(s["public_sandwiches"] for s in self.slots),
            "invariants_ok": not self.invariant_failures,
            "invariant_failures": list(self.invariant_failures),
            "final_state_root": self.final_state_root,
            "elapsed_s": round(self.elapsed_s, 4),
        }


class PrivateOrderflowSimulation:
    """Mixes public and private flow over several slots and checks the privacy guarantees.

    Per slot: a share of swaps (``config.private_flow_bps``) goes to the relay
    with hint profiles taken round-robin from ``profiles``; searchers bid on
    the hint stream; the relay auction's bundles are encrypted per builder and
    fed to the MEV-Boost auction next to public sandwich bundles. Checks: no
    private tx is ever the victim of an included sandwich, private txs only
    land through relay bundles, ETH is conserved through refund settlement,
    and every refund equals the user's share of the winning payment.
    """

    def __init__(self, pools: Iterable[Pool], config: SimulationConfig,
                 profiles: Optional[Mapping[str, PrivacyPreferences]] = None,
                 builders: Optional[Sequence[BlockBuilder]] = None) -> None:
        self.state = BuilderState.from_pools(pools)
        self.config = config
        self.profiles = dict(HINT_PROFILES if profiles is None else profiles)
        if not self.profiles:
            raise ConfigError("at least one hint profile is required")
        builders = list(builders or private_flow_builders(config))
        self.auction = MevBoostAuction(builders, proposer=Proposer())
        self.relay = PrivateRelay()
        self.receivers: Dict[str, PrivateFlowReceiver] = {}
        for b in builders:
            if b.strategy is not Strategy.FEE_ONLY and not b.name.startswith("public"):
                key = self.relay.register_builder(b.name, _derive_key(f"builder:{config.seed}:{b.name}"))
                self.receivers[b.name] = PrivateFlowReceiver(b.name, key)
        self.searchers: List[Any] = []
        for name, bid in (("searcher:alpha", 9_000), ("searcher:beta", 7_000)):
            acct = self.relay.register_searcher(name, _derive_key(f"searcher:{config.seed}:{name}"))
            self.searchers.append(BackrunSearcher(name, acct.key, config.native_token, bid,
                                                  max_hops=min(3, config.router_max_hops)))
        pred = self.relay.register_searcher("searcher:predator", _derive_key(f"predator:{config.seed}"))
        self.searchers.append(FrontrunSearcher(pred.name, pred.key, config.native_token))

    def run(self, executor: Optional[WorkerPool] = None, slots: Optional[int] = None,
            exposure: bool = True) -> PrivateFlowReport:
        cfg = self.config
        started = time.perf_counter()
        slots = cfg.blocks if slots is None else slots
        gas_limit = clamp_gas_limit(cfg.block_gas_limit)
        base_fee = cfg.base_fee_wei
        state = self.state
        txpool = TxPool(block_gas_limit=gas_limit)
        names = sorted(self.profiles)
        profile_i = 0
        cursor = 0
        rows: List[Dict[str, Any]] = []
        failures: List[str] = []
        exposures: List[ExposureResult] = []
        grid = cfg.frontrun_grid or 12
        refine = cfg.refine_iterations or 6
        own = executor is None
        wp = WorkerPool.for_config(cfg) if own else executor
        try:
            for slot in range(slots):
                block = state.block_number
                rng = random.Random(f"private-flow:{cfg.seed}:{slot}")
                private_now: List[Tuple[BuilderTx, PrivacyPreferences, str]] = []
                rejected_private: Dict[str, int] = defaultdict(int)
                for tx in generate_builder_mempool(state, cfg.txs_per_block, cfg.seed + 7_777, base_fee):
                    if tx.kind is TxKind.SWAP and rng.random() * 10_000 < cfg.private_flow_bps:
                        name = names[profile_i % len(names)]
                        profile_i += 1
                        prefs = replace(self.profiles[name], refund_bps=cfg.mev_share_refund_bps)
                        try:
                            self.relay.send_private_transaction(tx, state, block, prefs)
                            private_now.append((tx, prefs, name))
                        except RelayError as exc:
                            rejected_private[str(exc).split(":")[0]] += 1
                    else:
                        txpool.add(tx, base_fee, state.nonce(tx.sender))
                receipts: Dict[str, int] = defaultdict(int)
                for view in self.relay.stream(cursor):
                    for s in self.searchers:
                        for b in s.on_event(view, state, block, base_fee):
                            r = self.relay.send_bundle(b, block)
                            receipts["accepted" if r.accepted else r.reason.split(":")[0]] += 1
                cursor = len(self.relay.events)
                auction = self.relay.build_auction(state, block, base_fee, gas_limit)
                envelopes = self.relay.deliver(auction)
                private_bundles = {name: rcv.open(envelopes.get(name, ())) for name, rcv in self.receivers.items()}
                public = [t for t in txpool.pending(state.nonces) if not self.relay.is_private(t.tx_hash)]
                public_bundles = build_sandwich_bundles(state, public, base_fee, cfg.native_token, grid, refine,
                                                        cfg.bribe_bps, cfg.max_frontrun_bps)
                result = self.auction.run_slot(state, public_bundles, public, base_fee, slot, gas_limit, wp,
                                               seed=cfg.seed, native=cfg.native_token,
                                               extra_bundles=private_bundles)
                eth_before = _eth_total(state)
                settlement = self.relay.on_block(block, result.block, result.post_state)
                failures.extend(self._check(slot, result, settlement, eth_before))
                private_sandwiched, public_sandwiches = self._sandwich_counts(result.block)
                if private_sandwiched:
                    failures.append(f"slot {slot}: {private_sandwiched} private tx(s) sandwiched")
                if exposure and private_now:
                    exposures.extend(exposure_for_items(state, private_now, wp, cfg.native_token, grid,
                                                        refine, chunks=cfg.shards(RELAY_CHUNKS)).results)
                rows.append({
                    "slot": slot,
                    "block": block,
                    "private_txs": len(private_now),
                    "private_rejected": dict(rejected_private),
                    "bundle_receipts": dict(receipts),
                    "auction": auction.summary() | {"auctions": len(auction.entries)},
                    "settlement": settlement.to_dict(),
                    "winner": result.winner,
                    "private_sandwiched": private_sandwiched,
                    "public_sandwiches": public_sandwiches,
                    "block_value_wei": result.block.value_wei,
                    "decrypt_failures": sum(len(r.failures) for r in self.receivers.values()),
                })
                state = result.post_state
                txpool.prune(state.nonces)
                base_fee = next_base_fee(base_fee, result.block.gas_used, gas_limit)
                state.fork.mine()
        finally:
            if own:
                wp.close()
        self.state = state
        return PrivateFlowReport(rows, self.relay.summary(), ExposureMatrix(exposures), failures,
                                 state.state_root(), time.perf_counter() - started)

    def _sandwich_counts(self, block: SealedBlock) -> Tuple[int, int]:
        private_hit = public_hit = 0
        for item in block.layout:
            if not isinstance(item, Bundle) or f"bundle:{item.bundle_id}" not in block.included:
                continue
            labels = block.flags.get(f"bundle:{item.bundle_id}", [])
            if not any(lbl in ("sandwich", "multi_hop_sandwich") for lbl in labels):
                continue
            victims = [t.tx_hash for t in item.txs if t.sender != item.searcher]
            if any(self.relay.is_private(v) for v in victims):
                private_hit += 1
            else:
                public_hit += 1
        return private_hit, public_hit

    def _check(self, slot: int, result: Any, st: Settlement, eth_before: int) -> List[str]:
        out: List[str] = []
        tag = f"slot {slot}"
        if st.via_public:
            out.append(f"{tag}: private tx included as public item: {st.via_public}")
        if _eth_total(result.post_state) != eth_before:
            out.append(f"{tag}: ETH not conserved after refund settlement")
        for h, refund in st.refunds.items():
            rec = self.relay.records[h]
            b = self.relay.bundles[rec.winning_bundle]
            if refund != b.coinbase_transfer_wei * rec.prefs.refund_bps // 10_000:
                out.append(f"{tag}: refund mismatch for {h}")
        if result.block.gas_used > clamp_gas_limit(self.config.block_gas_limit):
            out.append(f"{tag}: gas over limit")
        return out


def _eth_total(state: BuilderState) -> int:
    return sum(toks.get(ETH, 0) for toks in state.fork.balances.values())


__all__ = [
    "DEFAULT_INCLUSION_WINDOW",
    "DEFAULT_REFUND_BPS",
    "FULL_DISCLOSURE",
    "HINT_NAMES",
    "HINT_PROFILES",
    "MEV_SHARE_DEFAULT",
    "RELAY_CHUNKS",
    "ROUTER_ADDRESS",
    "SELECTORS",
    "SELECTOR_RANGE",
    "BackrunSearcher",
    "BodyItem",
    "BundleReceipt",
    "Envelope",
    "ExposureMatrix",
    "ExposureResult",
    "FrontrunSearcher",
    "Hint",
    "HintView",
    "Inference",
    "Knowledge",
    "Log",
    "MevShareBundle",
    "PoolDirectory",
    "PrivacyPreferences",
    "PrivateFlowReceiver",
    "PrivateFlowReport",
    "PrivateOrderflowSimulation",
    "PrivateRelay",
    "PrivateTxRecord",
    "RelayAuction",
    "RelayError",
    "Settlement",
    "SimOutcome",
    "TxStatus",
    "VictimHypothesis",
    "Word",
    "address_of",
    "build_hint_view",
    "bundle_from_wire",
    "bundle_to_wire",
    "encode_calldata",
    "exposure_for_items",
    "hint_exposure_matrix",
    "hint_names",
    "infer_victim",
    "measure_exposure",
    "open_envelope",
    "parse_hints",
    "pool_word_range",
    "private_flow_builders",
    "realize_sandwich",
    "reveal_ranges",
    "revealed_selector",
    "revealed_words",
    "seal_envelope",
    "select_hypotheses",
    "selector_of",
    "sign_payload",
    "simulate_private_tx",
    "topic_of",
    "tx_from_dict",
    "tx_to_address",
    "verify_signature",
    "word_range",
]
