"""Layer 16: sandwich detector for real Ethereum mainnet blocks.

Everything else in this package works on seeded, in-memory markets. This
module is the one place that reads real chain data. It takes a mainnet block,
finds the Uniswap v2 / v3 swaps in it and reports every sandwich: who was
attacked, who attacked, how much output each victim lost and how much the
attacker made.

Input, in either of two shapes:

* **Receipts and logs** (``eth_getBlockByNumber`` + ``eth_getBlockReceipts``).
  v2 ``Swap`` events come with the pair's ``Sync`` (reserves right after the
  swap); v3 ``Swap`` events carry ``sqrtPriceX96``, ``liquidity`` and ``tick``
  after the swap. :func:`fetch_block_export` pulls these from any JSON-RPC
  node, and the result can be saved and analysed offline.
* **Call traces** (Parity ``trace_block`` format, which is also what
  ``flashbots/mev-inspect-py`` stores for its test blocks). v2 swaps are
  ``swap(...)`` calls on the pair, the input amount is the token transfer into
  the pair before the call, and reserves come from ``getReserves()`` outputs in
  the same block. Coinbase payments (bribes) are visible as value transfers to
  the miner.

How a sandwich is recognised (per pool, in block order):

1. a *front-run* swap A -> B;
2. one or more *victim* swaps A -> B in later transactions sent by someone
   else;
3. a *back-run* swap B -> A in a later transaction that shares the front-run's
   sender (EOA) or its target contract (a bot contract, never a public router),
   selling between 50% and 110% of what the front-run bought.

What is computed, and how exact it is:

* **Attacker profit** in token A is ``back_out - front_in`` straight from the
  chain data, so it is exact. Gas (``gasUsed * effectiveGasPrice`` of both
  transactions) and coinbase transfers (trace data only) are subtracted to give
  the net profit in wei when A is WETH; when A is another token and B is WETH
  the token amount is valued at the pool's price after the back-run.
* **Victim loss** is a counterfactual: the victim's actual output against the
  output the same input would have produced if the front-run had not been
  there (every other swap in between is replayed as an exact-input swap).
  For v2 the pre-front-run reserves are known from ``Sync`` / ``getReserves``
  and the counterfactual uses the bit-exact ``getAmountOut`` from
  :mod:`simulator.slippage_engine`. For v3 the pre-front-run price is
  reconstructed by inverting the front-run on the verified v3 swap loop. A
  victim's loss is marked ``exact`` only when the same model, started from the
  same state, reproduces the victim's real on-chain output to the wei.
* Limits: a v3 swap that crosses an initialized tick cannot be reconstructed
  from logs alone (the loss is then reported as approximate); v3 pools in trace
  data carry no price, so only detection and profit are reported there;
  coinbase transfers made through internal calls are invisible in logs-only
  data; fee-on-transfer tokens break the trace-based input amounts (those
  victims fail the reproduction check and are not marked exact).

The JSON-RPC client uses only ``urllib``; there is no third-party dependency.
"""

from __future__ import annotations

import gzip
import json
import os
import urllib.request
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .slippage_engine import (
    MAX_SQRT_RATIO,
    MIN_SQRT_RATIO,
    Q96,
    V2_FEE_BPS,
    ConcentratedPool,
    MASEError,
    sqrt_price_x96_to_price,
    v2_get_amount_in,
    v2_get_amount_out,
)

__all__ = [
    "BlockData",
    "DetectionReport",
    "JsonRpcClient",
    "KNOWN_ROUTERS",
    "SandwichFinding",
    "Swap",
    "VictimImpact",
    "WETH",
    "detect_block",
    "detect_export",
    "fetch_block_export",
    "keccak256",
    "load_export",
    "parse_logs_export",
    "parse_traces_export",
    "read_export_file",
    "write_export_file",
]

# ---------------------------------------------------------------------------
# Keccak-256 (the Ethereum hash; hashlib.sha3_256 uses different padding)
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1
_KECCAK_RC = (
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
)
_KECCAK_ROT = (
    (0, 36, 3, 41, 18),
    (1, 44, 10, 45, 2),
    (62, 6, 43, 15, 61),
    (28, 55, 25, 21, 56),
    (27, 20, 39, 8, 14),
)


def _rotl64(v: int, n: int) -> int:
    return ((v << n) | (v >> (64 - n))) & _MASK64 if n else v


def _keccak_f(a: List[List[int]]) -> List[List[int]]:
    for rc in _KECCAK_RC:
        c = [a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4] for x in range(5)]
        d = [c[(x - 1) % 5] ^ _rotl64(c[(x + 1) % 5], 1) for x in range(5)]
        a = [[a[x][y] ^ d[x] for y in range(5)] for x in range(5)]
        b = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                b[y][(2 * x + 3 * y) % 5] = _rotl64(a[x][y], _KECCAK_ROT[x][y])
        a = [[(b[x][y] ^ ((~b[(x + 1) % 5][y]) & b[(x + 2) % 5][y])) & _MASK64 for y in range(5)]
             for x in range(5)]
        a[0][0] ^= rc
    return a


def keccak256(data: bytes) -> bytes:
    """Keccak-256 as used by Ethereum (original Keccak padding 0x01 .. 0x80)."""
    rate = 136
    msg = bytearray(data)
    msg.append(0x01)
    while len(msg) % rate:
        msg.append(0)
    msg[-1] |= 0x80
    state = [[0] * 5 for _ in range(5)]
    for off in range(0, len(msg), rate):
        block = msg[off:off + rate]
        for i in range(rate // 8):
            state[i % 5][i // 5] ^= int.from_bytes(block[8 * i:8 * i + 8], "little")
        state = _keccak_f(state)
    return b"".join(state[i % 5][i // 5].to_bytes(8, "little") for i in range(4))


def event_topic(signature: str) -> str:
    return "0x" + keccak256(signature.encode()).hex()


def selector(signature: str) -> str:
    return "0x" + keccak256(signature.encode()).hex()[:8]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"

TOPIC_V2_SWAP = event_topic("Swap(address,uint256,uint256,uint256,uint256,address)")
TOPIC_V2_SYNC = event_topic("Sync(uint112,uint112)")
TOPIC_V3_SWAP = event_topic("Swap(address,address,int256,int256,uint160,uint128,int24)")
TOPIC_V3_MINT = event_topic("Mint(address,address,int24,int24,uint128,uint256,uint256)")
TOPIC_V3_BURN = event_topic("Burn(address,int24,int24,uint128,uint256,uint256)")

SEL_TRANSFER = selector("transfer(address,uint256)")
SEL_TRANSFER_FROM = selector("transferFrom(address,address,uint256)")
SEL_V2_SWAP = selector("swap(uint256,uint256,address,bytes)")
SEL_V3_SWAP = selector("swap(address,bool,int256,uint160,bytes)")
SEL_GET_RESERVES = selector("getReserves()")
SEL_WETH_DEPOSIT = selector("deposit()")
SEL_WETH_WITHDRAW = selector("withdraw(uint256)")
SEL_TOKEN0 = selector("token0()")
SEL_TOKEN1 = selector("token1()")
SEL_FEE = selector("fee()")
# Calls that change a v2 pair's reserves outside a swap: the reserve chain
# is broken there and has to be re-anchored by a later getReserves().
SEL_V2_RESETS = frozenset(selector(s) for s in ("mint(address)", "burn(address)", "sync()", "skim(address)"))

# Public routers and aggregators. A shared ``to`` address only links a
# front-run to a back-run when it is not one of these.
KNOWN_ROUTERS = frozenset({
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap V2 Router02
    "0xf164fc0ec4e93095b804a4795bbe1e041497b92a",  # Uniswap V2 Router01
    "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap V3 SwapRouter
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap V3 SwapRouter02
    "0xef1c6e67703c7bd7107eed8303fbe6ec2554bf6b",  # Uniswap Universal Router (old)
    "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad",  # Uniswap Universal Router
    "0x66a9893cc07d91d95644aedd05d03f95e1dba8af",  # Uniswap Universal Router v4
    "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f",  # SushiSwap Router
    "0x03f7724180aa6b939894b5ca4314783b0b36b329",  # ShibaSwap Router
    "0x1111111254fb6c44bac0bed2854e76f90643097d",  # 1inch v4
    "0x1111111254eeb25477b68fb85ed929f73a960582",  # 1inch v5
    "0x111111125421ca6dc452d289314280a0f8842a65",  # 1inch v6
    "0xdef1c0ded9bec7f1a1670819833240f027b25eff",  # 0x Exchange Proxy
    "0xdef171fe48cf0115b1d80b88dc8eab59176fee57",  # ParaSwap v5
    "0x6a000f20005980200259b80c5102003040001068",  # ParaSwap v6
    "0x881d40237659c251811cec9c364ef91dc08d300c",  # MetaMask Swap Router
    "0x9008d19f58aabd9ed0d60971565aa8510560ab41",  # CoW Protocol settlement
})

# Uniswap v2 forks charge different fees; the detector calibrates each pair
# against its own swaps and falls back to 0.3%.
V2_FEE_CANDIDATES_BPS = (30, 25, 20, 17, 10, 5, 100)
# v3 fee tier (pips) -> tick spacing, needed to step the swap loop exactly.
V3_TICK_SPACING = {100: 1, 500: 10, 3000: 60, 10000: 200}

Position = Tuple[int, int]  # (transaction index, order inside the transaction)


class DetectorError(ValueError):
    """Raised for malformed block data or RPC failures."""


# ---------------------------------------------------------------------------
# Small decoding helpers
# ---------------------------------------------------------------------------


def _int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.startswith(("0x", "0X")):
        return int(text, 16) if len(text) > 2 else 0
    return int(text)


def _addr(value: Any) -> str:
    if not value:
        return ""
    text = str(value).lower()
    if not text.startswith("0x"):
        text = "0x" + text
    return "0x" + text[2:].rjust(40, "0")[-40:]


def _words(hexdata: str) -> List[int]:
    body = (hexdata or "0x")[2:] if str(hexdata).startswith("0x") else str(hexdata or "")
    return [int(body[i:i + 64], 16) for i in range(0, len(body) - len(body) % 64, 64)]


def _signed(word: int, bits: int = 256) -> int:
    return word - (1 << bits) if word >> (bits - 1) else word


def _word_addr(word: int) -> str:
    return "0x" + format(word & ((1 << 160) - 1), "040x")


def _get(d: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


# ---------------------------------------------------------------------------
# Normalised block model
# ---------------------------------------------------------------------------


@dataclass
class TxInfo:
    index: int
    hash: str
    sender: str
    to: str
    gas_used: int = 0
    gas_price: int = 0
    coinbase_transfer: int = 0
    value: int = 0

    @property
    def fee_wei(self) -> int:
        return self.gas_used * self.gas_price


@dataclass
class Swap:
    """One swap on one pool, normalised from logs or traces."""

    pool: str
    version: str  # "v2" or "v3"
    tx_index: int
    seq: int
    zero_for_one: bool
    amount_in: int
    amount_out: int
    token_in: Optional[str] = None
    token_out: Optional[str] = None
    recipient: str = ""
    reserves_before: Optional[Tuple[int, int]] = None
    reserves_after: Optional[Tuple[int, int]] = None
    sqrt_after: Optional[int] = None
    liquidity_after: Optional[int] = None
    tick_after: Optional[int] = None

    @property
    def pos(self) -> Position:
        return (self.tx_index, self.seq)

    def v2_delta(self, reserves: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        r0, r1 = reserves
        if self.zero_for_one:
            r0, r1 = r0 + self.amount_in, r1 - self.amount_out
        else:
            r0, r1 = r0 - self.amount_out, r1 + self.amount_in
        return (r0, r1) if r0 > 0 and r1 > 0 else None

    def v2_undo(self, reserves: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        r0, r1 = reserves
        if self.zero_for_one:
            r0, r1 = r0 - self.amount_in, r1 + self.amount_out
        else:
            r0, r1 = r0 + self.amount_out, r1 - self.amount_in
        return (r0, r1) if r0 > 0 and r1 > 0 else None


@dataclass
class Anchor:
    """Known v2 reserves at a point in the block (None = unknown from here on)."""

    pool: str
    tx_index: int
    seq: int
    reserves: Optional[Tuple[int, int]]

    @property
    def pos(self) -> Position:
        return (self.tx_index, self.seq)


@dataclass
class TokenMove:
    """Token flow seen in a trace (transfer, transferFrom, WETH deposit / withdraw)."""

    tx_index: int
    token: str
    src: str
    dst: str
    amount: int


@dataclass
class BlockData:
    number: int
    miner: str
    source: str  # "logs" or "traces"
    txs: Dict[int, TxInfo] = field(default_factory=dict)
    swaps: List[Swap] = field(default_factory=list)
    anchors: List[Anchor] = field(default_factory=list)
    pools: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    moves: List[TokenMove] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def pool_swaps(self) -> Dict[str, List[Swap]]:
        out: Dict[str, List[Swap]] = {}
        for s in sorted(self.swaps, key=lambda s: s.pos):
            out.setdefault(s.pool, []).append(s)
        return out


# ---------------------------------------------------------------------------
# Parser: receipts with logs
# ---------------------------------------------------------------------------


def _tx_from_rpc(tx: Mapping[str, Any], receipt: Optional[Mapping[str, Any]]) -> TxInfo:
    r = receipt or {}
    gas_price = _get(r, "effectiveGasPrice", "effective_gas_price")
    if gas_price is None:
        gas_price = _get(tx, "gasPrice", "gas_price", default=0)
    return TxInfo(
        index=_int(_get(tx, "transactionIndex", "transaction_index")),
        hash=str(_get(tx, "hash", "transaction_hash", default="")).lower(),
        sender=_addr(_get(tx, "from")),
        to=_addr(_get(tx, "to")),
        gas_used=_int(_get(r, "gasUsed", "gas_used", default=0)),
        gas_price=_int(gas_price),
        value=_int(_get(tx, "value", default=0)),
    )


def parse_logs_export(block: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]],
                      pools: Optional[Mapping[str, Mapping[str, Any]]] = None) -> BlockData:
    """Normalise ``eth_getBlockByNumber(full=True)`` + its receipts."""
    data = BlockData(number=_int(_get(block, "number", "block_number")), miner=_addr(_get(block, "miner")),
                     source="logs", pools={_addr(k): dict(v) for k, v in (pools or {}).items()})
    consumed: set = set()
    by_hash = {str(_get(r, "transactionHash", "transaction_hash", default="")).lower(): r for r in receipts}
    for tx in _get(block, "transactions", default=[]):
        if not isinstance(tx, Mapping):
            raise DetectorError("block must be fetched with full transaction objects")
        info = _tx_from_rpc(tx, by_hash.get(str(tx.get("hash", "")).lower()))
        if info.to and info.to == data.miner:
            info.coinbase_transfer += info.value
        data.txs[info.index] = info

    for r in receipts:
        if _int(_get(r, "status", default=1)) != 1:
            continue
        tx_index = _int(_get(r, "transactionIndex", "transaction_index"))
        if tx_index not in data.txs:
            data.txs[tx_index] = TxInfo(index=tx_index, hash=str(_get(r, "transactionHash", default="")).lower(),
                                        sender=_addr(_get(r, "from")), to=_addr(_get(r, "to")),
                                        gas_used=_int(_get(r, "gasUsed", default=0)),
                                        gas_price=_int(_get(r, "effectiveGasPrice", default=0)))
        pending_sync: Dict[str, Anchor] = {}
        logs = sorted(_get(r, "logs", default=[]), key=lambda lg: _int(_get(lg, "logIndex", "log_index")))
        for lg in logs:
            if lg.get("removed"):
                continue
            topics = [str(t).lower() for t in _get(lg, "topics", default=[])]
            if not topics:
                continue
            pool = _addr(lg.get("address"))
            log_index = _int(_get(lg, "logIndex", "log_index"))
            words = _words(str(lg.get("data", "0x")))
            t0 = topics[0]
            if t0 == TOPIC_V2_SYNC and len(words) >= 2:
                # A pair emits Sync right before its Swap. A Sync that no Swap
                # claims belongs to a mint / burn / sync and stays an anchor.
                anchor = Anchor(pool, tx_index, log_index, (words[0], words[1]))
                pending_sync[pool] = anchor
                data.anchors.append(anchor)
            elif t0 == TOPIC_V2_SWAP and len(words) >= 4:
                a0_in, a1_in, a0_out, a1_out = words[:4]
                swap = _v2_swap_from_amounts(pool, tx_index, log_index, a0_in, a1_in, a0_out, a1_out,
                                             _word_addr(int(topics[2], 16)) if len(topics) > 2 else "")
                if swap is None:
                    data.warnings.append(f"tx {tx_index}: v2 swap on {pool} with two-sided amounts skipped")
                    continue
                sync = pending_sync.pop(pool, None)
                if sync is not None:
                    swap.reserves_after = sync.reserves
                    consumed.add(id(sync))
                _fill_tokens(swap, data.pools.get(pool))
                data.swaps.append(swap)
            elif t0 == TOPIC_V3_SWAP and len(words) >= 5:
                amount0, amount1 = _signed(words[0]), _signed(words[1])
                if (amount0 > 0) == (amount1 > 0) or amount0 == 0 or amount1 == 0:
                    continue
                zfo = amount0 > 0
                swap = Swap(pool=pool, version="v3", tx_index=tx_index, seq=log_index, zero_for_one=zfo,
                            amount_in=amount0 if zfo else amount1, amount_out=-(amount1 if zfo else amount0),
                            recipient=_word_addr(int(topics[2], 16)) if len(topics) > 2 else "",
                            sqrt_after=words[2], liquidity_after=words[3], tick_after=_signed(words[4], 24))
                _fill_tokens(swap, data.pools.get(pool))
                data.swaps.append(swap)
            elif t0 in (TOPIC_V3_MINT, TOPIC_V3_BURN):
                data.anchors.append(Anchor(pool, tx_index, log_index, None))
    data.anchors = [a for a in data.anchors if id(a) not in consumed]
    _resolve_v2_states(data)
    return data


def _v2_swap_from_amounts(pool: str, tx_index: int, seq: int, a0_in: int, a1_in: int, a0_out: int, a1_out: int,
                          recipient: str) -> Optional[Swap]:
    if a0_in > 0 and a1_out > 0 and a1_in == 0 and a0_out == 0:
        return Swap(pool, "v2", tx_index, seq, True, a0_in, a1_out, recipient=recipient)
    if a1_in > 0 and a0_out > 0 and a0_in == 0 and a1_out == 0:
        return Swap(pool, "v2", tx_index, seq, False, a1_in, a0_out, recipient=recipient)
    return None


def _fill_tokens(swap: Swap, meta: Optional[Mapping[str, Any]]) -> None:
    if not meta:
        return
    t0, t1 = _addr(meta.get("token0")), _addr(meta.get("token1"))
    if t0 and t1:
        swap.token_in, swap.token_out = (t0, t1) if swap.zero_for_one else (t1, t0)


# ---------------------------------------------------------------------------
# Parser: call traces (Parity trace_block / mev-inspect-py block files)
# ---------------------------------------------------------------------------


def parse_traces_export(export: Mapping[str, Any]) -> BlockData:
    """Normalise a trace export: ``{block_number, miner, traces, receipts}``."""
    data = BlockData(number=_int(_get(export, "block_number", "number")), miner=_addr(export.get("miner")),
                     source="traces", pools={_addr(k): dict(v) for k, v in (export.get("pools") or {}).items()})
    receipts = {_int(_get(r, "transaction_index", "transactionIndex")): r for r in export.get("receipts", [])}

    by_tx: Dict[int, List[Mapping[str, Any]]] = {}
    for tr in export.get("traces", []):
        pos = _get(tr, "transaction_position", "transactionPosition")
        if pos is None:
            continue
        by_tx.setdefault(_int(pos), []).append(tr)

    for tx_index in sorted(by_tx):
        traces = by_tx[tx_index]
        root = next((t for t in traces if not _get(t, "trace_address", "traceAddress", default=[])), traces[0])
        ra = root.get("action", {})
        rcpt = receipts.get(tx_index, {})
        info = TxInfo(index=tx_index,
                      hash=str(_get(root, "transaction_hash", "transactionHash", default="")).lower(),
                      sender=_addr(ra.get("from")), to=_addr(ra.get("to") or _get(rcpt, "to")),
                      gas_used=_int(_get(rcpt, "gas_used", "gasUsed", default=0)),
                      gas_price=_int(_get(rcpt, "effective_gas_price", "effectiveGasPrice", default=0)),
                      value=_int(ra.get("value")))
        data.txs[tx_index] = info
        if _get(root, "error") is not None:
            continue  # reverted transaction: nothing it did persisted
        failed: List[Tuple[int, ...]] = []
        live: List[Tuple[Tuple[int, ...], Mapping[str, Any]]] = []
        for tr in traces:
            addr = tuple(_get(tr, "trace_address", "traceAddress", default=[]))
            if _get(tr, "error") is not None:
                failed.append(addr)
                continue
            if any(addr[:len(f)] == f for f in failed):
                continue
            live.append((addr, tr))
        _scan_tx_traces(data, info, live)
    _resolve_v2_states(data)
    return data


def _scan_tx_traces(data: BlockData, info: TxInfo, live: List[Tuple[Tuple[int, ...], Mapping[str, Any]]]) -> None:
    moves: List[Tuple[int, TokenMove]] = []
    for seq, (addr, tr) in enumerate(live):
        if tr.get("type") != "call":
            continue
        a = tr.get("action", {})
        if str(a.get("callType", "call")).lower() not in ("call", ""):
            continue  # delegatecall / staticcall move no tokens of their own (getReserves is read below)
        src, dst = _addr(a.get("from")), _addr(a.get("to"))
        value = _int(a.get("value"))
        if value and dst == data.miner:
            info.coinbase_transfer += value
        inp = str(a.get("input", "0x")).lower()
        sel = inp[:10]
        args = _words("0x" + inp[10:])
        if sel == SEL_TRANSFER and len(args) >= 2:
            moves.append((seq, TokenMove(info.index, dst, src, _word_addr(args[0]), args[1])))
        elif sel == SEL_TRANSFER_FROM and len(args) >= 3:
            moves.append((seq, TokenMove(info.index, dst, _word_addr(args[0]), _word_addr(args[1]), args[2])))
        elif sel == SEL_WETH_DEPOSIT and dst == WETH and value:
            moves.append((seq, TokenMove(info.index, WETH, "", src, value)))
        elif sel == SEL_WETH_WITHDRAW and dst == WETH and args:
            moves.append((seq, TokenMove(info.index, WETH, src, "", args[0])))
        elif sel in SEL_V2_RESETS:
            data.anchors.append(Anchor(dst, info.index, seq, None))
        elif sel == SEL_V2_SWAP and len(args) >= 3:
            swap = _trace_v2_swap(data, info, seq, addr, live, moves, args)
            if swap is not None:
                data.swaps.append(swap)
        elif sel == SEL_V3_SWAP and len(args) >= 3:
            out = _words(str((tr.get("result") or {}).get("output", "0x")))
            if len(out) >= 2:
                amount0, amount1 = _signed(out[0]), _signed(out[1])
                if amount0 and amount1 and (amount0 > 0) != (amount1 > 0):
                    zfo = amount0 > 0
                    swap = Swap(dst, "v3", info.index, seq, zfo, amount0 if zfo else amount1,
                                -(amount1 if zfo else amount0), recipient=_word_addr(args[0]))
                    swap.token_out = _child_transfer_token(live, addr, dst)
                    _fill_tokens(swap, data.pools.get(dst))
                    data.swaps.append(swap)
        # getReserves output anchors the pair's reserves at this point (any call type).
    for seq, (addr, tr) in enumerate(live):
        a = tr.get("action", {})
        if str(a.get("input", "")).lower()[:10] == SEL_GET_RESERVES:
            out = _words(str((tr.get("result") or {}).get("output", "0x")))
            if len(out) >= 2 and out[0] and out[1]:
                data.anchors.append(Anchor(_addr(a.get("to")), info.index, seq, (out[0], out[1])))
    data.moves.extend(m for _, m in moves)


def _child_transfer_token(live: Sequence[Tuple[Tuple[int, ...], Mapping[str, Any]]], parent: Tuple[int, ...],
                          pool: str) -> Optional[str]:
    """Token the pool sent out during the swap call (first transfer it made)."""
    for addr, tr in live:
        if len(addr) > len(parent) and addr[:len(parent)] == parent:
            a = tr.get("action", {})
            if _addr(a.get("from")) == pool and str(a.get("input", "")).lower()[:10] == SEL_TRANSFER:
                return _addr(a.get("to"))
    return None


def _trace_v2_swap(data: BlockData, info: TxInfo, seq: int, addr: Tuple[int, ...],
                   live: Sequence[Tuple[Tuple[int, ...], Mapping[str, Any]]],
                   moves: Sequence[Tuple[int, TokenMove]], args: Sequence[int]) -> Optional[Swap]:
    pair = _addr(live[seq][1].get("action", {}).get("to"))
    a0_out, a1_out, recipient = args[0], args[1], _word_addr(args[2])
    if (a0_out == 0) == (a1_out == 0):
        return None
    token_out = _child_transfer_token(live, addr, pair)
    token_in = None
    amount_in = 0
    for mseq, m in reversed(moves):
        if mseq < seq and m.dst == pair and m.token != token_out:
            token_in, amount_in = m.token, m.amount
            break
    if token_in is None or amount_in <= 0:
        data.warnings.append(f"tx {info.index}: v2 swap on {pair} without a visible input transfer skipped")
        return None
    zfo = a1_out > 0
    swap = Swap(pair, "v2", info.index, seq, zfo, amount_in, a1_out if zfo else a0_out,
                token_in=token_in, token_out=token_out, recipient=recipient)
    if token_out and token_in and (token_in < token_out) != zfo:
        data.warnings.append(f"tx {info.index}: token order on {pair} disagrees with the swap direction")
    data.pools.setdefault(pair, {}).update(
        {"token0": token_in if zfo else token_out, "token1": token_out if zfo else token_in})
    return swap


# ---------------------------------------------------------------------------
# v2 reserve timeline
# ---------------------------------------------------------------------------


def _resolve_v2_states(data: BlockData) -> None:
    """Give every v2 swap its reserves before and after, where they are knowable.

    Anchors (``Sync`` after a mint/burn, ``getReserves()`` outputs) and swaps
    with a known post-state fix the reserves at a point; swaps in between move
    them by their exact deltas. A forward pass fills what follows an anchor, a
    backward pass what precedes the first one. A reset anchor (mint / burn /
    sync / skim seen in a trace) makes the state unknown until the next anchor.
    """
    per_pool: Dict[str, List[Union[Swap, Anchor]]] = {}
    for s in data.swaps:
        if s.version == "v2":
            per_pool.setdefault(s.pool, []).append(s)
    for a in data.anchors:
        if a.pool in per_pool:
            per_pool[a.pool].append(a)
    for pool, items in per_pool.items():
        items.sort(key=lambda it: (it.pos, 0 if isinstance(it, Anchor) else 1))
        state: Optional[Tuple[int, int]] = None
        for it in items:
            if isinstance(it, Anchor):
                if it.reserves is not None and state is not None and state != it.reserves:
                    data.warnings.append(f"{pool}: reserves jump at tx {it.tx_index} (transfer fee or donation)")
                state = it.reserves
                continue
            if it.reserves_after is not None:
                before = it.v2_undo(it.reserves_after)
                it.reserves_before = before
                state = it.reserves_after
            elif state is not None:
                it.reserves_before = state
                it.reserves_after = it.v2_delta(state)
                state = it.reserves_after
        later: Optional[Tuple[int, int]] = None
        for it in reversed(items):
            if isinstance(it, Anchor):
                later = it.reserves
                continue
            if it.reserves_after is None and later is not None:
                it.reserves_after = later
                it.reserves_before = it.v2_undo(later)
            later = it.reserves_before


def calibrate_v2_fee(swaps: Sequence[Swap], hint_bps: Optional[int] = None) -> Tuple[int, int]:
    """(fee_bps, swaps it reproduces exactly) for one pair."""
    candidates = ([hint_bps] if hint_bps is not None else []) + [f for f in V2_FEE_CANDIDATES_BPS if f != hint_bps]
    best = (V2_FEE_BPS, 0)
    for fee in candidates:
        hits = 0
        for s in swaps:
            if s.reserves_before is None:
                continue
            r_in, r_out = _v2_in_out(s.reserves_before, s.zero_for_one)
            try:
                if v2_get_amount_out(s.amount_in, r_in, r_out, fee) == s.amount_out:
                    hits += 1
            except MASEError:
                continue
        if hits > best[1]:
            best = (fee, hits)
    return best


def _v2_in_out(reserves: Tuple[int, int], zero_for_one: bool) -> Tuple[int, int]:
    return (reserves[0], reserves[1]) if zero_for_one else (reserves[1], reserves[0])


def _v2_step(reserves: Tuple[int, int], zero_for_one: bool, amount_in: int, fee_bps: int) -> Tuple[Tuple[int, int], int]:
    r_in, r_out = _v2_in_out(reserves, zero_for_one)
    out = v2_get_amount_out(amount_in, r_in, r_out, fee_bps)
    new = (reserves[0] + amount_in, reserves[1] - out) if zero_for_one else (reserves[0] - out, reserves[1] + amount_in)
    return new, out


# ---------------------------------------------------------------------------
# v3 state reconstruction (logs only)
# ---------------------------------------------------------------------------


def _v3_pool(sqrt_p: int, liquidity: int, fee: int) -> ConcentratedPool:
    return ConcentratedPool(token0="token0", token1="token1", sqrt_price_x96=sqrt_p, liquidity=liquidity,
                            fee_pips=fee, tick_spacing=V3_TICK_SPACING.get(fee, 60))


def _v3_forward(sqrt_p: int, liquidity: int, fee: int, swap: Swap) -> Optional[Tuple[int, int]]:
    """(amount_out, sqrt_after) of ``swap``'s exact input from (sqrt_p, liquidity)."""
    try:
        res = _v3_pool(sqrt_p, liquidity, fee).quote_exact_in(swap.amount_in, swap.zero_for_one)
    except MASEError:
        return None
    if not res.fully_filled or res.sqrt_price_after_x96 is None:
        return None
    return res.amount_out, res.sqrt_price_after_x96


def reconstruct_v3_before(swap: Swap, fee: int) -> Optional[int]:
    """sqrtPriceX96 before ``swap``, assuming it stayed inside one liquidity range.

    The post-swap price is monotone in the pre-swap price for a fixed exact
    input, so a binary search over the v3 swap loop finds the pre-swap price
    whose forward run lands exactly on the logged ``sqrtPriceX96``. Returns
    None when no price reproduces both the logged price and the output (the
    swap crossed an initialized tick, or was not a plain exact-input swap).
    """
    target, liq = swap.sqrt_after, swap.liquidity_after
    if not target or not liq:
        return None
    if swap.zero_for_one:  # price moves down: the start is above the end
        lo, hi = target, MAX_SQRT_RATIO - 1
    else:
        lo, hi = MIN_SQRT_RATIO + 1, target
    while lo < hi:
        mid = (lo + hi) // 2
        r = _v3_forward(mid, liq, fee, swap)
        if r is None:
            # Out of range or not filled: an impossible start, move towards the end price.
            if swap.zero_for_one:
                hi = mid
            else:
                lo = mid + 1
            continue
        if r[1] < target:
            lo = mid + 1
        else:
            hi = mid
    r = _v3_forward(lo, liq, fee, swap)
    if r is None or r[1] != target or r[0] != swap.amount_out:
        return None
    return lo


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


@dataclass
class VictimImpact:
    tx_index: int
    tx_hash: str
    address: str
    token_in: Optional[str]
    token_out: Optional[str]
    amount_in: int
    amount_out: int
    # "exact_in": the victim fixed its input and lost output; "exact_out": it
    # fixed its output and paid extra input. Decided by which form of the
    # AMM formula reproduces the victim's real amounts.
    mode: str = "exact_in"
    counterfactual_out: Optional[int] = None
    loss: Optional[int] = None  # exact_in: token_out units the victim did not get
    counterfactual_in: Optional[int] = None
    extra_in: Optional[int] = None  # exact_out: token_in units the victim overpaid
    loss_bps: Optional[float] = None
    loss_wei: Optional[int] = None
    exact: bool = False
    method: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tx_index": self.tx_index, "tx_hash": self.tx_hash, "address": self.address,
            "token_in": self.token_in, "token_out": self.token_out,
            "amount_in": self.amount_in, "amount_out": self.amount_out, "mode": self.mode,
            "counterfactual_out": self.counterfactual_out, "loss": self.loss,
            "counterfactual_in": self.counterfactual_in, "extra_in": self.extra_in,
            "loss_bps": None if self.loss_bps is None else round(self.loss_bps, 2),
            "loss_wei": self.loss_wei, "exact": self.exact, "method": self.method,
        }


@dataclass
class SandwichFinding:
    block: int
    pool: str
    version: str
    link: str  # "sender" or "contract"
    attacker_eoa: str
    attacker_contract: str
    front_tx: str
    back_tx: str
    front_index: int
    back_index: int
    token_a: Optional[str]
    token_b: Optional[str]
    front_in: int
    front_out: int
    back_in: int
    back_out: int
    victims: List[VictimImpact] = field(default_factory=list)
    gas_cost_wei: int = 0
    coinbase_transfer_wei: int = 0
    gross_profit_wei: Optional[int] = None
    net_profit_wei: Optional[int] = None
    profit_valuation: Optional[str] = None
    transfer_check: Optional[Dict[str, Any]] = None
    fee: Optional[int] = None

    @property
    def gross_profit(self) -> int:
        """Exact profit in token A (``back_out - front_in``)."""
        return self.back_out - self.front_in

    @property
    def leftover_b(self) -> int:
        """Token B bought in the front-run and not sold back (negative = sold more)."""
        return self.front_out - self.back_in

    @property
    def victim_loss_wei(self) -> Optional[int]:
        vals = [v.loss_wei for v in self.victims]
        return None if not vals or any(v is None for v in vals) else sum(vals)  # type: ignore[misc]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block": self.block, "pool": self.pool, "version": self.version, "fee": self.fee,
            "attacker": {"eoa": self.attacker_eoa, "contract": self.attacker_contract, "linked_by": self.link},
            "front_run": {"tx_index": self.front_index, "tx_hash": self.front_tx,
                          "amount_in": self.front_in, "amount_out": self.front_out},
            "back_run": {"tx_index": self.back_index, "tx_hash": self.back_tx,
                         "amount_in": self.back_in, "amount_out": self.back_out},
            "token_a": self.token_a, "token_b": self.token_b,
            "gross_profit_token_a": self.gross_profit, "leftover_token_b": self.leftover_b,
            "gas_cost_wei": self.gas_cost_wei, "coinbase_transfer_wei": self.coinbase_transfer_wei,
            "gross_profit_wei": self.gross_profit_wei, "net_profit_wei": self.net_profit_wei,
            "profit_valuation": self.profit_valuation,
            "victims": [v.to_dict() for v in self.victims],
            "victim_loss_wei": self.victim_loss_wei,
            "transfer_check": self.transfer_check,
        }


@dataclass
class DetectionReport:
    block: int
    source: str
    transactions: int
    swaps: int
    pools: int
    findings: List[SandwichFinding]
    warnings: List[str]

    def summary(self) -> Dict[str, Any]:
        victims = sum(len(f.victims) for f in self.findings)
        net = [f.net_profit_wei for f in self.findings if f.net_profit_wei is not None]
        loss = [v.loss_wei for f in self.findings for v in f.victims if v.loss_wei is not None]
        return {
            "block": self.block, "source": self.source, "transactions": self.transactions,
            "swaps": self.swaps, "pools": self.pools, "sandwiches": len(self.findings), "victims": victims,
            "attacker_net_profit_wei": sum(net), "victim_loss_wei": sum(loss),
            "exact_victims": sum(1 for f in self.findings for v in f.victims if v.exact),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary(), "sandwiches": [f.to_dict() for f in self.findings],
                "warnings": list(self.warnings)}


def _link(front: TxInfo, back: TxInfo) -> Optional[str]:
    if front.sender and front.sender == back.sender:
        return "sender"
    if front.to and front.to == back.to and front.to not in KNOWN_ROUTERS:
        return "contract"
    return None


def _is_attacker_tx(tx: TxInfo, front: TxInfo, back: TxInfo, link: str) -> bool:
    if tx.sender in (front.sender, back.sender):
        return True
    return link == "contract" and tx.to == front.to


def detect_block(data: BlockData, weth: str = WETH) -> DetectionReport:
    """Find every sandwich in ``data``."""
    findings: List[SandwichFinding] = []
    by_pool = data.pool_swaps()
    for pool, swaps in by_pool.items():
        fee_hint = data.pools.get(pool, {}).get("fee")
        used: set = set()
        for i, front in enumerate(swaps):
            if i in used:
                continue
            ftx = data.txs.get(front.tx_index)
            if ftx is None:
                continue
            for j in range(i + 1, len(swaps)):
                back = swaps[j]
                if back.tx_index == front.tx_index or back.zero_for_one == front.zero_for_one or j in used:
                    continue
                btx = data.txs.get(back.tx_index)
                if btx is None:
                    continue
                link = _link(ftx, btx)
                if link is None:
                    continue
                if not (2 * back.amount_in >= front.amount_out and 10 * back.amount_in <= 11 * front.amount_out):
                    break
                victims = [k for k in range(i + 1, j)
                           if swaps[k].tx_index not in (front.tx_index, back.tx_index)
                           and swaps[k].zero_for_one == front.zero_for_one
                           and not _is_attacker_tx(data.txs[swaps[k].tx_index], ftx, btx, link)]
                if victims:
                    used.update((i, j))
                    findings.append(_build_finding(data, swaps, i, j, victims, link, fee_hint, weth))
                break
    findings.sort(key=lambda f: (f.front_index, f.pool))
    return DetectionReport(block=data.number, source=data.source, transactions=len(data.txs),
                           swaps=len(data.swaps), pools=len(by_pool), findings=findings,
                           warnings=list(data.warnings))


def _build_finding(data: BlockData, swaps: List[Swap], i: int, j: int, victim_idx: List[int], link: str,
                   fee_hint: Any, weth: str) -> SandwichFinding:
    front, back = swaps[i], swaps[j]
    ftx, btx = data.txs[front.tx_index], data.txs[back.tx_index]
    meta = data.pools.get(front.pool, {})
    token_a = front.token_in or _token_label(meta, front.zero_for_one, True)
    token_b = front.token_out or _token_label(meta, front.zero_for_one, False)
    finding = SandwichFinding(
        block=data.number, pool=front.pool, version=front.version, link=link,
        attacker_eoa=ftx.sender, attacker_contract=ftx.to if link == "contract" or ftx.to == btx.to else "",
        front_tx=ftx.hash, back_tx=btx.hash, front_index=ftx.index, back_index=btx.index,
        token_a=token_a, token_b=token_b, front_in=front.amount_in, front_out=front.amount_out,
        back_in=back.amount_in, back_out=back.amount_out,
        gas_cost_wei=ftx.fee_wei + btx.fee_wei,
        coinbase_transfer_wei=ftx.coinbase_transfer + btx.coinbase_transfer,
    )
    # Profit in wei.
    if token_a == weth:
        finding.gross_profit_wei = finding.gross_profit
        finding.profit_valuation = "exact_weth"
    elif token_b == weth:
        price = _price_after(back)  # token_b per token_a
        if price is not None:
            finding.gross_profit_wei = _floor(finding.gross_profit * price)
            finding.profit_valuation = "pool_price_after_back_run"
    if finding.gross_profit_wei is not None:
        finding.net_profit_wei = finding.gross_profit_wei - finding.gas_cost_wei - finding.coinbase_transfer_wei
    # Victims.
    if front.version == "v2":
        fee, _ = calibrate_v2_fee(swaps, _int(fee_hint) if fee_hint is not None else None)
        finding.fee = fee
        impacts = _v2_victims(data, swaps, i, j, victim_idx, fee, ftx, btx, link)
    else:
        fee_v3 = _int(fee_hint) if fee_hint is not None else None
        impacts, finding.fee = _v3_victims(data, swaps, i, j, victim_idx, fee_v3, ftx, btx, link)
    for v in impacts:
        v.loss_wei = _loss_wei(v, weth)
    finding.victims = impacts
    if data.source == "traces" and token_a:
        finding.transfer_check = _transfer_check(data, finding, ftx, btx)
    return finding


def _token_label(meta: Mapping[str, Any], zfo: bool, is_in: bool) -> Optional[str]:
    t0, t1 = _addr(meta.get("token0")), _addr(meta.get("token1"))
    if not (t0 and t1):
        return None
    return (t0 if zfo else t1) if is_in else (t1 if zfo else t0)


def _floor(x: Fraction) -> int:
    return x.numerator // x.denominator


def _price_after(swap: Swap) -> Optional[Fraction]:
    """Marginal price of swap.token_out in units of swap.token_in after ``swap`` (B per A for a back-run)."""
    if swap.version == "v2" and swap.reserves_after:
        r0, r1 = swap.reserves_after
        # back-run sells B (token_in) for A (token_out): B per A = reserve_B / reserve_A
        return Fraction(r0, r1) if swap.zero_for_one else Fraction(r1, r0)
    if swap.version == "v3" and swap.sqrt_after:
        p = sqrt_price_x96_to_price(swap.sqrt_after)  # token1 per token0
        return 1 / p if swap.zero_for_one else p
    return None


def _loss_wei(v: VictimImpact, weth: str) -> Optional[int]:
    if v.mode == "exact_out":
        if v.extra_in is None or not v.counterfactual_in:
            return None
        if v.token_in == weth:
            return v.extra_in
        if v.token_out == weth:
            return v.extra_in * v.amount_out // v.counterfactual_in
        return None
    if v.loss is None:
        return None
    if v.token_out == weth:
        return v.loss
    if v.token_in == weth and v.counterfactual_out:
        # Value the lost output at the rate the victim would have got.
        return v.loss * v.amount_in // v.counterfactual_out
    return None


def _victim(data: BlockData, s: Swap) -> VictimImpact:
    tx = data.txs[s.tx_index]
    meta = data.pools.get(s.pool, {})
    return VictimImpact(tx_index=s.tx_index, tx_hash=tx.hash, address=tx.sender,
                        token_in=s.token_in or _token_label(meta, s.zero_for_one, True),
                        token_out=s.token_out or _token_label(meta, s.zero_for_one, False),
                        amount_in=s.amount_in, amount_out=s.amount_out)


def _set_loss(v: VictimImpact, cf_out: int) -> None:
    v.mode = "exact_in"
    v.counterfactual_out = cf_out
    v.loss = cf_out - v.amount_out
    v.loss_bps = (v.loss * 10_000 / cf_out) if cf_out else None


def _set_overpay(v: VictimImpact, cf_in: int) -> None:
    v.mode = "exact_out"
    v.counterfactual_in = cf_in
    v.extra_in = v.amount_in - cf_in
    v.loss_bps = (v.extra_in * 10_000 / cf_in) if cf_in else None


def _v2_replay(state: Tuple[int, int], swap: Swap, fee: int) -> Tuple[int, int]:
    """Replay ``swap`` on counterfactual reserves in the mode it was sent in.

    A swap whose real amounts match ``getAmountIn`` (and not ``getAmountOut``)
    fixed its output, so in the counterfactual it buys the same output; any
    other swap keeps its input.
    """
    if swap.reserves_before is not None:
        r_in, r_out = _v2_in_out(swap.reserves_before, swap.zero_for_one)
        if v2_get_amount_out(swap.amount_in, r_in, r_out, fee) != swap.amount_out \
                and v2_get_amount_in(swap.amount_out, r_in, r_out, fee) == swap.amount_in:
            c_in, c_out = _v2_in_out(state, swap.zero_for_one)
            paid = v2_get_amount_in(swap.amount_out, c_in, c_out, fee)
            if swap.zero_for_one:
                return state[0] + paid, state[1] - swap.amount_out
            return state[0] - swap.amount_out, state[1] + paid
    return _v2_step(state, swap.zero_for_one, swap.amount_in, fee)[0]


def _v2_victims(data: BlockData, swaps: List[Swap], i: int, j: int, victim_idx: List[int], fee: int,
                ftx: TxInfo, btx: TxInfo, link: str) -> List[VictimImpact]:
    front = swaps[i]
    resets = [a for a in data.anchors if a.pool == front.pool and a.reserves is None
              and front.pos < a.pos < swaps[victim_idx[-1]].pos]
    out: List[VictimImpact] = []
    for k in victim_idx:
        s = swaps[k]
        v = _victim(data, s)
        out.append(v)
        if front.reserves_before is None or s.reserves_before is None:
            v.method = "v2 reserves unknown before the front-run"
            continue
        try:
            state = front.reserves_before
            for m in range(i + 1, k):
                other = swaps[m]
                if _is_attacker_tx(data.txs[other.tx_index], ftx, btx, link):
                    continue
                state = _v2_replay(state, other, fee)
            r_in, r_out = _v2_in_out(s.reserves_before, s.zero_for_one)
            c_in, c_out = _v2_in_out(state, s.zero_for_one)
            if v2_get_amount_out(s.amount_in, r_in, r_out, fee) == s.amount_out:
                _set_loss(v, v2_get_amount_out(s.amount_in, c_in, c_out, fee))
                v.exact = not resets
                v.method = "v2 getAmountOut counterfactual (victim fixed its input)"
            elif v2_get_amount_in(s.amount_out, r_in, r_out, fee) == s.amount_in:
                _set_overpay(v, v2_get_amount_in(s.amount_out, c_in, c_out, fee))
                v.exact = not resets
                v.method = "v2 getAmountIn counterfactual (victim fixed its output)"
            else:
                model = v2_get_amount_out(s.amount_in, r_in, r_out, fee)
                _set_loss(v, v2_get_amount_out(s.amount_in, c_in, c_out, fee))
                v.method = (f"v2 getAmountOut counterfactual (approximate: the model misses the victim's real "
                            f"output by {s.amount_out - model} wei)")
        except MASEError as exc:
            v.method = f"v2 replay failed: {exc}"
            continue
        if resets:
            v.method += "; liquidity changed in between"
    return out


def _v3_victims(data: BlockData, swaps: List[Swap], i: int, j: int, victim_idx: List[int], fee_hint: Optional[int],
                ftx: TxInfo, btx: TxInfo, link: str) -> Tuple[List[VictimImpact], Optional[int]]:
    front = swaps[i]
    out = [_victim(data, swaps[k]) for k in victim_idx]
    if front.sqrt_after is None:
        for v in out:
            v.method = "v3 pool state is not in trace data; loss not computed"
        return out, fee_hint
    fees = [fee_hint] if fee_hint else list(V3_TICK_SPACING)
    pre = fee = None
    for f in fees:
        pre = reconstruct_v3_before(front, f)
        if pre is not None:
            fee = f
            break
    if pre is None or fee is None:
        for v in out:
            v.method = "v3 front-run crossed an initialized tick; pre-front-run state not recoverable from logs"
        return out, fee_hint
    liq = front.liquidity_after or 0
    for v, k in zip(out, victim_idx):
        s = swaps[k]
        try:
            actual = _v3_pool(pre, liq, fee)
            cf = _v3_pool(pre, liq, fee)
            actual.apply_swap(front.amount_in, front.zero_for_one)
            for m in range(i + 1, k):
                other = swaps[m]
                fixed_out = (actual.quote_exact_in(other.amount_in, other.zero_for_one).amount_out != other.amount_out
                             and actual.quote_exact_out(other.amount_out, other.zero_for_one).amount_in
                             == other.amount_in)
                actual.apply_swap(other.amount_in, other.zero_for_one)
                if not _is_attacker_tx(data.txs[other.tx_index], ftx, btx, link):
                    # Replay in the mode it was sent in (see _v2_replay).
                    paid = (cf.quote_exact_out(other.amount_out, other.zero_for_one).amount_in if fixed_out
                            else other.amount_in)
                    cf.apply_swap(paid, other.zero_for_one)
            model = actual.quote_exact_in(s.amount_in, s.zero_for_one)
            same_state = model.sqrt_price_after_x96 == s.sqrt_after and s.liquidity_after == liq
            if model.amount_out == s.amount_out and same_state:
                _set_loss(v, cf.quote_exact_in(s.amount_in, s.zero_for_one).amount_out)
                v.exact = True
                v.method = "v3 swap-loop counterfactual from the reconstructed pre-front-run price (fixed input)"
                continue
            model_out = actual.quote_exact_out(s.amount_out, s.zero_for_one)
            if model_out.amount_in == s.amount_in and model_out.sqrt_price_after_x96 == s.sqrt_after \
                    and s.liquidity_after == liq:
                _set_overpay(v, cf.quote_exact_out(s.amount_out, s.zero_for_one).amount_in)
                v.exact = True
                v.method = "v3 swap-loop counterfactual from the reconstructed pre-front-run price (fixed output)"
                continue
            _set_loss(v, cf.quote_exact_in(s.amount_in, s.zero_for_one).amount_out)
            v.method = ("v3 swap-loop counterfactual from the reconstructed pre-front-run price "
                        "(approximate: the victim crossed a tick, constant liquidity assumed)")
        except MASEError as exc:
            v.method = f"v3 replay failed: {exc}"
            continue
    return out, fee


def _transfer_check(data: BlockData, f: SandwichFinding, ftx: TxInfo, btx: TxInfo) -> Dict[str, Any]:
    """Net token-A flow of the attacker's addresses in the two attack transactions."""
    me = {a for a in (ftx.sender, btx.sender, f.attacker_contract) if a}
    net = 0
    for m in data.moves:
        if m.tx_index not in (ftx.index, btx.index) or m.token != f.token_a:
            continue
        if m.dst in me and m.src not in me:
            net += m.amount
        elif m.src in me and m.dst not in me:
            net -= m.amount
    return {"net_token_a_flow": net, "matches_gross_profit": net == f.gross_profit}


# ---------------------------------------------------------------------------
# Loading exports
# ---------------------------------------------------------------------------


def load_export(obj: Mapping[str, Any]) -> List[BlockData]:
    """Parse one export object (one block, or ``{"blocks": [...]}``)."""
    if "blocks" in obj:
        out: List[BlockData] = []
        for b in obj["blocks"]:
            out.extend(load_export(b))
        return out
    if "traces" in obj:
        return [parse_traces_export(obj)]
    if "block" in obj and "receipts" in obj:
        return [parse_logs_export(obj["block"], obj["receipts"], obj.get("pools"))]
    raise DetectorError("unknown export format: expected 'traces', or 'block' + 'receipts', or 'blocks'")


def read_export_file(path: str) -> Dict[str, Any]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def write_export_file(path: str, export: Mapping[str, Any]) -> None:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as fh:
        json.dump(export, fh)


def detect_export(obj: Mapping[str, Any], weth: str = WETH) -> List[DetectionReport]:
    return [detect_block(b, weth) for b in load_export(obj)]


# ---------------------------------------------------------------------------
# JSON-RPC
# ---------------------------------------------------------------------------

Transport = Callable[[str, List[Any]], Any]


class JsonRpcClient:
    """Minimal Ethereum JSON-RPC client over ``urllib`` (or any injected transport)."""

    def __init__(self, url: Optional[str] = None, transport: Optional[Transport] = None, timeout: float = 30.0) -> None:
        self.url = url or os.environ.get("ETH_RPC_URL", "")
        if transport is None and not self.url:
            raise DetectorError("no RPC endpoint: pass --rpc URL or set $ETH_RPC_URL")
        self._transport = transport
        self.timeout = timeout
        self._next_id = 1

    def request(self, method: str, params: List[Any]) -> Any:
        if self._transport is not None:
            return self._transport(method, params)
        body = json.dumps({"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params}).encode()
        self._next_id += 1
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json",
                                                                   "User-Agent": "mev-sandwich-simulator"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - user-supplied endpoint
                payload = json.loads(resp.read().decode())
        except OSError as exc:
            raise DetectorError(f"RPC {method} failed: {exc}") from exc
        if "error" in payload and payload["error"]:
            raise DetectorError(f"RPC {method} error: {payload['error']}")
        return payload.get("result")

    def block_number(self) -> int:
        return _int(self.request("eth_blockNumber", []))

    def block(self, number: Union[int, str]) -> Dict[str, Any]:
        tag = number if isinstance(number, str) else hex(number)
        result = self.request("eth_getBlockByNumber", [tag, True])
        if not result:
            raise DetectorError(f"block {number} not found")
        return result

    def receipts(self, block: Mapping[str, Any]) -> List[Dict[str, Any]]:
        try:
            result = self.request("eth_getBlockReceipts", [block["number"]])
            if isinstance(result, list):
                return result
        except DetectorError:
            pass
        return [self.request("eth_getTransactionReceipt", [tx["hash"]]) for tx in block.get("transactions", [])]

    def call(self, to: str, data: str, block_tag: str = "latest") -> str:
        return str(self.request("eth_call", [{"to": to, "data": data}, block_tag]) or "0x")


def fetch_block_export(client: JsonRpcClient, number: Union[int, str] = "latest",
                       with_pool_tokens: bool = True) -> Dict[str, Any]:
    """Block + receipts (+ token0/token1/fee of every swapping pool) as an export object."""
    block = client.block(number)
    receipts = client.receipts(block)
    pools: Dict[str, Dict[str, Any]] = {}
    if with_pool_tokens:
        seen: Dict[str, str] = {}
        for r in receipts:
            for lg in r.get("logs", []):
                topics = [str(t).lower() for t in lg.get("topics", [])]
                if topics and topics[0] in (TOPIC_V2_SWAP, TOPIC_V3_SWAP):
                    seen[_addr(lg["address"])] = "v3" if topics[0] == TOPIC_V3_SWAP else "v2"
        for pool, version in sorted(seen.items()):
            meta: Dict[str, Any] = {"version": version}
            try:
                meta["token0"] = _word_addr(_words(client.call(pool, SEL_TOKEN0, block["number"]))[0])
                meta["token1"] = _word_addr(_words(client.call(pool, SEL_TOKEN1, block["number"]))[0])
                if version == "v3":
                    meta["fee"] = _words(client.call(pool, SEL_FEE, block["number"]))[0]
            except (DetectorError, IndexError):
                pass
            pools[pool] = meta
    return {"format": "logs", "block": block, "receipts": receipts, "pools": pools}


def fetch_v2_pool(client: JsonRpcClient, pair: str, block_tag: str = "latest") -> Dict[str, Any]:
    """token0, token1 and reserves of a v2 pair (for the pre-trade checker)."""
    words = _words(client.call(pair, SEL_GET_RESERVES, block_tag))
    t0 = _words(client.call(pair, SEL_TOKEN0, block_tag))
    t1 = _words(client.call(pair, SEL_TOKEN1, block_tag))
    if len(words) < 2 or not t0 or not t1:
        raise DetectorError(f"{pair} does not look like a Uniswap v2 pair")
    return {"version": "v2", "address": _addr(pair), "token0": _word_addr(t0[0]), "token1": _word_addr(t1[0]),
            "reserve0": words[0], "reserve1": words[1]}


def fetch_v3_pool(client: JsonRpcClient, pool: str, block_tag: str = "latest") -> Dict[str, Any]:
    """token0, token1, fee, sqrtPriceX96 and active liquidity of a v3 pool (single-range view)."""
    slot0 = _words(client.call(pool, selector("slot0()"), block_tag))
    liq = _words(client.call(pool, selector("liquidity()"), block_tag))
    t0 = _words(client.call(pool, SEL_TOKEN0, block_tag))
    t1 = _words(client.call(pool, SEL_TOKEN1, block_tag))
    fee = _words(client.call(pool, SEL_FEE, block_tag))
    if not slot0 or not liq or not t0 or not t1 or not fee:
        raise DetectorError(f"{pool} does not look like a Uniswap v3 pool")
    return {"version": "v3", "address": _addr(pool), "token0": _word_addr(t0[0]), "token1": _word_addr(t1[0]),
            "fee": fee[0], "sqrt_price_x96": slot0[0], "liquidity": liq[0]}

