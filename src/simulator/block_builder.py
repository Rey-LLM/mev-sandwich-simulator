"""Layer 04: PBS block builder, MEV-Boost auction and block layout fuzzer.

This module models the proposer-builder separation (PBS) flow an Ethereum
validator runs through MEV-Boost, entirely offline on the simulator's pools:

1. Users send EIP-1559 transactions to a :class:`TxPool`; searchers send
   atomic :class:`Bundle` objects (front-run, victim, back-run plus a
   ``coinbase_transfer`` bribe to the builder).
2. Each :class:`BlockBuilder` orders bundles and public transactions into a
   block layout and seals it with :func:`seal_block`, which executes every
   item on a forked :class:`BuilderState` under a gas limit that is clamped to
   30,000,000.
3. Builders submit a :class:`BuilderBid` to one or more :class:`Relay`
   objects. A relay re-executes the block from the parent state and rejects
   any bid whose value, gas or state root does not reproduce.
4. The :class:`Proposer` takes the highest valid bid, or builds a local block
   when no bid clears its minimum.

Ordering
--------
* EIP-1559: a transaction is executable only if ``max_fee_per_gas >=
  base_fee``; its tip is ``min(max_priority_fee, max_fee - base_fee)``. Base
  fee is burned, the tip goes to the builder's coinbase.
* Nonces: public transactions are merged with a price-and-nonce heap (one head
  per sender, highest tip first), like geth's ``TransactionsByPriceAndNonce``.
  A transaction whose nonce is not the sender's next nonce is skipped.
* Bundles are scored by builder value per gas from a top-of-block simulation,
  then placed first and re-executed at their real position; a bundle that
  reverts (outside ``reverting_tx_hashes``), conflicts with an already
  included transaction, or does not fit the remaining gas is dropped whole.
* Gas overflow: an item whose gas limit exceeds the remaining block gas is
  rejected and the builder keeps trying smaller items until less than 21,000
  gas remains.

Guards
------
* Sandwich and JIT detection (:func:`classify_bundle`): a bundle where one
  account swaps on a pool before and after another account's swap through
  that pool is a sandwich (``multi_hop`` when the victim's route has more than
  one hop); one where an account adds liquidity before and removes it after
  another account's swap is JIT liquidity. :class:`GuardPolicy` decides
  whether builders allow, flag or reject each class.
* Searcher front-run legs carry an exact ``amount_out_min``, so any state
  change ahead of the bundle (including someone else's liquidity change)
  makes the bundle revert and drop instead of executing on stale quotes.

Layout fuzzing
--------------
:func:`fuzz_block_layout` hill-climbs over block layouts with swap / move /
reverse / drop / restore / promote mutations, repairing per-sender nonce
order after every mutation, and scores each layout by sealing it. Work is
split into a fixed number of seeded shards and run on the capped
:class:`~simulator.parallel.WorkerPool`, so results are identical for any
worker count. The fuzzed layout is never worth less than the greedy layout
it starts from.

Traces
------
:func:`seal_block` with ``trace=True`` records, per transaction, gas used,
cumulative gas, tip, fees, coinbase transfers and every pool-field and
balance change (``pool v2:WETH/USDC reserve0: a -> b``). ``measure_unrealized``
additionally prices the best sandwich still available before every public
swap: the block's unrealized MEV capacity. :meth:`SealedBlock.log_lines`
renders the trace and is emitted on the ``simulator.block_builder`` logger at
DEBUG level.

Limits: gas figures are per-hop estimates (see ``mempool_emulator``), not
EVM-metered; LP fee accrual is not modeled, so JIT bundles are detected and
policed but their LP profit is not computed; swap routes may not visit the
same pool twice.
"""

from __future__ import annotations

import copy
import hashlib
import heapq
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from .config import ConfigError, SimulationConfig
from .invariants import input_reserve_scale
from .mempool_emulator import GAS_TX_BASE, _apply_full, estimate_swap_gas, geometric_grid
from .parallel import WorkerPool
from .slippage_engine import (
    MAX_TICK,
    MIN_TICK,
    ConcentratedPool,
    MASEError,
    Pool,
    V2Pool,
    _direction,
    get_amount0_delta,
    get_amount1_delta,
    get_sqrt_ratio_at_tick,
    min_amount_out,
)
from .state_fork import StateFork, SwapReverted

log = logging.getLogger("simulator.block_builder")

BLOCK_GAS_LIMIT = 30_000_000
ELASTICITY_MULTIPLIER = 2
BASE_FEE_MAX_CHANGE_DENOMINATOR = 8
GAS_TRANSFER = 21_000
GAS_MODIFY_LIQUIDITY = 130_000
REPLACEMENT_BUMP_PCT = 10
ETH = "ETH"
BURN = "0x0burn"
FUZZ_SHARDS = 16  # fixed so fuzz results never depend on worker count
GWEI = 10**9


def clamp_gas_limit(gas_limit: Optional[int]) -> int:
    """Block gas limit, clamped into ``[21000, 30_000_000]``."""
    if gas_limit is None:
        return BLOCK_GAS_LIMIT
    return max(GAS_TRANSFER, min(int(gas_limit), BLOCK_GAS_LIMIT))


def next_base_fee(base_fee: int, gas_used: int, gas_limit: int) -> int:
    """EIP-1559 base fee update for the next block."""
    target = gas_limit // ELASTICITY_MULTIPLIER
    if target <= 0 or gas_used == target:
        return base_fee
    if gas_used > target:
        delta = max(1, base_fee * (gas_used - target) // target // BASE_FEE_MAX_CHANGE_DENOMINATOR)
        return base_fee + delta
    delta = base_fee * (target - gas_used) // target // BASE_FEE_MAX_CHANGE_DENOMINATOR
    return max(0, base_fee - delta)


# ---------------------------------------------------------------------------
# Transactions and bundles
# ---------------------------------------------------------------------------


class TxKind(str, Enum):
    SWAP = "swap"
    TRANSFER = "transfer"
    MINT = "mint"  # add concentrated liquidity
    BURN = "burn"  # remove concentrated liquidity


@dataclass(frozen=True, eq=False)
class BuilderTx:
    """EIP-1559 transaction as seen by a builder."""

    tx_hash: str
    sender: str
    nonce: int
    gas_limit: int
    max_fee_per_gas: int
    max_priority_fee_per_gas: int
    kind: TxKind = TxKind.SWAP
    payload: Mapping[str, Any] = field(default_factory=dict)

    def effective_tip(self, base_fee: int) -> Optional[int]:
        """Tip per gas paid to the builder, or None when the fee cap is below base fee."""
        if self.max_fee_per_gas < base_fee:
            return None
        return min(self.max_priority_fee_per_gas, self.max_fee_per_gas - base_fee)

    @property
    def route(self) -> Tuple[str, ...]:
        r = self.payload.get("route") or [self.payload.get("pool_id", "")]
        return tuple([r] if isinstance(r, str) else r)

    @property
    def token_in(self) -> str:
        return str(self.payload.get("token_in", ""))

    @property
    def amount_in(self) -> int:
        return int(self.payload.get("amount_in", 0))

    @property
    def amount_out_min(self) -> int:
        return int(self.payload.get("amount_out_min") or 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tx_hash": self.tx_hash,
            "sender": self.sender,
            "nonce": self.nonce,
            "gas_limit": self.gas_limit,
            "max_fee_per_gas": self.max_fee_per_gas,
            "max_priority_fee_per_gas": self.max_priority_fee_per_gas,
            "kind": self.kind.value,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True, eq=False)
class Bundle:
    """Atomic searcher bundle: all transactions land in order, or none do."""

    bundle_id: str
    txs: Tuple[BuilderTx, ...]
    coinbase_transfer_wei: int = 0
    searcher: str = ""
    reverting_tx_hashes: frozenset = frozenset()
    target_tx: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def tx_hashes(self) -> Tuple[str, ...]:
        return tuple(t.tx_hash for t in self.txs)

    @property
    def gas_limit(self) -> int:
        return sum(t.gas_limit for t in self.txs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "searcher": self.searcher,
            "txs": list(self.tx_hashes),
            "coinbase_transfer_wei": self.coinbase_transfer_wei,
            "target_tx": self.target_tx,
            "meta": dict(self.meta),
        }


Item = Union[BuilderTx, Bundle]


def item_id(item: Item) -> str:
    return f"bundle:{item.bundle_id}" if isinstance(item, Bundle) else f"tx:{item.tx_hash}"


# ---------------------------------------------------------------------------
# Sandwich / JIT classification and guard policy
# ---------------------------------------------------------------------------


class GuardAction(str, Enum):
    ALLOW = "allow"
    FLAG = "flag"
    REJECT = "reject"


@dataclass(frozen=True)
class GuardPolicy:
    sandwich: GuardAction = GuardAction.ALLOW
    multi_hop_sandwich: GuardAction = GuardAction.ALLOW
    jit_liquidity: GuardAction = GuardAction.FLAG

    def action_for(self, label: str) -> GuardAction:
        return {"sandwich": self.sandwich, "multi_hop_sandwich": self.multi_hop_sandwich,
                "jit_liquidity": self.jit_liquidity}.get(label, GuardAction.ALLOW)


def _swap_pool_set(tx: BuilderTx) -> Set[str]:
    return set(tx.route) if tx.kind is TxKind.SWAP else set()


def classify_bundle(bundle: Bundle) -> List[str]:
    """Labels for MEV patterns inside a bundle: sandwich, multi_hop_sandwich, jit_liquidity."""
    labels: List[str] = []
    txs = bundle.txs
    for vi, victim in enumerate(txs):
        if victim.kind is not TxKind.SWAP:
            continue
        victim_pools = _swap_pool_set(victim)
        before = txs[:vi]
        after = txs[vi + 1:]
        for b in before:
            if b.sender == victim.sender:
                continue
            if b.kind is TxKind.SWAP and _swap_pool_set(b) & victim_pools:
                if any(a.sender == b.sender and a.kind is TxKind.SWAP and _swap_pool_set(a) & _swap_pool_set(b)
                       for a in after):
                    label = "multi_hop_sandwich" if len(victim.route) > 1 else "sandwich"
                    if label not in labels:
                        labels.append(label)
            if b.kind is TxKind.MINT and b.payload.get("pool_id") in victim_pools:
                if any(a.sender == b.sender and a.kind is TxKind.BURN
                       and a.payload.get("pool_id") == b.payload.get("pool_id") for a in after):
                    if "jit_liquidity" not in labels:
                        labels.append("jit_liquidity")
    return labels


# ---------------------------------------------------------------------------
# Builder state: pools + balances + nonces + LP positions
# ---------------------------------------------------------------------------


PositionKey = Tuple[str, str, int, int]  # (owner, pool_id, tick_lower, tick_upper)


def clone_pool(pool: Pool) -> Pool:
    """Fast exact copy of a pool: every field is an int/str except the tick map and tick array."""
    twin = copy.copy(pool)
    if isinstance(pool, ConcentratedPool):
        twin.ticks = dict(pool.ticks)
        twin._tick_array = list(pool._tick_array)
    return twin


def _clone_fork(fork: StateFork) -> StateFork:
    child = StateFork(block_number=fork.block_number)
    child.pools = {pid: clone_pool(p) for pid, p in fork.pools.items()}
    for acct, toks in fork.balances.items():
        if toks:
            child.balances[acct].update(toks)
    return child


class BuilderState:
    def __init__(self, fork: StateFork, nonces: Optional[Mapping[str, int]] = None,
                 positions: Optional[Mapping[PositionKey, int]] = None) -> None:
        self.fork = fork
        self.nonces: Dict[str, int] = dict(nonces or {})
        self.positions: Dict[PositionKey, int] = dict(positions or {})

    @classmethod
    def from_pools(cls, pools: Iterable[Pool], block_number: int = 0) -> "BuilderState":
        return cls(StateFork([clone_pool(p) for p in pools], block_number=block_number))

    @property
    def pools(self) -> Dict[str, Pool]:
        return self.fork.pools

    @property
    def block_number(self) -> int:
        return self.fork.block_number

    def nonce(self, account: str) -> int:
        return self.nonces.get(account, 0)

    def copy(self) -> "BuilderState":
        return BuilderState(_clone_fork(self.fork), self.nonces, self.positions)

    def snapshot(self) -> Tuple[Any, ...]:
        """Cheap in-memory snapshot for atomic bundle execution."""
        return ({pid: clone_pool(p) for pid, p in self.fork.pools.items()},
                {a: dict(t) for a, t in self.fork.balances.items() if t},
                dict(self.nonces), dict(self.positions))

    def restore(self, snap: Tuple[Any, ...]) -> None:
        pools, balances, nonces, positions = snap
        self.fork.pools = pools
        self.fork.balances = defaultdict(lambda: defaultdict(int))
        for acct, toks in balances.items():
            self.fork.balances[acct].update(toks)
        self.nonces, self.positions = nonces, positions

    def to_payload(self) -> Tuple[Any, ...]:
        """Picklable form (StateFork's nested defaultdicts are not)."""
        balances = {a: {t: v for t, v in toks.items() if v} for a, toks in self.fork.balances.items()}
        return (self.fork.pools, balances, self.fork.block_number, dict(self.nonces), dict(self.positions))

    @classmethod
    def from_payload(cls, payload: Tuple[Any, ...]) -> "BuilderState":
        pools, balances, block_number, nonces, positions = payload
        fork = StateFork([clone_pool(p) for p in pools.values()], block_number=block_number, balances=balances)
        return cls(fork, nonces, positions)

    def state_root(self) -> str:
        h = hashlib.sha256(self.fork.state_root().encode())
        for acct in sorted(self.nonces):
            h.update(f"n|{acct}|{self.nonces[acct]}".encode())
        for key in sorted(self.positions):
            if self.positions[key]:
                h.update(f"p|{key}|{self.positions[key]}".encode())
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Liquidity modification (JIT / LP transactions)
# ---------------------------------------------------------------------------


def modify_liquidity(pool: Pool, tick_lower: int, tick_upper: int, delta: int) -> Tuple[int, int]:
    """Add (delta > 0) or remove (delta < 0) concentrated liquidity in place.

    Returns ``(amount0, amount1)`` owed to the pool (negative = paid out to
    the LP). Raises ``SwapReverted`` and leaves the pool untouched on bad input.
    """
    if not isinstance(pool, ConcentratedPool):
        raise SwapReverted("liquidity changes are modeled for concentrated pools only")
    if delta == 0:
        raise SwapReverted("zero liquidity delta")
    if tick_lower >= tick_upper or tick_lower < MIN_TICK or tick_upper > MAX_TICK:
        raise SwapReverted("bad tick range")
    if tick_lower % pool.tick_spacing or tick_upper % pool.tick_spacing:
        raise SwapReverted("ticks not aligned to spacing")
    adding = delta > 0
    liq = abs(delta)
    sa, sb = get_sqrt_ratio_at_tick(tick_lower), get_sqrt_ratio_at_tick(tick_upper)
    in_range = tick_lower <= pool.tick < tick_upper
    if pool.tick < tick_lower:
        a0, a1 = get_amount0_delta(sa, sb, liq, adding), 0
    elif pool.tick < tick_upper:
        a0 = get_amount0_delta(pool.sqrt_price_x96, sb, liq, adding)
        a1 = get_amount1_delta(sa, pool.sqrt_price_x96, liq, adding)
    else:
        a0, a1 = 0, get_amount1_delta(sa, sb, liq, adding)
    new_active = pool.liquidity + delta if in_range else pool.liquidity
    if new_active < 0:
        raise SwapReverted("removing more liquidity than is active")
    for tick, d in ((tick_lower, delta), (tick_upper, -delta)):
        net = pool.ticks.get(tick, 0) + d
        if net:
            pool.ticks[tick] = net
        else:
            pool.ticks.pop(tick, None)
    pool._tick_array = sorted(pool.ticks)
    pool.liquidity = new_active
    sign = 1 if adding else -1
    return sign * a0, sign * a1


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class Status(str, Enum):
    SUCCESS = "success"
    REVERTED = "reverted"
    OUT_OF_GAS = "out_of_gas"
    INVALID = "invalid"  # not includable: no gas charged, nonce unchanged


@dataclass
class TxTrace:
    tx_hash: str
    sender: str
    kind: str
    status: Status
    gas_used: int = 0
    tip_per_gas: int = 0
    priority_fee_wei: int = 0
    base_fee_burned_wei: int = 0
    error: str = ""
    index: int = -1
    cumulative_gas: int = 0
    item: str = ""
    coinbase_transfer_wei: int = 0
    state_changes: List[str] = field(default_factory=list)
    unrealized_mev_wei: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "item": self.item,
            "tx_hash": self.tx_hash,
            "sender": self.sender,
            "kind": self.kind,
            "status": self.status.value,
            "gas_used": self.gas_used,
            "cumulative_gas": self.cumulative_gas,
            "tip_per_gas": self.tip_per_gas,
            "priority_fee_wei": self.priority_fee_wei,
            "base_fee_burned_wei": self.base_fee_burned_wei,
            "coinbase_transfer_wei": self.coinbase_transfer_wei,
            "unrealized_mev_wei": self.unrealized_mev_wei,
            "error": self.error,
            "state_changes": list(self.state_changes),
        }


def _pool_fields(pool: Pool) -> Dict[str, int]:
    if isinstance(pool, V2Pool):
        return {"reserve0": pool.reserve0, "reserve1": pool.reserve1}
    return {"sqrt_price_x96": pool.sqrt_price_x96, "tick": pool.tick, "liquidity": pool.liquidity}


class _Recorder:
    """Captures line-level pool-field and balance changes around one transaction."""

    def __init__(self, state: BuilderState, pool_ids: Iterable[str], accounts: Iterable[str]) -> None:
        self.state = state
        self.pool_ids = sorted({p for p in pool_ids if p in state.pools})
        self.accounts = sorted(set(accounts))
        self.pools_before = {p: _pool_fields(state.pools[p]) for p in self.pool_ids}
        self.bal_before = {a: dict(state.fork.balances.get(a, {})) for a in self.accounts}

    def changes(self) -> List[str]:
        out: List[str] = []
        for pid in self.pool_ids:
            after = _pool_fields(self.state.pools[pid])
            for key, old in self.pools_before[pid].items():
                if after[key] != old:
                    out.append(f"pool {pid} {key}: {old} -> {after[key]}")
        for acct in self.accounts:
            before = self.bal_before[acct]
            after = self.state.fork.balances.get(acct, {})
            for tok in sorted(set(before) | set(after)):
                old, new = before.get(tok, 0), after.get(tok, 0)
                if old != new:
                    out.append(f"balance {acct} {tok}: {old} -> {new} ({new - old:+d})")
        return out


Hop = Tuple[str, str, int, Any]  # (pool_id, token_in, amount_in, SwapResult)


def _quote_hops(state: BuilderState, tx: BuilderTx) -> Tuple[List[Hop], int]:
    """Read-only route quote keeping each hop's SwapResult, plus total ticks crossed."""
    route = tx.route
    if not route or len(set(route)) != len(route):
        raise SwapReverted("route must be non-empty and visit each pool once")
    if tx.amount_in <= 0:
        raise SwapReverted("amount_in must be positive")
    hops: List[Hop] = []
    token, amount, ticks = tx.token_in, tx.amount_in, 0
    for pid in route:
        pool = state.pools.get(pid)
        if pool is None:
            raise SwapReverted(f"unknown pool {pid!r}")
        try:
            zfo, out_token = _direction(pool, token)
            res = pool.quote_exact_in(amount, zfo)
        except MASEError as exc:
            raise SwapReverted(str(exc)) from exc
        if not res.fully_filled or res.amount_out <= 0:
            raise SwapReverted("insufficient liquidity on route")
        hops.append((pid, token, amount, res))
        ticks += res.ticks_crossed
        token, amount = out_token, res.amount_out
    return hops, ticks


def _required_gas(state: BuilderState, tx: BuilderTx) -> Tuple[int, Optional[str], List[Hop]]:
    """Gas the transaction needs, the revert reason if it would revert, and swap hops."""
    if tx.kind is TxKind.TRANSFER:
        return GAS_TRANSFER, None, []
    if tx.kind in (TxKind.MINT, TxKind.BURN):
        return GAS_TX_BASE + GAS_MODIFY_LIQUIDITY, None, []
    try:
        hops, ticks = _quote_hops(state, tx)
    except SwapReverted as exc:
        pools = [state.pools[p] for p in tx.route if p in state.pools]
        return estimate_swap_gas(pools), str(exc), []
    gas = estimate_swap_gas([state.pools[p] for p in tx.route], ticks)
    out = hops[-1][3].amount_out
    if out < tx.amount_out_min:
        return gas, f"amount_out {out} below min {tx.amount_out_min}", hops
    return gas, None, hops


def _apply_body(state: BuilderState, tx: BuilderTx, hops: Sequence[Hop]) -> None:
    """Mutate state for a transaction. Every failure check runs before any mutation.

    Swap hops come from :func:`_quote_hops` on the same state; routes never
    repeat a pool, so each quote is still exact when its hop is applied.
    """
    bal = state.fork.balances
    if tx.kind is TxKind.SWAP:
        for pid, tok, amt, res in hops:
            pool = state.pools[pid]
            zfo, out_tok = _direction(pool, tok)
            if isinstance(pool, V2Pool):
                pool.apply_swap(res, zfo)
            else:
                pool.apply_swap(amt, zfo)
            bal[tx.sender][tok] -= amt
            bal[tx.sender][out_tok] += res.amount_out
        return
    if tx.kind is TxKind.TRANSFER:
        value = int(tx.payload.get("value", 0))
        bal[tx.sender][ETH] -= value
        bal[str(tx.payload.get("to", BURN))][ETH] += value
        return
    pid = str(tx.payload.get("pool_id", ""))
    pool = state.pools.get(pid)
    if pool is None:
        raise SwapReverted(f"unknown pool {pid!r}")
    try:
        lower, upper = int(tx.payload["tick_lower"]), int(tx.payload["tick_upper"])
        liq = int(tx.payload["liquidity"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SwapReverted(f"bad liquidity payload: {exc}") from exc
    if liq <= 0:
        raise SwapReverted("liquidity must be positive")
    key = (tx.sender, pid, lower, upper)
    if tx.kind is TxKind.BURN:
        owned = state.positions.get(key, 0)
        if owned < liq:
            raise SwapReverted(f"position holds {owned}, cannot burn {liq}")
        delta = -liq
    else:
        delta = liq
    a0, a1 = modify_liquidity(pool, lower, upper, delta)  # validates before mutating
    t0, t1 = pool.tokens()
    bal[tx.sender][t0] -= a0
    bal[tx.sender][t1] -= a1
    remaining = state.positions.get(key, 0) + delta
    if remaining:
        state.positions[key] = remaining
    else:
        state.positions.pop(key, None)


def execute_tx(state: BuilderState, tx: BuilderTx, base_fee: int, gas_remaining: int,
               coinbase: str = "builder", trace: bool = False) -> TxTrace:
    """Execute one transaction against ``state`` (mutated in place).

    Invalid transactions (fee cap below base fee, nonce mismatch, gas limit
    over the remaining block gas, gas limit below intrinsic gas) are not
    included and leave state untouched. Reverted and out-of-gas transactions
    are included: they consume gas, pay fees and bump the nonce, but their
    body's state changes are discarded.
    """
    out = TxTrace(tx.tx_hash, tx.sender, tx.kind.value, Status.INVALID)
    tip = tx.effective_tip(base_fee)
    if tip is None:
        out.error = "fee_cap_below_base_fee"
        return out
    if tx.max_priority_fee_per_gas > tx.max_fee_per_gas:
        out.error = "tip_above_fee_cap"
        return out
    if tx.gas_limit < GAS_TX_BASE:
        out.error = "intrinsic_gas_too_low"
        return out
    if tx.gas_limit > gas_remaining:
        out.error = "gas_limit_exceeds_remaining"
        return out
    expected = state.nonce(tx.sender)
    if tx.nonce != expected:
        out.error = "nonce_too_low" if tx.nonce < expected else "nonce_gap"
        return out

    touched_pools: List[str] = list(tx.route) if tx.kind is TxKind.SWAP else [str(tx.payload.get("pool_id", ""))]
    accounts = [tx.sender, coinbase, BURN]
    if tx.kind is TxKind.TRANSFER:
        accounts.append(str(tx.payload.get("to", BURN)))
    rec = _Recorder(state, touched_pools, accounts) if trace else None

    needed, revert_reason, hops = _required_gas(state, tx)
    if needed > tx.gas_limit:
        out.status, out.gas_used, out.error = Status.OUT_OF_GAS, tx.gas_limit, f"needs {needed} gas"
    elif revert_reason is not None:
        out.status, out.gas_used, out.error = Status.REVERTED, needed, revert_reason
    else:
        try:
            _apply_body(state, tx, hops)
        except (SwapReverted, MASEError) as exc:
            out.status, out.gas_used, out.error = Status.REVERTED, needed, str(exc)
        else:
            out.status, out.gas_used = Status.SUCCESS, needed

    out.tip_per_gas = tip
    out.priority_fee_wei = out.gas_used * tip
    out.base_fee_burned_wei = out.gas_used * base_fee
    bal = state.fork.balances
    bal[tx.sender][ETH] -= out.priority_fee_wei + out.base_fee_burned_wei
    bal[coinbase][ETH] += out.priority_fee_wei
    bal[BURN][ETH] += out.base_fee_burned_wei
    state.nonces[tx.sender] = expected + 1
    if rec is not None:
        out.state_changes = rec.changes()
    return out


@dataclass
class ItemResult:
    item: str
    included: bool
    reason: str = ""
    traces: List[TxTrace] = field(default_factory=list)
    gas_used: int = 0
    value_wei: int = 0
    labels: List[str] = field(default_factory=list)


def execute_bundle(state: BuilderState, bundle: Bundle, base_fee: int, gas_remaining: int,
                   included: Set[str], coinbase: str = "builder", trace: bool = False,
                   policy: Optional[GuardPolicy] = None) -> ItemResult:
    """Execute a bundle atomically; on any failure state is restored and nothing is included."""
    res = ItemResult(item_id(bundle), False)
    res.labels = classify_bundle(bundle)
    policy = policy or GuardPolicy()
    for label in res.labels:
        if policy.action_for(label) is GuardAction.REJECT:
            res.reason = f"guard_rejected:{label}"
            return res
    if not bundle.txs:
        res.reason = "empty_bundle"
        return res
    dup = next((h for h in bundle.tx_hashes if h in included), None)
    if dup is not None:
        res.reason = f"conflict_already_included:{dup}"
        return res
    if bundle.gas_limit > gas_remaining:
        res.reason = "gas_limit_exceeds_remaining"
        return res
    snap = state.snapshot()
    used = 0
    for tx in bundle.txs:
        t = execute_tx(state, tx, base_fee, gas_remaining - used, coinbase, trace)
        if t.status is Status.INVALID or (t.status is not Status.SUCCESS
                                          and tx.tx_hash not in bundle.reverting_tx_hashes):
            state.restore(snap)
            res.reason = f"tx_{t.status.value}:{tx.tx_hash}:{t.error}"
            res.traces = []
            return res
        used += t.gas_used
        res.traces.append(t)
    if bundle.coinbase_transfer_wei:
        payer = bundle.searcher or bundle.txs[-1].sender
        rec = _Recorder(state, (), [payer, coinbase]) if trace else None
        state.fork.balances[payer][ETH] -= bundle.coinbase_transfer_wei
        state.fork.balances[coinbase][ETH] += bundle.coinbase_transfer_wei
        res.traces[-1].coinbase_transfer_wei = bundle.coinbase_transfer_wei
        if rec is not None:
            res.traces[-1].state_changes.extend(rec.changes())
    res.included = True
    res.gas_used = used
    res.value_wei = sum(t.priority_fee_wei for t in res.traces) + bundle.coinbase_transfer_wei
    return res


# ---------------------------------------------------------------------------
# Sealing
# ---------------------------------------------------------------------------


@dataclass
class SealedBlock:
    builder: str
    coinbase: str
    block_number: int
    base_fee: int
    gas_limit: int
    parent_root: str
    layout: List[Item]
    gas_used: int = 0
    priority_fees_wei: int = 0
    coinbase_transfers_wei: int = 0
    burned_wei: int = 0
    txs: List[TxTrace] = field(default_factory=list)
    included: List[str] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)
    flags: Dict[str, List[str]] = field(default_factory=dict)
    gas_rejections: int = 0
    unrealized_mev_wei: int = 0
    state_root: str = ""

    @property
    def value_wei(self) -> int:
        """What the block pays its builder: tips plus coinbase transfers."""
        return self.priority_fees_wei + self.coinbase_transfers_wei

    @property
    def block_hash(self) -> str:
        h = hashlib.sha256(f"{self.parent_root}|{self.block_number}|{self.coinbase}|{self.state_root}".encode())
        for iid in self.included:
            h.update(iid.encode())
        return "0x" + h.hexdigest()

    def bundles_included(self) -> List[str]:
        return [i for i in self.included if i.startswith("bundle:")]

    def summary(self) -> Dict[str, Any]:
        statuses: Dict[str, int] = defaultdict(int)
        for t in self.txs:
            statuses[t.status.value] += 1
        return {
            "builder": self.builder,
            "block_number": self.block_number,
            "block_hash": self.block_hash,
            "base_fee": self.base_fee,
            "gas_limit": self.gas_limit,
            "gas_used": self.gas_used,
            "gas_used_pct": round(100 * self.gas_used / self.gas_limit, 3) if self.gas_limit else 0.0,
            "value_wei": self.value_wei,
            "priority_fees_wei": self.priority_fees_wei,
            "coinbase_transfers_wei": self.coinbase_transfers_wei,
            "burned_wei": self.burned_wei,
            "txs": len(self.txs),
            "tx_status": dict(statuses),
            "items_included": len(self.included),
            "bundles_included": len(self.bundles_included()),
            "items_skipped": len(self.skipped),
            "gas_rejections": self.gas_rejections,
            "flags": {k: list(v) for k, v in self.flags.items()},
            "unrealized_mev_wei": self.unrealized_mev_wei,
            "state_root": self.state_root,
        }

    def log_lines(self) -> List[str]:
        lines = [
            f"block {self.block_number} builder={self.builder} hash={self.block_hash[:18]} "
            f"gas={self.gas_used}/{self.gas_limit} ({100 * self.gas_used / max(1, self.gas_limit):.3f}%) "
            f"base_fee={self.base_fee} value={self.value_wei} burned={self.burned_wei} "
            f"unrealized_mev={self.unrealized_mev_wei}"
        ]
        for t in self.txs:
            lines.append(
                f"  [{t.index:03d}] {t.tx_hash} {t.kind} {t.status.value} gas={t.gas_used} "
                f"cum={t.cumulative_gas} tip={t.tip_per_gas} fee={t.priority_fee_wei} "
                f"bribe={t.coinbase_transfer_wei} unrealized={t.unrealized_mev_wei} via={t.item}"
                + (f" err={t.error}" if t.error else "")
            )
            lines.extend(f"        {c}" for c in t.state_changes)
        for iid, reason in self.skipped.items():
            lines.append(f"  [skip] {iid}: {reason}")
        return lines


def seal_block(parent: BuilderState, layout: Sequence[Item], base_fee: int,
               gas_limit: int = BLOCK_GAS_LIMIT, builder: str = "builder", coinbase: Optional[str] = None,
               trace: bool = True, policy: Optional[GuardPolicy] = None, measure_unrealized: bool = False,
               native: str = "WETH", unrealized_grid: int = 8,
               compute_root: bool = True) -> Tuple[SealedBlock, BuilderState]:
    """Execute ``layout`` in order on a fork of ``parent``; returns the block and post-state.

    ``parent`` is never mutated. Items that cannot be included are recorded in
    ``skipped`` with a reason and the next item is tried.
    """
    gas_limit = clamp_gas_limit(gas_limit)
    coinbase = coinbase or builder
    state = parent.copy()
    block = SealedBlock(builder, coinbase, parent.block_number, base_fee, gas_limit,
                        parent.state_root() if trace else "", list(layout))
    included_hashes: Set[str] = set()
    for item in layout:
        iid = item_id(item)
        remaining = gas_limit - block.gas_used
        if remaining < GAS_TX_BASE:
            block.skipped[iid] = "block_full"
            continue
        if isinstance(item, Bundle):
            r = execute_bundle(state, item, base_fee, remaining, included_hashes, coinbase, trace, policy)
            if r.labels:
                block.flags[iid] = r.labels
            if not r.included:
                block.skipped[iid] = r.reason
                if r.reason == "gas_limit_exceeds_remaining":
                    block.gas_rejections += 1
                continue
            traces = r.traces
            block.coinbase_transfers_wei += item.coinbase_transfer_wei
        else:
            if item.tx_hash in included_hashes:
                block.skipped[iid] = "duplicate"
                continue
            unrealized = 0
            if measure_unrealized and item.kind is TxKind.SWAP and item.effective_tip(base_fee) is not None:
                cand = best_sandwich(state.pools, item, unrealized_grid, refine=4, native=native)
                if cand is not None:
                    unrealized = max(0, native_value(state.pools, cand.token_in, cand.profit, native))
            t = execute_tx(state, item, base_fee, remaining, coinbase, trace)
            if t.status is Status.INVALID:
                block.skipped[iid] = t.error
                if t.error == "gas_limit_exceeds_remaining":
                    block.gas_rejections += 1
                continue
            t.unrealized_mev_wei = unrealized
            block.unrealized_mev_wei += unrealized
            traces = [t]
        for t in traces:
            block.gas_used += t.gas_used
            block.priority_fees_wei += t.priority_fee_wei
            block.burned_wei += t.base_fee_burned_wei
            t.index = len(block.txs)
            t.cumulative_gas = block.gas_used
            t.item = iid
            block.txs.append(t)
            included_hashes.add(t.tx_hash)
        block.included.append(iid)
    if block.gas_used > gas_limit:  # pragma: no cover - guarded per item above
        raise AssertionError(f"gas overflow {block.gas_used} > {gas_limit}")
    if compute_root:
        block.state_root = state.state_root()
    if trace and log.isEnabledFor(logging.DEBUG):
        for line in block.log_lines():
            log.debug(line)
    return block, state


# ---------------------------------------------------------------------------
# Transaction pool with replacement and eviction rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AddResult:
    status: str
    evicted: Tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.status in ("added", "replaced")


class TxPool:
    """Pending pool keyed by (sender, nonce).

    * Replacement of the same (sender, nonce) needs a >= 10% bump on both the
      fee cap and the priority fee.
    * When full, the cheapest transaction (lowest effective tip at the
      current base fee, then lowest fee cap) is evicted together with that
      sender's higher nonces, which would otherwise be stuck behind a gap; an
      incoming transaction no better than the cheapest is rejected.
    * ``pending()`` returns only nonce-contiguous (executable) transactions;
      ``queued()`` returns the rest.
    """

    def __init__(self, capacity: int = 4096, block_gas_limit: int = BLOCK_GAS_LIMIT) -> None:
        if capacity < 1:
            raise ConfigError("pool capacity must be positive")
        self.capacity = capacity
        self.block_gas_limit = clamp_gas_limit(block_gas_limit)
        self.by_sender: Dict[str, Dict[int, BuilderTx]] = defaultdict(dict)
        self.by_hash: Dict[str, BuilderTx] = {}

    def __len__(self) -> int:
        return len(self.by_hash)

    def __contains__(self, tx_hash: str) -> bool:
        return tx_hash in self.by_hash

    @staticmethod
    def _price(tx: BuilderTx, base_fee: int) -> Tuple[int, int]:
        tip = tx.effective_tip(base_fee)
        return (-1 if tip is None else tip, tx.max_fee_per_gas)

    def _remove(self, tx: BuilderTx) -> None:
        self.by_hash.pop(tx.tx_hash, None)
        slot = self.by_sender.get(tx.sender)
        if slot is not None and slot.get(tx.nonce) is tx:
            del slot[tx.nonce]
            if not slot:
                del self.by_sender[tx.sender]

    def _evict_from(self, tx: BuilderTx) -> List[str]:
        evicted = []
        for nonce in sorted(n for n in self.by_sender.get(tx.sender, {}) if n >= tx.nonce):
            victim = self.by_sender[tx.sender][nonce]
            self._remove(victim)
            evicted.append(victim.tx_hash)
        return evicted

    def add(self, tx: BuilderTx, base_fee: int, account_nonce: int = 0) -> AddResult:
        if tx.tx_hash in self.by_hash:
            return AddResult("rejected_known")
        if tx.gas_limit > self.block_gas_limit:
            return AddResult("rejected_gas_limit")
        if tx.gas_limit < GAS_TX_BASE:
            return AddResult("rejected_intrinsic_gas")
        if tx.max_priority_fee_per_gas > tx.max_fee_per_gas:
            return AddResult("rejected_tip_above_cap")
        if tx.nonce < account_nonce:
            return AddResult("rejected_nonce_too_low")
        existing = self.by_sender.get(tx.sender, {}).get(tx.nonce)
        if existing is not None:
            bump = 100 + REPLACEMENT_BUMP_PCT
            if (tx.max_fee_per_gas * 100 < existing.max_fee_per_gas * bump
                    or tx.max_priority_fee_per_gas * 100 < existing.max_priority_fee_per_gas * bump):
                return AddResult("rejected_replacement_underpriced")
            self._remove(existing)
            self._insert(tx)
            return AddResult("replaced", (existing.tx_hash,))
        evicted: List[str] = []
        if len(self.by_hash) >= self.capacity:
            cheapest = min(self.by_hash.values(), key=lambda t: (self._price(t, base_fee), -t.nonce, t.tx_hash))
            if self._price(tx, base_fee) <= self._price(cheapest, base_fee):
                return AddResult("rejected_pool_full")
            evicted = self._evict_from(cheapest)
        self._insert(tx)
        return AddResult("added", tuple(evicted))

    def _insert(self, tx: BuilderTx) -> None:
        self.by_sender[tx.sender][tx.nonce] = tx
        self.by_hash[tx.tx_hash] = tx

    def pending(self, nonces: Mapping[str, int]) -> List[BuilderTx]:
        out: List[BuilderTx] = []
        for sender, slots in self.by_sender.items():
            n = nonces.get(sender, 0)
            while n in slots:
                out.append(slots[n])
                n += 1
        return out

    def queued(self, nonces: Mapping[str, int]) -> List[BuilderTx]:
        ready = {t.tx_hash for t in self.pending(nonces)}
        return [t for t in self.by_hash.values() if t.tx_hash not in ready]

    def prune(self, nonces: Mapping[str, int]) -> List[str]:
        """Drop transactions whose nonce is already used (included or replaced on chain)."""
        stale = [t for t in self.by_hash.values() if t.nonce < nonces.get(t.sender, 0)]
        for t in stale:
            self._remove(t)
        return [t.tx_hash for t in stale]


def price_and_nonce_order(txs: Iterable[BuilderTx], base_fee: int,
                          nonces: Mapping[str, int]) -> List[BuilderTx]:
    """Highest effective tip first while keeping each sender's nonces contiguous and in order."""
    per_sender: Dict[str, Dict[int, BuilderTx]] = defaultdict(dict)
    for tx in txs:
        if tx.effective_tip(base_fee) is None:
            continue
        cur = per_sender[tx.sender].get(tx.nonce)
        if cur is None or (tx.effective_tip(base_fee), tx.tx_hash) > (cur.effective_tip(base_fee), cur.tx_hash):
            per_sender[tx.sender][tx.nonce] = tx
    heap: List[Tuple[int, str, str, int]] = []
    for sender, slots in per_sender.items():
        n = nonces.get(sender, 0)
        if n in slots:
            t = slots[n]
            heapq.heappush(heap, (-(t.effective_tip(base_fee) or 0), t.tx_hash, sender, n))
    out: List[BuilderTx] = []
    while heap:
        _, _, sender, n = heapq.heappop(heap)
        out.append(per_sender[sender][n])
        nxt = per_sender[sender].get(n + 1)
        if nxt is not None:
            heapq.heappush(heap, (-(nxt.effective_tip(base_fee) or 0), nxt.tx_hash, sender, n + 1))
    return out


# ---------------------------------------------------------------------------
# Sandwich search over single- and multi-hop victims
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SandwichCandidate:
    victim: str
    hop: int
    pool_id: str
    token_in: str
    token_out: str
    frontrun_in: int
    frontrun_out: int
    victim_out: int
    backrun_out: int
    multi_hop: bool

    @property
    def profit(self) -> int:
        return self.backrun_out - self.frontrun_in


def _route_hops(pools: Mapping[str, Pool], tx: BuilderTx) -> Optional[List[Tuple[str, str, str]]]:
    route = tx.route
    if not route or len(set(route)) != len(route):
        return None
    hops: List[Tuple[str, str, str]] = []
    token = tx.token_in
    for pid in route:
        pool = pools.get(pid)
        if pool is None:
            return None
        try:
            _, out = _direction(pool, token)
        except MASEError:
            return None
        hops.append((pid, token, out))
        token = out
    return hops


def simulate_route_sandwich(pools: Mapping[str, Pool], victim: BuilderTx, hop: int,
                            frontrun_in: int) -> Optional[Tuple[int, int, int]]:
    """(front_out, victim_out, back_out) for a sandwich on hop ``hop`` of the victim's route."""
    hops = _route_hops(pools, victim)
    if hops is None or not 0 <= hop < len(hops) or frontrun_in <= 0:
        return None
    work = {pid: clone_pool(pools[pid]) for pid, _, _ in hops}
    pid, tin, _ = hops[hop]
    target = work[pid]
    zfo, _ = _direction(target, tin)
    front = _apply_full(target, frontrun_in, zfo)
    if front is None:
        return None
    amount = victim.amount_in
    for vpid, vtin, _ in hops:
        pool = work[vpid]
        res = _apply_full(pool, amount, _direction(pool, vtin)[0])
        if res is None:
            return None
        amount = res.amount_out
    if amount < victim.amount_out_min:
        return None
    back = _apply_full(target, front.amount_out, not zfo)
    if back is None:
        return None
    return front.amount_out, amount, back.amount_out


def best_sandwich(pools: Mapping[str, Pool], victim: BuilderTx, grid: int = 16, refine: int = 8,
                  max_frontrun_bps: int = 2_000, native: str = "WETH") -> Optional[SandwichCandidate]:
    """Best sandwich over every hop of the victim's route (grid + ternary refine), or None.

    Hops are compared by profit converted to ``native`` at spot price.
    """
    if victim.kind is not TxKind.SWAP or grid <= 0:
        return None
    hops = _route_hops(pools, victim)
    if hops is None:
        return None
    best: Optional[SandwichCandidate] = None
    best_native = 0
    for h, (pid, tin, tout) in enumerate(hops):
        pool = pools[pid]
        zfo, _ = _direction(pool, tin)
        hi = input_reserve_scale(pool, zfo) * max_frontrun_bps // 10_000
        lo = max(1, hi // 100_000)
        cache: Dict[int, Optional[Tuple[int, int, int]]] = {}

        def score(a: int) -> float:
            if a not in cache:
                cache[a] = simulate_route_sandwich(pools, victim, h, a)
            r = cache[a]
            return float("-inf") if r is None else r[2] - a

        sizes = geometric_grid(lo, hi, grid)
        if not sizes:
            continue
        scores = [score(a) for a in sizes]
        i = max(range(len(sizes)), key=lambda k: (scores[k], -k))
        if scores[i] == float("-inf"):
            continue
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
        profit, neg_a = max((r[2] - a, -a) for a, r in cache.items() if r is not None)
        if profit <= 0:
            continue
        a = -neg_a
        fo, vo, bo = cache[a]  # type: ignore[misc]
        cand = SandwichCandidate(victim.tx_hash, h, pid, tin, tout, a, fo, vo, bo, len(hops) > 1)
        value = native_value(pools, tin, cand.profit, native)
        if best is None or value > best_native:
            best, best_native = cand, value
    return best


def native_value(pools: Mapping[str, Pool], token: str, amount: int, native: str = "WETH") -> int:
    """Convert ``amount`` of ``token`` to native units at the first direct pool's spot price."""
    if token == native or amount == 0:
        return amount
    for pid in sorted(pools):
        pool = pools[pid]
        if set(pool.tokens()) == {token, native}:
            zfo, _ = _direction(pool, token)
            price: Fraction = pool.spot_price(zfo)
            return int(amount * price)
    return 0


def build_sandwich_bundles(state: BuilderState, txs: Iterable[BuilderTx], base_fee: int,
                           native: str = "WETH", grid: int = 16, refine: int = 8,
                           bribe_bps: int = 8_000, max_frontrun_bps: int = 2_000,
                           tip_wei: int = GWEI) -> List[Bundle]:
    """One profitable sandwich bundle per victim swap (best hop by native value), bribe = share of net."""
    if not 0 <= bribe_bps <= 10_000:
        raise ConfigError("bribe_bps must be within [0, 10000]")
    bundles: List[Bundle] = []
    for victim in txs:
        if victim.kind is not TxKind.SWAP or victim.effective_tip(base_fee) is None:
            continue
        cand = best_sandwich(state.pools, victim, grid, refine, max_frontrun_bps, native)
        if cand is None:
            continue
        profit_native = native_value(state.pools, cand.token_in, cand.profit, native)
        pool = state.pools[cand.pool_id]
        leg_gas = estimate_swap_gas([pool], 0)
        gas_limit = leg_gas * 3 // 2 + 40_000  # headroom for tick crossings
        fee_cap = base_fee * 2 + tip_wei
        gas_cost = 2 * leg_gas * (base_fee + tip_wei)
        net = profit_native - gas_cost
        if net <= 0:
            continue
        searcher = f"searcher:{victim.tx_hash[:14]}"
        nonce = state.nonce(searcher)
        front = BuilderTx(f"{victim.tx_hash}:front", searcher, nonce, gas_limit, fee_cap, tip_wei, TxKind.SWAP,
                          {"route": [cand.pool_id], "token_in": cand.token_in, "amount_in": cand.frontrun_in,
                           "amount_out_min": cand.frontrun_out})
        back = BuilderTx(f"{victim.tx_hash}:back", searcher, nonce + 1, gas_limit, fee_cap, tip_wei, TxKind.SWAP,
                         {"route": [cand.pool_id], "token_in": cand.token_out, "amount_in": cand.frontrun_out,
                          "amount_out_min": cand.frontrun_in + 1})
        bundles.append(Bundle(
            bundle_id=f"sw:{victim.tx_hash}",
            txs=(front, victim, back),
            coinbase_transfer_wei=net * bribe_bps // 10_000,
            searcher=searcher,
            target_tx=victim.tx_hash,
            meta={"hop": cand.hop, "pool_id": cand.pool_id, "multi_hop": cand.multi_hop,
                  "profit": cand.profit, "profit_token": cand.token_in, "profit_native": profit_native},
        ))
    return bundles


def build_jit_bundle(state: BuilderState, victim: BuilderTx, lp: str, liquidity: int,
                     base_fee: int, width_spacings: int = 1, tip_wei: int = GWEI) -> Optional[Bundle]:
    """Mint a tight range around the current tick before ``victim`` and burn it after."""
    for pid in victim.route:
        pool = state.pools.get(pid)
        if isinstance(pool, ConcentratedPool):
            s = pool.tick_spacing
            lower = (pool.tick // s) * s - width_spacings * s
            upper = (pool.tick // s) * s + (width_spacings + 1) * s
            n = state.nonce(lp)
            body = {"pool_id": pid, "tick_lower": lower, "tick_upper": upper, "liquidity": liquidity}
            gas = GAS_TX_BASE + GAS_MODIFY_LIQUIDITY
            mint = BuilderTx(f"{victim.tx_hash}:jit-mint", lp, n, gas, base_fee * 2 + tip_wei, tip_wei,
                             TxKind.MINT, body)
            burn = BuilderTx(f"{victim.tx_hash}:jit-burn", lp, n + 1, gas, base_fee * 2 + tip_wei, tip_wei,
                             TxKind.BURN, body)
            return Bundle(f"jit:{victim.tx_hash}", (mint, victim, burn), searcher=lp, target_tx=victim.tx_hash)
    return None


# ---------------------------------------------------------------------------
# Mempool generation (EIP-1559, nonces, multi-hop, LP and transfers)
# ---------------------------------------------------------------------------


def _multi_hop_route(pools: Sequence[Pool], first: Pool, token_in: str, rng: random.Random) -> List[str]:
    _, mid = _direction(first, token_in)
    nexts = [p for p in pools if p.pool_id != first.pool_id and mid in p.tokens()]
    if not nexts:
        return [first.pool_id]
    return [first.pool_id, nexts[rng.randrange(len(nexts))].pool_id]


def generate_builder_mempool(state: BuilderState, count: int, seed: int, base_fee: int,
                             block: Optional[int] = None) -> List[BuilderTx]:
    """Deterministic EIP-1559 flow: swaps (~30% two-hop), transfers, LP mints and some underpriced txs.

    Senders can submit several sequential nonces; a few skip a nonce and so
    stay queued. Swap gas limits are drawn around the estimate, so some run
    out of gas.
    """
    pools = sorted(state.pools.values(), key=lambda p: p.pool_id)
    if not pools or count <= 0:
        return []
    block = state.block_number if block is None else block
    rng = random.Random(f"builder-mempool:{seed}:{block}")
    next_nonce: Dict[str, int] = {}
    out: List[BuilderTx] = []
    i = 0
    while len(out) < count and i < count * 4:
        i += 1
        sender = f"user{rng.randrange(max(4, count // 2)):03d}"
        nonce = next_nonce.get(sender, state.nonce(sender))
        if rng.random() < 0.04:
            nonce += 1  # nonce gap -> queued
        tip = rng.randrange(1, 8) * GWEI // 2
        fee_cap = base_fee * 2 + tip if rng.random() > 0.05 else max(0, base_fee - GWEI)
        roll = rng.random()
        tx_hash = f"0x{block:06x}{i:06x}{rng.getrandbits(96):024x}"
        if roll < 0.12:
            tx = BuilderTx(tx_hash, sender, nonce, GAS_TRANSFER, fee_cap, tip, TxKind.TRANSFER,
                           {"to": f"user{rng.randrange(1000):03d}", "value": rng.randrange(1, 10) * 10**17})
        elif roll < 0.20:
            cls = [p for p in pools if isinstance(p, ConcentratedPool)]
            if not cls:
                continue
            pool = cls[rng.randrange(len(cls))]
            s = pool.tick_spacing
            lower = (pool.tick // s - rng.randrange(1, 20)) * s
            upper = (pool.tick // s + rng.randrange(1, 20)) * s
            tx = BuilderTx(tx_hash, sender, nonce, GAS_TX_BASE + GAS_MODIFY_LIQUIDITY + 20_000, fee_cap, tip,
                           TxKind.MINT, {"pool_id": pool.pool_id, "tick_lower": lower, "tick_upper": upper,
                                         "liquidity": max(1, pool.liquidity // rng.randrange(50, 500))})
        else:
            pool = pools[rng.randrange(len(pools))]
            token_in = pool.tokens()[rng.randrange(2)]
            route = _multi_hop_route(pools, pool, token_in, rng) if rng.random() < 0.3 else [pool.pool_id]
            zfo, _ = _direction(pool, token_in)
            size_bps = 10 ** rng.uniform(0, 2.2)
            amount = max(1, int(input_reserve_scale(pool, zfo) * size_bps / 10_000))
            tolerance = rng.choice((10, 30, 50, 100, 300, 500, 1000))
            probe = BuilderTx("probe", sender, nonce, BLOCK_GAS_LIMIT, fee_cap, tip, TxKind.SWAP,
                              {"route": route, "token_in": token_in, "amount_in": amount})
            try:
                hops, ticks = _quote_hops(state, probe)
            except SwapReverted:
                continue
            gas_est = estimate_swap_gas([state.pools[p] for p in route], ticks)
            tx = BuilderTx(tx_hash, sender, nonce, int(gas_est * rng.choice((0.8, 1.1, 1.25, 1.5))), fee_cap,
                           tip, TxKind.SWAP,
                           {"route": route, "token_in": token_in, "amount_in": amount,
                            "amount_out_min": min_amount_out(hops[-1][3].amount_out, tolerance), "tolerance_bps": tolerance})
        out.append(tx)
        next_nonce[sender] = nonce + 1
    return out


# ---------------------------------------------------------------------------
# Layout fuzzer
# ---------------------------------------------------------------------------


MUTATIONS = ("swap", "move", "reverse", "drop", "restore", "promote_bundle")


def repair_nonce_order(layout: List[int], items: Sequence[Item]) -> List[int]:
    """Re-sort each sender's public transactions by nonce within the slots they occupy."""
    slots: Dict[str, List[int]] = defaultdict(list)
    for pos, idx in enumerate(layout):
        it = items[idx]
        if isinstance(it, BuilderTx):
            slots[it.sender].append(pos)
    out = list(layout)
    for positions in slots.values():
        if len(positions) < 2:
            continue
        ordered = sorted((layout[p] for p in positions), key=lambda i: (items[i].nonce, items[i].tx_hash))
        for p, idx in zip(positions, ordered):
            out[p] = idx
    return out


def mutate_layout(layout: List[int], dropped: List[int], items: Sequence[Item],
                  rng: random.Random) -> Tuple[List[int], List[int], str]:
    lay, drop = list(layout), list(dropped)
    op = MUTATIONS[rng.randrange(len(MUTATIONS))]
    n = len(lay)
    if op == "restore" and not drop:
        op = "swap"
    if op in ("swap", "move", "reverse", "drop") and n < 2:
        op = "restore" if drop else "noop"
    if op == "swap":
        i, j = rng.sample(range(n), 2)
        lay[i], lay[j] = lay[j], lay[i]
    elif op == "move":
        i, j = rng.randrange(n), rng.randrange(n)
        lay.insert(j, lay.pop(i))
    elif op == "reverse":
        i, j = sorted(rng.sample(range(n + 1), 2))
        lay[i:j] = lay[i:j][::-1]
    elif op == "drop":
        drop.append(lay.pop(rng.randrange(n)))
    elif op == "restore":
        lay.insert(rng.randrange(n + 1), drop.pop(rng.randrange(len(drop))))
    elif op == "promote_bundle":
        bundle_pos = [p for p, idx in enumerate(lay) if isinstance(items[idx], Bundle)]
        if bundle_pos:
            p = bundle_pos[rng.randrange(len(bundle_pos))]
            lay.insert(rng.randrange(min(p, 3) + 1), lay.pop(p))
    return repair_nonce_order(lay, items), drop, op


def _layout_value(state: BuilderState, items: Sequence[Item], layout: Sequence[int], base_fee: int,
                  gas_limit: int, policy: Optional[GuardPolicy]) -> int:
    block, _ = seal_block(state, [items[i] for i in layout], base_fee, gas_limit, trace=False, policy=policy,
                          compute_root=False)
    return block.value_wei


def _fuzz_shard(task: Tuple[Any, ...]) -> Dict[str, Any]:
    payload, items, start, base_fee, gas_limit, iterations, seed, policy = task
    state = BuilderState.from_payload(payload)
    rng = random.Random(f"layout-fuzz:{seed}")
    in_start = set(start)
    best, dropped = list(start), [i for i in range(len(items)) if i not in in_start]
    best_value = _layout_value(state, items, best, base_fee, gas_limit, policy)
    ops: Dict[str, int] = defaultdict(int)
    improvements = 0
    started = time.perf_counter()
    for _ in range(iterations):
        cand, cand_drop, op = mutate_layout(best, dropped, items, rng)
        ops[op] += 1
        value = _layout_value(state, items, cand, base_fee, gas_limit, policy)
        if value > best_value:
            improvements += 1
        if value >= best_value:
            best, dropped, best_value = cand, cand_drop, value
    return {"layout": best, "value": best_value, "evaluated": iterations + 1, "ops": dict(ops),
            "improvements": improvements, "cpu_s": time.perf_counter() - started}


@dataclass
class FuzzResult:
    layout: List[int]
    value_wei: int
    baseline_value_wei: int
    evaluated: int
    improvements: int
    ops: Dict[str, int]
    shards: int
    elapsed_s: float
    best_shard: int

    @property
    def layouts_per_s(self) -> float:
        return self.evaluated / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def gain_wei(self) -> int:
        return self.value_wei - self.baseline_value_wei

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value_wei": self.value_wei,
            "baseline_value_wei": self.baseline_value_wei,
            "gain_wei": self.gain_wei,
            "layouts_evaluated": self.evaluated,
            "improvements": self.improvements,
            "mutations": dict(self.ops),
            "shards": self.shards,
            "best_shard": self.best_shard,
            "elapsed_s": round(self.elapsed_s, 4),
            "layouts_per_s": round(self.layouts_per_s, 1),
        }


def fuzz_block_layout(state: BuilderState, items: Sequence[Item], start: Optional[Sequence[int]] = None,
                      base_fee: int = 0, gas_limit: int = BLOCK_GAS_LIMIT, iterations: int = 1024,
                      seed: int = 0, executor: Optional[WorkerPool] = None, shards: int = FUZZ_SHARDS,
                      policy: Optional[GuardPolicy] = None) -> FuzzResult:
    """Hill-climb block layouts across ``shards`` seeded shards on the capped worker pool.

    Every shard starts from ``start`` (default: ``items`` in order) and only
    accepts layouts worth at least as much, so the result is never below the
    starting layout. Ties between shards go to the lowest shard index, which
    keeps results independent of the worker count.
    """
    items = list(items)
    start = list(range(len(items)) if start is None else start)
    gas_limit = clamp_gas_limit(gas_limit)
    shards = max(1, shards)
    per, extra = divmod(max(0, iterations), shards)
    payload = state.to_payload()
    tasks = [(payload, items, start, base_fee, gas_limit, per + (1 if s < extra else 0), seed * 1_000 + s, policy)
             for s in range(shards)]
    started = time.perf_counter()
    parts = executor.map(_fuzz_shard, tasks) if executor is not None else [_fuzz_shard(t) for t in tasks]
    elapsed = time.perf_counter() - started
    baseline = _layout_value(state, items, start, base_fee, gas_limit, policy)
    best_i = max(range(len(parts)), key=lambda i: (parts[i]["value"], -i))
    ops: Dict[str, int] = defaultdict(int)
    for p in parts:
        for k, v in p["ops"].items():
            ops[k] += v
    return FuzzResult(
        layout=parts[best_i]["layout"],
        value_wei=parts[best_i]["value"],
        baseline_value_wei=baseline,
        evaluated=sum(p["evaluated"] for p in parts),
        improvements=sum(p["improvements"] for p in parts),
        ops=dict(ops),
        shards=shards,
        elapsed_s=elapsed,
        best_shard=best_i,
    )


# ---------------------------------------------------------------------------
# Builders, relays, proposer: the MEV-Boost auction
# ---------------------------------------------------------------------------


class Strategy(str, Enum):
    FEE_ONLY = "fee_only"  # vanilla local block: public txs by tip, no bundles
    GREEDY = "greedy"  # bundles by value/gas, then public txs
    FUZZ = "fuzz"  # greedy layout refined by the layout fuzzer


@dataclass
class BlockBuilder:
    name: str
    strategy: Strategy = Strategy.GREEDY
    margin_bps: int = 500
    fuzz_iterations: int = 0
    policy: GuardPolicy = field(default_factory=GuardPolicy)
    min_bundle_value_wei: int = 1

    def __post_init__(self) -> None:
        self.strategy = Strategy(self.strategy)
        if not 0 <= self.margin_bps <= 10_000:
            raise ConfigError("margin_bps must be within [0, 10000]")

    @property
    def coinbase(self) -> str:
        return f"coinbase:{self.name}"

    def order(self, parent: BuilderState, bundles: Sequence[Bundle], txs: Sequence[BuilderTx],
              base_fee: int, gas_limit: int) -> Tuple[List[Item], Dict[str, str]]:
        """Greedy layout and per-bundle admission notes."""
        notes: Dict[str, str] = {}
        public = price_and_nonce_order(txs, base_fee, parent.nonces)
        if self.strategy is Strategy.FEE_ONLY:
            return list(public), notes
        scored: List[Tuple[Fraction, str, Bundle]] = []
        for b in bundles:
            block, _ = seal_block(parent, [b], base_fee, gas_limit, self.name, self.coinbase,
                                  trace=False, policy=self.policy)
            if not block.included:
                notes[item_id(b)] = block.skipped.get(item_id(b), "not_included")
                continue
            if block.value_wei < self.min_bundle_value_wei:
                notes[item_id(b)] = "below_min_value"
                continue
            scored.append((Fraction(block.value_wei, max(1, block.gas_used)), b.bundle_id, b))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [b for _, _, b in scored] + list(public), notes

    def build(self, parent: BuilderState, bundles: Sequence[Bundle], txs: Sequence[BuilderTx], base_fee: int,
              gas_limit: int = BLOCK_GAS_LIMIT, executor: Optional[WorkerPool] = None, seed: int = 0,
              measure_unrealized: bool = False, native: str = "WETH"
              ) -> Tuple[SealedBlock, BuilderState, Dict[str, Any]]:
        gas_limit = clamp_gas_limit(gas_limit)
        layout, notes = self.order(parent, bundles, txs, base_fee, gas_limit)
        info: Dict[str, Any] = {"bundle_notes": notes}
        if self.strategy is Strategy.FUZZ and self.fuzz_iterations > 0 and len(layout) > 1:
            in_layout = {item_id(x) for x in layout}
            # Bundles that failed top-of-block simulation may still land after a reorder.
            items = layout + [b for b in bundles if item_id(b) not in in_layout
                              and not notes.get(item_id(b), "").startswith("guard_rejected")]
            start = list(range(len(layout)))
            fz = fuzz_block_layout(parent, items, start, base_fee, gas_limit, self.fuzz_iterations, seed,
                                   executor, policy=self.policy)
            layout = [items[i] for i in fz.layout]
            info["fuzz"] = fz.to_dict()
        block, post = seal_block(parent, layout, base_fee, gas_limit, self.name, self.coinbase, trace=True,
                                 policy=self.policy, measure_unrealized=measure_unrealized, native=native)
        return block, post, info

    def bid_for(self, block: SealedBlock, slot: int) -> "BuilderBid":
        value = block.value_wei * (10_000 - self.margin_bps) // 10_000
        return BuilderBid(self.name, slot, block.block_hash, value, block.value_wei, block.gas_used, block.gas_limit)


@dataclass(frozen=True)
class BuilderBid:
    builder: str
    slot: int
    block_hash: str
    value_wei: int  # paid to the proposer
    block_value_wei: int
    gas_used: int
    gas_limit: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "builder": self.builder,
            "slot": self.slot,
            "block_hash": self.block_hash,
            "value_wei": self.value_wei,
            "block_value_wei": self.block_value_wei,
            "gas_used": self.gas_used,
            "gas_limit": self.gas_limit,
        }


class Relay:
    """Validates builder submissions by re-executing them from the parent state."""

    def __init__(self, name: str = "relay") -> None:
        self.name = name
        self.accepted: List[Tuple[BuilderBid, SealedBlock]] = []
        self.rejected: List[Tuple[BuilderBid, str]] = []

    def reset(self) -> None:
        self.accepted.clear()
        self.rejected.clear()

    def validate(self, bid: BuilderBid, block: SealedBlock, parent: BuilderState, slot: int,
                 base_fee: int, policy: Optional[GuardPolicy] = None) -> Optional[str]:
        if bid.slot != slot:
            return "wrong_slot"
        if bid.gas_limit > BLOCK_GAS_LIMIT or block.gas_limit > BLOCK_GAS_LIMIT:
            return "gas_limit_above_cap"
        if bid.gas_used > bid.gas_limit:
            return "gas_used_exceeds_limit"
        if bid.value_wei > bid.block_value_wei:
            return "bid_exceeds_block_value"
        replay, _ = seal_block(parent, block.layout, base_fee, block.gas_limit, block.builder, block.coinbase,
                               trace=False, policy=policy)
        if replay.gas_used != bid.gas_used:
            return "gas_used_mismatch"
        if replay.value_wei != bid.block_value_wei:
            return "block_value_mismatch"
        replay.parent_root = block.parent_root
        if replay.block_hash != bid.block_hash:
            return "block_hash_mismatch"
        return None

    def submit(self, bid: BuilderBid, block: SealedBlock, parent: BuilderState, slot: int, base_fee: int,
               policy: Optional[GuardPolicy] = None) -> Optional[str]:
        reason = self.validate(bid, block, parent, slot, base_fee, policy)
        if reason is None:
            self.accepted.append((bid, block))
        else:
            self.rejected.append((bid, reason))
        return reason

    def best(self) -> Optional[Tuple[BuilderBid, SealedBlock]]:
        if not self.accepted:
            return None
        return max(self.accepted, key=lambda bb: (bb[0].value_wei, bb[0].builder))


@dataclass
class Proposer:
    name: str = "validator"
    min_bid_wei: int = 0

    @property
    def fee_recipient(self) -> str:
        return f"proposer:{self.name}"


@dataclass
class AuctionResult:
    slot: int
    winner: str
    source: str  # "relay:<name>" or "local"
    proposer_payment_wei: int
    block: SealedBlock
    post_state: BuilderState
    bids: List[BuilderBid]
    rejected: List[Dict[str, str]]
    builder_info: Dict[str, Dict[str, Any]]
    local_value_wei: int

    def summary(self) -> Dict[str, Any]:
        b = self.block
        sandwiches = [f for f in b.flags.items() if f[0] in b.included and
                      any(lbl in ("sandwich", "multi_hop_sandwich") for lbl in f[1])]
        return {
            "slot": self.slot,
            "winner": self.winner,
            "source": self.source,
            "proposer_payment_wei": self.proposer_payment_wei,
            "local_value_wei": self.local_value_wei,
            "builder_profit_wei": b.value_wei - self.proposer_payment_wei if self.source != "local" else 0,
            "bids": {bid.builder: bid.value_wei for bid in self.bids},
            "rejected": list(self.rejected),
            "sandwiches_included": len(sandwiches),
            "multi_hop_sandwiches": sum(1 for _, lbls in sandwiches if "multi_hop_sandwich" in lbls),
            "jit_bundles_flagged": sum(1 for lbls in b.flags.values() if "jit_liquidity" in lbls),
            "jit_bundles_included": sum(1 for iid, lbls in b.flags.items()
                                        if "jit_liquidity" in lbls and iid in b.included),
            "fuzz": {k: v["fuzz"] for k, v in self.builder_info.items() if "fuzz" in v},
            "block": b.summary(),
        }


class MevBoostAuction:
    def __init__(self, builders: Sequence[BlockBuilder], relays: Optional[Sequence[Relay]] = None,
                 proposer: Optional[Proposer] = None) -> None:
        if not builders:
            raise ConfigError("at least one builder is required")
        names = [b.name for b in builders]
        if len(set(names)) != len(names):
            raise ConfigError("builder names must be unique")
        self.builders = list(builders)
        self.relays = list(relays or [Relay()])
        self.proposer = proposer or Proposer()

    def run_slot(self, parent: BuilderState, bundles: Sequence[Bundle], txs: Sequence[BuilderTx],
                 base_fee: int, slot: int, gas_limit: int = BLOCK_GAS_LIMIT,
                 executor: Optional[WorkerPool] = None, seed: int = 0, measure_unrealized: bool = False,
                 native: str = "WETH", extra_bundles: Optional[Mapping[str, Sequence[Bundle]]] = None
                 ) -> AuctionResult:
        """Run one slot. ``extra_bundles`` maps builder name -> bundles only that builder received
        (Layer 05: private order flow decrypted from relay envelopes)."""
        gas_limit = clamp_gas_limit(gas_limit)
        extra_bundles = extra_bundles or {}
        for r in self.relays:
            r.reset()
        bids: List[BuilderBid] = []
        built: Dict[str, Tuple[SealedBlock, BuilderState]] = {}
        info: Dict[str, Dict[str, Any]] = {}
        for builder in self.builders:
            own_bundles = list(bundles) + list(extra_bundles.get(builder.name, ()))
            block, post, binfo = builder.build(parent, own_bundles, txs, base_fee, gas_limit, executor,
                                               seed=seed * 7919 + slot, measure_unrealized=measure_unrealized,
                                               native=native)
            built[builder.name] = (block, post)
            info[builder.name] = binfo
            bid = builder.bid_for(block, slot)
            bids.append(bid)
            for relay in self.relays:
                relay.submit(bid, block, parent, slot, base_fee, builder.policy)
        rejected = [{"relay": r.name, "builder": b.builder, "reason": why} for r in self.relays for b, why in r.rejected]
        local_builder = BlockBuilder("local", Strategy.FEE_ONLY, margin_bps=0)
        local_block, local_post, _ = local_builder.build(parent, (), txs, base_fee, gas_limit, native=native)
        best: Optional[Tuple[BuilderBid, SealedBlock, str]] = None
        for relay in self.relays:
            top = relay.best()
            if top is not None and (best is None or (top[0].value_wei, top[0].builder) > (best[0].value_wei,
                                                                                          best[0].builder)):
                best = (top[0], top[1], relay.name)
        if best is not None and best[0].value_wei >= max(self.proposer.min_bid_wei, 1) \
                and best[0].value_wei >= local_block.value_wei:
            bid, block, relay_name = best
            post = built[bid.builder][1]
            payment = bid.value_wei
            post.fork.balances[block.coinbase][ETH] -= payment
            post.fork.balances[self.proposer.fee_recipient][ETH] += payment
            return AuctionResult(slot, bid.builder, f"relay:{relay_name}", payment, block, post, bids, rejected,
                                 info, local_block.value_wei)
        payment = local_block.value_wei
        local_post.fork.balances[local_block.coinbase][ETH] -= payment
        local_post.fork.balances[self.proposer.fee_recipient][ETH] += payment
        return AuctionResult(slot, "local", "local", payment, local_block, local_post, bids, rejected, info,
                             local_block.value_wei)


# ---------------------------------------------------------------------------
# Multi-slot PBS simulation (wired into Brutal Aggressive Simulation)
# ---------------------------------------------------------------------------


def default_builders(config: SimulationConfig) -> List[BlockBuilder]:
    builders = [
        BlockBuilder("vanilla", Strategy.FEE_ONLY, margin_bps=0),
        BlockBuilder("greedy", Strategy.GREEDY, margin_bps=config.builder_margin_bps),
    ]
    if config.layout_fuzz_iterations > 0:
        builders.append(BlockBuilder("fuzzer", Strategy.FUZZ, margin_bps=config.builder_margin_bps,
                                     fuzz_iterations=config.layout_fuzz_iterations))
    return builders


@dataclass
class PBSReport:
    slots: List[AuctionResult]
    pool_stats: Dict[str, int]
    final_state: BuilderState
    base_fees: List[int]
    invariant_failures: List[str]

    def summary(self) -> Dict[str, Any]:
        wins: Dict[str, int] = defaultdict(int)
        for s in self.slots:
            wins[s.winner] += 1
        fuzz = [f for s in self.slots for f in s.summary()["fuzz"].values()]
        evaluated = sum(f["layouts_evaluated"] for f in fuzz)
        fuzz_time = sum(f["elapsed_s"] for f in fuzz)
        return {
            "slots": [s.summary() for s in self.slots],
            "wins": dict(wins),
            "proposer_revenue_wei": sum(s.proposer_payment_wei for s in self.slots),
            "total_gas_used": sum(s.block.gas_used for s in self.slots),
            "max_gas_used": max((s.block.gas_used for s in self.slots), default=0),
            "gas_limit": self.slots[0].block.gas_limit if self.slots else BLOCK_GAS_LIMIT,
            "unrealized_mev_wei": sum(s.block.unrealized_mev_wei for s in self.slots),
            "base_fees": list(self.base_fees),
            "txpool": dict(self.pool_stats),
            "fuzz_layouts_evaluated": evaluated,
            "fuzz_layouts_per_s": round(evaluated / fuzz_time, 1) if fuzz_time > 0 else 0.0,
            "invariants_ok": not self.invariant_failures,
            "invariant_failures": list(self.invariant_failures),
            "final_state_root": self.final_state.state_root(),
        }


class PBSSimulation:
    def __init__(self, pools: Iterable[Pool], config: SimulationConfig,
                 builders: Optional[Sequence[BlockBuilder]] = None, relays: Optional[Sequence[Relay]] = None,
                 proposer: Optional[Proposer] = None) -> None:
        self.state = BuilderState.from_pools(pools)
        self.config = config
        self.auction = MevBoostAuction(builders or default_builders(config), relays, proposer)

    def run(self, executor: Optional[WorkerPool] = None, slots: Optional[int] = None,
            measure_unrealized: bool = True) -> PBSReport:
        cfg = self.config
        slots = cfg.blocks if slots is None else slots
        gas_limit = clamp_gas_limit(cfg.block_gas_limit)
        base_fee = cfg.base_fee_wei
        state = self.state
        txpool = TxPool(block_gas_limit=gas_limit)
        stats: Dict[str, int] = defaultdict(int)
        results: List[AuctionResult] = []
        base_fees: List[int] = []
        failures: List[str] = []
        grid = cfg.frontrun_grid or 12
        refine = cfg.refine_iterations or 6
        own = executor is None
        wp = WorkerPool(cfg.workers, cfg.start_method, cfg.parallel_threshold) if own else executor
        try:
            for slot in range(slots):
                base_fees.append(base_fee)
                for tx in generate_builder_mempool(state, cfg.txs_per_block, cfg.seed, base_fee):
                    res = txpool.add(tx, base_fee, state.nonce(tx.sender))
                    stats[res.status] += 1
                    stats["evicted"] += len(res.evicted) if res.status == "added" else 0
                pending = txpool.pending(state.nonces)
                bundles = build_sandwich_bundles(state, pending, base_fee, cfg.native_token, grid, refine,
                                                 cfg.bribe_bps, cfg.max_frontrun_bps)
                bundles.extend(self._jit_bundles(state, pending, base_fee))
                result = self.auction.run_slot(state, bundles, pending, base_fee, slot, gas_limit, wp,
                                               seed=cfg.seed, measure_unrealized=measure_unrealized,
                                               native=cfg.native_token)
                failures.extend(self._check(state, result, gas_limit))
                results.append(result)
                state = result.post_state
                stats["pruned"] += len(txpool.prune(state.nonces))
                base_fee = next_base_fee(base_fee, result.block.gas_used, gas_limit)
                state.fork.mine()
        finally:
            if own:
                wp.close()
        return PBSReport(results, dict(stats), state, base_fees, failures)

    @staticmethod
    def _jit_bundles(state: BuilderState, pending: Sequence[BuilderTx], base_fee: int,
                     limit: int = 2) -> List[Bundle]:
        """JIT LP bundles around the largest swaps through concentrated pools (exercise the JIT guard)."""
        swaps = [t for t in pending if t.kind is TxKind.SWAP and t.effective_tip(base_fee) is not None
                 and any(isinstance(state.pools.get(p), ConcentratedPool) for p in t.route)]
        swaps.sort(key=lambda t: (-t.amount_in, t.tx_hash))
        out: List[Bundle] = []
        for n, victim in enumerate(swaps[:limit]):
            pool = next(state.pools[p] for p in victim.route if isinstance(state.pools.get(p), ConcentratedPool))
            b = build_jit_bundle(state, victim, f"jit-lp:{victim.tx_hash[:14]}", max(1, pool.liquidity * 4),
                                 base_fee)
            if b is not None:
                out.append(b)
        return out

    @staticmethod
    def _check(parent: BuilderState, result: AuctionResult, gas_limit: int) -> List[str]:
        out: List[str] = []
        b = result.block
        tag = f"slot {result.slot}"
        if b.gas_used > gas_limit or b.gas_used > BLOCK_GAS_LIMIT:
            out.append(f"{tag}: gas_used {b.gas_used} over limit")
        if b.txs and b.txs[-1].cumulative_gas != b.gas_used:
            out.append(f"{tag}: cumulative gas mismatch")
        if result.proposer_payment_wei > b.value_wei:
            out.append(f"{tag}: proposer paid more than block value")
        eth_delta = sum(toks.get(ETH, 0) for toks in result.post_state.fork.balances.values()) \
            - sum(toks.get(ETH, 0) for toks in parent.fork.balances.values())
        if eth_delta != 0:
            out.append(f"{tag}: ETH not conserved ({eth_delta:+d})")
        for pid, pool in parent.pools.items():
            after = result.post_state.pools[pid]
            if isinstance(pool, V2Pool) and isinstance(after, V2Pool) and after.k < pool.k:
                out.append(f"{tag}: {pid} k decreased")
        return out


__all__ = [
    "BLOCK_GAS_LIMIT",
    "ETH",
    "FUZZ_SHARDS",
    "MUTATIONS",
    "AddResult",
    "AuctionResult",
    "BlockBuilder",
    "Bundle",
    "BuilderBid",
    "BuilderState",
    "BuilderTx",
    "FuzzResult",
    "GuardAction",
    "GuardPolicy",
    "MevBoostAuction",
    "PBSReport",
    "PBSSimulation",
    "Proposer",
    "Relay",
    "SandwichCandidate",
    "SealedBlock",
    "Status",
    "Strategy",
    "TxKind",
    "TxPool",
    "TxTrace",
    "best_sandwich",
    "build_jit_bundle",
    "clone_pool",
    "build_sandwich_bundles",
    "clamp_gas_limit",
    "classify_bundle",
    "default_builders",
    "execute_bundle",
    "execute_tx",
    "fuzz_block_layout",
    "generate_builder_mempool",
    "item_id",
    "modify_liquidity",
    "mutate_layout",
    "native_value",
    "next_base_fee",
    "price_and_nonce_order",
    "repair_nonce_order",
    "seal_block",
    "simulate_route_sandwich",
]
