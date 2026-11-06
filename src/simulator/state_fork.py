"""Layer 03: in-memory state forking (Anvil-style snapshot / revert).

``StateFork`` holds pool objects and account balances for the simulated
chain. It mirrors the parts of Anvil's cheat RPCs the simulator needs:

* ``snapshot()`` / ``revert(id)`` - like ``evm_snapshot`` / ``evm_revert``.
  Reverting restores the state and discards that snapshot and every later
  one.
* ``fork()`` - an independent deep copy, like starting a second Anvil on the
  same fork block.
* ``checkpoint()`` - a context manager that always rolls back unless told to
  commit, and rolls back on any exception.
* ``mine(n)`` - advances the block number.

Snapshots are deep copies kept in process memory, so fork/mutate/rollback
never touches a network. Balances may go negative: that is borrowed (flash)
capital, and the net figure is what a strategy made or lost.

The constructor takes ownership of the pool objects it is given; pass
``copy.deepcopy(pools)`` to keep the originals unchanged.
"""

from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

from .dataflow import TokenFlowGraph
from .slippage_engine import (
    ConcentratedPool,
    MASEEngine,
    MASEError,
    Pool,
    SwapResult,
    V2Pool,
    _direction,
)


class SwapReverted(MASEError):
    """A simulated swap failed (min-out not met, liquidity exhausted, bad input)."""


@dataclass(frozen=True)
class _Snapshot:
    pools: Dict[str, Pool]
    balances: Dict[str, Dict[str, int]]
    block_number: int


class StateFork:
    def __init__(self, pools: Iterable[Pool] = (), block_number: int = 0,
                 balances: Optional[Dict[str, Dict[str, int]]] = None) -> None:
        self.pools: Dict[str, Pool] = {p.pool_id: p for p in pools}
        self.balances: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for acct, toks in (balances or {}).items():
            for tok, amt in toks.items():
                self.balances[acct][tok] = int(amt)
        self.block_number = block_number
        self._snapshots: Dict[int, _Snapshot] = {}
        self._next_id = 0
        self._commit_flags: Dict[int, bool] = {}

    # -- snapshot / revert ---------------------------------------------------

    def _capture(self) -> _Snapshot:
        return _Snapshot(
            pools=copy.deepcopy(self.pools),
            balances={a: dict(t) for a, t in self.balances.items()},
            block_number=self.block_number,
        )

    def _restore(self, snap: _Snapshot) -> None:
        self.pools = copy.deepcopy(snap.pools)
        self.balances = defaultdict(lambda: defaultdict(int))
        for acct, toks in snap.balances.items():
            self.balances[acct].update(toks)
        self.block_number = snap.block_number

    def snapshot(self) -> int:
        sid = self._next_id
        self._next_id += 1
        self._snapshots[sid] = self._capture()
        return sid

    def revert(self, snapshot_id: int) -> None:
        snap = self._snapshots.get(snapshot_id)
        if snap is None:
            raise KeyError(f"unknown or already reverted snapshot {snapshot_id}")
        self._restore(snap)
        self._drop_from(snapshot_id)

    def _drop_from(self, snapshot_id: int) -> None:
        for sid in [s for s in self._snapshots if s >= snapshot_id]:
            del self._snapshots[sid]

    @property
    def snapshot_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self._snapshots))

    @contextmanager
    def checkpoint(self, commit: bool = False) -> Iterator["StateFork"]:
        """Run a block of mutations; roll back unless ``commit`` (always on error).

        Inside the block, call ``commit_checkpoint()`` to keep the changes
        even when ``commit`` was False.
        """
        sid = self.snapshot()
        self._commit_flags[sid] = commit
        try:
            yield self
        except BaseException:
            if sid in self._snapshots:
                self.revert(sid)
            raise
        else:
            if sid in self._snapshots:
                if self._commit_flags.get(sid):
                    self._drop_from(sid)
                else:
                    self.revert(sid)
        finally:
            self._commit_flags.pop(sid, None)

    def commit_checkpoint(self) -> None:
        """Mark the innermost open checkpoint to be kept."""
        if not self._commit_flags:
            raise RuntimeError("no open checkpoint")
        self._commit_flags[max(self._commit_flags)] = True

    def fork(self) -> "StateFork":
        child = StateFork(block_number=self.block_number)
        child.pools = copy.deepcopy(self.pools)
        for acct, toks in self.balances.items():
            child.balances[acct].update(toks)
        return child

    def mine(self, blocks: int = 1) -> int:
        if blocks < 0:
            raise ValueError("cannot mine a negative number of blocks")
        self.block_number += blocks
        return self.block_number

    # -- execution -------------------------------------------------------------

    def engine(self, default_tolerance_bps: int = 50) -> MASEEngine:
        """Engine over the live pool objects (quotes see current state)."""
        return MASEEngine(self.pools.values(), default_tolerance_bps)

    def balance(self, account: str, token: str) -> int:
        return self.balances.get(account, {}).get(token, 0)

    def quote(self, pool_id: str, token_in: str, amount_in: int) -> Tuple[SwapResult, str]:
        pool = self._pool(pool_id)
        zero_for_one, token_out = _direction(pool, token_in)
        return pool.quote_exact_in(amount_in, zero_for_one), token_out

    def swap(self, account: str, pool_id: str, token_in: str, amount_in: int,
             min_amount_out: int = 0, tx: str = "", flow: Optional[TokenFlowGraph] = None,
             label: str = "") -> SwapResult:
        """Exact-input swap that mutates the pool and balances, or raises ``SwapReverted``.

        State is untouched when the swap reverts.
        """
        if amount_in <= 0:
            raise SwapReverted("amount_in must be positive")
        pool = self._pool(pool_id)
        try:
            zero_for_one, token_out = _direction(pool, token_in)
            result = pool.quote_exact_in(amount_in, zero_for_one)
        except MASEError as exc:
            raise SwapReverted(str(exc)) from exc
        if not result.fully_filled:
            raise SwapReverted("insufficient liquidity for full fill")
        if result.amount_out <= 0:
            raise SwapReverted("zero output")
        if result.amount_out < min_amount_out:
            raise SwapReverted(f"amount_out {result.amount_out} below min {min_amount_out}")
        if isinstance(pool, V2Pool):
            pool.apply_swap(result, zero_for_one)
        else:
            assert isinstance(pool, ConcentratedPool)
            pool.apply_swap(amount_in, zero_for_one)
        self.balances[account][token_in] -= amount_in
        self.balances[account][token_out] += result.amount_out
        if flow is not None:
            flow.record_swap(tx, account, pool_id, token_in, amount_in, token_out, result.amount_out, label)
        return result

    def _pool(self, pool_id: str) -> Pool:
        try:
            return self.pools[pool_id]
        except KeyError:
            raise SwapReverted(f"unknown pool {pool_id!r}") from None

    # -- fingerprint -----------------------------------------------------------

    def state_root(self) -> str:
        """Deterministic hash of pool state and balances, for cheap equality checks."""
        h = hashlib.sha256()
        h.update(str(self.block_number).encode())
        for pid in sorted(self.pools):
            pool = self.pools[pid]
            if isinstance(pool, V2Pool):
                h.update(f"{pid}|{pool.reserve0}|{pool.reserve1}".encode())
            else:
                h.update(f"{pid}|{pool.sqrt_price_x96}|{pool.tick}|{pool.liquidity}".encode())
        for acct in sorted(self.balances):
            for tok in sorted(self.balances[acct]):
                amt = self.balances[acct][tok]
                if amt:
                    h.update(f"{acct}|{tok}|{amt}".encode())
        return h.hexdigest()

    def pool_state(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for pid, pool in self.pools.items():
            if isinstance(pool, V2Pool):
                out[pid] = {"reserve0": pool.reserve0, "reserve1": pool.reserve1, "k": pool.k}
            else:
                out[pid] = {"sqrt_price_x96": pool.sqrt_price_x96, "tick": pool.tick, "liquidity": pool.liquidity}
        return out


__all__ = ["StateFork", "SwapReverted"]
