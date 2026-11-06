"""Layer 03: token dataflow graph (Slither-style flow tracing).

Slither builds data-dependency graphs over Solidity variables. The analogue
here is a graph over *accounts* (traders, pools, the attacker) whose edges are
token transfers recorded in execution order. From it we can answer:

* who ended up with which tokens (``net_flows``),
* which accounts were reached by value that left a given account, following
  transfers forward in time only (``taint_from``),
* which transfer paths connect two accounts (``paths``),
* what a sandwich extracted (``extraction``).

The graph is plain data; ``to_dot`` renders it for Graphviz.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class Transfer:
    step: int
    tx: str
    token: str
    src: str
    dst: str
    amount: int
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "tx": self.tx,
            "token": self.token,
            "src": self.src,
            "dst": self.dst,
            "amount": self.amount,
            "label": self.label,
        }


class TokenFlowGraph:
    def __init__(self) -> None:
        self.transfers: List[Transfer] = []

    def __len__(self) -> int:
        return len(self.transfers)

    # -- recording ------------------------------------------------------------

    def record(self, tx: str, token: str, src: str, dst: str, amount: int, label: str = "") -> Transfer:
        if amount < 0:
            raise ValueError("transfer amount must be non-negative")
        t = Transfer(len(self.transfers), tx, token, src, dst, int(amount), label)
        self.transfers.append(t)
        return t

    def record_swap(self, tx: str, account: str, pool_id: str, token_in: str, amount_in: int,
                    token_out: str, amount_out: int, label: str = "") -> Tuple[Transfer, Transfer]:
        """A swap is two transfers: account -> pool (in), pool -> account (out)."""
        a = self.record(tx, token_in, account, pool_id, amount_in, label)
        b = self.record(tx, token_out, pool_id, account, amount_out, label)
        return a, b

    def extend(self, other: "TokenFlowGraph") -> None:
        for t in other.transfers:
            self.record(t.tx, t.token, t.src, t.dst, t.amount, t.label)

    # -- views ------------------------------------------------------------------

    def nodes(self) -> Set[str]:
        out: Set[str] = set()
        for t in self.transfers:
            out.add(t.src)
            out.add(t.dst)
        return out

    def edges(self) -> Dict[Tuple[str, str, str], int]:
        """Aggregated ``(src, dst, token) -> total amount``."""
        agg: Dict[Tuple[str, str, str], int] = defaultdict(int)
        for t in self.transfers:
            agg[(t.src, t.dst, t.token)] += t.amount
        return dict(agg)

    def net_flows(self) -> Dict[str, Dict[str, int]]:
        """Per-account net token balance change (received minus sent)."""
        net: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for t in self.transfers:
            net[t.src][t.token] -= t.amount
            net[t.dst][t.token] += t.amount
        return {acct: {tok: v for tok, v in toks.items() if v} for acct, toks in net.items()}

    def by_tx(self, tx: str) -> List[Transfer]:
        return [t for t in self.transfers if t.tx == tx]

    def extraction(self, account: str) -> Dict[str, int]:
        return self.net_flows().get(account, {})

    def taint_from(self, source: str, token: Optional[str] = None) -> Dict[str, int]:
        """Accounts reached by value leaving ``source``, with the step each was first reached.

        A transfer propagates taint only if its sender was tainted at an
        earlier (or the same) step, so value cannot flow backwards in time.
        """
        tainted: Dict[str, int] = {source: -1}
        for t in self.transfers:
            if token is not None and t.token != token:
                continue
            if t.src in tainted and tainted[t.src] <= t.step and t.dst not in tainted:
                tainted[t.dst] = t.step
        del tainted[source]
        return tainted

    def paths(self, src: str, dst: str, max_depth: int = 6) -> List[List[Transfer]]:
        """Time-ordered transfer chains from ``src`` to ``dst`` (no revisited nodes)."""
        outgoing: Dict[str, List[Transfer]] = defaultdict(list)
        for t in self.transfers:
            outgoing[t.src].append(t)
        found: List[List[Transfer]] = []

        def walk(node: str, after: int, chain: List[Transfer], seen: Set[str]) -> None:
            if len(chain) > max_depth:
                return
            for t in outgoing.get(node, []):
                if t.step <= after or t.dst in seen:
                    continue
                nxt = chain + [t]
                if t.dst == dst:
                    found.append(nxt)
                else:
                    walk(t.dst, t.step, nxt, seen | {t.dst})

        walk(src, -1, [], {src})
        return found

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transfers": len(self.transfers),
            "nodes": sorted(self.nodes()),
            "net_flows": self.net_flows(),
        }

    def to_dot(self) -> str:
        lines = ["digraph token_flow {", "  rankdir=LR;"]
        for (src, dst, token), amount in sorted(self.edges().items()):
            lines.append(f'  "{src}" -> "{dst}" [label="{token} {amount}"];')
        lines.append("}")
        return "\n".join(lines)


def merge_graphs(graphs: Iterable[TokenFlowGraph]) -> TokenFlowGraph:
    out = TokenFlowGraph()
    for g in graphs:
        out.extend(g)
    return out


__all__ = ["Transfer", "TokenFlowGraph", "merge_graphs"]
