"""Layer 15: interactive BTC / ETH transaction dataflow graphs.

Status: the ETH graph is **exact** for the simulated block (every edge is an
integer number of wei or token base units, and the graph is checked against
a replay of the sealed block); the BTC side is a **model**: a seeded,
synthetic UTXO mempool (P2WPKH-sized transactions, BIP 125 replacement
rules, child-pays-for-parent packages, ancestor-feerate block templates),
because the simulator has no Bitcoin chain. Nothing is fetched from a
network.

Graph model
-----------
:class:`TxGraph` is a directed acyclic multigraph. Transactions are nodes;
state that a transaction reads and writes is versioned (an account or pool
gets a new node ``name#k`` every time a transaction changes it), so value
always flows forward in time and the graph stays acyclic even when one
account trades several times in a block. Edges carry ``(asset, amount)``
with ``amount`` an ``int`` (wei, token base units or satoshis), so
conservation is checked exactly:

* **ETH** (:func:`build_eth_graph`): every included transaction of a sealed
  block is replayed on the parent state. Swap legs become
  ``account -> tx -> pool`` and ``pool -> tx -> account`` edges per hop
  (multi-hop intermediates net to zero inside the transaction), gas becomes
  ``account -> tx -> builder`` (priority fee) and ``tx -> burn`` (base fee),
  and a bundle's coinbase transfer becomes a bribe edge. Each transaction's
  edges are compared with the account and pool deltas the replay actually
  produced; any difference, even 1 wei, is an anomaly, and so is a replayed
  state root that differs from the sealed block's.
* **BTC** (:func:`build_btc_graph`): outputs are edges from the creating
  transaction to the spending one (confirmed UTXOs are source nodes, unspent
  outputs end at address nodes, fees go to a miner node). Fees are exact
  ``inputs - outputs``.

Anomalies (:class:`Anomaly`) - block-building competition down to the wei:

* ETH: value-conservation breaks, state-root mismatch, sandwiches (same
  sender before and after a victim on the same pool, with the attacker's
  exact token gain), builder competition per victim (winner's top-of-block
  value minus the best rival's, in wei; a margin at or below
  ``photo_finish_wei`` is flagged as a photo finish), and evicted bundles
  with the builder's reason.
* BTC: double spends, BIP 125 replacements (valid, or the rule they break),
  negative fees, unknown inputs, dust outputs, CPFP packages.

Layout and rendering
--------------------
:meth:`TxGraph.topological_order` (Kahn's algorithm with a deterministic
tie-break) feeds :meth:`TxGraph.layout`: longest-path layering, then
barycentric crossing reduction sweeps (down and up, keeping the best order
seen), then coordinates. :func:`render_html` writes one self-contained page
(inline SVG, CSS and JavaScript; no network) with a tab per chain, wheel
zoom, drag to pan, tooltips with exact amounts, click-to-highlight of a
node's upstream and downstream flow, and an anomaly list. All labels go
through :mod:`simulator.i18n` (Russian by default).
"""

from __future__ import annotations

import hashlib
import heapq
import html
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from . import i18n
from .block_builder import (
    BURN,
    ETH,
    GWEI,
    BuilderState,
    BuilderTx,
    Bundle,
    CompetitionOutcome,
    Status,
    TxKind,
    _quote_hops,
    best_sandwich,
    default_competitors,
    execute_tx,
    run_competition,
    score_bundle,
)
from .config import ConfigError
from .slippage_engine import V2Pool, _direction
from .state_fork import SwapReverted

CHAINS: Tuple[str, ...] = ("eth", "btc")
FLOW_KINDS: Tuple[str, ...] = ("flow", "fee", "bribe", "spend")
TOKEN_DECIMALS: Dict[str, int] = {"ETH": 18, "WETH": 18, "DAI": 18, "USDC": 6, "USDT": 6, "WBTC": 8, "BTC": 8}
DUST_SAT = 546
MIN_RELAY_SAT_VB = 1
PHOTO_FINISH_WEI = GWEI
BLOCK_VSIZE_LIMIT = 1_000_000
CPFP_PARENT_RATE = 2  # sat/vB: parents at or below this rate count as "stuck" for CPFP detection


class GraphCycleError(ConfigError):
    """The graph has a cycle (it should never happen with versioned state)."""


@dataclass
class Node:
    id: str
    chain: str
    kind: str
    label: str
    attrs: Dict[str, Any] = field(default_factory=dict)
    layer: int = 0
    order: int = 0
    x: float = 0.0
    y: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "chain": self.chain, "kind": self.kind, "label": self.label, "attrs": dict(self.attrs),
                "layer": self.layer, "order": self.order, "x": round(self.x, 2), "y": round(self.y, 2)}


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    asset: str
    amount: int
    kind: str = "flow"
    tx: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "asset": self.asset, "amount": self.amount, "kind": self.kind,
                "tx": self.tx}


@dataclass(frozen=True)
class Anomaly:
    chain: str
    kind: str
    severity: str  # info | warning | error
    nodes: Tuple[str, ...]
    amount: int
    message: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"chain": self.chain, "kind": self.kind, "severity": self.severity, "nodes": list(self.nodes),
                "amount": self.amount, "message": self.message, "detail": dict(self.detail)}


class TxGraph:
    """Directed acyclic multigraph of value flows with exact integer amounts."""

    def __init__(self, chain: str, title: str = "") -> None:
        if chain not in CHAINS:
            raise ConfigError(f"unknown chain {chain!r}")
        self.chain = chain
        self.title = title
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.anomalies: List[Anomaly] = []
        self.meta: Dict[str, Any] = {}
        self._succ: Dict[str, List[int]] = defaultdict(list)
        self._pred: Dict[str, List[int]] = defaultdict(list)

    # -- construction --------------------------------------------------------

    def add_node(self, node_id: str, kind: str, label: str = "", **attrs: Any) -> Node:
        node = self.nodes.get(node_id)
        if node is None:
            node = Node(node_id, self.chain, kind, label or node_id, dict(attrs))
            self.nodes[node_id] = node
        else:
            node.attrs.update(attrs)
        return node

    def add_edge(self, src: str, dst: str, asset: str, amount: int, kind: str = "flow", tx: str = "") -> Edge:
        if src not in self.nodes or dst not in self.nodes:
            raise ConfigError(f"edge {src} -> {dst} references a missing node")
        if type(amount) is not int or amount < 0:
            raise ConfigError(f"edge amount must be a non-negative int, got {amount!r}")
        if src == dst:
            raise ConfigError(f"self loop on {src}")
        e = Edge(src, dst, asset, amount, kind, tx)
        self._succ[src].append(len(self.edges))
        self._pred[dst].append(len(self.edges))
        self.edges.append(e)
        return e

    def add_anomaly(self, kind: str, severity: str, nodes: Sequence[str], amount: int, message: str,
                    **detail: Any) -> Anomaly:
        a = Anomaly(self.chain, kind, severity, tuple(nodes), int(amount), message, detail)
        self.anomalies.append(a)
        for n in nodes:
            if n in self.nodes:
                self.nodes[n].attrs.setdefault("anomalies", []).append(kind)
        return a

    # -- structure -----------------------------------------------------------

    def successors(self, node_id: str) -> List[str]:
        return list(dict.fromkeys(self.edges[i].dst for i in self._succ.get(node_id, ())))

    def predecessors(self, node_id: str) -> List[str]:
        return list(dict.fromkeys(self.edges[i].src for i in self._pred.get(node_id, ())))

    def out_edges(self, node_id: str) -> List[Edge]:
        return [self.edges[i] for i in self._succ.get(node_id, ())]

    def in_edges(self, node_id: str) -> List[Edge]:
        return [self.edges[i] for i in self._pred.get(node_id, ())]

    def topological_order(self) -> List[str]:
        """Kahn's algorithm; ties broken by insertion order, so the order is deterministic."""
        index = {nid: i for i, nid in enumerate(self.nodes)}
        indeg = {nid: len(self.predecessors(nid)) for nid in self.nodes}
        heap = [(index[n], n) for n, d in indeg.items() if d == 0]
        heapq.heapify(heap)
        out: List[str] = []
        while heap:
            _, n = heapq.heappop(heap)
            out.append(n)
            for m in self.successors(n):
                indeg[m] -= 1
                if indeg[m] == 0:
                    heapq.heappush(heap, (index[m], m))
        if len(out) != len(self.nodes):
            stuck = sorted(n for n, d in indeg.items() if d > 0)
            raise GraphCycleError(f"cycle through {', '.join(stuck[:5])}")
        return out

    def is_acyclic(self) -> bool:
        try:
            self.topological_order()
        except GraphCycleError:
            return False
        return True

    def layers(self) -> Dict[str, int]:
        """Longest-path layering: sources on layer 0, every edge goes to a strictly higher layer."""
        layer: Dict[str, int] = {}
        for n in self.topological_order():
            preds = self.predecessors(n)
            layer[n] = 1 + max(layer[p] for p in preds) if preds else 0
        return layer

    def ancestors(self, node_id: str) -> Set[str]:
        return self._walk(node_id, self.predecessors)

    def descendants(self, node_id: str) -> Set[str]:
        return self._walk(node_id, self.successors)

    @staticmethod
    def _walk(start: str, step: Any) -> Set[str]:
        seen: Set[str] = set()
        stack = [start]
        while stack:
            for m in step(stack.pop()):
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        return seen

    def balance(self, node_id: str, kinds: Sequence[str] = FLOW_KINDS) -> Dict[str, int]:
        """Per asset: inflow minus outflow at ``node_id`` over edges of ``kinds`` (exact)."""
        out: Dict[str, int] = defaultdict(int)
        for e in self.in_edges(node_id):
            if e.kind in kinds:
                out[e.asset] += e.amount
        for e in self.out_edges(node_id):
            if e.kind in kinds:
                out[e.asset] -= e.amount
        return {k: v for k, v in sorted(out.items()) if v}

    # -- layout --------------------------------------------------------------

    def crossings(self, order: Optional[Mapping[str, int]] = None, layer: Optional[Mapping[str, int]] = None) -> int:
        """Edge crossings between consecutive layers (edges that span exactly one layer; exact count)."""
        layer = layer if layer is not None else {n: v.layer for n, v in self.nodes.items()}
        order = order if order is not None else {n: v.order for n, v in self.nodes.items()}
        by_gap: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for e in {(e.src, e.dst) for e in self.edges}:
            s, d = e
            if layer[d] == layer[s] + 1:
                by_gap[layer[s]].append((order[s], order[d]))
        total = 0
        for pairs in by_gap.values():
            pairs.sort()
            for i in range(len(pairs)):
                for j in range(i + 1, len(pairs)):
                    a, b = pairs[i], pairs[j]
                    if a[0] != b[0] and a[1] > b[1]:
                        total += 1
        return total

    def layout(self, sweeps: int = 6, dx: float = 190.0, dy: float = 64.0, margin: float = 40.0) -> Dict[str, Any]:
        """Assign ``layer``, ``order``, ``x`` and ``y`` to every node; returns layout statistics."""
        layer = self.layers()
        n_layers = 1 + max(layer.values()) if layer else 0
        rows: List[List[str]] = [[] for _ in range(n_layers)]
        for n in self.nodes:  # insertion order is the initial order
            rows[layer[n]].append(n)

        def positions(rs: Sequence[Sequence[str]]) -> Dict[str, int]:
            return {n: i for r in rs for i, n in enumerate(r)}

        def bary(r: Sequence[str], pos: Mapping[str, int], neighbours: Any) -> List[str]:
            keyed = []
            for i, n in enumerate(r):
                nb = [pos[m] for m in neighbours(n) if m in pos]
                keyed.append((Fraction(sum(nb), len(nb)) if nb else Fraction(i), i, n))
            return [n for _, _, n in sorted(keyed)]

        best = [list(r) for r in rows]
        best_x = self.crossings(positions(best), layer)
        initial = best_x
        for sweep in range(max(0, sweeps)):
            for li in (range(1, n_layers) if sweep % 2 == 0 else range(n_layers - 2, -1, -1)):
                pos = positions(rows)
                rows[li] = bary(rows[li], pos, self.predecessors if sweep % 2 == 0 else self.successors)
            x = self.crossings(positions(rows), layer)
            if x < best_x:
                best, best_x = [list(r) for r in rows], x
        for li, r in enumerate(best):
            for oi, n in enumerate(r):
                node = self.nodes[n]
                node.layer, node.order = li, oi
                node.x = margin + li * dx
                node.y = margin + oi * dy
        width = margin * 2 + max(0, n_layers - 1) * dx
        height = margin * 2 + max((len(r) for r in best), default=1) * dy
        self.meta.update(layers=n_layers, crossings=best_x, crossings_initial=initial, width=width, height=height)
        return {"layers": n_layers, "crossings": best_x, "crossings_initial": initial, "width": width,
                "height": height}

    # -- export --------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        kinds: Dict[str, int] = defaultdict(int)
        for n in self.nodes.values():
            kinds[n.kind] += 1
        akinds: Dict[str, int] = defaultdict(int)
        for a in self.anomalies:
            akinds[a.kind] += 1
        return {"chain": self.chain, "title": self.title, "nodes": len(self.nodes), "edges": len(self.edges),
                "node_kinds": dict(sorted(kinds.items())), "anomalies": dict(sorted(akinds.items())),
                "acyclic": self.is_acyclic(), "digest": self.digest(), **{k: v for k, v in self.meta.items()
                                                                          if isinstance(v, (int, float, str, bool))}}

    def to_dict(self) -> Dict[str, Any]:
        return {"chain": self.chain, "title": self.title, "meta": dict(self.meta),
                "nodes": [n.to_dict() for n in self.nodes.values()], "edges": [e.to_dict() for e in self.edges],
                "anomalies": [a.to_dict() for a in self.anomalies]}

    def digest(self) -> str:
        """SHA-256 of the structure (nodes, edges, anomaly kinds); independent of layout coordinates."""
        h = hashlib.sha256()
        for n in self.nodes.values():
            h.update(f"n|{n.id}|{n.kind}\n".encode())
        for e in self.edges:
            h.update(f"e|{e.src}|{e.dst}|{e.asset}|{e.amount}|{e.kind}\n".encode())
        for a in self.anomalies:
            h.update(f"a|{a.kind}|{a.amount}|{','.join(a.nodes)}\n".encode())
        return h.hexdigest()


def audit_conservation(graph: TxGraph, kinds: Sequence[str] = ("tx", "utxo_tx")) -> Dict[str, Dict[str, int]]:
    """Every transaction node whose value does not balance exactly: ``{node: {asset: inflow - outflow}}``.

    For ETH every asset must net to zero inside a transaction. For BTC the
    fee is an explicit edge to the miner, so inputs equal outputs plus fee.
    """
    out: Dict[str, Dict[str, int]] = {}
    for n in graph.nodes.values():
        if n.kind in kinds:
            net = graph.balance(n.id)
            if net:
                out[n.id] = net
    return out


# ---------------------------------------------------------------------------
# ETH: replay a sealed block into a versioned flow graph
# ---------------------------------------------------------------------------


class _Versions:
    """Current node id of every versioned entity (account or pool)."""

    def __init__(self, graph: TxGraph, prefix: str, kind: str) -> None:
        self.graph = graph
        self.prefix = prefix
        self.kind = kind
        self.version: Dict[str, int] = {}

    def current(self, name: str) -> str:
        v = self.version.setdefault(name, 0)
        nid = f"{self.prefix}:{name}#{v}"
        self.graph.add_node(nid, self.kind, name if v == 0 else f"{name} v{v}", entity=name, version=v)
        return nid

    def bump(self, name: str) -> str:
        self.current(name)
        self.version[name] += 1
        return self.current(name)


def _short(tx_hash: str) -> str:
    return tx_hash if len(tx_hash) <= 14 else f"{tx_hash[:8]}..{tx_hash[-4:]}"


def _balances(state: BuilderState, accounts: Iterable[str]) -> Dict[str, Dict[str, int]]:
    return {a: {t: v for t, v in state.fork.balances.get(a, {}).items() if v} for a in accounts}


def build_eth_graph(parent: BuilderState, outcome: CompetitionOutcome, items: Sequence[Any] = (),
                    base_fee: Optional[int] = None, lang: Optional[str] = None,
                    photo_finish_wei: int = PHOTO_FINISH_WEI, check_root: bool = True) -> TxGraph:
    """Flow graph of ``outcome.block`` replayed on ``parent`` (never mutated).

    ``items``: every transaction and bundle that was submitted (bundles that
    lost appear as evicted-bundle nodes, and victims with several bundles get
    a competition anomaly with the exact winning margin).
    """
    block = outcome.block
    base = block.base_fee if base_fee is None else base_fee
    g = TxGraph("eth", i18n.t("graph.tab.eth", lang))
    accounts = _Versions(g, "acct", "account")
    pools = _Versions(g, "pool", "pool")
    builder = g.add_node(f"builder:{block.coinbase}", "builder", block.coinbase).id
    burn = g.add_node("burn:base_fee", "burn", i18n.t("graph.kind.burn", lang)).id
    by_hash: Dict[str, BuilderTx] = {}
    bundles: Dict[str, Bundle] = {}
    for it in items:
        if isinstance(it, Bundle):
            bundles[it.bundle_id] = it
            for t in it.txs:
                by_hash.setdefault(t.tx_hash, t)
        else:
            by_hash.setdefault(it.tx_hash, it)
    for it in block.layout:
        if isinstance(it, Bundle):
            bundles.setdefault(it.bundle_id, it)
            for t in it.txs:
                by_hash.setdefault(t.tx_hash, t)
        else:
            by_hash.setdefault(it.tx_hash, it)
    state = parent.copy()
    used = 0
    swaps: List[Tuple[int, BuilderTx, List[Tuple[str, str, int, str, int]]]] = []
    last_tx_of_item: Dict[str, str] = {}
    for tr in block.txs:
        last_tx_of_item[tr.item] = tr.tx_hash
    for pos, trace in enumerate(block.txs):
        tx = by_hash.get(trace.tx_hash)
        if tx is None:
            raise ConfigError(f"transaction {trace.tx_hash} of the block was not supplied")
        tnode = g.add_node(f"tx:{tx.tx_hash}", "tx", _short(tx.tx_hash), sender=tx.sender, kind_tx=tx.kind.value,
                           index=pos, item=trace.item, status=trace.status.value).id
        hops: List[Tuple[str, str, int, Any]] = []
        if tx.kind is TxKind.SWAP:
            try:
                hops, _ = _quote_hops(state, tx)
            except SwapReverted:
                hops = []
        involved = {tx.sender, block.coinbase, BURN}
        if tx.kind is TxKind.TRANSFER:
            involved.add(str(tx.payload.get("to", BURN)))
        bundle = bundles.get(trace.item.split(":", 1)[1]) if trace.item.startswith("bundle:") else None
        bribe = bundle.coinbase_transfer_wei if bundle is not None and last_tx_of_item.get(trace.item) == \
            tx.tx_hash else 0
        payer = (bundle.searcher or bundle.txs[-1].sender) if bundle is not None else tx.sender
        if bribe:
            involved.add(payer)
        before = _balances(state, involved)
        pool_before = {pid: (p.reserve0, p.reserve1) for pid, p in state.pools.items() if isinstance(p, V2Pool)}
        replay = execute_tx(state, tx, base, block.gas_limit - used, block.coinbase, trace=False)
        if replay.status is Status.INVALID:
            raise ConfigError(f"replay of {tx.tx_hash} is invalid ({replay.error}); the block does not match "
                              "its parent state")
        used += replay.gas_used
        if bribe:
            state.fork.balances[payer][ETH] -= bribe
            state.fork.balances[block.coinbase][ETH] += bribe
        after = _balances(state, involved)
        # -- edges ----------------------------------------------------------
        expected: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        receipts: Dict[str, List[Tuple[str, int, str]]] = defaultdict(list)  # account -> (asset, amount, kind)
        touched: List[str] = [tx.sender]
        sender_in = accounts.current(tx.sender)
        legs: List[Tuple[str, str, int, str, int]] = []
        if replay.status is Status.SUCCESS and tx.kind is TxKind.SWAP and hops:
            first_tok, first_amt = hops[0][1], hops[0][2]
            g.add_edge(sender_in, tnode, first_tok, first_amt, "flow", tx.tx_hash)
            expected[tx.sender][first_tok] -= first_amt
            for pid, tok_in, amt_in, res in hops:
                _, tok_out = _direction(state.pools[pid], tok_in)
                p_in = pools.current(pid)
                p_out = pools.bump(pid)
                g.add_edge(p_in, tnode, tok_out, res.amount_out, "flow", tx.tx_hash)
                g.add_edge(tnode, p_out, tok_in, amt_in, "flow", tx.tx_hash)
                legs.append((pid, tok_in, amt_in, tok_out, res.amount_out))
            last_tok, last_amt = legs[-1][3], legs[-1][4]
            receipts[tx.sender].append((last_tok, last_amt, "flow"))
            expected[tx.sender][last_tok] += last_amt
        elif replay.status is Status.SUCCESS and tx.kind is TxKind.TRANSFER:
            value = int(tx.payload.get("value", 0))
            to = str(tx.payload.get("to", BURN))
            if value:
                g.add_edge(sender_in, tnode, ETH, value, "flow", tx.tx_hash)
                expected[tx.sender][ETH] -= value
                receipts[to].append((ETH, value, "flow"))
                expected[to][ETH] += value
                if to not in touched:
                    touched.append(to)
        gas_paid = replay.priority_fee_wei + replay.base_fee_burned_wei
        if gas_paid:
            g.add_edge(sender_in, tnode, ETH, gas_paid, "fee", tx.tx_hash)
            expected[tx.sender][ETH] -= gas_paid
        if replay.priority_fee_wei:
            g.add_edge(tnode, builder, ETH, replay.priority_fee_wei, "fee", tx.tx_hash)
            expected[block.coinbase][ETH] += replay.priority_fee_wei
        if replay.base_fee_burned_wei:
            g.add_edge(tnode, burn, ETH, replay.base_fee_burned_wei, "fee", tx.tx_hash)
            expected[BURN][ETH] += replay.base_fee_burned_wei
        if bribe:
            src = sender_in if payer == tx.sender else accounts.current(payer)
            g.add_edge(src, tnode, ETH, bribe, "bribe", tx.tx_hash)
            g.add_edge(tnode, builder, ETH, bribe, "bribe", tx.tx_hash)
            expected[payer][ETH] -= bribe
            expected[block.coinbase][ETH] += bribe
            if payer not in touched:
                touched.append(payer)
        for acct in touched:
            new_node = accounts.bump(acct)
            for asset, amount, kind in receipts.get(acct, ()):
                g.add_edge(tnode, new_node, asset, amount, kind, tx.tx_hash)
            if not receipts.get(acct):
                g.add_edge(tnode, new_node, "state", 0, "state", tx.tx_hash)
        # -- exact checks ---------------------------------------------------
        for acct in sorted(involved):
            for tok in sorted(set(before[acct]) | set(after[acct]) | set(expected.get(acct, {}))):
                observed = after[acct].get(tok, 0) - before[acct].get(tok, 0)
                want = expected.get(acct, {}).get(tok, 0)
                if observed != want:
                    g.add_anomaly("conservation", "error", [tnode], abs(observed - want),
                                  i18n.t("anomaly.conservation", lang, diff=abs(observed - want), unit=tok),
                                  account=acct, token=tok, observed=observed, expected=want)
        for pid, tok_in, amt_in, tok_out, amt_out in legs:
            pool = state.pools[pid]
            if isinstance(pool, V2Pool):
                r0, r1 = pool_before[pid]
                d = {pool.token0: pool.reserve0 - r0, pool.token1: pool.reserve1 - r1}
                if d[tok_in] != amt_in or d[tok_out] != -amt_out:
                    g.add_anomaly("conservation", "error", [tnode], abs(d[tok_in] - amt_in) + abs(d[tok_out] + amt_out),
                                  i18n.t("anomaly.conservation", lang,
                                         diff=abs(d[tok_in] - amt_in) + abs(d[tok_out] + amt_out), unit=tok_in),
                                  pool=pid)
        net = g.balance(tnode)
        if net:
            g.add_anomaly("conservation", "error", [tnode], sum(abs(v) for v in net.values()),
                          i18n.t("anomaly.conservation", lang, diff=sum(abs(v) for v in net.values()),
                                 unit="/".join(net)), net=net)
        if legs:
            swaps.append((pos, tx, legs))
    if check_root and block.state_root and state.state_root() != block.state_root:
        g.add_anomaly("state_root", "error", [], 0, i18n.t("anomaly.state_root", lang),
                      replayed=state.state_root(), sealed=block.state_root)
    _detect_sandwiches(g, swaps, lang)
    _detect_competition(g, parent, outcome, list(bundles.values()), base, lang, photo_finish_wei)
    g.meta.update(block_number=block.block_number, gas_used=block.gas_used, value_wei=block.value_wei,
                  base_fee=base, txs=len(block.txs), replayed_root_ok=not any(a.kind == "state_root"
                                                                               for a in g.anomalies))
    return g


def _detect_sandwiches(g: TxGraph, swaps: Sequence[Tuple[int, BuilderTx, List[Tuple[str, str, int, str, int]]]],
                       lang: Optional[str]) -> None:
    """Adjacent swaps A, V, B where A and B share a sender (not V's), touch V's pool, A trades with V and B back."""
    for i in range(len(swaps) - 2):
        (pa, a, la), (pv, v, lv), (pb, b, lb) = swaps[i], swaps[i + 1], swaps[i + 2]
        if not (pv == pa + 1 and pb == pv + 1 and a.sender == b.sender != v.sender):
            continue
        for pid, v_in, _, v_out, _ in lv:
            front = [x for x in la if x[0] == pid and x[1] == v_in]
            back = [x for x in lb if x[0] == pid and x[1] == v_out]
            if not front or not back:
                continue
            token = front[0][1]
            spent = front[0][2]
            got_back = back[0][4] if back[0][3] == token else 0
            profit = got_back - spent
            g.add_anomaly("sandwich", "warning", [f"tx:{a.tx_hash}", f"tx:{v.tx_hash}", f"tx:{b.tx_hash}"],
                          max(0, profit), i18n.t("anomaly.sandwich", lang, victim=_short(v.tx_hash),
                                                 attacker=a.sender, profit=profit, token=token),
                          pool=pid, token=token, spent=spent, returned=got_back, profit=profit)
            break


def _detect_competition(g: TxGraph, parent: BuilderState, outcome: CompetitionOutcome, bundles: Sequence[Bundle],
                        base_fee: int, lang: Optional[str], photo_finish_wei: int) -> None:
    by_target: Dict[str, List[Bundle]] = defaultdict(list)
    for b in bundles:
        if b.target_tx:
            by_target[b.target_tx].append(b)
    for target, group in sorted(by_target.items()):
        scores = {b.bundle_id: score_bundle(parent, b, base_fee, outcome.block.gas_limit) for b in group}
        winner = outcome.winners.get(target)
        victim_node = f"tx:{target}"
        for b in group:
            if outcome.won(b.bundle_id):
                continue
            nid = g.add_node(f"bundle:{b.bundle_id}", "bundle", b.searcher or b.bundle_id, searcher=b.searcher,
                             value_wei=scores[b.bundle_id].value_wei, reason=outcome.evicted.get(b.bundle_id, "")).id
            if victim_node in g.nodes:
                g.add_edge(nid, victim_node, ETH, scores[b.bundle_id].value_wei, "competes", "")
            g.add_anomaly("evicted", "info", [nid], scores[b.bundle_id].value_wei,
                          i18n.t("anomaly.evicted", lang, bundle=b.searcher or b.bundle_id,
                                 reason=outcome.evicted.get(b.bundle_id, "not_included")))
        if winner is None or len(group) < 2:
            continue
        rivals = [s.value_wei for bid, s in scores.items() if bid != winner and s.included]
        if not rivals:
            continue
        margin = scores[winner].value_wei - max(rivals)
        nodes = [victim_node] + [f"bundle:{b.bundle_id}" for b in group if not outcome.won(b.bundle_id)]
        g.add_anomaly("competition", "info", nodes, margin,
                      i18n.t("anomaly.competition", lang, victim=_short(target), margin=margin, rivals=len(rivals)),
                      winner=winner, winner_value_wei=scores[winner].value_wei, runner_up_wei=max(rivals))
        if margin <= photo_finish_wei:
            g.add_anomaly("photo_finish", "warning", nodes, margin,
                          i18n.t("anomaly.photo_finish", lang, victim=_short(target), margin=margin), winner=winner)


@dataclass
class EthScenario:
    parent: BuilderState
    victims: List[BuilderTx]
    bundles: List[Bundle]
    outcome: CompetitionOutcome
    base_fee: int


def eth_competition_scenario(seed: int = 2, victims: int = 4, regime: str = "extreme", bots: Optional[Sequence[Any]] = None,
                             base_fee: int = 20 * GWEI) -> EthScenario:
    """A seeded arena block: victim swaps, every bot's sandwich bundle, and the greedy builder's verdict.

    ``bots`` defaults to the four Layer 08 rival strategies; pass code-factory
    bots (e.g. :func:`simulator.block_builder.generated_competitors`) to put them in the race.
    """
    from .ai_agent_factory import ArenaConfig, BotArena, arena_market  # Layer 11 arena market and victims

    parent = BuilderState.from_pools(arena_market())
    arena = BotArena([], ArenaConfig(seed=seed, victims_per_slot=victims, regime=regime, base_fee_wei=base_fee,
                                     max_base_fee_wei=max(500 * GWEI, base_fee)))
    vic = arena._victims()
    roster = list(bots) if bots is not None else default_competitors()
    bundles: List[Bundle] = []
    for v in vic:
        oracle = best_sandwich(parent.pools, v)
        for bot in roster:
            b = bot.build(parent, v, oracle, base_fee)
            if b is not None:
                bundles.append(b)
    outcome = run_competition(parent, bundles, base_fee, vic, trace=True, rank=True)
    return EthScenario(parent, vic, bundles, outcome, base_fee)


def eth_graph_for_seed(seed: int = 2, victims: int = 4, lang: Optional[str] = None, **kw: Any) -> TxGraph:
    sc = eth_competition_scenario(seed, victims, **kw)
    return build_eth_graph(sc.parent, sc.outcome, [*sc.victims, *sc.bundles], sc.base_fee, lang)


# ---------------------------------------------------------------------------
# BTC: UTXO model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TxOut:
    address: str
    value: int  # satoshis


@dataclass(frozen=True)
class BtcTx:
    txid: str
    inputs: Tuple[Tuple[str, int], ...]
    outputs: Tuple[TxOut, ...]
    rbf: bool = True

    @property
    def weight(self) -> int:
        """P2WPKH estimate: 42 WU overhead (incl. segwit marker), 272 WU per input, 124 WU per output."""
        return 42 + 272 * len(self.inputs) + 124 * len(self.outputs)

    @property
    def vsize(self) -> int:
        return -(-self.weight // 4)

    @property
    def output_value(self) -> int:
        return sum(o.value for o in self.outputs)


Outpoint = Tuple[str, int]


def _txid(rng: random.Random) -> str:
    return f"{rng.getrandbits(256):064x}"


def generate_btc_mempool(seed: int = 0, txs: int = 18, funding_outputs: int = 8
                         ) -> Tuple[Dict[Outpoint, TxOut], List[BtcTx]]:
    """A seeded UTXO set and mempool: plain spends, chains, a CPFP child, one valid and one invalid RBF
    replacement, and an occasional dust output. Deterministic for a seed."""
    if txs < 4 or funding_outputs < 3:
        raise ConfigError("need at least 4 transactions and 3 funding outputs")
    rng = random.Random(f"btc-mempool:{seed}")
    fund = _txid(rng)
    utxos: Dict[Outpoint, TxOut] = {}
    for i in range(funding_outputs):
        utxos[(fund, i)] = TxOut(f"bc1q{rng.getrandbits(80):020x}", rng.randrange(20_000, 5_000_000))
    spendable: List[Tuple[Outpoint, TxOut]] = sorted(utxos.items())
    mempool: List[BtcTx] = []
    rbf_original: Optional[BtcTx] = None

    def make(inputs: Sequence[Tuple[Outpoint, TxOut]], n_out: int, rate: int, signal: bool = True,
             dust: bool = False) -> Optional[BtcTx]:
        """A transaction spending ``inputs`` at ``rate`` sat/vB, or None when the inputs cannot pay for it."""
        total = sum(o.value for _, o in inputs)
        draft = BtcTx("", tuple(op for op, _ in inputs), tuple(TxOut("x", 0) for _ in range(n_out)), signal)
        spend = total - rate * draft.vsize
        if spend < n_out * DUST_SAT:
            if n_out == 1 or spend < DUST_SAT:
                return None
            return make(inputs, 1, rate, signal, dust)
        outs: List[TxOut] = []
        left = spend
        for k in range(n_out):
            room = left - (n_out - 1 - k)  # leave at least 1 sat for every later output
            if k == n_out - 1:
                value = left
            elif dust and k == 0:
                value = min(room, rng.randrange(100, DUST_SAT))
            else:
                value = min(room, rng.randrange(1, max(2, left // 2)))
            outs.append(TxOut(f"bc1q{rng.getrandbits(80):020x}", value))
            left -= value
        return BtcTx(_txid(rng), draft.inputs, tuple(outs), signal)

    while len(mempool) < txs and spendable:
        roll = rng.random()
        k = 1 if len(spendable) < 3 or roll < 0.6 else 2
        picked = [spendable.pop(rng.randrange(len(spendable))) for _ in range(k)]
        rate = rng.choice((1, 2, 3, 5, 8, 13, 21, 34))
        built = make(picked, rng.choice((1, 2, 2, 3)), rate, signal=rng.random() < 0.8,
                     dust=rng.random() < 0.08)
        if built is None:
            continue  # inputs too small to pay the fee: they stay unspent
        tx = built
        mempool.append(tx)
        for vout, out in enumerate(tx.outputs):
            if out.value >= DUST_SAT:
                spendable.append(((tx.txid, vout), out))
        if rbf_original is None and tx.rbf and len(tx.inputs) == 1 and tx.output_value > 10_000:
            rbf_original = tx
    # A low-fee parent with a high-fee child (CPFP).
    while spendable:
        parent = make([spendable.pop(0)], 1, 1)
        child = make([((parent.txid, 0), parent.outputs[0])], 1, 60) if parent is not None else None
        if parent is not None and child is not None:
            mempool.extend((parent, child))
            break
    # A valid and an invalid BIP 125 replacement of the same original.
    if rbf_original is not None:
        op = rbf_original.inputs[0]
        value_in = (utxos.get(op) or next((t.outputs[op[1]] for t in mempool if t.txid == op[0]),
                                          TxOut("", rbf_original.output_value))).value
        orig_fee = value_in - rbf_original.output_value
        good_fee = orig_fee + MIN_RELAY_SAT_VB * rbf_original.vsize + 500
        bad_fee = max(0, orig_fee - 1)
        for fee in (good_fee, bad_fee):
            if value_in > fee:
                mempool.append(BtcTx(_txid(rng), (op,), (TxOut(f"bc1q{rng.getrandbits(80):020x}", value_in - fee),),
                                     True))
    return utxos, mempool


def _bip125(original: BtcTx, orig_fee: int, replacement: BtcTx, repl_fee: int,
            min_relay: int = MIN_RELAY_SAT_VB) -> Optional[str]:
    """None when ``replacement`` may replace ``original`` (rules 1, 3, 4 and the fee-rate rule), else the rule key."""
    if not original.rbf:
        return "rbf.not_signaled"
    if repl_fee < orig_fee + min_relay * replacement.vsize:
        return "rbf.fee_too_low"
    if Fraction(repl_fee, replacement.vsize) <= Fraction(orig_fee, original.vsize):
        return "rbf.rate_too_low"
    return None


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_btc_graph(utxos: Mapping[Outpoint, TxOut], mempool: Sequence[BtcTx], lang: Optional[str] = None,
                    min_relay: int = MIN_RELAY_SAT_VB, dust: int = DUST_SAT,
                    cpfp_parent_rate: int = CPFP_PARENT_RATE) -> TxGraph:
    """UTXO flow graph of ``mempool`` over the confirmed ``utxos``, with conflicts resolved by BIP 125."""
    g = TxGraph("btc", i18n.t("graph.tab.btc", lang))
    miner = g.add_node("miner:fees", "miner", i18n.t("graph.kind.miner", lang)).id
    created: Dict[Outpoint, TxOut] = {}
    for tx in mempool:
        for vout, out in enumerate(tx.outputs):
            created[(tx.txid, vout)] = out
    fees: Dict[str, int] = {}
    spenders: Dict[Outpoint, List[BtcTx]] = defaultdict(list)
    uf = _UnionFind()
    claims: Dict[Outpoint, int] = defaultdict(int)
    for tx in mempool:
        g.add_node(f"btx:{tx.txid}", "utxo_tx", _short(tx.txid), vsize=tx.vsize, rbf=tx.rbf)
        for op in tx.inputs:
            claims[op] += 1
    for tx in mempool:
        tnode = f"btx:{tx.txid}"
        value_in = 0
        addrs = []
        ok = True
        for op in tx.inputs:
            spenders[op].append(tx)
            if op in utxos:
                src = g.add_node(f"utxo:{op[0][:16]}:{op[1]}", "utxo", f"{op[0][:8]}:{op[1]}",
                                 address=utxos[op].address).id
                out = utxos[op]
            elif op in created:
                out = created[op]
                src = f"btx:{op[0]}"
                if claims[op] > 1:  # a contested output gets its own node, so the creator still balances
                    contested = f"out:{op[0][:16]}:{op[1]}"
                    if contested not in g.nodes:
                        g.add_node(contested, "utxo", f"{op[0][:8]}:{op[1]}", address=out.address, contested=True)
                        g.add_edge(src, contested, "BTC", out.value, "spend", op[0])
                    src = contested
            else:
                ok = False
                g.add_anomaly("missing_input", "error", [tnode], 0,
                              i18n.t("anomaly.missing_input", lang, tx=_short(tx.txid), outpoint=f"{op[0][:8]}:{op[1]}"))
                continue
            value_in += out.value
            addrs.append(out.address)
            g.add_edge(src, tnode, "BTC", out.value, "spend", tx.txid)
        for a in addrs[1:]:
            uf.union(addrs[0], a)
        fee = value_in - tx.output_value
        fees[tx.txid] = fee
        g.nodes[tnode].attrs.update(fee_sat=fee, fee_rate=round(fee / tx.vsize, 3) if tx.vsize else 0.0)
        if ok and fee < 0:
            g.add_anomaly("negative_fee", "error", [tnode], -fee,
                          i18n.t("anomaly.negative_fee", lang, tx=_short(tx.txid), missing=-fee))
        if fee > 0:
            g.add_edge(tnode, miner, "BTC", fee, "fee", tx.txid)
        for vout, out in enumerate(tx.outputs):
            if out.value < dust:
                g.add_anomaly("dust", "warning", [tnode], out.value,
                              i18n.t("anomaly.dust", lang, tx=_short(tx.txid), value=out.value))
    spent_by_mempool = {op for op, ts in spenders.items() if ts}
    for tx in mempool:
        for vout, out in enumerate(tx.outputs):
            if (tx.txid, vout) not in spent_by_mempool:
                addr = g.add_node(f"addr:{out.address}", "address", out.address[:14]).id
                g.add_edge(f"btx:{tx.txid}", addr, "BTC", out.value, "spend", tx.txid)
    # Conflicts and BIP 125.
    losers: Set[str] = set()
    for op, ts in sorted(spenders.items()):
        if len(ts) < 2:
            continue
        nodes = [f"btx:{t.txid}" for t in ts]
        g.add_anomaly("double_spend", "warning", nodes, len(ts),
                      i18n.t("anomaly.double_spend", lang, outpoint=f"{op[0][:8]}:{op[1]}", count=len(ts)))
        original = ts[0]
        best = original
        for repl in ts[1:]:
            rule = _bip125(best, fees[best.txid], repl, fees[repl.txid], min_relay)
            if rule is None:
                g.add_anomaly("rbf_ok", "info", [f"btx:{repl.txid}", f"btx:{best.txid}"],
                              fees[repl.txid] - fees[best.txid],
                              i18n.t("anomaly.rbf_ok", lang, replacement=_short(repl.txid),
                                     original=_short(best.txid), fee_delta=fees[repl.txid] - fees[best.txid]))
                losers.add(best.txid)
                best = repl
            else:
                g.add_anomaly("rbf_bad", "warning", [f"btx:{repl.txid}", f"btx:{best.txid}"], fees[repl.txid],
                              i18n.t("anomaly.rbf_bad", lang, replacement=_short(repl.txid),
                                     original=_short(best.txid), rule=i18n.t(rule, lang)), rule=rule)
                losers.add(repl.txid)
    # Losers' descendants are evicted too.
    by_id = {t.txid: t for t in mempool}
    frontier = list(losers)
    while frontier:
        dead = frontier.pop()
        for t in mempool:
            if t.txid not in losers and any(op[0] == dead for op in t.inputs):
                losers.add(t.txid)
                frontier.append(t.txid)
    for tid in losers:
        g.nodes[f"btx:{tid}"].attrs["replaced"] = True
    # CPFP: a low-fee parent (<= cpfp_parent_rate sat/vB) whose child lifts the package rate.
    for t in mempool:
        if t.txid in losers:
            continue
        for op in t.inputs:
            parent = by_id.get(op[0])
            if parent is None or parent.txid in losers:
                continue
            p_rate = Fraction(fees[parent.txid], parent.vsize)
            package = Fraction(fees[parent.txid] + fees[t.txid], parent.vsize + t.vsize)
            if p_rate <= cpfp_parent_rate and package > p_rate and Fraction(fees[t.txid], t.vsize) > package:
                g.add_anomaly("cpfp", "info", [f"btx:{t.txid}", f"btx:{parent.txid}"], fees[t.txid],
                              i18n.t("anomaly.cpfp", lang, child=_short(t.txid), parent=_short(parent.txid),
                                     rate=round(float(package), 2)))
    clusters: Dict[str, int] = {}
    for n in g.nodes.values():
        if n.kind == "utxo" and "address" in n.attrs:
            root = uf.find(n.attrs["address"])
            n.attrs["cluster"] = clusters.setdefault(root, len(clusters))
    template = block_template(mempool, fees, losers)
    g.meta.update(txs=len(mempool), replaced=len(losers), clusters=len(clusters),
                  template_txs=len(template["txids"]), template_fees_sat=template["fees_sat"],
                  template_vsize=template["vsize"])
    g.meta["template"] = template
    return g


def block_template(mempool: Sequence[BtcTx], fees: Mapping[str, int], excluded: Iterable[str] = (),
                   vsize_limit: int = BLOCK_VSIZE_LIMIT) -> Dict[str, Any]:
    """Greedy ancestor-feerate selection (the Bitcoin Core idea): repeatedly add the package (a tx plus its
    unselected in-mempool ancestors) with the highest fee rate that fits. Exact rational comparisons."""
    skip = set(excluded)
    txs = {t.txid: t for t in mempool if t.txid not in skip}
    parents = {tid: {op[0] for op in t.inputs if op[0] in txs} for tid, t in txs.items()}

    def ancestors(tid: str) -> Set[str]:
        seen: Set[str] = set()
        stack = [tid]
        while stack:
            for p in parents[stack.pop()]:
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    chosen: List[str] = []
    done: Set[str] = set()
    used = 0
    total = 0
    order = {t.txid: i for i, t in enumerate(mempool)}
    while True:
        best = None
        for tid in txs:
            if tid in done:
                continue
            pkg = ({tid} | ancestors(tid)) - done
            vs = sum(txs[x].vsize for x in pkg)
            if used + vs > vsize_limit:
                continue
            fee = sum(fees[x] for x in pkg)
            key = (Fraction(fee, vs), -order[tid])
            if best is None or key > best[0]:
                best = (key, pkg, vs, fee)
        if best is None:
            break
        _, pkg, vs, fee = best
        for x in sorted(pkg, key=lambda t: (len(ancestors(t) - done), order[t])):
            chosen.append(x)
            done.add(x)
        used += vs
        total += fee
    return {"txids": chosen, "fees_sat": total, "vsize": used}


def btc_graph_for_seed(seed: int = 0, txs: int = 18, lang: Optional[str] = None) -> TxGraph:
    utxos, mempool = generate_btc_mempool(seed, txs)
    return build_btc_graph(utxos, mempool, lang)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def format_amount(asset: str, amount: int, lang: Optional[str] = None) -> str:
    """Exact amount in whole units (``1,5 WETH``) from base units; unknown assets as raw integers."""
    if asset == "state":
        return ""
    dec = TOKEN_DECIMALS.get(asset)
    if dec is None:
        return f"{i18n.group_int(amount, lang)} {asset}"
    return f"{i18n.format_units(amount, dec, min(dec, 8), lang)} {asset}"


NODE_COLORS: Dict[str, str] = {"tx": "var(--tx)", "utxo_tx": "var(--tx)", "account": "var(--acct)",
                               "pool": "var(--pool)", "builder": "var(--sink)", "burn": "var(--sink)",
                               "miner": "var(--sink)", "bundle": "var(--warn)", "utxo": "var(--acct)",
                               "address": "var(--acct)"}

_CSS = """
:root{--bg:#0f1115;--panel:#171a21;--fg:#e8eaf0;--muted:#9aa3b2;--line:#2a2f3a;--tx:#6ea8fe;--acct:#7dd3a8;
--pool:#f5b86b;--sink:#c084fc;--warn:#f87171;--edge:#56607a;--hi:#facc15}
:root[data-theme="light"]{--bg:#f6f7f9;--panel:#fff;--fg:#1b1f27;--muted:#5b6474;--line:#d9dde5;--tx:#2563eb;
--acct:#15803d;--pool:#b45309;--sink:#7c3aed;--warn:#dc2626;--edge:#a3acbd;--hi:#ca8a04}
@media (prefers-color-scheme:light){:root:not([data-theme="dark"]){--bg:#f6f7f9;--panel:#fff;--fg:#1b1f27;
--muted:#5b6474;--line:#d9dde5;--tx:#2563eb;--acct:#15803d;--pool:#b45309;--sink:#7c3aed;--warn:#dc2626;
--edge:#a3acbd;--hi:#ca8a04}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{padding:16px}h1{font-size:20px;margin:0}p{margin:4px 0;color:var(--muted)}
.tabs{display:flex;gap:8px;padding:0 16px}.tabs button,.reset{background:var(--panel);color:var(--fg);
border:1px solid var(--line);border-radius:6px;padding:6px 12px;cursor:pointer;font:inherit}
.tabs button[aria-selected=true]{border-color:var(--tx)}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:12px;padding:12px 16px}
@media (max-width:820px){.wrap{grid-template-columns:minmax(0,1fr)}}
.pane{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden;position:relative}
svg{display:block;width:100%;height:70vh;cursor:grab;touch-action:none}
.edge{fill:none;stroke:var(--edge);opacity:.7}.edge.fee{stroke-dasharray:4 3}.edge.competes{stroke:var(--warn);
stroke-dasharray:2 4}.edge.hi{stroke:var(--hi);opacity:1}.node circle,.node rect{stroke:var(--bg);stroke-width:2}
.node.anom circle,.node.anom rect{stroke:var(--warn);stroke-width:3}.node text{fill:var(--fg);font-size:11px}
.node.dim,.edge.dim{opacity:.12}.node.hi circle,.node.hi rect{stroke:var(--hi);stroke-width:3}
.side{padding:12px;max-height:70vh;overflow:auto}.side h2{font-size:15px;margin:0 0 8px}
.side li{margin:0 0 8px;cursor:pointer;overflow-wrap:anywhere}.sev-error{color:var(--warn)}.stats{color:var(--muted);font-size:12px}
#tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;
padding:8px;font-size:12px;max-width:360px;display:none;white-space:pre-wrap;z-index:5}
"""

_JS = """
(function(){
const data=JSON.parse(document.getElementById('graph-data').textContent);
const tip=document.getElementById('tip');
document.querySelectorAll('.tabs button').forEach(b=>b.addEventListener('click',()=>{
 document.querySelectorAll('.tabs button').forEach(x=>x.setAttribute('aria-selected',x===b));
 document.querySelectorAll('.chain').forEach(c=>c.hidden=c.dataset.chain!==b.dataset.chain);}));
document.querySelectorAll('svg.graph').forEach(svg=>{
 const g=data[svg.dataset.chain];const vb=svg.viewBox.baseVal;const home=[vb.x,vb.y,vb.width,vb.height];
 let drag=null;
 svg.addEventListener('wheel',e=>{e.preventDefault();const k=e.deltaY>0?1.15:1/1.15;const r=svg.getBoundingClientRect();
  const px=vb.x+(e.clientX-r.left)/r.width*vb.width,py=vb.y+(e.clientY-r.top)/r.height*vb.height;
  vb.x=px-(px-vb.x)*k;vb.y=py-(py-vb.y)*k;vb.width*=k;vb.height*=k;},{passive:false});
 svg.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY,vb.x,vb.y];svg.setPointerCapture(e.pointerId);});
 svg.addEventListener('pointermove',e=>{if(!drag)return;const r=svg.getBoundingClientRect();
  vb.x=drag[2]-(e.clientX-drag[0])/r.width*vb.width;vb.y=drag[3]-(e.clientY-drag[1])/r.height*vb.height;});
 svg.addEventListener('pointerup',()=>{drag=null;});
 svg.parentElement.querySelector('.reset').addEventListener('click',()=>{[vb.x,vb.y,vb.width,vb.height]=home;clear(svg);});
 svg.querySelectorAll('.node').forEach(n=>{
  n.addEventListener('mousemove',e=>{tip.textContent=g.tips[n.dataset.id]||n.dataset.id;tip.style.display='block';
   tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';});
  n.addEventListener('mouseleave',()=>{tip.style.display='none';});
  n.addEventListener('click',e=>{e.stopPropagation();focus(svg,g,[n.dataset.id]);});});
});
function walk(adj,start){const seen=new Set(start),st=[...start];while(st.length){for(const m of (adj[st.pop()]||[])){
 if(!seen.has(m)){seen.add(m);st.push(m);}}}return seen;}
function clear(svg){svg.querySelectorAll('.dim,.hi').forEach(x=>x.classList.remove('dim','hi'));}
function focus(svg,g,ids){clear(svg);const keep=new Set([...walk(g.succ,ids),...walk(g.pred,ids)]);
 svg.querySelectorAll('.node').forEach(n=>{n.classList.toggle('dim',!keep.has(n.dataset.id));
  n.classList.toggle('hi',ids.includes(n.dataset.id));});
 svg.querySelectorAll('.edge').forEach(e=>{const on=keep.has(e.dataset.src)&&keep.has(e.dataset.dst);
  e.classList.toggle('dim',!on);e.classList.toggle('hi',on);});}
document.querySelectorAll('.side li[data-nodes]').forEach(li=>li.addEventListener('click',()=>{
 const sec=li.closest('.chain');const svg=sec.querySelector('svg');focus(svg,data[sec.dataset.chain],
 JSON.parse(li.dataset.nodes));}));
})();
"""


def _tooltip(g: TxGraph, n: Node, lang: Optional[str]) -> str:
    kind = i18n.t(f"graph.kind.{n.kind}", lang) if f"graph.kind.{n.kind}" in i18n.CATALOG else n.kind
    lines = [f"{n.label}", kind]
    for k, v in sorted(n.attrs.items()):
        if k in ("anomalies", "entity", "version"):
            continue
        lines.append(f"{k}: {v}")
    for e in g.in_edges(n.id)[:8]:
        if e.kind != "state":
            lines.append(f"<- {format_amount(e.asset, e.amount, lang)} ({e.amount} {e.kind})")
    for e in g.out_edges(n.id)[:8]:
        if e.kind != "state":
            lines.append(f"-> {format_amount(e.asset, e.amount, lang)} ({e.amount} {e.kind})")
    if n.attrs.get("anomalies"):
        lines.append("! " + ", ".join(dict.fromkeys(n.attrs["anomalies"])))
    return "\n".join(lines)


def render_svg(g: TxGraph, lang: Optional[str] = None) -> str:
    """Inline SVG of a laid-out graph (call :meth:`TxGraph.layout` first)."""
    esc = html.escape
    if "width" not in g.meta:
        g.layout()
    w, h = g.meta["width"], g.meta["height"]
    amounts = [e.amount for e in g.edges if e.amount > 0]
    top = max(amounts) if amounts else 1
    parts = [f'<svg class="graph" data-chain="{g.chain}" viewBox="0 0 {w:.0f} {h:.0f}" role="img" '
             f'aria-label="{esc(g.title)}">']
    for e in g.edges:
        a, b = g.nodes[e.src], g.nodes[e.dst]
        mid = (a.x + b.x) / 2
        width = 1.0 + 3.0 * (math.log10(1 + e.amount) / math.log10(1 + top) if e.amount > 0 else 0.0)
        parts.append(f'<path class="edge {esc(e.kind)}" data-src="{esc(e.src)}" data-dst="{esc(e.dst)}" '
                     f'stroke-width="{width:.2f}" d="M{a.x:.1f},{a.y:.1f} C{mid:.1f},{a.y:.1f} {mid:.1f},{b.y:.1f} '
                     f'{b.x:.1f},{b.y:.1f}"><title>{esc(format_amount(e.asset, e.amount, lang))}</title></path>')
    for n in g.nodes.values():
        cls = "node anom" if n.attrs.get("anomalies") else "node"
        color = NODE_COLORS.get(n.kind, "var(--muted)")
        shape = (f'<rect x="-9" y="-9" width="18" height="18" rx="4" fill="{color}"/>'
                 if n.kind in ("tx", "utxo_tx") else f'<circle r="8" fill="{color}"/>')
        parts.append(f'<g class="{cls}" data-id="{esc(n.id)}" transform="translate({n.x:.1f},{n.y:.1f})">{shape}'
                     f'<text x="12" y="4">{esc(n.label[:22])}</text></g>')
    parts.append("</svg>")
    return "".join(parts)


def _script_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def render_html(graphs: Sequence[TxGraph], lang: Optional[str] = None, title: Optional[str] = None) -> str:
    """One self-contained interactive page with a tab per graph."""
    esc = html.escape
    lg = i18n.resolve_language(lang)
    data: Dict[str, Any] = {}
    tabs, sections = [], []
    for i, g in enumerate(graphs):
        if "width" not in g.meta:
            g.layout()
        data[g.chain] = {"succ": {n: g.successors(n) for n in g.nodes}, "pred": {n: g.predecessors(n) for n in g.nodes},
                         "tips": {n.id: _tooltip(g, n, lg) for n in g.nodes.values()}}
        tabs.append(f'<button data-chain="{g.chain}" aria-selected="{"true" if i == 0 else "false"}">'
                    f'{esc(g.title)}</button>')
        if g.anomalies:
            items = "".join(f'<li class="sev-{esc(a.severity)}" data-nodes="{esc(json.dumps(list(a.nodes)))}">'
                            f"{esc(a.message)}</li>" for a in g.anomalies)
            listing = f"<ul>{items}</ul>"
        else:
            listing = f"<p>{esc(i18n.t('graph.no_anomalies', lg))}</p>"
        stats = (f'{len(g.nodes)} {i18n.t("graph.nodes", lg)} &middot; {len(g.edges)} {i18n.t("graph.edges", lg)} '
                 f'&middot; {g.meta.get("layers", 0)} {i18n.t("graph.layers", lg)} &middot; '
                 f'{g.meta.get("crossings", 0)} {i18n.t("graph.crossings", lg)}')
        sections.append(f'<section class="chain" data-chain="{g.chain}"{"" if i == 0 else " hidden"}>'
                        f'<div class="wrap"><div class="pane"><button class="reset" style="position:absolute;'
                        f'right:8px;top:8px">{esc(i18n.t("graph.reset", lg))}</button>{render_svg(g, lg)}</div>'
                        f'<aside class="pane side"><h2>{esc(i18n.t("graph.anomalies", lg))} ({len(g.anomalies)})</h2>'
                        f'<p class="stats">{stats}</p>{listing}</aside></div></section>')
    heading = title or i18n.t("graph.title", lg)
    return (f'<!doctype html><html lang="{lg}"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(heading)}</title>'
            f"<style>{_CSS}</style></head><body><header><h1>{esc(heading)}</h1>"
            f"<p>{esc(i18n.t('graph.subtitle', lg))}</p><p>{esc(i18n.t('graph.help', lg))}</p></header>"
            f'<nav class="tabs">{"".join(tabs)}</nav>{"".join(sections)}<div id="tip"></div>'
            f'<script type="application/json" id="graph-data">{_script_json(data)}</script>'
            f"<script>{_JS}</script></body></html>")


__all__ = [
    "BLOCK_VSIZE_LIMIT",
    "CHAINS",
    "CPFP_PARENT_RATE",
    "DUST_SAT",
    "FLOW_KINDS",
    "MIN_RELAY_SAT_VB",
    "PHOTO_FINISH_WEI",
    "TOKEN_DECIMALS",
    "Anomaly",
    "BtcTx",
    "Edge",
    "EthScenario",
    "GraphCycleError",
    "Node",
    "TxGraph",
    "TxOut",
    "audit_conservation",
    "block_template",
    "btc_graph_for_seed",
    "build_btc_graph",
    "build_eth_graph",
    "eth_competition_scenario",
    "eth_graph_for_seed",
    "format_amount",
    "generate_btc_mempool",
    "render_html",
    "render_svg",
]
