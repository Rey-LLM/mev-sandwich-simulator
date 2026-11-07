"""Layer 15: BTC / ETH transaction graph - topology, layout math, exact flows, anomalies, rendering."""

from __future__ import annotations

import json
import random
import re
from fractions import Fraction

import pytest

from simulator import tx_graph as tg
from simulator.block_builder import BURN, ETH, score_bundle


def _random_dag(seed: int, n: int = 30, p: float = 0.15) -> tg.TxGraph:
    rng = random.Random(seed)
    g = tg.TxGraph("eth")
    ids = [f"n{i}" for i in range(n)]
    rng.shuffle(ids)  # insertion order differs from the topological order
    for nid in ids:
        g.add_node(nid, "tx")
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                g.add_edge(f"n{i}", f"n{j}", "WETH", rng.randrange(1, 10**18))
    return g


@pytest.fixture(scope="module")
def eth_scenario():
    return tg.eth_competition_scenario(seed=2, victims=4)


@pytest.fixture(scope="module")
def eth(eth_scenario):
    sc = eth_scenario
    g = tg.build_eth_graph(sc.parent, sc.outcome, [*sc.victims, *sc.bundles], sc.base_fee)
    g.layout()
    return g


@pytest.fixture(scope="module")
def btc_data():
    return tg.generate_btc_mempool(0, 18)


@pytest.fixture(scope="module")
def btc(btc_data):
    g = tg.build_btc_graph(*btc_data)
    g.layout()
    return g


# -- graph core ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [
    dict(src="missing", dst="b", asset="X", amount=1),
    dict(src="a", dst="b", asset="X", amount=-1),
    dict(src="a", dst="b", asset="X", amount=1.5),
    dict(src="a", dst="b", asset="X", amount=True),
    dict(src="a", dst="a", asset="X", amount=1),
])
def test_add_edge_validation(bad):
    g = tg.TxGraph("eth")
    g.add_node("a", "tx")
    g.add_node("b", "tx")
    with pytest.raises(tg.ConfigError):
        g.add_edge(**bad)


def test_unknown_chain():
    with pytest.raises(tg.ConfigError):
        tg.TxGraph("sol")


@pytest.mark.parametrize("seed", range(10))
def test_topological_order_respects_every_edge(seed):
    g = _random_dag(seed)
    order = g.topological_order()
    pos = {n: i for i, n in enumerate(order)}
    assert sorted(order) == sorted(g.nodes)
    assert all(pos[e.src] < pos[e.dst] for e in g.edges)
    assert order == g.topological_order()  # deterministic


@pytest.mark.parametrize("seed", range(6))
def test_layers_strictly_increase_along_edges(seed):
    g = _random_dag(seed)
    layer = g.layers()
    assert all(layer[e.dst] > layer[e.src] for e in g.edges)
    assert all(layer[n] == 0 for n in g.nodes if not g.predecessors(n))
    # longest path: every non-source sits exactly one above its highest predecessor
    assert all(layer[n] == 1 + max(layer[p] for p in g.predecessors(n)) for n in g.nodes if g.predecessors(n))


@pytest.mark.parametrize("seed", range(8))
def test_layout_never_adds_crossings(seed):
    g = _random_dag(seed, 40, 0.1)
    stats = g.layout(sweeps=8)
    assert stats["crossings"] <= stats["crossings_initial"]
    assert stats["crossings"] == g.crossings()
    xs = {(n.layer, n.order) for n in g.nodes.values()}
    assert len(xs) == len(g.nodes)  # no two nodes share a slot


def test_cycle_is_detected():
    g = tg.TxGraph("btc")
    for n in "abc":
        g.add_node(n, "utxo_tx")
    g.add_edge("a", "b", "BTC", 1)
    g.add_edge("b", "c", "BTC", 1)
    g.add_edge("c", "a", "BTC", 1)
    assert not g.is_acyclic()
    with pytest.raises(tg.GraphCycleError):
        g.layers()


def test_crossings_count_known_example():
    g = tg.TxGraph("eth")
    for n in ("a", "b", "c", "d"):
        g.add_node(n, "tx")
    g.add_edge("a", "d", "X", 1)
    g.add_edge("b", "c", "X", 1)
    layer = {"a": 0, "b": 0, "c": 1, "d": 1}
    assert g.crossings({"a": 0, "b": 1, "c": 0, "d": 1}, layer) == 1
    assert g.crossings({"a": 0, "b": 1, "c": 1, "d": 0}, layer) == 0


def test_ancestors_descendants_and_balance():
    g = tg.TxGraph("eth")
    for n in ("s", "t", "u", "v"):
        g.add_node(n, "tx")
    g.add_edge("s", "t", "WETH", 10)
    g.add_edge("t", "u", "WETH", 7)
    g.add_edge("t", "v", "WETH", 3)
    assert g.ancestors("u") == {"s", "t"} and g.descendants("s") == {"t", "u", "v"}
    assert g.balance("t") == {}
    g.add_edge("t", "v", "WETH", 1)
    assert g.balance("t") == {"WETH": -1}
    assert tg.audit_conservation(g) == {"s": {"WETH": -10}, "t": {"WETH": -1}, "u": {"WETH": 7}, "v": {"WETH": 4}}


def test_digest_ignores_layout_but_not_amounts():
    a, b = _random_dag(3), _random_dag(3)
    a.layout(sweeps=0)
    b.layout(sweeps=6)
    assert a.digest() == b.digest()
    c = _random_dag(3)
    c.add_edge("n0", "n29", "WETH", 1)
    assert c.digest() != a.digest()


# -- ETH ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8])
def test_eth_graph_is_exact_for_every_seed(seed):
    sc = tg.eth_competition_scenario(seed=seed, victims=4)
    g = tg.build_eth_graph(sc.parent, sc.outcome, [*sc.victims, *sc.bundles], sc.base_fee)
    assert g.is_acyclic()
    assert not [a for a in g.anomalies if a.kind in ("conservation", "state_root")]
    assert tg.audit_conservation(g) == {}
    assert g.meta["replayed_root_ok"]
    builder = next(n for n in g.nodes.values() if n.kind == "builder")
    burn = next(n for n in g.nodes.values() if n.kind == "burn")
    assert g.balance(builder.id) == ({ETH: sc.outcome.block.value_wei} if sc.outcome.block.value_wei else {})
    assert g.balance(burn.id) == ({ETH: sc.outcome.block.burned_wei} if sc.outcome.block.burned_wei else {})


def test_eth_graph_tx_nodes_follow_block_order(eth, eth_scenario):
    txs = [n for n in eth.nodes.values() if n.kind == "tx"]
    assert [n.attrs["index"] for n in txs] == list(range(len(eth_scenario.outcome.block.txs)))
    order = eth.topological_order()
    pos = {n: i for i, n in enumerate(order)}
    assert all(pos[a.id] < pos[b.id] for a, b in zip(txs, txs[1:]) if a.attrs["sender"] == b.attrs["sender"])


def test_eth_account_deltas_match_post_state(eth, eth_scenario):
    """Sum of every versioned account node's balance equals the account's real balance change, to the wei."""
    sc = eth_scenario
    per_entity = {}
    for n in eth.nodes.values():
        if n.kind == "account":
            for asset, v in eth.balance(n.id).items():
                key = (n.attrs["entity"], asset)
                per_entity[key] = per_entity.get(key, 0) + v
    post = sc.outcome.post_state.fork.balances
    pre = sc.parent.fork.balances
    for (acct, asset), v in per_entity.items():
        assert v == post.get(acct, {}).get(asset, 0) - pre.get(acct, {}).get(asset, 0), (acct, asset)


def test_sandwich_profit_equals_real_balance_change(eth, eth_scenario):
    sands = [a for a in eth.anomalies if a.kind == "sandwich"]
    assert sands
    sc = eth_scenario
    for a in sands:
        attacker = eth.nodes[a.nodes[0]].attrs["sender"]
        tok = a.detail["token"]
        real = sc.outcome.post_state.fork.balances[attacker].get(tok, 0) - \
            sc.parent.fork.balances.get(attacker, {}).get(tok, 0)
        assert a.detail["profit"] == real and type(real) is int


def test_competition_margin_is_exact(eth, eth_scenario):
    sc = eth_scenario
    comps = [a for a in eth.anomalies if a.kind == "competition"]
    assert comps
    for a in comps:
        winner = a.detail["winner"]
        target = next(b.target_tx for b in sc.bundles if b.bundle_id == winner)
        values = {b.bundle_id: score_bundle(sc.parent, b, sc.base_fee) for b in sc.bundles if b.target_tx == target}
        rivals = [s.value_wei for bid, s in values.items() if bid != winner and s.included]
        assert a.amount == values[winner].value_wei - max(rivals)


def test_photo_finish_threshold(eth_scenario):
    sc = eth_scenario
    items = [*sc.victims, *sc.bundles]
    tight = tg.build_eth_graph(sc.parent, sc.outcome, items, sc.base_fee, photo_finish_wei=0)
    loose = tg.build_eth_graph(sc.parent, sc.outcome, items, sc.base_fee, photo_finish_wei=10**30)
    assert not [a for a in tight.anomalies if a.kind == "photo_finish" and a.amount > 0]
    assert len([a for a in loose.anomalies if a.kind == "photo_finish"]) == \
        len([a for a in loose.anomalies if a.kind == "competition"])


def test_evicted_bundles_are_nodes(eth, eth_scenario):
    sc = eth_scenario
    lost = [b for b in sc.bundles if not sc.outcome.won(b.bundle_id)]
    assert {f"bundle:{b.bundle_id}" for b in lost} <= set(eth.nodes)
    assert len([a for a in eth.anomalies if a.kind == "evicted"]) == len(lost)


def test_eth_graph_requires_every_block_tx(eth_scenario):
    sc = eth_scenario
    g = tg.build_eth_graph(sc.parent, sc.outcome, [], sc.base_fee)  # layout carries the items itself
    assert tg.audit_conservation(g) == {}


def test_eth_graph_does_not_mutate_parent(eth_scenario):
    sc = eth_scenario
    root = sc.parent.state_root()
    tg.build_eth_graph(sc.parent, sc.outcome, [*sc.victims, *sc.bundles], sc.base_fee)
    assert sc.parent.state_root() == root


def test_one_wei_tamper_is_caught(eth):
    import copy

    g = copy.deepcopy(eth)
    tx = next(n for n in g.nodes.values() if n.kind == "tx")
    builder = next(n for n in g.nodes.values() if n.kind == "builder")
    g.add_edge(tx.id, builder.id, ETH, 1, "fee")
    assert tg.audit_conservation(g) == {tx.id: {ETH: -1}}


def test_state_root_mismatch_is_reported(eth_scenario):
    sc = eth_scenario
    import dataclasses

    block = dataclasses.replace(sc.outcome.block, state_root="00" * 32)
    outcome = dataclasses.replace(sc.outcome, block=block)
    g = tg.build_eth_graph(sc.parent, outcome, [*sc.victims, *sc.bundles], sc.base_fee)
    assert [a.kind for a in g.anomalies if a.kind == "state_root"] == ["state_root"]


def test_eth_graph_is_deterministic():
    assert tg.eth_graph_for_seed(3).digest() == tg.eth_graph_for_seed(3).digest()


def test_burn_and_builder_nodes(eth):
    kinds = {n.kind for n in eth.nodes.values()}
    assert {"tx", "account", "pool", "builder", "burn"} <= kinds
    assert all(e.amount == 0 for e in eth.edges if e.kind == "state")
    assert not any(e.dst.startswith("acct:") and e.src.startswith("acct:") for e in eth.edges)
    assert BURN


# -- BTC ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_in,n_out,vsize", [(1, 1, 110), (1, 2, 141), (2, 2, 209), (3, 1, 246)])
def test_p2wpkh_vsize(n_in, n_out, vsize):
    tx = tg.BtcTx("t", tuple(("p", i) for i in range(n_in)), tuple(tg.TxOut("a", 1) for _ in range(n_out)))
    assert tx.weight == 42 + 272 * n_in + 124 * n_out
    assert tx.vsize == vsize


@pytest.mark.parametrize("seed", range(12))
def test_btc_mempool_is_deterministic_and_conserves_value(seed):
    utxos, mempool = tg.generate_btc_mempool(seed)
    assert tg.generate_btc_mempool(seed) == (utxos, mempool)
    g = tg.build_btc_graph(utxos, mempool)
    assert g.is_acyclic()
    bad = {k for k, _ in tg.audit_conservation(g).items()}
    neg = {a.nodes[0] for a in g.anomalies if a.kind in ("negative_fee", "missing_input")}
    assert bad <= neg
    assert all(o.value > 0 for t in mempool for o in t.outputs)
    assert not [a for a in g.anomalies if a.kind in ("negative_fee", "missing_input")]
    for tx in mempool:
        node = g.nodes[f"btx:{tx.txid}"]
        value_in = sum((utxos.get(op) or next(t.outputs[op[1]] for t in mempool if t.txid == op[0])).value
                       for op in tx.inputs)
        assert node.attrs["fee_sat"] == value_in - tx.output_value


def test_btc_conflicts_and_replacements(btc):
    kinds = [a.kind for a in btc.anomalies]
    assert kinds.count("double_spend") >= 1
    assert "rbf_ok" in kinds and "rbf_bad" in kinds
    ok = next(a for a in btc.anomalies if a.kind == "rbf_ok")
    assert ok.amount > 0 and "заменяет" in ok.message
    assert btc.meta["replaced"] >= 2


@pytest.mark.parametrize("orig_rbf,orig_fee,repl_fee,expected", [
    (False, 100, 10_000, "rbf.not_signaled"),
    (True, 1_000, 1_050, "rbf.fee_too_low"),
    (True, 1_000, 1_000 + 110, None),
    (True, 1_000, 1_109, "rbf.fee_too_low"),
])
def test_bip125_rules(orig_rbf, orig_fee, repl_fee, expected):
    orig = tg.BtcTx("o", (("p", 0),), (tg.TxOut("a", 1),), orig_rbf)
    repl = tg.BtcTx("r", (("p", 0),), (tg.TxOut("b", 1),), True)
    assert tg._bip125(orig, orig_fee, repl, repl_fee) == expected


def test_bip125_rate_rule():
    orig = tg.BtcTx("o", (("p", 0),), (tg.TxOut("a", 1),), True)
    big = tg.BtcTx("r", (("p", 0), ("q", 0), ("s", 0)), (tg.TxOut("b", 1),) * 4, True)
    # pays more absolute fee (rule 3 and 4) but at a lower rate than the original
    assert tg._bip125(orig, 2_000, big, 2_000 + big.vsize) == "rbf.rate_too_low"


def test_block_template_rules(btc_data, btc):
    utxos, mempool = btc_data
    template = btc.meta["template"]
    chosen = template["txids"]
    by_id = {t.txid: t for t in mempool}
    replaced = {n.id[4:] for n in btc.nodes.values() if n.attrs.get("replaced")}
    assert not set(chosen) & replaced
    pos = {t: i for i, t in enumerate(chosen)}
    for t in chosen:
        for op in by_id[t].inputs:
            if op[0] in pos:
                assert pos[op[0]] < pos[t]
    spent = [op for t in chosen for op in by_id[t].inputs]
    assert len(spent) == len(set(spent))  # no double spend inside the block
    fees = {n.id[4:]: n.attrs["fee_sat"] for n in btc.nodes.values() if n.kind == "utxo_tx"}
    assert template["fees_sat"] == sum(fees[t] for t in chosen)
    assert template["vsize"] == sum(by_id[t].vsize for t in chosen)


def test_block_template_respects_vsize_limit(btc_data):
    utxos, mempool = btc_data
    g = tg.build_btc_graph(utxos, mempool)
    fees = {n.id[4:]: n.attrs["fee_sat"] for n in g.nodes.values() if n.kind == "utxo_tx"}
    small = tg.block_template(mempool, fees, (), vsize_limit=400)
    assert 0 < small["vsize"] <= 400 and small["txids"]


def test_block_template_prefers_cpfp_package():
    parent = tg.BtcTx("p" * 64, (("f", 0),), (tg.TxOut("a", 1000),))
    child = tg.BtcTx("c" * 64, ((parent.txid, 0),), (tg.TxOut("b", 1),))
    other = tg.BtcTx("o" * 64, (("f", 1),), (tg.TxOut("c", 1),))
    fees = {parent.txid: parent.vsize * 1, child.txid: child.vsize * 50, other.txid: other.vsize * 10}
    t = tg.block_template([parent, child, other], fees)
    assert t["txids"][:2] == [parent.txid, child.txid]
    assert Fraction(fees[parent.txid] + fees[child.txid], parent.vsize + child.vsize) > 10


def test_missing_input_and_negative_fee():
    utxos = {("f", 0): tg.TxOut("a", 1_000)}
    txs = [tg.BtcTx("t1", (("f", 0),), (tg.TxOut("b", 2_000),)), tg.BtcTx("t2", (("zz", 0),), (tg.TxOut("c", 5),))]
    g = tg.build_btc_graph(utxos, txs)
    kinds = sorted(a.kind for a in g.anomalies)
    assert "negative_fee" in kinds and "missing_input" in kinds and "dust" in kinds
    neg = next(a for a in g.anomalies if a.kind == "negative_fee")
    assert neg.amount == 1_000


def test_common_input_clustering():
    utxos = {("f", 0): tg.TxOut("a1", 5_000), ("f", 1): tg.TxOut("a2", 5_000), ("f", 2): tg.TxOut("a3", 5_000)}
    txs = [tg.BtcTx("t1", (("f", 0), ("f", 1)), (tg.TxOut("x", 9_000),))]
    g = tg.build_btc_graph(utxos, txs)
    clusters = {n.attrs["address"]: n.attrs["cluster"] for n in g.nodes.values() if n.kind == "utxo"}
    assert clusters["a1"] == clusters["a2"]


def test_generate_btc_mempool_rejects_tiny_inputs():
    with pytest.raises(tg.ConfigError):
        tg.generate_btc_mempool(0, txs=2)


# -- rendering ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("asset,amount,lang,text", [
    ("WETH", 1_500_000_000_000_000_000, "en", "1.50000000 WETH"),
    ("USDC", 1_234_567, "ru", "1,234567 USDC"),
    ("BTC", 150_000_000, "en", "1.50000000 BTC"),
    ("FOO", 1234, "en", "1,234 FOO"),
    ("state", 0, "en", ""),
])
def test_format_amount(asset, amount, lang, text):
    assert tg.format_amount(asset, amount, lang) == text


def test_render_html_structure(eth, btc):
    page = tg.render_html([eth, btc], "ru")
    assert page.startswith('<!doctype html><html lang="ru">')
    assert "Граф транзакций" in page and "Эфириум" in page and "Биткоин" in page
    assert page.count('<svg class="graph"') == 2
    assert "http://" not in page.replace('xmlns="http://www.w3.org/2000/svg"', "") and "https://" not in page
    data = re.search(r'<script type="application/json" id="graph-data">(.*?)</script>', page, re.S).group(1)
    parsed = json.loads(data)
    assert set(parsed) == {"eth", "btc"}
    assert set(parsed["eth"]["succ"]) == set(eth.nodes)


def test_render_svg_draws_every_node_and_edge(eth):
    svg = tg.render_svg(eth)
    assert svg.count('<g class="node') == len(eth.nodes)
    assert svg.count('<path class="edge') == len(eth.edges)


def test_render_html_escapes_hostile_labels():
    g = tg.TxGraph("eth", "</script><script>alert(1)</script>")
    g.add_node("a", "tx", "<img src=x onerror=alert(1)>")
    g.add_node("b", "account", "&<>\"'")
    g.add_edge("a", "b", "WETH", 1)
    page = tg.render_html([g], "en")
    assert "<img src=x" not in page and "<script>alert" not in page
    payload = re.search(r'id="graph-data">(.*?)</script>', page, re.S).group(1)
    assert "</script" not in payload and json.loads(payload)


def test_render_html_english(eth):
    page = tg.render_html([eth], "en")
    assert '<html lang="en">' in page and "Transaction graph" in page
