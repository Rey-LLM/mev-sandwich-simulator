"""Layer 16: sandwich detector on real block data.

Three kinds of evidence:

* a real mainnet block (12775690, call traces from flashbots/mev-inspect-py's
  test suite) with a known sandwich: four victims, every number checked
  against the raw token transfers in the traces;
* synthetic blocks encoded exactly as a node returns them (receipts and logs
  with the real event topics), whose expected losses are computed here,
  independently of the detector, with the AMM math;
* negative cases that look like sandwiches but are not.
"""

import copy
import gzip
import json
from pathlib import Path

import pytest

from simulator import sandwich_detector as sd
from simulator.__main__ import main as cli_main
from simulator.slippage_engine import (
    ConcentratedPool,
    Position,
    V2Pool,
    get_sqrt_ratio_at_tick,
    v2_get_amount_in,
    v2_get_amount_out,
)

FIXTURE = Path(__file__).parent / "vectors" / "mainnet_block_12775690_traces.json.gz"

MINER = "0x00000000000000000000000000000000000000ee"
TOKEN = "0x1111111111111111111111111111111111111111"  # sorts before WETH -> token0
USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
WETH = sd.WETH
ROUTER = "0x7a250d5630b4cf539739df2c5dacb4c659f2488d"
BOT_EOA = "0x000000000000000000000000000000000000b075"
BOT = "0x00000000000000000000000000000000000b0b0b"
ALICE = "0x000000000000000000000000000000000000a11c"
BOB = "0x0000000000000000000000000000000000000b0b"
CAROL = "0x000000000000000000000000000000000000ca01"
PAIR = "0x2222222222222222222222222222222222222222"
V3POOL = "0x3333333333333333333333333333333333333333"


# ---------------------------------------------------------------------------
# Keccak and constants
# ---------------------------------------------------------------------------


def test_keccak_matches_known_vectors():
    assert sd.keccak256(b"").hex() == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    assert sd.keccak256(b"abc").hex() == "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"
    # The function selector every ERC-20 shares, from a known signature.
    assert sd.keccak256(b"transfer(address,uint256)")[:4].hex() == "a9059cbb"


def test_keccak_rate_boundary_padding():
    # 135 bytes leaves room for exactly one padding byte (0x81); 136 needs a full extra block.
    assert len(sd.keccak256(b"x" * 135)) == 32
    assert sd.keccak256(b"x" * 135) != sd.keccak256(b"x" * 136)


@pytest.mark.parametrize("value, literal", [
    (sd.TOPIC_V2_SWAP, "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"),
    (sd.TOPIC_V2_SYNC, "0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1"),
    (sd.TOPIC_V3_SWAP, "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"),
    (sd.TOPIC_V3_MINT, "0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde"),
    (sd.TOPIC_V3_BURN, "0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c"),
    (sd.event_topic("Transfer(address,address,uint256)"),
     "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"),
    (sd.SEL_TRANSFER, "0xa9059cbb"),
    (sd.SEL_TRANSFER_FROM, "0x23b872dd"),
    (sd.SEL_V2_SWAP, "0x022c0d9f"),
    (sd.SEL_V3_SWAP, "0x128acb08"),
    (sd.SEL_GET_RESERVES, "0x0902f1ac"),
    (sd.SEL_WETH_DEPOSIT, "0xd0e30db0"),
    (sd.SEL_WETH_WITHDRAW, "0x2e1a7d4d"),
    (sd.SEL_TOKEN0, "0x0dfe1681"),
    (sd.SEL_TOKEN1, "0xd21220a7"),
    (sd.SEL_FEE, "0xddca3f43"),
])
def test_event_topics_and_selectors_are_the_real_ones(value, literal):
    assert value == literal


# ---------------------------------------------------------------------------
# Real mainnet block 12775690
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mainnet_export():
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def mainnet_report(mainnet_export):
    (report,) = sd.detect_export(mainnet_export)
    return report


def _raw_transfers(export, tx_index, token):
    """Every successful ERC-20 transfer of ``token`` in a transaction, read straight from the traces."""
    out = []
    for tr in export["traces"]:
        if tr.get("transaction_position") != tx_index or tr.get("error"):
            continue
        a = tr["action"]
        inp = a.get("input", "")
        if a.get("to") != token:
            continue
        if inp.startswith("0xa9059cbb"):
            out.append((a["from"], "0x" + inp[34:74], int(inp[74:138], 16)))
        elif inp.startswith("0x23b872dd"):
            out.append(("0x" + inp[34:74], "0x" + inp[98:138], int(inp[138:202], 16)))
    return out


def test_mainnet_block_has_exactly_one_sandwich(mainnet_report):
    s = mainnet_report.summary()
    assert s["block"] == 12775690
    assert s["sandwiches"] == 1
    assert s["victims"] == 4
    assert s["exact_victims"] == 4
    assert mainnet_report.warnings == []


def test_mainnet_sandwich_parties(mainnet_report):
    (f,) = mainnet_report.findings
    assert f.pool == "0xefb47fcfcad4f96c83d4ca676842fb03ef20a477"
    assert f.version == "v2"
    assert f.fee == 30
    assert f.attacker_eoa == "0x0000000000c521824eaff97eac7b73b084ef9306"
    assert f.attacker_contract == "0x000000000027d2efc283613d0c3e24a8b430c4d8"
    assert f.front_tx == "0x91a3abe5f3b806426542252820ba0ab6d56c098fdef6864ecaf4d352f64217a0"
    assert f.back_tx == "0xc300d1ff79d3901b58dc56489fc7d083a6c13d422bfc1425a0579379300c95a2"
    assert (f.front_index, f.back_index) == (2, 7)
    assert [v.tx_index for v in f.victims] == [3, 4, 5, 6]
    assert f.victims[0].tx_hash == "0x9b40deca1f53593b7631ca25485d0c6faf90279b9872845acfd5c98afb185934"
    assert f.token_a == WETH
    assert f.token_b == "0x9813037ee2218799597d83d4a5b6f3b6778218d9"  # BONE


def test_mainnet_profit_matches_raw_transfers(mainnet_export, mainnet_report):
    (f,) = mainnet_report.findings
    bot, pair = f.attacker_contract, f.pool
    sent = sum(amt for src, dst, amt in _raw_transfers(mainnet_export, 2, WETH) if src == bot and dst == pair)
    got = sum(amt for src, dst, amt in _raw_transfers(mainnet_export, 7, WETH) if src == pair and dst == bot)
    assert (f.front_in, f.back_out) == (sent, got)
    assert f.gross_profit == got - sent == 49_991_481_915_322_274
    assert f.leftover_b == 0  # sold back exactly what it bought
    assert f.transfer_check == {"net_token_a_flow": f.gross_profit, "matches_gross_profit": True}
    # Flashbots-era bundle: 1 wei gas price, paid the miner directly from the back-run.
    assert f.gas_cost_wei == 103_303 + 111_682
    assert f.coinbase_transfer_wei == 0x70B1B725FE030C
    assert f.net_profit_wei == f.gross_profit - f.gas_cost_wei - f.coinbase_transfer_wei == 18_270_884_348_978_381


def test_mainnet_victim_losses(mainnet_report):
    (f,) = mainnet_report.findings
    v3, v4, v5, v6 = f.victims
    # tx 3 is swapETHForExactTokens: it fixed its output and overpaid input.
    assert v3.mode == "exact_out" and v3.amount_out == 60 * 10 ** 18
    assert v3.extra_in == v3.amount_in - v3.counterfactual_in == 2_580_635_624_826_764
    assert v3.loss_wei == v3.extra_in
    # txs 4-6 are swapExactETHForTokens: fixed input, lost output.
    for v in (v4, v5, v6):
        assert v.mode == "exact_in" and v.exact
        assert v.loss == v.counterfactual_out - v.amount_out > 0
        assert v.token_in == WETH and v.loss_wei == v.loss * v.amount_in // v.counterfactual_out
    assert v6.amount_in == 30 * 10 ** 18
    assert v6.loss == 10_855_006_322_639_006_651
    assert f.victim_loss_wei == sum(v.loss_wei for v in f.victims) == 122_533_327_961_626_097
    assert all(39 < v.loss_bps < 40 for v in f.victims)


def test_mainnet_victim_counterfactual_by_hand(mainnet_export, mainnet_report):
    """Recompute victim 4 from getReserves() outputs in the raw traces, without the detector's timeline."""
    (f,) = mainnet_report.findings

    def reserves_at(tx_index):
        for tr in mainnet_export["traces"]:
            a = tr["action"]
            if tr.get("transaction_position") == tx_index and a.get("to") == f.pool \
                    and a.get("input", "").startswith("0x0902f1ac"):
                out = tr["result"]["output"]
                return int(out[2:66], 16), int(out[66:130], 16)
        raise AssertionError(tx_index)

    r0, r1 = reserves_at(2)  # before the front-run: token0 = BONE, token1 = WETH
    victim3 = f.victims[0]
    # Without the front-run, victim 3 (exact out) would have paid getAmountIn on these reserves...
    cf_in3 = v2_get_amount_in(victim3.amount_out, r1, r0)
    assert cf_in3 == victim3.counterfactual_in
    # ...and victim 4 then trades on the reserves victim 3 leaves behind.
    r0, r1 = r0 - victim3.amount_out, r1 + cf_in3
    v4 = f.victims[1]
    assert v2_get_amount_out(v4.amount_in, r1, r0) == v4.counterfactual_out
    # The real victim 4 traded on the post-front-run reserves read in its own transaction.
    a0, a1 = reserves_at(4)
    assert v2_get_amount_out(v4.amount_in, a1, a0) == v4.amount_out


def test_mainnet_v2_timeline_matches_every_getreserves(mainnet_export):
    """Each swap's reconstructed post-state equals the next getReserves() read on that pair."""
    (data,) = sd.load_export(mainnet_export)
    anchors = {}
    for a in data.anchors:
        if a.reserves is not None:
            anchors.setdefault(a.pool, []).append(a)
    checked = 0
    for s in data.swaps:
        if s.version != "v2" or s.reserves_after is None:
            continue
        nxt = [a for a in anchors.get(s.pool, []) if a.pos > s.pos]
        prev_swaps = [o for o in data.swaps if o.pool == s.pool and s.pos < o.pos < (nxt[0].pos if nxt else (1 << 30, 0))]
        if nxt and not prev_swaps:
            assert s.reserves_after == nxt[0].reserves
            checked += 1
    assert checked >= 6


def test_mainnet_logs_format_gives_the_same_result(mainnet_export, mainnet_report):
    """Re-encode the real swaps as node logs (Swap + Sync) and parse them with the logs parser."""
    (data,) = sd.load_export(mainnet_export)
    block, receipts = _logs_export_from_block(data)
    (logs_report,) = sd.detect_export({"block": block, "receipts": receipts, "pools": data.pools})
    a, b = logs_report.findings[0], mainnet_report.findings[0]
    assert len(logs_report.findings) == 1
    assert (a.front_in, a.front_out, a.back_in, a.back_out) == (b.front_in, b.front_out, b.back_in, b.back_out)
    strip = lambda v: {k: x for k, x in v.to_dict().items() if k != "tx_hash"}  # noqa: E731
    assert [strip(v) for v in a.victims] == [strip(v) for v in b.victims]
    # Logs carry no internal coinbase payments: the bribe is invisible there.
    assert a.coinbase_transfer_wei == 0 and a.net_profit_wei == b.net_profit_wei + b.coinbase_transfer_wei


# ---------------------------------------------------------------------------
# Synthetic blocks in node format
# ---------------------------------------------------------------------------


def _w(x):
    return format(x % (1 << 256), "064x")


def _t(addr):
    return "0x" + "0" * 24 + addr[2:]


class Chain:
    """Builds ``eth_getBlockByNumber`` + ``eth_getBlockReceipts`` objects."""

    def __init__(self, number=19_000_000, miner=MINER):
        self.number, self.miner = number, miner
        self.txs, self.receipts = [], []
        self._log_index = 0

    def tx(self, sender, to, logs=(), gas_used=150_000, gas_price=20 * 10 ** 9, value=0):
        idx = len(self.txs)
        h = "0x%064x" % (self.number * 1000 + idx)
        self.txs.append({"hash": h, "transactionIndex": hex(idx), "from": sender, "to": to, "value": hex(value),
                         "gasPrice": hex(gas_price), "blockNumber": hex(self.number)})
        out = []
        for lg in logs:
            out.append({**lg, "logIndex": hex(self._log_index), "transactionIndex": hex(idx),
                        "transactionHash": h})
            self._log_index += 1
        self.receipts.append({"transactionHash": h, "transactionIndex": hex(idx), "status": "0x1",
                              "from": sender, "to": to, "gasUsed": hex(gas_used),
                              "effectiveGasPrice": hex(gas_price), "logs": out})
        return h

    def export(self, pools=None):
        block = {"number": hex(self.number), "miner": self.miner, "transactions": self.txs}
        return {"block": block, "receipts": self.receipts, "pools": pools or {}}


def v2_logs(addr, pool, zfo, amount_in=None, amount_out=None, recipient=ALICE):
    """Apply a swap to a V2Pool and return its Sync + Swap logs (exact in, or exact out)."""
    if amount_out is None:
        res = pool.quote_exact_in(amount_in, zfo)
    else:
        res = pool.quote_exact_out(amount_out, zfo)
    pool.apply_swap(res, zfo)
    a_in, a_out = res.amount_in, res.amount_out
    words = (a_in, 0, 0, a_out) if zfo else (0, a_in, a_out, 0)
    sync = {"address": addr, "topics": [sd.TOPIC_V2_SYNC], "data": "0x" + _w(pool.reserve0) + _w(pool.reserve1)}
    swap = {"address": addr, "topics": [sd.TOPIC_V2_SWAP, _t(ROUTER), _t(recipient)],
            "data": "0x" + "".join(_w(x) for x in words)}
    return [sync, swap], res


def v3_log(addr, pool, zfo, amount_in, recipient=ALICE):
    res = pool.apply_swap(amount_in, zfo)
    a0, a1 = (res.amount_in, -res.amount_out) if zfo else (-res.amount_out, res.amount_in)
    data = "0x" + _w(a0) + _w(a1) + _w(pool.sqrt_price_x96) + _w(pool.liquidity) + _w(pool.tick)
    return {"address": addr, "topics": [sd.TOPIC_V3_SWAP, _t(ROUTER), _t(recipient)], "data": data}, res


def _logs_export_from_block(data):
    chain = Chain(number=data.number, miner=data.miner)
    by_tx = {}
    for s in data.swaps:
        by_tx.setdefault(s.tx_index, []).append(s)
    for idx in sorted(data.txs):
        tx = data.txs[idx]
        logs = []
        for s in by_tx.get(idx, []):
            if s.version != "v2" or s.reserves_after is None:
                continue
            words = (s.amount_in, 0, 0, s.amount_out) if s.zero_for_one else (0, s.amount_in, s.amount_out, 0)
            logs.append({"address": s.pool, "topics": [sd.TOPIC_V2_SYNC],
                         "data": "0x" + _w(s.reserves_after[0]) + _w(s.reserves_after[1])})
            logs.append({"address": s.pool, "topics": [sd.TOPIC_V2_SWAP, _t(tx.to), _t(s.recipient)],
                         "data": "0x" + "".join(_w(x) for x in words)})
        while len(chain.txs) < idx:
            chain.tx("0x" + "0" * 40, "0x" + "0" * 40)
        chain.tx(tx.sender, tx.to, logs, gas_used=tx.gas_used, gas_price=tx.gas_price)
    exp = chain.export()
    return exp["block"], exp["receipts"]


def _v2_pool():
    # token0 = TOKEN (address sorts first), token1 = WETH; 2,000,000 TOKEN / 1,000 WETH.
    return V2Pool(TOKEN, WETH, 2_000_000 * 10 ** 18, 1_000 * 10 ** 18)


@pytest.fixture
def v2_block():
    """Front-run, exact-in victim, unrelated opposite swap, exact-out victim, back-run."""
    pool = _v2_pool()
    pre = (pool.reserve0, pool.reserve1)
    chain = Chain()
    zfo_buy = False  # WETH (token1) -> TOKEN (token0)
    chain.tx(CAROL, "0x" + "9" * 40)  # unrelated tx without swaps
    logs, front = v2_logs(PAIR, pool, zfo_buy, amount_in=20 * 10 ** 18, recipient=BOT)
    front_h = chain.tx(BOT_EOA, BOT, logs, gas_used=110_000, gas_price=30 * 10 ** 9)
    logs, victim1 = v2_logs(PAIR, pool, zfo_buy, amount_in=10 * 10 ** 18)
    v1_h = chain.tx(ALICE, ROUTER, logs)
    logs, other = v2_logs(PAIR, pool, True, amount_in=5_000 * 10 ** 18, recipient=CAROL)  # sells TOKEN
    chain.tx(CAROL, ROUTER, logs)
    logs, victim2 = v2_logs(PAIR, pool, zfo_buy, amount_out=8_000 * 10 ** 18, recipient=BOB)
    v2_h = chain.tx(BOB, ROUTER, logs)
    logs, back = v2_logs(PAIR, pool, True, amount_in=front.amount_out, recipient=BOT)
    back_h = chain.tx(BOT_EOA, BOT, logs, gas_used=100_000, gas_price=20 * 10 ** 9)
    return {"export": chain.export({PAIR: {"token0": TOKEN, "token1": WETH}}), "pre": pre, "front": front,
            "victim1": victim1, "other": other, "victim2": victim2, "back": back,
            "hashes": (front_h, v1_h, v2_h, back_h)}


def test_v2_logs_sandwich_detected(v2_block):
    (report,) = sd.detect_export(v2_block["export"])
    assert report.source == "logs"
    (f,) = report.findings
    front_h, v1_h, v2_h, back_h = v2_block["hashes"]
    assert (f.front_tx, f.back_tx) == (front_h, back_h)
    assert [v.tx_hash for v in f.victims] == [v1_h, v2_h]  # Carol's opposite-direction sell is not a victim
    assert f.link == "sender" and f.attacker_eoa == BOT_EOA and f.attacker_contract == BOT
    assert (f.token_a, f.token_b) == (WETH, TOKEN)
    assert f.gross_profit == v2_block["back"].amount_out - v2_block["front"].amount_in
    assert f.gas_cost_wei == 110_000 * 30 * 10 ** 9 + 100_000 * 20 * 10 ** 9
    assert f.net_profit_wei == f.gross_profit - f.gas_cost_wei
    assert f.profit_valuation == "exact_weth"


def test_v2_logs_victim_losses_match_independent_counterfactual(v2_block):
    (report,) = sd.detect_export(v2_block["export"])
    (f,) = report.findings
    r_tok, r_weth = v2_block["pre"]
    # Counterfactual world: no front-run; victim 1, Carol and victim 2 trade in the same order.
    cf1 = v2_get_amount_out(10 * 10 ** 18, r_weth, r_tok)
    r_tok, r_weth = r_tok - cf1, r_weth + 10 * 10 ** 18
    carol = v2_get_amount_out(5_000 * 10 ** 18, r_tok, r_weth)
    r_tok, r_weth = r_tok + 5_000 * 10 ** 18, r_weth - carol
    cf2_in = v2_get_amount_in(8_000 * 10 ** 18, r_weth, r_tok)
    v1, v2 = f.victims
    assert v1.mode == "exact_in" and v1.exact
    assert v1.counterfactual_out == cf1
    assert v1.loss == cf1 - v2_block["victim1"].amount_out
    assert v1.loss_wei == v1.loss * v1.amount_in // cf1
    assert v2.mode == "exact_out" and v2.exact
    assert v2.counterfactual_in == cf2_in
    assert v2.extra_in == v2_block["victim2"].amount_in - cf2_in > 0
    assert v2.loss_wei == v2.extra_in


def test_v2_fee_calibration_handles_a_fork_fee():
    pool = V2Pool(TOKEN, WETH, 10 ** 24, 10 ** 21, fee_bps=25)
    chain = Chain()
    logs, front = v2_logs(PAIR, pool, False, amount_in=10 ** 19, recipient=BOT)
    chain.tx(BOT_EOA, BOT, logs)
    logs, _ = v2_logs(PAIR, pool, False, amount_in=5 * 10 ** 18)
    chain.tx(ALICE, ROUTER, logs)
    logs, _ = v2_logs(PAIR, pool, True, amount_in=front.amount_out, recipient=BOT)
    chain.tx(BOT_EOA, BOT, logs)
    (f,) = sd.detect_export(chain.export({PAIR: {"token0": TOKEN, "token1": WETH}}))[0].findings
    assert f.fee == 25
    assert f.victims[0].exact


def test_unknown_tokens_still_detected_without_wei_values(v2_block):
    exp = copy.deepcopy(v2_block["export"])
    exp["pools"] = {}
    (f,) = sd.detect_export(exp)[0].findings
    assert f.token_a is None and f.gross_profit_wei is None and f.net_profit_wei is None
    assert all(v.loss_wei is None for v in f.victims)
    assert f.victims[0].loss > 0 and f.victims[0].exact


def test_non_weth_pair_profit_is_left_in_tokens():
    pool = V2Pool(TOKEN, USDC, 10 ** 24, 10 ** 12)
    chain = Chain()
    logs, front = v2_logs(PAIR, pool, True, amount_in=10 ** 21, recipient=BOT)
    chain.tx(BOT_EOA, BOT, logs)
    logs, _ = v2_logs(PAIR, pool, True, amount_in=10 ** 22)
    chain.tx(ALICE, ROUTER, logs)
    logs, _ = v2_logs(PAIR, pool, False, amount_in=front.amount_out, recipient=BOT)
    chain.tx(BOT_EOA, BOT, logs)
    (f,) = sd.detect_export(chain.export({PAIR: {"token0": TOKEN, "token1": USDC}}))[0].findings
    assert f.token_a == TOKEN and f.gross_profit > 0
    assert f.gross_profit_wei is None and f.victims[0].loss_wei is None


def test_token_a_not_weth_is_valued_at_pool_price():
    # token0 = TOKEN, token1 = WETH; attacker trades TOKEN -> WETH -> TOKEN.
    pool = _v2_pool()
    chain = Chain()
    logs, front = v2_logs(PAIR, pool, True, amount_in=50_000 * 10 ** 18, recipient=BOT)
    chain.tx(BOT_EOA, BOT, logs)
    logs, _ = v2_logs(PAIR, pool, True, amount_in=30_000 * 10 ** 18)
    chain.tx(ALICE, ROUTER, logs)
    logs, back = v2_logs(PAIR, pool, False, amount_in=front.amount_out, recipient=BOT)
    chain.tx(BOT_EOA, BOT, logs)
    (f,) = sd.detect_export(chain.export({PAIR: {"token0": TOKEN, "token1": WETH}}))[0].findings
    assert f.token_a == TOKEN and f.profit_valuation == "pool_price_after_back_run"
    assert f.gross_profit_wei == f.gross_profit * pool.reserve1 // pool.reserve0
    assert f.victims[0].token_out == WETH and f.victims[0].loss_wei == f.victims[0].loss


# -- negatives ---------------------------------------------------------------


def _three(front_sender, front_to, victim_sender, victim_to, back_sender, back_to, back_ratio=(1, 1), victim_zfo=False):
    pool = _v2_pool()
    chain = Chain()
    logs, front = v2_logs(PAIR, pool, False, amount_in=20 * 10 ** 18, recipient=BOT)
    chain.tx(front_sender, front_to, logs)
    logs, _ = v2_logs(PAIR, pool, victim_zfo, amount_in=10 * 10 ** 18 if not victim_zfo else 10 ** 22)
    chain.tx(victim_sender, victim_to, logs)
    logs, _ = v2_logs(PAIR, pool, True, amount_in=front.amount_out * back_ratio[0] // back_ratio[1], recipient=BOT)
    chain.tx(back_sender, back_to, logs)
    return sd.detect_export(chain.export({PAIR: {"token0": TOKEN, "token1": WETH}}))[0]


def test_baseline_shape_is_detected():
    assert len(_three(BOT_EOA, BOT, ALICE, ROUTER, BOT_EOA, BOT).findings) == 1


def test_contract_link_is_enough():
    report = _three(BOT_EOA, BOT, ALICE, ROUTER, "0x" + "5" * 40, BOT)
    assert report.findings[0].link == "contract"


def test_public_router_alone_does_not_link_two_users():
    assert _three(BOB, ROUTER, ALICE, ROUTER, CAROL, ROUTER).findings == []


def test_victim_in_the_opposite_direction_is_not_sandwiched():
    assert _three(BOT_EOA, BOT, ALICE, ROUTER, BOT_EOA, BOT, victim_zfo=True).findings == []


def test_victim_sent_by_the_attacker_is_not_a_victim():
    assert _three(BOT_EOA, BOT, BOT_EOA, ROUTER, BOT_EOA, BOT).findings == []


def test_back_run_must_sell_what_was_bought():
    assert _three(BOT_EOA, BOT, ALICE, ROUTER, BOT_EOA, BOT, back_ratio=(3, 10)).findings == []
    assert _three(BOT_EOA, BOT, ALICE, ROUTER, BOT_EOA, BOT, back_ratio=(12, 10)).findings == []
    assert len(_three(BOT_EOA, BOT, ALICE, ROUTER, BOT_EOA, BOT, back_ratio=(9, 10)).findings) == 1


def test_round_trip_without_victim_is_not_a_sandwich():
    pool = _v2_pool()
    chain = Chain()
    logs, front = v2_logs(PAIR, pool, False, amount_in=20 * 10 ** 18, recipient=BOT)
    chain.tx(BOT_EOA, BOT, logs)
    logs, _ = v2_logs(PAIR, pool, True, amount_in=front.amount_out, recipient=BOT)
    chain.tx(BOT_EOA, BOT, logs)
    assert sd.detect_export(chain.export())[0].findings == []


def test_swaps_on_other_pools_do_not_mix():
    pool_a, pool_b = _v2_pool(), _v2_pool()
    other = "0x4444444444444444444444444444444444444444"
    chain = Chain()
    logs, front = v2_logs(PAIR, pool_a, False, amount_in=20 * 10 ** 18, recipient=BOT)
    chain.tx(BOT_EOA, BOT, logs)
    logs, _ = v2_logs(other, pool_b, False, amount_in=10 * 10 ** 18)
    chain.tx(ALICE, ROUTER, logs)
    logs, _ = v2_logs(PAIR, pool_a, True, amount_in=front.amount_out, recipient=BOT)
    chain.tx(BOT_EOA, BOT, logs)
    assert sd.detect_export(chain.export())[0].findings == []


def test_failed_receipt_is_ignored(v2_block):
    exp = copy.deepcopy(v2_block["export"])
    for r in exp["receipts"]:
        if r["transactionHash"] == v2_block["hashes"][3]:
            r["status"] = "0x0"
    assert sd.detect_export(exp)[0].findings == []


# -- v3 ----------------------------------------------------------------------


def _v3_pool(ticks=False):
    sqrt_p = get_sqrt_ratio_at_tick(-80_000)  # token0 = TOKEN, token1 = WETH
    if ticks:
        return ConcentratedPool.from_positions(TOKEN, WETH, sqrt_p, [
            Position(-80_100, -79_900, 3 * 10 ** 22), Position(-81_000, -79_000, 10 ** 22)],
            fee_pips=500, tick_spacing=10, pool_id=V3POOL)
    return ConcentratedPool(TOKEN, WETH, sqrt_p, 10 ** 24, fee_pips=500, tick_spacing=10, pool_id=V3POOL)


def _v3_block(pool, front_in, victim_in):
    pre = copy.copy(pool)
    chain = Chain()
    lg, front = v3_log(V3POOL, pool, False, front_in, recipient=BOT)
    chain.tx(BOT_EOA, BOT, [lg])
    lg, victim = v3_log(V3POOL, pool, False, victim_in)
    chain.tx(ALICE, ROUTER, [lg])
    lg, back = v3_log(V3POOL, pool, True, front.amount_out, recipient=BOT)
    chain.tx(BOT_EOA, BOT, [lg])
    return chain.export({V3POOL: {"token0": TOKEN, "token1": WETH}}), pre, front, victim, back


def test_v3_reconstruction_recovers_the_exact_pre_price():
    pool = _v3_pool()
    start = pool.sqrt_price_x96
    lg, _ = v3_log(V3POOL, pool, False, 7 * 10 ** 18)
    (data,) = sd.load_export(Chain().export() | {"receipts": [{"transactionHash": "0x1", "transactionIndex": "0x0",
                                                              "status": "0x1", "logs": [lg | {"logIndex": "0x0"}]}]})
    (swap,) = data.swaps
    assert sd.reconstruct_v3_before(swap, 500) == start
    assert sd.reconstruct_v3_before(swap, 3000) is None  # wrong fee tier cannot reproduce it


def test_v3_sandwich_inside_one_range_is_exact():
    export, pre, front, victim, back = _v3_block(_v3_pool(), 30 * 10 ** 18, 10 * 10 ** 18)
    (f,) = sd.detect_export(export)[0].findings
    assert f.version == "v3" and f.fee == 500
    assert f.gross_profit == back.amount_out - front.amount_in
    (v,) = f.victims
    cf = copy.copy(pre).quote_exact_in(10 * 10 ** 18, False).amount_out
    assert v.exact and v.counterfactual_out == cf and v.loss == cf - victim.amount_out > 0
    assert v.loss_wei == v.loss * v.amount_in // cf


def test_v3_fee_from_metadata_is_used():
    export, *_ = _v3_block(_v3_pool(), 30 * 10 ** 18, 10 * 10 ** 18)
    export["pools"][V3POOL]["fee"] = 500
    (f,) = sd.detect_export(export)[0].findings
    assert f.fee == 500 and f.victims[0].exact


def test_v3_tick_crossing_is_not_reported_as_exact():
    # The 3e22 position ends at tick -79,900: the front-run stays below it, the victim crosses it.
    export, *_ = _v3_block(_v3_pool(ticks=True), 3 * 10 ** 18, 2 * 10 ** 18)
    (f,) = sd.detect_export(export)[0].findings
    (v,) = f.victims
    assert not v.exact
    assert "tick" in v.method


def test_v3_mint_between_breaks_nothing_in_detection():
    export, *_ = _v3_block(_v3_pool(), 30 * 10 ** 18, 10 * 10 ** 18)
    mint = {"address": V3POOL, "topics": [sd.TOPIC_V3_MINT], "data": "0x", "logIndex": "0x63"}
    export["receipts"][1]["logs"].append(mint)
    assert len(sd.detect_export(export)[0].findings) == 1


# ---------------------------------------------------------------------------
# JSON-RPC path (fake transport serving node-format objects)
# ---------------------------------------------------------------------------


class FakeNode:
    def __init__(self, export, block_receipts=True):
        self.block = export["block"]
        self.receipts = export["receipts"]
        self.block_receipts = block_receipts
        self.calls = []

    def __call__(self, method, params):
        self.calls.append(method)
        if method == "eth_blockNumber":
            return self.block["number"]
        if method == "eth_getBlockByNumber":
            assert params[1] is True
            return self.block if params[0] in (self.block["number"], "latest") else None
        if method == "eth_getBlockReceipts":
            if not self.block_receipts:
                raise sd.DetectorError("method not found")
            return self.receipts
        if method == "eth_getTransactionReceipt":
            return next(r for r in self.receipts if r["transactionHash"] == params[0])
        if method == "eth_call":
            to, data = params[0]["to"], params[0]["data"]
            if to == PAIR and data == sd.SEL_TOKEN0:
                return "0x" + _w(int(TOKEN, 16))
            if to == PAIR and data == sd.SEL_TOKEN1:
                return "0x" + _w(int(WETH, 16))
            if to == PAIR and data == sd.SEL_GET_RESERVES:
                return "0x" + _w(10 ** 24) + _w(10 ** 21) + _w(0)
            raise sd.DetectorError("execution reverted")
        raise AssertionError(method)


@pytest.mark.parametrize("block_receipts", [True, False])
def test_rpc_fetch_then_detect(v2_block, block_receipts):
    node = FakeNode(v2_block["export"], block_receipts)
    client = sd.JsonRpcClient(transport=node)
    export = sd.fetch_block_export(client, client.block_number())
    assert export["pools"][PAIR] == {"version": "v2", "token0": TOKEN, "token1": WETH}
    (report,) = sd.detect_export(export)
    assert len(report.findings) == 1 and report.findings[0].token_a == WETH
    assert ("eth_getTransactionReceipt" in node.calls) is not block_receipts


def test_export_file_round_trip(tmp_path, v2_block):
    for name in ("block.json", "block.json.gz"):
        path = tmp_path / name
        sd.write_export_file(str(path), {"blocks": [v2_block["export"]]})
        reports = sd.detect_export(sd.read_export_file(str(path)))
        assert [len(r.findings) for r in reports] == [1]


def test_rpc_requires_an_endpoint(monkeypatch):
    monkeypatch.delenv("ETH_RPC_URL", raising=False)
    with pytest.raises(sd.DetectorError):
        sd.JsonRpcClient()


def test_fetch_v2_pool_reads_reserves_and_tokens(v2_block):
    client = sd.JsonRpcClient(transport=FakeNode(v2_block["export"]))
    spec = sd.fetch_v2_pool(client, PAIR)
    assert spec == {"version": "v2", "address": PAIR, "token0": TOKEN, "token1": WETH,
                    "reserve0": 10 ** 24, "reserve1": 10 ** 21}


def test_unknown_export_format_is_rejected():
    with pytest.raises(sd.DetectorError):
        sd.load_export({"foo": 1})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_detect_file_on_the_mainnet_fixture(capsys):
    assert cli_main(["--detect-file", str(FIXTURE)]) == 0
    out = json.loads(capsys.readouterr().out)
    (block,) = out["blocks"]
    assert block["summary"]["sandwiches"] == 1
    assert block["sandwiches"][0]["attacker"]["contract"] == "0x000000000027d2efc283613d0c3e24a8b430c4d8"
    assert block["sandwiches"][0]["net_profit_wei"] == 18_270_884_348_978_381


def test_cli_detect_block_without_rpc_fails_cleanly(monkeypatch, capsys):
    monkeypatch.delenv("ETH_RPC_URL", raising=False)
    assert cli_main(["--detect-block", "latest"]) == 2
    assert "RPC" in capsys.readouterr().err


def test_cli_missing_file_fails_cleanly(tmp_path, capsys):
    assert cli_main(["--detect-file", str(tmp_path / "nope.json")]) == 2
