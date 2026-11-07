"""Layer 12: MEV pattern ingestion, the pattern store, slippage evaluation and pattern-seeded AI bots."""

import json
import os
import random
import threading
from pathlib import Path

import numpy as np
import pytest

from simulator.__main__ import main as cli_main
from simulator.ai_agent_factory import (
    ARENA_POOLS,
    CATEGORY_HOP_POLICY,
    CATEGORY_STRATEGY,
    FACTORY_RULES,
    N_GENES,
    AgentFactory,
    PatternPrior,
    decode_profile,
    encode_profile,
    pattern_latent,
    profile_from_pattern,
    profile_issues,
)
from simulator.config import PRESETS, ConfigError, RunMode, SimulationConfig, detect_cpu_count
from simulator.controller import SimulationController, ingest_patterns, invariants_ok
from simulator.parallel import WorkerPool
from simulator.pattern_ingestor import (
    CATEGORIES,
    HEADER,
    MATRIX_FIELDS,
    NOMINAL_RESERVE_WEI,
    ByteLRU,
    CorruptFrameError,
    IngestReport,
    PatternError,
    PatternRecord,
    PatternStore,
    ScanResult,
    StorageFullError,
    _as_int,
    classify,
    decode_frame,
    detect_source,
    encode_frame,
    evaluate_pattern,
    iter_lines,
    iter_raw,
    normalize,
    parallel_scan,
    parse_advisory,
    parse_flashbots_block,
    parse_mev_inspect,
    parse_normalized,
    pattern_weight,
    record_issues,
    scan_shards,
    synthetic_corpus,
    synthetic_record,
    write_jsonl,
)
from simulator.slippage_engine import V2Pool
from simulator.validator import Collector, check_signals

E18 = 10**18
GWEI = 10**9
CPUS = detect_cpu_count()
needs_two_cpus = pytest.mark.skipif(CPUS < 2, reason="needs at least two usable CPUs")


def rec(**kw):
    base = dict(category="multi_pool_sandwich", source="normalized", victim_tolerance_bps=100,
                frontrun_impact_bps=40, victim_size_bps=50, reserve_in_wei=1_000 * E18, bribe_bps=8_000,
                priority_fee_wei=3 * GWEI, latency_ms=150, capital_wei=20 * E18)
    base.update(kw)
    return PatternRecord(**base)


@pytest.fixture(scope="module")
def corpus():
    return synthetic_corpus(240, seed=11)


@pytest.fixture(scope="module")
def store_root(tmp_path_factory, corpus):
    root = tmp_path_factory.mktemp("patterns") / "store"
    with PatternStore(root, segment_bytes=16 * 1024) as st:
        st.add(corpus)
    return root


@pytest.fixture(scope="module")
def full_pool():
    with WorkerPool(policy="full", parallel_threshold=1, instrument=True) as wp:
        yield wp


@pytest.fixture
def fresh(tmp_path):
    with PatternStore(tmp_path / "s", segment_bytes=8 * 1024) as st:
        yield st


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("category", CATEGORIES)
def test_record_round_trips_through_canonical_bytes(category):
    r = synthetic_record(category, random.Random(category))
    back = PatternRecord.from_bytes(r.to_bytes())
    assert back == r and back.digest() == r.digest()


@pytest.mark.parametrize("field,value", [
    ("hops", 2), ("pools", 3), ("victim_tolerance_bps", 51), ("frontrun_impact_bps", 41), ("bribe_bps", 7_999),
    ("priority_fee_wei", 4 * GWEI), ("latency_ms", 151), ("capital_wei", 21 * E18), ("block_number", 9),
    ("tx_hashes", ("0xabc",)), ("reference", "x"), ("source", "advisory"),
])
def test_digest_changes_with_every_field(field, value):
    assert rec().digest() != rec(**{field: value}).digest()


def test_digest_is_stable_and_order_independent():
    a = rec()
    b = PatternRecord.from_canonical(dict(reversed(list(a.canonical().items()))))
    assert a.digest() == b.digest() == rec().digest()


def test_big_integers_survive_json_as_strings():
    r = rec(extracted_wei=2**200 + 1)
    assert r.canonical()["extracted_wei"] == str(2**200 + 1)
    assert PatternRecord.from_bytes(r.to_bytes()).extracted_wei == 2**200 + 1


@pytest.mark.parametrize("kw", [
    {"category": "laser"}, {"source": "rumour"}, {"hops": 0}, {"hops": 9}, {"pools": 0}, {"pools": 17},
    {"victim_tolerance_bps": 10_001}, {"bribe_bps": -1}, {"frontrun_impact_bps": 20_000}, {"latency_ms": -5},
    {"latency_ms": 60_001}, {"chain_id": 0}, {"capital_wei": -1}, {"extracted_wei": 2**256}, {"drift_blocks": 10_001},
    {"hops": 1.0}, {"hops": True}, {"tx_hashes": ["0x1"]}, {"reference": 5},
])
def test_invalid_records_are_rejected(kw):
    with pytest.raises(PatternError):
        rec(**kw)


def test_record_issues_lists_every_problem():
    class Fake:
        category, source, tx_hashes, reference = "x", "y", (), ""
        chain_id = block_number = dst_chain_id = drift_blocks = 0
        hops = pools = 0
        victim_tolerance_bps = victim_slippage_bps = frontrun_impact_bps = victim_size_bps = bribe_bps = 0
        reserve_in_wei = extracted_wei = priority_fee_wei = capital_wei = jit_liquidity_wei = latency_ms = 0

    issues = record_issues(Fake())
    assert len(issues) >= 5 and any("category" in i for i in issues) and any("hops" in i for i in issues)


def test_unknown_canonical_fields_are_rejected():
    with pytest.raises(PatternError, match="unknown pattern fields"):
        PatternRecord.from_canonical({**rec().canonical(), "colour": 1})


@pytest.mark.parametrize("raw,want", [
    (5, 5), ("12", 12), ("1_000", 1000), ("0x10", 16), (3.0, 3), ("2e3", 2000), (" 7 ", 7), ("-4", -4),
])
def test_as_int_accepts_whole_numbers(raw, want):
    assert _as_int(raw) == want


@pytest.mark.parametrize("raw", [1.5, "abc", float("nan"), float("inf"), None, "0xzz", [1]])
def test_as_int_rejects_everything_else(raw):
    with pytest.raises(PatternError):
        _as_int(raw)


@pytest.mark.parametrize("args,want", [
    (dict(hops=1, pools=1), "multi_pool_sandwich"),
    (dict(hops=1, pools=4), "multi_pool_sandwich"),
    (dict(hops=2, pools=2), "multi_hop_frontrun"),
    (dict(hops=1, pools=1, jit_liquidity_wei=1), "jit_liquidity"),
    (dict(hops=3, pools=3, jit_liquidity_wei=1), "jit_liquidity"),
    (dict(hops=3, pools=1, chain_id=1, dst_chain_id=10), "cross_chain_drift"),
    (dict(hops=1, pools=1, drift_blocks=2, jit_liquidity_wei=5), "cross_chain_drift"),
    (dict(hops=1, pools=1, chain_id=10, dst_chain_id=10), "multi_pool_sandwich"),
])
def test_classify_follows_the_documented_order(args, want):
    assert classify(**args) == want


# ---------------------------------------------------------------------------
# Normalisers
# ---------------------------------------------------------------------------


def fb_block(bundles):
    txs = []
    for b, eoas in enumerate(bundles):
        for i, eoa in enumerate(eoas):
            txs.append({"transaction_hash": f"0x{b}{i}", "tx_index": b * 10 + i, "bundle_index": b,
                        "eoa_address": eoa, "coinbase_transfer": str(10**16) if i == len(eoas) - 1 else "0",
                        "total_miner_reward": str(2 * 10**16) if i == len(eoas) - 1 else str(10**15),
                        "gas_price": str(5 * GWEI)})
    return {"block_number": 17_500_000, "miner_reward": "1", "transactions": txs}


@pytest.mark.parametrize("raw,want", [
    (fb_block([]), "flashbots_blocks"),
    ({"frontrun_swap_transaction_hash": "0x1"}, "mev_inspect"),
    ({"sandwiched_swaps": []}, "mev_inspect"),
    ({"advisory_id": "A-1"}, "advisory"),
    ({"source": "synthetic", "hops": 1}, "synthetic"),
    ({"category": "jit_liquidity"}, "normalized"),
])
def test_detect_source(raw, want):
    assert detect_source(raw) == want


@pytest.mark.parametrize("raw", [{"foo": 1}, [1, 2], "text"])
def test_detect_source_rejects_unknown_shapes(raw):
    with pytest.raises(PatternError):
        detect_source(raw)


def test_flashbots_sandwich_bundle_becomes_a_pattern():
    out = parse_flashbots_block(fb_block([["0xA", "0xV", "0xA"]]))
    assert len(out) == 1
    r = out[0]
    assert r.source == "flashbots_blocks" and r.block_number == 17_500_000 and r.category == "multi_pool_sandwich"
    assert r.bribe_bps == 10**16 * 10_000 // (2 * 10**16 + 10**15 * 2)
    assert r.tx_hashes == ("0x00", "0x01", "0x02") and r.priority_fee_wei == 5 * GWEI


@pytest.mark.parametrize("bundle", [["0xA", "0xA"], ["0xA", "0xV", "0xB"], ["0xA", "0xA", "0xA"]])
def test_flashbots_non_sandwich_bundles_are_skipped(bundle):
    assert parse_flashbots_block(fb_block([bundle])) == []


def test_flashbots_transactions_outside_bundles_are_ignored():
    block = fb_block([["0xA", "0xV", "0xA"]])
    block["transactions"].append({"transaction_hash": "0xff", "tx_index": 99, "eoa_address": "0xZ"})
    assert len(parse_flashbots_block(block)) == 1


def test_flashbots_wide_sandwich_counts_victims_as_pools():
    out = parse_flashbots_block(fb_block([["0xA", "0xV", "0xW", "0xX", "0xA"]]))
    assert out[0].pools == 3


def mi_row(**kw):
    row = {"block_number": 16_000_001, "sandwicher_address": "0xS", "frontrun_swap_transaction_hash": "0xF",
           "backrun_swap_transaction_hash": "0xB", "profit_amount": str(3 * 10**17), "reserve_in": str(1_000 * E18),
           "frontrun_swap": {"contract_address": "0xP1", "token_in_amount": str(5 * E18)},
           "backrun_swap": {"gas_price": str(9 * GWEI)},
           "sandwiched_swaps": [{"transaction_hash": "0xV", "contract_address": "0xP1",
                                 "token_in_amount": str(2 * E18)}]}
    row.update(kw)
    return row


def test_mev_inspect_row_is_normalised():
    r = parse_mev_inspect(mi_row())
    assert r.frontrun_impact_bps == 50 and r.victim_size_bps == 20 and r.extracted_wei == 3 * 10**17
    assert r.tx_hashes == ("0xF", "0xV", "0xB") and r.priority_fee_wei == 9 * GWEI and r.pools == 1


def test_mev_inspect_multi_pool_and_multi_hop():
    victims = [{"transaction_hash": "0xV1", "contract_address": "0xP2", "path": ["a", "b", "c"]},
               {"transaction_hash": "0xV2", "contract_address": "0xP3"}]
    r = parse_mev_inspect(mi_row(sandwiched_swaps=victims))
    assert r.pools == 3 and r.hops == 3 and r.category == "multi_hop_frontrun"


def test_mev_inspect_negative_profit_is_clamped():
    assert parse_mev_inspect(mi_row(profit_amount="-5")).extracted_wei == 0


def test_advisory_cross_chain_entry():
    r = parse_advisory({"advisory_id": "SEC-7", "chain_id": 1, "dst_chain_id": 42161,
                        "attack": {"hops": 2, "latency_ms": 4_000, "drift_blocks": 3}})
    assert r.category == "cross_chain_drift" and r.reference == "advisory:SEC-7" and r.latency_ms == 4_000


def test_advisory_explicit_category_wins():
    r = parse_advisory({"advisory_id": "SEC-8", "category": "jit_liquidity", "attack": {"hops": 2}})
    assert r.category == "jit_liquidity"


def test_advisory_unknown_attack_field_is_rejected():
    with pytest.raises(PatternError):
        parse_advisory({"advisory_id": "SEC-9", "attack": {"warp_speed": 9}})


def test_normalized_csv_cells():
    r = parse_normalized({"category": "", "hops": "2", "pools": "2", "tx_hashes": "0x1|0x2", "reference": "r",
                          "bribe_bps": "", "latency_ms": "300"})
    assert r.category == "multi_hop_frontrun" and r.tx_hashes == ("0x1", "0x2") and r.bribe_bps == 0


def test_normalize_clamps_out_of_range_inputs():
    r = normalize({"category": "multi_pool_sandwich", "hops": 40, "pools": 99, "bribe_bps": 50_000,
                   "latency_ms": 10**9})[0]
    assert (r.hops, r.pools, r.bribe_bps, r.latency_ms) == (8, 16, 10_000, 60_000)


def test_normalize_with_forced_source():
    assert normalize(mi_row(), "mev_inspect")[0].source == "mev_inspect"
    with pytest.raises(PatternError):
        normalize({}, "telepathy")


# ---------------------------------------------------------------------------
# Chunked readers
# ---------------------------------------------------------------------------

TEXT = "alpha\nbeta\r\n\ngamma delta\n" + "x" * 5000 + "\nlast"


@pytest.mark.parametrize("chunk", [1, 2, 7, 64, 4096, 1 << 20])
def test_iter_lines_matches_splitlines_for_any_chunk(tmp_path, chunk):
    p = tmp_path / "t.txt"
    p.write_bytes(TEXT.encode())
    assert list(iter_lines(p, chunk)) == TEXT.splitlines()


def test_iter_lines_utf8_split_across_chunks(tmp_path):
    p = tmp_path / "u.txt"
    p.write_bytes("ünïcødé\n€€€\n".encode())
    assert list(iter_lines(p, 1)) == ["ünïcødé", "€€€"]


def test_iter_lines_rejects_bad_chunk(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("a", encoding="utf-8")
    with pytest.raises(PatternError):
        list(iter_lines(p, 0))


@pytest.mark.parametrize("name,content,n", [
    ("a.json", json.dumps([{"category": "jit_liquidity"}, {"hops": 2}]), 2),
    ("b.json", json.dumps({"blocks": [fb_block([["0xA", "0xV", "0xA"]])]}), 1),
    ("c.jsonl", "\n".join(json.dumps({"hops": i}) for i in range(1, 5)) + "\n\n", 4),
    ("d.csv", "category,hops,pools\nmulti_hop_frontrun,2,2\n,1,1\n", 2),
    ("e.data", "\n".join(json.dumps({"hops": 1}) for _ in range(3)), 3),
])
def test_iter_raw_reads_every_format(tmp_path, name, content, n):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    assert len(list(iter_raw(p, 3))) == n


def test_ingest_file_keeps_going_past_bad_rows(fresh, tmp_path):
    p = tmp_path / "mixed.jsonl"
    rows = [{"hops": 1}, {"hops": "zz"}, {"category": "nope"}, {"hops": 2, "pools": 2}, {"what": 1}]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    rep = fresh.ingest_file(p)
    assert rep.added == 2 and rep.rejected == 3 and rep.raw_objects == 5
    assert all(str(p) in e for e in rep.errors) and ":2:" in rep.errors[0]


def test_ingest_report_merge_and_dict():
    a = IngestReport(added=1, by_category={"x": 1}, errors=["e1"])
    a.merge(IngestReport(added=2, duplicates=1, by_category={"x": 2, "y": 1}, errors=["e2", "e3"]), error_limit=2)
    d = a.to_dict()
    assert d["added"] == 3 and d["duplicates"] == 1 and d["by_category"] == {"x": 3, "y": 1}
    assert d["errors"] == ["e1", "e2"]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_lru_evicts_least_recent_first():
    c = ByteLRU(10)
    c.put("a", 1, 4)
    c.put("b", 2, 4)
    c.get("a")
    c.put("c", 3, 4)
    assert "a" in c and "c" in c and "b" not in c and c.used == 8 and c.evictions == 1


def test_lru_does_not_keep_oversized_values():
    c = ByteLRU(10)
    assert not c.put("big", 1, 11) and len(c) == 0


def test_lru_refresh_replaces_size():
    c = ByteLRU(10)
    c.put("a", 1, 6)
    c.put("a", 2, 3)
    assert c.get("a") == 2 and c.used == 3


def test_lru_stats_and_clear():
    c = ByteLRU(100)
    c.put("a", 1, 1)
    c.get("a")
    c.get("b")
    assert c.stats()["hit_rate"] == 0.5
    c.clear()
    assert len(c) == 0 and c.used == 0


@pytest.mark.parametrize("bad", [-1])
def test_lru_rejects_negative_sizes(bad):
    with pytest.raises(PatternError):
        ByteLRU(bad)
    with pytest.raises(PatternError):
        ByteLRU(10).put("a", 1, bad)


def test_lru_is_consistent_under_concurrent_writers():
    c = ByteLRU(1_000)

    def work(t):
        for i in range(2_000):
            c.put((t, i % 97), i, 7)
            c.get((t, (i * 31) % 97))

    threads = [threading.Thread(target=work, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert c.used == 7 * len(c) <= 1_000


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def test_frame_round_trip():
    r = rec(tx_hashes=("0x1", "0x2"))
    assert decode_frame(encode_frame(r)) == r


@pytest.mark.parametrize("damage", ["magic", "crc", "truncate", "short"])
def test_corrupt_frames_are_detected(damage):
    f = bytearray(encode_frame(rec()))
    if damage == "magic":
        f[0] ^= 0xFF
    elif damage == "crc":
        f[-1] ^= 0x01
    elif damage == "truncate":
        f = f[:-3]
    else:
        f = f[:HEADER.size - 1]
    with pytest.raises(CorruptFrameError):
        decode_frame(bytes(f))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_store_adds_and_counts(fresh, corpus):
    rep = fresh.add(corpus[:50])
    assert rep.added == 50 and fresh.count() == len(fresh) == 50 and rep.bytes_written > 0
    assert sum(rep.by_category.values()) == 50


def test_store_skips_duplicates_within_a_batch(fresh):
    rep = fresh.add([rec(), rec(), rec(hops=2)])
    assert rep.added == 2 and rep.duplicates == 1


def test_store_skips_duplicates_across_batches(fresh, corpus):
    fresh.add(corpus[:30], batch=7)
    rep = fresh.add(corpus[:40], batch=11)
    assert rep.added == 10 and rep.duplicates == 30 and fresh.count() == 40


def test_store_rotates_segments_and_verifies(store_root, corpus):
    with PatternStore(store_root, readonly=True) as st:
        assert len(st.segments()) > 3
        assert all(size <= 16 * 1024 for size in st.segment_sizes().values())
        v = st.verify()
        assert v["ok"] and v["frames"] == v["indexed"] == len(corpus)


def test_store_reads_every_record_back(store_root, corpus):
    with PatternStore(store_root, readonly=True) as st:
        assert {r.digest() for r in st.records()} == {r.digest() for r in corpus}


@pytest.mark.parametrize("batch", [1, 3, 64, 10_000])
def test_batch_size_does_not_change_contents(tmp_path, corpus, batch):
    with PatternStore(tmp_path / "s", segment_bytes=4 * 1024) as st:
        st.add(corpus[:60], batch=batch)
        assert sorted(r.hash for r in st.rows()) == sorted(r.digest() for r in corpus[:60])
        assert st.verify()["ok"]


def test_storage_cap_refuses_the_write_and_changes_nothing(tmp_path, corpus):
    with PatternStore(tmp_path / "s") as st:
        st.add(corpus[:5])
        st.max_storage_bytes = st.bytes_used() + 10
        before = (st.count(), st.segment_sizes())
        with pytest.raises(StorageFullError):
            st.add(corpus[5:50])
        assert (st.count(), st.segment_sizes()) == before and st.verify()["ok"]


def test_default_cap_is_500_gib(fresh):
    assert fresh.max_storage_bytes == 500 * (1 << 30)
    assert fresh.stats()["capacity_used"] < 1e-6


def test_readonly_store_rejects_writes(store_root):
    with PatternStore(store_root, readonly=True) as st:
        with pytest.raises(PatternError, match="read-only"):
            st.add([rec()])


def test_missing_readonly_store_raises(tmp_path):
    with pytest.raises(PatternError, match="no pattern store"):
        PatternStore(tmp_path / "nothing", readonly=True)


@pytest.mark.parametrize("kw", [dict(max_storage_bytes=0), dict(segment_bytes=4), dict(chunk_bytes=0)])
def test_store_rejects_bad_sizes(tmp_path, kw):
    with pytest.raises(PatternError):
        PatternStore(tmp_path / "s", **kw)


def test_closed_store_refuses_use(tmp_path):
    st = PatternStore(tmp_path / "s")
    st.close()
    with pytest.raises(PatternError, match="closed"):
        st.count()


def test_add_rejects_bad_batch(fresh):
    with pytest.raises(PatternError):
        fresh.add([rec()], batch=0)


@pytest.mark.parametrize("category", CATEGORIES)
def test_rows_filter_by_category(store_root, category):
    with PatternStore(store_root, readonly=True) as st:
        rows = st.rows(category=category)
        assert rows and all(r.category == category for r in rows) and len(rows) == st.count(category)


def test_rows_filter_by_source_extraction_and_ids(store_root):
    with PatternStore(store_root, readonly=True) as st:
        assert len(st.rows(source="synthetic")) == st.count() and st.rows(source="advisory") == []
        assert all(r.field("extraction_bps") >= 50 for r in st.rows(min_extraction_bps=50))
        lo, hi = st.id_range()
        assert [r.id for r in st.rows(id_lo=lo + 5, id_hi=lo + 9)] == list(range(lo + 5, lo + 10))


def test_rows_order_by_extraction_and_limit(store_root):
    with PatternStore(store_root, readonly=True) as st:
        best = st.rows(order="extraction", limit=10)
        vals = [r.field("extraction_bps") for r in best]
        assert len(best) == 10 and vals == sorted(vals, reverse=True)
        assert vals[0] == max(r.field("extraction_bps") for r in st.rows())


@pytest.mark.parametrize("kw", [dict(category="nope"), dict(order="random"), dict(limit=-1)])
def test_rows_rejects_bad_queries(store_root, kw):
    with PatternStore(store_root, readonly=True) as st:
        with pytest.raises(PatternError):
            st.rows(**kw)


def test_get_uses_the_cache(store_root, corpus):
    with PatternStore(store_root, readonly=True) as st:
        h = corpus[3].digest()
        assert st.get(h) == corpus[3] and st.get(h) == corpus[3]
        assert st.cache.hits >= 1 and st.get("0" * 64) is None


def test_read_many_preserves_order_with_partial_cache(store_root):
    with PatternStore(store_root, readonly=True, chunk_bytes=512) as st:
        rows = st.rows()
        st.read(rows[7])
        shuffled = rows[:]
        random.Random(3).shuffle(shuffled)
        got = st.read_many(shuffled)
        assert [r.digest() for r in got] == [r.hash for r in shuffled]


def test_cache_budget_bounds_memory(store_root):
    with PatternStore(store_root, readonly=True, cache_bytes=4096) as st:
        st.records()
        assert st.cache.used <= 4096 and st.cache.evictions > 0


@pytest.mark.parametrize("chunk", [1, 13, 100, 4096, 1 << 20])
def test_iter_segment_is_chunk_size_independent(store_root, chunk):
    with PatternStore(store_root, readonly=True) as st:
        seg = st.segments()[0]
        ref = [(o, n, r.digest()) for o, n, r in st.iter_segment(seg, 1 << 20)]
        assert [(o, n, r.digest()) for o, n, r in st.iter_segment(seg, chunk)] == ref and ref


def test_verify_detects_a_flipped_byte(tmp_path, corpus):
    with PatternStore(tmp_path / "s") as st:
        st.add(corpus[:20])
        path = st.segment_path(st.segments()[0])
        data = bytearray(path.read_bytes())
        data[HEADER.size + 5] ^= 0xFF
        path.write_bytes(bytes(data))
        v = st.verify()
        assert not v["ok"] and v["errors"]


def test_verify_detects_trailing_garbage(tmp_path, corpus):
    with PatternStore(tmp_path / "s") as st:
        st.add(corpus[:5])
        with open(st.segment_path(st.segments()[-1]), "ab") as fh:
            fh.write(b"MEV")
        assert not st.verify()["ok"]


def test_read_detects_a_frame_that_does_not_match_its_hash(tmp_path):
    with PatternStore(tmp_path / "s") as st:
        st.add([rec(), rec(hops=2)])
        a, b = st.rows()
        bad = type(a)(a.id, b.hash, a.category, a.source, a.chain_id, a.block_number, a.values, a.segment,
                      a.offset, a.length)
        with pytest.raises(CorruptFrameError):
            st.read(bad)


@pytest.mark.parametrize("category", CATEGORIES)
def test_category_matrix_matches_records(store_root, category):
    with PatternStore(store_root, readonly=True) as st:
        m = st.matrix(category)
        recs = st.records(category=category)
        assert m.shape == (len(recs), len(MATRIX_FIELDS)) and m.dtype == np.int64
        assert [tuple(row) for row in m.tolist()] == [r.matrix_row() for r in recs]


def test_empty_category_matrix(fresh):
    assert fresh.matrix("jit_liquidity").shape == (0, len(MATRIX_FIELDS))
    assert set(fresh.category_matrices()) == set(CATEGORIES)


def test_store_stats(store_root, corpus):
    with PatternStore(store_root, readonly=True) as st:
        s = st.stats()
        assert s["patterns"] == len(corpus) and sum(s["by_category"].values()) == len(corpus)
        assert s["by_source"] == {"synthetic": len(corpus)} and s["bytes_used"] >= s["segment_bytes"] > 0
        assert set(s["category_means"]) == set(CATEGORIES)


def test_reopened_store_sees_everything(tmp_path, corpus):
    with PatternStore(tmp_path / "s", segment_bytes=4096) as st:
        st.add(corpus[:30])
    with PatternStore(tmp_path / "s", segment_bytes=4096) as st:
        st.add(corpus[30:60])
        assert st.count() == 60 and st.verify()["ok"]


def test_concurrent_readers_agree(store_root):
    with PatternStore(store_root, readonly=True, cache_bytes=8192) as st:
        ref = [r.hash for r in st.rows()]
        out = []

        def read():
            out.append([r.digest() for r in st.records()])

        threads = [threading.Thread(target=read) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(out) == 8 and all(o == ref for o in out)


def test_concurrent_writers_never_duplicate(tmp_path, corpus):
    with PatternStore(tmp_path / "s", segment_bytes=4096) as st:
        parts = [corpus[i * 20:i * 20 + 60] for i in range(6)]
        errors = []

        def write(p):
            try:
                st.add(p, batch=9)
            except Exception as exc:  # surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(p,)) for p in parts]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert st.count() == len({r.digest() for p in parts for r in p}) and st.verify()["ok"]


def test_reader_sees_consistent_snapshots_while_writing(tmp_path, corpus):
    root = tmp_path / "s"
    with PatternStore(root, segment_bytes=4096) as writer:
        writer.add(corpus[:10])
        stop = threading.Event()
        seen = []

        def read():
            with PatternStore(root, readonly=True) as reader:
                while not stop.is_set():
                    rows = reader.rows()
                    seen.append(len(reader.read_many(rows)) == len(rows))

        t = threading.Thread(target=read)
        t.start()
        for i in range(10, 120, 10):
            writer.add(corpus[i:i + 10])
        stop.set()
        t.join()
        assert seen and all(seen)


# ---------------------------------------------------------------------------
# Slippage evaluation
# ---------------------------------------------------------------------------


def test_evaluation_without_frontrun_has_no_slippage():
    ev = evaluate_pattern(rec(frontrun_impact_bps=0))
    assert ev.realized_slippage_bps == 0 and ev.profit_wei == 0 and ev.victim_limit_held


@pytest.mark.parametrize("fraction,held", [(0.25, True), (0.5, True), (0.9, True), (1.2, False), (2.0, False)])
def test_victim_limit_holds_exactly_up_to_the_max_frontrun(fraction, held):
    base = rec(frontrun_impact_bps=0, victim_tolerance_bps=200, victim_size_bps=100)
    room = evaluate_pattern(base).max_frontrun_in_wei
    impact = int(room * fraction * 10_000 // base.reserve_in_wei)
    ev = evaluate_pattern(rec(frontrun_impact_bps=impact, victim_tolerance_bps=200, victim_size_bps=100))
    assert ev.victim_limit_held is held


def test_max_frontrun_is_the_exact_boundary():
    r = rec(frontrun_impact_bps=0, victim_tolerance_bps=150)
    ev = evaluate_pattern(r)
    from simulator.pattern_ingestor import _sandwich

    m = ev.max_frontrun_in_wei
    res = ev.reserve_in_wei
    assert _sandwich(res, res, m, ev.victim_in_wei, 30)[0] >= ev.victim_min_out
    assert _sandwich(res, res, m + 1, ev.victim_in_wei, 30)[0] < ev.victim_min_out


def test_slippage_grows_with_frontrun_impact():
    slips = [evaluate_pattern(rec(frontrun_impact_bps=i)).realized_slippage_bps for i in (0, 10, 40, 160, 640)]
    assert slips == sorted(slips) and slips[-1] > slips[0]


def test_evaluation_matches_the_v2_pool_quote():
    r = rec(frontrun_impact_bps=30, victim_size_bps=40)
    ev = evaluate_pattern(r)
    pool = V2Pool("A", "B", r.reserve_in_wei, r.reserve_in_wei)
    front = pool.quote_exact_in(ev.frontrun_in_wei, True)
    after = V2Pool("A", "B", r.reserve_in_wei + ev.frontrun_in_wei, r.reserve_in_wei - front.amount_out)
    assert after.quote_exact_in(ev.victim_in_wei, True).amount_out == ev.victim_actual_out


def test_failed_victim_limit_extracts_nothing():
    ev = evaluate_pattern(rec(frontrun_impact_bps=5_000, victim_tolerance_bps=10))
    assert not ev.victim_limit_held and ev.extraction_bps == 0
    assert pattern_weight(rec(frontrun_impact_bps=5_000, victim_tolerance_bps=10)) == 0.05


def test_nominal_pool_is_used_without_reserves():
    assert evaluate_pattern(rec(reserve_in_wei=0)).reserve_in_wei == NOMINAL_RESERVE_WEI


def test_pattern_weights_are_bounded(corpus):
    ws = [pattern_weight(r) for r in corpus]
    assert all(0.05 <= w <= 1.0 for w in ws) and len(set(ws)) > 10


# ---------------------------------------------------------------------------
# Parallel scan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lo,hi,shards", [(1, 1, 1), (1, 10, 3), (1, 10, 10), (1, 10, 64), (5, 1004, 7),
                                          (1, 3072, 64)])
def test_scan_shards_partition_the_id_range(lo, hi, shards):
    parts = scan_shards(lo, hi, shards)
    ids = [i for a, b in parts for i in range(a, b + 1)]
    sizes = [b - a + 1 for a, b in parts]
    assert ids == list(range(lo, hi + 1)) and max(sizes) - min(sizes) <= 1 and len(parts) <= shards


def test_scan_shards_edge_cases():
    assert scan_shards(0, -1, 4) == []
    with pytest.raises(PatternError):
        scan_shards(1, 2, 0)


def test_inline_scan_reads_everything(store_root, corpus):
    res = parallel_scan(store_root, None, shards=5)
    assert isinstance(res, ScanResult) and len(res.records) == len(corpus) and res.shards == 5
    assert res.hashes == [r.digest() for r in res.records]


@pytest.mark.parametrize("shards", [1, 4, 16, 64])
def test_scan_selection_is_shard_count_independent(store_root, shards):
    ref = parallel_scan(store_root, None, shards=1, limit=50)
    res = parallel_scan(store_root, None, shards=shards, limit=50)
    assert res.hashes == ref.hashes and res.weights == ref.weights


def test_limit_keeps_the_heaviest(store_root):
    full = parallel_scan(store_root, None, shards=3)
    top = parallel_scan(store_root, None, shards=3, limit=20)
    cutoff = sorted(full.weights, reverse=True)[19]
    assert len(top.records) == 20 and min(top.weights) >= cutoff - 1e-12


@pytest.mark.parametrize("category", CATEGORIES)
def test_scan_category_filter(store_root, category):
    res = parallel_scan(store_root, None, shards=4, category=category)
    assert res.records and all(r.category == category for r in res.records)


@needs_two_cpus
def test_scan_on_every_core_matches_inline(store_root, full_pool):
    ref = parallel_scan(store_root, None, shards=16, limit=100)
    res = parallel_scan(store_root, full_pool, shards=16, limit=100)
    assert res.hashes == ref.hashes and res.parallel and len(set(res.worker_pids)) > 1


@needs_two_cpus
def test_repeated_full_core_scans_are_identical(store_root, full_pool):
    runs = [parallel_scan(store_root, full_pool, shards=32).hashes for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_scan_summary_is_json(store_root):
    s = parallel_scan(store_root, None, shards=2, limit=10).summary()
    json.dumps(s)
    assert s["patterns"] == 10 and set(s["by_category"]) == set(CATEGORIES)


# ---------------------------------------------------------------------------
# Synthetic corpus and export
# ---------------------------------------------------------------------------


def test_synthetic_corpus_is_seeded():
    assert [r.digest() for r in synthetic_corpus(40, 5)] == [r.digest() for r in synthetic_corpus(40, 5)]
    assert synthetic_corpus(10, 5) != synthetic_corpus(10, 6)


@pytest.mark.parametrize("n", [0, 1, 4, 9, 101])
def test_synthetic_corpus_is_balanced_and_labelled(n):
    recs = synthetic_corpus(n, 1)
    counts = [sum(r.category == c for r in recs) for c in CATEGORIES]
    assert len(recs) == n and max(counts, default=0) - min(counts, default=0) <= 1
    assert all(r.source == "synthetic" and r.reference.startswith("synthetic:") for r in recs)


def test_synthetic_corpus_is_mostly_feasible(corpus):
    held = sum(evaluate_pattern(r).victim_limit_held for r in corpus)
    assert 0.75 * len(corpus) <= held < len(corpus)


def test_synthetic_rejects_bad_input():
    with pytest.raises(PatternError):
        synthetic_corpus(-1)
    with pytest.raises(PatternError):
        synthetic_record("nope", random.Random(0))


def test_jsonl_export_reingests_identically(tmp_path, store_root):
    with PatternStore(store_root, readonly=True) as st:
        n = write_jsonl(st.records(), tmp_path / "x.jsonl")
        want = sorted(r.hash for r in st.rows())
    with PatternStore(tmp_path / "copy") as st:
        rep = st.ingest_file(tmp_path / "x.jsonl")
        assert rep.added == n and rep.rejected == 0 and sorted(r.hash for r in st.rows()) == want


# ---------------------------------------------------------------------------
# Pattern-seeded AI bots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("category", CATEGORIES)
def test_pattern_profiles_follow_the_category(corpus, category):
    for r in [x for x in corpus if x.category == category][:15]:
        p = profile_from_pattern(r, "p")
        assert not profile_issues(p)
        assert p.strategy == CATEGORY_STRATEGY[category] and p.hop_policy == CATEGORY_HOP_POLICY[category]
        assert p.max_hops == (2 if r.hops >= 2 else 1) and set(p.target_pools) <= set(ARENA_POOLS)


def test_pattern_profile_fields_track_the_record():
    p = profile_from_pattern(rec(bribe_bps=6_000, frontrun_impact_bps=120, victim_tolerance_bps=300,
                                 latency_ms=400, pools=2), "p")
    assert (p.share_bps, p.max_impact_bps, p.slippage_floor_bps, p.latency_ms, len(p.target_pools)) == (
        6_000, 120, 300, 400, 2)


def test_pattern_profile_survives_gene_round_trip(corpus):
    for r in corpus[:20]:
        p = profile_from_pattern(r, "p")
        q = decode_profile(encode_profile(p), "p")
        assert (q.strategy, q.hop_policy, q.max_hops, q.target_pools, q.bundle_slots) == (
            p.strategy, p.hop_policy, p.max_hops, p.target_pools, p.bundle_slots)


def test_pattern_latents_are_deterministic():
    a, b = pattern_latent("ab" * 32, 8), pattern_latent("ab" * 32, 8)
    assert np.array_equal(a, b) and not np.array_equal(a, pattern_latent("cd" * 32, 8)) and a.shape == (8,)


def test_prior_shapes_and_dedup(corpus):
    prior = PatternPrior.build(corpus[:30] + corpus[:5], latent_dim=6)
    assert len(prior) == 30 and prior.genes.shape == (30, N_GENES) and prior.latents.shape == (30, 6)
    assert np.all((prior.genes >= 0) & (prior.genes <= 1)) and prior.weights.shape == (30,)


def test_prior_ranking_is_heaviest_first(corpus):
    prior = PatternPrior.build(corpus[:40])
    w = [prior.weights[i] for i in prior.ranked()]
    assert w == sorted(w, reverse=True)


def test_prior_needs_one_weight_per_pattern(corpus):
    with pytest.raises(ConfigError):
        PatternPrior.build(corpus[:3], [1.0])


def small_factory(**kw):
    base = dict(bots=16, generations=2, heat_size=4, heat_slots=2, seed=3, pattern_seed_bps=5_000, epochs=5,
                pattern_epochs=20)
    base.update(kw)
    return AgentFactory(**base)


def test_ingest_from_list_store_path_and_scan_agree(store_root, corpus):
    a = small_factory().ingest_patterns(store_root, limit=40)
    with PatternStore(store_root, readonly=True) as st:
        b = small_factory().ingest_patterns(st, limit=40)
    c = small_factory().ingest_patterns(parallel_scan(store_root, None, limit=40))
    d = small_factory().ingest_patterns(corpus, limit=40)
    strip = {k: v for k, v in a.items() if k != "scan"}
    assert strip == {k: v for k, v in b.items() if k != "scan"} == {k: v for k, v in c.items() if k != "scan"}
    assert d["patterns"] == 40 and d["by_category"] == strip["by_category"]


def test_pretraining_lowers_the_loss(corpus):
    f = small_factory()
    f.ingest_patterns(corpus[:64])
    before, after = f.pretrain()
    assert after < before


@pytest.mark.parametrize("bps,want", [(0, 0), (2_500, 4), (5_000, 8), (10_000, 16)])
def test_seed_plan_size(corpus, bps, want):
    f = small_factory(pattern_seed_bps=bps)
    f.ingest_patterns(corpus[:5])
    plan = f.seed_plan()
    assert len(plan) == want and plan[:5] == f.prior.ranked()[:len(plan)]


def test_generation_zero_starts_from_pattern_latents(corpus):
    f = small_factory()
    f.ingest_patterns(corpus[:32])
    pop = f.synthesize(0)
    plan = f.seed_plan()
    assert pop.seeded == len(plan) == 8
    assert np.array_equal(pop.latents[:8], f.prior.latents[plan])
    assert all(h is None for h in pop.pattern_sources[8:])


def test_later_generations_are_not_reseeded(corpus):
    f = small_factory()
    f.ingest_patterns(corpus[:32])
    pop = f.synthesize(1, parents=[0] * 16, parent_latents=np.zeros((16, f.latent_dim)))
    assert pop.seeded == 0


def test_pattern_seeded_run_reports_and_passes_invariants(store_root):
    f = small_factory()
    f.ingest_patterns(store_root, limit=60)
    s = f.run().summary()
    assert s["invariants"]["ok"] and "ai.pattern_seeding" in s["invariants"]["rules"]
    p = s["patterns"]
    assert p["seeded_bots"] == 8 and sum(p["seeded_by_category"].values()) == 8
    assert p["pretrain_loss_after"] < p["pretrain_loss_before"] and 0 <= p["gene_error_mean"] < 0.5
    json.dumps(s)


def test_factory_without_patterns_is_unchanged():
    a = AgentFactory(bots=12, generations=2, heat_size=4, heat_slots=2, seed=9, epochs=5).run().summary()
    b = AgentFactory(bots=12, generations=2, heat_size=4, heat_slots=2, seed=9, epochs=5,
                     pattern_seed_bps=5_000).run().summary()
    for s in (a, b):
        for k in ("elapsed_s", "cpu_s", "bots_per_s", "parallel_efficiency"):
            s.pop(k)
    assert a == b and "patterns" not in a


def test_pattern_rule_is_registered():
    assert FACTORY_RULES[-1] == "ai.pattern_seeding" and len(set(FACTORY_RULES)) == len(FACTORY_RULES)


@needs_two_cpus
def test_pattern_seeded_run_is_worker_count_independent(store_root, full_pool):
    def run(executor):
        f = small_factory()
        f.ingest_patterns(store_root, executor, limit=60)
        s = f.run(executor=executor).summary()
        return s["history"], s["top_bots"], s["patterns"]["seeded_mean_pnl_wei"]

    assert run(None) == run(full_pool)


@pytest.mark.parametrize("kw", [dict(pattern_seed_bps=-1), dict(pattern_seed_bps=10_001), dict(pattern_epochs=-1)])
def test_factory_rejects_bad_pattern_settings(kw):
    with pytest.raises(ConfigError):
        small_factory(**kw)


# ---------------------------------------------------------------------------
# Config, controller, validator, CLI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kw", [
    dict(pattern_store=5), dict(pattern_max_storage_gb=0), dict(pattern_segment_mb=0), dict(pattern_cache_mb=-1),
    dict(pattern_seed_bps=10_001), dict(pattern_limit=-1), dict(pattern_synthetic=-2),
])
def test_config_rejects_bad_pattern_settings(kw):
    with pytest.raises(ConfigError):
        SimulationConfig.for_mode("max", cpus=4, **kw)


def test_presets_enable_patterns_only_in_max():
    assert PRESETS[RunMode.MAX_BRUTAL]["pattern_seed_bps"] == 5_000
    assert PRESETS[RunMode.MAX_BRUTAL]["pattern_synthetic"] == 3_072
    assert PRESETS[RunMode.MODERATE_SCAN]["pattern_seed_bps"] == PRESETS[RunMode.BRUTAL_AGGRESSIVE][
        "pattern_seed_bps"] == 0
    cfg = SimulationConfig.for_mode("max", cpus=4)
    assert cfg.pattern_max_storage_gb == 500 and cfg.to_dict()["pattern_limit"] == 4_096


@pytest.mark.parametrize("kw", [dict(pattern_seed_bps=0), dict(pattern_limit=0), dict(pattern_synthetic=0)])
def test_controller_ingestion_is_off_without_patterns(kw):
    cfg = SimulationConfig.for_mode("max", cpus=4, **kw)
    assert ingest_patterns(cfg, small_factory(), None) == {}


def test_controller_ingests_a_configured_store(store_root):
    cfg = SimulationConfig.for_mode("max", cpus=4, pattern_store=str(store_root), pattern_limit=30)
    f = small_factory()
    out = ingest_patterns(cfg, f, None)
    assert out["store"] == str(store_root) and not out["synthetic"] and out["patterns"] == 30
    assert out["stored_patterns"] == 240 and len(f.prior) == 30


def test_controller_builds_a_synthetic_reference_store():
    cfg = SimulationConfig.for_mode("max", cpus=4, pattern_synthetic=40, pattern_limit=25)
    out = ingest_patterns(cfg, small_factory(), None)
    assert out["synthetic"] and out["stored_patterns"] == 40 and out["patterns"] == 25


def test_max_controller_run_carries_the_pattern_sections():
    cfg = SimulationConfig.for_mode(
        "max", cpus=4, blocks=1, txs_per_block=4, frontrun_grid=2, refine_iterations=1, fuzz_iterations=4,
        layout_fuzz_iterations=2, rl_rounds=1, rl_steps_per_shard=2, rl_eval_steps=2, pg_paths=4, pg_path_steps=2,
        pg_batch=2, shard_density=1, private_flow_bps=0, agent_bots=8, agent_generations=1, agent_heat_size=4,
        agent_heat_slots=1, pattern_synthetic=32, pattern_limit=16)
    out = SimulationController(cfg).run()
    af = out["result"]["agent_factory"]
    assert invariants_ok(out) and af["pattern_ingestion"]["synthetic"] and af["patterns"]["seeded_bots"] == 4
    json.dumps(out)


def pattern_signals(cfg):
    col = Collector()
    check_signals(cfg, col, 4)
    return {f.rule for f in col.findings}


def test_validator_flags_a_missing_store(tmp_path):
    cfg = SimulationConfig.for_mode("max", cpus=4, pattern_store=str(tmp_path / "none"))
    assert "signal.pattern_store" in pattern_signals(cfg)


def test_validator_accepts_an_existing_store(store_root):
    cfg = SimulationConfig.for_mode("max", cpus=4, pattern_store=str(store_root))
    rules = pattern_signals(cfg)
    assert "signal.pattern_store" not in rules and "signal.pattern_seed_ignored" not in rules


def test_validator_notes_unused_pattern_seeding():
    assert "signal.pattern_seed_ignored" in pattern_signals(
        SimulationConfig.for_mode("moderate", cpus=4, pattern_seed_bps=1_000))


def cli(capsys, *argv):
    code = cli_main(["--no-config", *argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_builds_reports_and_verifies(tmp_path, capsys):
    store = str(tmp_path / "cli")
    code, out, _ = cli(capsys, "--pattern-store", store, "--build-synthetic-patterns", "48")
    assert code == 0 and json.loads(out)["added"]["added"] == 48
    code, out, _ = cli(capsys, "--pattern-store", store, "--pattern-stats")
    assert code == 0 and json.loads(out)["patterns"] == 48
    code, out, _ = cli(capsys, "--pattern-store", store, "--pattern-verify")
    assert code == 0 and json.loads(out)["ok"]


def test_cli_export_and_ingest(tmp_path, capsys):
    src, dst, f = str(tmp_path / "a"), str(tmp_path / "b"), str(tmp_path / "x.jsonl")
    cli(capsys, "--pattern-store", src, "--build-synthetic-patterns", "20")
    code, out, _ = cli(capsys, "--pattern-store", src, "--export-patterns", f)
    assert code == 0 and json.loads(out)["exported"] == 20
    code, out, _ = cli(capsys, "--pattern-store", dst, "--ingest-patterns", f, f)
    total = json.loads(out)["total"]
    assert code == 0 and total["added"] == 20 and total["duplicates"] == 20


def test_cli_forced_parser(tmp_path, capsys):
    f = tmp_path / "mi.jsonl"
    f.write_text(json.dumps(mi_row()) + "\n", encoding="utf-8")
    code, out, _ = cli(capsys, "--pattern-store", str(tmp_path / "s"), "--ingest-patterns", str(f),
                       "--pattern-source", "mev_inspect")
    assert code == 0 and json.loads(out)["total"]["by_source"] == {"mev_inspect": 1}


def test_cli_needs_a_store(capsys):
    code, _, err = cli(capsys, "--pattern-stats")
    assert code == 2 and "--pattern-store" in err


def test_cli_verify_fails_on_corruption(tmp_path, capsys):
    store = tmp_path / "bad"
    cli(capsys, "--pattern-store", str(store), "--build-synthetic-patterns", "8")
    seg = sorted((store / "segments").iterdir())[0]
    data = bytearray(seg.read_bytes())
    data[-1] ^= 0xFF
    seg.write_bytes(bytes(data))
    code, out, _ = cli(capsys, "--pattern-store", str(store), "--pattern-verify")
    assert code == 1 and not json.loads(out)["ok"]


def test_cli_rejects_negative_synthetic_count(tmp_path, capsys):
    code, _, _ = cli(capsys, "--pattern-store", str(tmp_path / "s"), "--build-synthetic-patterns", "-1")
    assert code == 2


def test_store_files_stay_inside_the_root(tmp_path):
    with PatternStore(tmp_path / "s") as st:
        st.add(synthetic_corpus(12, 2))
    names = {p.relative_to(tmp_path / "s").parts[0] for p in (tmp_path / "s").rglob("*")}
    assert names <= {"index.sqlite", "index.sqlite-wal", "index.sqlite-shm", "segments"}
    assert os.path.isdir(Path(tmp_path / "s" / "segments"))
