"""Layer 13: local discovery of new attack compositions and the strategy library."""

import json

import pytest

from simulator.__main__ import main
from simulator.attack_vectors import build_scenario
from simulator.strategy_discovery import (
    LIBRARY_ENV,
    DiscoveryEngine,
    StrategyLibrary,
    classify,
    default_library_dir,
    evaluate_genome,
    label_for,
    minimize,
    mutate,
    random_genome,
    template_genomes,
)


@pytest.fixture(scope="module")
def scn():
    return build_scenario(seed=0, victims=10)


@pytest.fixture(scope="module")
def report_and_lib(tmp_path_factory):
    lib = StrategyLibrary(tmp_path_factory.mktemp("lib"))
    report = DiscoveryEngine(seed=1, generations=25, population=40, victims=10, scenario_seeds=2,
                             holdout=2).run(library=lib)
    return report, lib


def _first_profitable(scn, genomes):
    for g in genomes:
        ev = evaluate_genome(scn, g)
        if ev.profitable:
            return ev
    return None


@pytest.mark.parametrize("template", ["sandwich", "jit_liquidity", "backrun_arbitrage"])
def test_template_genomes_are_recognised_as_known(scn, template):
    names = set()
    for g in template_genomes(scn):
        ev = evaluate_genome(scn, g)
        if ev.valid:
            names.add(classify(ev.signature))
    assert template in names


def test_signature_roles_and_collapse(scn):
    roles = [g for g in template_genomes(scn) if g[0][0] == "swap"]
    ev = _first_profitable(scn, roles)
    assert ev is not None
    assert ev.signature.startswith("SWAP[V1=] VICTIM[V1] SWAP[V1~]")
    assert classify(ev.signature) == "sandwich"
    doubled = evaluate_genome(scn, (ev.genome[0],) + ev.genome)
    assert doubled.signature.startswith("SWAP[V1=]+ VICTIM[V1]")
    assert classify(doubled.signature) == "sandwich"


@pytest.mark.parametrize("sig,expected", [
    ("SWAP[V1=] VICTIM[V1]+ SWAP[V1~]", "multi_victim_sandwich"),
    ("MINT[V1] VICTIM[V1] BURN[V1] SETTLE[V1~]", "jit_liquidity"),
    ("VICTIM[V1] LIQ[M1] SETTLE[P1b]", "liquidation_backrun"),
    ("SWAP[P1b] SWAP[P2x]", "cyclic_arbitrage"),
    ("MINT[V1] SWAP[V1=] VICTIM[V1] BURN[V1] SETTLE[V1~]", None),
    ("SWAP[O1b] LIQ[M1] SETTLE[P1b]", None),
])
def test_classify(sig, expected):
    assert classify(sig) == expected


def test_labels_name_the_components():
    assert label_for("MINT[V1] SWAP[V1=] VICTIM[V1] BURN[V1] SETTLE[V1~]", core_only=True) == \
        "JIT liquidity on V1 + front-run on V1 + back-run on V1"
    assert label_for("SWAP[O1b] LIQ[M1] SETTLE[P1b]", core_only=True) == "oracle push via O1 + liquidation"
    assert "2 victim tx on 2 pool(s)" in label_for("VICTIM[V1] SWAP[V2=] VICTIM[V2] SETTLE[V2~]")


def test_invalid_genomes_are_rejected(scn):
    assert not evaluate_genome(scn, ()).valid
    assert not evaluate_genome(scn, (("burn",),)).valid
    assert not evaluate_genome(scn, (("victim", 0, 0, 99),)).valid
    assert not evaluate_genome(scn, (("close", 0, 0),)).valid


def test_evaluation_is_deterministic_and_settles_to_base(scn):
    import random

    rng = random.Random(3)
    for _ in range(60):
        g = random_genome(rng, scn)
        a, b = evaluate_genome(scn, g), evaluate_genome(scn, g)
        assert (a.valid, a.net_profit_base, a.signature) == (b.valid, b.net_profit_base, b.signature)
        if a.valid and a.result.flat:
            assert set(a.result.ledger) <= {"WETH"}
        g2 = mutate(rng, scn, g)
        assert 1 <= len(g2) <= 10


def test_minimize_never_lowers_profit(scn):
    ev = _first_profitable(scn, template_genomes(scn))
    padded = evaluate_genome(scn, ev.genome + (("burn",),))
    assert not padded.valid
    noisy = evaluate_genome(scn, (("swap", 0, 0, 10),) + ev.genome)
    if noisy.profitable:
        m = minimize(scn, noisy)
        assert m.net_profit_base >= noisy.net_profit_base
        assert len(m.genome) <= len(noisy.genome)


def test_engine_finds_novel_structures_and_saves_them(report_and_lib):
    report, lib = report_and_lib
    d = report.to_dict()
    assert d["evaluations"] > 1000 and d["structures_found"] > 10
    assert d["best_known_by_template"].get("sandwich", 0) > 0
    assert report.novel, "the search should find at least one composition outside the templates"
    families = [r["family"] for r in report.novel]
    assert len(families) == len(set(families))
    for r in report.novel:
        assert classify(r["signature"]) is None
        assert r["net_profit_base"] > 0
        assert r["label"].startswith(r["family"])
        assert len(r["holdout"]) == 2
    assert set(report.saved) <= {r["id"] for r in report.novel}
    assert lib.path.exists()


def test_library_replays_every_record_to_the_wei(report_and_lib):
    _, lib = report_and_lib
    v = lib.verify()
    assert v["ok"] and v["records"] == len(lib.best())


def test_library_keeps_the_best_record_per_family(report_and_lib, tmp_path):
    report, _ = report_and_lib
    lib = StrategyLibrary(tmp_path)
    rec = report.novel[0]
    assert lib.add(rec)
    assert not lib.add(rec)
    worse = dict(rec, net_profit_base=rec["net_profit_base"] - 1)
    assert not lib.add(worse)
    better = dict(rec, net_profit_base=rec["net_profit_base"] + 1)
    assert lib.add(better)
    assert lib.best()[rec["family"]]["net_profit_base"] == rec["net_profit_base"] + 1
    assert not lib.verify()["ok"]  # the doctored record no longer replays exactly


def test_engine_is_deterministic():
    a = DiscoveryEngine(seed=4, generations=4, population=16, scenario_seeds=1, holdout=1).run().to_dict()
    b = DiscoveryEngine(seed=4, generations=4, population=16, scenario_seeds=1, holdout=1).run().to_dict()
    assert a == b


def test_default_library_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(LIBRARY_ENV, str(tmp_path))
    assert default_library_dir() == tmp_path
    monkeypatch.delenv(LIBRARY_ENV)
    assert default_library_dir().parts[-2:] == (".mev_sim", "strategies")


def test_cli_discover_list_verify_and_catalog(tmp_path, capsys):
    d = str(tmp_path / "lib")
    assert main(["--discover", "--generations", "3", "--population", "16", "--strategy-dir", d,
                 "--workers", "1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["evaluations"] > 0 and out["library"].startswith(d)
    assert main(["--strategies", "--strategy-dir", d]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert len(listed["strategies"]) == len(out["saved"])
    assert main(["--verify-strategies", "--strategy-dir", d]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    assert main(["--attack-catalog", "--scenario-victims", "4"]) == 0
    cat = json.loads(capsys.readouterr().out)
    assert set(cat["summary"]) >= {"sandwich", "jit_liquidity", "oracle_manipulation"}
    assert main(["--discover", "--population", "0"]) == 2
