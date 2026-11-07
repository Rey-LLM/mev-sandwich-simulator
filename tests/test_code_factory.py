"""Layer 15: on-disk code factory - generated source safety, compilation, disk deployment, training loop."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from simulator import code_factory as cf
from simulator.ai_agent_factory import (
    N_GENES,
    AgentFactory,
    ArenaConfig,
    BotArena,
    SynthesizedBot,
    arena_market,
    class_name,
    decode_profile,
)
from simulator.block_builder import BuilderState, generated_competitors, run_generated_competition
from simulator.parallel import WorkerPool


def _population(bots=6, seed=0, generation=0):
    return AgentFactory(bots=bots, generations=1, seed=seed).synthesize(generation)


@pytest.fixture(scope="module")
def pop():
    return _population(8, seed=3)


@pytest.fixture(scope="module")
def sample(pop):
    profile, genes = pop.profiles[0], [float(x) for x in pop.genes[0]]
    return profile, genes, cf.render_python(profile, genes), cf.render_solidity(profile)


@pytest.fixture(scope="module")
def factory_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("bots")
    events = []
    f = cf.CodeFactory(d, bots=10, generations=3, heat_size=4, heat_slots=4, seed=5, listener=events.append)
    return d, f.run(), events


# -- rendering -------------------------------------------------------------------------------------------------------


def test_render_python_is_deterministic(sample, pop):
    profile, genes, src, _ = sample
    assert cf.render_python(profile, genes) == src
    assert src.startswith("# -*- coding: utf-8 -*-\n")


def test_rendered_python_passes_allowlist_and_compiles(sample):
    _, _, src, _ = sample
    assert cf.python_source_issues(src) == []
    compile(src, "bot.py", "exec")


@pytest.mark.parametrize("i", range(8))
def test_every_profile_of_a_population_renders_safely(pop, i):
    src = cf.render_python(pop.profiles[i], [float(x) for x in pop.genes[i]])
    assert cf.python_source_issues(src) == []
    sol = cf.render_solidity(pop.profiles[i])
    issues, consts = cf.solidity_source_issues(sol)
    assert issues == []
    assert cf.profile_from_solidity(consts, pop.profiles[i].name) == pop.profiles[i]


def test_header_is_localized(sample):
    profile, genes, _, _ = sample
    ru = cf.render_python(profile, genes, lang="ru")
    en = cf.render_python(profile, genes, lang="en")
    assert "Только для симуляции" in ru and "Simulation only" in en
    assert "Исполняемой логики нет" in cf.render_solidity(profile, lang="ru")


def test_render_rejects_bad_inputs(sample):
    profile, genes, _, _ = sample
    with pytest.raises(cf.CodeRejected):
        cf.render_python(profile, genes[:-1])
    with pytest.raises(cf.CodeRejected):
        cf.render_python(profile, genes[:-1] + [float("nan")])


def test_genes_literal_roundtrips_exactly(sample):
    profile, genes, src, _ = sample
    tree = ast.parse(src)
    consts = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)}
    assert consts["GENES"] == tuple(genes)
    assert consts["FINGERPRINT"] == profile.fingerprint()
    assert consts["PROFILE_FIELDS"] == cf.profile_fields(profile)


def test_solidity_constants_are_uint256(sample):
    profile = sample[0]
    consts = cf.solidity_constants(profile)
    assert all(type(v) is int and 0 <= v <= cf.UINT256_MAX for v in consts.values())
    assert consts["TARGET_POOLS_MASK"] > 0


def test_solidity_record_has_no_functions(sample):
    sol = sample[3]
    for word in ("function", "call", "assembly", "address", "external", "public", "payable"):
        assert not re.search(rf"\b{word}\b", sol)


# -- the allow-list rejects tampered code ------------------------------------------------------------------------------


TAMPERS = {
    "import_os": ("from simulator.ai_agent_factory import BotProfile, SynthesizedBot",
                  "from simulator.ai_agent_factory import BotProfile, SynthesizedBot\nimport os"),
    "other_module": ("from simulator.ai_agent_factory import", "from os import"),
    "alias": ("import BotProfile, SynthesizedBot", "import BotProfile as P, SynthesizedBot"),
    "exec_call": ("CODEGEN_VERSION = 1", "CODEGEN_VERSION = 1\nexec('print(1)')"),
    "call_in_constant": ("GENERATION = ", "GENERATION = len('ab') + "),
    "unknown_constant": ("CODEGEN_VERSION = 1", "CODEGEN_VERSION = 1\nPAYLOAD = 1"),
    "dup_constant": ("CODEGEN_VERSION = 1", "CODEGEN_VERSION = 1\nCODEGEN_VERSION = 1"),
    "extra_function": ("def build():", "def evil():\n    return 1\n\n\ndef build():"),
    "decorator": ("def build():", "@staticmethod\ndef build():"),
    "class_method": ("    PROFILE = BotProfile(**PROFILE_FIELDS)",
                     "    PROFILE = BotProfile(**PROFILE_FIELDS)\n\n    def plan(self):\n        return None"),
    "other_base": ("(SynthesizedBot):", "(SynthesizedBot, object):"),
    "loop": ("def build():", "while False:\n    pass\n\n\ndef build():"),
    "attribute_in_build": ("    return AIBot", "    return __import__('os').system or AIBot"),
    "build_args": ("def build():", "def build(x=1):"),
    "version": ("CODEGEN_VERSION = 1", "CODEGEN_VERSION = 99"),
    "fingerprint": ('FINGERPRINT = "', 'FINGERPRINT = "0'),
    "syntax": ("def build():", "def build(:"),
    "with_stmt": ("def build():", "with open('x') as f:\n    pass\n\n\ndef build():"),
}


@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_allowlist_rejects_tampering(sample, name, tmp_path):
    old, new = TAMPERS[name]
    src = sample[2]
    assert old in src, name
    bad = src.replace(old, new, 1)
    assert cf.python_source_issues(bad), name
    path = tmp_path / "bot_0000.py"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(cf.CodeRejected):
        cf.load_bot(path)


def test_allowlist_rejects_wrong_class_name(sample):
    src = sample[2]
    cls = class_name(sample[0])
    assert cf.python_source_issues(src.replace(cls, "Renamed"))


def test_allowlist_rejects_invalid_profile_values(sample):
    src = sample[2].replace('"max_hops": ', '"max_hops": 9 + 0 * ', 1)
    assert cf.python_source_issues(src)
    src2 = re.sub(r'"bundle_slots": \d+', '"bundle_slots": 99', sample[2])
    assert any("not a valid profile" in i for i in cf.python_source_issues(src2))


SOL_TAMPERS = {
    "function": ("}\n", "    function f() external {}\n}\n"),
    "assembly": ("}\n", "    assembly { }\n}\n"),
    "overflow": ("CODEGEN_VERSION = 1;", "CODEGEN_VERSION = " + str(1 << 256) + ";"),
    "duplicate": ("CODEGEN_VERSION = 1;", "CODEGEN_VERSION = 1;\n    uint256 internal constant CODEGEN_VERSION = 1;"),
    "spdx": ("// SPDX-License-Identifier: MIT", "// licence"),
    "pragma": ("pragma solidity ^0.8.20;", "pragma solidity ^0.4.0;"),
    "unclosed": ("}\n", "\n"),
    "trailing": ("}\n", "}\ncontract X {}\n"),
    "fingerprint": ("bytes8 internal constant FINGERPRINT = 0x", "bytes8 internal constant FINGERPRINT = 0xf"),
}


@pytest.mark.parametrize("name", sorted(SOL_TAMPERS))
def test_solidity_validator_rejects_tampering(sample, name):
    old, new = SOL_TAMPERS[name]
    sol = sample[3]
    assert old in sol, name
    issues, _ = cf.solidity_source_issues(sol.replace(old, new, 1) if name != "unclosed" else sol[:-2] + "\n")
    assert issues, name


def test_solidity_roundtrip_detects_value_change(sample, tmp_path):
    profile, _, _, sol = sample
    path = tmp_path / "b.sol"
    path.write_text(sol, encoding="utf-8")
    assert cf.check_solidity_file(path, profile) == []
    changed = re.sub(r"SHARE_BPS = (\d+);", lambda m: f"SHARE_BPS = {int(m.group(1)) + 1};", sol, count=1)
    path.write_text(changed, encoding="utf-8")
    assert cf.check_solidity_file(path, profile)


def test_profile_from_solidity_rejects_bad_records(sample):
    _, consts = cf.solidity_source_issues(sample[3])
    for key, value in (("STRATEGY", 99), ("HOP_POLICY", 99), ("TARGET_POOLS_MASK", 0), ("TARGET_POOLS_MASK", 8)):
        with pytest.raises(cf.CodeRejected):
            cf.profile_from_solidity({**consts, key: value}, "x")
    with pytest.raises(cf.CodeRejected):
        cf.profile_from_solidity({k: v for k, v in consts.items() if k != "SIZE_BPS"}, "x")


# -- loading -----------------------------------------------------------------------------------------------------------


def test_load_bot_roundtrip(sample, tmp_path):
    profile, genes, src, _ = sample
    path = tmp_path / "bot_0000.py"
    digest = cf.write_atomic(path, src)
    loaded = cf.load_bot(path, digest)
    assert loaded.profile == profile and loaded.genes == tuple(genes)
    assert loaded.cls.__name__ == class_name(profile)
    assert loaded.module_name.startswith(cf.GENERATED_PACKAGE + ".")
    bot = loaded.spawn()
    assert isinstance(bot, SynthesizedBot) and bot.profile == profile
    assert cf.decode_check(loaded)


def test_load_bot_rejects_digest_mismatch(sample, tmp_path):
    path = tmp_path / "bot_0000.py"
    cf.write_atomic(path, sample[2])
    with pytest.raises(cf.CodeRejected, match="digest"):
        cf.load_bot(path, "0" * 64)


def test_load_bot_rejects_non_utf8_and_oversize(tmp_path):
    p = tmp_path / "bot_0001.py"
    p.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(cf.CodeRejected):
        cf.load_bot(p)
    p.write_bytes(b"#" * (cf.MAX_SOURCE_BYTES + 1))
    with pytest.raises(cf.CodeRejected):
        cf.load_bot(p)


def test_loaded_bot_plays_in_the_arena(sample, tmp_path):
    path = tmp_path / "bot_0000.py"
    cf.write_atomic(path, sample[2])
    bot = cf.load_bot(path).spawn()
    out = BotArena([bot], ArenaConfig(seed=1, slots=3)).run()
    assert out["bots"][0]["class"] == class_name(sample[0])
    assert type(out["bots"][0]["pnl_wei"]) is int


def test_write_atomic_digest_and_no_temp_files(tmp_path):
    p = tmp_path / "sub" / "f.txt"
    d = cf.write_atomic(p, "привет")
    assert d == hashlib.sha256("привет".encode()).hexdigest() == cf.sha256_text("привет")
    assert p.read_text(encoding="utf-8") == "привет"
    assert [x.name for x in p.parent.iterdir()] == ["f.txt"]


def test_concurrent_atomic_writes_never_tear(tmp_path):
    p = tmp_path / "same.py"
    texts = [f"# writer {i}\n" + "x = 1\n" * (200 + i) for i in range(24)]
    with ThreadPoolExecutor(12) as ex:
        digests = list(ex.map(lambda t: cf.write_atomic(p, t), texts))
    final = p.read_text(encoding="utf-8")
    assert final in texts and cf.sha256_text(final) in digests
    assert [x.name for x in tmp_path.iterdir()] == ["same.py"]


def test_concurrent_loads_of_one_file(sample, tmp_path):
    path = tmp_path / "bot_0000.py"
    digest = cf.write_atomic(path, sample[2])
    with ThreadPoolExecutor(16) as ex:
        loaded = list(ex.map(lambda _: cf.load_bot(path, digest), range(64)))
    assert {lb.profile for lb in loaded} == {sample[0]}
    assert len({lb.digest for lb in loaded}) == 1


# -- the factory loop ------------------------------------------------------------------------------------------------------


def test_factory_report_is_clean(factory_run):
    d, report, _ = factory_run
    s = report.summary()
    assert report.ok, s["invariants"]
    assert s["generations"] == 3 and len(s["history"]) == 3
    assert s["files_written"] == 3 * 10 * 2
    assert s["invariants"]["checks"] > 0
    assert set(s["invariants"]["rules"]) == set(cf.FACTORY_CODE_RULES)


def test_factory_files_match_manifest(factory_run):
    d, _, _ = factory_run
    manifest = cf.read_manifest(d)
    assert manifest["generation"] == 2 and len(manifest["bots"]) == 10
    for entry in manifest["bots"]:
        py = Path(entry["py_path"])
        assert hashlib.sha256(py.read_bytes()).hexdigest() == entry["py_digest"]
        assert hashlib.sha256(Path(entry["sol_path"]).read_bytes()).hexdigest() == entry["sol_digest"]
        loaded = cf.load_bot(py, entry["py_digest"])
        assert loaded.profile.generation == 2 and loaded.profile.name == entry["name"]
        assert cf.check_solidity_file(entry["sol_path"], loaded.profile, entry["sol_digest"]) == []


def test_factory_history_and_champions(factory_run):
    d, report, _ = factory_run
    assert len(report.champions) == 3
    for g, champ in enumerate(report.champions):
        assert champ["generation"] == g and Path(champ["py_path"]).exists()
        loaded = cf.load_bot(champ["py_path"], champ["py_digest"])
        assert loaded.fitness is not None and loaded.fitness["pnl_wei"] == champ["pnl_wei"]
    for h in report.history:
        assert h["loss_after"] <= h["loss_before"] + 1e-12
        assert type(h["best_pnl_wei"]) is int and h["best_pnl_wei"] >= 0


def test_factory_events_reach_the_listener(factory_run):
    _, _, events = factory_run
    kinds = [e["event"] for e in events]
    assert kinds.count("generation") == 3 and kinds.count("deployed") == 3
    assert kinds.count("file_written") == 30
    gen = [e for e in events if e["event"] == "generation"]
    assert all("поколение" in e["message"] for e in gen)


def test_factory_rewrites_files_each_generation(tmp_path):
    seen = {}

    def listener(e):
        if e["event"] == "file_written" and e["slot"] == 0:
            seen[e["generation"]] = Path(e["path"]).read_bytes()

    cf.CodeFactory(tmp_path, bots=4, generations=2, heat_size=4, heat_slots=2, listener=listener).run()
    assert seen[0] != seen[1]
    assert b"GENERATION = 1" in Path(tmp_path / "bot_0000.py").read_bytes()


def test_factory_is_deterministic(tmp_path):
    a = cf.CodeFactory(tmp_path / "a", bots=6, generations=2, heat_size=3, heat_slots=3, seed=9).run()
    b = cf.CodeFactory(tmp_path / "b", bots=6, generations=2, heat_size=3, heat_slots=3, seed=9).run()
    da = [e["py_digest"] for e in cf.read_manifest(tmp_path / "a")["bots"]]
    db = [e["py_digest"] for e in cf.read_manifest(tmp_path / "b")["bots"]]
    assert da == db
    assert [h["best_pnl_wei"] for h in a.history] == [h["best_pnl_wei"] for h in b.history]


def test_factory_prunes_stale_slots(tmp_path):
    cf.CodeFactory(tmp_path, bots=6, generations=1, heat_size=3, heat_slots=2).run()
    assert (tmp_path / "bot_0005.py").exists()
    (tmp_path / "notes.txt").write_text("keep", encoding="utf-8")
    cf.CodeFactory(tmp_path, bots=3, generations=1, heat_size=3, heat_slots=2).run()
    names = sorted(p.name for p in tmp_path.iterdir() if p.is_file())
    assert "bot_0005.py" not in names and "bot_0002.sol" in names and "notes.txt" in names


def test_factory_without_solidity(tmp_path):
    r = cf.CodeFactory(tmp_path, bots=3, generations=1, heat_size=3, heat_slots=2, write_solidity=False,
                       keep_champions=False).run()
    assert r.ok and not list(tmp_path.glob("*.sol")) and not (tmp_path / cf.CHAMPIONS_DIR).exists()


def test_factory_stop_event(tmp_path):
    stop = threading.Event()
    stop.set()
    r = cf.CodeFactory(tmp_path, bots=3, generations=3, heat_size=3, heat_slots=2).run(stop=stop)
    assert r.generations == 0


def test_factory_rejects_zero_bots(tmp_path):
    with pytest.raises(cf.ConfigError):
        cf.CodeFactory(tmp_path, bots=0)


def test_disk_heat_task_refuses_tampered_file(factory_run, tmp_path):
    d, _, _ = factory_run
    entry = cf.read_manifest(d)["bots"][0]
    copy = tmp_path / "bot_0000.py"
    copy.write_bytes(Path(entry["py_path"]).read_bytes() + b"\nimport os\n")
    task = (0, 0, [(str(copy), entry["py_digest"])], ArenaConfig().to_dict(), None, 8)
    with pytest.raises(cf.CodeRejected):
        cf._disk_heat_task(task)


def test_parallel_disk_deploy_matches_serial(tmp_path):
    serial = cf.CodeFactory(tmp_path / "s", bots=8, generations=1, heat_size=2, heat_slots=2, seed=4).run()
    with WorkerPool(2, parallel_threshold=1) as wp:
        par = cf.CodeFactory(tmp_path / "p", bots=8, generations=1, heat_size=2, heat_slots=2, seed=4).run(executor=wp)
    assert par.ok and serial.ok
    assert serial.history[0]["best_pnl_wei"] == par.history[0]["best_pnl_wei"]
    assert serial.history[0]["total_pnl_wei"] == par.history[0]["total_pnl_wei"]


# -- block builder integration -----------------------------------------------------------------------------------------------


def test_generated_competitors_from_disk(factory_run):
    d, _, _ = factory_run
    bots = generated_competitors(d)
    assert len(bots) == 10 and all(type(b).__module__.startswith(cf.GENERATED_PACKAGE) for b in bots)
    assert len(generated_competitors(d, limit=3)) == 3


def test_generated_competitors_refuse_modified_sol(factory_run, tmp_path):
    import shutil

    d, _, _ = factory_run
    copy = tmp_path / "bots"
    shutil.copytree(d, copy)
    manifest = json.loads((copy / cf.MANIFEST_NAME).read_text(encoding="utf-8"))
    sol = copy / Path(manifest["bots"][0]["sol_path"]).name
    sol.write_text(sol.read_text(encoding="utf-8").replace("GENERATION = 2;", "GENERATION = 3;"), encoding="utf-8")
    with pytest.raises(cf.CodeRejected):
        generated_competitors(copy)


def test_run_generated_competition_lands_at_most_one_per_victim(factory_run):
    from simulator.tx_graph import eth_competition_scenario

    d, _, _ = factory_run
    sc = eth_competition_scenario(seed=2, victims=4)
    outcome, bundles = run_generated_competition(sc.parent, sc.victims, sc.base_fee, directory=d)
    landed = [b.target_tx for b in bundles if outcome.won(b.bundle_id)]
    assert len(landed) == len(set(landed))
    assert outcome.block.gas_used <= outcome.block.gas_limit


def test_manifest_requires_codegen_version(tmp_path):
    (tmp_path / cf.MANIFEST_NAME).write_text(json.dumps({"codegen_version": 0, "bots": []}), encoding="utf-8")
    with pytest.raises(cf.CodeRejected):
        cf.read_manifest(tmp_path)


def test_resolve_bot_dir(monkeypatch, tmp_path):
    monkeypatch.delenv(cf.BOT_DIR_ENV, raising=False)
    assert cf.resolve_bot_dir() == cf.DEFAULT_BOT_DIR
    monkeypatch.setenv(cf.BOT_DIR_ENV, str(tmp_path))
    assert cf.resolve_bot_dir() == tmp_path
    assert cf.resolve_bot_dir(tmp_path / "x") == tmp_path / "x"


def test_decode_check_matches_factory_genes(pop):
    for i in range(len(pop.profiles)):
        back = decode_profile(np.asarray(pop.genes[i]), pop.profiles[i].name, 0)
        assert back.strategy_key() == pop.profiles[i].strategy_key()
    assert pop.genes.shape[1] == N_GENES


def test_market_untouched_by_loading(sample, tmp_path):
    before = [(p.reserve0, p.reserve1) for p in arena_market()]
    path = tmp_path / "bot_0000.py"
    cf.write_atomic(path, sample[2])
    cf.load_bot(path)
    assert [(p.reserve0, p.reserve1) for p in arena_market()] == before
    assert BuilderState.from_pools(arena_market()).state_root()
