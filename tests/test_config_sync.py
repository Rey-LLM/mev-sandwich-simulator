"""Layers 09 / 10: the sandwich.toml sync engine, its CLI and GUI views, and safe overwrites.

Covers the schema (every SimulationConfig field maps to one TOML location, one
CLI flag and one GUI field), the TOML parser and format-preserving updates,
validation errors, and every way an overwrite can fail: stale digests, held
and stale locks, full disks, short writes, failed renames, failed backups,
symlinks, permissions, and concurrent writers in threads and processes.
"""

import errno
import json
import os
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import fields
from pathlib import Path

import pytest

from simulator.__main__ import build_parser, main
from simulator.config import MODE_KEYS, PRESETS, ConfigError, RunMode, SimulationConfig, resolve_worker_count
from simulator.config_sync import (
    BY_ATTR,
    CONFIG_FILENAME,
    MODE_NAMES,
    PER_MODE_BY_KEY,
    SCHEMA_VERSION,
    SETTINGS,
    FileOps,
    MiniToml,
    SettingsConflictError,
    SettingsDocument,
    SettingsError,
    SettingsFileError,
    SettingsLockError,
    SettingsStore,
    SettingsWriteError,
    TomlSyntaxError,
    apply_patch,
    config_locations,
    default_locations,
    discover_config,
    form_schema,
    lookup,
    parse_assignment,
    parse_toml,
    patch_from_attrs,
    render_layout,
    render_value,
    resolve_config,
    validate_data,
    values_from_args,
)
from simulator.settings_gui import SettingsGui

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
try:
    import tomllib
except ImportError:  # Python 3.10: only the parity tests need it; the rest use the built-in parser
    tomllib = None
needs_tomllib = pytest.mark.skipif(tomllib is None, reason="tomllib ships with Python 3.11+")


@pytest.fixture
def path(tmp_path):
    return tmp_path / CONFIG_FILENAME


@pytest.fixture
def store(path):
    s = SettingsStore(path)
    s.init()
    return s


def leftovers(directory):
    return sorted(p.name for p in Path(directory).iterdir() if p.name.endswith(".tmp") or p.name.endswith(".lock"))


# ---------------------------------------------------------------------------
# One schema, three views
# ---------------------------------------------------------------------------


def test_every_config_field_has_exactly_one_setting():
    assert {f.name for f in fields(SimulationConfig)} - {"extra"} == set(BY_ATTR)
    assert len(BY_ATTR) == len(SETTINGS)


def test_per_mode_settings_mirror_the_run_matrices():
    assert set(PER_MODE_BY_KEY) == set(MODE_KEYS) | {"workers"}


def test_cli_flags_and_destinations_are_unique_and_keep_the_legacy_flags():
    flags = [f for s in SETTINGS for f in s.flags]
    assert len(flags) == len(set(flags)) and len({s.dest for s in SETTINGS}) == len(SETTINGS)
    legacy = {"--mode", "--workers", "--seed", "--blocks", "--txs", "--fuzz", "--layouts", "--gas-limit",
              "--private-share", "--refund", "--max-hops", "--rl-rounds", "--rl-regime", "--trace"}
    assert legacy <= set(flags)


SAMPLES = {
    "mode": ("max", "max"), "seed": ("99", 99), "native_token": ("DAI", "DAI"),
    "default_tolerance_bps": ("75", 75), "start_method": ("forkserver", "forkserver"),
    "parallel_threshold": ("9", 9), "base_fee_wei": ("1_000", 1_000), "priority_fee_wei": ("0", 0),
    "block_gas_limit": ("12000000", 12_000_000), "max_frontrun_bps": ("1500", 1_500),
    "builder_margin_bps": ("250", 250), "bribe_bps": ("7000", 7_000), "mev_share_refund_bps": ("8000", 8_000),
    "rl_regime": ("calm", "calm"), "trace": (None, True), "trace_path": ("trace.log", "trace.log"),
    "audit": (None, False), "audit_fail_on": ("warning", "warning"), "audit_code_scan": (None, False),
    "audit_tick_samples": ("32", 32), "workers": ("1", 1), "cpu_policy": ("full", "full"), "blocks": ("2", 2),
    "txs_per_block": ("5", 5), "frontrun_grid": ("6", 6), "refine_iterations": ("3", 3),
    "fuzz_iterations": ("17", 17), "layout_fuzz_iterations": ("9", 9), "private_flow_bps": ("1234", 1_234),
    "router_max_hops": ("2", 2), "rl_rounds": ("2", 2), "rl_steps_per_shard": ("7", 7),
    "rl_eval_steps": ("11", 11), "shard_density": ("3", 3), "pg_paths": ("13", 13), "pg_path_steps": ("5", 5),
    "pg_batch": ("4", 4), "agent_bots": ("24", 24), "agent_generations": ("3", 3), "agent_heat_size": ("6", 6),
    "agent_heat_slots": ("5", 5),
}


def test_samples_cover_every_setting():
    assert set(SAMPLES) == set(BY_ATTR)


@pytest.mark.parametrize("attr", sorted(SAMPLES))
def test_cli_flag_maps_onto_its_toml_location_and_config_field(attr):
    setting = BY_ATTR[attr]
    text, expected = SAMPLES[attr]
    if setting.kind is bool:
        argv = [setting.flags[0] if expected else "--no-" + setting.flags[0][2:]]
    else:
        argv = [setting.flags[0], text]
    if attr != "mode":
        argv += ["--mode", "brutal"]
    args = build_parser().parse_args(argv)
    values = values_from_args(args)
    assert values[attr] == expected
    mode = "max" if attr == "mode" else "brutal"
    patch = patch_from_attrs(values, mode)
    assert patch[setting.location(mode)] == expected
    cfg = resolve_config({}, patch, cpus=8)
    if attr == "workers":
        assert cfg.workers == resolve_worker_count(1, 8, cfg.policy)
    elif attr == "mode":
        assert cfg.mode is RunMode.MAX_BRUTAL
    else:
        assert getattr(cfg, attr) == expected


def test_form_schema_describes_every_setting():
    schema = form_schema()
    json.dumps(schema)
    common = [f for sec in schema["sections"] for f in sec["fields"]]
    assert {f["attr"] for f in common} | {f["attr"] for f in schema["modes"]["fields"]} == set(BY_ATTR)
    assert [c["mode"] for c in schema["modes"]["columns"]] == list(MODE_NAMES)
    for mode in RunMode:
        for key, value in PRESETS[mode].items():
            assert schema["modes"]["presets"][mode.short][key] == value
    widgets = {f["key"]: f["widget"] for f in common}
    assert widgets["enabled"] == "toggle" and widgets["fail_on"] == "select" and widgets["path"] == "path"


# ---------------------------------------------------------------------------
# Layout, parser and rendering
# ---------------------------------------------------------------------------


def test_repository_sandwich_toml_is_the_canonical_layout():
    text = (REPO / CONFIG_FILENAME).read_text(encoding="utf-8")
    assert text == render_layout()
    doc = SettingsDocument.from_text(text)
    for mode in RunMode:
        for key, value in PRESETS[mode].items():
            assert doc.values[(f"modes.{mode.short}", key)] == value
    assert doc.values[("meta", "schema_version")] == SCHEMA_VERSION


def test_layout_without_comments_parses_to_the_same_values():
    a = parse_toml(render_layout())
    b = parse_toml(render_layout(comments=False))
    assert a == b and "#" not in render_layout(comments=False)


VALID_DOCS = [
    'a = 1\nb = "x\\ty"\n[t]\nc = [1, 2, [3]]\nd = {x = 1, y.z = 2}\n',
    'a = """\nline1\nline2"""\n',
    "a = '''\nraw\\n'''\n",
    'a = """x""""\n',
    'a = """x"""""\n',
    "a = '''x''''\n",
    'a = """\\\n   folded \\\n  text"""\n',
    'a = 0xDEAD_BEEF\nb = 0o755\nc = 0b1010\n',
    'a = 1_000_000\nb = -1.5e-3\nc = +inf\nd = -inf\ne = 1e1_0\n',
    '"quoted key" = 1\n\'lit\' = 2\n',
    '[a.b]\nx = 1\n[a]\ny = 2\n',
    'a.b.c = 1\na.b.d = 2\n',
    '# comment\r\n[x]\r\nk = 1 # trailing\r\n',
    'a = "\\u00e9\\U0001F600"\n',
    'a = []\nb = [ ]\nc = [\n  1,\n  2,\n]\n',
    'a = {}\n',
    'a = 0\nb = 0.0\nc = -0\n',
]


@needs_tomllib
@pytest.mark.parametrize("doc", VALID_DOCS + [render_layout(), render_layout(comments=False)])
def test_builtin_parser_matches_tomllib(doc):
    assert json.dumps(MiniToml(doc).parse(), sort_keys=True) == json.dumps(tomllib.loads(doc), sort_keys=True)


INVALID_DOCS = [
    'a = 1\na = 2\n', '[t]\n[t]\n', 'a = 01\n', 'a = 1__0\n', 'a = _1\n', 'a = 1_\n', 'a = 1_.5\n',
    'a = -0x10\n', 'a = "unterminated\n', 'a = "x"y\n', 'a = [1 2]\n', 'a = {x = 1,}\n', 'a = """x""""""\n',
    'a = """\\  x"""\n', 'a = tru\n', 'a = 1.\n', 'a = .5\n', 'a = "\\q"\n', 'a = 1 b = 2\n', '= 1\n',
    'a.b = 1\na.b.c = 2\n', 'a = "x\ry"\n', 'a = 0x\n',
]


@needs_tomllib
@pytest.mark.parametrize("doc", INVALID_DOCS)
def test_builtin_parser_rejects_what_tomllib_rejects(doc):
    with pytest.raises(tomllib.TOMLDecodeError):
        tomllib.loads(doc)
    with pytest.raises(TomlSyntaxError):
        MiniToml(doc).parse()


@pytest.mark.parametrize("doc", ['d = 1979-05-27\n', '[[arr]]\nx = 1\n'])
def test_builtin_parser_rejects_the_unsupported_subset_clearly(doc):
    with pytest.raises(TomlSyntaxError):
        MiniToml(doc).parse()


def test_syntax_errors_carry_line_numbers():
    with pytest.raises(TomlSyntaxError) as exc:
        MiniToml("[run]\nseed = 1\nseed = 2\n").parse()
    assert exc.value.line == 3
    with pytest.raises(SettingsError) as err:
        SettingsDocument.from_text("[run]\nseed = 1\nseed = 2\n")
    assert err.value.issues[0].line == 3


@pytest.mark.parametrize("value", [0, 7, 12_345, -12_345, 10**30, "", 'a"b\\c', "tab\tnl\nret\r", "\x01ctl\x7f",
                                   "ünïcødé ✓", True, False, [1, 2, 3], 1.5])
@pytest.mark.parametrize("builtin", [False, True])
def test_render_value_round_trips(value, builtin):
    assert parse_toml(f"v = {render_value(value)}\n", prefer_builtin=builtin)["v"] == value


def test_large_integers_get_digit_separators():
    assert render_value(20_000_000_000) == "20_000_000_000" and render_value(1234) == "1234"


def test_bom_is_ignored_by_the_parser():
    assert parse_toml("\ufeffa = 1\n") == {"a": 1} == parse_toml("\ufeffa = 1\n", prefer_builtin=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("doc,needle", [
    ('[gas]\nbase_fee_wei = "lots"\n', "expected an integer"),
    ("[gas]\nbase_fee_wei = 0\n", "below the minimum"),
    ("[modes.max]\nshard_density = 65\n", "above the maximum 64"),
    ('[run]\nmode = "turbo"\n', "unknown run mode"),
    ('[rl]\nregime = "calmm"\n', "did you mean 'calm'"),
    ("[gass]\nx = 1\n", "did you mean 'gas'"),
    ("[gas]\nbase_fee = 1\n", "did you mean 'base_fee_wei'"),
    ("[gas]\nworkers = 2\n", "set per mode"),
    ("[run]\nblocks = 2\n", "set per mode"),
    ("workers = 2\n", "top-level key"),
    ("[modes.turbo]\nworkers = 1\n", "unknown mode"),
    ("[meta]\nschema_version = 99\n", "newer simulator"),
    ('[audit]\nenabled = "yes"\n', "expected true or false"),
    ("[trace]\npath = 5\n", "expected a string"),
    ("[modes]\nmax = 3\n", "must be a table"),
    ("[modes.max]\nworkers = true\n", "expected an integer, got a boolean"),
    ('[modes.max]\ncpu_policy = "turbo"\n', "unknown cpu policy"),
])
def test_invalid_settings_are_rejected_with_a_useful_message(doc, needle):
    with pytest.raises(SettingsError) as exc:
        SettingsDocument.from_text(doc)
    assert needle in str(exc.value)


def test_every_problem_is_reported_at_once_with_lines():
    doc = "[gas]\nbase_fee_wei = 0\n\n[modes.max]\nshard_density = 99\npg_batch = 0\n"
    with pytest.raises(SettingsError) as exc:
        SettingsDocument.from_text(doc)
    assert [(i.location, i.line) for i in exc.value.issues] == [
        ("gas.base_fee_wei", 2), ("modes.max.shard_density", 5), ("modes.max.pg_batch", 6)]


def test_unknown_keys_are_warnings_only_when_allowed():
    data = parse_toml("[extras]\nfoo = 1\n[gas]\nbase_fee_wei = 5\n")
    values, warnings = validate_data(data, allow_unknown=True)
    assert values == {("gas", "base_fee_wei"): 5} and [w.location for w in warnings] == ["extras"]
    with pytest.raises(SettingsError):
        validate_data(data)


def test_values_are_normalised():
    doc = SettingsDocument.from_text('[run]\nmode = "MAX Brutal Simulation"\n[gas]\nblock_gas_limit = 99_000_000\n'
                                     '[modes.max]\ncpu_policy = "100%"\n')
    assert doc.mode == "max" and doc.values[("gas", "block_gas_limit")] == 30_000_000
    assert doc.values[("modes.max", "cpu_policy")] == "full"


def test_invalid_utf8_is_a_settings_error():
    with pytest.raises(SettingsError, match="UTF-8"):
        SettingsDocument.from_bytes(b"[run]\nnative_token = \"\xff\"\n")


@pytest.mark.parametrize("text,location,value", [
    ("gas.base_fee_wei=1_000", ("gas", "base_fee_wei"), 1_000),
    ("modes.max.workers = 3", ("modes.max", "workers"), 3),
    ("trace.enabled=yes", ("trace", "enabled"), True),
    ("run.native_token='DAI'", ("run", "native_token"), "DAI"),
    ("run.mode=maximum", ("run", "mode"), "max"),
    ("gas.priority_fee_wei=0x10", ("gas", "priority_fee_wei"), 16),
])
def test_parse_assignment(text, location, value):
    assert parse_assignment(text) == (location, value)


@pytest.mark.parametrize("text", ["gas.base_fee_wei", "gas.nope=1", "modes.turbo.workers=1", "a.b.c.d=1",
                                  "trace.enabled=maybe", "modes.max.shard_density=0"])
def test_parse_assignment_rejects(text):
    with pytest.raises(SettingsError):
        parse_assignment(text)


def test_lookup_rejects_unknown_locations():
    assert lookup("modes.max", "workers").attr == "workers"
    for table, key in (("modes.turbo", "workers"), ("nowhere", "x"), ("gas", "x")):
        with pytest.raises(SettingsError):
            lookup(table, key)


# ---------------------------------------------------------------------------
# Format-preserving updates
# ---------------------------------------------------------------------------


HAND_WRITTEN = """\
# my settings
[run]
mode = "brutal"   # what I usually run
seed = 7

[gas]
base_fee_wei = 20_000_000_000   # keep in sync with the testnet
"""


def test_apply_patch_changes_only_value_text():
    out = apply_patch(HAND_WRITTEN, {("run", "seed"): 8, ("gas", "base_fee_wei"): 1})
    before, after = HAND_WRITTEN.splitlines(), out.splitlines()
    assert len(before) == len(after)
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert changed == [("seed = 7", "seed = 8"),
                       ("base_fee_wei = 20_000_000_000   # keep in sync with the testnet",
                        "base_fee_wei = 1                # keep in sync with the testnet")]


def test_apply_patch_inserts_missing_keys_and_tables_in_canonical_order():
    out = apply_patch(HAND_WRITTEN, {("gas", "priority_fee_wei"): 9, ("modes.max", "workers"): 2,
                                     ("audit", "fail_on"): "warning"})
    data = parse_toml(out)
    assert data["gas"] == {"base_fee_wei": 20_000_000_000, "priority_fee_wei": 9}
    assert data["audit"] == {"fail_on": "warning"} and data["modes"] == {"max": {"workers": 2}}
    assert out.index("[audit]") < out.index("[modes.max]")
    assert "# MAX Brutal Simulation" in out and out.startswith("# my settings\n")


def test_apply_patch_preserves_crlf_and_missing_final_newline():
    crlf = HAND_WRITTEN.replace("\n", "\r\n")
    out = apply_patch(crlf, {("run", "seed"): 9, ("pbs", "bribe_bps"): 1})
    assert "\r\n" in out and "\n" not in out.replace("\r\n", "")
    bare = "[run]\nseed = 1"
    assert apply_patch(bare, {("run", "seed"): 2}) == "[run]\nseed = 2"


def test_apply_patch_escapes_strings_and_handles_quoted_keys():
    out = apply_patch('[run]\n"native_token" = "WETH"\n', {("run", "native_token"): 'W"E\\TH'})
    assert parse_toml(out)["run"]["native_token"] == 'W"E\\TH'


def test_apply_patch_is_idempotent():
    patch = {("run", "seed"): 3, ("modes.brutal", "blocks"): 4}
    once = apply_patch(render_layout(), patch)
    assert apply_patch(once, patch) == once


def test_apply_patch_refuses_multiline_values():
    with pytest.raises(SettingsError, match="multi-line"):
        apply_patch('[run]\nnative_token = """\nWETH"""\n', {("run", "native_token"): "DAI"})


# ---------------------------------------------------------------------------
# The store: init, update, sync, conflicts
# ---------------------------------------------------------------------------


def test_init_writes_the_layout_and_refuses_to_overwrite(path):
    s = SettingsStore(path)
    doc = s.init()
    assert path.read_text(encoding="utf-8") == render_layout() and doc.values == default_locations()
    with pytest.raises(SettingsFileError, match="already exists"):
        s.init()
    path.write_text(render_layout().replace("seed = 7", "seed = 1"), encoding="utf-8")
    s.init(force=True)
    assert "seed = 1 " in Path(str(path) + ".bak").read_text(encoding="utf-8")
    assert path.read_text(encoding="utf-8") == render_layout() and leftovers(path.parent) == []


def test_update_with_a_stale_digest_is_refused(store, path):
    doc = store.load()
    store.update({("run", "seed"): 1})
    before = path.read_bytes()
    with pytest.raises(SettingsConflictError, match="changed on disk"):
        store.update({("run", "seed"): 2}, expected_digest=doc.digest)
    assert path.read_bytes() == before
    store.update({("run", "seed"): 2}, expected_digest=doc.digest, force=True)
    assert store.load().values[("run", "seed")] == 2


def test_update_of_a_deleted_file_is_a_conflict(store, path):
    doc = store.load()
    path.unlink()
    with pytest.raises(SettingsConflictError, match="was deleted"):
        store.update({("run", "seed"): 2}, expected_digest=doc.digest)
    assert not path.exists()


def test_update_without_a_file(path):
    s = SettingsStore(path)
    with pytest.raises(SettingsFileError, match="does not exist"):
        s.update({("run", "seed"): 2}, create=False)
    s.update({("run", "seed"): 2})
    assert s.load().values[("run", "seed")] == 2 and s.load().values[("gas", "base_fee_wei")] == 20 * 10**9


def test_invalid_update_never_touches_the_file(store, path):
    before = path.read_bytes()
    with pytest.raises(SettingsError):
        store.update({("modes.max", "shard_density"): 10_000})
    with pytest.raises(SettingsError):
        store.update({("nowhere", "x"): 1})
    assert path.read_bytes() == before and leftovers(path.parent) == []


def test_update_keeps_comments_and_unknown_keys(path):
    path.write_text(HAND_WRITTEN + "\n[extras]\nnote = 'kept'\n", encoding="utf-8")
    s = SettingsStore(path)
    s.update({("run", "seed"): 11})
    text = path.read_text(encoding="utf-8")
    assert "# what I usually run" in text and "note = 'kept'" in text and "seed = 11" in text
    assert s.load(allow_unknown=True).warnings[0].location == "extras"


def test_save_text_validates_and_skips_identical_content(store, path):
    with pytest.raises(SettingsError):
        store.save_text("[gas]\nbase_fee_wei = -1\n")
    assert path.read_text(encoding="utf-8") == render_layout()
    mtime = path.stat().st_mtime_ns
    bak = Path(str(path) + ".bak")
    store.save_text(render_layout())
    assert path.stat().st_mtime_ns == mtime and not bak.exists()


def test_backup_holds_the_previous_version(store, path):
    store.update({("run", "seed"): 5})
    first = path.read_bytes()
    store.update({("run", "seed"): 6})
    assert Path(str(path) + ".bak").read_bytes() == first


def test_sync_adds_only_missing_keys(path):
    path.write_text(HAND_WRITTEN + "\n[extras]\nfoo = 1\n", encoding="utf-8")
    s = SettingsStore(path)
    report = s.sync()
    assert report.written and not report.created and report.unknown == ["extras"]
    assert "run.seed" not in report.added and "modes.max.pg_paths" in report.added
    doc = s.load(allow_unknown=True)
    assert set(default_locations()) <= set(doc.values)
    assert doc.values[("run", "mode")] == "brutal" and doc.values[("gas", "base_fee_wei")] == 20 * 10**9
    assert "# keep in sync with the testnet" in path.read_text(encoding="utf-8")
    again = s.sync()
    assert again.added == [] and not again.written


def test_sync_creates_a_missing_file_and_dry_run_writes_nothing(path):
    dry = SettingsStore(path).sync(write=False)
    assert dry.created and not path.exists()
    rep = SettingsStore(path).sync()
    assert rep.created and path.read_text(encoding="utf-8") == render_layout()


def test_sync_refuses_an_invalid_file(path):
    path.write_text("[gas]\nbase_fee_wei = 0\n", encoding="utf-8")
    with pytest.raises(SettingsError):
        SettingsStore(path).sync()
    assert path.read_text(encoding="utf-8") == "[gas]\nbase_fee_wei = 0\n"


def test_bom_and_crlf_survive_updates(path):
    path.write_bytes(b"\xef\xbb\xbf" + render_layout().replace("\n", "\r\n").encode())
    s = SettingsStore(path)
    s.update({("run", "seed"): 3})
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")
    assert s.load().values[("run", "seed")] == 3


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlinks and permission bits")
def test_symlinked_settings_are_written_through(tmp_path):
    target = tmp_path / "real" / CONFIG_FILENAME
    target.parent.mkdir()
    SettingsStore(target).init()
    link = tmp_path / CONFIG_FILENAME
    link.symlink_to(target)
    SettingsStore(link).update({("run", "seed"): 42})
    assert link.is_symlink() and SettingsStore(target).load().values[("run", "seed")] == 42


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_permission_bits_are_preserved(store, path):
    os.chmod(path, 0o600)
    store.update({("run", "seed"): 4})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(Path(str(path) + ".bak").stat().st_mode) == 0o600


def test_a_directory_in_place_of_the_file_is_an_error(path):
    path.mkdir()
    with pytest.raises(SettingsFileError):
        SettingsStore(path).load()
    with pytest.raises(SettingsFileError):
        SettingsStore(path).update({("run", "seed"): 1})


# ---------------------------------------------------------------------------
# Injected faults: the original file always survives byte-for-byte
# ---------------------------------------------------------------------------


class FaultyOps(FileOps):
    def __init__(self, **faults):
        self.faults = faults
        self.replace_calls = 0
        self.sleeps = []

    def write_new(self, path, data):
        mode = self.faults.get("write")
        if mode and (self.faults.get("only") is None or self.faults["only"] in path.name):
            if mode == "enospc":
                super().write_new(path, data[: len(data) // 2])  # a partial temp file is left behind
                raise OSError(errno.ENOSPC, "No space left on device")
            if mode == "short":
                return super().write_new(path, data[:-5])
            if mode == "flip":
                return super().write_new(path, data[:-1] + bytes([data[-1] ^ 1]))
        return super().write_new(path, data)

    def replace(self, src, dst):
        self.replace_calls += 1
        mode = self.faults.get("replace")
        if mode == "sharing" and self.replace_calls <= self.faults.get("times", 99):
            raise PermissionError(errno.EACCES, "sharing violation")
        if mode == "xdev":
            raise OSError(errno.EXDEV, "cross-device link")
        return super().replace(src, dst)

    def chmod(self, path, mode):
        if self.faults.get("chmod"):
            raise PermissionError(errno.EPERM, "chmod refused")
        return super().chmod(path, mode)

    def fsync_dir(self, path):
        if self.faults.get("fsync_dir"):
            raise OSError(errno.EIO, "fsync failed")
        return super().fsync_dir(path)

    def read_bytes(self, path):
        if self.faults.get("read") and path.name == CONFIG_FILENAME:
            raise PermissionError(errno.EACCES, "read refused")
        return super().read_bytes(path)

    def sleep(self, seconds):
        self.sleeps.append(seconds)


@pytest.mark.parametrize("faults,error", [
    (dict(write="enospc"), SettingsWriteError),
    (dict(write="short"), SettingsWriteError),
    (dict(write="flip"), SettingsWriteError),
    (dict(replace="sharing"), SettingsWriteError),
    (dict(replace="xdev"), SettingsWriteError),
    (dict(chmod=True), SettingsWriteError),
    (dict(write="enospc", only=".bak"), SettingsWriteError),
    (dict(read=True), SettingsFileError),
])
def test_failed_overwrites_leave_the_original_intact(store, path, faults, error):
    before = path.read_bytes()
    faulty = SettingsStore(path, ops=FaultyOps(**faults))
    with pytest.raises(error):
        faulty.update({("run", "seed"): 99})
    assert path.read_bytes() == before
    assert leftovers(path.parent) == []
    assert SettingsStore(path).load().values[("run", "seed")] == 7


def test_failed_main_write_after_backup_keeps_the_backup_identical(store, path):
    before = path.read_bytes()
    ops = FaultyOps(write="enospc", only=f".{CONFIG_FILENAME}.{os.getpid()}.")  # the main temp file only
    with pytest.raises(SettingsWriteError):
        SettingsStore(path, ops=ops).update({("run", "seed"): 99})
    assert path.read_bytes() == before == Path(str(path) + ".bak").read_bytes()


def test_transient_sharing_violations_are_retried(store, path):
    ops = FaultyOps(replace="sharing", times=2)
    SettingsStore(path, ops=ops, backup=False).update({("run", "seed"): 5})
    assert SettingsStore(path).load().values[("run", "seed")] == 5
    assert ops.replace_calls == 3 and ops.sleeps == [0.01, 0.02]


def test_persistent_sharing_violation_gives_up_after_the_retry_budget(store, path):
    ops = FaultyOps(replace="sharing")
    with pytest.raises(SettingsWriteError):
        SettingsStore(path, ops=ops, backup=False, replace_retries=4).update({("run", "seed"): 5})
    assert ops.replace_calls == 4 and len(ops.sleeps) == 3


def test_non_transient_rename_errors_are_not_retried(store, path):
    ops = FaultyOps(replace="xdev")
    with pytest.raises(SettingsWriteError):
        SettingsStore(path, ops=ops, backup=False).update({("run", "seed"): 5})
    assert ops.replace_calls == 1 and ops.sleeps == []


def test_directory_fsync_failure_does_not_fail_a_completed_write(store, path):
    SettingsStore(path, ops=FaultyOps(fsync_dir=True)).update({("run", "seed"): 8})
    assert SettingsStore(path).load().values[("run", "seed")] == 8


# ---------------------------------------------------------------------------
# Locks
# ---------------------------------------------------------------------------


def test_a_held_lock_times_out(store, path):
    other = SettingsStore(path, lock_timeout=0.05)
    with store.lock():
        with pytest.raises(SettingsLockError, match="held by another writer"):
            other.update({("run", "seed"): 1})
    other.update({("run", "seed"): 1})
    assert leftovers(path.parent) == []


def test_lock_is_reentrant_within_a_store_and_released_on_errors(store, path):
    lock = Path(str(path) + ".lock")
    with pytest.raises(RuntimeError):
        with store.lock():
            with store.lock():
                store.update({("run", "seed"): 3})
                assert lock.exists()
            assert lock.exists()
            raise RuntimeError("boom")
    assert not lock.exists()


def _write_lock(path, payload, age_s=0.0):
    lock = Path(str(path) + ".lock")
    lock.write_text(payload, encoding="utf-8")
    if age_s:
        old = time.time() - age_s
        os.utime(lock, (old, old))
    return lock


def _dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


@pytest.mark.parametrize("payload,age,breaks", [
    ("1 elsewhere 0\n", 600.0, True),            # older than stale_lock_s
    ("1 elsewhere 0\n", 0.0, False),             # fresh lock from another host: wait
    ("garbage\n", 5.0, False),                   # unreadable payload without a host: wait for the age limit
    ("{pid} {host} 0\n", 0.0, False),            # our own live process
])
def test_stale_lock_detection(store, path, payload, age, breaks):
    _write_lock(path, payload.format(pid=os.getpid(), host=socket.gethostname()), age)
    s = SettingsStore(path, lock_timeout=0.05, stale_lock_s=300)
    if breaks:
        s.update({("run", "seed"): 2})
        assert s.load().values[("run", "seed")] == 2
    else:
        with pytest.raises(SettingsLockError):
            s.update({("run", "seed"): 2})


@pytest.mark.skipif(os.name != "posix", reason="liveness check uses POSIX signals")
def test_lock_of_a_dead_local_process_is_broken(store, path):
    _write_lock(path, f"{_dead_pid()} {socket.gethostname()} 0\n")
    SettingsStore(path, lock_timeout=1.0).update({("run", "seed"): 4})
    assert SettingsStore(path).load().values[("run", "seed")] == 4


def test_garbage_local_lock_older_than_a_second_is_broken(store, path):
    _write_lock(path, f"notapid {socket.gethostname()} 0\n", age_s=5.0)
    SettingsStore(path, lock_timeout=1.0).update({("run", "seed"): 6})
    assert SettingsStore(path).load().values[("run", "seed")] == 6


def test_lock_in_a_missing_directory(tmp_path):
    with pytest.raises(SettingsLockError, match="does not exist"):
        SettingsStore(tmp_path / "missing" / CONFIG_FILENAME).update({("run", "seed"): 1})


def test_pid_liveness():
    ops = FileOps()
    if os.name == "posix":
        assert ops.pid_alive(os.getpid()) is True and ops.pid_alive(_dead_pid()) is False
    assert ops.pid_alive(0) is None


# ---------------------------------------------------------------------------
# Concurrent writers
# ---------------------------------------------------------------------------


def _increment(path, times):
    s = SettingsStore(path, lock_timeout=30)
    done = conflicts = 0
    while done < times:
        doc = s.load()
        try:
            s.update({("run", "seed"): doc.values[("run", "seed")] + 1}, expected_digest=doc.digest)
            done += 1
        except SettingsConflictError:
            conflicts += 1
    return conflicts


def test_threaded_read_modify_write_loses_no_update(store, path):
    errors, conflicts = [], []

    def worker():
        try:
            conflicts.append(_increment(path, 8))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert not errors
    assert SettingsStore(path).load().values[("run", "seed")] == 7 + 48
    assert leftovers(path.parent) == []


def test_concurrent_writers_of_different_keys_keep_every_value(store, path):
    keys = ["blocks", "txs_per_block", "frontrun_grid", "fuzz_iterations", "rl_rounds", "pg_paths"]
    errors = []

    def worker(key):
        try:
            s = SettingsStore(path, lock_timeout=30)
            for i in range(1, 6):
                s.update({("modes.max", key): i})
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in keys]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert not errors
    doc = SettingsStore(path).load()
    assert all(doc.values[("modes.max", k)] == 5 for k in keys)


def test_readers_never_observe_a_torn_file(store, path):
    stop = threading.Event()
    seen = {i: [] for i in range(3)}
    bad = []

    def reader(i):
        while not stop.is_set():
            raw = path.read_bytes()
            try:
                seen[i].append(SettingsDocument.from_bytes(raw).values[("run", "seed")])
            except Exception as exc:  # pragma: no cover - reported below
                bad.append((exc, raw[:80]))

    readers = [threading.Thread(target=reader, args=(i,)) for i in seen]
    for t in readers:
        t.start()
    try:
        _increment(path, 30)
    finally:
        stop.set()
        for t in readers:
            t.join(30)
    assert bad == []
    for values in seen.values():
        assert values and values == sorted(values) and 7 <= values[0] and values[-1] <= 37


WORKER_SCRIPT = """
import sys
from simulator.config_sync import SettingsConflictError, SettingsStore

path, times = sys.argv[1], int(sys.argv[2])
store = SettingsStore(path, lock_timeout=60)
done = 0
while done < times:
    doc = store.load()
    try:
        store.update({("run", "seed"): doc.values[("run", "seed")] + 1}, expected_digest=doc.digest)
        done += 1
    except SettingsConflictError:
        pass
"""


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["MEV_SIM_CONFIG"] = "none"
    return env


def test_processes_increment_concurrently_without_lost_updates(store, path, tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(WORKER_SCRIPT, encoding="utf-8")
    procs = [subprocess.Popen([sys.executable, str(script), str(path), "10"], env=_env()) for _ in range(4)]
    assert [p.wait(timeout=180) for p in procs] == [0, 0, 0, 0]
    assert SettingsStore(path).load().values[("run", "seed")] == 7 + 40
    assert leftovers(path.parent) == []


def test_concurrent_cli_saves_either_land_or_report_a_conflict(store, path):
    flags = [("--blocks", "3", "blocks"), ("--txs", "9", "txs_per_block"), ("--fuzz", "77", "fuzz_iterations"),
             ("--pg-paths", "64", "pg_paths"), ("--layouts", "5", "layout_fuzz_iterations")]
    procs = [subprocess.Popen([sys.executable, "-m", "simulator", "--config", str(path), "--mode", "max",
                               "--save-config", flag, value], env=_env(), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for flag, value, _ in flags]
    results = [(p.wait(timeout=180), p.stderr.read()) for p in procs]
    for p in procs:
        p.stdout.close()
        p.stderr.close()
    doc = SettingsStore(path).load()
    landed = 0
    for (rc, err), (_, value, key) in zip(results, flags):
        assert rc in (0, 2), err
        if rc == 0:
            landed += 1
            assert doc.values[("modes.max", key)] == int(value)
        else:
            assert "changed on disk" in err or "held by another writer" in err
    assert landed >= 1 and leftovers(path.parent) == []


# ---------------------------------------------------------------------------
# Discovery and precedence
# ---------------------------------------------------------------------------


def test_discovery_walks_up_to_the_repository_root(tmp_path, monkeypatch):
    monkeypatch.delenv("MEV_SIM_CONFIG", raising=False)
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "a" / "b").mkdir(parents=True)
    (tmp_path / CONFIG_FILENAME).write_text("", encoding="utf-8")  # above the repository: never used
    assert discover_config(repo / "a" / "b") is None
    (repo / CONFIG_FILENAME).write_text("", encoding="utf-8")
    assert discover_config(repo / "a" / "b") == (repo / CONFIG_FILENAME).resolve()
    (repo / "a" / CONFIG_FILENAME).write_text("", encoding="utf-8")
    assert discover_config(repo / "a" / "b") == (repo / "a" / CONFIG_FILENAME).resolve()


@pytest.mark.parametrize("value,expected", [("none", None), ("off", None), ("", None), ("/x/y.toml", "/x/y.toml")])
def test_discovery_environment_override(monkeypatch, value, expected):
    monkeypatch.setenv("MEV_SIM_CONFIG", value)
    got = discover_config()
    assert (None if got is None else str(got)) == expected


def test_precedence_preset_file_env_cli(monkeypatch):
    file_values = {("run", "mode"): "max", ("modes.max", "blocks"): 3, ("modes.max", "workers"): 2,
                   ("modes.brutal", "blocks"): 99}
    cfg = resolve_config(file_values, {}, cpus=8)
    assert cfg.mode is RunMode.MAX_BRUTAL and cfg.blocks == 3 and cfg.workers == 2
    assert cfg.txs_per_block == PRESETS[RunMode.MAX_BRUTAL]["txs_per_block"]
    monkeypatch.setenv("MEV_SIM_WORKERS", "5")
    assert resolve_config(file_values, {}, cpus=8).workers == 5
    cli = {("modes.max", "workers"): 7, ("modes.max", "blocks"): 4}
    cfg = resolve_config(file_values, cli, cpus=8)
    assert cfg.workers == 7 and cfg.blocks == 4
    other = resolve_config(file_values, {("run", "mode"): "brutal"}, cpus=8)
    assert other.mode is RunMode.BRUTAL_AGGRESSIVE and other.blocks == 99


def test_zero_workers_in_the_file_means_automatic():
    cfg = resolve_config({("modes.max", "workers"): 0, ("run", "mode"): "max"}, {}, cpus=6)
    assert cfg.workers == 6


@pytest.mark.parametrize("mode", list(MODE_NAMES))
def test_config_locations_round_trip(mode):
    cfg = SimulationConfig.for_mode(mode, cpus=8, seed=3, trace=True, trace_path="t.log")
    assert resolve_config(config_locations(cfg), {}, cpus=8) == cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_init_refuse_and_force(path, capsys):
    assert main(["--config", str(path), "--init-config"]) == 0
    assert json.loads(capsys.readouterr().out)["written"]
    path.write_text(path.read_text(encoding="utf-8").replace("seed = 7", "seed = 5"), encoding="utf-8")
    before = path.read_bytes()
    assert main(["--config", str(path), "--init-config"]) == 2
    assert "already exists" in capsys.readouterr().err and path.read_bytes() == before
    assert main(["--config", str(path), "--init-config", "--force"]) == 0
    assert Path(str(path) + ".bak").read_bytes() == before


def test_cli_save_writes_flags_into_the_active_mode_table(store, path, capsys):
    rc = main(["--config", str(path), "--save-config", "--mode", "max", "--workers", "3", "--pg-paths", "64",
               "--trace", "--trace-file", "out.log", "--set", "gas.base_fee_wei=5"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and "modes.max.workers" in out["written"]
    data = parse_toml(path.read_text(encoding="utf-8"))
    assert data["run"]["mode"] == "max" and data["modes"]["max"]["workers"] == 3
    assert data["modes"]["max"]["pg_paths"] == 64 and data["modes"]["brutal"]["pg_paths"] == 0
    assert data["trace"] == {"enabled": True, "path": "out.log"} and data["gas"]["base_fee_wei"] == 5


def test_cli_save_uses_the_files_mode_when_no_mode_flag(store, path, capsys):
    store.update({("run", "mode"): "brutal"})
    assert main(["--config", str(path), "--save-config", "--blocks", "2"]) == 0
    assert parse_toml(path.read_text(encoding="utf-8"))["modes"]["brutal"]["blocks"] == 2


@pytest.mark.parametrize("argv,needle", [
    (["--save-config"], "needs at least one"),
    (["--save-config", "--set", "modes.turbo.workers=1"], "unknown mode"),
    (["--save-config", "--set", "gas.base_fee_wei=0"], "below the minimum"),
])
def test_cli_save_errors_exit_2_and_leave_the_file(store, path, capsys, argv, needle):
    before = path.read_bytes()
    assert main(["--config", str(path)] + argv) == 2
    assert needle in capsys.readouterr().err and path.read_bytes() == before


def test_cli_sync_reports_added_and_unknown_keys(path, capsys):
    path.write_text("[run]\nseed = 3\n[extras]\nx = 1\n", encoding="utf-8")
    assert main(["--config", str(path), "--sync-config"]) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert "gas.base_fee_wei" in report["added"] and report["unknown"] == ["extras"]
    assert "extras: unknown setting kept as is" in captured.err
    assert SettingsStore(path).load(allow_unknown=True).values[("run", "seed")] == 3


def test_cli_show_config_applies_precedence(store, path, capsys, monkeypatch):
    store.update({("run", "mode"): "brutal", ("modes.brutal", "blocks"): 3, ("modes.brutal", "workers"): 1})
    assert main(["--config", str(path), "--show-config"]) == 0
    shown = parse_toml(capsys.readouterr().out)
    assert shown["run"]["mode"] == "brutal" and shown["modes"]["brutal"]["blocks"] == 3
    assert main(["--config", str(path), "--show-config", "--blocks", "5"]) == 0
    assert parse_toml(capsys.readouterr().out)["modes"]["brutal"]["blocks"] == 5
    monkeypatch.setenv("MEV_SIM_WORKERS", "1")
    assert main(["--config", str(path), "--show-config", "--mode", "max"]) == 0
    assert parse_toml(capsys.readouterr().out)["modes"]["max"]["workers"] == 1


def test_cli_reads_the_file_named_by_the_environment(store, path, capsys, monkeypatch):
    store.update({("run", "mode"): "max"})
    monkeypatch.setenv("MEV_SIM_CONFIG", str(path))
    assert main(["--show-config"]) == 0
    assert parse_toml(capsys.readouterr().out)["run"]["mode"] == "max"
    assert main(["--show-config", "--no-config"]) == 0
    assert parse_toml(capsys.readouterr().out)["run"]["mode"] == "moderate"


def test_cli_rejects_an_invalid_settings_file_with_its_line(path, capsys):
    path.write_text("[run]\nseed = 1\n\n[modes.max]\nshard_density = 0\n", encoding="utf-8")
    assert main(["--config", str(path), "--show-config"]) == 2
    err = capsys.readouterr().err
    assert "modes.max.shard_density (line 5)" in err and "below the minimum" in err


def test_cli_no_config_cannot_write(capsys):
    assert main(["--no-config", "--sync-config"]) == 2
    assert "--no-config" in capsys.readouterr().err


def test_cli_form_schema(capsys):
    assert main(["--form-schema"]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(form_schema()))


def test_cli_run_uses_the_settings_file(store, path, capsys):
    store.update({("modes.moderate", "txs_per_block"): 5, ("modes.moderate", "fuzz_iterations"): 8})
    assert main(["--config", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["result"]["txs"] == 5 and report["config"]["fuzz_iterations"] == 8
    assert report["audit"]["enabled"] and report["resources"]["cpu_policy"] == "capped"


def test_cli_cpu_probe(capsys):
    assert main(["--mode", "max", "--workers", "2", "--cpu-probe", "0.1"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["cpu_policy"] == "full" and data["rendezvous"]["released"]
    assert data["saturation"]["tasks"] == data["rendezvous"]["workers"]
    assert main(["--cpu-probe", "0"]) == 2


# ---------------------------------------------------------------------------
# GUI (browser settings editor)
# ---------------------------------------------------------------------------


def _request(gui, route, data=None, headers=None, follow=True):
    req = urllib.request.Request(gui.url.rstrip("/") + route, data=data, headers=headers or {})
    opener = urllib.request.build_opener() if follow else urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8"), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8"), dict(exc.headers)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


@pytest.fixture
def gui(store):
    with SettingsGui(store) as g:
        yield g


def _post_json(gui, values, digest=None, token=None, force=False):
    body = json.dumps({"values": values, "digest": digest, "force": force}).encode()
    return _request(gui, "/api/settings", body, {"X-Settings-Token": gui.token if token is None else token})


def test_gui_page_has_a_field_for_every_location(gui):
    status, page, headers = _request(gui, "/")
    assert status == 200 and "no-store" in headers["Cache-Control"]
    for table, key in default_locations():
        if table != "meta":
            assert f'name="{table}.{key}"' in page, (table, key)
    assert gui.token in page


def test_gui_schema_and_settings_endpoints(gui, store):
    status, body, _ = _request(gui, "/api/schema")
    assert status == 200 and json.loads(body) == json.loads(json.dumps(form_schema()))
    status, body, _ = _request(gui, "/api/settings")
    data = json.loads(body)
    assert data["digest"] == store.load().digest and data["values"]["modes.max.cpu_policy"] == "full"
    assert _request(gui, "/nope")[0] == 404


def test_gui_json_save_and_conflict(gui, store, path):
    digest = store.load().digest
    status, body, _ = _post_json(gui, {"modes.max.workers": 3, "audit.fail_on": "warning"}, digest)
    assert status == 200 and json.loads(body)["written"] == ["audit.fail_on", "modes.max.workers"]
    before = path.read_bytes()
    status, body, _ = _post_json(gui, {"modes.max.workers": 9}, digest)  # page loaded before the save
    assert status == 409 and path.read_bytes() == before
    assert _post_json(gui, {"modes.max.workers": 9}, digest, force=True)[0] == 200


def test_gui_rejects_invalid_values_without_writing(gui, path):
    before = path.read_bytes()
    status, body, _ = _post_json(gui, {"modes.max.shard_density": 999, "gas.nope": 1})
    issues = json.loads(body)["issues"]
    assert status == 400 and len(issues) == 2 and path.read_bytes() == before


def test_gui_requires_the_token_and_a_loopback_host(gui, path):
    before = path.read_bytes()
    assert _post_json(gui, {"run.seed": 1}, token="wrong")[0] == 403
    status, _, _ = _request(gui, "/api/settings", headers={"Host": "evil.example"})
    assert status == 403 and path.read_bytes() == before


def test_gui_form_post_saves_changed_values(gui, store):
    doc = store.load()
    form = {"_token": gui.token, "_digest": doc.digest, "trace.enabled": ["false", "true"],
            "audit.code_scan": ["false"], "modes.brutal.blocks": "3", "run.seed": "7"}
    body = urllib.parse.urlencode(form, doseq=True).encode()
    status, _, headers = _request(gui, "/", body, follow=False)
    assert status == 303 and headers["Location"] == "/?saved=1"
    after = store.load()
    assert after.values[("trace", "enabled")] is True and after.values[("audit", "code_scan")] is False
    assert after.values[("modes.brutal", "blocks")] == 3
    assert "Saved." in _request(gui, "/?saved=1")[1]


def test_gui_form_errors_and_stale_pages(gui, store, path):
    doc = store.load()
    bad = urllib.parse.urlencode({"_token": gui.token, "_digest": doc.digest, "modes.max.pg_batch": "0"}).encode()
    status, page, _ = _request(gui, "/", bad)
    assert status == 400 and "below the minimum" in page and 'value="0"' in page
    store.update({("run", "seed"): 1})
    before = path.read_bytes()
    stale = urllib.parse.urlencode({"_token": gui.token, "_digest": doc.digest, "run.seed": "2"}).encode()
    status, page, _ = _request(gui, "/", stale)
    assert status == 409 and "changed after this page was loaded" in page and path.read_bytes() == before
    assert _request(gui, "/", urllib.parse.urlencode({"_token": "x"}).encode())[0] == 403


def test_gui_reports_a_held_lock(path):
    s = SettingsStore(path, lock_timeout=0.05)
    s.init()
    holder = SettingsStore(path)
    with SettingsGui(s) as g, holder.lock():
        assert _post_json(g, {"run.seed": 1})[0] == 423


def test_gui_rejects_oversized_bodies(gui):
    status, _, _ = _request(gui, "/api/settings", b"x" * (300 * 1024), {"X-Settings-Token": gui.token})
    assert status == 413


def test_cli_gui_flag_is_wired(monkeypatch, path, capsys):
    import simulator.settings_gui as sg

    calls = {}
    monkeypatch.setattr(sg, "serve", lambda store, port, out: calls.update(path=store.path, port=port))
    assert main(["--config", str(path), "--gui", "--port", "0"]) == 0
    assert calls == {"path": path, "port": 0}
