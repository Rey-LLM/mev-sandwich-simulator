"""Layer 15: Russian localization (catalog, sandwich.toml, settings editor, CLI)."""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

from simulator import i18n
from simulator.__main__ import build_parser, main
from simulator.config import ConfigError, SimulationConfig
from simulator.config_sync import (
    COMMON_TABLES,
    SETTINGS,
    SettingsDocument,
    SettingsStore,
    apply_patch,
    default_locations,
    form_schema,
    parse_toml,
    render_layout,
    text_language,
)
from simulator.settings_gui import render_page

ROOT = Path(__file__).resolve().parents[1]
CYR = re.compile(r"[Ѐ-ӿ]")


@pytest.fixture(autouse=True)
def _no_lang_env(monkeypatch):
    monkeypatch.delenv(i18n.LANG_ENV_VAR, raising=False)


# -- catalog -----------------------------------------------------------------------------------------------------


def test_catalog_has_no_structural_issues():
    assert i18n.catalog_issues() == []


@pytest.mark.parametrize("key", sorted(i18n.CATALOG))
def test_every_catalog_entry_is_complete_and_russian(key):
    entry = i18n.CATALOG[key]
    assert set(entry) == set(i18n.LANGUAGES)
    assert i18n.fields_of(entry["ru"]) == i18n.fields_of(entry["en"])
    assert i18n.is_russian(entry["ru"])
    params = {f: 1 for f in i18n.fields_of(entry["en"])}
    assert i18n.t(key, "ru", **params) and i18n.t(key, "en", **params)


@pytest.mark.parametrize("setting", SETTINGS, ids=lambda s: s.attr)
def test_every_setting_has_russian_help(setting):
    text = i18n.SETTING_HELP_RU[setting.attr]
    assert CYR.search(text), setting.attr
    assert i18n.setting_help(setting.attr, setting.help, "ru") == text
    assert i18n.setting_help(setting.attr, setting.help, "en") == setting.help


def test_no_orphan_russian_setting_help():
    assert set(i18n.SETTING_HELP_RU) == {s.attr for s in SETTINGS}


@pytest.mark.parametrize("table", list(COMMON_TABLES) + ["modes", "meta"])
def test_every_table_title_is_translated(table):
    assert CYR.search(i18n.table_title(table, "ru"))


def test_catalog_issues_detects_problems():
    bad = {"a": {"en": "x {n}", "ru": "икс {m}"}, "b": {"en": "y"}, "c": {"en": "z", "ru": "zz"},
           "d": {"en": "q", "ru": "кю", "de": "q"}}
    issues = i18n.catalog_issues(bad)
    assert any(i.startswith("a: fields differ") for i in issues)
    assert any(i.startswith("b: missing ru") for i in issues)
    assert any(i.startswith("c: Russian text has no Cyrillic") for i in issues)
    assert any(i.startswith("d: unknown languages") for i in issues)


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        i18n.t("no.such.key")


@pytest.mark.parametrize("value,expected", [("ru", "ru"), ("RU", "ru"), ("russian", "ru"), ("русский", "ru"),
                                            ("ru_RU", "ru"), ("en", "en"), ("English", "en"), ("en-US", "en")])
def test_normalize_language(value, expected):
    assert i18n.normalize_language(value) == expected


@pytest.mark.parametrize("value", ["de", "", None, "fr-FR"])
def test_normalize_language_rejects(value):
    with pytest.raises(ValueError):
        i18n.normalize_language(value)


def test_resolve_language_precedence(monkeypatch):
    assert i18n.resolve_language() == "ru"
    assert i18n.resolve_language(configured="en") == "en"
    monkeypatch.setenv(i18n.LANG_ENV_VAR, "en")
    assert i18n.resolve_language(configured="ru") == "en"
    assert i18n.resolve_language("ru", configured="en") == "ru"
    monkeypatch.setenv(i18n.LANG_ENV_VAR, "klingon")
    assert i18n.resolve_language(configured="en") == "en"


@pytest.mark.parametrize("n,form", [(0, "файлов"), (1, "файл"), (2, "файла"), (4, "файла"), (5, "файлов"),
                                    (11, "файлов"), (12, "файлов"), (14, "файлов"), (21, "файл"), (22, "файла"),
                                    (101, "файл"), (111, "файлов"), (-3, "файла")])
def test_plural_ru(n, form):
    assert i18n.plural_ru(n, "файл", "файла", "файлов") == form


def test_format_units_is_exact_beyond_float_precision():
    wei = 123_456_789_012_345_678_901_234_567
    assert i18n.format_units(wei, 18, 18, "en") == "123,456,789.012345678901234567"
    assert i18n.format_units(wei, 18, 6, "ru") == f"123{i18n.NBSP}456{i18n.NBSP}789,012345"
    assert i18n.format_units(-1, 18, 18, "en") == "-0.000000000000000001"
    assert i18n.format_units(5, 0, 0, "en") == "5"


def test_group_int_and_decimal():
    assert i18n.group_int(1_234_567, "en") == "1,234,567"
    assert i18n.group_int(-1_234_567, "ru") == f"-1{i18n.NBSP}234{i18n.NBSP}567"
    assert i18n.format_decimal(3.14159, 2, "ru") == "3,14"
    assert i18n.format_decimal(3.14159, 3, "en") == "3.142"


@pytest.mark.parametrize("mode", ["moderate", "moderate_scan", "brutal", "brutal_aggressive", "max", "max_brutal"])
def test_mode_labels_accept_every_spelling(mode):
    assert CYR.search(i18n.mode_label(mode, "ru"))
    assert i18n.mode_note(mode, "en")


# -- sandwich.toml --------------------------------------------------------------------------------------------------


def test_root_sandwich_toml_is_russian_and_canonical():
    text = (ROOT / "sandwich.toml").read_text(encoding="utf-8")
    assert text == render_layout()
    assert "центральный файл настроек" in text
    assert 'language = "ru"' in text


def test_render_layout_languages_hold_the_same_values():
    ru, en = render_layout(lang="ru"), render_layout(lang="en")
    assert CYR.search(ru) and not CYR.search(en)
    assert parse_toml(ru) == parse_toml(en)


def test_render_layout_follows_ui_language_value():
    vals = dict(default_locations())
    vals[("ui", "language")] = "en"
    text = render_layout(vals)
    assert not CYR.search(text) and 'language = "en"' in text


def test_every_comment_line_of_russian_layout_is_russian():
    for line in render_layout(lang="ru").splitlines():
        if "#" in line and "=" in line and not line.lstrip().startswith("#"):
            comment = line.split("#", 1)[1]
            assert CYR.search(comment), line


def test_apply_patch_keeps_the_file_language():
    en = "\n".join(line for line in render_layout(lang="en").splitlines()
                   if not line.startswith(("[ui]", "language "))) + "\n"
    patched = apply_patch(en, {("ui", "language"): "en"})
    assert "[ui]" in patched and 'language = "en"' in patched and not CYR.search(patched)
    ru = render_layout(lang="ru")
    trimmed = "\n".join(line for line in ru.splitlines() if not line.startswith("seed ")) + "\n"
    patched_ru = apply_patch(trimmed, {("run", "seed"): 3})
    assert "seed = 3" in patched_ru and "зерно" in patched_ru


@pytest.mark.parametrize("text,expected", [("", "ru"), ("# комментарий\n", "ru"), ("# comment\n", "en"),
                                           ("[run]\nseed = 1\n", "en")])
def test_text_language(text, expected):
    assert text_language(text) == expected


def test_language_setting_validation():
    assert SimulationConfig.for_mode("moderate").language == "ru"
    assert SimulationConfig.for_mode("moderate", language="en").to_dict()["language"] == "en"
    with pytest.raises(ConfigError):
        SimulationConfig.for_mode("moderate", language="de")


def test_store_init_writes_russian_file(tmp_path):
    store = SettingsStore(tmp_path / "sandwich.toml")
    store.init()
    text = (tmp_path / "sandwich.toml").read_text(encoding="utf-8")
    assert CYR.search(text) and store.load().values[("ui", "language")] == "ru"


# -- settings editor and form schema -------------------------------------------------------------------------------


def _doc(lang):
    vals = dict(default_locations())
    vals[("ui", "language")] = lang
    return SettingsDocument.from_text(render_layout(vals))


def test_settings_editor_page_is_russian_by_default(tmp_path):
    page = render_page(SettingsStore(tmp_path / "s.toml"), _doc("ru"), "tok")
    assert "<html lang=ru>" in page and "Настройки симулятора" in page and "Сохранить" in page
    assert "Матрицы режимов" in page and "Умеренное сканирование" in page


def test_settings_editor_page_follows_file_language(tmp_path):
    page = render_page(SettingsStore(tmp_path / "s.toml"), _doc("en"), "tok")
    assert "<html lang=en>" in page and "Simulator settings" in page and not CYR.search(page.split("<form")[0])


def test_settings_editor_explicit_lang_wins(tmp_path):
    page = render_page(SettingsStore(tmp_path / "s.toml"), _doc("en"), "tok", lang="ru")
    assert "<html lang=ru>" in page


def test_form_schema_local_fields():
    schema = form_schema("ru")
    assert schema["modes"]["language"] == "ru"
    for section in schema["sections"]:
        assert CYR.search(section["title_local"])
        for f in section["fields"]:
            assert CYR.search(f["help_local"])
    assert all(CYR.search(c["label_local"]) for c in schema["modes"]["columns"])
    en = form_schema("en")
    assert en["sections"][0]["fields"][0]["help_local"] == en["sections"][0]["fields"][0]["help"]


# -- CLI -------------------------------------------------------------------------------------------------------------


def test_cli_help_is_russian_by_default():
    text = build_parser().format_help()
    assert "Офлайн-симулятор" in text and "зерно всех детерминированных потоков" in text


def test_cli_help_in_english_from_env(monkeypatch):
    monkeypatch.setenv(i18n.LANG_ENV_VAR, "en")
    text = build_parser().format_help()
    assert "Offline MEV sandwich simulator" in text


def test_cli_error_prefix_is_localized(monkeypatch, capsys):
    rc = main(["--no-config", "--set", "run.seed=oops"])
    assert rc == 2
    assert "simulator: ошибка:" in capsys.readouterr().err
    monkeypatch.setenv(i18n.LANG_ENV_VAR, "en")
    rc = main(["--no-config", "--set", "run.seed=oops"])
    assert rc == 2 and "simulator: error:" in capsys.readouterr().err


def test_cli_lang_flag_sets_language(capsys):
    assert main(["--no-config", "--lang", "en", "--show-config"]) == 0
    out = capsys.readouterr().out
    assert 'language = "en"' in out and "Effective settings" in out
    assert main(["--no-config", "--show-config"]) == 0
    assert "Действующие настройки" in capsys.readouterr().out


def test_cli_json_output_keeps_cyrillic_readable(tmp_path, capsys):
    path = tmp_path / "g.html"
    assert main(["--no-config", "--tx-graph", str(path)]) == 0
    out = capsys.readouterr().out
    assert "Эфириум" in out and "\\u042d" not in out
    assert path.read_text(encoding="utf-8").startswith('<!doctype html><html lang="ru">')


def test_monitor_without_tk_reports_in_russian(monkeypatch, capsys):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **kw):
        if name == "tkinter":
            raise ImportError("no tk")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake)
    assert main(["--no-config", "--monitor"]) == 2
    assert "tkinter недоступен" in capsys.readouterr().err


def test_stream_output_language_independent_keys(capsys):
    assert main(["--no-config", "--form-schema"]) == 0
    out = io.StringIO(capsys.readouterr().out).read()
    assert '"sections"' in out and '"title_local"' in out
