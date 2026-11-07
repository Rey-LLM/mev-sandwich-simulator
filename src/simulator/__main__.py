"""Command line: ``python -m simulator [--mode moderate|brutal|max] [setting flags]``.

Every setting flag is generated from :data:`simulator.config_sync.SETTINGS`,
the same schema that lays out ``sandwich.toml`` and describes the GUI form
(``--form-schema``), so the three views cannot drift apart. Settings resolve
as: the mode's built-in preset < ``sandwich.toml`` < ``$MEV_SIM_WORKERS``
(worker count only) < ``--set TABLE.KEY=VALUE`` < setting flags.

The settings file is ``--config PATH``, else ``$MEV_SIM_CONFIG`` (``none``
disables it), else the nearest ``sandwich.toml`` from the working directory
up to the repository root; ``--no-config`` ignores it.

Actions (each runs instead of a simulation and then exits):

``--init-config``   write the default layout (``--force`` overwrites)
``--sync-config``   add keys this version knows but the file lacks
``--save-config``   write the setting flags and ``--set`` values given into the file
``--show-config``   print the effective settings as TOML
``--form-schema``   print the GUI form description (JSON)
``--gui``           serve the settings editor on http://127.0.0.1 (``--port``)
``--audit-only``    run the integrity audit and print it
``--audit-rules``   print the integrity rule catalog (JSON)
``--cpu-probe S``   saturate the worker pool for S seconds and print its telemetry

MEV pattern store actions (Layer 12; the store is ``[patterns] store`` /
``--pattern-store DIR``):

``--ingest-patterns FILE...``       normalise JSON / JSONL / CSV exports into the store
``--build-synthetic-patterns N``    add N generated reference patterns (``source = "synthetic"``)
``--pattern-stats``                 print counts, category means and storage use
``--pattern-verify``                stream every segment and check it against the index
``--export-patterns FILE``          write the store as normalised JSONL
``--pattern-source NAME``           force the parser for ``--ingest-patterns``

Attack catalog and strategy discovery (Layer 13):

``--attack-catalog``      run every attack type on a seeded scenario (``--scenario-victims``)
``--discover``            search for new attack compositions (``--generations``,
                          ``--population``) and save novel ones to the local library
``--strategies``          list the local strategy library (``--strategy-dir``)
``--verify-strategies``   replay every stored strategy and check its profit to the wei

Exit codes: 0 success; 1 an invariant failed during the run; 2 invalid
settings, a settings file that cannot be read or written safely, or a
blocking integrity-audit finding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from .config import ConfigError, SimulationConfig
from .config_sync import (
    CONFIG_FILENAME,
    Location,
    SettingsFileError,
    SettingsStore,
    add_setting_arguments,
    discover_config,
    effective_mode,
    form_schema,
    parse_assignment,
    patch_from_attrs,
    render_effective,
    resolve_config,
    values_from_args,
)
from .controller import SimulationController, invariants_ok
from .parallel import WorkerPool, rendezvous_probe, saturation_probe
from .validator import IntegrityError, IntegrityValidator, rule_catalog

EXIT_OK = 0
EXIT_INVARIANT = 1
EXIT_SETTINGS = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simulator",
        description="Offline MEV sandwich simulator. Settings come from the mode preset, sandwich.toml, "
                    "$MEV_SIM_WORKERS, --set and the flags below, in that order.",
    )
    files = parser.add_argument_group("Settings file")
    source = files.add_mutually_exclusive_group()
    source.add_argument("--config", metavar="PATH",
                        help=f"settings file (default: $MEV_SIM_CONFIG, else the nearest {CONFIG_FILENAME} "
                             "up to the repository root)")
    source.add_argument("--no-config", action="store_true", help="ignore every settings file")
    files.add_argument("--set", dest="assignments", action="append", default=[], metavar="TABLE.KEY=VALUE",
                       help="set any value by its sandwich.toml location, e.g. gas.base_fee_wei=1e9 or "
                            "modes.max.workers=8 (repeatable)")
    files.add_argument("--force", action="store_true", help="let --init-config overwrite an existing file")
    files.add_argument("--port", type=int, default=8765, help="port for --gui (0 = any free port)")
    actions = parser.add_argument_group("Actions (instead of a simulation)").add_mutually_exclusive_group()
    actions.add_argument("--init-config", action="store_true", help="write the default settings layout")
    actions.add_argument("--sync-config", action="store_true", help="add keys this version knows but the file lacks")
    actions.add_argument("--save-config", action="store_true",
                         help="write the setting flags and --set values given here into the settings file")
    actions.add_argument("--show-config", action="store_true", help="print the effective settings as TOML")
    actions.add_argument("--form-schema", action="store_true", help="print the GUI form description as JSON")
    actions.add_argument("--gui", action="store_true", help="serve the settings editor on http://127.0.0.1:PORT")
    actions.add_argument("--audit-only", action="store_true", help="run the integrity audit only")
    actions.add_argument("--audit-rules", action="store_true", help="print the integrity rule catalog as JSON")
    actions.add_argument("--cpu-probe", type=float, metavar="SECONDS",
                         help="saturate the worker pool for SECONDS and print utilization telemetry")
    actions.add_argument("--ingest-patterns", nargs="+", metavar="FILE",
                         help="normalise MEV pattern exports (JSON, JSONL, CSV) into the pattern store")
    actions.add_argument("--build-synthetic-patterns", type=int, metavar="N",
                         help="add N generated reference patterns to the pattern store")
    actions.add_argument("--pattern-stats", action="store_true", help="print pattern store statistics")
    actions.add_argument("--pattern-verify", action="store_true",
                         help="check every pattern segment frame against the index")
    actions.add_argument("--export-patterns", metavar="FILE", help="export the pattern store as JSONL")
    actions.add_argument("--attack-catalog", action="store_true",
                         help="run every attack type (sandwich, multi-victim, JIT, back-run arbitrage, "
                              "liquidation back-run, oracle manipulation) on a seeded scenario")
    actions.add_argument("--discover", action="store_true",
                         help="search for new attack compositions and save the novel ones to the local library")
    actions.add_argument("--strategies", action="store_true", help="list the local strategy library")
    actions.add_argument("--verify-strategies", action="store_true",
                         help="replay every stored strategy and check its profit to the wei")
    disc = parser.add_argument_group("Attack catalog and strategy discovery")
    disc.add_argument("--scenario-victims", type=int, default=10, metavar="N",
                      help="pending victim swaps per generated scenario (default 10)")
    disc.add_argument("--generations", type=int, default=40, help="discovery generations (default 40)")
    disc.add_argument("--population", type=int, default=64, help="candidates per generation (default 64)")
    disc.add_argument("--strategy-dir", metavar="DIR",
                      help="local strategy library (default $MEV_SIM_STRATEGY_DIR, else ~/.mev_sim/strategies)")
    disc.add_argument("--no-save", action="store_true", help="--discover: report only, write nothing")
    files.add_argument("--pattern-source", choices=("flashbots_blocks", "mev_inspect", "advisory", "normalized"),
                       help="parser for --ingest-patterns (default: detected per object)")
    add_setting_arguments(parser)
    return parser


def _dump(data: Any, out: TextIO) -> None:
    json.dump(data, out, indent=2, default=str)
    out.write("\n")


def _settings_path(args: argparse.Namespace, for_writing: bool) -> Optional[Path]:
    if args.no_config:
        return None
    if args.config:
        return Path(args.config)
    found = discover_config()
    if found is None and for_writing:
        return Path.cwd() / CONFIG_FILENAME
    return found


def _cli_values(args: argparse.Namespace, file_values: Dict[Location, Any]) -> Dict[Location, Any]:
    """Locations set on the command line: ``--set`` first, then setting flags (which win)."""
    out: Dict[Location, Any] = {}
    for text in args.assignments:
        location, value = parse_assignment(text)
        out[location] = value
    attrs = values_from_args(args)
    mode = effective_mode({**file_values, **out}, attrs.get("mode"))
    out.update(patch_from_attrs(attrs, mode))
    return out


def _warn(issues: List[Any], err: TextIO) -> None:
    for issue in issues:
        err.write(f"simulator: warning: {issue}\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out, err = sys.stdout, sys.stderr
    try:
        return _main(args, out, err)
    except IntegrityError as exc:
        err.write(f"simulator: {exc}\n")
        return EXIT_SETTINGS
    except (ConfigError, SettingsFileError) as exc:  # SettingsError is a ConfigError
        err.write(f"simulator: error: {exc}\n")
        return EXIT_SETTINGS


def _main(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if args.form_schema:
        _dump(form_schema(), out)
        return EXIT_OK
    if args.audit_rules:
        _dump(rule_catalog(), out)
        return EXIT_OK
    writing = args.init_config or args.sync_config or args.save_config or args.gui
    if writing and args.no_config:
        raise ConfigError("--no-config cannot be combined with an action that writes the settings file")
    path = _settings_path(args, writing)
    store = SettingsStore(path) if path is not None else None

    if args.init_config:
        assert store is not None
        doc = store.init(force=args.force)
        _dump({"path": str(store.path), "written": True, "digest": doc.digest}, out)
        return EXIT_OK
    if args.sync_config:
        assert store is not None
        report = store.sync()
        _warn([f"{u}: unknown setting kept as is" for u in report.unknown], err)
        _dump(report.to_dict(), out)
        return EXIT_OK
    if args.gui:
        assert store is not None
        from .settings_gui import serve

        serve(store, port=args.port, out=out)
        return EXIT_OK

    doc = store.load(allow_unknown=True) if store is not None else None
    if doc is not None:
        _warn(doc.warnings, err)
    file_values = dict(doc.values) if doc is not None else {}
    cli_values = _cli_values(args, file_values)

    if args.save_config:
        assert store is not None
        if not cli_values:
            raise ConfigError("--save-config needs at least one setting flag or --set value to write")
        saved = store.update(cli_values, expected_digest=doc.digest if doc is not None else None)
        _dump({"path": str(store.path), "written": sorted(f"{t}.{k}" for t, k in cli_values),
               "digest": saved.digest}, out)
        return EXIT_OK

    cfg: SimulationConfig = resolve_config(file_values, cli_values)
    if args.show_config:
        out.write(render_effective(cfg, file_values))
        return EXIT_OK
    if args.audit_only:
        report = IntegrityValidator.for_config(cfg).run()
        _dump({"mode": cfg.mode.value, "mode_label": cfg.mode.label, "audit": report.to_dict(max_findings=500)}, out)
        return EXIT_OK if report.ok else EXIT_SETTINGS
    if args.cpu_probe is not None:
        if args.cpu_probe <= 0:
            raise ConfigError("--cpu-probe needs a positive number of seconds")
        with WorkerPool.for_config(cfg, instrument=True) as wp:
            barrier = rendezvous_probe(wp)
            saturation = saturation_probe(wp, seconds=args.cpu_probe)
            _dump({"mode": cfg.mode.value, "cpu_policy": cfg.cpu_policy, "rendezvous": barrier,
                   "saturation": saturation, "telemetry": wp.telemetry()}, out)
        return EXIT_OK

    if (args.ingest_patterns or args.build_synthetic_patterns is not None or args.pattern_stats
            or args.pattern_verify or args.export_patterns):
        return _pattern_action(args, cfg, out, err)

    if args.attack_catalog or args.discover or args.strategies or args.verify_strategies:
        return _strategy_action(args, cfg, out)

    controller = SimulationController(cfg)
    try:
        report = controller.run()
    except IntegrityError as exc:
        _dump({"mode": cfg.mode.value, "mode_label": cfg.mode.label,
               "audit": exc.report.to_dict(max_findings=500)}, out)
        raise
    _dump(report, out)
    return EXIT_OK if invariants_ok(report) else EXIT_INVARIANT


def _pattern_action(args: argparse.Namespace, cfg: SimulationConfig, out: TextIO, err: TextIO) -> int:
    from .pattern_ingestor import IngestReport, open_store, synthetic_corpus, write_jsonl

    if not cfg.pattern_store:
        raise ConfigError("pattern store actions need --pattern-store DIR (or [patterns] store in sandwich.toml)")
    reading = args.pattern_stats or args.pattern_verify or args.export_patterns
    with open_store(cfg, readonly=bool(reading)) as store:
        if args.ingest_patterns:
            total = IngestReport()
            files = []
            for path in args.ingest_patterns:
                rep = store.ingest_file(path, source=args.pattern_source)
                files.append({"file": path, **rep.to_dict()})
                total.merge(rep)
            _warn(total.errors[:20], err)
            _dump({"total": total.to_dict(), "files": files, "store": store.stats()}, out)
        elif args.build_synthetic_patterns is not None:
            if args.build_synthetic_patterns < 0:
                raise ConfigError("--build-synthetic-patterns needs a non-negative count")
            rep = store.add(synthetic_corpus(args.build_synthetic_patterns, cfg.seed))
            _dump({"added": rep.to_dict(), "store": store.stats()}, out)
        elif args.pattern_stats:
            _dump(store.stats(), out)
        elif args.pattern_verify:
            report = store.verify()
            _dump(report, out)
            return EXIT_OK if report["ok"] else EXIT_INVARIANT
        else:
            n = write_jsonl(store.records(), args.export_patterns)
            _dump({"exported": n, "file": args.export_patterns}, out)
    return EXIT_OK


def _strategy_action(args: argparse.Namespace, cfg: SimulationConfig, out: TextIO) -> int:
    from .attack_vectors import build_scenario, run_attack_catalog
    from .strategy_discovery import DiscoveryEngine, StrategyLibrary

    if args.scenario_victims < 0 or args.generations < 0 or args.population <= 0:
        raise ConfigError("--scenario-victims and --generations must be >= 0, --population > 0")
    library = StrategyLibrary(args.strategy_dir)
    if args.attack_catalog:
        _dump(run_attack_catalog(build_scenario(cfg.seed, args.scenario_victims)), out)
        return EXIT_OK
    if args.strategies:
        _dump({"path": str(library.path), "strategies": library.summary()}, out)
        return EXIT_OK
    if args.verify_strategies:
        report = library.verify()
        _dump(report, out)
        return EXIT_OK if report["ok"] else EXIT_INVARIANT
    engine = DiscoveryEngine(seed=cfg.seed, generations=args.generations, population=args.population,
                             victims=args.scenario_victims)
    with WorkerPool.for_config(cfg) as wp:
        report = engine.run(executor=wp, library=None if args.no_save else library)
    _dump(report.to_dict(), out)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
