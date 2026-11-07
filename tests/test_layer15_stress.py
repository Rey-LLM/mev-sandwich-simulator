"""Layer 15: stability under 100% CPU load - every core busy while the factory, graphs and monitors run."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from simulator import code_factory as cf
from simulator import gui, i18n
from simulator import tx_graph as tg
from simulator.parallel import WorkerPool, saturation_probe

CPUS = os.cpu_count() or 1


@pytest.fixture(scope="module")
def baselines(tmp_path_factory):
    """Results computed on an idle machine, single-threaded."""
    d = tmp_path_factory.mktemp("baseline")
    report = cf.CodeFactory(d, bots=6, generations=2, heat_size=3, heat_slots=3, seed=21).run()
    digests = [e["py_digest"] for e in cf.read_manifest(d)["bots"]]
    return {
        "factory_pnl": [h["best_pnl_wei"] for h in report.history],
        "factory_digests": digests,
        "eth": {s: tg.eth_graph_for_seed(s).digest() for s in (1, 2, 3)},
        "btc": {s: tg.btc_graph_for_seed(s).digest() for s in (0, 1, 2)},
    }


class Saturation:
    """Every logical core busy (MAX Brutal CPU policy) on a background thread for ``seconds``."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.result = None
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            with WorkerPool(None, policy="full") as wp:
                self.result = saturation_probe(wp, seconds=self.seconds)
        except Exception as exc:  # noqa: BLE001 - surfaced by the test
            self.error = exc

    def __enter__(self) -> "Saturation":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.thread.join(120)


def test_full_policy_pool_uses_every_core():
    with WorkerPool(None, policy="full") as wp:
        assert wp.workers == wp.cpus == CPUS
        r = saturation_probe(wp, seconds=0.3)
    assert r["tasks"] == CPUS and r["cpu_s"] > 0


def test_factory_is_deterministic_under_full_load(tmp_path, baselines):
    with Saturation(2.0) as sat:
        report = cf.CodeFactory(tmp_path, bots=6, generations=2, heat_size=3, heat_slots=3, seed=21).run()
    assert sat.error is None and sat.result["workers"] == CPUS
    assert report.ok
    assert [h["best_pnl_wei"] for h in report.history] == baselines["factory_pnl"]
    assert [e["py_digest"] for e in cf.read_manifest(tmp_path)["bots"]] == baselines["factory_digests"]


def test_graphs_are_identical_under_full_load(baselines):
    with Saturation(1.5) as sat:
        with ThreadPoolExecutor(8) as ex:
            eth = list(ex.map(lambda s: (s, tg.eth_graph_for_seed(s).digest()), [1, 2, 3] * 3))
            btc = list(ex.map(lambda s: (s, tg.btc_graph_for_seed(s).digest()), [0, 1, 2] * 3))
    assert sat.error is None
    assert all(d == baselines["eth"][s] for s, d in eth)
    assert all(d == baselines["btc"][s] for s, d in btc)


def test_parallel_factories_in_threads_under_load(tmp_path, baselines):
    def run(i):
        r = cf.CodeFactory(tmp_path / f"f{i}", bots=6, generations=2, heat_size=3, heat_slots=3, seed=21).run()
        return r.ok, [h["best_pnl_wei"] for h in r.history]

    with Saturation(2.0):
        with ThreadPoolExecutor(6) as ex:
            results = list(ex.map(run, range(6)))
    assert all(ok for ok, _ in results)
    assert all(p == baselines["factory_pnl"] for _, p in results)


def test_cpu_sampler_reports_load_within_ceiling_while_saturated():
    hub = gui.TelemetryHub("ru")
    hub.start_cpu(interval=0.05)
    with Saturation(1.0) as sat:
        pass
    hub.sample_cpu()
    hub.stop()
    assert sat.error is None
    pts = hub.cpu_total.points()
    assert len(pts) >= 3
    assert all(0.0 <= v <= gui.CPU_CEILING for _, v in pts)
    for core in hub.cores:
        assert all(0.0 <= v <= gui.CPU_CEILING for _, v in core.points())
    if os.path.exists(gui.PROC_STAT):
        assert max(v for _, v in pts) > 0.0


def test_concurrent_loads_and_writes_on_one_directory(tmp_path):
    pop = cf.AgentFactory(bots=12, generations=1, seed=2).synthesize(0)
    srcs = [cf.render_python(p, [float(x) for x in g]) for p, g in zip(pop.profiles, pop.genes)]

    def write_then_load(i):
        path = tmp_path / f"bot_{i:04d}.py"
        digest = cf.write_atomic(path, srcs[i])
        return cf.load_bot(path, digest).profile

    with Saturation(1.0):
        with ThreadPoolExecutor(12) as ex:
            profiles = list(ex.map(write_then_load, range(12)))
    assert profiles == pop.profiles


def test_render_html_concurrently_is_byte_identical():
    graphs = [tg.eth_graph_for_seed(2), tg.btc_graph_for_seed(1)]
    for g in graphs:
        g.layout()
    with ThreadPoolExecutor(8) as ex:
        pages = list(ex.map(lambda _: tg.render_html(graphs, "ru"), range(24)))
    assert len(set(pages)) == 1


def test_translations_are_thread_safe():
    keys = sorted(k for k, v in i18n.CATALOG.items() if not i18n.fields_of(v["en"]))

    def check(k):
        return i18n.t(k, "ru") == i18n.CATALOG[k]["ru"] and i18n.t(k, "en") == i18n.CATALOG[k]["en"]

    with ThreadPoolExecutor(16) as ex:
        assert all(ex.map(check, keys * 8))


def test_hub_survives_event_storm():
    hub = gui.TelemetryHub("ru", sampler=gui.CpuSampler(lambda: ""))

    def storm(k):
        for g in range(200):
            hub.on_factory_event({"event": "file_written", "bytes": 100})
            hub.on_factory_event({"event": "generation", "generation": g, "files_per_s": 1.0, "write_s": 0.1,
                                  "bytes": 1024, "loss_before": 0.5, "loss_after": 0.25, "best_pnl_wei": k,
                                  "message": f"{k}:{g}"})
            hub.sample_cpu()

    threads = [threading.Thread(target=storm, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(hub.writes()) == 1600
    assert len(hub.files_per_s) == min(1600, hub.history)
    assert len(hub.snapshot()["log"]) == 200
    gui.plans_for(hub)


@pytest.mark.parametrize("workers", sorted({1, 2, CPUS}))
def test_disk_deploy_same_result_for_any_worker_count(tmp_path, workers):
    serial = cf.CodeFactory(tmp_path / "s", bots=8, generations=1, heat_size=2, heat_slots=2, seed=13).run()
    with WorkerPool(workers, parallel_threshold=1, policy="full") as wp:
        par = cf.CodeFactory(tmp_path / "p", bots=8, generations=1, heat_size=2, heat_slots=2, seed=13).run(
            executor=wp)
    assert par.ok
    assert par.history[0]["total_pnl_wei"] == serial.history[0]["total_pnl_wei"]
    assert par.history[0]["best_bot"] == serial.history[0]["best_bot"]
