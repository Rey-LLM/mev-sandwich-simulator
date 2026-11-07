"""Layer 15: telemetry monitors - CPU sampling math, thread-safe series, chart plans, dashboard, window plumbing."""

from __future__ import annotations

import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import pytest

from simulator import gui

STAT_A = """cpu  100 0 100 800 0 0 0 0 0 0
cpu0 50 0 50 400 0 0 0 0 0 0
cpu1 50 0 50 400 0 0 0 0 0 0
intr 12345
"""
STAT_B = """cpu  300 0 200 900 0 0 0 0 0 0
cpu0 200 0 100 400 0 0 0 0 0 0
cpu1 100 0 100 500 0 0 0 0 0 0
"""


def test_parse_proc_stat():
    s = gui.parse_proc_stat(STAT_A)
    assert s == {"cpu": (200, 1000), "cpu0": (100, 500), "cpu1": (100, 500)}


def test_parse_proc_stat_counts_iowait_as_idle_and_ignores_guest():
    s = gui.parse_proc_stat("cpu0 10 0 10 70 10 0 0 0 99 99\n")
    assert s["cpu0"] == (20, 100)


def test_cpu_loads_math():
    loads = gui.cpu_loads(gui.parse_proc_stat(STAT_A), gui.parse_proc_stat(STAT_B))
    assert loads["cpu"] == pytest.approx(100 * 300 / 400)
    assert loads["cpu0"] == pytest.approx(100.0)
    assert loads["cpu1"] == pytest.approx(50.0)


@pytest.mark.parametrize("prev,cur,expected", [
    ({"cpu": (100, 100)}, {"cpu": (100, 100)}, 0.0),  # no time passed
    ({"cpu": (100, 200)}, {"cpu": (50, 300)}, 0.0),  # counter went backwards: clamp
    ({"cpu": (0, 0)}, {"cpu": (500, 400)}, 100.0),  # inconsistent counters: ceiling
])
def test_cpu_loads_clamp(prev, cur, expected):
    assert gui.cpu_loads(prev, cur)["cpu"] == expected


def test_core_keys_numeric_order():
    loads = {f"cpu{i}": 0.0 for i in (10, 2, 0, 1)}
    loads["cpu"] = 0.0
    assert gui.core_keys(loads) == ["cpu0", "cpu1", "cpu2", "cpu10"]


def test_sampler_with_injected_reader():
    texts = iter([STAT_A, STAT_B])
    s = gui.CpuSampler(lambda: next(texts))
    first = s.sample()
    assert first["total"] == 0.0 and first["cores"] == [0.0, 0.0]
    second = s.sample()
    assert second["cores"] == pytest.approx([100.0, 50.0]) and second["source"] == "proc"


def test_sampler_fallback_without_proc():
    s = gui.CpuSampler(lambda: "").sample()
    assert s["source"] == "loadavg" and 0.0 <= s["total"] <= gui.CPU_CEILING and s["cores"] == []


def test_real_sampler_stays_within_ceiling():
    s = gui.CpuSampler()
    for _ in range(3):
        r = s.sample()
        assert 0.0 <= r["total"] <= 100.0 and all(0.0 <= c <= 100.0 for c in r["cores"])


# -- series and hub -----------------------------------------------------------------------------------------------


def test_series_is_bounded_and_thread_safe():
    series = gui.Series(maxlen=500)

    def push(k):
        for i in range(1_000):
            series.append(k * 1_000 + i, float(i))

    threads = [threading.Thread(target=push, args=(k,)) for k in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(series) == 500
    st = series.stats()
    assert st["count"] == 500 and 0.0 <= st["min"] <= st["max"] <= 999.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_series_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        gui.Series().append(0.0, bad)


def test_hub_consumes_factory_events():
    hub = gui.TelemetryHub("ru", sampler=gui.CpuSampler(lambda: ""))
    hub.on_factory_event({"event": "file_written", "bytes": 2048})
    hub.on_factory_event({"event": "generation", "generation": 0, "files_per_s": 500.0, "write_s": 0.5,
                          "bytes": 10_240, "loss_before": 0.02, "loss_after": 0.001, "best_pnl_wei": 7,
                          "message": "поколение 0"})
    assert hub.files_per_s.last() == 500.0
    assert hub.kib_per_s.last() == pytest.approx(20.0)
    assert hub.loss_after.last() == 0.001 and hub.best_pnl.last() == 7.0
    assert hub.writes()[0][1] == 2048
    assert hub.snapshot()["log"][-1].endswith("поколение 0")


def test_hub_tasks_report_success_and_failure():
    hub = gui.TelemetryHub("ru", sampler=gui.CpuSampler(lambda: ""))
    ok = hub.run_task("ok", lambda: 42, "задача")
    bad = hub.run_task("bad", lambda: 1 // 0, "сбой")
    ok.join()
    bad.join()
    assert hub.tasks["ok"].status == "done" and hub.tasks["ok"].result == 42
    assert hub.tasks["bad"].status == "failed" and "ZeroDivisionError" in hub.tasks["bad"].error
    log = "\n".join(hub.snapshot()["log"])
    assert "Готово: задача" in log and "Ошибка: сбой" in log


def test_hub_cpu_thread_samples_and_stops():
    texts = iter([STAT_A] + [STAT_B] * 10_000)
    hub = gui.TelemetryHub("en", sampler=gui.CpuSampler(lambda: next(texts)))
    hub.start_cpu(interval=0.01)
    deadline = threading.Event()
    deadline.wait(0.2)
    hub.stop()
    assert len(hub.cpu_total) >= 2 and len(hub.cores) == 2
    n = len(hub.cpu_total)
    deadline.wait(0.05)
    assert len(hub.cpu_total) == n  # the sampler thread really stopped


# -- plans ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lo,hi", [(0, 1), (0, 100), (-3.5, 7.2), (0.0001, 0.0009), (1e6, 5e9), (5, 5), (10, 0)])
def test_nice_ticks_cover_range_with_round_steps(lo, hi):
    ticks = gui.nice_ticks(lo, hi)
    assert ticks == sorted(ticks) and len(ticks) >= 2
    a, b = min(lo, hi), max(lo, hi) if hi != lo else lo + 1
    assert all(a - 1e-9 * abs(b - a) <= t <= b + 1e-9 * abs(b - a) for t in ticks)
    steps = {round(ticks[i + 1] - ticks[i], 12) for i in range(len(ticks) - 1)}
    assert len(steps) == 1
    step = steps.pop()
    mantissa = step / 10 ** __import__("math").floor(__import__("math").log10(step))
    assert round(mantissa, 6) in (1.0, 2.0, 5.0, 10.0)


def test_nice_ticks_rejects_non_finite():
    with pytest.raises(ValueError):
        gui.nice_ticks(0, float("inf"))


@pytest.mark.parametrize("pts", [[(0, 0), (10, 100)], [(1, -5), (2, 500), (3, 50)], [(0, 1)], [(5, 5), (5, 5)]])
def test_scale_points_stay_in_box(pts):
    box = (10.0, 20.0, 200.0, 100.0)
    out = gui.scale_points(pts, box, y_range=(0, 100))
    assert len(out) == len(pts)
    assert all(10.0 <= x <= 210.0 and 20.0 <= y <= 120.0 for x, y in out)


def test_scale_points_orientation():
    out = gui.scale_points([(0, 0), (1, 100)], (0, 0, 100, 100), (0, 1), (0, 100))
    assert out == [(0.0, 100.0), (100.0, 0.0)]  # y grows downwards


def test_line_chart_plan_with_ceiling():
    plan = gui.plan_line_chart("CPU", [("a", [(0, 10), (1, 90), (2, 100)], "#fff")], 400, 200, (0, 100),
                               ceiling=100, lang="ru")
    dashed = [op for op in plan.ops if op[0] == "line" and op[4]]
    assert len(dashed) == 1
    y_ceiling = dashed[0][1][0][1]
    series_line = [op for op in plan.ops if op[0] == "line" and op[2] == "#fff"][0]
    assert series_line[1][-1][1] == pytest.approx(y_ceiling)
    assert any(op[0] == "text" and "потолок" in op[3] for op in plan.ops)


def test_line_chart_without_data():
    plan = gui.plan_line_chart("x", [("a", [], "#fff")], lang="ru")
    assert any(op[0] == "text" and op[3] == "данных пока нет" for op in plan.ops)
    assert plan.kinds().get("line", 0) == 0


@pytest.mark.parametrize("loads", [[0.0], [10.0, 95.0], [100.0] * 4, [150.0, -20.0], [50.0] * 64])
def test_core_bars(loads):
    plan = gui.plan_core_bars(loads, 520, 220, "ru")
    bars = [op for op in plan.ops if op[0] == "rect"][1:]  # first rect is the background
    assert len(bars) == len(loads)
    full = 220 - 62.0
    for value, bar in zip(loads, bars):
        clamped = max(0.0, min(100.0, value))
        assert bar[4] == pytest.approx(full * clamped / 100.0, abs=0.01)
        assert bar[5] == (gui.THEME["hot"] if clamped >= 90.0 else gui.THEME["cpu"])


def test_plan_to_svg_is_valid_xml_and_escaped():
    plan = gui.CanvasPlan(100, 50)
    plan.rect(0, 0, 100, 50, "#000")
    plan.line([(0, 0), (10, 10)], "#fff", dash=True)
    plan.text(5, 5, "<script>&'\"", "#fff")
    svg = gui.plan_to_svg(plan)
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "<script>" not in svg and "stroke-dasharray" in svg


def test_plans_for_hub_are_localized():
    hub = gui.TelemetryHub("ru", sampler=gui.CpuSampler(lambda: ""))
    hub.sample_cpu()
    plans = gui.plans_for(hub)
    assert set(plans) == {"cpu", "cpu_history", "synthesis", "convergence"}
    titles = [op[3] for p in plans.values() for op in p.ops if op[0] == "text"]
    assert any("Загрузка CPU" in t for t in titles) and any("Сходимость" in t for t in titles)


def test_dashboard_html():
    hub = gui.TelemetryHub("ru", sampler=gui.CpuSampler(lambda: ""))
    hub.sample_cpu()
    hub.add_log("<b>журнал</b>")
    page = gui.build_dashboard_html(hub, {"ключ": "<значение>"})
    assert page.startswith('<!doctype html><html lang="ru">')
    assert "Панель телеметрии" in page and "<b>журнал</b>" not in page and "&lt;значение&gt;" in page
    assert page.count("<svg") == 4


# -- toolkit plumbing -----------------------------------------------------------------------------------------------


class FakeCanvas:
    def __init__(self):
        self.calls = []

    def delete(self, what):
        self.calls.append(("delete", what))

    def create_line(self, *coords, **kw):
        self.calls.append(("line", coords, kw))

    def create_rectangle(self, *coords, **kw):
        self.calls.append(("rect", coords, kw))

    def create_text(self, *coords, **kw):
        self.calls.append(("text", coords, kw))


def test_tk_renderer_replays_plan():
    plan = gui.plan_core_bars([10.0, 99.0], lang="en", total=55.0)
    canvas = FakeCanvas()
    n = gui.TkRenderer(canvas).draw(plan)
    assert n == len(plan.ops) and canvas.calls[0] == ("delete", "all")
    assert len(canvas.calls) == n + 1
    assert {c[0] for c in canvas.calls[1:]} == {"line", "rect", "text"}


def test_load_toolkit_without_tkinter(monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **kw):
        if name == "tkinter":
            raise ImportError("missing")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake)
    monkeypatch.delenv("MEV_SIM_LANG", raising=False)
    with pytest.raises(RuntimeError, match="tkinter недоступен"):
        gui.load_toolkit()


def test_record_dashboard_end_to_end(tmp_path):
    out = gui.record_dashboard(str(tmp_path / "dash.html"), lang="ru", stress_seconds=0.3, interval=0.05,
                               directory=tmp_path / "bots", bots=6, generations=2, heat_size=3, heat_slots=2)
    assert out["tasks"]["factory"]["status"] == "done" and out["tasks"]["stress"]["status"] == "done"
    assert 0.0 <= out["cpu"]["max"] <= 100.0 and out["cpu"]["count"] >= 2
    assert out["factory"]["invariants"]["ok"]
    page = (tmp_path / "dash.html").read_text(encoding="utf-8")
    assert "Скорость синтеза кода" in page and "Сходимость обучения" in page


def test_graph_task_writes_page(tmp_path):
    r = gui.run_graph_task(str(tmp_path / "g.html"), seed=3, lang="en")
    assert r["eth"]["acyclic"] and r["btc"]["acyclic"]
    assert (tmp_path / "g.html").read_text(encoding="utf-8").startswith('<!doctype html><html lang="en">')


def test_concurrent_plan_building_is_deterministic():
    hub = gui.TelemetryHub("ru", sampler=gui.CpuSampler(lambda: ""))
    for g in range(5):
        hub.on_factory_event({"event": "generation", "generation": g, "files_per_s": 100.0 + g, "write_s": 0.1,
                              "bytes": 4096, "loss_before": 0.1 / (g + 1), "loss_after": 0.01 / (g + 1),
                              "best_pnl_wei": g})
    with ThreadPoolExecutor(8) as ex:
        svgs = list(ex.map(lambda _: gui.plan_to_svg(gui.plans_for(hub)["convergence"]), range(32)))
    assert len(set(svgs)) == 1
