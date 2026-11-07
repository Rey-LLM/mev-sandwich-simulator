"""Layer 15: telemetry window and HTML telemetry dashboard (Russian by default).

``python -m simulator --monitor`` opens a dark desktop window (CustomTkinter
when it is installed, plain tkinter otherwise) with three live monitors:

* **CPU load per core** - read from ``/proc/stat`` deltas on a sampling
  thread, every core clamped to the 100% ceiling (on systems without
  ``/proc/stat`` only the total is shown, estimated from the load average);
* **code synthesis speed** - files per second and KiB per second of the
  Layer 15 :class:`~simulator.code_factory.CodeFactory`, per generation;
* **learning convergence** - the generator's training loss before / after
  each generation and the best bot's P&L.

Buttons start the code factory, a *MAX Brutal Simulation* CPU stress (the
worker pool under the ``full`` CPU policy running the
:func:`~simulator.parallel.saturation_probe` kernel on every logical core)
and the transaction graph builder, each on a background thread; the window
redraws from a thread-safe :class:`TelemetryHub` every few hundred
milliseconds, so the UI thread never blocks.

``python -m simulator --monitor-html FILE`` records the same telemetry while
the code factory (and optionally the stress) runs and writes a
self-contained HTML dashboard: no display is needed.

Drawing is split from the toolkit: every chart is first built as a
:class:`CanvasPlan` (lines, rectangles, text in pixel coordinates) by pure
functions, then replayed on a Tk canvas (:class:`TkRenderer`) or turned into
SVG (:func:`plan_to_svg`). The plans are what the tests check.
"""

from __future__ import annotations

import html
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from . import i18n

PROC_STAT = "/proc/stat"
CPU_CEILING = 100.0
DEFAULT_HISTORY = 600

THEME: Dict[str, str] = {
    "bg": "#0f1115", "panel": "#171a21", "fg": "#e8eaf0", "muted": "#9aa3b2", "grid": "#2a2f3a",
    "cpu": "#6ea8fe", "hot": "#f87171", "synth": "#7dd3a8", "kib": "#f5b86b", "loss": "#c084fc",
    "loss2": "#6ea8fe", "pnl": "#facc15", "ceiling": "#f87171",
}


# ---------------------------------------------------------------------------
# CPU sampling
# ---------------------------------------------------------------------------


def parse_proc_stat(text: str) -> Dict[str, Tuple[int, int]]:
    """``{"cpu": (busy, total), "cpu0": ..., ...}`` in jiffies from ``/proc/stat`` text.

    total = user + nice + system + idle + iowait + irq + softirq + steal (guest time is already
    inside user / nice); busy = total - idle - iowait.
    """
    out: Dict[str, Tuple[int, int]] = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts or not parts[0].startswith("cpu"):
            continue
        try:
            vals = [int(x) for x in parts[1:9]]
        except ValueError:
            continue
        vals += [0] * (8 - len(vals))
        total = sum(vals)
        idle = vals[3] + vals[4]
        out[parts[0]] = (total - idle, total)
    return out


def cpu_loads(prev: Dict[str, Tuple[int, int]], cur: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
    """Percent busy per key between two snapshots, clamped to ``[0, CPU_CEILING]``."""
    out: Dict[str, float] = {}
    for key, (busy, total) in cur.items():
        if key not in prev:
            continue
        pb, pt = prev[key]
        dt = total - pt
        if dt <= 0:
            out[key] = 0.0
            continue
        out[key] = max(0.0, min(CPU_CEILING, 100.0 * (busy - pb) / dt))
    return out


def core_keys(loads: Dict[str, float]) -> List[str]:
    """``cpu0, cpu1, ...`` in numeric order."""
    return sorted((k for k in loads if k != "cpu" and k[3:].isdigit()), key=lambda k: int(k[3:]))


class CpuSampler:
    """Per-core load from successive ``/proc/stat`` reads (``reader`` is injectable for tests)."""

    def __init__(self, reader: Optional[Callable[[], str]] = None) -> None:
        self.reader = reader if reader is not None else self._read
        self._prev: Optional[Dict[str, Tuple[int, int]]] = None
        self._lock = threading.Lock()

    @staticmethod
    def _read() -> str:
        try:
            with open(PROC_STAT, encoding="ascii") as fh:
                return fh.read()
        except OSError:
            return ""

    def sample(self) -> Dict[str, Any]:
        text = self.reader()
        with self._lock:
            if not text:
                return self._fallback()
            cur = parse_proc_stat(text)
            prev, self._prev = self._prev, cur
        if prev is None:
            return {"total": 0.0, "cores": [0.0] * len(core_keys({k: 0.0 for k in cur})), "source": "proc"}
        loads = cpu_loads(prev, cur)
        return {"total": loads.get("cpu", 0.0), "cores": [loads[k] for k in core_keys(loads)], "source": "proc"}

    @staticmethod
    def _fallback() -> Dict[str, Any]:
        try:
            one = os.getloadavg()[0]
        except (OSError, AttributeError):
            one = 0.0
        cpus = os.cpu_count() or 1
        return {"total": max(0.0, min(CPU_CEILING, 100.0 * one / cpus)), "cores": [], "source": "loadavg"}


# ---------------------------------------------------------------------------
# Thread-safe series and the hub
# ---------------------------------------------------------------------------


class Series:
    """Bounded, thread-safe ``(t, value)`` series."""

    def __init__(self, maxlen: int = DEFAULT_HISTORY) -> None:
        self._data: Deque[Tuple[float, float]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, t: float, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError(f"non-finite sample {value!r}")
        with self._lock:
            self._data.append((float(t), float(value)))

    def points(self) -> List[Tuple[float, float]]:
        with self._lock:
            return list(self._data)

    def last(self, default: float = 0.0) -> float:
        with self._lock:
            return self._data[-1][1] if self._data else default

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def stats(self) -> Dict[str, float]:
        pts = self.points()
        if not pts:
            return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
        vals = [v for _, v in pts]
        return {"count": len(vals), "min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals)}


@dataclass
class TaskState:
    name: str
    status: str = "idle"  # idle | running | done | failed
    error: str = ""
    result: Any = None
    started: float = 0.0
    finished: float = 0.0


class TelemetryHub:
    """Everything the monitors show, filled by background threads; safe to read from any thread."""

    def __init__(self, lang: Optional[str] = None, sampler: Optional[CpuSampler] = None,
                 history: int = DEFAULT_HISTORY, clock: Callable[[], float] = time.perf_counter) -> None:
        self.lang = i18n.resolve_language(lang)
        self.sampler = sampler if sampler is not None else CpuSampler()
        self.clock = clock
        self.t0 = clock()
        self.history = history
        self.cpu_total = Series(history)
        self.cores: List[Series] = []
        self.files_per_s = Series(history)
        self.kib_per_s = Series(history)
        self.loss_before = Series(history)
        self.loss_after = Series(history)
        self.best_pnl = Series(history)
        self.log: Deque[str] = deque(maxlen=200)
        self.tasks: Dict[str, TaskState] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._writes: List[Tuple[float, int]] = []

    def now(self) -> float:
        return self.clock() - self.t0

    # -- CPU ------------------------------------------------------------------

    def sample_cpu(self) -> Dict[str, Any]:
        s = self.sampler.sample()
        t = self.now()
        self.cpu_total.append(t, s["total"])
        with self._lock:
            while len(self.cores) < len(s["cores"]):
                self.cores.append(Series(self.history))
            cores = list(self.cores)
        for series, value in zip(cores, s["cores"]):
            series.append(t, value)
        return s

    def start_cpu(self, interval: float = 0.25) -> threading.Thread:
        def loop() -> None:
            while not self._stop.wait(interval):
                self.sample_cpu()

        self.sample_cpu()
        th = threading.Thread(target=loop, name="telemetry-cpu", daemon=True)
        th.start()
        self._threads.append(th)
        return th

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for th in list(self._threads):
            th.join(timeout)

    @property
    def stopping(self) -> threading.Event:
        return self._stop

    # -- code factory events ------------------------------------------------------

    def on_factory_event(self, event: Dict[str, Any]) -> None:
        """Listener for :class:`simulator.code_factory.CodeFactory` (called from the factory's thread)."""
        kind = event.get("event")
        if kind == "file_written":
            with self._lock:
                self._writes.append((self.now(), int(event.get("bytes", 0))))
        elif kind == "generation":
            g = float(event.get("generation", 0))
            self.files_per_s.append(g, float(event.get("files_per_s", 0.0)))
            write_s = float(event.get("write_s", 0.0))
            kib = float(event.get("bytes", 0)) / 1024.0
            self.kib_per_s.append(g, kib / write_s if write_s > 0 else 0.0)
            self.loss_before.append(g, float(event.get("loss_before", 0.0)))
            self.loss_after.append(g, float(event.get("loss_after", 0.0)))
            self.best_pnl.append(g, float(event.get("best_pnl_wei", 0)))
        message = event.get("message")
        if message:
            self.add_log(str(message))

    def add_log(self, line: str) -> None:
        with self._lock:
            self.log.append(f"[{self.now():7.2f}] {line}")

    def writes(self) -> List[Tuple[float, int]]:
        with self._lock:
            return list(self._writes)

    # -- background tasks ---------------------------------------------------------

    def run_task(self, name: str, fn: Callable[[], Any], label: Optional[str] = None) -> threading.Thread:
        """Run ``fn`` on a daemon thread and track its state under ``name``."""
        state = TaskState(name, "running", started=self.now())
        with self._lock:
            self.tasks[name] = state
        shown = label or name
        self.add_log(i18n.t("gui.running", self.lang, task=shown))

        def body() -> None:
            try:
                state.result = fn()
                state.status = "done"
                self.add_log(i18n.t("gui.done", self.lang, task=shown))
            except Exception as exc:  # noqa: BLE001 - reported in the window, never swallowed
                state.status, state.error = "failed", f"{type(exc).__name__}: {exc}"
                self.add_log(i18n.t("gui.failed", self.lang, task=shown, error=state.error))
            finally:
                state.finished = self.now()

        th = threading.Thread(target=body, name=f"telemetry-{name}", daemon=True)
        th.start()
        self._threads.append(th)
        return th

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            cores = [s.last() for s in self.cores]
            tasks = {k: {"status": v.status, "error": v.error} for k, v in self.tasks.items()}
            log = list(self.log)
        return {"t": self.now(), "cpu_total": self.cpu_total.last(), "cores": cores,
                "files_per_s": self.files_per_s.last(), "kib_per_s": self.kib_per_s.last(),
                "loss_after": self.loss_after.last(), "best_pnl_wei": self.best_pnl.last(), "tasks": tasks,
                "log": log, "language": self.lang}


def run_factory_task(hub: TelemetryHub, **factory_kwargs: Any) -> Any:
    """Code factory run wired to ``hub`` (blocking; call it through :meth:`TelemetryHub.run_task`)."""
    from .code_factory import CodeFactory  # NumPy-backed; imported when the button is pressed

    factory = CodeFactory(listener=hub.on_factory_event, lang=hub.lang, **factory_kwargs)
    return factory.run(stop=hub.stopping).summary()


def run_stress_task(seconds: float = 2.0, workers: Optional[int] = None) -> Dict[str, Any]:
    """MAX Brutal Simulation CPU stress: every logical core busy for ``seconds`` (full CPU policy)."""
    from .parallel import WorkerPool, saturation_probe

    with WorkerPool(workers, policy="full") as wp:
        return saturation_probe(wp, seconds=seconds)


def run_graph_task(path: str, seed: int = 2, lang: Optional[str] = None) -> Dict[str, Any]:
    from .tx_graph import btc_graph_for_seed, eth_graph_for_seed, render_html

    eth, btc = eth_graph_for_seed(seed, lang=lang), btc_graph_for_seed(seed, lang=lang)
    page = render_html([eth, btc], lang)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)
    return {"path": path, "eth": eth.summary(), "btc": btc.summary()}


# ---------------------------------------------------------------------------
# Chart plans
# ---------------------------------------------------------------------------


@dataclass
class CanvasPlan:
    """Toolkit-neutral drawing: a list of primitive operations in pixel coordinates."""

    width: int
    height: int
    ops: List[Tuple[Any, ...]] = field(default_factory=list)

    def line(self, points: Sequence[Tuple[float, float]], color: str, width: float = 1.5, dash: bool = False) -> None:
        if len(points) >= 2:
            self.ops.append(("line", [(round(x, 2), round(y, 2)) for x, y in points], color, width, dash))

    def rect(self, x: float, y: float, w: float, h: float, fill: str) -> None:
        self.ops.append(("rect", round(x, 2), round(y, 2), round(max(0.0, w), 2), round(max(0.0, h), 2), fill))

    def text(self, x: float, y: float, text: str, color: str, anchor: str = "start", size: int = 11) -> None:
        self.ops.append(("text", round(x, 2), round(y, 2), text, color, anchor, size))

    def kinds(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for op in self.ops:
            out[op[0]] = out.get(op[0], 0) + 1
        return out


def nice_ticks(lo: float, hi: float, count: int = 5) -> List[float]:
    """Round tick values (1-2-5 steps) covering ``[lo, hi]``."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError("tick range must be finite")
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(1, count)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 1e-9:
        if v >= lo - step * 1e-9:
            ticks.append(round(v, 12))
        v += step
    return ticks


Box = Tuple[float, float, float, float]  # x, y, w, h


def scale_points(points: Sequence[Tuple[float, float]], box: Box, x_range: Optional[Tuple[float, float]] = None,
                 y_range: Optional[Tuple[float, float]] = None) -> List[Tuple[float, float]]:
    """Map data points into ``box`` (y grows downwards); values outside the ranges are clamped to the box."""
    if not points:
        return []
    x0, y0, w, h = box
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xl, xh = x_range if x_range is not None else (min(xs), max(xs))
    yl, yh = y_range if y_range is not None else (min(ys), max(ys))
    if xh == xl:
        xh = xl + 1.0
    if yh == yl:
        yh = yl + 1.0
    out = []
    for x, y in points:
        fx = min(1.0, max(0.0, (x - xl) / (xh - xl)))
        fy = min(1.0, max(0.0, (y - yl) / (yh - yl)))
        out.append((x0 + fx * w, y0 + (1.0 - fy) * h))
    return out


def _fmt(v: float, lang: str) -> str:
    """Short axis label: integers as is, small values with three significant digits, huge ones in e-notation."""
    if abs(v) >= 1e6:
        text = f"{v:.2e}"
    elif v == int(v):
        text = str(int(v))
    else:
        text = f"{v:.3g}"
    return text.replace(".", ",") if lang == "ru" else text


def plan_line_chart(title: str, series: Sequence[Tuple[str, Sequence[Tuple[float, float]], str]], width: int = 520,
                    height: int = 220, y_range: Optional[Tuple[float, float]] = None,
                    ceiling: Optional[float] = None, lang: Optional[str] = None, x_label: str = "") -> CanvasPlan:
    """Line chart plan: frame, grid with 1-2-5 ticks, one polyline per series, legend and optional ceiling."""
    lg = i18n.resolve_language(lang)
    plan = CanvasPlan(width, height)
    plan.rect(0, 0, width, height, THEME["panel"])
    plan.text(12, 18, title, THEME["fg"], size=13)
    box: Box = (48.0, 34.0, width - 64.0, height - 62.0)
    pts_all = [p for _, pts, _ in series for p in pts]
    if not pts_all:
        plan.text(width / 2, height / 2, i18n.t("gui.no_data", lg), THEME["muted"], anchor="middle")
        return plan
    xs = [p[0] for p in pts_all]
    ys = [p[1] for p in pts_all]
    yr = y_range if y_range is not None else (min(0.0, min(ys)), max(ys) if max(ys) > min(0.0, min(ys)) else 1.0)
    xr = (min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1.0)
    for tick in nice_ticks(yr[0], yr[1], 4):
        (_, py), = scale_points([(xr[0], tick)], box, xr, yr)
        plan.line([(box[0], py), (box[0] + box[2], py)], THEME["grid"], 1.0)
        plan.text(box[0] - 6, py + 4, _fmt(tick, lg), THEME["muted"], anchor="end", size=10)
    if ceiling is not None:
        (_, py), = scale_points([(xr[0], ceiling)], box, xr, yr)
        plan.line([(box[0], py), (box[0] + box[2], py)], THEME["ceiling"], 1.0, dash=True)
        plan.text(box[0] + box[2], py - 4, i18n.t("gui.ceiling", lg, value=_fmt(ceiling, lg)), THEME["ceiling"],
                  anchor="end", size=10)
    legend_x = box[0]
    for label, pts, color in series:
        plan.line(scale_points(pts, box, xr, yr), color, 2.0)
        plan.rect(legend_x, height - 18, 10, 10, color)
        plan.text(legend_x + 14, height - 9, label, THEME["fg"], size=10)
        legend_x += 24 + 7 * len(label)
    if x_label:
        plan.text(box[0] + box[2], height - 9, x_label, THEME["muted"], anchor="end", size=10)
    return plan


def plan_core_bars(loads: Sequence[float], width: int = 520, height: int = 220, lang: Optional[str] = None,
                   ceiling: float = CPU_CEILING, total: Optional[float] = None) -> CanvasPlan:
    """One bar per core (0..ceiling %), red above 90%, with the ceiling line and the total in the title."""
    lg = i18n.resolve_language(lang)
    plan = CanvasPlan(width, height)
    plan.rect(0, 0, width, height, THEME["panel"])
    title = i18n.t("gui.cpu", lg)
    if total is not None:
        title += f" - {i18n.t('gui.cpu_total', lg)}: {_fmt(total, lg)}%"
    plan.text(12, 18, title, THEME["fg"], size=13)
    box: Box = (48.0, 34.0, width - 64.0, height - 62.0)
    for tick in (0.0, 25.0, 50.0, 75.0, 100.0):
        y = box[1] + (1.0 - tick / ceiling) * box[3]
        plan.line([(box[0], y), (box[0] + box[2], y)], THEME["grid"], 1.0)
        plan.text(box[0] - 6, y + 4, f"{tick:.0f}%", THEME["muted"], anchor="end", size=10)
    plan.line([(box[0], box[1]), (box[0] + box[2], box[1])], THEME["ceiling"], 1.0, dash=True)
    n = len(loads)
    if n == 0:
        plan.text(width / 2, height / 2, i18n.t("gui.no_data", lg), THEME["muted"], anchor="middle")
        return plan
    slot = box[2] / n
    bar = max(2.0, slot * 0.7)
    for i, v in enumerate(loads):
        v = max(0.0, min(ceiling, float(v)))
        h = box[3] * v / ceiling
        x = box[0] + i * slot + (slot - bar) / 2
        plan.rect(x, box[1] + box[3] - h, bar, h, THEME["hot"] if v >= 90.0 else THEME["cpu"])
        if n <= 32:
            plan.text(x + bar / 2, height - 9, str(i), THEME["muted"], anchor="middle", size=9)
    return plan


def plans_for(hub: TelemetryHub, width: int = 520, height: int = 220) -> Dict[str, CanvasPlan]:
    """The three monitor plans for the hub's current data."""
    lg = hub.lang
    snap = hub.snapshot()
    cpu = plan_core_bars(snap["cores"] or [snap["cpu_total"]], width, height, lg, total=snap["cpu_total"])
    synth = plan_line_chart(i18n.t("gui.synthesis", lg),
                            [(i18n.t("gui.files_per_s", lg), hub.files_per_s.points(), THEME["synth"]),
                             (i18n.t("gui.kib_per_s", lg), hub.kib_per_s.points(), THEME["kib"])],
                            width, height, lang=lg, x_label=i18n.t("gui.generation", lg))
    conv = plan_line_chart(i18n.t("gui.convergence", lg),
                           [(i18n.t("gui.loss_before", lg), hub.loss_before.points(), THEME["loss"]),
                            (i18n.t("gui.loss_after", lg), hub.loss_after.points(), THEME["loss2"])],
                           width, height, lang=lg, x_label=i18n.t("gui.generation", lg))
    load = plan_line_chart(i18n.t("gui.cpu_total", lg), [("CPU %", hub.cpu_total.points(), THEME["cpu"])],
                           width, height, y_range=(0.0, CPU_CEILING), ceiling=CPU_CEILING, lang=lg,
                           x_label=i18n.t("gui.time_s", lg))
    return {"cpu": cpu, "cpu_history": load, "synthesis": synth, "convergence": conv}


def plan_to_svg(plan: CanvasPlan) -> str:
    """SVG text of a plan (all text escaped)."""
    esc = html.escape
    out = [f'<svg viewBox="0 0 {plan.width} {plan.height}" width="100%" role="img" xmlns="http://www.w3.org/2000/svg">']
    for op in plan.ops:
        if op[0] == "line":
            pts = " ".join(f"{x},{y}" for x, y in op[1])
            dash = ' stroke-dasharray="5 4"' if op[4] else ""
            out.append(f'<polyline fill="none" stroke="{esc(op[2])}" stroke-width="{op[3]}"{dash} points="{pts}"/>')
        elif op[0] == "rect":
            out.append(f'<rect x="{op[1]}" y="{op[2]}" width="{op[3]}" height="{op[4]}" fill="{esc(op[5])}" rx="2"/>')
        elif op[0] == "text":
            out.append(f'<text x="{op[1]}" y="{op[2]}" fill="{esc(op[4])}" text-anchor="{esc(op[5])}" '
                       f'font-size="{op[6]}" font-family="system-ui,sans-serif">{esc(op[3])}</text>')
    out.append("</svg>")
    return "".join(out)


def build_dashboard_html(hub: TelemetryHub, extra: Optional[Dict[str, Any]] = None) -> str:
    """Self-contained HTML telemetry dashboard of the hub's current data."""
    esc = html.escape
    lg = hub.lang
    plans = plans_for(hub, 560, 240)
    cards = "".join(f'<section class="card">{plan_to_svg(p)}</section>' for p in plans.values())
    log = "".join(f"<li>{esc(line)}</li>" for line in hub.snapshot()["log"])
    facts = ""
    if extra:
        facts = "<dl>" + "".join(f"<dt>{esc(str(k))}</dt><dd>{esc(str(v))}</dd>" for k, v in extra.items()) + "</dl>"
    title = i18n.t("gui.dashboard", lg)
    return (f'<!doctype html><html lang="{lg}"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>'
            f':root{{--bg:{THEME["bg"]};--fg:{THEME["fg"]};--panel:{THEME["panel"]};--muted:{THEME["muted"]};'
            f'--line:{THEME["grid"]}}}'
            "body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}"
            "main{max-width:1180px;margin:0 auto;padding:16px}h1{font-size:20px;margin:0 0 12px}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:12px}"
            ".card{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}"
            "dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 12px;color:var(--muted)}dd{margin:0;"
            "color:var(--fg)}ul{color:var(--muted);font:12px ui-monospace,monospace;padding-left:18px}"
            f"</style></head><body><main><h1>{esc(title)}</h1>{facts}<div class=\"grid\">{cards}</div>"
            f"<h2>{esc(i18n.t('gui.log', lg))}</h2><ul>{log}</ul></main></body></html>")


def record_dashboard(path: str, lang: Optional[str] = None, stress_seconds: float = 0.0, interval: float = 0.1,
                     **factory_kwargs: Any) -> Dict[str, Any]:
    """Sample CPU while the code factory (and optionally a MAX stress) runs, then write the dashboard."""
    hub = TelemetryHub(lang)
    hub.start_cpu(interval)
    try:
        threads = [hub.run_task("factory", lambda: run_factory_task(hub, **factory_kwargs),
                                i18n.t("gui.start_factory", hub.lang))]
        if stress_seconds > 0:
            threads.append(hub.run_task("stress", lambda: run_stress_task(stress_seconds),
                                        i18n.t("gui.start_max", hub.lang)))
        for th in threads:
            th.join()
        hub.sample_cpu()
    finally:
        hub.stop()
    summary = {k: {"status": v.status, "error": v.error} for k, v in hub.tasks.items()}
    factory = hub.tasks["factory"].result or {}
    extra = {i18n.t("gui.synthesis", hub.lang): f"{factory.get('files_per_s', 0)} {i18n.t('gui.files_per_s', hub.lang)}",
             i18n.t("gui.cpu_total", hub.lang): f"{hub.cpu_total.stats()['max']:.1f}% max",
             i18n.t("gui.samples", hub.lang, count=len(hub.cpu_total)): ""}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(build_dashboard_html(hub, extra))
    return {"path": path, "tasks": summary, "cpu": hub.cpu_total.stats(), "factory": factory,
            "language": hub.lang}


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


def load_toolkit() -> Tuple[Any, Optional[Any]]:
    """``(tkinter, customtkinter or None)``; raises RuntimeError (localized) when tkinter is missing."""
    try:
        import tkinter as tk
    except ImportError as exc:
        raise RuntimeError(i18n.t("gui.no_tk")) from exc
    try:
        import customtkinter as ctk  # optional premium dark widgets
    except ImportError:
        ctk = None
    return tk, ctk


class TkRenderer:
    """Replays a :class:`CanvasPlan` on a ``tkinter.Canvas``."""

    def __init__(self, canvas: Any) -> None:
        self.canvas = canvas

    def draw(self, plan: CanvasPlan) -> int:
        c = self.canvas
        c.delete("all")
        for op in plan.ops:
            if op[0] == "line":
                flat = [v for pt in op[1] for v in pt]
                c.create_line(*flat, fill=op[2], width=op[3], dash=(5, 4) if op[4] else None)
            elif op[0] == "rect":
                c.create_rectangle(op[1], op[2], op[1] + op[3], op[2] + op[4], fill=op[5], outline="")
            elif op[0] == "text":
                anchor = {"start": "w", "middle": "center", "end": "e"}[op[5]]
                c.create_text(op[1], op[2], text=op[3], fill=op[4], anchor=anchor, font=("TkDefaultFont", op[6]))
        return len(plan.ops)


class TelemetryApp:
    """The dark telemetry window (needs a display)."""

    def __init__(self, lang: Optional[str] = None, refresh_ms: int = 300, graph_path: str = "tx_graph.html",
                 factory_kwargs: Optional[Dict[str, Any]] = None) -> None:
        tk, ctk = load_toolkit()
        self.tk, self.ctk = tk, ctk
        self.hub = TelemetryHub(lang)
        self.refresh_ms = refresh_ms
        self.graph_path = graph_path
        self.factory_kwargs = dict(factory_kwargs or {})
        lg = self.hub.lang
        if ctk is not None:
            ctk.set_appearance_mode("dark")
            self.root = ctk.CTk()
        else:
            self.root = tk.Tk()
            self.root.configure(bg=THEME["bg"])
        self.root.title(i18n.t("gui.title", lg))
        bar = tk.Frame(self.root, bg=THEME["bg"])
        bar.pack(fill="x", padx=10, pady=8)
        buttons = (("gui.start_factory", self.start_factory), ("gui.start_max", self.start_stress),
                   ("gui.build_graph", self.build_graph), ("gui.stop", self.close))
        for key, cmd in buttons:
            if ctk is not None:
                ctk.CTkButton(bar, text=i18n.t(key, lg), command=cmd).pack(side="left", padx=4)
            else:
                tk.Button(bar, text=i18n.t(key, lg), command=cmd, bg=THEME["panel"], fg=THEME["fg"],
                          activebackground=THEME["grid"], relief="flat").pack(side="left", padx=4)
        self.status = tk.Label(self.root, text=i18n.t("gui.idle", lg), bg=THEME["bg"], fg=THEME["muted"], anchor="w")
        self.status.pack(fill="x", padx=10)
        grid = tk.Frame(self.root, bg=THEME["bg"])
        grid.pack(fill="both", expand=True, padx=10, pady=8)
        self.canvases: Dict[str, Any] = {}
        for i, name in enumerate(("cpu", "cpu_history", "synthesis", "convergence")):
            c = tk.Canvas(grid, width=520, height=220, bg=THEME["panel"], highlightthickness=0)
            c.grid(row=i // 2, column=i % 2, padx=4, pady=4, sticky="nsew")
            self.canvases[name] = c
        self.renderers = {k: TkRenderer(c) for k, c in self.canvases.items()}
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def start_factory(self) -> None:
        self.hub.run_task("factory", lambda: run_factory_task(self.hub, **self.factory_kwargs),
                          i18n.t("gui.start_factory", self.hub.lang))

    def start_stress(self, seconds: float = 5.0) -> None:
        self.hub.run_task("stress", lambda: run_stress_task(seconds), i18n.t("gui.start_max", self.hub.lang))

    def build_graph(self) -> None:
        self.hub.run_task("graph", lambda: run_graph_task(self.graph_path, lang=self.hub.lang),
                          i18n.t("gui.build_graph", self.hub.lang))

    def refresh(self) -> None:
        for name, plan in plans_for(self.hub).items():
            self.renderers[name].draw(plan)
        log = self.hub.snapshot()["log"]
        self.status.configure(text=log[-1] if log else i18n.t("gui.idle", self.hub.lang))
        self.root.after(self.refresh_ms, self.refresh)

    def run(self) -> None:
        self.hub.start_cpu()
        self.refresh()
        self.root.mainloop()

    def close(self) -> None:
        self.hub.stop(timeout=0.5)
        self.root.destroy()


def launch(lang: Optional[str] = None, **kwargs: Any) -> None:
    TelemetryApp(lang, **kwargs).run()


__all__ = [
    "CPU_CEILING",
    "PROC_STAT",
    "THEME",
    "CanvasPlan",
    "CpuSampler",
    "Series",
    "TaskState",
    "TelemetryApp",
    "TelemetryHub",
    "TkRenderer",
    "build_dashboard_html",
    "core_keys",
    "cpu_loads",
    "launch",
    "load_toolkit",
    "nice_ticks",
    "parse_proc_stat",
    "plan_core_bars",
    "plan_line_chart",
    "plan_to_svg",
    "plans_for",
    "record_dashboard",
    "run_factory_task",
    "run_graph_task",
    "run_stress_task",
    "scale_points",
]
