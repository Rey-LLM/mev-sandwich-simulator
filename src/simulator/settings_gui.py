"""Layer 10: browser settings editor for ``sandwich.toml``.

``python -m simulator --gui`` serves a form on ``http://127.0.0.1:PORT``. The
form is generated from :func:`simulator.config_sync.form_schema`, the same
schema that generates the CLI flags and the TOML layout, so every setting the
CLI accepts (multiprocessing caps per run matrix, trace output, audit
configuration, ...) appears here with the same name, bounds and choices.

Saving goes through :meth:`SettingsStore.update`, so the GUI gets the same
guarantees as the CLI: validation before anything touches the disk, the
writer lock, a format-preserving edit that keeps comments, a ``.bak`` copy
and an atomic replace. The page carries the digest of the file it was
rendered from; if the file changed in the meantime (another tab, the CLI, an
editor) the save is refused with 409 instead of overwriting that change.

Endpoints::

    GET  /               the form
    POST /               form submission (application/x-www-form-urlencoded)
    GET  /api/schema     form_schema() as JSON
    GET  /api/settings   {"path", "exists", "digest", "values": {"table.key": value}, "warnings"}
    POST /api/settings   {"digest": ..., "values": {"table.key": value}, "force": false} -> new digest

The server binds to 127.0.0.1 only. Every POST must carry the per-server
token embedded in the page (form field ``_token`` or header
``X-Settings-Token``), and requests whose ``Host`` header is not this
server's loopback address are refused, which blocks cross-site form posts
and DNS-rebinding pages from changing settings.
"""

from __future__ import annotations

import html
import json
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Mapping, Optional, TextIO, Tuple
from urllib.parse import parse_qs, urlsplit

from .config_sync import (
    BY_ATTR,
    COMMON_TABLES,
    MODE_NAMES,
    SETTINGS,
    Location,
    Setting,
    SettingsConflictError,
    SettingsDocument,
    SettingsError,
    SettingsFileError,
    SettingsIssue,
    SettingsLockError,
    SettingsStore,
    default_locations,
    form_schema,
    lookup,
)
from . import i18n

MAX_BODY = 256 * 1024
TOKEN_HEADER = "X-Settings-Token"


def _dotted(location: Location) -> str:
    return f"{location[0]}.{location[1]}"


def _split(dotted: str) -> Location:
    table, _, key = dotted.rpartition(".")
    if not table or not key:
        raise SettingsError([SettingsIssue(dotted, "expected table.key or modes.<mode>.key")])
    return table, key


def current_values(doc: SettingsDocument) -> Dict[Location, Any]:
    """Every location the form shows: the file's value, else the built-in default."""
    values = {loc: v for loc, v in default_locations().items() if loc[0] != "meta"}
    values.update({loc: v for loc, v in doc.values.items() if loc[0] != "meta"})
    return values


def parse_form(fields: Mapping[str, List[str]], shown: Mapping[Location, Any]) -> Dict[Location, Any]:
    """Changed values from a submitted form (all problems are reported together)."""
    patch: Dict[Location, Any] = {}
    issues: List[SettingsIssue] = []
    for name, raw_values in fields.items():
        if name.startswith("_"):
            continue
        try:
            location = _split(name)
            setting = lookup(*location)
            value = setting.parse_text(raw_values[-1], name)  # checkboxes post "false" then "true"
        except SettingsError as exc:
            issues.extend(exc.issues)
            continue
        if shown.get(location) != value:
            patch[location] = value
    if issues:
        raise SettingsError(issues)
    return patch


def parse_json_values(values: Mapping[str, Any]) -> Dict[Location, Any]:
    patch: Dict[Location, Any] = {}
    issues: List[SettingsIssue] = []
    for name, value in values.items():
        try:
            location = _split(name)
            setting = lookup(*location)
            patch[location] = setting.coerce(value, name)
        except SettingsError as exc:
            issues.extend(exc.issues)
    if issues:
        raise SettingsError(issues)
    return patch


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """
:root{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#6b6b66;--line:#d9d9d4;--accent:#2f5d8a;--err:#a32020;--ok:#26663a;
--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--muted:#a3a39e;--line:#3a3a37;
--accent:#8ab4de;--err:#f08a8a;--ok:#8fd19e;--card:#1f1f1d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}
p.sub{color:var(--muted);margin:0 0 20px}section{background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:16px;margin:0 0 16px}h2{font-size:16px;margin:0 0 12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
label{display:block;font-weight:600;font-size:13px}small{display:block;color:var(--muted);font-size:12px}
input,select{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);
color:var(--fg);font:inherit}input[type=checkbox]{width:auto}
.matrix{overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:560px}
th,td{border-bottom:1px solid var(--line);padding:6px;text-align:left;vertical-align:top}
th{font-size:13px}td input,td select{min-width:110px}.msg{padding:10px 12px;border-radius:6px;margin:0 0 16px}
.msg.err{border:1px solid var(--err);color:var(--err)}.msg.ok{border:1px solid var(--ok);color:var(--ok)}
button{background:var(--accent);color:var(--bg);border:0;border-radius:6px;padding:8px 18px;font:inherit;
font-weight:600;cursor:pointer}code{font-size:12px}
"""


def _input(setting: Setting, name: str, value: Any) -> str:
    esc = html.escape
    if setting.kind is bool:
        checked = " checked" if value else ""
        return (f'<input type="hidden" name="{esc(name)}" value="false">'
                f'<input type="checkbox" id="{esc(name)}" name="{esc(name)}" value="true"{checked}>')
    if setting.choices:
        opts = "".join(f'<option value="{esc(c)}"{" selected" if c == value else ""}>{esc(c)}</option>'
                       for c in setting.choices)
        return f'<select id="{esc(name)}" name="{esc(name)}">{opts}</select>'
    if setting.kind is int:
        bounds = "".join(f' {attr}="{bound}"' for attr, bound in (("min", setting.minimum), ("max", setting.maximum))
                         if bound is not None)
        return f'<input type="number" id="{esc(name)}" name="{esc(name)}" value="{value}" step="1"{bounds}>'
    return f'<input type="text" id="{esc(name)}" name="{esc(name)}" value="{esc(str(value))}">'


def page_language(shown: Mapping[Location, Any], lang: Optional[str] = None) -> str:
    """Language of the page: ``lang`` > ``$MEV_SIM_LANG`` > the file's ``[ui] language`` > Russian."""
    return i18n.resolve_language(lang, str(shown.get(("ui", "language")) or ""))


def render_page(store: SettingsStore, doc: SettingsDocument, token: str, values: Optional[Mapping[Location, Any]] = None,
                errors: Optional[List[str]] = None, saved: bool = False, lang: Optional[str] = None) -> str:
    esc = html.escape
    shown = dict(current_values(doc))
    if values:
        shown.update(values)
    lg = page_language(shown, lang)

    def tr(key: str, **params: Any) -> str:
        return i18n.t(key, lg, **params)

    def helptext(s: Setting) -> str:
        return i18n.setting_help(s.attr, s.help, lg)

    status = tr("editor.saved_as", digest=doc.digest[:12]) if doc.exists else tr("editor.not_created")
    parts = [f"<!doctype html><html lang={lg}><head><meta charset=utf-8>"
             f'<meta name=viewport content="width=device-width,initial-scale=1"><title>{esc(tr("editor.title"))}</title>'
             f"<style>{_CSS}</style></head><body><main>",
             f"<h1>{esc(tr('editor.title'))}</h1>",
             f'<p class=sub><code>{esc(str(store.path))}</code> &middot; {esc(status)}</p>']
    if errors:
        items = "".join(f"<li>{esc(e)}</li>" for e in errors)
        parts.append(f'<div class="msg err" role=alert><strong>{esc(tr("editor.not_saved"))}</strong>'
                     f"<ul>{items}</ul></div>")
    elif saved:
        parts.append(f'<div class="msg ok" role=status>{esc(tr("editor.saved"))}</div>')
    parts.append(f'<form method=post action="/"><input type=hidden name=_token value="{esc(token)}">'
                 f'<input type=hidden name=_digest value="{esc(doc.digest)}">')
    for table in COMMON_TABLES:
        parts.append(f"<section><h2>{esc(i18n.table_title(table, lg))}</h2><div class=grid>")
        for s in SETTINGS:
            if s.table != table:
                continue
            name = _dotted(s.location())
            parts.append(f'<div><label for="{esc(name)}">{esc(s.key)}</label>{_input(s, name, shown[s.location()])}'
                         f"<small>{esc(helptext(s))} &middot; <code>{esc(name)}</code></small></div>")
        parts.append("</div></section>")
    head = "".join(f"<th>{esc(i18n.mode_label(m, lg))}</th>" for m in MODE_NAMES)
    parts.append(f"<section><h2>{esc(i18n.table_title('modes', lg))}</h2><div class=matrix><table>"
                 f"<thead><tr><th>{esc(tr('editor.setting'))}</th>{head}</tr></thead><tbody>")
    for s in SETTINGS:
        if not s.per_mode:
            continue
        cells = "".join(f"<td>{_input(s, _dotted(s.location(m)), shown[s.location(m)])}</td>" for m in MODE_NAMES)
        parts.append(f"<tr><th>{esc(s.key)}<small>{esc(helptext(s))}</small></th>{cells}</tr>")
    parts.append(f"</tbody></table></div></section><button type=submit>{esc(tr('editor.save'))}</button></form>"
                 "</main></body></html>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class SettingsGui:
    """The settings editor server (``serve_forever`` blocks; ``start`` runs it on a thread)."""

    def __init__(self, store: SettingsStore, host: str = "127.0.0.1", port: int = 0) -> None:
        self.store = store
        self.token = secrets.token_urlsafe(24)
        self.server = ThreadingHTTPServer((host, port), _make_handler(self))
        self.server.daemon_threads = True
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self) -> Tuple[str, int]:
        host, port = self.server.server_address[:2]
        return str(host), int(port)

    @property
    def url(self) -> str:
        host, port = self.address
        return f"http://{host}:{port}/"

    def allowed_hosts(self) -> Tuple[str, ...]:
        _, port = self.address
        return (f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}")

    def serve_forever(self) -> None:
        self.server.serve_forever()

    def start(self) -> threading.Thread:
        self._thread = threading.Thread(target=self.server.serve_forever, name="settings-gui", daemon=True)
        self._thread.start()
        return self._thread

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> "SettingsGui":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _make_handler(gui: SettingsGui) -> type:
    class Handler(BaseHTTPRequestHandler):
        server_version = "simulator-settings/1"

        def log_message(self, format: str, *args: Any) -> None:  # quiet by default
            return

        # -- plumbing ---------------------------------------------------------

        def _send(self, status: int, body: bytes, content_type: str, extra: Optional[Dict[str, str]] = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; "
                                                        "form-action 'self'")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status: int, data: Any) -> None:
            self._send(status, json.dumps(data, default=str).encode("utf-8"), "application/json")

        def _html(self, status: int, page: str) -> None:
            self._send(status, page.encode("utf-8"), "text/html; charset=utf-8")

        def _host_ok(self) -> bool:
            return self.headers.get("Host", "") in gui.allowed_hosts()

        def _body(self) -> Optional[bytes]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return None
            if length < 0 or length > MAX_BODY:
                return None
            return self.rfile.read(length)

        def _load(self) -> SettingsDocument:
            return gui.store.load(allow_unknown=True)

        # -- GET --------------------------------------------------------------

        def do_GET(self) -> None:
            if not self._host_ok():
                self._json(HTTPStatus.FORBIDDEN, {"error": "unexpected Host header"})
                return
            url = urlsplit(self.path)
            try:
                if url.path == "/":
                    saved = "saved=1" in (url.query or "")
                    self._html(HTTPStatus.OK, render_page(gui.store, self._load(), gui.token, saved=saved))
                elif url.path == "/api/schema":
                    self._json(HTTPStatus.OK, form_schema())
                elif url.path == "/api/settings":
                    doc = self._load()
                    self._json(HTTPStatus.OK, {
                        "path": str(gui.store.path), "exists": doc.exists, "digest": doc.digest,
                        "values": {_dotted(loc): v for loc, v in sorted(current_values(doc).items())},
                        "warnings": [str(w) for w in doc.warnings]})
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except SettingsError as exc:
                self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "the settings file is invalid",
                                                             "issues": [str(i) for i in exc.issues]})
            except SettingsFileError as exc:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

        do_HEAD = do_GET

        # -- POST -------------------------------------------------------------

        def do_POST(self) -> None:
            if not self._host_ok():
                self._json(HTTPStatus.FORBIDDEN, {"error": "unexpected Host header"})
                return
            body = self._body()
            if body is None:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "missing or oversized body"})
                return
            path = urlsplit(self.path).path
            if path == "/api/settings":
                self._post_json(body)
            elif path == "/":
                self._post_form(body)
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def _post_json(self, body: bytes) -> None:
            if not secrets.compare_digest(self.headers.get(TOKEN_HEADER, ""), gui.token):
                self._json(HTTPStatus.FORBIDDEN, {"error": f"missing or wrong {TOKEN_HEADER}"})
                return
            try:
                data = json.loads(body.decode("utf-8"))
                if not isinstance(data, dict) or not isinstance(data.get("values"), dict):
                    raise ValueError("expected {\"values\": {...}}")
            except (ValueError, UnicodeDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": f"invalid JSON body: {exc}"})
                return
            digest = data.get("digest")
            try:
                patch = parse_json_values(data["values"])
                doc = gui.store.update(patch, expected_digest=digest if isinstance(digest, str) else None,
                                       force=bool(data.get("force", False)))
            except SettingsError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid settings", "issues": [str(i) for i in exc.issues]})
            except SettingsConflictError as exc:
                self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
            except SettingsLockError as exc:
                self._json(HTTPStatus.LOCKED, {"error": str(exc)})
            except SettingsFileError as exc:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            else:
                self._json(HTTPStatus.OK, {"digest": doc.digest, "written": sorted(_dotted(loc) for loc in patch)})

        def _post_form(self, body: bytes) -> None:
            try:
                fields = parse_qs(body.decode("utf-8"), keep_blank_values=True, max_num_fields=1000)
            except (UnicodeDecodeError, ValueError):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid form body"})
                return
            if not secrets.compare_digest(fields.get("_token", [""])[-1], gui.token):
                self._json(HTTPStatus.FORBIDDEN, {"error": "missing or wrong form token"})
                return
            digest = fields.get("_digest", [""])[-1]
            try:
                doc = self._load()
                submitted = {}
                try:
                    patch = parse_form(fields, current_values(doc))
                except SettingsError as exc:
                    submitted = _raw_form_values(fields)
                    self._html(HTTPStatus.BAD_REQUEST, render_page(gui.store, doc, gui.token, submitted,
                                                                   [str(i) for i in exc.issues]))
                    return
                if patch:
                    gui.store.update(patch, expected_digest=digest)
            except SettingsConflictError:
                self._html(HTTPStatus.CONFLICT, render_page(gui.store, self._load(), gui.token, None, [
                    "The settings file changed after this page was loaded (another tab, the CLI or an editor). "
                    "The form now shows the current file; apply your change again."]))
                return
            except SettingsError as exc:
                self._html(HTTPStatus.BAD_REQUEST, render_page(gui.store, self._load(), gui.token, None,
                                                               [str(i) for i in exc.issues]))
                return
            except SettingsFileError as exc:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return
            self._send(HTTPStatus.SEE_OTHER, b"", "text/plain", {"Location": "/?saved=1"})

    return Handler


def _raw_form_values(fields: Mapping[str, List[str]]) -> Dict[Location, Any]:
    """Submitted text by location, so a rejected form re-renders with what the user typed."""
    out: Dict[Location, Any] = {}
    for name, raw in fields.items():
        if name.startswith("_"):
            continue
        try:
            location = _split(name)
            setting = lookup(*location)
        except SettingsError:
            continue
        text = raw[-1]
        out[location] = text.lower() in ("1", "true", "yes", "on") if setting.kind is bool else text
    return out


def serve(store: SettingsStore, host: str = "127.0.0.1", port: int = 8765, out: Optional[TextIO] = None) -> None:
    """Run the editor until interrupted (``python -m simulator --gui``)."""
    gui = SettingsGui(store, host, port)
    if out is not None:
        out.write(f"settings editor for {store.path} on {gui.url} (Ctrl+C to stop)\n")
        out.flush()
    try:
        gui.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        gui.server.server_close()


__all__ = [
    "SettingsGui",
    "current_values",
    "parse_form",
    "parse_json_values",
    "render_page",
    "serve",
]
