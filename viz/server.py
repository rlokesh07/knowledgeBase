"""Serve the graph viewer and persist tag filter state."""

from __future__ import annotations

import json
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
VIZ_DIR = Path(__file__).resolve().parent
DB_PATH = ROOT / "database.pkl"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import graphStore  # noqa: E402
import node_tags  # noqa: E402
import agent_dialog  # noqa: E402


def _load_store() -> graphStore.GraphStore:
    store = graphStore.GraphStore()
    if DB_PATH.exists():
        store.loadFromFile(str(DB_PATH))
    return store


def _export_graph_json(store: graphStore.GraphStore) -> None:
    from viz.export import main as export_viz

    export_viz()


class GraphViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VIZ_DIR), **kwargs)

    def log_message(self, format: str, *args) -> None:
        if self.path.startswith("/api/"):
            super().log_message(format, *args)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/tags":
            self._handle_get_tags()
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/disabled-tags":
            self._handle_set_disabled_tags()
            return
        if path == "/api/chat":
            self._handle_chat()
            return
        self.send_error(404, "Not found")

    def _handle_get_tags(self) -> None:
        store = _load_store()
        payload = {
            "tags": node_tags.tag_summary(store),
            "disabled_tags": sorted(node_tags.load_disabled_tags()),
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_set_disabled_tags(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        disabled = data.get("disabled_tags")
        if not isinstance(disabled, list) or not all(isinstance(t, str) for t in disabled):
            self.send_error(400, "disabled_tags must be a list of strings")
            return

        if not DB_PATH.exists():
            self.send_error(404, "database.pkl not found")
            return

        store = _load_store()
        node_tags.set_disabled_tags(store, set(disabled), db_path=DB_PATH)
        try:
            _export_graph_json(store)
        except Exception as exc:
            self.send_error(500, f"graph export failed: {exc}")
            return

        payload = {
            "ok": True,
            "disabled_tags": sorted(node_tags.load_disabled_tags()),
            "disabled_nodes": sum(1 for n in store.nodes if getattr(n, "disabled", False)),
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_chat(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        message = data.get("message")
        if not isinstance(message, str) or not message.strip():
            self.send_error(400, "message is required")
            return

        selected = data.get("selected_node_uuid")
        if selected is not None and not isinstance(selected, str):
            self.send_error(400, "selected_node_uuid must be a string")
            return

        try:
            result = agent_dialog.handle_chat(
                message.strip(),
                project_root=ROOT,
                selected_node_uuid=(selected or "").strip() or None,
            )
        except Exception as exc:
            self.send_error(500, str(exc))
            return

        payload = {
            "reply": result.reply,
            "logs": result.logs,
            "graph_updated": result.graph_updated,
            "action": result.action,
            "propagation": result.propagation,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main() -> None:
    port = 8765
    server = ThreadedHTTPServer(("127.0.0.1", port), GraphViewerHandler)
    print(f"Graph viewer → http://127.0.0.1:{port}/")
    print("Tag toggles persist to disabled_tags.json and database.pkl")
    print("Agent chat → POST /api/chat (use the panel in the viewer)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
