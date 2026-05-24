"""Chat orchestrator for the viz agent dialog."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

import graphStore
import pipeline
import state_propagation
from pipeline import format_duration


@dataclass
class ChatResponse:
    reply: str
    logs: list[str] = field(default_factory=list)
    graph_updated: bool = False
    action: str = ""
    propagation: dict | None = None


def _progress_factory(logs: list[str]):
    def progress(msg: str) -> None:
        logs.append(msg)

    return progress


def _find_candidates(
    store: graphStore.GraphStore,
    engine,
    question: str,
    *,
    prefer_uuid: str | None = None,
) -> list[graphStore.Node]:
    from agent import _find_candidate_nodes

    candidates = _find_candidate_nodes(store, engine, question)
    if not prefer_uuid:
        return candidates

    preferred = next((n for n in store.nodes if n.uuid == prefer_uuid), None)
    if preferred is None or getattr(preferred, "disabled", False):
        return candidates

    out = [preferred]
    seen = {preferred.uuid}
    for node in candidates:
        if node.uuid not in seen:
            out.append(node)
            seen.add(node.uuid)
    return out


def _run_predict(
    store: graphStore.GraphStore,
    engine,
    message: str,
    logs: list[str],
    *,
    selected_node_uuid: str | None = None,
) -> ChatResponse:
    progress = _progress_factory(logs)
    candidates = _find_candidates(
        store,
        engine,
        message,
        prefer_uuid=selected_node_uuid,
    )
    if not candidates:
        return ChatResponse(
            reply="No nodes in the graph to run predict mode on.",
            logs=logs,
            action="predict",
        )

    if selected_node_uuid:
        progress(f"Preferred seed: {candidates[0].label}")

    result = state_propagation.run_propagation(
        store,
        engine,
        message,
        candidates,
        progress,
    )
    if result is None:
        return ChatResponse(
            reply="Predict mode failed (no seed selected or content filter).",
            logs=logs,
            action="predict",
        )

    live = [
        {"label": c.label, "object_type": c.object_type, "new_state": c.new_state[:400]}
        for c in result.changes
        if c.is_live_signal
    ]
    return ChatResponse(
        reply=result.answer,
        logs=logs,
        graph_updated=False,
        action="predict",
        propagation={
            "seed_label": result.seed_label,
            "change_count": len(result.changes),
            "live_signals": live,
        },
    )


def _help_text() -> str:
    return (
        "Commands:\n"
        "• Ask any question — runs predict mode (ephemeral state propagation)\n"
        "• Click a node to use it as the preferred seed, then ask your question\n"
        "• /sync polymarket — fetch relevant prediction markets\n"
        "• /sync instruments — refresh live price/level nodes\n"
        "• /link markets — PREDICTS edges from markets to objects\n"
        "• /link instruments — INFLUENCES/DEPENDS_ON from instruments to objects\n"
        "• /link market-markets — implication edges between markets\n"
        "• /link all — all linking passes\n"
        "• /refresh — re-export graph.json\n"
        "• /build — full pipeline (python main.py all)\n"
        "• /help — this message\n\n"
        "Or run from terminal: python main.py"
    )


_SLASH_COMMANDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^/help\b", re.I), "help"),
    (re.compile(r"^/build\b", re.I), "build_all"),
    (re.compile(r"^/sync\s+polymarket\b", re.I), "sync_polymarket"),
    (re.compile(r"^/sync\s+instruments?\b", re.I), "sync_instruments"),
    (re.compile(r"^/link\s+all\b", re.I), "link_all"),
    (re.compile(r"^/link\s+markets?\b", re.I), "link_market_edges"),
    (re.compile(r"^/link\s+instruments?\b", re.I), "link_instrument_edges"),
    (re.compile(r"^/link\s+market-markets?\b", re.I), "link_market_market_edges"),
    (re.compile(r"^/refresh\b", re.I), "refresh_graph"),
]


def _parse_slash_command(message: str) -> str | None:
    text = message.strip()
    for pattern, action in _SLASH_COMMANDS:
        if pattern.search(text):
            return action
    return None


def handle_chat(
    message: str,
    *,
    project_root: Path | None = None,
    selected_node_uuid: str | None = None,
) -> ChatResponse:
    root = project_root or pipeline.project_root()
    load_dotenv(root / ".env")

    msg = (message or "").strip()
    if not msg:
        return ChatResponse(reply="Send a question or type /help for commands.")

    chat_start = time.perf_counter()
    logs: list[str] = []
    progress = _progress_factory(logs)
    db_path = pipeline.db_path(root)
    store = pipeline.load_store(root)

    action = _parse_slash_command(msg)
    if action is None:
        action = "predict"

    if action == "help":
        return ChatResponse(reply=_help_text(), action="help")

    def _finish(response: ChatResponse) -> ChatResponse:
        elapsed = time.perf_counter() - chat_start
        merged = list(response.logs) if response.logs else list(logs)
        merged.append(f"Elapsed: {format_duration(elapsed)}")
        response.logs = merged
        return response

    if action == "refresh_graph":
        if not db_path.exists():
            return _finish(ChatResponse(reply="database.pkl not found.", action="refresh_graph"))
        pipeline.export_viz(root, progress)
        return _finish(ChatResponse(
            reply="Graph exported to viz/graph.json.",
            graph_updated=True,
            action="refresh_graph",
        ))

    if action == "predict":
        if not db_path.exists():
            return _finish(ChatResponse(
                reply="database.pkl not found — run `python main.py` first.",
                action="predict",
            ))
        engine = pipeline.build_engine()
        return _finish(_run_predict(
            store,
            engine,
            msg,
            logs,
            selected_node_uuid=selected_node_uuid,
        ))

    if action == "build_all":
        pipeline.run_pipeline(
            index=True,
            sync=True,
            link=True,
            export=True,
            root=root,
            progress_fn=progress,
        )
        return _finish(ChatResponse(
            reply="Full pipeline complete. Graph rebuilt and exported.",
            graph_updated=True,
            action="build_all",
        ))

    if not db_path.exists():
        return _finish(ChatResponse(
            reply="database.pkl not found — run `python main.py` first.",
            action=action,
        ))

    engine = pipeline.build_engine()

    if action == "sync_polymarket":
        pipeline.sync_polymarket(store, engine, progress)
        pipeline.save_store(store, root, progress)
        pipeline.export_viz(root, progress)
        return _finish(ChatResponse(
            reply="Polymarket sync complete.",
            graph_updated=True,
            action="sync_polymarket",
        ))

    if action == "sync_instruments":
        pipeline.sync_instruments(store, progress)
        pipeline.save_store(store, root, progress)
        pipeline.export_viz(root, progress)
        return _finish(ChatResponse(
            reply="Instrument sync complete.",
            graph_updated=True,
            action="sync_instruments",
        ))

    if action == "link_all":
        pipeline.link_all_edges(store, engine, progress)
        pipeline.save_store(store, root, progress)
        pipeline.export_viz(root, progress)
        return _finish(ChatResponse(
            reply="All edge linking complete.",
            graph_updated=True,
            action="link_all",
        ))

    if action == "link_market_edges":
        import market_edges

        count = market_edges.build_market_edges(store, engine, progress)
        pipeline.save_store(store, root, progress)
        pipeline.export_viz(root, progress)
        return _finish(ChatResponse(
            reply=f"Market→object linking complete: {count} edge(s).",
            graph_updated=True,
            action="link_market_edges",
        ))

    if action == "link_instrument_edges":
        import instrument_edges

        count = instrument_edges.build_instrument_edges(
            store, engine, progress, replace_all_instrument_edges=True
        )
        pipeline.save_store(store, root, progress)
        pipeline.export_viz(root, progress)
        return _finish(ChatResponse(
            reply=f"Instrument→object linking complete: {count} edge(s).",
            graph_updated=True,
            action="link_instrument_edges",
        ))

    if action == "link_market_market_edges":
        import market_market_edges

        count = market_market_edges.build_market_market_edges(store, engine, progress)
        pipeline.save_store(store, root, progress)
        pipeline.export_viz(root, progress)
        return _finish(ChatResponse(
            reply=f"Market→market linking complete: {count} edge(s).",
            graph_updated=True,
            action="link_market_market_edges",
        ))

    return _finish(ChatResponse(reply="Unknown action.", action=action))
