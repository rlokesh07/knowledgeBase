"""Extract and update holistic current-state snapshots for object nodes."""

import os
import threading

import graphStore
import aiEngine
from aiEngine import _content_filter_blocked
from gpt_parallel import gpt_worker_count, run_parallel_batches


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _excerpts_for_node(
    node: graphStore.Node,
    chunks_by_uuid: dict[str, graphStore.Chunk],
    *,
    max_chars: int,
    max_chunks: int,
) -> str:
    parts: list[str] = []
    total = 0
    for uid in node.chunk_uuids[:max_chunks]:
        chunk = chunks_by_uuid.get(uid)
        if not chunk:
            continue
        text = (chunk.text or "").strip()
        if not text:
            continue
        if total + len(text) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                parts.append(text[:remaining])
            break
        parts.append(text)
        total += len(text) + 10
    return "\n\n---\n\n".join(parts)


def _nodes_to_update(
    store: graphStore.GraphStore,
    *,
    new_chunk_uuids: set[str],
    full_refresh: bool,
) -> list[graphStore.Node]:
    if full_refresh:
        return [
            n
            for n in store.nodes
            if n.chunk_uuids and not getattr(n, "disabled", False)
        ]

    out: list[graphStore.Node] = []
    for node in store.nodes:
        if not node.chunk_uuids or getattr(node, "disabled", False):
            continue
        if not (getattr(node, "state", "") or "").strip():
            out.append(node)
            continue
        if new_chunk_uuids and any(uid in new_chunk_uuids for uid in node.chunk_uuids):
            out.append(node)
    return out


def _apply_state_results(
    results: list,
    node_index: dict[str, graphStore.Node],
) -> None:
    for row in results:
        node = node_index.get(row.uuid)
        if node and (row.state or "").strip():
            node.state = row.state.strip()


def _safe_extract_object_states(
    engine: aiEngine.aiEngine,
    payload: list[dict],
) -> tuple[list, bool]:
    try:
        return engine.extract_object_states(payload)
    except Exception as exc:
        if _content_filter_blocked(exc):
            return [], True
        raise


def _extract_state_rows(
    engine: aiEngine.aiEngine,
    payload: list[dict],
    progress,
) -> list:
    """Extract state for a batch; on content filter, retry objects individually."""
    if not payload:
        return []

    results, filtered = _safe_extract_object_states(engine, payload)
    if filtered and len(payload) > 1:
        labels = ", ".join(item["label"] for item in payload[:3])
        if len(payload) > 3:
            labels += ", …"
        progress(
            f"  Batch content-filtered ({labels}); "
            f"retrying {len(payload)} object(s) individually…"
        )
        rows: list = []
        skipped = 0
        for item in payload:
            single, single_filtered = _safe_extract_object_states(engine, [item])
            if single_filtered:
                progress(f"  Skipping state for {item['label']!r} (content filter).")
                skipped += 1
            elif single:
                rows.extend(single)
        if skipped:
            progress(f"  {skipped} object(s) skipped due to content filter in this batch.")
        return rows

    if filtered:
        progress(f"  Skipping state for {payload[0]['label']!r} (content filter).")
        return []

    return results


def _payload_for_nodes(
    batch: list[graphStore.Node],
    chunks_by_uuid: dict[str, graphStore.Chunk],
    *,
    max_chars: int,
    max_chunks: int,
) -> list[dict]:
    payload: list[dict] = []
    for node in batch:
        excerpts = _excerpts_for_node(
            node,
            chunks_by_uuid,
            max_chars=max_chars,
            max_chunks=max_chunks,
        )
        if not excerpts.strip():
            continue
        payload.append(
            {
                "uuid": node.uuid,
                "label": node.label,
                "object_type": node.object_type,
                "description": node.description,
                "excerpts": excerpts,
            }
        )
    return payload


def update_object_states(
    store: graphStore.GraphStore,
    engine: aiEngine.aiEngine,
    progress,
    *,
    new_chunk_uuids: set[str] | None = None,
    full_refresh: bool = False,
) -> None:
    """Refresh state snapshots for nodes with new evidence or missing state."""
    new_chunk_uuids = new_chunk_uuids or set()
    nodes = _nodes_to_update(store, new_chunk_uuids=new_chunk_uuids, full_refresh=full_refresh)
    if not nodes:
        progress("Skipping state extraction (no nodes need updating).")
        return

    chunks_by_uuid = {c.uuid: c for c in store.chunks}
    batch_size = _env_int("STATE_GPT_BATCH_SIZE", 4)
    max_chars = _env_int("STATE_MAX_EXCERPT_CHARS", 8000)
    max_chunks = _env_int("STATE_MAX_CHUNKS_PER_NODE", 12)

    progress(
        f"Extracting current state for {len(nodes)} object(s) "
        f"(batch={batch_size}, excerpt≤{max_chars} chars)…"
    )

    node_index = {n.uuid: n for n in nodes}
    batches = [nodes[i : i + batch_size] for i in range(0, len(nodes), batch_size)]
    payloads = [
        _payload_for_nodes(
            batch,
            chunks_by_uuid,
            max_chars=max_chars,
            max_chunks=max_chunks,
        )
        for batch in batches
    ]
    payloads = [p for p in payloads if p]
    merge_lock = threading.Lock()

    def _merge_state_rows(results: list) -> None:
        _apply_state_results(results, node_index)

    def _extract_batch(_batch_idx: int, payload: list[dict]) -> list:
        return _extract_state_rows(engine, payload, progress)

    run_parallel_batches(
        payloads,
        worker=_extract_batch,
        on_results=_merge_state_rows,
        progress=progress,
        label="State extraction",
        workers=gpt_worker_count(),
        lock=merge_lock,
    )

    filled = sum(1 for n in store.nodes if (getattr(n, "state", "") or "").strip())
    progress(f"State extraction done: {filled}/{len(store.nodes)} object(s) have state.")
