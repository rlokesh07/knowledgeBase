"""Link financial instrument nodes to ontology objects via INFLUENCES / DEPENDS_ON edges."""

import os
import threading
import uuid

import graphStore
import aiEngine
from gpt_parallel import gpt_worker_count, run_parallel_batches


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _instrument_nodes(
    store: graphStore.GraphStore,
    *,
    only_uuids: set[str] | None = None,
) -> list[graphStore.Node]:
    nodes = [
        n
        for n in store.nodes
        if (getattr(n, "object_type", "") or "") == "instrument"
        and not getattr(n, "disabled", False)
    ]
    if only_uuids is not None:
        nodes = [n for n in nodes if n.uuid in only_uuids]
    return nodes


def _object_nodes(store: graphStore.GraphStore, *, max_objects: int) -> list[graphStore.Node]:
    skip = {"market", "instrument"}
    objs = [
        n
        for n in store.nodes
        if (getattr(n, "object_type", "") or "") not in skip
        and not getattr(n, "disabled", False)
    ]
    objs.sort(key=lambda n: len(n.chunk_uuids), reverse=True)
    return objs[:max_objects]


def _edge_weight_for_instrument(instrument: graphStore.Node) -> float:
    props = getattr(instrument, "properties", None) or {}
    for key in ("change_1d_pct", "change_1w_pct"):
        raw = props.get(key)
        if raw is None:
            continue
        try:
            return min(1.0, max(0.3, abs(float(raw)) / 5.0))
        except (TypeError, ValueError):
            pass
    return 0.5


def build_instrument_edges(
    store: graphStore.GraphStore,
    engine: aiEngine.aiEngine,
    progress,
    *,
    only_uuids: set[str] | None = None,
    replace_all_instrument_edges: bool = False,
) -> int:
    """Create INFLUENCES/DEPENDS_ON edges from instruments to ontology objects."""
    instruments = _instrument_nodes(store, only_uuids=only_uuids)
    if not instruments:
        progress("No instrument nodes found — run sync_instruments.py first.")
        return 0

    max_objects = _env_int("INSTRUMENT_LINK_MAX_OBJECTS", 100)
    batch_size = _env_int("INSTRUMENT_LINK_GPT_BATCH_SIZE", 5)
    objects = _object_nodes(store, max_objects=max_objects)
    if not objects:
        progress("No ontology objects to link instruments to.")
        return 0

    instrument_uuids = {n.uuid for n in _instrument_nodes(store)}
    target_uuids = {n.uuid for n in instruments}
    instrument_by_uuid = {n.uuid: n for n in instruments}
    objects_payload = [
        {
            "uuid": o.uuid,
            "label": o.label,
            "object_type": o.object_type,
            "description": (o.description or "")[:200],
            "state": (getattr(o, "state", "") or "")[:600],
        }
        for o in objects
    ]

    batches = [instruments[i : i + batch_size] for i in range(0, len(instruments), batch_size)]
    resolved: dict[tuple[str, str], tuple[str, float, str]] = {}
    graph_lock = threading.Lock()

    def _merge_links(results: list[tuple[str, str, str, float, str]]) -> None:
        for inst_uuid, target_uuid, rel_type, weight, mechanism in results:
            key = (inst_uuid, target_uuid)
            if key not in resolved:
                resolved[key] = (rel_type, weight, mechanism)

    def _link_batch(_batch_idx: int, batch: list[graphStore.Node]) -> list[tuple[str, str, str, float, str]]:
        instruments_payload = [
            {
                "uuid": n.uuid,
                "label": n.label,
                "description": (n.description or "")[:600],
                "state": (getattr(n, "state", "") or "")[:600],
                "symbol": (n.properties or {}).get("symbol") or "",
                "asset_class": (n.properties or {}).get("asset_class") or "",
            }
            for n in batch
        ]
        links = engine.link_instruments_to_objects(instruments_payload, objects_payload)
        out: list[tuple[str, str, str, float, str]] = []
        for link in links:
            inst = instrument_by_uuid.get(link.instrument_uuid)
            if not inst:
                continue
            weight = _edge_weight_for_instrument(inst)
            rel = link.relationship_type.strip().upper()
            if rel not in {"INFLUENCES", "DEPENDS_ON"}:
                rel = "INFLUENCES"
            out.append(
                (
                    link.instrument_uuid,
                    link.target_uuid,
                    rel,
                    weight,
                    link.mechanism.strip(),
                )
            )
        return out

    run_parallel_batches(
        batches,
        worker=_link_batch,
        on_results=_merge_links,
        progress=progress,
        label="Instrument linking",
        workers=gpt_worker_count(),
        lock=graph_lock,
    )

    new_edges: list[graphStore.Edge] = []
    for (src, tgt), (rel_type, weight, mechanism) in resolved.items():
        rel_phrase = "Influences" if rel_type == "INFLUENCES" else "Depends on"
        new_edges.append(
            graphStore.Edge(
                str(uuid.uuid4()),
                src,
                tgt,
                rel_phrase,
                weight=weight,
                relationship_type=rel_type,
                mechanism=mechanism,
            )
        )

    if replace_all_instrument_edges:
        remove_uuids = instrument_uuids
    else:
        remove_uuids = target_uuids

    kept = [
        e
        for e in store.edges
        if e.uuid1 not in remove_uuids and e.uuid2 not in remove_uuids
    ]
    with graph_lock:
        store.edges = kept + new_edges

    scope = "all instrument" if replace_all_instrument_edges else "new instrument"
    progress(
        f"Instrument edges: {len(new_edges)} INFLUENCES/DEPENDS_ON link(s) "
        f"for {scope} node(s) ({len(kept)} other edge(s) kept)."
    )
    return len(new_edges)
