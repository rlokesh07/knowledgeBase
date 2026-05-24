"""Link Polymarket market nodes to ontology objects via PREDICTS edges."""

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


def _market_nodes(store: graphStore.GraphStore) -> list[graphStore.Node]:
    return [
        n
        for n in store.nodes
        if (getattr(n, "object_type", "") or "") == "market"
        and not getattr(n, "disabled", False)
    ]


def _object_nodes(store: graphStore.GraphStore, *, max_objects: int) -> list[graphStore.Node]:
    objs = [
        n
        for n in store.nodes
        if (getattr(n, "object_type", "") or "") != "market"
        and not getattr(n, "disabled", False)
    ]
    objs.sort(key=lambda n: len(n.chunk_uuids), reverse=True)
    return objs[:max_objects]


def _edge_weight_for_market(market: graphStore.Node) -> float:
    props = getattr(market, "properties", None) or {}
    prices = props.get("outcome_prices") or []
    if prices:
        try:
            return max(float(p) for p in prices)
        except (TypeError, ValueError):
            pass
    volume = props.get("volume")
    if volume:
        try:
            return min(1.0, float(volume) / 1_000_000.0)
        except (TypeError, ValueError):
            pass
    return 0.5


def build_market_edges(
    store: graphStore.GraphStore,
    engine: aiEngine.aiEngine,
    progress,
) -> int:
    """Create PREDICTS edges from markets to ontology objects. Returns number of edges added."""
    markets = _market_nodes(store)
    if not markets:
        progress("No market nodes found — run sync_polymarket.py first.")
        return 0

    max_objects = _env_int("MARKET_LINK_MAX_OBJECTS", 100)
    batch_size = _env_int("MARKET_LINK_GPT_BATCH_SIZE", 5)
    objects = _object_nodes(store, max_objects=max_objects)
    if not objects:
        progress("No ontology objects to link markets to.")
        return 0

    market_uuids = {n.uuid for n in markets}
    market_by_uuid = {n.uuid: n for n in markets}
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

    batches = [markets[i : i + batch_size] for i in range(0, len(markets), batch_size)]
    resolved: dict[tuple[str, str], tuple[float, str]] = {}
    graph_lock = threading.Lock()

    def _merge_links(results: list[tuple[str, str, float, str]]) -> None:
        for market_uuid, target_uuid, weight, mechanism in results:
            key = (market_uuid, target_uuid)
            if key not in resolved:
                resolved[key] = (weight, mechanism)

    def _link_batch(_batch_idx: int, batch: list[graphStore.Node]) -> list[tuple[str, str, float, str]]:
        markets_payload = [
            {
                "uuid": m.uuid,
                "question": m.label,
                "description": (m.description or "")[:600],
            }
            for m in batch
        ]
        links = engine.link_markets_to_objects(markets_payload, objects_payload)
        out: list[tuple[str, str, float, str]] = []
        for link in links:
            market = market_by_uuid.get(link.market_uuid)
            if not market:
                continue
            weight = _edge_weight_for_market(market)
            out.append((link.market_uuid, link.target_uuid, weight, link.mechanism.strip()))
        return out

    run_parallel_batches(
        batches,
        worker=_link_batch,
        on_results=_merge_links,
        progress=progress,
        label="Market linking",
        workers=gpt_worker_count(),
        lock=graph_lock,
    )

    new_edges: list[graphStore.Edge] = []
    for (src, tgt), (weight, mechanism) in resolved.items():
        new_edges.append(
            graphStore.Edge(
                str(uuid.uuid4()),
                src,
                tgt,
                "Predicts",
                weight=weight,
                relationship_type="PREDICTS",
                mechanism=mechanism,
            )
        )

    kept = [
        e
        for e in store.edges
        if e.uuid1 not in market_uuids and e.uuid2 not in market_uuids
    ]
    with graph_lock:
        store.edges = kept + new_edges
    progress(f"Market edges: {len(new_edges)} PREDICTS link(s) ({len(kept)} document edge(s) kept).")
    return len(new_edges)
