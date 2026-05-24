"""Link Polymarket markets to each other via INFLUENCES / DEPENDS_ON implication edges."""

import os
import threading
import uuid
from collections import defaultdict

import numpy as np

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


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _market_nodes(store: graphStore.GraphStore) -> list[graphStore.Node]:
    return [
        n
        for n in store.nodes
        if (getattr(n, "object_type", "") or "") == "market"
        and not getattr(n, "disabled", False)
    ]


def _market_catalog_entry(node: graphStore.Node) -> dict:
    props = getattr(node, "properties", None) or {}
    return {
        "uuid": node.uuid,
        "question": node.label,
        "description": (node.description or "")[:500],
        "event_slug": props.get("event_slug") or "",
    }


def _market_embed_text(node: graphStore.Node) -> str:
    props = getattr(node, "properties", None) or {}
    slug = (props.get("event_slug") or "").strip()
    parts = [node.label or "", (node.description or "")[:500]]
    if slug:
        parts.append(f"event: {slug}")
    return "\n".join(p for p in parts if p.strip())


def _compute_market_neighbors(
    markets: list[graphStore.Node],
    engine: aiEngine.aiEngine,
    progress,
) -> dict[str, list[str]]:
    """Return top-K similar market uuids per market (cosine on embedded label+description)."""
    top_k = _env_int("MARKET_MARKET_TOP_K", 15)
    min_cos = _env_float("MARKET_MARKET_MIN_COSINE", 0.40)

    texts = [_market_embed_text(m) for m in markets]
    progress(f"Embedding {len(markets)} market(s) for similarity filter (top_k={top_k}, min_cos={min_cos:g})…")
    embeddings = engine.embed_batch(texts)
    if len(embeddings) != len(markets):
        raise RuntimeError("Market embedding batch size mismatch.")

    mat = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    mat_n = mat / norms

    uuids = [m.uuid for m in markets]
    neighbors: dict[str, list[str]] = {}

    for i, uid in enumerate(uuids):
        sims = mat_n @ mat_n[i]
        sims[i] = -1.0
        order = np.argsort(-sims)
        picks: list[str] = []
        for j in order:
            if len(picks) >= top_k:
                break
            if float(sims[j]) >= min_cos:
                picks.append(uuids[int(j)])
        neighbors[uid] = picks

    with_neighbors = sum(1 for v in neighbors.values() if v)
    progress(f"Similarity filter: {with_neighbors}/{len(markets)} market(s) have at least one neighbor.")
    return neighbors


def _catalog_for_cross_batch(
    batch: list[graphStore.Node],
    neighbor_map: dict[str, list[str]],
    catalog_by_uuid: dict[str, dict],
) -> list[dict]:
    max_catalog = _env_int("MARKET_MARKET_MAX_CATALOG", 40)
    ordered: list[str] = []
    seen: set[str] = set()

    for market in batch:
        for nbr_uuid in neighbor_map.get(market.uuid, []):
            if nbr_uuid in seen or nbr_uuid not in catalog_by_uuid:
                continue
            seen.add(nbr_uuid)
            ordered.append(nbr_uuid)
            if len(ordered) >= max_catalog:
                break
        if len(ordered) >= max_catalog:
            break

    return [catalog_by_uuid[uid] for uid in ordered]


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


def _is_market_market_edge(edge: graphStore.Edge, market_uuids: set[str]) -> bool:
    return edge.uuid1 in market_uuids and edge.uuid2 in market_uuids


def _links_from_gpt(
    batch: list[graphStore.Node],
    catalog: list[dict],
    market_by_uuid: dict[str, graphStore.Node],
    engine: aiEngine.aiEngine,
) -> list[tuple[str, str, str, float, str]]:
    if not catalog:
        return []

    sources = [_market_catalog_entry(m) for m in batch]
    links = engine.link_markets_to_markets(sources, catalog)
    out: list[tuple[str, str, str, float, str]] = []
    for link in links:
        source = market_by_uuid.get(link.source_uuid)
        if not source:
            continue
        weight = _edge_weight_for_market(source)
        out.append(
            (
                link.source_uuid,
                link.target_uuid,
                link.relationship_type,
                weight,
                link.mechanism.strip(),
            )
        )
    return out


def build_market_market_edges(
    store: graphStore.GraphStore,
    engine: aiEngine.aiEngine,
    progress,
) -> int:
    """Create INFLUENCES/DEPENDS_ON edges between markets with logical implication."""
    markets = _market_nodes(store)
    if len(markets) < 2:
        progress("Need at least two market nodes for market–market linking.")
        return 0

    market_uuids = {m.uuid for m in markets}
    market_by_uuid = {m.uuid: m for m in markets}
    catalog_by_uuid = {m.uuid: _market_catalog_entry(m) for m in markets}
    batch_size = _env_int("MARKET_MARKET_LINK_GPT_BATCH_SIZE", 8)

    neighbor_map = _compute_market_neighbors(markets, engine, progress)

    resolved: dict[tuple[str, str, str], tuple[float, str]] = {}
    graph_lock = threading.Lock()

    def _merge(results: list[tuple[str, str, str, float, str]]) -> None:
        for src, tgt, rel_type, weight, mechanism in results:
            key = (src, tgt, rel_type)
            if key not in resolved:
                resolved[key] = (weight, mechanism)

    def _link_same_event_batch(
        _batch_idx: int,
        batch: list[graphStore.Node],
    ) -> list[tuple[str, str, str, float, str]]:
        catalog = [_market_catalog_entry(m) for m in batch]
        return _links_from_gpt(batch, catalog, market_by_uuid, engine)

    def _link_cross_event_batch(
        _batch_idx: int,
        batch: list[graphStore.Node],
    ) -> list[tuple[str, str, str, float, str]]:
        catalog = _catalog_for_cross_batch(batch, neighbor_map, catalog_by_uuid)
        return _links_from_gpt(batch, catalog, market_by_uuid, engine)

    # Same-event groups first (full within-group catalog — usually small)
    by_event: dict[str, list[graphStore.Node]] = defaultdict(list)
    for m in markets:
        slug = ((m.properties or {}).get("event_slug") or "").strip()
        by_event[slug or m.uuid].append(m)

    event_batches = [group for group in by_event.values() if len(group) >= 2]
    if event_batches:
        progress(f"Market–market linking: {len(event_batches)} same-event group(s)…")
        run_parallel_batches(
            event_batches,
            worker=_link_same_event_batch,
            on_results=_merge,
            progress=progress,
            label="Market implication (same event)",
            workers=gpt_worker_count(),
            lock=graph_lock,
        )

    # Cross-event pass — GPT sees only embedding-similar neighbors per batch
    all_batches = [markets[i : i + batch_size] for i in range(0, len(markets), batch_size)]
    cross_batches = [
        batch
        for batch in all_batches
        if _catalog_for_cross_batch(batch, neighbor_map, catalog_by_uuid)
    ]
    skipped = len(all_batches) - len(cross_batches)
    progress(
        f"Market–market linking: {len(cross_batches)} cross-event batch(es)"
        + (f" ({skipped} skipped — no similar neighbors)" if skipped else "")
        + "…"
    )
    if cross_batches:
        run_parallel_batches(
            cross_batches,
            worker=_link_cross_event_batch,
            on_results=_merge,
            progress=progress,
            label="Market implication (cross-event)",
            workers=gpt_worker_count(),
            lock=graph_lock,
        )

    new_edges: list[graphStore.Edge] = []
    for (src, tgt, rel_type), (weight, mechanism) in resolved.items():
        label = rel_type.replace("_", " ").title()
        new_edges.append(
            graphStore.Edge(
                str(uuid.uuid4()),
                src,
                tgt,
                label,
                weight=weight,
                relationship_type=rel_type,
                mechanism=mechanism,
            )
        )

    kept = [e for e in store.edges if not _is_market_market_edge(e, market_uuids)]
    with graph_lock:
        store.edges = kept + new_edges
    progress(
        f"Market–market edges: {len(new_edges)} INFLUENCES/DEPENDS_ON link(s) "
        f"({len(kept)} other edge(s) kept)."
    )
    return len(new_edges)
