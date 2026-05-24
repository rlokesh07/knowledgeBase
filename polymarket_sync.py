"""Sync relevant Polymarket markets into the knowledge graph as prediction-market nodes."""

import os
import threading
import uuid

import graphStore
import aiEngine
import node_tags
from gpt_parallel import gpt_worker_count, run_parallel_batches
from polymarket_client import PolymarketClient, PolymarketMarket, search_terms_from_graph

TOPIC_UUID_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "kms.topic-node.v1")


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _ontology_summary(store: graphStore.GraphStore, *, max_objects: int = 60) -> str:
    ranked = sorted(
        (n for n in store.nodes if not getattr(n, "disabled", False)),
        key=lambda n: len(n.chunk_uuids),
        reverse=True,
    )
    lines = []
    for node in ranked[:max_objects]:
        otype = getattr(node, "object_type", "") or "other"
        desc = (getattr(node, "description", "") or "")[:120]
        state = (getattr(node, "state", "") or "")[:200]
        line = f"• {node.label} ({otype})"
        if desc:
            line += f" — {desc}"
        if state:
            line += f" | state: {state}"
        lines.append(line)
    return "\n".join(lines)


def _market_node_uuid(market_id: str) -> str:
    return str(uuid.uuid5(TOPIC_UUID_NS, f"polymarket:{market_id}"))


def _existing_polymarket_ids(store: graphStore.GraphStore) -> set[str]:
    ids: set[str] = set()
    for node in store.nodes:
        props = getattr(node, "properties", None) or {}
        pid = props.get("polymarket_id")
        if pid:
            ids.add(str(pid))
    return ids


def _market_description(market: PolymarketMarket) -> str:
    if market.event_title and market.event_title != market.question:
        return f"Polymarket market under «{market.event_title}»: {market.question}"
    return f"Polymarket prediction market: {market.question}"


def _market_properties(market: PolymarketMarket) -> dict:
    return {
        "polymarket_id": market.id,
        "slug": market.slug,
        "event_slug": market.event_slug,
        "url": market.url,
        "outcomes": market.outcomes,
        "outcome_prices": market.outcome_prices,
        "volume": market.volume,
        "liquidity": market.liquidity,
        "end_date": market.end_date,
        "source": "polymarket",
    }


def _upsert_market_node(
    store: graphStore.GraphStore,
    market: PolymarketMarket,
    *,
    next_cluster_id: list[int],
) -> bool:
    """Insert or update a market node. Returns True if newly created."""
    nid = _market_node_uuid(market.id)
    state = market.format_state()
    props = _market_properties(market)

    for node in store.nodes:
        if node.uuid == nid or (node.properties or {}).get("polymarket_id") == market.id:
            node.label = market.question
            node.object_type = "market"
            node.description = _market_description(market)
            node.state = state
            node.properties = props
            return False

    store.nodes.append(
        graphStore.Node(
            nid,
            market.question,
            [],
            cluster_id=next_cluster_id[0],
            object_type="market",
            description=_market_description(market),
            state=state,
            properties=props,
        )
    )
    next_cluster_id[0] += 1
    return True


def _assign_market_tags(
    store: graphStore.GraphStore,
    engine: aiEngine.aiEngine,
    market_nodes: list[graphStore.Node],
    progress,
) -> None:
    if not market_nodes:
        return

    batch_size = _env_int("POLYMARKET_TAG_GPT_BATCH_SIZE", 20)
    by_uuid = {n.uuid: n for n in market_nodes}
    batches = [
        market_nodes[i : i + batch_size]
        for i in range(0, len(market_nodes), batch_size)
    ]
    progress(f"Assigning tags to {len(market_nodes)} market node(s)…")
    merge_lock = threading.Lock()

    def _merge_tags(results: list[tuple[str, list[str]]]) -> None:
        for node_uuid, tags in results:
            node = by_uuid.get(node_uuid)
            if node:
                node.tags = tags

    def _tag_batch(_batch_idx: int, batch: list[graphStore.Node]) -> list[tuple[str, list[str]]]:
        payload = [
            {
                "uuid": n.uuid,
                "label": n.label,
                "object_type": n.object_type,
                "description": (n.description or "")[:600],
            }
            for n in batch
        ]
        rows = engine.assign_node_tags(payload)
        out: list[tuple[str, list[str]]] = []
        for row in rows:
            tags = node_tags.normalize_tags(row.tags)
            if "polymarket" not in tags:
                tags.append("polymarket")
            out.append((row.uuid, tags))
        return out

    run_parallel_batches(
        batches,
        worker=_tag_batch,
        on_results=_merge_tags,
        progress=progress,
        label="Market tagging",
        workers=gpt_worker_count(),
        lock=merge_lock,
    )

    node_tags.apply_disabled_tags(store)


def fetch_candidate_markets(
    store: graphStore.GraphStore,
    client: PolymarketClient,
    progress,
) -> dict[str, PolymarketMarket]:
    per_query = _env_int("POLYMARKET_SEARCH_LIMIT", 20)
    max_terms = _env_int("POLYMARKET_MAX_SEARCH_TERMS", 25)

    terms = search_terms_from_graph(
        [n for n in store.nodes if not getattr(n, "disabled", False)],
        max_terms=max_terms,
    )
    progress(f"Searching Polymarket for {len(terms)} ontology term(s)…")

    candidates: dict[str, PolymarketMarket] = {}
    for term in terms:
        progress(f"  Query: {term!r}")
        try:
            hits = client.search(term, limit=per_query)
        except Exception as exc:
            progress(f"  Search failed for {term!r}: {exc}")
            continue
        for market in hits:
            candidates[market.id] = market
        progress(f"    {len(hits)} market(s), {len(candidates)} unique total")

    return candidates


def filter_relevant_markets(
    engine: aiEngine.aiEngine,
    store: graphStore.GraphStore,
    candidates: dict[str, PolymarketMarket],
    progress,
) -> list[PolymarketMarket]:
    if not candidates:
        return []

    batch_size = _env_int("POLYMARKET_GPT_BATCH_SIZE", 15)
    summary = _ontology_summary(store)
    markets = list(candidates.values())
    relevant_ids: set[str] = set()
    merge_lock = threading.Lock()

    batches = [markets[i : i + batch_size] for i in range(0, len(markets), batch_size)]

    def _merge_relevance(results: list[str]) -> None:
        for market_id in results:
            relevant_ids.add(market_id)

    def _assess_batch(_batch_idx: int, batch: list[PolymarketMarket]) -> list[str]:
        payload = [
            {
                "id": m.id,
                "question": m.question,
                "description": (m.description or "")[:800],
                "event_title": m.event_title,
            }
            for m in batch
        ]
        rows = engine.assess_market_relevance(summary, payload)
        return [row.market_id for row in rows if row.relevant]

    run_parallel_batches(
        batches,
        worker=_assess_batch,
        on_results=_merge_relevance,
        progress=progress,
        label="Market relevance",
        workers=gpt_worker_count(),
        lock=merge_lock,
    )

    if not relevant_ids and markets:
        progress("GPT returned no relevance matches; falling back to keyword overlap.")
        entity_terms = {
            n.label.lower()
            for n in store.nodes
            if len(n.chunk_uuids) >= 20 and not getattr(n, "disabled", False)
        }
        for m in markets:
            text = f"{m.question} {m.description} {m.event_title}".lower()
            if any(term in text for term in entity_terms if len(term) >= 4):
                relevant_ids.add(m.id)

    selected = [candidates[mid] for mid in relevant_ids if mid in candidates]
    progress(f"{len(selected)} relevant market(s) of {len(candidates)} candidate(s).")
    return selected


def sync_polymarket_markets(
    store: graphStore.GraphStore,
    engine: aiEngine.aiEngine,
    progress,
) -> tuple[int, int]:
    """Fetch, filter, and upsert relevant Polymarket markets. Returns (added, updated)."""
    client = PolymarketClient()
    candidates = fetch_candidate_markets(store, client, progress)
    relevant = filter_relevant_markets(engine, store, candidates, progress)

    next_id = max((n.cluster_id for n in store.nodes), default=-1) + 1
    counter = [next_id]
    added = 0
    updated = 0
    touched: list[graphStore.Node] = []

    for market in relevant:
        nid = _market_node_uuid(market.id)
        if _upsert_market_node(store, market, next_cluster_id=counter):
            added += 1
        else:
            updated += 1
        node = next((n for n in store.nodes if n.uuid == nid), None)
        if node:
            touched.append(node)

    _assign_market_tags(store, engine, touched, progress)

    progress(f"Polymarket sync done: {added} added, {updated} updated.")
    return added, updated
