"""Object–object edges: per-chunk top-K similarity, directed typed relationship extraction."""

import os
import threading
import uuid

import numpy as np

import aiEngine
import graphStore
from gpt_parallel import gpt_worker_count, run_parallel_batches


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms


def build_object_edges(
    store: graphStore.GraphStore,
    engine: aiEngine.aiEngine,
    progress,
) -> None:
    graphStore.recomputed_node_centroids(store)

    nodes = [
        n
        for n in store.nodes
        if n.centroid and n.chunk_uuids and not getattr(n, "disabled", False)
    ]
    if len(nodes) < 2:
        store.edges = []
        progress("Skipping edges (need at least two object nodes with centroids).")
        return

    top_k = _env_int("EDGE_TOP_NODES_PER_CHUNK", 6)
    node_min_cos = _env_float("EDGE_NODE_MIN_COSINE", 0.35)
    max_pairs = _env_int("EDGE_MAX_PAIRS", 300)

    C = np.asarray([n.centroid for n in nodes], dtype=np.float64)
    C_n = _l2_normalize_rows(C)
    node_index = {n.uuid: n for n in nodes}

    candidate_best: dict[tuple[str, str], tuple[float, str]] = {}

    for chunk in store.chunks:
        emb = np.asarray(chunk.embedding, dtype=np.float64)
        norm = float(np.linalg.norm(emb))
        if norm < 1e-12:
            continue
        sims = C_n @ (emb / norm)

        above = [(float(sims[i]), i) for i in range(len(nodes)) if float(sims[i]) >= node_min_cos]
        above.sort(reverse=True)
        top = above[:top_k]

        if len(top) < 2:
            continue

        for pi in range(len(top)):
            for pj in range(pi + 1, len(top)):
                sim_a, ia = top[pi]
                sim_b, ib = top[pj]
                na, nb = nodes[ia], nodes[ib]
                key = (na.uuid, nb.uuid) if na.uuid < nb.uuid else (nb.uuid, na.uuid)
                combined = sim_a + sim_b
                existing = candidate_best.get(key)
                if existing is None or combined > existing[0]:
                    candidate_best[key] = (combined, chunk.text)

    if not candidate_best:
        store.edges = []
        progress("No candidate pairs found above similarity threshold.")
        return

    sorted_candidates = sorted(candidate_best.items(), key=lambda x: x[1][0], reverse=True)
    sorted_candidates = sorted_candidates[:max_pairs]
    progress(
        f"Edge candidates: {len(sorted_candidates)} unique pair(s) "
        f"(top_k={top_k}, node_min_cos={node_min_cos:g}, cap={max_pairs})."
    )

    resolved: dict[tuple[str, str, str], tuple[float, str]] = {}
    graph_lock = threading.Lock()

    def _merge_relationships(results: list[tuple[tuple[str, str, str], float, str]]) -> None:
        for triple, weight, mechanism in results:
            if triple not in resolved:
                resolved[triple] = (weight, mechanism)

    def _classify_pair(
        _batch_idx: int,
        item: tuple[tuple[str, str], tuple[float, str]],
    ) -> list[tuple[tuple[str, str, str], float, str]]:
        (ua, ub), (combined, chunk_text) = item
        na = node_index.get(ua)
        nb = node_index.get(ub)
        if not na or not nb:
            return []

        payload = [
            {
                "uuid_a": ua,
                "label_a": na.label,
                "description_a": na.description,
                "uuid_b": ub,
                "label_b": nb.label,
                "description_b": nb.description,
            }
        ]
        relationships = engine.extract_relationships(chunk_text, payload)
        out: list[tuple[tuple[str, str, str], float, str]] = []
        for rel in relationships:
            triple = (rel.source_uuid, rel.target_uuid, rel.relationship_type)
            pair_key = (
                (rel.source_uuid, rel.target_uuid)
                if rel.source_uuid < rel.target_uuid
                else (rel.target_uuid, rel.source_uuid)
            )
            weight = candidate_best.get(pair_key, (combined, chunk_text))[0]
            out.append((triple, weight, rel.mechanism))
        return out

    run_parallel_batches(
        sorted_candidates,
        worker=_classify_pair,
        on_results=_merge_relationships,
        progress=progress,
        label="Relationship classification",
        workers=gpt_worker_count(),
        lock=graph_lock,
    )

    edges_out: list[graphStore.Edge] = []
    for (src, tgt, rel_type), (weight, mechanism) in resolved.items():
        label = rel_type.replace("_", " ").title()
        eid = str(uuid.uuid4())
        edges_out.append(
            graphStore.Edge(eid, src, tgt, label, weight=weight, relationship_type=rel_type, mechanism=mechanism)
        )

    with graph_lock:
        store.edges = edges_out
    progress(f"Stored {len(store.edges)} typed directed edge(s).")


def format_edges_short(store: graphStore.GraphStore, limit: int = 40) -> None:
    for e in store.edges[:limit]:
        src = e.uuid1[:8]
        tgt = e.uuid2[:8]
        rtype = f"[{e.relationship_type}] " if e.relationship_type else ""
        print(f"  [{src}] --{rtype}{e.relationship}--> [{tgt}]  (w={e.weight:.3f})", flush=True)
    if len(store.edges) > limit:
        print(f"  … {len(store.edges) - limit} more edge(s)", flush=True)
