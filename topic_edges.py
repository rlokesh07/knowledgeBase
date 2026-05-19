"""Topic–topic edges: centroid similarity, difference-vector K-means, batched GPT labels."""

import os
import uuid
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans

import aiEngine
import graphStore


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


def _diff_cluster_count(n_pairs: int, env_override: str | None) -> int:
    if n_pairs <= 0:
        return 0
    raw = (env_override or "").strip()
    if raw:
        try:
            k = int(raw)
            return max(1, min(k, n_pairs))
        except ValueError:
            pass
    if n_pairs == 1:
        return 1
    return max(2, min(12, max(2, n_pairs // 8)))


def _chunk_excerpts(
    store: graphStore.GraphStore, node: graphStore.Node, max_chunks: int = 2, clip: int = 400
) -> str:
    by_u = {c.uuid: c for c in store.chunks}
    parts: list[str] = []
    for uid in node.chunk_uuids[:max_chunks]:
        ch = by_u.get(uid)
        if not ch:
            continue
        t = ch.text.replace("\n", " ").strip()
        if len(t) > clip:
            t = t[: clip - 3] + "..."
        parts.append(t)
    return " ".join(parts)


def build_topic_edges(
    store: graphStore.GraphStore,
    engine: aiEngine.aiEngine,
    progress,
) -> None:
    graphStore.recomputed_node_centroids(store)

    nodes = [n for n in store.nodes if n.centroid and len(n.centroid) > 0 and n.chunk_uuids]
    if len(nodes) < 2:
        store.edges = []
        progress("Skipping edges (need at least two topics with centroids).")
        return

    X = np.asarray([n.centroid for n in nodes], dtype=np.float64)
    Xn = _l2_normalize_rows(X)
    sim = Xn @ Xn.T

    thresh = _env_float("NODE_EDGE_SIM_THRESHOLD", 0.5)
    max_pairs = _env_int("NODE_EDGE_MAX_PAIRS", 300)

    pairs: list[tuple[int, int, float]] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            s = float(sim[i, j])
            if s >= thresh:
                pairs.append((i, j, s))

    pairs.sort(key=lambda t: t[2], reverse=True)
    pairs = pairs[:max_pairs]

    if not pairs:
        store.edges = []
        progress(f"No topic pairs ≥ cosine {thresh:g} (edges cleared).")
        return

    progress(f"Edge candidates: {len(pairs)} pair(s) (threshold {thresh:g}, cap {max_pairs}).")

    diff_rows: list[np.ndarray] = []
    meta: list[tuple[str, str, float]] = []

    for i, j, s in pairs:
        ni, nj = nodes[i], nodes[j]
        if ni.uuid < nj.uuid:
            ua, ub = ni.uuid, nj.uuid
            ea, eb = Xn[i], Xn[j]
        else:
            ua, ub = nj.uuid, ni.uuid
            ea, eb = Xn[j], Xn[i]
        d = eb - ea
        dn = float(np.linalg.norm(d))
        if dn < 1e-12:
            dnorm = np.zeros_like(d)
        else:
            dnorm = d / dn
        diff_rows.append(dnorm)
        meta.append((ua, ub, s))

    Dmat = np.stack(diff_rows, axis=0)
    k2 = _diff_cluster_count(len(pairs), os.environ.get("NODE_EDGE_DIFF_CLUSTER_K"))
    labels = KMeans(n_clusters=k2, random_state=42, n_init=10).fit_predict(Dmat)

    by_lab: dict[int, list[tuple[str, str, float]]] = defaultdict(list)
    for lab, trip in zip(labels, meta):
        by_lab[int(lab)].append(trip)

    uuid_to_node = {n.uuid: n for n in nodes}
    resolved: dict[tuple[str, str], str] = {}
    batch_max = _env_int("NODE_EDGE_GPT_BATCH_SIZE", 14)

    for lab in sorted(by_lab.keys()):
        group = by_lab[lab]
        progress(f"Labeling edge batch cluster {lab + 1}/{len(by_lab)} ({len(group)} pair(s))…")
        for start in range(0, len(group), batch_max):
            sub = group[start : start + batch_max]
            payload = []
            for ua, ub, s in sub:
                na, nb = uuid_to_node[ua], uuid_to_node[ub]
                payload.append(
                    {
                        "uuid1": ua,
                        "uuid2": ub,
                        "label1": na.label,
                        "label2": nb.label,
                        "excerpts1": _chunk_excerpts(store, na),
                        "excerpts2": _chunk_excerpts(store, nb),
                        "centroid_cosine": round(s, 4),
                    }
                )
            rels = engine.edge_labels_for_pairs(
                payload,
                contrast_cluster_id=lab,
                contrast_clusters_total=k2,
                contrast_cluster_pair_count=len(group),
                sub_batch_pair_count=len(sub),
            )
            for (ua, ub, _), rel in zip(sub, rels):
                resolved[(ua, ub)] = rel

    fallback = "Related topics"
    edges_out: list[graphStore.Edge] = []
    for ua, ub, w in meta:
        rel = resolved.get((ua, ub), fallback)
        rel = (rel or fallback).strip() or fallback
        if len(rel) > 220:
            rel = rel[:217] + "..."
        eid = str(uuid.uuid4())
        edges_out.append(graphStore.Edge(eid, ua, ub, rel, weight=w))

    store.edges = edges_out
    progress(f"Stored {len(store.edges)} topic edge(s).")


def format_edges_short(store: graphStore.GraphStore, limit: int = 40) -> None:
    for e in store.edges[:limit]:
        a = e.uuid1[:8]
        b = e.uuid2[:8]
        print(f"  [{a}] — {e.relationship} — [{b}]  (w={e.weight:.3f})", flush=True)
    if len(store.edges) > limit:
        print(f"  … {len(store.edges) - limit} more edge(s)", flush=True)
