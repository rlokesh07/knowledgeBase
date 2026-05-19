import json
import os
import uuid
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse, urlunparse

import boto3
import numpy as np
from dotenv import load_dotenv

import aiEngine
import graphStore
import text_chunking
import topic_cluster
import topic_edges
from encoder import Encoder


TOPIC_UUID_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "kms.topic-node.v1")


def _env_truthy(name: str) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _incremental_topics_ready(store: graphStore.GraphStore) -> bool:
    if not store.nodes or not store.chunks:
        return False
    dim = len(store.chunks[0].embedding)
    for node in sorted(store.nodes, key=lambda n: n.cluster_id):
        c = getattr(node, "centroid", None)
        if not c or len(c) != dim or not node.chunk_uuids:
            return False
    return True


def _incremental_assign_min_cosine() -> float:
    """Minimum cosine(similarity between chunk embedding and a topic centroid) to join."""
    raw = (os.environ.get("CLUSTER_ASSIGN_MIN_COSINE") or "0.6").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.6
    return max(-1.0, min(1.0, value))


def _next_cluster_id(nodes: list[graphStore.Node]) -> int:
    return max((n.cluster_id for n in nodes), default=-1) + 1


def _singleton_topic_node(
    engine: aiEngine.aiEngine, chunk: graphStore.Chunk, cluster_id: int
) -> graphStore.Node:
    excerpt = chunk.text[:800]
    label = engine.cluster_topic_label([excerpt])
    centroid_vec = np.asarray(chunk.embedding, dtype=float).tolist()
    nid = str(uuid.uuid5(TOPIC_UUID_NS, "singleton:" + chunk.uuid))
    return graphStore.Node(
        nid,
        label,
        [chunk.uuid],
        cluster_id=cluster_id,
        centroid=centroid_vec,
    )


def _progress(message: str) -> None:
    print(message, flush=True)


def _canonical_source_key(path: Path, project_root: Path) -> str:
    """Stable absolute path for deduping notes (same logical file → same key)."""
    p = path.expanduser()
    if not p.is_absolute():
        p = project_root / p
    try:
        return str(p.resolve(strict=False))
    except OSError:
        return str(p)


def _indexed_source_paths(store: graphStore.GraphStore, project_root: Path) -> set[str]:
    paths: set[str] = set()
    for chunk in store.chunks:
        raw = (chunk.source_path or "").strip()
        if not raw:
            continue
        paths.add(_canonical_source_key(Path(raw), project_root))
    return paths


def _write_index_manifest(store: graphStore.GraphStore, project_root: Path, db_path: Path) -> None:
    paths = sorted(
        {
            _canonical_source_key(Path(c.source_path), project_root)
            for c in store.chunks
            if (c.source_path or "").strip()
        }
    )
    out = db_path.with_name("indexed_sources.json")
    out.write_text(json.dumps(paths, indent=2), encoding="utf-8")


def _rebuild_topic_nodes(
    engine: aiEngine.aiEngine, chunks: list[graphStore.Chunk], labels: np.ndarray
) -> list[graphStore.Node]:
    groups: dict[int, list[graphStore.Chunk]] = defaultdict(list)
    for chunk, lab in zip(chunks, labels):
        groups[int(lab)].append(chunk)

    nodes: list[graphStore.Node] = []
    for cid in sorted(groups.keys()):
        members = groups[cid]
        excerpts = [m.text[:800] for m in members[:5]]
        label = engine.cluster_topic_label(excerpts)
        centroid_vec = np.mean([m.embedding for m in members], axis=0).tolist()
        finger = ",".join(sorted(m.uuid for m in members))
        nid = str(uuid.uuid5(TOPIC_UUID_NS, finger))
        nodes.append(
            graphStore.Node(
                nid,
                label,
                [m.uuid for m in members],
                cluster_id=cid,
                centroid=centroid_vec,
            )
        )
    return nodes


def _run_global_kmeans_and_topics(
    engine: aiEngine.aiEngine, store: graphStore.GraphStore, preamble: str | None = None
) -> None:
    """Assign every chunk to topics via global K-means and LLM naming."""
    if preamble:
        _progress(preamble)
    n = len(store.chunks)
    _progress(f"Clustering {n} chunk embedding(s)…")
    k = topic_cluster.effective_k(n, os.environ.get("CLUSTER_K"))
    X = np.array([c.embedding for c in store.chunks])
    labels = topic_cluster.cluster_labels(X, k)
    _progress(f"Naming {k} topic cluster(s)…")
    store.nodes = _rebuild_topic_nodes(engine, store.chunks, labels)


def _print_topic_snapshot(store: graphStore.GraphStore) -> None:
    print(
        f"\n=== Topic graph ({len(store.nodes)} topics, {len(store.chunks)} chunks) ===",
        flush=True,
    )
    by_uuid = {c.uuid: c for c in store.chunks}
    for node in sorted(store.nodes, key=lambda n: n.cluster_id):
        print(
            f"\n--- [{node.cluster_id}] {node.label}  ({len(node.chunk_uuids)} chunks) ---",
            flush=True,
        )
        for uid in node.chunk_uuids[:4]:
            chunk = by_uuid.get(uid)
            if not chunk:
                continue
            preview = chunk.text[:140].replace("\n", " ").strip()
            if len(chunk.text) > 140:
                preview += "..."
            print(f"  • {preview}", flush=True)
        if len(node.chunk_uuids) > 4:
            _progress(f"  … and {len(node.chunk_uuids) - 4} more chunks")


def _print_edges_summary(store: graphStore.GraphStore) -> None:
    if not store.edges:
        print("\n=== Topic edges (none) ===", flush=True)
        return
    print(f"\n=== Topic edges ({len(store.edges)}) ===", flush=True)
    topic_edges.format_edges_short(store, limit=60)


# Azure OpenAI REST API revision (fallback if URL / env omit api-version).
AZURE_OPENAI_HTTP_API_VERSION = "2024-02-15-preview"

_MISTRAL_OCR_ROUTE = "/providers/mistral/azure/ocr"


def _normalize_mistral_ocr_endpoint(raw: str) -> str:
    """Rewrite common mis-paste OpenAI Cognitive Services host → Foundry OCR URL.

    Runtime evidence showed POST to ``*.cognitiveservices.azure.com`` with no OCR path yields
    HTTP 200 ``Content-Length: 0``. Microsoft samples use ``<name>.services.ai.azure.com``.
    """
    s = raw.strip()
    if not s:
        return s
    u = urlparse(s)
    host = (u.hostname or "").lower()
    netloc_original = u.netloc
    query = u.query
    fragment = u.fragment
    scheme = u.scheme or "https"
    path = (u.path or "").rstrip("/")

    if _MISTRAL_OCR_ROUTE in path.lower():
        return urlunparse((scheme, netloc_original, path or "/", "", query, fragment))

    cog_suffix = ".cognitiveservices.azure.com"
    if host.endswith(cog_suffix):
        stem = host[: -len(cog_suffix)].rstrip(".")
        if stem:
            new_netloc = f"{stem}.services.ai.azure.com"
            rebuilt = urlunparse(
                (
                    scheme,
                    new_netloc,
                    _MISTRAL_OCR_ROUTE,
                    "",
                    query,
                    fragment,
                )
            )
            return rebuilt

    sia_suffix = ".services.ai.azure.com"
    if host.endswith(sia_suffix) and path in ("", "/"):
        return urlunparse(
            (scheme, netloc_original, _MISTRAL_OCR_ROUTE, "", query, fragment)
        )

    return s


def _normalize_azure_openai_endpoint(raw: str) -> tuple[str, str | None]:
    """Collapse a pasted portal URL to scheme://host only.

    If the URL includes ``api-version`` in its query string, returns that value
    (used when AZURE_OPENAI_API_VERSION is unset).
    """
    u = urlparse(raw.strip())
    if not u.scheme or not u.netloc:
        raise SystemExit(
            "AZURE_OPENAI_ENDPOINT must be a URL like "
            "`https://<resource>.cognitiveservices.azure.com` or "
            "`https://<resource>.openai.azure.com` "
            "(not an empty host)."
        )
    base_url = f"{u.scheme}://{u.netloc}"
    qp = parse_qs(u.query)
    api_from_url = (qp.get("api-version") or [None])[0]
    if isinstance(api_from_url, str):
        api_from_url = api_from_url.strip() or None
    return base_url, api_from_url


def _require_env(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and set them."
        )


def main() -> None:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    _progress("Indexing starting… loading environment")

    _require_env(
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "MISTRAL_OCR_ENDPOINT",
        "S3_BUCKET",
    )

    mistral_key = (
        (os.environ.get("AZURE_API_KEY") or os.environ.get("MISTRAL_API_KEY") or "").strip()
    )
    if not mistral_key:
        raise SystemExit(
            "Missing Mistral OCR API key: set AZURE_API_KEY (Azure samples) "
            "or MISTRAL_API_KEY in .env to the key from Deployments + Endpoint."
        )

    notes_dir = Path(os.environ.get("NOTES_DIR", "notes"))
    if not notes_dir.is_absolute():
        notes_dir = project_root / notes_dir
    try:
        notes_dir = notes_dir.resolve(strict=False)
    except OSError:
        pass

    if notes_dir.exists() and not notes_dir.is_dir():
        raise SystemExit(f"NOTES_DIR exists but is not a directory: {notes_dir}")
    notes_dir.mkdir(parents=True, exist_ok=True)

    chat_deployment = (os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"] or "").strip()
    raw_aoai = (os.environ["AZURE_OPENAI_ENDPOINT"] or "").strip()
    azure_base, api_hint_from_url = _normalize_azure_openai_endpoint(raw_aoai)

    api_version = (
        (os.environ.get("AZURE_OPENAI_API_VERSION") or "").strip()
        or (api_hint_from_url or "").strip()
        or AZURE_OPENAI_HTTP_API_VERSION
    )

    embedding_deploy = (
        (os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or "").strip()
        or (os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT") or "").strip()
        or None
    )

    engine = aiEngine.aiEngine(
        apiKey=os.environ["AZURE_OPENAI_API_KEY"],
        baseURL=azure_base,
        apiVersion=api_version,
        primaryModel=chat_deployment,
        embeddingDeployment=embedding_deploy,
    )

    store = graphStore.GraphStore()
    db_path = project_root / "database.pkl"

    chunks_snapshot_env = (os.environ.get("DATABASE_CHUNKS_JSONL") or "").strip()
    if chunks_snapshot_env:
        snap = Path(chunks_snapshot_env)
        chunks_jsonl_path = snap if snap.is_absolute() else project_root / snap
    else:
        chunks_jsonl_path = project_root / "chunks" / "chunks.jsonl"

    if db_path.exists():
        store.loadFromFile(str(db_path))
    chunk_uuids_at_start = {c.uuid for c in store.chunks}

    mistral_doc_model = (os.environ.get("MISTRAL_OCR_MODEL") or "").strip() or (
        "mistral-document-ai-2512"
    )
    mistral_ep = _normalize_mistral_ocr_endpoint(os.environ["MISTRAL_OCR_ENDPOINT"])
    file_encoder = Encoder(
        mistral_key,
        mistral_ep,
        model=mistral_doc_model,
    )

    s3 = boto3.client("s3")
    bucket = os.environ["S3_BUCKET"]

    max_chars, overlap = topic_cluster.default_chunk_window_env()

    entries = sorted(os.listdir(notes_dir))
    _progress(f"Processing {len(entries)} entries from {notes_dir}")

    seen_sources = _indexed_source_paths(store, project_root)
    _progress(f"{len(seen_sources)} source file(s) already indexed (loaded from database).")

    for entry in entries:
        local_path = notes_dir / entry
        if not local_path.is_file():
            continue

        source_key = _canonical_source_key(local_path, project_root)
        if source_key in seen_sources:
            _progress(f"Skipping already indexed: {entry}")
            continue

        _progress(f"Uploading to S3: {entry}")
        s3.upload_file(str(local_path), bucket, entry)

        _progress(f"OCR (Mistral)… this can take minutes for large PDFs: {entry}")
        page_markdowns = file_encoder.encodeDocuments(str(local_path))
        _progress(f"OCR produced {len(page_markdowns)} page(s): {entry}")

        texts = text_chunking.split_pages_into_chunks(page_markdowns, max_chars, overlap)
        _progress(f"Split into {len(texts)} text chunk(s): {entry}")

        embeddings = engine.embed_batch(texts)
        if len(embeddings) != len(texts):
            raise RuntimeError("Embedding batch size mismatch.")

        for piece, vector in zip(texts, embeddings):
            store.chunks.append(
                graphStore.Chunk(str(uuid.uuid4()), piece, vector, source_key)
            )

        seen_sources.add(source_key)

    new_chunks = [c for c in store.chunks if c.uuid not in chunk_uuids_at_start]
    n_chunks = len(store.chunks)

    if n_chunks == 0:
        _progress("No chunks in store; skipping clustering.")
        store.nodes = []
        store.edges = []
    elif _env_truthy("CLUSTER_FULL_REFRESH"):
        _run_global_kmeans_and_topics(
            engine, store, "CLUSTER_FULL_REFRESH: rebuilding all topics."
        )
    elif new_chunks:
        if store.nodes and _incremental_topics_ready(store):
            cos_th = _incremental_assign_min_cosine()
            _progress(
                f"Incremental assignment ({len(new_chunks)} new chunk(s)), "
                f"CLUSTER_ASSIGN_MIN_COSINE≥{cos_th:g} …"
            )
            attached = 0
            spawned = 0
            for ch in new_chunks:
                ordered = sorted(store.nodes, key=lambda n: n.cluster_id)
                centroid_matrix = np.array([n.centroid for n in ordered], dtype=np.float64)
                emb = np.asarray(ch.embedding, dtype=np.float64)
                ix = topic_cluster.best_cluster_above_cos_threshold(
                    emb,
                    centroid_matrix,
                    cos_th,
                )
                if ix is None:
                    nid = _next_cluster_id(store.nodes)
                    store.nodes.append(_singleton_topic_node(engine, ch, nid))
                    spawned += 1
                else:
                    ordered[ix].chunk_uuids.append(ch.uuid)
                    attached += 1
                graphStore.recomputed_node_centroids(store)
            _progress(
                f"  → joined existing topics {attached} time(s), created {spawned} new topic(s)."
            )
        elif store.nodes:
            _run_global_kmeans_and_topics(
                engine,
                store,
                "Existing topics lack usable centroids — one-time global K-means migrate.",
            )
        else:
            _run_global_kmeans_and_topics(engine, store)
    else:
        _progress("No new chunks; keeping existing topics (centroid refresh).")
        if store.nodes:
            graphStore.recomputed_node_centroids(store)

    _progress("Topic snapshot:")
    _print_topic_snapshot(store)
    _progress("Building topic–topic edges…")
    topic_edges.build_topic_edges(store, engine, _progress)
    _print_edges_summary(store)
    store.exportToFile(str(db_path))
    store.export_chunks_jsonl(chunks_jsonl_path)
    _write_index_manifest(store, project_root, db_path)
    try:
        chunks_rel = chunks_jsonl_path.relative_to(project_root)
    except ValueError:
        chunks_rel = chunks_jsonl_path
    _progress(f"Done. Wrote database.pkl + {chunks_rel}")


if __name__ == "__main__":
    main()
