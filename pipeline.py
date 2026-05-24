"""Unified knowledge-graph build pipeline."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlparse, urlunparse

import boto3
import numpy as np
from dotenv import load_dotenv

import aiEngine
import graphStore
import instrument_edges
import instrument_sync
import market_edges
import market_market_edges
import node_tags
import object_state
import polymarket_sync
import text_chunking
import topic_edges
from encoder import Encoder
from gpt_parallel import gpt_worker_count, run_parallel_batches

TOPIC_UUID_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "kms.topic-node.v1")
AZURE_OPENAI_HTTP_API_VERSION = "2024-02-15-preview"
_MISTRAL_OCR_ROUTE = "/providers/mistral/azure/ocr"


def progress(message: str) -> None:
    print(message, flush=True)


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = seconds % 60
    if mins < 60:
        return f"{mins}m {secs:.1f}s"
    hours = mins // 60
    mins = mins % 60
    return f"{hours}h {mins}m {secs:.0f}s"


class _PhaseTimer:
    def __init__(self, label: str, progress_fn=progress):
        self.label = label
        self.progress_fn = progress_fn
        self.elapsed: float = 0.0
        self._start = 0.0

    def __enter__(self) -> "_PhaseTimer":
        self._start = time.perf_counter()
        self.progress_fn(f"=== {self.label} ===")
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self._start
        self.progress_fn(f"--- {self.label}: {format_duration(self.elapsed)} ---")


def _env_truthy(name: str, *, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_enabled(name: str) -> bool:
    """Pipeline phase toggles default to on."""
    return _env_truthy(name, default=True)


def require_env(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and set them."
        )


def normalize_azure_openai_endpoint(raw: str) -> tuple[str, str | None]:
    u = urlparse(raw.strip())
    if not u.scheme or not u.netloc:
        raise SystemExit(
            "AZURE_OPENAI_ENDPOINT must be a URL like "
            "`https://<resource>.cognitiveservices.azure.com` "
            "(not an empty host)."
        )
    base_url = f"{u.scheme}://{u.netloc}"
    qp = parse_qs(u.query)
    api_from_url = (qp.get("api-version") or [None])[0]
    if isinstance(api_from_url, str):
        api_from_url = api_from_url.strip() or None
    return base_url, api_from_url


def _normalize_mistral_ocr_endpoint(raw: str) -> str:
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
            return urlunparse((scheme, new_netloc, _MISTRAL_OCR_ROUTE, "", query, fragment))

    sia_suffix = ".services.ai.azure.com"
    if host.endswith(sia_suffix) and path in ("", "/"):
        return urlunparse((scheme, netloc_original, _MISTRAL_OCR_ROUTE, "", query, fragment))

    return s


def build_engine() -> aiEngine.aiEngine:
    require_env(
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
    )
    chat_deployment = (os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"] or "").strip()
    raw_aoai = (os.environ["AZURE_OPENAI_ENDPOINT"] or "").strip()
    azure_base, api_hint_from_url = normalize_azure_openai_endpoint(raw_aoai)
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
    return aiEngine.aiEngine(
        apiKey=os.environ["AZURE_OPENAI_API_KEY"],
        baseURL=azure_base,
        apiVersion=api_version,
        primaryModel=chat_deployment,
        embeddingDeployment=embedding_deploy,
    )


def project_root() -> Path:
    return Path(__file__).resolve().parent


def db_path(root: Path | None = None) -> Path:
    return (root or project_root()) / "database.pkl"


def chunks_jsonl_path(root: Path | None = None) -> Path:
    root = root or project_root()
    chunks_snapshot_env = (os.environ.get("DATABASE_CHUNKS_JSONL") or "").strip()
    if chunks_snapshot_env:
        snap = Path(chunks_snapshot_env)
        return snap if snap.is_absolute() else root / snap
    return root / "chunks" / "chunks.jsonl"


def load_store(root: Path | None = None) -> graphStore.GraphStore:
    store = graphStore.GraphStore()
    path = db_path(root)
    if path.exists():
        store.loadFromFile(str(path))
    return store


def _canonical_source_key(path: Path, root: Path) -> str:
    p = path.expanduser()
    if not p.is_absolute():
        p = root / p
    try:
        return str(p.resolve(strict=False))
    except OSError:
        return str(p)


def _indexed_source_paths(store: graphStore.GraphStore, root: Path) -> set[str]:
    paths: set[str] = set()
    for chunk in store.chunks:
        raw = (chunk.source_path or "").strip()
        if raw:
            paths.add(_canonical_source_key(Path(raw), root))
    return paths


def _write_index_manifest(store: graphStore.GraphStore, root: Path, path: Path) -> None:
    paths = sorted(
        {
            _canonical_source_key(Path(c.source_path), root)
            for c in store.chunks
            if (c.source_path or "").strip()
        }
    )
    out = path.with_name("indexed_sources.json")
    out.write_text(json.dumps(paths, indent=2), encoding="utf-8")


def assign_chunks_to_objects(
    store: graphStore.GraphStore,
    min_cosine: float,
    progress_fn=progress,
) -> None:
    nodes = [n for n in store.nodes if n.centroid and not getattr(n, "disabled", False)]
    if not nodes or not store.chunks:
        return

    C = np.asarray([n.centroid for n in nodes], dtype=np.float64)
    norms = np.linalg.norm(C, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    C_n = C / norms

    assigned = 0
    for chunk in store.chunks:
        emb = np.asarray(chunk.embedding, dtype=np.float64)
        norm = float(np.linalg.norm(emb))
        if norm < 1e-12:
            continue
        sims = C_n @ (emb / norm)
        for i, sim in enumerate(sims):
            if float(sim) >= min_cosine and chunk.uuid not in nodes[i].chunk_uuids:
                nodes[i].chunk_uuids.append(chunk.uuid)
                assigned += 1

    progress_fn(
        f"Similarity pass: {assigned} additional chunk→object link(s) "
        f"(cosine≥{min_cosine:g}, {len(store.chunks)} chunk(s) × {len(nodes)} object(s))."
    )


def _normalize_object_name(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def extract_and_merge_objects(
    engine: aiEngine.aiEngine,
    store: graphStore.GraphStore,
    chunks: list[graphStore.Chunk],
    progress_fn=progress,
) -> None:
    name_map: dict[str, graphStore.Node] = {}
    for node in store.nodes:
        key = _normalize_object_name(node.label)
        name_map[key] = node

    next_id = [max((n.cluster_id for n in store.nodes), default=-1) + 1]
    batch_size = int(os.environ.get("OBJECT_EXTRACT_BATCH_SIZE") or 20)
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    merge_lock = threading.Lock()

    def _merge_extracted(
        results: list[tuple[list, list[str]]],
    ) -> None:
        for objects, batch_uuids in results:
            for obj in objects:
                key = _normalize_object_name(obj.name)
                if not key:
                    continue
                if key in name_map:
                    existing = name_map[key]
                    for uid in batch_uuids:
                        if uid not in existing.chunk_uuids:
                            existing.chunk_uuids.append(uid)
                else:
                    nid = str(uuid.uuid5(TOPIC_UUID_NS, "object:" + key))
                    node = graphStore.Node(
                        nid,
                        obj.name,
                        list(batch_uuids),
                        cluster_id=next_id[0],
                        object_type=obj.object_type,
                        description=obj.description,
                        tags=node_tags.normalize_tags(getattr(obj, "tags", None)),
                    )
                    store.nodes.append(node)
                    name_map[key] = node
                    next_id[0] += 1

    def _extract_batch(
        _batch_idx: int,
        batch: list[graphStore.Chunk],
    ) -> list[tuple[list, list[str]]]:
        objects = engine.extract_objects([c.text for c in batch])
        batch_uuids = [c.uuid for c in batch]
        return [(objects, batch_uuids)]

    run_parallel_batches(
        batches,
        worker=_extract_batch,
        on_results=_merge_extracted,
        progress=progress_fn,
        label="Object extraction",
        workers=gpt_worker_count(),
        lock=merge_lock,
    )

    progress_fn(f"Object extraction done: {len(store.nodes)} unique object(s) in store.")
    node_tags.apply_disabled_tags(store)


def _print_object_snapshot(store: graphStore.GraphStore) -> None:
    print(
        f"\n=== Object graph ({len(store.nodes)} objects, {len(store.chunks)} chunks) ===",
        flush=True,
    )
    for node in sorted(store.nodes, key=lambda n: n.label.lower()):
        type_str = f" [{node.object_type}]" if node.object_type else ""
        print(f"\n  • {node.label}{type_str}  ({len(node.chunk_uuids)} chunk ref(s))", flush=True)
        if node.description:
            print(f"    {node.description}", flush=True)
        if getattr(node, "state", ""):
            preview = node.state.replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:157] + "…"
            print(f"    state: {preview}", flush=True)


def _print_edges_summary(store: graphStore.GraphStore) -> None:
    if not store.edges:
        print("\n=== Object edges (none) ===", flush=True)
        return
    print(f"\n=== Object edges ({len(store.edges)}) ===", flush=True)
    topic_edges.format_edges_short(store, limit=60)


def _notes_dir(root: Path) -> Path:
    notes = Path(os.environ.get("NOTES_DIR", "notes"))
    if not notes.is_absolute():
        notes = root / notes
    try:
        notes = notes.resolve(strict=False)
    except OSError:
        pass
    if notes.exists() and not notes.is_dir():
        raise SystemExit(f"NOTES_DIR exists but is not a directory: {notes}")
    notes.mkdir(parents=True, exist_ok=True)
    return notes


def index_documents(
    store: graphStore.GraphStore,
    engine: aiEngine.aiEngine,
    root: Path,
    progress_fn=progress,
) -> None:
    """OCR notes → chunks → objects → states → document edges."""
    progress_fn("Indexing documents…")

    require_env(
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
            "Missing Mistral OCR API key: set AZURE_API_KEY or MISTRAL_API_KEY in .env."
        )

    notes_dir = _notes_dir(root)
    chunk_uuids_at_start = {c.uuid for c in store.chunks}

    mistral_doc_model = (os.environ.get("MISTRAL_OCR_MODEL") or "").strip() or (
        "mistral-document-ai-2512"
    )
    mistral_ep = _normalize_mistral_ocr_endpoint(os.environ["MISTRAL_OCR_ENDPOINT"])
    file_encoder = Encoder(mistral_key, mistral_ep, model=mistral_doc_model)

    s3 = boto3.client("s3")
    bucket = os.environ["S3_BUCKET"]
    max_chars = int(os.environ.get("TEXT_CHUNK_MAX_CHARS") or 900)
    overlap = int(os.environ.get("TEXT_CHUNK_OVERLAP") or 120)

    entries = sorted(os.listdir(notes_dir))
    progress_fn(f"Processing {len(entries)} entries from {notes_dir}")

    seen_sources = _indexed_source_paths(store, root)
    progress_fn(f"{len(seen_sources)} source file(s) already indexed.")

    for entry in entries:
        local_path = notes_dir / entry
        if not local_path.is_file():
            continue
        source_key = _canonical_source_key(local_path, root)
        if source_key in seen_sources:
            progress_fn(f"Skipping already indexed: {entry}")
            continue

        progress_fn(f"Uploading to S3: {entry}")
        s3.upload_file(str(local_path), bucket, entry)

        progress_fn(f"OCR (Mistral)… this can take minutes for large PDFs: {entry}")
        page_markdowns = file_encoder.encodeDocuments(str(local_path))
        progress_fn(f"OCR produced {len(page_markdowns)} page(s): {entry}")

        texts = text_chunking.split_pages_into_chunks(page_markdowns, max_chars, overlap)
        progress_fn(f"Split into {len(texts)} text chunk(s): {entry}")

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
        progress_fn("No chunks in store; skipping object extraction.")
        if not store.nodes:
            store.edges = []
    else:
        if _env_truthy("OBJECT_FULL_REFRESH"):
            store.nodes = []
            progress_fn("OBJECT_FULL_REFRESH: re-extracting all objects.")
            chunks_to_process = store.chunks
        else:
            chunks_to_process = new_chunks
        if chunks_to_process:
            extract_and_merge_objects(engine, store, chunks_to_process, progress_fn)
        else:
            progress_fn("No new chunks; keeping existing objects.")
        graphStore.recomputed_node_centroids(store)
        raw_cos = (os.environ.get("OBJECT_ASSIGN_MIN_COSINE") or "0.45").strip()
        try:
            min_cosine = max(0.0, min(1.0, float(raw_cos)))
        except ValueError:
            min_cosine = 0.45
        assign_chunks_to_objects(store, min_cosine, progress_fn)
        graphStore.recomputed_node_centroids(store)
        object_state.update_object_states(
            store,
            engine,
            progress_fn,
            new_chunk_uuids={c.uuid for c in new_chunks},
            full_refresh=_env_truthy("STATE_FULL_REFRESH"),
        )

    progress_fn("Object snapshot:")
    _print_object_snapshot(store)
    progress_fn("Building object–object edges…")
    topic_edges.build_object_edges(store, engine, progress_fn)
    _print_edges_summary(store)


def sync_instruments(store: graphStore.GraphStore, progress_fn=progress) -> None:
    progress_fn("Syncing instruments…")
    instrument_sync.sync_instruments_from_config(store, progress_fn)


def sync_polymarket(
    store: graphStore.GraphStore,
    engine: aiEngine.aiEngine,
    progress_fn=progress,
) -> None:
    progress_fn("Syncing Polymarket…")
    polymarket_sync.sync_polymarket_markets(store, engine, progress_fn)


def link_all_edges(
    store: graphStore.GraphStore,
    engine: aiEngine.aiEngine,
    progress_fn=progress,
) -> None:
    progress_fn("Linking edges…")
    instrument_edges.build_instrument_edges(
        store,
        engine,
        progress_fn,
        replace_all_instrument_edges=True,
    )
    market_edges.build_market_edges(store, engine, progress_fn)
    market_market_edges.build_market_market_edges(store, engine, progress_fn)


def save_store(store: graphStore.GraphStore, root: Path, progress_fn=progress) -> None:
    path = db_path(root)
    store.exportToFile(str(path))
    store.export_chunks_jsonl(chunks_jsonl_path(root))
    _write_index_manifest(store, root, path)
    try:
        rel = chunks_jsonl_path(root).relative_to(root)
    except ValueError:
        rel = chunks_jsonl_path(root)
    progress_fn(f"Saved database.pkl + {rel}")


def export_viz(root: Path | None = None, progress_fn=progress) -> None:
    progress_fn("Exporting viz…")
    from viz.export import main as export_viz_main

    export_viz_main()
    out = (root or project_root()) / "viz" / "graph.json"
    progress_fn(f"Exported → {out}")


def run_pipeline(
    *,
    index: bool = False,
    sync: bool = False,
    sync_instruments_only: bool = False,
    sync_polymarket_only: bool = False,
    link: bool = False,
    export: bool = False,
    root: Path | None = None,
    progress_fn=progress,
) -> None:
    """Run selected pipeline phases. Saves store after mutating phases."""
    root = root or project_root()
    load_dotenv(root / ".env")
    pipeline_start = time.perf_counter()
    phase_times: list[tuple[str, float]] = []
    progress_fn("Pipeline starting…")

    store = load_store(root)
    engine: aiEngine.aiEngine | None = None
    mutated = False

    def _engine() -> aiEngine.aiEngine:
        nonlocal engine
        if engine is None:
            engine = build_engine()
        return engine

    if index:
        with _PhaseTimer("Index documents", progress_fn) as timer:
            _engine()
            index_documents(store, engine, root, progress_fn)
            mutated = True
        phase_times.append((timer.label, timer.elapsed))

    do_inst = sync or sync_instruments_only
    do_poly = sync or sync_polymarket_only

    if do_inst:
        if _env_enabled("PIPELINE_INSTRUMENTS"):
            with _PhaseTimer("Sync instruments", progress_fn) as timer:
                sync_instruments(store, progress_fn)
                mutated = True
            phase_times.append((timer.label, timer.elapsed))
        else:
            progress_fn("(Skipping instruments — PIPELINE_INSTRUMENTS=0)")

    if do_poly:
        if _env_enabled("PIPELINE_POLYMARKET"):
            with _PhaseTimer("Sync Polymarket", progress_fn) as timer:
                sync_polymarket(store, _engine(), progress_fn)
                mutated = True
            phase_times.append((timer.label, timer.elapsed))
        else:
            progress_fn("(Skipping Polymarket — PIPELINE_POLYMARKET=0)")

    if link:
        if _env_enabled("PIPELINE_LINK"):
            with _PhaseTimer("Link edges", progress_fn) as timer:
                link_all_edges(store, _engine(), progress_fn)
                mutated = True
            phase_times.append((timer.label, timer.elapsed))
        else:
            progress_fn("(Skipping link — PIPELINE_LINK=0)")

    if mutated:
        with _PhaseTimer("Save store", progress_fn) as timer:
            save_store(store, root, progress_fn)
        phase_times.append((timer.label, timer.elapsed))

    if export:
        if _env_enabled("PIPELINE_EXPORT"):
            with _PhaseTimer("Export viz", progress_fn) as timer:
                export_viz(root, progress_fn)
            phase_times.append((timer.label, timer.elapsed))
        else:
            progress_fn("(Skipping export — PIPELINE_EXPORT=0)")

    total = time.perf_counter() - pipeline_start
    progress_fn("")
    progress_fn("=== Stopwatch ===")
    for label, elapsed in phase_times:
        progress_fn(f"  {label}: {format_duration(elapsed)}")
    progress_fn(f"  Total: {format_duration(total)}")
    progress_fn("Pipeline complete.")
