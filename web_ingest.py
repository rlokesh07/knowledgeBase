"""Ingest validated web content into the knowledge graph."""

import os
import uuid

import graphStore
import text_chunking
import aiEngine


def _indexed_urls(store: graphStore.GraphStore) -> set[str]:
    return {(c.source_path or "").strip() for c in store.chunks if (c.source_path or "").strip()}


def ingest_web_document(
    store: graphStore.GraphStore,
    engine: aiEngine.aiEngine,
    *,
    url: str,
    title: str,
    markdown: str,
    progress,
    extract_objects_fn,
    assign_chunks_fn,
) -> list[graphStore.Chunk]:
    """Chunk, embed, and merge a validated web page into the store. Returns new chunks."""
    url = (url or "").strip()
    if not url:
        return []

    if url in _indexed_urls(store):
        progress(f"  Already in graph: {url}")
        return []

    body = (markdown or "").strip()
    if not body:
        progress(f"  No content to ingest for {url}")
        return []

    header = f"# {title}\n\nSource: {url}\n\n" if title else f"Source: {url}\n\n"
    full_text = header + body

    max_chars = int(os.environ.get("TEXT_CHUNK_MAX_CHARS") or 900)
    overlap = int(os.environ.get("TEXT_CHUNK_OVERLAP") or 120)
    texts = text_chunking.split_pages_into_chunks([full_text], max_chars, overlap)
    if not texts:
        return []

    embeddings = engine.embed_batch(texts)
    if len(embeddings) != len(texts):
        raise RuntimeError("Embedding batch size mismatch during web ingest.")

    new_chunks: list[graphStore.Chunk] = []
    for piece, vector in zip(texts, embeddings):
        chunk = graphStore.Chunk(str(uuid.uuid4()), piece, vector, url)
        store.chunks.append(chunk)
        new_chunks.append(chunk)

    progress(f"  Ingested {len(new_chunks)} chunk(s) from {url}")

    extract_objects_fn(engine, store, new_chunks, progress)
    graphStore.recomputed_node_centroids(store)

    raw_cos = (os.environ.get("OBJECT_ASSIGN_MIN_COSINE") or "0.45").strip()
    try:
        min_cosine = max(0.0, min(1.0, float(raw_cos)))
    except ValueError:
        min_cosine = 0.45
    assign_chunks_fn(store, min_cosine, progress)
    graphStore.recomputed_node_centroids(store)

    return new_chunks
