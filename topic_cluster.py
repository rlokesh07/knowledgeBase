"""Global K-means on chunk embeddings."""

import os

import numpy as np
from sklearn.cluster import KMeans


def effective_k(n_chunks: int, env_override: str | None = None) -> int:
    if n_chunks <= 0:
        return 0
    raw = (env_override or "").strip()
    if raw:
        try:
            k = int(raw)
            return max(1, min(k, n_chunks))
        except ValueError:
            pass
    # No fixed upper cap — large corpora get more clusters (still ≤ n_chunks).
    k = max(2, n_chunks // 40)
    return max(1, min(k, n_chunks))


def cluster_labels(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Return integer cluster id per row (shape (n_samples,))."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(embeddings)


def best_cluster_above_cos_threshold(
    embedding: np.ndarray,
    centroids: np.ndarray,
    min_cosine: float,
) -> int | None:
    """Index of centroid row with highest cosine sim to ``embedding``, if ``>= min_cosine``.

    Rows of ``centroids`` need not be unit-normalized. Returns ``None`` if no centroid qualifies
    (including degenerate embeddings or ``centroids`` shape (0, d)).
    """
    e = np.asarray(embedding, dtype=np.float64).ravel()
    ne = float(np.linalg.norm(e))
    if ne <= 1e-12:
        return None
    en = e / ne
    c = np.atleast_2d(np.asarray(centroids, dtype=np.float64))
    if c.shape[0] == 0:
        return None
    norms = np.linalg.norm(c, axis=1)
    norms = np.maximum(norms, 1e-12).reshape(-1, 1)
    cn = c / norms
    sims = cn @ en
    i = int(np.argmax(sims))
    return i if float(sims[i]) >= min_cosine else None


def default_chunk_window_env() -> tuple[int, int]:
    """(max_chars, overlap) from env TEXT_CHUNK_MAX_CHARS / TEXT_CHUNK_OVERLAP."""
    max_c = int(os.environ.get("TEXT_CHUNK_MAX_CHARS") or 900)
    overlap = int(os.environ.get("TEXT_CHUNK_OVERLAP") or 120)
    return max_c, overlap
