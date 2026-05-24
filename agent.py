"""Knowledge-graph agent — predict mode with ephemeral state propagation."""

import argparse
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

import graphStore
import pipeline
import state_propagation


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _question_keywords(question: str) -> set[str]:
    words: set[str] = set()
    for token in question.lower().replace("/", " ").replace("-", " ").split():
        cleaned = "".join(ch for ch in token if ch.isalnum())
        if len(cleaned) >= 3:
            words.add(cleaned)
    return words


def _find_candidate_nodes(
    store: graphStore.GraphStore,
    engine,
    question: str,
) -> list[graphStore.Node]:
    max_candidates = _env_int("PREDICT_SEED_CANDIDATES", 20)
    keywords = _question_keywords(question)
    active = [n for n in store.nodes if not getattr(n, "disabled", False)]
    if not active:
        return []

    scored: dict[str, float] = {}
    node_by_uuid = {n.uuid: n for n in active}

    if store.chunks:
        q_emb = np.asarray(engine.embed(question), dtype=np.float64)
        q_norm = float(np.linalg.norm(q_emb))
        if q_norm >= 1e-12:
            chunk_scores: dict[str, float] = {}
            for chunk in store.chunks:
                emb = np.asarray(chunk.embedding, dtype=np.float64)
                norm = float(np.linalg.norm(emb))
                if norm < 1e-12:
                    continue
                chunk_scores[chunk.uuid] = float(q_emb @ emb / (q_norm * norm))

            for node in active:
                if not node.chunk_uuids:
                    continue
                sims = [chunk_scores[uid] for uid in node.chunk_uuids if uid in chunk_scores]
                if sims:
                    scored[node.uuid] = max(sims)

    for node in active:
        otype = getattr(node, "object_type", "") or ""
        if otype not in {"market", "instrument"}:
            continue
        label = (node.label or "").lower()
        symbol = str((node.properties or {}).get("symbol") or "").lower()
        haystack = f"{label} {symbol} {(node.description or '').lower()}"
        if any(kw in haystack for kw in keywords):
            scored[node.uuid] = max(scored.get(node.uuid, 0.0), 0.85)

    for node in active:
        label = (node.label or "").lower()
        if any(kw in label for kw in keywords):
            scored[node.uuid] = max(scored.get(node.uuid, 0.0), 0.75)

    if not scored:
        ranked = sorted(active, key=lambda n: len(n.chunk_uuids), reverse=True)
        return ranked[:max_candidates]

    ranked_uuids = sorted(scored.keys(), key=lambda uid: scored[uid], reverse=True)
    return [node_by_uuid[uid] for uid in ranked_uuids[:max_candidates]]


def run_agent(question: str) -> str:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")

    db_path = project_root / "database.pkl"
    store = graphStore.GraphStore()
    if db_path.exists():
        store.loadFromFile(str(db_path))
    else:
        raise SystemExit("database.pkl not found — run `python main.py` first.")

    engine = pipeline.build_engine()

    pipeline.progress(f"Question: {question}")
    candidates = _find_candidate_nodes(store, engine, question)
    pipeline.progress(f"Predict mode: {len(candidates)} candidate node(s) for seed selection.")

    result = state_propagation.run_propagation(
        store,
        engine,
        question,
        candidates,
        pipeline.progress,
    )
    if result is None:
        return "Could not run predict-mode propagation (no seed or content filter)."

    return result.answer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Knowledge graph agent — predict mode with state propagation"
    )
    parser.add_argument("question", help="Question or scenario to simulate")
    args = parser.parse_args()

    answer = run_agent(args.question)
    print("\n=== Predict Mode Report ===\n")
    print(answer)


if __name__ == "__main__":
    main()
