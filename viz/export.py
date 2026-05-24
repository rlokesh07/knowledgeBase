"""Export database.pkl → graph.json for the web visualizer."""

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import graphStore  # noqa: E402 — must come after sys.path patch

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "database.pkl"
OUT_PATH = Path(__file__).resolve().parent / "graph.json"


def main() -> None:
    if not DB_PATH.exists():
        print(f"database.pkl not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(DB_PATH, "rb") as f:
        data = pickle.load(f)

    raw_nodes = data.get("nodes", [])
    raw_edges = data.get("edges", [])

    node_uuids = {n.uuid for n in raw_nodes}

    nodes = []
    for n in raw_nodes:
        nodes.append({
            "uuid": n.uuid,
            "label": n.label,
            "object_type": getattr(n, "object_type", "") or "other",
            "description": getattr(n, "description", "") or "",
            "state": getattr(n, "state", "") or "",
            "chunk_count": len(n.chunk_uuids),
            "tags": list(getattr(n, "tags", None) or []),
            "disabled": bool(getattr(n, "disabled", False)),
        })

    edges = []
    for e in raw_edges:
        if e.uuid1 not in node_uuids or e.uuid2 not in node_uuids:
            continue
        edges.append({
            "uuid": e.uuid,
            "source": e.uuid1,
            "target": e.uuid2,
            "relationship_type": getattr(e, "relationship_type", "") or "",
            "relationship": e.relationship or "",
            "mechanism": getattr(e, "mechanism", "") or "",
            "weight": round(float(e.weight), 4),
        })

    out = {"nodes": nodes, "edges": edges}
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(nodes)} node(s) and {len(edges)} edge(s) → {OUT_PATH}")


if __name__ == "__main__":
    main()
