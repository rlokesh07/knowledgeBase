import json
import pickle
from pathlib import Path


class Chunk:
    def __init__(self, uuid: str, text: str, embedding: list[float], source_path: str):
        self.uuid = uuid
        self.text = text
        self.embedding = embedding
        self.source_path = source_path


class Node:
    """A distinct object/entity node in the knowledge graph."""

    def __init__(
        self,
        uuid: str,
        label: str,
        chunk_uuids: list[str],
        cluster_id: int = -1,
        centroid: list[float] | None = None,
        object_type: str = "",
        description: str = "",
        state: str = "",
        properties: dict | None = None,
        tags: list[str] | None = None,
        disabled: bool = False,
    ):
        self.uuid = uuid
        self.label = label
        self.chunk_uuids = chunk_uuids
        self.cluster_id = cluster_id
        self.centroid = centroid
        self.object_type = object_type
        self.description = description
        self.state = state
        self.properties = properties or {}
        self.tags = tags or []
        self.disabled = disabled


class Edge:
    """Directed typed link between two object nodes (uuid1 → uuid2)."""

    def __init__(
        self,
        uuid: str,
        uuid1: str,
        uuid2: str,
        relationship: str,
        weight: float = 1.0,
        relationship_type: str = "",
        mechanism: str = "",
    ):
        self.uuid = uuid
        self.uuid1 = uuid1
        self.uuid2 = uuid2
        self.relationship = relationship
        self.weight = weight
        self.relationship_type = relationship_type
        self.mechanism = mechanism


class GraphStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []

    def export(self):
        return {"chunks": self.chunks, "nodes": self.nodes, "edges": self.edges}

    def exportToFile(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self.export(), f)

    def loadFromFile(self, filename: str) -> None:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        try:
            self.chunks = data["chunks"]
            self.nodes = data["nodes"]
        except KeyError as exc:
            missing = exc.args[0]
            raise SystemExit(
                f"database.pkl is missing {missing!r}. "
                "Remove the file and re-run indexing (old graph format is not supported)."
            ) from exc
        self.edges = data.get("edges", [])
        # Back-fill new Edge fields that may be absent in older pickled objects.
        for e in self.edges:
            if not hasattr(e, "relationship_type"):
                e.relationship_type = ""
            if not hasattr(e, "mechanism"):
                e.mechanism = ""
        # Back-fill new Node fields that may be absent in older pickled objects.
        for n in self.nodes:
            if not hasattr(n, "object_type"):
                n.object_type = ""
            if not hasattr(n, "description"):
                n.description = ""
            if not hasattr(n, "state"):
                n.state = ""
            if not hasattr(n, "properties"):
                n.properties = {}
            if not hasattr(n, "tags"):
                n.tags = []
            if not hasattr(n, "disabled"):
                n.disabled = False
        try:
            import node_tags

            node_tags.sync_disabled_state(self)
        except ImportError:
            pass

    def chunk_cluster_ids(self) -> dict[str, int]:
        """Map chunk uuid → cluster id (topics only — unassigned chunks are omitted)."""
        out: dict[str, int] = {}
        for n in self.nodes:
            for uid in n.chunk_uuids:
                out[uid] = n.cluster_id
        return out

    def export_chunks_jsonl(self, path: Path) -> None:
        """Dense JSONL snapshot of chunks (text + embeddings) for redundancy / tooling."""
        path.parent.mkdir(parents=True, exist_ok=True)
        cluster_ids = self.chunk_cluster_ids()
        with path.open("w", encoding="utf-8") as f:
            for c in self.chunks:
                row = {
                    "uuid": c.uuid,
                    "source_path": c.source_path,
                    "text": c.text,
                    "embedding": c.embedding,
                    "cluster_id": cluster_ids.get(c.uuid),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def recomputed_node_centroids(store: GraphStore) -> None:
    """Set each node's centroid to the mean embedding of its member chunks (or None)."""
    by_uuid = {c.uuid: c for c in store.chunks}
    for n in store.nodes:
        mats = []
        for uid in n.chunk_uuids:
            chunk = by_uuid.get(uid)
            if chunk:
                mats.append(chunk.embedding)
        if not mats:
            n.centroid = None
            continue
        dim = len(mats[0])
        tot = [0.0] * dim
        for row in mats:
            tot = [t + float(x) for t, x in zip(tot, row)]
        n.centroid = [t / len(mats) for t in tot]
