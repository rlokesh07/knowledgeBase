import graphStore
import aiEngine

class Brain:
    def __init__(self, graphStore, aiEngine, maxLayer, weightFunction=None):
        self.graphStore = graphStore
        self.aiEngine = aiEngine
        self.maxLayer = maxLayer 
        if weightFunction is None:
            self.weightFunction = lambda current, similarity: current + similarity * 0.1
        else:
            self.weightFunction = weightFunction

    def print(self):
        data = self.graphStore.export()
        layer_nodes: dict[int, list] = {}
        layer_edges: dict[int, list] = {}

        nodes = data["nodes"]
        edges = data["edges"]

        for node in nodes:
            layer_nodes.setdefault(node.layer, []).append(node)

        for edge in edges:
            layer_edges.setdefault(edge.layer, []).append(edge)

        print(
            f"\n=== Knowledge graph ({len(nodes)} nodes, {len(edges)} edges) ===",
            flush=True,
        )

        def clip(text: str, max_len: int) -> str:
            t = (text or "").replace("\n", " ").strip()
            if len(t) > max_len:
                return t[: max_len - 1] + "..."
            return t

        by_uuid = {n.uuid: n for n in nodes}

        for layer in sorted(set(layer_nodes) | set(layer_edges)):
            print(f"\n--- Layer {layer} ---", flush=True)
            ln = sorted(layer_nodes.get(layer, []), key=lambda n: n.uuid)
            le = sorted(layer_edges.get(layer, []), key=lambda e: (e.uuid1, e.uuid2))

            if ln:
                print("  Nodes:", flush=True)
                for node in ln:
                    sid = node.uuid[:8]
                    print(f"    [{sid}]  {node.link}", flush=True)
                    print(f"        {clip(node.summary, 90)}", flush=True)

            if le:
                print("  Connections:", flush=True)
                for edge in le:
                    a = edge.uuid1[:8]
                    b = edge.uuid2[:8]
                    rel = clip(edge.relationship, 64)
                    w = edge.weight
                    w_txt = f"{w:.4g}" if isinstance(w, float) else str(w)
                    print(
                        f"    [{a}] -- {rel} (weight {w_txt}) --> [{b}]",
                        flush=True,
                    )
                    n1 = by_uuid.get(edge.uuid1)
                    n2 = by_uuid.get(edge.uuid2)
                    if n1 or n2:
                        s1 = clip(n1.summary, 42) if n1 else "(missing node)"
                        s2 = clip(n2.summary, 42) if n2 else "(missing node)"
                        print(f"        |  {s1}", flush=True)
                        print(f"        +> {s2}", flush=True)

    def addKnowledge(self, node, newRelationships, layer):
        # Exclude the node being linked — it is already in the store and would cause self-edges.
        nodes = [
            n
            for n in self.graphStore.getNodesByLayer(layer)
            if n.uuid != node.uuid
        ]
        for edgeNumber in range(newRelationships):
            if(len(nodes) == 0): break
            newEdge = self.aiEngine.newEdge(node, nodes)
            newEdge.setLayer(layer)

            self.graphStore.addEdge(newEdge)

            nodes[:] = [n for n in nodes if n.uuid != newEdge.uuid2]

