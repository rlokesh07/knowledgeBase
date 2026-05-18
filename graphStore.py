import pickle

class Node:
    def __init__(self, uuid, summary, layer, link):
        self.uuid = uuid
        self.summary = summary
        self.layer = layer
        self.link = link

class Edge:
    def __init__(self, uuid, uuid1, uuid2, weight, relationship, layer):
        self.uuid = uuid
        self.uuid1 = uuid1 # uuid for the first node connected
        self.uuid2 = uuid2 # uuid for the second node connected
        self.weight = weight 
        self.relationship = relationship
        self.layer = layer
    def __repr__(self):
        return f"""uuid: {self.uuid}, nodes:{self.uuid1} & {self.uuid2}, Weight: {self.weight}
    relationship: {self.relationship}"""

    def setLayer(self, layer):
        self.layer = layer


class GraphStore:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def addNode(self, node):
        self.nodes.append(node)

    def addEdge(self, edge):
        self.edges.append(edge)

    def copyNodes(self):
        return self.nodes.copy()

    def getSize(self):
        return len(self.nodes)

    def getNodesByLayer(self, layer):
        sameLayerNodes = []
        for node in self.nodes:
            if node.layer == layer:
                sameLayerNodes.append(node)

        return sameLayerNodes

    def getNodeByUUID(self, uuid):
        for node in self.nodes:
            if node.uuid == uuid:
                return node

        return None

    def getNeighbors(self, node):
        neighborNodes = []
        for edge in self.edges:
            if edge.uuid1 == node.uuid:
                neighborNodes.append(self.getNodeByUUID(edge.uuid2))
            elif edge.uuid2 == node.uuid:
                neighborNodes.append(self.getNodeByUUID(edge.uuid1))
        return neighborNodes

    def updateEdgeWeight(self, edge, newWeight):
        edge.weight = newWeight

    def getEdge(self, node1, node2):
        for edge in self.edges:
            if {edge.uuid1, edge.uuid2} == {node1.uuid, node2.uuid}:
                return edge
        return None

    def exportToFile(self, filename):
        with open(filename, "wb") as f:
            pickle.dump({"nodes": self.nodes, "edges": self.edges}, f)
    def export(self):
        return {"nodes": self.nodes, "edges": self.edges}
    def loadFromFile(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.nodes = data["nodes"]
            self.edges = data["edges"]
