import queue
import string
from itertools import count
import copy

INPUT_FILE = "cor3.net"

class Vertex():
    """
    vertex_example: {
        "a": Vertex(),
        "b": Vertex()
    }
    """
    def __init__(self, label) -> None:
        self.label = label                      # Label of vertex
        self.neighbors = []                     # List neighbors
        self.outgoingNeighbors = []             # List of outgoing neighbors N(u)+
        self.incomingNeighbors = []             # List of incoming neighbors N(u)-
        self.isOrigin = False                   # Is the vertex origin?
        self.BeenVisited = False                # Has the vertex been visited?
        self.distance = 2147483647              # Distance from origin
        self.ancestor = None                    # Ancestor - used in tree-like structures
        self.time = 2147483647                  # Time counter
        self.visit_begin_time = 2147483647      # Begin of visit counter
        self.visit_end_time = 2147483647        # End of visit counter
        self.k = 2147483647                     # Key value for later to be used in key structure
        self.beenSetInRG = False                # Has the vertex been set in the Residual Graph (RG)?
        self.mate = None                        # vertex mate
        self.isTheNullVertex = False            # is the vertex the Null vertex used in Hopcroft-Karp?

    def getDegree(self) -> int:
        return len(self.getNeighbors())

    def getLabel(self) -> string:
        return self.label

    def getNeighbors(self) -> list:
        return self.neighbors
    
    def getOutgoingNeighbors(self) -> list:
        return self.outgoingNeighbors
    
    def getIncomingNeighbors(self) -> list:
        return self.incomingNeighbors

    def isVertexOrigin(self) -> bool:
        return self.isOrigin

    def hasBeenVisited(self) -> bool:
        return self.BeenVisited

    def getDistance(self) -> int:
        return self.distance
    
    def getAncestor(self) -> string:
        return self.ancestor

    def getTime(self) -> int:
        return self.time
    
    def getVisitBeginTime(self) -> int:
        return self.visit_begin_time
    
    def getVisitEndTime(self) -> int:
        return self.visit_end_time
    
    def getK(self) -> int:
        return self.k
    
    def getMate(self) -> string:
        return self.mate
    
    def hasBeenSetInRG(self) -> bool:
        return self.beenSetInRG
    
    def getIsTheNullVertex(self) -> bool:
        return self.isTheNullVertex
    
    def setLabel(self, label) -> None:
        self.label = label
    
    def appendNeighbor(self, neighbor) -> None:
        self.neighbors.append(neighbor)
    
    def eraseNeighbors(self) -> None:
        self.neighbors = list()

    def appendOutgoingNeighbor(self, neighbor) -> None:
        self.outgoingNeighbors.append(neighbor)
    
    def setWholeOutgoingNeighbors(self, outgoingNeighbors) -> None:
        self.outgoingNeighbors = outgoingNeighbors
    
    def appendIncomingNeighbor(self, neighbor) -> None:
        self.incomingNeighbors.append(neighbor)
    
    def setWholeIncomingNeighbors(self, incomingNeighbors) -> None:
        self.incomingNeighbors = incomingNeighbors

    def setAsOrigin(self) -> None:
        self.isOrigin = True

    def setAsVisited(self) -> None:
        self.BeenVisited = True
    
    def setDistance(self, distance) -> None:
        self.distance = distance

    def setAncestor(self, ancestor) -> None:
        self.ancestor = ancestor
    
    def setTime(self, time) -> None:
        self.time = time

    def setVisitBeginTime(self, time) -> None:
        self.visit_begin_time = time
    
    def setVisitEndTime(self, time) -> None:
        self.visit_end_time = time
    
    def setK(self, k) -> None:
        self.k = k
    
    def setAsSetInRG(self) -> None:
        self.beenSetInRG = True
    
    def setMate(self, mate) -> None:
        self.mate = mate
    
    def setAsTheNullVertex(self) -> None:
        self.isTheNullVertex = True
    
    def getAllInfos(self) -> dict:
        return {
            "label": self.getLabel(),
            "neighbors": self.getNeighbors(),
            "incomingNeighbors": self.getIncomingNeighbors(),
            "outgoingNeighbors": self.getOutgoingNeighbors(),
            "isOrigin": self.isVertexOrigin(),
            "beenVisited": self.hasBeenVisited(),
            "distance": self.getDistance(),
            "ancestor": self.getAncestor().getLabel() if self.getAncestor() != None else self.getAncestor(),
            "time": self.getTime(),
            "visit begin time": self.getVisitBeginTime(),
            "visit end time": self.getVisitEndTime(),
            "k": self.getK()
        }

class Edge():
    """
    edge_example: {
        "ab": Edge(),
        "ac": Edge()
    }
    """
    def __init__(self, vertex_A, vertex_B, weight) -> None:
        self.label = (vertex_A+vertex_B)   # label of the edge "labelOfVertex1LabelOfVertex2"
        self.source = vertex_A             # source vertex
        self.target = vertex_B             # target vertex
        self.weight = weight               # weight of the edge
        self.flow = 0                      # flow of the edge

    def getWeight(self) -> float:
        return self.weight
    
    def getLabel(self) -> string:
        return self.label
    
    def getSource(self) -> string:
        return self.source

    def getTarget(self) -> string:
        return self.target
    
    def getFlow(self) -> int:
        return self.flow

    def setLabel(self, label) -> None:
        self.label = label
    
    def setWeight(self, weight) -> None:
        self.weight = weight
    
    def setSource(self,source) -> None:
        self.source = source

    def setTarget(self,target) -> None:
        self.target = target
    
    def setFlow(self, flow) -> None:
        self.flow = flow

    def getAllInfos(self) -> dict:
        return {
            "label": self.getLabel(),
            "source": self.getSource(),
            "target": self.getTarget(),
            "weight": self.getWeight(),
            "flow": self.getFlow()
        }


class Graph:
    """
    Main class with two main attributes: edges and vertices.
    """
    def __init__(self) -> None:
        self.raw_vertices = []                      # A list of raw vertices ['1 rotulo_de_1', '2 rotulo_de_2']
        self.raw_edges = []                         # A list of raw edges ['a b valor_do_peso', 'a c valor_do_peso']
        self.vertices = dict()                      # A dictionary in which the key is the vertex label and the value is an object Vertex
        self.edges = dict()                         # A dictionary in which the key is the edge label and the value is an object Edge
        self.biggestNeighborQuantity = 0            # Biggest neighbor quantity
        self.visited = []                           # A list of all visited vertices. Not 100% necessary, but created to facilitate Dijkstra algorithm
        self.directed = False                       # Is the graph directed?
        self.time = 2147483647                      # Time counter for CFC algorithm
        self.verticesOrderedByVisitTime = list()    # A list of all vertices ordered by visit time ascending 
        self.o = list()                             # The final list of vertices obeying the topological order
        self.s = list()                             # Unused - to be removed TODO: remove
        self.edgesOrderedByWeight = list()          # A list with all edges ordered by weight ascending
        self.flow = 2147483647                      # Flow
        self.p = list()                             # List of increasing paths - used for Edmonds-karp
        self.X = dict()                             # Indexed dict - used for coloring vertices
        self.XVertices = list()                     # List of vertices for paring algorithm
        self.YVertices = list()                     # List of vertices for paring algorithm
        self.availableColors = list()               
        self.usedColors = list()
        self.vertexColorMap = dict()

    def build(self):
        """
        Main method.
        """
        self.readFile(INPUT_FILE)
        self.buildEdgesAndVertices()
        self.checkRemainingVertices()

    def prettyPrint(self) -> None:
        """
        Completely optional method, just to prettify the graph and print it.
        """
        print("#VERTICES#")
        for vertex in self.getVertices():
            print(vertex,self.getVertices()[vertex].getAllInfos())

        print("#EDGES#")
        for edge in self.getEdges():
            print(edge, self.getEdges()[edge].getAllInfos())

            

    def buildEdgesAndVertices(self) -> None:
        """
        Builds vertices and edges based solely on the *edges section of the input file.
        If graph is directed, it sets incoming and outgoing neighbors instead of just neighbors.
        Calulates the biggest quantity of neighbors so it can be used later in BFS algorithm.
        """
        for edge in self.getRawEdges():
            source = Vertex(self.getRawVertices()[int(edge[0])-1][2:]) if (self.getRawVertices()[int(edge[0])-1][2:] not in self.getVertices()) else self.getVertices()[self.getRawVertices()[int(edge[0])-1][2:]]
            target = Vertex(self.getRawVertices()[int(edge[2])-1][2:]) if (self.getRawVertices()[int(edge[2])-1][2:] not in self.getVertices()) else self.getVertices()[self.getRawVertices()[int(edge[2])-1][2:]]
            weight = None if edge[4] == "" else float(edge[4:])

            self.vertices[source.getLabel()] = source
            self.vertices[target.getLabel()] = target
            self.edges[source.getLabel()+target.getLabel()] = Edge(source.getLabel(), target.getLabel(), weight)

            if (self.isDirected()):
                source.appendOutgoingNeighbor(target.getLabel())
                target.appendIncomingNeighbor(source.getLabel())
            else:
                source.appendNeighbor(target.getLabel())
                target.appendNeighbor(source.getLabel())
            
        # for vertexLabel in self.getVertices().keys():
        #     QuantityOfNeighbors = len(self.getVertices()[vertexLabel].getNeighbors())
        #     if  QuantityOfNeighbors > self.biggestNeighborQuantity:
        #         self.biggestNeighborQuantity = QuantityOfNeighbors
        
        # for key in self.getVertices().keys():
        #     print(key + " outgoing: ", self.getVertices()[key].getOutgoingNeighbors())
        #     print(key + " incoming: ", self.getVertices()[key].getIncomingNeighbors())
        #     print("")
        #     #print(key, self.getVertices()[key].getNeighbors())

    
    def checkRemainingVertices(self) -> None:
        """
        If there's a vertex under the *vertices section that it's not under the *edges section,
        it creates it.
        """
        vertices_labels = self.getVertices().keys()
        for vertex in self.getRawVertices():
            if vertex[2:] not in vertices_labels:
                self.getVertices()[vertex[2:]] = Vertex(vertex[2:])

    
    def printVertices(self) -> None:
        print(self.getVertices())

    def printEdges(self) -> None:
        print(self.getEdges())

    def getVertices(self) -> dict:
        return self.vertices
    
    def getEdges(self) -> dict:
        return self.edges
    
    def getRawEdges(self) -> list:
        return self.raw_edges

    def getRawVertices(self) -> list:
        return self.raw_vertices

    def getBiggestNeighborQuantity(self) -> int:
        return self.biggestNeighborQuantity

    def getVisitedVertices(self) -> list:
        return self.visited

    def getNotVisitedVertices(self) -> list:
        notVisited = []
        for vertex in self.getVertices():
            if not vertex.hasBeenVisited():
                notVisited.append(vertex)
        return notVisited
    
    def isDirected(self) -> bool:
        return self.directed
    
    def getTime(self) -> int:
        return self.time

    def getVerticesOrderedByVisitTime(self) -> list:
        return self.verticesOrderedByVisitTime
    
    def getO(self) -> list:
        return self.o
    
    def getS(self) -> list:
        return self.s
    
    def getFlow(self) -> int:
        return self.flow
    
    def getP(self) -> list:
        return self.p

    def getX(self) -> dict:
        return self.X
    
    def getAvailableColors(self) -> list:
        return self.availableColors
    
    def getUsedColors(self) -> list:
        return self.usedColors
    
    def getVertexColorMap(self) -> dict:
        return self.vertexColorMap
    
    def getTheNullVertex(self) -> string:
        for vertexLabel in self.getVertices().keys():
            if (self.getVertices()[vertexLabel].getIstheNullVertex() == True):
                return vertexLabel
        return ''
    
    def getXVertices(self) -> list:
        return self.XVertices
    
    def getYVertices(self) -> list:
        return self.YVertices

    def setRawEdges(self, raw_edges) -> None:
        self.raw_edges = raw_edges

    def setRawVertices(self, raw_vertices) -> None:
        self.raw_vertices = raw_vertices

    def setAsDirected(self) -> None:
        self.directed = True
    
    def setTime(self, time) -> None:
        self.time = time

    def setFlow(self, flow) -> None:
        self.flow = flow
    
    def setX(self, quantity_vertices) -> None:
        for i in range(2**(quantity_vertices-1)):
            self.X[i] = {}
    
    def setAvailableColors(self, quantity_vertices):
        self.availableColors = [True] * quantity_vertices
    
    def eraseEdges(self) -> None:
        self.edges = dict()
    
    def eraseNeighborsOfAllVertices(self) -> None:
        for vertex_label in self.getVertices():
            vertex = self.getVertices()[vertex_label]
            vertex.eraseNeighbors()
            vertex.setWholeOutgoingNeighbors(list())
            vertex.setWholeIncomingNeighbors(list())
    
    def insertToVerticesOrderedByVisitTime(self, vertex) -> None:
        self.verticesOrderedByVisitTime.insert(0, vertex)
    
    def insertToO(self, vertex) -> None:
        self.o.insert(0, vertex)
    
    def appendToS(self, vertex) -> None:
        self.s.append(vertex)
    
    def appendToP(self, vertex) -> None:
        self.p.append(vertex)
    
    def insertToP(self, vertex) -> None:
        self.p.insert(0, vertex)
    
    def appendToUsedColors(self, color):
        self.usedColors.append(color)
    
    def appendNewEdge(self, edge) -> None:
        source_label = edge.getSource()
        target_label = edge.getTarget()
        key = source_label+target_label
        self.getEdges()[key] = edge
    
    def appendNewVertex(self, vertex) -> None:
        label = vertex.getLabel()
        self.getVertices()[label] = vertex

    def readFile(self, file) -> None:
        """
        Reads input and sets variables raw_vertices and raw_edges.
        """
        f = open(file, "r")
        f_content = f.readlines()
        f.close()

        numberOfVertices = int(f_content[0][10:])
        raw_vertices = []
        raw_edges = []

        # get raw_vertices
        for line in range(1,numberOfVertices+1):
            vertex = (f_content[line]).split("\n")[0]
            raw_vertices.append(vertex)

        # check if graph is directed or not
        isDirected = f_content[numberOfVertices+1].split("\n")[0] == '*arcs'
        if isDirected:
            self.setAsDirected()

        # get raw_edge
        for line in range(numberOfVertices+2, len(f_content)):
            edge = (f_content[line]).split("\n")[0]
            raw_edges.append(edge)

        self.setRawVertices(raw_vertices)
        self.setRawEdges(raw_edges)

    ###########REQUESTED BASIC METHODS/OPERATIONS##################
    """
    This section aims to define all the trivial methods related to graphs.
    All these operations have O(1) complexity.
    """
    def qtdVertices(self) -> int:
        return len(self.getVertices())

    def qtdArestas(self) -> int:
        return len(self.getEdges())

    def grau(self, v) -> int:
        return self.getVertices()[v].getDegree()

    def rotulo(self, v) -> string:
        return self.getVertices()[v].getLabel()

    def vizinhos(self, v) -> list:
        return self.getVertices()[v].getNeighbors()

    def haAresta(self, u, v) -> bool:
        return u+v in self.getEdges()

    def peso(self, u, v) -> float:
        if self.haAresta(u,v):
            return self.getEdges()[u+v].getWeight()
        return 2147483647.0

    def ler(self, arquivo) -> None:
        self.readFile(arquivo)


class Algorithms():
    def __init__(self) -> None:
        pass
    
    # Busca em Largura
    def BFS(self, graph: Graph, originIndex) -> None:
        originVertexLabel = graph.getRawVertices()[originIndex-1][2]
        originVertex = graph.getVertices()[originVertexLabel]
        originVertex.setAsVisited()
        originVertex.setDistance(0)
        d = {k: [] for k in range(graph.getBiggestNeighborQuantity())} # Should I change to range(graph.qtdVertices())?
        level = 0
        q = queue.Queue()
        q.put(originVertex) # enqueue
        while not q.empty():
            u = q.get() # dequeue
            d[u.getDistance()].append(u.getLabel())
            level = level + 1
            for neighbor in u.getNeighbors():
                current_neighbor = graph.getVertices()[neighbor]
                if (current_neighbor.hasBeenVisited() == False):
                    current_neighbor.setAsVisited()
                    current_neighbor.setDistance(u.getDistance() + 1)
                    current_neighbor.setAncestor(u)
                    q.put(current_neighbor)
        for i in range(len(d.items())):
            print(str(i) + ": ", d[i])

    # Dijkstra
    def Dijkstra(self, graph: Graph, vertex_s_label) -> None:
        d = {k:2147483647 for k in graph.getVertices().keys()}
        vertex_s = graph.getVertices()[vertex_s_label]
        d[vertex_s_label] = 0

        unique = count()
        pq = queue.PriorityQueue()
        pq.put((0, unique ,vertex_s))

        while not pq.empty():
            (dist, unique_, current_vertex) = pq.get()
            graph.visited.append(current_vertex)
            for neighbor in current_vertex.getNeighbors():
                if graph.haAresta(current_vertex.getLabel(), neighbor):
                    distance = graph.peso(current_vertex.getLabel(), neighbor)
                    if graph.getVertices()[neighbor] not in graph.visited:
                        old_cost = d[graph.getVertices()[neighbor].getLabel()]
                        new_cost = d[current_vertex.getLabel()] + distance
                        if new_cost < old_cost:
                            pq.put((new_cost, next(unique), graph.getVertices()[neighbor]))
                            d[neighbor] = new_cost
        for key in d.keys():
            print(key + ":" + str(d[key]))
            
    # Floyd-Warhsall - This has an error.
    def FloydWarshall(self, graph: Graph):
        dist = {k: {l: {} for l in graph.getVertices().keys()} for k in graph.getVertices().keys()}
        dist1 = {k: {l: {} for l in graph.getVertices().keys()} for k in graph.getVertices().keys()}
        # Set D(0)
        for r in graph.getVertices().keys():
            for c in graph.getVertices().keys():
                if r==c:
                    dist[r][c] = 0
                elif ((r != c) and (graph.haAresta(r, c))):
                    dist[r][c] = graph.peso(r, c)
                elif ((r != c) and (not graph.haAresta(r, c))):
                    dist[r][c] = 2147483647
                else:
                    dist[r][c] = -1
        # algorithm itself
        for k in graph.getVertices().keys():
            for u in graph.getVertices().keys():
                for v in graph.getVertices().keys():
                    dist1[u][v] = min(dist[u][v], dist[u][k] + dist[k][v])

        print(dist1)
    
    # Ordenação Topológica
    def topologicalOrder(self, graph: Graph) -> None:
        # algorithm itself
        for vertex_label in graph.getVertices().keys():
            vertex = graph.getVertices()[vertex_label]
            if not vertex.hasBeenVisited():
                self.visitTO(graph, vertex)
        
        # formatting final print
        pretty_printed_o = list()
        for v in graph.getO():
            pretty_printed_o.append(v.getLabel())
            pretty_printed_o.append("->")
        print(' '.join(pretty_printed_o[:len(pretty_printed_o)-1]))
        

    def visitTO(self, graph: Graph, vertex) -> None:
        vertex.setAsVisited()
        for neighbor_label in vertex.getOutgoingNeighbors():
            neighbor = graph.getVertices()[neighbor_label]
            if not neighbor.hasBeenVisited():
                self.visitTO(graph, neighbor)
        graph.insertToO(vertex)

    # Prim
    def Prim(self, graph: Graph):
        r = graph.getVertices()[list(graph.getVertices().keys())[0]]
        r.setK(0)
        q = list()
        for vertex_label in graph.getVertices().keys():
            q.append(graph.getVertices()[vertex_label])
        q.sort(key=self.keySortCriteria)    # This costs a lot of power. I thought it would compensate later because we can always just pop the first item of the list, but it's not worth it.
        while not len(q) == 0:
            u = q[0]
            q.pop(0)
            for neighbor_label in (u.getNeighbors()):
                neighbor = graph.getVertices()[neighbor_label]
                if (neighbor in q and graph.peso(u.getLabel(), neighbor.getLabel()) < neighbor.getK()):
                    neighbor.setAncestor(u)
                    neighbor.setK(graph.peso(u.getLabel(), neighbor.getLabel()))

        # formatting final print
        self.formatedPrint(graph)

    # util method to sort the array by K
    def keySortCriteria(self, edge):
        return edge.getK()

    # util method to pretty print the Prim method
    def formatedPrint(self, graph: Graph):
        weightSum = 0
        vertexNumbers = dict()
        ancestors = list()
        for raw_vertex in graph.getRawVertices():
            vertexNumbers[(raw_vertex[0:2]).strip()] = raw_vertex[3:]
        for vertex_label in graph.getVertices().keys():
            vertex = graph.getVertices()[vertex_label]
            ancestor = vertex.getAncestor()
            if (not ancestor == None):
                weightSum = weightSum + graph.peso(ancestor.getLabel(),vertex.getLabel())
                ancestors.append((list(vertexNumbers.keys())[list(vertexNumbers.values()).index(vertex.getAncestor().getLabel())],list(vertexNumbers.keys())[list(vertexNumbers.values()).index(vertex_label )]))
            else:
                ancestors.append((list(vertexNumbers.keys())[list(vertexNumbers.values()).index(vertex_label)], None))
        print(weightSum)
        print(ancestors)

    # Edmonds-Karp
    def Edmonds_Karp(self, graph: Graph, s, t) -> None:
        residual_graph = self.createResidualGraph(graph)
        residual_graph.setFlow(0)
        #residual_graph.prettyPrint()
        while True:
            paths = self.findIncreasingPaths(residual_graph, s, t)
            if (paths):
                paths2 = list()
                paths2.append(paths[0].getLabel())
                for i in range(1,len(paths)):
                    if (i < len(paths)):
                        paths2.append(paths[i-1].getLabel()+paths[i].getLabel())
                pathsCapacity = self.getMinimalCapacity(residual_graph, paths2)         # get the smallest weight
                for edge_label in paths2:
                    if len(edge_label) > 1 and edge_label in graph.getEdges().keys():
                        edge = residual_graph.getEdges()[edge_label]
                        edge.setWeight(edge.getWeight()-pathsCapacity)
                        return_edge = residual_graph.getEdges()[edge.getTarget()+edge.getSource()]
                        return_edge.setWeight(return_edge.getWeight()+pathsCapacity)
                residual_graph.setFlow(residual_graph.getFlow()+pathsCapacity)
            else:
                print("residual graph flow: ", residual_graph.getFlow())
                #residual_graph.prettyPrint()
                break
        return None

    # Hopcroft-Karp
    def Hopcroft_Karp(self, graph: Graph) -> None:
        m = 0
        # set X and Y
        for i in range(len(graph.getVertices().keys())):
            if (i <= (len(graph.getVertices().keys()))/2):
                graph.getXVertices().append(list(graph.getVertices().keys())[i])
            else:
                graph.getYVertices().append(list(graph.getVertices().keys())[i])

        while self.BFS_Hopcroft_Karp(graph) == True: #I might need to reset the lists like I did with the other algo.
            for vertexLabel in graph.getXVertices():
                vertex = graph.getVertices()[vertexLabel]
                if vertex.getMate() == None:
                    if self.DFS_Hopcroft_Karp(graph, vertex) == True:
                        m = m + 1
        print(m)

    def BFS_Hopcroft_Karp(self, graph: Graph) -> bool:
        q = queue.Queue()
        for vertexLabel in graph.getXVertices():
            vertex = graph.getVertices()[vertexLabel]
            if vertex.getMate() == None:
                vertex.setDistance(0)
                q.put(vertex)
            else:
                vertex.setDistance(2147483647)
        nullVertex = Vertex("nullVertex")
        nullVertex.setAsTheNullVertex()
        while not q.empty():
            x = q.get()
            if x.getDistance() < nullVertex.getDistance():
                for neighborLabel in x.getNeighbors():
                    neighbor = graph.getVertices()[neighborLabel]
                    if (neighbor.getMate()):
                        mate = graph.getVertices()[neighbor.getMate()]
                        if mate.getDistance() == 2147483647:
                            mate.setDistance(x.getDistance()+1)
                            q.put(mate)
        return nullVertex.getDistance() != 2147483647
    
    def DFS_Hopcroft_Karp(self, graph: Graph, x) -> None:
        if x != None:
            for vertexLabel in x.getNeighbors():
                neighbor = graph.getVertices()[vertexLabel]
                mate = graph.getVertices()[neighbor.getMate()]
                if mate.getDistance() == x.getDistance() + 1:
                    if self.DFS_Hopcroft_Karp(graph, mate) == True:
                        neighbor.setMate(x)
                        x.setMate(neighbor)
                        return True
            x.setDistance(2147483647)
            return False
        return True

    def verticesColoring(self, graph: Graph) -> None:
        vertices = sorted(list(graph.getVertices().keys()), key=lambda x: len(graph.getVertices()[x].getNeighbors()), reverse=True)

        for vertexLabel in vertices:
            graph.setAvailableColors(len(vertices))
            vertex = graph.getVertices()[vertexLabel]
            for neighbor in vertex.getNeighbors():
                if neighbor in graph.getVertexColorMap():
                    color = graph.getVertexColorMap()[neighbor]
                    graph.getAvailableColors()[color] = False

            for color, available in enumerate(graph.getAvailableColors()):
                if available:
                    graph.getVertexColorMap()[vertexLabel] = color
                    if color not in graph.getUsedColors():
                        graph.appendToUsedColors(color)
                    break

        print("Minimum quantity of colors: ",len(graph.getUsedColors()))
        print("Vertex-Color map",graph.getVertexColorMap())

        return graph.getVertexColorMap()

    # Achar os caminhos aumentantes
    # BFS for Edmonds-Karp
    def findIncreasingPaths(self, residual_graph: Graph, s_label, t_label) -> list or None:
        residual_graph.p = []
        for vLabel in residual_graph.getVertices().keys():
            v = residual_graph.getVertices()[vLabel]
            v.BeenVisited=False

        s = residual_graph.getVertices()[s_label]
        t = residual_graph.getVertices()[t_label]
        s.setAsVisited()
        q = queue.Queue()
        q.put(s) # enqueue

        while q.empty() == False:
            u = q.get() # dequeue
            for neighbor_label in u.getOutgoingNeighbors():
                neighbor = residual_graph.getVertices()[neighbor_label]
                edge = residual_graph.getEdges()[u.getLabel()+neighbor.getLabel()]
                if (neighbor.hasBeenVisited() == False and edge.getWeight() > 0):
                    neighbor.setAsVisited()
                    neighbor.setAncestor(u)
                    if neighbor.getLabel() == t.getLabel():
                        residual_graph.appendToP(t)
                        w = t
                        while w.getLabel() != s.getLabel():
                            w = w.getAncestor()
                            residual_graph.insertToP(w)
                        #print("Path: ", [v.getLabel() for v in residual_graph.getP()])
                        return residual_graph.getP()
                    q.put(neighbor)
        return None
    
    # util method to create residual graph Gf = (V, Af, Cf)
    def createResidualGraph(self, graph: Graph) -> Graph:
        residual_graph = copy.deepcopy(graph)
        for edge_label in graph.getEdges().keys():
            edge = graph.getEdges()[edge_label]
            original_source = graph.getVertices()[edge.getSource()]
            original_target = graph.getVertices()[edge.getTarget()]
            source = residual_graph.getVertices()[edge.getSource()]
            target = residual_graph.getVertices()[edge.getTarget()]
            if (edge.getTarget() + edge.getSource()) not in graph.getEdges().keys():
                # build the return arc between target and source and reset the neighbors
                return_edge = Edge(edge.getTarget(), edge.getSource(), 0)
                residual_graph.appendNewEdge(return_edge)                                   # add the return edge to residual graph
                if (source.hasBeenSetInRG() == False):
                    source.outgoingNeighbors = original_source.getOutgoingNeighbors() \
                        + original_source.getIncomingNeighbors()
                    source.incomingNeighbors = [] + source.getOutgoingNeighbors()
                    source.setAsSetInRG()
                if (not target.hasBeenSetInRG() == False):
                    target.outgoing = original_target.getOutgoingNeighbors() \
                        + original_target.getIncomingNeighbors()
                    target.incomingNeighbors = [] +  target.getOutgoingNeighbors()
                    target.setAsSetInRG()
            else:
                # add an extra vertex: 
                # target -> vertex_x -> source (with the original arc weight)
                # source -> vertex_x -> target (with weight 0)
                if ('x_' + edge.getTarget() + edge.getSource()) not in (residual_graph.getVertices().keys()):
                    vertex_x = Vertex('x_' + edge.getSource() + edge.getTarget())
                    new_edge_target_to_x = Edge(edge.getTarget(), vertex_x.getLabel(), edge.getWeight())
                    new_edge_x_to_source = Edge(vertex_x.getLabel(), edge.getSource(), edge.getWeight())
                    new_edge_source_to_x = Edge(edge.getSource(), vertex_x.getLabel(), 0)
                    new_edge_x_to_target = Edge(vertex_x.getLabel(), edge.getTarget(), 0)
                    residual_graph.appendNewVertex(vertex_x)
                    residual_graph.appendNewEdge(new_edge_target_to_x)
                    residual_graph.appendNewEdge(new_edge_x_to_source)
                    residual_graph.appendNewEdge(new_edge_source_to_x)
                    residual_graph.appendNewEdge(new_edge_x_to_target)
                    if (source.hasBeenSetInRG() == False):
                        source.outgoingNeighbors = [] + original_source.getIncomingNeighbors() + [vertex_x.getLabel()]
                        source.incomingNeighbors = [] + source.outgoingNeighbors
                        source.setAsSetInRG()
                    if (not target.hasBeenSetInRG() == False):
                        target.outgoingNeighbors = [] + original_target.getOutgoingNeighbors() + [vertex for vertex in original_target.getIncomingNeighbors() if vertex not in original_target.getOutgoingNeighbors()] + [vertex_x.getLabel()]
                        target.incomingNeighbors = [] + target.getOutgoingNeighbors()
                        target.setAsSetInRG()
                    vertex_x.outgoingNeighbors = vertex_x.outgoingNeighbors + [edge.getSource()] + [edge.getTarget()]
                    vertex_x.incomingNeighbors = vertex_x.outgoingNeighbors
                    vertex_x.setAsSetInRG()
                    residual_graph.getEdges()[edge.getTarget() + edge.getSource()].setWeight(0)
        return residual_graph
    
    def getMinimalCapacity(self, graph: Graph, p) -> float or int:
        minimal = 2147483647
        for edge_label in p:
            if (len(edge_label)>1):
                edge = graph.getEdges()[edge_label]
                if edge.getWeight() < minimal:
                    minimal = edge.getWeight()
        return minimal

