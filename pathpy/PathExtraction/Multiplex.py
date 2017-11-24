import igraph
import pathpy as pp
import random

class Multiplex():
    """
    Nomenclature:

    Graph: iGraph graph object
    Path: PathPy Paths object
    Plot: iGraph Plot object

    Represents a multiplex network, i.e. a graph that consists of different link types between nodes, using a graph object as delegate.

    The graph can be accessed through the attitribute 'g'.

    Each edge of the multiplex has an attribute 'layer', which determines the layer that it belongs to. The value of the
    attribute 'layer' is an integer between 0 and n-1, where n is the total number of layers.
    """


    def __init__(self, graph=None):
        """
        @param graph: If given, the multiplex will use the graph as its delegate.
        """

        if not graph:
            self.g = igraph.Graph()
        else:
            self.g = graph


    def MergeGraphs(graphList):
        """
        Merges several graphs consisting of the same nodes into a multiplex.
        
        @param graphList: The list of graphs which should be merged into a multiplex.
        
        @return: The multiplex
        """
        
        g = graphList[0]
        g.to_directed()
        g.es["layer"] = 0
        mp = Multiplex(g)
        for l in range(1,len(graphList)):
            mp.g.add_edges(graphList[l].get_edgelist())
            mp.g.es.select(layer=None)["layer"] = l
            
        return mp


    def layerCount(self):
        """
        Returns the number of layers of the multiplex
        """
        return max(self.g.es['layer']) + 1 


    def add_edge(self, source, target, layer):
        """
        Adds an edge to the multiplex

        @param source: source vertex
        @param target: target vertex
        @param layer: the layer of the multiplex to which the edge should be added
        """
        self.g.add_edge(source, target, layer=layer)


    def shortestPaths(self, all_shortest_paths=True):
        """
        Returns a Paths object which consists of all (shortest) paths of all layers of a multiplex.
       
        @param all_shortest_paths: If the shortest path between two vertices is not unique, consider either all shortest paths
        or pick only one of them
        
        @return: The paths object
        """
        
        paths = pp.Paths()

        for l in range(self.layerCount()):

            # create graph which represents one layer of the multiplex
            g = igraph.Graph(self.g.vcount(), directed=True)
            for e in self.g.es.select(layer=l): g.add_edge(*e.tuple)

            for v in g.vs:
                     
                # get all shortest paths from v to all other nodes
                # NOTE: Why does get_all_shortest_paths() return duplicates???
                sp = g.get_all_shortest_paths(v,mode="OUT") if all_shortest_paths else g.get_shortest_paths(v,mode="OUT")

                # create set to remove duplicates
                sp_set = set()
                for p in sp:
                    sp_set.add(tuple(p))

                for p in sp_set: 
                    path_str = ",".join(map(str,p))
                    paths.addPath(path_str, expandSubPaths=False)
        
        # expand subpaths later. This might be faster if we got the same path in different layers.
        paths.expandSubPaths()

        return paths
