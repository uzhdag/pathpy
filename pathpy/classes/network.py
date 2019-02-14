# -*- coding: utf-8 -*-

#    pathpy is an OpenSource python package for the analysis of time series data
#    on networks using higher- and multi order graphical models.
#
#    Copyright (C) 2016-2018 Ingo Scholtes, ETH Zürich/Universität Zürich
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program. If not, see <http://www.gnu.org/licenses/>.
#
#    Contact the developer:
#
#    E-mail: scholtes@ifi.uzh.ch
#    Web:    http://www.ingoscholtes.net
import collections as _co
import copy
import itertools

import numpy as _np

import scipy.sparse as _sparse
import scipy.sparse.linalg as _sla

from pathpy.utils import Log, Severity
from pathpy.utils.exceptions import PathpyError, PathpyNotImplemented


class Network:
    r"""A graph or network that can be directed, undirected, unweighted or weighted
    and whose edges can contain arbitrary attributes. This is the base class for 
    HigherOrderNetwork

    Attributes
    ----------

    nodes : list
        A list of (string) nodes.
    edges : dictionary
        A dictionary containing edges (as tuple-valued keys) and their attributes (as value)
    """

    def __init__(self, directed=False):
        """
        Generates an empty network.
        """

        # Boolean value that inidcates whether the network is directed or undirected
        self.directed = directed

        # A dictionary containing nodes as well as node properties
        self.nodes = _co.defaultdict(dict)

        # A dictionary containing edges as well as edge properties
        if not directed:
            self.edges = UnorderedDict()
        else:
            self.edges = _co.defaultdict(dict)

        # A dictionary containing the sets of successors of all nodes
        self.successors = _co.defaultdict(set)

        # A dictionary containing the sets of predecessors of all nodes
        self.predecessors = _co.defaultdict(set)


    def __add__(self, other):
        r"""Add two networks and return the union of both

        Parameters
        ----------
        other : Network

        Returns
        -------
        Network
            Default operator +, which returns the sum of two Network objects
        """
        n_sum = Network(directed = self.directed or other.directed)
        n_sum.nodes = copy.deepcopy(self.nodes)
        n_sum.edges = copy.deepcopy(self.edges)
        n_sum.successors = copy.deepcopy(self.successors)
        n_sum.predecessors = copy.deepcopy(self.predecessors)
        for edge in other.edges:
            n_sum.add_edge(edge[0], edge[1], weight=other.edges[edge]['weight'])
        return n_sum

    @classmethod
    def read_file(cls, filename, separator=',', weighted=False, directed=False, header=False):
        r"""Reads a network from an edge list file.

        Reads data from a file containing multiple lines of *edges* of the
        form "v,w,frequency,X" (where frequency is optional and X are
        arbitrary additional columns). The default separating character ','
        can be changed. In order to calculate the statistics of paths of any length,
        by default all subpaths of length 0 (i.e. single nodes) contained in an edge
        will be considered.

        Parameters
        ----------
        filename : str
            path to edgelist file
        separator : str
            character separating the nodes
        weighted : bool
            is a weight given? if ``True`` it is the last element in the edge
            (i.e. ``a,b,2``)
        directed : bool
            are the edges directed or undirected
        header : bool
            if true skip the first row, useful if header row in file

        Returns
        -------
        Network
            a ``Network`` object obtained from the edgelist
        """
        net = cls(directed)

        with open(filename, 'r') as f:
            Log.add('Reading edge list ... ')
            header_offset = 0
            if header:
                f.readline()
                header_offset = 1

            for n, line in enumerate(f):
                fields = line.rstrip().split(separator)
                fields = [field.strip() for field in fields]
                if len(fields) < 2:
                    Log.add('Ignoring malformed line {0}: {1}'.format(n, line+header_offset), Severity.WARNING)
                else:
                    if weighted:
                        net.add_edge(fields[0], fields[1], weight=int(fields[2]))
                    else:
                        net.add_edge(fields[0], fields[1])

        Log.add('finished.')

        return net


    def write_file(self, filename, separator=',', weighted=False, header=False):
        r"""Writes a network to an edge file"""
        with open(filename, 'w+') as f:
            if header:
                if weighted:
                    f.write('source' + separator + 'target' + separator + 'weight' + '\n')
                else:
                    f.write('source' + separator + 'target' + '\n')
            for edge in self.edges:
                if weighted:       
                    f.write(str(edge[0]) + separator + str(edge[1]) + separator + str(self.edges[edge]['weight'])+'\n')
                else:
                    f.write(str(edge[0]) + separator + str(edge[1]) + '\n')

    @classmethod
    def from_sqlite(cls, cursor, directed=True):
        r"""Returns a new Network instance generated from links obtained 
        from an SQLite cursor. The cursor must refer to a table with at least
        two columns

                source target

        in which each row contains one link. Additional columns will be used as
        named edge properties. Since columns are accessed by name this function requires that a
        row factory object is set for the SQLite connection prior to cursor creation,
        i.e. you should set

                connection.row_factory = sqlite3.Row

        Parameters
        ----------
        cursor : 
            The SQLite cursor to fetch rows from. 
        directed : bool
            Whether or not links should be interpreted as directed. Default is True.

        Returns
        -------
        Network
            A Network instance created from the SQLite database.

        """
        from pathpy.classes import DAG
        if cls == DAG:
            n = cls()
        else:
            n = cls(directed=directed)

        assert cursor.connection.row_factory, \
            'Cannot access columns by name. Please set ' \
            'connection.row_factory = sqlite3.Row before creating DB cursor.'

        Log.add('Retrieving links from database ...')

        for row in cursor:
            n.add_edge(str(row['source']), str(row['target']))

        return n


    @classmethod
    def from_paths(cls, paths):
        r"""Generates a weighted directed network from a Paths
            object. The weight of directed links will correspond 
            to the statistics of (sub)-paths of length one"""
        network = cls(directed=True)

        # check all sub-paths of length one
        for p in paths.paths[1]:
            network.add_edge(p[0], p[1], weight=paths.paths[1][p].sum())

        return network

    @classmethod
    def from_temporal_network(cls, tempnet, min_time=None, max_time=None, directed=True):
        r"""Returns a time-aggregated directed network representation
        of a temporal network. The number of occurrences of
        the same edge at different time stamps is captured
        by edge weights.
        """
        network = cls(directed=directed)

        for (v, w, t) in tempnet.tedges:
            if (min_time is None or t >= min_time) and (max_time is None or t < max_time):
                if (v, w) in network.edges:
                    network.add_edge(v, w, weight=network.edges[(v, w)]['weight']+1.0)
                else:
                    network.add_edge(v, w)

        return network


    def to_unweighted(self):
        r"""Returns an unweighted copy of a directed or undirected network.
        In this copy all edge and node properties of the original network
        are removed, but the directionality of links is retained.
        """
        n = Network(directed = self.directed)

        for (v,w) in self.edges:
            n.add_edge(v, w)
        return n


    def to_undirected(self):
        r"""Returns an undirected copy of the network, in which all
        node and edge properties are removed.
        """
        n = Network(directed = False)

        for (v,w) in self.edges:
            #if v!=w:
            n.add_edge(v, w)
        return n


    def add_node(self, v, **node_attributes):
        r"""Adds a node to a network and assigns arbitrary
        node attributes.

        Parameters
        ----------
        node_attributes : dict
            Key-value pairs that will be stored as
            named node attributes in a dictionary. An
            attribute set via network.add_node(v, x=42) can be
            assessed via network.nodes[v]['x']. Any node in an undirected
            network will have the default attributes 'degree', 'inweight',
            and 'outweight'. Any node in a directed network will have the 
            default attributes 'indegree', 'outdegree', 'inweight', and 'outweight'.
            See examples below.

        Examples
        --------
            >>> network = pathpy.Network(directed=False)
            >>> network.add_node(v)
            >>> print(network.nodes[v])
            >>> {'inweight': 0.0, 'outweight': 0.0, 'degree': 0}            
            >>> network = pathpy.Network(directed=True)
            >>> network.add_node(v)
            >>> print(network.nodes[v])
            >>> {'inweight': 0.0, 'outweight': 0.0, 'indegree': 0, 'outdegree': 0}
        """
        if v not in self.nodes:
            self.nodes[v] = {**self.nodes[v], **node_attributes}

            # set default values if not set already
            if 'inweight' not in self.nodes[v]:
                self.nodes[v]['inweight'] = 0.0
            if 'outweight' not in self.nodes[v]:
                self.nodes[v]['outweight'] = 0.0
            if self.directed:
                self.nodes[v]['indegree'] = 0
                self.nodes[v]['outdegree'] = 0
            else:
                self.nodes[v]['degree'] = 0


    def remove_node(self, v):
        r"""Removes a node and all of its attributes from the network."""
        if v in self.nodes:
            # remove all incident edges and update neighbors
            if not self.directed:
                for w in list(self.successors[v]):
                    edge = (v, w)
                    self.nodes[w]['degree'] -= 1
                    self.nodes[w]['inweight'] -= self.edges[edge]['weight']
                    self.nodes[w]['outweight'] -= self.edges[edge]['weight']
                    self.successors[w].remove(v)
                    self.predecessors[w].remove(v)
                    del self.edges[edge]
            else:
                for w in list(self.successors[v]):
                    self.nodes[w]['indegree'] -= 1
                    self.nodes[w]['inweight'] -= self.edges[(v, w)]['weight']
                    self.predecessors[w].remove(v)
                    del self.edges[(v, w)]
                for w in list(self.predecessors[v]):
                    self.nodes[w]['outdegree'] -= 1
                    self.nodes[w]['outweight'] -= self.edges[(w, v)]['weight']
                    self.successors[w].remove(v)
                    del self.edges[(w, v)]
            del self.nodes[v]
        if v in self.successors:
            del self.successors[v]
        if v in self.predecessors:
            del self.predecessors[v]


    def remove_edge(self, source, target):
        r"""
        Remove an edge and all of its attributes from the network.

        Parameters
        ----------
        source : str
            Source node of the edge to remove
        target : str
            Target node of the edge to remove
        """
        if not (source in self.nodes and target in self.nodes):
            return None

        if self.directed:
            # take care of source
            self.nodes[source]['outdegree'] -= 1
            self.nodes[source]['outweight'] -= self.edges[(source, target)]['weight']
            self.successors[source].remove(target)

            # take care of target
            self.nodes[target]['indegree'] -= 1
            self.nodes[target]['inweight'] -= self.edges[(source, target)]['weight']
            self.predecessors[target].remove(source)

            del self.edges[(source, target)]
        else:
            # take care of source
            self.nodes[source]['degree'] -= 1
            self.nodes[source]['outweight'] -= self.edges[(source, target)]['weight']
            self.nodes[source]['inweight'] -= self.edges[(source, target)]['weight']
            self.successors[source].remove(target)
            self.predecessors[source].remove(target)

            # take care of target
            if source != target:
                self.nodes[target]['degree'] -= 1
                self.nodes[target]['outweight'] -= self.edges[(source, target)]['weight']
                self.nodes[target]['inweight'] -= self.edges[(source, target)]['weight']
                self.successors[target].remove(source)
                self.predecessors[target].remove(source)

            del self.edges[(source, target)]


    def add_clique(self, node_list, **edge_attributes):
        r"""
        Adds a fully connected clique to the network. This will 
        automatically create all edges between all pairs of nodes
        (without self-loops). Depending on the network type
        edges will be directed or undirected.

        Parameters
        ----------
        node_list: iterable
            the list of nodes for which all pairs will be connected
        edge_attributes: dict
            edge attributes that will be assigned to all generated edges
        """        
        for v, w in itertools.combinations(node_list, 2):            
            self.add_edge(v, w, **edge_attributes)
            if self.directed:
                self.add_edge(w, v, **edge_attributes)



    def add_edge(self, v, w, **edge_attributes):
        r"""
        Adds an edge to a network and assigns arbitrary
        key-value pairs as edge attribute.

        Parameters
        ----------
        v : str
            String label of the source node
        w : str
            String label of the target node
        edge_attributes : dict
            Key-value pairs that will be stored as
            named edge attributes in a dictionary. An
            attribute set via network.add_edge(v, w, x=42) can be
            assessed via network.edges[(v,w)]['x'].
            Any edge will have the default attribute 'weight', which
            is set to 1.0 by default. Edge weights are overwritten
            if an additional weighted edge is added later
            (see example below).

        Examples
        --------
            >>> network.add_edge('a','b')
            >>> print(network.edges[('a', 'b')]['weight'])
            >>> 1.0 
            >>> network.add_edge('a','b', weight = 2.0)
            >>> print(network.edges[('a', 'b')]['weight'])
            >>> 2.0
        """

        # Add nodes if they don't exist
        self.add_node(v)
        self.add_node(w)

        e = (v, w)

        if 'weight' in edge_attributes and isinstance(edge_attributes['weight'], int):
            edge_attributes['weight'] = float(edge_attributes['weight'])

        # add any new atributes to the edge
        self.edges[e] = {**self.edges[e], **edge_attributes}

        # add default weight of one, if no weight is specified
        if 'weight' not in self.edges[e]:
            self.edges[e]['weight'] = 1.0

        # update predecessor and successor lists
        self.successors[v].add(w)
        self.predecessors[w].add(v)
        if not self.directed:
            self.successors[w].add(v)
            self.predecessors[v].add(w)

        # update degrees and node weights
        if not self.directed:
            # update degree, in- and outweight
            self.nodes[v]['degree'] = len(self.successors[v])
            self.nodes[w]['degree'] = len(self.successors[w])

            S = [self.edges[(v,w)]['weight'] for w in self.successors[v]]
            if S:
                self.nodes[v]['outweight'] = sum(S)
                self.nodes[v]['inweight'] = self.nodes[v]['outweight']

            S = [self.edges[(v,w)]['weight'] for v in self.predecessors[w]]
            if S:
                self.nodes[w]['outweight'] = sum(S)
                self.nodes[w]['inweight'] = self.nodes[w]['outweight']
        else:
            self.nodes[v]['outdegree'] = len(self.successors[v])
            self.nodes[v]['indegree'] = len(self.predecessors[v])
            self.nodes[w]['outdegree'] = len(self.successors[w])
            self.nodes[w]['indegree'] = len(self.predecessors[w])

            # Note: Weights will be 0 for nodes with empty successors or predecessors. This is a
            # problem for higher-order networks, where the zero weight is assumed to be a vector
            # (0,0), Not updating weights in this case will ensure that we keep the initial value
            # of weights

            S = [self.edges[(v, x)]['weight'] for x in self.successors[v]]
            if S:
                self.nodes[v]['outweight'] = sum(S)
            S = [self.edges[(x, v)]['weight'] for x in self.predecessors[v]]
            if S:
                self.nodes[v]['inweight'] = sum(S)
            S = [self.edges[(w, x)]['weight'] for x in self.successors[w]]
            if S:
                self.nodes[w]['outweight'] = sum(S)
            S = [self.edges[(x, w)]['weight'] for x in self.predecessors[w]]
            if S:
                self.nodes[w]['inweight'] = sum(S)


    def find_nodes(self, select_node=lambda v: True):
        r"""
        Returns all nodes that satisfy a given condition. In the select_node
        lambda function, node attributes can be accessed by calling v['attr']
        """
        return [n for n in self.nodes if select_node(self.nodes[n])]


    def find_edges(self, select_nodes=lambda v, w: True, select_edges=lambda e: True):
        r"""
        Returns all edges that satisfy a given condition. Edges can be selected based
        on attributes of the adjacent nodes as well as attributes of the edge. In the select_edges
        lambda function,.

        Parameters
        ----------
        select_nodes : lambda
            a lambda function that takes two parameters v, w corresponding to the source and 
            target node of an edge. All edges for which the lambda function returns True will be 
            selected. Default is lambda v,w: True.
        select_edges : lambda
            a lambda function that takes a single parameter e corresponding to an edge tuple. 
            Edge attributes can be accessed by e['attr']. All edges for which the lambda function 
            returns True will be selected.  Default is lambda e: True.

        Example:
        >>> network.find_edges(select_nodes = lambda v,w: True if v['desired_node_property'] else False, 
                               select_edges = lambda e: True if e['desired_edge_property'] else False)
        """
        return [e for e in self.edges if (select_nodes(self.nodes[e[0]], self.nodes[e[1]]) and select_edges(self.edges[e]))]


    def ncount(self):
        """ Returns the number of nodes """
        return len(self.nodes)


    def ecount(self):
        r"""Returns the number of links """
        return len(self.edges)


    def total_edge_weight(self):
        r"""Returns the sum of all edge weights """
        if self.edges:
            return _np.sum(e['weight'] for e in self.edges.values())
        return 0


    def node_properties(self, prop):
        r"""Returns a list of arbitrary node properties in the network, 
        where entries have the same order as in network.nodes. If a property
        is not present for a given node, None will be added to the list.
        """
        properties = []
        for v in self.nodes:
            if prop in self.nodes[v]:
                properties.append(self.nodes[v][prop])
            else:
                properties.append(None)
        return properties


    def degrees(self, mode='degree'):
        r"""Returns the sequence of node degrees in the network, where
        entries have the same order as in network.nodes. Note that 
        if mode == 'degree' for a directed network, the degree sequence
        of the undirected network will be returned.

        Parameters
        ----------
        mode : str
            either 'degree', 'indegree', or 'outdegree'
        """
        assert mode is 'degree' or mode is 'indegree' or mode is 'outdegree', \
            'Only "degree", "indegree", or "outdegree" are supported.'
        
        if self.directed and mode == 'degree':
            return self.to_undirected().degrees()

        return self.node_properties(mode)


    def node_to_name_map(self):
        """Returns a dictionary that can be used to map nodes to matrix/vector indices"""
        return {v: idx for idx, v in enumerate(self.nodes)}


    def adjacency_matrix(self, weighted=True, transposed=False):
        """Returns a sparse adjacency matrix of the higher-order network. Unless transposed
        is set to true, the entry corresponding to a directed link s->t is stored in row s and
        column t and can be accessed via A[s,t].

        Parameters
        ----------
        weighted: bool
            if set to False, the function returns a binary adjacency matrix.
            If set to True, adjacency matrix entries contain edge weights.
        transposed: bool
            whether to transpose the matrix or not.

        Returns
        -------
        numpy cooc matrix
        """
        row = []
        col = []
        data = []

        edgeC = self.ecount()
        if not self.directed:
            n_self_loops = sum(s == t for (s, t) in self.edges)
            edgeC *= 2
            edgeC -= n_self_loops

        node_to_coord = self.node_to_name_map()

        for (s, t), e in self.edges.items():
            row.append(node_to_coord[s])
            col.append(node_to_coord[t])
            if weighted:
                data.append(e['weight'])
            else:
                data.append(1)

            if not self.directed and t != s:
                row.append(node_to_coord[t])
                col.append(node_to_coord[s])
                if weighted:
                    data.append(e['weight'])
                else:
                    data.append(1)

        shape = (self.ncount(), self.ncount())
        A = _sparse.coo_matrix((data, (row, col)), shape=shape).tocsr()

        if transposed:
            return A.transpose()
        return A


    def transition_matrix(self):
        """Returns a (transposed) transition matrix of a random walk process
        on the network

        Parameters
        ----------

        Returns
        -------

        """
        row = []
        col = []
        data = []

        # calculate weighted out-degrees of all nodes
        D = {n: self.nodes[n]['outweight'] for n in self.nodes}

        node_to_coord = self.node_to_name_map()

        for s, t in self.edges:
            # the following makes sure that we do not accidentally consider zero-weight
            # edges (automatically added by default_dic)
            weight = self.edges[(s, t)]['weight']
            if weight > 0:
                # add transition from s to t
                row.append(node_to_coord[t])
                col.append(node_to_coord[s])
                assert D[s] > 0, \
                    'Encountered zero out-weight or out-degree for node "{s}" ' \
                    'while weight of link ({s}, {t}) is non-zero.'.format(s=s, t=t)
                prob = weight / D[s]
                if prob < 0 or prob > 1:  # pragma: no cover
                    raise ValueError('Encountered transition probability {p} outside '
                                     '[0,1] range.'.format(p=prob))
                data.append(prob)

                # add transition from t to s for undirected network
                if not self.directed and s!=t:
                    row.append(node_to_coord[s])
                    col.append(node_to_coord[t])
                    assert D[t] > 0, \
                    'Encountered zero out-degree for node "{t}" ' \
                    'while weight of link ({t}, {s}) is non-zero.'.format(s=s, t=t)
                    prob = weight / D[t]
                    if prob < 0 or prob > 1:  # pragma: no cover
                        raise ValueError('Encountered transition probability {p} outside '
                                        '[0,1] range.'.format(p=prob))
                    data.append(prob)

        data = _np.array(data)
        data = data.reshape(data.size, )

        shape = self.ncount(), self.ncount()
        return _sparse.coo_matrix((data, (row, col)), shape=shape).tocsr()


    def laplacian_matrix(self, weighted=False, transposed=False):
        """
        Returns the transposed normalized Laplacian matrix corresponding to the network.

        Parameters
        ----------

        Returns
        -------

        """
        if weighted:
            A = self.transition_matrix().transpose()
            D = _sparse.identity(self.ncount())
        else:
            A = self.adjacency_matrix(weighted=False)
            D = _sparse.diags(_np.array([float(self.nodes[v]['degree']) for v in self.nodes]))
        L = D - A
        if transposed:
            return L.transpose()
        return L


    @staticmethod
    def leading_eigenvector(A, normalized=True, lanczos_vecs=None, maxiter=None):
        """Compute normalized leading eigenvector of a given matrix A.

        Parameters
        ----------
        A:
            sparse matrix for which leading eigenvector will be computed
        normalized: bool
            whether or not to normalize, default is True
        lanczos_vecs: int
            number of Lanczos vectors to be used in the approximate
            calculation of eigenvectors and eigenvalues. This maps to the ncv parameter
            of scipy's underlying function eigs.
        maxiter: int
            scaling factor for the number of iterations to be used in the
            approximate calculation of eigenvectors and eigenvalues.

        Returns
        -------

        """
        if not _sparse.issparse(A):  # pragma: no cover
            raise TypeError("A must be a sparse matrix")

        # NOTE: ncv sets additional auxiliary eigenvectors that are computed
        # NOTE: in order to be more confident to find the one with the largest
        # NOTE: magnitude, see https://github.com/scipy/scipy/issues/4987
        if lanczos_vecs == None or maxiter == None:
            w, pi = _sla.eigs(A, k=1, which="LM")
        else:
            w, pi = _sla.eigs(A, k=1, which="LM", ncv=lanczos_vecs, maxiter=maxiter)
        pi = pi.reshape(pi.size, )
        if normalized:
            pi /= sum(pi)
        return pi


    def summary(self):
        """Returns a string containing basic summary statistics of this network instance
        """
        summary_fmt = (
            '{directed_str} network\n'
            'Nodes:\t\t\t\t{ncount}\n'
            'Links:\t\t\t\t{ecount}\n'
        )
        if self.directed:
            directed_str = 'Directed'
        else:
            directed_str = 'Undirected'
        summary = summary_fmt.format(directed_str=directed_str, ncount=self.ncount(), ecount=self.ecount())
        return summary

    def __str__(self):
        """Returns the default string representation of this network instance"""
        return self.summary()

    def _repr_(self):
        """Returns the default string representation for jupyter"""
        return self.summary()

    def _repr_html_(self):
        """
        display an interactive d3js visualisation of the network in jupyter
        """
        from pathpy.visualisation.html import generate_html
        return generate_html(self)
        


def network_from_networkx(graph):
    """method to load a networkx graph into a pathpy.Network instance

    Parameters
    ----------
    garph

    Returns
    -------
    Network
    """
    try:
        import networkx as nx
    except ImportError:
        raise PathpyError("To load a network from networkx it must be installed")

    if isinstance(graph, nx.DiGraph):
        directed = True
    elif isinstance(graph, nx.Graph):
        directed = False
    else:
        raise PathpyNotImplemented("At the moment only DiGraph and Graph are supported.")

    net = Network(directed=directed)
    for node_id in graph.nodes:
        net.add_node(str(node_id), **graph.node[node_id])

    for edge in graph.edges:
        net.add_edge(str(edge[0]), str(edge[1]), **graph.edges[edge])

    return net


def network_to_networkx(network):
    """method to export a pathpy Network to a networkx compatible graph

    Parameters
    ----------
    network: Network

    Returns
    -------
    networkx Graph or DiGraph
    """
    # keys to exclude since they are handled differently in networkx
    excluded_node_props = {"degree", "inweight", "outweight", "indegree", "outdegree"}
    try:
        import networkx as nx
    except ImportError:
        raise PathpyError("To export a network to networkx it must be installed")

    directed = network.directed
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    for node_id, node_props in network.nodes.items():
        valid_props = {k: v for k, v in node_props.items() if k not in excluded_node_props}
        graph.add_node(node_id, **valid_props)

    for edge, edge_props in network.edges.items():
        graph.add_edge(*edge, **edge_props)

    return graph


class UnorderedDict(dict):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys

       Source: https://stackoverflow.com/questions/3387691/how-to-perfectly-override-a-dict
       """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.store = _co.defaultdict(dict)
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __contains__(self, item):
        return self.__keytransform__(item) in self.store

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __missing__(self, key):
        return {}

    def keys(self):
        return self.store.keys()

    def values(self):
        return self.store.values()

    def items(self):
        return self.store.items()

    def __keytransform__(self, key):
        return tuple(sorted(key))
