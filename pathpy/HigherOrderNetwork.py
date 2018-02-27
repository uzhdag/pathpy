# -*- coding: utf-8 -*-
#    pathpy is an OpenSource python package for the analysis of time series data
#    on networks using higher- and multi order graphical models.
#
#    Copyright (C) 2016-2017 Ingo Scholtes, ETH Zürich
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
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#    Contact the developer:

#    E-mail: ischoltes@ethz.ch
#    Web:    http://www.ingoscholtes.net

import collections as _co

import numpy as _np

import scipy.sparse as _sparse
import scipy.sparse.linalg as _sla

from pathpy.Log import Log
from pathpy.Log import Severity


class HigherOrderNetwork:
    """
    Instances of this class capture a k-th-order representation
    of path statistics. Path statistics can originate from pathway
    data, temporal networks, or from processes observed on top
    of a network topology.
    """

    def __init__(self, paths, k=1, separator='-', null_model=False,
                 method='FirstOrderTransitions', lanczos_vecs=15, maxiter=1000):
        """Generates a k-th-order representation based on the given path statistics.

        Parameters
        ----------
        paths: Path
            An instance of class Paths, which contains the path statistics to be used in
            the generation of the k-th order representation
        k: int
            The order of the network representation to generate. For the default case of
            k=1, the resulting representation corresponds to the usual (first-order)
            aggregate network, i.e. links connect nodes and link weights are given by the
            frequency of each interaction. For k>1, a k-th order node corresponds to a
            sequence of k nodes. The weight of a k-th order link captures the frequency
            of a path of length k.
        separator: str
            The separator character to be used in higher-order node names.
        null_model: bool
            For the default value False, link weights are generated based on the
            statistics of paths of length k in the underlying path statistics instance.
            If True, link weights are generated from the first-order model (k=1) based on
            the assumption of independent links (i.e. corresponding) to a first-order
            Markov model.
        method: str
            specifies how the null model link weights in the k-th order model are
            calculated. For the default method='FirstOrderTransitions', the weight
            w('v_1-v_2-...v_k', 'v_2-...-v_k-v_k+1') of a k-order edge is set to the
            transition probability T['v_k', 'v_k+1'] in the first order network.
            For method='KOrderPi' the entry pi['v1-...-v_k'] in the stationary
            distribution of the k-order network is used instead.
        lanczos_vecs: int
        maxiter: int
        """
        assert not null_model or (null_model and k > 1)

        assert method in ['FirstOrderTransitions', 'KOrderPi'], \
            'Error: unknown method to build null model'

        assert paths.paths.keys() and max(paths.paths.keys()) >= k, \
            'Error: constructing a model of order k requires paths of at least length k'

        # The order of this HigherOrderNetwork
        self.order = k

        # The paths object used to generate this instance
        self.paths = paths

        # The nodes in this HigherOrderNetwork
        self.nodes = []

        # The separator character used to label higher-order nodes.
        # For separator '-', a second-order node will be 'a-b'.
        self.separator = separator

        # A dictionary containing the sets of successors of all nodes
        self.successors = _co.defaultdict(set)

        # A dictionary containing the sets of predecessors of all nodes
        self.predecessors = _co.defaultdict(set)

        # A dictionary containing the out-degrees of all nodes
        self.outdegrees = _co.defaultdict(lambda: 0.0)

        # A dictionary containing the in-degrees of all nodes
        self.indegrees = _co.defaultdict(lambda: 0.0)

        # NOTE: edge weights, as well as in- and out weights of nodes are
        # numpy arrays consisting of two weight components [w0, w1]. w0
        # counts the weight of an edge based on its occurrence in a subpaths
        # while w1 counts the weight of an edge based on its occurrence in
        # a longest path. As an illustrating example, consider the single
        # path a -> b -> c. In the first-order network, the weights of edges
        # (a,b) and (b,c) are both (1,0). In the second-order network, the
        # weight of edge (a-b, b-c) is (0,1).

        # A dictionary containing edges as well as edge weights
        self.edges = _co.defaultdict(lambda: _np.array([0., 0.]))

        # A dictionary containing the weighted in-degrees of all nodes
        self.inweights = _co.defaultdict(lambda: _np.array([0., 0.]))

        # A dictionary containing the weighted out-degrees of all nodes
        self.outweights = _co.defaultdict(lambda: _np.array([0., 0.]))

        if k > 1:
            # For k>1 we need the first-order network to generate the null model
            # and calculate the degrees of freedom

            # For a multi-order model, the first-order network is generated multiple
            # times!
            # TODO: Make this more efficient
            g1 = HigherOrderNetwork(paths, k=1)
            g1_node_mapping = g1.node_to_name_map()
            A = g1.adjacency_matrix(include_subpaths=True, weighted=False,
                                    transposed=True)

        if not null_model:
            # Calculate the frequency of all paths of
            # length k, generate k-order nodes and set
            # edge weights accordingly
            node_set = set()
            iterator = paths.paths[k].items()

            if k == 0:
                # For a 0-order model, we generate a dummy start node
                node_set.add('start')
                for key, val in iterator:
                    w = key[0]
                    node_set.add(w)
                    self.edges[('start', w)] += val
                    self.successors['start'].add(w)
                    self.predecessors[w].add('start')
                    self.indegrees[w] = len(self.predecessors[w])
                    self.inweights[w] += val
                    self.outdegrees['start'] = len(self.successors['start'])
                    self.outweights['start'] += val
            else:
                for key, val in iterator:
                    # Generate names of k-order nodes v and w
                    v = separator.join(key[0:-1])
                    w = separator.join(key[1:])
                    node_set.add(v)
                    node_set.add(w)
                    self.edges[(v, w)] += val
                    self.successors[v].add(w)
                    self.predecessors[w].add(v)
                    self.indegrees[w] = len(self.predecessors[w])
                    self.inweights[w] += val
                    self.outdegrees[v] = len(self.successors[v])
                    self.outweights[v] += val

            self.nodes = list(sorted(node_set))

            # Note: For all sequences of length k which (i) have never been observed, but
            #       (ii) do actually represent paths of length k in the first-order
            #       network, we may want to include some 'escape' mechanism along the
            #       lines of (Cleary and Witten 1994)

        else:
            # generate the *expected* frequencies of all possible
            # paths based on independently occurring (first-order) links

            # generate all possible paths of length k
            # based on edges in the first-order network
            possiblePaths = list(g1.edges.keys())

            for _ in range(k - 1):
                E_new = list()
                for e1 in possiblePaths:
                    for e2 in g1.edges:
                        if e1[-1] == e2[0]:
                            p = e1 + (e2[1],)
                            E_new.append(p)
                possiblePaths = E_new

            # validate that the number of unique generated paths corresponds to the sum
            # of entries in A**k
            A_sum = _np.sum(A ** k)
            assert A_sum == len(possiblePaths), \
                'Expected {ak} paths but got {re}'.format(ak=A_sum, re=len(possiblePaths))

            if method == 'KOrderPi':
                # compute stationary distribution of a random walker in the k-th order
                # network
                g_k = HigherOrderNetwork(paths, k, separator, null_model=False)
                transition_m = g_k.transition_matrix(include_subpaths=True)
                pi_k = HigherOrderNetwork.leading_eigenvector(
                    transition_m,
                    normalized=True,
                    lanczos_vecs=lanczos_vecs,
                    maxiter=maxiter
                )
                gk_node_mapping = g_k.node_to_name_map()
            else:
                # A = g1.adjacency_matrix(includeSubPaths=True, weighted=True,
                # transposed=False)
                T = g1.transition_matrix(include_subpaths=True)

            # assign link weights in k-order null model
            for p in possiblePaths:
                v = p[0]
                # add k-order nodes and edges
                for l in range(1, k):
                    v = v + separator + p[l]
                w = p[1]
                for l in range(2, k + 1):
                    w = w + separator + p[l]
                if v not in self.nodes:
                    self.nodes.append(v)
                if w not in self.nodes:
                    self.nodes.append(w)

                # NOTE: under the null model's assumption of independent events, we
                # have P(B|A) = P(A ^ B)/P(A) = P(A)*P(B)/P(A) = P(B)
                # In other words: we are encoding a k-1-order Markov process in a k-order
                # Markov model and for the transition probabilities T_AB in the k-order
                #  model
                # we simply have to set the k-1-order probabilities, i.e. T_AB = P(B)

                # Solution A: Use entries of stationary distribution,
                # which give stationary visitation frequencies of k-order node w
                if method == 'KOrderPi':
                    if w in gk_node_mapping:
                        w_coordinate = gk_node_mapping[w]
                        eigen_value = pi_k[w_coordinate]
                        if _np.abs(_np.imag(eigen_value)) < 1e-16:
                            self.edges[(v, w)] = _np.array([0, _np.real(eigen_value)])

                # Solution B: Use relative edge weight in first-order network
                # Note that A is *not* transposed
                # self.edges[(v,w)] = A[(g1.nodes.index(p[-2]),g1.nodes.index(p[-1]))]
                # / A.sum()

                # Solution C: Use transition probability in first-order network
                # Note that T is transposed (!)
                elif method == 'FirstOrderTransitions':
                    v_i, w_i = g1_node_mapping[p[-1]], g1_node_mapping[p[-2]]
                    p_vw = T[v_i, w_i]
                    self.edges[(v, w)] = _np.array([0, p_vw])

                # Solution D: calculate k-path weights based on entries of squared
                # k-1-order adjacency matrix

                # Note: Solution B and C are equivalent
                self.successors[v].add(w)
                self.indegrees[w] = len(self.predecessors[w])
                self.inweights[w] += self.edges[(v, w)]
                self.outdegrees[v] = len(self.successors[v])
                self.outweights[v] += self.edges[(v, w)]

        # Compute degrees of freedom of models
        if k == 0:
            # for a zero-order model, we just fit node probabilities
            # (excluding the special 'start' node)
            # Since probabilities must sum to one, the effective degree
            # of freedom is one less than the number of nodes
            # This holds for both the paths and the ngrams model
            self.dof_paths = self.vcount() - 2
            self.dof_ngrams = self.vcount() - 2
        else:
            # for a first-order model, self is the first-order network
            if k == 1:
                g1 = self
                A = g1.adjacency_matrix(include_subpaths=True, weighted=False,
                                        transposed=True)

            # Degrees of freedom in a higher-order ngram model
            s = g1.vcount()

            # The degrees of freedom of the higher-order model, under the ngram
            # assumption
            self.dof_ngrams = (s ** k) * (s - 1)

            # For k>0, the degrees of freedom of a path-based model depend on
            # the number of possible paths of length k in the first-order network.
            # Since probabilities in each row must sum to one, the degrees
            # of freedom must be reduced by one for each k-order node
            # that has at least one possible transition.

            # (A**k).sum() counts the number of different paths of exactly length k
            # based on the first-order network, which corresponds to the number of
            # possible transitions in the transition matrix of a k-th order model.
            paths_k = (A ** k).sum()

            # For the degrees of freedom, we must additionally consider that
            # rows in the transition matrix must sum to one, i.e. we have to
            # subtract one degree of freedom for every non-zero row in the (null-model)
            # transition matrix. In other words, we subtract one for every path of
            # length k-1
            # that can possibly be followed by at least one edge to a path of length k

            # This can be calculated by counting the number of non-zero elements in the
            # vector containing the row sums of A**k
            non_zero = _np.count_nonzero((A ** k).sum(axis=0))

            # The degrees of freedom of the higher-order model, under the paths
            # assumption
            self.dof_paths = paths_k - non_zero

    def vcount(self):
        """ Returns the number of nodes """
        return len(self.nodes)

    def ecount(self):
        """ Returns the number of links """
        return len(self.edges)

    def total_edge_weight(self):
        """ Returns the sum of all edge weights """
        if self.edges:
            return sum(self.edges.values())
        return _np.array([0, 0])

    def model_size(self):
        """
        Returns the number of non-zero elements in the adjacency matrix
        of the higher-order model.
        """
        return self.adjacency_matrix().count_nonzero()

    def higher_order_node_to_path(self, node):
        """Helper function that transforms a node in a higher-order network of order k
        into a corresponding path of length k-1. For a higher-order node 'a-b-c-d'
        this function will return ('a','b','c','d')

        Parameters
        ----------
        node: str
            The higher-order node to be transformed to a path.
            TODO: this function assumes that the separator is '-', but it is not sure
            TODO: that the user will use it.

        Returns
        -------
        tuple
        """
        return tuple(node.split(self.separator))

    def path_to_higher_order_nodes(self, path, k=None):
        """Helper function that transforms a path into a sequence of k-order nodes
        using the separator character of the HigherOrderNetwork instance

        Parameters
        ----------
        path:
            the path tuple to turn into a sequence of higher-order nodes
        k: int
            the order of the representation to use (default: order of the
            HigherOrderNetwork instance)

        Returns
        -------
        list

        Examples
        --------

        Consider an example path (a,b,c,d) with a separator string '-'

        >>> path_tuple = ('a', 'b', 'c', 'd')
        >>> paths = Paths()
        >>> paths.add_path_tuple(path_tuple)
        >>> hon = HigherOrderNetwork(paths, separator='-')
        >>> hon.path_to_higher_order_nodes(path_tuple, k=1)
        ['a', 'b', 'c', 'd']
        >>> hon.path_to_higher_order_nodes(path_tuple, k=2)
        ['a-b', 'b-c', 'c-d']
        >>> hon.path_to_higher_order_nodes(path_tuple, k=3)
        ['a-b-c', 'b-c-d']
        """

        if k is None:
            k = self.order
        assert len(path) > k, 'Error: Path must be longer than k'

        if k == 0 and len(path) == 1:
            return ['start', path[0]]

        return [self.separator.join(path[n:n + k]) for n in range(len(path) - k + 1)]

    def node_to_name_map(self):
        """Returns a dictionary that can be used to map nodes to matrix/vector indices"""
        return {v: idx for idx, v in enumerate(self.nodes)}

    def degrees_of_freedom(self, assumption="paths"):
        """Calculates the degrees of freedom (i.e. number of parameters) of
        this k-order model. Depending on the modeling assumptions, this either
        corresponds to the number of paths of length k in the first-order network
        or to the number of all possible k-grams. The degrees of freedom of a model
        can be used to assess the model complexity when calculating, e.g., the
        Bayesian Information Criterion (BIC).

        Parameters
        ----------
        assumption: str
            if set to 'paths', for the degree of freedom calculation in the BIC, only
            paths in the first-order network topology will be considered. This is needed
            whenever we are interested in a modeling of paths in a given network topology.
            If set to 'ngrams' all possible n-grams will be considered, independent of
            whether they are valid paths in the first-order network or not. The 'ngrams'
            and the 'paths' assumption coincide if the first-order network is fully
            connected.

        Returns
        -------
        int
        """
        assert assumption in ['paths', 'ngrams'], 'Error: Invalid assumption'

        if assumption == 'paths':
            return self.dof_paths
        return self.dof_ngrams

    def distance_matrix(self):
        """Calculates shortest path distances between all pairs of higher-order nodes
        using the Floyd-Warshall algorithm."""

        Log.add('Calculating distance matrix in higher-order network '
                '(k = %s) ...' % self.order, Severity.INFO)

        dist = _co.defaultdict(lambda: _co.defaultdict(lambda: _np.inf))

        # assign first the default weight of 1
        for e in self.edges:
            dist[e[0]][e[1]] = 1

        # set all self-loop edges to 0
        for v in self.nodes:
            dist[v][v] = 0

        for k in self.nodes:
            for v in self.nodes:
                for w in self.nodes:
                    if dist[v][w] > dist[v][k] + dist[k][w]:
                        dist[v][w] = dist[v][k] + dist[k][w]

        Log.add('finished.', Severity.INFO)

        return dist

    def shortest_paths(self):
        """
        Calculates all shortest paths between all pairs of
        higher-order nodes using the Floyd-Warshall algorithm.
        """

        Log.add('Calculating shortest paths in higher-order network '
                '(k = %s) ...' % self.order, Severity.INFO)

        dist = _co.defaultdict(lambda: _co.defaultdict(lambda: _np.inf))
        shortest_paths = _co.defaultdict(lambda: _co.defaultdict(set))

        for e in self.edges:
            dist[e[0]][e[1]] = 1
            shortest_paths[e[0]][e[1]].add(e)

        for k in self.nodes:
            for v in self.nodes:
                for w in self.nodes:
                    if v != w:
                        if dist[v][w] > dist[v][k] + dist[k][w]:
                            dist[v][w] = dist[v][k] + dist[k][w]
                            shortest_paths[v][w] = set()
                            for p in list(shortest_paths[v][k]):
                                for q in list(shortest_paths[k][w]):
                                    shortest_paths[v][w].add(p + q[1:])
                        elif dist[v][w] == dist[v][k] + dist[k][w]:
                            for p in list(shortest_paths[v][k]):
                                for q in list(shortest_paths[k][w]):
                                    shortest_paths[v][w].add(p + q[1:])

        for v in self.nodes:
            dist[v][v] = 0
            shortest_paths[v][v].add((v,))

        Log.add('finished.', Severity.INFO)

        return shortest_paths

    def distance_matrix_first_order(self):
        """
        Projects a distance matrix from a higher-order to first-order nodes, while path
        lengths are calculated based on the higher-order topology
        """

        dist = self.distance_matrix()
        dist_first = _co.defaultdict(lambda: _co.defaultdict(lambda: _np.inf))

        # calculate distances between first-order nodes based on distance in
        # higher-order topology
        for vk in dist:
            for wk in dist[vk]:
                v1 = self.higher_order_node_to_path(vk)[0]
                w1 = self.higher_order_node_to_path(wk)[-1]
                if dist[vk][wk] + self.order - 1 < dist_first[v1][w1]:
                    dist_first[v1][w1] = dist[vk][wk] + self.order - 1

        return dist_first

    def higher_order_path_to_first_order(self, path):
        """Maps a path in the higher-order network to a path in the first-order network.
        As an example, the second-order path ('a-b', 'b-c', 'c-d') of length two is mapped
        to the first-order path ('a','b','c','d') of length four.
        In general, a path of length l in a network of order k is mapped to a path of
        length l+k-1 in the first-order network.

        Parameters
        ----------
        path: str
            The higher-order path that shall be mapped to the first-order network

        Returns
        -------
        tuple
        """
        p1 = self.higher_order_node_to_path(path[0])
        for x in path[1:]:
            p1 += (self.higher_order_node_to_path(x)[-1],)
        return p1

    def reduce_to_gcc(self):
        """Reduces the higher-order network to its largest (giant) strongly connected
        component (using Tarjan's algorithm).
        """

        # nonlocal variables (!)
        index = 0
        S = []
        indices = _co.defaultdict(lambda: None)
        low_link = _co.defaultdict(lambda: None)
        on_stack = _co.defaultdict(lambda: False)

        # Tarjan's algorithm
        def strong_connect(v):
            nonlocal index
            nonlocal S
            nonlocal indices
            nonlocal low_link
            nonlocal on_stack

            indices[v] = index
            low_link[v] = index
            index += 1
            S.append(v)
            on_stack[v] = True

            for w in self.successors[v]:
                if indices[w] is None:
                    strong_connect(w)
                    low_link[v] = min(low_link[v], low_link[w])
                elif on_stack[w]:
                    low_link[v] = min(low_link[v], indices[w])

            # Generate SCC of node v
            component = set()
            if low_link[v] == indices[v]:
                while True:
                    w = S.pop()
                    on_stack[w] = False
                    component.add(w)
                    if v == w:
                        break
            return component

        # Get largest strongly connected component
        components = _co.defaultdict(set)
        max_size = 0
        max_head = None
        for v in self.nodes:
            if indices[v] is None:
                components[v] = strong_connect(v)
                if len(components[v]) > max_size:
                    max_head = v
                    max_size = len(components[v])

        scc = components[max_head]

        # Reduce higher-order network to SCC
        for v in list(self.nodes):
            if v not in scc:
                self.nodes.remove(v)
                del self.successors[v]

        for (v, w) in list(self.edges):
            if v not in scc or w not in scc:
                del self.edges[(v, w)]

    def summary(self):
        """Returns a string containing basic summary statistics of this higher-order
        graphical model instance
        """
        summary_fmt = (
            'Graphical model of order k = {order}\n'
            '\n'
            'Nodes:\t\t\t\t{vcount}\n'
            'Links:\t\t\t\t{ecount}\n'
            'Total weight (sub/longest):\t{sub_w}/{uni_w}\n'
        )
        summary = summary_fmt.format(
            order=self.order, vcount=self.vcount(), ecount=self.ecount(),
            sub_w=self.total_edge_weight()[0], uni_w=self.total_edge_weight()[1]
        )
        return summary

    def __str__(self):
        """Returns the default string representation of this graphical model instance"""
        return self.summary()

    def adjacency_matrix(self, include_subpaths=True, weighted=True, transposed=False):
        """Returns a sparse adjacency matrix of the higher-order network. By default,
        the entry corresponding to a directed link source -> target is stored in row s and
        column t and can be accessed via A[s,t].

        Parameters
        ----------
        include_subpaths: bool
            if set to True, the returned adjacency matrix will account for the occurrence
            of links of order k (i.e. paths of length k-1) as subpaths
        weighted: bool
            if set to False, the function returns a binary adjacency matrix.
            If set to True, adjacency matrix entries will contain the weight of an edge.
        transposed: bool
            whether to transpose the matrix or not.

        Returns
        -------
        numpy cooc matrix
        """
        row = []
        col = []
        data = []

        node_to_coord = self.node_to_name_map()

        if transposed:
            for s, t in self.edges:
                row.append(node_to_coord[t])
                col.append(node_to_coord[s])
        else:
            for s, t in self.edges:
                row.append(node_to_coord[s])
                col.append(node_to_coord[t])

        # create array with non-zero entries
        if not weighted:
            data = _np.ones(len(self.edges.keys()))
        else:
            if include_subpaths:
                data = _np.array([float(x.sum()) for x in self.edges.values()])
            else:
                data = _np.array([float(x[1]) for x in self.edges.values()])

        shape = (self.vcount(), self.vcount())
        return _sparse.coo_matrix((data, (row, col)), shape=shape).tocsr()

    def transition_matrix(self, include_subpaths=True):
        """Returns a (transposed) random walk transition matrix corresponding to the
        higher-order network.

        Parameters
        ----------
        include_subpaths: bool
            whether or not to include subpath statistics in the transition probability
            calculation (default True)

        Returns
        -------

        """
        row = []
        col = []
        data = []
        # calculate weighted out-degrees (with or without subpaths)
        if include_subpaths:
            D = {n: w.sum() for n, w in self.outweights.items()}
        else:
            D = {n: w[1] for n, w in self.outweights.items()}

        node_to_coord = self.node_to_name_map()

        for (s, t) in self.edges:
            # either s->t has been observed as a longest path, or we are interested in
            # subpaths as well

            # the following makes sure that we do not accidentally consider zero-weight
            # edges (automatically added by default_dic)
            unique_weight = self.edges[(s, t)][1]
            subpath_weight = self.edges[(s, t)][0]
            is_valid = (unique_weight > 0 or (include_subpaths and subpath_weight > 0))
            if is_valid:
                row.append(node_to_coord[t])
                col.append(node_to_coord[s])
                if include_subpaths:
                    count = self.edges[(s, t)].sum()
                else:
                    count = self.edges[(s, t)][1]
                assert D[s] > 0, \
                    'Encountered zero out-degree for node "{s}" ' \
                    'while weight of link ({s}, {t}) is non-zero.'.format(s=s, t=t)
                prob = count / D[s]
                if prob < 0 or prob > 1:  # pragma: no cover
                    raise ValueError('Encountered transition probability {p} outside '
                                     '[0,1] range.'.format(p=prob))
                data.append(prob)

        data = _np.array(data)
        data = data.reshape(data.size, )

        shape = self.vcount(), self.vcount()
        return _sparse.coo_matrix((data, (row, col)), shape=shape).tocsr()

    @staticmethod
    def leading_eigenvector(A, normalized=True, lanczos_vecs=15, maxiter=1000):
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
            approximate calculation of eigenvectors and eigenvalues. The number of
            iterations passed to scipy's underlying eigs function will be n*maxiter
            where n is the number of rows/columns of the Laplacian matrix.

        Returns
        -------

        """
        if not _sparse.issparse(A):  # pragma: no cover
            raise TypeError("A must be a sparse matrix")

        # NOTE: ncv sets additional auxiliary eigenvectors that are computed
        # NOTE: in order to be more confident to find the one with the largest
        # NOTE: magnitude, see https://github.com/scipy/scipy/issues/4987
        w, pi = _sla.eigs(A, k=1, which="LM", ncv=lanczos_vecs, maxiter=maxiter)
        pi = pi.reshape(pi.size, )
        if normalized:
            pi /= sum(pi)
        return pi

    def laplacian_matrix(self, include_subpaths=True):
        """
        Returns the transposed Laplacian matrix corresponding to the higher-order network.

        Parameters
        ----------
        include_subpaths: bool
            Whether or not subpath statistics shall be included in the calculation of
            matrix weights

        Returns
        -------

        """
        transition_matrix = self.transition_matrix(include_subpaths)
        identity_matrix = _sparse.identity(self.vcount())

        return identity_matrix - transition_matrix

    def _to_html(self, width=600, height=600, require=True):
        import json
        import os
        from string import Template

        # prefix nodes starting with number
        def fix_node_name(v):
            if v[0].isdigit():
                return "n_" + v
            else:
                return v

        network_data = {
            'nodes': [{'id': fix_node_name(v), 'group': 1} for v in self.nodes],
            'links': [
                {'source': fix_node_name(e[0]),
                 'target': fix_node_name(e[1]),
                 'value': 1} for e, weight in self.edges.items()
            ]
        }

        import string
        import random

        all_chars = string.ascii_letters + string.digits
        div_id = "".join(random.choice(all_chars) for x in range(8))

        template_file = 'higherordernet_require.html'
        if not require:
            template_file = 'higherordernet.html'

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'js',
                               template_file)) as f:
            html_str = f.read()

        html_template = Template(html_str)

        return html_template.substitute({
            'network_data': json.dumps(network_data),
            'width': width,
            'height': height,
            'div_id': div_id})

    def _repr_html_(self, require=True):
        """
        display an interactive D3 visualisation of the higher-order network in jupyter
        """
        from IPython.core.display import display, HTML
        display(HTML(self._to_html(require=require)))

    def write_html(self, filename, width=600, height=600):
        html = self._to_html(width=width, height=height, require=False)
        with open(filename, 'w+') as f:
            f.write(html)
