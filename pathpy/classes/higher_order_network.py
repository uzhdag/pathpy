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

from collections import defaultdict
import numpy as _np
import scipy.sparse as _sparse

from pathpy.utils.exceptions import PathsTooShort
from pathpy.classes.network import Network
from pathpy.algorithms import shortest_paths


class HigherOrderNetwork(Network):
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
            transition probability transition_matrices['v_k', 'v_k+1'] in the first order network.
            For method='KOrderPi' the entry pi['v1-...-v_k'] in the stationary
            distribution of the k-order network is used instead.
        lanczos_vecs: int
        maxiter: int
        """
        assert not null_model or (null_model and k > 1)

        assert method in ['FirstOrderTransitions', 'KOrderPi'], \
            'Error: unknown method to build null model'

        if not (paths.paths.keys() and max(paths.paths.keys()) >= k):
            msg = ('Constructing a model of order %d requires paths of at least length %d, '
                   'found paths of max length %d ' % (k, k, max(paths.paths.keys())))
            raise PathsTooShort(msg)

        super().__init__(directed=True)

        # The order of this HigherOrderNetwork
        self.order = k

        # The paths object used to generate this instance
        self.paths = paths

        # The separator character used to label higher-order nodes.
        # For separator '-', the name of a second-order node will be 'a-b'.
        self.separator = separator

        # NOTE: In a higher-order network, edge weights as well as in- and out
        # weights of nodes are numpy arrays consisting of two weight components [w0, w1].
        # w0 counts the weight of an edge based on its occurrence in a subpaths
        # while w1 counts the weight of an edge based on its occurrence in
        # a longest path. As an illustrating example, consider the single
        # path a -> b -> c. In the first-order network, the weights of edges
        # (a,b) and (b,c) are both (1,0). In the second-order network, the
        # weight of edge (a-b, b-c) is (0,1).
        # Here, we will store these weights (as well as in- and out-degrees in
        # node and edge attributes)

        if k > 1:
            # For k>1 we need the first-order network to generate the null model
            # and calculate the degrees of freedom

            # TODO: For a multi-order model, the first-order network is generated multiple
            # times! Make this more efficient
            g1 = HigherOrderNetwork(paths, k=1)
            g1_node_mapping = g1.node_to_name_map()
            A = g1.adjacency_matrix(include_subpaths=True, weighted=False,
                                    transposed=True)

        if not null_model:
            # Calculate the frequency of all paths of
            # length k, generate k-order nodes and set
            # edge weights accordingly
            iterator = paths.paths[k].items()

            if k == 0:
                # For a 0-order model, we generate a "dummy" start node
                self.add_node('start', inweight=_np.array([0.0, 0.0]), outweight=_np.array([0.0, 0.0]))
                for key, val in iterator:
                    w = key[0]
                    # add weight val to edge ('start', w)
                    self.add_node(w, inweight=_np.array([0.0, 0.0]), outweight=_np.array([0.0, 0.0]))
                    self.add_edge('start', w, weight=val)
            else:
                for key, val in iterator:
                    # Generate names of k-order nodes v and w
                    v = separator.join(key[0:-1])
                    w = separator.join(key[1:])
                    self.add_node(v, inweight=_np.array([0.0, 0.0]), outweight=_np.array([0.0, 0.0]))
                    self.add_node(w, inweight=_np.array([0.0, 0.0]), outweight=_np.array([0.0, 0.0]))
                    # add weight val to directed edge (v,w)
                    self.add_edge(v, w, weight=val)


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
                pi_k = Network.leading_eigenvector(
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
                self.add_node(v, inweight=_np.array([0.0, 0.0]), outweight=_np.array([0.0, 0.0]))
                self.add_node(w, inweight=_np.array([0.0, 0.0]), outweight=_np.array([0.0, 0.0]))

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
                            self.add_edge(v, w, weight = _np.array([0, _np.real(eigen_value)]))

                # Solution B: Use relative edge weight in first-order network
                # Note that A is *not* transposed
                # self.edges[(v,w)] = A[(g1.nodes.index(p[-2]),g1.nodes.index(p[-1]))]
                # / A.sum()

                # Solution C: Use transition probability in first-order network
                # Note that transition_matrices is transposed (!)
                elif method == 'FirstOrderTransitions':
                    v_i, w_i = g1_node_mapping[p[-1]], g1_node_mapping[p[-2]]
                    p_vw = T[v_i, w_i]
                    self.add_edge(v, w, weight = _np.array([0, p_vw]))

                # Solution D: calculate k-path weights based on entries of squared
                # k-1-order adjacency matrix

                # Note: Solution B and C are equivalent

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


    def total_edge_weight(self):
        """ Returns the sum of all edge weights """
        if self.edges:
            return sum(e['weight'] for e in self.edges.values())
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

        >>> from pathpy import Paths
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

    def distance_matrix_first_order(self):
        """
        Projects a distance matrix from a higher-order to first-order nodes, while path
        lengths are calculated based on the higher-order topology
        """

        dist = shortest_paths.distance_matrix(self)
        dist_first = defaultdict(lambda: defaultdict(lambda: _np.inf))

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

    def summary(self):
        """Returns a string summary of this higher-order
           network
        """
        summary_fmt = (
            'Higher-order network of order k = {order}\n'
            '\n'
            'Nodes:\t\t\t\t{vcount}\n'
            'Links:\t\t\t\t{ecount}\n'
            'Total weight (subpaths/longest paths):\t{sub_w}/{uni_w}\n'
        )
        summary = summary_fmt.format(
            order=self.order, vcount=self.vcount(), ecount=self.ecount(),
            sub_w=self.total_edge_weight()[0], uni_w=self.total_edge_weight()[1]
        )
        return summary


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
                data = _np.array([float(self.edges[e]['weight'].sum()) for e in self.edges])
            else:
                data = _np.array([float(self.edges[e]['weight'][1]) for e in self.edges])

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
            D = {n: self.nodes[n]['outweight'].sum() for n in self.nodes}
        else:
            D = {n: self.nodes[n]['outweight'][1] for n in self.nodes}

        node_to_coord = self.node_to_name_map()

        for (s, t) in self.edges:
            # either s->t has been observed as a longest path, or we are interested in
            # subpaths as well

            # the following makes sure that we do not accidentally consider zero-weight
            # edges (automatically added by default_dic)
            unique_weight = self.edges[(s, t)]['weight'][1]
            subpath_weight = self.edges[(s, t)]['weight'][0]
            is_valid = (unique_weight > 0 or (include_subpaths and subpath_weight > 0))
            if is_valid:
                row.append(node_to_coord[t])
                col.append(node_to_coord[s])
                if include_subpaths:
                    count = self.edges[(s, t)]['weight'].sum()
                else:
                    count = self.edges[(s, t)]['weight'][1]
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
