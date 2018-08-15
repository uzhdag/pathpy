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

from collections import defaultdict
import numpy as _np
import scipy.sparse as _sparse

from pathpy.utils.exceptions import PathsTooShort
from pathpy.classes.network import Network


class HigherOrderNetwork(Network):
    """
    A higher-order graphical model of path statistics with order k.

    Attributes:
    -----------

    edges: dict
        In a higher-order network, edge weights as well as in- and out
        weights of nodes are numpy arrays consisting of two weight components [w0, w1].
        w0 counts the weight of an edge based on its occurrence in a subpaths
        while w1 counts the weight of an edge based on its occurrence in
        a longest path. As an illustrating example, consider the single
        path a -> b -> c. In the first-order network, the weights of edges
        # (a,b) and (b,c) are both (1,0). In the second-order network, the
        weight of edge (a-b, b-c) is (0,1).
        Here, we will store these weights (as well as in- and out-degrees in
        node and edge attributes)
    """

    def __init__(self, paths, k=1, null_model=False, separator=None):
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
        null_model: bool
            For the default value False, link weights capture the frequencies of paths of length k 
            in the underlying paths object. If True, link weights capture expected frequencies
            under the assumption of independent links (i.e. corresponding to a first-order
            Markov model).
        separator: str
            The separator character to be used in higher-order node names. If this parameter 
            is not specified, the separator character of the underlying paths object will be 
            used.
        """
        assert not null_model or (null_model and k > 1)

        if not (paths.paths.keys() and max(paths.paths.keys()) >= k):
            msg = ('Constructing a model of order %d requires paths of at least length %d, '
                   'found paths of max length %d ' % (k, k, max(paths.paths.keys())))
            raise PathsTooShort(msg)

        super().__init__(directed=True)

        # The order of this HigherOrderNetwork
        self.order = k

        # The paths object used to generate this instance
        self.paths = paths

        self.is_null_model = null_model

        # The separator character used to label higher-order nodes.
        # For separator '-', the name of a second-order node will be 'a-b'.
        if separator is None:
            self.separator = paths.separator
        else:
            self.separator = separator

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
                    v = self.separator.join(key[0:-1])
                    w = self.separator.join(key[1:])
                    self.add_node(v, inweight=_np.array([0.0, 0.0]), outweight=_np.array([0.0, 0.0]))
                    self.add_node(w, inweight=_np.array([0.0, 0.0]), outweight=_np.array([0.0, 0.0]))
                    # add weight val to directed edge (v,w)
                    self.add_edge(v, w, weight=val)
            
                # create all possible higher-order nodes
                if k > 1:

                    nodes = HigherOrderNetwork.generate_possible_paths(g1, k-1)
                    for p in nodes:
                        v = p[0]                
                        for l in range(1, k):
                            v = v + self.separator + p[l]
                        
                        # create nodes and make sure that in- and out-weights are numpy arrays
                        if v not in self.nodes:
                            self.add_node(v, inweight=_np.array([0.0, 0.0]), outweight=_np.array([0.0, 0.0]))

            # Note: For all sequences of length k which (i) have never been observed, but
            #       (ii) do actually represent paths of length k in the first-order
            #       network, we may want to include some 'escape' mechanism along the
            #       lines of (Cleary and Witten 1994)

        else:
            # generate the *expected* frequencies of all possible
            # paths of length k based on independently occurring (first-order) links
            possible_paths = HigherOrderNetwork.generate_possible_paths(g1, k)

            # validate that the number of unique paths corresponds to the sum
            # of entries in A**k
            A_sum = _np.sum(A ** k)
            assert A_sum == len(possible_paths), \
                'Expected {ak} paths but got {re}'.format(ak=A_sum, re=len(possible_paths))

            T = g1.transition_matrix(include_subpaths=True)

            # create nodes and links in k-th-order null model
            for p in possible_paths:
                # create higher-order nodes (a,b,c,...) and (b,c,d,...)
                v = p[0]
                for l in range(1, k):
                    v = v + self.separator + p[l]

                w = p[1]
                for l in range(2, k + 1):
                    w = w + self.separator + p[l]                    

                # create nodes and make sure that in- and out-weights are numpy arrays
                self.add_node(v, inweight=_np.array([0.0, 0.0]), outweight=_np.array([0.0, 0.0]))
                self.add_node(w, inweight=_np.array([0.0, 0.0]), outweight=_np.array([0.0, 0.0]))

                # In the null model, we encode a first-order Markov process in a k-th-order
                # model. For the transition probabilities e.g. (a,b) -> (b,c) in a second-order
                # null model, we simply use the first-order transition probabilities, i.e. P(b->c).             
                # Note that transition_matrices are transposed (!)
                v_1, w_1 = g1_node_mapping[p[-2]], g1_node_mapping[p[-1]]
                p_vw = T[w_1, v_1]

                # We use first-order transition probabilities to create an expected frequency 
                # of paths of length k. For a path (a,b,c) we use the count of (a,b) and 
                # "distribute" it to all possible paths (a,b,*) according to the first-order 
                # transition probabilities of (b,*)
                expected_vw = paths.paths[k-1][p[:k]].sum() * p_vw

                self.add_edge(v, w, weight = _np.array([0, expected_vw]))

        # Compute degrees of freedom of models
        if k == 0:
            # for a zero-order model, we just fit node probabilities
            # (excluding the special 'start' node)
            # Since probabilities must sum to one, the effective degree
            # of freedom is one less than the number of nodes
            # This holds for both the paths and the ngrams model
            self.dof_paths = self.ncount() - 2
            self.dof_ngrams = self.ncount() - 2
        else:
            # for a first-order model, self is the first-order network
            if k == 1:
                g1 = self
                A = g1.adjacency_matrix(include_subpaths=True, weighted=False,
                                        transposed=True)

            # Degrees of freedom in a higher-order ngram model
            s = g1.ncount()

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


    @staticmethod
    def generate_possible_paths(network, k):
        """ Returns all paths of length k that can
        possibly exist in a given network """

        assert k > 0, 'This function only calculates possible paths of length k > 0'

        # start with edges, i.e. paths of length one
        possible_paths = list(network.edges.keys())

        # extend all of those paths by an edge k-1 times
        for _ in range(k - 1):
            E_new = list()
            for e1 in possible_paths:
                for e2 in network.edges:
                    if e1[-1] == e2[0]:
                        p = e1 + (e2[1],)
                        E_new.append(p)
            possible_paths = E_new
        return possible_paths


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

    def first_order_nodes(self):
        """
        Returns a set of nodes projected to a first-order network
        """
        nodes = set()
        for v in self.nodes:
            for w in self.higher_order_node_to_path(v):
                nodes.add(w)
        return nodes


    def higher_order_node_to_path(self, node):
        """Helper function that transforms a node in a higher-order network of order k
        into a corresponding path of length k-1. For a higher-order node 'a-b-c-d'
        this function will return ('a','b','c','d')

        Parameters
        ----------
        node: str
            The higher-order node to be transformed to a path.

        Returns
        -------
        tuple
        """
        return tuple(node.split(self.separator))

    def path_to_higher_order_nodes(self, path, k=None):
        """Helper function that transforms a path of first-order nodes into a
        sequence of k-order nodes using the separator character of the
        HigherOrderNetwork instance

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
        assert len(path) >= k, 'Error: Path length must be at least k'

        if k == 0 and len(path) == 1:
            return ['start', path[0]]

        return [self.separator.join(path[n:n + k]) for n in range(len(path) - k + 1)]

    def node_to_name_map(self):
        """Returns a dictionary that can be used to map node names to matrix/vector indices"""
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
            'Nodes:\t\t\t\t{ncount}\n'
            'Links:\t\t\t\t{ecount}\n'
            'Total weight (subpaths/longest paths):\t{sub_w}/{uni_w}\n'
        )
        summary = summary_fmt.format(
            order=self.order, ncount=self.ncount(), ecount=self.ecount(),
            sub_w=self.total_edge_weight()[0], uni_w=self.total_edge_weight()[1]
        )
        return summary

    def likelihood(self, paths, log=True):
        """
        Calculates the likelihood of this higher-order model under the observed path 
        statistics given in paths.
        """
        if log:
            L = 0.0
        else: 
            L = 1.0
        T = self.transition_matrix()
        node_map = self.node_to_name_map()
        for l in paths.paths:
            if l>=self.order:
                for p in paths.paths[l]:
                    if paths.paths[l][p][1]>0:
                        if log:
                            path_L = 0.0
                        else:
                            path_L = 1.0
                        node_sequence = self.path_to_higher_order_nodes(p)
                        prev = node_sequence[0]
                        for n in node_sequence[1:]:
                            if log:
                                path_L += _np.log(T[node_map[n], node_map[prev]])
                            else:
                                path_L *= T[node_map[n], node_map[prev]]
                            prev = n
                        if log:
                            L += path_L * paths.paths[l][p][1]
                        else: 
                            L *= path_L ** paths.paths[l][p][1]
        return L


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

        shape = (self.ncount(), self.ncount())
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

        shape = self.ncount(), self.ncount()
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
        identity_matrix = _sparse.identity(self.ncount())

        return identity_matrix - transition_matrix
