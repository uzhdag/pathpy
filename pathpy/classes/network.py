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

from pathpy.utils import Log, Severity


class Network:
    """
    Instances of this class capture a graph or network
    that can be directed, undirected, unweighted or weighted
    """

    def __init__(self, directed=False):
        """
        Generates an empty network.
        """

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

    @classmethod
    def read_edges(cls, filename, separator=',', weighted=False, directed=False):
        """
        Reads a network from an edge list file

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

        Returns
        -------
        Network
            a ``Network`` object obtained from the edgelist
        """
        n = cls(directed)

        with open(filename, 'r') as f:
            Log.add('Reading edge list ... ')
            for n, line in enumerate(f):
                fields = line.rstrip().split(separator)
                assert len(fields) >= 2, 'Error: malformed line: {0}'.format(line)

                if weighted:
                    n.add_edge(fields[0], fields[1], weight=int(fields[2]))
                else:
                    n.add_edge(fields[0], fields[1])

        Log.add('finished.')

        return n


    @classmethod
    def from_sqlite(cls, cursor, directed=True):
        """Reads links from an SQLite cursor and returns a new instance of
        the class Network. The cursor is assumed to refer to a table that
        minimally has two columns

                source target

        and where each row refers to a link. Any additional columns will be used as
        edge properties

        Important: Since columns are accessed by name this function requires that a
        row factory object is set for the SQLite connection prior to cursor creation,
        i.e. you should set

                connection.row_factory = sqlite3.Row

        Parameters
        ----------
        cursor:
            The SQLite cursor to fetch rows
        directed: bool

        Returns
        -------

        """
        n = cls(directed=directed)

        assert cursor.connection.row_factory, \
            'Cannot access columns by name. Please set ' \
            'connection.row_factory = sqlite3.Row before creating DB cursor.'

        Log.add('Retrieving links from database ...')

        for row in cursor:
            n.add_edge(str(row['source']), str(row['target']))

        return n


    def add_node(self, v, **kwargs):
        """
        Adds a node to a network
        """
        if v not in self.nodes:
            self.nodes[v] = {**self.nodes[v], **kwargs}

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
        """
        Removes a node from the network
        """
        if v in self.nodes:
            # remove all incident edges and update neighbors
            if not self.directed:
                for w in self.successors[v]:
                    edge = (v, w)
                    self.nodes[w]['degree'] -= 1
                    self.nodes[w]['inweight'] -= self.edges[edge]['weight']
                    self.nodes[w]['outweight'] -= self.edges[edge]['weight']
                    self.successors[w].remove(v)
                    self.predecessors[w].remove(v)
                    del self.edges[edge]
            else:
                for w in self.successors[v]:
                    self.nodes[w]['indegree'] -= 1
                    self.nodes[w]['inweight'] -= self.edges[(v, w)]['weight']
                    self.predecessors[w].remove(v)
                    del self.edges[(v, w)]
                for w in self.predecessors[v]:
                    self.nodes[w]['outdegree'] -= 1
                    self.nodes[w]['outweight'] -= self.edges[(w, v)]['weight']
                    self.successors[w].remove(v)
                    del self.edges[(w, v)]
            del self.nodes[v]
            del self.successors[v]
            del self.predecessors[v]


    def add_edge(self, v, w, **kwargs):
        """
        Adds an edge to a network
        """
        self.add_node(v)
        self.add_node(w)

        e = (v, w)

        self.edges[e] = {**self.edges[e], **kwargs}

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
        """
        Returns all nodes that satisfy a given condition
        """
        return [n for n in self.nodes if select_node(self.nodes[n])]

    def find_edges(self, select_nodes=lambda v, w: True, select_edges=lambda e: True):
        """
        Returns all edges that satisfy a given condition. Edges can be selected based
        on attributes of the adjacent nodes as well as attributes of the edge
        """
        return [e for e in self.edges if (select_nodes(self.nodes[e[0]], self.nodes[e[1]]) and select_edges(self.edges[e]))]

    def vcount(self):
        """ Returns the number of nodes """
        return len(self.nodes)

    def ecount(self):
        """ Returns the number of links """
        return len(self.edges)

    def total_edge_weight(self):
        """ Returns the sum of all edge weights """
        if self.edges:
            return _np.sum(e['weight'] for e in self.edges.values())
        return 0

    def node_to_name_map(self):
        """Returns a dictionary that can be used to map nodes to matrix/vector indices"""
        return {v: idx for idx, v in enumerate(self.nodes)}

    def adjacency_matrix(self, weighted=True, transposed=False):
        """Returns a sparse adjacency matrix of the higher-order network. By default,
        the entry corresponding to a directed link source -> target is stored in row s and
        column t and can be accessed via A[s,t].

        Parameters
        ----------
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
            data = _np.array([float(e['weight']) for e in self.edges.values()])

        shape = (self.vcount(), self.vcount())
        return _sparse.coo_matrix((data, (row, col)), shape=shape).tocsr()


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
        # calculate weighted out-degrees
        D = {n: self.nodes[n]['outweight'] for n in self.nodes}

        node_to_coord = self.node_to_name_map()

        for (s, t) in self.edges:
            # either s->t has been observed as a longest path, or we are interested in
            # subpaths as well

            # the following makes sure that we do not accidentally consider zero-weight
            # edges (automatically added by default_dic)
            weight = self.edges[(s, t)]['weight']
            if weight > 0:
                row.append(node_to_coord[t])
                col.append(node_to_coord[s])
                assert D[s] > 0, \
                    'Encountered zero out-degree for node "{s}" ' \
                    'while weight of link ({s}, {t}) is non-zero.'.format(s=s, t=t)
                prob = weight / D[s]
                if prob < 0 or prob > 1:  # pragma: no cover
                    raise ValueError('Encountered transition probability {p} outside '
                                     '[0,1] range.'.format(p=prob))
                data.append(prob)

        data = _np.array(data)
        data = data.reshape(data.size, )

        shape = self.vcount(), self.vcount()
        return _sparse.coo_matrix((data, (row, col)), shape=shape).tocsr()


    def laplacian_matrix(self):
        """
        Returns the transposed normalized Laplacian matrix corresponding to the network.

        Parameters
        ----------

        Returns
        -------

        """
        transition_matrix = self.transition_matrix()
        identity_matrix = _sparse.identity(self.vcount())

        return identity_matrix - transition_matrix


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


    def summary(self):
        """Returns a string containing basic summary statistics of this network instance
        """
        summary_fmt = (
            '{directed_str} network\n'
            '\n'
            'Nodes:\t\t\t\t{vcount}\n'
            'Links:\t\t\t\t{ecount}\n'
        )
        if self.directed:
            directed_str = 'Directed'
        else:
            directed_str = 'Undirected'
        summary = summary_fmt.format(directed_str=directed_str, vcount=self.vcount(), ecount=self.ecount())
        return summary

    def __str__(self):
        """Returns the default string representation of this graphical model instance"""
        return self.summary()

    def _to_html(self, width=600, height=600, use_requirejs=True):
        import json
        import os
        from string import Template

        # prefix nodes starting with number
        def fix_node_name(v):
            if v[0].isdigit():
                return "n_" + v
            return v

        network_data = {
            'nodes': [{'id': fix_node_name(v), 'group': 1} for v in self.nodes],
            'links': [
                {'source': fix_node_name(e[0]),
                 'target': fix_node_name(e[1]),
                 'value': 1} for e in self.edges.keys()
            ]
        }

        import string
        import random

        div_id = "".join(random.choice(string.ascii_letters) for x in range(8))

        if not use_requirejs:
            template_file = 'higherordernet.html'
        else:
            template_file = 'higherordernet_require.html'

        module_dir = os.path.dirname(os.path.realpath(__file__))
        html_dir = os.path.join(module_dir, os.path.pardir, 'html_templates')

        with open(os.path.join(html_dir, template_file)) as f:
            html_str = f.read()

        html_template = Template(html_str)

        return html_template.substitute({
            'network_data': json.dumps(network_data),
            'width': width,
            'height': height,
            'div_id': div_id})

    def _repr_html_(self, use_requirejs=True):
        """
        display an interactive D3 visualisation of the higher-order network in jupyter
        """
        from IPython.core.display import display, HTML
        display(HTML(self._to_html(use_requirejs=use_requirejs)))

    def write_html(self, filename, width=600, height=600):
        html = self._to_html(width=width, height=height, use_requirejs=False)
        with open(filename, 'w+') as f:
            f.write(html)


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
