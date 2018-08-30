"""
Algorithms to generate random graphs according to various models
"""
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

from pathpy.utils import PathpyNotImplemented
from pathpy.classes import Network
from pathpy.classes import TemporalNetwork


def is_graphic_sequence(degree_sequence, directed=False):
    """
    Checks whether a degree sequence is graphic, i.e. whether 
    there exists an undirected or directed graph that has the
    given degree sequence. A graphic degree sequence is the 
    precondition to apply the Molloy-Reed random graph generation.

    Parameters
    ----------
    degree_sequence: list or tuple
        the degree sequence for which to test the graphic property

    directed: bool
        whether or not to check for the sequence of a directed 
        graph

    Returns
    -------
    bool
    """

    # fast check using result of Behzad and Chartrand 1967,
    # showing that a graphic degree sequence msut have at least one
    # degree occurring twice
    if len(set(degree_sequence)) == len(degree_sequence):
        return False
    
    S = sum(degree_sequence)
    n = len(degree_sequence)

    # for undirected graphs, the sum of degrees must be even
    if not directed and S%2 != 0:
        return False

    # check necessary and sufficient condition given by Erdös and Gallai (1960)
    # see http://mathworld.wolfram.com/GraphicSequence.html
    for r in range(1, n):
        M = 0
        S = 0
        for i in range(1, r+1):
            S += degree_sequence[i-1]
        for i in range(r+1, n+1):
            M += min(r, degree_sequence[i-1])
        if S > r * (r-1) + M:
            return False

    return True


def molloy_reed(degree_sequence, node_names=None, self_loops=True):
    """
    Generates a random undirected network with a given degree sequence.
    The generated network is guaranteed to have the given degree sequence.
    Multiple edges are forbidden in the network generation. Raises an exception
    if the given degree sequence is not graphic, i.e. if no possible simple
    graph exisits with the desired degree sequence.

    Parameters:
    -----------
    degree_sequence: list or tuple
        The degree sequence of the randomly generated network. The degree
        sequence must have at least two entries. The sequence must be graphic,
        or an exception will be raised.
    node_names: list or tuple
        Node names to be used in the network creation. If None (default)
        nodes will be numbered from 0 to n-1, where n is the length of
        the degree sequence.
    self_loops: bol
        Whether or not to allow the generation of self_loops. Default is True.
    """    

    assert is_graphic_sequence(degree_sequence, directed = False), 'Error: degree sequence is not graphic'

    n = len(degree_sequence)

    if node_names is None:
        node_names = [str(x) for x in range(n)]
    assert len(node_names) >= n, 'Error: Number of node names not matching degree sequence length'

    network = Network(directed=False)

    # generate a list with node stubs
    stubs = []
    for i in range(n):
        for j in range(degree_sequence[i]):
            stubs.append(str(node_names[i]))

    while len(stubs) > 1:
        random_nodes = _np.random.choice(stubs, size=2, replace=False)
        (v, w) = (random_nodes[0], random_nodes[1])
        if (v, w) not in network.edges and (self_loops or v != w):
            network.add_edge(v, w)
            stubs.remove(v)
            if v != w: # ensures that self-loops are counted as degree 1
                stubs.remove(w)
        elif network.ecount()>0: # randomly remove edge
            edges = list(network.edges)
            edge = edges[_np.random.choice(len(edges))]
            network.remove_edge(edge[0], edge[1])
            stubs.append(edge[0])
            stubs.append(edge[1])
    return network


def random_k_regular(n, k, self_loops=True, node_names=None):
    """
    Generates an undirected random k-regular network, i.e. a random
    network where all nodes have exactly degree k. A call to this
    function is equivalent to generating a random network with a 
    given degree sequence [k]*n.

    Parameters:
    -----------
    n: int
        The number of nodes in the generated network.
    k: int
        The degree of all nodes
    self_loops: bool
        Whether or not to allow self_loops in the network generation. 
        Default is True.
    node_names: list or sequence
        Node names to be used in the network creation. If None (default)
        nodes will be numbered from 0 to n-1.
    """
    assert n*k%2 == 0, 'Error: parameters lead to non-graphic degree sequence.'
    return molloy_reed([k]*n, self_loops, node_names)


def erdoes_renyi_gnm(n, m, node_names=None, self_loops=True, directed=False, temporal=False):
    """
    """
    if node_names is None:
        node_names = [str(x) for x in range(n)]

    assert len(node_names) >= n, 'Error: Number of node names not matching degree sequence length'

    if not temporal:
        network = Network(directed=directed)
    else:
        network = TemporalNetwork()

    # generate nodes
    if not temporal:
        for i in range(n):
            network.add_node(str(node_names[i]))

    time = -1
    edges = defaultdict(lambda: False)

    # add edges
    while network.ecount() < m:
        edge = _np.random.choice(node_names[:n], size=2, replace=self_loops)
        if not edges[(edge[0], edge[1])]:
            edges[(edge[0], edge[1])] = True
            if not directed:
                edges[(edge[1], edge[0])] = True
            if not temporal:
                network.add_edge(edge[0], edge[1])                
            else:
                time += 1
                network.add_edge(edge[0], edge[1], time, directed=directed)
    return network


def erdoes_renyi_gnp(n, p, node_names=None, self_loops=True, directed=False, temporal=False):
    """
    """
    if node_names is None: 
        node_names = [str(x) for x in range(n)]
    assert len(node_names) >= n, 'Error: Number of node names not matching degree sequence length'

    if not temporal:
        network = Network(directed=directed)
    else:
        network = TemporalNetwork()

    # make sure that isolated nodes exist
    if not temporal:
        for i in range(n):
            network.add_node(str(node_names[i]))

    time = -1

    # add edges
    for i in range(n):
        for j in range(i+1):
            if i != j or self_loops:                
                if _np.random.rand() <= p:
                    if not temporal:
                        network.add_edge(node_names[i], node_names[j])
                    else:
                        time += 1
                        network.add_edge(node_names[i], node_names[j], time, directed=directed)
    return network


def watts_strogatz(n, p, node_names=None, directed=False):
    """
    """
    if node_names is None: 
        node_names = [str(x) for x in range(n)]
    assert len(node_names) >= n, 'Error: Number of node names not matching degree sequence length'

    raise PathpyNotImplemented('Watts-Strogatz model is not implemented yet')


def barabasi_albert(n, n_init, k=1, node_names=None, directed=False, temporal=False):
    """
    """

    if node_names is None:
        node_names = [str(x) for x in range(n)]
    assert len(node_names) >= n, 'Error: Number of node names not matching degree sequence length'
    if not temporal:
        network = Network(directed=directed)
    else:
        network = TemporalNetwork()

    endpoints = []
    time =  0

    # initial network
    for i in range(n_init):
        for j in range(n_init):
            if i < j:
                if not temporal:
                    network.add_edge(str(node_names[i]), str(node_names[j]))
                else:
                    network.add_edge(str(node_names[i]), str(node_names[j]), time, directed=directed)
                endpoints.append(str(node_names[i]))
                endpoints.append(str(node_names[j]))

    for i in range(n_init, n):
        time += 1
        # TODO: for k>1 we can choose different stubs of the same node in one step!
        targets = _np.random.choice(endpoints, size=k, replace=False)
        for t in targets:
            if not temporal:
                network.add_edge(str(node_names[i]), t)
            else:
                network.add_edge(str(node_names[i]), t, time, directed=directed)
            endpoints.append(str(node_names[i]))
            endpoints.append(t)
        
    return network