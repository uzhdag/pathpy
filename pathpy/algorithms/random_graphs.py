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


def molloy_reed(degree_sequence, self_loops=True, node_names=None):

    n = len(degree_sequence)

    assert n > 1, 'Error: degree sequence must contain at least two entries'

    assert sum(degree_sequence)%2 == 0 and \
            sum(degree_sequence) <= n**2, 'Error: degree sequence is not graphical'

    if node_names is None:
        node_names = [str(x) for x in range(n)]
    assert len(node_names) >= n, 'Error: Number of node names not matching degree sequence length'

    if not self_loops:
        raise PathpyNotImplemented('Molly-Reed model without self-loops is not implemented yet')

    network = Network(directed=False)

    # generate node stubs
    stubs = []
    for i in range(n):
        for j in range(degree_sequence[i]):
            stubs.append(str(node_names[i]))

    while len(stubs) > 1:
        random_nodes = _np.random.choice(stubs, size=2, replace=False)
        (v, w) = (random_nodes[0], random_nodes[1])
        if (v, w) not in network.edges:
            network.add_edge(v, w)
            stubs.remove(v)
            stubs.remove(w)
    return network


def random_k_regular(n, k, self_loops=True, node_names=None):
    """
    """
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
                network.add_edge(edge[0], edge[1], time)
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
                        network.add_edge(node_names[i], node_names[j], time)
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
                    network.add_edge(str(node_names[i]), str(node_names[j]), time)
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
                network.add_edge(str(node_names[i]), t, time)
            endpoints.append(str(node_names[i]))
            endpoints.append(t)        
        
    return network