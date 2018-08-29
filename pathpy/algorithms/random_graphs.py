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

from pathpy.utils import Log, Severity
from pathpy.utils import PathpyNotImplemented
from pathpy.classes import Network


def molloy_reed(degree_sequence, self_loops=True, node_names=None):

    assert len(degree_sequence)>1, 'Error: degree sequence must contain at least two entries'

    assert sum(degree_sequence)%2 == 0 and sum(degree_sequence) <= len(degree_sequence)**2, 'Error: degree sequence is not graphical'

    if node_names is None: 
        node_names = [str(x) for x in range(len(degree_sequence))]
    assert len(node_names)>=len(degree_sequence), 'Error: Number of node names not matching length of degree sequence'

    if not self_loops:
        raise PathpyNotImplemented('Network generation without self-loops is not implemented yet')
    

    network = Network(directed=False)

    # generate node stubs
    stubs = []
    for i in range(len(degree_sequence)):
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
    
    return molloy_reed([k]*n, self_loops, node_names)


def erdoes_renyi_gnm(n, m, node_names=None, self_loops=True, directed=False):
    
    if node_names is None: 
        node_names = [str(x) for x in range(n)]

    assert len(node_names)>=n, 'Error: Number of node names not matching length of degree sequence'

    network = Network(directed=directed)

    # generate nodes
    for i in range(n):
        network.add_node(str(node_names[i]))
    node_list = [v for v in network.nodes]

    # add edges
    while network.ecount() < m:
        edge = _np.random.choice(node_list, size=2, replace=self_loops)
        if (edge[0], edge[1]) not in network.edges:
            network.add_edge(edge[0], edge[1])
    return network


def erdoes_renyi_gnp(n, p, node_names=None, self_loops=True, directed=False):    
    """
    """
    if node_names is None: 
        node_names = [str(x) for x in range(n)]
    assert len(node_names)>=n, 'Error: Number of node names not matching length of degree sequence'

    network = Network(directed=directed)

    # generate nodes
    for i in range(n):
        network.add_node(str(node_names[i]))        
    
    # add edges
    for i in range(n):
        for j in range(n):
            if i != j or self_loops:
                if _np.random.rand() <= p:
                    network.add_edge(node_names[i], node_names[j])        
    return network


def watts_strogatz(n, p, node_names=None, directed=False):
    """
    """
    if node_names is None: 
        node_names = [str(x) for x in range(n)]
    assert len(node_names)>=n, 'Error: Number of node names not matching length of degree sequence'

    raise PathpyNotImplemented('Watts-Strogatz model is not implemented yet')


def barabasi_albert(n, n_init, k=1, node_names=None, directed=False):
    """
    """

    if node_names is None:
        node_names = [str(x) for x in range(n)]
    assert len(node_names)>=n, 'Error: Number of node names not matching length of degree sequence'

    network = Network(directed=directed)

    # initial network
    for i in range(n_init):
        for j in range(n_init):
            if i < j:
                network.add_edge(str(node_names[i]), str(node_names[j]))

    node_list = [v for v in network.nodes]

    for i in range(n_init, n):
        targets = _np.random.choice(node_list, size=k, replace=False)
        for t in targets:
            network.add_edge(str(node_names[i]), t)
        node_list.append(str(node_names[i]))
    return network
