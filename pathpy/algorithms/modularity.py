"""
Spectral and information-theoretic measures that can be calculated
based on higher-order models of paths.
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

import numpy as np
import scipy.sparse.linalg as sla

from pathpy import HigherOrderNetwork
from pathpy.utils import Log, Severity
from pathpy import Paths
from pathpy.utils import PathpyError


__all__ = ['q', 'q_max', 'assortativity_coeff', 'find_communities']

def q(network, C=None, delta=None):
    assert C is None or delta is None, 'Error: Cannot use clustering and delta-function simultaneously'

    m = network.ecount()
    A = network.adjacency_matrix(weighted=False)
    idx = network.node_to_name_map()
    q = 0.0
    for v in network.nodes:
        for w in network.nodes:
            if (C != None and C[v] == C[w]) or (delta != None and delta(v,w)):
                q += A[idx[v], idx[w]] - network.nodes[v]['degree']*network.nodes[w]['degree']/(2*m)
    q /= 2*m
    return q


def q_max(network, C=None, delta=None):
    assert C is None or delta is None, 'Error: Cannot use clustering and delta-function simultaneously'

    m = network.ecount()
    idx = network.node_to_name_map()
    q = 0.0
    for v in network.nodes:
        for w in network.nodes:
            if (C != None and C[v] == C[w]) or (delta != None and delta(v,w)):
                q -= network.nodes[v]['degree']*network.nodes[w]['degree']/(2*m)
    q /= 2*m
    return q

def assortativity_coeff(network, C=None):
    C, q_opt = find_communities(network)
    return q_opt/q_max(network, C)


def q_merge(network, C, merge=None):
    m = network.ecount()
    n = network.ncount()    
    A = network.adjacency_matrix(weighted=False)
    idx = network.node_to_name_map()
    q = 0.0
    for v in network.nodes:
        for w in network.nodes:
            if C[v] == C[w] or (merge is not None and C[v] in merge and C[w] in merge):
                q += A[idx[v], idx[w]] - network.nodes[v]['degree']*network.nodes[w]['degree']/(2*m)
    q /= 2*m
    return q


def find_communities(network, iterations=100):
    # start with each node being in a separate cluster
    C = {}
    community_to_nodes = {}
    c = 0
    for n in network.nodes:
        C[n] = c
        community_to_nodes[c] = set([n])
        c += 1
    q_current = q(network, C)
    communities = list(C.values())
    
    for i in range(iterations):
        # randomly choose two communities
        x, y = np.random.choice(communities, size=2)
        # check Q of merged communities
        q_new = q_merge(network, C, merge=set([x, y]))
        if q_new > q_current:
            # actually merge the communities
            for n in community_to_nodes[x]:
                C[n] = y
            community_to_nodes[y] = community_to_nodes[y] | community_to_nodes[x]
            q_current = q_new
            communities.remove(x)
            del community_to_nodes[x]
    return C, q_current