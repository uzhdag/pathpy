"""
Algorithms to calculate connected components in networks.
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
import scipy.sparse.linalg as _sla

from pathpy import Network
from pathpy.utils import Log, Severity
from pathpy.utils import PathpyError

__all__ = ['connected_components']


def connected_components(network, lanczos_vecs=None, maxiter=None):
    """
    Calculates connected components based on the spectrum of the Laplacian matrix
    """
    L = network.laplacian_matrix(weighted=True)
    n = network.ncount()-2
    if lanczos_vecs is None:
        lanczos_vecs = min(n, max(2*n + 1, 20))
    if maxiter is None:
        maxiter = n*10
    vals, vecs = _sla.eigs(L, k=n, which="SM", return_eigenvectors=True)

    components = defaultdict(set)
    c = 0

    # use eigenvectors of zero eigenvalues to map nodes to components
    for i in range(n):
        if _np.isclose(vals[i], 0, atol=1.e-12):
            min_v = _np.min(vecs[:,i])
            for i in _np.where(_np.isclose(vecs[:,i], min_v))[0]:
                components[c].add(i)
            c += 1
    return components



def reduce_to_gcc(network):
    """
    Reduces the network to the largest connected component.
    Connected components are calculated using Tarjan's algorithm.
    """

    # these are used as nonlocal variables (!)
    index = 0
    S = []
    indices = defaultdict(lambda: None)
    low_link = defaultdict(lambda: None)
    on_stack = defaultdict(lambda: False)
    components = {}

    # Tarjan's algorithm
    def strong_connect(v):
        nonlocal index
        nonlocal S
        nonlocal indices
        nonlocal low_link
        nonlocal on_stack
        nonlocal components

        indices[v] = index
        low_link[v] = index
        index += 1
        S.append(v)
        on_stack[v] = True

        for w in network.successors[v]:
            if indices[w] is None:
                strong_connect(w)
                low_link[v] = min(low_link[v], low_link[w])
            elif on_stack[w]:
                low_link[v] = min(low_link[v], indices[w])

        # create component of node v
        if low_link[v] == indices[v]:
            components[v] = set()
            while True:
                w = S.pop()
                on_stack[w] = False
                components[v].add(w)
                if v == w:
                    break

    # compute strongly connected components
    for v in network.nodes:
        if indices[v] is None:
            strong_connect(v)
            # print('node {v}, size = {n}, component = {component}'.format(v=v, component=components[v], n = len(components[v]) ))

    max_size = 0
    for v in components:
        if len(components[v]) > max_size:            
            scc = components[v]
            max_size = len(components[v])

    # Reduce higher-order network to SCC
    for v in list(network.nodes):
        if v not in scc:
            network.remove_node(v)
