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

from pathpy import Network
from pathpy.utils import Log, Severity
from pathpy.utils import PathpyError

__all__ = ['connected_components']


def connected_components(network):
    """
    Returns a list of instances of the class Network that represent the 
    (strongly) connected components of a network. Connected components
    are calculated using Tarjan's algorithm.
    """

    # nonlocal variables (!)
    index = 0
    S = []
    indices = defaultdict(lambda: None)
    low_link = defaultdict(lambda: None)
    on_stack = defaultdict(lambda: False)

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

        for w in network.successors[v]:
            if indices[w] is None:
                strong_connect(w)
                low_link[v] = min(low_link[v], low_link[w])
            elif on_stack[w]:
                low_link[v] = min(low_link[v], indices[w])

        # generate SCC of node v
        component = set()
        if low_link[v] == indices[v]:
            while True:
                w = S.pop()
                on_stack[w] = False
                component.add(w)
                if v == w:
                    break
        return component

    # compute strongly connected components
    components = defaultdict(set)
    max_size = 0
    max_head = None
    for v in network.nodes:
        if indices[v] is None:
            components[v] = strong_connect(v)
            if len(components[v]) > max_size:
                max_head = v
                max_size = len(components[v])

    scc = components[max_head]

    # Reduce higher-order network to SCC
    for v in list(network.nodes):
        if v not in scc:
            network.remove_node(v)
