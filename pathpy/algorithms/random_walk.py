"""
Algorithms to calculate shortest paths and distances in higher-order networks and paths.
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
from functools import singledispatch
from collections import defaultdict

from pathpy.utils import Log, Severity
from pathpy.utils import PathpyNotImplemented
from pathpy.classes import TemporalNetwork
from pathpy.classes import Network
from pathpy.classes import HigherOrderNetwork
import numpy as _np

@singledispatch
def generate_walk(network, l=100, start_node=None):
    """
    """
    T = network.transition_matrix().todense().transpose()
    idx_map = network.node_to_name_map()
    nodes = _np.array([v for v in network.nodes])

    itinerary = []

    if start_node is None:
        start_node = _np.random.choice(nodes)
    
    # choose random start node
    itinerary.append(start_node)
    for j in range(l):
        # get transition probability vector T[idx ->  . ]
        prob = _np.array(T[idx_map[itinerary[-1]],:])[0,:]
        nz = prob.nonzero()[0]
        # make one random transition
        if nz.shape[0]>0:
            next_node = _np.random.choice(a=nodes[nz], p=prob[nz])
            # add node to path
            itinerary.append(next_node)
        else: # no neighbor
            break
    return itinerary


@generate_walk.register(HigherOrderNetwork)
def _temporal_walk(higher_order_net, l=100, start_node=None):
    raise PathpyNotImplemented('Walk in higher order network is not implemented')

@generate_walk.register(TemporalNetwork)
def _temporal_walk(tempnet, l=100, start_node=None):
    itinerary = []
    if start_node is None:
        current_node = _np.random.choice(tempnet.nodes)
    else:
        current_node = start_node
    itinerary.append(current_node)
    steps = 0
    for t in tempnet.ordered_times:
        # find possible targets in time t
        targets = set()
        for (v, w, time) in tempnet.time[t]:
            if v == current_node:
               targets.add(w)
        # move to random target
        if targets:
            current_node = _np.random.choice(list(targets))
            steps += 1
        # add currently visited node
        itinerary.append(current_node)
        if steps == l:
            break
    return itinerary