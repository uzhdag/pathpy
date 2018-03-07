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

from pathpy.classes.network import Network
from pathpy.utils import Log, Severity
from pathpy.utils import PathpyError


__all__ = ['distance_matrix', 'shortest_paths']


def distance_matrix(network):
    """Calculates shortest path distances between all pairs of nodes
    in a network using the Floyd-Warshall algorithm."""

    Log.add('Calculating distance matrix in network ', Severity.INFO)

    dist = defaultdict(lambda: defaultdict(lambda: _np.inf))

    # assign first the default weight of 1
    for e in network.edges:
        dist[e[0]][e[1]] = 1
        if not network.directed:
            dist[e[1]][e[0]] = 1

    # set all self-loop edges to 0
    for v in network.nodes:
        dist[v][v] = 0

    for k in network.nodes:
        for v in network.nodes:
            for w in network.nodes:
                if dist[v][w] > dist[v][k] + dist[k][w]:
                    dist[v][w] = dist[v][k] + dist[k][w]

    Log.add('finished.', Severity.INFO)

    return dist

def shortest_paths(network):
    """
    Calculates all shortest paths between all pairs of
    nodes using the Floyd-Warshall algorithm.
    """

    Log.add('Calculating shortest paths in network ', Severity.INFO)

    dist = defaultdict(lambda: defaultdict(lambda: _np.inf))
    shortest_paths = defaultdict(lambda: defaultdict(set))

    for e in network.edges:
        dist[e[0]][e[1]] = 1
        shortest_paths[e[0]][e[1]].add(e)
        if not network.directed:
            dist[e[1]][e[0]] = 1
            shortest_paths[e[1]][e[0]].add((e[1], e[0]))

    for k in network.nodes:
        for v in network.nodes:
            for w in network.nodes:
                if v != w:
                    if dist[v][w] > dist[v][k] + dist[k][w]:
                        dist[v][w] = dist[v][k] + dist[k][w]
                        shortest_paths[v][w] = set()
                        for p in list(shortest_paths[v][k]):
                            for q in list(shortest_paths[k][w]):
                                shortest_paths[v][w].add(p + q[1:])
                    elif dist[v][w] == dist[v][k] + dist[k][w]:
                        for p in list(shortest_paths[v][k]):
                            for q in list(shortest_paths[k][w]):
                                shortest_paths[v][w].add(p + q[1:])

    for v in network.nodes:
        dist[v][v] = 0
        shortest_paths[v][v].add((v,))

    Log.add('finished.', Severity.INFO)

    return shortest_paths