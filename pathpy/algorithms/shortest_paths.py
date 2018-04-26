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
from functools import singledispatch
from collections import defaultdict

import numpy as _np

from pathpy.utils import Log, Severity
from pathpy.utils import PathpyNotImplemented
from pathpy.classes import HigherOrderNetwork
from pathpy.classes import Network
from pathpy.classes import Paths

__all__ = ['distance_matrix', 'shortest_paths', 'diameter']

@singledispatch
def distance_matrix(network):
    """Calculates shortest path distances between all pairs of nodes
    in a network using the Floyd-Warshall algorithm."""

    assert isinstance(network, Network), \
        "network must be an instance of Network"

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

    return dist


@distance_matrix.register(Paths)
def _dm(paths):
    """
    Calculates shortest path distances between all pairs of
    nodes based on the observed shortest paths (and subpaths)
    """
    dist = defaultdict(lambda: defaultdict(lambda: _np.inf))

    Log.add('Calculating distance matrix based on empirical paths ...', Severity.INFO)
    # Node: no need to initialize shortest_path_lengths[v][v] = 0
    # since paths of length zero are contained in self.paths

    for v in paths.nodes:
        dist[v][v] = 0

    for p_length in paths.paths:
        for p in paths.paths[p_length]:
            start = p[0]
            end = p[-1]
            if p_length < dist[start][end]:
                dist[start][end] = p_length

    Log.add('finished.', Severity.INFO)

    return dist


@distance_matrix.register(HigherOrderNetwork)
def _dm_ho(network):
    """
    Returns a matrix capturing distances between (first-order)
    nodes, based on a given higher-order topology. 
    As an example, the second-order network [a-b] -> [b-c]
    will lead to the distance matrix:
    dist[a][c] = 2
    """

    assert isinstance(network, HigherOrderNetwork), \
        "network must be an instance of HigherOrderNetwork"

    dist = defaultdict(lambda: defaultdict(lambda: _np.inf))

    # Note: higher-order networks are always directed
    for e in network.edges:
        dist[e[0]][e[1]] = 1

    # k, v, and w are *higher-order* nodes, i.e. paths of length k
    for k in network.nodes:
        for v in network.nodes:
            for w in network.nodes:
                if dist[v][w] > dist[v][k] + dist[k][w]:
                    dist[v][w] = dist[v][k] + dist[k][w]
    
    # project distances to first-order nodes
    dist_first = defaultdict(lambda: defaultdict(lambda: _np.inf))

    # set distance between nodes based on higher-order nodes (paths)
    for v in network.nodes:
        v1 = network.higher_order_node_to_path(v)[0]
        w1 = network.higher_order_node_to_path(v)[-1]
        dist_first[v1][w1] = network.order - 1

    # set diagonal entries to zero
    for v in network.paths.nodes:
        dist_first[v][v] = 0

    # set distances between first-order nodes
    for vk in dist:
        for wk in dist[vk]:
            v1 = network.higher_order_node_to_path(vk)[0]
            w1 = network.higher_order_node_to_path(wk)[-1]
            if dist[vk][wk] + network.order - 1 < dist_first[v1][w1]:
                dist_first[v1][w1] = dist[vk][wk] + network.order - 1

    return dist_first


@singledispatch
def shortest_paths(network):
    """
    Calculates all shortest paths between all pairs of
    nodes using the Floyd-Warshall algorithm.
    """
    assert isinstance(network, Network), \
        "network must be an instance of Network"

    dist = defaultdict(lambda: defaultdict(lambda: _np.inf))
    s_p = defaultdict(lambda: defaultdict(set))

    for e in network.edges:
        dist[e[0]][e[1]] = 1
        s_p[e[0]][e[1]].add(e)
        if not network.directed:
            dist[e[1]][e[0]] = 1
            s_p[e[1]][e[0]].add((e[1], e[0]))

    for k in network.nodes:
        for v in network.nodes:
            for w in network.nodes:
                if v != w:
                    if dist[v][w] > dist[v][k] + dist[k][w]:
                        dist[v][w] = dist[v][k] + dist[k][w]
                        s_p[v][w] = set()
                        for p in list(s_p[v][k]):
                            for q in list(s_p[k][w]):
                                s_p[v][w].add(p + q[1:])
                    elif dist[v][w] == dist[v][k] + dist[k][w]:
                        for p in list(s_p[v][k]):
                            for q in list(s_p[k][w]):
                                s_p[v][w].add(p + q[1:])

    for v in network.nodes:
        dist[v][v] = 0
        s_p[v][v].add((v,))

    return s_p


@shortest_paths.register(Paths)
def _sp(paths):
    """
    Calculates the set of shortest 
    paths between each pair of nodes based on 
    a given set of empirically observed paths
    """

    assert isinstance(paths, Paths), \
        "paths must be an instance of Paths"

    s_p = defaultdict(lambda: defaultdict(set))
    s_p_lengths = defaultdict(lambda: defaultdict(lambda: _np.inf))

    for p_length in paths.paths:
        for p in paths.paths[p_length]:
            if _np.sum(paths.paths[p_length][p])>0:
                # make sure we only consider paths with non-zero 
                # observation count (as path or sub-path)
                s = p[0]
                d = p[-1]
                # we found a shorter path of length l between s and d
                if p_length < s_p_lengths[s][d]:
                    # update shortest path length
                    s_p_lengths[s][d] = p_length
                    # redefine set
                    s_p[s][d] = set()
                    s_p[s][d].add(p)
                elif p_length == s_p_lengths[s][d]:
                    s_p[s][d].add(p)
    return s_p


@singledispatch
def diameter(network):
    """
    Return the maximal path length.
    """
    assert isinstance(network, Network), \
        "network must be an instance of Network"

    raise PathpyNotImplemented()
