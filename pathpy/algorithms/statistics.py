"""
Collection of statistical measures for  paths, (higher-order) networks, and temporal networks
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
import scipy as sp

from pathpy .utils import Log, Severity
from pathpy import Network

from pathpy.utils import PathpyNotImplemented


def local_clustering_coefficient(network, v):
    r"""Calculates the local clustering coefficient of a node in a directed or undirected network.
    The local clustering coefficient of any node with an (out-)degree smaller than two is defined
    as zero. For all other nodes, it is defined as:

        cc(c) := 2*k(i)/(d_i(d_i-1))
    
        or
    
        cc(c) := k(i)/(d_out_i(d_out_i-1))

        in undirected and directed networks respectively.

    Parameters
    ----------
    network:    Network
        The network in which to calculate the local clustering coefficient.
    node:   str
        The node for which the local clustering coefficient shall be calculated.
    """
    if network.directed and network.nodes[v]['outdegree'] < 2:
        return 0.0
    if not network.directed and network.nodes[v]['degree'] < 2:
        return 0.0
    k_i = 0.0
    for i in network.successors[v]:
        for j in network.successors[v]:
            if (i, j) in network.edges:
                k_i += 1.0
    if not network.directed:
        return k_i/(network.nodes[v]['degree']*(network.nodes[v]['degree']-1.0))
    return k_i/(network.nodes[v]['outdegree']*(network.nodes[v]['outdegree']-1.0))


def avg_clustering_coefficient(network):
    r"""Calculates the average (global) clustering coefficient of a directed or undirected network.

    Parameters
    ----------
    network:    Network
        The network in which to calculate the local clustering coefficient.
    """
    return _np.mean([ local_clustering_coefficient(network, v) for v in network.nodes])
