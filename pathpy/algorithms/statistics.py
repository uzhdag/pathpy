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
from collections import Counter

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


def mean_degree(network, degree='degree'):
    r"""Calculates the mean (in/out)-degree of a directed or undirected network.

    Parameters
    ----------
    network:    Network
        The network in which to calculate the mean degree
    """
    assert degree is 'degree' or degree is 'indegree' or degree is 'outdegree', \
            'Unknown degree property'
    return _np.mean([network.nodes[x][degree] for x in network.nodes])


def degree_dist(network, degree='degree'):
    r"""Calculates the (in/out)-degree distribution of a directed or undirected network.

    Parameters
    ----------
    network:    Network
        The network for which to calculate the degree distribution
    """
    assert degree is 'degree' or degree is 'indegree' or degree is 'outdegree',\
            'Unknown degree property'
    p_k = Counter([network.nodes[x][degree] for x in network.nodes])
    for x in p_k:
        p_k[x] = p_k[x]/network.ncount()
    return p_k


def degree_moment(network, k, degree='degree'):
    r"""Calculates the k-th moment of the (in/out)-degree distribution of a
    directed or undirected network.

    Parameters
    ----------
    network:    Network
        The network in which to calculate the k-th moment of the degree distribution
    """
    p_k = degree_dist(network, degree)
    mom = 0
    for x in p_k:
        mom += x**k * p_k[x]
    return mom


def generating_func(network, x, degree='degree'):
    r"""Returns f(x) where f is the probability generating function for the
    (in/out)-degree distribution P(k) for a network. The function is defined in the interval [0,1].
    The value returned is from the range [0,1]. The following properties hold:

    [1/k! d^k/dx f]_{x=0} = P(k)    with d^k/dx f being the k-th derivative of f by x
    f'(1) = <k>                     with f' being the first derivative and <k> the mean degree
    [(x d/dx)^m f]_{x=1} = <k^m>    with <k^m> being the m-th raw moment of P

    Parameters
    ----------
    x:  float, list, numpy.ndarray
        The argument(s) for which the value f(x) shall be computed.

    Returns
    -------
        Either a single float value f(x) (if x is float) or a numpy.ndarray
        containing the function values f(x) for all arguments in x

    Example
    -------
    >>> import pathpy as pp
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> n = pp.Network()
    >>> n.add_edge('a', 'b')
    >>> n.add_edge('b', 'c')
    >>> n.add_edge('a', 'c')
    >>> n.add_edge('c', 'd')
    >>> n.add_edge('d', 'e')
    >>> n.add_edge('d', 'f')
    >>> n.add_edge('e', 'f')
    
    >>> # print single value f(x)
    >>> print(pp.statistics.generating_func(n, 0.3))

    >>> # plot generating function
    >>> x = np.linspace(0, 1, 20)
    >>> y = pp.statistics.generating_func(n, x)
    >>> x = plt.plot(x, y)
    """

    assert isinstance(x, (float, list, _np.ndarray)), \
        'Argument can only be float, list or numpy.ndarray'

    p_k = degree_dist(network, degree)

    if isinstance(x, float):
        x_range = [x]
    else:
        x_range = x

    values = defaultdict(lambda: 0)
    for k in p_k:
        for v in x_range:
            values[v] += p_k[k] * v**k

    if len(x_range) > 1:
        return _np.array(list(values.values()))
    else:
        return values[x]


def molloy_reed_fraction(network, degree='degree'):
    r"""Calculates the Molloy-Reed fraction <k**2>/<k> based on the (in/out)-degree
    distribution of a directed or undirected network.

    Parameters
    ----------
    network:    Network
        The network in which to calculate the Molloy-Reed fraction
    """
    return degree_moment(network, k=2, degree=degree)/degree_moment(network, k=1, degree=degree)
