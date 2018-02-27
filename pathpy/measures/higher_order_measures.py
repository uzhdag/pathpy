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
#   GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#    Contact the developer:

#    E-mail: ischoltes@ethz.ch
#    Web:    http://www.ingoscholtes.net
"""
This class can be used to calculate path statistics based on
origin-destination data available for a known network topology.
The path statistics generated from such data will be based on
the assumption that each observed path from an origin to a destination
node follows a shortest path in the network topology.
"""
import collections as _co
import operator

import numpy as _np
import scipy.sparse as _sparse
import scipy.sparse.linalg as _sla
import scipy.linalg as _la
import scipy as _sp

from pathpy import Log as _Log
from pathpy.log import Severity as _Severity
from pathpy import HigherOrderNetwork


__all__ = ["rank_centralities", "closeness_centrality", "betweenness_centrality",
           "eigenvector_centrality", "pagerank", "eigenvalue_gap",
           "fiedler_vector_sparse", "fiedler_vector_dense", "algebraic_connectivity",
           ]


def rank_centralities(centralities):
    """Returns a dictionary with node centrality values as a rank_centralities list

    Parameters
    ----------
    centralities: dict
        dictionary of centralities

    Examples
    --------
    >>> centralities = {('a', 'b', 'c'): .2, ('b', 'a', 'b'): .8}
    >>> rank_centralities(centralities)
    [(('b', 'a', 'b'), 0.8), (('a', 'b', 'c'), 0.2)]

    Returns
    -------
    list

    """
    ranked_nodes = sorted(centralities.items(), key=operator.itemgetter(1))
    ranked_nodes.reverse()
    return ranked_nodes


def closeness_centrality(network):
    """Calculates the closeness centralities of all nodes.

    If the order of the higher-order network is larger than one
    centralities calculated based on the higher-order
    topology will automatically be projected back to first-order
    nodes.

    Parameters
    ----------
    network: HigherOrderNetwork

    Returns
    -------
    dict
    """
    assert isinstance(network, HigherOrderNetwork), \
        "network must be an instance of HigherOrderNetwork"
    dist_first = network.distance_matrix_first_order()
    node_centralities = _co.defaultdict(lambda: 0)

    _Log.add('Calculating closeness centralities (k = %s) ...' % network.order,
             _Severity.INFO)

    # calculate closeness values
    for v1 in dist_first:
        for w1 in dist_first[v1]:
            if v1 != w1 and dist_first[v1][w1] < _np.inf:
                node_centralities[v1] += 1.0 / dist_first[v1][w1]

    # assign centrality zero to nodes not occurring on higher-order shortest paths
    nodes = network.paths.nodes()
    for v in nodes:
        node_centralities[v] += 0

    _Log.add('finished.', _Severity.INFO)

    return node_centralities


def betweenness_centrality(network, normalized=False):
    """Calculates the betweenness centralities of all nodes.

    If the order of the higher-order network is larger than one
    centralities calculated based on the higher-order
    topology will automatically be projected back to first-order
    nodes.

    Parameters
    ----------
    network: HigherOrderNetwork
        an instance of a pathpy HigherOrderNetwork
    normalized:
        If set to True, betweenness centralities of nodes will be scaled by the maximum
        value (default False)

    Returns
    -------
    dict
        Dictionary containing as the keys the higher order node and as values their
        centralities
    """
    assert isinstance(network, HigherOrderNetwork), \
        "network must be an instance of HigherOrderNetwork"
    shortest_paths = network.shortest_paths()
    node_centralities = _co.defaultdict(lambda: 0)

    shortest_paths_first_order = _co.defaultdict(lambda: _co.defaultdict(lambda: set()))

    _Log.add('Calculating betweenness centralities (k = %s) ...' % network.order,
             _Severity.INFO)

    for sk in shortest_paths:
        for dk in shortest_paths:
            s1 = network.higher_order_node_to_path(sk)[0]
            d1 = network.higher_order_node_to_path(dk)[-1]

            # we consider a path in a k-th order network
            # connecting first-order node s1 to d1
            for pk in shortest_paths[sk][dk]:
                # convert k-th order path to first-order path and add
                shortest_paths_first_order[s1][d1].add(
                    network.higher_order_path_to_first_order(pk))

    for s1 in shortest_paths_first_order:
        for d1 in shortest_paths_first_order[s1]:
            for p1 in shortest_paths_first_order[s1][d1]:
                # increase betweenness centrality of all intermediary nodes
                # on path from s1 to d1
                for v in p1[1:-1]:
                    if s1 != v != d1:
                        # print('node ' + x + ': ' + str(1.0 / len(shortest_paths[vk][
                        # wk])))
                        node_centralities[v] += 1.0 / (len(
                            shortest_paths_first_order[s1][d1]) + network.order - 1)
                        # else:
                        #    node_centralities[v] += 1.0
    if normalized:
        m = max(node_centralities.values())
        for v in node_centralities:
            node_centralities[v] /= m

    # assign centrality zero to nodes not occurring on higher-order shortest paths
    nodes = network.paths.nodes()
    for v in nodes:
        node_centralities[v] += 0

    _Log.add('finished.', _Severity.INFO)

    return node_centralities


def eigenvector_centrality(network, projection='scaled', include_sub_paths=True):
    """Calculates the eigenvector centralities of higher-order nodes.

    If the order of the HigherOrderNetwork is larger than one, the centralities
    will be projected to the first-order nodes.

    Parameters
    ----------
    network
    projection: str
        Indicates how the projection from k-th-order nodes (v1, v2, ... , v{k-1}) shall be
        performed. For the method 'all', the eigenvector centrality of the higher-order
        node will be added to *all* first-order nodes on the path corresponding to the
        higher-order node. For the method 'last', the centrality of the higher-order node
        will only be assigned to *last* first-order node v{k-1}. For the method 'scaled'
        (default), the eigenvector centrality of higher-order nodes will be assigned
        proportionally to first-order nodes, i.e. each of the three nodes in the
        third-order node (a,b,c) will receive one third of the eigenvector centrality
        of (a,b,c).
    include_sub_paths: bool
        whether or not to include subpath statistics in the calculation (default True)

    Returns
    -------
    dict

    """
    assert isinstance(network, HigherOrderNetwork), \
        "network must be an instance of HigherOrderNetwork"
    A = network.adjacency_matrix(include_subpaths=include_sub_paths, weighted=False,
                                 transposed=True)

    # calculate leading eigenvector of A
    w, v = _sla.eigs(A, k=1, which="LM", ncv=13)

    v = v.reshape(v.size, )

    higher_order_eigen_vec_cent = dict(zip(network.nodes, map(_np.abs, v)))

    # project eigen_vec_cent of higher-order nodes to first-order network
    first_order_eigen_vec_cent = _co.defaultdict(lambda: 0.0)

    # sum eigen_vec_cent values based on higher-order nodes
    # and normalize the result
    for v in network.nodes:
        # turns node a-b-c in path tuple (a,b,c)
        p = network.higher_order_node_to_path(v)
        if projection == 'all':
            # assign eigen_vec_cent of higher-order node to all first-order nodes
            for x in p:
                first_order_eigen_vec_cent[x] += higher_order_eigen_vec_cent[v]
        elif projection == 'scaled':
            for x in p:
                n_p = float(len(p))
                first_order_eigen_vec_cent[x] += higher_order_eigen_vec_cent[v] / n_p
        elif projection == 'last':
            # assign eigen_vec_cent  of higher-order node to last first-order node
            first_order_eigen_vec_cent[p[-1]] += higher_order_eigen_vec_cent[v]
        elif projection == 'first':
            # assign eigen_vec_cent  of higher-order node to last first-order node
            first_order_eigen_vec_cent[p[0]] += higher_order_eigen_vec_cent[v]

    # for scaled, values sum to one anyway
    if projection != 'scaled':
        for v in first_order_eigen_vec_cent:
            first_order_eigen_vec_cent[v] /= sum(first_order_eigen_vec_cent.values())

    _Log.add('finished.', _Severity.INFO)

    return first_order_eigen_vec_cent


def pagerank(network, alpha=0.85, max_iter=100, tol=1.0e-6, projection='scaled',
             include_sub_paths=True, weighted=False):
    """Calculates the PageRank of higher-order nodes based on a power iteration.

    If the order of the higher-order network is larger than one, the PageRank calculated
    based on the higher-order topology will automatically be projected back to first-order
    nodes.

    Parameters
    ----------
    network: HigherOrderNetwork
    alpha: float
        damping factor
    max_iter: int
        maximum number or iterations in solver
    tol: float
        accepted tolerance for convergence check
    projection: str
        Indicates how the projection from k-th-order nodes (v1, v2, ... , v{k-1}) shall be
        performed. For the method 'all', the PageRank value of the higher-order node will
        be added to *all* first-order nodes on the path corresponding to the higher-order
        node. For the method 'last', the PR value of the higher-order node will only be
        assigned to *last* first-order node v{k-1}. For the method 'scaled' (default), the
        PageRank of higher-order nodes will be assigned proportionally to first-order
        nodes, i.e. each of the three nodes in the third-order node (a,b,c) will receive
        one third of the PageRank of (a,b,c).
    include_sub_paths: bool
        whether or not to use subpath statistics in the PageRank calculation
    weighted: bool
        use path weights in the calculation

    Returns
    -------
    dict

    """
    assert isinstance(network, HigherOrderNetwork), \
        "network must be an instance of HigherOrderNetwork"
    assert projection in ['all', 'last', 'first', 'scaled'], 'Invalid projection method'

    _Log.add('Calculating PageRank in ' + str(network.order) + '-th order network...',
             _Severity.INFO)

    higher_order_PR = _co.defaultdict(lambda: 0)

    n = float(len(network.nodes))

    assert n > 0, "Number of nodes is zero"

    # entries A[s,t] give directed link s -> t
    A = network.adjacency_matrix(include_subpaths=include_sub_paths, weighted=weighted,
                                 transposed=False)

    # sum of outgoing node degrees
    row_sums = _sp.array(A.sum(axis=1)).flatten()

    # replace non-zero entries x by 1/x
    row_sums[row_sums != 0] = 1.0 / row_sums[row_sums != 0]

    # indices of zero entries in row_sums
    d = _sp.where(row_sums == 0)[0]

    # create sparse matrix with row_sums as diagonal elements
    Q = _sparse.spdiags(row_sums.T, 0, A.shape[0], A.shape[1], format='csr')

    # with this, we have divided elements in non-zero rows in A by 1 over the row sum
    Q = Q * A

    # vector with n entries 1/n
    p = _sp.array([1.0 / n] * int(n))

    pr = p

    # Power iteration
    for i in range(max_iter):
        last = pr

        # sum(pr[d]) is the sum of PageRanks for nodes with zero out-degree
        # sum(pr[d]) * p yields a vector with length n
        pr = alpha * (pr * Q + sum(pr[d]) * p) + (1 - alpha) * p

        if _sp.absolute(pr - last).sum() < n * tol:
            higher_order_PR = dict(zip(network.nodes, map(float, pr)))
            break

    if network.order == 1:
        return higher_order_PR

    # project PageRank of higher-order nodes to first-order network
    first_order_PR = _co.defaultdict(lambda: 0.0)

    # sum PageRank values based on higher-order nodes
    # and normalize the result
    for v in network.nodes:
        # turns node a-b-c in path tuple (a,b,c)
        p = network.higher_order_node_to_path(v)
        if projection == 'all':
            # assign PR of higher-order node to all first-order nodes
            for x in p:
                first_order_PR[x] += higher_order_PR[v] / len(p)
        elif projection == 'scaled':
            for x in p:
                # each node on e.g. a 4-th-order path a-b-c-d receives one fourth of the
                # PageRank value, to ensure that the resulting first-order PageRank sums
                # to one
                first_order_PR[x] += higher_order_PR[v] / float(len(p))
        elif projection == 'last':
            # assign PR of higher-order node to last first-order node
            first_order_PR[p[-1]] += higher_order_PR[v]
        elif projection == 'first':
            # assign PR of higher-order node to last first-order node
            first_order_PR[p[0]] += higher_order_PR[v]

    # for projection method 'scaled', the values sum to one anyway
    if projection != 'scaled':
        for v in first_order_PR:
            first_order_PR[v] /= sum(first_order_PR.values())

    # assign centrality zero to nodes not occurring in higher-order PR
    nodes = network.paths.nodes()
    for v in nodes:
        first_order_PR[v] += 0

    _Log.add('finished.', _Severity.INFO)

    return first_order_PR


def eigenvalue_gap(network, include_sub_paths=True, lanczos_vectors=15, maxiter=20):
    """Returns the eigenvalue gap of the transition matrix.

    Parameters
    ----------
    network
    include_sub_paths: bool
        whether or not to include subpath statistics in the calculation of transition
        probabilities.
    lanczos_vectors: int
        number of Lanczos vectors to be used in the approximate
        calculation of eigenvectors and eigenvalues. This maps to the ncv parameter
        of scipy's underlying function eigs.
    maxiter: int
        scaling factor for the number of iterations to be used in the
        approximate calculation of eigenvectors and eigenvalues. The number of iterations
        passed to scipy's underlying eigs function will be n*maxiter where n is the
        number of rows/columns of the Laplacian matrix.

    Returns
    -------
    float
    """
    assert isinstance(network, HigherOrderNetwork), \
        "network must be an instance of HigherOrderNetwork"
    # NOTE to myself: most of the time goes for construction of the 2nd order
    # NOTE            null graph, then for the 2nd order null transition matrix

    _Log.add('Calculating eigenvalue gap ... ', _Severity.INFO)

    # Build transition matrices
    T = network.transition_matrix(include_sub_paths)

    # Compute the two largest eigenvalues
    # NOTE: ncv sets additional auxiliary eigenvectors that are computed
    # NOTE: in order to be more confident to actually find the one with the largest
    # NOTE: magnitude, see https://github.com/scipy/scipy/issues/4987
    w2 = _sla.eigs(T, which="LM", k=2, ncv=lanczos_vectors, return_eigenvectors=False,
                   maxiter=maxiter)
    eigen_values2_sorted = _np.sort(-_np.absolute(w2))

    _Log.add('finished.', _Severity.INFO)

    return _np.abs(eigen_values2_sorted[1])


def fiedler_vector_sparse(network, normalized=True, lanczos_vectors=15, maxiter=20):
    """Returns the (sparse) Fiedler vector of the higher-order network. The Fiedler
    vector can be used for a spectral bisection of the network.

    Note that sparse linear algebra for eigenvalue problems with small eigenvalues
    is problematic in terms of numerical stability. Consider using the dense version
    of this method in this case. Note also that the sparse Fiedler vector might be scaled
    by a factor (-1) compared to the dense version.

    Parameters
    ----------
    network
    normalized: bool
        whether (default) or not to normalize the fiedler vector.
        Normalization is done such that the sum of squares equals one in order to
        get reasonable values as entries might be positive and negative.
    lanczos_vectors: int
        number of Lanczos vectors to be used in the approximate
        calculation of eigenvectors and eigenvalues. This maps to the ncv parameter
        of scipy's underlying function eigs.
    maxiter: int
        scaling factor for the number of iterations to be used in the
        approximate calculation of eigenvectors and eigenvalues. The number of iterations
        passed to scipy's underlying eigs function will be n*maxiter where n is the
        number of rows/columns of the Laplacian matrix.

    Returns
    -------

    """
    assert isinstance(network, HigherOrderNetwork), \
        "network must be an instance of HigherOrderNetwork"
    # NOTE: The transposed matrix is needed to get the "left" eigenvectors
    L = network.laplacian_matrix()

    # NOTE: ncv sets additional auxiliary eigenvectors that are computed
    # NOTE: in order to be more confident to find the one with the largest
    # NOTE: magnitude, see https://github.com/scipy/scipy/issues/4987
    maxiter = maxiter * L.get_shape()[0]
    w = _sla.eigs(L, k=2, which="SM", ncv=lanczos_vectors, return_eigenvectors=False,
                  maxiter=maxiter)

    # compute a sparse LU decomposition and solve for the eigenvector
    # corresponding to the second largest eigenvalue
    n = L.get_shape()[0]
    b = _np.ones(n)
    eigen_value = _np.sort(_np.abs(w))[1]
    A = (L[1:n, :].tocsc()[:, 1:n] - _sparse.identity(n - 1).multiply(eigen_value))
    b[1:n] = A[0, :].toarray()

    lu = _sla.splu(A)
    b[1:n] = lu.solve(b[1:n])

    if normalized:
        b /= _np.sqrt(_np.inner(b, b))
    return b


def fiedler_vector_dense(network):
    """Returns the (dense)Fiedler vector of the higher-order network.
    The Fiedler vector can be used for a spectral bisection of the network.

    Parameters
    ----------
    network: HigherOrderNetwork

    Returns
    -------

    """
    assert isinstance(network, HigherOrderNetwork), \
        "network must be an instance of HigherOrderNetwork"
    # NOTE: The Laplacian is transposed for the sparse case to get the left
    # NOTE: eigenvalue.
    L = network.laplacian_matrix()
    # convert to dense matrix and transpose again to have the un-transposed
    # laplacian again.
    laplacian_transposed = L.todense().transpose()
    w, v = _la.eig(laplacian_transposed, right=False, left=True)

    return v[:, _np.argsort(_np.absolute(w))][:, 1]


def algebraic_connectivity(network, lanczos_vectors=15, maxiter=20):
    """

    Parameters
    ----------
    network: HigherOrderNetwork
    lanczos_vectors: int
        number of Lanczos vectors to be used in the approximate calculation of
        eigenvectors and eigenvalues. This maps to the ncv parameter of scipy's underlying
        function eigs.
    maxiter: int
        scaling factor for the number of iterations to be used in the approximate
        calculation of eigenvectors and eigenvalues. The number of iterations passed to
        scipy's underlying eigs function will be n*maxiter where n is the number of
        rows/columns of the Laplacian matrix.

    Returns
    -------

    """
    assert isinstance(network, HigherOrderNetwork), \
        "network must be an instance of HigherOrderNetwork"
    _Log.add('Calculating algebraic connectivity ... ', _Severity.INFO)

    L = network.laplacian_matrix()
    # NOTE: ncv sets additional auxiliary eigenvectors that are computed
    # NOTE: in order to be more confident to find the one with the largest
    # NOTE: magnitude, see https://github.com/scipy/scipy/issues/4987
    w = _sla.eigs(L, which="SM", k=2, ncv=lanczos_vectors, return_eigenvectors=False,
                  maxiter=maxiter)
    eigen_values_sorted = _np.sort(_np.absolute(w))

    _Log.add('finished.', _Severity.INFO)

    return _np.abs(eigen_values_sorted[1])
