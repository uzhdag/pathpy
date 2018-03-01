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

from pathpy.exception import PathpyNotImplemented


__all__ = ["rank_centralities", "closeness_centrality", "betweenness_centrality",
           "eigenvector_centrality", "pagerank", "eigenvalue_gap",
           "fiedler_vector_sparse", "fiedler_vector_dense", "algebraic_connectivity"]


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
    if not isinstance(network, HigherOrderNetwork):
        raise PathpyNotImplemented(
            "`network` must be an instance of HigherOrderNetwork "
            "not `{}`".format(type(network))
        )

    dist_first = network.distance_matrix_first_order()
    node_centralities = _co.defaultdict(lambda: 0)

    _Log.add('Calculating closeness centralities (k = %s) ...' % network.order,
             _Severity.INFO)

    # calculate closeness values
    for v_node in dist_first:
        for w_node in dist_first[v_node]:
            if v_node != w_node and dist_first[v_node][w_node] < _np.inf:
                node_centralities[v_node] += 1.0 / dist_first[v_node][w_node]

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

    shortest_paths_first_order = _co.defaultdict(lambda: _co.defaultdict(set))

    _Log.add('Calculating betweenness centralities (k = %s) ...' % network.order,
             _Severity.INFO)

    for path_1_ord_k in shortest_paths:
        for path_2_ord_k in shortest_paths:
            source_k1 = network.higher_order_node_to_path(path_1_ord_k)[0]
            dest_k1 = network.higher_order_node_to_path(path_2_ord_k)[-1]

            # we consider a path in a k-th order network
            # connecting first-order node s1 to d1
            for path_ord_k in shortest_paths[path_1_ord_k][path_2_ord_k]:
                # convert k-th order path to first-order path and add
                shortest_paths_first_order[source_k1][dest_k1].add(
                    network.higher_order_path_to_first_order(path_ord_k))

    for source_k1 in shortest_paths_first_order:
        for dest_k1 in shortest_paths_first_order[source_k1]:
            for path_k1 in shortest_paths_first_order[source_k1][dest_k1]:
                # increase betweenness centrality of all intermediary nodes
                # on path from s1 to d1
                for v in path_k1[1:-1]:
                    if source_k1 != v != dest_k1:
                        # print('node ' + x + ': ' + str(1.0 / len(shortest_paths[vk][
                        # wk])))
                        node_centralities[v] += 1.0 / (len(
                            shortest_paths_first_order[source_k1][dest_k1]) + network.order - 1)
                        # else:
                        #    node_centralities[v] += 1.0
    if normalized:
        max_centr = max(node_centralities.values())
        for v in node_centralities:
            node_centralities[v] /= max_centr

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
    adj_mat = network.adjacency_matrix(include_subpaths=include_sub_paths,
                                       weighted=False, transposed=True)

    # calculate leading eigenvector of A
    _, v = _sla.eigs(adj_mat, k=1, which="LM", ncv=13)

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

    higher_order_pr = _co.defaultdict(lambda: 0)

    n_nodes = float(len(network.nodes))

    assert n_nodes > 0, "Number of nodes is zero"

    # entries A[s,t] give directed link s -> t
    adj_mat = network.adjacency_matrix(include_subpaths=include_sub_paths,
                                       weighted=weighted, transposed=False)

    # sum of outgoing node degrees
    row_sums = _sp.array(adj_mat.sum(axis=1)).flatten()

    # replace non-zero entries x by 1/x
    row_sums[row_sums != 0] = 1.0 / row_sums[row_sums != 0]

    # indices of zero entries in row_sums
    d = _sp.where(row_sums == 0)[0]

    # create sparse matrix with row_sums as diagonal elements
    q_mat = _sparse.spdiags(row_sums.T, 0, adj_mat.shape[0], adj_mat.shape[1],
                            format='csr')

    # with this, we have divided elements in non-zero rows in A by 1 over the row sum
    q_mat = q_mat * adj_mat

    # vector with n entries 1/n
    inv_n_nodes = _sp.array([1.0 / n_nodes] * int(n_nodes))

    p_rank = inv_n_nodes

    # Power iteration
    for _ in range(max_iter):
        last = p_rank

        # sum(pr[d]) is the sum of PageRanks for nodes with zero out-degree
        # sum(pr[d]) * p yields a vector with length n
        p_rank = (alpha * (p_rank * q_mat + sum(p_rank[d]) * inv_n_nodes) +
                  (1 - alpha) * inv_n_nodes)

        if _sp.absolute(p_rank - last).sum() < n_nodes * tol:
            higher_order_pr = dict(zip(network.nodes, map(float, p_rank)))
            break

    if network.order == 1:
        return higher_order_pr

    # project PageRank of higher-order nodes to first-order network
    first_order_pr = _co.defaultdict(lambda: 0.0)

    # sum PageRank values based on higher-order nodes
    # and normalize the result
    for v in network.nodes:
        # turns node a-b-c in path tuple (a,b,c)
        inv_n_nodes = network.higher_order_node_to_path(v)
        if projection == 'all':
            # assign PR of higher-order node to all first-order nodes
            for x in inv_n_nodes:
                first_order_pr[x] += higher_order_pr[v] / len(inv_n_nodes)
        elif projection == 'scaled':
            for x in inv_n_nodes:
                # each node on e.g. a 4-th-order path a-b-c-d receives one fourth of the
                # PageRank value, to ensure that the resulting first-order PageRank sums
                # to one
                first_order_pr[x] += higher_order_pr[v] / float(len(inv_n_nodes))
        elif projection == 'last':
            # assign PR of higher-order node to last first-order node
            first_order_pr[inv_n_nodes[-1]] += higher_order_pr[v]
        elif projection == 'first':
            # assign PR of higher-order node to last first-order node
            first_order_pr[inv_n_nodes[0]] += higher_order_pr[v]

    # for projection method 'scaled', the values sum to one anyway
    if projection != 'scaled':
        for v in first_order_pr:
            first_order_pr[v] /= sum(first_order_pr.values())

    # assign centrality zero to nodes not occurring in higher-order PR
    nodes = network.paths.nodes()
    for v in nodes:
        first_order_pr[v] += 0

    _Log.add('finished.', _Severity.INFO)

    return first_order_pr


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
    trans_mat = network.transition_matrix(include_sub_paths)

    # Compute the two largest eigenvalues
    # NOTE: ncv sets additional auxiliary eigenvectors that are computed
    # NOTE: in order to be more confident to actually find the one with the largest
    # NOTE: magnitude, see https://github.com/scipy/scipy/issues/4987
    eig_vals = _sla.eigs(trans_mat, which="LM", k=2, ncv=lanczos_vectors,
                         return_eigenvectors=False, maxiter=maxiter)
    eigen_values2_sorted = _np.sort(-_np.absolute(eig_vals))

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
    lapl_mat = network.laplacian_matrix()

    # NOTE: ncv sets additional auxiliary eigenvectors that are computed
    # NOTE: in order to be more confident to find the one with the largest
    # NOTE: magnitude, see https://github.com/scipy/scipy/issues/4987
    maxiter = maxiter * lapl_mat.get_shape()[0]
    w = _sla.eigs(lapl_mat, k=2, which="SM", ncv=lanczos_vectors,
                  return_eigenvectors=False, maxiter=maxiter)

    # compute a sparse LU decomposition and solve for the eigenvector
    # corresponding to the second largest eigenvalue
    lapl_n = lapl_mat.get_shape()[0]
    fiedler_v = _np.ones(lapl_n)
    eigen_value = _np.sort(_np.abs(w))[1]
    mat = (lapl_mat[1:lapl_n, :].tocsc()[:, 1:lapl_n] -
           _sparse.identity(lapl_n - 1).multiply(eigen_value))
    fiedler_v[1:lapl_n] = mat[0, :].toarray()

    lu_decom = _sla.splu(mat)
    fiedler_v[1:lapl_n] = lu_decom.solve(fiedler_v[1:lapl_n])

    if normalized:
        fiedler_v /= _np.sqrt(_np.inner(fiedler_v, fiedler_v))
    return fiedler_v


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
    lapl_mat = network.laplacian_matrix()
    # convert to dense matrix and transpose again to have the un-transposed
    # laplacian again.
    laplacian_transposed = lapl_mat.todense().transpose()
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

    lapl_mat = network.laplacian_matrix()
    # NOTE: ncv sets additional auxiliary eigenvectors that are computed
    # NOTE: in order to be more confident to find the one with the largest
    # NOTE: magnitude, see https://github.com/scipy/scipy/issues/4987
    w = _sla.eigs(lapl_mat, which="SM", k=2, ncv=lanczos_vectors,
                  return_eigenvectors=False, maxiter=maxiter)
    eigen_values_sorted = _np.sort(_np.absolute(w))

    _Log.add('finished.', _Severity.INFO)

    # TODO: result is unstable, it looks like it depends on a "warm start"
    # (i.e. run after other eigen velue calculations) see test_algebraic_connectivity
    # problems with order k=3

    return _np.abs(eigen_values_sorted[1])
