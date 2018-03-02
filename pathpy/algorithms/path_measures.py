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
import collections as _co

import numpy as _np
import scipy.sparse.linalg as _sla

from pathpy import HigherOrderNetwork as _HigherOrderNetwork
from pathpy import Log as _Log
from pathpy.log import Severity as _Severity
from pathpy import Paths
from pathpy.exception import PathpyError


__all__ = ['betweenness_centrality', 'closeness_centrality', 'node_traversals',
           'visitation_probabilities', 'slow_down_factor', 'entropy_growth_rate_ratio',
           'betweenness_preference', 'betweenness_preference_matrix']


def betweenness_centrality(paths, normalized=False):
    """Calculates the betweenness centrality of nodes based on observed shortest paths
    between all pairs of nodes

    Parameters
    ----------
    paths:
        Paths object
    normalized: bool
        normalize such that largest value is 1.0

    Returns
    -------
    dict
    """
    assert isinstance(paths, Paths), "paths must be an instance of pathpy.Paths"
    node_centralities = _co.defaultdict(lambda: 0)
    shortest_paths = paths.shortest_paths()

    for s in shortest_paths:
        for d in shortest_paths[s]:
            for p in shortest_paths[s][d]:
                for x in p[1:-1]:
                    if s != d != x:
                        node_centralities[x] += 1.0 / len(shortest_paths[s][d])
                        # node_centralities[x] += 1.0
    if normalized:
        m = max(node_centralities.values())
        for v in node_centralities:
            node_centralities[v] /= m

    # assign zero values to nodes not occurring on shortest paths
    nodes = paths.nodes()
    for v in nodes:
        node_centralities[v] += 0

    return node_centralities


def closeness_centrality(paths, normalized=False):
    """Calculates the closeness centrality of nodes based on observed shortest paths
    between all nodes

    Parameters
    ----------
    paths: Paths
    normalized: bool
        normalize such that largest value is 1.0

    Returns
    -------
    dict
    """
    node_centralities = _co.defaultdict(lambda: 0)
    shortest_path_lengths = paths.distance_matrix()

    for x in shortest_path_lengths:
        for d in shortest_path_lengths[x]:
            if x != d and shortest_path_lengths[x][d] < _np.inf:
                node_centralities[x] += 1.0 / shortest_path_lengths[x][d]

    # assign zero values to nodes not occurring
    nodes = paths.nodes()
    for v in nodes:
        node_centralities[v] += 0

    if normalized:
        m = max(node_centralities.values())
        for v in nodes:
            node_centralities[v] /= m

    return node_centralities


def node_traversals(paths):
    """Calculates the number of times any path traverses each of the nodes.

    Parameters
    ----------
    paths: Paths

    Returns
    -------
    dict
    """
    _Log.add('Calculating node traversals...', _Severity.INFO)

    # entries capture the number of times nodes are "visited by paths"
    # Note: this is identical to the subpath count of zero-length paths
    traversals = _co.defaultdict(lambda: 0)

    for p in paths.paths[0]:
        traversals[p[0]] += paths.paths[0][p].sum()

    _Log.add('finished.', _Severity.INFO)

    return traversals


def visitation_probabilities(paths):
    """Calculates the probabilities that a randomly chosen path passes through each of
    the nodes. If 5 out of 100 paths (of any length) traverse node v, node v will be
    assigned a visitation probability of 0.05. This measure can be interpreted as ground
    truth for the notion of importance captured by PageRank applied to a graphical
    abstraction of the paths.

    Parameters
    ----------
    paths: Paths

    Returns
    -------
    dict
    """
    _Log.add('Calculating visitation probabilities...', _Severity.INFO)

    # entries capture the probability that a given node is visited on an arbitrary path
    # Note: this is identical to the subpath count of zero-length paths
    # (i.e. the relative frequencies of nodes across all pathways)
    visit_probabilities = node_traversals(paths)

    # total number of visits
    visits = 0.0
    for v in visit_probabilities:
        visits += visit_probabilities[v]

    for v in visit_probabilities:
        visit_probabilities[v] /= visits

    _Log.add('finished.', _Severity.INFO)

    return visit_probabilities


def slow_down_factor(paths, k=2, lanczos_vectors=15, maxiter=1000):
    """Returns a factor S that indicates how much slower (S>1) or faster (S<1)
    a diffusion process evolves in a k-order model of the path statistics
    compared to what is expected based on a first-order model. This value captures
    the effect of order correlations of length k on a diffusion process which evolves
    based on the observed paths.

    Parameters
    ----------
    paths: Paths
        the Paths object to compute the slow down factor for
    k: int
        order of the model to consider
    lanczos_vectors: int
        number of Lanczos vectors to be used in the approximate calculation of
        eigenvectors and eigenvalues. This maps to the ncv parameter of scipy's
        underlying function eigs.
    maxiter: int
        scaling factor for the number of iterations to be used in the approximate
        calculation of eigenvectors and eigenvalues. The number of iterations passed to
        scipy's underlying eigs function will be n*maxiter where n is the number of
        rows/columns of the Laplacian matrix.

    Returns
    -------
    float
    """
    assert k > 1, 'Slow-down factor can only be calculated for orders larger than one'

    # NOTE to myself: most of the time goes for construction of the 2nd order
    # NOTE            null graph, then for the 2nd order null transition matrix

    gk = _HigherOrderNetwork(paths, k=k)
    gkn = _HigherOrderNetwork(paths, k=k, null_model=True)

    _Log.add('Calculating slow down factor ... ', _Severity.INFO)

    # Build transition matrices
    Tk = gk.transition_matrix()
    Tkn = gkn.transition_matrix()

    # Compute eigenvector sequences
    # NOTE: ncv=13 sets additional auxiliary eigenvectors that are computed
    # NOTE: in order to be more confident to find the one with the largest
    # NOTE: magnitude, see
    # NOTE: https://github.com/scipy/scipy/issues/4987
    w2 = _sla.eigs(Tk, which="LM", k=2, ncv=lanczos_vectors, return_eigenvectors=False,
                   maxiter=maxiter)
    eigen_values2_sorted = _np.sort(-_np.absolute(w2))

    w2n = _sla.eigs(Tkn, which="LM", k=2, ncv=lanczos_vectors, return_eigenvectors=False,
                    maxiter=maxiter)
    eigen_values2n_sorted = _np.sort(-_np.absolute(w2n))

    _Log.add('finished.', _Severity.INFO)

    abs_eigen_values2 = _np.abs(eigen_values2_sorted[1])
    abs_eigen_values2n = _np.abs(eigen_values2n_sorted[1])
    return _np.log(abs_eigen_values2n) / _np.log(abs_eigen_values2)


def entropy_growth_rate_ratio(paths, method='MLE', k=2, lanczos_vectors=15, maxiter=1000):
    """Computes the ratio between the entropy growth rate ratio between
    the k-order and first-order model of a temporal network t. Ratios smaller
    than one indicate that the temporal network exhibits non-Markovian characteristics

    Parameters
    ----------
    paths: Paths
    method:
        which method to use for the entropy calculation. The default 'MLE' uses
        the standard Maximum-Likelihood estimation of entropy. Setting method to
        'Miller' additionally applies a Miller-correction.
        see e.g.
    k: int
        order of the higher order network to generate
    lanczos_vectors:
        number of Lanczos vectors to be used in the approximate calculation of
        eigenvectors and eigenvalues. This maps to the ncv parameter of scipy's underlying
        function eigs.
    maxiter:
        Maximum number of Arnoldi update iterations allowed Default: n*10
        See scipy.sparse.linalg.eigs for more details


    References
    ---------
        Liam Paninski: Estimation of Entropy and Mutual Information, Neural Computation 5,
        2003 or
        http://www.nowozin.net/sebastian/blog/estimating-discrete-entropy-part-2.html

    Returns
    -------
    float
    """
    # NOTE to myself: most of the time here goes into computation of the
    # NOTE            EV of the transition matrix for the bigger of the
    # NOTE            two graphs below (either 2nd-order or 2nd-order null)

    assert (method == 'MLE' or method == 'Miller'), \
        'Only methods MLE or Miller are supported'

    # Generate k-order network
    gk = _HigherOrderNetwork(paths, k=k)
    g1 = _HigherOrderNetwork(paths, k=1)

    _Log.add('Calculating entropy growth rate ratio ... ', _Severity.INFO)

    # Compute entropy growth rate of observed transition matrix
    adj_matrix = g1.adjacency_matrix(weighted=False, transposed=True)
    tran_mat_k = gk.transition_matrix()
    leading_eigen_vec_k = _HigherOrderNetwork.leading_eigenvector(
        tran_mat_k, normalized=True, lanczos_vecs=lanczos_vectors, maxiter=maxiter
    )

    tran_mat_k.data *= _np.log2(tran_mat_k.data)

    # Apply Miller correction to the entropy estimation
    if method == 'Miller':
        # Here, 'possible_k_paths' is the number of different k-paths that can exist
        # based on the observed edges
        possible_k_paths = (adj_matrix ** k).sum()
        print('K = ', possible_k_paths)

        # N is the number of observations used to estimate the transition probabilities
        # in the second-order network. This corresponds to the total edge weight in the
        # k-order network, or - alternatively - to the number of paths of length k
        num_obs = 0
        for p in paths.paths[k]:
            num_obs += paths.paths[k][p].sum()
        print('N = ', num_obs)
        h_mat_k = (- _np.sum(tran_mat_k * leading_eigen_vec_k) +
                   (possible_k_paths - 1) / (2 * num_obs))
    else:
        # simple MLE estimation
        h_mat_k = - _np.sum(tran_mat_k * leading_eigen_vec_k)

    h_mat_k = _np.absolute(h_mat_k)

    # Compute entropy rate of null model
    gk_n = _HigherOrderNetwork(paths, k=k, null_model=True)

    # For the entropy rate of the null model, no Miller correction is needed
    # since we assume that transitions correspond to the true probabilities
    trans_mat_null = gk_n.transition_matrix()
    leading_eigen_v_null = _HigherOrderNetwork.leading_eigenvector(trans_mat_null)
    trans_mat_null.data *= _np.log2(trans_mat_null.data)
    h_mat_null = -_np.sum(trans_mat_null * leading_eigen_v_null)
    h_mat_null = _np.absolute(h_mat_null)

    _Log.add('finished.', _Severity.INFO)

    # Return ratio
    return h_mat_k / h_mat_null


def betweenness_preference_matrix(paths, v):
    """Computes a betweenness preference matrix for a node v

    Parameters
    ----------
    paths: Paths
    v:
        Node for which the betweenness preference matrix shall
        be calculated

    Returns
    -------

    """
    # create first-order network
    g = _HigherOrderNetwork(paths)

    in_degree = len(g.predecessors[v])
    out_degree = len(g.successors[v])

    index_succ = {}
    index_pred = {}

    B_v = _np.zeros(shape=(in_degree, out_degree))

    # Create an index-to-node mapping for predecessors and successors
    i = 0
    for u in g.predecessors[v]:
        index_pred[u] = i
        i = i + 1

    i = 0
    for w in g.successors[v]:
        index_succ[w] = i
        i = i + 1

    # Calculate entries of betweenness preference matrix
    for p in paths.paths[2]:
        if p[1] == v:
            B_v[index_pred[p[0]], index_succ[p[2]]] += paths.paths[2][p].sum()

    return B_v


def betweenness_preference(paths, v, normalized=False, method='MLE'):
    """Calculates the betweenness preferences of a node v based on the mutual information
     of path statistics of length two.

    Parameters
    ----------
    paths: Paths
    v:
        Node for which the betweenness preference will be calculated
    normalized: bool
        whether or not to normalize betweenness preference values
    method: str
        which method to use for the entropy calculation. The default 'MLE' uses the
        standard Maximum-Likelihood estimation of entropy. Setting method to 'Miller'
        additionally applies a Miller-correction.

    References
    ---------
        see e.g. Liam Paninski: Estimation of Entropy and Mutual Information,
        Neural Computation 5, 2003 or
        http://www.nowozin.net/sebastian/blog/estimating-discrete-entropy-part-2.html

    Returns
    -------

    """
    assert method in ['MLE', 'Miller'], "method must be one of 'MLE' or 'Miller'"

    # If the network is empty, just return zero
    if not paths.nodes():
        return 0.0

    # First create the betweenness preference matrix (equation (2) of the paper)
    B_v = betweenness_preference_matrix(paths, v)

    if B_v.shape[0] == 0 or B_v.shape[1] == 0:
        return None

    # Normalize matrix (equation (3) of the paper)
    # NOTE: P_v has the same shape as B_v
    P_v = _np.zeros(shape=B_v.shape)
    S = _np.sum(B_v)

    if S > 0:
        P_v = B_v / S

    # Compute marginal probabilities
    # Marginal probabilities P^v_d = \sum_s'{P_{s'd}}
    marginal_d = _np.sum(P_v, axis=0)

    # Marginal probabilities P^v_s = \sum_d'{P_{sd'}}
    marginal_s = _np.sum(P_v, axis=1)

    if method == 'Miller':

        # total number of samples, i.e. observed two-paths
        N = _np.sum(B_v)

        # print('N = ', N)
        # print('B = ', B_v)
        # print('marginal_s = ', marginal_s)
        # print('marginal_d = ', marginal_d)

        # marginal entropy H(S)
        H_s = _entropy(marginal_s, len(marginal_s), N, method='Miller')

        # print('H(S) = ', H_s)
        # marginal entropy H(D)

        H_d = _entropy(marginal_d, len(marginal_d), N, method='Miller')

        # print('H(D) = ', H_d)
        # we need the conditional entropy H(D|S)

        H_ds = 0
        for s in range(len(marginal_s)):
            # number of two paths s -> v -> * observed in the data
            N_s = _np.sum(B_v[s, :])

            # print('N(s=' + str(s) + ') = ' +  str(N_s))

            # probabilities of all destinations, given the particular source s
            p_ds = B_v[s, :] / _np.sum(B_v[s, :])

            # print('P(D|S=' + str(s) + ') = '+ str(p_ds))

            # number of possible destinations d
            K_s = len(p_ds)

            # print('K(s=' + str(s) + ') = ' +  str(K_s))

            # marginal_s[s] is the overall probability of source s
            p_s = marginal_s[s]

            # add to conditional entropy
            H_ds += p_s * _entropy(p_ds, K_s, N_s, method='Miller')

            I = H_d - H_ds

        # print('H(D|S) = ', H_ds)

    else:
        # use MLE estimation
        H_s = _entropy(marginal_s)
        H_d = _entropy(marginal_d)
        # H_ds = 0

        # for s in range(len(marginal_s)):
        #    print('s = ' + str(s) + ': ' + str(_np.sum(P_v[s,:])))
        #    p_ds = P_v[s,:]/_np.sum(P_v[s,:])
        #    H_ds += marginal_s[s] * Paths.__Entropy(p_ds)

        # Alternative calculation (without explicit entropies)
        # build mask for non-zero elements
        row, col = _np.nonzero(P_v)
        pv = P_v[(row, col)]
        marginal = _np.outer(marginal_s, marginal_d)
        log_argument = _np.divide(pv, marginal[(row, col)])
        I = _np.dot(pv, _np.log2(log_argument))

    # I = H_d - H_ds

    if normalized:
        I = I / _np.min([H_s, H_d])

    return I


def _entropy(prob, possible_outcomes=None, sample_size=None, method='MLE'):
    """Calculates the entropy of an (observed) probability distribution based on Maximum
    Likelihood Estimation (MLE) (default) or using a Miller correction.

    Parameters
    ----------
    prob:
        numpy array of the observed probabilities
    possible_outcomes: int
        the number of possible outcomes, i.e. outcomes with non-zero probability to be
        used for the Miller correction (default None)
    sample_size: int
        number of samples based on which observed probabilities where computed. This is
        needed for the Miller correction (default None)
    method: str
        The method to be used to calculate entropy. Can be 'MLE' (default) or 'Miller'

    Returns
    -------

    """
    assert method in ["MLE", "Miller"], "method must be one of 'MLE' or 'Miller'"
    if method == 'MLE':
        idx = _np.nonzero(prob)
        return -_np.inner(_np.log2(prob[idx]), prob[idx])
    elif method == 'Miller':
        assert (possible_outcomes is not None) and (sample_size is not None)
        if sample_size == 0:
            return 0

        idx = _np.nonzero(prob)

        addition = (possible_outcomes - 1) / (2 * sample_size)
        return -_np.inner(_np.log2(prob[idx]), prob[idx]) + addition
    else:
        raise PathpyError("method {} not supported".format(method))
