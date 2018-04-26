from functools import singledispatch
from collections import defaultdict
import operator

import numpy as _np
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import scipy.linalg as la
import scipy as sp

from pathpy.utils import Log, Severity
from pathpy import HigherOrderNetwork
from pathpy import Network
from pathpy import Paths
from pathpy.algorithms.shortest_paths import *

from pathpy.utils import PathpyNotImplemented


__all__ = ["rank_centralities", "closeness_centrality", "betweenness_centrality",
           "eigenvector_centrality", "pagerank", 'node_traversals',
           'visitation_probabilities']


def rank_centralities(centralities):
    """Returns a list of (node, centrality) tuples in which tuples are ordered 
    by centrality in descending order

    Parameters
    ----------
    centralities: dict
        dictionary of centralities

    Examples
    --------
    >>> centralities = {'a': .2, 'b': .8, 'c': .5}
    >>> rank_centralities(centralities)
    [('b', 0.8), ('c', 0.5), ('a', 0.2)]

    Returns
    -------
    list 
        list of (node,centrality) tuples

    """
    ranked_nodes = sorted(centralities.items(), key=operator.itemgetter(1))
    ranked_nodes.reverse()
    return ranked_nodes



@singledispatch
def betweenness_centrality(network, normalized=False):
    assert isinstance(network, Network), \
        "network must be an instance of Network"

    Log.add('Calculating betweenness centralities in network ...', Severity.INFO)

    all_paths = shortest_paths(network)
    node_centralities = defaultdict(lambda: 0)

    for s in all_paths:
        for d in all_paths[s]:
            for p in all_paths[s][d]:
                for x in p[1:-1]:
                    if s != d != x:
                        node_centralities[x] += 1.0 / len(all_paths[s][d])
    if normalized:
        max_centr = max(node_centralities.values())
        for v in node_centralities:
            node_centralities[v] /= max_centr

    # assign zero values to nodes not occurring on shortest paths
    for v in network.nodes:
        node_centralities[v] += 0

    return node_centralities
    


@betweenness_centrality.register(HigherOrderNetwork)
def _bw(higher_order_net, normalized=False):
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
    assert isinstance(higher_order_net, HigherOrderNetwork), \
        "arguments must be an instance of HigherOrderNetwork"

    Log.add('Calculating betweenness centralities (k = %s) ...' % \
        higher_order_net.order, Severity.INFO)

    all_paths = shortest_paths(higher_order_net)
    node_centralities = defaultdict(lambda: 0)

    shortest_paths_first_order = defaultdict(lambda: defaultdict(set))
    shortest_paths_first_order_lengths = defaultdict(lambda: defaultdict(lambda: _np.inf))

    for path_1_ord_k in all_paths:
        for path_2_ord_k in all_paths:
            source_k1 = higher_order_net.higher_order_node_to_path(path_1_ord_k)[0]
            dest_k1 = higher_order_net.higher_order_node_to_path(path_2_ord_k)[-1]

            # we consider a path in a k-th order network
            # connecting first-order node s1 to d1
            for path_ord_k in all_paths[path_1_ord_k][path_2_ord_k]:
                # convert k-th order path to first-order path and add
                #shortest_paths_first_order[source_k1][dest_k1].add(
                #    higher_order_net.higher_order_path_to_first_order(path_ord_k))

                p1 = higher_order_net.higher_order_path_to_first_order(path_ord_k)
                # obtain start node and end node
                s1 = p1[0]
                d1 = p1[-1]
                # compute the length of the first-order path
                l = len(p1) - 1
                # if path is a shortest path add it to dictionary
                if l < shortest_paths_first_order_lengths[s1][d1]:
                    shortest_paths_first_order_lengths[s1][d1] = l
                    shortest_paths_first_order[s1][d1] = set()
                    shortest_paths_first_order[s1][d1].add(p1)
                elif l == shortest_paths_first_order_lengths[s1][d1]:
                    shortest_paths_first_order[s1][d1].add(p1)

    for source_k1 in shortest_paths_first_order:
        for dest_k1 in shortest_paths_first_order[source_k1]:
            for path_k1 in shortest_paths_first_order[source_k1][dest_k1]:
                # increase betweenness centrality of all intermediary nodes
                # on path from s1 to d1
                for v in path_k1[1:-1]:
                    if source_k1 != v != dest_k1:
                        l_p = len(shortest_paths_first_order[source_k1][dest_k1])
                        node_centralities[v] += 1.0 / l_p
    if normalized:
        max_centr = max(node_centralities.values())
        for v in node_centralities:
            node_centralities[v] /= max_centr

    # assign centrality zero to nodes not occurring on higher-order shortest paths
    nodes = higher_order_net.paths.nodes
    for v in nodes:
        node_centralities[v] += 0

    Log.add('finished.', Severity.INFO)

    return node_centralities



@betweenness_centrality.register(Paths)
def _bw(paths, normalized=False):
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
    assert isinstance(paths, Paths), "argument must be an instance of pathpy.Paths"
    node_centralities = defaultdict(lambda: 0)

    Log.add('Calculating betweenness centralities in paths ...', Severity.INFO)

    all_paths = shortest_paths(paths)

    for s in all_paths:
        for d in all_paths[s]:
            for p in all_paths[s][d]:
                for x in p[1:-1]:
                    if s != d != x:
                        node_centralities[x] += 1.0 / len(all_paths[s][d])
    if normalized:
        max_centr = max(node_centralities.values())
        for v in node_centralities:
            node_centralities[v] /= max_centr

    # assign zero values to nodes not occurring on shortest paths
    nodes = paths.nodes
    for v in nodes:
        node_centralities[v] += 0
    Log.add('finished.')
    return node_centralities



@singledispatch
def closeness_centrality(network, normalized=False):
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
    if not isinstance(network, Network):
        raise PathpyNotImplemented("`network` must be an instance of Network")

    distances = distance_matrix(network)
    node_centralities = defaultdict(lambda: 0)

    Log.add('Calculating closeness centralities in network ...', Severity.INFO)

    # calculate closeness values
    for x in network.nodes:
        for d in network.nodes:
            if d != x and distances[d][x] < _np.inf:
                node_centralities[x] += 1.0 / distances[d][x]

    # assign centrality zero to nodes not occurring on higher-order shortest paths
    for v in network.nodes:
        node_centralities[v] += 0

    if normalized:
        max_centr = max(node_centralities.values())
        for v in network.nodes:
            node_centralities[v] /= max_centr

    Log.add('finished.', Severity.INFO)

    return node_centralities



@closeness_centrality.register(Paths)
def _cl(paths, normalized=False):
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
    node_centralities = defaultdict(lambda: 0)
    distances = distance_matrix(paths)
    nodes = paths.nodes

    for x in nodes:
        # calculate closeness centrality of x
        for d in nodes:
            if x != d and distances[d][x] < _np.inf:
                node_centralities[x] += 1.0 / distances[d][x]

    # assign zero values to nodes not occurring
    
    for v in nodes:
        node_centralities[v] += 0

    if normalized:
        m = max(node_centralities.values())
        for v in nodes:
            node_centralities[v] /= m

    return node_centralities



@closeness_centrality.register(HigherOrderNetwork)
def _cl(higher_order_net, normalized=False):

    if not isinstance(higher_order_net, HigherOrderNetwork):
        raise PathpyNotImplemented("`higher_order_net` must be an instance of HigherOrderNetwork")

    distances = distance_matrix(higher_order_net)
    node_centralities = defaultdict(lambda: 0)
    nodes = higher_order_net.paths.nodes

    Log.add('Calculating closeness centralities (k = %s) ...' % higher_order_net.order,
            Severity.INFO)
    
    for x in nodes:
        # calculate closeness centrality of x
        for d in nodes:
            if x != d and distances[d][x] < _np.inf:
                node_centralities[x] += 1.0 / distances[d][x]

    # assign centrality zero to those nodes for which no higher-order path exists
    for v in nodes:
        node_centralities[v] += 0

    if normalized:
        m = max(node_centralities.values())
        for v in nodes:
            node_centralities[v] /= m

    Log.add('finished.', Severity.INFO)

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
    if not isinstance(paths, Paths):
        raise PathpyNotImplemented("`paths` must be an instance of Paths")

    Log.add('Calculating node traversals...', Severity.INFO)

    # entries capture the number of times nodes are "visited by paths"
    # Note: this is identical to the subpath count of zero-length paths
    traversals = defaultdict(lambda: 0)

    for p in paths.paths[0]:
        traversals[p[0]] += paths.paths[0][p].sum()

    Log.add('finished.', Severity.INFO)

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
    if not isinstance(paths, Paths):
        raise PathpyNotImplemented("`paths` must be an instance of Paths")
    Log.add('Calculating visitation probabilities...', Severity.INFO)

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

    Log.add('finished.', Severity.INFO)

    return visit_probabilities



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
    _, v = sla.eigs(adj_mat, k=1, which="LM", ncv=13)

    v = v.reshape(v.size, )

    higher_order_eigen_vec_cent = dict(zip(network.nodes, map(_np.abs, v)))

    # project eigen_vec_cent of higher-order nodes to first-order network
    first_order_eigen_vec_cent = defaultdict(lambda: 0.0)

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

    Log.add('finished.', Severity.INFO)

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

    Log.add('Calculating PageRank in ' + str(network.order) + '-th order network...',
            Severity.INFO)

    higher_order_pr = defaultdict(lambda: 0)

    n_nodes = float(len(network.nodes))

    assert n_nodes > 0, "Number of nodes is zero"

    # entries A[s,t] give directed link s -> t
    adj_mat = network.adjacency_matrix(include_subpaths=include_sub_paths,
                                       weighted=weighted, transposed=False)

    # sum of outgoing node degrees
    row_sums = sp.array(adj_mat.sum(axis=1)).flatten()

    # replace non-zero entries x by 1/x
    row_sums[row_sums != 0] = 1.0 / row_sums[row_sums != 0]

    # indices of zero entries in row_sums
    d = sp.where(row_sums == 0)[0]

    # create sparse matrix with row_sums as diagonal elements
    q_mat = sparse.spdiags(row_sums.T, 0, adj_mat.shape[0], adj_mat.shape[1],
                           format='csr')

    # with this, we have divided elements in non-zero rows in A by 1 over the row sum
    q_mat = q_mat * adj_mat

    # vector with n entries 1/n
    inv_n_nodes = sp.array([1.0 / n_nodes] * int(n_nodes))

    p_rank = inv_n_nodes

    # Power iteration
    for _ in range(max_iter):
        last = p_rank

        # sum(pr[d]) is the sum of PageRanks for nodes with zero out-degree
        # sum(pr[d]) * p yields a vector with length n
        p_rank = (alpha * (p_rank * q_mat + sum(p_rank[d]) * inv_n_nodes) +
                  (1 - alpha) * inv_n_nodes)

        if sp.absolute(p_rank - last).sum() < n_nodes * tol:
            higher_order_pr = dict(zip(network.nodes, map(float, p_rank)))
            break

    if network.order == 1:
        return higher_order_pr

    # project PageRank of higher-order nodes to first-order network
    first_order_pr = defaultdict(lambda: 0.0)

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
    nodes = network.paths.nodes
    for v in nodes:
        first_order_pr[v] += 0

    Log.add('finished.', Severity.INFO)

    return first_order_pr
