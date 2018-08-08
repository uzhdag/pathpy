import pathpy as pp
import pytest
import numpy as np
import itertools
from pathpy.algorithms import shortest_paths


slow = pytest.mark.slow


def dict_of_dicts_to_matrix(network, max_val=np.inf, agg=None):
    """return a numpy matrix representation fo the given dict of dicts
    optionally apply an aggregator function and set a maximum value"""
    N = max(max(len(x) for x in network.values()), len(network))
    matrix = np.zeros(shape=(N, N))
    print(sorted(network))
    for i, source in enumerate(sorted(network)):
        for j, target in enumerate(sorted(network[source])):
            values = network[source][target]
            if agg:
                value = agg(values)
            else:
                value = values

            if max_val is not None and value >= max_val:
                value = 0
            matrix[i, j] = value

    return matrix


def test_summary(path_from_edge_file):
    hon_1 = pp.HigherOrderNetwork(path_from_edge_file, k=1)
    print(hon_1)


@pytest.mark.parametrize('k, n_nodes, expected', (
        (1, 10, np.array([121., 9.])),
        (2, 10, np.array([91., 6.]))
))
def test_edge_weight(random_paths, k, n_nodes, expected):
    p = random_paths(20, 10, n_nodes)
    hon_1 = pp.HigherOrderNetwork(p, k=k)
    assert np.allclose(hon_1.total_edge_weight(), expected)


@pytest.mark.parametrize('k, n_nodes, expected', (
        (1, 10, 44),
        (2, 10, 46),
        (3, 10, 36),
        (4, 10, 27),
        (5, 10, 19),
        (6, 10, 12)
))
def test_model_size(random_paths, k, n_nodes, expected):
    p = random_paths(20, 10, n_nodes)
    hon_1 = pp.HigherOrderNetwork(p, k=k)
    assert np.allclose(hon_1.model_size(), expected)


def test_degrees(path_from_edge_file):
    hon_1 = pp.HigherOrderNetwork(path_from_edge_file, k=1)
    expected_degrees = {'1': 52, '2': 0, '3': 2, '5': 5}
    for v in hon_1.nodes:
        assert expected_degrees[v] == hon_1.nodes[v]["outweight"][1], \
            "Wrong degree calculation in HigherOrderNetwork"


def test_distance_matrix_from_file(path_from_edge_file):
    p = path_from_edge_file
    hon = pp.HigherOrderNetwork(paths=p, k=1)
    d_matrix = shortest_paths.distance_matrix(hon)

    np_matrix = dict_of_dicts_to_matrix(d_matrix)
    assert np.sum(np_matrix) == 8
    assert np.min(np_matrix) == 0
    assert np.max(np_matrix) == 2


def test_distance_matrix_equal_across_objects(random_paths):
    """test that the distance matrix is the same if constructed from to path objects with
    the same paths but different instances"""
    p1 = random_paths(40, 20, num_nodes=9)
    p2 = random_paths(40, 20, num_nodes=9)
    hon1 = pp.HigherOrderNetwork(paths=p1, k=1)
    hon2 = pp.HigherOrderNetwork(paths=p2, k=1)
    d_matrix1 = shortest_paths.distance_matrix(hon1)
    d_matrix2 = shortest_paths.distance_matrix(hon2)
    assert d_matrix1 == d_matrix2


@pytest.mark.parametrize('paths, n_nodes, k, e_var, e_sum', (
        (7, 9, 1, 1.015394, 121),
        (60, 20, 1, 0.346694, 577),
        (7, 9, 2, 3.628105, 116),
))
def test_distance_matrix(random_paths, paths, n_nodes, k, e_var, e_sum):
    p = random_paths(paths, 20, num_nodes=n_nodes)
    hon = pp.HigherOrderNetwork(paths=p, k=k)
    d_matrix = shortest_paths.distance_matrix(hon)

    np_matrix = dict_of_dicts_to_matrix(d_matrix)

    assert np.var(np_matrix) == pytest.approx(e_var)
    assert np.sum(np_matrix) == e_sum


@pytest.mark.parametrize('paths, k, num_nodes, s_mean, s_var, s_max', (
        (20, 1, 10, 1.47, 0.4891, 4),
        (20, 2, 10, 0.693877, 0.42556342, 2)
))
def test_shortest_path_length(random_paths, paths, k, num_nodes, s_mean, s_var, s_max):
    p = random_paths(paths, 10, num_nodes=num_nodes)
    hon = pp.HigherOrderNetwork(p, k=k)

    all_paths = shortest_paths.shortest_paths(hon)

    distances = dict_of_dicts_to_matrix(all_paths, agg=len)
    assert np.mean(distances) == pytest.approx(s_mean)
    assert np.var(distances) == pytest.approx(s_var)
    assert np.max(distances) == s_max


def test_node_name_map(random_paths):
    p = random_paths(20, 10, 20)
    hon = pp.HigherOrderNetwork(p, k=1)
    node_map = hon.node_to_name_map()

    assert set(node_map) == set(hon.nodes)


@pytest.mark.parametrize('paths, k_order, sub, num_nodes, s_sum, s_mean', (
        (20, 1, True, 10, 130, 1.3),
        (20, 2, True, 10, 97, 0.0549887),
        (20, 1, False, 10, 9, 0.09),
        (20, 2, False, 10, 6, 0.0034013605)
))
def test_get_adjacency_mat(random_paths, paths, k_order, sub, num_nodes, s_sum, s_mean):
    p = random_paths(paths, 10, num_nodes)
    hon = pp.HigherOrderNetwork(p, k=k_order)
    adj = hon.adjacency_matrix(include_subpaths=sub)
    assert adj.sum() == s_sum
    assert adj.mean() == pytest.approx(s_mean)


def test_laplacian_matrix(random_paths):
    paths = random_paths(30, 10, 5)
    hon = pp.HigherOrderNetwork(paths, k=1)
    L = hon.laplacian_matrix().toarray()
    assert np.trace(L) > 0
    assert np.tril(L, k=-1).sum() < 0
    assert np.triu(L, k=1).sum() < 0


@pytest.mark.parametrize('sub', (True, False))
@pytest.mark.parametrize('k', (1, 2, 5))
def test_transition_probability(random_paths, k, sub):
    paths = random_paths(30, 45, 14)
    hon = pp.HigherOrderNetwork(paths, k=k)
    T = hon.transition_matrix(include_subpaths=sub).toarray()
    if sub:
        transitions = sum(hon.nodes[w]["outweight"].sum() > 0 for w in hon.nodes)
    else:
        transitions = sum(hon.nodes[x]["outweight"][1] > 0 for x in hon.nodes)
    assert T.sum() == pytest.approx(transitions)
    assert np.all(T <= 1), "not all probabilities are smaller then 1"
    assert np.all(T >= 0), "not all probabilities are positive"


@pytest.mark.parametrize('num_nodes', (5, 8, 10))
@pytest.mark.parametrize('paths', (10, 20, 50))
def test_distance_matrix_first_order_eq_dist_matrix(random_paths, paths, num_nodes):
    """test that the distance matrix of k=1 is equal to
    distance_matrix_first_order"""
    p = random_paths(paths, 10, num_nodes)
    hon = pp.HigherOrderNetwork(p, k=1)
    dist = shortest_paths.distance_matrix(hon)
    dist_alt = shortest_paths.distance_matrix(hon)
    m = dict_of_dicts_to_matrix(dist)
    m_alt = dict_of_dicts_to_matrix(dist_alt)
    assert np.allclose(m, m_alt)


@pytest.mark.parametrize('paths, n_nodes, k, e_sum', (
        (7, 9, 1, 79),
        (60, 20, 1, 613),
        (10, 9, 2, 140),  # creates problems
))
def test_distance_matrix_first_order(random_paths, n_nodes, k, paths, e_sum):
    p = random_paths(paths, 10, n_nodes)
    hon_k, hon_1 = pp.HigherOrderNetwork(p, k=k), pp.HigherOrderNetwork(p, k=1)
    dist_k = shortest_paths.distance_matrix(hon_k)
    dist_1 = shortest_paths.distance_matrix(hon_1)
    total_distance = 0
    for source, target in itertools.product(hon_1.nodes, hon_1.nodes):
        dist_st = dist_k[source][target]
        assert dist_1[source][target] <= dist_k[source][target], \
            "not all distances at order k are at least as long as at order 1"
        if dist_st < np.inf:
            total_distance += dist_st

    assert total_distance == e_sum


def test_d3_html(random_paths, tmpdir):
    file_path = str(tmpdir.mkdir("sub").join("d3_hon.html"))
    p = random_paths(50, 10, 20)
    hon = pp.HigherOrderNetwork(p, k=3)
    pp.visualisation.export_html(hon, file_path)
