# -*- coding: utf-8 -*-
"""
    pathpy is an OpenSource python package for the analysis of sequential data
    on pathways and temporal networks using higher- and multi order graphical models

    Copyright (C) 2016-2017 Ingo Scholtes, ETH Zürich

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Contact the developer:

    E-mail: ischoltes@ethz.ch
    Web:    http://www.ingoscholtes.net
"""

import pathpy as pp
import pytest
import numpy as np


slow = pytest.mark.slow


def dict_of_dicts_to_matrix(network, max_val=1e9, agg=None):
    """return a numpy matrix representation fo the given dict of dicts
    optionally apply an aggregator function"""
    N = len(network)
    matrix = np.zeros(shape=(N, N))
    for i, source in enumerate(sorted(network)):
        for j, target in enumerate(sorted(network[source])):
            values = network[source][target]
            if agg:
                value = agg(values)
            else:
                value = values
            matrix[i, j] = value if value < max_val else 0

    return matrix


def test_summary(path_from_edge_file):
    hon_1 = pp.HigherOrderNetwork(path_from_edge_file, k=1)
    print(hon_1)


def test_degrees(path_from_edge_file):
    hon_1 = pp.HigherOrderNetwork(path_from_edge_file, k=1)
    expected_degrees = {'1': 52, '2' : 0, '3': 2, '5': 5}
    for v in hon_1.nodes:
        assert expected_degrees[v] == hon_1.outweights[v][1], \
        "Wrong degree calculation in HigherOrderNetwork"


def test_distance_matrix_from_file(path_from_edge_file):
    p = path_from_edge_file
    hon = pp.HigherOrderNetwork(paths=p, k=1)
    d_matrix = hon.getDistanceMatrix()

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
    d_matrix1 = hon1.getDistanceMatrix()
    d_matrix2 = hon2.getDistanceMatrix()
    assert d_matrix1 == d_matrix2


@pytest.mark.parametrize('paths, n_nodes, k, e_var, e_sum', (
        (7, 9, 1, 1.015394, 121),
        (60, 20, 1, 0.346694, 577),
        (7, 9, 2, 2.69493, 314),
))
def test_distance_matrix(random_paths, paths, n_nodes, k, e_var, e_sum):
    p = random_paths(paths, 20, num_nodes=n_nodes)
    hon = pp.HigherOrderNetwork(paths=p, k=k)
    d_matrix = hon.getDistanceMatrix()

    np_matrix = dict_of_dicts_to_matrix(d_matrix)

    assert np.var(np_matrix) == pytest.approx(e_var)
    assert np.sum(np_matrix) == e_sum


@pytest.mark.parametrize('paths, k_order, num_nodes, s_mean, s_var, s_max', (
        (20, 1, 10, 1.47, 0.4891, 4),
        (20, 2, 10, 0.693877, 0.42556342, 2)
))
def test_shortest_path_length(random_paths, paths, k_order, num_nodes, s_mean, s_var, s_max):
    p = random_paths(paths, 10, num_nodes=num_nodes)
    hon = pp.HigherOrderNetwork(p, k=k_order)

    shortest_paths = hon.getShortestPaths()

    distances = dict_of_dicts_to_matrix(shortest_paths, agg=len)
    assert np.mean(distances) == pytest.approx(s_mean)
    assert np.var(distances) == pytest.approx(s_var)
    assert np.max(distances) == s_max


@pytest.mark.xfail
def test_node_name_map(random_paths):
    p = random_paths(20, 10, 20)
    hon = pp.HigherOrderNetwork(p, k=1)
    node_map = hon.getNodeNameMap()
    # TODO: this is just an idea of how the mapping could be unique
    assert node_map == {str(i): i+1 for i in range(20)}


@pytest.mark.parametrize('paths, k_order, num_nodes, s_sum, s_mean', (
        (20, 1, 10, 130, 1.3),
        (20, 2, 10, 97, 0.0549887),
))
def test_get_adjacency_matrix(random_paths, paths, k_order, num_nodes, s_sum, s_mean):
    p = random_paths(paths, 10, num_nodes)
    hon = pp.HigherOrderNetwork(p, k=k_order)
    adj = hon.getAdjacencyMatrix()
    assert adj.sum() == s_sum
    assert adj.mean() == pytest.approx(s_mean)


def test_laplacian_matrix(random_paths):
    paths = random_paths(30, 10, 5)
    hon = pp.HigherOrderNetwork(paths, k=1)
    L = hon.getLaplacianMatrix().toarray()
    assert np.trace(L) > 0
    assert np.tril(L, k=-1).sum() < 0
    assert np.triu(L, k=1).sum() < 0


@pytest.mark.parametrize('k', (1, 2, 3))
def test_transition_probability(random_paths, k):
    paths = random_paths(10, 10, 3)
    hon = pp.HigherOrderNetwork(paths, k=k)
    T = hon.getTransitionMatrix(includeSubPaths=True).toarray()
    num_nodes = len(hon.nodes)
    # count nodes only at head of edge
    # these will have a column of 0 since they are absorbing
    # and therefore the sum of probabilities will be the number of nodes
    # minus the absorbing set
    head_nodes = {e[1] for e in hon.edges}
    tail_nodes = {e[0] for e in hon.edges}
    only_head = head_nodes - tail_nodes
    num_transitions = num_nodes - len(only_head)
    assert T.sum() == pytest.approx(num_transitions)
    assert np.all(T <= 1), "not all probabilities are smaller then 1"
    assert np.all(T >= 0), "not all probabilities are positive"


@pytest.mark.parametrize('num_nodes', (5, 8, 10))
@pytest.mark.parametrize('paths', (10, 20, 50))
def test_distance_matrix_first_order_eq_dist_matrix(random_paths, paths, num_nodes):
    """test that the distance matrix of k=1 is equal to
    getDistanceMatrixFirstOrder"""
    p = random_paths(paths, 10, num_nodes)
    hon = pp.HigherOrderNetwork(p, k=1)
    dist = hon.getDistanceMatrixFirstOrder()
    dist_alt = hon.getDistanceMatrix()
    m = dict_of_dicts_to_matrix(dist)
    m_alt = dict_of_dicts_to_matrix(dist_alt)
    assert np.allclose(m, m_alt)


# @pytest.mark.parametrize('k', (1, 2, 3))
# @pytest.mark.parametrize('num_nodes', (5, 8, 10))
# @pytest.mark.parametrize('paths', (10, 20, 50))
def test_distance_matrix_first_order(random_paths):
    """test that the distance matrix of k=1 is equal to
    getDistanceMatrixFirstOrder"""
    p = random_paths(30, 10, 5)
    hon = pp.HigherOrderNetwork(p, k=2)
    dist = hon.getDistanceMatrixFirstOrder()
    m = dict_of_dicts_to_matrix(dist)
    print(m)
    assert 0
