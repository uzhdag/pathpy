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


def test_degrees(path_from_edge_file):
    hon_1 = pp.HigherOrderNetwork(path_from_edge_file, k=1)
    expected_degrees = {'1': 52, '2' : 0, '3': 2, '5': 5}
    for v in hon_1.nodes:
        assert expected_degrees[v] == hon_1.outweights[v][1], \
        "Wrong degree calculation in HigherOrderNetwork"


def test_distance_matrix(path_from_edge_file):
    p = path_from_edge_file
    hon = pp.HigherOrderNetwork(paths=p, k=1)
    d_matrix = hon.getDistanceMatrix()
    distances = []
    for source in sorted(d_matrix):
        for target in sorted(d_matrix[source]):
            distance = d_matrix[source][target]
            if distance < 1e6:
                distances.append(d_matrix[source][target])

    assert np.sum(distances) == 8
    assert np.min(distances) == 0
    assert np.max(distances) == 2


def test_distance_matrix_equal_across_objects(random_paths):
    p1 = random_paths(40, 20, num_nodes=9)
    p2 = random_paths(40, 20, num_nodes=9)
    hon1 = pp.HigherOrderNetwork(paths=p1, k=1)
    hon2 = pp.HigherOrderNetwork(paths=p2, k=1)
    d_matrix1 = hon1.getDistanceMatrix()
    d_matrix2 = hon2.getDistanceMatrix()
    assert d_matrix1 == d_matrix2


@pytest.mark.parametrize('paths,n_nodes,k,e_var,e_sum', (
        (7, 9, 1, 0.7911428035, 123),
        (20, 9, 1, 0.310318549, 112),
        (60, 20, 1, 0.2941, 588),
))
def test_distance_matrix_large(random_paths, paths, n_nodes, k, e_var, e_sum):
    p = random_paths(paths, 20, num_nodes=n_nodes)
    hon = pp.HigherOrderNetwork(paths=p, k=1)
    d_matrix = hon.getDistanceMatrix()
    distances = []
    for i, source in enumerate(sorted(d_matrix)):
        for j, target in enumerate(sorted(d_matrix[source])):
            distance = d_matrix[source][target]
            if distance < 1e16:
                distances.append(d_matrix[source][target])

    assert np.var(distances) == pytest.approx(e_var)
    assert np.sum(distances) == e_sum
