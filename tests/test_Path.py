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
from collections import Counter

slow = pytest.mark.slow


def test_readfile_import(path_from_ngram_file):
    levels = list(path_from_ngram_file.paths.keys())
    max_level = max(levels)
    expected = 5
    assert max_level == expected, \
        "The nodes have not been imported correctly"

    assert path_from_ngram_file.getNodes() == {'a', 'b', 'c', 'd', 'e'}, \
        "Wrong node labels"


def test_write_file(tmpdir, random_paths):
    dir_path = tmpdir.mkdir("sub").join("test.edges")
    p = random_paths(30, 50)

    expected_seq = ''.join(p.getSequence())
    expected_paths = sorted(expected_seq.split('|'))

    p.writeFile(dir_path.strpath)
    p2 = pp.Paths.readFile(dir_path.strpath, pathFrequency=True)

    read_back = ''.join(p2.getSequence())
    read_back_paths = sorted(read_back.split('|'))

    assert expected_paths == read_back_paths


def test_read_edges_import(path_from_edge_file):
    """test if the Paths.readEdges functions works"""
    levels = list(path_from_edge_file.paths.keys())
    max_level = max(levels)
    expected_max = 1
    assert expected_max == max_level, \
        "The nodes have not been imported correctly"

    assert path_from_edge_file.getNodes() == {'1', '2', '3', '5'}, \
        "Nodes not imported correctly"


def test_read_edges_undirected(path_from_edge_file_undirected):
    p = path_from_edge_file_undirected
    layers = list(p.paths.keys())
    max_layers = max(layers)
    expected_layers = 1
    assert max_layers == expected_layers, \
        "The nodes have not been imported correctly"

    assert p.getNodes() == {'1', '2', '3', '5'}, \
        "Nodes not imported correctly"


def test_get_sequence(path_from_ngram_file):
    """Test if the Paths.getSequence function works correctly"""
    sequence_raw = "".join(path_from_ngram_file.getSequence())
    sequence = Counter(sequence_raw.split('|'))
    assert sequence == {'dedab': 4, 'abcdab': 2, '': 1}, \
        "Returned the wrong sequence"


def test_get_unique_paths(random_paths):
    p = random_paths(90, 90)
    assert p.getUniquePaths() == 87, \
        "Wrong number of paths detected"


def test_observation_count_file(path_from_ngram_file):
    assert path_from_ngram_file.ObservationCount() == 6, \
        "Wrong number of observations detected"


def test_observation_count_large(random_paths):
    p = random_paths(90, 90)
    assert p.ObservationCount() == 193, \
        "Wrong number of observations detected"


def test_path_summary(random_paths):
    p = random_paths(90, 90)
    print(p)


def test_summary_multi_order_model(random_paths):
    p = random_paths(90, 90)
    multi = pp.MultiOrderModel(paths=p, maxOrder=3)
    print(multi)


def test_get_shortest_paths(path_from_ngram_file):
    path_from_ngram_file.getShortestPaths()
    paths_dict = path_from_ngram_file.getShortestPaths()
    expected_paths = {('d', 'a'): {('d', 'a')},
                      ('b', 'd'): {('b', 'c', 'd')},
                      ('d', 'e'): {('d', 'e')},
                      ('a', 'c'): {('a', 'b', 'c')},
                      ('a', 'a'): {('a',)},
                      ('e', 'a'): {('e', 'd', 'a')},
                      ('e', 'b'): {('e', 'd', 'a', 'b')},
                      ('e', 'e'): {('e',)},
                      ('a', 'b'): {('a', 'b')},
                      ('b', 'b'): {('b',)},
                      ('c', 'd'): {('c', 'd')},
                      ('d', 'b'): {('d', 'a', 'b')},
                      ('c', 'a'): {('c', 'd', 'a')},
                      ('b', 'a'): {('b', 'c', 'd', 'a')},
                      ('c', 'b'): {('c', 'd', 'a', 'b')},
                      ('e', 'd'): {('e', 'd')},
                      ('a', 'd'): {('a', 'b', 'c', 'd')},
                      ('d', 'd'): {('d',)},
                      ('c', 'c'): {('c',)},
                      ('b', 'c'): {('b', 'c')}
                      }
    paths_to_check = dict()
    for k in paths_dict:
        for p in paths_dict[k]:
            paths_to_check[(k, p)] = paths_dict[k][p]
    assert paths_to_check == expected_paths


def test_get_contained_paths():
    path_to_check = ('a', 'b', 'c', 'd', 'e', 'f', 'g')
    node_filter = ('a', 'b', 'd', 'f', 'g')
    cont_paths = pp.Paths.getContainedPaths(path_to_check, node_filter)
    expected = [('a', 'b'), ('d',), ('f', 'g')]
    assert cont_paths == expected


def test_filter_paths(path_from_ngram_file):
    p = path_from_ngram_file
    new_paths = p.filterPaths(node_filter=['a', 'b', 'c'])
    expected_sequence = {'ab': 6, 'abc': 2, '': 1}
    new_sequence_raw = ''.join(new_paths.getSequence())
    new_sequence = Counter(new_sequence_raw.split('|'))
    assert new_sequence == expected_sequence


def test_project_paths(path_from_ngram_file):
    p = path_from_ngram_file
    mapping = {'a': 'x', 'b': 'x', 'c': 'y', 'd': 'y', 'e': 'y'}
    new_p = p.projectPaths(mapping=mapping)
    new_sequence_raw = ''.join(new_p.getSequence())
    new_sequence = Counter(new_sequence_raw.split('|'))
    expected_sequence = {'yyyxx': 4, 'xxyyxx': 2, '': 1}
    assert new_sequence == expected_sequence


def test_get_nodes(random_paths):
    p = random_paths(3, 9)
    rest = p.getNodes()
    expected = {'b', 'o', 'u', 'v', 'w', 'y'}
    assert rest == expected


def test_get_path_lengths(path_from_ngram_file):
    p = path_from_ngram_file
    expected = {0: [32, 0], 1: [26, 0], 2: [20, 0], 3: [14, 0], 4: [4, 4], 5: [0, 2]}
    plengths = p.getPathLengths()
    assert np.all([plengths[x] == expected[x] for x in expected])
    assert np.all([plengths[x] == expected[x] for x in plengths])


def test_dag_acyclic(dag_object):
    dag = dag_object
    dag.topsort()
    assert dag.isAcyclic == True

    # Add cycle to the graph
    dag.addEdge('b', 'c')
    dag.topsort()
    assert dag.edge_classes[('b', 'c')] == 'back' or dag.edge_classes[('c', 'b')] == 'back'
    assert dag.isAcyclic == False

    dag.makeAcyclic()
    assert ('b', 'c') not in dag.edges or ('c', 'b') not in dag.edges
    assert len(dag.edges) == 9


def test_dag_path_extraction(dag_object):
    dag = dag_object
    dag.topsort()

    # Extract paths between nodes in DAG
    paths = pp.Paths.fromDAG(dag)
    assert paths.ObservationCount() == 7


def test_dag_path_mapping(dag_object):
    dag = dag_object
    dag.topsort()

    mapping = {'a': 'A', 'b': 'B', 'c': 'A', 'e': 'B', 'f': 'B', 'g': 'A', 'h': 'A',
               'i': 'B', 'j': 'A'}
    paths_mapped2 = pp.Paths.fromDAG(dag, node_mapping=mapping)
    assert paths_mapped2.paths[1][('A', 'B')][1] == 1
    assert paths_mapped2.paths[1][('A', 'A')][1] == 1
    assert paths_mapped2.paths[2][('A', 'B', 'B')][1] == 1
    assert paths_mapped2.paths[2][('A', 'A', 'A')][1] == 1
    assert paths_mapped2.paths[3][('A', 'B', 'B', 'A')][1] == 1
    assert paths_mapped2.paths[3][('A', 'A', 'B', 'B')][1] == 1
    assert paths_mapped2.paths[4][('A', 'A', 'B', 'B', 'A')][1] == 1
    assert paths_mapped2.ObservationCount() == 7
