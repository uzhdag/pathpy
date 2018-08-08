import pathpy as pp
from pathpy.algorithms.shortest_paths import *

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

    assert path_from_ngram_file.nodes == {'a', 'b', 'c', 'd', 'e'}, \
        "Wrong node labels"


def test_write_read_file(tmpdir, random_paths):
    dir_path = tmpdir.mkdir("sub").join("test.edges")
    p = random_paths(30, 50)

    expected_seq = ''.join(p.sequence())
    expected_paths = sorted(expected_seq.split('|'))

    p.write_file(dir_path.strpath)
    p2 = pp.Paths.read_file(dir_path.strpath, frequency=True)

    read_back = ''.join(p2.sequence())
    read_back_paths = sorted(read_back.split('|'))

    assert expected_paths == read_back_paths


@pytest.mark.parametrize('maxN', (2, 90))
@pytest.mark.parametrize('max_line', (90, 3))
@pytest.mark.parametrize('freq', (True, False))
def test_write_file(tmpdir, random_paths, max_line, freq, maxN):
    dir_path = tmpdir.mkdir("sub").join("test.edges")
    p = random_paths(30, 50)

    p.write_file(dir_path.strpath)
    p2 = pp.Paths.read_file(dir_path.strpath, frequency=freq,
                            maxlines=max_line, max_ngram_length=maxN)
    assert p2


def test_read_edges_import(path_from_edge_file):
    """test if the Paths.read_edges functions works"""
    levels = list(path_from_edge_file.paths.keys())
    max_level = max(levels)
    expected_max = 1
    assert expected_max == max_level, \
        "The nodes have not been imported correctly"

    assert path_from_edge_file.nodes == {'1', '2', '3', '5'}, \
        "Nodes not imported correctly"


def test_read_edges_undirected(path_from_edge_file_undirected):
    p = path_from_edge_file_undirected
    layers = list(p.paths.keys())
    max_layers = max(layers)
    expected_layers = 1
    assert max_layers == expected_layers, \
        "The nodes have not been imported correctly"

    assert p.nodes == {'1', '2', '3', '5'}, \
        "Nodes not imported correctly"


def test_get_sequence(path_from_ngram_file):
    """Test if the Paths.sequence function works correctly"""
    sequence_raw = "".join(path_from_ngram_file.sequence())
    sequence = Counter(sequence_raw.split('|'))
    assert sequence == {'dedab': 4, 'abcdab': 2, '': 1}, \
        "Returned the wrong sequence"


def test_get_unique_paths(random_paths):
    p = random_paths(90, 90)
    assert p.unique_paths() == 87, \
        "Wrong number of paths detected"


def test_observation_count_file(path_from_ngram_file):
    assert path_from_ngram_file.observation_count == 6, \
        "Wrong number of observations detected"


def test_observation_count_large(random_paths):
    p = random_paths(90, 90)
    assert p.observation_count == 193, \
        "Wrong number of observations detected"


def test_path_summary(random_paths):
    p = random_paths(90, 90)
    print(p)


def test_summary_multi_order_model(random_paths):
    p = random_paths(90, 90)
    multi = pp.MultiOrderModel(paths=p, max_order=3)
    print(multi)


def test_get_shortest_paths(path_from_ngram_file):
    paths_dict = shortest_paths(path_from_ngram_file)
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
    cont_paths = pp.Paths.contained_paths(path_to_check, node_filter)
    expected = [('a', 'b'), ('d',), ('f', 'g')]
    assert cont_paths == expected


def test_filter_paths(path_from_ngram_file):
    p = path_from_ngram_file
    new_paths = p.filter_nodes(node_filter=['a', 'b', 'c'])
    expected_sequence = {'ab': 6, 'abc': 2, '': 1}
    new_sequence_raw = ''.join(new_paths.sequence())
    new_sequence = Counter(new_sequence_raw.split('|'))
    assert new_sequence == expected_sequence


def test_project_paths(path_from_ngram_file):
    p = path_from_ngram_file
    mapping = {'a': 'x', 'b': 'x', 'c': 'y', 'd': 'y', 'e': 'y'}
    new_p = p.project_paths(mapping=mapping)
    new_sequence_raw = ''.join(new_p.sequence())
    new_sequence = Counter(new_sequence_raw.split('|'))
    expected_sequence = {'yyyxx': 4, 'xxyyxx': 2, '': 1}
    assert new_sequence == expected_sequence


def test_get_nodes(random_paths):
    p = random_paths(3, 9)
    rest = p.nodes
    expected = {'b', 'o', 'u', 'v', 'w', 'y'}
    assert rest == expected


def test_get_path_lengths(path_from_ngram_file):
    p = path_from_ngram_file
    expected = {0: [32, 0], 1: [26, 0], 2: [20, 0], 3: [14, 0], 4: [4, 4], 5: [0, 2]}
    plengths = p.path_lengths()
    assert np.all([plengths[x] == expected[x] for x in expected])
    assert np.all([plengths[x] == expected[x] for x in plengths])


#def test_diameter(random_paths):
#    p = random_paths(3, 9)
#    assert p.diameter() == 5


def test_path_addition(random_paths):
    p1 = random_paths(20, 0, 5)
    p2 = random_paths(10, 0, 5)

    p12 = p1 + p2

    assert p12.nodes == (set(p1.nodes | set(p2.nodes)))


@pytest.mark.parametrize('factor', (1, 2, 3, 4, 10))
def test_path_multiplication(random_paths, factor):
    TEST_PATH = ('2', '3', '2')
    p = random_paths(20, 0, 5)  # base path

    # incrementally add path p using __add__
    sum_paths = random_paths(20, 0, 5)
    for i in range(factor-1):
        sum_paths = sum_paths + p

    assert sum(sum_paths.paths[2][TEST_PATH]) == sum(p.paths[2][TEST_PATH]) * factor

    # add in place p times to sum_inplace
    sum_inplace = random_paths(20, 0, 5)
    for i in range(factor-1):
        sum_inplace += p

    assert sum(sum_inplace.paths[2][TEST_PATH]) == sum(p.paths[2][TEST_PATH]) * factor

    # simple multiplication
    rmult_paths = factor * p
    mult_paths = p * factor
    assert sum(rmult_paths.paths[2][TEST_PATH]) == sum(mult_paths.paths[2][TEST_PATH])

    assert sum(mult_paths.paths[2][TEST_PATH]) == sum(sum_paths.paths[2][TEST_PATH])

    # multiplication inplace
    mult_inplace = random_paths(20, 0, 5)
    mult_inplace *= factor

    assert sum(mult_paths.paths[2][TEST_PATH]) == sum(mult_inplace.paths[2][TEST_PATH])


def test_pickle(random_paths, tmpdir):
    import pickle
    from pathpy import Paths

    dir_path = tmpdir.join("test_path.pkl")
    paths = random_paths(90, 0, 20)

    with open(str(dir_path), 'wb') as f:
        pickle.dump(paths, f)

    with open(str(dir_path), 'rb') as f:
        paths_back = pickle.load(f)  # type: Paths

    # assert diamter(paths) == diamter(paths_back)
    assert paths.paths.keys() == paths_back.paths.keys()
    assert paths.observation_count == paths.observation_count


# def test_expand_subpaths(random_paths):
#     paths = random_paths(2000, 0, 400)
#     paths.expand_subpaths()
#
#
# import pytest
# pytest.main(__file__)



