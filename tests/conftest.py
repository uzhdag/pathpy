import pathpy as pp
import pytest
import numpy as np
import os

test_directory = os.path.dirname(os.path.abspath(__file__))
test_data_dir = os.path.join(test_directory, 'test_data')


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")


def pytest_runtest_setup(item):
    if 'slow' in item.keywords and not item.config.getvalue("runslow"):
        pytest.skip("need --runslow option to run")


@pytest.fixture()
def test_data_directory():
    return test_data_dir


@pytest.fixture()
def path_from_ngram_file():
    """load the example file as pypath.Path"""
    ngram_file_path = os.path.join(test_data_dir, 'ngram_simple.ngram')
    path = pp.Paths.read_file(ngram_file_path, frequency=True)
    return path


@pytest.fixture()
def path_from_edge_file():
    file_path = os.path.join(test_data_dir, 'edge_frequency.edge')
    path = pp.Paths.read_edges(file_path, weight=True)
    return path


@pytest.fixture()
def path_from_edge_file_undirected():
    file_path = os.path.join(test_data_dir, 'edge_frequency.edge')
    path = pp.Paths.read_edges(file_path, weight=True, undirected=True, maxlines=5)
    return path


def generate_random_path(size, rnd_seed, num_nodes=None):
    """Generate a Path with random path sequences"""
    if num_nodes is None:
        import string
        node_set = string.ascii_lowercase
    else:
        node_set = [str(x) for x in range(num_nodes)]

    def random_ngram(p_len, nodes):
        num_elements = len(nodes)
        sequence = np.random.choice(num_elements, p_len)
        path = [nodes[i] for i in sequence]
        return ','.join(path)

    np.random.seed(rnd_seed)
    paths = pp.Paths()
    for _ in range(size):
        frequency = np.random.randint(1, 4)
        path_length = np.random.randint(1, 10)
        path_to_add = random_ngram(path_length, node_set)
        paths.add_path_ngram(path_to_add, frequency=frequency)

    return paths


@pytest.fixture(scope='function')
def random_paths():
    """Generate a Path with random path sequences"""
    return generate_random_path


@pytest.fixture()
def temporal_network_object():
    t = pp.TemporalNetwork()
    # Path of length two
    t.add_edge("c", "e", 1)
    t.add_edge("e", "f", 2)

    # Path of length two
    t.add_edge("a", "e", 3)
    t.add_edge("e", "g", 4)

    # Path of length two
    t.add_edge("c", "e", 5)
    t.add_edge("e", "f", 6)

    # Path of length two
    t.add_edge("a", "e", 7)
    t.add_edge("e", "g", 8)

    # Path of length two
    t.add_edge("c", "e", 9)
    t.add_edge("e", "f", 10)

    # The next two edges continue the previous path to ( c-> e-> f-> e -> b )
    t.add_edge("f", "e", 11)
    t.add_edge("e", "b", 12)

    # This is an isolated edge (i.e. path of length one)
    t.add_edge("e", "b", 13)

    # Path of length two
    t.add_edge("c", "e", 14)
    t.add_edge("e", "f", 15)

    # Path of length two
    t.add_edge("b", "e", 16)
    t.add_edge("e", "g", 17)

    # Path of length two
    t.add_edge("c", "e", 18)
    t.add_edge("e", "f", 19)

    # Path of length two
    t.add_edge("c", "e", 20)
    t.add_edge("e", "f", 21)

    return t


@pytest.fixture()
def dag_object():
    dag = pp.DAG()
    # For this DAG, the following five paths between the root and the leaves exist
    # for the following mapping:
    # mapping = {'a': 'A', 'b': 'B', 'c': 'A', 'e': 'B',
    # 'f': 'B', 'g': 'A', 'h': 'A','i': 'B', 'j': 'A' }

    #   h -> i                  ( A -> B )
    #   h -> j                  ( A -> A )
    #   a -> b -> e             ( A -> B -> B )
    #   a -> c -> g             ( A -> A -> A )
    #   a -> b -> f -> g        ( A -> B -> B -> A )
    #   a -> c -> b -> e        ( A -> A -> B -> B )
    #   a -> c -> b -> f -> g   ( A -> A -> B -> B -> A )
    dag.add_edge('a', 'b')
    dag.add_edge('a', 'c')
    dag.add_edge('c', 'b')
    dag.add_edge('b', 'e')
    dag.add_edge('b', 'f')
    dag.add_edge('f', 'g')
    dag.add_edge('c', 'g')
    dag.add_edge('h', 'i')
    dag.add_edge('h', 'j')
    return dag
