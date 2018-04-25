import pathpy as pp
import pytest
import numpy as np


@pytest.mark.parametrize('edge_list', (
        [(1, 2), (2, 3), (5, 2)],
        list(zip(range(9), range(2, 11))) + list(zip(range(10), range(2, 12))),  # redundant
        list(zip(range(8), range(2, 10))) + list(zip(range(8), range(8)))  # self-loops
))
def test_dag_init(edge_list):
    dag = pp.DAG(edges=edge_list)
    print(dag)


def test_dag_acyclic(dag_object):
    dag = dag_object
    dag.topsort()
    assert dag.is_acyclic is True

    # Add cycle to the graph
    dag.add_edge('b', 'c')
    dag.topsort()
    assert (dag.edge_classes[('b', 'c')] == 'back' or
            dag.edge_classes[('c', 'b')] == 'back')
    assert dag.is_acyclic is False

    dag.make_acyclic()
    assert ('b', 'c') not in dag.edges or ('c', 'b') not in dag.edges
    assert len(dag.edges) == 9


def test_dag_path_extraction(dag_object):
    dag = dag_object
    dag.topsort()

    # Extract paths between nodes in DAG
    paths = pp.path_extraction.paths_from_dag(dag)
    assert paths.observation_count == 7


def test_dag_path_extraction_cyclic(dag_object: pp.DAG):
    dag_object.add_edge('g', 'a')  # adds a cycle to the dag object
    with pytest.raises(ValueError):
        pp.path_extraction.paths_from_dag(dag_object)


def test_route_from_node(dag_object: pp.DAG):
    root = 'a'
    routes = list(sorted(dag_object.routes_from_node(root)))
    expected = [['a', 'b', 'e'], ['a', 'b', 'f', 'g'], ['a', 'c', 'b', 'e'],
                ['a', 'c', 'b', 'f', 'g'], ['a', 'c', 'g']]
    assert routes == expected


def test_route_to_node(dag_object: pp.DAG):
    route_i = dag_object.routes_to_node('i')
    assert route_i == [['h', 'i']]

    route_e = dag_object.routes_to_node('e')
    assert route_e == [['a', 'b', 'e'], ['a', 'c', 'b', 'e']]

    route_g = list(sorted(dag_object.routes_to_node('g')))
    assert len(route_g) == 3
    expected = list(sorted([['a', 'c', 'g'], ['a', 'b', 'f', 'g'], ['a', 'c', 'b', 'f', 'g']]))
    for rout, e_route in zip(route_g, expected):
        assert rout == e_route


def test_dag_path_mapping(dag_object):
    dag = dag_object
    dag.topsort()

    mapping = {'a': 'A', 'b': 'B', 'c': 'A', 'e': 'B', 'f': 'B', 'g': 'A', 'h': 'A',
               'i': 'B', 'j': 'A'}
    paths_mapped2 = pp.path_extraction.paths_from_dag(dag, node_mapping=mapping)
    assert paths_mapped2.paths[1][('A', 'B')][1] == 1
    assert paths_mapped2.paths[1][('A', 'A')][1] == 1
    assert paths_mapped2.paths[2][('A', 'B', 'B')][1] == 1
    assert paths_mapped2.paths[2][('A', 'A', 'A')][1] == 1
    assert paths_mapped2.paths[3][('A', 'B', 'B', 'A')][1] == 1
    assert paths_mapped2.paths[3][('A', 'A', 'B', 'B')][1] == 1
    assert paths_mapped2.paths[4][('A', 'A', 'B', 'B', 'A')][1] == 1
    assert paths_mapped2.observation_count == 7


def test_dag_path_mapping_to_many(dag_object):
    dag = dag_object
    dag.topsort()

    mapping = {
        'a': {1, 2}, 'b': {2, 5},
        'c': {5}, 'e': {1, 2},
        'f': {2, 3}, 'g': {2, 5},
        'h': {1}, 'i': {1, 5},
        'j': {4}
    }
    paths_mapped2 = pp.path_extraction.paths_from_dag(dag, node_mapping=mapping)

    assert paths_mapped2.observation_count == 55
    assert set(paths_mapped2.nodes) == {'1', '2', '3', '4', '5'}


edges1, types1 = [(1, 2), (1, 3), (2, 3)], ({1}, {2}, {3})
edges2 = [(1, 2), (1, 3), (2, 3), (3, 7), (4, 2), (4, 5), (4, 6), (5, 7), (6, 5)]
types2 = ({1, 4}, {2, 6, 3, 5}, {7})
edges3 = 2*[(1, 2), (1, 3), (2, 3), (3, 3)]  # self loop + redundant
types3 = ({1}, {2}, {3})


@pytest.mark.parametrize('edges, types', (
        (edges1, types1),
        (edges2, types2),
        (edges3, types3)
))
def test_add_edges(edges, types):
    roots, neither, leafs = types
    from pathpy import DAG

    D = DAG(edges=edges)
    assert D.leafs == leafs
    assert D.roots == roots
    assert neither not in leafs
    assert neither not in roots


def test_dag_io(dag_object, tmpdir):
    file_path = str(tmpdir.mkdir("sub").join("test.edges"))
    dag_orig = dag_object
    dag_orig.make_acyclic()

    dag_orig.write_file(file_path)
    dag_back = pp.DAG.read_file(file_path)
    assert set(dag_back.edges) == set(dag_orig.edges)

    # filter_nodes out nodes not in the mapping
    mapping = {'a': 'A', 'b': 'B', 'c': 'A'}
    dag_back_map = pp.DAG.read_file(file_path, mapping=mapping)
    assert set(dag_back_map.nodes.keys()) == {'a', 'b', 'c'}


def test_remove_edge(dag_object: pp.DAG):
    dag_object.remove_edge('b', 'f')
    dag_object.remove_edge('b', 'e')
    assert 'f' in dag_object.roots
    assert 'e' in dag_object.roots
    assert 'e' in dag_object.leafs
    assert 'e' in dag_object.isolate_nodes()


def test_dag_from_temporal_network_basic():
    tn = pp.TemporalNetwork()
    tn.add_edge('a', 'b', 1)
    tn.add_edge('b', 'c', 2)
    tn.add_edge('a', 'c', 2)

    dag, mapping = pp.DAG.from_temporal_network(tn, delta=1)
    assert dag.routes_to_node('c_3') == [['a_2', 'c_3'], ['a_1', 'b_2', 'c_3']]


def test_dag_from_temporal_network():
    """
    The patterns is:
    1. o x x
        \
    2. x o o
          /
    3. x o x
          \
    4. x o o
        / /
    5. o o x
    """
    tn = pp.TemporalNetwork()
    tn.add_edge('a', 'b', 1)
    tn.add_edge('c', 'b', 2)
    tn.add_edge('b', 'c', 3)
    tn.add_edge('b', 'c', 3)
    tn.add_edge('c', 'b', 4)
    tn.add_edge('b', 'a', 4)

    dag, mapping = pp.DAG.from_temporal_network(tn, delta=1)
    assert dag.routes_to_node('c_4') == [['c_2', 'b_3', 'c_4']]
    assert dag.routes_to_node('a_5') == [['b_4', 'a_5']]

    dag, mapping = pp.DAG.from_temporal_network(tn, delta=2)
    assert dag.routes_to_node('c_4') == [['c_2', 'b_3', 'c_4'], ['a_1', 'b_2', 'b_3', 'c_4']]
    assert dag.routes_to_node('a_5') == [
        ['c_2', 'b_3', 'b_4', 'a_5'],
        ['a_1', 'b_2', 'b_3', 'b_4', 'a_5']  # TODO: why is b_3 listed here? it is not on the path
    ]

    # network as before but with the node at 3 moved from b to a
    tn = pp.TemporalNetwork()
    tn.add_edge('a', 'b', 1)
    tn.add_edge('c', 'a', 2)
    tn.add_edge('a', 'c', 3)
    tn.add_edge('c', 'b', 4)
    tn.add_edge('b', 'a', 4)
    dag, mapping = pp.DAG.from_temporal_network(tn, delta=3)
    assert dag.routes_to_node('c_4') == [['c_2', 'a_3', 'c_4']]
    assert dag.routes_to_node('a_5') == [['a_1', 'b_2', 'b_4', 'a_5']]


@pytest.mark.networkx
def test_strong_connected_components(random_network):
    from pathpy.classes.network import network_to_networkx
    from networkx import strongly_connected_components
    from pathpy.algorithms.components import reduce_to_gcc

    wrong_gcc = 0
    for i in range(200):
        hn = random_network(n=10, m=30, directed=True, seed=i)

        hn_nx = network_to_networkx(hn)
        size_largest_nx = len(max(strongly_connected_components(hn_nx), key=len))

        reduce_to_gcc(hn)  # need to run after because reduce_to_gss changes in place

        size_largest = len(hn.nodes)

        if size_largest_nx != size_largest:
            # print(f'seed {i} | nx: {size_largest_nx}, pp: {size_largest}')
            wrong_gcc += 1

    assert wrong_gcc < 1, f'wrong results {wrong_gcc/i:0.1%}'


@pytest.mark.networkx
def test_strong_connected_tmp(random_temp_network):
    from pathpy.path_extraction.temporal_paths import paths_from_temporal_network_dag
    from pathpy.algorithms.components import reduce_to_gcc
    from pathpy.classes.network import network_to_networkx
    from networkx import strongly_connected_components
    from pathpy.utils.log import Log, Severity
    Log.set_min_severity(Severity.WARNING)
    comp_sizes_nx = []
    comp_sizes_pp = []
    for delta in range(1, 900, 50):
        print(delta)
        tn = random_temp_network(n=10, m=100, min_t=0, max_t=800, seed=90)  # type: pp.TemporalNetwork
        obs_times = np.array([t[-1] for t in tn.tedges])
        obs_times.sort()

        p = paths_from_temporal_network_dag(tn, delta=delta)
        hn = pp.HigherOrderNetwork(p, k=2)
        total_size = len(hn.nodes)

        # using NetworkX
        nx_network = network_to_networkx(hn)
        giant_size_nx = len(max(strongly_connected_components(nx_network), key=len))
        comp_sizes_nx.append([delta, giant_size_nx, total_size, giant_size_nx/total_size])

        # using pathpy
        reduce_to_gcc(hn)
        giant_size_pp = len(hn.nodes)
        comp_sizes_pp.append([delta, giant_size_pp, total_size, giant_size_pp / total_size])
        # if abs(giant_size_nx - giant_size_pp) > 8:
        #     pass

    # import pandas as pd
    # df_nx = pd.DataFrame(comp_sizes_nx, columns=['delta', 'gcc_size', 'n_nodes', 'giant_prop'])
    # df_pp = pd.DataFrame(comp_sizes_pp, columns=['delta', 'gcc_size', 'n_nodes', 'giant_prop'])
    # print(df_nx)
    # print(df_pp)
    # print(df_nx - df_pp)
    assert comp_sizes_nx == 13








