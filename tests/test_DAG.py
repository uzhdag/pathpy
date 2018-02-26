import pathpy as pp
import pytest


@pytest.mark.parametrize('edge_list', (
        [(1, 2), (2, 3), (5, 2)],
        list(zip(range(9), range(2, 11))) + list(zip(range(10), range(2, 12))),  # redund
        list(zip(range(8), range(2, 10))) + list(zip(range(8), range(8)))  # self-loops
))
def test_dag_init(edge_list):
    dag = pp.DAG(edges=edge_list)
    print(dag)


def test_dag_acyclic(dag_object):
    dag = dag_object
    dag.topsort()
    assert dag.isAcyclic is True

    # Add cycle to the graph
    dag.addEdge('b', 'c')
    dag.topsort()
    assert (dag.edge_classes[('b', 'c')] == 'back' or
            dag.edge_classes[('c', 'b')] == 'back')
    assert dag.isAcyclic is False

    dag.makeAcyclic()
    assert ('b', 'c') not in dag.edges or ('c', 'b') not in dag.edges
    assert len(dag.edges) == 9


def test_dag_path_extraction(dag_object):
    dag = dag_object
    dag.topsort()

    # Extract paths between nodes in DAG
    paths = pp.PathExtraction.DAGPaths.extract(dag)
    assert paths.observation_count == 7


def test_dag_path_extraction_cyclic(dag_object: pp.DAG):
    dag_object.addEdge('g', 'a')  # adds a cycle to the dag object
    with pytest.raises(ValueError):
        pp.PathExtraction.DAGPaths.extract(dag_object)


def test_dag_path_mapping(dag_object):
    dag = dag_object
    dag.topsort()

    mapping = {'a': 'A', 'b': 'B', 'c': 'A', 'e': 'B', 'f': 'B', 'g': 'A', 'h': 'A',
               'i': 'B', 'j': 'A'}
    paths_mapped2 = pp.PathExtraction.DAGPaths.extract(dag, node_mapping=mapping)
    assert paths_mapped2.paths[1][('A', 'B')][1] == 1
    assert paths_mapped2.paths[1][('A', 'A')][1] == 1
    assert paths_mapped2.paths[2][('A', 'B', 'B')][1] == 1
    assert paths_mapped2.paths[2][('A', 'A', 'A')][1] == 1
    assert paths_mapped2.paths[3][('A', 'B', 'B', 'A')][1] == 1
    assert paths_mapped2.paths[3][('A', 'A', 'B', 'B')][1] == 1
    assert paths_mapped2.paths[4][('A', 'A', 'B', 'B', 'A')][1] == 1
    assert paths_mapped2.observation_count == 7


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
    dag_orig.makeAcyclic()

    dag_orig.writeFile(file_path)
    dag_back = pp.DAG.readFile(file_path)
    assert set(dag_back.edges) == set(dag_orig.edges)

    # filter out nodes not in the mapping
    mapping = {'a': 'A', 'b': 'B', 'c': 'A'}
    dag_back_map = pp.DAG.readFile(file_path, mapping=mapping)
    assert dag_back_map.nodes == {'a', 'b', 'c'}
