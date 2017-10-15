import pathpy as pp
import pytest


@pytest.mark.parametrize('edge_list', (
        [(1, 2), (2, 3), (5, 2)],
        list(zip(range(9), range(2, 11))) + list(zip(range(10), range(2, 12))),  # redund
        list(zip(range(8), range(2, 10))) + list(zip(range(8), range(8)))  # self-loops
))
def test_dag_init(edge_list):
    pp.DAG(edges=edge_list)


def test_dag_acyclic(dag_object):
    dag = dag_object
    dag.topsort()
    assert dag.isAcyclic is True

    # Add cycle to the graph
    dag.addEdge('b', 'c')
    dag.topsort()
    assert dag.edge_classes[('b', 'c')] == 'back' or \
        dag.edge_classes[('c', 'b')] == 'back'
    assert dag.isAcyclic is False

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
