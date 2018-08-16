import random
import pytest

@pytest.mark.parametrize('directed', (True, False))
@pytest.mark.parametrize('weighted', (True, False))
def test_add_node(random_network, directed, weighted):
    """
    Test node creation
    """
    net = random_network(n=10, m=20, directed=directed, weighted=weighted)

    assert net.directed == directed

    vc_before = net.ncount()
    ec_before = net.ecount()

    assert 'v' not in net.nodes

    net.add_node('v', test1='x', test2=42)

    assert 'v' in net.nodes

    assert net.nodes['v']['test1'] == 'x'
    assert net.nodes['v']['test2'] == 42
    if directed:
        assert net.nodes['v']['indegree'] == 0
        assert net.nodes['v']['outdegree'] == 0
    else:
        assert net.nodes['v']['degree'] == 0
    assert net.nodes['v']['inweight'] == 0
    assert net.nodes['v']['outweight'] == 0

    assert net.ncount() == vc_before + 1
    assert net.ecount() == ec_before


@pytest.mark.parametrize('directed', (True, False))
@pytest.mark.parametrize('weighted', (True, False))
def test_remove_node(random_network, directed, weighted):
    """
    Test node removal
    """
    net = random_network(n=10, m=20, directed=directed, weighted=weighted)

    to_remove = random.choice(list(net.nodes))

    # collect values before removal
    v_c = net.ncount()
    e_c = net.ecount()
    t_w = net.total_edge_weight()

    incident_edges = [(v, w) for (v, w) in net.edges if v == to_remove or w == to_remove]
    weight_incident = sum([net.edges[e]['weight'] for e in incident_edges])
    successors = [w for w in net.successors[to_remove]]
    predecessors = [v for v in net.predecessors[to_remove]]

    net.remove_node(to_remove)

    # test values after removal
    assert to_remove not in net.nodes
    assert net.ncount() == v_c-1
    assert net.ecount() == e_c - len(incident_edges)
    assert net.total_edge_weight() == t_w - weight_incident

    for e in incident_edges:
        assert e not in net.edges

    for w in successors:
        assert to_remove not in net.predecessors[w]
        assert to_remove not in net.successors[w]
    for v in predecessors:
        assert to_remove not in net.predecessors[v]
        assert to_remove not in net.successors[v]



@pytest.mark.parametrize('directed', (True, False))
@pytest.mark.parametrize('weighted', (True, False))
def test_add_edge(random_network, directed, weighted):
    """
    Test edge creation
    """
    net = random_network(n=10, m=20, directed=directed, weighted=weighted)

    # draw pair of nodes that are not already connected
    (v, w) = random.choice(list(net.edges))
    while (v, w) in net.edges:
        v, w = random.sample(list(net.nodes), 2)

    if weighted:
        weight_to_add = random.randint(1, 10)
    else:
        weight_to_add = 1

    # collect values before removal
    v_c = net.ncount()
    e_c = net.ecount()
    t_w = net.total_edge_weight()

    if weighted:
        net.add_edge(v, w, weight=weight_to_add)
    else:
        net.add_edge(v, w)

    # test values after removal
    assert v in net.nodes
    assert w in net.nodes
    assert net.ncount() == v_c
    assert net.ecount() == e_c + 1
    assert net.total_edge_weight() == t_w + weight_to_add
    assert (v, w) in net.edges

    assert w in net.successors[v]
    assert v in net.predecessors[w]

    if not directed:
        assert w in net.predecessors[v]
        assert v in net.successors[w]


def test_import_from_networkx():
    # TODO: add test for weighted networks
    from pathpy.classes.network import network_from_networkx
    import networkx as nx

    g = nx.generators.barabasi_albert_graph(20, 10)
    relabeling = {i: str(i) for i in g}
    nx.relabel_nodes(g, relabeling, copy=False)
    for i, edge in enumerate(g.edges):
        g.edges[edge]['custom'] = i

    net = network_from_networkx(g)
    assert net.ncount() == len(g)
    assert net.ecount() == len(g.edges)
    for edge in net.edges:
        assert net.edges[edge]['custom'] == g.edges[edge]['custom']


def test_export_netwokx():
    # TODO: test directed graph
    from pathpy.classes.network import network_from_networkx
    from pathpy.classes.network import network_to_networkx
    import networkx as nx

    g = nx.generators.karate_club_graph()
    # pathpy.Network will implicitely recast all labels to str so to have
    # a comparable network to start with do the same here
    relabel = {i: str(i) for i in g}
    nx.relabel_nodes(g, relabel, copy=False)
    for i, edge in enumerate(g.edges):
        g.edges[edge]['custom'] = i
        g.edges[edge]['weight'] = (i % 4) + 100

    for i, node in enumerate(g.nodes):
        g.nodes[node]['custom'] = "{} unique string".format(i)

    net = network_from_networkx(g)
    g_back = network_to_networkx(net)

    nx_degrees = g.degree(weight='weight')

    assert len(g_back) == len(g)
    assert len(g_back.edges) == len(g.edges)
    assert dict(g_back.degree) == dict(g.degree)
    for edge in g_back.edges:
        assert net.edges[edge]['weight'] == g.edges[edge]['weight']
        assert net.edges[edge]['custom'] == g.edges[edge]['custom']
        assert g_back.edges[edge]['custom'] == g.edges[edge]['custom']
        assert g_back.edges[edge]['weight'] == g.edges[edge]['weight']

    for node in g_back.nodes:
        assert g_back.nodes[node]['custom'] == g.nodes[node]['custom']
        assert nx_degrees[node] == net.nodes[node]['inweight']
        assert nx_degrees[node] == net.nodes[node]['outweight']


def test_read_edges(test_data_directory):
    import os
    import pathpy

    edge_file = os.path.join(test_data_directory, "example_int.tedges")

    net = pathpy.Network.read_file(edge_file, weighted=True, header=True, directed=True)
    assert net.ncount() == 5
    assert net.ecount() == 6


def test_diagonal_values():
    from pathpy.classes.network import Network
    net = Network()
    net.add_edge('a', 'b')
    net.add_edge('a', 'a')
    adj = net.adjacency_matrix().todense()
    assert adj.sum() == 3
    assert adj[0, 0] == 1
    assert adj[1, 1] == 0
    assert adj.diagonal().sum() == 1

    # test directed case
    net = Network(directed=True)
    net.add_edge('a', 'b')
    net.add_edge('a', 'a')
    adj = net.adjacency_matrix().todense()
    assert adj.sum() == 2
    assert adj[0, 0] == 1
    assert adj[1, 1] == 0
    assert adj.diagonal().sum() == 1

