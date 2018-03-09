# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 09:32:22 2018
@author: Ingo Scholtes

(c) Copyright ETH Zurich, Chair of Systems Design, 2015-2018
"""
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

    vc_before = net.vcount()
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

    assert net.vcount() == vc_before + 1
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
    v_c = net.vcount()
    e_c = net.ecount()
    t_w = net.total_edge_weight()

    incident_edges = [(v, w) for (v, w) in net.edges if v == to_remove or w == to_remove]
    weight_incident = sum([net.edges[e]['weight'] for e in incident_edges])
    successors = [w for w in net.successors[to_remove]]
    predecessors = [v for v in net.predecessors[to_remove]]

    net.remove_node(to_remove)

    # test values after removal
    assert to_remove not in net.nodes
    assert net.vcount() == v_c-1
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
    v_c = net.vcount()
    e_c = net.ecount()
    t_w = net.total_edge_weight()

    if weighted:
        net.add_edge(v, w, weight=weight_to_add)
    else:
        net.add_edge(v, w)

    # test values after removal
    assert v in net.nodes
    assert w in net.nodes
    assert net.vcount() == v_c
    assert net.ecount() == e_c + 1
    assert net.total_edge_weight() == t_w + weight_to_add
    assert (v, w) in net.edges

    assert w in net.successors[v]
    assert v in net.predecessors[w]

    if not directed:
        assert w in net.predecessors[v]
        assert v in net.successors[w]
