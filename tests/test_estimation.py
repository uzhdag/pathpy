# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 11:59:22 2015
@author: Ingo Scholtes

(c) Copyright ETH Zurich, Chair of Systems Design, 2015-2017
"""

import pathpy as pp
import numpy as _np
import pytest

# mark to be used as decorator on slow functions such that they are only run
# when explicitly called with `$ pytest --runslow`
slow = pytest.mark.slow


def test_estimate_order_1():
    """Example without second-order correlations"""
    paths = pp.Paths()

    paths.add_path_ngram('a,c')
    paths.add_path_ngram('b,c')
    paths.add_path_ngram('c,d')
    paths.add_path_ngram('c,e')

    for k in range(4):
        paths.add_path_ngram('a,c,d')
        paths.add_path_ngram('b,c,e')
        paths.add_path_ngram('b,c,d')
        paths.add_path_ngram('a,c,e')

    m = pp.MultiOrderModel(paths, maxOrder=2)
    assert m.estimate_order(paths) == 1, \
        "Error, wrongly detected higher-order correlations"


def test_estimate_order_2():
    # Example with second-order correlations
    paths = pp.Paths()

    paths.add_path_ngram('a,c')
    paths.add_path_ngram('b,c')
    paths.add_path_ngram('c,d')
    paths.add_path_ngram('c,e')

    for k in range(4):
        paths.add_path_ngram('a,c,d')
        paths.add_path_ngram('b,c,e')

    m = pp.MultiOrderModel(paths, maxOrder=2)
    assert m.estimate_order(paths) == 2, \
        "Error, did not detect second-order correlations"

    g1 = pp.HigherOrderNetwork(paths, k=1)
    assert g1.vcount() == 5, \
        "Error, wrong number of nodes in first-order network"
    assert g1.ecount() == 4, \
        "Error, wrong number of links in first-order network"

    g2 = pp.HigherOrderNetwork(paths, k=2)
    assert g2.vcount() == 4, \
        "Error, wrong number of nodes in second-order network"
    assert g2.ecount() == 2, \
        "Error, wrong number of links in second-order network"

    g2.reduce_to_gcc()
    assert g2.vcount() == 1, \
        "Error, wrong number of nodes in giant connected component"
    assert g2.ecount() == 0, \
        "Error, wrong number of links in giant connected component"


@pytest.mark.parametrize('method', ('BIC', 'AIC'))
def test_markov_sequence(method):
    _np.random.seed(90)
    x = list(map(str, _np.random.choice(range(10), 1000)))
    ms = pp.MarkovSequence(x)
    detected_order = ms.estimate_order(maxOrder=4, method=method)
    assert detected_order == 1, \
        "Error, wrongly detected higher-order correlations"


def test_estimate_order_strongly_connected():
    """
    Example with single strongly connected component in first-
    and two connected components in second-order network
    """
    paths = pp.Paths()

    ngram_list = ['a,b,c', 'b,c,b', 'c,b,a',
                  'b,a,b', 'e,b,f', 'b,f,b',
                  'f,b,e', 'b,e,b']

    for ngram in ngram_list:
        paths.add_path_ngram(ngram)

    g1 = pp.HigherOrderNetwork(paths, k=1)
    g1.reduce_to_gcc()
    assert g1.vcount() == 5, "Error, wrong number of nodes in first-order network"
    assert g1.ecount() == 8, "Error, wrong number of links in first-order network"

    g2 = pp.HigherOrderNetwork(paths, k=2)
    g2.reduce_to_gcc()
    assert g2.vcount() == 4, "Error, wrong number of nodes in second-order network"
    assert g2.ecount() == 4, "Error, wrong number of links in second-order network"

    # test mapping of higher-order nodes and paths
    assert g2.higher_order_node_to_path('a-b') == ('a', 'b'), \
        "Error: mapping from higher-order node to first-order path failed"
    assert g2.higher_order_path_to_first_order(('a-b', 'b-c')) == ('a', 'b', 'c'), \
        "Error: mapping from higher-order path to first-order path failed"


def test_temp_net_extraction(temporal_network_object):
    t = temporal_network_object
    paths = pp.path_extraction.paths_from_temporal_network(t, delta=1)

    assert paths.observation_count == 10, \
        "Extracted wrong number of time-respecting paths"


def test_betweenness_preference_empty():
    t = pp.TemporalNetwork()
    paths = pp.path_extraction.paths_from_temporal_network(t, delta=3)
    assert len(paths.nodes()) == 0

    betweenness_pref = pp.path_measures.betweenness_preference(paths, 'e', method='MLE')
    expected = 0.0
    assert betweenness_pref == pytest.approx(expected)


def test_betweenness_preference_mle(temporal_network_object):
    t = temporal_network_object

    # Extract (time-respecting) paths
    p = pp.path_extraction.paths_from_temporal_network(t, delta=1)
    betweenness_pref = pp.path_measures.betweenness_preference(p, 'e', method='MLE')
    expected = 1.2954618442383219
    assert betweenness_pref == pytest.approx(expected)


def test_betweenness_preference_miller(temporal_network_object):
    t = temporal_network_object
    p = pp.path_extraction.paths_from_temporal_network(t, delta=1)

    betweenness_pref = pp.path_measures.betweenness_preference(p, 'e', method='Miller')
    expected = 0.99546184423832196
    assert betweenness_pref == pytest.approx(expected)


def test_betweenness_preference_normalized(temporal_network_object):
    t = temporal_network_object
    p = pp.path_extraction.paths_from_temporal_network(t, delta=1)
    # test normalize
    betweenness_pref = pp.path_measures.betweenness_preference(p, 'e', normalized=True)
    expected_norm = 1
    assert betweenness_pref == pytest.approx(expected_norm)


def test_slow_down_factor_random(random_paths):
    paths = random_paths(90, 90)
    slow_down_factor = pp.path_measures.slow_down_factor(paths)
    expected = 4.05
    assert slow_down_factor == pytest.approx(expected, rel=1e-2), \
        "Got slowdown factor %f but expected %f +- 1e-2" % (slow_down_factor, expected)


def test_get_distance_matrix_temporal(temporal_network_object):
    p = pp.path_extraction.paths_from_temporal_network(temporal_network_object)
    shortest_paths_dict = p.distance_matrix()

    path_distances = dict()
    for k in shortest_paths_dict:
        for p in shortest_paths_dict[k]:
            path_distances[(k, p)] = shortest_paths_dict[k][p]

    expected_distances = {
        ('c', 'e'): 1,
        ('c', 'f'): 2,
        ('c', 'c'): 0,
        ('b', 'g'): 2,
        ('f', 'e'): 1,
        ('c', 'b'): 4,
        ('a', 'a'): 0,
        ('a', 'g'): 2,
        ('g', 'g'): 0,
        ('e', 'g'): 1,
        ('e', 'e'): 0,
        ('b', 'b'): 0,
        ('e', 'b'): 1,
        ('e', 'f'): 1,
        ('f', 'b'): 2,
        ('a', 'e'): 1,
        ('f', 'f'): 0,
        ('b', 'e'): 1
    }
    assert path_distances == expected_distances


def test_get_distance_matrix_empty():
    p = pp.Paths()
    shortest_paths_dict = p.distance_matrix()
    assert len(shortest_paths_dict) == 0


def test_betweenness_centrality(path_from_ngram_file):
    p = path_from_ngram_file
    betweenness_centrality = pp.path_measures.betweenness_centrality(p, normalized=False)
    betweenness = {n: c for n, c in betweenness_centrality.items()}
    expected = {'b': 2.0, 'a': 3.0, 'e': 0, 'c': 3.0, 'd': 5.0}
    assert betweenness == expected


def test_betweenness_centrality_norm(path_from_ngram_file):
    p = path_from_ngram_file
    betweenness_centrality = pp.path_measures.betweenness_centrality(p, normalized=True)
    betweenness = max(c for c in betweenness_centrality.values())
    expected_norm_max = 1
    assert pytest.approx(betweenness) == expected_norm_max


def test_closeness_centrality(path_from_ngram_file):
    p = path_from_ngram_file
    closeness_centrality = pp.path_measures.closeness_centrality(p, normalized=False)
    closeness_sum = sum(c for c in closeness_centrality.values())
    expected_sum = 9.833333333333332
    assert closeness_sum == pytest.approx(expected_sum)

    nodes = {n for n in closeness_centrality}
    expected_nodes = {'a', 'b', 'c', 'd', 'e'}
    assert nodes == expected_nodes


def test_closeness_centrality_norm(path_from_ngram_file):
    p = path_from_ngram_file
    closeness_centrality = pp.path_measures.closeness_centrality(p, normalized=True)
    closeness_max = max(c for c in closeness_centrality.values())
    expected_max = 1
    assert closeness_max == pytest.approx(expected_max)


def test_visitation_probabilities(path_from_ngram_file):
    p = path_from_ngram_file
    v_prob = pp.path_measures.visitation_probabilities(p)
    prob_sum = sum(p for p in v_prob.values())
    assert prob_sum == pytest.approx(1)

    max_prob = max(p for p in v_prob.values())
    expected_max = 0.3125
    assert max_prob == pytest.approx(expected_max)


@slow
def test_entropy_growth_rate_ratio_mle(random_paths):
    p = random_paths(100, 500)
    mle_ratio = pp.path_measures.entropy_growth_rate_ratio(p, method="MLE")
    mle_expected = 0.10515408343772015
    assert mle_ratio == pytest.approx(mle_expected)


@slow
def test_entropy_growth_rate_ratio_miller(random_paths):
    p = random_paths(100, 500)
    miller_ratio = pp.path_measures.entropy_growth_rate_ratio(p, method="Miller")
    miller_expected = 0.88685603746914599
    assert miller_ratio == pytest.approx(miller_expected)
