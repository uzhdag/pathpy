import pathpy as pp
import pytest

from pathpy import centralities

@pytest.mark.parametrize('k, e_sum, e_var', (
        (3, 27.5833333, 0.0085720486),
        (2, 55.0, 0.046875),
        (1, 55, 0.046875),
))
def test_closeness_centrality_hon(random_paths, k, e_sum, e_var):
    import numpy as np
    p = random_paths(50, 0, 8)
    hon = pp.HigherOrderNetwork(p, k=k)
    closeness = centralities.closeness_centrality(hon)
    np_closeness = np.array(list(closeness.values()))
    assert np_closeness.sum() == pytest.approx(e_sum)
    assert np_closeness.var() == pytest.approx(e_var)


@pytest.mark.parametrize('k, norm, e_sum, e_var, e_max', (
        (2, False, 213.404507128, 43.674855957, 37.5657124911),
        (1, False, 2, 0.00694444, 0.333333333),
        (2, True, 5.68083215, 0.030949114, 1),
))
def test_betweenness_centrality_hon(random_paths, norm, k, e_sum, e_var, e_max):
    import numpy as np
    p = random_paths(50, 0, 8)
    hon = pp.HigherOrderNetwork(p, k=k)
    betweenness = centralities.betweenness_centrality(hon, normalized=norm)
    values = np.array(list(betweenness.values()))
    assert values.sum() == pytest.approx(e_sum)
    assert max(values) == pytest.approx(e_max)
    assert values.var() == pytest.approx(e_var)


@pytest.mark.parametrize('k, sub, projection, e_sum, e_var', (
        (2, False, 'all', 2.030946758666, 0.0168478112),
        (1, False, 'scaled', 2.82310329017, 0.00047012207),
        (2, False, 'last', 1.7463870380802424, 0.0077742413305),
        (2, False, 'first', 1.7461339874793731, 0.0083696967427),
        (2, True, 'all', 2.030946758, 0.0168478112489),
        (1, True, 'scaled', 2.823103290, 0.0004701220779),
        (2, True, 'last', 1.746387038080242, 0.007774241),
        (2, True, 'first', 1.7461339874793727, 0.0083696967427313),
))
def test_eigen_centrality_hon(random_paths, sub, projection, k, e_sum, e_var):
    import numpy as np
    p = random_paths(50, 0, 8)
    hon = pp.HigherOrderNetwork(p, k=k)
    eigen = centralities.eigenvector_centrality(hon, projection, sub)
    values = np.array(list(eigen.values()))
    assert values.sum() == pytest.approx(e_sum)
    assert values.var() == pytest.approx(e_var)


@pytest.mark.parametrize('k, sub, proj, e_sum, e_var', (
        (2, False, 'all', 1, 0.000399240558236),
        (1, False, 'scaled', 1, 6.111199022e-05),
        (2, False, 'scaled', 1, 0.00039924055823),
        (2, False, 'last', 1, 0.00045826544),
        (2, False, 'first', 1, 0.000345796913),
        (2, True, 'all', 1, 0.000399240558),
        (1, True, 'scaled', 1, 6.111199022e-05),
        (2, True, 'scaled', 1, 0.000399240558236666),
        (2, True, 'last', 1, 0.000458265),
        (2, True, 'first', 1, 0.0003457969),
))
def test_pagerank_centrality_hon(random_paths, sub, proj, k, e_sum, e_var):
    import numpy as np
    p = random_paths(50, 0, 8)
    hon = pp.HigherOrderNetwork(p, k=k)
    page = centralities.pagerank(hon, include_sub_paths=sub, projection=proj)
    values = np.array(list(page.values()))
    assert values.sum() == pytest.approx(e_sum)
    assert values.var() == pytest.approx(e_var)


def test_betweenness_centrality_paths(path_from_ngram_file):
    p = path_from_ngram_file
    betweenness_centrality = centralities.betweenness_centrality(p, normalized=False)
    betweenness = {n: c for n, c in betweenness_centrality.items()}
    expected = {'b': 2.0, 'a': 3.0, 'e': 0, 'c': 3.0, 'd': 5.0}
    assert betweenness == expected


def test_betweenness_centrality_paths_norm(path_from_ngram_file):
    p = path_from_ngram_file
    betweenness_centrality = centralities.betweenness_centrality(p, normalized=True)
    betweenness = max(c for c in betweenness_centrality.values())
    expected_norm_max = 1
    assert pytest.approx(betweenness) == expected_norm_max


def test_closeness_centrality_paths(path_from_ngram_file):
    p = path_from_ngram_file
    closeness_centrality = centralities.closeness_centrality(p, normalized=False)
    closeness_sum = sum(c for c in closeness_centrality.values())
    expected_sum = 9.833333333333332
    assert closeness_sum == pytest.approx(expected_sum)

    nodes = {n for n in closeness_centrality}
    expected_nodes = {'a', 'b', 'c', 'd', 'e'}
    assert nodes == expected_nodes


def test_closeness_centrality_paths_norm(path_from_ngram_file):
    p = path_from_ngram_file
    closeness_centrality = centralities.closeness_centrality(p, normalized=True)
    closeness_max = max(c for c in closeness_centrality.values())
    expected_max = 1
    assert closeness_max == pytest.approx(expected_max)


def test_visitation_probabilities(path_from_ngram_file):
    p = path_from_ngram_file
    v_prob = centralities.visitation_probabilities(p)
    prob_sum = sum(p for p in v_prob.values())
    assert prob_sum == pytest.approx(1)

    max_prob = max(p for p in v_prob.values())
    expected_max = 0.3125
    assert max_prob == pytest.approx(expected_max)

