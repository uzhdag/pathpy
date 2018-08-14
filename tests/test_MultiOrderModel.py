import pytest
import numpy as np

import pathpy as pp


def test_print(random_paths):
    p = random_paths(90, 0, 20)
    multi = pp.MultiOrderModel(p, max_order=3)
    print(multi)


@pytest.mark.parametrize('k', (1, 2, 3))
def test_init(random_paths, k):
    p = random_paths(90, 0, 20)
    multi = pp.MultiOrderModel(p, max_order=k)
    assert len(multi.layers) == k+1


# @pytest.mark.slow
# @pytest.mark.parametrize('k', (1, 2))
# def test_parallel(random_paths, k):
#     """assert that the parallel calculation is equal to the
#     sequential"""
#     p = random_paths(90, 0, 20)
#     multi_seq = pp.MultiOrderModel(p, max_order=k)
#
#     pp.ENABLE_MULTICORE_SUPPORT = True
#     assert pp.ENABLE_MULTICORE_SUPPORT
#
#     multi_parallel = pp.MultiOrderModel(p, max_order=k)
#
#     assert multi_parallel.model_size(k) == multi_seq.model_size(k)
#     for k in multi_parallel.transition_matrices:
#         assert np.sum(multi_parallel.transition_matrices[k] - multi_seq.transition_matrices[k]) == pytest.approx(0)


# TODO: how to properly test this function?
@pytest.mark.parametrize('method', ('AIC', 'BIC', 'AICc'))
@pytest.mark.parametrize('k', (2, 3))
def test_test_network_hypothesis(random_paths, k, method):
    p = random_paths(20, 40, 6)
    multi = pp.MultiOrderModel(p, max_order=k)
    (is_net, ic0, ic1) = multi.test_network_hypothesis(p, method=method)


@pytest.mark.parametrize(
    'method, k, e_ic0, e_ic1', (
            ('AIC', 1, 853.7904463041854, 829.9533867847043),
            ('BIC', 3, 862.234843574755, 885.6864087704643),
            ('AICc', 3, 856.3359008496399, 1305.9533867847044)
    )
)
def test_test_network_hypothesis_values(random_paths, k, method, e_ic0, e_ic1):
    p = random_paths(20, 40, 6)
    multi = pp.MultiOrderModel(p, max_order=k)
    (is_net, ic0, ic1) = multi.test_network_hypothesis(p, method=method)
    assert e_ic0 == pytest.approx(ic0)
    assert e_ic1 == pytest.approx(ic1)


@pytest.mark.parametrize('k', (1, 2, 3))
def test_write_state_file(random_paths, k, tmpdir):
    file_path = str(tmpdir.mkdir("sub").join("multi_order_state"))
    p = random_paths(20, 40, 6)
    multi = pp.MultiOrderModel(p, max_order=k)

    for i in range(1, k+1):
        multi.save_state_file(file_path + '.' + str(i), layer=i)


def test_estimate_order_1():
    """Example without second-order correlations"""
    paths = pp.Paths()

    paths.add_path('a,c')
    paths.add_path('b,c')
    paths.add_path('c,d')
    paths.add_path('c,e')

    for k in range(4):
        paths.add_path('a,c,d')
        paths.add_path('b,c,e')
        paths.add_path('b,c,d')
        paths.add_path('a,c,e')

    m = pp.MultiOrderModel(paths, max_order=2)
    assert m.estimate_order() == 1, \
        "Error, wrongly detected higher-order correlations"


def test_estimate_order_2():
    # Example with second-order correlations
    paths = pp.Paths()

    paths.add_path('a,c')
    paths.add_path('b,c')
    paths.add_path('c,d')
    paths.add_path('c,e')

    for k in range(4):
        paths.add_path('a,c,d')
        paths.add_path('b,c,e')

    m = pp.MultiOrderModel(paths, max_order=2)
    assert m.estimate_order() == 2


def test_save_statefile(random_paths, tmpdir):
    file_path = str(tmpdir.join("statefile.sf"))
    p = random_paths(3, 20, 6)
    multi = pp.MultiOrderModel(p, max_order=2)
    multi.save_state_file(file_path, layer=2)
    with open(file_path) as f:
        for line in f:
            assert '{' not in line  # make sure that we did not write a dictionary


def test_single_path_likelihood(random_paths):
    p1 = random_paths(size=10, rnd_seed=20, num_nodes=10)  # type: pp.Paths
    p2 = random_paths(size=100, rnd_seed=0, num_nodes=50)
    p12 = p1 + p2
    mom = pp.MultiOrderModel(p12, max_order=3)
    lkh1 = mom.likelihood(p1)
    lkh2 = mom.likelihood(p2)
    lkh12 = mom.likelihood(p12)

    assert lkh1 > lkh2  # second paths must be
    assert (lkh1 + lkh2) == pytest.approx(lkh12)

    assert mom.path_likelihood(('1', '2'), layer=0, freq=4) < 0

    lkl_last = None
    for i in range(3):  # likelihoods must be increasing
        lkl = mom.path_likelihood(('6', '7', '2', '0', '6'), layer=i, freq=9)
        if lkl_last is not None:
            assert lkl >= lkl_last
            lkl_last = lkl

    path_likelihoods = []
    for p, freq in p12.paths[3].items():  # print the path with the highest likelihood
        lkl = mom.path_likelihood(p, layer=2, freq=freq.sum(), log=False)
        path_likelihoods.append((lkl, p))

    assert max(path_likelihoods)[1] == ('23', '32', '19', '8')
