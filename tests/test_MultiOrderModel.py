import pytest
import numpy as np

import pathpy as pp


def test_print(random_paths):
    p = random_paths(90, 0, 20)
    multi = pp.MultiOrderModel(p, maxOrder=3)
    print(multi)


@pytest.mark.parametrize('k', (1, 2, 3))
def test_init(random_paths, k):
    p = random_paths(90, 0, 20)
    multi = pp.MultiOrderModel(p, maxOrder=k)
    assert len(multi.layers) == k+1


@pytest.mark.slow
@pytest.mark.parametrize('k', (1, 2))
def test_parallel(random_paths, k):
    """assert that the parallel calculation is equal to the
    sequential"""
    p = random_paths(90, 0, 20)
    multi_seq = pp.MultiOrderModel(p, maxOrder=k)

    pp.ENABLE_MULTICORE_SUPPORT = True
    assert pp.ENABLE_MULTICORE_SUPPORT

    multi_parallel = pp.MultiOrderModel(p, maxOrder=k)

    assert multi_parallel.model_size(k) == multi_seq.model_size(k)
    for k in multi_parallel.T:
        assert np.sum(multi_parallel.T[k] - multi_seq.T[k]) == pytest.approx(0)


# TODO: how to properly test this function?
@pytest.mark.parametrize('method', ('AIC', 'BIC', 'AICc'))
@pytest.mark.parametrize('k', (2, 3))
def test_test_network_hypothesis(random_paths, k, method):
    p = random_paths(20, 40, 6)
    multi = pp.MultiOrderModel(p, maxOrder=k)
    (is_net, ic0, ic1) = multi.test_network_hypothesis(p, method=method)


@pytest.mark.parametrize('k', (1, 2, 3))
def test_write_state_file(random_paths, k, tmpdir):
    file_path = str(tmpdir.mkdir("sub").join("multi_order_state"))
    p = random_paths(20, 40, 6)
    multi = pp.MultiOrderModel(p, maxOrder=k)

    for i in range(1, k+1):
        multi.save_state_file(file_path + '.' + str(i), layer=i)
