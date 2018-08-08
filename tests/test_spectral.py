import pathpy as pp
import pytest

# absolute eigenvalue difference tolerance
EIGEN_ABS_TOL = 1e-2


@pytest.mark.parametrize('k, sub, e_gap', (
        (2, False, 1e-9),
        (1, False, 1e-5),
        (2, True, 1),
))
def test_eigen_value_gap(random_paths, k, sub, e_gap):
    import numpy as np
    p = random_paths(200, 0, 40)
    hon = pp.HigherOrderNetwork(p, k=k)
    np.random.seed(0)
    eigen_gap = pp.algorithms.spectral.eigenvalue_gap(hon, include_sub_paths=sub, lanczos_vectors=90)
    assert eigen_gap


@pytest.mark.xfail
@pytest.mark.parametrize('k, norm, e_sum, e_var', (
        (3, True, 1, 0.0036494914419765924),
        (2, False, 2765.72998141474, 8.661474971012986),
        (1, True, 1, 0.04948386659908706),
))
def test_fiedler_vector_sparse(random_paths, k, norm, e_sum, e_var):
    import numpy as np
    p = random_paths(90, 0, 20)
    hon = pp.HigherOrderNetwork(p, k=k)
    fv = pp.algorithms.spectral.fiedler_vector_sparse(hon, normalized=norm)
    assert fv.var() == pytest.approx(e_var, abs=EIGEN_ABS_TOL)
    assert np.sum(fv) == pytest.approx(e_sum, abs=EIGEN_ABS_TOL)


@pytest.mark.xfail
@pytest.mark.parametrize('k, e_sum, e_var', (
        (3, 1, 0.003649586067168485),
        (2, (1.0000000000000002+0j), 0.0031136096467386416),
        (1, (-0.0009514819500764382+0.1190367717310192j), 0.049999999999999996),
))
def test_fiedler_vector_dense(random_paths, k, e_sum, e_var):
    import numpy as np
    p = random_paths(90, 0, 20)
    hon = pp.HigherOrderNetwork(p, k=k)
    fv = pp.algorithms.spectral.fiedler_vector_dense(hon)
    assert fv.var() == pytest.approx(e_var, abs=EIGEN_ABS_TOL)
    assert np.sum(fv) == pytest.approx(e_sum, abs=EIGEN_ABS_TOL)


@pytest.mark.xfail
@pytest.mark.parametrize('k, e_sum', (
        (3, 0.9967398214809227),
        (2, 0.24345712528855065),
        (1, 0.7143571081268268),
))
def test_algebraic_connectivity(random_paths, k, e_sum):
    import pathpy
    p = random_paths(120, 0, 40)
    hon = pp.HigherOrderNetwork(p, k=k)
    ac = pp.algorithms.spectral.algebraic_connectivity(hon, lanczos_vectors=60, maxiter=40)
    assert ac == pytest.approx(e_sum, rel=1e-7)
