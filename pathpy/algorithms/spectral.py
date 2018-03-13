# -*- coding: utf-8 -*-

#    pathpy is an OpenSource python package for the analysis of time series data
#    on networks using higher- and multi order graphical models.
#
#    Copyright (C) 2016-2017 Ingo Scholtes, ETH Zürich
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#    Contact the developer:

#    E-mail: ischoltes@ethz.ch
#    Web:    http://www.ingoscholtes.net
"""
This class can be used to calculate path statistics based on
origin-destination data available for a known network topology.
The path statistics generated from such data will be based on
the assumption that each observed path from an origin to a destination
node follows a shortest path in the network topology.
"""
from collections import defaultdict
import operator

import numpy as _np
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import scipy.linalg as la
import scipy as sp

from pathpy .utils import Log, Severity
from pathpy import HigherOrderNetwork
from pathpy.algorithms.shortest_paths import *

from pathpy.utils import PathpyNotImplemented


__all__ = ['eigenvalue_gap', 'fiedler_vector_sparse', 'fiedler_vector_dense', 'algebraic_connectivity']

def eigenvalue_gap(network, include_sub_paths=True, lanczos_vectors=15, maxiter=20):
    """Returns the eigenvalue gap of the transition matrix.

    Parameters
    ----------
    network
    include_sub_paths: bool
        whether or not to include subpath statistics in the calculation of transition
        probabilities.
    lanczos_vectors: int
        number of Lanczos vectors to be used in the approximate
        calculation of eigenvectors and eigenvalues. This maps to the ncv parameter
        of scipy's underlying function eigs.
    maxiter: int
        scaling factor for the number of iterations to be used in the
        approximate calculation of eigenvectors and eigenvalues. The number of iterations
        passed to scipy's underlying eigs function will be n*maxiter where n is the
        number of rows/columns of the Laplacian matrix.

    Returns
    -------
    float
    """
    assert isinstance(network, HigherOrderNetwork), \
        "network must be an instance of HigherOrderNetwork"
    # NOTE to myself: most of the time goes for construction of the 2nd order
    # NOTE            null graph, then for the 2nd order null transition matrix

    Log.add('Calculating eigenvalue gap ... ', Severity.INFO)

    # Build transition matrices
    trans_mat = network.transition_matrix(include_sub_paths)

    # Compute the two largest eigenvalues
    # NOTE: ncv sets additional auxiliary eigenvectors that are computed
    # NOTE: in order to be more confident to actually find the one with the largest
    # NOTE: magnitude, see https://github.com/scipy/scipy/issues/4987
    eig_vals = sla.eigs(trans_mat, which="LM", k=2, ncv=lanczos_vectors,
                        return_eigenvectors=False, maxiter=maxiter)
    eigen_values2_sorted = _np.sort(-_np.absolute(eig_vals))

    Log.add('finished.', Severity.INFO)

    return _np.abs(eigen_values2_sorted[1])


def fiedler_vector_sparse(network, normalized=True, lanczos_vectors=15, maxiter=20):
    """Returns the (sparse) Fiedler vector of the higher-order network. The Fiedler
    vector can be used for a spectral bisection of the network.

    Note that sparse linear algebra for eigenvalue problems with small eigenvalues
    is problematic in terms of numerical stability. Consider using the dense version
    of this method in this case. Note also that the sparse Fiedler vector might be scaled
    by a factor (-1) compared to the dense version.

    Parameters
    ----------
    network
    normalized: bool
        whether (default) or not to normalize the fiedler vector.
        Normalization is done such that the sum of squares equals one in order to
        get reasonable values as entries might be positive and negative.
    lanczos_vectors: int
        number of Lanczos vectors to be used in the approximate
        calculation of eigenvectors and eigenvalues. This maps to the ncv parameter
        of scipy's underlying function eigs.
    maxiter: int
        scaling factor for the number of iterations to be used in the
        approximate calculation of eigenvectors and eigenvalues. The number of iterations
        passed to scipy's underlying eigs function will be n*maxiter where n is the
        number of rows/columns of the Laplacian matrix.

    Returns
    -------

    """
    assert isinstance(network, HigherOrderNetwork), \
        "network must be an instance of HigherOrderNetwork"
    # NOTE: The transposed matrix is needed to get the "left" eigenvectors
    lapl_mat = network.laplacian_matrix()

    # NOTE: ncv sets additional auxiliary eigenvectors that are computed
    # NOTE: in order to be more confident to find the one with the largest
    # NOTE: magnitude, see https://github.com/scipy/scipy/issues/4987
    maxiter = maxiter * lapl_mat.get_shape()[0]
    w = sla.eigs(lapl_mat, k=2, which="SM", ncv=lanczos_vectors,
                 return_eigenvectors=False, maxiter=maxiter)

    # compute a sparse LU decomposition and solve for the eigenvector
    # corresponding to the second largest eigenvalue
    lapl_n = lapl_mat.get_shape()[0]
    fiedler_v = _np.ones(lapl_n)
    eigen_value = _np.sort(_np.abs(w))[1]
    mat = (lapl_mat[1:lapl_n, :].tocsc()[:, 1:lapl_n] -
           sparse.identity(lapl_n - 1).multiply(eigen_value))
    fiedler_v[1:lapl_n] = mat[0, :].toarray()

    lu_decom = sla.splu(mat)
    fiedler_v[1:lapl_n] = lu_decom.solve(fiedler_v[1:lapl_n])

    if normalized:
        fiedler_v /= _np.sqrt(_np.inner(fiedler_v, fiedler_v))
    return fiedler_v


def fiedler_vector_dense(network):
    """Returns the (dense)Fiedler vector of the higher-order network.
    The Fiedler vector can be used for a spectral bisection of the network.

    Parameters
    ----------
    network: HigherOrderNetwork

    Returns
    -------

    """
    assert isinstance(network, HigherOrderNetwork), \
        "network must be an instance of HigherOrderNetwork"
    # NOTE: The Laplacian is transposed for the sparse case to get the left
    # NOTE: eigenvalue.
    lapl_mat = network.laplacian_matrix()
    # convert to dense matrix and transpose again to have the un-transposed
    # laplacian again.
    laplacian_transposed = lapl_mat.todense().transpose()
    w, v = la.eig(laplacian_transposed, right=False, left=True)

    return v[:, _np.argsort(_np.absolute(w))][:, 1]


def algebraic_connectivity(network, lanczos_vectors=15, maxiter=20):
    """

    Parameters
    ----------
    network: HigherOrderNetwork
    lanczos_vectors: int
        number of Lanczos vectors to be used in the approximate calculation of
        eigenvectors and eigenvalues. This maps to the ncv parameter of scipy's underlying
        function eigs.
    maxiter: int
        scaling factor for the number of iterations to be used in the approximate
        calculation of eigenvectors and eigenvalues. The number of iterations passed to
        scipy's underlying eigs function will be n*maxiter where n is the number of
        rows/columns of the Laplacian matrix.

    Returns
    -------

    """
    assert isinstance(network, HigherOrderNetwork), \
        "network must be an instance of HigherOrderNetwork"
    Log.add('Calculating algebraic connectivity ... ', Severity.INFO)

    lapl_mat = network.laplacian_matrix()
    # NOTE: ncv sets additional auxiliary eigenvectors that are computed
    # NOTE: in order to be more confident to find the one with the largest
    # NOTE: magnitude, see https://github.com/scipy/scipy/issues/4987
    w = sla.eigs(lapl_mat, which="SM", k=2, ncv=lanczos_vectors,
                 return_eigenvectors=False, maxiter=maxiter)
    eigen_values_sorted = _np.sort(_np.absolute(w))

    Log.add('finished.', Severity.INFO)

    # TODO: result is unstable, it looks like it depends on a "warm start"
    # (i.e. run after other eigen velue calculations) see test_algebraic_connectivity
    # problems with order k=3

    return _np.abs(eigen_values_sorted[1])
