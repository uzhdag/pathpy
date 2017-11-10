# -*- coding: utf-8 -*-
"""
    pathpy is an OpenSource python package for the analysis of time series data
    on networks using higher- and multi order graphical models.

    Copyright (C) 2016-2017 Ingo Scholtes, ETH Zürich

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Contact the developer:

    E-mail: ischoltes@ethz.ch
    Web:    http://www.ingoscholtes.net
"""

import collections as _co
import sys as _sys

import numpy as _np
import scipy.sparse.linalg as _sla

import pathpy.Paths as _Paths
import pathpy.HigherOrderNetwork as _HigherOrderNetwork
from pathpy.Log import Log as _Log
from pathpy.Log import Severity as _Severity


class PathMeasures:
    """
    This class contains measures based on the Paths object
    """


    @staticmethod
    def BetweennessCentrality(paths, normalized=False):
        """
        Calculates the betweenness centrality of nodes based on
        observed shortest paths between all pairs of nodes
        """

        node_centralities = _co.defaultdict(lambda: 0)
        shortest_paths = paths.getShortestPaths()

        for s in shortest_paths:
            for d in shortest_paths[s]:
                for p in shortest_paths[s][d]:
                    for x in p[1:-1]:
                        if s != d != x:
                            # print('node ' + x + ': ' + str(1.0 / len(shortest_paths[start][end])))
                            node_centralities[x] += 1.0 / len(shortest_paths[s][d])
                            # node_centralities[x] += 1.0
        if normalized:
            m = max(node_centralities.values())
            for v in node_centralities:
                node_centralities[v] /= m

        # assign zero values to nodes not occurring on shortest paths
        nodes = paths.getNodes()
        for v in nodes:
            node_centralities[v] += 0

        return node_centralities


    @staticmethod
    def ClosenessCentrality(paths, normalized=False):
        """
        Calculates the closeness centrality of nodes based on
        observed shortest paths between all nodes
        """

        node_centralities = _co.defaultdict(lambda: 0)
        shortest_path_lengths = paths.getDistanceMatrix()

        for x in shortest_path_lengths:
            for d in shortest_path_lengths[x]:
                if x != d and shortest_path_lengths[x][d] < _np.inf:
                    node_centralities[x] += 1.0 / shortest_path_lengths[x][d]

        # assign zero values to nodes not occurring
        nodes = paths.getNodes()
        for v in nodes:
            node_centralities[v] += 0

        if normalized:
            m = max(node_centralities.values())
            for v in nodes:
                node_centralities[v] /= m

        return node_centralities


    @staticmethod
    def VisitationProbabilities(paths):
        """
        Calculates the probabilities that randomly chosen paths
        pass through nodes. If 5 out of 100 paths (of any length) contain
        node v, it will be assigned a value of 0.05. This measure can be
        interpreted as path-based ground truth for the notion of importance
        captured by PageRank applied to a graphical abstraction of the paths.
        """

        _Log.add('Calculating path visitation probabilities...', _Severity.INFO)

        # entries capture the probability that a given node is visited on an arbitrary path
        # Note: this is identical to the subpath count of zero-length paths
        # (i.e. the relative frequencies of nodes across all pathways)
        visitation_probabilities = _co.defaultdict(lambda: 0)

        # total number of visits
        visits = 0.0

        for l in paths.paths:
            for p in paths.paths[l]:
                for v in p:
                    # count occurrences in longest paths only!
                    visitation_probabilities[v] += paths.paths[l][p][1]
                    visits += paths.paths[l][p][1]

        for v in visitation_probabilities:
            visitation_probabilities[v] /= visits

        _Log.add('finished.', _Severity.INFO)

        return visitation_probabilities



    @staticmethod
    def getSlowDownFactor(paths, k=2, lanczosVecs=15, maxiter=1000):
        """
        Returns a factor S that indicates how much slower (S>1) or faster (S<1)
        a diffusion process evolves in a k-order model of the path statistics
        compared to what is expected based on a first-order model. This value captures
        the effect of order correlations of length k on a diffusion process which evolves
        based on the observed paths.
        """

        assert k > 1, 'Slow-down factor can only be calculated for orders larger than one'

        # NOTE to myself: most of the time goes for construction of the 2nd order
        # NOTE            null graph, then for the 2nd order null transition matrix

        gk = _HigherOrderNetwork(paths, k=k)
        gkn = _HigherOrderNetwork(paths, k=k, nullModel=True)

        _Log.add('Calculating slow down factor ... ', _Severity.INFO)

        # Build transition matrices
        Tk = gk.getTransitionMatrix()
        Tkn = gkn.getTransitionMatrix()

        # Compute eigenvector sequences
        # NOTE: ncv=13 sets additional auxiliary eigenvectors that are computed
        # NOTE: in order to be more confident to find the one with the largest
        # NOTE: magnitude, see
        # NOTE: https://github.com/scipy/scipy/issues/4987
        w2 = _sla.eigs(Tk, which="LM", k=2, ncv=lanczosVecs, return_eigenvectors=False, maxiter=maxiter)
        evals2_sorted = _np.sort(-_np.absolute(w2))

        w2n = _sla.eigs(Tkn, which="LM", k=2, ncv=lanczosVecs, return_eigenvectors=False, maxiter=maxiter)
        evals2n_sorted = _np.sort(-_np.absolute(w2n))

        _Log.add('finished.', _Severity.INFO)

        return _np.log(_np.abs(evals2n_sorted[1]))/_np.log(_np.abs(evals2_sorted[1]))


    @staticmethod
    def getEntropyGrowthRateRatio(paths, method='MLE', k=2, lanczosVecs=15, maxiter=1000):
        """
        Computes the ratio between the entropy growth rate ratio between
        the k-order and first-order model of a temporal network t. Ratios smaller
        than one indicate that the temporal network exhibits non-Markovian characteristics
        """

        # NOTE to myself: most of the time here goes into computation of the
        # NOTE            EV of the transition matrix for the bigger of the
        # NOTE            two graphs below (either 2nd-order or 2nd-order null)

        assert (method == 'MLE' or method == 'Miller'), 'Only methods MLE or Miller are supported'

        # Generate k-order network
        gk = _HigherOrderNetwork(paths, k=k)
        g1 = _HigherOrderNetwork(paths, k=1)

        _Log.add('Calculating entropy growth rate ratio ... ', _Severity.INFO)

        # Compute entropy growth rate of observed transition matrix
        A = g1.getAdjacencyMatrix(weighted=False, transposed=True)
        Tk = gk.getTransitionMatrix()
        Tk_pi = _HigherOrderNetwork.getLeadingEigenvector(Tk, normalized=True, lanczosVecs=lanczosVecs, maxiter=maxiter)

        Tk.data *= _np.log2(Tk.data)

        # Apply Miller correction to the entropy estimation
        if method == 'Miller':
            # Here, K is the number of different k-paths that can exist based on the
            # observed edges
            K = (A**k).sum()
            print('K = ', K)

            # N is the number of observations used to estimate the transition probabilities
            # in the second-order network. This corresponds to the total edge weight in the
            # k-order network, or - alternatively - to the number of paths of length k
            N = 0
            for p in paths.paths[k]:
                N += paths.paths[k][p].sum()
            print('N = ', N)
            Hk = _np.sum(Tk * Tk_pi) + (K-1)/(2*N)
        else:
            # simple MLE estimation
            Hk = -_np.sum(Tk * Tk_pi)

        Hk = _np.absolute(Hk)

        # Compute entropy rate of null model
        gk_n = _HigherOrderNetwork(paths, k=k, nullModel=True)

        # For the entropy rate of the null model, no Miller correction is needed
        # since we assume that transitions correspond to the true probabilities
        Tk_n = gk_n.getTransitionMatrix()
        Tk_n_pi = _HigherOrderNetwork.getLeadingEigenvector(Tk_n)
        Tk_n.data *= _np.log2(Tk_n.data)
        Hk_n = -_np.sum(Tk_n * Tk_n_pi)
        Hk_n = _np.absolute(Hk_n)

        _Log.add('finished.', _Severity.INFO)

        # Return ratio
        return Hk/Hk_n


    @staticmethod
    def BWPrefMatrix(paths, v):
        """Computes a betweenness preference matrix for a node v

        @param v: Node for which the betweenness preference matrix shall
            be calculated
        """
        # create first-order network
        g = _HigherOrderNetwork(paths)

        indeg = len(g.predecessors[v])
        outdeg = len(g.successors[v])

        index_succ = {}
        index_pred = {}

        B_v = _np.zeros(shape=(indeg, outdeg))

        # Create an index-to-node mapping for predecessors and successors
        i = 0
        for u in g.predecessors[v]:
            index_pred[u] = i
            i = i+1

        i = 0
        for w in g.successors[v]:
            index_succ[w] = i
            i = i+1

        # Calculate entries of betweenness preference matrix
        for p in paths.paths[2]:
            if p[1] == v:
                B_v[index_pred[p[0]], index_succ[p[2]]] += paths.paths[2][p].sum()

        return B_v


    @staticmethod
    def __Entropy(prob, K=None, N=None, method='MLE'):
        """
        Calculates the entropy of an (observed) probability ditribution
        based on Maximum Likelihood Estimation (MLE) (default) or using
        a Miller correction.

        @param prob: the observed probabilities
        @param K: the number of possible outcomes, i.e. outcomes with non-zero probability to be used
            for the Miller correction (default None)
        @param N: number of samples based on which observed probabilities where computed. This
            is needed for the Miller correaction (default None)
        @param method: The method to be used to calculate entropy. Can be 'MLE' (default) or 'Miller'
        """

        if method == 'MLE':
            idx = _np.nonzero(prob)
            return -_np.inner(_np.log2(prob[idx]), prob[idx])
        if method == 'Miller':
            assert K != None and N != None
            if N == 0:
                return 0

            idx = _np.nonzero(prob)
            return -_np.inner(_np.log2(prob[idx]), prob[idx]) + (K-1)/(2*N)


    @staticmethod
    def BetweennessPreference(paths, v, normalized=False, method='MLE'):
        """
        Calculates the betweenness preferences of a
        node v based on the mutual information of path
        statistics of length two.

        @nornalized: whether or not to normalize betweenness preference values

        @method: which method to use for the entropy calculation. The default 'MLE' uses
            the standard Maximum-Likelihood estimation of entropy. Setting method to
            'Miller' additionally applies a Miller-correction. see e.g.
            Liam Paninski: Estimation of Entropy and Mutual Information, Neural Computation 5, 2003 or
            http://www.nowozin.net/sebastian/blog/estimating-discrete-entropy-part-2.html
        """

        assert method == 'MLE' or method == 'Miller'

        # If the network is empty, just return zero
        if not paths.getNodes():
            return 0.0

        # First create the betweenness preference matrix (equation (2) of the paper)
        B_v = PathMeasures.BWPrefMatrix(paths, v)

        if B_v.shape[0] == 0 or B_v.shape[1] == 0:
            return None

        # Normalize matrix (equation (3) of the paper)
        # NOTE: P_v has the same shape as B_v
        P_v = _np.zeros(shape=B_v.shape)
        S = _np.sum(B_v)

        if S > 0:
            P_v = B_v / S

        # Compute marginal probabilities
        # Marginal probabilities P^v_d = \sum_s'{P_{s'd}}
        marginal_d = _np.sum(P_v, axis=0)

        # Marginal probabilities P^v_s = \sum_d'{P_{sd'}}
        marginal_s = _np.sum(P_v, axis=1)

        if method == 'Miller':

            # total number of samples, i.e. observed two-paths
            N = _np.sum(B_v)

            # print('N = ', N)
            # print('B = ', B_v)
            # print('marginal_s = ', marginal_s)
            # print('marginal_d = ', marginal_d)

            # marginal entropy H(S)
            H_s = PathMeasures.__Entropy(marginal_s, len(marginal_s), N, method='Miller')

            # print('H(S) = ', H_s)
            # marginal entropy H(D)

            H_d = PathMeasures.__Entropy(marginal_d, len(marginal_d), N, method='Miller')

            # print('H(D) = ', H_d)
            # we need the conditional entropy H(D|S)

            H_ds = 0
            for s in range(len(marginal_s)):

                # number of two paths s -> v -> * observed in the data
                N_s = _np.sum(B_v[s, :])

                # print('N(s=' + str(s) + ') = ' +  str(N_s))

                # probabilities of all destinations, given the particular source s
                p_ds = B_v[s, :]/_np.sum(B_v[s, :])

                # print('P(D|S=' + str(s) + ') = '+ str(p_ds))

                # number of possible destinations d
                K_s = len(p_ds)

                # print('K(s=' + str(s) + ') = ' +  str(K_s))

                # marginal_s[s] is the overall probability of source s
                p_s = marginal_s[s]

                # add to conditional entropy
                H_ds += p_s * PathMeasures.__Entropy(p_ds, K_s, N_s, method='Miller')

                I = H_d - H_ds

            # print('H(D|S) = ', H_ds)

        else:
            # use MLE estimation
            H_s = PathMeasures.__Entropy(marginal_s)
            H_d = PathMeasures.__Entropy(marginal_d)
            # H_ds = 0

            # for s in range(len(marginal_s)):
            #    print('s = ' + str(s) + ': ' + str(_np.sum(P_v[s,:])))
            #    p_ds = P_v[s,:]/_np.sum(P_v[s,:])
            #    H_ds += marginal_s[s] * Paths.__Entropy(p_ds)

            # Alternative calculation (without explicit entropies)
            # build mask for non-zero elements
            row, col = _np.nonzero(P_v)
            pv = P_v[(row, col)]
            marginal = _np.outer(marginal_s, marginal_d)
            log_argument = _np.divide(pv, marginal[(row, col)])
            I = _np.dot(pv, _np.log2(log_argument))

        # I = H_d - H_ds

        if normalized:
            I = I/_np.min([H_s, H_d])

        return I
