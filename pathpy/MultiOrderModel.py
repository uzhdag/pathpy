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

import numpy as _np
from scipy.stats import chi2

import pathpy
from pathpy.HigherOrderNetwork import HigherOrderNetwork
from pathpy.Log import Log
from pathpy.Log import Severity
import pathpy as _pp


_np.seterr(all='warn')


class MultiOrderModel:
    """Instances of this class represent a hierarchy of
        higher-order networks which collectively represent
        a multi-order model for path statistics. """

    def __init__(self, paths, maxOrder=1):
        """
        Generates a hierarchy of higher-order
        models for the given path statistics,
        up to a given maximum order

        @param paths: the paths instance for which the model should be created
        @param maxOrder: the maximum order of the multi-order model
        """

        assert paths.maxSubPathLength >= maxOrder, \
            'Error: Construction of multi-order model with maximum order M ' \
            'requires sub path statistics up to length M'

        ## A dictionary containing the layers of HigherOrderNetworks, where
        ## layers[k] contains the network of order k
        self.layers = {}

        ## the maximum order of this multi-order model
        self.maxOrder = maxOrder

        ## the paths object from which this multi-order model was created
        self.paths = paths

        ## a dictionary of transition matrices for all layers of the model
        self.T = {}

        if pathpy.ENABLE_MULTICORE_SUPPORT:
            try:
                import pathos as _pa
            except ImportError:  # pragma: no cover
                raise ImportError('Error while importing module "pathos",'
                                  'to use the parallel feature "pathos" '
                                  'needs to be installed')

            def parallel(k):  # pragma: no cover
                Log.add('Generating ' + str(k) + '-th order network layer ...')
                layer = HigherOrderNetwork(paths, k, paths.separator, False)

                # compute transition matrices for all layers. In order to use the
                # maximally available statistics, we always use sub paths in the
                # calculation
                T = layer.transition_matrix(include_subpaths=True)

                Log.add('Generating ' + str(k) + '-th order network layer ... finished')
                return [k, layer, T]

            pool = _pa.multiprocessing.ProcessPool()
            results = pool.map(parallel, range(maxOrder+1))

            # save results
            for k, layer, T in results:
                self.layers[k] = layer
                self.T[k] = T

        else:
            for k in range(maxOrder+1):
                Log.add('Generating ' + str(k) + '-th order network layer ...')
                self.layers[k] = HigherOrderNetwork(paths, k, paths.separator, False)

                # compute transition matrices for all layers. In order to use the maximally
                # available statistics, we always use sub paths in the calculation
                self.T[k] = self.layers[k].transition_matrix(include_subpaths=True)

            Log.add('finished.')

    def summary(self):
        """
        Returns a string containing basic summary information
        of this multi-order model instance.
        """
        summary_fmt = (
            "Multi-order model (max. order = {order}, "
            "DoF (paths/ngrams) = {dof_path} / {dof_ngram})\n"
            '==========================================================================\n'
            )
        summary = summary_fmt.format(
            order=self.maxOrder,
            dof_path=self.degrees_od_freedom(assumption='paths'),
            dof_ngram=self.degrees_od_freedom(assumption='ngrams')
        )
        layer_fmt = ("Layer k = {k} \t {vcount} nodes, {ecount} links, {sum_path} paths, "
                     "DoF (paths/ngrams) = {dof_paths} / {dof_ngram} \n")

        for k in range(self.maxOrder+1):
            vcount = self.layers[k].vcount()
            ecount = self.layers[k].ecount()
            sum_path = self.layers[k].total_edge_weight().sum()
            dof_paths = int(self.layers[k].degrees_of_freedom('paths'))
            dof_ngram = int(self.layers[k].degrees_of_freedom('ngrams'))

            layer_sum = layer_fmt.format(k=k, vcount=vcount, ecount=ecount,
                                         sum_path=sum_path, dof_paths=dof_paths,
                                         dof_ngram=dof_ngram)
            summary += layer_sum
        return summary

    def save_state_file(self, filename, layer=None, infomapIndexing=None):
        """
        Saves the multi-order model in state file format
        suitable to be used with InfoMap

        @param layer: if none, all layers will be export. If set to k, only
            the k-th layer of the model will be exported.
        @param infomapIndexing: if none, standard pathpy indices will be used
            in the export of state files. This can be set to a custom index dictionary
            in which infomapIndexing[k] contains a dictionary that maps k-th order nodes
            to a custum node index. This is useful to create state files with consistent indices
            from multiple MultiOrderModels
        """

        assert layer, 'Export of all layers is currently not supported'


        name_map = self.layers[layer].node_to_name_map()
        first_layer_map = self.layers[1].node_to_name_map()

        T = self.layers[layer].transition_matrix()

        file = open(filename,'w')

        file.write('# this file was generated by pathpy\n')

        # Note: InfoMap requires consecutive indexing of nodes!
        if infomapIndexing:
            file.write('*Vertices {0}\n'.format(len(infomapIndexing[1])))
            # sort indices
            reverse_index = {}
            for name, node_idx in infomapIndexing[1].items():
                reverse_index[node_idx] = name
            sorted_indices = list(reverse_index.keys())
            for node_idx in sorted_indices:
                file.write('{0} "{1}"\n'.format(node_idx, reverse_index[node_idx]))
        else:
            file.write('*Vertices {0}\n'.format(self.layers[1].vcount()))
            for i in range(self.layers[1].vcount()):
                file.write('{0} "{1}"\n'.format(first_layer_map[self.layers[1].nodes[i]]+1, self.layers[1].nodes[i]))

        # Write higher-order nodes to states section
        file.write('*States {0}\n'.format(self.layers[layer].vcount()))
        for i in range(self.layers[layer].vcount()):
            v = self.layers[layer].nodes[i]
            if infomapIndexing:
                v_ix = infomapIndexing[layer][v]
            else:
                v_ix = name_map[v]+1
            v_path = self.layers[layer].higher_order_node_to_path(v)

            # each line contains uniqueID physicalID [name]
            if infomapIndexing:
                file.write('{0} {1} "{2}"\n'.format(v_ix, infomapIndexing[1][v_path[-1]], v))
            else:
                file.write('{0} {1} "{2}"\n'.format(v_ix, first_layer_map[v_path[-1]]+1, v))

        file.write('*Links {0}\n'.format(self.layers[layer].ecount()))
        for e in self.layers[layer].edges:
            source = e[0]
            target = e[1]

            # Get source and target paths
            #source_p = self.layers[layer].higher_order_node_to_path(source)
            #source_t = self.layers[layer].higher_order_node_to_path(target)

            source_ix = name_map[source]
            target_ix = name_map[target]

            # Get edge weight
            #w_st = self.layers[layer].edges[e][1]

            # Get transition probability
            T_st = T[target_ix, source_ix]

            # Write entry to file
            # each line contains from to [weight]
            if infomapIndexing:
                file.write('{0} {1} {2}\n'.format(infomapIndexing[layer][source], infomapIndexing[layer][target], T_st))
            else:
                file.write('{0} {1} {2}\n'.format(source_ix+1, target_ix+1, T_st))

        file.close()

    def __str__(self):
        """
        Returns the default string representation of
        this multi-order model instance
        """
        return self.summary()

    def likelihood(self, paths, maxOrder=None, log=True):
        """Calculates the likelihood of a multi-order
        network model up to a maximum order maxOrder based on all
        path statistics.

        @param paths: the path statistics to be used in the likelihood
            calculation

        @param maxOrder: the maximum layer order to take into
            account for the likelihood calculation. For the default
            value None, all orders will be used for the
            likelihood calculation.

        @log: Whether or not to return the log likelihood (default: True)
        """
        maxOrder = self.maxOrder if maxOrder is None else maxOrder
        assert maxOrder <= self.maxOrder, \
            'Error: maxOrder cannot be larger than maximum order of multi-order network'

        # add log-likelihoods of multiple model layers,
        # assuming that paths are independent
        L = _np.float64(0)

        for k in range(0, maxOrder+1):
            if k < maxOrder:
                p = self.layer_likelihood(paths, k, considerLongerPaths=False, log=True)[0]
            else:
                p = self.layer_likelihood(paths, k, considerLongerPaths=True, log=True)[0]
            # print('Log L(k=' + str(k) + ') = ' + str(p))
            assert p <= 0, 'Layer Log-Likelihood out of bounds'
            L += p
        assert L <= 0, 'Log-Likelihood out of bounds'

        return L if log else _np.exp(L)

    @staticmethod
    def factorial(n, log=True):  # pragma: no cover
        """
        Calculates (or approximates) the (log of the) factorial n!.
        The function applies Stirling's approximation if n>20.

        @param n: computes factorial of n
        @param log: whether or not to return the (natural) logarithm of the factorial
        """
        f = _np.float64(0)
        n_ = _np.float64(n)
        if n > 20:  # use Stirling's approximation
            try:
                f = n_ * _np.log(n_)-n_ + 0.5 * _np.log(2.0*_np.pi*n_)+1.0/(12.0*n_)-1/(360.0*n_**3.0)
            except Warning as w:
                Log.add('Factorial calculation for n = ' + str(n)+ ': ' + str(w), severity=Severity.WARNING)

        else:
            f = _np.log(_np.math.factorial(n))

        if log:
            return f
        else:
            return _np.exp(f)

    def layer_likelihood(self, paths, l=1, considerLongerPaths=True, log=True, minL=None):
        """
        Calculates the (log-)likelihood of the **first** l layers of a multi-order network model
        using all observed paths of (at least) length l

        @param paths: the path statistics for which to calculate the layer likelihood

        @param l: number of layers for which likelihood shall be calculated
            Paths of length l (and possibly longer) will be used to calculate the likelihood
            of model layers for all orders up to l

        @param considerLongerPaths: whether or not to include paths longer than l
            in the calculation of the likelihood. In general, when calculating the likelihood
            of a multi-order model which combines orders from 1 to l, this should be set to
            true only for the value of l that corresponds to the largest order in the model.

        @param log: whether to compute Log-Likelihood (default: True)

        @param minL: minimum length of paths which enter the likelihood calculation. For the
            default value None, all paths with at least length l will be considered.

        @returns: the (log-)likelihood of the model layer given the path statistics
        """

        # m is the maximum length of any path in the data
        m = max(paths.paths)

        assert m >= l and len(paths.paths[l])>0, 'Error: there are no paths of length l or longer'

        if minL == None:
            minL = l

         # Set maximum length of paths to consider in likelihood calculation
        if considerLongerPaths:
            maxL = m
        else:
            maxL = l

        # For the paths S_k of length k (or longer) that we observe, we need to calculate
        # the probability of observing all paths in S_k based on the probabilities of
        # individual paths (which are calculated using the underlying Markov model(s))

        # n is the total number of path observations
        n = 0

        # Initialize likelihood
        L = 0

        # compute likelihood for all longest paths
        # up to the maximum path length maxL
        for k in range(minL, maxL+1):
            for p in paths.paths[k]:

                # Only consider observations as *longest* path
                if paths.paths[k][p][1]>0:

                    # Add m_i observations of path p to total number of observations n
                    n += paths.paths[k][p][1]

                    # special case: to calculate the likelihood of the path based on a zero-order model we
                    # use the 'start' -> v transitions in the respective model instance
                    if l == 0:
                        for s in range(len(p)):
                            L += _np.log(self.T[0][self.layers[0].nodes.index(p[s]), self.layers[0].nodes.index('start')]) * paths.paths[k][p][1]

                    # general case: compute likelihood of path based on
                    # hierarchy of higher-order models as follows ...
                    else:

                        # 1.) transform the path into a sequence of (two or more) l-th-order nodes
                        nodes = self.layers[l].path_to_higher_order_nodes(p)
                        # print('l-th order path = ', str(nodes))

                        # 2.) nodes[0] is the prefix of the k-th order transitions, which we
                        #   can transform into multiple transitions in lower order models.
                        #   Example: for a path a-b-c-d of length three, the node sequence
                        #   at order l=3 is ['a-b-c', 'b-c-d'] and thus the prefix is 'a-b-c'.
                        prefix = nodes[0]

                        # 3.) We extract the transitions for the prefix based on models of
                        #   orders k_<l. In our example, we have the transitions ...
                        #   (a-b, b-c) for k_=2
                        #   (a, b) for k_=1, and
                        #   (start, a) for k_=0
                        transitions = {}

                        # for all k_<l in descending order
                        for k_ in range(l-1, -1, -1):
                            #print('prefix = ', prefix)
                            x = prefix.split(self.layers[k_].separator)
                            transitions[k_] = self.layers[k_].path_to_higher_order_nodes(x)
                            #print('transition (k_=', k_,') = ', transitions[k_])
                            prefix = transitions[k_][0]

                        # 4.) Using Bayes theorem, we calculate the likelihood of a path a-b-c-d-e
                        #   of length four for l=4 as a single transition in a fourth-order model, and
                        #   four additional transitions in the k_=0, 1, 2 and 3-order models, i.e. we have ...
                        #   P(a-b-c-d-e) = P(e|a-b-c-d) * [ P(d|a-b-c) * P(c|a-b) * P(b|a) * P(a) ]
                        #   If we were to model the same path based on model hierarchy with a maximum order of l=2,
                        #   we instead have three transitions in the second-order model and two additional transitions
                        #   in the k_=0 and k_=1 order models for the prefix 'a-b' ...
                        #   P(a-b-c-d-e) = P(e|c-d) * P(d|b-c) * P(c|a-b) * [ P(b|a) * P(a) ]

                        # First multiply the transitions in the l-th order model ...
                        transition_matrix = self.T[l]
                        nodes_ = self.layers[l].nodes
                        factor_ = paths.paths[k][p][1]
                        for s in range(len(nodes)-1):
                            # print((nodes[s], nodes[s+1]))
                            # print(T[model.nodes.index(nodes[s+1]), model.nodes.index(nodes[s])])
                            L += _np.log(transition_matrix[nodes_.index(nodes[s+1]), nodes_.index(nodes[s])]) * factor_


                        # ... then multiply additional transition probabilities for the prefix ...
                        for k_ in range(l):
                            L += _np.log(self.T[k_][self.layers[k_].nodes.index(transitions[k_][1]), self.layers[k_].nodes.index(transitions[k_][0])]) * factor_

            if n == 0:
                L = 0
        if log:
            assert L <= 0, 'Log-Likelihood out of bounds'
            return L, n
        else:
            assert 0 <= L <= 1, 'Likelihood out of bounds'
            return _np.exp(L), n

    def degrees_od_freedom(self, maxOrder=None, assumption="paths"):
        """
        Calculates the degrees of freedom of the model based on
        different assumptions, and taking into account layers up to
        a maximum order.

        @param: maxOrder: the maximum order up to which model layers shall be
            taken into account

        @param assumption: if set to 'paths', for the degree of freedom calculation
            only paths in the first-order network topology will be considered. This is
            needed whenever we model paths in a *given* network topology.
            If set to 'ngrams' all possible n-grams will be considered, independent of whether they
            are valid paths in the first-order network or not. The 'ngrams' and the 'paths' assumption
            coincide if the first-order network is fully connected, i.e. if all possible paths actually occur.
        """
        if maxOrder is None:
            maxOrder = self.maxOrder
        assert maxOrder <= self.maxOrder, \
            'Error: maxOrder cannot be larger than maximum order of multi-order network'

        dof = 0

        # Sum degrees of freedom of all model layers up to maxOrder
        for i in range(0, maxOrder+1):
            dof += self.layers[i].degrees_of_freedom(assumption)

        return int(dof)

    def model_size(self, maxOrder):
        """
        Returns the total number of non-zero
        transition matrix entries in all
        model layers
        """
        maxOrder = self.maxOrder if maxOrder is None else maxOrder
        assert maxOrder <= self.maxOrder, \
            'Error: maxOrder cannot be larger than maximum order of multi-order network'

        size = 0
        for i in range(0, maxOrder+1):
            size += self.layers[i].model_size()
        return int(size)

    def likelihood_ratio_test(self, paths, maxOrderNull=0, maxOrder=1, assumption='paths', significanceThreshold=0.01):
        """
        Performs a likelihood-ratio test between two multi-order models with given maximum orders, where maxOrderNull serves
        as null hypothesis and maxOrder serves as alternative hypothesis. The null hypothesis is rejected if the p-value for
        the observed paths under the null hypothesis is smaller than the given significance threshold.

        Applying this test makes the assumption that we have nested models, i.e. that the null model is contained
        as a special case in the parameter space of the more complex model. If we assume that the path constraint holds,
        this is not true for the test of the first- against the zero-order model (since some sequences of the zero order model
        cannot be generated in the first-order model). However, since the set of possible higher-order transitions is generated
        based on the first-order model, the nestedness property holds for all higher order models.

        @param paths: the path data to be used in the liklihood ratio test
        @param maxOrderNull: maximum order of the multi-order model
                to be used as a null hypothesis
        @param maxOrder: maximum order of the multi-order model to be used as
                alternative hypothesis
        @param assumption: paths or ngrams
        @param significanceThreshold: the threshold for the p-value
                below which to accept the alternative hypothesis
        @returns: a tuple of the format (reject, p) which captures whether or
                not the null hypothesis is rejected in favor of the alternative
                hypothesis, as well as the p-value that led to the decision
        """

        assert maxOrderNull < maxOrder, 'Error: order of null hypothesis must be smaller than order of alternative hypothesis'
        # let L0 be the likelihood for the null model and L1 be the likelihood for the alternative model

        # we first compute a test statistic x = -2 * log (L0/L1) = -2 * (log L0 - log L1)
        x = -2 * (self.likelihood(paths, maxOrder=maxOrderNull, log=True) - self.likelihood(paths, maxOrder=maxOrder, log=True))

        # we calculate the additional degrees of freedom in the alternative model
        dof_diff = self.degrees_od_freedom(maxOrder=maxOrder, assumption=assumption) - self.degrees_od_freedom(maxOrder=maxOrderNull, assumption=assumption)

        Log.add('Likelihood ratio test for K_opt = ' + str(maxOrder) + ', x = ' + str(x))
        Log.add('Likelihood ratio test, d_1-d_0 = ' + str(dof_diff))

        # if the p-value is *below* the significance threshold, we reject the null hypothesis
        p = 1-chi2.cdf(x, dof_diff)

        Log.add('Likelihood ratio test, p = ' + str(p))
        return (p<significanceThreshold), p

    def estimate_order(self, paths, maxOrder=None, significanceThreshold=0.01):
        """
        Selects the optimal maximum order of a multi-order network model for the
        observed paths, based on a likelihood ratio test with p-value threshold of p
        By default, all orders up to the maximum order of the multi-order model will be tested.

        @param paths: The path statistics for which to perform the order selection

        @param maxOrder: The maximum order up to which the multi-order model shall be tested.
        """
        if maxOrder is None:
            maxOrder = self.maxOrder
        assert maxOrder <= self.maxOrder, 'Error: maxOrder cannot be larger than maximum order of multi-order network'
        assert maxOrder > 1, 'Error: maxOrder must be larger than one'

        maxAcceptedOrder = 1

        # Test for highest order that passes
        # likelihood ratio test against null model
        for k in range(2, maxOrder+1):
            if self.likelihood_ratio_test(paths, maxOrderNull=k - 1, maxOrder=k, significanceThreshold=significanceThreshold)[0]:
                maxAcceptedOrder = k

        return maxAcceptedOrder

    def test_network_hypothesis(self, paths, method='AIC'):
        """
        Tests whether the assumption that paths are constrained
        to the (first-order) network topology is justified.
        Roughly speaking, this test yields true if the gain in
        explanatory power that is due to the network topology
        justifies the additional model complexity.

        The decision will be made based on a comparison between the zero-
        and the first-order layer of the model. Different from the multi-order
        model selection method implemented in estimate_order and likelihoodRatioTest,
        here we do *not* consider nested models, so we cannot use a likelihood ratio
        test. We instead use the AIC or BIC.
        """

        assert method in ['AIC', 'BIC', 'AICc'],\
            'Expected method AIC, AICc or BIC "%s" given.' % method

        # count number of omitted paths with length zero
        p_sum = 0
        for p in paths.paths[0]:
            p_sum += paths.paths[0][p][1]
        if p_sum > 0:
            Log.add('Omitting ' + str(p_sum) + ' zero-length paths for test of network assumption', Severity.INFO)

        # log-likelihood and observation count of zero-order model
        L0, n0 = self.layer_likelihood(paths, l=0, considerLongerPaths=True, log=True, minL=1)

        # log-likelihood and observation count of first-order model
        L1, n1 = self.layer_likelihood(paths, l=1, considerLongerPaths=True, log=True, minL=1)

        # By definition, the number of observations for both models should be the total
        # weighted degree of the first-order network
        if n0 != n1:
            raise _pp.PathpyError(
                'Observation count for 0-order ({n0}) and '
                '1-st order model ({n1}) do not match'.format(n0=n0, n1=n1)
            )

        # degrees of freedom = |V|-1
        dof0 = self.layers[0].degrees_of_freedom(assumption='ngrams')

        # degrees of freedom based on network assumption
        dof1 = self.layers[1].degrees_of_freedom(assumption='paths')

        Log.add('Log-Likelihood (k=0) = ' + str(L0), Severity.INFO)
        Log.add('Degrees of freedom (k=0) = ' + str(dof0), Severity.INFO)

        Log.add('Log-Likelihood (k=1) = ' + str(L1), Severity.INFO)
        Log.add('Degrees of freedom (k=1) = ' + str(dof0 + dof1), Severity.INFO)

        if method == 'AIC':
            ic0 = 2 * dof0 - 2 * L0
            ic1 = 2 * (dof0 + dof1) - 2 * L1
        elif method == 'AICc':
            assert n1 > dof0+dof1-2, 'Error: number of samples too small for model complexit'
            ic0 = 2 * dof0 - 2 * L0 + (2*(dof0+1)*(dof0+2))/(n0-dof0-2)
            ic1 = 2 * (dof0 + dof1) - 2 * L1 + (2*(dof0 + dof1 + 1)*(dof0 + dof1 + 2))/(n1 - (dof0 + dof1) - 2)
        elif method == 'BIC':
            ic0 = _np.log(n0) * dof0 - 2 * L0
            ic1 = _np.log(n1) * (dof0 + dof1) - 2 * L1
        else:
            raise _pp.PathpyError("Method check has not filtered out illegal "
                                  "method %s " % method)

        # if the AIC/AICc/BIC of the zero-order model is larger than that of the
        # first-order model, we do not reject the network hypothesis
        return ic0 > ic1, ic0, ic1
