# -*- coding: utf-8 -*-

#    pathpy is an OpenSource python package for the analysis of time series data
#    on networks using higher- and multi order graphical models.
#
#    Copyright (C) 2016-2018 Ingo Scholtes, ETH Zürich/Universität Zürich
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
#    along with this program. If not, see <http://www.gnu.org/licenses/>.
#
#    Contact the developer:
#
#    E-mail: scholtes@ifi.uzh.ch
#    Web:    http://www.ingoscholtes.net
import numpy as np
from scipy.stats import chi2

from pathpy.utils import Log, Severity
from pathpy.utils.exceptions import PathpyError, PathsTooShort, PathpyNotImplemented
from pathpy.classes.higher_order_network import HigherOrderNetwork
from pathpy.classes.paths import Paths


np.seterr(all='warn')


class MultiOrderModel:
    """
    A hierarchy of higher-order networks which jointly represent
    a multi-order model for path statistics.

    Attributes
    ----------
    paths: Paths
        The Paths instance that this multi-order model was generated from.
    layers: dict
        A dictionary where layers[k] contains the higher-order model with order k
    """

    def __init__(self, paths, max_order=1):
        """Generates a hierarchy of higher-order models for the given path statistics
        up to a given maximum order

        Parameters
        ----------
        paths: Paths
            the paths instance for which the model should be created
        max_order: int
            the maximum order of the multi-order model
        """
        assert paths.max_subpath_length >= max_order, \
            'Error: Construction of multi-order model with maximum order M ' \
            'requires sub path statistics up to length M'

        # the paths object from which this multi-order model was created
        self.paths = paths

        """A dictionary containing the layers of HigherOrderNetworks, where
        # layers[k] contains the network of order k"""
        self.layers = {}
        
        # a dictionary of transition matrices for all layers of the model
        self.transition_matrices = {}

        self.add_layers(max_order)

    @property
    def max_order(self):
        """The current maximum order of the multi-order model"""
        orders = list(self.layers.keys())
        if not orders:
            return None
        else:
            return max(orders)

    def __add_layers_parallel(self, orders):
        paths = self.paths
        try:
            import pathos as _pa
        except ImportError:  # pragma: no cover
            raise ImportError('Error while importing module "pathos",'
                              'to use the parallel feature "pathos" '
                              'needs to be installed')

        def parallel(order_k):  # pragma: no cover
            Log.add('Generating ' + str(order_k) + '-th order network layer ...')
            p_layer = HigherOrderNetwork(paths, k=order_k, null_model=False)

            # compute transition matrices for all layers. In order to use the
            # maximally available statistics, we always use sub paths in the
            # calculation
            trans_mat = p_layer.transition_matrix(include_subpaths=True)

            Log.add('... finished')
            return [order_k, p_layer, trans_mat]

        pool = _pa.multiprocessing.ProcessPool()
        results = pool.map(parallel, orders)

        # save results
        for k, layer, transition_mat in results:
            self.layers[k] = layer
            self.transition_matrices[k] = transition_mat

    def __add_layers_sequential(self, orders):
        paths = self.paths

        for k in sorted(orders):
            Log.add('Generating %d-th order layer ...' % k)
            self.layers[k] = HigherOrderNetwork(paths, k, null_model=False)

            # compute transition matrices for all layers. In order to use the
            # maximally available statistics, we always use sub paths in the
            # calculation
            self.transition_matrices[k] = self.layers[k].transition_matrix(include_subpaths=True)

        Log.add('finished.')

    def add_layers(self, max_order):
        """Add higher-order layers up to the given maximum order.

        Parameters
        ----------
        max_order: int
            up to which order to add higher order layers, if below the current maximum the
            operation will have no effect and the HigherOrderNetwork will remain unchanged.

        """
        from pathpy import ENABLE_MULTICORE_SUPPORT

        current_max_order = self.max_order if self.max_order else -1
        if max_order < 0:
            raise PathpyError("max_order must be a positive integer not %d" % max_order)

        if max_order <= current_max_order:
            return
#             Log.add("Layers up to order %d already added. Nothing changed." % self.max_order)

        orders_to_add = list(range(current_max_order+1, max_order+1))
        if len(orders_to_add) > 1 and ENABLE_MULTICORE_SUPPORT:
            self.__add_layers_parallel(orders_to_add)
        else:
            self.__add_layers_sequential(orders_to_add)

    def summary(self):
        """
        Returns a string containing summary information
        on a multi-order model.
        """
        summary_fmt = (
            "Multi-order model (max. order = {order}, "
            "DoF (paths/ngrams) = {dof_path} / {dof_ngram})\n"
            '==========================================================================\n'
        )
        summary = summary_fmt.format(
            order=self.max_order,
            dof_path=self.degrees_of_freedom(assumption='paths'),
            dof_ngram=self.degrees_of_freedom(assumption='ngrams')
        )
        layer_fmt = ("Layer k = {k} \t {ncount} nodes, {ecount} links, {sum_path} paths, "
                     "DoF (paths/ngrams) = {dof_paths} / {dof_ngram} \n")

        for k in range(self.max_order + 1):
            ncount = self.layers[k].ncount()
            ecount = self.layers[k].ecount()
            sum_path = self.layers[k].total_edge_weight().sum()
            dof_paths = int(self.layers[k].degrees_of_freedom('paths'))
            dof_ngram = int(self.layers[k].degrees_of_freedom('ngrams'))

            layer_sum = layer_fmt.format(k=k, ncount=ncount, ecount=ecount,
                                         sum_path=sum_path, dof_paths=dof_paths,
                                         dof_ngram=dof_ngram)
            summary += layer_sum
        return summary

    def save_state_file(self, filename, layer=None, infomap_indexing=None):
        """Saves the multi-order model in state file format suitable to be used with
         InfoMap

        Parameters
        ----------
        filename
        layer: int
            if none, all layers will be export. If set to k, only the k-th layer of the
            model will be exported.
        infomap_indexing: dict
            if none, standard pathpy indices will be used in the export of state files.
            This can be set to a custom index dictionary in which infomap_indexing[k]
            contains a dictionary that maps k-th order nodes to a custom node index.
            This is useful to create state files with consistent indices from multiple
            MultiOrderModels

        Returns
        -------

        """
        assert layer, 'Export of all layers is currently not supported'

        name_map = self.layers[layer].node_to_name_map()
        first_layer_map = self.layers[1].node_to_name_map()

        trans_mat = self.layers[layer].transition_matrix()

        file = open(filename, 'w')

        file.write('# this file was generated by pathpy\n')

        # Note: InfoMap requires consecutive indexing of nodes!
        if infomap_indexing:
            file.write('*Vertices {0}\n'.format(len(infomap_indexing[1])))
            # sort indices
            reverse_index = {}
            for name, node_idx in infomap_indexing[1].items():
                reverse_index[node_idx] = name
            sorted_indices = list(reverse_index.keys())
            for node_idx in sorted_indices:
                file.write('{0} "{1}"\n'.format(node_idx, reverse_index[node_idx]))
        else:
            file.write('*Vertices {0}\n'.format(self.layers[1].ncount()))
            for i in self.layers[1].nodes:
                idx = first_layer_map[i]

                file.write('{0} "{1}"\n'.format(idx, i))

        # Write higher-order nodes to states section
        file.write('*States {0}\n'.format(self.layers[layer].ncount()))
        for v in self.layers[layer].nodes:
            if infomap_indexing:
                v_ix = infomap_indexing[layer][v]
            else:
                v_ix = name_map[v]
            v_path = self.layers[layer].higher_order_node_to_path(v)

            # each line contains uniqueID physicalID [name]
            if infomap_indexing:
                file.write(
                    '{0} {1} "{2}"\n'.format(v_ix, infomap_indexing[1][v_path[-1]], v))
            else:
                file.write(
                    '{0} {1} "{2}"\n'.format(v_ix, first_layer_map[v_path[-1]], v))

        file.write('*Links {0}\n'.format(self.layers[layer].ecount()))
        for e in self.layers[layer].edges:
            source = e[0]
            target = e[1]

            # Get source and target paths
            # source_p = self.layers[layer].higher_order_node_to_path(source)
            # source_t = self.layers[layer].higher_order_node_to_path(target)

            source_ix = name_map[source]
            target_ix = name_map[target]

            # Get edge weight
            # w_st = self.layers[layer].edges[e][1]

            # Get transition probability
            trans_prop = trans_mat[target_ix, source_ix]

            # Write entry to file
            # each line contains from to [weight]
            if infomap_indexing:
                idx_s = infomap_indexing[layer][source]
                idx_t = infomap_indexing[layer][target]
                file.write('{} {} {}\n'.format(idx_s, idx_t, trans_prop))
            else:
                file.write('{} {} {}\n'.format(source_ix, target_ix, trans_prop))

        file.close()

    def __str__(self):
        """
        Returns the default string representation of
        this multi-order model instance
        """
        return self.summary()

    def likelihood(self, paths=None, max_order=None, log=True):
        """Calculates the likelihood of a multi-order
        network model up to a maximum order max_order based on all
        path statistics.

        Parameters
        ----------
        paths:
            the path statistics to be used in the likelihood calculation
        max_order:
            the maximum layer order to take into account for the likelihood calculation.
            For the default value None, all orders will be used for the likelihood
            calculation.
        log: bool
            Whether or not to return the log likelihood (default: True)

        Returns
        -------

        """
        max_order = self.max_order if max_order is None else max_order
        assert max_order <= self.max_order, \
            'Error: max_order cannot be larger than maximum order of multi-order network'

        # add log-likelihoods of multiple model layers,
        # assuming that paths are independent
        likelihood = np.float64(0)

        for k in range(0, max_order + 1):
            if k < max_order:
                p = self.layer_likelihood(paths, k, consider_longer_paths=False, log=True)[0]
            else:
                p = self.layer_likelihood(paths, k, consider_longer_paths=True, log=True)[0]
            # print('Log L(k=' + str(k) + ') = ' + str(p))
            assert p <= 0, 'Layer Log-Likelihood out of bounds'
            likelihood += p
        assert likelihood <= 0, 'Log-Likelihood out of bounds'

        return likelihood if log else np.exp(likelihood)

    @staticmethod
    def factorial(n, log=True):  # pragma: no cover
        """
        Calculates the factorial of n, automatically switching to 
        Stirling's approaximation for n>20.

        Parameters
        ----------
        n: int
            The value n for which the fatorial should be calculated.
        log: bool
            Whether or not to return the (natural) logarithm of the factorial. Default is True.

        Returns
        -------
        float
        """

        f = np.float64(0)
        n_ = np.float64(n)
        if n > 20:  # use Stirling's approximation
            try:
                f = (n_ * np.log(n_) - n_ + 0.5 * np.log(2.0 * np.pi * n_)
                     + 1.0 / (12.0 * n_) - 1 / (360.0 * n_ ** 3.0))
            except Warning as w:
                msg = 'Factorial calculation for n={}: {}'.format(n, w)
                Log.add(msg, severity=Severity.WARNING)

        else:
            f = np.log(np.math.factorial(n))

        if log:
            return f
        else:
            return np.exp(f)

    def layer_likelihood(self, paths=None, l=1, consider_longer_paths=True, log=True,
                         min_path_length=None):
        """
        Calculates the (log-)likelihood of the **first** l layers of a multi-order
        network model using all observed paths of (at least) length l

        Parameters
        ----------
        paths: Paths
            the path statistics for which to calculate the layer likelihood
        l: int
            number of layers for which likelihood shall be calculated. Paths of length l
            (and possibly longer) will be used to calculate the likelihood of model layers
            for all orders up to l
        consider_longer_paths: bool
            whether or not to include paths longer than l in the calculation of the
            likelihood. In general, when calculating the likelihood of a multi-order model
            which combines orders from 1 to l, this should be set to true only for the
            value of l that corresponds to the largest order in the model.
        log: bool
             whether to compute Log-Likelihood (default: True)
        min_path_length: int
            minimum length of paths which enter the likelihood calculation. For the
            default value None, all paths with at least length l will be considered.

        Returns
        -------
        float
            the (log-)likelihood of the model layer given the path statistics
        """
        # m is the maximum length of any path in the data
        if paths is None:
            paths = self.paths
        max_len_obs = max(paths.paths)

        if min_path_length is None:
            min_path_length = l

        # Set maximum length of paths to consider in likelihood calculation
        if consider_longer_paths:
            maxL = max_len_obs
        else:
            maxL = l

        # create index maps to map node names to matrix indices
        indexmaps = {}
        for k in range(0, l+1):
            indexmaps[k] = self.layers[k].node_to_name_map()

        # For the paths S_k of length k (or longer) that we observe, we need to calculate
        # the probability of observing all paths in S_k based on the probabilities of
        # individual paths (which are calculated using the underlying Markov model(s))

        # n is the total number of path observations
        n = 0

        # Initialize likelihood
        likelihood = 0

        # compute likelihood for all longest paths
        # up to the maximum path length maxL
        for k in range(min_path_length, maxL + 1):
            for p in paths.paths[k]:
                # Only consider observations as *longest* path
                freq = paths.paths[k][p][1]
                if freq > 0:
                    n += freq  # Add m_i observations of path p to total number of observations n
                    likelihood += self.path_likelihood(p, freq, l, log=True, index_maps=indexmaps)
            if n == 0:
                likelihood = 0
        if log:
            assert likelihood <= 0, 'Log-Likelihood out of bounds'
            return likelihood, n
        else:
            assert 0 <= likelihood <= 1, 'Likelihood out of bounds'
            return np.exp(likelihood), n

    def path_likelihood(self, path, freq=1, layer=1, log=True, index_maps=None):
        """Computes the model likelihood given a single path.

        Parameters
        ----------
        path: tuple
            the path for which the likelihood should be computed, the path must be a tuple whose
            elements represent the path, i.e. ('a', 'b', 'd')
        freq: int
            the frequency of the path
        layer: int
            the layer up to which the likelihood should be computed
        log: bool
            true if the log-likelihood should be returned
        index_maps: dict
            a dictionary mapping the nodes for all orders to their index in the transition matrix.
            For simple calls this is computed automatically, however if the a lot of paths will be
            calculated it is best to pre-compute it and pass it.

        Returns
        -------
        float:
            likelihood or log-likelihood of path

        Raises
        -----
        PathpyNotImplemented

        Examples
        -------
        >>> layer = 2
        >>> p = Paths()
        >>> p.add_path(('1', '3', '2'))
        >>> p.add_path(('3', '2', '1'))
        >>> p.add_path(('1', '2', '1'))
        >>> mom = MultiOrderModel(p, max_order=layer)
        >>> # a precomputed index map can be obtained as follows
        >>> index_maps = {k: mom.layers[k].node_to_name_map() for k in range(0, layer+1)}
        >>> format(mom.path_likelihood(('1', '2', '1'), log=False, layer=layer), '.2f')
        '0.22'
        >>> # trying to compute the likelihood of an nonexistent path
        >>> format(mom.path_likelihood(('1', '2', '2'), log=False, layer=layer), '.2f')
        Traceback (most recent call last):
        ...
        pathpy.utils.exceptions.PathpyNotImplemented: The path segment '(2,2)' has not been \
observed and therefore the likelihood cannot be computed.


        """
        if index_maps is None:
            index_maps = {k: self.layers[k].node_to_name_map() for k in range(0, layer+1)}

        likelihood = 0
        # special case: to calculate the likelihood of the path based on a
        # zero-order model we use the 'start' -> v transitions in the
        # respective model instance
        if layer == 0:
            try:
                for s in range(len(path)):
                    source = index_maps[0]['start']
                    target = index_maps[0][path[s]]
                    likelihood += np.log(self.transition_matrices[0][target, source]) * freq
            except KeyError as e:
                msg = ("The path segment '({})' has not been observed and therefore the "
                       "likelihood cannot be computed.").format(e.args[0])
                raise PathpyNotImplemented(msg)

        # general case: compute likelihood of path based on hierarchy of higher-order models
        else:
            # 1.) transform the path into a sequence of (two or more) l-th-order nodes
            nodes = self.layers[layer].path_to_higher_order_nodes(path)
            # print('l-th order path = ', str(nodes))

            # 2.) nodes[0] is the prefix of the k-th order transitions,  which we can transform
            # into multiple transitions in lower order models. Example: for a path a-b-c-d of
            # length three, the node sequence at order l=3 is ['a-b-c', 'b-c-d'] and thus the
            # prefix is 'a-b-c'.
            prefix = nodes[0]

            # 3.) We extract the transitions for the prefix based on models of orders k_<l. In
            # our example, we have the transitions ... (a-b, b-c) for k_=2 (a, b) for k_=1,
            # and (start, a) for k_=0
            transitions = {}

            # for all k_<l in descending order
            for k_ in range(layer - 1, -1, -1):
                x = prefix.split(self.layers[k_].separator)
                transitions[k_] = self.layers[k_].path_to_higher_order_nodes(x)
                prefix = transitions[k_][0]

            # 4.) Using Bayes theorem, we calculate the likelihood of a path a-b-c-d-e of length
            # four for l=4 as a single transition in a fourth-order model, and four additional
            # transitions in the k_=0, 1, 2 and 3-order models, i.e. we have ... P(a-b-c-d-e) =
            # P(e|a-b-c-d) * [ P(d|a-b-c) * P(c|a-b) * P(b|a) * P(a) ] If we were to model the
            # same path based on model hierarchy with a maximum order of l=2, we instead have
            # three transitions in the second-order model and two additional transitions in the
            # k_=0 and k_=1 order models for the prefix 'a-b' ... P(a-b-c-d-e) = P(e|c-d) * P(
            # d|b-c) * P(c|a-b) * [ P(b|a) * P(a) ]

            # First multiply the transitions in the l-th order model ...
            transition_matrix = self.transition_matrices[layer]
            try:
                for s in range(len(nodes)-1):
                    idx_s1 = index_maps[layer][nodes[s + 1]]
                    idx_s0 = index_maps[layer][nodes[s]]
                    trans_mat = transition_matrix[idx_s1, idx_s0]
                    likelihood += np.log(trans_mat) * freq
                # ... then multiply additional transition probabilities for the prefix ...
                for k_ in range(layer):
                    trans_idx0 = index_maps[k_][transitions[k_][0]]
                    trans_idx1 = index_maps[k_][transitions[k_][1]]
                    trans_mat = self.transition_matrices[k_]
                    likelihood += np.log(trans_mat[trans_idx1, trans_idx0]) * freq
            except KeyError as e:
                msg = ("The path segment '({})' has not been observed and therefore the "
                       "likelihood cannot be computed.").format(e.args[0])
                raise PathpyNotImplemented(msg)

        if log:
            return likelihood
        else:
            return np.exp(likelihood)

    def degrees_of_freedom(self, max_order=None, assumption="paths"):
        """
        Calculates the degrees of freedom of the model based on
        different assumptions, and taking into account layers up to
        a maximum order.

        Parameters
        ----------
        max_order: int
            the maximum order up to which model layers shall be taken into account
        assumption: str
            if set to 'paths', for the degree of freedom calculation only paths in the
            first-order network topology will be considered. This is needed whenever we
            model paths in a *given* network topology. If set to 'ngrams' all possible
            n-grams will be considered, independent of whether they are valid paths in the
            first-order network or not. The 'ngrams' and the 'paths' assumption coincide
            if the first-order network is fully connected, i.e. if all possible paths
            actually occur.

        Returns
        -------

        """
        if max_order is None:
            max_order = self.max_order
        assert max_order <= self.max_order, \
            'Error: max_order cannot be larger than maximum order of multi-order network'

        dof = 0

        # Sum degrees of freedom of all model layers up to max_order
        for i in range(0, max_order + 1):
            dof += self.layers[i].degrees_of_freedom(assumption)

        return int(dof)

    def model_size(self, max_order):
        """
        Returns the total number of non-zero
        transition matrix entries in all
        model layers
        """
        max_order = self.max_order if max_order is None else max_order
        assert max_order <= self.max_order, \
            'Error: max_order cannot be larger than maximum order of multi-order network'

        size = 0
        for i in range(0, max_order + 1):
            size += self.layers[i].model_size()
        return int(size)

    def likelihood_ratio_test(self, paths=None, max_order_null=0, max_order=1,
                              assumption='paths', significance_threshold=0.01):
        """
        Performs a likelihood-ratio test between two multi-order models with given
        maximum orders, where maxOrderNull serves as null hypothesis and max_order
        serves as alternative hypothesis. The null hypothesis is rejected if the
        p-value for the observed paths under the null hypothesis is smaller than the
        given significance threshold.

        Applying this test makes the assumption that we have nested models, i.e. that
        the null model is contained as a special case in the parameter space of the
        more complex model. If we assume that the path constraint holds, this is not
        true for the test of the first- against the zero-order model (since some
        sequences of the zero order model cannot be generated in the first-order
        model). However, since the set of possible higher-order transitions is
        generated based on the first-order model, the nestedness property holds for all
        higher order models.

        Parameters
        ----------
        paths:
            the path data to be used in the likelihood ratio test
        max_order_null:
            maximum order of the multi-order model to be used as a null hypothesis
        max_order: int
            maximum order of the multi-order model to be used as alternative hypothesis
        assumption: str
            paths or ngrams
        significance_threshold: float
            the threshold for the p-value below which to accept the alternative hypothesis

        Returns
        -------
        tuple
            a tuple of the format (reject, p) which captures whether or not the null
            hypothesis is rejected in favor of the alternative hypothesis,
            as well as the p-value that led to the decision
        """
        assert max_order_null < max_order, \
            'Error: order of null hypothesis must be smaller than order of ' \
            'alternative hypothesis'
        # let L0 be the likelihood for the null model and L1 be the likelihood for the
        # alternative model

        # we first compute a test statistic x = -2 * log (L0/L1) = -2 * (log L0 - log L1)
        x = -2 * (self.likelihood(paths, max_order=max_order_null, log=True) -
                  self.likelihood(paths, max_order=max_order, log=True))

        # we calculate the additional degrees of freedom in the alternative model
        dof_diff = (
            self.degrees_of_freedom(max_order=max_order, assumption=assumption) -
            self.degrees_of_freedom(max_order=max_order_null, assumption=assumption)
        )

        Log.add('Likelihood ratio test for K_opt = ' + str(max_order) + ', x = ' + str(x))
        Log.add('Likelihood ratio test, d_1-d_0 = ' + str(dof_diff))

        # if the p-value is *below* the significance threshold, we reject the null
        # hypothesis
        p = 1 - chi2.cdf(x, dof_diff)

        Log.add('Likelihood ratio test, p = ' + str(p))
        return (p < significance_threshold), p

    def estimate_order(self, paths=None, stop_at_order=None, significance_threshold=0.01):
        """Selects the optimal maximum order of a multi-order network model for the
        observed paths, based on a likelihood ratio test with p-value threshold of p

        By default, all orders up to the maximum order of the multi-order model will be
        tested.

        Parameters
        ----------
        paths: Paths
             The path statistics for which to perform the order selection.
             It defaults to the path statistics the MultiOrderModel was created from.
        stop_at_order: int
            The maximum order up to which the multi-order model shall be tested. By 
            default (None), the test will be performed up to the max_order of the 
            MultiOrderModel instance. If the order up to which the test shall be done 
            is larger than the max_order of this model, additional model layers will 
            automatically be created. 
            Default is None.            
        significance_threshold: float
            the threshold for the p-value below which to accept the alternative hypothesis

        Returns
        -------
        int
        """
        
        if stop_at_order is None:
            stop_at_order = self.max_order
        else:
            assert stop_at_order > 1, 'Order to be tested must be larger than one'

        # Test for highest order that passes, likelihood ratio test against null model
        max_accepted_order = 1
        for k in range(2, stop_at_order + 1):

            if k >= self.max_order:
                try:
                    self.add_layers(k)
                except PathsTooShort:
                    msg = ("Optimal order is at least %d, but could be higher. Paths too short"
                           "to create higher orders layers." % max_accepted_order)
                    Log.add(msg, Severity.WARNING)
                    break

            accept, p_value = self.likelihood_ratio_test(
                paths, max_order_null=k - 1, max_order=k,
                significance_threshold=significance_threshold
            )
            if accept:
                max_accepted_order = k
        if paths is None:
            max_len = max(self.paths.paths)
        else:
            max_len = max(paths.paths)
        if stop_at_order == max_accepted_order and max_len>stop_at_order:
            msg = ("Optimal order is at least %d, but may be higher."
                   "Try to increase `stop_at_order`" % stop_at_order)
            Log.add(msg, Severity.WARNING)
        return max_accepted_order


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
        from pathpy.utils.exceptions import PathpyError
        assert method in ['AIC', 'BIC', 'AICc'], \
            'Expected method AIC, AICc or BIC "%s" given.' % method

        # count number of omitted paths with length zero
        p_sum = 0
        for p in paths.paths[0]:
            p_sum += paths.paths[0][p][1]
        if p_sum > 0:
            msg = 'Omitting {} zero-length paths ' \
                  'for test of network assumption'.format(p_sum)
            Log.add(msg, Severity.INFO)

        # log-likelihood and observation count of zero-order model
        likelihood_0, n_0 = self.layer_likelihood(paths, l=0, consider_longer_paths=True,
                                                  log=True, min_path_length=1)

        # log-likelihood and observation count of first-order model
        likelihood_1, n_1 = self.layer_likelihood(paths, l=1, consider_longer_paths=True,
                                                  log=True, min_path_length=1)

        # By definition, the number of observations for both models should be the total
        # weighted degree of the first-order network
        if n_0 != n_1:
            raise PathpyError(
                'Observation count for 0-order ({n0}) and '
                '1-st order model ({n1}) do not match'.format(n0=n_0, n1=n_1)
            )

        # degrees of freedom = |V|-1
        dof0 = self.layers[0].degrees_of_freedom(assumption='ngrams')

        # degrees of freedom based on network assumption
        dof1 = self.layers[1].degrees_of_freedom(assumption='paths')

        Log.add('Log-Likelihood (k=0) = ' + str(likelihood_0), Severity.INFO)
        Log.add('Degrees of freedom (k=0) = ' + str(dof0), Severity.INFO)

        Log.add('Log-Likelihood (k=1) = ' + str(likelihood_1), Severity.INFO)
        Log.add('Degrees of freedom (k=1) = ' + str(dof0 + dof1), Severity.INFO)

        if method == 'AIC':
            ic0 = 2 * dof0 - 2 * likelihood_0
            ic1 = 2 * (dof0 + dof1) - 2 * likelihood_1
        elif method == 'AICc':
            dof10 = dof0 + dof1
            assert n_1 > dof10 - 2, \
                'Error: number of samples too small for model complexity'
            dof10 = dof0 + dof1
            ic0 = 2 * dof0 - 2 * likelihood_0 + (2 * (dof0 + 1) * (dof0 + 2)) / (n_0 - dof0 - 2)
            ic1 = 2 * dof10 - 2 * likelihood_1 + (2 * (dof10+1) * (dof10 + 2)) / (n_1 - dof10 - 2)
        elif method == 'BIC':
            ic0 = np.log(n_0) * dof0 - 2 * likelihood_0
            ic1 = np.log(n_1) * (dof0 + dof1) - 2 * likelihood_1
        else:
            raise PathpyError("Method check has not filtered out illegal "
                                  "method %s " % method)

        # if the AIC/AICc/BIC of the zero-order model is larger than that of the
        # first-order model, we do not reject the network hypothesis
        return ic0 > ic1, ic0, ic1
