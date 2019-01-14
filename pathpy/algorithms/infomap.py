"""
Implementation of the flow compression algorithm InfoMap
"""

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
from functools import singledispatch
from collections import defaultdict

import numpy as _np

from pathpy import Network, Paths, HigherOrderNetwork, MultiOrderModel
from pathpy.utils import Log, Severity
from pathpy.utils import PathpyError, PathpyNotImplemented

__all__ = ['modular_description_length', 'find_communities_sa']


@singledispatch
def modular_description_length(network, module_map):
    """
    Calculates the modular description length L(M) of random walks 
    in an undirected network, using the mapping of nodes
    to modules given in the dictionary M. The calculation 
    is based on the MapEquation proposed in:

    M. Rosvall and C. T. Bergstrom: Maps of random walks on complex
    networks reveal community structure, PNAS, Vol. 105, No. 4, Jan. 2008
    """
    assert isinstance(network, Network), "argument must be an instance of pathpy.Network"
    assert not network.directed, "this function is currently only implemented for undirected and unweighted networks"

    module_to_nodes = defaultdict(list)

    for v in network.nodes:
        module_to_nodes[module_map[v]].append(v)

    # count edges within and across modules
    across_count = 0
    total = 0
    total_within = 0
    total_exit = 0
    within_count = defaultdict(lambda: 0)
    module_count = defaultdict(lambda: 0)
    exit_count = defaultdict(lambda: 0)
    exit_count_node = defaultdict(lambda: 0)
    node_count = defaultdict(lambda: 0)
    for (v,w) in network.edges:
        total += 1
        module_count[module_map[v]] += 1
        module_count[module_map[w]] += 1
        if module_map[v] != module_map[w]:
            across_count += 1
            exit_count[module_map[v]] += 1
            exit_count_node[v] += 1
            exit_count[module_map[w]] += 1
            exit_count_node[w] += 1
            total_exit += 2
        else:
            node_count[v] += 1
            node_count[w] += 1
            within_count[module_map[v]] += 1
            total_within += 1

    Log.add('total_within = {0}'.format(total_within), Severity.DEBUG)
    Log.add('across_count = {0}'.format(across_count), Severity.DEBUG)
    Log.add('total = {0}'.format(total), Severity.DEBUG)

    # contribution of module changes to code length
    q = across_count/total
    Log.add('q = {0}'.format(q), Severity.DEBUG)
    H_Q = 0
    for m in module_count:
        x = module_count[m] / (2*total)
        H_Q += x * _np.log2(x)
    Log.add('H_Q = {0}'.format(-H_Q), Severity.DEBUG)

    # contribution of node changes to code length
    S = 0
    for m in module_to_nodes:
        # problem if there are no links within clusters!
        p_i = (within_count[m]+exit_count[m])/ (total_within + total_exit)
        Log.add('p_i = {0}'.format(p_i), Severity.DEBUG)
        H_Pi = 0
        for n in module_to_nodes[m]:
            x = (node_count[n] + exit_count_node[n]) / (2*within_count[m]+exit_count[m])
            H_Pi += x * _np.log2(x)
        Log.add('H_Pi = {0}'.format(-H_Pi), Severity.DEBUG)
        S += - p_i * H_Pi

    return - q * H_Q + S


@modular_description_length.register(Paths)
def _mdl_paths(paths, module_map):
    """
    Calculates the modular description length L(M) of paths in the 
    given paths objects, using the mapping of nodes
    to modules given in the dictionary M. The calculation 
    is based on an adapted version of the MapEquation proposed in:

    M. Rosvall and C. T. Bergstrom: Maps of random walks on complex
    networks reveal community structure, PNAS, Vol. 105, No. 4, Jan. 2008
    """
    assert isinstance(paths, Paths), "argument must be an instance of pathpy.Paths"

    # create reverse mapping from modules to nodes
    module_to_nodes = defaultdict(list)
    for node in module_map:
        module_to_nodes[module_map[node]].append(node)

    module_labels = set(module_map.values())

    # number of all transitions (in all paths)
    transitions = 0.0

    # transitions within a module, for all modules
    within_prob = defaultdict(lambda: 0.0)

    # transitions that exit a module, for all modules
    exit_prob = defaultdict(lambda: 0.0)

    # transitions through all nodes
    node_prob = defaultdict(lambda: 0.0)

    for l in paths.paths:
        for p in paths.paths[l]:
            # only consider paths that occur as longest path
            if paths.paths[l][p][1]>0:
                # look up module of first node and set as current_module
                current_module = module_map[p[0]]
                # for each transition
                for x in p[1:]:
                    # count transitions (total and through node x)
                    transitions += paths.paths[l][p][1]
                    node_prob[x] += paths.paths[l][p][1]
                    # count transition within module 
                    if module_map[x] == current_module:
                        within_prob[current_module] += paths.paths[l][p][1]
                    else: # or count exit transition
                        exit_prob[current_module] += paths.paths[l][p][1]                        
                        # and update current module
                        current_module = module_map[x]
                exit_prob[current_module] += paths.paths[l][p][1]
                        
    # STEP 1: Calculate contribution of transitions BETWEEN modules ...

    for i in module_labels:
        # exit_prob corresponds to q_i->, i.e. per step probability of 
        # transitions in i to exit module i
        exit_prob[i] = exit_prob[i] / transitions
        within_prob[i] = within_prob[i] / transitions

    # Eq. (2) in SI
    q = 0.0
    for i in module_labels:
        q += exit_prob[i]

    # Eq. (3) in SI
    H_Q = 0.0
    for i in module_labels:
        H_Q += exit_prob[i] * _np.log2(exit_prob[i])

    Log.add('q = {0}'.format(q), severity=Severity.DEBUG)
    Log.add('H_Q = {0}'.format(-H_Q), severity=Severity.DEBUG)
    Log.add('q * H(Q) = {0}'.format(-q*H_Q), severity=Severity.DEBUG)

    # STEP 2: Calculate the sum (S) that accounts for the contribution 
    # of transitions WITHIN modules
    S = 0.0
    
    # we can directly calculate node transitions p_alpha
    # from the paths
    for n in paths.nodes:
        node_prob[n] = node_prob[n] / transitions

    # sum entropies for within-module transitions across modules
    for m in module_labels:
        # Eq. (5a)
        A = 0.0
        for v in module_to_nodes[m]:
            A += node_prob[v]
        B = exit_prob[m] / (exit_prob[m] + A)
        H_Pi = B * _np.log(B)

        # Eq. (5b)
        C = 0.0
        for v in module_to_nodes[m]:
            D = 0.0
            for w in module_to_nodes[m]:
                D += node_prob[w]
            E = node_prob[v] / (exit_prob[m] + D)
            C += E * _np.log2(E)        
        H_Pi = H_Pi + C

        # p^i<-> from Eq. (4)
        F = exit_prob[m]
        for v in module_to_nodes[m]:
            F += node_prob[v]

        # Sum in Eq. (1)
        S += F * H_Pi

    return -(q * H_Q + S)


def find_communities_sa(obj, iterations=100, initial_map=None, T=0.1, t_i=1):
    """
    A simple simulated annealing algorithm to optimize the modular description length 
    calculated either based on a Paths or Network object
    """
    assert isinstance(obj, Paths) or isinstance(obj, Network), 'Can only find communities based on paths or undirected network'
    nodes = [v for v in obj.nodes]

    if initial_map != None:
        module_map = initial_map
    else:
        module_map = {}
        i = 0
        for n in nodes:
            module_map[n] = str(i)            
            i += 1

    mdl = modular_description_length(obj, module_map)

    if mdl == _np.nan:
        raise PathpyError('Could not calculate initial MDL')

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for i in range(iterations):
            # pick a random pair of nodes
            c = _np.random.choice(_np.array(nodes), 2, replace=False)
            n1 = c[0]
            n2 = c[1]

            # merge nodes into same module
            old_module = module_map[n1]
            module_map[n1] = module_map[n2]

            mdl_new = modular_description_length(obj, module_map)
            # accept change if it decreases MDL or based on temperature
            if mdl_new != _np.nan and (mdl_new <= mdl or _np.random.ranf() < _np.exp(-(mdl_new-mdl)/T)):
                mdl = mdl_new
            else: # revert change
                module_map[n1] = old_module

            # cooling schedule
            if i % t_i == 0:
                T = T/1.1

    return module_map
