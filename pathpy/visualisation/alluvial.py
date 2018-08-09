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
import json
import os
from string import Template

import string
import random

import collections as _co

from pathpy.classes.higher_order_network import HigherOrderNetwork
from pathpy.classes.paths import Paths
from pathpy.classes.network import Network

import numpy as _np

def generate_memory_net(paths, node, self_loops=True):
    """
    Helper class that generates a directed and weighted
    memory network where weights capture path statistics.
    """
    n = Network(directed=True)

    # consider all (sub-)paths of length two 
    # through the focal node
    for p in paths.paths[2]:
        if p[1] == node:
            if self_loops or (p[0] != node and p[2] != node):
                src = 'src_{0}'.format(p[0])
                tgt = 'tgt_{0}'.format(p[2])
                mem = 'mem_{0}_{1}'.format(p[0], p[1])
                # calculate frequency of sub-paths src->focal_node->*, i.e. paths that 
                # continue through the focal_node
                # w_1 = 0
                # for x in paths.nodes:
                #         ct = p[:2]+(x,)
                #         if ct in paths.paths[2] and x != focal_node:
                #             w_1 += paths.paths[2][ct].sum()

                # calculate frequency of (sub-)path src -> focal_node -> tgt
                w_2 = paths.paths[2][p].sum()
                n.add_edge(src, mem, weight=1)
                n.add_edge(mem, tgt, weight=w_2)


    # adjust weights of links to memory nodes:
    for m in n.nodes:
        if m.startswith('mem'):
            for u in n.predecessors[m]:
                n.edges[(u,m)]['weight'] = n.nodes[m]['outweight']
                n.nodes[m]['inweight'] = n.nodes[m]['outweight']
    return n


def generate_memory_net_markov(network, focal_node, self_loops=True):
    """
    Generates a directed and weighted network with flow values based
    on a network and an assumption of Markov flows.
    """
    n = Network(directed=True)

    out_weight = _np.sum(network.nodes[focal_node]['outweight'])

    for u in network.predecessors[focal_node]:
        for w in network.successors[focal_node]:
            if self_loops or (u!= focal_node and w != focal_node):
                src = 'src_{0}'.format(u)
                tgt = 'tgt_{0}'.format(w)
                mem = 'mem_{0}_{1}'.format(u, focal_node)

                w_1 = _np.sum(network.edges[(u, focal_node)]['weight'])

                # at random, we expect the flow to be proportional to the relative edge weight
                w_2 = w_1 * (_np.sum(network.edges[(focal_node, w)]['weight'])/out_weight)
                n.add_edge(src, mem, weight=w_1)
                n.add_edge(mem, tgt, weight=w_2)
    return n


def generate_diffusion_net(paths, node=None, markov=True, steps=5):
    """
    """
    g1 = HigherOrderNetwork(paths, k=1)
    map_1 = g1.node_to_name_map()

    prob = _np.zeros(g1.ncount())
    prob = prob.transpose()
    if node is None:
        node = g1.nodes[0]
    
    prob[map_1[node]] = 1.0
    
    T = g1.transition_matrix()

    flow_net = Network(directed=True)

    if markov:
        # if markov == True flows are given by first-order transition matrix
        for t in range(1, steps+1):
            # calculate flow from i to j in step t
            for i in g1.nodes:
                for j in g1.nodes:                    
                    i_to_j = prob[map_1[i]] * T[map_1[j], map_1[i]]
                    if i_to_j > 0:
                        flow_net.add_edge('{0}_{1}'.format(i, t-1), '{0}_{1}'.format(j, t), weight = i_to_j)
            prob = T.dot(prob)
    else:
        # if markov == False calculate flows based on paths starting in initial_node
        for p in paths.paths[steps]:
            if p[0] == node:
                for t in range(len(p)-1):
                    flow_net.add_edge('{0}_{1}'.format(p[t], t), '{0}_{1}'.format(p[t+1], t+1), weight = paths.paths[steps][p].sum())
        
        # normalize flows and balance in- and out-weight for all nodes
        # normalization = flow_net.nodes['{0}_{1}'.format(initial_node, 0)]['outweight']

        flow_net.nodes[node+'_0']['inweight'] = 1.0
        Q = [node+'_0']
        # adjust weights using BFS
        while Q:
            v = Q.pop()
            # print(v)
            inweight = flow_net.nodes[v]['inweight']
            outweight = flow_net.nodes[v]['outweight']

            for w in flow_net.successors[v]:
                flow_net.nodes[w]['inweight'] = flow_net.nodes[w]['inweight'] - flow_net.edges[(v,w)]['weight']
                flow_net.edges[(v,w)]['weight'] = (inweight/outweight) * flow_net.edges[(v,w)]['weight']
                flow_net.nodes[w]['inweight'] =  flow_net.nodes[w]['inweight'] + flow_net.edges[(v,w)]['weight']
                Q.append(w)
    return flow_net
