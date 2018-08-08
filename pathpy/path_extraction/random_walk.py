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

import collections as _co
import random

import numpy as _np

from pathpy.classes.network import Network
from pathpy.classes.paths import Paths


def random_walk(network, l, n=1):
    """
    Generates n paths of a random walker in the given network
    and returns them as a paths object. 
    Each path has a length of l steps   
    Parameters
    ----------
    network: pathpy.Network
    int: l
    int: n
    """

    p = Paths()
    T = network.transition_matrix().todense().transpose()
    idx_map = network.node_to_name_map()
    nodes = _np.array([v for v in network.nodes])

    for i in range(n):
        # choose random start node
        path = (random.choice(nodes),)
        for j in range(l):
            # get transition probability vector T[idx ->  . ]
            prob = _np.array(T[idx_map[path[-1]],:])[0,:]
            nz = prob.nonzero()[0]
            # make one random transition
            if nz.shape[0]>0:
                next_node = _np.random.choice(a=nodes[nz], p=prob[nz])
                # add node to path
                path = path + (next_node,)
            else:
                print('no neighbor')
                break
        p.add_path_tuple(path)
    return p

