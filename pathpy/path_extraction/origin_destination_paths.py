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

from pathpy import Paths
from pathpy.utils import Log
from pathpy.algorithms.shortest_paths import shortest_paths

import numpy as np


__all__ = ["paths_from_origin_destination", "read_origin_destination"]


def paths_from_origin_destination(origin_destination_list, network,
                                  distribute_weight=True):
    """extracts shortest path statistics based on origin/destination data.
    Such data capture the statistics of the origin (i.e. the start node) and destination
    (i.e. the target) node of itineraries in a given network.

    Common examples include passenger origin and destination statistics in transportation
    networks. The methods in this class can be used to read  origin/destination data from
    a file and generate path statistics based on the assumption that all paths from an
    origin and a destination follow the shortest path in the network.

    Extracts pathway statistics by calculating shortest paths between all origin and
    destination pairs in a given network.

    Parameters
    ----------
    origin_destination_list: list
        a list of tuples (o, d, w) containing the origin (o), destination (d),
        and float weight (w) of paths.
    network:
        the network topology for which shortest paths will be calculated. Names of nodes
        in the network must match the node names used in the origin destination list.
    distribute_weight: bool


    Returns
    -------
    paths
    """
    assert network is not None, \
        'Error: extraction of origin destination paths requires a network topology'

    all_paths = shortest_paths(network)

    paths = Paths()

    # OD is a list of tuples of the form (origin_node, destination_node, weight)
    # that indicates that the shortest path from origin_node to destination_node was
    # observed weight times
    Log.add('Starting origin destination path calculation ...')
    for (o, d, w) in origin_destination_list:
        assert o in network.nodes, 'Error: could not find node ' + str(o) + ' in network'
        assert d in network.nodes, 'Error: could not find node ' + str(d) + ' in network'
        sp = list(all_paths[o][d])
        num_paths = len(sp)
        if distribute_weight and num_paths > 1:
            # to avoid introducing false correlations that are not justified by the
            # available data, the (integer) weight of an origin destination pair can be
            # distributed among all possible shortest paths between a pair of nodes,
            # while constraining the weight of shortest paths to integers.
            for i in range(int(w)):
                paths.add_path_tuple(sp[i % num_paths], frequency=(0, 1))
        else:
            # in this case, the full weight of an origin destination path will be
            # assigned to a random single shortest path in the network
            paths.add_path_tuple(sp[np.random.randint(num_paths)], frequency=(0, w))
    Log.add('finished.')
    return paths


def read_origin_destination(filename, separator=','):
    """Helper function to read origin/destination statistics from a csv file.
    The file is assumed to have the following structure:

    origin1,destination1,weight
    origin2,destination2,weight
    origin3,destination3,weight

    Parameters
    ----------
    filename: str
        the path to the file from which to reach origin/destination statistics
    separator: str
        arbitrary separation character (default: ',')

    Returns
    -------
    list
    """
    origin_destination_list = []
    Log.add('Reading origin/destination statistics from file ...')

    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            fields = line.rstrip().split(separator)
            origin_destination_list.append((fields[0].strip(), fields[1].strip(), float(fields[2].strip())))
            line = f.readline()
    Log.add('Finished.')

    return origin_destination_list

