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

import sys as _sys

import pathpy.Paths as _Paths
from pathpy.Log import Log as _Log
from pathpy.Log import Severity as _Severity


class OriginDestinationPaths:
    """
    This class can be used to extract shortest path statistics based on statistics of shortest path origin and destinations in a given network topology.
    """

    @staticmethod
    def extract(OD, network):
        """
        Extracts pathway statistics by calculating the shortest paths between origin and destination pairs in a given network. 
        """

        assert network is not None, 'Error: extraction of origin destination paths requires a network topology'

        shortest_paths = network.getShortestPaths()

        paths = _Paths()

        # OD is a list of tuples of the form (origin_node, destination_node, weight)
        # that indicates that the shortest path from origin_node to destination_node was 
        # observed weight times
        for (o, d, w) in OD:
            # for new we assume that each of the multiple shortest paths occurs w times
            assert o in network.nodes and d in network.nodes, 'Error: could not find origin or destination node in network'
            for p in shortest_paths[o][d]: 
                paths.addPathTuple(p, frequency=(0,w))
        
        return paths


    def readFile(filename, separator=','):
        """
        Helper function to read origin destination statistics from a csv file
        """

        OD = []
        _Log.add('Reading origin destination data from file ... ')

        with open(filename, 'r') as f:                
            line = f.readline()
            while line:
                fields = line.rstrip().split(separator)
                OD.append((fields[0], fields[1], float(fields[2])))
                line = f.readline()
        _Log.add('Finished.')

        return OD

