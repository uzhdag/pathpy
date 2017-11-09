# -*- coding: utf-8 -*-
"""
    pathpy is an OpenSource python package for the analysis of sequential data on
    pathways and temporal networks using higher- and multi order graphical models

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
import pathpy.Paths as _Paths
import numpy as _np
from pathpy.Log import Log as _Log
from pathpy.Log import Severity as _Severity


class DAGPaths:
    """
    This class can be used to calculate time-respecting paths in directed acyclic graphs
    """

    @staticmethod
    def extract(dag, node_mapping=None, maxSubPathLength=_sys.maxsize):
        """
        Extracts pathway statistics from a directed acyclic graph.
        For this, all paths between all roots (zero incoming links)
        and all leafs (zero outgoing link) will be constructed.
        """

        # Try to topo_logically sort the graph if not already sorted
        if dag.isAcyclic is None:
            dag.topsort()
        # issue error if graph contains cycles
        if dag.isAcyclic is False:
            _Log.add('Cannot extract path statistics from a cyclic graph', _Severity.ERROR)
            raise ValueError('Cannot extract path statistics from a cyclic graph')
        else:
            # path object which will hold the detected (projected) paths
            p = _Paths()
            p.maxSubPathLength = maxSubPathLength
            _Log.add('Creating paths from directed acyclic graph', _Severity.INFO)
            n = 0

            # construct all paths originating from root nodes
            for s in dag.roots:
                if n % 100 == 0:
                    _Log.add('Processed ' + str(n) + '/' + str(len(dag.roots)) + ' root nodes', _Severity.TIMING)
                if node_mapping is None:
                    paths = dag.constructPaths(s)
                    # add detected paths to paths object
                    for d in paths:
                        for x in paths[d]:
                            p.addPathTuple(x, expandSubPaths=False, frequency=(0, 1))
                else:
                    dag.constructMappedPaths(s, node_mapping, p)
                n += 1
            p.expandSubPaths()
            _Log.add('finished.', _Severity.INFO)
            return p