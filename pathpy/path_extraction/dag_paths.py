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

import sys

from pathpy.classes.paths import Paths
from pathpy.utils import Log, Severity
from pathpy import DAG


def paths_from_dag(dag, node_mapping=None, max_subpath_length=sys.maxsize, separator=','):
    """    Extracts pathway statistics from a directed acyclic graph.
    For this, all paths between all roots (zero incoming links)
    and all leafs (zero outgoing link) will be constructed.

    Parameters
    ----------
    dag: DAG
    node_mapping
    max_subpath_length

    Returns
    -------

    """
    # Try to topologically sort the graph if not already sorted
    if dag.is_acyclic is None:
        dag.topsort()
    # issue error if graph contains cycles
    if dag.is_acyclic is False:
        Log.add('Cannot extract statistics from a cyclic graph', Severity.ERROR)
        raise ValueError('Cannot extract path statistics from a cyclic graph')
    else:
        # path object which will hold the detected (projected) paths
        p = Paths(separator=separator)
        p.max_subpath_length = max_subpath_length
        Log.add('Creating paths from directed acyclic graph', Severity.INFO)
        n = 0

        # construct all paths originating from root nodes
        for s in dag.roots:
            if n % 100 == 0:
                msg = 'Processed {} / {} root nodes'.format(n, len(dag.roots))
                Log.add(msg, Severity.TIMING)
            if node_mapping is None:
                paths = dag.construct_paths(s)
                # add detected paths to paths object
                for d in paths:
                    for x in paths[d]:
                        p.add_path_tuple(x, expand_subpaths=False, frequency=(0, 1))
            else:
                dag.construct_mapped_paths(s, node_mapping, p)
            n += 1
        p.expand_subpaths()
        Log.add('finished.', Severity.INFO)
        return p
