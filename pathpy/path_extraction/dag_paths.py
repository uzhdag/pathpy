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
import sys
import itertools as it
import functools as ft
from collections import defaultdict

from pathpy.classes.paths import Paths
from pathpy.utils import Log, Severity
from pathpy import DAG


def remove_repetitions(path):
    """
    Remove repeated nodes in the path

    Parameters
    ----------
    path

    Returns
    -------

    Examples
    -------
    >>> remove_repetitions((1, 2, 2, 3, 4, 1))
    (1, 2, 3, 4, 1)
    >>> remove_repetitions((1, 2, 2, 2, 3)) == remove_repetitions((1, 2, 2, 3, 3))
    True
    """
    return tuple(p[0] for p in it.groupby(path))


def paths_from_dag(dag, node_mapping=None, max_subpath_length=None, separator=',', repetitions=True, unique=False):
    """
    Calculates path statistics in a directed acyclic graph.
    All paths between all roots (nodes with zero indegree)
    and all leafs (nodes with zero outdegree) are generated.

    Parameters
    ----------
    dag: DAG
        the directed acyclic graph instance for which paths are calculated
    node_mapping: dict
        can be a simple mapping (1-to-1) or a 1-to-many (a dict with sets as values)
    max_subpath_length: int
        This can be used to limit the calculation of sub path statistics to a given
        maximum length. This is useful, as the statistics of sub paths of length k
        are only needed to fit a higher-order model with order k. Hence, if we know
        that the model selection is limited to a given maximum order K, we can safely
        set the maximum sub path length to K. By default, sub paths of any length
        will be calculated. Note that, independent of the sub path calculation
        longest path of any length will be considered in the likelihood calculation!
    separator: str
        separator to use to separate nodes in the generated Paths object. Default is ','.
    repetitions: bool
        whether or not to remove repeated nodes in the paths. Such repeated paths can occur
        if a non-injective node_mapping is applied. If set to True, a path a,a,b,b,c,c,d is
        returned as a,b,c,d.
    unique: bool
        whether or not multiple identical mapped paths should be counted separately. For
        DAG representations of temporal networks with delta > 1, where nodes are temporal copies,
        we do not want to count multiple paths from the same root that pass through different
        temporal copies of the same physical node. For instance with delta=2, time-stamped edges
        (a,b;1), (b,c;3) are transformed into a DAG a1->b2, a1->b3, b3->c4. With the mapping to
        physical nodes we would find two different paths a->b->c of length two, which only differ
        in terms of WHEN they arrive in node c


    Returns
    -------
    Paths

    """
    # Try to topologically sort the graph if not already sorted
    if node_mapping:
        test_key = list(node_mapping.keys())[0]
        ONE_TO_MANY = isinstance(node_mapping[test_key], set)
    else:
        ONE_TO_MANY = False

    if dag.is_acyclic is None:
        dag.topsort()
    # issue error if graph contains cycles
    if dag.is_acyclic is False:
        Log.add('Cannot extract statistics from a cyclic graph', Severity.ERROR)
        raise ValueError('Cannot extract path statistics from a cyclic graph')
    else:
        # path object which will hold the detected (projected) paths
        p = Paths(separator=separator)
        if max_subpath_length:
            p.max_subpath_length = max_subpath_length
        else:
            p.max_subpath_length = sys.maxsize

        Log.add('Creating paths from directed acyclic graph', Severity.INFO)

        # construct all paths originating from root nodes for 1 to 1
        if not ONE_TO_MANY:
            for s in dag.roots:
                extracted_paths = dag.routes_from_node(s, node_mapping)
                if unique:
                    extracted_paths = set(tuple(x) for x in extracted_paths)
                for path in extracted_paths:   # add detected paths to paths object                    
                    if repetitions:
                        p.add_path(path, expand_subpaths=False, frequency=(0, 1))
                    else:
                        p.add_path(remove_repetitions(path), expand_subpaths=False, frequency=(0, 1))
        else:
            path_counter = defaultdict(lambda: 0)
            for root in dag.roots:
                for set_path in dag.routes_from_node(root, node_mapping):
                    for blown_up_path in expand_set_paths(set_path):
                        path_counter[blown_up_path] += 1

            for path, count in path_counter.items():
                if repetitions:
                    p.add_path(path, expand_subpaths=False, frequency=(0, count))
                else:
                    p.add_path(remove_repetitions(path), expand_subpaths=False, frequency=(0, count))

        Log.add('Expanding Subpaths', Severity.INFO)
        p.expand_subpaths()
        Log.add('finished.', Severity.INFO)
        return p


def expand_set_paths(set_path):
    """returns all possible paths which are consistent with the sequence of sets

    Parameters
    ----------
    set_path: list
        a list of sets or other iterable

    Examples
    -------
    >>> node_path = [{1, 2}, {2, 5}, {1, 2}]
    >>> list(expand_set_paths(node_path))
    [(1, 2, 1), (2, 2, 1), (1, 5, 1), (2, 5, 1), (1, 2, 2), (2, 2, 2), (1, 5, 2), (2, 5, 2)]
    >>> node_path = [{1, 2}, {5}, {2, 5}]
    >>> list(expand_set_paths(node_path))
    [(1, 5, 2), (2, 5, 2), (1, 5, 5), (2, 5, 5)]


    Yields
    ------
    tuple
        a possible path
    """
    # how many possible combinations are there
    node_sizes = [len(n) for n in set_path]
    num_possibilities = ft.reduce(lambda x, y: x * y, node_sizes, 1)

    # create a list of lists such that each iterator is repeated the number of times
    # his predecessors have completed their cycle
    all_periodics = []
    current_length = 1
    for node_set in set_path:
        periodic_num = []
        for num in node_set:
            periodic_num.extend([num] * current_length)
        current_length *= len(node_set)
        all_periodics.append(periodic_num)

    iterator = [it.cycle(periodic) for periodic in all_periodics]
    for i, elements in enumerate(zip(*iterator)):
        if i >= num_possibilities:
            break
        yield elements
