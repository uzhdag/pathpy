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
import itertools as it
import functools as ft
from collections import defaultdict

from pathpy.classes.paths import Paths
from pathpy.utils import Log, Severity
from pathpy import DAG


def remove_repetitions(path):
    """Remove repeated nodes in the path

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


def paths_from_dag(dag, node_mapping=None, max_subpath_length=None, separator=',', repetitions=True):
    """    Extracts pathway statistics from a directed acyclic graph.
    For this, all paths between all roots (zero incoming links)
    and all leafs (zero outgoing link) will be constructed.

    Parameters
    ----------
    dag: DAG
    node_mapping: dict
        can be a simple mapping (1-to-1) or a 1-to-many (a dict with sets as values)
    max_subpath_length: int
    separator: str
        separator to use to separate nodes on paths

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
                for path in extracted_paths:   # add detected paths to paths object
                    if repetitions:
                        p.add_path_tuple(path, expand_subpaths=False, frequency=(0, 1))
                    else:
                        p.add_path_tuple(remove_repetitions(path), expand_subpaths=False, frequency=(0, 1))
        else:
            path_counter = defaultdict(lambda: 0)
            for root in dag.roots:
                for set_path in dag.routes_from_node(root, node_mapping):
                    for blown_up_path in expand_set_paths(set_path):
                        path_counter[blown_up_path] += 1

            for path, count in path_counter.items():
                if repetitions:
                    p.add_path_tuple(path, expand_subpaths=False, frequency=(0, count))
                else:
                    p.add_path_tuple(remove_repetitions(path), expand_subpaths=False, frequency=(0, count))

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
