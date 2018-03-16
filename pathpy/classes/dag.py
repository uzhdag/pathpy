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

from collections import defaultdict
import sys
from pathpy.utils import Log, Severity


class DAG(object):
    """
        Represents a directed acyclic graph (DAG) which
        can be used to generate pathway statistics.
    """

    def __init__(self, edges=None):
        """
        Constructs a directed acyclic graph from an edge list
        """

        self.nodes = set()
        self.edges = set()

        # Whether or not this graph is acyclic. None indicates that it is unknown
        self.is_acyclic = None

        # list of topologically sorted nodes
        self.sorting = []

        # Set of nodes with no incoming edges
        self.roots = set()

        # Set of nodes with no outgoing edges
        self.leafs = set()

        # The dictionary of successors of each node
        self.successors = defaultdict(set)

        # The dictionary of predecessors of each node
        self.predecessors = defaultdict(set)
        self_loops = 0
        redundant_edges = 0
        if edges is not None:
            for e in edges:
                is_redundant = False
                has_self_loop = False
                if e[0] == e[1]:
                    has_self_loop = True
                    self_loops += 1
                if (e[0], e[1]) in self.edges:
                    is_redundant = True
                    redundant_edges += 1
                if not has_self_loop and not is_redundant:
                    self.add_edge(e[0], e[1])
            if self_loops > 0:
                Log.add('Warning: omitted %d self-loops' % self_loops, Severity.WARNING)
            if redundant_edges > 0:
                Log.add('Warning: omitted %d redundant edges' % redundant_edges,
                        Severity.WARNING)

        # placeholder properties for topological sort
        self.parent = {}
        self.start_time = {}
        self.finish_time = {}
        self.edge_classes = {}
        self.top_sort_count = 0

    def construct_paths(self, v):
        """
        Constructs all paths from node v to any leaf nodes
        """

        # Collect temporary paths, indexed by the target node
        temp_paths = defaultdict(list)
        temp_paths[v] = [(v,)]

        # set of unprocessed nodes
        queue = {v}

        while queue:
            # take one unprocessed node
            x = queue.pop()

            # successors of x expand all temporary
            # paths, currently ending in x
            if self.successors[x]:
                for w in self.successors[x]:
                    for p in temp_paths[x]:
                        temp_paths[w].append(p + (w,))
                    queue.add(w)
                del temp_paths[x]

        return temp_paths

    def construct_mapped_paths(self, v, node_mapping, paths):
        """
        Constructs all paths from node v to any leaf nodes,
        while applying a surjective projection function
        given in terms of a mapping.
        """

        # (mapped) paths that can be continued
        # for a given endpoint node (key)
        continuable = defaultdict(list)
        continuable[v] = [(node_mapping[v],)]

        while continuable:

            # process one node for which path can possibly continued
            x, cp = continuable.popitem()

            if not self.successors[x]:
                # x is a leaf, so any path ending in x are longest paths in the DAG
                for p in cp:
                    paths.add_path_tuple(p, expand_subpaths=False, frequency=(0, 1))

            else:
                # extend all paths to x by successors of x
                for p in cp:
                    for w in self.successors[x]:
                        continuable[w].append(p + (node_mapping[w],))

    def dfs_visit(self, v, parent=None):
        """Recursively visits nodes in the graph, classifying edges as (1) tree, (2)
        forward, (3) back or (4) cross edges.

        Parameters
        ----------
        v:
            node to be visited
        parent:
            the parent of this node (None for nodes) with no parents
        """
        self.parent[v] = parent
        self.top_sort_count += 1
        self.start_time[v] = self.top_sort_count
        if parent:
            self.edge_classes[(parent, v)] = 'tree'

        for w in self.successors[v]:
            if w not in self.parent:
                self.dfs_visit(w, v)
            elif w not in self.finish_time:
                self.edge_classes[(v, w)] = 'back'
                self.is_acyclic = False
            elif self.start_time[v] < self.start_time[w]:
                self.edge_classes[(v, w)] = 'forward'
            else:
                self.edge_classes[(v, w)] = 'cross'
        self.top_sort_count += 1
        self.finish_time[v] = self.top_sort_count
        self.sorting.append(v)

    def topsort(self):
        """
        Performs a topological sorting of the graph, classifying
        all edges as (1) tree, (2) forward, (3) back or (4) cross
        edges in the process.

        see Cormen 2001 for details
        """
        self.sorting = []
        self.parent = {}
        self.start_time = {}
        self.finish_time = {}
        self.edge_classes = {}
        self.top_sort_count = 0
        self.is_acyclic = True
        for v in self.nodes:
            if v not in self.parent:
                self.dfs_visit(v)
        self.sorting.reverse()

    def make_acyclic(self):
        """Removes all back-links from the graph to make it acyclic, then performs another
        topological sorting of the DAG
        """
        if self.is_acyclic is None:
            self.topsort()
        removed_links = 0
        if not self.is_acyclic:
            # Remove all back links
            for e in list(self.edge_classes):
                if self.edge_classes[e] == 'back':
                    self.edges.remove(e)
                    removed_links += 1
                    self.successors[e[0]].remove(e[1])
                    self.predecessors[e[1]].remove(e[0])
                    del self.edge_classes[e]
            self.topsort()
            assert self.is_acyclic, "Error: make_acyclic did not generate acyclic graph!"
            Log.add('Removed ' + str(removed_links) +
                    ' back links to make graph acyclic', Severity.INFO)

    def summary(self):
        """
        Returns a string representation of this directed acyclic graph
        """

        summary = 'Directed Acyclic Graph'
        summary += '\n'
        summary += 'Nodes:\t\t' + str(len(self.nodes)) + '\n'
        summary += 'Roots:\t\t' + str(len(self.roots)) + '\n'
        summary += 'Leaves:\t\t' + str(len(self.leafs)) + '\n'
        summary += 'Links:\t\t' + str(len(self.edges)) + '\n'
        summary += 'Acyclic:\t' + str(self.is_acyclic) + '\n'
        return summary

    def __str__(self):
        """
        Returns the default string representation of this object
        """
        return self.summary()

    def add_edge(self, source, target):
        """
        Adds a directed edge to the graph
        """

        if source not in self.nodes:
            self.nodes.add(source)
            self.roots.add(source)
        if target not in self.nodes:
            self.nodes.add(target)
            self.leafs.add(target)

        self.leafs.discard(source)
        self.roots.discard(target)
        self.edges.add((source, target))
        self.successors[source].add(target)
        self.predecessors[target].add(source)
        self.is_acyclic = None

    def write_file(self, filename, sep=','):
        """Writes a dag as an adjaceny list to file

        Parameters
        ----------
        filename
        sep

        Returns
        -------
        dag
        """
        with open(filename, 'w') as file:
            for edge in self.edges:
                file.write(sep.join(edge)+'\n')

    @classmethod
    def read_file(cls, filename, sep=',', maxlines=None, mapping=None, header=False):
        """
        Reads a directed acyclic graph from a file
        containing an edge list of the form

        source,target

        where ',' can be an arbitrary separator character
        """
        with open(filename, 'r') as f:
            edges = []

            if mapping is not None:
                Log.add('Filtering mapped edges')

            Log.add('Reading edge list ...')

            if header:  # Read header
                f.readline()
            for i, line in enumerate(f):
                if maxlines and i > maxlines:
                    break
                fields = line.rstrip().split(sep)
                try:
                    if mapping is None or (fields[0] in mapping and fields[1] in mapping):
                        edges.append((fields[0], fields[1]))

                except (IndexError, ValueError):  # pragma: no cover
                    msg = 'Ignoring malformed data in ' \
                          'line {}: "{}"'.format((i+header), line.strip())
                    Log.add(msg, Severity.WARNING)

        return cls(edges=edges)
