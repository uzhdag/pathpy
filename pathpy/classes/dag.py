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
from collections import defaultdict
from pathpy.utils import Log, Severity
from pathpy.classes.network import Network


class DAG(Network):
    """
        A directed acyclic graph (DAG) which
        can be used to generate pathway statistics.
    """

    def __init__(self, edges=None):
        """
        Constructs a directed acyclic graph from an edge list
        """

        super().__init__(directed=True)

        # Whether or not this graph is acyclic. None indicates that it is unknown
        self.is_acyclic = None

        # list of topologically sorted nodes
        self.sorting = []

        # Set of nodes with no incoming edges
        self.roots = set()

        # Set of nodes with no outgoing edges
        self.leafs = set()

        if edges is not None:
            self.add_edges(edges)

        # placeholder properties for topological sort
        self.parent = {}
        self.start_time = {}
        self.finish_time = {}
        self.edge_classes = {}
        self.top_sort_count = 0


    @classmethod
    def from_node_activity_sequence(cls, sequence, self_loops=True, connect_prior=0):
        """
        Generates a directed acyclic graph from a sequence of node
        activations, where each activation is assumed to be influenced
        by previous activations. This is useful to extract a
        directed acyclic graph, e.g., from a sequence of user activities
        in a communication forum. The sequence of interactions is taken into
        account in the generation of the DAG. The number of prior activations to
        each node will be linked can be customized using the connect_prior parameter.
        For the default 0 (which means connect to all prior activations), a sequence
        ['a', 'b', 'c', 'a'] will result in a dag (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
        The returned node_map maps the nodes in the DAG to the actual nodes, i.e. in the example
        node_map = { '0': 'a', '1': 'b', '2': c, '3': a}.

        Parameter:
        ----------
        sequence: iterable
            any iterable containing a sequence of string node names
        self_loops: bool
            whether or not to consider links linking nodes to themselves, i.e. whether
            to consider an edge (x, y) in the dag where node_map[x] == node_map[y]
        connect_prior: bool
            to how many prior nodes each node shall be linked. For a value of 1, a chain 
            will be generated, for 2 a lattice where each node is connected to two previous
            nodes is generated. For the default 0 each node is connected to all prior nodes.
        """
        dag = cls()

        # dictionary that maps nodes in dag to actual nodes
        node_map = {}

        i = 0
        for v in sequence:
            name = str(i)
            node_map[name] = v
            dag.add_node(name)
            if connect_prior == 0:
                m = i
            else:
                m = min(connect_prior, i)
            for j in range(i-m, i):
                if self_loops or node_map[str(j)] != node_map[name]:
                    dag.add_edge(str(j), name)            
            i += 1
        return dag, node_map


    @classmethod
    def from_temporal_network(cls, tempnet, delta=1):
        """
        Creates a time-unfolded directed acyclic graph from a
        temporal network for a given maximum time difference delta.

        Parameters
        ----------
        tempnet: pp.TemporalNetwork
            The temporal network instance to use for the DAG generation.

        delta: int
            The maximum time difference t'- t between two time-stamped edges
            (a,b,t) and (b,c,t') up to which a causal path (a, b, c) is a
            created in the DAG

        """
        dag = cls()

        # dictionary that maps time-unfolded nodes to actual nodes
        node_map = {}

        for (v, w, t) in tempnet.tedges:
            # create time-unfolded nodes v_t and w_{t+1}
            v_t = "{0}_{1}".format(v, t)
            node_map[v_t] = v

            # create one time-unfolded link for all delta in [1, delta]
            # this implies that for delta = 2 and an edge (a,b,1) two 
            # time-unfolded links (a_1, b_2) and (a_1, b_3) will be created
            for x in range(1, int(delta)+1):
                w_t = "{0}_{1}".format(w, t+x)
                node_map[w_t] = w
                dag.add_edge(v_t, w_t)

        return dag, node_map


    def add_edges(self, edges):
        """Add a list of edges

        Parameters
        ----------
        edges: list
            a list of edges [(s_1, t_1), (s_1, t_2), ...]

        """
        self_loops = 0
        redundant_edges = 0
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

    def routes_from_node(self, v, node_mapping=None):
        """
        Constructs all paths from node v to any leaf nodes

        Parameters
        ----------
        v:
            node from which to start
        node_mapping: dict
            an optional mapping from node to a different set.

        Returns
        -------
        list
            a list of lists, where each list contains one path from the source
            node v until a leaf node is reached
        """
        # Collect temporary paths, indexed by the target node
        temp_paths = defaultdict(list)
        temp_paths[v] = [[v]]

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
                        temp_paths[w].append(p + [w])
                    queue.add(w)
                del temp_paths[x]

        # flatten list
        final_paths = []
        for possible_paths in temp_paths.values():
            for path in possible_paths:
                if node_mapping:
                    path = [node_mapping[k] for k in path]
                final_paths.append(path)

        return final_paths

    def routes_to_node(self, v, node_mapping=None):
        """
        Constructs all paths to node v from any root node

        Parameters
        ----------
        v:
            node from which to start
        node_mapping: dict
            an optional mapping from node to a different set.

        Returns
        -------
        list
            a list of lists, where each list contains one path from the source
            node v until a leaf node is reached
        """
        # Collect temporary paths, indexed by the target node
        temp_paths = defaultdict(list)
        temp_paths[v] = [[v]]

        # set of unprocessed nodes
        queue = {v}

        while queue:
            # take one unprocessed node
            x = queue.pop()

            # successors of x expand all temporary
            # paths, currently ending in x
            if self.predecessors[x]:
                for w in self.predecessors[x]:
                    for p in temp_paths[x]:
                        temp_paths[w].append(p + [w])
                    queue.add(w)
                del temp_paths[x]

        # flatten list
        final_paths = []
        for possible_paths in temp_paths.values():
            for path in possible_paths:
                path = list(reversed(path))
                if node_mapping:
                    path = [node_mapping[k] for k in path]
                final_paths.append(path)

        return final_paths

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
                    self.remove_edge(*e)
                    removed_links += 1
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

    def isolate_nodes(self):
        """Return a set of isolated nodes"""
        return self.leafs & self.roots

    def __str__(self):
        """
        Returns the default string representation of this object
        """
        return self.summary()

    def add_node(self, node):
        """
        Adds a node to a graph
        """
        if node not in self.nodes:
            self.roots.add(node)
            self.leafs.add(node)
            super().add_node(node)

    def add_edge(self, source, target, **kwargs):
        """
        Adds a directed edge to the graph
        """

        if source not in self.nodes:
            self.add_node(source, **kwargs)
            self.roots.add(source)
        if target not in self.nodes:
            self.add_node(target, **kwargs)
            self.leafs.add(target)

        self.leafs.discard(source)
        self.roots.discard(target)
        super().add_edge(source, target)
        self.is_acyclic = None

    def remove_edge(self, source, target):
        """remove an edge"""
        super().remove_edge(source, target)

        # if no more predecessors for the target then it becomes a root
        if not self.predecessors[target]:
            self.roots.add(target)

        # if no more successors for the source node then it becomes a leaf
        if not self.successors[source]:
            self.leafs.add(target)


    def write_file(self, filename, separator=','):
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
                file.write(separator.join(edge)+'\n')

    @classmethod
    def read_file(cls, filename, separator=',', maxlines=None, mapping=None, header=False):
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
                fields = line.rstrip().split(separator)
                try:
                    if mapping is None or (fields[0] in mapping and fields[1] in mapping):
                        edges.append((fields[0], fields[1]))

                except (IndexError, ValueError):  # pragma: no cover
                    msg = 'Ignoring malformed data in ' \
                          'line {}: "{}"'.format((i+header), line.strip())
                    Log.add(msg, Severity.WARNING)

        return cls(edges=edges)
