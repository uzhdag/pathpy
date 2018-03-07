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

import numpy as _np

import scipy.sparse as _sparse
import scipy.sparse.linalg as _sla

from pathpy.utils import Log, Severity


class Network:
    """
    Instances of this class capture a graph or network 
    that can be directed, undirected, unweighted or weighted
    """

    def __init__(self, directed=False):
        """
        Generates an empty network.       
        """

        self.directed = directed 

        # A dictionary containing nodes as well as node properties
        self.nodes = _co.defaultdict(lambda: {})        

        # A dictionary containing edges as well as edge properties
        self.edges = _co.defaultdict(lambda: {})

         # A dictionary containing the sets of successors of all nodes
        self.successors = _co.defaultdict(set)

        # A dictionary containing the sets of predecessors of all nodes
        self.predecessors = _co.defaultdict(set)       


    @classmethod
    def read_edges(cls, filename, separator=',', weighted=False, directed=False):
        """
        Reads a network from an edge list file

        Reads data from a file containing multiple lines of *edges* of the
        form "v,w,frequency,X" (where frequency is optional and X are
        arbitrary additional columns). The default separating character ','
        can be changed. In order to calculate the statistics of paths of any length,
        by default all subpaths of length 0 (i.e. single nodes) contained in an edge
        will be considered.

        Parameters
        ----------
        filename : str
            path to edgelist file
        separator : str
            character separating the nodes
        weight : bool
            is a weight given? if ``True`` it is the last element in the edge
            (i.e. ``a,b,2``)
        undirected : bool
            are the edges directed or undirected
        
        Returns
        -------
        Network
            a ``Network`` object obtained from the edgelist
        """
        n = Network(directed)

        with open(filename, 'r') as f:
            Log.add('Reading edge list ... ')
            for n, line in enumerate(f):
                fields = line.rstrip().split(separator)
                assert len(fields) >= 2, 'Error: malformed line: {0}'.format(line)

                if weighted:
                    n.add_edge(fields[0], fields[1], weight = int(fields[2]))
                else:
                    n.add_edge(fields[0], fields[1])

        Log.add('finished.')

        return n


    @classmethod
    def from_sqlite(cls, cursor, directed=True):
        """Reads links from an SQLite cursor and returns a new instance of
        the class Network. The cursor is assumed to refer to a table that
        minimally has two columns

                source target

        and where each row refers to a link. Any additional columns will be used as 
        edge properties

        Important: Since columns are accessed by name this function requires that a
        row factory object is set for the SQLite connection prior to cursor creation,
        i.e. you should set

                connection.row_factory = sqlite3.Row

        Parameters
        ----------
        cursor:
            The SQLite cursor to fetch rows
        directed: bool        

        Returns
        -------

        """
        n = Network(directed=directed)

        assert cursor.connection.row_factory, \
            'Cannot access columns by name. Please set ' \
            'connection.row_factory = sqlite3.Row before creating DB cursor.'

        Log.add('Retrieving links from database ...')

        for row in cursor:    
            n.add_edge(str(row['source']), str(row['target']))

        return n


    def add_node(self, v, **kwargs):
        """
        Adds a node to a network
        """
        self.nodes[v] = { **self.nodes[v], **kwargs }


    def add_edge(self, v, w, **kwargs):
        """
        Adds an edge to a network
        """
        self.add_node(v)
        self.add_node(w)

        if not self.directed:
            edge = tuple(x for x in sorted([v,w]))
        self.edges[edge] = { **self.edges[edge], **kwargs}

        self.successors[v].add(w)
        self.predecessors[w].add(w)

        if not self.directed:            
            self.successors[w].add(v)
            self.predecessors[v].add(v)

    def find_nodes(self, select_node = lambda v: True):
        """ 
        Returns all nodes that satisfy a given condition
        """
        return [ n for n in self.nodes if select_node(self.nodes[n])]

    def find_edges(self, select_nodes = lambda v,w: True, select_edges = lambda e: True):
        """ 
        Returns all edges that satisfy a given condition. Edges can be selected based 
        on attributes of the adjacent nodes as well as attributes of the edge
        """
        return [ e for e in self.edges if (select_nodes(self.nodes[e[0]], self.nodes[e[1]]) and select_edges(self.edges[e])) ]
        
    def vcount(self):
        """ Returns the number of nodes """
        return len(self.nodes)

    def ecount(self):
        """ Returns the number of links """
        return len(self.edges)

    def total_edge_weight(self):
        """ Returns the sum of all edge weights """
        if self.edges:
            return sum(self.edges.values())
        return 0

    def node_to_name_map(self):
        """Returns a dictionary that can be used to map nodes to matrix/vector indices"""
        return {v: idx for idx, v in enumerate(self.nodes)}

    def _to_html(self, width=600, height=600, use_requirejs=True):
        import json
        import os
        from string import Template

        # prefix nodes starting with number
        def fix_node_name(v):
            if v[0].isdigit():
                return "n_" + v
            else:
                return v

        network_data = {
            'nodes': [{'id': fix_node_name(v), 'group': 1} for v in self.nodes],
            'links': [
                {'source': fix_node_name(e[0]),
                 'target': fix_node_name(e[1]),
                 'value': 1} for e in self.edges.keys()
            ]
        }

        import string
        import random

        all_chars = string.ascii_letters + string.digits
        div_id = "".join(random.choice(all_chars) for x in range(8))

        if not use_requirejs:
            template_file = 'higherordernet.html'
        else:
            template_file = 'higherordernet_require.html'

        module_dir = os.path.dirname(os.path.realpath(__file__))
        html_dir = os.path.join(module_dir, os.path.pardir, 'html_templates')

        with open(os.path.join(html_dir, template_file)) as f:
            html_str = f.read()

        html_template = Template(html_str)

        return html_template.substitute({
            'network_data': json.dumps(network_data),
            'width': width,
            'height': height,
            'div_id': div_id})

    def _repr_html_(self, use_requirejs=True):
        """
        display an interactive D3 visualisation of the higher-order network in jupyter
        """
        from IPython.core.display import display, HTML
        display(HTML(self._to_html(use_requirejs=use_requirejs)))

    def write_html(self, filename, width=600, height=600):
        html = self._to_html(width=width, height=height, use_requirejs=False)
        with open(filename, 'w+') as f:
            f.write(html)
