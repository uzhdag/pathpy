# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 12:06:00 2016
@author: Ingo Scholtes

(c) Copyright ETH Zurich, Chair of Systems Design, 2015-2017
"""

import collections as _co
import bisect as _bs
import itertools as _iter

import sys as _sys

import numpy as _np 

import scipy.sparse as _sparse
import scipy.sparse.linalg as _sla
import scipy.linalg as _la
import scipy as _sp

from pathpy.Log import Log
from pathpy.Log import Severity

import requests
from lxml import etree


class DAG(object):
    """
        Represents a directed acyclic graph (DAG) which 
        can be used to generate pathway statistics.
    """


    def __init__(self, edges = None):
        """
        Constructs a directed acyclic graph from an edge list
        """

        self.nodes = set()
        self.edges = set()

        ## Whether or not this graph is acyclic. None indicates that it is unknown
        self.isAcyclic = None

        ## list of topologically sorted nodes
        self.sorting = []

        ## Set of nodes with no incoming edges
        self.roots = set()

        ## Set of nodes with no outgoing edges
        self.leafs = set()

        ## The dictionary of successors of each node
        self.successors = _co.defaultdict( lambda: set() )

        ## The dictionary of predecessors of each node
        self.predecessors = _co.defaultdict( lambda: set() )
        self_loops = 0
        redundant_edges = 0
        if edges != None:
            for e in edges:
                redundant = False
                selfLoop = False
                if e[0] == e[1]:
                    selfLoop = True
                    self_loops += 1
                if (e[0], e[1]) in self.edges:
                    redundant = True
                    redundant_edges += 1
                if not selfLoop and not redundant:
                    self.addEdge(e[0], e[1])
            if self_loops>0:
                Log.add('Warning: omitted ' + str(self_loops) + ' self-loops', Severity.WARNING)
            if redundant_edges>0:
                Log.add('Warning: omitted ' + str(redundant_edges) + ' redundant edges', Severity.WARNING)



    def constructPaths(self, v, node_mapping = {}):
        """
        Constructs all paths from node v to any leaf nodes
        """
        paths = _co.defaultdict( lambda: [] )
        if v not in node_mapping:
            paths[v] = [ (v,) ]
        else:
            paths[v] = [ (node_mapping[v],) ]

        # set of unprocessed nodes
        Q = set([v])

        while len(Q)>0:
            # take one unprocessed node
            x = Q.pop()
            # expand paths if it has successors
            if len(self.successors[x])>0:
                for w in self.successors[x]:
                    for p in paths[x]:
                        if w not in node_mapping:
                            paths[w].append(p + (w,))
                        else:
                            paths[w].append(p + (node_mapping[w],))
                    Q.add(w)
                del paths[x]

        return paths


    def dfs_visit(self, v, parent = None):
        """
        Recursively visits nodes in the graph, classifying 
        edges as (1) tree, (2) forward, (3) back or (4) cross 
        edges.

        @param v: the node to be visited
        @param parent: the parent of this node (None for nodes)
            with no parents
        """
        self.parent[v] = parent
        self.count += 1 
        self.start_time[v] = self.count
        if parent:
            self.edge_classes[(parent, v)] = 'tree'

        for w in self.successors[v]:
            if w not in self.parent: 
                self.dfs_visit(w, v)
            elif w not in self.finish_time:
                self.edge_classes[(v,w)] = 'back'
                self.isAcyclic = False
            elif self.start_time[v] < self.start_time[w]:
                self.edge_classes[(v,w)] = 'forward'
            else:
                self.edge_classes[(v,w)] = 'cross'
        self.count += 1
        self.finish_time[v] = self.count
        self.sorting.append(v)


    def topsort(self):
        """
        Performs a topological sorting of the graph, classifying 
        all edges as (1) tree, (2) forward, (3) back or (4) cross 
        edges in the process. 

        see Cormen 2001 for details
        """
        self.parent = {}
        self.start_time = {}
        self.finish_time = {}
        self.edge_classes = {}
        self.sorting = []
        self.count = 0
        self.isAcyclic = True
        for v in self.nodes:
            if v not in self.parent:
                self.dfs_visit(v)
        self.sorting.reverse()


    def makeAcyclic(self):
        """
        Removes all backlinks from the graph to make it 
        acyclic, then performs another topological sorting 
        of the DAG
        """

        if self.isAcyclic==None:
            self.topsort()
        removed_links = 0
        if self.isAcyclic == False:
            # Remove all back links            
            for e in list(self.edge_classes):
                if self.edge_classes[e] == 'back':
                    self.edges.remove(e)
                    removed_links += 1
                    self.successors[e[0]].remove(e[1])
                    self.predecessors[e[1]].remove(e[0])
                    del self.edge_classes[e]
            self.topsort()
            assert self.isAcyclic, "Error: makeAcyclic did not generate acyclic graph!"
            Log.add('Removed ' + str(removed_links) + ' back links to make graph acyclic', Severity.INFO)


    def summary(self):
        """
        Returns a string representation of this directed acyclic graph
        """

        summary = 'Directed Acyclic Graph'
        summary += '\n'
        summary += 'Nodes:\t\t' +  str(len(self.nodes)) + '\n'
        summary += 'Links:\t\t' + str(len(self.edges)) + '\n'
        summary += 'Acyclic:\t' +  str(self.isAcyclic) + '\n'
        return summary  


    def __str__(self):
        """
        Returns the default string representation of this object
        """
        return self.summary()


    def addEdge(self, source, target):
        """
        Adds a directed edge to the graph
        """

        if source not in self.nodes:
            self.nodes.add(source)
            self.roots.add(source)
            self.leafs.discard(source)
        if target not in self.nodes:
            self.nodes.add(target)    
            self.roots.discard(target)
            self.leafs.add(target)

        self.edges.add((source, target))
        self.successors[source].add(target)
        self.predecessors[target].add(source)
        self.isAcyclic = None   


    @staticmethod
    def readFile(filename, sep=',', maxlines=_sys.maxsize, mapping=None):
        """
        Reads a directed acyclic graph from a file
        containing an edge list of the form 
        
        source,target 

        where ',' can be an arbitrary separator character

        """

        assert (filename != ''), 'Empty filename given'
        
        # Read header
        with open(filename, 'r') as f:
            edges = []                             
        
            if mapping != None:
                Log.add('Filtering mapped edges')

            Log.add('Reading edge list ...')

            line = f.readline()
            n = 1           
            while line and n <= maxlines:
                fields = line.rstrip().split(sep)
                try:
                    if mapping == None or (fields[0] in mapping and fields[1] in mapping):
                        edges.append((fields[0], fields[1]))
                    
                except (IndexError, ValueError):
                    Log.add('Ignoring malformed data in line ' + str(n+1) + ': "' +  line.strip() + '"', Severity.WARNING)
                line = f.readline()
                n += 1
        # end of with open()

        return DAG(edges = edges)




