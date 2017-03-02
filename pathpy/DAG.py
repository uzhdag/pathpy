# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 12:06:00 2016
@author: Ingo Scholtes

(c) Copyright ETH Zurich, Chair of Systems Design, 2015-2017
"""

import collections as _co
import bisect as _bs
import itertools as _iter

import numpy as _np 

import scipy.sparse as _sparse
import scipy.sparse.linalg as _sla
import scipy.linalg as _la
import scipy as _sp

from pathpy.Log import Log
from pathpy.Log import Severity


class CycleError(Exception):
    """
    This exception is thrown whenever a a cycle is found 
    in what is supposed to be a directed acyclic graph
    """
    pass


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
        self.isSorted = False

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

        if edges != None:
            for e in edges:
                self.addEdge(e[0], e[1], testAcyclic = False)



    def constructPaths(self, v):
        """
        Constructs all paths from node v to any leaf nodes
        """
        paths = _co.defaultdict( lambda: [] )
        paths[v] = [ (v,) ]

        # set of unprocessed nodes
        Q = set(v)

        while len(Q)>0:
            # take one unprocessed node
            x = Q.pop()

            # expand paths if it has successors
            if len(self.successors[x])>0:
                for w in self.successors[x]:
                    for p in paths[x]:
                        paths[w].append(p + (w,))
                    Q.add(w)
                del paths[x]

        return paths


    def dfs(self, v):
        """
        dfs function for topological sorting of DAG
        """
        if v in self.tempmark:
            raise CycleError()
        if v in self.unmarked:
            self.tempmark.add(v)
            for w in self.successors[v]:
                self.dfs(w)
            self.unmarked.remove(v)
            self.tempmark.discard(v)            
            self.sorting.insert(0, v)


    def topsort(self):
        """
        Sorts the nodes in the DAG topologically 
        Raises a CycleError if the graph is not acyclic
        """        
        self.sorting = []
        self.unmarked = list(self.nodes)
        self.tempmark = set()
        while len(self.unmarked)>0:
            v = self.unmarked[0]
            self.dfs(v)
        self.isSorted = True


    def summary(self):
        """
        Returns a string representation of this directed acyclic graph
        """

        try:
            if not self.isSorted:
                self.topsort()
        except CycleError:
            Log.add('Warning: cycle detected', Severity.ERROR)

        summary = 'Directed Acyclic Graph'
        summary += '\n'        
        summary += 'Nodes:\t\t' +  str(len(self.nodes)) + '\n'
        summary += 'Links:\t\t' + str(len(self.edges)) + '\n'
        summary += 'Acyclic:\t' +  str(self.isSorted) + '\n'
        return summary


    def __str__(self):
        """
        Returns the default string representation of this object
        """
        return self.summary()


    def addEdge(self, source, target, testAcyclic=True):
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
        self.isSorted = False


    @staticmethod
    def readFile(filename, sep=','):
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
        
            Log.add('Reading edge list ...')

            line = f.readline()
            n = 1 
            while line:
                fields = line.rstrip().split(sep)
                try:                   
                        edges.append((fields[0], fields[1]))
                except (IndexError, ValueError):
                    Log.add('Ignoring malformed data in line ' + str(n+1) + ': "' +  line.strip() + '"', Severity.WARNING)
                line = f.readline()
                n += 1
        # end of with open()

        return DAG(edges = edges)




