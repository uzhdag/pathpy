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

        ## The dictionary of successors of each node
        self.successors = _co.defaultdict( lambda: set() )

        ## The dictionary of predecessors of each node
        self.successors = _co.defaultdict( lambda: set() )


    def isAcyclic(self):
        """
        Returns whether the graph is acyclic
        """

        raise NotImplementedError()


    def addEdge(self, source, target):
        """
        Adds a directed edge to the graph
        """

        if source not in self.nodes:
            self.nodes.add(source)
        if target not in self.nodes:
            self.nodes.add(target)

        self.edges.add((source, target))
        self.successors[source].add(target)
        self.predecessors[target].add(source)


    @staticmethod
    def readFile(filename, sep=','):
        """
        Reads a directed acyclic graph from a file
        """

        return DAG(edges = None)




