"""
An OpenSource python package to analyze and
visualize time series data on complex networks.
"""

__author__ = """Ingo Scholtes"""
__email__ = 'scholtes@ifi.uzh.ch'
__version__ = '2.1.0'

from .classes import *
import pathpy.path_extraction
import pathpy.visualisation
import pathpy.algorithms.centralities
import pathpy.algorithms.components
import pathpy.algorithms.shortest_paths
import pathpy.algorithms.centralities
import pathpy.algorithms.random_walk
import pathpy.algorithms.temporal_walk
import pathpy.algorithms.spectral
import pathpy.algorithms.path_measures
import pathpy.algorithms.infomap
import pathpy.algorithms.modularity
import pathpy.algorithms.random_graphs
from .algorithms import statistics

import pathpy.utils

global ENABLE_MULTICORE_SUPPORT
ENABLE_MULTICORE_SUPPORT = False
