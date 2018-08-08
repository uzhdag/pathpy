"""base module of pathpy"""

__author__ = """Ingo Scholtes"""
__email__ = 'scholtes@ifi.uzh.ch'
__version__ = '2.0.0'

from .classes import *
from .path_extraction import *
from .visualisation import *
import pathpy.utils

from .algorithms import centralities
from .algorithms import shortest_paths
from .algorithms import path_measures
from .algorithms import components
from .algorithms import infomap
from .algorithms import spectral

global ENABLE_MULTICORE_SUPPORT
ENABLE_MULTICORE_SUPPORT = False
