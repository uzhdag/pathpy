"""base module for pathpy"""

__author__ = """Ingo Scholtes"""
__email__ = 'ischoltes@ethz.ch'
__version__ = '2.0.0.dev1'

from .classes import DAG
from .classes import Paths
from .classes import TemporalNetwork
from .classes import HigherOrderNetwork
from .classes import MultiOrderModel
from .classes import MarkovSequence
from .classes import Network
from .algorithms import spectral
from .algorithms import path_measures
from .algorithms import shortest_paths
from .algorithms import components
from .algorithms import centralities
from .algorithms import infomap
from .path_extraction import paths_from_dag
from .path_extraction import paths_from_origin_destination
from .path_extraction import paths_from_temporal_network
from .visualisation import alluvial_diagram
import pathpy.utils

global ENABLE_MULTICORE_SUPPORT
ENABLE_MULTICORE_SUPPORT = False
