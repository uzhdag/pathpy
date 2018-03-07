"""base module for pathpy"""

__author__ = """Ingo Scholtes"""
__email__ = 'ischoltes@ethz.ch'
__version__ = '2.0.0'

from .log import Log, Severity
from .temporal_network import TemporalNetwork
from .network import Network
from .paths import Paths
from .higher_order_network import HigherOrderNetwork
from .multi_order_model import MultiOrderModel
from .markov_sequence import MarkovSequence
from .dag import DAG
from .algorithms import higher_order_measures
from .algorithms import path_measures
from .path_extraction import paths_from_dag
from .path_extraction import paths_from_origin_destination
from .path_extraction import paths_from_temporal_network
from .exception import PathpyNotImplemented
from .exception import PathpyException
from .exception import PathpyError


global ENABLE_MULTICORE_SUPPORT
ENABLE_MULTICORE_SUPPORT = False
