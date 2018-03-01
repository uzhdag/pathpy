"""base module for pathpy"""

__author__ = """Ingo Scholtes"""
__email__ = 'ischoltes@ethz.ch'
__version__ = '1.2.1'


from pathpy.log import Log, Severity
from pathpy.temporal_network import TemporalNetwork
from pathpy.paths import Paths
from pathpy.higher_order_network import HigherOrderNetwork
from pathpy.multi_order_model import MultiOrderModel
from pathpy.markov_sequence import MarkovSequence
from pathpy.dag import DAG
from pathpy.algorithms import higher_order_measures
from pathpy.algorithms import path_measures

from pathpy.exception import PathpyNotImplemented, PathpyException, PathpyError

global ENABLE_MULTICORE_SUPPORT
ENABLE_MULTICORE_SUPPORT = False
