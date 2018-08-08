"""base module of pathpy"""

__author__ = """Ingo Scholtes"""
__email__ = 'scholtes@ifi.uzh.ch'
__version__ = '2.0.0'

from .classes import *
from .algorithms import *
from .path_extraction import *
from .visualisation import *
import pathpy.utils

global ENABLE_MULTICORE_SUPPORT
ENABLE_MULTICORE_SUPPORT = False
