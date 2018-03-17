"""This module contains the containers for the various classes
which are used to store nodes, edges and similar

To make the various classes pickle-able the defaultdicts need to be publicly addressable
function names, this means that no lambda functions are allowed.

All Pathpy classes which required a default value as a container, should use these here.
"""
from collections import defaultdict
import numpy as np


def nested_zero_default():
    """returns a nested default dict (2 levels)
    with a numpy zero array of length 0 as default
    """
    return defaultdict(zero_array_default)


def _zero_array():
    """returns a zero numpy array of length 2"""
    return np.array([0.0, 0.0])


def zero_array_default():
    """returns a default dict with numpy zero array af length 2 as default"""
    return defaultdict(_zero_array)
