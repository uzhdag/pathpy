# -*- coding: utf-8 -*-
#    pathpy is an OpenSource python package for the analysis of time series data
#    on networks using higher- and multi order graphical models.
#
#    Copyright (C) 2016-2018 Ingo Scholtes, ETH Zürich/Universität Zürich
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program. If not, see <http://www.gnu.org/licenses/>.
#
#    Contact the developer:
#
#    E-mail: scholtes@ifi.uzh.ch
#    Web:    http://www.ingoscholtes.net

"""
Provides default containers for various classes
which are used to store nodes, edges and similar objects.

To make the various classes pickle-able the defaultdicts need to be publicly addressable
function names, this means that no lambda functions are allowed.

All pathpy classes which required a default value as a container, should use these here.
"""

from collections import defaultdict
import numpy as np


def nested_zero_default():
    """
    Returns a nested default dict (2 levels)
    with a numpy zero array of length 0 as default
    """
    return defaultdict(zero_array_default)


def _zero_array():
    """
    Returns a zero numpy array of length 2
    """
    return np.array([0.0, 0.0])


def zero_array_default():
    """
    Returns a default dict with numpy zero array af length 2 as default
    """
    return defaultdict(_zero_array)
