"""
Algorithms to calculate shortest paths and distances in higher-order networks and paths.
"""
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
from functools import singledispatch
from collections import defaultdict

import random

from pathpy.utils import Log, Severity
from pathpy.utils import PathpyNotImplemented
from pathpy.classes import TemporalNetwork
from pathpy.classes import Network
from pathpy.classes import Paths

def generate_walk(tempnet, l=100, start_node=None):
    """
    """
    itinerary = []
    if start_node is None:
        current_node = random.sample(tempnet.nodes, k=1)[0]
    else:
        current_node = start_node
    itinerary.append(current_node)
    steps = 0
    for t in tempnet.ordered_times:
        targets = set()
        for (v, w, time) in tempnet.time[t]:
            if v == current_node:
               targets.add(w)
        if targets:
            current_node = random.sample(targets, k=1)[0]
            itinerary.append(current_node)
            steps += 1
        if steps == l:
            break
    return itinerary
