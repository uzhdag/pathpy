# -*- coding: utf-8 -*-
#    pathpy is an OpenSource python package for the analysis of time series data
#    on networks using higher- and multi order graphical models.
#
#    Copyright (C) 2016-2017 Ingo Scholtes, ETH Zürich
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
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#    Contact the developer:

#    E-mail: ischoltes@ethz.ch
#    Web:    http://www.ingoscholtes.net

import collections as _co
import sys as _sys
import numpy as _np

from pathpy import Paths as _Paths
from pathpy import Log as _Log


def paths_from_temporal_network(tempnet, delta=1, maxLength=_sys.maxsize,
                                maxSubPathLength=_sys.maxsize):
    """create from a temporal network a Paths object
    Calculates the frequency of all time-respecting paths up to maximum length
    of maxLength, assuming a maximum temporal distance of delta between consecutive
    time-stamped links on a path. This (static) method returns an instance of the
    class Paths, which can subsequently be used to generate higher-order network
    representations based on the path statistics.

    Parameters
    ----------

    tempnet : pytest.TemporalNetwork
    delta : int
        Indicates the maximum temporal distance up to which time-stamped
        links will be considered to contribute to time-respecting paths.
        For (u, v;3) and (v,w;7) a time-respecting path (u,v)->(v,w) will be inferred
        for all 0 < delta <= 4, while no time-respecting path will be inferred for all
        delta > 4. If the max time diff is not set specifically, the default value of
        delta=1 will be used, meaning that a time-respecting path u -> v -> w will
        only be inferred if there are *directly consecutive* time-stamped links
        (u,v;t) (v,w;t+1). Every time-stamped edge is further considered a path of
        length one, i.e. for maxLength=1 this function will simply return the
        statistics of time-stamped edges.
    maxLength : int
        Indicates the maximum length up to which time-respecting paths should be
        calculated, which can be limited due to computational efficiency.
        A value of k will generate all time-respecting paths consisting of up to k
        time-stamped links. Note that generating a multi-order model with a maximum
        order of k requires to paths_from_temporal_network time-respecting paths with
        *at least* length k.
        If a limitation of the maxLength is not required for computational reasons,
        this parameter should not be set (as it will change the statistics of paths)
    maxSubPathLength : int

    Returns
    -------
    paths
    """

    if maxLength == _sys.maxsize:  # pragma: no cover
        _Log.add('Extracting time-respecting paths for delta = ' + str(delta) + ' ...')
    else:  # pragma: no cover
        _Log.add('Extracting time-respecting paths up to length ' + str(maxLength) +
                ' for delta = ' + str(delta) + ' ...')

    # for dictionary p.paths paths[k] contains a list of all
    # time-respecting paths p with length k and paths[k][p] contains
    # a two-dimensional counter whose first component counts the number of
    # occurrences of p as subpath of a longer path and whose second component counts
    # the number of occurrences of p as "real" path
    p = _Paths()

    p.maxSubPathLength = maxSubPathLength
    # a dictionary containing paths that can still be extended
    # by future time-stamped links
    # candidates[t][v] is a set of paths which end at time t in node v
    candidates = _co.defaultdict(lambda: _co.defaultdict(lambda: set()))

    # Note that here we only paths_from_temporal_network **longest** time-respecting paths, since we will
    # use later the expandSubpaths function to calculate statistics of shorter paths

    # set of longest time-respecting paths (i.e. those paths which are
    # NOT sub path of a longer time-respecting path)
    longest_paths = set()

    # loop over all time stamps t of edges
    for t in tempnet.ordered_times:

        for e in tempnet.time[t]:
            # assume that this edge is the root of a longest time-respecting path
            root = True

            # check whether this edge extends existing candidates
            for t_prev in list(candidates):
                # time stamp of candidate has to be in [t-delta, t) ...
                if t-delta <= t_prev < t:
                    # ... and last node has to be e[0] ...
                    if e[0] in candidates[t_prev]:
                        for c in list(candidates[t_prev][e[0]]):

                            # c is path (p_0, p_1, ...) which ends in node e[0] at
                            # time t_prev
                            new_path = c + ((e[0], e[1], t),)

                            # we now know that (e[0], e[1]) is not the root of a
                            # new longest path as it continues a previous path c
                            root = False

                            # if c has previously been considered a longest path,
                            # we discard it from the list of longest paths. We also
                            # add the extended path as a new longest path
                            # (possible removing it later if it is further extended)
                            longest_paths.discard(c)
                            longest_paths.add(new_path)

                            # we add the newly found path as a candidate for paths
                            # which can be continued by future edges
                            if len(new_path) < maxLength:
                                candidates[t][e[1]].add(new_path)

                            # delete candidate c, because from now on
                            # we only extend new_path
                            candidates[t_prev][e[0]].discard(c)

            # if edge e does not continue a previous path
            # we start a new longest path
            if root:
                longest_paths.add(((e[0], e[1], t), ))
                # add edge as candidate path of length one that can be extended by
                # future edges
                if maxLength > 1:
                    candidates[t][e[1]].add(((e[0], e[1], t), ))

        # we finished processing time stamp t, so
        # we can remove all candidates which finish
        # at a time smaller than t-delta. Since they cannot
        # be extended, these are longest paths
        for t_prev in list(candidates.keys()):
            if t_prev < t-delta:
                del candidates[t_prev]

    # once we reached the last time stamp, add all candidates
    # as longest paths
    # for t_prev in candidates:
    #    for x in candidates[t_prev]:
    #        for p in candidates[t_prev][x]:
    #            longest_paths.add(p)

    # Count occurrences as longest time-respecting path
    for x in longest_paths:
        path = (x[0][0], )
        for edge in x:
            path += (edge[1], )
        p.paths[len(x)][path] += _np.array([0, 1])

    # expand sub paths of longest paths
    p.expand_subpaths()

    _Log.add('finished.')

    return p
