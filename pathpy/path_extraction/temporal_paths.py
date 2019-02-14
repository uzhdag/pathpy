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

from collections import defaultdict
from collections import deque
import sys

from pathpy import Paths
from pathpy import DAG
from pathpy.path_extraction import paths_from_dag
from pathpy.utils import Log
from pathpy.utils import Severity

def paths_from_temporal_network(tempnet, delta=1, max_length=sys.maxsize,
                                max_subpath_length=sys.maxsize):
    """
    Warning: This function is deprecated. Calls will be rerouted to to paths_from_temporal_network_dag. 
    
    If you intended to calculate paths with a single continuing edge, use paths_from_temporal_network_single instead (see documentation of this function for details).
    """
    Log.add('This function is deprecated. Rerouting call to paths_from_temporal_network_dag. If you intended to calculate paths with a single continuing edge, use paths_from_temporal_network_single instead.', Severity.WARNING)
    return paths_from_temporal_network_dag(tempnet, delta, max_subpath_length=max_subpath_length)


def paths_from_temporal_network_single(tempnet, delta=1, max_length=sys.maxsize,
                                max_subpath_length=sys.maxsize):
    """
    Calculates the frequency of time-respecting paths up to maximum length
    of maxLength, assuming a maximum temporal distance of delta between consecutive
    time-stamped links on a path. This method uses a fast but heuristic approach that
    only considers the FIRST continuation of a time-respecting path. I.e., for time-
    stamped links (a,b,1), (b,c,5), (b,d,7) and a delta=6 only the time-respecting
    path (a,b,c) will be found, while (a,b,d) is ignored. If the next time-stamp includes 
    two possible continuations, i.e. (b,c,5) and (b,d,5), only the path continued by the first 
    edge will be used.
    
    This (static) method returns an instance of the class Paths, which can
    be used to generate higher- and multi-order models of time-respecting paths.

    Parameters
    ----------
    tempnet : pathpy.TemporalNetwork
        TemporalNetwork to extract the path from
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
    max_length : int
        Indicates the maximum length up to which time-respecting paths should be
        calculated, which can be limited due to computational efficiency.
        A value of k will generate all time-respecting paths consisting of up to k
        time-stamped links. Note that generating a multi-order model with a maximum
        order of k requires to extract time-respecting paths with *at least* length k.
        If a limitation of the maxLength is not required for computational reasons,
        this parameter should not be set (as it will change the statistics of paths)
    max_subpath_length : int
        This can be used to limit the calculation of sub path statistics to a given
        maximum length. This is useful, as the statistics of sub paths of length k
        are only needed to fit a higher-order model with order k. Hence, if we know
        that the model selection is limited to a given maximum order K, we can safely
        set the maximum sub path length to K. By default, sub paths of any length
        will be calculated. Note that, independent of the sub path calculation
        longest path of any length will be considered in the likelihood calculation!

    Returns
    -------
    Paths

    Examples
    --------
    >>> t = pp.TemporalNetwork()
    >>> t.add_edge('a', 'b', 1)
    >>> t.add_edge('b', 'c', 5)
    >>> t.add_edge('b', 'd', 7)

    >>> p = pp.path_extraction.paths_from_temporal_network(t, delta=6)    
    >>> [Severity.INFO]	Extracting time-respecting paths for delta = 7 ...
    >>> [Severity.INFO]	Calculating sub path statistics ...
    >>> [Severity.INFO]	finished.
    >>> print(p)
    >>> Total path count: 		2.0 
    >>> [Unique / Sub paths / Total]: 	[2.0 / 7.0 / 9.0]
    >>> Nodes:				4 
    >>> Edges:				3
    >>> Max. path length:		2
    >>> Avg path length:		1.5 
    >>> Paths of length k = 0		0.0 [ 0.0 / 5.0 / 5.0 ]
    >>> Paths of length k = 1		1.0 [ 1.0 / 2.0 / 3.0 ]
    >>> Paths of length k = 2		1.0 [ 1.0 / 0.0 / 1.0 ]

    """

    if max_length == sys.maxsize:  # pragma: no cover
        Log.add('Extracting time-respecting paths for delta = ' + str(delta) + ' ...')
    else:  # pragma: no cover
        Log.add('Extracting time-respecting paths up to length ' + str(max_length) +
                 ' for delta = ' + str(delta) + ' ...')

    # for dictionary p.paths paths[k] contains a list of all
    # time-respecting paths p with length k and paths[k][p] contains
    # a two-dimensional counter whose first component counts the number of
    # occurrences of p as subpath of a longer path and whose second component counts
    # the number of occurrences of p as "real" path
    p = Paths()

    p.max_subpath_length = max_subpath_length
    # a dictionary containing paths that can still be extended
    # by future time-stamped links
    # candidates[t][v] is a set of paths which end at time t in node v
    candidates = defaultdict(lambda: defaultdict(lambda: set()))

    # Note that here we only extract **longest** time-respecting
    # paths, since we will use later the expandSubpaths function to calculate statistics
    # of shorter paths

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
                if t - delta <= t_prev < t:
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
                            if len(new_path) < max_length:
                                candidates[t][e[1]].add(new_path)

                            # if we delete candidate c we can only extend new_path from now on
                            # In this case, for every path we only consider the first link that 
                            # extends it to a longer path

                            candidates[t_prev][e[0]].discard(c)
            
            # if edge e does not continue a previous path
            # we start a new longest path
            if root:
                longest_paths.add(((e[0], e[1], t),))
                # add edge as candidate path of length one that can be extended by
                # future edges
                if max_length > 1:
                    candidates[t][e[1]].add(((e[0], e[1], t),))

        # we finished processing time stamp t, so
        # we can remove all candidates which finish
        # at a time smaller than t-delta. Since they cannot
        # be extended, these are longest paths
        for t_prev in list(candidates.keys()):
            if t_prev < t - delta:
                del candidates[t_prev]

    # once we reached the last time stamp, add all candidates
    # as longest paths
    # for t_prev in candidates:
    #    for x in candidates[t_prev]:
    #        for p in candidates[t_prev][x]:
    #            longest_paths.add(p)

    # Count occurrences as longest time-respecting path
    for x in longest_paths:
        path = (x[0][0],)
        for edge in x:
            path += (edge[1],)
        p.paths[len(x)][path][1] += 1

    # expand sub paths of longest paths
    p.expand_subpaths()

    Log.add('finished.')

    return p



def generate_causal_tree(dag, root, node_map):
    """
    For a directed acyclic graph and a non-injective mapping of nodes,
    this method creates a *causal tree* for a given root node.
    This is useful for the extraction of causal paths in time-unfolded DAG
    representations of temporal networks. The nodes "{v}_{d}" in the resulting
    causal tree capture that - starting from the root node at step 0 - there is
    a causal path to node v at distance d from the root. Note that the same node
    can be represented by multiple nodes in the causal tree (at different distances d).
    """    
    causal_tree = DAG()
    causal_mapping = {}
    visited = defaultdict(lambda: False)
    queue = deque()

    # launch breadth-first-search at root of tree
    # root nodes are necessarily at depth 0
    queue.append((root, 0))
    edges = []
    while queue:
        # take out left-most element from FIFO queue
        v, depth = queue.popleft()

        # x is the node ID of the node in the causal tree
        # the second component captures the distance from
        # the root of the causal tree. These IDs ensure
        # that the same physical nodes can occur at different
        # distances from the root
        x = '{0}_{1}'.format(node_map[v], depth)
        causal_mapping[x] = node_map[v]

        # process nodes at next level
        for w in dag.successors[v]:
            if (w, depth+1) not in queue:
                queue.append((w, depth+1))
                # only consider nodes that have not already
                # been added to this level
                if not visited[node_map[w], depth+1]:
                    # add edge to causal tree
                    y = '{0}_{1}'.format(node_map[w], depth+1)
                    edges.append((x, y))

                    visited[node_map[w], depth+1] = True
                    causal_mapping[y] = node_map[w]
    
    # Adding all edges at once is more efficient!
    causal_tree.add_edges(edges)

    return causal_tree, causal_mapping


def paths_from_temporal_network_dag(tempnet, delta=1, max_subpath_length=None):
    """
    Calculates the frequency of causal paths in a temporal network assuming a 
    maximum temporal distance of delta between consecutive
    time-stamped links on a path. This method first creates a directed and acyclic
    time-unfolded graph based on the given parameter delta. This directed acyclic
    graph is used to calculate all time-respecting paths for a given delta.
    I.e., for time-stamped links (a,b,1), (b,c,5), (b,d,7) and delta = 5 the
    time-respecting path (a,b,c) will be found.

    Parameters
    ----------
    tempnet : pathpy.TemporalNetwork
        TemporalNetwork to extract the time-respecting paths from
    delta : int
        Indicates the maximum temporal distance up to which time-stamped
        links will be considered to contribute to a causal path.
        For (u,v;3) and (v,w;7) a causal path (u,v,w) is generated
        for 0 < delta <= 4, while no causal path is generated for
        delta > 4. Every time-stamped edge is a causal path of
        length one. Default value is 1.
    max_subpath_length : int
        Can be used to limit the calculation of sub path statistics to a given
        maximum length. This is useful as statistics of sub paths of length k
        are only needed to fit higher-order model with order k and larger. If model
        selection is limited to a maximum order K, we can set the maximum sub path length
        to K. Default is None, which means all subpaths are calculated.

    Returns
    -------
    Paths
        An instance of the class Paths, which can be used to generate higher- and multi-order
        models of causal paths in temporal networks.

    Examples
    ---------
    >>> t = pp.TemporalNetwork()
    >>> t.add_edge('a', 'b', 1)
    >>> t.add_edge('b', 'a', 3)
    >>> t.add_edge('b', 'c', 3)
    >>> t.add_edge('d', 'c', 4)
    >>> t.add_edge('c', 'd', 5)
    >>> t.add_edge('c', 'b', 6)

    >>> >>>causal_paths = pp.path_extraction.paths_from_temporal_network_dag(t, delta=2)
    >>> [Severity.INFO]	Constructing time-unfolded DAG ...
    >>> [Severity.INFO]	finished.
    >>> [Severity.INFO]	Generating causal trees for 2 root nodes ...
    >>> [Severity.INFO]	finished.
    >>> print(causal_paths)
    >>> Total path count: 		4.0 
    >>> [Unique / Sub paths / Total]: 	[4.0 / 24.0 / 28.0]
    >>> Nodes:				    4 
    >>> Edges:				    6
    >>> Max. path length:		3
    >>> Avg path length:		2.25 
    >>> Paths of length k = 0		0.0 [ 0.0 / 13.0 / 13.0 ]
    >>> Paths of length k = 1		0.0 [ 0.0 / 9.0 / 9.0 ]
    >>> Paths of length k = 2		3.0 [ 3.0 / 2.0 / 5.0 ]
    >>> Paths of length k = 3		1.0 [ 1.0 / 0.0 / 1.0 ]

    >>> The calculated (longest) causal paths in this example are:
    >>> (a, b, c, d), (d, c, b), (d, c, d), (a, b, a)
    """
    # generate a single time-unfolded DAG
    Log.add('Constructing time-unfolded DAG ...')
    dag, node_map = DAG.from_temporal_network(tempnet, delta)
    Log.add('finished.')
    print(dag)

    causal_paths = Paths()
    
    # For each root in the time-unfolded DAG, we generate a
    # causal tree and use it to count all causal paths
    # that originate at this root
    num_roots = len(dag.roots)
    current_root = 1
    Log.add('Generating causal trees for {0} root nodes ...'.format(num_roots))
    for root in dag.roots:
        causal_tree, causal_mapping = generate_causal_tree(dag, root, node_map)
        if num_roots > 10:
            step = num_roots/10
            if current_root % step == 0:
                Log.add('Analyzing tree {0}/{1} ...'.format(current_root, num_roots))
        # elevate Logging level
        x = Log.min_severity
        Log.set_min_severity(Severity.WARNING)

        # calculate all unique longest path in causal tree
        causal_paths += paths_from_dag(causal_tree, causal_mapping, repetitions=False, max_subpath_length=max_subpath_length)
        current_root += 1

        # restore log level
        Log.set_min_severity(x)
    Log.add('finished.')
    
    return causal_paths

def sample_paths_from_temporal_network_dag(tempnet, delta=1, num_roots=1, max_subpath_length=None):
    """
    Estimates the frequency of causal paths in a temporal network assuming a
    maximum temporal distance of delta between consecutive
    time-stamped links on a path. This method first creates a directed and acyclic
    time-unfolded graph based on the given parameter delta. This directed acyclic
    graph is used to calculate causal paths for a given delta, randomly sampling num_roots
    roots in the time-unfolded DAG.

    Parameters
    ----------
    tempnet : pathpy.TemporalNetwork
        TemporalNetwork to extract the time-respecting paths from
    delta : int
        Indicates the maximum temporal distance up to which time-stamped
        links will be considered to contribute to a causal path.
        For (u,v;3) and (v,w;7) a causal path (u,v,w) is generated
        for 0 < delta <= 4, while no causal path is generated for
        delta > 4. Every time-stamped edge is a causal path of
        length one. Default value is 1.
    num_roots : int
        The number of randomly chosen roots that will be used to estimate path statistics.

    Returns
    -------
    Paths
        An instance of the class Paths, which can be used to generate higher- and multi-order
        models of causal paths in temporal networks.
    """
    # generate a single time-unfolded DAG
    Log.add('Constructing time-unfolded DAG ...')
    dag, node_map = DAG.from_temporal_network(tempnet, delta)
    # dag.topsort()
    # assert dag.is_acyclic
    Log.add('finished.')
    print(dag)

    causal_paths = Paths()
    
    # For each root in the time-unfolded DAG, we generate a
    # causal tree and use it to count all causal paths
    # that originate at this root
    current_root = 1
    Log.add('Generating causal trees for {0} root nodes ...'.format(num_roots))
    import random
    for root in random.sample(dag.roots, num_roots):
        causal_tree, causal_mapping = generate_causal_tree(dag, root, node_map)
        if num_roots > 10:
            step = num_roots/10
            if current_root % step == 0:
                Log.add('Analyzing tree {0}/{1} ...'.format(current_root, num_roots))
        # elevate Logging level
        x = Log.min_severity
        Log.set_min_severity(Severity.WARNING)

        # calculate all unique longest path in causal tree
        causal_paths += paths_from_dag(causal_tree, causal_mapping, repetitions=False, max_subpath_length=max_subpath_length)
        current_root += 1

        # restore log level
        Log.set_min_severity(x)
    Log.add('finished.')
    
    return causal_paths
