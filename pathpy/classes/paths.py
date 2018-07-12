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
from collections import defaultdict
import sys
import copy

import numpy as np
from pathpy.utils import Log, Severity
from pathpy.utils.exceptions import PathpyError
from pathpy.utils.default_containers import nested_zero_default as _nested_zero_default
from pathpy.utils.default_containers import zero_array_default as _zero_array_default


class Paths:
    """
    Instances of this class represent path statistics
    which can be analyzed using higher- and multi-order network
    models. The origin of the path statistics can be (i) n-gram
    files which provide us with a list of paths in terms of n-grams
    of varying lengths, or (ii) a temporal network instance which
    provides us with a set of time-respecting paths based on a
    given maximum time difference delta.
    """

    def __init__(self, separator=','):
        """
        Creates an empty Paths object
        """

        # A dictionary of paths that has the following structure:
        # - paths[k] is a dictionary containing all paths of length k,
        #    indexed by a path tuple p = (u,v,w,...)
        # - for each tuple p of length k, paths[k][p] contains a tuple
        #    (i,j) where i refers to the number of times p occurs as a
        #    subpath of a longer path, and j refers to the number of times p
        #    occurs as a *real* or *longest* path (i.e. not being a subpath
        #    of a longer path)
        self.paths = _nested_zero_default()

        # The character used to separate nodes on paths
        self.separator = separator

        # This can be used to limit the calculation of sub path statistics to a given
        # maximum length. This is useful, as the statistics of sub paths of length k
        # are only needed to fit a higher-order model with order k. Hence, if we know
        # that the model selection is limited to a given maximum order K, we can safely
        #  set the maximum sub path length to K. By default, sub paths of any length
        # will be calculated. Note that, independent of the sub path calculation
        # longest path of any length will be considered in the likelihood calculation!
        self.max_subpath_length = sys.maxsize

    def summary(self):
        """

        Returns
        -------
        str
            Returns a string containing basic summary info of this Paths instance
        """
        total_paths = []
        sub_path_sum = []
        l_path_sum = []
        max_path_length = 0
        average_length = 0
        for k in sorted(self.paths):
            paths_ = self.paths[k]
            values_ = np.array(list(paths_.values()))
            v_0 = np.sum(values_[:, 0])
            v_1 = np.sum(values_[:, 1])
            total_paths += [v_0 + v_1]
            sub_path_sum += [v_0]
            l_path_sum += [v_1]
            average_length += v_1 * k
            if paths_:
                max_path_length = max(max_path_length, k)
        if np.sum(l_path_sum) > 0:
            average_length = average_length / np.sum(l_path_sum)

        summary_fmt = (
            "Total path count: \t\t{lpsum} \n"
            "[Unique / Sub paths / Total]: \t[{unique_paths} / {spsum} / {total_paths}]\n"
            "Nodes:\t\t\t\t{len_nodes} \n"
            "Edges:\t\t\t\t{len_edges}\n"
            "Max. path length:\t\t{maxL}\n"
            "Avg path length:\t\t{avgL} \n"
        )

        k_path_info_fmt = 'Paths of length k = {k}\t\t{lpsum} ' \
                          '[ {unique_paths_longer} / {spsum} / {total_paths} ]\n'

        # Count number of nodes and edges
        if 0 not in self.paths:
            len_0 = 0
        else:
            len_0 = len(self.paths[0])
        if 1 not in self.paths:
            len_1 = 0
        else:
            len_1 = len(self.paths[1])

        summary_info = {
            "lpsum": np.sum(l_path_sum),
            "unique_paths": self.unique_paths(),
            "spsum": np.sum(sub_path_sum),
            "total_paths": np.sum(total_paths),
            "len_nodes": len_0,
            "len_edges": len_1,
            "maxL": max_path_length,
            "avgL": average_length
        }

        summary = summary_fmt.format(**summary_info)

        for k in sorted(self.paths):
            k_info = k_path_info_fmt.format(
                k=k, lpsum=l_path_sum[k], spsum=sub_path_sum[k],
                total_paths=total_paths[k],
                unique_paths_longer=self.unique_paths(l=k, consider_longer_paths=False)
            )
            summary += k_info

        return summary

    def path_lengths(self):
        """compute the length of all paths

        Returns
        -------
        dict
            Returns a dictionary containing the distribution of path lengths
            in this Path object. In the returned dictionary, entry
            lengths ``k`` is a ``numpy.array`` ``x`` where
            ``x[0]`` is the number of sub paths with length ``k``, and ``x[1]``
            is the number of (longest) paths with length ``k``


        """
        lengths = _zero_array_default()

        for k in self.paths:
            for p in self.paths[k]:
                lengths[k] += self.paths[k][p]
        return lengths

    def __add__(self, other):
        """add path statistics of one object to the other

        Parameters
        ----------
        other : Paths

        Returns
        -------
        Paths
            Default operator +, which returns the sum of two Path objects
        """
        p_sum = Paths()
        p_sum.paths = copy.deepcopy(self.paths)
        for p_length in other.paths:
            for p in other.paths[p_length]:
                p_sum.paths[p_length][p] += other.paths[p_length][p]
        return p_sum

    def __iadd__(self, other):
        """in place addition avoids unnecessary copies of the object

        Parameters
        ----------
        other

        Returns
        -------
        None

        """
        for p_length in other.paths:
            for p in other.paths[p_length]:
                self.paths[p_length][p] += other.paths[p_length][p]
        return self

    def __mul__(self, factor):
        """multiplies all path statistics by factor

        Parameters
        ----------
        factor

        Returns
        -------
        a Paths object with multiplied frequencies

        """
        p_mult = Paths()
        for p_length in self.paths:
            for p in self.paths[p_length]:
                p_mult.paths[p_length][p] = self.paths[p_length][p] * factor

        return p_mult

    def __rmul__(self, factor):
        """right multiply"""
        return self * factor

    def __imul__(self, factor):
        """in-place scaling of path statistics

        Parameters
        ----------
        factor

        Returns
        -------
        None


        """
        for l in self.paths:
            for p in self.paths[l]:
                self.paths[l][p] = self.paths[l][p] * factor

        return self

    def sequence(self, stop_char='|'):
        """

        Parameters
        ----------
        stop_char : str
            the character used to separate paths

        Returns
        -------
        tuple:
            Returns a single sequence in which all paths have been concatenated.
            Individual paths are separated by a stop character.
        """
        Log.add('Concatenating paths to sequence ...')
        sequence = []
        for p_length in self.paths:
            for p in self.paths[p_length]:
                segment = []
                for s in p:
                    segment.append(s)
                if stop_char != '':
                    segment.append(stop_char)
                for _ in range(int(self.paths[p_length][p][1])):
                    sequence += segment

        Log.add('finished')
        return sequence

    def unique_paths(self, l=0, consider_longer_paths=True):
        """Returns the number of unique paths of a given length l (and possibly longer)

        Parameters
        ----------
        l : int
            the (inclusive) maximum length up to which path shall be counted.
        consider_longer_paths : bool
            TODO: add parameter description

        Returns
        -------
        int
            number of unique paths satisfying parameter ``l``
        """
        num_l = 0.0
        if not self.paths:
            return num_l

        max_length = l
        if consider_longer_paths:
            max_length = max(self.paths) if self.paths else 0
        for j in range(l, max_length + 1):
            for p in self.paths[j]:
                if self.paths[j][p][1] > 0:
                    num_l += 1.0
        return num_l

    def __str__(self):
        """
        Returns the default string representation of
        this Paths instance
        """
        return self.summary()

    @property
    def nodes(self):
        """
        Returns the list of nodes for the underlying
        set of paths
        """
        nodes = set()
        for p in self.paths[0]:
            nodes.add(p[0])
        return nodes

    @staticmethod
    def read_edges(filename, separator=',', weight=False, undirected=False,
                   maxlines=None, expand_sub_paths=True, max_subpath_length=None):
        """
        Read path in edgelist format

        Reads data from a file containing multiple lines of *edges* of the
        form "v,w,frequency,X" (where frequency is optional and X are
        arbitrary additional columns). The default separating character ','
        can be changed. In order to calculate the statistics of paths of any length,
        by default all subpaths of length 0 (i.e. single nodes) contained in an edge
        will be considered.

        Parameters
        ----------
        filename : str
            path to edgelist file
        separator : str
            character separating the nodes
        weight : bool
            is a weight given? if ``True`` it is the last element in the edge
            (i.e. ``a,b,2``)
        undirected : bool
            are the edges directed or undirected
        maxlines : int
            number of lines to read (useful to test large files)
        expand_sub_paths : bool
        max_subpath_length : int (default None)
            maximum length for subpaths to consider, ``None`` means the entire file is
            read
            TODO: this parameter is unused.

        Returns
        -------
        Paths
            a ``Paths`` object obtained from the edgelist
        """
        p = Paths()

        p.separator = separator
        p.max_subpath_length = sys.maxsize

        with open(filename, 'r') as f:
            Log.add('Reading edge data ... ')
            for n, line in enumerate(f):
                fields = line.rstrip().split(separator)
                assert len(fields) >= 2, 'Error: malformed line: {0}'.format(line)
                path = (fields[0], fields[1])

                frequency = int(fields[2]) if weight else 1

                p.paths[1][path] += (0, frequency)
                if undirected:
                    reverse_path = (fields[1], fields[0])
                    p.paths[1][reverse_path] += (0, frequency)

                if maxlines is not None and n >= maxlines:
                    break
        if expand_sub_paths:
            p.expand_subpaths()
        Log.add('finished.')

        return p

    @classmethod
    def read_file(cls, filename, separator=',', frequency=False, maxlines=sys.maxsize,
                  max_ngram_length=sys.maxsize, expand_sub_paths=True,
                  max_subpath_length=sys.maxsize):
        """Read path data in ngram format.

        Reads path data from a file containing multiple lines of n-grams of the form
        ``a,b,c,d,frequency`` (where frequency is optional).
        The default separating character ',' can be changed. Each n-gram will be
        interpreted as a path of
        length n-1, i.e. bigrams a,b are considered as path of length one, trigrams a,
        b,c as path of length two, etc. In order to calculate the statistics of paths
        of any length, by default all subpaths of length k < n-1 contained in an n-gram
        will be considered. I.e. for n=4 the four-gram a,b,c,d will be considered as a
        single (longest) path of length n-1 = 3 and three subpaths a->b, b->c, c->d of
        length k=1 and two subpaths a->b->c amd b->c->d of length k=2 will be
        additionally counted.

        Parameters
        ----------
        filename : str
            path to the n-gram file to read the data from
        separator : str
            the character used to separate nodes on the path, i.e. using a
            separator character of ';' n-grams are represented as ``a;b;c;...``
        frequency : bool
            if set to ``True``, the last entry in each n-gram will be interpreted as
            weight (i.e. frequency of the path), e.g. ``a,b,c,d,4`` means that four-gram
            ``a,b,c,d`` has weight four. ``False`` by default, which means each path
            occurrence is assigned a default weight of 1 (adding weights for multiple
            occurrences).
        maxlines : int
            number of lines/n-grams to read, if left at None the whole file is read in.
        max_ngram_length : int
            The maximum n for the n-grams to read, i.e. setting max_ngram_length to 15
            will ignore
            all n-grams of length 16 and longer, which means that only paths up to length
            n-1 are considered.
        expand_sub_paths : bool
            by default all subpaths of the given n-grams are generated, i.e.
            for an input file with a single trigram a;b;c a path a->b->c of length two
            will be generated as well as two subpaths a->b and b->c of length one
        max_subpath_length : int

        Returns
        -------
        Paths
            a ``Paths`` object obtained from the n-grams file
        """
        assert filename != "", 'Empty filename given'

        # If subpath expansion is applied, we keep the information how many times a path
        # has been observed as a subpath, and how many times as a "real" path

        p = cls()

        p.max_subpath_length = max_subpath_length
        p.separator = separator
        max_length = 0

        with open(filename, 'r') as f:
            Log.add('Reading ngram data ... ')
            line = f.readline()
            n = 1
            while line and n <= maxlines:
                fields = line.rstrip().split(separator)
                path = ()
                # Add frequency of "real" path to second component of occurrence counter
                if frequency:
                    for i in range(0, len(fields) - 1):
                        # Omit empty fields
                        v = fields[i].strip()
                        if v:
                            path += (v,)                                         
                    freq = float(fields[len(fields) - 1])
                    if freq >0:
                        if len(path) <= max_ngram_length:
                            p.paths[len(path) - 1][path] += (0, freq)
                            max_length = max(max_length, len(path) - 1)
                        else:  # cut path at max_ngram_length
                            mnl = max_ngram_length
                            p.paths[mnl - 1][path[:mnl]] += (0, freq)
                            max_length = max(max_length, max_ngram_length - 1)
                    else:
                        Log.add('Non-positive path count in line {0}'.format(n), Severity.WARNING)
                else:
                    for field in fields:
                        # Omit empty fields
                        v = field.strip()
                        if v:
                            path += (v,)
                    if len(path) <= max_ngram_length:
                        p.paths[len(path) - 1][path] += (0, 1)
                        max_length = max(max_length, len(path) - 1)
                    else:  # cut path at max_ngram_length
                        p.paths[max_ngram_length - 1][path[:max_ngram_length]] += (0, 1)
                        max_length = max(max_length, max_ngram_length - 1)
                line = f.readline()
                n += 1
        # end of with open()
        Log.add(
            'finished. Read ' + str(n - 1) + ' paths with maximum length ' + str(max_length))

        if expand_sub_paths:
            p.expand_subpaths()
        Log.add('finished.')

        return p

    def write_file(self, filename, separator=','):
        """Writes path statistics data to a file. Each line in this file captures a
        longest path (v0,v1,...,vl), as well as its frequency f as follows

        Parameters
        ----------
        filename: str
            name of the file to write to
        separator: str
            character that shall be used to separate nodes and frequencies

        Returns
        -------

        """
        with open(filename, 'w') as f:
            for p_length in self.paths:
                for p in self.paths[p_length]:
                    if self.paths[p_length][p][1] > 0:
                        line = ""
                        for x in p:
                            line += x
                            line += separator
                        line += str(self.paths[p_length][p][1])
                        f.write(line + '\n')
        f.close()

    @property
    def observation_count(self):
        """
        Returns the total number of observed pathways of any length
        (includes multiple observations for paths with a frequency weight)
        """

        obs_count = 0
        for k in self.paths:
            for p in self.paths[k]:
                obs_count += self.paths[k][p][1]
        return obs_count

    def expand_subpaths(self):
        """
        This function implements the sub path expansion, i.e.
        for a four-gram a,b,c,d, the paths a->b, b->c, c->d of
        length one and the paths a->b->c and b->c->d of length
        two will be counted.

        This process will consider restrictions to the maximum
        sub path length defined in self.max_subpath_length
        """

        # nothing to see here ...
        if not self.paths:
            return

        Log.add('Calculating sub path statistics ... ')

        # the expansion of all subpaths in paths with a maximum path length of maxL
        # necessarily generates paths of *any* length up to MaxL.
        # Forcing the generation of all these indices here, prevents us
        # from mutating indices during subpath creation. The fact that indices are
        # immutable allows us to use efficient iterators and prevent unnecessarily copying

        # Thanks to the use of defaultdict, the following trick will prevent us from
        # repeatedly testing whether l already exists as a key
        for p_length in range(max(self.paths)):
            self.paths[p_length] = self.paths[p_length]

        # expand subpaths in paths of any length ...
        for path_length in self.paths:
            for path, value in self.paths[path_length].items():

                # The frequency is given by the number of occurrences as longest
                # path, which is stored in the second entry of the numpy array
                frequency = value[1]

                # compute maximum length of sub paths to consider
                # (maximum up to pathLength)
                max_length = min(self.max_subpath_length + 1, path_length)

                # Generate all subpaths of length k for k = 0 to k = max_len-1 (inclusive)
                for k in range(max_length):
                    # Generate subpaths of length k for all start indices s
                    # for s = 0 to s = pathLength-k (inclusive)
                    for s in range(path_length - k + 1):
                        # Add frequency as a subpath to *first* entry of occurrence
                        # counter
                        path_slice = path[s:s + k + 1]
                        self.paths[k][path_slice][0] += frequency

    def add_path_tuple(self, path, expand_subpaths=True, frequency=(0, 1)):
        """Adds a tuple of elements as a path. If the elements are not strings,
        a conversion to strings will be made. This function can be used to
        to set custom subpath statistics, via the frequency tuple (see below).

        Parameters
        ----------
        path: tuple
            The path tuple to be added, e.g. ('a', 'b', 'c')
        expand_subpaths: bool
            Whether or not to calculate subpath statistics for this path
        frequency: tuple
            A tuple (x,y) indicating the frequency of this path as subpath
            (first component) and longest path (second component). Default is (0,1).

        Returns
        -------

        """
        assert path, 'Error: paths needs to contain at least one element'

        for x in path:
            if isinstance(x, str) and self.separator in x:
                raise PathpyError('Error: Node name contains separator character. '
                                  'Choose different separator.')

        path_str = path if isinstance(path, str) else tuple(map(str, path))

        self.paths[len(path) - 1][path_str] += frequency

        if expand_subpaths:

            max_length = min(self.max_subpath_length + 1, len(path_str) - 1)

            for k in range(0, max_length):
                for s in range(len(path_str) - k):
                    # for all start indices from 0 to n-k

                    subpath = ()
                    # construct subpath
                    for i in range(s, s + k + 1):
                        subpath += (path_str[i],)
                    # add subpath weight to first component of occurrences
                    self.paths[k][subpath][0] += frequency[1]

    def add_path_ngram(self, ngram, frequency=(0, 1), separator=',', expand_subpaths=True):
        """Adds the path(s) of a single n-gram to the path statistics object.

        Parameters
        ----------
        ngram: str
            An ngram representing a path between nodes, separated by the separator
            character, e.g. the 4-gram a;b;c;d represents a path of length three
            (with separator ';')
        frequency: tuple, int
            the number of occurrences (i.e. frequency) of the ngram (n_subpaths, n_observed)
            the default is (0, 1) (i.e. 0 subpaths and one observed longest path). If an integer x
            is passed, it will be automatically converted to (0, x).
        separator: str
            The character used as separator for the ngrams (';' by default)
        expand_subpaths: bool
            by default all subpaths of the given ngram are generated, i.e.
            for the trigram a;b;c a path a->b->c of length two will be generated
            as well as two subpaths a->b and b->c of length one        

        """
        path = tuple(ngram.split(separator))
        for x in path:
            if isinstance(x, str) and self.separator in x:
                raise PathpyError('Error: Node name contains separator character.'
                                  'Choose different separator.')

        path_length = len(path) - 1

        # add the occurrences as *longest* path to the second component of the numpy array
        if isinstance(frequency, int):
            frequency = (0, frequency)
        self.paths[path_length][path][1] += frequency[1]

        if expand_subpaths:
            max_length = min(self.max_subpath_length + 1, path_length)
            for k in range(0, max_length):
                for s in range(len(path) - k):
                    # for all start indices from 0 to n-k

                    subpath = ()
                    # construct subpath
                    for i in range(s, s + k + 1):
                        subpath += (path[i],)
                    # add subpath weight to first component of occurrences
                    self.paths[k][subpath][0] += frequency[1]

    @staticmethod
    def contained_paths(p, node_filter):
        """Returns the set of maximum-length sub-paths of the path p, which only contain
        nodes that appear in the node_filter. As an example, for the path (a,b,c,d,e,f,g)
        and a node_filter [a,b,d,f,g], the method will return [(a,b), (d,), (f,g)].

        Parameters
        ----------
        p: tuple
            a path tuple to check for contained paths
        node_filter: set
            a set of nodes to which the contained paths should be limited

        Returns
        -------
        """

        contained_paths = []
        current_path = ()
        for node in p:
            if node in node_filter:
                current_path += (node,)
            else:
                if current_path:
                    contained_paths.append(current_path)
                    current_path = ()
        if current_path:
            contained_paths.append(current_path)

        return contained_paths

    def filter_nodes(self, node_filter, min_length=0, max_length=sys.maxsize):
        """Returns a new paths object which contains only paths between nodes in a given
        filter set. For each of the paths in the current Paths object, the set of
        maximally contained subpaths between nodes in node_filter is extracted.
        This method is useful when studying (sub-)paths passing through a subset of nodes.

        Parameters
        ----------
        node_filter: set
            the nodes for which paths with be extracted from the current set of paths
        min_length: int
            the minimum length of paths to extract (default 0)
        max_length: int
            the maximum length of paths to extract (default sys.maxsize)

        Returns
        -------
        Paths
        """
        p = Paths()
        for p_length in self.paths:
            for x in self.paths[p_length]:
                if self.paths[p_length][x][1] > 0:
                    # determine all contained subpaths which only pass through
                    # nodes in node_filter
                    contained = Paths.contained_paths(x, node_filter)
                    for s in contained:
                        if min_length <= len(s) - 1 <= max_length:
                            freq = (0, self.paths[p_length][x][1])
                            p.add_path_tuple(s, expand_subpaths=True, frequency=freq)
        return p

    def project_paths(self, mapping):
        """Returns a new path object in which nodes have been mapped to different labels
        given by an arbitrary mapping function. For instance, for the mapping
        {'a': 'x', 'b': 'x', 'c': 'y', 'd': 'y'} the path (a,b,c,d) is mapped to
        (x,x,y,y). This is useful, e.g., to map page page click streams to topic
        click streams, using a mapping from pages to topics.

        Parameters
        ----------
        mapping: dict
            a dictionary that maps nodes to the new labels

        Returns
        -------

        """
        p = Paths()
        p.max_subpath_length = self.max_subpath_length
        for p_length in self.paths:
            for x in self.paths[p_length]:
                # if this path occurred as longest path
                if self.paths[p_length][x][1] > 0:
                    # construct projected path
                    new_p = ()
                    for v in x:
                        new_p += (mapping[v],)
                    # add to new path object and expand sub paths
                    freq = (0, self.paths[p_length][x][1])
                    p.add_path_tuple(new_p, expand_subpaths=True, frequency=freq)
        return p
