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
import sys
from collections import defaultdict
import bisect
import datetime
from time import mktime

import numpy as _np

from pathpy.utils import Log, Severity


class TemporalNetwork:
    """
      This class represents a sequence of time-stamped edges.
       Instances of this class can be used to generate path statistics
       based on the time-respecting paths resulting from a given maximum
       time difference between consecutive time-stamped edges.
    """

    def __init__(self, tedges=None):
        """Constructor that generates a temporal network instance.

        Parameters
        ----------
        tedges:
            an optional list of directed time-stamped edges from which to construct a
            temporal network instance. For the default value None an empty temporal
            network will be created.
        """
        # A list of time-stamped edges of this temporal network
        self.tedges = []

        # A list of nodes of this temporal network
        self.nodes = []

        # A dictionary storing all time-stamped links, indexed by time-stamps
        self.time = defaultdict(lambda: list())

        # A dictionary storing all time-stamped links, indexed by time and target node
        self.targets = defaultdict(lambda: dict())

        # A dictionary storing all time-stamped links, indexed by time and source node
        self.sources = defaultdict(lambda: dict())

        # A dictionary storing time stamps at which links (v,*;t) originate from node v
        self.activities = defaultdict(lambda: list())

        # A dictionary storing sets of time stamps at which links (v,*;t) originate from
        # node v
        # Note that the insertion into a set is much faster than repeatedly checking
        # whether an element already exists in a list!
        self.activities_sets = defaultdict(lambda: set())

        # An ordered list of time-stamps
        self.ordered_times = []

        nodes_seen = defaultdict(lambda: False)

        if tedges is not None:
            Log.add('Building index data structures ...')

            for e in tedges:
                self.activities_sets[e[0]].add(e[2])
                self.time[e[2]].append(e)
                self.targets[e[2]].setdefault(e[1], []).append(e)
                self.sources[e[2]].setdefault(e[0], []).append(e)
                if not nodes_seen[e[0]]:
                    nodes_seen[e[0]] = True
                if not nodes_seen[e[1]]:
                    nodes_seen[e[1]] = True
            self.tedges = tedges
            self.nodes = list(nodes_seen.keys())

            Log.add('Sorting time stamps ...')

            self.ordered_times = sorted(list(self.time.keys()))
            for v in self.nodes:
                self.activities[v] = sorted(self.activities_sets[v])
            Log.add('finished.')

    @classmethod
    def from_sqlite(cls, cursor, directed=True, timestamp_format='%Y-%m-%d %H:%M:%S'):
        """Reads time-stamped links from an SQLite cursor and returns a new instance of
        the class TemporalNetwork. The cursor is assumed to refer to a table that
        minimally has three columns

                source target time

        and where each row refers to a directed link. Time stamps can be integers,
        or strings to be converted to UNIX time stamps via a custom timestamp format.
        For this, the python function datetime.strptime will be used.

        Important: Since columns are accessed by name this function requires that a
        row factory object is set for the SQLite connection prior to cursor creation,
        i.e. you should set

                connection.row_factory = sqlite3.Row

        Parameters
        ----------
        cursor:
            The SQLite cursor to fetch rows
        directed: bool
        timestamp_format: str
            used to convert string timestamps to UNIX timestamps. This parameter is
            ignored, if the timestamps are digit types (like a simple int).

        Returns
        -------

        """
        tedges = []

        assert cursor.connection.row_factory, \
            'Cannot access columns by name. Please set ' \
            'connection.row_factory = sqlite3.Row before creating DB cursor.'

        if not directed:
            Log.add('Retrieving undirected time-stamped links ...')
        else:
            Log.add('Retrieving directed time-stamped links ...')

        for row in cursor:
            # r = sqlite3.Row(row)
            timestamp = row['time']
            assert isinstance(timestamp, int) or isinstance(timestamp, str), \
                'Error: pathpy only supports integer or string timestamps'
            # if the timestamp is a number, we use this
            if isinstance(timestamp, int):
                t = timestamp
            else:
                # if it is a string, we use the timestamp format to convert it to
                # a UNIX timestamp
                x = datetime.datetime.strptime(timestamp, timestamp_format)
                t = int(mktime(x.timetuple()))
            tedges.append((str(row['source']), str(row['target']), t))
            if not directed:
                tedges.append((str(row['target']), str(row['source']), t))

        return cls(tedges=tedges)

    @classmethod
    def read_file(cls, filename, sep=',', directed=True,
                  timestamp_format='%Y-%m-%d %H:%M:%S', maxlines=sys.maxsize):
        """
        Reads time-stamped links from a file and returns a new instance of the class
        TemporalNetwork. The file is assumed to have a header

                source target time

        where columns can be in arbitrary order and separated by arbitrary characters.
        Each time-stamped link must occur in a separate line and links are assumed to be
        directed.

        The time column can be omitted and in this case all links are assumed to occur
        in consecutive time stamps (that have a distance of one). Time stamps can be
        simple integers, or strings to be converted to UNIX time stamps via a custom
        timestamp format. For this, the python function datetime.strptime will be used.

        Parameters
        ----------
        filename: str
            path of the file to read from
        sep: str
            the character that separates columns (default ',')
        directed: bool
            whether to read edges as directed (default True)
        timestamp_format: str
            used to convert string timestamps to UNIX timestamps. This parameter is
            ignored, if timestamps are digit types (like a simple int).
            The default is '%Y-%m-%d %H:%M'
        maxlines: int
            limit reading of file to a given number of lines (default sys.maxsize)

        Returns
        -------

        """
        assert (filename != ''), 'Empty filename given'

        # Read header
        with open(filename, 'r') as f:
            tedges = []

            header = f.readline()
            header = header.split(sep)

            # If header columns are included, arbitrary column orders are supported
            time_ix = -1
            source_ix = -1
            target_ix = -1
            for i in range(len(header)):
                header[i] = header[i].strip()
                if header[i] == 'node1' or header[i] == 'source':
                    source_ix = i
                elif header[i] == 'node2' or header[i] == 'target':
                    target_ix = i
                elif header[i] == 'time' or header[i] == 'timestamp':
                    time_ix = i            

            assert (source_ix >= 0 and target_ix >= 0), \
                "Detected invalid header columns: %s" % header

            if time_ix < 0:  # pragma: no cover
                Log.add('No time stamps found in data, assuming consecutive links',
                        Severity.WARNING)

            if not directed:
                Log.add('Reading undirected time-stamped links ...')
            else:
                Log.add('Reading directed time-stamped links ...')

            line = f.readline()
            n = 1
            while line and n <= maxlines:
                fields = line.rstrip().split(sep)
                try:
                    if time_ix >= 0:
                        timestamp = fields[time_ix]
                        # if the timestamp is a number, we use this
                        if timestamp.isdigit():
                            t = int(timestamp)
                        else:
                            # if it is a string, we use the timestamp format to convert
                            # it to a UNIX timestamp
                            x = datetime.datetime.strptime(timestamp, timestamp_format)
                            t = int(mktime(x.timetuple()))
                    else:
                        t = n
                    if t >= 0 and fields[source_ix] != '' and fields[target_ix] != '':
                        tedge = (fields[source_ix], fields[target_ix], t)
                        tedges.append(tedge)
                        if not directed:
                            tedges.append((fields[target_ix], fields[source_ix], t))
                    else:  # pragma: no cover
                        s_line = line.strip()
                        if fields[source_ix] == '' or fields[target_ix] == '':
                            msg = 'Empty node in line {0}: {1}'.format(n+1, s_line)
                        else: 
                            msg = 'Negative timestamp in line {0}: {1}'.format(n+1, s_line)
                        Log.add(msg, Severity.WARNING)
                except (IndexError, ValueError):  # pragma: no cover
                    s_line = line.strip()
                    msg = 'Malformed line {0}: {1}'.format(n+1, s_line)
                    Log.add(msg, Severity.WARNING)
                line = f.readline()
                n += 1
        # end of with open()

        return cls(tedges=tedges)

    def write_file(self, filename, sep=','):
        """Writes the time-stamped edge list of this temporal network instance as CSV file

        Parameters
        ----------
        filename: str
            name of CSV file to save data to
        sep: str
            character used to separate columns in generated CSV file

        Returns
        -------

        """
        msg = 'Writing {0} time-stamped edges to file {1}'.format(self.ecount(), filename)
        Log.add(msg, Severity.INFO)
        with open(filename, 'w+') as f:
            f.write('source' + sep + 'target' + sep + 'time' + '\n')
            for time in self.ordered_times:
                for (v, w, t) in self.time[time]:
                    f.write(str(v) + sep + str(w) + sep + str(t)+'\n')


    def filter_nodes(self, nodes):
        """Returns a copy of the temporal network where time-stamped edges are filtered 
        according to a given set of nodes.

        Parameters
        ----------
        node_filter: iterable
            a list or set of nodes that shall be included in the returned temporal network

        Returns
        -------
        """
        def edge_filter(v, w, t):
            if v in nodes and w in nodes: 
                return True
            return False

        return self.filter_edges(edge_filter)

    def filter_edges(self, edge_filter):
        """Returns a copy of the temporal network where time-stamped edges are filtered 
        according to a given filter expression. This can be used, e.g., to create time 
        slice networks by filtering edges within certain time windows, or to reduce a 
        temporal network to interactions between a subset of nodes.

        Parameters
        ----------
        edge_filter: callable
            an arbitrary filter function of the form filter_func(v, w, time) that returns
            True for time-stamped edges that shall pass the filter, and False for time-stamped
            edges that shall be filtered out.

        Returns
        -------

        """
        Log.add('Starting filtering ...', Severity.INFO)
        new_t_edges = []

        for (v, w, t) in self.tedges:
            if edge_filter(v, w, t):
                new_t_edges.append((v, w, t))

        n_filtered = self.ecount() - len(new_t_edges)
        msg = 'finished. Filtered out {} time-stamped edges.'.format(n_filtered)
        Log.add(msg,  Severity.INFO)

        return TemporalNetwork(tedges=new_t_edges)

    def add_edge(self, source, target, ts, directed=True):
        """Adds a time-stamped edge (source,target;time) to the temporal network.
        Unless specified otherwise, time-stamped edges are assumed to be directed.

        Parameters
        ----------
        source:
            name of the source node of a directed, time-stamped link
        target:
            name of the target node of a directed, time-stamped link
        ts: int
            time-stamp of the time-stamped link
        directed: bool

        Returns
        -------

        """
        e = (source, target, ts)
        self.tedges.append(e)
        if source not in self.nodes:
            self.nodes.append(source)
        if target not in self.nodes:
            self.nodes.append(target)

        # Add edge to index structures
        self.time[ts].append(e)
        self.targets[ts].setdefault(target, []).append(e)
        self.sources[ts].setdefault(source, []).append(e)

        if ts not in self.activities[source]:
            self.activities[source].append(ts)
            self.activities[source].sort()

        # Maintain order of time stamps
        index = bisect.bisect_left(self.ordered_times, ts)
        # add if ts is not already in list
        if index == len(self.ordered_times) or self.ordered_times[index] != ts:
            self.ordered_times.insert(index, ts)

        # make edge undirected by adding another directed edge
        if not directed:
            self.add_edge(target, source, ts)

    def vcount(self):
        """Returns the number of vertices in the temporal network.
        This number corresponds to the number of nodes in the (first-order)
        time-aggregated network.
        """

        return len(self.nodes)

    def ecount(self):
        """Returns the number of time-stamped edges (u,v;t) in the temporal network.
        This number corresponds to the sum of link weights in the (first-order)
        time-aggregated network.
        """

        return len(self.tedges)

    def observation_length(self):
        """Returns the length of the observation time in time units."""

        return max(self.ordered_times)-min(self.ordered_times)

    def inter_event_times(self):
        """
        Returns an array containing all time differences between any
        two consecutive time-stamped links (involving any node)
        """
        time_diffs = []
        for i in range(1, len(self.ordered_times)):
            time_diffs += [self.ordered_times[i] - self.ordered_times[i-1]]
        return _np.array(time_diffs)

    def inter_path_times(self):
        """Returns a dictionary which, for each node v, contains all time differences
        between any time-stamped link (*,v;t) and the next link (v,*;t') (t'>t) in the
        temporal network
        """
        ip_times = defaultdict(list)
        for e in self.tedges:
            # Get target v of current edge e=(u,v,t)
            v = e[1]
            t = e[2]

            # Get time stamp of link (v,*,t_next)
            # with smallest t_next such that t_next > t
            i = bisect.bisect_right(self.activities[v], t)
            if i != len(self.activities[v]):
                ip_times[v].append(self.activities[v][i]-t)
        return ip_times

    def summary(self):
        """
        Returns a string containing basic summary statistics of this temporal network
        """

        summary = ''

        summary += 'Nodes:\t\t\t' + str(self.vcount()) + '\n'
        summary += 'Time-stamped links:\t' + str(self.ecount()) + '\n'
        if self.vcount() > 0:
            summary += 'Links/Nodes:\t\t' + str(self.ecount()/self.vcount()) + '\n'
        else:
            summary += 'Links/Nodes:\t\tN/A\n'
        if len(self.ordered_times) > 1:
            min_o = min(self.ordered_times)
            max_o = max(self.ordered_times)
            obs_len = max_o - min_o
            n_stamps = len(self.ordered_times)
            summary += 'Observation period:\t[{}, {}]\n'.format(min_o, max_o)
            summary += 'Observation length:\t {} \n'.format(obs_len)
            summary += 'Time stamps:\t\t {} \n'.format(n_stamps)

            d = self.inter_event_times()
            mean_d = _np.mean(d)

            summary += 'Avg. inter-event dt:\t {}\n'.format(mean_d)
            summary += 'Min/Max inter-event dt:\t {}/{}'.format(min(d), max(d))

        return summary

    def __str__(self):
        """Returns the default string representation of this temporal network instance.
        """
        return self.summary()

    def shuffle_edges(self, l=0, with_replacement=False, window_splits=None,
                      maintain_undirected=True):
        """Generates a shuffled version of the temporal network in which edge statistics
        (i.e. the frequencies of time-stamped edges) and inter-event time  statistics are
        preserved, while all order correlations are destroyed by randomly reshuffling the
        time stamps of links.

        Parameters
        ----------
        l: int
            For the default value l=0, the length of the original temporal network is
            used.
        with_replacement: bool
            Whether or not the sampling of new time-stamped edges should be with
            replacement (default False). If False, the exact edge frequencies and
            inter-event time statistics in the original network will be preserved.
        window_splits: list
            a list of time stamps that separate shuffling windows. E.g. specifying
            window_splits = [7,14,21] will separately shuffle edges within intervals
            [min_timestamp,7], (7,14], (14,21], (21,max_timestamp] (default None). The
            number of edges l to generate applies separately for each time window. For
            l=0, the original number of edges in each time window will be used.
        maintain_undirected: bool
            if True, two directed edges (a,b,t) (b,a,t) occurring at the same time will be
            treated as a single undirected edge, i.e. both are reassigned to a different
            time stamp at once (default True). This ensures that undirected edges are
            preserved as atomic objects.

        Returns
        -------

        """
        tedges = []

        if window_splits is None:
            window_splits = [max(self.time)]
        else:
            window_splits.append(max(self.time))

        window_min = min(self.time)-1
        for window_max in window_splits:

            timestamps = []
            edges = []
            for e in self.tedges:
                if window_min < e[2] <= window_max:
                    timestamps.append(e[2])
                    edges.append(e)

            if l == 0:
                l = len(edges)
            if with_replacement:  # sample l edges with replacement
                for i in range(l):
                    # Pick random link
                    edge = edges[_np.random.randint(0, len(edges))]
                    # Pick random time stamp
                    time = timestamps[_np.random.randint(0, len(timestamps))]
                    # Generate new time-stamped link
                    tedges.append((edge[0], edge[1], time))
            else:
                # shuffle edges while avoiding multiple identical edges at same time stamp
                i = 0
                while i < l:
                    # Pick random link
                    edge = edges.pop(_np.random.randint(0, len(edges)))

                    # Pick random time stamp
                    time = timestamps.pop(_np.random.randint(0, len(timestamps)))
                    rewired = False

                    # for undirected edges, rewire both directed edges at once
                    if maintain_undirected and (edge[1], edge[0], edge[2]) in edges:

                        # check whether one of the time-stamped edges already exists
                        if (
                                (edge[0], edge[1], time) not in tedges and
                                (edge[1], edge[0], time) not in tedges
                        ):
                            tedges.append((edge[0], edge[1], time))
                            tedges.append((edge[1], edge[0], time))
                            edges.remove((edge[1], edge[0], edge[2]))
                            rewired = True

                    # rewire a single directed edge individually
                    elif (edge[0], edge[1], time) not in tedges:
                        tedges.append((edge[0], edge[1], time))
                        rewired = True

                    # edge could not be rewired to the chosen time stamp, so re-append
                    # both to the list for future sampling
                    if not rewired:
                        edges.append(edge)
                        timestamps.append(time)
                    else:
                        i += 1

            window_min = window_max

        # Generate temporal network
        t = TemporalNetwork(tedges=tedges)

        # Fix node order to correspond to original network
        t.nodes = self.nodes
        return t

    def write_tikz(self, filename, dag=True, angle=20, layer_dist='0.3cm',
                   split_directions=True):
        """Generates a tex file that can be compiled to a time-unfolded representation of
         the temporal network.

        Parameters
        ----------
        filename: str
            the name of the tex file to be generated.
        dag: bool
            whether or not to draw the unfolded network as a directed acyclic graph,
            in which a link (v,w,t) connects node-time elements (v_t, w_t+1) (default
            True). If False, a simple sequence of links will be generated.
        angle: float
            the angle of curved edges
        layer_dist: str
            LaTex distance parameter string specifying the distance between adjacent
            node-time elements (default '0.3cm')
        split_directions: bool
            whether or not the curve angle of edges shall be split depending on
            direction (default True) If this is set to True, arrows from left to right
            bend upwards, while arrows from right to left bend downwards This helps
            readability in temporal networks with multiple edges per time step. For
            temporal networks with single edges per time, False is recommended.

        Returns
        -------

        """
        import os as _os

        output = [
            '\\documentclass{article}',
            '\\usepackage{tikz}',
            '\\usepackage{verbatim}',
            '\\usepackage[active,tightpage]{preview}',
            '\\PreviewEnvironment{tikzpicture}',
            '\\setlength\\PreviewBorder{5pt}%',
            '\\usetikzlibrary{arrows}',
            '\\usetikzlibrary{positioning}',
            '\\renewcommand{\\familydefault}{\\sfdefault}',
            '\\begin{document}',
            '\\begin{center}',
            '\\newcounter{a}',
            "\\begin{tikzpicture}[->,>=stealth',auto,"
            "scale=1, every node/.style={scale=1}]",
            "\\tikzstyle{node} = [fill=lightgray,text=black,circle]",
            "\\tikzstyle{v} = [fill=lightgray,draw=black,"
            "text=white,circle,minimum size=0.5cm]",
            "\\tikzstyle{dst} = [fill=lightgray,text=black,circle]",
            "\\tikzstyle{lbl} = [text=black,circle]"]

        last = ''

        for n in _np.sort(self.nodes):
            if last == '':
                node = r"\node[lbl]  ({n}-0) {{\bf \Huge {n} }};".format(n=n)
            else:
                node_fmt = r"\node[lbl, right=0.4cm of {last}-0] ({n}-0) " \
                           r"{{\bf \Huge {n} }};"
                node = node_fmt.format(last=last, n=n)

            output.append(node)
            last = n

        output.append("\\setcounter{a}{0}\n")
        min_t = min(self.ordered_times)
        max_t = max(self.ordered_times)
        if dag:
            dag_num = r'\foreach \number in {{ {min_t}, ..., {max_t} }} {{'
            output.append(dag_num.format(min_t=min_t, max_t=max_t+1))
        else:
            dag_num = r'\foreach \number in {{ {min_t}, ..., {max_t} }} {{'
            output.append(dag_num.format(min_t=min_t, max_t=max_t))
        output.append("\\setcounter{a}{\\number}")
        output.append("\\addtocounter{a}{-1}")
        output.append("\\pgfmathparse{\\thea}")

        for n in _np.sort(self.nodes):
            layer_fm = r"\node[v,below={layer} of {n}-\pgfmathresult] ({n}-\number) {{}};"
            output.append(layer_fm.format(layer=layer_dist, n=n))

        first_node = _np.sort(self.nodes)[0]
        fmt = r'\node[lbl,left=0.4cm of {f_node}-\number] (col-\pgfmathresult) ' \
              r'{{ \selectfont {{ \bf \Huge \number}} }};'

        output.append(fmt.format(f_node=first_node))
        output.append(r"}")
        output.append(r"\path[->,line width=2pt]")
        # draw only directed edges
        for ts in self.ordered_times:
            for edge in self.time[ts]:
                if dag:
                    edge_str = r'({}-{}) edge ({}-{})'.format(edge[0], ts, edge[1], ts+1)
                    output.append(edge_str)
                else:
                    if (edge[1], edge[0], ts) not in self.time[ts]:
                        bend_direction = 'right'
                        if not split_directions and edge[0] < edge[1]:
                            bend_direction = 'left'
                        edge_fmt = r'({}-{}) edge[bend {} = {}] ({}-{})'
                        edge_str = edge_fmt.format(edge[0], ts, bend_direction, angle,
                                                   edge[1], ts)
                        output.append(edge_str)
        output.append(";\n")

        # separately draw undirected edges if we don't output a DAG
        if not dag:
            output.append("\\path[-,line width=2pt]\n")
            for ts in self.ordered_times:
                for edge in self.time[ts]:
                    if (edge[1], edge[0], ts) in self.time[ts]:
                        # admittedly, this is an ugly trick: I avoid keeping state on
                        # which of the directed edges has been drawn already as an
                        # undirected edge, by simply drawing them twice in the same way
                        #  :-)
                        s = max(edge[0], edge[1])
                        t = min(edge[0], edge[1])
                        edge = "({s}-{ts}) edge[bend right={angle}] ({t}={ts})\n"
                        output.append(edge.format(s=s, ts=ts, angle=angle, t=t))
            output.append(";\n")
        output.append("\\end{tikzpicture} \n"
                      "\\end{center} \n"
                      "\\end{document}")

        # create directory if necessary to avoid IO errors
        directory = _os.path.dirname(filename)
        if directory != '':
            if not _os.path.exists(directory):
                _os.makedirs(directory)

        with open(filename, "w") as tex_file:
            tex_file.writelines('\n'.join(output))


    def _to_html(self, width=600, height=600, ms_per_frame=50, ts_per_frame=20, radius=6,
                look_behind=1500, look_ahead=150, template_file=None, **kwargs):
        """
        Generates an html snippet with an interactive d3js visualisation of the
        temporal network. This function can be used to embed interactive visualisations
        into a jupyter notebook, or to export stand-alone html visualisations based on
        a customizable template.

        Parameters
        ----------
        width:  int
            the width of the div that contains the visualisation (default 600)
        height: int
            the height of the div that contains the visualisation (default 600)
        ms_per_frame: int
            how many milliseconds each frame of the visualisation shall be displayed.
            The inverse of this value gives the framerate of the resulting visualisation.
            The default value of 100 yields a framerate of 10 fps
        ts_per_frame: int
            how many timestamps in the temporal network shall be displayed in every frame
            of the visualisation (default 1). For the default value of 1, each timestamp
            is shown in a new frame. For higher values, multiple timestamps will be aggregated
            in a single frame.
        radius: int
            radius of nodes in the visualisation. Unfortunately this can only be set via
            the style file starting from SVG2.
        template_file: str
            path to the template file that shall be used to output the html visualisation (default None)
            If a custom-tailored template_file is specified, python's string.Template mechanism is used
            to replace the following strings by JavaScript variables:
                $network_data:  replaced by a JSON dictionary that contains nodes and time-stamped links
                $width: width of the DIV to be generated
                $height: height of the DIV to be generated
                $msperframe: see above
                $tsperframe: see above
                $div_id: a unique ID for the generated DIV. This is important when including
                    multiple visualisations into a single output file (e.g. in multiple jupyter cells)
            If this is set to None (default value), a default template provided by pathpy will be used.
        use_requirejs: bool
            whether or not the generated html shall import d3js via the requirejs framework
            (default True). The use of requirejs is needed to include html inside a jupyter
            notebook. For the generation of stand-alone files, the value should be set to
            False. This parameter will be ignored when using a custom templatefile
        **kwargs: keyword args
            arbitrary key-value pairs, that will be exported (via json.dumps) to the corresponding
            placeholder values in a custom template-file. As an example, if the template file contains
            JavaScript code like x=f($x); and console.log($y);, and we set parameters
            write_html(..., x=42, y='knockknock'), the exported JavaScript will contain x=f(42) and
            console.log('knockknock').
        Returns
        -------

        """
        import json
        import os
        from string import Template

        import string
        import random

        # auto-adjust simulation speed to temporal characteristics
        if ts_per_frame == 0:
            d = self.inter_event_times()
            avg_ts_bw_interactions = _np.mean(d)
            fps = 1000.0/float(ms_per_frame)
            x = avg_ts_bw_interactions/fps
            # set time scale so that we expect 5 interactions per frame
            ts_per_frame = _np.max([1, int(20 * x)])

        div_id = "".join(random.choice(string.ascii_letters) for x in range(8))

        # prefix nodes starting with number
        def fix_node_name(v):
            new_v = v
            if v[0].isdigit():
                new_v = "n_" + v
            if new_v[0] == '_':
                new_v = "n_" + v
            if '-' in new_v:
                new_v = new_v.replace('-', '_')
            return new_v

        network_data = {
            'nodes': [{'id': fix_node_name(v), 'group': 1} for v in self.nodes],
            'links': [{'source': fix_node_name(s),
                       'target': fix_node_name(v),
                       'value': 1,
                       'time': t} for s, v, t in self.tedges
                      ]
        }

        # use standard template if no custom template is specified
        if template_file is None:
            module_dir = os.path.dirname(os.path.realpath(__file__))
            html_dir = os.path.join(module_dir, os.path.pardir, 'html_templates')
            template_file = os.path.join(html_dir, 'tempnet.html')            

        with open(template_file) as f:
            html_str = f.read()

        default_args = {
            'network_data': json.dumps(network_data),
            'width': width,
            'height': height,
            'div_id': div_id,
            'msperframe': ms_per_frame,
            'tsperframe': ts_per_frame,
            'radius': radius,
            'look_ahead': look_ahead,
            'look_behind': look_behind
        }

        # replace all placeholders in the template
        html = Template(html_str).substitute({**default_args, **kwargs})

        return html


    def _repr_html_(self, width=800, height=800, msperframe=50, tsperframe=20, radius=6,
            look_behind=1500, look_ahead=150, template_file=None, **kwargs):
        from IPython.core.display import display, HTML
        display(HTML(self._to_html(width, height, msperframe, tsperframe=tsperframe, radius=radius,
            template_file=template_file, **kwargs)))


    def write_html(self, filename, width=800, height=800, msperframe=50, tsperframe=20, radius=6,
            look_behind=1500, look_ahead=150, template_file=None, **kwargs):

        html = self._to_html(width, height, msperframe, tsperframe=tsperframe, radius=radius,
            template_file=template_file, **kwargs)

        # for the inner HTML generated from the default templates, we add the surrounding DOCTYPE and body
        # needed for a stand-alone HTML file
        if template_file is None:
            html = '<!DOCTYPE html>\n<html><body>\n' + html + '</body>\n</html>'
        with open(filename, 'w+') as f:
            f.write(html)
