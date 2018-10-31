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
from pathpy.utils import Log, Severity
from pathpy.classes import Network

class RollingTimeWindow:
    r"""
    An iterable rolling time window that can be used to perform time slice
    analysis of time-stamped network data.
    """

    def __init__(self, temporal_net, window_size, step_size=1, directed=True, return_window=False):
        r"""
        Initialises a RollingTimeWindow instance that can be used to
        iterate through a sequence of time-slice networks for a given
        TemporalNetwork instance.

        Parameters:
        -----------
        temporal_net:   TemporalNetwork
            TemporalNetwork instance that will be used to generate the
            sequence of time-slice networks.
        window_size:    int
            The width of the rolling time window used to create
            time-slice networks.
        step_size:      int
            The step size in time units by which the starting time of the rolling
            window will be incremented on each iteration. Default is 1.
        directed:       bool
            Whether or not the generated time-slice networks should be directed.
            Default is true.
        return_window: bool
            Whether or not the iterator shall return the current time window
            as a second return value. Default is False.

        Returns
        -------
        RollingTimeWindow
            An iterable sequence of tuples Network, [window_start, window_end]

        Examples
        --------
            >>> t = pathpy.TemporalNetwork.read_file(DATA)
            >>>
            >>> for n in pathpy.RollingTimeWindow(t, window_size=100):
            >>>     print(n)
            >>>
            >>> for n, w in pathpy.RollingTimeWindow(t, window_size=100, step_size=10, return_window=True):
            >>>     print('Time window starting at {0} and ending at {1}'.format(w[0], w[1]))
            >>>     print(network)
        """
        self.temporal_network = temporal_net
        self.window_size = window_size
        self.step_size = step_size
        self.current_time = min(temporal_net.ordered_times)
        self.max_time = max(temporal_net.ordered_times)
        self.directed = directed
        self.return_window = return_window

    def __iter__(self):
        return self


    def __next__(self):
        if self.current_time+self.window_size <= self.max_time:
            time_window = [self.current_time, self.current_time+self.window_size]
            n = Network.from_temporal_network(self.temporal_network, min_time=self.current_time,
                                              max_time=self.current_time+self.window_size,
                                              directed=self.directed)
            self.current_time += self.step_size
            if self.return_window:
                return n, time_window
            else:
                return n
        else:
            raise StopIteration()
