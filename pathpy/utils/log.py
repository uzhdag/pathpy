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
import enum
from datetime import datetime
import sys


__all__ = ["Severity", "Log"]


class Severity(enum.IntEnum):
    """ An enumeration that can be used to indicate
        the severity of log messages, and which can be
        used to filter messages based on severities.
    """

    # Error messages
    ERROR = 4

    # Warning messages
    WARNING = 3

    # Informational messages (default minimum level)
    INFO = 2

    # Messages regarding timing and performance
    TIMING = 1

    # Debug messages (really verbose)
    DEBUG = 0


class Log:
    """ A simple logging class, that allows to select what messages should
        be recorded in the output, and where these message should be directed.
    """

    # the output stream to which log entries will be written
    output_stream = sys.stdout

    # The minimum severity level of messages to be logged
    min_severity = Severity.INFO

    @staticmethod
    def set_min_severity(severity):  # pragma: no cover
        """ Sets the minimum sveerity level a message
        needs to have in order to be recorded in the output stream.
        By default, any message which has a severity of at least
        Severity.INFO will be written to the output stream. All messages
        with lower priority will be surpressed.
        """
        Log.min_severity = severity

    @staticmethod
    def set_output_stream(stream):  # pragma: no cover
        """ Sets the output stream to which all messages will be
            written. By default, this is sys.stdout, but it can be
            changed in order to redirect the log to a logfile.
        """
        Log.output_stream = stream

    @staticmethod
    def add(msg, severity=Severity.INFO):  # pragma: no cover
        """ Adds a message with the given severity to the log. This message will be written
            to the log output stream, which by default is sys.stdout. A newline character
            will be added to the message by default.
        """
        if severity >= Log.min_severity:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            Log.output_stream.write(ts + ' [' + str(severity) + ']\t' + msg + '\n')
            Log.output_stream.flush()
