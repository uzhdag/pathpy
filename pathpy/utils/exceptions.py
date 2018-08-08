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
Classes to signal errors and exceptions in pathpy.
"""

class PathpyException(Exception):
    """
    Base class for exceptions in Pathpy.
    """


class PathpyError(PathpyException):
    """
    Exception for a serious error in Pathpy
    """


class PathpyNotImplemented(PathpyException):
    """
    Exception for procedure not implemented in pathpy
    """


class EmptySCCError(PathpyException):
    """
    This exception is thrown whenever a non-empty strongly
    connected component is needed, but when we encounter an empty one.
    """


class PathsTooShort(PathpyException):
    """
    This exception if thrown if available paths are too
    short for the requested operation.
    """
