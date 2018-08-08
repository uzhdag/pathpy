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

def svg_to_pdf(svg_file, output_file):
    """
    Method to convert an SVG file to a PDF file, suitable for
    scholarly publications. This method requires the third-party library
    svglib.
    """
    # uses svglib to render a SVG to PDF
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF

    drawing = svg2rlg(svg_file)
    renderPDF.drawToFile(drawing, output_file)


def svg_to_png(svg_file, output_file):
    """
    Method to convert an SVG file to a PNG file. This method
    requires the third-party library svglib.
    """
    # uses svglib to render a SVG to PDF
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM

    drawing = svg2rlg(svg_file)
    renderPM.drawToFile(drawing, output_file, fmt='PNG')
