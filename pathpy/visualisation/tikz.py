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

import numpy as _np

from pathpy.classes.network import Network
from pathpy.classes.temporal_network import TemporalNetwork


@singledispatch
def export_tikz(tempnet, filename, dag=True, angle=20, layer_dist='0.3cm',
                   split_directions=True):
        """Generates a tex file that can be compiled to a time-unfolded representation of
         the temporal network. This method is intended for small illustratiions of toy examples.

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

        for n in _np.sort(tempnet.nodes):
            if last == '':
                node = r"\node[lbl]  ({n}-0) {{\bf \Huge {n} }};".format(n=n)
            else:
                node_fmt = r"\node[lbl, right=0.4cm of {last}-0] ({n}-0) " \
                           r"{{\bf \Huge {n} }};"
                node = node_fmt.format(last=last, n=n)

            output.append(node)
            last = n

        output.append("\\setcounter{a}{0}\n")
        min_t = min(tempnet.ordered_times)
        max_t = max(tempnet.ordered_times)
        if dag:
            dag_num = r'\foreach \number in {{ {min_t}, ..., {max_t} }} {{'
            output.append(dag_num.format(min_t=min_t, max_t=max_t+1))
        else:
            dag_num = r'\foreach \number in {{ {min_t}, ..., {max_t} }} {{'
            output.append(dag_num.format(min_t=min_t, max_t=max_t))
        output.append("\\setcounter{a}{\\number}")
        output.append("\\addtocounter{a}{-1}")
        output.append("\\pgfmathparse{\\thea}")

        for n in _np.sort(tempnet.nodes):
            layer_fm = r"\node[v,below={layer} of {n}-\pgfmathresult] ({n}-\number) {{}};"
            output.append(layer_fm.format(layer=layer_dist, n=n))

        first_node = _np.sort(tempnet.nodes)[0]
        fmt = r'\node[lbl,left=0.4cm of {f_node}-\number] (col-\pgfmathresult) ' \
              r'{{ \selectfont {{ \bf \Huge \number}} }};'

        output.append(fmt.format(f_node=first_node))
        output.append(r"}")
        output.append(r"\path[->,line width=2pt]")
        # draw only directed edges
        for ts in tempnet.ordered_times:
            for edge in tempnet.time[ts]:
                if dag:
                    edge_str = r'({}-{}) edge ({}-{})'.format(edge[0], ts, edge[1], ts+1)
                    output.append(edge_str)
                else:
                    if (edge[1], edge[0], ts) not in tempnet.time[ts]:
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
            for ts in tempnet.ordered_times:
                for edge in tempnet.time[ts]:
                    if (edge[1], edge[0], ts) in tempnet.time[ts]:
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
