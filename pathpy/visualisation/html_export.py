# -*- coding: utf-8 -*-
#    pathpy is an OpenSource python package for the analysis of time series data
#    on networks using higher- and multi order graphical models.
#
#    Copyright (C) 2016-2018 Ingo Scholtes, ETH Zürich, University of Zurich
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

#    E-mail: scholtes@ifi.uzh.ch
#    Web:    http://www.ingoscholtes.net
from functools import singledispatch

import numpy as _np

from pathpy.classes.network import Network
from pathpy.classes.temporal_network import TemporalNetwork

__all__ = ["generate_html", "write_html"]


@singledispatch
def generate_html(network, width=800, height=800, clusters=None, sizes=None,
            template_file=None, **kwargs):
    """
    ...
    """
    assert isinstance(network, Network), \
        "network must be an instance of Network"
    import json
    import os
    from string import Template

    # prefix nodes starting with number
    def fix_node_name(v):
        if v[0].isdigit():
            return "n_" + v
        return v

    network_data = {
        'links': [{'source': fix_node_name(e[0]),
                   'target': fix_node_name(e[1]),
                   'value': 1} for e in network.edges.keys()]
    }

    def get_cluster(v):
        if clusters is None or v not in clusters:
            return 'None'
        else:
            return clusters[v]

    def get_size(v):
        if sizes is None or v not in sizes:
            return 6
        else:
            return sizes[v]

    network_data['nodes'] = [{'id': fix_node_name(v),
                              'group': get_cluster(v),
                              'size': get_size(v)} for v in network.nodes]

    import string
    import random

    div_id = "".join(random.choice(string.ascii_letters) for x in range(8))

    if template_file is None:
        module_dir = os.path.dirname(os.path.realpath(__file__))
        html_dir = os.path.join(module_dir, os.path.pardir, 'html_templates')
        template_file = os.path.join(html_dir, 'network.html')

    with open(template_file) as f:
        html_str = f.read()

    if network.directed:
        directedness = 'true'
    else:
        directedness = 'false'

    default_args = {
        'network_data': json.dumps(network_data),
        'directed' : directedness,
        'width': width,
        'height': height,
        'div_id': div_id
    }

    # replace all placeholders in template
    html = Template(html_str).substitute({**default_args, **kwargs})

    return html


@generate_html.register(TemporalNetwork)
def _generate_html_tempnet(tempnet, width=600, height=600, ms_per_frame=50, ts_per_frame=20, radius=6,
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
        d = tempnet.inter_event_times()
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
        'nodes': [{'id': fix_node_name(v), 'group': 1} for v in tempnet.nodes],
        'links': [{'source': fix_node_name(s),
                   'target': fix_node_name(v),
                   'value': 1,
                   'time': t} for s, v, t in tempnet.tedges
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


@singledispatch
def write_html(network, filename, width=800, height=800, clusters=None, sizes=None, 
               template_file=None, **kwargs):
    """
    ...
    """
    assert isinstance(network, Network), \
        "network must be an instance of Network"
    html = network.to_html(width=width, height=height, clusters=clusters,
                           sizes=sizes, template_file=template_file, **kwargs)
    with open(filename, 'w+') as f:
        f.write(html)


@write_html.register(TemporalNetwork)
def _write_html_tempnet(tempnet, filename, width=800, height=800, msperframe=50, tsperframe=20, radius=6,
               look_behind=1500, look_ahead=150, template_file=None, **kwargs):
    """
    ...
    """
    html = tempnet.to_html(width, height, msperframe, tsperframe=tsperframe, 
                           radius=radius, template_file=template_file, **kwargs)

    # for the inner HTML generated from the default templates, we add the surrounding DOCTYPE
    # and body needed for a stand-alone HTML file
    if template_file is None:
        html = '<!DOCTYPE html>\n<html><body>\n' + html + '</body>\n</html>'
    with open(filename, 'w+') as f:
        f.write(html)
