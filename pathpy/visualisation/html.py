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
import json
import os
import string
from string import Template
import random

import numpy as _np

from pathpy.classes.network import Network
from pathpy.classes.higher_order_network import HigherOrderNetwork
from pathpy.classes.temporal_network import TemporalNetwork
from pathpy.classes.paths import Paths

from pathpy.visualisation.alluvial import generate_memory_net
from pathpy.visualisation.alluvial import generate_diffusion_net
from pathpy.visualisation.alluvial import generate_memory_net_markov

@singledispatch
def plot(network, **params):
    """
    Plots an interactive visualisation of a network in a jupyter notebook.

    Parameters
    ----------
    network: Network
        The network to visualize
    params: dict
        A dictionary with visualization parameters to be passed to the HTML
        generation function. These parameters can be processed by custom
        visualisation templates extendable by the user. The default pathpy template
        supports the following parameters:
            width: int
                Width of the div element containing the jupyter visualization.
                Default value is 400.
            height: int
                Height of the div element containing the jupyter visualization.
                Default value is 400.
            node_size: int, dict
                Either an int value that specifies the radius of all nodes, or
                a dictionary that assigns custom node sizes to invidual nodes.
                Default value is 5.0.
            edge_width: int, float, dict
                Either an int value that specifies the radius of all edges, or
                a dictionary that assigns custom edge width to invidual edges.
                Default value is 0.5.
            node_color: string, dict
                Either a string value that specifies the HTML color of all nodes,
                or a dictionary that assigns custom node colors to invidual nodes.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#99ccff"
            edge_color: string, dict
                Either a string value that specifies the HTML color of all edges, 
                or a dictionary that assigns custom edge color to invidual edges.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#ffffff".
            edge_opacity: float
                The opacity of all edges in a range from 0.0 to 1.0. Default value is 1.0.                
            edge_arrows: bool
                Whether to draw edge arrows for directed networks. Default value is True.
            label_color: string
                The HTML color of node labels. Default value is #ffffff.
            label_opacity: float
                The opacity of the label. Default is 1.0.
            label_size: int
                Size of the font to be used for labels.
            label_offset: list
                The offset [x,y] of the label from the center of a node. For [0,0] labels will be 
                displayed in the center of a node. Positive values for the first and second component
                move the label to the right and top respectively. Default is [0, -10], which 
                displays labels above the nodes.
            force_charge: float, int
                The charge strength of nodes to be used in the force-directed layout. Default value is -20
            force_repel: float, int
                The strength of the repulsive force between nodes. Larger negative values will increase the distance
                between nodes. Default value is -200.
            force_alpha: float
                The alpha target (convergence threshold) to be passed to the underlying force-directed 
                layout algorithm. Default value is 0.0.                          
            template: string
                Path to custom visualization template file. If this parameter is omitted, the
                default pathpy network template will be used.
            d3js_path: string
                URL to the d3js library. By default, d3js will be loaded from https://d3js.org/d3.v4.min.js.
                For offline operation, the URL to a local copy of d3js can be specified instead.
    """
    assert isinstance(network, Network), \
        "network must be an instance of Network"

    from IPython.core.display import display, HTML
    display(HTML(generate_html(network, **params)))


@singledispatch
def generate_html(network, **params):
    """
    Generates an HTML snippet that contains an interactive d3js visualization
    of the given network instance.

    Parameters
    ----------
    network: Network
        The network to visualize
    params: dict
        A dictionary with visualization parameters to be passed to the HTML
        generation function. These parameters can be processed by custom
        visualisation templates extendable by the user. The default pathpy template
        supports the following parameters:
            width: int
                Width of the div element containing the jupyter visualization.
                Default value is 400.
            height: int
                Height of the div element containing the jupyter visualization.
                Default value is 400.
            node_size: int, dict
                Either an int value that specifies the radius of all nodes, or
                a dictionary that assigns custom node sizes to invidual nodes.
                Default value is 5.0.
            edge_width: int, float, dict
                Either an int value that specifies the radius of all edges, or
                a dictionary that assigns custom edge width to invidual edges.
                Default value is 0.5.
            node_color: string, dict
                Either a string value that specifies the HTML color of all nodes,
                or a dictionary that assigns custom node colors to invidual nodes.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#99ccff"
            edge_color: string, dict
                Either a string value that specifies the HTML color of all edges, 
                or a dictionary that assigns custom edge color to invidual edges.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#999999".
            edge_opacity: float
                The opacity of all edges in a range from 0.0 to 1.0. Default value is 1.0.
            edge_arrows: bool
                Whether to draw edge arrows for directed networks. Default value is True.
            label_color: string
                The HTML color of node labels. Default value is #ffffff.
            label_opacity: float
                The opacity of the label. Default is 1.0.
            label_size: string
                CSS-style font size of the font to be used for labels. Default is '8px'.
            label_offset: list
                The offset [x,y] of the label from the center of a node. For [0,0] labels will be
                displayed in the center of a node. Positive values for the first and second component
                move the label to the right and top respectively. Default is [0, -10], which
                displays labels above the nodes.
            force_charge: float, int
                The charge strength of nodes to be used in the force-directed layout. Default value is -20
            force_repel: float, int
                The strength of the repulsive force between nodes. Larger negative values will increase the distance
                between nodes. Default value is -200.
            force_alpha: float
                The alpha target (convergence threshold) to be passed to the underlying force-directed
                layout algorithm. Default value is 0.0.                          
            template: string
                Path to custom visualization template file. If this parameter is omitted, the
                default pathpy network template will be used.
            d3js_path: string
                URL to the d3js library. By default, d3js will be loaded from https://d3js.org/d3.v4.min.js.
                For offline operation, the URL to a local copy of d3js can be specified instead.
    """
    assert isinstance(network, Network), \
        "network must be an instance of Network"    

    # prefix nodes starting with number as such IDs are not supported in HTML5
    def fix_node_name(v):
        if v[0].isdigit():
            return "n_" + v
        return v

    # function to assign node/edge attributes based on params
    def get_attr(key, attr_name, attr_default):
        # If parameter does not exist assign default
        if attr_name not in params:
            return attr_default
        # if parameter is a scalar, assign the specified value 
        elif isinstance(params[attr_name], type(attr_default)):
            return params[attr_name]
        # if parameter is a dictionary, assign node/edge specific value
        elif isinstance(params[attr_name], dict):
            if key in params[attr_name]:
                return params[attr_name][key]
            # ... or default value if node/edge is not in dictionary
            else:
                return attr_default
        # raise exception if parameter is of unexpected type
        else:
            raise Exception(
                'Edge and node attribute must either be dict or {0}'.format(type(attr_default))
                )

    # Create network data that will be passed as JSON object
    network_data = {
        'links': [{'source': fix_node_name(e[0]),
                   'target': fix_node_name(e[1]),
                   'color': get_attr((e[0], e[1]), 'edge_color', '#999999'),
                   'width': get_attr((e[0], e[1]), 'edge_width', 0.5)} for e in network.edges.keys()]
    }
    network_data['nodes'] = [{'id': fix_node_name(v),
                              'color': get_attr(v, 'node_color', '#99ccff'),
                              'size': get_attr(v, 'node_size', 5.0)} for v in network.nodes]

    # DIV params
    if 'height' not in params:
        params['height'] = 400

    if 'width' not in params:
        params['width'] = 400

    # label params
    if 'label_size' not in params:
        params['label_size'] = '8px'
    
    if 'label_offset' not in params:
        params['label_offset'] = [0,-10]
    
    if 'label_color' not in params:
        params['label_color'] = '#ffffff'

    if 'label_opacity' not in params:
        params['label_opacity'] = 1.0

    if 'edge_opacity' not in params:
        params['edge_opacity'] = 1.0

    # layout params 
    if 'force_repel' not in params: 
        params['force_repel'] = -200

    if 'force_charge' not in params: 
        params['force_charge'] = -20

    if 'force_alpha' not in params: 
        params['force_alpha'] = 0.0

    # arrows
    if 'edge_arrows' not in params:
        params['edge_arrows'] = 'true'
    else:
        params['edge_arrows'] = str(params['edge_arrows']).lower()

    # Create a random DIV ID to avoid conflicts within the same notebook
    div_id = "".join(random.choice(string.ascii_letters) for x in range(8))

    module_dir = os.path.dirname(os.path.realpath(__file__))
    html_dir = os.path.join(module_dir, os.path.pardir, 'visualisation_assets')
    
    # We have three options to lod the d3js library: 

    # 1.) Via a URL of a local copy in pathpy's visualisation assets
    # from urllib.parse import urljoin
    # from urllib.request import pathname2url    
    # d3js_path = urljoin('file://', pathname2url(os.path.abspath(os.path.join(html_dir, 'd3.v4.min.js'))))

    # 2.) Assuming a local copy in the startup folder of jupyter
    # d3js_path = 'd3.v4.min.js'

    # 3.) Direct import from the d3js server (default)

    # All three options work with the export to a stand-alone HTML viewed via the browser
    # in jupyter, options 1 + 2 do not work at all, while option 3 unfortunately requires us to be online
    if 'd3js_path' not in params:
        params['d3js_path'] = 'https://d3js.org/d3.v4.min.js'

    d3js_params = {
        'network_data': json.dumps(network_data),
        'div_id': div_id,
    }

    # Resolve the template ...
    if 'template' not in params:
        module_dir = os.path.dirname(os.path.realpath(__file__))
        html_dir = os.path.join(module_dir, os.path.pardir, 'visualisation_assets')
        template_file = os.path.abspath(os.path.join(html_dir, 'network_template.html'))
    else:
        template_file = params['template']

    # Read template file ... 
    with open(template_file) as f:
        html_str = f.read()

    # substitute variables in template file
    html = Template(html_str).substitute({**d3js_params, **params})

    return html


@singledispatch
def export_html(network, filename, **params):
    """
    Exports a stand-alone HTML file that contains an interactive d3js visualization
    of the given network instance.

    Parameters
    ----------
    network: Network
        The network to visualize
    filename: string
        Path where the HTML file will be saved
    params: dict
        A dictionary with visualization parameters to be passed to the HTML
        generation function. These parameters can be processed by custom
        visualisation templates extendable by the user. The default pathpy template
        supports the following parameters:
            width: int
                Width of the div element containing the jupyter visualization.
                Default value is 400.
            height: int
                Height of the div element containing the jupyter visualization.
                Default value is 400.
            node_size: int, dict
                Either an int value that specifies the radius of all nodes, or
                a dictionary that assigns custom node sizes to invidual nodes.
                Default value is 5.0.
            edge_width: int, dict
                Either an int value that specifies the radius of all edges, or
                a dictionary that assigns custom edge width to invidual edges.
                Default value is 0.5.
            node_color: string, dict
                Either a string value that specifies the HTML color of all nodes,
                or a dictionary that assigns custom node colors to invidual nodes.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#99ccff"
            edge_color: string, dict
                Either a string value that specifies the HTML color of all edges, 
                or a dictionary that assigns custom edge color to invidual edges.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#999999".
            edge_opacity: float
                The opacity of all edges in a range from 0.0 to 1.0. Default value is 1.0.                
            edge_arrows: bool
                Whether to draw edge arrows for directed networks. Default value is True.
            label_color: string
                The HTML color of node labels. Default value is #ffffff.
            label_opacity: float
                The opacity of the label. Default is 1.0.
            label_size: int
                Size of the font to be used for labels.
            label_offset: list
                The offset [x,y] of the label from the center of a node. For [0,0] labels will be 
                displayed in the center of a node. Positive values for the first and second component
                move the label to the right and top respectively. Default is [0, -10], which 
                displays labels above the nodes.
            label_opacity: float
                Opacity of node labels. Default is 0.7.                
            force_charge: float, int
                The charge strength of nodes to be used in the force-directed layout. Default value is -20
            force_repel: float, int
                The strength of the repulsive force between nodes. Larger negative values will increase the distance
                between nodes. Default value is -200.
            force_alpha: float
                The alpha target (convergence threshold) to be passed to the underlying force-directed 
                layout algorithm. Default value is 0.0.                          
            template: string
                Path to custom visualization template file. If this parameter is omitted, the
                default pathpy network template will be used.
            d3js_path: string
                URL to the d3js library. By default, d3js will be loaded from https://d3js.org/d3.v4.min.js.
                For offline operation, the URL to a local copy of d3js can be specified instead.
    """
    assert isinstance(network, Network), \
        "network must be an instance of Network"
    html = generate_html(network, **params)
    if 'template' not in params:
        html = '<!DOCTYPE html>\n<html><body>\n' + html + '</body>\n</html>'
    with open(filename, 'w+') as f:
        f.write(html)


@plot.register(HigherOrderNetwork)
def _plot_hon(hon, **params):
    """
    Testing implementation of higher-order layout algorithm
    """ 
    from IPython.core.display import display, HTML
    display(HTML(generate_html(hon, **params)))


@export_html.register(HigherOrderNetwork)
def export_html_hon(hon, filename, **params):
    html = generate_html(hon, **params)

    # for the inner HTML generated from the default templates, we add the surrounding DOCTYPE
    # and body needed for a stand-alone HTML file.
    if 'template' not in params:
        html = '<!DOCTYPE html>\n<html><body>\n' + html + '</body>\n</html>'
    with open(filename, 'w+') as f:
        f.write(html)


@generate_html.register(HigherOrderNetwork)
def _generate_html_hon(hon, **params):
    """
    """

    network = Network.from_paths(hon.paths)

    # prefix nodes starting with number as such IDs are not supported in HTML5
    def fix_node_name(v):
        if v[0].isdigit():
            return "n_" + v
        return v

    # function to assign node/edge attributes based on params
    def get_attr(key, attr_name, attr_default):
        # If parameter does not exist assign default
        if attr_name not in params:
            return attr_default
        # if parameter is a scalar, assign the specified value 
        elif isinstance(params[attr_name], type(attr_default)):
            return params[attr_name]
        # if parameter is a dictionary, assign node/edge specific value
        elif isinstance(params[attr_name], dict):
            if key in params[attr_name]:
                return params[attr_name][key]
            # ... or default value if node/edge is not in dictionary
            else:
                return attr_default
        # raise exception if parameter is of unexpected type
        else:
            raise Exception(
                'Edge and node attribute must either be dict or {0}'.format(type(attr_default))
                )

    # Create network data that will be passed as JSON object
    network_data = {
        'links': [{'source': fix_node_name(e[0]),
                   'target': fix_node_name(e[1]),
                   'color': get_attr((e[0], e[1]), 'edge_color', '#999999'),
                   'width': get_attr((e[0], e[1]), 'edge_width', 0.5)} for e in network.edges.keys()]
    }
    network_data['nodes'] = [{'id': fix_node_name(v),
                              'color': get_attr(v, 'node_color', '#99ccff'),
                              'size': get_attr(v, 'node_size', 5.0)} for v in network.nodes]

    
    # add invisible higher-order links
    for e in hon.edges:
        network_data['links'].append(
                                     {'source': fix_node_name(hon.higher_order_node_to_path(e[0])[0]),
                                     'target': fix_node_name(hon.higher_order_node_to_path(e[1])[-1]), 
                                     'width': 0.0, 
                                     'color': ' #ffffff' })

    # DIV params
    if 'height' not in params:
        params['height'] = 400

    if 'width' not in params:
        params['width'] = 400

    # label params
    if 'label_size' not in params:
        params['label_size'] = '8px'
    
    if 'label_offset' not in params:
        params['label_offset'] = [0,-10]
    
    if 'label_color' not in params:
        params['label_color'] = '#ffffff'

    if 'label_opacity' not in params:
        params['label_opacity'] = 1.0

    if 'edge_opacity' not in params:
        params['edge_opacity'] = 1.0

    # layout params 
    if 'force_repel' not in params: 
        params['force_repel'] = -200

    if 'force_charge' not in params: 
        params['force_charge'] = -20

    if 'force_alpha' not in params: 
        params['force_alpha'] = 0.0

    # arrows
    if 'edge_arrows' not in params:
        params['edge_arrows'] = 'true'
    else:
        params['edge_arrows'] = str(params['edge_arrows']).lower()

    # Create a random DIV ID to avoid conflicts within the same notebook
    div_id = "".join(random.choice(string.ascii_letters) for x in range(8))

    module_dir = os.path.dirname(os.path.realpath(__file__))
    html_dir = os.path.join(module_dir, os.path.pardir, 'visualisation_assets')
    
    # We have three options to lod the d3js library: 

    # 1.) Via a URL of a local copy in pathpy's visualisation assets
    # from urllib.parse import urljoin
    # from urllib.request import pathname2url    
    # d3js_path = urljoin('file://', pathname2url(os.path.abspath(os.path.join(html_dir, 'd3.v4.min.js'))))

    # 2.) Assuming a local copy in the startup folder of jupyter
    # d3js_path = 'd3.v4.min.js'

    # 3.) Direct import from the d3js server (default)

    # All three options work with the export to a stand-alone HTML viewed via the browser
    # in jupyter, options 1 + 2 do not work at all, while option 3 unfortunately requires us to be online
    if 'd3js_path' not in params:
        params['d3js_path'] = 'https://d3js.org/d3.v4.min.js'

    d3js_params = {
        'network_data': json.dumps(network_data),
        'div_id': div_id,
    }

    # Resolve the template ...
    if 'template' not in params:
        module_dir = os.path.dirname(os.path.realpath(__file__))
        html_dir = os.path.join(module_dir, os.path.pardir, 'visualisation_assets')
        template_file = os.path.abspath(os.path.join(html_dir, 'network_template.html'))
    else:
        template_file = params['template']

    # Read template file ... 
    with open(template_file) as f:
        html_str = f.read()

    # substitute variables in template file
    html = Template(html_str).substitute({**d3js_params, **params})

    return html

@plot.register(TemporalNetwork)
def _plot_tempnet(tempnet, **params):
    """
    Plots an interactive dynamic visualization of a temporal network in a jupyter notebook.

    Parameters
    ----------
    tempnet: TemporalNetwork
        The temporal network to visualize.
    params: dict
        A dictionary with visualization parameters to be passed to the HTML
        generation function. These parameters can be processed by custom
        visualisation templates extendable by the user. The default pathpy template
        supports the following parameters:
            width: int
                Width of the div element containing the jupyter visualization.
                Default value is 400.
            height: int
                Height of the div element containing the jupyter visualization.
                Default value is 400.
            ms_per_frame: int
                how many milliseconds each frame of the visualisation shall be displayed.
                The inverse of this value gives the framerate of the resulting visualisation.
                The default value of 20 yields a framerate of 50 fps.
            ts_per_frame: int
                How many timestamps in the temporal network shall be displayed in every frame
                of the visualisation. For a value of 1 each timestamp is shown in a separate frame.
                For higher values, multiple timestamps will be aggregated in a single frame. For a
                value of zero, simulation speed is adjusted to the inter event time distribution such
                that on average five interactions are shown per second. Default value is 10.
            look_behind: int
                The look_ahead and look_behind parameters define a temporal range around the current time
                stamp within which time-stamped edges will be considered for the force-directed layout. 
                Values larger than one result in smoothly changing layouts.
                Default value is 10.
            look_ahead: int
                The look_ahead and look_behind parameters define a temporal range around the current time
                stamp within which time-stamped edges will be considered for the force-directed layout. 
                Values larger than one result in smoothly changing layouts.
                Default value is 10.
            node_size: float
                An float value that specifies the radius of all nodes.
                Default value is 5.0.            
            active_edge_width: float
                A float value that specifies the width of active edges.
                Default value is 4.0.
            inactive_edge_width: float
                A float value that specifies the width of active edges.
                Default value is 0.5.
            node_color: string, dict
                Either a string value that specifies the HTML color of all nodes,
                or a dictionary that assigns custom node colors to invidual nodes.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#99ccff"
            active_edge_color: string
                A string value that specifies the HTML color of active edges.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#ff0000".
            inactive_edge_color: string
                A string value that specifies the HTML color of inactive edges.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#999999".
            edge_opacity: float
                The opacity of active edges in a range from 0.0 to 1.0. Default value is 1.0.                
            active_node_color: string
                A string value that specifies the HTML color of active nodes.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#ff0000".
            inactive_node_color: string
                A string value that specifies the HTML color of inactive nodes.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#999999".
            label_color: string
                The HTML color of node labels. Default value is #ffffff.
            label_opacity: float
                The opacity of the label. Default is 1.0.
            label_size: int
                Size of the font to be used for labels.
            label_offset: list
                The offset [x,y] of the label from the center of a node. For [0,0] labels will be 
                displayed in the center of a node. Positive values for the first and second component
                move the label to the right and top respectively. Default is [0, -10], which 
                displays labels above the nodes.
            force_charge: float, int
                The charge strength of nodes to be used in the force-directed layout. Default value is -20
            force_repel: float, int
                The strength of the repulsive force between nodes. Larger negative values will increase the distance
                between nodes. Default value is -200.
            force_alpha: float
                The alpha target (convergence threshold) to be passed to the underlying force-directed 
                layout algorithm. Default value is 0.0.
            template: string
                Path to custom visualization template file. If this parameter is omitted, the
                default pathpy network template will be used.
            d3js_path: string
                URL to the d3js library. By default, d3js will be loaded from https://d3js.org/d3.v4.min.js.
                For offline operation, the URL to a local copy of d3js can be specified instead.     
    """
    assert isinstance(tempnet, TemporalNetwork), \
        "tempnet must be an instance of TemporalNetwork"
    from IPython.core.display import display, HTML
    display(HTML(generate_html(tempnet, **params)))


@generate_html.register(TemporalNetwork)
def _generate_html_tempnet(tempnet, **params):
    """
    Generates an HTML snippet that contains an interactive dynamic d3js visualization
    of the given temporal network instance.

    Parameters
    ----------
    tempnet: TemporalNetwork
        The temporal network to visualize
    params: dict
        A dictionary with visualization parameters to be passed to the HTML
        generation function. These parameters can be processed by custom
        visualisation templates extendable by the user. Arbitrary parameters x
        can be passed to such templates, from where their value can be accessed
        via a variable $x. The default pathpy template supports the following parameters:
            width: int
                Width of the div element containing the jupyter visualization.
                Default value is 400.
            height: int
                Height of the div element containing the jupyter visualization.
                Default value is 400.
            ms_per_frame: int
                how many milliseconds each frame of the visualisation shall be displayed.
                The inverse of this value gives the framerate of the resulting visualisation.
                The default value of 20 yields a framerate of 50 fps.
            ts_per_frame: int
                How many timestamps in the temporal network shall be displayed in every frame
                of the visualisation. For a value of 1 each timestamp is shown in a separate frame.
                For higher values, multiple timestamps will be aggregated in a single frame. For a
                value of zero, simulation speed is adjusted to the inter event time distribution such
                that on average five interactions are shown per second. Default value is 10.
            look_behind: int
                The look_ahead and look_behind parameters define a temporal range around the current time
                stamp within which time-stamped edges will be considered for the force-directed layout. 
                Values larger than one result in smoothly changing layouts.
                Default value is 10.
            look_ahead: int
                The look_ahead and look_behind parameters define a temporal range around the current time
                stamp within which time-stamped edges will be considered for the force-directed layout. 
                Values larger than one result in smoothly changing layouts.
                Default value is 10.
            node_size: float
                An float value that specifies the radius of all nodes.
                Default value is 5.0.            
            active_edge_width: float
                A float value that specifies the width of active edges.
                Default value is 4.0.
            inactive_edge_width: float
                A float value that specifies the width of active edges.
                Default value is 0.5.
            node_color: string, dict
                Either a string value that specifies the HTML color of all nodes,
                or a dictionary that assigns custom node colors to invidual nodes.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#99ccff"
            active_edge_color: string
                A string value that specifies the HTML color of active edges.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#ff0000".
            inactive_edge_color: string
                A string value that specifies the HTML color of inactive edges.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#999999".
            edge_opacity: float
                The opacity of active edges in a range from 0.0 to 1.0. Default value is 1.0.                
            active_node_color: string
                A string value that specifies the HTML color of active nodes.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#ff0000".
            inactive_node_color: string
                A string value that specifies the HTML color of inactive nodes.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#999999".
            label_color: string
                The HTML color of node labels. Default value is #ffffff. 
            label_size: str
                CSS-style size of the font to be used for labels. Default value is '8px'
            label_offset: list
                The offset [x,y] of the label from the center of a node. For [0,0] labels will be 
                displayed in the center of a node. Positive values for the first and second component
                move the label to the right and top respectively. Default is [0, -10], which 
                displays labels above the nodes.
            label_opacity: float
                Opacity of node labels. Default is 0.7.                
            force_charge: float, int
                The charge strength of nodes to be used in the force-directed layout. Default value is -20
            force_repel: float, int
                The strength of the repulsive force between nodes. Larger negative values will increase the distance
                between nodes. Default value is -200.
            force_alpha: float
                The alpha target (convergence threshold) to be passed to the underlying force-directed 
                layout algorithm. Default value is 0.0.                
            template: string
                Path to custom visualization template file. If this parameter is omitted, the
                default pathpy network template will be used.
            d3js_path: string
                URL to the d3js library. By default, d3js will be loaded from https://d3js.org/d3.v4.min.js.
                For offline operation, the URL to a local copy of d3js can be specified instead.
    Returns
    -------

    """
    # Temporal parameters
    if 'ms_per_frame' not in params:
        params['ms_per_frame'] = 50

    if 'ts_per_frame' not in params:
        params['ts_per_frame'] = 1

    # auto-adjust simulation speed to temporal characteristics
    if params['ts_per_frame'] == 0:
        d = tempnet.inter_event_times()
        avg_ts_bw_interactions = _np.mean(d)
        fps = 1000.0/float(params['ms_per_frame'])
        x = avg_ts_bw_interactions/fps
        # set time scale so that we expect 5 interactions per frame
        params['ts_per_frame'] = _np.max([1, int(20 * x)])

    if 'look_ahead' not in params:
        params['look_ahead'] = 10

    if 'look_behind' not in params:
        params['look_behind'] = 10

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
                   'width': 1,
                   'time': t} for s, v, t in tempnet.tedges
                 ]
    }

    # Size of DIV
    if 'width' not in params:
        params['width'] = 400

    if 'height' not in params:
        params['height'] = 400

    # label params
    if 'label_size' not in params:
        params['label_size'] = '8px'
    
    if 'label_offset' not in params:
        params['label_offset'] = [0,-10]
    
    if 'label_color' not in params:
        params['label_color'] = '#ffffff'

    if 'label_opacity' not in params:
        params['label_opacity'] = 1.0

    # layout params 
    if 'force_repel' not in params: 
        params['force_repel'] = -200

    if 'force_charge' not in params: 
        params['force_charge'] = -20

    if 'force_alpha' not in params: 
        params['force_alpha'] = 0.0

    # Colors and sizes
    if 'node_size' not in params:
        params['node_size'] = 5.0

    if 'active_edge_color' not in params:
        params['active_edge_color'] = '#ff0000'

    if 'edge_opacity' not in params:
        params['edge_opacity'] = 1.0

    if 'inactive_edge_width' not in params:
        params['inactive_edge_width'] = 0.5

    if 'active_edge_width' not in params:
        params['active_edge_width'] = 4.0

    if 'inactive_edge_color' not in params:
        params['inactive_edge_color'] = '#999999'

    if 'active_node_color' not in params:
        params['active_node_color'] = '#ff0000'

    if 'inactive_node_color' not in params:
        params['inactive_node_color'] = '#999999'  

    # Create a random DIV ID to avoid conflicts within the same notebook
    div_id = "".join(random.choice(string.ascii_letters) for x in range(8))

    # use standard template file if no custom template is specified
    if 'template' not in params:
        module_dir = os.path.dirname(os.path.realpath(__file__))
        html_dir = os.path.join(module_dir, os.path.pardir, 'visualisation_assets')
        template_file = os.path.abspath(os.path.join(html_dir, 'tempnet_template.html'))
    else:
        template_file = params['template']

    with open(template_file) as f:
        html_str = f.read()

    if 'd3js_path' not in params:
        params['d3js_path'] = 'https://d3js.org/d3.v4.min.js'

    d3js_params = {
        'network_data': json.dumps(network_data),
        'div_id': div_id
    }

    # substitute variables in template file
    html = Template(html_str).substitute({**d3js_params, **params})

    return html


@export_html.register(TemporalNetwork)
def _export_html_tempnet(tempnet, filename, **params):
    """
    Exports a stand-alone HTML file that contains an interactive d3js visualization
    of the given temporal network instance.

    Parameters
    ----------
    tempnet: TemporalNetwork
        The temporal network to visualize
    filename: string
        Path where the HTML file will be saved
    params: dict
        A dictionary with visualization parameters to be passed to the HTML
        generation function. These parameters can be processed by custom
        visualisation templates extendable by the user. The default pathpy template
        supports the following parameters:
                        width: int
                Width of the div element containing the jupyter visualization.
                Default value is 400.
            height: int
                Height of the div element containing the jupyter visualization.
                Default value is 400.
            ms_per_frame: int
                how many milliseconds each frame of the visualisation shall be displayed.
                The inverse of this value gives the framerate of the resulting visualisation.
                The default value of 20 yields a framerate of 50 fps.
            ts_per_frame: int
                How many timestamps in the temporal network shall be displayed in every frame
                of the visualisation. For a value of 1 each timestamp is shown in a separate frame.
                For higher values, multiple timestamps will be aggregated in a single frame. For a
                value of zero, simulation speed is adjusted to the inter event time distribution such
                that on average five interactions are shown per second. Default value is 10.
            look_behind: int
                The look_ahead and look_behind parameters define a temporal range around the current time
                stamp within which time-stamped edges will be considered for the force-directed layout. 
                Values larger than one result in smoothly changing layouts.
                Default value is 10.
            look_ahead: int
                The look_ahead and look_behind parameters define a temporal range around the current time
                stamp within which time-stamped edges will be considered for the force-directed layout. 
                Values larger than one result in smoothly changing layouts.
                Default value is 10.
            node_size: float
                An float value that specifies the radius of all nodes.
                Default value is 5.0.            
            active_edge_width: float
                A float value that specifies the width of active edges.
                Default value is 4.0.
            inactive_edge_width: float
                A float value that specifies the width of active edges.
                Default value is 0.5.
            node_color: string, dict
                Either a string value that specifies the HTML color of all nodes,
                or a dictionary that assigns custom node colors to invidual nodes.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#99ccff"
            active_edge_color: string
                A string value that specifies the HTML color of active edges.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#ff0000".
            inactive_edge_color: string
                A string value that specifies the HTML color of inactive edges.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#999999".
            edge_opacity: float
                The opacity of active edges in a range from 0.0 to 1.0. Default value is 1.0.                
            active_node_color: string
                A string value that specifies the HTML color of active nodes.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#ff0000".
            inactive_node_color: string
                A string value that specifies the HTML color of inactive nodes.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#999999".
            label_color: string
                The HTML color of node labels. Default value is #ffffff. 
            label_size: int
                Size of the font to be used for labels.
            label_offset: list
                The offset [x,y] of the label from the center of a node. For [0,0] labels will be 
                displayed in the center of a node. Positive values for the first and second component
                move the label to the right and top respectively. Default is [0, -10], which 
                displays labels above the nodes.
            label_opacity: float
                Opacity of node labels. Default is 0.7.
            force_charge: float, int
                The charge strength of nodes to be used in the force-directed layout. Default value is -20
            force_repel: float, int
                The strength of the repulsive force between nodes. Larger negative values will increase the distance
                between nodes. Default value is -200.
            force_alpha: float
                The alpha target (convergence threshold) to be passed to the underlying force-directed 
                layout algorithm. Default value is 0.0.                
            template: string
                Path to custom visualization template file. If this parameter is omitted, the
                default pathpy network template will be used.
            d3js_path: string
                URL to the d3js library. By default, d3js will be loaded from https://d3js.org/d3.v4.min.js.
                For offline operation, the URL to a local copy of d3js can be specified instead.
    """
    html = generate_html(tempnet, **params)

    # for the inner HTML generated from the default templates, we add the surrounding DOCTYPE
    # and body needed for a stand-alone HTML file.
    if 'template' not in params:
        html = '<!DOCTYPE html>\n<html><body>\n' + html + '</body>\n</html>'
    with open(filename, 'w+') as f:
        f.write(html)


@plot.register(Paths)
def _plot_paths(paths, **params):
    """
    Parameters
    ----------    
    params: dict
        node : str
        markov : bool
        self_loops : bool
        width: int
            Width of the div element containing the jupyter visualization.
            Default value is 400.
        height: int
            Height of the div element containing the jupyter visualization.
            Default value is 400.
        template: str
                Path to custom visualization template file. If this parameter is omitted, the
                default pathpy network template will be used.
        d3js_path: str
            URL to the d3js library. By default, d3js will be loaded from https://d3js.org/d3.v3.min.js.
            For offline operation, the URL to a local copy of d3js can be specified instead.
    """
    html = generate_html(paths, **params)
    from IPython.core.display import display, HTML
    include = """<script id="d3js" src="https://d3js.org/d3.v3.min.js"></script>
        <script id="sankey1" src="http://cdn.rawgit.com/newrelic-forks/d3-plugins-sankey/master/sankey.js"></script>
        <script id="sankey2" src="http://cdn.rawgit.com/misoproject/d3.chart/master/d3.chart.min.js"></script>
        <script id="sankey3" src="http://cdn.rawgit.com/q-m/d3.chart.sankey/master/d3.chart.sankey.min.js"></script>"""
    display(HTML(html))


@generate_html.register(Paths)
def _generate_html_paths(paths, **params):
    """
    Parameters
    ----------    
    params: dict
        node : str
        markov : bool
        self_loops : bool
        width: int
            Width of the div element containing the jupyter visualization.
            Default value is 400.
        height: int
            Height of the div element containing the jupyter visualization.
            Default value is 400.
        template: str
            Path to custom visualization template file. If this parameter is omitted, the
            default pathpy network template will be used.
        d3js_path: str
            URL to the d3js library. By default, d3js will be loaded from https://d3js.org/d3.v3.min.js.
            For offline operation, the URL to a local copy of d3js can be specified instead.
    """
    if 'self_loops' in params:
        self_loops = params['self_loops']
    else:
        self_loops = True

    if 'node' in params:
        node = params['node']
    else: 
        params['node'] = list(paths.nodes)[0]
        node = params['node']
    
    if 'markov' in params and params['markov']:
        n = generate_memory_net_markov(HigherOrderNetwork(paths, k=1), node, self_loops=self_loops)
    else:
        n = generate_memory_net(paths, node, self_loops=self_loops)

    node_idx = {}
    i = 0
    for v in n.nodes:
        node_idx[v] = i
        i += 1

    data = {
        'nodes': [{'name': v, 'id': v} for v in n.nodes],
        'links': [{'source': int(node_idx[e[0]]),
                   'target': int(node_idx[e[1]]),
                   'value': n.edges[e]['weight']
                  } for e in n.edges]
    }

    div_id = "".join(random.choice(string.ascii_letters) for x in range(8))

    if 'width' not in params:
        params['width'] = 400

    if 'height' not in params:
        params['height'] = 400

    if 'd3js_path' not in params:
        params['d3js_path'] = 'http://d3js.org/d3.v3.min.js'

    if 'template' not in params:
        module_dir = os.path.dirname(os.path.realpath(__file__))
        html_dir = os.path.join(module_dir, os.path.pardir, 'visualisation_assets')
        template_file = os.path.join(html_dir, 'paths_template.html')
    else:
        template_file = params['template']

    with open(template_file) as f:
        html_str = f.read()

    d3js_params = {
        'flow_data': json.dumps(data),
        'div_id': div_id,
    }
    html = Template(html_str).substitute({**d3js_params, **params})
    return html


@export_html.register(Paths)
def _export_html_paths(paths, filename, **params):
    """
    Parameters
    ----------    
    params: dict
        node : str
        markov : bool
        self_loops : bool
        width: int
            Width of the div element containing the jupyter visualization.
            Default value is 400.
        height: int
            Height of the div element containing the jupyter visualization.
            Default value is 400.
        template: str
            Path to custom visualization template file. If this parameter is omitted, the
            default pathpy network template will be used.
        d3js_path: str
            URL to the d3js library. By default, d3js will be loaded from https://d3js.org/d3.v3.min.js.
            For offline operation, the URL to a local copy of d3js can be specified instead.
    """
    html = generate_html(paths, **params)
    if 'template' not in params:
        html = '<!DOCTYPE html>\n<html><body>\n' + html + '</body>\n</html>'
    with open(filename, 'w+') as f:
        f.write(html)


def generate_html_diffusion(paths, **params):
    """
    Parameters
    ----------
    params : dict
        paths: Paths
        node : str
        markov : bool
        steps : int
        width : int
        height : int
        template : str
    """
    if 'node' in params:
        node = params['node']
    else:
        params['node'] = list(paths.nodes)[0]
        node = params['node']
    
    if 'markov' in params:
        markov = params['markov']
    else:
        markov = False
    if 'steps' in params:
        steps = params['steps']
    else:
        steps = 5
    n = generate_diffusion_net(paths, node=node, markov=markov, steps=steps)

    node_map = {v: idx for idx, v in enumerate(n.nodes)}

    data = {
        'nodes': [{'name': v, 'id': v} for v in n.nodes],
        'links': [{'source': node_map[e[0]], 'target': node_map[e[1]], 'value': n.edges[e]['weight']} for e in n.edges]
    }

    div_id = "".join(random.choice(string.ascii_letters) for x in range(8))

    if 'd3js_path' not in params:
        params['d3js_path'] = 'http://d3js.org/d3.v3.min.js'

    if 'template' not in params:
        module_dir = os.path.dirname(os.path.realpath(__file__))
        html_dir = os.path.join(module_dir, os.path.pardir, 'visualisation_assets')
        template_file = os.path.join(html_dir, 'diffusion_template.html')

    with open(template_file) as f:
        html_str = f.read()

    d3js_params = {
        'flow_data': json.dumps(data),
        'div_id': div_id
    }

    # replace all placeholders in template
    html = Template(html_str).substitute({**d3js_params, **params})
    return html


def plot_diffusion(paths, **params):
    html = generate_html_diffusion(paths, **params)
    from IPython.core.display import display, HTML
    display(HTML(html))


def export_html_diffusion(paths, filename, **params):
    html = generate_html_diffusion(paths, **params)
    if 'template' not in params:
        html = '<!DOCTYPE html>\n<html><body>\n' + html + '</body>\n</html>'
    with open(filename, 'w+') as f:
        f.write(html)