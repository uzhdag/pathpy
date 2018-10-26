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
from pathpy.classes.multi_order_model import MultiOrderModel
from pathpy.classes.temporal_network import TemporalNetwork
from pathpy.classes.paths import Paths

from pathpy.visualisation.alluvial import generate_memory_net
from pathpy.visualisation.alluvial import generate_diffusion_net
from pathpy.visualisation.alluvial import generate_memory_net_markov

@singledispatch
def plot(network, **params):
    """
    Plots an interactive visualisation of pathpy objects
    in a jupyter notebook. This generic function supports instances of
    pathpy.Network, pathpy.TemporalNetwork, pathpy.HigherOrderNetwork,
    pathpy.MultiOrderModel, and pathpy.Paths. See description of different
    visualisations in the parameter description.

    Parameters
    ----------
    network: Network, TemporalNetwork, HigherOrderNetwork, MultiOrderModel, Paths
        The object to visualize. Depending on the type of the object passed, the following
        visualisations are generated:
            Network: interactive visualisation of a network with a force-directed layout.
            HigherOrderNetwork: interactive visualisation of the first-order network
                with forces calculated based on the higher-order network. By setting
                plot_higher_order_nodes=True a network with unprojected
                higher-order nodes can be plotted instead.
            MultiOrderModel: interactive visualisation of the first-order network
                with forces calculated based on the multi-order model.
            TemporalNetwork: interactive and dynamic visualisation of a temporal
                network.
            Paths: alluvial diagram showing markov or non-Markov trajectories through
                a given focal node.
    params: dict
        A dictionary with visualisation parameters. These parameters are passed through to
        visualisation templates that are extendable by the user. The default pathpy templates
        support the following parameters, depending on the type of object being visualised
        (see brackets).
            width: int (all)
                Width of the div element containing the jupyter visualization.
                Default value is 400.
            height: int (all)
                Height of the div element containing the jupyter visualization.
                Default value is 400.
            template: string (all)
                Path to custom visualization template file. If this parameter is omitted, the
                default pathpy visualistion template fitting the corresponding object will be used.
            d3js_path: string (all)
                URL to the d3js library. By default, d3js will be loaded from https://d3js.org/d3.v4.min.js.
                For offline operation, the URL to a local copy of d3js can be specified instead. For custom
                templates, a specific d3js version can be used.
            node_size: int, dict (Network, HigherOrderNetwork, TemporalNetwork, MultiOrderModel)
                Either an int value that specifies the radius of all nodes, or
                a dictionary that assigns custom node sizes to invidual nodes.
                Default value is 5.0.
            edge_width: int, float, dict (Network, HigherOrderNetwork, TemporalNetwork, MultiOrderModel)
                Either an int value that specifies the radius of all edges, or
                a dictionary that assigns custom edge widths to invidual edges.
                Default value is 0.5.
            node_color: string, dict (Network, HigherOrderNetwork, TemporalNetwork, MultiOrderModel)
                Either a string value that specifies the HTML color of all nodes,
                or a dictionary that assigns custom node colors to invidual nodes.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#99ccff". (lightblue)
            node_text: string, dict (Network, HigherOrderNetwork, MultiOrderModel)
                A text displayed when hovering over nodes, e.g. containing node properties, full names, etc. 
                Defaults to node names.
            edge_color: string, dict (Network, HigherOrderNetwork, TemporalNetwork, MultiOrderModel)
                Either a string value that specifies the HTML color of all edges,
                or a dictionary that assigns custom edge color to invidual edges.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#cccccc" (lightgray).
            edge_opacity: float (Network, HigherOrderNetwork, TemporalNetwork, MultiOrderModel)
                The opacity of all edges in a range from 0.0 to 1.0. Default value is 1.0.             
            edge_arrows: bool (Network, HigherOrderNetwork, MultiOrderNetwork)
                Whether to draw edge arrows for directed networks. Default value is True.
            label_color: string (Network, HigherOrderNetwork, TemporalNetwork, MultiOrderModel)
                The HTML color of node labels.  Both HTML named colors ('red, 'blue', 'yellow') 
                or HEX-RGB values can be used.Default value is '#cccccc' (lightgray).
            label_opacity: float (Network, HigherOrderNetwork, TemporalNetwork, MultiOrderModel)
                The opacity of the label. Default value is 1.0.
            label_size: str (Network, HigherOrderNetwork, TemporalNetwork, MultiOrderNetwork)
                CSS string specifying the font size to be used for labels. Default is '8px'.
            label_offset: list (Network, HigherOrderNetwork, TemporalNetwork, MultiOrderNetwork)
                The offset [x,y] of the label from the center of a node. For [0,0] labels will be
                displayed in the center of nodes. Positive values for the first and second component
                move the label to the right and top respectively. Default is [0, -10], which
                displays labels above nodes to accomodate for labels wider than nodes.
            force_charge: float, int ()
                The charge strength of nodes to be used in the force-directed layout. Default value is -20
            force_repel: float, int (all)
                The strength of the repulsive force between nodes. Larger negative values will increase the distance
                between nodes. Default value is -200.
            force_alpha: float (all)
                The alpha target (convergence threshold) to be passed to the underlying force-directed
                layout algorithm. Default value is 0.0.
            plot_higher_order_nodes: HigherOrderNetwork
                If set to True, a raw higher-order network with higher-order nodes will be plotted. If
                False, a first-order projection with a higher-order force-directed layout will be plotted.
                The default value is True.
            ms_per_frame: int (TemporalNetwork)
                how many milliseconds each frame shall be displayed in the visualisation of a TemporalNetwork.
                The 1000/ms_per_frame specifies the framerate of the visualisation. The default value of 20 yields a
                framerate of 50 fps.
            ts_per_frame: int (TemporalNetwork)
                How many timestamps in a temporal network shall be displayed in every frame
                of the visualisation. For a value of 1 each timestamp is shown in a separate frame.
                For higher values, multiple timestamps will be aggregated in a single frame. For a
                value of zero, simulation speed is adjusted to the inter event time distribution such
                that on average five interactions are shown per second. Default value is 1.
            look_behind: int (TemporalNetwork)
                The look_ahead and look_behind parameters define a temporal range around the current time
                stamp within which time-stamped edges will be considered for the force-directed layout.
                Values larger than one result in smoothly changing layouts.
                Default value is 10.
            look_ahead: int (TemporalNetwork)
                The look_ahead and look_behind parameters define a temporal range around the current time
                stamp within which time-stamped edges will be considered for the force-directed layout.
                Values larger than one result in smoothly changing layouts.
                Default value is 10.
            max_time: int (TemporalNetwork)
                maximum time stamp to visualise. Useful to limit visualisation of very long Temporal Networks. 
                If None, the whole sequence will be shown. Default is None.
            active_edge_width: float (TemporalNetwork)
                A float value that specifies the width of currently active edges.
                Default value is 4.0.
            inactive_edge_width: float (TemporalNetwork)
                A float value that specifies the width of currently active edges.
                Default value is 0.5.
            active_edge_color: string (TemporalNetwork)
                A string value that specifies the HTML color of currently active edges.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#ff0000" (red).
            inactive_edge_color: string (TemporalNetwork)
                A string value that specifies the HTML color of inactive edges.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#cccccc" (lightgray).
            active_node_color: string (TemporalNetwork)
                A string value that specifies the HTML color of active nodes.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#ff0000" (red).
            inactive_node_color: string (TemporalNetwork)
                A string value that specifies the HTML color of inactive nodes.
                Both HTML named colors ('red, 'blue', 'yellow') or HEX-RGB values can
                be used. Default value is "#cccccc" (lightgray).

    Examples:
    ---------
    >>> paths = pp.Paths()
    >>> paths.add_path('a,b,c')
    >>> n = pp.Network.from_paths(paths)
    >>> params = {'label_color': '#ff0000',
                  'node_color': { 'a': '#ff0000', 'b': '#00ff00', 'c': '#0000ff'}
    >>>          }
    >>> pp.visualisation.plot(n, **params)
    >>> [inline visualisation]
    >>> pp.visualisation.export_html(n, filename='myvisualisation.html', **params)
    """
    assert isinstance(network, Network) or isinstance(network, MultiOrderModel), \
        "network must be an instance of Network"

    from IPython.core.display import display, HTML
    display(HTML(generate_html(network, **params)))


@singledispatch
def generate_html(network, **params):
    """
    Generates an HTML snippet that contains an interactive d3js visualization
    of the given instance. This function supports instances of pathpy.Network,
    pathpy.TemporalNetwork, pathpy.HigherOrderNetwork, pathpy.Paths, and pathpy.Paths.
    
    Parameters:
    -----------
        network: Network, TemporalNetwork, HigherOrderNetwork, MultiOrderModel, Paths
            The pathpy object that should be visualised.
        params: dict
            A dictionary with visualization parameters to be passed to the HTML
            generation function. These parameters can be processed by custom
            visualisation templates extendable by the user. For supported parameters
            see docstring of plot.
    """
    assert isinstance(network, Network) or isinstance(network, MultiOrderModel),\
        "Argument must be an instance of Network, HigherOrderNetwork, or MultiOrderModel"

    if 'plot_higher_order_nodes' not in params:
        params['plot_higher_order_nodes'] = True

    if isinstance(network, HigherOrderNetwork):
        mog = None
        hon = None
        if not params['plot_higher_order_nodes']:
            hon = network
            network = Network.from_paths(hon.paths)
    elif isinstance(network, Network):
        hon = None
        mog = None
    elif isinstance(network, MultiOrderModel):
        hon = None
        mog = network
        network = mog.layers[1]

    # prefix nodes starting with number as such IDs are not supported in HTML5
    def fix_node_name(v):
        if str(v)[0].isdigit():
            return "n_" + str(v)
        return str(v)

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

    def compute_weight(network, e):
        """
        Calculates a normalized force weight for an edge in a network
        """
        if 'force_weighted' in params and not params['force_weighted']:
            weight = network.edges[e]['degree']
            source_weight = network.nodes[e[0]]['indegree'] + network.nodes[e[0]]['outdegree']
            target_weight = network.nodes[e[1]]['indegree'] + network.nodes[e[1]]['outdegree']
        else:
            weight = network.edges[e]['weight']
            source_weight = network.nodes[e[0]]['inweight'] + network.nodes[e[0]]['outweight']
            target_weight = network.nodes[e[1]]['inweight'] + network.nodes[e[1]]['outweight']

        if isinstance(weight, _np.ndarray):
            weight = weight.sum()
            source_weight = source_weight.sum()
            target_weight = target_weight.sum()
        s = min(source_weight, target_weight)
        if s>0.0:
            return weight/s
        else:
            return 0.0

    # Create network data that will be passed as JSON object
    network_data = {'links': [{'source': fix_node_name(e[0]),
                               'target': fix_node_name(e[1]),
                               'color': get_attr((e[0], e[1]), 'edge_color', '#999999'),
                               'width': get_attr((e[0], e[1]), 'edge_width', 0.5),
                               'weight': compute_weight(network, e) if hon is None and mog is None else 0.0
                              } for e in network.edges.keys()]
                   }
    network_data['nodes'] = [{'id': fix_node_name(v),
                              'text': get_attr(v, 'node_text', fix_node_name(v)),
                              'color': get_attr(v, 'node_color', '#99ccff'),
                              'size': get_attr(v, 'node_size', 5.0)} for v in network.nodes]
    
    # calculate network of higher-order forces between nodes
    from collections import defaultdict
    higher_order_forces = Network()

    if hon is not None:
        for e in hon.edges:
            v = fix_node_name(hon.higher_order_node_to_path(e[0])[0])
            w = fix_node_name(hon.higher_order_node_to_path(e[1])[-1])
            if (v,w) in higher_order_forces.edges:
                weight = higher_order_forces.edges[(v,w)]['weight'] + hon.edges[e]['weight']
            else: 
                weight = hon.edges[e]['weight']
            higher_order_forces.add_edge(v, w, weight=weight)

    if mog is not None:
        for l in range(1, mog.max_order+1):
            for e in mog.layers[l].edges:
                v = fix_node_name(mog.layers[l].higher_order_node_to_path(e[0])[0])
                w = fix_node_name(mog.layers[l].higher_order_node_to_path(e[1])[-1])
                if (v, w) in higher_order_forces.edges:
                    weight = higher_order_forces.edges[(v, w)]['weight'] + mog.layers[l].edges[e]['weight']
                else:
                    weight = mog.layers[l].edges[e]['weight']
                higher_order_forces.add_edge(v, w, weight=weight)

    for (v, w) in higher_order_forces.edges:
        # add invisible links for higher-order forces
        network_data['links'].append({'source': v,
                                      'target': w,
                                      'width': 0.0,
                                      'weight': compute_weight(higher_order_forces, (v, w)),
                                      'color': ' #999999'})

    # DIV params
    if 'height' not in params:
        params['height'] = 400

    if 'width' not in params:
        params['width'] = 400

    # label params
    if 'label_size' not in params:
        params['label_size'] = '8px'

    if 'label_offset' not in params:
        params['label_offset'] = [0, -10]

    if 'label_color' not in params:
        params['label_color'] = '#999999'

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

    if not network.directed:
        params['edge_arrows'] = 'false'

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
    of the given pathpy instance. function supports instances of pathpy.Network, 
    pathpy.TemporalNetwork, pathpy.HigherOrderNetwork, pathpy.Paths, and pathpy.Paths.

    Parameters
    ----------
    network: Network
        The network to visualize
    filename: string
        Path where the HTML file will be saved
    params: dict
        A dictionary with visualization parameters to be passed to the HTML
        generation function. These parameters can be processed by custom
        visualisation templates extendable by the user. For supported parameters
        see docstring of plot.
    """
    assert isinstance(network, Network) or isinstance(network, MultiOrderModel), \
        "network must be an instance of Network"
    html = generate_html(network, **params)
    if 'template' not in params:
        html = '<!DOCTYPE html>\n<html><body>\n' + html + '</body>\n</html>'
    with open(filename, 'w+') as f:
        f.write(html)


@plot.register(TemporalNetwork)
def _plot_tempnet(tempnet, **params):
    assert isinstance(tempnet, TemporalNetwork), \
        "tempnet must be an instance of TemporalNetwork"
    from IPython.core.display import display, HTML
    display(HTML(generate_html(tempnet, **params)))


@generate_html.register(TemporalNetwork)
def _generate_html_tempnet(tempnet, **params):
    
    if 'ms_per_frame' not in params:
        params['ms_per_frame'] = 50

    if 'ts_per_frame' not in params:
        params['ts_per_frame'] = 1

    if 'max_time' not in params:
        params['max_time'] = None

    if params['max_time'] is not None:
        tempnet = tempnet.filter_edges(lambda u, v, t: t <= params['max_time'])

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
        new_v = str(v)
        if str(v)[0].isdigit():
            new_v = "n_" + str(v)
        if new_v[0] == '_':
            new_v = "n_" + str(v)
        if '-' in new_v:
            new_v = new_v.replace('-', '_')
        return new_v

    network_data = {
        'nodes': [{'id': fix_node_name(v),
                   'group': 1} for v in tempnet.nodes],
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
        params['label_offset'] = [0, -10]

    if 'label_color' not in params:
        params['label_color'] = '#cccccc'

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
        params['inactive_edge_color'] = '#cccccc'

    if 'active_node_color' not in params:
        params['active_node_color'] = '#ff0000'

    if 'inactive_node_color' not in params:
        params['inactive_node_color'] = '#cccccc'

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

    html = generate_html(tempnet, **params)

    # for the inner HTML generated from the default templates, we add the surrounding DOCTYPE
    # and body needed for a stand-alone HTML file.
    if 'template' not in params:
        html = '<!DOCTYPE html>\n<html><body>\n' + html + '</body>\n</html>'
    with open(filename, 'w+') as f:
        f.write(html)


@plot.register(Paths)
def _plot_paths(paths, **params):
    html = generate_html(paths, **params)
    from IPython.core.display import display, HTML
    display(HTML(html))


@generate_html.register(Paths)
def _generate_html_paths(paths, **params):
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


def plot_walk(network, walk, **params):
    """
    Plots an interactive visualisation of a random walk trajectory in 
    a network.
    """

    def fix_node_name(v):
        new_v = v
        if v[0].isdigit():
            new_v = "n_" + v
        if new_v[0] == '_':
            new_v = "n_" + v
        if '-' in new_v:
            new_v = new_v.replace('-', '_')
        return new_v

    from IPython.core.display import display, HTML
    module_dir = os.path.dirname(os.path.realpath(__file__))
    html_dir = os.path.join(module_dir, os.path.pardir, 'visualisation_assets')
    params['template'] = os.path.join(html_dir, 'walk_template.html')
    params['itinerary'] = [ fix_node_name(v) for v in walk ]
    if 'active_node_color' not in params:
        params['active_node_color'] = 'red'
    if 'inactive_node_color' not in params:
        params['inactive_node_color'] = 'lightblue'
    params['plot_higher_order_nodes'] = False
    if 'ms_per_frame' not in params:
        params['ms_per_frame'] = 500
    display(HTML(generate_html(network, **params)))


def export_html_walk(network, walk, filename, **params):
    """
    Exports an interactive visualisation of a random walk trajectory in 
    a network to a file.
    """
    
    def fix_node_name(v):
        new_v = v
        if v[0].isdigit():
            new_v = "n_" + v
        if new_v[0] == '_':
            new_v = "n_" + v
        if '-' in new_v:
            new_v = new_v.replace('-', '_')
        return new_v

    module_dir = os.path.dirname(os.path.realpath(__file__))
    html_dir = os.path.join(module_dir, os.path.pardir, 'visualisation_assets')
    params['template'] = os.path.join(html_dir, 'walk_template.html')
    params['itinerary'] = [ fix_node_name(v) for v in walk ]
    if 'active_node_color' not in params:
        params['active_node_color'] = 'red'
    if 'inactive_node_color' not in params:
        params['inactive_node_color'] = 'lightblue'
    params['plot_higher_order_nodes'] = False
    if 'ms_per_frame' not in params:
        params['ms_per_frame'] = 500
    html = generate_html(network, **params)
    with open(filename, 'w+') as f:
        f.write(html)