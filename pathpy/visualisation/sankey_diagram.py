from pathpy.classes.higher_order_network import HigherOrderNetwork
from pathpy.classes.paths import Paths
from pathpy.classes.network import Network

import numpy as _np


def _flow_net(paths, focal_node):
    g1 = HigherOrderNetwork(paths, k=1)
    g2 = HigherOrderNetwork(paths, k=2)
    n = Network(directed=True)

    # get frequencies of all sub-paths of length two 
    # where the focal_node is the middle node
    for p in paths.paths[2]:
        if p[1] == focal_node:
            g1_nodes = g1.path_to_higher_order_nodes(p)
            g2_nodes = g2.path_to_higher_order_nodes(p)
            src = 'src_{0}'.format(p[0])
            tgt = 'tgt_{0}'.format(p[2])
            mem = 'mem_{0}_{1}'.format(p[0], p[1])
            n.add_edge(src, mem, weight=_np.sum(g1.edges[(g1_nodes[0], g1_nodes[1])]['weight']))
            n.add_edge(mem, tgt, weight=_np.sum(g2.edges[(g2_nodes[0], g2_nodes[1])]['weight']))

    return n


def _flow_net_markov(network, focal_node):
    n = Network(directed=True)

    out_weight = _np.sum(network.nodes[focal_node]['outweight'])

    for u in network.predecessors[focal_node]:
        for w in network.successors[focal_node]:
            src = 'src_{0}'.format(u)
            tgt = 'tgt_{0}'.format(w)
            mem = 'mem_{0}_{1}'.format(u, focal_node)

            w_1 = _np.sum(network.edges[(u, focal_node)]['weight'])

            # at random, we expect the flow to be proportional to the relative edge weight
            w_2 = w_1 * (_np.sum(network.edges[(focal_node, w)]['weight'])/out_weight)
            n.add_edge(src, mem, weight=w_1)
            n.add_edge(mem, tgt, weight=w_2)
    return n


def _to_html(paths, focal_node, markov=False, width=600, height=600):
    import json
    import os
    from string import Template

    import string
    import random

    if markov:
        g1 = HigherOrderNetwork(paths, k=1)
        n = _flow_net_markov(g1, focal_node=focal_node)
    else:
        n = _flow_net(paths, focal_node)

    
    node_idx = {}
    i = 0
    for v in n.nodes:
        node_idx[v] = i
        i += 1

    data = {
        'nodes': [ {'name': v, 'id': v} for v in n.nodes ],
        'links': [ {'source': int(node_idx[e[0]]), 'target': int(node_idx[e[1]]), 'value': n.edges[e]['weight']} for e in n.edges ]
    }

    div_id = "".join(random.choice(string.ascii_letters) for x in range(8))

    module_dir = os.path.dirname(os.path.realpath(__file__))
    html_dir = os.path.join(module_dir, os.path.pardir, 'html_templates')            
    template_file = os.path.join(html_dir, 'sankey.html') 

    with open(template_file) as f:
        html_str = f.read()

    args = {
        'flow_data': json.dumps(data),        
        'width': width,
        'height': height,
        'div_id': div_id,
        'focal_node': focal_node
    }

    # replace all placeholders in template
    return Template(html_str).substitute(args)


def write_html(paths, focal_node, filename, markov=False, width=600, height=600):
    html = _to_html(paths, focal_node, markov, width, height)
    with open(filename, 'w+') as f:
        f.write(html)


def show_flow(paths, focal_node, markov=False, width=600, height=600):
    html = _to_html(paths, focal_node, markov, width, height)
    from IPython.core.display import display, HTML
    display(HTML(html))
