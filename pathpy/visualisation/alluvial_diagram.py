from pathpy.classes.higher_order_network import HigherOrderNetwork
from pathpy.classes.paths import Paths
from pathpy.classes.network import Network

import collections as _co

import numpy as _np



def _flow_net(paths, focal_node, self_loops=True):

    n = Network(directed=True)

    # consider all (sub-)paths of length two 
    # through the focal node
    for p in paths.paths[2]:
        if p[1] == focal_node:
            if self_loops or (p[0] != focal_node and p[2] != focal_node):
                src = 'src_{0}'.format(p[0])
                tgt = 'tgt_{0}'.format(p[2])
                mem = 'mem_{0}_{1}'.format(p[0], p[1])
                # calculate frequency of sub-paths src->focal_node->*, i.e. paths that 
                # continue through the focal_node
                # w_1 = 0
                # for x in paths.nodes:
                #         ct = p[:2]+(x,)
                #         if ct in paths.paths[2] and x != focal_node:
                #             w_1 += paths.paths[2][ct].sum()

                # calculate frequency of (sub-)path src -> focal_node -> tgt
                w_2 = paths.paths[2][p].sum()
                n.add_edge(src, mem, weight = 1)
                n.add_edge(mem, tgt, weight = w_2)


    # adjust weights of links to memory nodes:
    for m in n.nodes:
        if m.startswith('mem'):
            for u in n.predecessors[m]:
                n.edges[(u,m)]['weight'] = n.nodes[m]['outweight']
                n.nodes[m]['inweight'] = n.nodes[m]['outweight']
    return n


def _flow_net_markov(network, focal_node, self_loops=True):
    n = Network(directed=True)

    out_weight = _np.sum(network.nodes[focal_node]['outweight'])

    for u in network.predecessors[focal_node]:
        for w in network.successors[focal_node]:
            if self_loops or (u!= focal_node and w != focal_node):
                src = 'src_{0}'.format(u)
                tgt = 'tgt_{0}'.format(w)
                mem = 'mem_{0}_{1}'.format(u, focal_node)

                w_1 = _np.sum(network.edges[(u, focal_node)]['weight'])

                # at random, we expect the flow to be proportional to the relative edge weight
                w_2 = w_1 * (_np.sum(network.edges[(focal_node, w)]['weight'])/out_weight)
                n.add_edge(src, mem, weight=w_1)
                n.add_edge(mem, tgt, weight=w_2)
    return n


def _to_html(paths, focal_node, self_loops=True, markov=False, width=600, height=600, template_file=None):
    import json
    import os
    from string import Template

    import string
    import random

    if markov:
        g1 = HigherOrderNetwork(paths, k=1)
        n = _flow_net_markov(g1, focal_node=focal_node, self_loops=self_loops)
    else:
        n = _flow_net(paths, focal_node, self_loops=self_loops)

    
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

    if template_file is None:
        module_dir = os.path.dirname(os.path.realpath(__file__))
        html_dir = os.path.join(module_dir, os.path.pardir, 'html_templates')            
        template_file = os.path.join(html_dir, 'alluvial_node.html') 

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


def write_html_flow(paths, focal_node, filename, markov=False, self_loops=True, width=600, height=600, template_file=None):
    html = _to_html(paths, focal_node, self_loops, markov, width, height, template_file)
    with open(filename, 'w+') as f:
        f.write(html)


def show_flow(paths, focal_node, markov=False, self_loops=True, width=600, height=600):
    html = _to_html(paths, focal_node, self_loops, markov, width, height)
    from IPython.core.display import display, HTML
    display(HTML(html))


def diffusion_to_flow_net(paths, markov=True, initial_node=None, steps=5):

    g1 = HigherOrderNetwork(paths, k=1)
    map_1 = g1.node_to_name_map()

    prob = _np.zeros(g1.vcount())
    prob = prob.transpose()
    if initial_node is None:
        initial_node = g1.nodes[0]    
    
    prob[map_1[initial_node]] = 1.0
    
    T = g1.transition_matrix()

    flow_net = Network(directed=True)

    if markov:
        # if markov == True flows are given by first-order transition matrix
        for t in range(1, steps+1):
            # calculate flow from i to j in step t
            for i in g1.nodes:
                for j in g1.nodes:                    
                    i_to_j = prob[map_1[i]] * T[map_1[j], map_1[i]]
                    if i_to_j > 0:
                        flow_net.add_edge('{0}_{1}'.format(i, t-1), '{0}_{1}'.format(j, t), weight = i_to_j)
            prob = T.dot(prob)
    else:
        # if markov == False calculate flows based on paths starting in initial_node
        for p in paths.paths[steps]:
            if p[0] == initial_node:
                for t in range(len(p)-1):
                    flow_net.add_edge('{0}_{1}'.format(p[t], t), '{0}_{1}'.format(p[t+1], t+1), weight = paths.paths[steps][p].sum())
        
        # normalize flows and balance in- and out-weight for all nodes
        # normalization = flow_net.nodes['{0}_{1}'.format(initial_node, 0)]['outweight']

        flow_net.nodes[initial_node+'_0']['inweight'] = 1.0
        Q = [initial_node+'_0']
        # adjust weights using BFS
        while Q:
            v = Q.pop()
            # print(v)
            inweight = flow_net.nodes[v]['inweight']
            outweight = flow_net.nodes[v]['outweight']

            for w in flow_net.successors[v]:
                flow_net.nodes[w]['inweight'] = flow_net.nodes[w]['inweight'] - flow_net.edges[(v,w)]['weight']
                flow_net.edges[(v,w)]['weight'] = (inweight/outweight) * flow_net.edges[(v,w)]['weight']
                flow_net.nodes[w]['inweight'] =  flow_net.nodes[w]['inweight'] + flow_net.edges[(v,w)]['weight']
                Q.append(w)
    return flow_net


def diffusion_to_html(paths, markov=True, initial_node=None, steps=5, width=600, height=600, template_file=None):
    import json
    import os
    from string import Template

    import string
    import random

    n = diffusion_to_flow_net(paths, markov=markov, initial_node=initial_node, steps=steps)

    node_map = {v: idx for idx, v in enumerate(n.nodes)}

    data = {
        'nodes': [ {'name': v, 'id': v} for v in n.nodes ],
        'links': [ {'source': node_map[e[0]], 'target': node_map[e[1]], 'value': n.edges[e]['weight']} for e in n.edges ]
    }

    div_id = "".join(random.choice(string.ascii_letters) for x in range(8))

    if template_file is None:
        module_dir = os.path.dirname(os.path.realpath(__file__))
        html_dir = os.path.join(module_dir, os.path.pardir, 'html_templates')            
        template_file = os.path.join(html_dir, 'alluvial_diffusion.html') 

    with open(template_file) as f:
        html_str = f.read()

    args = {
        'flow_data': json.dumps(data),
        'width': width,
        'height': height,
        'div_id': div_id
    }

    # replace all placeholders in template
    return Template(html_str).substitute(args)


def show_diffusion(paths, markov=True, initial_node=None, steps=5, width=600, height=600):
    html = diffusion_to_html(paths, markov=markov, initial_node=initial_node, steps=steps, width=width, height=height)
    from IPython.core.display import display, HTML
    display(HTML(html))


def write_html_diffusion(paths, filename, markov=True, initial_node=None, steps=5, width=600, height=600, template_file=None):
    html = diffusion_to_html(paths, markov=markov, initial_node=initial_node, steps=steps, width=width, height=height, template_file=template_file)
    with open(filename, 'w+') as f:
        f.write(html)