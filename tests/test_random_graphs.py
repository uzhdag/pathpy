import pathpy as pp

import pytest
import numpy as np


def test_is_graphic_sequence():

    sequence = [2, 2, 90]
    assert pp.algorithms.random_graphs.is_graphic_sequence(sequence) is False, \
        'Wrongly detected graphic sequence'

    sequence = [1, 1]
    assert pp.algorithms.random_graphs.is_graphic_sequence(sequence), \
        'Wrongly rejected graphic sequence'

    sequence = [1, 2, 3]
    assert pp.algorithms.random_graphs.is_graphic_sequence(sequence) is False, \
        'Wrongly detected graphic sequence'

    sequence = [2, 2]
    assert pp.algorithms.random_graphs.is_graphic_sequence(sequence, self_loops=True), \
        'Wrongly rejected graphic sequence'

    sequence = [2]
    assert pp.algorithms.random_graphs.is_graphic_sequence(sequence, self_loops=True), \
        'Wrongly rejected graphic sequence'

    sequence = [2]
    assert pp.algorithms.random_graphs.is_graphic_sequence(sequence, self_loops=True,
                                                           multi_edges=True), \
        'Wrongly rejected graphic sequence'

    sequence = [2]
    assert pp.algorithms.random_graphs.is_graphic_sequence(sequence, self_loops=False) is False, \
        'Wrongly detected graphic sequence'

    sequence = [3, 3]
    assert pp.algorithms.random_graphs.is_graphic_sequence(sequence, multi_edges=True,
                                                           self_loops=True), \
        'Wrongly rejected graphic sequence'

    sequence = [1, 3]
    assert pp.algorithms.random_graphs.is_graphic_sequence(sequence, self_loops=True), \
        'Wrongly rejected graphic sequence'

    sequence = [1, 2]
    assert pp.algorithms.random_graphs.is_graphic_sequence(sequence, self_loops=True) is False, \
        'Wrongly detected graphic sequence'

    for i in range(10):
        g = pp.algorithms.random_graphs.erdoes_renyi_gnp(n=100, p=0.03, self_loops=False)
        assert pp.algorithms.random_graphs.is_graphic_sequence([x for x in g.degrees() if x > 0]), \
            'Wrongly rejected degree sequence of randomly generated graph'

    for i in range(10):
        g = pp.algorithms.random_graphs.erdoes_renyi_gnp(n=100, p=0.03, self_loops=True)
        # HACK: correct degrees for self_loops. Need to consistently define degrees of self-loops as two in pathpy!
        for e in g.edges:
            if e[0] == e[1]:
                g.nodes[e[0]]['degree'] += 1
        assert pp.algorithms.random_graphs.is_graphic_sequence([x for x in g.degrees() if x > 0],
                                                               self_loops=True), \
            'Wrongly rejected degree sequence of randomly generated graph'
