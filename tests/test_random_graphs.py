import pathpy as pp

import pytest
import numpy as np

def test_is_graphic_sequence():

    sequence = [2, 2, 90]
    assert pp.algorithms.random_graphs.is_graphic_sequence(sequence) == False, \
        'Wrongly detected graphic sequence'

    sequence = [1, 1]
    assert pp.algorithms.random_graphs.is_graphic_sequence(sequence) == True, \
        'Wrongly detected non-graphic sequence'

    sequence = [1, 2, 3]
    assert pp.algorithms.random_graphs.is_graphic_sequence(sequence) == False, \
        'Wrongly detected graphic sequence'