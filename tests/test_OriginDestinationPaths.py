import pytest
import numpy as np

import pathpy as pp

def test_extract():
    network_path = os.path.join(test_data_directory, 'example_network.edges')
    od_path = os.path.join(test_data_directory, 'example_origin_destination.csv')
    p = pp.Paths.readEdges(file_path, undirected=True)
    network = pp.HigherOrderNetwork(p)

    OD = pp.PathExtraction.OriginDestinationPaths.readFile(od_path)

    paths = pp.PathExtraction.OriginDestinationPaths.extract(OD, network)
    m = pp.MultiOrderModel(paths, maxOrder=3)
    k_opt = m.estimateOrder(paths, maxOrder=3)
    assert k_opt == 3

    
