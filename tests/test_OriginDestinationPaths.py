import os
import pathpy as pp

def test_extract_distribute(test_data_directory, ):
    network_path = os.path.join(test_data_directory, 'example_network.edges')
    od_path = os.path.join(test_data_directory, 'example_origin_destination.csv')

    # read the network topology
    p = pp.Paths.read_edges(network_path, undirected=True)
    network = pp.HigherOrderNetwork(p)

    OD = pp.path_extraction.read_origin_destination(od_path)

    paths = pp.path_extraction.paths_from_origin_destination(OD, network)

    assert (paths.paths[3][('A', 'B', 'F', 'H')][1] == 2.0 and
            paths.paths[3][('A', 'C', 'G', 'H')][1] == 3.0) or \
           (paths.paths[3][('A', 'B', 'F', 'H')][1] == 3.0 and
            paths.paths[3][('A', 'C', 'G', 'H')][1] == 2.0)
    assert paths.paths[3][('D', 'B', 'C', 'E')][1] == 7.0
    assert paths.paths[2][('A', 'B', 'F')][1] == 3.0
    assert paths.paths[2][('B', 'C', 'E')][1] == 3.0


def test_extract_single(test_data_directory, ):
    network_path = os.path.join(test_data_directory, 'example_network.edges')
    od_path = os.path.join(test_data_directory, 'example_origin_destination.csv')

    # read the network topology
    p = pp.Paths.read_edges(network_path, undirected=True)
    network = pp.HigherOrderNetwork(p)

    OD = pp.path_extraction.read_origin_destination(od_path)

    paths = pp.path_extraction.paths_from_origin_destination(OD, network,
                                                             distribute_weight=False)

    assert (paths.paths[3][('A', 'B', 'F', 'H')][1] == 5.0 and
            paths.paths[3][('A', 'C', 'G', 'H')][1] == 0.0) or \
           (paths.paths[3][('A', 'B', 'F', 'H')][1] == 0.0 and
            paths.paths[3][('A', 'C', 'G', 'H')][1] == 5.0)
    assert paths.paths[3][('D', 'B', 'C', 'E')][1] == 7.0
    assert paths.paths[2][('A', 'B', 'F')][1] == 3.0
    assert paths.paths[2][('B', 'C', 'E')][1] == 3.0
