import pathpy as pp
import os
import numpy as np
import sqlite3
from pytest import mark

def test_read_temporal_file_int(test_data_directory, ):
    file_path = os.path.join(test_data_directory, 'example_int.tedges')
    t = pp.TemporalNetwork.read_file(file_path)
    times = t.ordered_times
    expected_times = [0, 2, 4, 5, 6, 8]
    assert times == expected_times

    activities = sorted(list(t.activities.values()))
    expected_activities = [[], [], [], [], [0, 2, 5], [2], [4], [6], [8]]
    assert expected_activities == activities


def test_read_temporal_file_time_stamp(test_data_directory, ):
    file_path = os.path.join(test_data_directory, 'example_timestamp.tedges')
    t = pp.TemporalNetwork.read_file(file_path, timestamp_format="%Y-%m-%d %H:%M")
    times = t.ordered_times
    time_diffs = [j - i for i, j in zip(times[:-1], times[1:])]
    expected_diffs = [10800, 15060, 264960]
    # TODO: The actual time number depends on local set by the user
    assert time_diffs == expected_diffs


def test_filter_temporal_edges(temporal_network_object):
    t = temporal_network_object

    def filter_func(v, w, time):
        return time % 2 == 0

    filtered = t.filter_edges(filter_func)
    times = filtered.ordered_times
    expected = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    assert times == expected


def test_get_interpath_times(temporal_network_object):
    t = temporal_network_object
    inter_time = dict(t.inter_path_times())
    expected = {'e': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                'b': [4, 3], 'f': [9, 5, 1]
                }
    assert inter_time == expected


def test_shuffle_edges(temporal_network_object):
    t = temporal_network_object

    np.random.seed(90)
    t1 = t.shuffle_edges(with_replacement=True)
    times1 = len(t1.tedges)
    expected1 = len(t.tedges)
    assert times1 == expected1

    np.random.seed(90)
    t2 = t.shuffle_edges(l=4, with_replacement=False)
    edges2 = len(t2.tedges)
    expected2 = 4
    assert edges2 == expected2


def test_inter_event_times(temporal_network_object):
    time_diffs = temporal_network_object.inter_event_times()
    # all time differences are 1
    assert (time_diffs == 1).all()


def test_inter_path_times(temporal_network_object):
    t = temporal_network_object
    path_times = dict(t.inter_path_times())
    expected = {'f': [9, 5, 1],
                'e': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                'b': [4, 3]}
    assert path_times == expected


def test_temporal_summary(temporal_network_object):
    print(temporal_network_object)


def test_export_tikz_unfolded_network(temporal_network_object, tmpdir):
    t = temporal_network_object  # type: pp.TemporalNetwork
    file_path = str(tmpdir.mkdir("sub").join("multi_order_state"))
    pp.visualisation.export_tikz(t, file_path)


def test_from_sqlite_int(test_data_directory, ):
    file_path = os.path.join(test_data_directory, 'test_tempnets.db')
    con = sqlite3.connect(file_path)
    con.row_factory = sqlite3.Row
    cursor = con.execute('SELECT source, target, time FROM example_int')

    t = pp.TemporalNetwork.from_sqlite(cursor)
    times = t.ordered_times
    expected_times = [0, 2, 4, 5, 6, 8]
    assert times == expected_times

    activities = sorted(list(t.activities.values()))
    expected_activities = [[], [], [], [], [0, 2, 5], [2], [4], [6], [8]]
    assert expected_activities == activities


def test_from_sqlite_timestamps(test_data_directory, ):
    file_path = os.path.join(test_data_directory, 'test_tempnets.db')
    con = sqlite3.connect(file_path)
    con.row_factory = sqlite3.Row
    cursor = con.execute('SELECT source, target, time FROM example_timestamp')
    t = pp.TemporalNetwork.from_sqlite(cursor, timestamp_format="%Y-%m-%d %H:%M")
    times = t.ordered_times
    time_diffs = [j - i for i, j in zip(times[:-1], times[1:])]
    expected_diffs = [10800, 15060, 264960]
    # TODO: The actual time number depends on local set by the user
    assert time_diffs == expected_diffs


def test_write_html(temporal_network_object, tmpdir):
    file_path = str(tmpdir.mkdir("sub").join("d3_temp.html"))
    t = temporal_network_object
    pp.visualisation.export_html(t, file_path)


@mark.latex
@mark.parametrize('is_dag', (False, True))
@mark.parametrize('split_dir', (False, True))
def test_write_tikz(temporal_network_object, tmpdir, is_dag, split_dir):
    dir_path = tmpdir
    file_path = str(dir_path.join("temp.tikz"))
    print(file_path)
    t = temporal_network_object
    t.write_tikz(file_path, dag=is_dag, split_directions=split_dir)

    cmd = "cd {}; pdflatex " \
          " -interaction nonstopmode {} > /dev/null".format(str(dir_path), file_path)
    exit_code = os.system(cmd)
    print(dir_path)
    assert exit_code == 0
