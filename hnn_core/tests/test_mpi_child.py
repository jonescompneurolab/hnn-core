import os.path as op
import io
from contextlib import redirect_stdout, redirect_stderr
from queue import Queue

import pytest

import hnn_core
from hnn_core import read_params, Network, jones_2009_model
from hnn_core.mpi_child import MPISimulation, _str_to_net, _pickle_data
from hnn_core.parallel_backends import (
    _gather_trial_data,
    _process_child_data,
    _echo_child_output,
    _get_data_from_child_err,
    _extract_data,
    _extract_data_length,
)


def test_get_data_from_child_err():
    """Test _get_data_from_child_err for handling stderr"""
    # write data to queue
    err_q = Queue()
    test_string = "this gets printed to stdout"
    err_q.put(test_string)

    with io.StringIO() as buf_out, redirect_stdout(buf_out):
        _get_data_from_child_err(err_q)
        output = buf_out.getvalue()
    assert output == test_string


def test_echo_child_output():
    """Test _echo_child_output for handling stdout, i.e. status messages"""
    # write data to queue
    out_q = Queue()
    test_string = "Test output"
    out_q.put(test_string)

    with io.StringIO() as buf_out, redirect_stdout(buf_out):
        got_output = _echo_child_output(out_q)
        output = buf_out.getvalue()
    assert got_output
    assert output == test_string


def test_extract_data():
    """Test _extract_data for extraction between signals"""

    # no ending
    test_string = "@start_of_data@start of data"
    output = _extract_data(test_string, "data")
    assert output == ""

    # valid end, but no start to data
    test_string = "end of data@end_of_data:11@"
    output = _extract_data(test_string, "data")
    assert output == ""

    test_string = "@start_of_data@all data@end_of_data:8@"
    output = _extract_data(test_string, "data")
    assert output == "all data"


def test_extract_data_length():
    """Test _extract_data_length for data length in signal"""

    test_string = "end of data@end_of_data:@"
    with pytest.raises(ValueError, match="Couldn't find data length in string"):
        _extract_data_length(test_string, "data")

    test_string = "all data@end_of_data:8@"
    output = _extract_data_length(test_string, "data")
    assert output == 8


def test_str_to_net():
    """Test reading the network via a string"""

    hnn_core_root = op.dirname(hnn_core.__file__)

    # prepare network
    params_fname = op.join(hnn_core_root, "param", "default.json")
    params = read_params(params_fname)
    net = jones_2009_model(params, add_drives_from_params=True)

    pickled_net = _pickle_data(net)

    input_str = (
        "@start_of_net@"
        + pickled_net.decode()
        + "@end_of_net:%d@\n" % (len(pickled_net))
    )

    received_net = _str_to_net(input_str)
    assert isinstance(received_net, Network)

    # muck with the data size in the signal
    input_str = (
        "@start_of_net@"
        + pickled_net.decode()
        + "@end_of_net:%d@\n" % (len(pickled_net) + 1)
    )

    expected_string = "Got incorrect network size: %d bytes " % len(
        pickled_net
    ) + "expected length: %d" % (len(pickled_net) + 1)

    # process input from queue
    with pytest.raises(ValueError, match=expected_string):
        _str_to_net(input_str)


def test_child_run():
    """Test running the child process without MPI"""

    hnn_core_root = op.dirname(hnn_core.__file__)

    # prepare params
    params_fname = op.join(hnn_core_root, "param", "default.json")
    params = read_params(params_fname)
    params_reduced = params.copy()
    params_reduced.update({"t_evprox_1": 5, "t_evdist_1": 10, "t_evprox_2": 20})
    tstop, n_trials = 25, 2
    net_reduced = jones_2009_model(
        params_reduced, add_drives_from_params=True, mesh_shape=(3, 3)
    )
    net_reduced._instantiate_drives(tstop=tstop, n_trials=n_trials)

    with MPISimulation(skip_mpi_import=True) as mpi_sim:
        with io.StringIO() as buf, redirect_stdout(buf):
            sim_data = mpi_sim.run(
                net_reduced, tstop=tstop, dt=0.025, n_trials=n_trials
            )
            stdout = buf.getvalue()
        assert "Trial 1: 0.03 ms..." in stdout

        with io.StringIO() as buf_err, redirect_stderr(buf_err):
            mpi_sim._write_data_stderr(sim_data)
            stderr_str = buf_err.getvalue()
        assert "@start_of_data@" in stderr_str
        assert "@end_of_data:" in stderr_str

        # write data to queue
        err_q = Queue()
        err_q.put(stderr_str)

        # use _read_stderr to get data_len (but not the data this time)
        data_len, data = _get_data_from_child_err(err_q)
        sim_data = _process_child_data(data, data_len)
        n_trials = 1
        postproc = False
        dpls = _gather_trial_data(sim_data, net_reduced, n_trials, postproc)
        assert len(dpls) == 1


def test_empty_data():
    """Test that an empty string raises RuntimeError"""
    data_bytes = b""
    with pytest.raises(RuntimeError, match="MPI simulation didn't return any data"):
        _process_child_data(data_bytes, len(data_bytes))


def test_data_len_mismatch():
    """Test that padded data can be unpickled with warning for length"""

    pickled_bytes = _pickle_data({})

    expected_len = len(pickled_bytes) + 1

    with pytest.warns(UserWarning) as record:
        _process_child_data(pickled_bytes, expected_len)

    expected_string = (
        "Length of received data unexpected. "
        + "Expecting %d bytes, got %d" % (expected_len, len(pickled_bytes))
    )

    assert len(record) == 1
    assert record[0].message.args[0] == expected_string
