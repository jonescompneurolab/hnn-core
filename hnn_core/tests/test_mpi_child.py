import os.path as op
import os
import io
from contextlib import redirect_stdout, redirect_stderr
from queue import Queue
import pickle
import base64

import pytest

import hnn_core
from hnn_core import read_params, Network
from hnn_core.mpi_child import MPISimulation
from hnn_core.parallel_backends import (MPIBackend, _gather_trial_data,
                                        _extract_data, _extract_data_length)


def test_process_out_stderr():
    """Test _process_output for handling stderr, i.e. error messages"""
    # write data to queue
    out_q = Queue()
    err_q = Queue()
    test_string = "this gets printed to stdout"
    err_q.put(test_string)

    with MPIBackend() as backend:
        with io.StringIO() as buf_out, redirect_stdout(buf_out):
            data_len = backend._process_output(out_q, err_q)
            output = buf_out.getvalue()
        assert data_len is None
        assert output == test_string


def test_process_out_stdout():
    """Test _process_output for handling stdout, i.e. simulation messages"""
    # write data to queue
    out_q = Queue()
    err_q = Queue()
    test_string = "Test output"
    out_q.put(test_string)

    with MPIBackend() as backend:
        with io.StringIO() as buf_out, redirect_stdout(buf_out):
            data_len = backend._process_output(out_q, err_q)
            output = buf_out.getvalue()
        assert data_len is None
        assert output == test_string


def test_extract_data():
    """Test _extract_data for extraction between signals"""

    # no ending
    test_string = "@start_of_data@start of data"
    output = _extract_data(test_string, 'data')
    assert output == ''

    # valid end, but no start to data
    test_string = "end of data@end_of_data:11@"
    output = _extract_data(test_string, 'data')
    assert output == ''

    test_string = "@start_of_data@all data@end_of_data:8@"
    output = _extract_data(test_string, 'data')
    assert output == 'all data'


def test_extract_data_length():
    """Test _extract_data_length for data length in signal"""

    test_string = "end of data@end_of_data:@"
    with pytest.raises(ValueError, match="Couldn't find data length in "
                       "string"):
        _extract_data_length(test_string, 'data')

    test_string = "all data@end_of_data:8@"
    output = _extract_data_length(test_string, 'data')
    assert output == 8


def test_process_input():
    """Test reading the network via a Queue"""

    hnn_core_root = op.dirname(hnn_core.__file__)

    # prepare network
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    net = Network(params, add_drives_from_params=True)

    with MPISimulation(skip_mpi_import=True) as mpi_sim:
        pickled_net = mpi_sim._pickle_data(net)

        data_str = '@start_of_net@' + pickled_net.decode() + \
            '@end_of_net:%d@\n' % (len(pickled_net))

        # write contents to a Queue
        in_q = Queue()
        in_q.put(data_str)

        # process input from queue
        data_len = mpi_sim._process_input(in_q)
        assert isinstance(data_len, int)

        # unpickle net
        received_net = pickle.loads(base64.b64decode(mpi_sim.input_bytes,
                                                     validate=True))
        assert isinstance(received_net, Network)

        # muck with the data size in the signal
        data_str = '@start_of_net@' + pickled_net.decode() + \
            '@end_of_net:%d@\n' % (len(pickled_net) + 1)

        # write contents to a Queue
        in_q = Queue()
        in_q.put(data_str)

        expected_string = "Got incorrect network size: %d bytes " % \
            len(mpi_sim.input_bytes) + "expected length: %d" % \
            (len(pickled_net) + 1)

        # process input from queue
        with pytest.raises(ValueError, match=expected_string):
            mpi_sim._process_input(in_q)


def test_child_run():
    """Test running the child process without MPI"""

    hnn_core_root = op.dirname(hnn_core.__file__)

    # prepare params
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    params_reduced = params.copy()
    params_reduced.update({'N_pyr_x': 3,
                           'N_pyr_y': 3,
                           'tstop': 25,
                           't_evprox_1': 5,
                           't_evdist_1': 10,
                           't_evprox_2': 20,
                           'N_trials': 2})
    net_reduced = Network(params_reduced, add_drives_from_params=True)

    with MPISimulation(skip_mpi_import=True) as mpi_sim:
        with io.StringIO() as buf, redirect_stdout(buf):
            sim_data = mpi_sim.run(net_reduced)
            stdout = buf.getvalue()
        assert "Simulation time:" in stdout

        with io.StringIO() as buf_err, redirect_stderr(buf_err):
            mpi_sim._write_data_stderr(sim_data)
            stderr_str = buf_err.getvalue()
        assert "@start_of_data@" in stderr_str
        assert "@end_of_data:" in stderr_str

        # write data to queue
        out_q = Queue()
        err_q = Queue()
        err_q.put(stderr_str)

        # use _read_stderr to get data_len (but not the data this time)
        with MPIBackend() as backend:
            data_len = backend._process_output(out_q, err_q)
            sim_data = backend._process_child_data(backend.proc_data_bytes,
                                                   data_len)
        n_trials = 1
        postproc = False
        dpls = _gather_trial_data(sim_data, net_reduced, n_trials, postproc)
        assert len(dpls) == 1


def test_empty_data():
    """Test that an empty string raises RuntimeError"""
    data_bytes = b''
    expected_len = len(data_bytes)
    backend = MPIBackend()
    with pytest.raises(RuntimeError, match="MPI simulation didn't return any "
                       "data"):
        backend._process_child_data(data_bytes, expected_len)


def test_data_len_mismatch():
    """Test that padded data can be unpickled with warning for length """

    with MPISimulation(skip_mpi_import=True) as mpi_sim:
        pickled_bytes = mpi_sim._pickle_data({})

    expected_len = len(pickled_bytes) + 1

    backend = MPIBackend()
    with pytest.warns(UserWarning) as record:
        backend._process_child_data(pickled_bytes, expected_len)

    expected_string = "Length of received data unexpected. " + \
        "Expecting %d bytes, got %d" % (expected_len, len(pickled_bytes))

    assert len(record) == 1
    assert record[0].message.args[0] == expected_string
