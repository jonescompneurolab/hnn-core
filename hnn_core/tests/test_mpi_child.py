import os.path as op
import os
import io
from contextlib import redirect_stdout, redirect_stderr
import selectors

import pytest

import hnn_core
from hnn_core import read_params
from hnn_core.mpi_child import MPISimulation
from hnn_core.parallel_backends import MPIBackend


def test_read_stderr():
    """Test the _read_stderr handler for processing data and signals"""
    (pipe_stderr_r, pipe_stderr_w) = os.pipe()
    stderr = os.fdopen(pipe_stderr_w, 'w')
    backend = MPIBackend()

    stderr.write("test_data")
    stderr.flush()
    backend._read_stderr(pipe_stderr_r, selectors.EVENT_READ)
    assert backend.proc_data_bytes == "test_data".encode()
    backend.proc_data_bytes = b''

    stderr.write("@")
    stderr.flush()
    with pytest.raises(ValueError, match="Invalid signal start"):
        backend._read_stderr(pipe_stderr_r, selectors.EVENT_READ)

    stderr.write("@end_of_data:")
    stderr.flush()
    with pytest.raises(ValueError, match="Invalid signal start"):
        backend._read_stderr(pipe_stderr_r, selectors.EVENT_READ)

    stderr.write("@end_of_data:@")
    stderr.flush()
    with pytest.raises(ValueError, match="Completion signal from child MPI "
                       "process did not contain data length."):
        backend._read_stderr(pipe_stderr_r, selectors.EVENT_READ)

    stderr.write("blahblah@end_of_data:1000@blah")
    stderr.flush()
    data_len = backend._read_stderr(pipe_stderr_r, selectors.EVENT_READ)
    assert data_len == 1000
    assert backend.proc_data_bytes == "blahblahblah".encode()


def test_read_stdout():
    """Test the _read_stdout handler for processing simulation messages"""
    (pipe_stdout_r, pipe_stdout_w) = os.pipe()
    stdout = os.fdopen(pipe_stdout_w, 'w')
    backend = MPIBackend()

    stdout.write("Test output")
    stdout.flush()
    with io.StringIO() as buf_out, redirect_stdout(buf_out):
        backend._read_stdout(pipe_stdout_r, selectors.EVENT_READ)
        output = buf_out.getvalue()
    assert output == "Test output"

    stdout.write("end_of_sim")
    stdout.flush()
    signal = backend._read_stdout(pipe_stdout_r, selectors.EVENT_READ)
    assert signal == "end_of_sim"

    stdout.write("blahend_of_simblahblah")
    stdout.flush()
    with io.StringIO() as buf_out, redirect_stdout(buf_out):
        backend._read_stdout(pipe_stdout_r, selectors.EVENT_READ)
        output = buf_out.getvalue()
    assert output == "blahblahblah"
    assert signal == "end_of_sim"


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

    with MPISimulation(skip_mpi_import=True) as mpi_sim:
        with io.StringIO() as buf, redirect_stdout(buf):
            sim_data = mpi_sim.run(params_reduced)
            stdout = buf.getvalue()
        assert "end_of_sim" in stdout

        with io.StringIO() as buf_err, redirect_stderr(buf_err):
            with io.StringIO() as buf_out, redirect_stdout(buf_out):
                mpi_sim._write_data_stderr(sim_data)
                stdout = buf_out.getvalue()
            stderr_str = buf_err.getvalue()
        assert "@end_of_data:" in stderr_str

        # data will be before "@end_of_data:"
        signal_index_start = stderr_str.rfind('@end_of_data:')
        data = stderr_str[0:signal_index_start].encode()

        # setup stderr pipe just for signal (with data_len)
        (pipe_stderr_r, pipe_stderr_w) = os.pipe()
        stderr_fd = os.fdopen(pipe_stderr_w, 'w')
        stderr_fd.write(stderr_str[signal_index_start:])
        stderr_fd.flush()

        # use _read_stderr to get data_len (but not the data this time)
        backend = MPIBackend()
        data_len = backend._read_stderr(pipe_stderr_r, selectors.EVENT_READ)
        sim_data = backend._process_child_data(data, data_len)


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
