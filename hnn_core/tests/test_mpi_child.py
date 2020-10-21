import os.path as op
import io
from contextlib import redirect_stdout, redirect_stderr
import pytest

import hnn_core
from hnn_core import read_params
from hnn_core.mpi_child import MPISimulation
from hnn_core.parallel_backends import MPIBackend


def test_child_run():
    """Test running the MPI child process"""

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
            stderr = buf_err.getvalue()
        assert "end_of_data:" in stderr

        # data will all be before "end_of_data"
        data = stderr.split('end_of_data')[0].encode()
        expected_len = len(data)
        backend = MPIBackend()
        sim_data = backend._process_child_data(data, expected_len)


def test_empty_data():
    """Test that an empty string raises RuntimeError"""
    data_bytes = b''
    expected_len = len(data_bytes)
    backend = MPIBackend()
    with pytest.raises(RuntimeError, match="MPI simulation didn't return any "
                       "data"):
        backend._process_child_data(data_bytes, expected_len)


def test_data_len_mismatch():
    """Test that an unexpected data length raises RuntimeError"""
    data_bytes = b'\0'
    expected_len = 2
    backend = MPIBackend()
    with pytest.raises(RuntimeError, match="Failed to receive all data from "
                       "the child MPI process. Expecting 2 bytes, got 1"):
        backend._process_child_data(data_bytes, expected_len)
