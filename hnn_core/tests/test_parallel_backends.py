import os.path as op
from os import environ
import io
from contextlib import redirect_stdout
from multiprocessing import cpu_count
from numpy import loadtxt
from numpy.testing import assert_array_equal, assert_allclose, assert_raises

import pytest
from mne.utils import _fetch_file

import hnn_core
from hnn_core import read_params
from hnn_core import MPIBackend
from hnn_core.parallel_backends import requires_mpi4py

# The purpose of this incremental mark is to avoid running the full length
# simulation when there are failures in previous (faster) tests. When a test
# in the sequence fails, all subsequent tests will be marked "xfailed" rather
# than skipped.


@pytest.mark.incremental
class TestParallelBackends():
    dpls_reduced_mpi = None
    dpls_reduced_default = None
    dpls_reduced_joblib = None

    def test_run_default(self, run_hnn_core_fixture):
        """Test consistency between default backend simulation and master"""
        global dpls_reduced_default
        dpls_reduced_default, _ = run_hnn_core_fixture(None, reduced=True)
        # test consistency across all parallel backends for multiple trials
        assert_raises(AssertionError, assert_array_equal,
                      dpls_reduced_default[0].data['agg'],
                      dpls_reduced_default[1].data['agg'])

    def test_run_joblibbackend(self, run_hnn_core_fixture):
        """Test consistency between joblib backend simulation with master"""
        global dpls_reduced_default, dpls_reduced_joblib

        dpls_reduced_joblib, _ = run_hnn_core_fixture(backend='joblib',
                                                      n_jobs=2, reduced=True)

        for trial_idx in range(len(dpls_reduced_default)):
            assert_array_equal(dpls_reduced_default[trial_idx].data['agg'],
                               dpls_reduced_joblib[trial_idx].data['agg'])

    @requires_mpi4py
    def test_mpi_nprocs(self):
        """Test that MPIBackend can use more than 1 processor"""
        # if only 1 processor is available, then MPIBackend tests will not
        # be valid
        backend = MPIBackend()
        assert backend.n_procs > 1

    @requires_mpi4py
    def test_run_mpibackend(self, run_hnn_core_fixture):
        """Test running a MPIBackend on reduced model"""
        global dpls_reduced_default, dpls_reduced_mpi
        dpls_reduced_mpi, _ = run_hnn_core_fixture(backend='mpi', reduced=True)
        for trial_idx in range(len(dpls_reduced_default)):
            # account for rounding error incured during MPI parallelization
            assert_allclose(dpls_reduced_default[trial_idx].data['agg'],
                            dpls_reduced_mpi[trial_idx].data['agg'], rtol=0,
                            atol=1e-14)

    @requires_mpi4py
    def test_run_mpibackend_oversubscribed(self, run_hnn_core_fixture):
        """Test running MPIBackend with oversubscribed number of procs"""
        oversubscribed = round(cpu_count() * 1.5)
        run_hnn_core_fixture(backend='mpi', n_procs=oversubscribed,
                             reduced=True)

    @pytest.mark.parametrize("backend", ['mpi', 'joblib'])
    def test_compare_hnn_core(self, run_hnn_core_fixture, backend, n_jobs=1):
        """Test hnn-core does not break."""
        # small snippet of data on data branch for now. To be deleted
        # later. Data branch should have only commit so it does not
        # pollute the history.
        data_url = ('https://raw.githubusercontent.com/jonescompneurolab/'
                    'hnn-core/test_data/dpl.txt')
        if not op.exists('dpl.txt'):
            _fetch_file(data_url, 'dpl.txt')
        dpl_master = loadtxt('dpl.txt')

        hnn_core_root = op.dirname(hnn_core.__file__)

        # default params
        params_fname = op.join(hnn_core_root, 'param', 'default.json')
        params = read_params(params_fname)

        dpls, net = run_hnn_core_fixture(params, backend)
        dpl = dpls[0]

        # write the dipole to a file and compare
        fname = './dpl2.txt'
        dpl.write(fname)

        dpl_pr = loadtxt(fname)
        assert_array_equal(dpl_pr[:, 2], dpl_master[:, 2])  # L2
        assert_array_equal(dpl_pr[:, 3], dpl_master[:, 3])  # L5

        # Test spike type counts
        spike_type_counts = {}
        for spike_gid in net.cell_response.spike_gids[0]:
            if net.gid_to_type(spike_gid) not in spike_type_counts:
                spike_type_counts[net.gid_to_type(spike_gid)] = 0
            else:
                spike_type_counts[net.gid_to_type(spike_gid)] += 1
        assert 'common' not in spike_type_counts
        assert 'exgauss' not in spike_type_counts
        assert 'extpois' not in spike_type_counts
        assert spike_type_counts == {'evprox1': 269,
                                     'L2_basket': 54,
                                     'L2_pyramidal': 113,
                                     'L5_pyramidal': 395,
                                     'L5_basket': 85,
                                     'evdist1': 234,
                                     'evprox2': 269}


# there are no dependencies if this unit tests fails; no need to be in
# class marked incremental
@requires_mpi4py
def test_mpi_failure(run_hnn_core_fixture):
    """Test that an MPI failure is handled and messages are printed"""
    # this MPI paramter will cause a MPI job to fail
    environ["OMPI_MCA_btl"] = "self"

    with pytest.warns(UserWarning) as record:
        with io.StringIO() as buf, redirect_stdout(buf):
            with pytest.raises(RuntimeError, match="MPI simulation failed"):
                run_hnn_core_fixture(backend='mpi', reduced=True)
            stdout = buf.getvalue()

    assert "MPI processes are unable to reach each other" in stdout

    expected_string = "Timed out (5s) waiting for end of data " + \
        "after child process stopped"
    assert len(record) == 1
    assert record[0].message.args[0] == expected_string

    del environ["OMPI_MCA_btl"]
