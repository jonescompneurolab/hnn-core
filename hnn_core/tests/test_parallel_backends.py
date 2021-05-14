import os.path as op
from os import environ
import io
from contextlib import redirect_stdout
from multiprocessing import cpu_count
from numpy import loadtxt
from numpy.testing import assert_array_equal, assert_allclose, assert_raises
from threading import Thread, Event
from time import sleep

import pytest
from mne.utils import _fetch_file

import hnn_core
from hnn_core import MPIBackend, default_network, read_params
from hnn_core.dipole import simulate_dipole
from hnn_core.parallel_backends import requires_mpi4py, requires_psutil


def _terminate_mpibackend(event, backend):
    # wait for run_subprocess to start MPI proc and put handle on queue
    proc = backend.proc_queue.get()
    # put the proc back in the queue. used by backend.terminate()
    backend.proc_queue.put(proc)

    # give the process a little time to startup
    sleep(0.1)

    # run terminate until it is successful
    while not event.isSet():
        backend.terminate()
        sleep(0.01)


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
    @requires_psutil
    def test_mpi_nprocs(self):
        """Test that MPIBackend can use more than 1 processor"""
        # if only 1 processor is available, then MPIBackend tests will not
        # be valid
        with MPIBackend() as backend:
            assert backend.n_procs > 1

    @requires_mpi4py
    @requires_psutil
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
    @requires_psutil
    def test_terminate_mpibackend(self, run_hnn_core_fixture):
        """Test terminating MPIBackend from thread"""
        hnn_core_root = op.dirname(hnn_core.__file__)
        params_fname = op.join(hnn_core_root, 'param', 'default.json')
        params = read_params(params_fname)
        params.update({'N_pyr_x': 3,
                       'N_pyr_y': 3,
                       'tstop': 40,
                       't_evprox_1': 5,
                       't_evdist_1': 10,
                       't_evprox_2': 20,
                       'N_trials': 2})
        net = default_network(params, add_drives_from_params=True)

        with MPIBackend() as backend:
            event = Event()
            # start background thread that will kill all MPIBackends
            # until event.set()
            kill_t = Thread(target=_terminate_mpibackend,
                            args=(event, backend))
            # make thread a daemon in case we throw an exception
            # and don't run event.set() so that py.test will
            # not hang before exiting
            kill_t.daemon = True
            kill_t.start()

            with pytest.warns(UserWarning) as record:
                with pytest.raises(
                        RuntimeError,
                        match="MPI simulation failed. Return code: 1"):
                    simulate_dipole(net)

            event.set()
        expected_string = "Child process failed unexpectedly"
        assert expected_string in record[0].message.args[0]

    @requires_mpi4py
    @requires_psutil
    def test_run_mpibackend_oversubscribed(self, run_hnn_core_fixture):
        """Test running MPIBackend with oversubscribed number of procs"""
        hnn_core_root = op.dirname(hnn_core.__file__)
        params_fname = op.join(hnn_core_root, 'param', 'default.json')
        params = read_params(params_fname)
        params.update({'N_pyr_x': 3,
                       'N_pyr_y': 3,
                       'tstop': 40,
                       't_evprox_1': 5,
                       't_evdist_1': 10,
                       't_evprox_2': 20,
                       'N_trials': 2})
        net = default_network(params, add_drives_from_params=True)

        oversubscribed = round(cpu_count() * 1.5)
        with MPIBackend(n_procs=oversubscribed) as backend:
            assert backend.n_procs == oversubscribed
            simulate_dipole(net)

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

        dpls, net = run_hnn_core_fixture(backend=backend)
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
@requires_psutil
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

    expected_string = "Child process failed unexpectedly"
    assert len(record) == 1
    assert record[0].message.args[0] == expected_string

    del environ["OMPI_MCA_btl"]
