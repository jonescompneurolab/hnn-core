import os.path as op
from os import environ
import pytest
import io
from contextlib import redirect_stdout

from numpy import loadtxt
from numpy.testing import assert_array_equal, assert_allclose, assert_raises

from mne.utils import _fetch_file
import hnn_core
from hnn_core import simulate_dipole, Network, read_params
from hnn_core import MPIBackend, JoblibBackend


def run_hnn_core(backend=None, n_jobs=1):
    """Test to check if hnn-core does not break."""
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
    params_reduced = params.copy()
    params_reduced.update({'N_pyr_x': 3,
                           'N_pyr_y': 3,
                           'tstop': 25,
                           't_evprox_1': 5,
                           't_evdist_1': 10,
                           't_evprox_2': 20,
                           'N_trials': 2})

    # run the simulation on full model (1 trial) and a reduced model (2 trials)
    net = Network(params)
    net_reduced = Network(params_reduced)

    if backend == 'mpi':
        with MPIBackend(n_procs=2, mpi_cmd='mpiexec'):
            dpl = simulate_dipole(net)[0]
            dpls_reduced = simulate_dipole(net_reduced)
    elif backend == 'joblib':
        with JoblibBackend(n_jobs=n_jobs):
            dpl = simulate_dipole(net)[0]
            dpls_reduced = simulate_dipole(net_reduced)
    else:
        dpl = simulate_dipole(net)[0]
        dpls_reduced = simulate_dipole(net_reduced)

    # write the dipole to a file and compare
    fname = './dpl2.txt'
    dpl.write(fname)

    dpl_pr = loadtxt(fname)
    assert_array_equal(dpl_pr[:, 2], dpl_master[:, 2])  # L2
    assert_array_equal(dpl_pr[:, 3], dpl_master[:, 3])  # L5

    # Test spike type counts
    spiketype_counts = {}
    for spikegid in net.spikes.gids[0]:
        if net.gid_to_type(spikegid) not in spiketype_counts:
            spiketype_counts[net.gid_to_type(spikegid)] = 0
        else:
            spiketype_counts[net.gid_to_type(spikegid)] += 1
    assert 'common' not in spiketype_counts
    assert 'exgauss' not in spiketype_counts
    assert 'extpois' not in spiketype_counts
    assert spiketype_counts == {'evprox1': 269,
                                'L2_basket': 54,
                                'L2_pyramidal': 113,
                                'L5_pyramidal': 395,
                                'L5_basket': 85,
                                'evdist1': 234,
                                'evprox2': 269}
    return dpls_reduced


def test_compare_across_backends():
    """Test that trials are generated consistently across parallel backends."""

    # test consistency between default backend simulation and master
    dpls_reduced_default = run_hnn_core(None)

    try:
        import mpi4py
        mpi4py.__file__
        # test consistency between mpi backend simulation (n_procs=2) & master
        dpls_reduced_mpi = run_hnn_core(backend='mpi')
    except ImportError:
        print("Skipping MPIBackend test and dipole comparison because mpi4py "
              "could not be imported...")
        dpls_reduced_mpi = None

    # test consistency between joblib backend simulation (n_jobs=2) with master
    dpls_reduced_joblib = run_hnn_core(backend='joblib', n_jobs=2)

    # test consistency across all parallel backends for multiple trials
    assert_raises(AssertionError, assert_array_equal,
                  dpls_reduced_default[0].data['agg'],
                  dpls_reduced_default[1].data['agg'])

    for trial_idx in range(len(dpls_reduced_default)):
        # account for rounding error incured during MPI parallelization
        if dpls_reduced_mpi:
            assert_allclose(dpls_reduced_default[trial_idx].data['agg'],
                            dpls_reduced_mpi[trial_idx].data['agg'], rtol=0,
                            atol=1e-14)
        assert_array_equal(dpls_reduced_default[trial_idx].data['agg'],
                           dpls_reduced_joblib[trial_idx].data['agg'])


def test_mpi_failure():
    """Test that an MPI failure is handled and error messages pass through"""
    pytest.importorskip("mpi4py", reason="mpi4py not available")

    # this MPI paramter will cause a MPI job with more than one process to fail
    environ["OMPI_MCA_btl"] = "self"

    with io.StringIO() as buf, redirect_stdout(buf):
        with pytest.raises(RuntimeError, match="MPI simulation failed"):
            run_hnn_core(backend='mpi')
        stdout = buf.getvalue()
    assert "MPI processes are unable to reach each other" in stdout

    del environ["OMPI_MCA_btl"]
