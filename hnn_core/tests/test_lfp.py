# Authors: Nick Tolley <nicholas_tolley@brown.edu>
#          Christopher Bailey <cjb@cfin.au.dk>

from copy import deepcopy
import os.path as op
import numpy as np
from numpy.testing import assert_allclose
import pytest

import hnn_core
from hnn_core import read_params, default_network, simulate_dipole
from hnn_core.parallel_backends import requires_mpi4py, requires_psutil
from hnn_core.parallel_backends import MPIBackend


def test_lfp_api():
    """Test LFP recording API."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    net = default_network(deepcopy(params), add_drives_from_params=True)

    # Test LFP electrodes
    electrode_pos = (2, 400, 2)
    sigma, method = 0.3, 'psa'
    net.add_electrode(electrode_pos, sigma, method)
    electrode_pos = [(2, 400, 2), (6, 800, 6)]
    net.add_electrode(electrode_pos, sigma, method)
    assert len(net.lfp) == 3

    pytest.raises(AssertionError, net.add_electrode,
                  [(2, 2), (6, 6, 800)], sigma, method)
    pytest.raises(AssertionError, net.add_electrode,
                  electrode_pos, -1.0, method)

    pytest.raises(ValueError, net.add_electrode,
                  electrode_pos, sigma, 'LSA')

    pytest.raises(TypeError, net.add_electrode,
                  '(2, 2, 400)', sigma, method)
    pytest.raises(TypeError, net.add_electrode,
                  (2, '2', 400), sigma, method)
    pytest.raises(TypeError, net.add_electrode,
                  electrode_pos, '0.3', method)
    pytest.raises(TypeError, net.add_electrode,
                  electrode_pos, sigma, 3.0)


@requires_mpi4py
@requires_psutil
def test_lfp_backends(run_hnn_core_fixture):
    """Test lfp outputs across backends."""

    # reduced simulation has n_trials=2
    # trial_idx, n_trials = 0, 2
    electrode_pos = [(2, 400, 2), (6, 800, 6)]
    _, joblib_net = run_hnn_core_fixture(
        backend='joblib', n_jobs=1, reduced=True, record_isoma=True,
        record_vsoma=True, electrode_pos=electrode_pos)
    _, mpi_net = run_hnn_core_fixture(
        backend='mpi', n_procs=2, reduced=True, record_isoma=True,
        record_vsoma=True, electrode_pos=electrode_pos)

    assert len(electrode_pos) == len(joblib_net.lfp) == len(mpi_net.lfp)
    assert_allclose(joblib_net.lfp[0]['data'], mpi_net.lfp[0]['data'])
    assert_allclose(joblib_net.lfp[1]['data'], mpi_net.lfp[1]['data'])


def test_lfp_calculation():
    """Test LFP calculation."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    params.update({'N_pyr_x': 3,
                   'N_pyr_y': 3,
                   'tstop': 25,
                   'dipole_smooth_win': 5,
                   't_evprox_1': 7,
                   't_evdist_1': 17})
    net = default_network(params, add_drives_from_params=True)

    sigma, method = 0.3, 'psa'
    electrode_pos = [(2, 400, 2), (6, 800, 6)]  # one inside, one outside net
    net.add_electrode(electrode_pos, sigma, method)
    with MPIBackend(n_procs=2):
        _ = simulate_dipole(net, n_trials=1)

    # temporary, while working on PSA and LSA implementations
    # "gold standard" data are based on the output of c75cf239 (PSA)
    # Note that the lfp calculation was since extended by 1 data sample
    # (to match the length of Dipole); the first N-1 samples are compared
    test_data = np.load(op.join(op.dirname(__file__),
                                'temp_lfp_test_data.npy'))
    for ele_idx in range(len(electrode_pos)):
        sim_data = net.lfp[ele_idx]['data'][0]
        assert_allclose(test_data[:, ele_idx], sim_data[:-1])
        assert len(net.cell_response.times) == len(net.lfp[ele_idx]['data'][0])

    net = default_network(params, add_drives_from_params=True)

    sigma, method = 0.3, 'lsa'
    electrode_pos = [(6, 800, 6)]  # LSA and PSA should agree far away
    net.add_electrode(electrode_pos, sigma, method)
    # make sure no sinister segfaults are triggered when running mult. trials
    _ = simulate_dipole(net, n_trials=10)
    assert_allclose(test_data[:, 1], net.lfp[0]['data'][0][:-1],
                    rtol=1e-1, atol=1e-1)
