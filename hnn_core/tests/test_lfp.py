# Authors: Nick Tolley <nicholas_tolley@brown.edu>
#          Christopher Bailey <cjb@cfin.au.dk>

from copy import deepcopy
import os.path as op
import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import hnn_core
from hnn_core import read_params, default_network, simulate_dipole, Network
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
    net.add_electrode_array('el1', electrode_pos, sigma, method)
    electrode_pos = [(2, 400, 2), (6, 800, 6)]
    net.add_electrode_array('arr1', electrode_pos, sigma, method)
    assert len(net.lfp_array) == 2
    assert len(net.lfp_array['arr1']) == 2  # length == n.o. electrodes

    pytest.raises(ValueError, net.add_electrode_array,
                  'arr1', [(6, 6, 800)], sigma, method)
    pytest.raises(TypeError, net.add_electrode_array,
                  42, [(6, 6, 800)], sigma, method)
    pytest.raises(AssertionError, net.add_electrode_array,
                  'arr2', [(2, 2), (6, 6, 800)], sigma, method)
    pytest.raises(AssertionError, net.add_electrode_array,
                  'arr2', electrode_pos, -1.0, method)

    pytest.raises(ValueError, net.add_electrode_array,
                  'arr2', electrode_pos, sigma, 'LSA')

    pytest.raises(TypeError, net.add_electrode_array,
                  'arr2', '(2, 2, 400)', sigma, method)
    pytest.raises(TypeError, net.add_electrode_array,
                  'arr2', (2, '2', 400), sigma, method)
    pytest.raises(TypeError, net.add_electrode_array,
                  'arr2', electrode_pos, '0.3', method)
    pytest.raises(TypeError, net.add_electrode_array,
                  'arr2', electrode_pos, sigma, 3.0)


@requires_mpi4py
@requires_psutil
def test_lfp_backends(run_hnn_core_fixture):
    """Test lfp outputs across backends."""

    electrode_array = {'arr1': [(2, 400, 2), (6, 800, 6)]}
    _, joblib_net = run_hnn_core_fixture(
        backend='joblib', n_jobs=1, reduced=True, record_isoma=True,
        record_vsoma=True, electrode_array=electrode_array)
    _, mpi_net = run_hnn_core_fixture(
        backend='mpi', n_procs=2, reduced=True, record_isoma=True,
        record_vsoma=True, electrode_array=electrode_array)

    assert (len(electrode_array['arr1']) ==
            len(joblib_net.lfp_array['arr1']) ==
            len(mpi_net.lfp_array['arr1']))

    # reduced simulation has n_trials=2
    # trial_idx, n_trials = 0, 2
    for tr_idx, el_idx in zip([0, 1], [0, 1]):
        assert_allclose(joblib_net.lfp_array['arr1']._data[tr_idx][el_idx],
                        mpi_net.lfp_array['arr1']._data[tr_idx][el_idx])

    assert isinstance(joblib_net.lfp_array['arr1'].get_data(), np.ndarray)
    assert_array_equal(joblib_net.lfp_array['arr1'].get_data().shape,
                       [len(joblib_net.lfp_array['arr1']._data),
                        len(joblib_net.lfp_array['arr1']._data[0]),
                        len(joblib_net.lfp_array['arr1']._data[0][0])])


def _mathematical_dipole(e_pos, d_pos, d_Q):
    rr = e_pos - d_pos
    R = norm(rr)
    Q = norm(d_Q)
    cosT = np.dot(rr, d_Q) / (R * Q)
    return (Q * cosT) / (4 * np.pi * R ** 2)


# require MPI to speed up due to large number of LFP electrodes
@requires_mpi4py
def test_dipolar_far_field():
    """Test that LFP in the far field is dipolar when expected."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    params.update({'N_pyr_x': 3,
                   'N_pyr_y': 3,
                   'tstop': 25,
                   })
    # initialise an unconnected network
    net = Network(params)

    # issue _weak_ excitatory drive to distal apical dendrites
    # NB must not cause Na- or Ca-spiking, as these are not associated with
    # dipolar fields
    weights_nmda = {'L2_basket': .0, 'L2_pyramidal': .0005,
                    'L2_basket': .0, 'L5_pyramidal': .0005}
    net.add_evoked_drive('d', mu=10., sigma=0., numspikes=1, location='distal',
                         sync_within_trial=True, weights_nmda=weights_nmda)

    sigma = 0.3
    method = 'psa'  # at these distances, psa and lsa are identical

    # create far-field grid of LFP electrodes; note that cells are assumed
    # to lie in the XY-plane
    xmin, xmax = -5e4, 5e4
    ymin, ymax = -5e4, 5e4
    step = 5e3
    posz = 1e2  # out-of-plane
    electrode_pos = list()
    for posx in np.arange(xmin, xmax, step):
        for posy in np.arange(ymin, ymax, step):
            electrode_pos.append((posx, posy, posz))
    net.add_electrode_array('grid', electrode_pos, sigma=sigma, method=method)

    with MPIBackend(n_procs=2):
        dpl = simulate_dipole(net, postproc=False)

    X_p = np.arange(xmin, xmax, step) / 1000
    Y_p = np.arange(ymin, ymax, step) / 1000
    Z_p = posz / 1000
    idt = np.argmin(np.abs(dpl[0].times - 15.))
    phi_p = np.zeros((len(X_p), len(Y_p)))
    phi_p_theory = np.zeros((len(X_p), len(Y_p)))

    # location of equivalent current dipole for this stimulation (manual)
    d_pos = np.array((0, 1600, 0)) / 1000  # um -> mm
    # dipole orientation is along the apical dendrite, towards the soma
    # the amplitude is really irrelevant, only shape is compared
    d_Q = 5e2 * np.array((0, -1, 0))

    for ii, row in enumerate(X_p):
        for jj, col in enumerate(Y_p):

            e_pos = np.array((row, col, Z_p))

            # ignore 10 mm radius closest to dipole
            if norm(e_pos - d_pos) < 10:
                phi_p[ii][jj] = 0
                phi_p_theory[ii][jj] = 0
                continue

            phi_p[ii][jj] = net.lfp_array['grid']._data[0][
                ii * len(X_p) + jj][idt] * 1e3
            phi_p_theory[ii][jj] = \
                _mathematical_dipole(e_pos, d_pos, d_Q) / sigma

    # compare the shape of the far fields
    cosT = np.dot(phi_p.ravel(), phi_p_theory.ravel()) / (
        norm(phi_p.ravel()) * norm(phi_p_theory.ravel()))

    # the far field should be very close to dipolar, though threshold may need
    # adjusting when new mechanisms are included in the cells
    assert 1 - cosT < 1e-3

    # for diagnostic plots, uncomment the following:
    # import matplotlib.pyplot as plt
    # from matplotlib.colors import SymLogNorm
    # fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    # for ax, phi in zip(axs, [phi_p, phi_p_theory]):
    #     ax.pcolormesh(X_p, Y_p, phi.T,
    #                   norm=SymLogNorm(linthresh=1e-2, linscale=1.,
    #                                   vmin=-5e0, vmax=5e0, base=10),
    #                   cmap='BrBG_r', shading='auto')
    # plt.show()


def test_lfp_array_calculation():
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
    net.add_electrode_array('arr1', electrode_pos, sigma, method)
    _ = simulate_dipole(net, n_trials=1)
    assert len(net.lfp_array['arr1']._data) == 1  # n_trials
    assert len(net.lfp_array['arr1']._data[0]) == 2  # n_contacts
    assert (len(net.lfp_array['arr1']._data[0][0]) ==
            len(net.lfp_array['arr1'].times))
    # temporary, while working on PSA and LSA implementations
    # "gold standard" data are based on the output of c75cf239 (PSA)
    # Note that the lfp calculation was since extended by 1 data sample
    # (to match the length of Dipole); the first N-1 samples are compared
    test_data = np.load(op.join(op.dirname(__file__),
                                'temp_lfp_test_data.npy'))
    trial_idx = 0
    for ele_idx in range(len(electrode_pos)):
        sim_data = net.lfp_array['arr1']._data[trial_idx][ele_idx]
        assert_allclose(test_data[:, ele_idx], sim_data)
        assert (len(sim_data) == len(net.lfp_array['arr1'].times))

    sigma, method = 0.3, 'lsa'
    electrode_pos = [(6, 800, 6)]  # same as 2nd contact in 'arr1'
    net.add_electrode_array('arr2', electrode_pos, sigma, method)

    # make sure no sinister segfaults are triggered when running mult. trials
    n_trials = 10  # NB 10 trials!
    _ = simulate_dipole(net, n_trials=n_trials)

    # simulate_dipole is run twice above, first 1 then 10 trials.
    # Make sure that previous results are discarded on each run
    assert len(net.lfp_array['arr1']._data) == n_trials

    # LSA and PSA should agree far away
    assert_allclose(net.lfp_array['arr1']._data[trial_idx][1],
                    net.lfp_array['arr2']._data[trial_idx][0],
                    rtol=1e-1, atol=1e-1)
