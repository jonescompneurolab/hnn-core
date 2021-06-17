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
from hnn_core.extracellular import ExtracellularArray
from hnn_core.parallel_backends import requires_mpi4py, requires_psutil
from hnn_core.parallel_backends import MPIBackend


hnn_core_root = op.dirname(hnn_core.__file__)
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)


def test_extracellular_api():
    """Test extracellular recording API."""
    net = default_network(deepcopy(params), add_drives_from_params=True)

    # Test LFP electrodes
    electrode_pos = (1, 2, 3)
    net.add_electrode_array('el1', electrode_pos)
    electrode_pos = [(1, 2, 3), (-1, -2, -3)]
    net.add_electrode_array('arr1', electrode_pos)
    assert len(net.rec_arrays) == 2
    assert len(net.rec_arrays['arr1'].positions) == 2

    # ensure unique names
    pytest.raises(ValueError, net.add_electrode_array, 'arr1', [(6, 6, 800)])
    # all remaining input arguments checked by ExtracellularArray

    rec_arr = ExtracellularArray(electrode_pos)
    with pytest.raises(AttributeError, match="can't set attribute"):
        rec_arr.times = [1, 2, 3]
    with pytest.raises(AttributeError, match="can't set attribute"):
        rec_arr.voltages = [1, 2, 3]
    with pytest.raises(TypeError, match="trial index must be int"):
        _ = rec_arr['0']
    with pytest.raises(IndexError, match="the data contain"):
        _ = rec_arr[42]

    # positions are 3-tuples
    bad_positions = [[(1, 2), (1, 2, 3)], [42, (1, 2, 3)]]
    for bogus_pos in bad_positions:
        pytest.raises((ValueError, TypeError), ExtracellularArray, bogus_pos)

    good_positions = [(1, 2, 3), (100, 200, 300)]
    for cond in ['0.3', [0.3], -1]:  # conductivity is positive float
        pytest.raises((TypeError, AssertionError), ExtracellularArray,
                      good_positions, conductivity=cond)
    for meth in ['foo', 0.3]:  # method is 'psa' or 'lsa' (or None for test)
        pytest.raises((TypeError, AssertionError, ValueError),
                      ExtracellularArray, good_positions, method=meth)
    for mind in ['foo', -1, None]:  # minimum distance to segment boundary
        pytest.raises((TypeError, AssertionError), ExtracellularArray,
                      good_positions, min_distance=mind)

    pytest.raises(ValueError, ExtracellularArray,  # more chans than voltages
                  good_positions, times=[1], voltages=[[[42]]])
    pytest.raises(ValueError, ExtracellularArray,  # less times than voltages
                  good_positions, times=[1], voltages=[[[42, 42], [84, 84]]])

    rec_arr = ExtracellularArray(good_positions,
                                 times=[0, 0.1, 0.21, 0.3],  # uneven sampling
                                 voltages=[[[0, 0, 0, 0], [0, 0, 0, 0]]])
    with pytest.raises(RuntimeError, match="Extracellular sampling times"):
        _ = rec_arr.sfreq
    rec_arr._reset()
    assert len(rec_arr.times) == len(rec_arr.voltages) == 0
    assert rec_arr.sfreq is None
    rec_arr = ExtracellularArray(good_positions,
                                 times=[0], voltages=[[[0], [0]]])
    with pytest.raises(RuntimeError, match="Sampling rate is not defined"):
        _ = rec_arr.sfreq


def test_transmembrane_currents():
    """Test that net transmembrane current is zero at all times."""
    params.update({'N_pyr_x': 3,
                   'N_pyr_y': 3,
                   'tstop': 40,
                   't_evprox_1': 5,
                   't_evdist_1': 10,
                   't_evprox_2': 20,
                   'N_trials': 1})
    net = default_network(params, add_drives_from_params=True)
    electrode_pos = (0, 0, 0)  # irrelevant where electrode is
    # all transfer resistances set to unity
    net.add_electrode_array('net_Im', electrode_pos, method=None)
    _ = simulate_dipole(net, postproc=False)
    currents = net.rec_arrays['net_Im'].get_data()
    assert_allclose(currents, 0, rtol=1e-10, atol=1e-10)


def test_transfer_resistance():
    """Test transfer resistances calculated correctly"""
    from neuron import h
    from hnn_core.extracellular import _transfer_resistance
    sec = h.Section(name='dend')
    h.pt3dclear(sec=sec)
    h.pt3dadd(0, 0, 0, 1, sec=sec)
    h.pt3dadd(0, 1, 0, 1, sec=sec)  # section oriented along y-axis
    sec.L = 300
    sec.diam = 8
    sec.nseg = 5
    # NB segment lengths aren't equal! First/last segment center point is
    # closer to respective end point than to next/previous segment!
    seg_ctr_pts = [0]
    seg_ctr_pts.extend([seg.x * sec.L for seg in sec])
    seg_ctr_pts.append(sec.L)
    seg_lens = np.diff(seg_ctr_pts)
    first_len = seg_lens[0]
    seg_lens = np.array([first_len] + list(seg_lens[2:]))
    seg_ctr_pts = seg_ctr_pts[1:-1]  # remove end points again

    conductivity = 0.3

    elec_pos = (10, 150, 0)
    target_vals = {'psa': list(), 'lsa': list()}
    for seg_idx in range(sec.nseg):
        # PSA: distance to middle segment == electrode x-position
        var_r_psa = np.sqrt(elec_pos[0] ** 2 +
                            (elec_pos[1] - seg_ctr_pts[seg_idx]) ** 2)
        target_vals['psa'].append(
            1000 / (4. * np.pi * conductivity * var_r_psa))

        # LSA: calculate L and H variables relative to segment endpoints
        var_l = elec_pos[1] - (seg_ctr_pts[seg_idx] - seg_lens[seg_idx])
        var_h = elec_pos[1] - (seg_ctr_pts[seg_idx] + seg_lens[seg_idx])
        var_r_lsa = elec_pos[0]  # just use the axial distance
        target_vals['lsa'].append(
            1000 * np.log(np.abs(
                (np.sqrt(var_h ** 2 + var_r_lsa ** 2) - var_h) /
                (np.sqrt(var_l ** 2 + var_r_lsa ** 2) - var_l)
            )) / (4. * np.pi * conductivity * 2 * seg_lens[seg_idx]))

    for method in ['psa', 'lsa']:
        res = _transfer_resistance(sec, elec_pos, conductivity, method)
        assert_allclose(res, target_vals[method], rtol=1e-12, atol=0.)


@requires_mpi4py
@requires_psutil
def test_extracellular_backends(run_hnn_core_fixture):
    """Test extracellular outputs across backends."""

    electrode_array = {'arr1': [(2, 2, 400), (6, 6, 800)]}
    _, joblib_net = run_hnn_core_fixture(
        backend='joblib', n_jobs=1, reduced=True, record_isoma=True,
        record_vsoma=True, electrode_array=electrode_array)
    _, mpi_net = run_hnn_core_fixture(
        backend='mpi', n_procs=2, reduced=True, record_isoma=True,
        record_vsoma=True, electrode_array=electrode_array)

    assert (len(electrode_array['arr1']) ==
            len(joblib_net.rec_arrays['arr1'].positions) ==
            len(mpi_net.rec_arrays['arr1'].positions))
    assert (len(joblib_net.rec_arrays['arr1']) ==
            len(mpi_net.rec_arrays['arr1']) ==
            2)  # length == n.o. trials

    # reduced simulation has n_trials=2
    # trial_idx, n_trials = 0, 2
    for tr_idx, el_idx in zip([0, 1], [0, 1]):
        assert_allclose(joblib_net.rec_arrays['arr1']._data[tr_idx][el_idx],
                        mpi_net.rec_arrays['arr1']._data[tr_idx][el_idx])

    assert isinstance(joblib_net.rec_arrays['arr1'].get_data(), np.ndarray)
    assert_array_equal(joblib_net.rec_arrays['arr1'].get_data().shape,
                       [len(joblib_net.rec_arrays['arr1']._data),
                        len(joblib_net.rec_arrays['arr1']._data[0]),
                        len(joblib_net.rec_arrays['arr1']._data[0][0])])

    # make sure sampling rate is fixed (raises RuntimeError if not)
    _ = joblib_net.rec_arrays['arr1'].sfreq
    # check plotting works
    joblib_net.rec_arrays['arr1'].plot(show=False)


def _mathematical_dipole(e_pos, d_pos, d_Q):
    rr = e_pos - d_pos
    R = norm(rr)
    Q = norm(d_Q)
    cosT = np.dot(rr, d_Q) / (R * Q)
    return (Q * cosT) / (4 * np.pi * R ** 2)


# require MPI to speed up due to large number of extracellular electrodes
@requires_mpi4py
def test_dipolar_far_field():
    """Test that LFP in the far field is dipolar when expected."""
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

    conductivity = 0.3

    # create far-field grid of LFP electrodes; note that cells are assumed
    # to lie in the XZ-plane
    xmin, xmax = -5e4, 5e4
    zmin, zmax = -5e4, 5e4
    step = 5e3
    posy = 1e2  # out-of-plane
    electrode_pos = list()
    for posx in np.arange(xmin, xmax, step):
        for posz in np.arange(zmin, zmax, step):
            electrode_pos.append((posx, posy, posz))
    net.add_electrode_array('grid_psa', electrode_pos,
                            conductivity=conductivity, method='psa')
    net.add_electrode_array('grid_lsa', electrode_pos,
                            conductivity=conductivity, method='lsa')

    with MPIBackend(n_procs=2):
        dpl = simulate_dipole(net, postproc=False)

    X_p = np.arange(xmin, xmax, step) / 1000
    Z_p = np.arange(zmin, zmax, step) / 1000
    Y_p = posy / 1000
    idt = np.argmin(np.abs(dpl[0].times - 15.))
    phi_p_psa = np.zeros((len(X_p), len(Z_p)))
    phi_p_lsa = np.zeros((len(X_p), len(Z_p)))
    phi_p_theory = np.zeros((len(X_p), len(Z_p)))

    # location of equivalent current dipole for this stimulation (manual)
    d_pos = np.array((0, 0, 800)) / 1000  # um -> mm
    # dipole orientation is along the apical dendrite, towards the soma
    # the amplitude is really irrelevant, only shape is compared
    d_Q = 5e2 * np.array((0, 0, -1))

    for ii, row in enumerate(X_p):
        for jj, col in enumerate(Z_p):

            e_pos = np.array((row, Y_p, col))

            # ignore 10 mm radius closest to dipole
            if norm(e_pos - d_pos) < 10:
                phi_p_psa[ii][jj] = 0
                phi_p_lsa[ii][jj] = 0
                phi_p_theory[ii][jj] = 0
                continue

            phi_p_psa[ii][jj] = net.rec_arrays['grid_psa']._data[0][
                ii * len(X_p) + jj][idt] * 1e3
            phi_p_lsa[ii][jj] = net.rec_arrays['grid_lsa']._data[0][
                ii * len(X_p) + jj][idt] * 1e3
            phi_p_theory[ii][jj] = \
                _mathematical_dipole(e_pos, d_pos, d_Q) / conductivity

    # compare the shape of the far fields
    for phi_p in [phi_p_psa, phi_p_lsa]:
        cosT = np.dot(phi_p.ravel(), phi_p_theory.ravel()) / (
            norm(phi_p.ravel()) * norm(phi_p_theory.ravel()))
        # the far field should be very close to dipolar, though threshold may
        # need adjusting when new mechanisms are included in the cells
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


def test_rec_array_calculation():
    """Test LFP calculation."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    params.update({'N_pyr_x': 3,
                   'N_pyr_y': 3,
                   'tstop': 25,
                   't_evprox_1': 7,
                   't_evdist_1': 17})
    net = default_network(params, add_drives_from_params=True)

    # one electrode inside, one above the active elements of the network
    electrode_pos = [(1.5, 1.5, 1000), (1.5, 1.5, 3000)]
    net.add_electrode_array('arr1', electrode_pos)
    _ = simulate_dipole(net, n_trials=1, postproc=False)

    # test accessing simulated voltages
    assert (len(net.rec_arrays['arr1']) ==
            len(net.rec_arrays['arr1'].voltages) == 1)  # n_trials
    assert len(net.rec_arrays['arr1'].voltages[0]) == 2  # n_contacts
    assert (len(net.rec_arrays['arr1'].voltages[0][0]) ==
            len(net.rec_arrays['arr1'].times))
    # ensure copy drops data (but retains electrode position information etc.)
    net_copy = net.copy()
    assert isinstance(net_copy.rec_arrays['arr1'], ExtracellularArray)
    assert len(net_copy.rec_arrays['arr1'].voltages) == 0

    data, times = net.rec_arrays['arr1'].get_data(return_times=True)
    assert isinstance(data, np.ndarray)
    assert isinstance(times, np.ndarray)
    data_only = net.rec_arrays['arr1'].get_data()
    assert_allclose(data, data_only)

    # using the same electrode positions, but a different method: LSA
    net.add_electrode_array('arr2', electrode_pos, method='lsa')

    # make sure no sinister segfaults are triggered when running mult. trials
    n_trials = 5  # NB 5 trials!
    _ = simulate_dipole(net, n_trials=n_trials, postproc=False)

    # simulate_dipole is run twice above, first 1 then 5 trials.
    # Make sure that previous results are discarded on each run
    assert len(net.rec_arrays['arr1']._data) == n_trials

    for trial_idx in range(n_trials):
        # LSA and PSA should agree far away (second electrode)
        assert_allclose(net.rec_arrays['arr1']._data[trial_idx][1],
                        net.rec_arrays['arr2']._data[trial_idx][1],
                        rtol=1e-3, atol=1e-3)
