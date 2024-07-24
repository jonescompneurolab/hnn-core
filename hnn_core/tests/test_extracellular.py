# Authors: Nick Tolley <nicholas_tolley@brown.edu>
#          Christopher Bailey <cjb@cfin.au.dk>

from copy import deepcopy
import os.path as op
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import hnn_core
from hnn_core import read_params, jones_2009_model, simulate_dipole
from hnn_core.extracellular import (ExtracellularArray, calculate_csd2d,
                                    _get_laminar_z_coords)
from hnn_core.parallel_backends import requires_mpi4py, requires_psutil

import matplotlib.pyplot as plt


hnn_core_root = op.dirname(hnn_core.__file__)
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)


def test_extracellular_api():
    """Test extracellular recording API."""
    net = jones_2009_model(deepcopy(params), add_drives_from_params=True)

    # Test LFP electrodes
    electrode_pos = (1, 2, 3)
    net.add_electrode_array('el1', electrode_pos)
    electrode_pos = [(1, 2, 3), (-1, -2, -3)]
    net.add_electrode_array('arr1', electrode_pos)
    assert len(net.rec_arrays) == 2
    assert len(net.rec_arrays['arr1'].positions) == 2

    # Test other not NotImplemented for ExtracellularArray Class
    assert (net.rec_arrays['arr1'] == "extArr") is False

    # ensure unique names
    pytest.raises(ValueError, net.add_electrode_array, 'arr1', [(6, 6, 800)])
    # all remaining input arguments checked by ExtracellularArray

    rec_arr = ExtracellularArray(electrode_pos)

    # Added second string in the match pattern due to changes in python >=3.11
    # AttributeError message changed to "property X of object Y has no setter"
    with pytest.raises(AttributeError,
                       match="has no setter|can't set attribute"):
        rec_arr.times = [1, 2, 3]
    with pytest.raises(AttributeError,
                       match="has no setter|can't set attribute"):
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

    # test colinearity and equal spacing between electrode contacts for laminar
    # profiling (e.g., for platting laminar LFP or CSD)
    electrode_pos = [(1, 2, 1000), (2, 3, 3000), (3, 4, 5000),
                     (4, 5, 7000)]
    z_coords, z_delta = _get_laminar_z_coords(electrode_pos)
    assert np.array_equal(z_coords, [1000, 3000, 5000, 7000])
    assert z_delta == 2000
    with pytest.raises(ValueError, match='Electrode array positions must '
                       'contain more than 1 contact'):
        _, _ = _get_laminar_z_coords([(1, 2, 3)])
    with pytest.raises(ValueError, match='Make sure the electrode positions '
                       'are equispaced, colinear'):
        _, _ = _get_laminar_z_coords([(1, 1, 3), (1, 1, 4), (1, 1, 3.5)])


def test_transmembrane_currents():
    """Test that net transmembrane current is zero at all times."""
    params.update({'N_pyr_x': 3,
                   'N_pyr_y': 3,
                   't_evprox_1': 5,
                   't_evdist_1': 10,
                   't_evprox_2': 20,
                   'N_trials': 1})
    net = jones_2009_model(params, add_drives_from_params=True)
    electrode_pos = (0, 0, 0)  # irrelevant where electrode is
    # all transfer resistances set to unity
    net.add_electrode_array('net_Im', electrode_pos, method=None)
    _ = simulate_dipole(net, tstop=40.)
    assert_allclose(net.rec_arrays['net_Im'].voltages, 0,
                    rtol=1e-10, atol=1e-10)


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
    # calculation of CSD requires >=4 electrode contacts
    electrode_array = {'arr1': [(2, 2, 400), (2, 2, 600), (2, 2, 800),
                                (2, 2, 1000)]}
    _, joblib_net = run_hnn_core_fixture(
        backend='joblib', n_jobs=1, reduced=True, record_isec='soma',
        record_vsec='soma', record_ca='soma', electrode_array=electrode_array)
    _, mpi_net = run_hnn_core_fixture(
        backend='mpi', n_procs=2, reduced=True, record_isec='soma',
        record_vsec='soma', record_ca='soma', electrode_array=electrode_array)

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

    assert isinstance(joblib_net.rec_arrays['arr1'].voltages, np.ndarray)
    assert_array_equal(joblib_net.rec_arrays['arr1'].voltages.shape,
                       [len(joblib_net.rec_arrays['arr1']._data),
                        len(joblib_net.rec_arrays['arr1']._data[0]),
                        len(joblib_net.rec_arrays['arr1']._data[0][0])])

    # make sure sampling rate is fixed (raises RuntimeError if not)
    _ = joblib_net.rec_arrays['arr1'].sfreq
    # check plotting works
    joblib_net.rec_arrays['arr1'].plot_lfp(show=False)
    joblib_net.rec_arrays['arr1'].plot_csd(show=False)

    plt.close('all')


def test_rec_array_calculation():
    """Test LFP/CSD calculation."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    params.update({'t_evprox_1': 7,
                   't_evdist_1': 17})
    net = jones_2009_model(params, mesh_shape=(3, 3),
                           add_drives_from_params=True)

    # one electrode inside, one above the active elements of the network,
    # and two more to allow calculation of CSD (2nd spatial derivative)
    electrode_pos = [(1, 2, 1000), (2, 3, 3000), (3, 4, 5000),
                     (4, 5, 7000)]
    net.add_electrode_array('arr1', electrode_pos)
    _ = simulate_dipole(net, tstop=5, n_trials=1)

    # test accessing simulated voltages
    assert (len(net.rec_arrays['arr1']) ==
            len(net.rec_arrays['arr1'].voltages) == 1)  # n_trials
    assert len(net.rec_arrays['arr1'].voltages[0]) == 4  # n_contacts
    assert (len(net.rec_arrays['arr1'].voltages[0][0]) ==
            len(net.rec_arrays['arr1'].times))

    # test dimensionality of LFP and CSD matrices
    lfp_data = net.rec_arrays['arr1'].voltages[0]
    csd_data = calculate_csd2d(lfp_data)
    assert lfp_data.shape == csd_data.shape

    # ensure copy drops data (but retains electrode position information etc.)
    net_copy = net.copy()
    assert isinstance(net_copy.rec_arrays['arr1'], ExtracellularArray)
    assert len(net_copy.rec_arrays['arr1'].voltages) == 0

    assert isinstance(net.rec_arrays['arr1'].voltages, np.ndarray)
    assert isinstance(net.rec_arrays['arr1'].times, np.ndarray)

    # using the same electrode positions, but a different method: LSA
    net.add_electrode_array('arr2', electrode_pos, method='lsa')

    # make sure no sinister segfaults are triggered when running mult. trials
    n_trials = 5  # NB 5 trials!
    _ = simulate_dipole(net, tstop=5, n_trials=n_trials)

    # simulate_dipole is run twice above, first 1 then 5 trials.
    # Make sure that previous results are discarded on each run
    assert len(net.rec_arrays['arr1']._data) == n_trials

    for trial_idx in range(n_trials):
        # LSA and PSA should agree far away (second electrode)
        assert_allclose(net.rec_arrays['arr1']._data[trial_idx][1],
                        net.rec_arrays['arr2']._data[trial_idx][1],
                        rtol=1e-3, atol=1e-3)


def test_extracellular_viz():
    """Test if deprecation warning is raised in plot_laminar_lfp."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    params.update({'t_evprox_1': 7, 't_evdist_1': 17})
    net = jones_2009_model(params, mesh_shape=(3, 3),
                           add_drives_from_params=True)

    # one electrode inside, one above the active elements of the network,
    # and two more to allow calculation of CSD (2nd spatial derivative)
    electrode_pos = [(1, 2, 1000), (2, 3, 3000), (3, 4, 5000), (4, 5, 7000)]
    net.add_electrode_array('arr1', electrode_pos)
    _ = simulate_dipole(net, tstop=5, n_trials=1)

    with pytest.deprecated_call():
        net.rec_arrays['arr1'].plot_lfp(show=False, tmin=10, tmax=100)
    with pytest.raises(RuntimeError, match='Please use sink = "b" or '
                       'sink = "r". Only colormap "jet" is supported '
                       'for CSD.'):
        net.rec_arrays['arr1'].plot_csd(show=False, sink='g')
