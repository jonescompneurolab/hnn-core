import os.path as op

import matplotlib
import numpy as np
from numpy.testing import assert_allclose
import pytest

import hnn_core
from hnn_core import read_params, read_dipole, average_dipoles
from hnn_core import Network, default_network
from hnn_core.viz import plot_dipole
from hnn_core.dipole import Dipole, simulate_dipole
from hnn_core.parallel_backends import requires_mpi4py, requires_psutil

matplotlib.use('agg')


def test_dipole(tmpdir, run_hnn_core_fixture):
    """Test dipole object."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    dpl_out_fname = tmpdir.join('dpl1.txt')
    params = read_params(params_fname)
    times = np.arange(0, 6000 * params['dt'], params['dt'])
    data = np.random.random((6000, 3))
    dipole = Dipole(times, data)
    dipole._baseline_renormalize(params['N_pyr_x'], params['N_pyr_y'])
    dipole._convert_fAm_to_nAm()
    dipole.scale(params['dipole_scalefctr'])

    # test smoothing logic
    with pytest.raises(ValueError,
                       match='Window length must be a non-negative number'):
        dipole.smooth(window_len=None)
    with pytest.raises(ValueError,
                       match='Window length must be a non-negative number'):
        dipole.smooth(window_len=-1)

    with pytest.raises(ValueError, match='Window length too long:'):
        dipole.smooth(window_len=1e6)
    with pytest.raises(ValueError, match='Window length less than 1 ms'):
        dipole.smooth(window_len=1e-1)
    dipole.smooth(window_len=params['dipole_smooth_win'])

    dipole.plot(show=False)
    plot_dipole([dipole, dipole], show=False)
    dipole.write(dpl_out_fname)
    dipole_read = read_dipole(dpl_out_fname)
    assert_allclose(dipole_read.times, dipole.times, rtol=0, atol=0.00051)
    for dpl_key in dipole.data.keys():
        assert_allclose(dipole_read.data[dpl_key],
                        dipole.data[dpl_key], rtol=0, atol=0.000051)

    # average two identical dipole objects
    dipole_avg = average_dipoles([dipole, dipole_read])
    for dpl_key in dipole_avg.data.keys():
        assert_allclose(dipole_read.data[dpl_key],
                        dipole_avg.data[dpl_key], rtol=0, atol=0.000051)

    with pytest.raises(ValueError, match="Dipole at index 0 was already an "
                       "average of 2 trials"):
        dipole_avg = average_dipoles([dipole_avg, dipole_read])

    # test postproc
    dpls_raw, net = run_hnn_core_fixture(backend='joblib', n_jobs=1,
                                         reduced=True, record_isoma=True,
                                         record_vsoma=True, postproc=False)
    dpls, _ = run_hnn_core_fixture(backend='joblib', n_jobs=1, reduced=True,
                                   record_isoma=True, record_vsoma=True,
                                   postproc=True)
    with pytest.raises(AssertionError):
        assert_allclose(dpls[0].data['agg'], dpls_raw[0].data['agg'])

    dpls_raw[0]._post_proc(net._params['dipole_smooth_win'],
                           net._params['dipole_scalefctr'])
    assert_allclose(dpls_raw[0].data['agg'], dpls[0].data['agg'])


def test_dipole_simulation():
    """Test data produced from simulate_dipole() call."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    params.update({'N_pyr_x': 3,
                   'N_pyr_y': 3,
                   'tstop': 25,
                   'dipole_smooth_win': 5,
                   't_evprox_1': 5,
                   't_evdist_1': 10,
                   't_evprox_2': 20})
    net = default_network(params, add_drives_from_params=True)
    with pytest.raises(ValueError, match="Invalid number of simulations: 0"):
        simulate_dipole(net, n_trials=0)
    with pytest.raises(TypeError, match="record_vsoma must be bool, got int"):
        simulate_dipole(net, n_trials=1, record_vsoma=0)
    with pytest.raises(TypeError, match="record_isoma must be bool, got int"):
        simulate_dipole(net, n_trials=1, record_vsoma=False, record_isoma=0)

    # test Network.copy() returns 'bare' network after simulating
    dpl = simulate_dipole(net, n_trials=1)[0]
    net_copy = net.copy()
    assert len(net_copy.external_drives['evprox1']['events']) == 0
    assert len(net_copy.cell_response.vsoma) == 0

    # test that Dipole.copy() returns the expected exact copy
    assert_allclose(dpl.data['agg'], dpl.copy().data['agg'])

    # Test raster plot with no spikes
    params['tstop'] = 0.1
    net = Network(params)
    simulate_dipole(net, n_trials=1, postproc=False)
    net.cell_response.plot_spikes_raster()


@requires_mpi4py
@requires_psutil
def test_cell_response_backends(run_hnn_core_fixture):
    """Test cell_response outputs across backends."""

    # reduced simulation has n_trials=2
    trial_idx, n_trials, gid = 0, 2, 7
    _, joblib_net = run_hnn_core_fixture(backend='joblib', n_jobs=1,
                                         reduced=True, record_isoma=True,
                                         record_vsoma=True)
    _, mpi_net = run_hnn_core_fixture(backend='mpi', n_procs=2, reduced=True,
                                      record_isoma=True, record_vsoma=True)
    n_times = len(joblib_net.cell_response.times)

    assert len(joblib_net.cell_response.vsoma) == n_trials
    assert len(joblib_net.cell_response.isoma) == n_trials
    assert len(joblib_net.cell_response.vsoma[trial_idx][gid]) == n_times
    assert len(joblib_net.cell_response.isoma[
               trial_idx][gid]['soma_gabaa']) == n_times

    assert len(mpi_net.cell_response.vsoma) == n_trials
    assert len(mpi_net.cell_response.isoma) == n_trials
    assert len(mpi_net.cell_response.vsoma[trial_idx][gid]) == n_times
    assert len(mpi_net.cell_response.isoma[
               trial_idx][gid]['soma_gabaa']) == n_times
    assert mpi_net.cell_response.vsoma == joblib_net.cell_response.vsoma
    assert mpi_net.cell_response.isoma == joblib_net.cell_response.isoma

    # Test if spike time falls within depolarization window above v_thresh
    v_thresh = 0.0
    times = np.array(joblib_net.cell_response.times)
    spike_times = np.array(joblib_net.cell_response.spike_times[trial_idx])
    spike_gids = np.array(joblib_net.cell_response.spike_gids[trial_idx])
    vsoma = np.array(joblib_net.cell_response.vsoma[trial_idx][gid])

    v_mask = vsoma > v_thresh
    assert np.all([spike_times[spike_gids == gid] > times[v_mask][0],
                   spike_times[spike_gids == gid] < times[v_mask][-1]])

    # test that event times before and after simulation are the same
    for drive_name, drive in joblib_net.external_drives.items():
        gid_ran = joblib_net.gid_ranges[drive_name]
        for idx_drive, event_times in enumerate(drive['events'][trial_idx]):
            net_ets = [spike_times[i] for i, g in enumerate(spike_gids) if
                       g == gid_ran[idx_drive]]
            assert_allclose(np.array(event_times), np.array(net_ets))
