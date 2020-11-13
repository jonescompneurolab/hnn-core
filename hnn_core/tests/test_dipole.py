import os.path as op

import matplotlib
import numpy as np
from numpy.testing import assert_allclose
import pytest

import hnn_core
from hnn_core import read_params, read_dipole, average_dipoles, viz, Network
from hnn_core.dipole import Dipole, simulate_dipole

matplotlib.use('agg')


def test_dipole(tmpdir):
    """Test dipole object."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    dpl_out_fname = tmpdir.join('dpl1.txt')
    params = read_params(params_fname)
    times = np.random.random(6000)
    data = np.random.random((6000, 3))
    dipole = Dipole(times, data)
    dipole.baseline_renormalize(params)
    dipole.convert_fAm_to_nAm()
    dipole.scale(params['dipole_scalefctr'])
    dipole.smooth(params['dipole_smooth_win'] / params['dt'])
    dipole.plot(show=False)
    viz.plot_dipole([dipole, dipole], show=False)
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


def test_dipole_simulation():
    """Test data produced from simulate_dipole() call."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    params.update({'N_pyr_x': 3,
                   'N_pyr_y': 3,
                   'tstop': 25,
                   't_evprox_1': 5,
                   't_evdist_1': 10,
                   't_evprox_2': 20})
    net = Network(params)
    with pytest.raises(ValueError, match="Invalid number of simulations: 0"):
        simulate_dipole(net, n_trials=0)
    with pytest.raises(TypeError, match="record_vsoma must be bool, got int"):
        simulate_dipole(net, n_trials=1, record_vsoma=0)

    simulate_dipole(net, n_trials=2, record_vsoma=True)
    assert len(net.spikes.vsoma) == 2
    assert len(net.spikes.vsoma[0]) == 24
