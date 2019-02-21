import matplotlib
import os.path as op

import numpy as np

import mne_neuron
from mne_neuron import Params
from mne_neuron.dipole import Dipole

matplotlib.use('agg')


def test_dipole():
    """Test params object."""
    mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')
    params_fname = op.join(mne_neuron_root, 'param', 'default.json')
    params = Params(params_fname)

    times = np.random.random(6000)
    data = np.random.random((6000, 3))
    dipole = Dipole(times, data)
    dipole.baseline_renormalize(params)
    dipole.convert_fAm_to_nAm()
    dipole.scale(params['dipole_scalefctr'])
    dipole.smooth(params['dipole_smooth_win'] / params['dt'])
    dipole.plot(layer='agg')
