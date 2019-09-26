import matplotlib
import os.path as op

import numpy as np

import hnn_core
from hnn_core import read_params
from hnn_core.dipole import Dipole

matplotlib.use('agg')


def test_dipole():
    """Test params object."""
    hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)

    times = np.random.random(6000)
    data = np.random.random((6000, 3))
    dipole = Dipole(times, data)
    dipole.baseline_renormalize(params)
    dipole.convert_fAm_to_nAm()
    dipole.scale(params['dipole_scalefctr'])
    dipole.smooth(params['dipole_smooth_win'] / params['dt'])
    dipole.plot(layer='agg')
    dipole.write('/tmp/dpl1.txt')
