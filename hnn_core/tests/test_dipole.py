import matplotlib
import os.path as op

import numpy as np
#from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

import hnn_core
from hnn_core import read_params, read_dipole
from hnn_core.dipole import Dipole

matplotlib.use('agg')


def test_dipole():
    """Test dipole object."""
    hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    dpl_out_fname = '/tmp/dpl1.txt'
    params = read_params(params_fname)
    times = np.random.random(6000)
    data = np.random.random((6000, 3))
    dipole = Dipole(times, data)
    dipole.baseline_renormalize(params)
    dipole.convert_fAm_to_nAm()
    dipole.scale(params['dipole_scalefctr'])
    dipole.smooth(params['dipole_smooth_win'] / params['dt'])
    dipole.plot(layer='agg')
    dipole.write(dpl_out_fname)
    dipole_read = read_dipole(dpl_out_fname)
    assert_allclose(dipole_read.t, dipole.t, rtol=0, atol=0.00051)
    for dpl_key in dipole.dpl.keys():
        assert_allclose(dipole_read.dpl[dpl_key],
                        dipole.dpl[dpl_key], rtol=0, atol=0.000051)
