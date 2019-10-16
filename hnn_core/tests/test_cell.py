import matplotlib
import os.path as op

import hnn_core
from hnn_core import Params, Network

matplotlib.use('agg')


def test_dipole():
    """Test params object."""
    hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = Params(params_fname)

    net = Network(params)
    net.build()

    net.cells[0].plot_voltage()
