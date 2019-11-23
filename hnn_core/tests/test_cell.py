import matplotlib
import os.path as op

import hnn_core
from hnn_core import read_params, Network

matplotlib.use('agg')


def test_dipole():
    """Test params object."""
    hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)

    with Network(params) as net:
        net.build()
        net.cells[0].plot_voltage()
