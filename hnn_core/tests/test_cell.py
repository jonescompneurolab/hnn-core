import matplotlib
import os.path as op

import hnn_core
from hnn_core import read_params, Network
from hnn_core.neuron import _neuron_network

matplotlib.use('agg')


def test_cell():
    """Test cells object."""
    hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)

    net = Network(params)
    with _neuron_network(net) as neuron_network:
        neuron_network.cells[0].plot_voltage()
