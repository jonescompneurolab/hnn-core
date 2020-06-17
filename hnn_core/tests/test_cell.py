import matplotlib
import os.path as op

import hnn_core
from hnn_core import read_params, Network
from hnn_core.neuron import NeuronNetwork
from hnn_core.cell import _ArtificialCell

matplotlib.use('agg')


def test_cell():
    """Test cells object."""
    hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)

    net = Network(params)
    with NeuronNetwork(net) as neuron_net:
        neuron_net.cells[0].plot_voltage()


def test_artificial_cell():
    """Test artificial cell object."""
    event_times = [1, 2, 3]
    threshold = 0.0
    artificial_cell = _ArtificialCell(event_times, threshold)
    assert artificial_cell.nrn_eventvec.to_python() == event_times
    # the h.VecStim() object defined in vecevent.mod should contain a 'play()'
    # method
    assert hasattr(artificial_cell.nrn_vecstim, 'play')
    # the h.Netcon() instance should reference the h.VecStim() instance
    assert artificial_cell.nrn_netcon.pre() == artificial_cell.nrn_vecstim
    assert artificial_cell.nrn_netcon.threshold == threshold
