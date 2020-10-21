import pytest

import matplotlib
import os.path as op

import hnn_core
from hnn_core import read_params, Network
from hnn_core.network_builder import NetworkBuilder
from hnn_core.cell import _ArtificialCell, _Cell

matplotlib.use('agg')


def test_cell():
    """Test cells object."""
    # test that ExpSyn always takes nrn.Segment, not float
    soma_props = {"L": 22.1, "diam": 23.4, "cm": 0.6195, "Ra": 200.0,
                  "pos": (0., 0., 0.), 'name': 'test_cell'}

    with pytest.raises(TypeError, match='with abstract methods get_sections'):
        cell = _Cell(gid=1, soma_props=soma_props)

    class Cell(_Cell):
        def get_sections(self):
            return [self.soma]

    cell = Cell(gid=1, soma_props=soma_props)
    with pytest.raises(TypeError, match='secloc must be instance of'):
        cell.syn_create(0.5, e=0., tau1=0.5, tau2=5.)


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
