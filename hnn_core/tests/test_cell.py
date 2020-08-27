import pytest

import matplotlib
import os.path as op

import hnn_core
from hnn_core import read_params, Network
from hnn_core.network_builder import NetworkBuilder
from hnn_core.cell import _ArtificialCell, _Cell
from hnn_core.pyramidal import L5Pyr
from hnn_core.params_default import get_L5Pyr_params_default

matplotlib.use('agg')


def test_cell():
    """Test cells object."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)

    net = Network(params)
    with NetworkBuilder(net) as neuron_net:
        neuron_net.cells[0].plot_voltage()

    # test that ExpSyn always takes nrn.Segment, not float
    soma_props = {"L": 22.1, "diam": 23.4, "cm": 0.6195, "Ra": 200.0,
                  "pos": (0., 0., 0.), 'name': 'test_cell'}
    cell = _Cell(gid=1, soma_props=soma_props)
    with pytest.raises(TypeError, match='secloc must be instance of'):
        cell.syn_create(0.5, e=0., tau1=0.5, tau2=5.)

    # test connecting feed at a location
    cell_params = get_L5Pyr_params_default()
    cell = L5Pyr(gid=0, pos=(0., 0., 0.), p=cell_params)
    nc_dict = {
        'pos_src': (1., 1., 1.),
        'A_weight': 0.01,
        'A_delay': 0.01,
        'lamtha': 3,
        'threshold': 0.5,
        'type_src': 'extgauss'
    }
    gid_src = 999  # doesn't matter as gid_connect makes virtual connection
    n_connections = dict(proximal=3, distal=1)
    for feed_type in n_connections:
        nc_list = []
        cell._connect_feed_at_loc(feed_type, receptor='ampa', gid_src=gid_src,
                                  nc_dict=nc_dict, nc_list=nc_list)
        assert len(nc_list) == n_connections[feed_type]
        # XXX: poor man's type check since couldn't do isinstance(nc, h.NetCon)
        assert all(isinstance(nc, type(nc_list[0])) for nc in nc_list)


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
