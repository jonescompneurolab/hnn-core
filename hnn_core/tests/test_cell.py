import pytest
import pickle

import matplotlib

from hnn_core.network_builder import load_custom_mechanisms
from hnn_core.cell import _ArtificialCell, Cell

matplotlib.use('agg')


def test_cell():
    """Test cells object."""
    load_custom_mechanisms()

    name = 'test'
    pos = (0., 0., 0.)
    p_secs = {'blah': 1}
    p_syn = {'ampa': dict(e=0, tau1=0.5, tau2=5.)}
    topology = None
    sect_loc = {'proximal': 'soma'}
    # GID is assigned exactly once for each cell, either at initialisation...
    cell = Cell(name, pos, p_secs, p_syn, topology, sect_loc, gid=42)
    assert cell.gid == 42
    with pytest.raises(RuntimeError,
                       match='Global ID for this cell already assigned!'):
        cell.gid += 1
    # ... or later
    # cells can exist fine without gid
    cell = Cell(name, pos, p_secs, p_syn, topology, sect_loc)
    assert cell.gid is None  # check that it's initialised to None
    with pytest.raises(ValueError,
                       match='gid must be an integer'):
        cell.gid = [1]
    cell.gid = 42
    assert cell.gid == 42
    with pytest.raises(ValueError,
                       match='gid must be an integer'):
        # test init checks gid
        cell = Cell(name, pos, p_secs, p_syn, topology, sect_loc,
                    gid='one')

    # test that ExpSyn always takes nrn.Segment, not float
    with pytest.raises(TypeError, match='secloc must be instance of'):
        cell.syn_create(0.5, e=0., tau1=0.5, tau2=5.)

    pickle.dumps(cell)  # check cell object is picklable until built
    with pytest.raises(KeyError, match='soma must be defined'):
        cell.build()

    p_secs = {
        'soma':
        {
            'L': 39,
            'diam': 20,
            'cm': 0.85,
            'Ra': 200.,
            'sec_pts': [[0, 0, 0], [0, 39., 0]],
            'syns': ['ampa'],
            'mechs': {
                'km': {
                    'gbar_km': 60
                },
                'ca': {
                    'gbar_ca': lambda x: 3e-3 * x
                }
            }
        }
    }
    cell = Cell(name, pos, p_secs, p_syn, topology, sect_loc)
    # test successful build
    cell.build()
    assert 'soma' in cell.sections
    assert cell.sections['soma'].L == p_secs['soma']['L']
    assert cell.sections['soma'].gbar_km == p_secs[
        'soma']['mechs']['km']['gbar_km']
    # test building cell with a dipole oriented to a nonexitent section
    with pytest.raises(ValueError, match='sec_name_apical must be an'):
        cell.build(sec_name_apical='blah')


def test_artificial_cell():
    """Test artificial cell object."""
    load_custom_mechanisms()
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

    # GID is assigned exactly once for each cell, either at initialisation...
    cell = _ArtificialCell(event_times, threshold, gid=42)
    assert cell.gid == 42
    with pytest.raises(RuntimeError,
                       match='Global ID for this cell already assigned!'):
        cell.gid += 1
    with pytest.raises(ValueError,
                       match='gid must be an integer'):
        cell.gid = [1]
    # ... or later
    cell = _ArtificialCell(event_times, threshold)  # fine without gid
    assert cell.gid is None  # check that it's initialised to None
    cell.gid = 42
    assert cell.gid == 42
    with pytest.raises(ValueError,  # test init checks gid
                       match='gid must be an integer'):
        cell = _ArtificialCell(event_times, threshold, gid='one')
