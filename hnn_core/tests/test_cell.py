import pytest
import pickle
import numpy as np

import matplotlib

from hnn_core.network_builder import load_custom_mechanisms
from hnn_core.cell import _ArtificialCell, Cell, Section

matplotlib.use('agg')


def test_cell():
    """Test cells object."""
    load_custom_mechanisms()

    name = 'test'
    pos = (0., 0., 0.)
    sections = {'soma': Section(L=1, diam=5, Ra=3, cm=100,
                                end_pts=[[0, 0, 0], [0, 39., 0]])}
    synapses = {'ampa': dict(e=0, tau1=0.5, tau2=5.)}
    topology = None
    sect_loc = {'proximal': 'soma'}
    # GID is assigned exactly once for each cell, either at initialisation...
    cell = Cell(name, pos, sections, synapses, topology, sect_loc, gid=42)
    assert cell.gid == 42
    with pytest.raises(RuntimeError,
                       match='Global ID for this cell already assigned!'):
        cell.gid += 1
    # ... or later
    # cells can exist fine without gid
    cell = Cell(name, pos, sections, synapses, topology, sect_loc)
    assert cell.gid is None  # check that it's initialised to None
    with pytest.raises(ValueError,
                       match='gid must be an integer'):
        cell.gid = [1]
    cell.gid = 42
    assert cell.gid == 42
    with pytest.raises(ValueError,
                       match='gid must be an integer'):
        # test init checks gid
        cell = Cell(name, pos, sections, synapses, topology, sect_loc,
                    gid='one')

    # test that ExpSyn always takes nrn.Segment, not float
    with pytest.raises(TypeError, match='secloc must be instance of'):
        cell.syn_create(0.5, e=0., tau1=0.5, tau2=5.)

    pickle.dumps(cell)  # check cell object is picklable until built

    bad_sections = {'blah': Section(L=1, diam=5, Ra=3, cm=100,
                    end_pts=[[0, 0, 0], [0, 39., 0]])}
    # Check soma must be included in sections
    with pytest.raises(KeyError, match='soma must be defined'):
        cell = Cell(name, pos, bad_sections, synapses, topology, sect_loc)

    sections = {
        'soma': Section(
            L=39,
            diam=20,
            cm=0.85,
            Ra=200.,
            end_pts=[[0, 0, 0], [0, 39., 0]]
        )
    }
    sections['soma'].syns = ['ampa']
    sections['soma'].mechs = {
        'km': {
            'gbar_km': 60
        },
        'ca': {
            'gbar_ca': lambda x: 3e-3 * x
        }
    }

    cell = Cell(name, pos, sections, synapses, topology, sect_loc)

    # test successful build
    cell.build()
    assert 'soma' in cell._nrn_sections
    assert cell._nrn_sections['soma'].L == sections['soma'].L
    assert cell._nrn_sections['soma'].gbar_km == sections[
        'soma'].mechs['km']['gbar_km']
    # test building cell with a dipole oriented to a nonexitent section
    with pytest.raises(ValueError, match='sec_name_apical must be an'):
        cell.build(sec_name_apical='blah')

    # Test section modification
    sec_name = 'soma'
    new_L = 1.0
    new_diam = 2.0
    new_cm = 3.0
    new_Ra = 4.0
    cell.modify_section(sec_name, L=new_L, diam=new_diam, cm=new_cm, Ra=new_Ra)

    # Make sure distance betweeen `Section.end_pts` matches `Section.L`
    new_pts = np.array(cell.sections[sec_name].end_pts)
    new_dist = np.linalg.norm(new_pts[0, :] - new_pts[1, :])
    np.isclose(new_L, new_dist)

    assert cell.sections[sec_name].L == new_L
    assert cell.sections[sec_name].diam == new_diam
    assert cell.sections[sec_name].cm == new_cm
    assert cell.sections[sec_name].Ra == new_Ra


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
