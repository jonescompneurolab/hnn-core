import pytest
import pickle
import numpy as np

import matplotlib

from hnn_core.network_builder import load_custom_mechanisms
from hnn_core.cell import _ArtificialCell, Cell, Section
from hnn_core import pyramidal

matplotlib.use("agg")


def test_cell():
    """Test cells object."""
    load_custom_mechanisms()

    name = "test"
    pos = (0.0, 0.0, 0.0)
    sections = {
        "soma": Section(L=1, diam=5, Ra=3, cm=100, end_pts=[[0, 0, 0], [0, 39.0, 0]])
    }
    synapses = {"ampa": dict(e=0, tau1=0.5, tau2=5.0)}
    cell_tree = {("soma", 0): [("soma", 1)]}
    sect_loc = {"proximal": "soma"}
    # GID is assigned exactly once for each cell, either at initialisation...
    cell = Cell(name, pos, sections, synapses, sect_loc, cell_tree, gid=42)
    assert cell.gid == 42
    with pytest.raises(RuntimeError, match="Global ID for this cell already assigned!"):
        cell.gid += 1
    # ... or later
    # cells can exist fine without gid
    cell = Cell(name, pos, sections, synapses, sect_loc, cell_tree)
    assert cell.gid is None  # check that it's initialised to None
    with pytest.raises(ValueError, match="gid must be an integer"):
        cell.gid = [1]
    cell.gid = 42
    assert cell.gid == 42
    with pytest.raises(ValueError, match="gid must be an integer"):
        # test init checks gid
        cell = Cell(name, pos, sections, synapses, sect_loc, cell_tree, gid="one")

    # test that ExpSyn always takes nrn.Segment, not float
    with pytest.raises(TypeError, match="secloc must be instance of"):
        cell.syn_create(0.5, e=0.0, tau1=0.5, tau2=5.0)

    pickle.dumps(cell)  # check cell object is picklable until built

    bad_sections = {
        "blah": Section(L=1, diam=5, Ra=3, cm=100, end_pts=[[0, 0, 0], [0, 39.0, 0]])
    }
    # Check soma must be included in sections
    with pytest.raises(KeyError, match="soma must be defined"):
        cell = Cell(name, pos, bad_sections, synapses, sect_loc, cell_tree)

    sections = {
        "soma": Section(
            L=39, diam=20, cm=0.85, Ra=200.0, end_pts=[[0, 0, 0], [0, 39.0, 0]]
        )
    }
    sections["soma"].syns = ["ampa"]
    sections["soma"].mechs = {
        "km": {"gbar_km": 60},
        "ca": {"gbar_ca": lambda x: 3e-3 * x},
    }

    cell = Cell(name, pos, sections, synapses, sect_loc, cell_tree)

    # test successful build
    cell.build()
    assert "soma" in cell._nrn_sections
    assert cell._nrn_sections["soma"].L == sections["soma"].L
    assert cell._nrn_sections["soma"].gbar_km == sections["soma"].mechs["km"]["gbar_km"]
    # test building cell with a dipole oriented to a nonexitent section
    with pytest.raises(ValueError, match="sec_name_apical must be an"):
        cell.build(sec_name_apical="blah")

    # Test section modification
    sec_name = "soma"
    new_L = 1.0
    new_diam = 2.0
    new_cm = 3.0
    new_Ra = 4.0
    cell.modify_section(sec_name, L=new_L, diam=new_diam, cm=new_cm, Ra=new_Ra)

    # Make sure distance between `Section.end_pts` matches `Section.L`
    new_pts = np.array(cell.sections[sec_name].end_pts)
    new_dist = np.linalg.norm(new_pts[0, :] - new_pts[1, :])
    np.isclose(new_L, new_dist)

    assert cell.sections[sec_name].L == new_L
    assert cell.sections[sec_name].diam == new_diam
    assert cell.sections[sec_name].cm == new_cm
    assert cell.sections[sec_name].Ra == new_Ra

    # Testing update end pts using template cell
    cell1 = pyramidal(cell_name="L5Pyr")

    # Test other not NotImplemented for Cell Class
    assert (cell1 == "cell") is False

    # Test other not NotImplemented for Section Class
    assert (cell1.sections["soma"] == "section") is False

    end_pts_original = list()
    end_pts_new = list()
    for sec_name in cell1.sections.keys():
        section = cell1.sections[sec_name]
        end_pts_original.append(section.end_pts)
        section._L = section._L * 2
        cell1.sections[sec_name] = section
    cell1._update_end_pts()
    for sec_name in cell1.sections.keys():
        end_pts_new.append(cell1.sections[sec_name].end_pts)

    # All coordinates are multiplied by 2 since all section
    # lengths are doubled
    cell1.plot_morphology(show=True)
    for end_pt_original, end_pt_new in zip(end_pts_original, end_pts_new):
        for pt_original, pt_new in zip(end_pt_original, end_pt_new):
            np.testing.assert_almost_equal(list(np.array(pt_original) * 2), pt_new, 5)

    for sec_name in cell1.sections.keys():
        section = cell1.sections[sec_name]
        section._L = section._L / 2
        cell1.sections[sec_name] = section

    end_pts_new = list()
    cell1._update_end_pts()
    for sec_name in cell1.sections.keys():
        section = cell1.sections[sec_name]
        cell1.sections[sec_name] = section
        end_pts_new.append(section.end_pts)
    cell1.plot_morphology(show=False)
    # Checking equality till 5 decimal places
    np.testing.assert_almost_equal(end_pts_original, end_pts_new, 5)

    # Testing distance function using template cell (L5pyr)
    sec_dist = dict()
    sec_dist["soma"] = 19.5
    sec_dist["apical_trunk"] = 90
    sec_dist["apical_oblique"] = 268.5
    sec_dist["apical_1"] = 481
    sec_dist["apical_2"] = 1161
    sec_dist["apical_tuft"] = 1713.5
    sec_dist["basal_1"] = 42.5
    sec_dist["basal_2"] = 212.5
    sec_dist["basal_3"] = 212.5
    for sec_name in cell1.sections.keys():
        sec_dist_test = cell1.distance_section(sec_name, ("soma", 0))
        assert sec_dist_test == sec_dist[sec_name]


def test_artificial_cell():
    """Test artificial cell object."""
    load_custom_mechanisms()
    event_times = [1, 2, 3]
    threshold = 0.0
    artificial_cell = _ArtificialCell(event_times, threshold)
    assert artificial_cell.nrn_eventvec.to_python() == event_times
    # the h.VecStim() object defined in vecevent.mod should contain a 'play()'
    # method
    assert hasattr(artificial_cell.nrn_vecstim, "play")
    # the h.Netcon() instance should reference the h.VecStim() instance
    assert artificial_cell.nrn_netcon.pre() == artificial_cell.nrn_vecstim
    assert artificial_cell.nrn_netcon.threshold == threshold

    # GID is assigned exactly once for each cell, either at initialisation...
    cell = _ArtificialCell(event_times, threshold, gid=42)
    assert cell.gid == 42
    with pytest.raises(RuntimeError, match="Global ID for this cell already assigned!"):
        cell.gid += 1
    with pytest.raises(ValueError, match="gid must be an integer"):
        cell.gid = [1]
    # ... or later
    cell = _ArtificialCell(event_times, threshold)  # fine without gid
    assert cell.gid is None  # check that it's initialised to None
    cell.gid = 42
    assert cell.gid == 42
    with pytest.raises(
        ValueError,  # test init checks gid
        match="gid must be an integer",
    ):
        cell = _ArtificialCell(event_times, threshold, gid="one")
