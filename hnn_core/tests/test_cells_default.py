import pytest

from neuron import h
import numpy as np

from hnn_core.cells_default import pyramidal, basket
from hnn_core.network_builder import load_custom_mechanisms


def test_cells_default():
    """Test default cell objects."""
    load_custom_mechanisms()

    # Test that an invalid name raises an error
    with pytest.raises(ValueError, match="Unknown pyramidal cell type"):
        pyramidal(cell_name="blah")

    # Test all new valid cell names
    l5p = pyramidal(cell_name="L5_pyramidal")
    l5p.build(sec_name_apical="apical_trunk")
    assert len(l5p.sections) == 9
    assert "apical_2" in l5p.sections

    l2p = pyramidal(cell_name="L2_pyramidal")
    assert len(l2p.sections) == 8
    assert "apical_2" not in l2p.sections

    l5b = basket(cell_name="L5_basket")
    assert len(l5b.sections) == 1

    l2b = basket(cell_name="L2_basket")
    assert len(l2b.sections) == 1

    # Test that old names now raise errors
    old_pyr_names = ["L2Pyr", "L5Pyr"]
    for name in old_pyr_names:
        with pytest.raises(ValueError, match="Unknown pyramidal cell type"):
            pyramidal(cell_name=name)

    old_basket_names = ["L2Basket", "L5Basket"]
    for name in old_basket_names:
        with pytest.raises(ValueError, match="Unknown basket cell type"):
            basket(cell_name=name)

    # check that after building, the vertical sections have the length
    # specified in get_L5Pyr_params_default (or overridden in a params file).
    # Note that the lengths implied by _secs_L5Pyr are completely ignored:
    # NEURON extends the sections as needed to match the sec.L 's
    vertical_secs = [
        "basal_1",
        "soma",
        "apical_trunk",
        "apical_1",
        "apical_2",
        "apical_tuft",
    ]
    for sec_name in vertical_secs:
        sec = l5p._nrn_sections[sec_name]
        vert_len = np.abs(sec.z3d(1) - sec.z3d(0))
        assert np.allclose(vert_len, sec.L)

    # smoke test to check if cell can be used in simulation
    h.load_file("stdrun.hoc")
    h.tstop = 40
    h.dt = 0.025
    h.celsius = 37

    l5p.record(record_vsec="soma")
    vsoma = l5p.vsec["soma"].record(l5p._nrn_sections["soma"](0.5)._ref_v)
    times = h.Vector().record(h._ref_t)

    stim = h.IClamp(l5p._nrn_sections["soma"](0.5))
    stim.delay = 5
    stim.dur = 5.0
    stim.amp = 2.0

    h.finitialize()
    h.fcurrent()
    h.run()

    times = times.to_python()
    vsoma = vsoma.to_python()
    assert len(times) == len(vsoma)

    with pytest.raises(ValueError, match="Unknown basket cell type"):
        basket(cell_name="blah")
