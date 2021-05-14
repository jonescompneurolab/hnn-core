import pytest

from neuron import h

from hnn_core.cells_default import pyramidal, basket
from hnn_core.network_builder import load_custom_mechanisms


def test_cells_default():
    """Test default cell objects."""
    load_custom_mechanisms()

    with pytest.raises(ValueError, match='Unknown pyramidal cell type'):
        l5p = pyramidal(cell_name='blah')

    l5p = pyramidal(cell_name='L5Pyr')
    l5p.build(sec_name_apical='apical_trunk')
    assert len(l5p.sections) == 9
    assert 'apical_2' in l5p.sections

    # smoke test to check if cell can be used in simulation
    h.load_file("stdrun.hoc")
    h.tstop = 40
    h.dt = 0.025
    h.celsius = 37

    vsoma = l5p.rec_v.record(l5p.sections['soma'](0.5)._ref_v)
    times = h.Vector().record(h._ref_t)

    stim = h.IClamp(l5p.sections['soma'](0.5))
    stim.delay = 5
    stim.dur = 5.
    stim.amp = 2.

    h.finitialize()
    h.fcurrent()
    h.run()

    times = times.to_python()
    vsoma = vsoma.to_python()
    assert len(times) == len(vsoma)

    with pytest.raises(ValueError, match='Unknown basket cell type'):
        l5p = basket(cell_name='blah')
