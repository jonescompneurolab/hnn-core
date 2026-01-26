import pytest

from neuron import h
import numpy as np

from hnn_core.cells_default import pyramidal, basket, _exp_g_at_dist, _linear_g_at_dist
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


def test_exp_g_at_dist():
    """Test exponential distance-dependent conductance function."""
    # Test at x=0 with default slope
    gbar = _exp_g_at_dist(x=0, gbar_at_zero=1e-6, exp_term=3e-3, offset=0.0)
    expected = 1e-6 * (1 * np.exp(0) + 0.0)  # 1e-6 * 1 = 1e-6
    assert np.isclose(gbar, expected)

    # Test at x=0 with non-default slope and offset
    gbar = _exp_g_at_dist(x=0, gbar_at_zero=2e-6, exp_term=3e-3, offset=0.5, slope=2)
    expected = 2e-6 * (2 * np.exp(0) + 0.5)  # 2e-6 * 2.5 = 5e-6
    assert np.isclose(gbar, expected)

    # Test at positive distance
    x = 100
    gbar = _exp_g_at_dist(x=x, gbar_at_zero=1e-6, exp_term=3e-3, offset=0.0)
    expected = 1e-6 * (1 * np.exp(3e-3 * x) + 0.0)
    assert np.isclose(gbar, expected)

    # Test with negative exp_term (decay)
    x = 100
    gbar = _exp_g_at_dist(x=x, gbar_at_zero=0.06, exp_term=-0.006, offset=1e-4, slope=1)
    expected = 0.06 * (1 * np.exp(-0.006 * x) + 1e-4)
    assert np.isclose(gbar, expected)

    # Test with array input
    x_array = np.array([0, 50, 100, 200])
    gbar_array = _exp_g_at_dist(
        x=x_array, gbar_at_zero=1e-6, exp_term=3e-3, offset=0.0, slope=1
    )
    expected_array = 1e-6 * (np.exp(3e-3 * x_array) + 0.0)
    assert np.allclose(gbar_array, expected_array)

    # Test that function is monotonically increasing with positive exp_term
    x_values = np.linspace(0, 100, 10)
    gbar_values = _exp_g_at_dist(
        x=x_values, gbar_at_zero=1e-6, exp_term=3e-3, offset=0.0, slope=1
    )
    assert np.all(np.diff(gbar_values) > 0)

    # Test that function is monotonically decreasing with negative exp_term
    x_values = np.linspace(0, 100, 10)
    gbar_values = _exp_g_at_dist(
        x=x_values, gbar_at_zero=1.0, exp_term=-0.01, offset=0.0, slope=1
    )
    assert np.all(np.diff(gbar_values) < 0)


def test_linear_g_at_dist():
    """Test linear distance-dependent conductance function."""
    # Test at x=0 (should return gsoma)
    gbar = _linear_g_at_dist(x=0, gsoma=10.0, gdend=40.0, xkink=1501)
    assert np.isclose(gbar, 10.0)

    # Test at x < xkink (linear interpolation)
    x = 750  # halfway to xkink
    gbar = _linear_g_at_dist(x=x, gsoma=10.0, gdend=40.0, xkink=1501)
    expected = 10.0 + 750 * (40.0 - 10.0) / 1501
    assert np.isclose(gbar, expected)

    # Test at x = xkink (should return gdend)
    gbar = _linear_g_at_dist(x=1501, gsoma=10.0, gdend=40.0, xkink=1501)
    assert np.isclose(gbar, 40.0)

    # Test at x > xkink (should still return gdend)
    gbar = _linear_g_at_dist(x=2000, gsoma=10.0, gdend=40.0, xkink=1501)
    assert np.isclose(gbar, 40.0)

    # Test with hotzone (x inside hotzone boundaries)
    x = 500
    gbar = _linear_g_at_dist(
        x=x,
        gsoma=10.0,
        gdend=40.0,
        xkink=1501,
        hotzone_factor=2.0,
        hotzone_boundaries=[400, 600],
    )
    expected_base = 10.0 + 500 * (40.0 - 10.0) / 1501
    expected = expected_base * 2.0
    assert np.isclose(gbar, expected)

    # Test outside hotzone (x before hotzone)
    x = 300
    gbar = _linear_g_at_dist(
        x=x,
        gsoma=10.0,
        gdend=40.0,
        xkink=1501,
        hotzone_factor=2.0,
        hotzone_boundaries=[400, 600],
    )
    expected = 10.0 + 300 * (40.0 - 10.0) / 1501
    assert np.isclose(gbar, expected)

    # Test outside hotzone (x after hotzone)
    x = 700
    gbar = _linear_g_at_dist(
        x=x,
        gsoma=10.0,
        gdend=40.0,
        xkink=1501,
        hotzone_factor=2.0,
        hotzone_boundaries=[400, 600],
    )
    expected = 10.0 + 700 * (40.0 - 10.0) / 1501
    assert np.isclose(gbar, expected)

    # Test at hotzone boundary (exactly at start, should not apply factor)
    x = 400
    gbar = _linear_g_at_dist(
        x=x,
        gsoma=10.0,
        gdend=40.0,
        xkink=1501,
        hotzone_factor=2.0,
        hotzone_boundaries=[400, 600],
    )
    expected = 10.0 + 400 * (40.0 - 10.0) / 1501
    assert np.isclose(gbar, expected)

    # Test at hotzone boundary (exactly at end, should not apply factor)
    x = 600
    gbar = _linear_g_at_dist(
        x=x,
        gsoma=10.0,
        gdend=40.0,
        xkink=1501,
        hotzone_factor=2.0,
        hotzone_boundaries=[400, 600],
    )
    expected = 10.0 + 600 * (40.0 - 10.0) / 1501
    assert np.isclose(gbar, expected)

    # Test with hotzone_factor = 1 (no effect)
    x = 500
    gbar = _linear_g_at_dist(
        x=x,
        gsoma=10.0,
        gdend=40.0,
        xkink=1501,
        hotzone_factor=1.0,
        hotzone_boundaries=[400, 600],
    )
    expected = 10.0 + 500 * (40.0 - 10.0) / 1501
    assert np.isclose(gbar, expected)

    # Test decreasing conductance (gdend < gsoma)
    x = 500
    gbar = _linear_g_at_dist(x=x, gsoma=40.0, gdend=10.0, xkink=1000)
    expected = 40.0 + 500 * (10.0 - 40.0) / 1000
    assert np.isclose(gbar, expected)
    assert gbar < 40.0  # Should be less than gsoma


def test_basket_voltage_init():
    """Test voltage initialization for Basket cell types."""
    # Current section initial voltage, for soma (only section) of both basket types
    expected_basket_v0 = -64.9737

    # Initial setup
    load_custom_mechanisms()
    l5b = basket(cell_name="L5_basket")
    l2b = basket(cell_name="L2_basket")

    # Test initial voltages (v0) for basket cells
    assert np.isclose(l5b.sections["soma"].v0, expected_basket_v0), (
        f"L5_basket soma v0={l5b.sections['soma'].v0}, expected {expected_basket_v0}"
    )
    assert np.isclose(l2b.sections["soma"].v0, expected_basket_v0), (
        f"L2_basket soma v0={l2b.sections['soma'].v0}, expected {expected_basket_v0}"
    )

    # Test that v0 values are correctly applied to built NEURON sections
    l2b.build()
    l5b.build()

    # Initialize with finitialize() which should use the v0 values
    h.finitialize()

    # All L2Pyr sections should have the same v0
    l2b_soma_v = l2b._nrn_sections["soma"](0.5).v
    l5b_soma_v = l5b._nrn_sections["soma"](0.5).v

    assert np.isclose(l2b_soma_v, expected_basket_v0, atol=0.1), (
        f"L2_Basket soma initial voltage is {l2b_soma_v}, expected {expected_basket_v0}"
    )
    assert np.isclose(l5b_soma_v, expected_basket_v0, atol=0.1), (
        f"L5_Basket soma initial voltage is {l5b_soma_v}, expected {expected_basket_v0}"
    )


def test_l2pyr_voltage_init():
    """Test voltage initialization for Layer 2/3 Pyramidal cell type."""
    # Current section initial voltage, for all sections
    expected_l2pyr_v0 = -71.46

    # Initial setup
    load_custom_mechanisms()
    l2pyr = pyramidal(cell_name="L2_pyramidal")

    # Test initial voltages (v0) for L2Pyr
    for sec_name, sec in l2pyr.sections.items():
        assert np.isclose(sec.v0, expected_l2pyr_v0), (
            f"L2Pyr {sec_name} v0={sec.v0}, expected {expected_l2pyr_v0}"
        )

    # Test that v0 values are correctly applied to built NEURON sections
    l2pyr.build(sec_name_apical="apical_trunk")

    # Initialize with finitialize() which should use the v0 values
    h.finitialize()

    # All L2Pyr sections should have the same v0
    soma_v = l2pyr._nrn_sections["soma"](0.5).v
    apical_1_v = l2pyr._nrn_sections["apical_1"](0.5).v
    basal_1_v = l2pyr._nrn_sections["basal_1"](0.5).v

    assert np.isclose(soma_v, expected_l2pyr_v0, atol=0.1), (
        f"L2Pyr soma initial voltage is {soma_v}, expected {expected_l2pyr_v0}"
    )
    assert np.isclose(apical_1_v, expected_l2pyr_v0, atol=0.1), (
        f"L2Pyr apical_1 initial voltage is {apical_1_v}, expected {expected_l2pyr_v0}"
    )
    assert np.isclose(basal_1_v, expected_l2pyr_v0, atol=0.1), (
        f"L2Pyr basal_1 initial voltage is {basal_1_v}, expected {expected_l2pyr_v0}"
    )


def test_l5pyr_voltage_init():
    """Test voltage initialization for Layer 2/3 Pyramidal cell type."""
    # Current section initial voltages
    expected_l5pyr_v0 = {
        "apical_1": -71.32,
        "apical_2": -69.08,
        "apical_tuft": -67.30,
        "apical_trunk": -72,
        "soma": -72.0,
        "basal_1": -72,
        "basal_2": -72,
        "basal_3": -72,
        "apical_oblique": -72,
    }

    # Initial setup
    load_custom_mechanisms()
    l5pyr = pyramidal(cell_name="L5_pyramidal")

    # Test initial voltages (v0) for L5Pyr - different sections have different values
    for sec_name, sec in l5pyr.sections.items():
        expected_v0 = expected_l5pyr_v0[sec_name]
        assert np.isclose(sec.v0, expected_v0), (
            f"L5Pyr {sec_name} v0={sec.v0}, expected {expected_v0}"
        )

    # Test that v0 values are correctly applied to built NEURON sections
    l5pyr.build(sec_name_apical="apical_trunk")

    # After building, NEURON sections should have v initialized to v0
    # Check a few key sections with different v0 values
    apical_2_nrn_sec = l5pyr._nrn_sections["apical_2"]
    apical_tuft_nrn_sec = l5pyr._nrn_sections["apical_tuft"]
    soma_nrn_sec = l5pyr._nrn_sections["soma"]

    # Initialize with finitialize() which should use the v0 values
    h.finitialize()

    # Check that voltages match expected v0 values
    # Note: We check at the midpoint (0.5) of each section
    assert np.isclose(apical_2_nrn_sec(0.5).v, -69.08, atol=0.1), (
        f"apical_2 initial voltage is {apical_2_nrn_sec(0.5).v}, expected -69.08"
    )
    assert np.isclose(apical_tuft_nrn_sec(0.5).v, -67.30, atol=0.1), (
        f"apical_tuft initial voltage is {apical_tuft_nrn_sec(0.5).v}, expected -67.30"
    )
    assert np.isclose(soma_nrn_sec(0.5).v, -72.0, atol=0.1), (
        f"soma initial voltage is {soma_nrn_sec(0.5).v}, expected -72.0"
    )
