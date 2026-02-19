import pytest

from neuron import h
import numpy as np

from hnn_core.cells_default import (
    pyramidal,
    basket,
    _exp_g_at_dist,
    _linear_g_at_dist,
    _get_syn_props,
    _get_basket_syn_props,
)
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


def test_get_syn_props_backwards_compat():
    """_get_syn_props with no type key in p_all defaults to Exp2Syn."""
    # Simulate a p_all dict without any _type keys (legacy / backwards-compat case)
    cell_type = "L2Pyr"
    syn_type = "ampa"
    p_all = {
        f"{cell_type}_{syn_type}_e": 0.0,
        f"{cell_type}_{syn_type}_tau1": 0.5,
        f"{cell_type}_{syn_type}_tau2": 5.0,
    }
    result1 = _get_syn_props(p_all, cell_type, syn_types=[syn_type])
    for syn_type, syn_param in zip([syn_type], ["e", "tau1", "tau2"]):
        assert (
            result1[syn_type][syn_param] == p_all[f"{cell_type}_{syn_type}_{syn_param}"]
        )
    # Test that type defaults to Exp2Syn when no type key is present
    assert result1[syn_type]["type"] == "Exp2Syn"

    # Add the type key to test that "Exp2Syn" is correctly read when present
    p_all.update(
        {
            f"{cell_type}_{syn_type}_type": "Exp2Syn",
        }
    )
    result2 = _get_syn_props(p_all, cell_type, syn_types=[syn_type])
    # Test that type key and value are correctly present
    for syn_type, syn_param in zip([syn_type], ["e", "tau1", "tau2", "type"]):
        assert (
            result2[syn_type][syn_param] == p_all[f"{cell_type}_{syn_type}_{syn_param}"]
        )


def test_get_syn_props_custom_type():
    """_get_syn_props with a custom synapse type collects all custom params."""
    cell_type = "L2Pyr"
    syn_type = "ampa"
    p_all = {
        f"{cell_type}_{syn_type}_type": "ExpSyn",
        f"{cell_type}_{syn_type}_e": 0.0,
        f"{cell_type}_{syn_type}_tau": 5.0,
    }
    result = _get_syn_props(p_all, cell_type, syn_types=[syn_type])
    for syn_type, syn_param in zip([syn_type], ["e", "tau", "type"]):
        assert (
            result[syn_type][syn_param] == p_all[f"{cell_type}_{syn_type}_{syn_param}"]
        )
    # tau1/tau2 should NOT be present since they weren't in p_all
    assert "tau1" not in result[syn_type]
    assert "tau2" not in result[syn_type]


def test_get_syn_props_NMDA_gao_type_params():
    """_get_syn_props with the new NMDA_gao synapse type works for all custom params."""
    cell_type = "L5Pyr"
    syn_type = "nmda"
    p_all = {
        f"{cell_type}_{syn_type}_type": "NMDA_gao",
        f"{cell_type}_{syn_type}_e": 0.0,
        f"{cell_type}_{syn_type}_Beta": 5.0,
        f"{cell_type}_{syn_type}_Cdur": 1.2,
    }
    result = _get_syn_props(p_all, cell_type, syn_types=[syn_type])
    result = _get_syn_props(p_all, cell_type, syn_types=[syn_type])
    for syn_type, syn_param in zip([syn_type], ["e", "Beta", "Cdur", "type"]):
        assert (
            result[syn_type][syn_param] == p_all[f"{cell_type}_{syn_type}_{syn_param}"]
        )
    # tau1/tau2 should NOT be present since they weren't in p_all
    assert "tau1" not in result[syn_type]
    assert "tau2" not in result[syn_type]


def test_get_syn_props_multiple_syn_types():
    """_get_syn_props returns all requested syn_types, including mixed custom types."""
    p_all = {
        "L2Pyr_ampa_e": 0.0,
        "L2Pyr_ampa_tau1": 0.5,
        "L2Pyr_ampa_tau2": 5.0,
        "L2Pyr_gabaa_e": -80.0,
        "L2Pyr_gabaa_tau1": 0.5,
        "L2Pyr_gabaa_tau2": 5.0,
        "L2Pyr_nmda_type": "NMDA_gao",
        "L2Pyr_nmda_e": 0.0,
        "L2Pyr_nmda_Beta": 5.0,
        "L2Pyr_nmda_Cdur": 1.2,
    }
    result1 = _get_syn_props(p_all, "L2Pyr", syn_types=["ampa"])
    assert "ampa" in result1
    assert "gabaa" not in result1

    result2 = _get_syn_props(p_all, "L2Pyr", syn_types=["ampa", "gabaa"])
    assert set(result2.keys()) == {"ampa", "gabaa"}
    assert result2["ampa"]["type"] == "Exp2Syn"
    assert result2["gabaa"]["e"] == -80.0
    assert result2["gabaa"]["type"] == "Exp2Syn"

    result3 = _get_syn_props(p_all, "L2Pyr", syn_types=["ampa", "nmda"])
    assert set(result3.keys()) == {"ampa", "nmda"}
    assert result3["ampa"]["type"] == "Exp2Syn"
    assert result3["nmda"]["Beta"] == 5.0
    assert result3["nmda"]["type"] == "NMDA_gao"


def test_get_basket_syn_props_includes_type():
    """_get_basket_syn_props should include 'type': 'Exp2Syn' for each synapse."""
    result = _get_basket_syn_props()
    for syn_name, syn_props in result.items():
        assert "type" in syn_props, f"Missing 'type' key in basket synapse '{syn_name}'"
        assert syn_props["type"] == "Exp2Syn"


def test_pyramidal_synapses_have_type():
    """Pyramidal cells should have 'type' key in each synapse after construction."""
    load_custom_mechanisms()
    for cell_name in ["L2_pyramidal", "L5_pyramidal"]:
        cell = pyramidal(cell_name=cell_name)
        for syn_name, syn_props in cell.synapses.items():
            assert "type" in syn_props, (
                f"Synapse '{syn_name}' on {cell_name} is missing 'type' key"
            )


def test_basket_synapses_have_type():
    """Basket cells should have 'type' key in each synapse after construction."""
    load_custom_mechanisms()
    for cell_name in ["L2_basket", "L5_basket"]:
        cell = basket(cell_name=cell_name)
        for syn_name, syn_props in cell.synapses.items():
            assert "type" in syn_props, (
                f"Synapse '{syn_name}' on {cell_name} is missing 'type' key"
            )
