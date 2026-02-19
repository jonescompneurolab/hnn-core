from hnn_core.cells_default import _get_syn_props
from hnn_core.params_default import get_L2Pyr_params_default, get_L5Pyr_params_default

# AES: yeah yeah we're eventually going to deprecate params_default. Still, as we do
# that, it may be useful to have a place to put tests for it.


def test_get_syn_props_l2pyr_default_params_backwards_compat():
    """L2Pyr default params have mixed type/no-type keys; all synapses should resolve."""
    p_all = get_L2Pyr_params_default()
    result = _get_syn_props(p_all, "L2Pyr")
    # All four default synapse types should be present
    for syn in ["ampa", "nmda", "gabaa", "gabab"]:
        assert syn in result
        # All should fall back to or explicitly declare Exp2Syn
        assert result[syn]["type"] == "Exp2Syn"
        assert "e" in result[syn]
        assert "tau1" in result[syn]
        assert "tau2" in result[syn]


def test_get_syn_props_l5pyr_default_params_backwards_compat():
    """L5Pyr default params have no type keys; all synapses should resolve via backwards compat."""
    p_all = get_L5Pyr_params_default()
    result = _get_syn_props(p_all, "L5Pyr")
    for syn in ["ampa", "nmda", "gabaa", "gabab"]:
        assert syn in result
        assert result[syn]["type"] == "Exp2Syn"
        assert "e" in result[syn]
        assert "tau1" in result[syn]
        assert "tau2" in result[syn]
