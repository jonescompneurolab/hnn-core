# Authors: Nick Tolley <nicholas_tolley@brown.edu>

from copy import deepcopy
import os.path as op
import pytest

import hnn_core
from hnn_core import read_params, Network


def test_lfp():
    """Test LFP recording API."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    # add rhythmic inputs (i.e., a type of common input)
    params.update({'input_dist_A_weight_L2Pyr_ampa': 5.4e-5,
                   'input_dist_A_weight_L5Pyr_ampa': 5.4e-5,
                   't0_input_dist': 50,
                   'input_prox_A_weight_L2Pyr_ampa': 5.4e-5,
                   'input_prox_A_weight_L5Pyr_ampa': 5.4e-5,
                   't0_input_prox': 50})
    net = Network(deepcopy(params), add_drives_from_params=True)

    # Test LFP electrodes
    electrode_pos = (2, 2, 400)
    sigma, method = 3.0, 'psa'
    net.add_electrode(electrode_pos, sigma, method)
    electrode_pos = [(2, 2, 400), (6, 6, 800)]
    net.add_electrode(electrode_pos, sigma, method)
    assert len(net.lfp) == 3

    pytest.raises(AssertionError, net.add_electrode,
                  [(2, 2), (6, 6, 800)], sigma, method)
    pytest.raises(AssertionError, net.add_electrode,
                  electrode_pos, -1.0, method)

    pytest.raises(ValueError, net.add_electrode,
                  electrode_pos, sigma, 'LSA')

    pytest.raises(TypeError, net.add_electrode,
                  '(2, 2, 400)', sigma, method)
    pytest.raises(TypeError, net.add_electrode,
                  (2, '2', 400), sigma, method)
    pytest.raises(TypeError, net.add_electrode,
                  electrode_pos, '3.0', method)
    pytest.raises(TypeError, net.add_electrode,
                  electrode_pos, sigma, 3.0)
