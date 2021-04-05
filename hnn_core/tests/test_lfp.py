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
    kwargs_default = {
        'electrode_pos': (2, 2, 400),
        'sigma': 3.0, 'method': 'psa'}
    net.add_electrode(**kwargs_default)
    kwargs_default['electrode_pos'] = [(2, 2, 400), (6, 6, 800)]
    net.add_electrode(**kwargs_default)
    assert len(net.lfp) == 3
    with pytest.raises(AssertionError):
        kwargs = kwargs_default.copy()
        kwargs['electrode_pos'] = [(2, 2), (6, 6, 800)]
        net.add_electrode(**kwargs)
    with pytest.raises(AssertionError):
        kwargs = kwargs_default.copy()
        kwargs['sigma'] = -1.0
        net.add_electrode(**kwargs)

    match = "Invalid value for the 'method' parameter"
    with pytest.raises(ValueError, match=match):
        kwargs = kwargs_default.copy()
        kwargs['method'] = 'LSA'
        net.add_electrode(**kwargs)

    bad_kwargs = [
        ('electrode_pos', '[(2, 2, 400), (6, 6, 800)]'),
        ('electrode_pos', (2, '2', 400)),
        ('sigma', '3.0'), ('method', 3.0)]
    for arg, item in bad_kwargs:
        kwargs = kwargs_default.copy()
        kwargs[arg] = item
        match = 'must be an instance of'
        with pytest.raises(TypeError, match=match):
            net.add_electrode(**kwargs)
