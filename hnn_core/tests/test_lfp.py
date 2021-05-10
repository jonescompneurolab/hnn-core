# Authors: Nick Tolley <nicholas_tolley@brown.edu>

from copy import deepcopy
import os.path as op
import pytest

import hnn_core
from hnn_core import read_params, Network, simulate_dipole


def test_lfp_api():
    """Test LFP recording API."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    net = Network(deepcopy(params), add_drives_from_params=True)

    # Test LFP electrodes
    electrode_pos = (2, 400, 2)
    sigma, method = 0.3, 'psa'
    net.add_electrode(electrode_pos, sigma, method)
    electrode_pos = [(2, 400, 2), (6, 800, 6)]
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
                  electrode_pos, '0.3', method)
    pytest.raises(TypeError, net.add_electrode,
                  electrode_pos, sigma, 3.0)


def test_lfp_calculation():
    """Test LFP calculation."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    params.update({'N_pyr_x': 3,
                   'N_pyr_y': 3,
                   'tstop': 25,
                   'dipole_smooth_win': 5,
                   't_evprox_1': 5,
                   't_evdist_1': 10,
                   't_evprox_2': 20})
    net = Network(params, add_drives_from_params=True)

    sigma, method = 0.3, 'psa'
    electrode_pos = [(2, 400, 2), (6, 800, 6)]
    net.add_electrode(electrode_pos, sigma, method)
    _ = simulate_dipole(net, n_trials=1)
