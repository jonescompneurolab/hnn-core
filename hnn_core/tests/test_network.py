# Authors: Mainak Jas <mainakjas@gmail.com>

from copy import deepcopy
import os.path as op

import hnn_core
from hnn_core import read_params, Network


def test_network():
    """Test network object."""
    hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    net = Network(deepcopy(params))

    # Assert that params are conserved across Network initialization
    for p in params:
        assert params[p] == net.params[p]
    assert len(params) == len(net.params)
    print(net)
    print(net.cells[:2])

    # Assert that proper number of gids are created for Network inputs
    assert len(net.gid_dict['common']) == 0
    assert len(net.gid_dict['extgauss']) == net.n_cells
    assert len(net.gid_dict['extpois']) == net.n_cells
    for ev_input in params['t_ev*']:
        type_key = ev_input[2: -2] + ev_input[-1]
        assert len(net.gid_dict[type_key]) == net.N_cells
    net.write_spikes('/tmp/spk1.txt', trial_idx=None)
    assert op.exists('/tmp/spk1.txt', trial_idx=None)
