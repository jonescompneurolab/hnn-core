# Authors: Mainak Jas <mainakjas@gmail.com>
#          Christopher Bailey <bailey.cj@gmail.com>

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
    assert len(net.gid_dict['extinput']) == 2
    assert len(net.gid_dict['extgauss']) == net.N_cells
    assert len(net.gid_dict['extpois']) == net.N_cells
    for ev_input in params['t_ev*']:
        type_key = ev_input[2: -2] + ev_input[-1]
        assert len(net.gid_dict[type_key]) == net.N_cells


def test_external_rhythmic_feeds():
    """Test external rhythmic feeds to proximal and distal dendrites."""
    hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)

    # default parameters have no rhythmic inputs (distal or proximal),
    params.update({'input_dist_A_weight_L2Pyr_ampa': 5.4e-5,
                   'input_dist_A_weight_L5Pyr_ampa': 5.4e-5,
                   't0_input_dist': 50,
                   'input_prox_A_weight_L2Pyr_ampa': 5.4e-5,
                   'input_prox_A_weight_L5Pyr_ampa': 5.4e-5,
                   't0_input_prox': 50})
    net = Network(deepcopy(params))

    # from Network.build: creates ExtFeeds in < 1 sec
    net._create_all_src()

    # annoyingly, net.extinput_list is always 2, whether populated
    # or not, so the first few assertions have no effect
    # the last one needs to be > 0, though (catches #101, tests #102)
    # Assert number of rhythmic inputs is 2 (distal & proximal)
    assert len(net.extinput_list) == 2
    for ei in net.extinput_list:
        # naming could be better
        assert ei.ty == 'extinput'
        # eventvec is of type h.Vector
        assert ei.eventvec.hname().startswith('Vector')
        # not sure why this is 40 for both, just test > 0
        assert len(ei.eventvec.as_numpy()) > 0
