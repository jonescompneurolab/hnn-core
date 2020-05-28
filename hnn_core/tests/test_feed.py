# Authors: Mainak Jas <mainakjas@gmail.com>
#          Christopher Bailey <bailey.cj@gmail.com>

from copy import deepcopy
import os.path as op

import hnn_core
from hnn_core import read_params, Network


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

    with Network(deepcopy(params)) as net:
        net._create_all_src()

        assert len(net.extinput_list) == 2  # (distal & proximal)
        for ei in net.extinput_list:
            # XXX: need to rename ei.ty to 'rhythmic'
            assert ei.ty == 'extinput'
            assert ei.eventvec.hname().startswith('Vector')
            # not sure why this is 40 for both, just test > 0
            assert len(ei.eventvec.as_numpy()) > 0

            # check that ei.p_ext matches params
            loc = ei.p_ext['loc'][:4]  # loc=prox or dist
            for layer in ['L2', 'L5']:
                key = 'input_{}_A_weight_{}Pyr_ampa'.format(loc, layer)
                assert ei.p_ext[layer + 'Pyr_ampa'][0] == params[key]
