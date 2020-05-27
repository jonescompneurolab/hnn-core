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
    net = Network(deepcopy(params))

    # from Network.build: creates ExtFeeds in < 1 sec
    net._create_all_src()

    # annoyingly, net.extinput_list is always 2, whether populated
    # or not, so the first few assertions have no effect
    # the eventvec needs to be > 0, though (catches #101, tests #102)
    assert len(net.extinput_list) == 2  # (distal & proximal)
    for ei in net.extinput_list:
        # naming could be better
        assert ei.ty == 'extinput'
        # eventvec is of type h.Vector
        assert ei.eventvec.hname().startswith('Vector')
        # not sure why this is 40 for both, just test > 0
        assert len(ei.eventvec.as_numpy()) > 0
        # move this to new test of create_pext in params?
        # tests that cryptically named input parameters indeed
        # copied into p_ext
        loc = ei.p_ext['loc'][:4]
        for lay in ['L2', 'L5']:
            pname = 'input_' + loc + '_A_weight_' + lay + 'Pyr_ampa'
            assert ei.p_ext[lay + 'Pyr_ampa'][0] == params[pname]
