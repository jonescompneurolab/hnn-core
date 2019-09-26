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
    for p in params:
        assert params[p] == net.params[p]
    assert len(params) == len(net.params)
    print(net)
    print(net.cells[:2])
