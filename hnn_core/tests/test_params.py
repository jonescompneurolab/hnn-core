# Authors: Mainak Jas <mainakjas@gmail.com>

import os.path as op

import hnn_core
from hnn_core import Params


def test_params():
    """Test params object."""
    hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = Params(params_fname)
    print(params)
    print(params['L2Pyr*'])
