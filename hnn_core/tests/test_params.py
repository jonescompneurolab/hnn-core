# Authors: Mainak Jas <mainakjas@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>

import os.path as op
import json

import pytest
from mne.utils import _fetch_file

import hnn_core
from hnn_core import read_params, Params
hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')


def test_params():
    """Test params object."""
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    print(params)
    print(params['L2Pyr*'])


def test_base_params():
    """Test params object with base params"""
    param_url = ('https://raw.githubusercontent.com/jonescompneurolab/'
                 'hnn-core/test_data/base.json')
    params_base_fname = op.join(hnn_core_root, 'param', 'base.json')
    if not op.exists(params_base_fname):
        _fetch_file(param_url, params_base_fname)

    params_base = read_params(params_base_fname)
    params = Params()
    assert params == params_base

    # unsupported extension
    pytest.raises(ValueError, read_params, 'params.txt')
    # empty file
    empty_fname = op.join(hnn_core_root, 'param', 'empty.json')
    with open(empty_fname, 'w') as json_data:
        json.dump({}, json_data)
    pytest.raises(ValueError, read_params, empty_fname)
    # non dict type
    pytest.raises(ValueError, Params, [])
    pytest.raises(ValueError, Params, 'sdfdfdf')
