# Authors: Mainak Jas <mainakjas@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>

import os.path as op
import json

import pytest
from mne.utils import _fetch_file

import hnn_core
from hnn_core import read_params, Params
hnn_core_root = op.dirname(hnn_core.__file__)


def test_read_params():
    """Test reading of params object."""
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    print(params)
    print(params['L2Pyr*'])

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


def test_read_legacy_params():
    """Test reading of legacy .param file."""
    param_url = ('https://raw.githubusercontent.com/hnnsolver/'
                 'hnn-core/test_data/default.param')
    params_legacy_fname = op.join(hnn_core_root, 'param', 'default.param')
    if not op.exists(params_legacy_fname):
        _fetch_file(param_url, params_legacy_fname)

    params_new_fname = op.join(hnn_core_root, 'param', 'default.json')
    params_legacy = read_params(params_legacy_fname)
    params_new = read_params(params_new_fname)

    params_new_seedless = {key: val for key, val in params_new.items()
                           if key not in params_new['prng_seedcore*'].keys()}
    params_legacy_seedless = {key: val for key, val in params_legacy.items()
                              if key not in
                              params_legacy['prng_seedcore*'].keys()}
    assert params_new_seedless == params_legacy_seedless


def test_base_params():
    """Test default params object matches base params"""
    param_url = ('https://raw.githubusercontent.com/jonescompneurolab/'
                 'hnn-core/test_data/base.json')
    params_base_fname = op.join(hnn_core_root, 'param', 'base.json')
    if not op.exists(params_base_fname):
        _fetch_file(param_url, params_base_fname)

    params_base = read_params(params_base_fname)
    params = Params()
    assert params == params_base

    params_base['spec_cmap'] = 'viridis'
    params = Params(params_base)
    assert params == params_base
