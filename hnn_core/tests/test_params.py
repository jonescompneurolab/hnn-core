# Authors: Mainak Jas <mainakjas@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>

import os.path as op
import json
from pathlib import Path
from urllib.request import urlretrieve

import pytest

from hnn_core import (read_params, Params, jones_2009_model, convert_to_hdf5,
                      Network)
from hnn_core.hnn_io import read_network

hnn_core_root = Path(__file__).parents[1]


def test_read_params():
    """Test reading of params object."""
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    # Smoke test that network loads params
    _ = jones_2009_model(
        params, add_drives_from_params=True, legacy_mode=False)
    _ = jones_2009_model(
        params, add_drives_from_params=True, legacy_mode=True)
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
        urlretrieve(param_url, params_legacy_fname)

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
        urlretrieve(param_url, params_base_fname)

    params_base = read_params(params_base_fname)
    params = Params()
    assert params == params_base

    params_base['spec_cmap'] = 'viridis'
    params = Params(params_base)
    assert params == params_base


def test_convert_to_hdf5(tmp_path):
    """Tests conversion of a json file to hdf5"""
    # Download params
    param_url = ('https://raw.githubusercontent.com/hnn-core/'
                 'hnn_core/param/default.json')
    params_base_fname = Path(hnn_core_root, 'param', 'default.json')
    if not op.exists(params_base_fname):
        urlretrieve(param_url, params_base_fname)
    net_params = Network(read_params(params_base_fname),
                         add_drives_from_params=True,
                         )

    # Write hdf5 and check if constructed network is equal
    outpath = Path(tmp_path, 'default.hdf5')
    convert_to_hdf5(params_base_fname, outpath)
    net_hdf5 = read_network(outpath)
    assert net_hdf5 == net_params

    # Write hdf5 without drives
    outpath_no_drives = Path(tmp_path, 'default_no_drives.hdf5')
    convert_to_hdf5(params_base_fname, outpath_no_drives, include_drives=False)
    net_hdf5_no_drives = read_network(outpath_no_drives)
    assert net_hdf5_no_drives != net_hdf5
    assert bool(net_hdf5_no_drives.external_drives) is False

    # Check that writing with no extension will add one
    outpath_no_ext = Path(tmp_path, 'default_no_ext')
    convert_to_hdf5(params_base_fname, outpath_no_ext)
    assert outpath_no_ext.with_suffix('.hdf5').exists()


def test_convert_to_hdf5_legacy(tmp_path):
    """Tests conversion of a param legacy file to hdf5"""
    # Download params
    param_url = ('https://raw.githubusercontent.com/hnnsolver/'
                 'hnn-core/test_data/default.param')
    params_base_fname = Path(hnn_core_root, 'param', 'default.param')
    if not op.exists(params_base_fname):
        urlretrieve(param_url, params_base_fname)
    net_params = Network(read_params(params_base_fname),
                         add_drives_from_params=True,
                         legacy_mode=True
                         )

    # Write hdf5 and check if constructed network is equal
    outpath = Path(tmp_path, 'default.hdf5')
    convert_to_hdf5(params_base_fname, outpath)
    net_hdf5 = read_network(outpath)
    assert net_hdf5 == net_params


def test_convert_to_hdf5_bad_type():
    """Tests type validation in convert_to_hdf5 function"""
    good_path = hnn_core_root
    path_str = str(good_path)
    bad_path = 5

    # Valid path and string, but not actual files
    with pytest.raises(
            ValueError,
            match="Extension must be .param or .json"
    ):
        convert_to_hdf5(good_path, path_str)

    # Bad params_fname
    with pytest.raises(
            TypeError,
            match="params_fname must be an instance of str or Path"
    ):
        convert_to_hdf5(bad_path, good_path)

    # Bad out_fname
    with pytest.raises(
            TypeError,
            match="out_fname must be an instance of str or Path"
    ):
        convert_to_hdf5(good_path, bad_path)
