# Authors: Mainak Jas <mainakjas@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>

import os.path as op
import json
from pathlib import Path
from urllib.request import urlretrieve

import pytest

from hnn_core import read_params, Params, convert_to_json
from hnn_core.hnn_io import read_network_configuration
from hnn_core.network_models import jones_2009_model
from hnn_core.params import remove_nulled_drives


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


def test_remove_nulled_drives(tmp_path):
    param_url = ("https://raw.githubusercontent.com/jonescompneurolab/hnn/"
                 "master/param/ERPYes100Trials.param")
    params_fname = Path(hnn_core_root, 'param', 'ERPYes100Trials.param')
    if not op.exists(params_fname):
        urlretrieve(param_url, params_fname)

    net = jones_2009_model(params=read_params(params_fname),
                           add_drives_from_params=True,
                           legacy_mode=True,
                           )
    net_removed = remove_nulled_drives(net)

    assert net_removed != net

    # External drives were removed
    drives_removed = ['bursty1', 'bursty2', 'extgauss', 'extpois']
    assert all([drive not in net_removed.external_drives.keys()
                for drive in drives_removed])

    # Connections were removed
    conn_src_types = set([conn['src_type']
                          for conn in net_removed.connectivity])
    assert all([drive not in conn_src_types for drive in drives_removed])

    # gid ranges were updated
    assert all([drive not in net_removed.gid_ranges.keys()
                for drive in drives_removed])

    # position dictionary was updated
    assert all([drive not in net_removed.pos_dict.keys()
                for drive in drives_removed])


class TestConvertToJson:
    """Tests convert_to_json function"""

    path_default = Path(hnn_core_root, 'param', 'default.json')

    def test_default_network_connectivity(self, tmp_path):
        """Tests conversion with default parameters"""

        net_params = jones_2009_model(params=read_params(self.path_default),
                                      add_drives_from_params=True)

        # Write json and check if constructed network is equal
        outpath = Path(tmp_path, 'default.json')
        convert_to_json(self.path_default,
                        outpath
                        )
        net_json = read_network_configuration(outpath)
        assert net_json == net_params

        # Write json without drives
        outpath_no_drives = Path(tmp_path, 'default_no_drives.json')
        convert_to_json(self.path_default,
                        outpath_no_drives,
                        include_drives=False
                        )
        net_json_no_drives = read_network_configuration(outpath_no_drives)
        assert net_json_no_drives != net_json
        assert bool(net_json_no_drives.external_drives) is False

        # Check that writing with no extension will add one
        outpath_no_ext = Path(tmp_path, 'default_no_ext')
        convert_to_json(self.path_default,
                        outpath_no_ext
                        )
        assert outpath_no_ext.with_suffix('.json').exists()

    def test_convert_to_json_legacy(self, tmp_path):
        """Tests conversion of a param legacy file to json"""

        # Download params
        param_url = ('https://raw.githubusercontent.com/hnnsolver/'
                     'hnn-core/test_data/default.param')
        params_base_fname = Path(hnn_core_root, 'param', 'default.param')
        if not op.exists(params_base_fname):
            urlretrieve(param_url, params_base_fname)
        net_params = jones_2009_model(read_params(params_base_fname),
                                      add_drives_from_params=True,
                                      legacy_mode=True
                                      )

        # Write json and check if constructed network is correct
        outpath = Path(tmp_path, 'default.json')
        convert_to_json(params_base_fname, outpath)
        net_json = read_network_configuration(outpath)
        # Nulled drives are removed, so they should not be equal
        assert net_json != net_params
        assert len(net_json.external_drives) == 3
        assert len(net_json.connectivity) == 39
        assert len(net_json.gid_ranges) == 7

    def test_convert_to_json_bad_type(self):
        """Tests type validation in convert_to_json function"""

        good_path = hnn_core_root
        path_str = str(good_path)
        bad_path = 5

        # Valid path and string, but not actual files
        with pytest.raises(
                ValueError,
                match="Unrecognized extension, expected one of"
        ):
            convert_to_json(good_path, path_str)

        # Bad params_fname
        with pytest.raises(
                TypeError,
                match="params_fname must be an instance of str or Path"
        ):
            convert_to_json(bad_path, good_path)

        # Bad out_fname
        with pytest.raises(
                TypeError,
                match="out_fname must be an instance of str or Path"
        ):
            convert_to_json(good_path, bad_path)
