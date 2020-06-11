# Authors: Mainak Jas <mainakjas@gmail.com>

from copy import deepcopy
import os.path as op
from glob import glob
import numpy as np
import pytest

import hnn_core
from hnn_core import read_params, Network, Spikes, read_spikes


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
        assert len(net.gid_dict[type_key]) == net.n_cells

    # Assert that an empty Spikes object is created as an attribute
    assert net.spikes == Spikes()


def test_spikes():
    '''Test spikes object.'''

    # Round-trip test
    spiketimes = [[2.3456, 7.89], [4.2812, 93.2]]
    spikegids = [[1, 3], [5, 7]]
    spiketypes = [['L2_pyramidal', 'L2_basket'], ['L5_pyramidal', 'L5_basket']]
    spikes = Spikes(times=spiketimes, gids=spikegids, types=spiketypes)
    spikes.write('/tmp/spk_%d.txt')
    assert spikes == read_spikes('/tmp/spk_*.txt')

    # TypeError should be raised when one of the args is entered as a non-list
    # (e.g., when times is a tuple of lists)
    with pytest.raises(TypeError) as excinfo:
        spiketimes_tuple = ([2.3456, 7.89], [4.2812, 93.2])
        spikes = Spikes(times=spiketimes_tuple, gids=spikegids,
                        types=spiketypes)
    assert "times should be a list of lists" in str(excinfo.value)

    # TypeError should be raised when one of the args is entered as a list of
    # non-lists (e.g., when times is a list of ints)
    with pytest.raises(TypeError) as excinfo:
        spikes = Spikes(times=[1, 2], gids=spikegids, types=spiketypes)
    assert "times should be a list of lists" in str(excinfo.value)

    # ValueError should be raised when one of the args is entered as an
    # incongruent number of trials (e.g., 1 trial when all other args have 2)
    with pytest.raises(ValueError) as excinfo:
        spiketimes_1_trial = [[2.3456, 7.89]]
        spikes = Spikes(times=spiketimes_1_trial, gids=spikegids,
                        types=spiketypes)
    assert ("times, gids, and types should be lists of the same length"
            in str(excinfo.value))

    # Write spike file with no 'types' column
    # Check for error when read back in without providing gid_dict
    for fname in sorted(glob('/tmp/spk_*.txt')):
        times_gids_only = np.loadtxt(fname, dtype=str)[:, (0, 1)]
        np.savetxt(fname, times_gids_only, delimiter='\t', fmt='%s')
    with pytest.raises(ValueError) as excinfo:
        spikes = read_spikes('/tmp/spk_*.txt')
    assert ("gid_dict must be provided if spike types are unspecified in "
            "'spk.txt' file" in str(excinfo.value))
