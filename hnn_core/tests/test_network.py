# Authors: Mainak Jas <mainakjas@gmail.com>

from copy import deepcopy
import os.path as op
from glob import glob
import numpy as np
import pytest

import hnn_core
from hnn_core import read_params, Network, Spikes, read_spikes, simulate_dipole
from hnn_core.neuron import NeuronNetwork


def test_network():
    """Test network object."""
    hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    # add rhythmic inputs (i.e., a type of common input)
    params.update({'input_dist_A_weight_L2Pyr_ampa': 5.4e-5,
                   'input_dist_A_weight_L5Pyr_ampa': 5.4e-5,
                   't0_input_dist': 50,
                   'input_prox_A_weight_L2Pyr_ampa': 5.4e-5,
                   'input_prox_A_weight_L5Pyr_ampa': 5.4e-5,
                   't0_input_prox': 50})
    net = Network(deepcopy(params))
    neuron_network = NeuronNetwork(net)  # needed to populate net.cells

    # Assert that params are conserved across Network initialization
    for p in params:
        assert params[p] == net.params[p]
    assert len(params) == len(net.params)
    print(neuron_network)
    print(neuron_network.cells[:2])

    # Assert that proper number of gids are created for Network inputs
    assert len(net.gid_dict['common']) == 2
    assert len(net.gid_dict['extgauss']) == net.n_cells
    assert len(net.gid_dict['extpois']) == net.n_cells
    for ev_input in params['t_ev*']:
        type_key = ev_input[2: -2] + ev_input[-1]
        assert len(net.gid_dict[type_key]) == net.n_cells

    # Assert that an empty Spikes object is created as an attribute
    assert net.spikes == Spikes()

    # Assert that all external feeds are initialized
    net._create_all_spike_sources()
    n_evoked_sources = 270 * 3
    n_pois_sources = 270
    n_gaus_sources = 270
    n_common_sources = 2
    assert len(net._feed_cells) == (n_evoked_sources +
                                    n_pois_sources +
                                    n_gaus_sources +
                                    n_common_sources)


def test_spikes():
    """Test spikes object."""

    # Round-trip test
    spiketimes = [[2.3456, 7.89], [4.2812, 93.2]]
    spikegids = [[1, 3], [5, 7]]
    spiketypes = [['L2_pyramidal', 'L2_basket'], ['L5_pyramidal', 'L5_basket']]
    spikes = Spikes(times=spiketimes, gids=spikegids, types=spiketypes)
    spikes.write('/tmp/spk_%d.txt')
    assert spikes == read_spikes('/tmp/spk_*.txt')
    assert ("Spikes | 2 simulation trials" in repr(spikes))

    with pytest.raises(TypeError, match="times should be a list of lists"):
        spikes = Spikes(times=([2.3456, 7.89], [4.2812, 93.2]), gids=spikegids,
                        types=spiketypes)

    with pytest.raises(TypeError, match="times should be a list of lists"):
        spikes = Spikes(times=[1, 2], gids=spikegids, types=spiketypes)

    with pytest.raises(ValueError, match="times, gids, and types should be "
                       "lists of the same length"):
        spikes = Spikes(times=[[2.3456, 7.89]], gids=spikegids,
                        types=spiketypes)

    # Write spike file with no 'types' column
    # Check for gid_dict errors
    for fname in sorted(glob('/tmp/spk_*.txt')):
        times_gids_only = np.loadtxt(fname, dtype=str)[:, (0, 1)]
        np.savetxt(fname, times_gids_only, delimiter='\t', fmt='%s')
    with pytest.raises(ValueError, match="gid_dict must be provided if spike "
                       "types are unspecified in the file /tmp/spk_0.txt"):
        spikes = read_spikes('/tmp/spk_*.txt')
    with pytest.raises(ValueError, match="gid_dict should contain only "
                       "disjoint sets of gid values"):
        gid_dict = {'L2_pyramidal': range(3), 'L2_basket': range(2, 4),
                    'L5_pyramidal': range(4, 6), 'L5_basket': range(6, 8)}
        spikes = read_spikes('/tmp/spk_*.txt', gid_dict=gid_dict)


def test_plots():
    """Tests plots."""

    hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    net = Network(params)
    simulate_dipole(net)
    net.plot_input()
    net.spikes.plot()
