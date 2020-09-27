# Authors: Mainak Jas <mainakjas@gmail.com>

from copy import deepcopy
import os.path as op
from glob import glob
import numpy as np
import pytest

import hnn_core
from hnn_core import read_params, Network, Spikes, read_spikes
from hnn_core.network_builder import NetworkBuilder


def test_network():
    """Test network object."""
    hnn_core_root = op.dirname(hnn_core.__file__)
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
    network_builder = NetworkBuilder(net)  # needed to populate net.cells

    # Assert that params are conserved across Network initialization
    for p in params:
        assert params[p] == net.params[p]
    assert len(params) == len(net.params)
    print(network_builder)
    print(network_builder.cells[:2])

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
    n_evoked_sources = 270 * 3
    n_pois_sources = 270
    n_gaus_sources = 270
    n_common_sources = 2
    assert len(network_builder._feed_cells) == (n_evoked_sources +
                                                n_pois_sources +
                                                n_gaus_sources +
                                                n_common_sources)

    # Assert that netcons are created properly
    # proximal
    assert 'L2Pyr_L2Pyr_nmda' in network_builder.ncs
    n_pyr = len(net.gid_dict['L2_pyramidal'])
    n_connections = 3 * (n_pyr ** 2 - n_pyr)  # 3 synapses / cell
    assert len(network_builder.ncs['L2Pyr_L2Pyr_nmda']) == n_connections
    nc = network_builder.ncs['L2Pyr_L2Pyr_nmda'][0]
    assert nc.threshold == params['threshold']

    # create a new connection between cell types
    nc_dict = {'A_delay': 1, 'A_weight': 1e-5, 'lamtha': 20,
               'threshold': 0.5}
    network_builder._connect_celltypes(
        'common', 'L5Basket', 'soma', 'gabaa', nc_dict,
        unique=False)
    assert 'common_L5Basket_gabaa' in network_builder.ncs
    n_conn = len(net.gid_dict['common']) * len(net.gid_dict['L5_basket'])
    assert len(network_builder.ncs['common_L5Basket_gabaa']) == n_conn

    # try unique=True
    network_builder._connect_celltypes(
        'extgauss', 'L5Basket', 'soma', 'gabaa', nc_dict,
        unique=True)
    n_conn = len(net.gid_dict['L5_basket'])
    assert len(network_builder.ncs['extgauss_L5Basket_gabaa']) == n_conn


def test_spikes(tmpdir):
    """Test spikes object."""

    # Round-trip test
    spiketimes = [[2.3456, 7.89], [4.2812, 93.2]]
    spikegids = [[1, 3], [5, 7]]
    spiketypes = [['L2_pyramidal', 'L2_basket'], ['L5_pyramidal', 'L5_basket']]
    tstart, tstop = 0.1, 98.4
    gid_dict = {'L2_pyramidal': range(1, 2), 'L2_basket': range(3, 4),
                'L5_pyramidal': range(5, 6), 'L5_basket': range(7, 8)}
    spikes = Spikes(times=spiketimes, gids=spikegids, types=spiketypes)
    spikes.plot_hist(show=False)
    spikes.write(tmpdir.join('spk_%d.txt'))
    assert spikes == read_spikes(tmpdir.join('spk_*.txt'))

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

    spikes = Spikes(times=spiketimes, gids=spikegids, types=spiketypes)
    
    with pytest.raises(TypeError, match="indices must be integers or slices, "
                       "not str"):
        spikes['1']

    with pytest.raises(TypeError, match="spike_types should be str, "
                                        "list, dict, or None"):
        spikes.plot_hist(spike_types=1, show=False)

    with pytest.raises(TypeError, match=r"spike_types\[ev\] must be a list\. "
                                        r"Got int\."):
        spikes.plot_hist(spike_types={'ev': 1}, show=False)

    with pytest.raises(ValueError, match=r"Elements of spike_types must map to"
                       r" mutually exclusive input types\. L2_basket is found"
                       r" more than once\."):
        spikes.plot_hist(spike_types={'ev': ['L2_basket', 'L2_b']}, show=False)

    with pytest.raises(ValueError, match="No input types found for ABC"):
        spikes.plot_hist(spike_types='ABC', show=False)

    with pytest.raises(ValueError, match="tstart and tstop must be of type "
                       "int or float"):
        spikes.mean_rates(tstart=0.1, tstop='ABC', gid_dict=gid_dict)

    with pytest.raises(ValueError, match="tstop must be greater than tstart"):
        spikes.mean_rates(tstart=0.1, tstop=-1.0, gid_dict=gid_dict)

    with pytest.raises(ValueError, match="Invalid mean_type. Valid "
                       "arguments include 'all', 'trial', or 'cell'."):
        spikes.mean_rates(tstart=tstart, tstop=tstop, gid_dict=gid_dict,
                          mean_type='ABC')

    test_rate = (1 / (tstop - tstart)) * 1000

    assert spikes.mean_rates(tstart, tstop, gid_dict) == {
        'L5_pyramidal': test_rate / 2,
        'L5_basket': test_rate / 2,
        'L2_pyramidal': test_rate / 2,
        'L2_basket': test_rate / 2}
    assert spikes.mean_rates(tstart, tstop, gid_dict, mean_type='trial') == {
        'L5_pyramidal': [0.0, test_rate],
        'L5_basket': [0.0, test_rate],
        'L2_pyramidal': [test_rate, 0.0],
        'L2_basket': [test_rate, 0.0]}
    assert spikes.mean_rates(tstart, tstop, gid_dict, mean_type='cell') == {
        'L5_pyramidal': [[0.0], [test_rate]],
        'L5_basket': [[0.0], [test_rate]],
        'L2_pyramidal': [[test_rate], [0.0]],
        'L2_basket': [[test_rate], [0.0]]}

    # Write spike file with no 'types' column
    # Check for gid_dict errors
    for fname in sorted(glob(str(tmpdir.join('spk_*.txt')))):
        times_gids_only = np.loadtxt(fname, dtype=str)[:, (0, 1)]
        np.savetxt(fname, times_gids_only, delimiter='\t', fmt='%s')
    with pytest.raises(ValueError, match="gid_dict must be provided if spike "
                       "types are unspecified in the file "):
        spikes = read_spikes(tmpdir.join('spk_*.txt'))
    with pytest.raises(ValueError, match="gid_dict should contain only "
                       "disjoint sets of gid values"):
        gid_dict = {'L2_pyramidal': range(3), 'L2_basket': range(2, 4),
                    'L5_pyramidal': range(4, 6), 'L5_basket': range(6, 8)}
        spikes = read_spikes(tmpdir.join('spk_*.txt'), gid_dict=gid_dict)
