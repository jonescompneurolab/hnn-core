# Authors: Mainak Jas <mainakjas@gmail.com>

from copy import deepcopy
import os.path as op
from glob import glob
import numpy as np
from numpy.testing import assert_allclose
import pytest

import hnn_core
from hnn_core import read_params, Network, CellResponse, read_spikes
from hnn_core.network_builder import NetworkBuilder


def test_network(tmpdir):
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
    assert len(net.gid_ranges['common']) == 2
    assert len(net.gid_ranges['extgauss']) == net.n_cells
    assert len(net.gid_ranges['extpois']) == net.n_cells
    for ev_input in params['t_ev*']:
        type_key = ev_input[2: -2] + ev_input[-1]
        assert len(net.gid_ranges[type_key]) == net.n_cells

    # Assert that an empty CellResponse object is created as an attribute
    cell_response = []
    for cell_type, gid_range in net.gid_ranges.items():
        if cell_type in net.cellname_list:
            for gid in gid_range:
                cell_response.append(CellResponse(gid=gid,
                                                  cell_type=cell_type))
    assert net.cell_response == cell_response

    # array of simulation times is created in Network.__init__, but passed
    # to CellResponse-constructor for storage (Network is agnostic of time)
    with pytest.raises(TypeError,
                       match="'times' is an np.ndarray of simulation times"):
        _ = CellResponse(times=[1, 2, 3])

    # Assert that all external feeds are initialized
    n_evoked_sources = net.n_cells * 3
    n_pois_sources = net.n_cells
    n_gaus_sources = net.n_cells
    n_common_sources = 2

    # test that expected number of external driving events are created, and
    # make sure the PRNGs are consistent.
    assert isinstance(net.feed_times, dict)
    # single trial simulated
    assert all(len(src_feed_times) == 1 for
               src_type, src_feed_times in net.feed_times.items()
               if src_type != 'tonic')
    assert len(net.feed_times['common'][0]) == n_common_sources
    assert len(net.feed_times['common'][0][0]) == 40  # 40 spikes
    assert isinstance(net.feed_times['evprox1'][0][0], list)
    assert len(net.feed_times['evprox1'][0]) == net.n_cells
    assert_allclose(net.feed_times['evprox1'][0][0],
                    [23.80641637082997], rtol=1e-12)

    assert len(network_builder._feed_cells) == (n_evoked_sources +
                                                n_pois_sources +
                                                n_gaus_sources +
                                                n_common_sources)
    assert len(network_builder._gid_list) ==\
        len(network_builder._feed_cells) + net.n_cells
    # first 'evoked feed' comes after real cells and common inputs
    assert network_builder._feed_cells[2].gid == net.n_cells + n_common_sources

    # Assert that netcons are created properly
    # proximal
    assert 'L2Pyr_L2Pyr_nmda' in network_builder.ncs
    n_pyr = len(net.gid_ranges['L2_pyramidal'])
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
    n_conn = len(net.gid_ranges['common']) * len(net.gid_ranges['L5_basket'])
    assert len(network_builder.ncs['common_L5Basket_gabaa']) == n_conn

    # try unique=True
    network_builder._connect_celltypes(
        'extgauss', 'L5Basket', 'soma', 'gabaa', nc_dict,
        unique=True)
    n_conn = len(net.gid_ranges['L5_basket'])
    assert len(network_builder.ncs['extgauss_L5Basket_gabaa']) == n_conn

    # Test Network plotting functions
    net._spike_times = [[2.3456, 7.89], [4.2812, 93.2]]
    net._spike_gids = [[1, 3], [5, 7]]
    net._spike_types = [['L2_pyramidal', 'L2_basket'],
                        ['L5_pyramidal', 'L5_basket']]
    tstart, tstop = 0.1, 98.4
    gid_ranges = {'L2_pyramidal': range(0, 2), 'L2_basket': range(2, 4),
                  'L5_pyramidal': range(4, 6), 'L5_basket': range(6, 8)}

    net.plot_spikes_hist(show=False)

    cell_response = []
    for cell_type, gid_range in gid_ranges.items():
        for gid in gid_range:
            cell_response.append(CellResponse(gid=gid, cell_type=cell_type))

    for trial in range(len(net._spike_times)):
        for cell_resp in cell_response:
            gid = cell_resp.gid
            gid_mask = np.array(net._spike_gids[trial]) == gid
            gid_spike_times = np.array(
                net._spike_times)[trial][gid_mask].tolist()
            cell_response[gid].spike_times.append(gid_spike_times)

    net.cell_response = cell_response

    net.write(tmpdir.join('spk_%d.txt'))
    cell_response_read = read_spikes(tmpdir.join('spk_*.txt'), gid_ranges)
    assert ("CellResponse | 2 simulation trials" in repr(cell_response[0]))
    assert net.cell_response == cell_response_read

    with pytest.raises(TypeError, match="spike_types should be str, "
                                        "list, dict, or None"):
        net.plot_spikes_hist(spike_types=1, show=False)

    with pytest.raises(TypeError, match=r"spike_types\[ev\] must be a list\. "
                                        r"Got int\."):
        net.plot_spikes_hist(spike_types={'ev': 1}, show=False)

    with pytest.raises(ValueError, match=r"Elements of spike_types must map to"
                       r" mutually exclusive input types\. L2_basket is found"
                       r" more than once\."):
        net.plot_spikes_hist(spike_types={'ev':
                             ['L2_basket', 'L2_b']}, show=False)

    with pytest.raises(ValueError, match="No input types found for ABC"):
        net.plot_spikes_hist(spike_types='ABC', show=False)

    with pytest.raises(ValueError, match="tstart and tstop must be of type "
                       "int or float"):
        net.mean_rates(tstart=0.1, tstop='ABC',
                       gid_ranges=gid_ranges)

    with pytest.raises(ValueError, match="tstop must be greater than tstart"):
        net.mean_rates(tstart=0.1, tstop=-1.0, gid_ranges=gid_ranges)

    with pytest.raises(ValueError, match="Invalid mean_type. Valid "
                       "arguments include 'all', 'trial', or 'cell'."):
        net.mean_rates(tstart=tstart, tstop=tstop,
                       gid_ranges=gid_ranges, mean_type='ABC')

    test_rate = (1 / (tstop - tstart)) * 1000

    assert net.mean_rates(tstart, tstop, gid_ranges) == {
        'L5_pyramidal': test_rate / 4,
        'L5_basket': test_rate / 4,
        'L2_pyramidal': test_rate / 4,
        'L2_basket': test_rate / 4}
    assert net.mean_rates(tstart, tstop, gid_ranges,
                          mean_type='trial') == {
        'L5_pyramidal': [0.0, test_rate / 2],
        'L5_basket': [0.0, test_rate / 2],
        'L2_pyramidal': [test_rate / 2, 0.0],
        'L2_basket': [test_rate / 2, 0.0]}
    assert net.mean_rates(tstart, tstop, gid_ranges,
                          mean_type='cell') == {
        'L5_pyramidal': [[0.0, 0.0], [0.0, test_rate]],
        'L5_basket': [[0.0, 0.0], [0.0, test_rate]],
        'L2_pyramidal': [[0.0, test_rate], [0.0, 0.0]],
        'L2_basket': [[0.0, test_rate], [0.0, 0.0]]}

    # Write spike file with no 'types' column
    # Check for gid_ranges errors
    for fname in sorted(glob(str(tmpdir.join('spk_*.txt')))):
        times_gids_only = np.loadtxt(fname, dtype=str)[:, (0, 1)]
        np.savetxt(fname, times_gids_only, delimiter='\t', fmt='%s')
    with pytest.raises(ValueError, match="gid_ranges must be provided if "
                       "spike types are unspecified in the file "):
        cell_response = read_spikes(tmpdir.join('spk_*.txt'))
    with pytest.raises(ValueError, match="gid_ranges should contain only "
                       "disjoint sets of gid values"):
        gid_ranges = {'L2_pyramidal': range(3), 'L2_basket': range(2, 4),
                      'L5_pyramidal': range(4, 6), 'L5_basket': range(6, 8)}
        cell_response = read_spikes(tmpdir.join('spk_*.txt'),
                                    gid_ranges=gid_ranges)


def test_cell_response(tmpdir):
    """Test CellResponse object."""

    # Round-trip test
    gid = 1
    cell_type = 'L2_pyramidal'

    with pytest.raises(TypeError,
                       match="spike_times should be a list of lists"):
        CellResponse(spike_times=([2.3456, 7.89], [4.2812, 93.2]),
                     gid=gid, cell_type=cell_type)

    with pytest.raises(TypeError,
                       match="spike_times should be a list of lists"):
        CellResponse(spike_times=[1, 2], gid=gid, cell_type=cell_type)
