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
    net = Network(deepcopy(params), add_drives_from_params=True)
    network_builder = NetworkBuilder(net)  # needed to populate net.cells

    # Assert that params are conserved across Network initialization
    for p in params:
        assert params[p] == net.params[p]
    assert len(params) == len(net.params)
    print(network_builder)
    print(network_builder.cells[:2])

    # Assert that proper number of gids are created for Network drives
    dns_from_gids = [name for name in net.gid_ranges.keys() if
                     name not in net.cellname_list]
    assert len(dns_from_gids) == len(net.external_drives)
    for dn in dns_from_gids:
        assert dn in net.external_drives.keys()
        this_src_gids = set([gid for drive_conn in
                             net.external_drives[dn]['conn'].values() for
                             gid in drive_conn['src_gids']])  # NB set: globals
        assert len(net.gid_ranges[dn]) == len(this_src_gids)
        assert len(net.external_drives[dn]['events']) == 1  # single trial!

    assert len(net.gid_ranges['bursty1']) == 1
    for drive in net.external_drives.values():
        assert len(drive['events']) == 1  # single trial simulated
        if drive['type'] == 'evoked':
            for kw in ['mu', 'sigma', 'numspikes']:
                assert kw in drive['dynamics'].keys()
            assert len(drive['events'][0]) == net.n_cells
            # this also implicitly tests that events are always a list
            assert len(drive['events'][0][0]) == drive['dynamics']['numspikes']
        elif drive['type'] == 'gaussian':
            for kw in ['mu', 'sigma', 'numspikes']:
                assert kw in drive['dynamics'].keys()
            assert len(drive['events'][0]) == net.n_cells
        elif drive['type'] == 'poisson':
            for kw in ['tstart', 'tstop', 'rate_constant']:
                assert kw in drive['dynamics'].keys()
            assert len(drive['events'][0]) == net.n_cells

        elif drive['type'] == 'bursty':
            for kw in ['distribution', 'tstart', 'tstart_std', 'tstop',
                       'burst_rate', 'burst_std', 'numspikes', 'repeats']:
                assert kw in drive['dynamics'].keys()
            assert len(drive['events'][0]) == 1
            n_events = (
                drive['dynamics']['numspikes'] *  # 2
                drive['dynamics']['repeats'] *  # 10
                (1 + (drive['dynamics']['tstop'] -
                      drive['dynamics']['tstart'] - 1) //
                    (1000. / drive['dynamics']['burst_rate'])))
            assert len(drive['events'][0][0]) == n_events  # 40

    # make sure the PRNGs are consistent.
    target_times = {'evdist1': [66.30498327062551, 61.54362532343694],
                    'evprox1': [23.80641637082997, 30.857310915553647],
                    'evprox2': [141.76252038319825, 137.73942375578602]}
    for drive_name in target_times:
        for idx in [0, -1]:  # first and last
            assert_allclose(net.external_drives[drive_name]['events'][0][idx],
                            target_times[drive_name][idx], rtol=1e-12)

    # check select AMPA weights
    target_weights = {'evdist1': {'L2_basket': 0.006562,
                                  'L5_pyramidal': 0.142300},
                      'evprox1': {'L2_basket': 0.08831,
                                  'L5_pyramidal': 0.00865},
                      'evprox2': {'L2_basket': 0.000003,
                                  'L5_pyramidal': 0.684013}}
    for drive_name in target_weights:
        for cellname in target_weights[drive_name]:
            assert_allclose(
                net.external_drives[
                    drive_name]['conn'][cellname]['ampa']['A_weight'],
                target_weights[drive_name][cellname], rtol=1e-12)

    # check select synaptic delays
    target_delays = {'evdist1': {'L2_basket': 0.1, 'L5_pyramidal': 0.1},
                     'evprox1': {'L2_basket': 0.1, 'L5_pyramidal': 1.},
                     'evprox2': {'L2_basket': 0.1, 'L5_pyramidal': 1.}}
    for drive_name in target_delays:
        for cellname in target_delays[drive_name]:
            assert_allclose(
                net.external_drives[
                    drive_name]['conn'][cellname]['ampa']['A_delay'],
                target_delays[drive_name][cellname], rtol=1e-12)

    # Assert that an empty CellResponse object is created as an attribute
    assert net.cell_response == CellResponse()
    # array of simulation times is created in Network.__init__, but passed
    # to CellResponse-constructor for storage (Network is agnostic of time)
    with pytest.raises(TypeError,
                       match="'times' is an np.ndarray of simulation times"):
        _ = CellResponse(times=[1, 2, 3])

    # Assert that all external drives are initialized
    n_evoked_sources = net.n_cells * 3
    n_pois_sources = net.n_cells
    n_gaus_sources = net.n_cells
    n_bursty_sources = 2

    # test that expected number of external driving events are created
    assert len(network_builder._drive_cells) == (n_evoked_sources +
                                                 n_pois_sources +
                                                 n_gaus_sources +
                                                 n_bursty_sources)
    assert len(network_builder._gid_list) ==\
        len(network_builder._drive_cells) + net.n_cells
    # first 'evoked drive' comes after real cells and common inputs
    assert network_builder._drive_cells[2].gid ==\
        net.n_cells + n_bursty_sources

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
        'bursty1', 'L5Basket', 'soma', 'gabaa', nc_dict,
        unique=False)
    assert 'bursty1_L5Basket_gabaa' in network_builder.ncs
    n_conn = len(net.gid_ranges['bursty1']) * len(net.gid_ranges['L5_basket'])
    assert len(network_builder.ncs['bursty1_L5Basket_gabaa']) == n_conn

    # try unique=True
    network_builder._connect_celltypes(
        'extgauss', 'L5Basket', 'soma', 'gabaa', nc_dict,
        unique=True)
    n_conn = len(net.gid_ranges['L5_basket'])
    assert len(network_builder.ncs['extgauss_L5Basket_gabaa']) == n_conn


def test_tonic_biases():
    """Test tonic biases."""
    hnn_core_root = op.dirname(hnn_core.__file__)

    # default params
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)

    net = hnn_core.Network(params, add_drives_from_params=True)
    with pytest.raises(ValueError, match=r'cell_type must be one of .*$'):
        net.add_tonic_bias(cell_type='name_nonexistent', amplitude=1.0,
                           t0=0.0, T=4.0)

    with pytest.raises(ValueError, match='Duration of tonic input cannot be'
                       ' negative'):
        net.add_tonic_bias(cell_type='L2_pyramidal', amplitude=1.0,
                           t0=5.0, T=4.0)

    with pytest.raises(ValueError, match='End time of tonic input cannot be'
                       ' negative'):
        net.add_tonic_bias(cell_type='L2_pyramidal', amplitude=1.0,
                           t0=5.0, T=-1.)

    with pytest.raises(ValueError, match='parameter may be missing'):
        params['Itonic_T_L2Pyr_soma'] = 5.0
        net = hnn_core.Network(params, add_drives_from_params=True)

    params.update({
        'N_pyr_x': 3, 'N_pyr_y': 3,
        'tstop': 25, 'N_trials': 1,
        'dipole_smooth_win': 5,
        't_evprox_1': 5,
        't_evdist_1': 10,
        't_evprox_2': 20,
        # tonic inputs
        'Itonic_A_L2Pyr_soma': 1.0,
        'Itonic_t0_L2Pyr_soma': 5.0,
        'Itonic_T_L2Pyr_soma': 15.0
    })
    # old API
    net = hnn_core.Network(params, add_drives_from_params=True)
    assert 'tonic' in net.external_biases
    assert 'L2_pyramidal' in net.external_biases['tonic']

    # new API
    net = hnn_core.Network(params)
    net.add_tonic_bias(cell_type='L2_pyramidal', amplitude=1.0)
    assert 'tonic' in net.external_biases
    assert 'L5_pyramidal' not in net.external_biases['tonic']
    assert net.external_biases['tonic']['L2_pyramidal']['t0'] == 0
    assert net.external_biases[
        'tonic']['L2_pyramidal']['T'] == net.params['tstop']
    with pytest.raises(ValueError, match=r'Tonic bias already defined for.*$'):
        net.add_tonic_bias(cell_type='L2_pyramidal', amplitude=1.0)


def test_cell_response(tmpdir):
    """Test CellResponse object."""

    # Round-trip test
    spike_times = [[2.3456, 7.89], [4.2812, 93.2]]
    spike_gids = [[1, 3], [5, 7]]
    spike_types = [['L2_pyramidal', 'L2_basket'],
                   ['L5_pyramidal', 'L5_basket']]
    tstart, tstop = 0.1, 98.4
    gid_ranges = {'L2_pyramidal': range(1, 2), 'L2_basket': range(3, 4),
                  'L5_pyramidal': range(5, 6), 'L5_basket': range(7, 8)}
    cell_response = CellResponse(spike_times=spike_times,
                                 spike_gids=spike_gids,
                                 spike_types=spike_types)
    cell_response.plot_spikes_hist(show=False)
    cell_response.write(tmpdir.join('spk_%d.txt'))
    assert cell_response == read_spikes(tmpdir.join('spk_*.txt'))

    assert ("CellResponse | 2 simulation trials" in repr(cell_response))

    # Test recovery of empty spike files
    empty_spike = CellResponse(spike_times=[[], []], spike_gids=[[], []],
                               spike_types=[[], []])
    empty_spike.write(tmpdir.join('empty_spk_%d.txt'))
    assert empty_spike == read_spikes(tmpdir.join('empty_spk_*.txt'))

    assert ("CellResponse | 2 simulation trials" in repr(empty_spike))

    with pytest.raises(TypeError,
                       match="spike_times should be a list of lists"):
        cell_response = CellResponse(spike_times=([2.3456, 7.89],
                                     [4.2812, 93.2]),
                                     spike_gids=spike_gids,
                                     spike_types=spike_types)

    with pytest.raises(TypeError,
                       match="spike_times should be a list of lists"):
        cell_response = CellResponse(spike_times=[1, 2], spike_gids=spike_gids,
                                     spike_types=spike_types)

    with pytest.raises(ValueError, match="spike times, gids, and types should "
                       "be lists of the same length"):
        cell_response = CellResponse(spike_times=[[2.3456, 7.89]],
                                     spike_gids=spike_gids,
                                     spike_types=spike_types)

    cell_response = CellResponse(spike_times=spike_times,
                                 spike_gids=spike_gids,
                                 spike_types=spike_types)

    with pytest.raises(TypeError, match="indices must be int, slice, or "
                       "array-like, not str"):
        cell_response['1']

    with pytest.raises(TypeError, match="indices must be int, slice, or "
                       "array-like, not float"):
        cell_response[1.0]

    with pytest.raises(ValueError, match="ndarray cannot exceed 1 dimension"):
        cell_response[np.array([[1, 2], [3, 4]])]

    with pytest.raises(TypeError, match="gids must be of dtype int, "
                       "not float64"):
        cell_response[np.array([1, 2, 3.0])]

    with pytest.raises(TypeError, match="gids must be of dtype int, "
                       "not float64"):
        cell_response[[0, 1, 2, 2.0]]

    with pytest.raises(TypeError, match="spike_types should be str, "
                                        "list, dict, or None"):
        cell_response.plot_spikes_hist(spike_types=1, show=False)

    with pytest.raises(TypeError, match=r"spike_types\[ev\] must be a list\. "
                                        r"Got int\."):
        cell_response.plot_spikes_hist(spike_types={'ev': 1}, show=False)

    with pytest.raises(ValueError, match=r"Elements of spike_types must map to"
                       r" mutually exclusive input types\. L2_basket is found"
                       r" more than once\."):
        cell_response.plot_spikes_hist(spike_types={'ev':
                                       ['L2_basket', 'L2_b']},
                                       show=False)

    with pytest.raises(ValueError, match="No input types found for ABC"):
        cell_response.plot_spikes_hist(spike_types='ABC', show=False)

    with pytest.raises(ValueError, match="tstart and tstop must be of type "
                       "int or float"):
        cell_response.mean_rates(tstart=0.1, tstop='ABC',
                                 gid_ranges=gid_ranges)

    with pytest.raises(ValueError, match="tstop must be greater than tstart"):
        cell_response.mean_rates(tstart=0.1, tstop=-1.0, gid_ranges=gid_ranges)

    with pytest.raises(ValueError, match="Invalid mean_type. Valid "
                       "arguments include 'all', 'trial', or 'cell'."):
        cell_response.mean_rates(tstart=tstart, tstop=tstop,
                                 gid_ranges=gid_ranges, mean_type='ABC')

    test_rate = (1 / (tstop - tstart)) * 1000

    assert cell_response.mean_rates(tstart, tstop, gid_ranges) == {
        'L5_pyramidal': test_rate / 2,
        'L5_basket': test_rate / 2,
        'L2_pyramidal': test_rate / 2,
        'L2_basket': test_rate / 2}
    assert cell_response.mean_rates(tstart, tstop, gid_ranges,
                                    mean_type='trial') == {
        'L5_pyramidal': [0.0, test_rate],
        'L5_basket': [0.0, test_rate],
        'L2_pyramidal': [test_rate, 0.0],
        'L2_basket': [test_rate, 0.0]}
    assert cell_response.mean_rates(tstart, tstop, gid_ranges,
                                    mean_type='cell') == {
        'L5_pyramidal': [[0.0], [test_rate]],
        'L5_basket': [[0.0], [test_rate]],
        'L2_pyramidal': [[test_rate], [0.0]],
        'L2_basket': [[test_rate], [0.0]]}

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
