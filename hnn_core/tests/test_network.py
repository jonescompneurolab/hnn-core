# Authors: Mainak Jas <mainakjas@gmail.com>

from copy import deepcopy
import os.path as op
import numpy as np
from numpy.testing import assert_allclose
import pytest

import hnn_core
from hnn_core import read_params, CellResponse
from hnn_core import jones_2009_model, law_2021_model, calcium_model
from hnn_core.network_models import add_erp_drives_to_jones_model
from hnn_core.network_builder import NetworkBuilder

hnn_core_root = op.dirname(hnn_core.__file__)
params_fname = op.join(hnn_core_root, 'param', 'default.json')


def test_network_models():
    """"Test instantiations of the network object"""
    # Make sure critical biophysics for Law model are updated
    net_law = law_2021_model()
    for cell_name in ['L5_pyramidal', 'L2_pyramidal']:
        assert net_law.cell_types[cell_name].p_syn['gabab']['tau1'] == 45.0
        assert net_law.cell_types[cell_name].p_syn['gabab']['tau2'] == 200.0

    # Check add_default_erp()
    net_default = jones_2009_model()
    with pytest.raises(TypeError, match='net must be'):
        add_erp_drives_to_jones_model(net='invalid_input')
    with pytest.raises(TypeError, match='tstart must be'):
        add_erp_drives_to_jones_model(net=net_default,
                                      tstart='invalid_input')
    n_conn = len(net_default.connectivity)
    add_erp_drives_to_jones_model(net_default)
    for drive_name in ['evdist1', 'evprox1', 'evprox2']:
        assert drive_name in net_default.external_drives.keys()
    assert len(net_default.connectivity) == n_conn + 14

    # Ensure distant dependent calcium gbar
    net_calcium = calcium_model()
    network_builder = NetworkBuilder(net_calcium)
    gid = net_calcium.gid_ranges['L5_pyramidal'][0]
    for section_name, section in network_builder._cells[gid].sections.items():
        # Section endpoints where seg.x == 0.0 or 1.0 don't have 'ca' mech
        ca_gbar = [seg.__getattribute__('ca').gbar for
                   seg in list(section.allseg())[1:-1]]
        if section_name == 'apical_tuft':
            assert np.all(np.diff(ca_gbar) == 0)
        else:
            assert np.all(np.diff(ca_gbar) > 0)


def test_network():
    """Test network object."""
    params = read_params(params_fname)
    # add rhythmic inputs (i.e., a type of common input)
    params.update({'input_dist_A_weight_L2Pyr_ampa': 1.4e-5,
                   'input_dist_A_weight_L5Pyr_ampa': 2.4e-5,
                   't0_input_dist': 50,
                   'input_prox_A_weight_L2Pyr_ampa': 3.4e-5,
                   'input_prox_A_weight_L5Pyr_ampa': 4.4e-5,
                   't0_input_prox': 50})
    net = jones_2009_model(deepcopy(params), add_drives_from_params=True)
    network_builder = NetworkBuilder(net)  # needed to populate net.cells

    # Assert that params are conserved across Network initialization
    for p in params:
        assert params[p] == net._params[p]
    assert len(params) == len(net._params)
    print(network_builder)
    print(network_builder._cells[:2])

    # Assert that proper number of gids are created for Network drives
    dns_from_gids = [name for name in net.gid_ranges.keys() if
                     name not in net.cell_types]
    assert len(dns_from_gids) == len(net.external_drives)
    for dn in dns_from_gids:
        assert dn in net.external_drives.keys()
        this_src_gids = set([gid for drive_conn in
                             net.external_drives[dn]['conn'].values() for
                             gid in drive_conn['src_gids']])  # NB set: globals
        assert len(net.gid_ranges[dn]) == len(this_src_gids)
        assert len(net.external_drives[dn]['events']) == 1  # single trial!

    # Check src_gids match between drives/connections
    for conn in net.connectivity:
        src_type = conn['src_type']
        target_type = conn['target_type']
        if src_type not in net.cell_types.keys():
            assert set(conn['gid_pairs'].keys()) == set(
                net.external_drives[src_type]['conn'][target_type]['src_gids'])

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
            for kw in ['tstart', 'tstart_std', 'tstop',
                       'burst_rate', 'burst_std', 'numspikes', 'n_drive_cells']:
                assert kw in drive['dynamics'].keys()
            assert len(drive['events'][0]) == 1
            n_events = (
                drive['dynamics']['numspikes'] *  # 2
                drive['dynamics']['n_drive_cells'] *  # 10
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
                                  'L5_pyramidal': 0.684013},
                      'bursty1': {'L2_pyramidal': 0.000034,
                                  'L5_pyramidal': 0.000044},
                      'bursty2': {'L2_pyramidal': 0.000014,
                                  'L5_pyramidal': 0.000024}
                      }
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
    n_pyr = len(net.gid_ranges['L2_pyramidal'])
    n_basket = len(net.gid_ranges['L2_basket'])

    # Check basket-basket connection where allow_autapses=False
    assert 'L2Pyr_L2Pyr_nmda' in network_builder.ncs
    n_connections = 3 * (n_pyr ** 2 - n_pyr)  # 3 synapses / cell
    assert len(network_builder.ncs['L2Pyr_L2Pyr_nmda']) == n_connections
    nc = network_builder.ncs['L2Pyr_L2Pyr_nmda'][0]
    assert nc.threshold == params['threshold']

    # Check bursty drives which use cell_specific=False
    assert 'bursty1_L2Pyr_ampa' in network_builder.ncs
    n_connections = 3 * n_pyr  # 3 synapses / cell
    assert len(network_builder.ncs['bursty1_L2Pyr_ampa']) == n_connections
    nc = network_builder.ncs['bursty1_L2Pyr_ampa'][0]
    assert nc.threshold == params['threshold']

    # Check basket-basket connection where allow_autapses=True
    assert 'L2Basket_L2Basket_gabaa' in network_builder.ncs
    n_connections = n_basket ** 2  # 1 synapse / cell
    assert len(network_builder.ncs['L2Basket_L2Basket_gabaa']) == n_connections
    nc = network_builder.ncs['L2Basket_L2Basket_gabaa'][0]
    assert nc.threshold == params['threshold']

    # Check evoked drives which use cell_specific=True
    assert 'evdist1_L2Basket_nmda' in network_builder.ncs
    n_connections = n_basket  # 1 synapse / cell
    assert len(network_builder.ncs['evdist1_L2Basket_nmda']) == n_connections
    nc = network_builder.ncs['evdist1_L2Basket_nmda'][0]
    assert nc.threshold == params['threshold']

    # Test inputs for connectivity API
    net = jones_2009_model(deepcopy(params), add_drives_from_params=True)
    n_conn = len(network_builder.ncs['L2Basket_L2Pyr_gabaa'])
    kwargs_default = dict(src_gids=[0, 1], target_gids=[35, 36],
                          loc='soma', receptor='gabaa',
                          weight=5e-4, delay=1.0, lamtha=1e9,
                          probability=1.0)
    net.add_connection(**kwargs_default)  # smoke test
    network_builder = NetworkBuilder(net)
    assert len(network_builder.ncs['L2Basket_L2Pyr_gabaa']) == n_conn + 4
    nc = network_builder.ncs['L2Basket_L2Pyr_gabaa'][-1]
    assert_allclose(nc.weight[0], kwargs_default['weight'])

    kwargs_good = [
        ('src_gids', 0), ('src_gids', 'L2_pyramidal'), ('src_gids', range(2)),
        ('target_gids', 35), ('target_gids', range(2)),
        ('target_gids', 'L2_pyramidal'),
        ('target_gids', [[35, 36], [37, 38]]), ('probability', 0.5)]
    for arg, item in kwargs_good:
        kwargs = kwargs_default.copy()
        kwargs[arg] = item
        net.add_connection(**kwargs)

    kwargs_bad = [
        ('src_gids', 0.0), ('src_gids', [0.0]),
        ('target_gids', 35.0), ('target_gids', [35.0]),
        ('target_gids', [[35], [36.0]]), ('loc', 1.0),
        ('receptor', 1.0), ('weight', '1.0'), ('delay', '1.0'),
        ('lamtha', '1.0'), ('probability', '0.5')]
    for arg, item in kwargs_bad:
        match = ('must be an instance of')
        with pytest.raises(TypeError, match=match):
            kwargs = kwargs_default.copy()
            kwargs[arg] = item
            net.add_connection(**kwargs)

    kwargs_bad = [
        ('src_gids', -1), ('src_gids', [-1]),
        ('target_gids', -1), ('target_gids', [-1]),
        ('target_gids', [[35], [-1]]), ('target_gids', [[35]]),
        ('src_gids', [0, 100]), ('target_gids', [0, 100])]
    for arg, item in kwargs_bad:
        with pytest.raises(AssertionError):
            kwargs = kwargs_default.copy()
            kwargs[arg] = item
            net.add_connection(**kwargs)

    for arg in ['src_gids', 'target_gids', 'loc', 'receptor']:
        string_arg = 'invalid_string'
        match = f"Invalid value for the '{arg}' parameter"
        with pytest.raises(ValueError, match=match):
            kwargs = kwargs_default.copy()
            kwargs[arg] = string_arg
            net.add_connection(**kwargs)

    # Check probability=0.5 produces half as many connections as default
    net.add_connection(**kwargs_default)
    kwargs = kwargs_default.copy()
    kwargs['probability'] = 0.5
    net.add_connection(**kwargs)
    n_connections = np.sum(
        [len(t_gids) for
         t_gids in net.connectivity[-2]['gid_pairs'].values()])
    n_connections_new = np.sum(
        [len(t_gids) for
         t_gids in net.connectivity[-1]['gid_pairs'].values()])
    assert n_connections_new == np.round(n_connections * 0.5).astype(int)
    assert net.connectivity[-1]['probability'] == 0.5
    with pytest.raises(ValueError, match='probability must be'):
        kwargs = kwargs_default.copy()
        kwargs['probability'] = -1.0
        net.add_connection(**kwargs)

    # Test removing connections from net.connectivity
    # Needs to be updated if number of drives change in preceeding tests
    net.clear_connectivity()
    assert len(net.connectivity) == 18
    net.clear_drives()
    assert len(net.connectivity) == 0


def test_add_cell_type():
    """Test adding a new cell type."""
    params = read_params(params_fname)
    net = jones_2009_model(params)

    n_total_cells = net.n_cells
    pos = [(0, idx, 0) for idx in range(10)]
    tau1 = 0.6

    new_cell = net.cell_types['L2_basket'].copy()
    net._add_cell_type('new_type', pos=pos, cell_template=new_cell)
    net.cell_types['new_type'].p_syn['gabaa']['tau1'] = tau1

    n_new_type = len(net.gid_ranges['new_type'])
    assert n_new_type == len(pos)
    assert 'new_type' in net.cell_response._cell_type_names
    net.add_connection('L2_basket', 'new_type', loc='proximal',
                       receptor='gabaa', weight=8e-3, delay=1,
                       lamtha=2)

    network_builder = NetworkBuilder(net)
    assert net.n_cells == n_total_cells + len(pos)
    n_basket = len(net.gid_ranges['L2_basket'])
    n_connections = n_basket * n_new_type
    assert len(network_builder.ncs['L2Basket_new_type_gabaa']) == n_connections
    nc = network_builder.ncs['L2Basket_new_type_gabaa'][0]
    assert nc.syn().tau1 == tau1


def test_tonic_biases():
    """Test tonic biases."""
    hnn_core_root = op.dirname(hnn_core.__file__)

    # default params
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)

    net = hnn_core.Network(params, add_drives_from_params=True)
    with pytest.raises(ValueError, match=r'cell_type must be one of .*$'):
        net.add_tonic_bias(cell_type='name_nonexistent', amplitude=1.0,
                           t0=0.0, tstop=4.0)

    with pytest.raises(ValueError, match='Duration of tonic input cannot be'
                       ' negative'):
        net.add_tonic_bias(cell_type='L2_pyramidal', amplitude=1.0,
                           t0=5.0, tstop=4.0)

    with pytest.raises(ValueError, match='End time of tonic input cannot be'
                       ' negative'):
        net.add_tonic_bias(cell_type='L2_pyramidal', amplitude=1.0,
                           t0=5.0, tstop=-1.)

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
        'tonic']['L2_pyramidal']['tstop'] == net._params['tstop']
    with pytest.raises(ValueError, match=r'Tonic bias already defined for.*$'):
        net.add_tonic_bias(cell_type='L2_pyramidal', amplitude=1.0)
