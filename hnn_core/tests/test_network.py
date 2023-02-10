# Authors: Mainak Jas <mainakjas@gmail.com>

from copy import deepcopy
from hnn_core.dipole import simulate_dipole
import os.path as op
import numpy as np
from numpy.testing import assert_allclose
import pytest

import hnn_core
from hnn_core import read_params, CellResponse, Network
from hnn_core import jones_2009_model, law_2021_model, calcium_model
from hnn_core.network_models import add_erp_drives_to_jones_model
from hnn_core.network_builder import NetworkBuilder
from hnn_core.network import pick_connection

hnn_core_root = op.dirname(hnn_core.__file__)
params_fname = op.join(hnn_core_root, 'param', 'default.json')


def test_network_models():
    """"Test instantiations of the network object"""
    # Make sure critical biophysics for Law model are updated
    net_law = law_2021_model()
    # instantiate drive events for NetworkBuilder
    net_law._instantiate_drives(tstop=net_law._params['tstop'],
                                n_trials=net_law._params['N_trials'])

    for cell_name in ['L5_pyramidal', 'L2_pyramidal']:
        assert net_law.cell_types[cell_name].synapses['gabab']['tau1'] == 45.0
        assert net_law.cell_types[cell_name].synapses['gabab']['tau2'] == 200.0

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
    # 15 drive connections are added as follows: evdist1: 3 ampa + 3 nmda,
    # evprox1: 4 ampa, evprox2: 4 ampa, and 1 extra zero-weighted ampa
    # evdist1->L5_basket connection is added to comply with legacy_mode
    assert len(net_default.connectivity) == n_conn + 15

    # Ensure distant dependent calcium gbar
    net_calcium = calcium_model()
    # instantiate drive events for NetworkBuilder
    net_calcium._instantiate_drives(tstop=net_calcium._params['tstop'],
                                    n_trials=net_calcium._params['N_trials'])
    network_builder = NetworkBuilder(net_calcium)
    gid = net_calcium.gid_ranges['L5_pyramidal'][0]
    for section_name, section in \
            network_builder._cells[gid]._nrn_sections.items():
        # Section endpoints where seg.x == 0.0 or 1.0 don't have 'ca' mech
        ca_gbar = [seg.__getattribute__('ca').gbar for
                   seg in list(section.allseg())[1:-1]]
        na_gbar = [seg.__getattribute__('hh2').gnabar for
                   seg in list(section.allseg())[1:-1]]
        k_gbar = [seg.__getattribute__('hh2').gkbar for
                  seg in list(section.allseg())[1:-1]]

        # Ensure positive distance dependent calcium gbar with plateau
        if section_name == 'apical_tuft':
            assert np.all(np.diff(ca_gbar) == 0)
        else:
            assert np.all(np.diff(ca_gbar) > 0)

        # Ensure negative distance dependent sodium gbar with plateau
        if section_name == 'apical_2':
            assert np.all(np.diff(na_gbar[0:3]) < 0)
            assert np.all(np.diff(na_gbar[3:]) == 0)
        elif section_name == 'apical_tuft':
            assert np.all(np.diff(na_gbar) == 0)
        else:
            assert np.all(np.diff(na_gbar) < 0)

        # Ensure negative exponential distance dependent K gbar
        assert np.all(np.diff(k_gbar) < 0)
        assert np.all(np.diff(k_gbar, n=2) > 0)  # positive 2nd derivative


def test_network_cell_positions():
    """"Test manipulation of cell positions in the network object"""

    net = jones_2009_model()
    assert np.isclose(net._inplane_distance, 1.)  # default
    assert np.isclose(net._layer_separation, 1307.4)  # default

    # change both from their default values
    net.set_cell_positions(inplane_distance=2.)
    assert np.isclose(net._layer_separation, 1307.4)  # still the default
    net.set_cell_positions(layer_separation=1000.)
    assert np.isclose(net._inplane_distance, 2.)  # mustn't change

    # check that in-plane distance is now 2. for the default 10 x 10 grid
    assert np.allclose(  # x-coordinate jumps every 10th gid
        np.diff(np.array(net.pos_dict['L5_pyramidal'])[9::10, 0], axis=0), 2.)
    assert np.allclose(  # test first 10 y-coordinates
        np.diff(np.array(net.pos_dict['L5_pyramidal'])[:9, 1], axis=0), 2.)

    # check that layer separation has changed (L5 is zero) tp 1000.
    assert np.isclose(net.pos_dict['L2_pyramidal'][0][2], 1000.)

    with pytest.raises(ValueError,
                       match='In-plane distance must be positive'):
        net.set_cell_positions(inplane_distance=0.)
    with pytest.raises(ValueError,
                       match='Layer separation must be positive'):
        net.set_cell_positions(layer_separation=0.)

    # Check that the origin of the drive cells matches the new 'origin'
    # when set_cell_positions is called after adding drives.
    # As the network dimensions increase, so does the center-of-mass of the
    # grid points, which is where all hnn drives should be located. The lamtha-
    # dependent weights and delays of the drives are calculated with respect to
    # this origin.
    add_erp_drives_to_jones_model(net)
    net.set_cell_positions(inplane_distance=20.)
    for drive_name, drive in net.external_drives.items():
        assert len(net.pos_dict[drive_name]) == drive['n_drive_cells']
        # just test the 0th index, assume all others then fine too
        for idx in range(3):  # x,y,z coords
            assert (net.pos_dict[drive_name][0][idx] ==
                    net.pos_dict['origin'][idx])


def test_network_drives():
    """Test manipulation of drives in the network object."""
    with pytest.raises(TypeError, match='params must be an instance of dict'):
        Network('hello')
    params = read_params(params_fname)
    # add rhythmic inputs (i.e., a type of common input)
    params.update({'input_dist_A_weight_L2Pyr_ampa': 1.4e-5,
                   'input_dist_A_weight_L5Pyr_ampa': 2.4e-5,
                   't0_input_dist': 50,
                   'input_prox_A_weight_L2Pyr_ampa': 3.4e-5,
                   'input_prox_A_weight_L5Pyr_ampa': 4.4e-5,
                   't0_input_prox': 50})

    net = jones_2009_model(deepcopy(params))

    # add all drives explicitly and ensure that the expected number of drive
    # cells get instantiated for each case
    n_drive_cells_list = list()

    '''
    weights_ampa_d1 = {'L2_basket': 0.006562,
                       'L2_pyramidal': 7e-6,
                       'L5_pyramidal': 0.142300}
    weights_nmda_d1 = {'L2_basket': 0.019482,
                       'L2_pyramidal': 0.004317,
                       'L5_pyramidal': 0.080074}
    synaptic_delays_d1 = {'L2_basket': 0.1,
                          'L2_pyramidal': 0.1,
                          'L5_pyramidal': 0.1}
    n_drive_cells = 'n_cells'
    n_drive_cells_list.append(n_drive_cells)
    net.add_evoked_drive(
        name='evdist1',
        mu=63.53,
        sigma=3.85,
        numspikes=1,
        location='distal',
        n_drive_cells=n_drive_cells,
        cell_specific=True,
        weights_ampa=weights_ampa_d1,
        weights_nmda=weights_nmda_d1,
        synaptic_delays=synaptic_delays_d1,
        event_seed=4)
    '''

    add_erp_drives_to_jones_model(net)
    n_drive_cells = 'n_cells'
    n_drive_cells_list.append(n_drive_cells)

    weights_ampa_d2 = {'L2_basket': 0.006562,
                       'L2_pyramidal': 7e-6,
                       'L5_basket': 0.01,
                       'L5_pyramidal': 0.142300}
    weights_nmda_d2 = {'L2_basket': 0.019482,
                       'L2_pyramidal': 0.004317,
                       'L5_basket': 0.001,
                       'L5_pyramidal': 0.080074}
    synaptic_delays_d2 = {'L2_basket': 0.1,
                          'L2_pyramidal': 0.1,
                          'L5_basket': 0.1,
                          'L5_pyramidal': 0.1}
    n_drive_cells = 10
    n_drive_cells_list.append(n_drive_cells)
    net.add_bursty_drive(
        name='bursty1',
        tstart=10.,
        tstart_std=0.5,
        tstop=30.,
        location='proximal',
        burst_rate=100.,
        burst_std=0.,
        numspikes=2,
        spike_isi=1.,
        n_drive_cells=n_drive_cells,
        cell_specific=False,
        weights_ampa=weights_ampa_d2,
        weights_nmda=weights_nmda_d2,
        synaptic_delays=synaptic_delays_d2,
        event_seed=4)

    # instantiate drive events for NetworkBuilder
    net._instantiate_drives(tstop=params['tstop'],
                            n_trials=params['N_trials'])
    network_builder = NetworkBuilder(net)  # needed to instantiate cells

    # Assert that params are conserved across Network initialization
    for p in params:
        assert params[p] == net._params[p]
    assert len(params) == len(net._params)
    print(network_builder)
    print(network_builder._cells[:2])

    # Assert that proper number/types of gids are created for Network drives
    dns_from_gids = [name for name in net.gid_ranges.keys() if
                     name not in net.cell_types]
    assert sorted(dns_from_gids) == sorted(net.external_drives.keys())
    for dn in dns_from_gids:
        n_drive_cells = net.external_drives[dn]['n_drive_cells']
        assert len(net.gid_ranges[dn]) == n_drive_cells

    # Check drive dict structure for each external drive
    for drive_idx, drive in enumerate(net.external_drives.values()):
        # Check that connectivity sources correspond to gid_ranges
        conn_idxs = pick_connection(net, src_gids=drive['name'])
        this_src_gids = set([gid for conn_idx in conn_idxs
                             for gid in net.connectivity[conn_idx]['src_gids']
                             ])  # NB set: globals
        assert sorted(this_src_gids) == list(net.gid_ranges[drive['name']])
        # Check type-specific dynamics and events
        n_drive_cells = drive['n_drive_cells']
        if n_drive_cells_list[drive_idx] != 'n_cells':
            assert n_drive_cells_list[drive_idx] == n_drive_cells
        assert len(drive['events']) == 1  # single trial simulated
        if drive['type'] == 'evoked':
            for kw in ['mu', 'sigma', 'numspikes']:
                assert kw in drive['dynamics'].keys()
            assert len(drive['events'][0]) == n_drive_cells
            # this also implicitly tests that events are always a list
            assert len(drive['events'][0][0]) == drive['dynamics']['numspikes']
        elif drive['type'] == 'gaussian':
            for kw in ['mu', 'sigma', 'numspikes']:
                assert kw in drive['dynamics'].keys()
            assert len(drive['events'][0]) == n_drive_cells
        elif drive['type'] == 'poisson':
            for kw in ['tstart', 'tstop', 'rate_constant']:
                assert kw in drive['dynamics'].keys()
            assert len(drive['events'][0]) == n_drive_cells
        elif drive['type'] == 'bursty':
            for kw in ['tstart', 'tstart_std', 'tstop',
                       'burst_rate', 'burst_std', 'numspikes']:
                assert kw in drive['dynamics'].keys()
            assert len(drive['events'][0]) == n_drive_cells
            n_events = (
                drive['dynamics']['numspikes'] *  # 2
                (1 + (drive['dynamics']['tstop'] -
                      drive['dynamics']['tstart'] - 1) //
                    (1000. / drive['dynamics']['burst_rate'])))
            assert len(drive['events'][0][0]) == n_events  # 4

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
        for target_type in target_weights[drive_name]:
            conn_idxs = pick_connection(net, src_gids=drive_name,
                                        target_gids=target_type,
                                        receptor='ampa')
            for conn_idx in conn_idxs:
                drive_conn = net.connectivity[conn_idx]
                assert_allclose(drive_conn['nc_dict']['A_weight'],
                                target_weights[drive_name][target_type],
                                rtol=1e-12)

    # check select synaptic delays
    target_delays = {'evdist1': {'L2_basket': 0.1, 'L5_pyramidal': 0.1},
                     'evprox1': {'L2_basket': 0.1, 'L5_pyramidal': 1.},
                     'evprox2': {'L2_basket': 0.1, 'L5_pyramidal': 1.}}
    for drive_name in target_delays:
        for target_type in target_delays[drive_name]:
            conn_idxs = pick_connection(net, src_gids=drive_name,
                                        target_gids=target_type,
                                        receptor='ampa')
            for conn_idx in conn_idxs:
                drive_conn = net.connectivity[conn_idx]
                assert_allclose(drive_conn['nc_dict']['A_delay'],
                                target_delays[drive_name][target_type],
                                rtol=1e-12)

    # array of simulation times is created in Network.__init__, but passed
    # to CellResponse-constructor for storage (Network is agnostic of time)
    with pytest.raises(TypeError,
                       match="'times' is an np.ndarray of simulation times"):
        _ = CellResponse(times='blah')

    # Assert that all external drives are initialized
    # Assumes legacy mode where cell-specific drives create artificial cells
    # for all network cells regardless of connectivity
    n_evoked_sources = 3 * net._n_cells
    n_pois_sources = net._n_cells
    n_gaus_sources = net._n_cells
    n_bursty_sources = (net.external_drives['bursty1']['n_drive_cells'] +
                        net.external_drives['bursty2']['n_drive_cells'])
    # test that expected number of external driving events are created
    assert len(network_builder._drive_cells) == (n_evoked_sources +
                                                 n_pois_sources +
                                                 n_gaus_sources +
                                                 n_bursty_sources)
    assert len(network_builder._gid_list) ==\
        len(network_builder._drive_cells) + net._n_cells
    # first 'evoked drive' comes after real cells and bursty drive cells
    assert network_builder._drive_cells[n_bursty_sources].gid ==\
        net._n_cells + n_bursty_sources

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
    n_bursty1_sources = net.external_drives['bursty1']['n_drive_cells']
    n_connections = n_bursty1_sources * 3 * n_pyr  # 3 synapses / cell
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


def test_network_connectivity():
    """Test manipulation of local network connectivity."""
    params = read_params(params_fname)
    net = Network(params, legacy_mode=False)

    # add some basic local network connectivity
    # layer2 Pyr -> layer2 Pyr
    # layer5 Pyr -> layer5 Pyr
    for target_cell in ['L2_pyramidal', 'L5_pyramidal']:
        for receptor in ['nmda', 'ampa']:
            net.add_connection(
                target_cell, target_cell, loc='proximal', receptor=receptor,
                weight=5e-4, delay=net.delay, lamtha=3.0, allow_autapses=False)
    # layer2 Basket -> layer2 Pyr
    # layer5 Basket -> layer5 Pyr
    for receptor in ['gabaa', 'gabab']:
        net.add_connection(
            src_gids='L2_basket', target_gids='L2_pyramidal', loc='soma',
            receptor=receptor, weight=5e-4, delay=net.delay, lamtha=50.0)
        net.add_connection(
            src_gids='L5_basket', target_gids='L2_pyramidal', loc='soma',
            receptor=receptor, weight=5e-4, delay=net.delay, lamtha=70.0)

    # add arbitrary drives that contribute artificial cells to network
    net.add_evoked_drive(name='evdist1', mu=5.0, sigma=1.0,
                         numspikes=1, location='distal',
                         weights_ampa={'L2_basket': 0.1,
                                       'L2_pyramidal': 0.1})
    net.add_evoked_drive(name='evprox1', mu=5.0, sigma=1.0,
                         numspikes=1, location='proximal',
                         weights_ampa={'L2_basket': 0.1,
                                       'L2_pyramidal': 0.1})

    # instantiate drive events and artificial cells for NetworkBuilder
    net._instantiate_drives(tstop=10.0,
                            n_trials=1)
    network_builder = NetworkBuilder(net)

    # get initial number of connections targeting a single section
    n_conn_prox = len(network_builder.ncs['L2Pyr_L2Pyr_ampa'])
    n_conn_trunk = len(network_builder.ncs['L2Pyr_L2Pyr_nmda'])

    # add connections targeting single section and rebuild
    kwargs_default = dict(src_gids=[35, 36], target_gids=[35, 36],
                          loc='proximal', receptor='ampa',
                          weight=5e-4, delay=1.0, lamtha=1e9,
                          probability=1.0)
    net.add_connection(**kwargs_default)  # smoke test
    kwargs_trunk = kwargs_default.copy()
    kwargs_trunk['loc'] = 'apical_trunk'
    kwargs_trunk['receptor'] = 'nmda'
    net.add_connection(**kwargs_trunk)
    network_builder = NetworkBuilder(net)

    # Check proximal targeted connection count increased by right number
    # (2*2 connections between cells, 3 sections in proximal target)
    assert len(network_builder.ncs['L2Pyr_L2Pyr_ampa']) == n_conn_prox + 4 * 3
    nc = network_builder.ncs['L2Pyr_L2Pyr_ampa'][-1]
    assert_allclose(nc.weight[0], kwargs_default['weight'])

    # Check apical_trunk targeted connection count increased by right number
    # (2*2 connections between cells, 1 section i.e. apical_turnk)
    assert len(network_builder.ncs['L2Pyr_L2Pyr_nmda']) == n_conn_trunk + 4
    nc = network_builder.ncs['L2Pyr_L2Pyr_nmda'][-1]
    assert_allclose(nc.weight[0], kwargs_trunk['weight'])
    # Check that exactly 4 apical_trunk connections appended
    for idx in range(1, 5):
        assert network_builder.ncs['L2Pyr_L2Pyr_nmda'][
            -idx].postseg().__str__() == 'L2Pyr_apical_trunk(0.5)'
    assert network_builder.ncs['L2Pyr_L2Pyr_nmda'][
        -5].postseg().__str__() == 'L2Pyr_basal_3(0.5)'

    kwargs_good = [
        ('src_gids', 0), ('src_gids', 'L2_pyramidal'), ('src_gids', range(2)),
        ('target_gids', 35), ('target_gids', range(2)),
        ('target_gids', 'L2_pyramidal'),
        ('target_gids', [[35, 36], [37, 38]]), ('probability', 0.5),
        ('loc', 'apical_trunk')]
    for arg, item in kwargs_good:
        kwargs = kwargs_default.copy()
        kwargs[arg] = item
        net.add_connection(**kwargs)

    kwargs_bad = [
        ('src_gids', 0.0), ('src_gids', [0.0]),
        ('target_gids', 35.0), ('target_gids', [35.0]),
        ('target_gids', [[35], [36.0]]), ('loc', 1.0),
        ('receptor', 1.0), ('weight', '1.0'), ('delay', '1.0'),
        ('lamtha', '1.0'), ('probability', '0.5'), ('allow_autapses', 1.0)]
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

    # Make sure warning raised if section targeted doesn't contain synapse
    match = ('Invalid value for')
    with pytest.raises(ValueError, match=match):
        kwargs = kwargs_default.copy()
        kwargs['target_gids'] = 'L5_pyramidal'
        kwargs['loc'] = 'soma'
        kwargs['receptor'] = 'ampa'
        net.add_connection(**kwargs)

    # Test net.pick_connection()
    kwargs_default = dict(net=net, src_gids=None, target_gids=None,
                          loc=None, receptor=None)

    kwargs_good = [
        ('src_gids', 0),
        ('src_gids', 'L2_pyramidal'),
        ('src_gids', range(2)),
        ('src_gids', None),
        ('target_gids', 35),
        ('target_gids', range(2)),
        ('target_gids', 'L2_pyramidal'),
        ('target_gids', None),
        ('loc', 'soma'),
        ('loc', None),
        ('receptor', 'gabaa'),
        ('receptor', None)]
    for arg, item in kwargs_good:
        kwargs = kwargs_default.copy()
        kwargs[arg] = item
        indices = pick_connection(**kwargs)
        for conn_idx in indices:
            if (arg == 'src_gids' or arg == 'target_gids') and \
                    isinstance(item, str):
                assert np.all(np.in1d(net.connectivity[conn_idx][arg],
                              net.gid_ranges[item]))
            elif item is None:
                pass
            else:
                assert np.any(np.in1d([item], net.connectivity[conn_idx][arg]))

    # Test searching a list of src or target types
    src_cell_type_list = ['L2_basket', 'L5_basket']
    true_gid_set = set(list(net.gid_ranges['L2_basket']) + list(
        net.gid_ranges['L5_basket']))
    indices = pick_connection(net, src_gids=src_cell_type_list)
    pick_gid_list = list()
    for conn_idx in indices:
        pick_gid_list.extend(
            net.connectivity[conn_idx]['src_gids'])
    assert true_gid_set == set(pick_gid_list)

    target_cell_type_list = ['L2_pyramidal', 'L5_pyramidal']
    true_gid_set = set(list(net.gid_ranges['L2_pyramidal']) + list(
        net.gid_ranges['L5_pyramidal']))
    indices = pick_connection(net, target_gids=target_cell_type_list)
    pick_gid_list = list()
    for conn_idx in indices:
        pick_gid_list.extend(
            net.connectivity[conn_idx]['target_gids'])
    assert true_gid_set == set(pick_gid_list)

    # Check that a given gid isn't present in any connection profile that
    # pick_connection can't identify
    conn_idxs = pick_connection(net, src_gids=0)
    for conn_idx in range(len(net.connectivity)):
        if conn_idx not in conn_idxs:
            assert 0 not in net.connectivity[conn_idx]['src_gids']

    # Check that pick_connection returns empty lists when searching for
    # a drive targetting the wrong location
    conn_idxs = pick_connection(net, src_gids='evdist1', loc='proximal')
    assert len(conn_idxs) == 0
    assert not pick_connection(net, src_gids='evprox1', loc='distal')

    # Check conditions where no connections match
    assert pick_connection(net, loc='distal', receptor='gabab') == list()
    assert pick_connection(
        net, src_gids='L2_pyramidal', receptor='gabab') == list()
    assert pick_connection(
        net, src_gids='L2_basket', target_gids='L2_basket',
        loc='proximal', receptor='nmda') == list()
    assert pick_connection(
        net, src_gids='L2_pyramidal', target_gids='L2_basket',
        loc='distal', receptor='gabab') == list()

    kwargs_bad = [
        ('src_gids', 0.0), ('src_gids', [0.0]),
        ('target_gids', 35.0), ('target_gids', [35.0]),
        ('target_gids', [35, [36.0]]), ('loc', 1.0),
        ('receptor', 1.0)]
    for arg, item in kwargs_bad:
        match = ('must be an instance of')
        with pytest.raises(TypeError, match=match):
            kwargs = kwargs_default.copy()
            kwargs[arg] = item
            pick_connection(**kwargs)

    kwargs_bad = [
        ('src_gids', -1), ('src_gids', [-1]),
        ('target_gids', -1), ('target_gids', [-1]),
        ('src_gids', [35, -1]), ('target_gids', [35, -1])]
    for arg, item in kwargs_bad:
        with pytest.raises(AssertionError):
            kwargs = kwargs_default.copy()
            kwargs[arg] = item
            pick_connection(**kwargs)

    for arg in ['src_gids', 'target_gids', 'loc', 'receptor']:
        string_arg = 'invalid_string'
        match = f"Invalid value for the '{arg}' parameter"
        with pytest.raises(ValueError, match=match):
            kwargs = kwargs_default.copy()
            kwargs[arg] = string_arg
            pick_connection(**kwargs)

    # Test removing connections from net.connectivity
    # Needs to be updated if number of drives change in preceeding tests
    net.clear_connectivity()
    assert len(net.connectivity) == 4  # 2 drives x 4 target cell types
    net.clear_drives()
    assert len(net.connectivity) == 0

    with pytest.warns(UserWarning, match='No connections'):
        simulate_dipole(net, tstop=10)


def test_add_cell_type():
    """Test adding a new cell type."""
    params = read_params(params_fname)
    net = jones_2009_model(params)
    # instantiate drive events for NetworkBuilder
    net._instantiate_drives(tstop=params['tstop'],
                            n_trials=params['N_trials'])

    n_total_cells = net._n_cells
    pos = [(0, idx, 0) for idx in range(10)]
    tau1 = 0.6

    new_cell = net.cell_types['L2_basket'].copy()
    net._add_cell_type('new_type', pos=pos, cell_template=new_cell)
    net.cell_types['new_type'].synapses['gabaa']['tau1'] = tau1

    n_new_type = len(net.gid_ranges['new_type'])
    assert n_new_type == len(pos)
    net.add_connection('L2_basket', 'new_type', loc='proximal',
                       receptor='gabaa', weight=8e-3, delay=1,
                       lamtha=2)

    network_builder = NetworkBuilder(net)
    assert net._n_cells == n_total_cells + len(pos)
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

    net = Network(params)
    # add arbitrary local network connection to avoid simulation warning
    net.add_connection(src_gids='L2_pyramidal',
                       target_gids='L2_basket',
                       loc='soma', receptor='ampa', weight=1e-3,
                       delay=1.0, lamtha=3.0)
    with pytest.raises(ValueError, match=r'cell_type must be one of .*$'):
        net.add_tonic_bias(cell_type='name_nonexistent', amplitude=1.0,
                           t0=0.0, tstop=4.0)

    with pytest.raises(ValueError, match='Duration of tonic input cannot be'
                       ' negative'):
        net.add_tonic_bias(cell_type='L2_pyramidal', amplitude=1.0,
                           t0=5.0, tstop=4.0)
        simulate_dipole(net, tstop=20.)
    net.external_biases = dict()

    with pytest.raises(ValueError, match='End time of tonic input cannot be'
                       ' negative'):
        net.add_tonic_bias(cell_type='L2_pyramidal', amplitude=1.0,
                           t0=5.0, tstop=-1.0)
        simulate_dipole(net, tstop=5.)

    with pytest.raises(ValueError, match='parameter may be missing'):
        params['Itonic_T_L2Pyr_soma'] = 5.0
        net = Network(params, add_drives_from_params=True)

    params.update({
        'N_pyr_x': 3, 'N_pyr_y': 3,
        'N_trials': 1,
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
    net = Network(params, add_drives_from_params=True)
    assert 'tonic' in net.external_biases
    assert 'L2_pyramidal' in net.external_biases['tonic']

    # new API
    net = Network(params)
    net.add_tonic_bias(cell_type='L2_pyramidal', amplitude=1.0)
    assert 'tonic' in net.external_biases
    assert 'L5_pyramidal' not in net.external_biases['tonic']
    assert net.external_biases['tonic']['L2_pyramidal']['t0'] == 0
    with pytest.raises(ValueError, match=r'Tonic bias already defined for.*$'):
        net.add_tonic_bias(cell_type='L2_pyramidal', amplitude=1.0)
