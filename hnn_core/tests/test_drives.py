# Authors: Mainak Jas <mainakjas@gmail.com>
#          Christopher Bailey <bailey.cj@gmail.com>

import pytest
import os.path as op

import numpy as np

import hnn_core
from hnn_core import Params, Network, read_params
from hnn_core.drives import (drive_event_times, _get_prng, _create_extpois,
                             _create_bursty_input)
from hnn_core.params import create_pext
from hnn_core.network import pick_connection


def test_external_drive_times():
    """Test the different external drives."""

    params = Params()
    p_common, p_unique = create_pext(params,
                                     params['tstop'])

    # drive name must be valid and unambiguous
    p_bogus = {'prng_seedcore': 0}
    pytest.raises(ValueError, drive_event_times,
                  'invalid_drive', None, p_bogus, 0)
    pytest.raises(ValueError, drive_event_times,
                  'ev', None, p_bogus, 0)  # ambiguous

    # 'unique' external drives are always created
    for drive_type in ['extpois', 'extgauss']:
        event_times = drive_event_times(
            drive_type=drive_type,
            target_cell_type='L2_basket',
            params=p_unique[drive_type],
            gid=0)

    # but 'common' (rhythmic) drives are not
    for ii in range(len(p_common)):  # len == 0 for def. params
        event_times = drive_event_times(
            drive_type='common',
            target_cell_type=None,
            params=p_common[ii],
            gid=0)
        # parameters should lead to 0 input spikes for default params
        assert len(event_times) == 0
        # check that ei.p_ext matches params
        loc = p_common[ii]['loc'][:4]  # loc=prox or dist
        for layer in ['L2', 'L5']:
            key = 'input_{}_A_weight_{}Pyr_ampa'.format(loc, layer)
            assert p_common[ii][layer + 'Pyr_ampa'][0] == params[key]

    # validate poisson input time interval
    p_extpois = p_unique['extpois']
    p_extpois['L2_basket'] = (1., 1., 0., 0.)
    with pytest.raises(ValueError, match='The end time for Poisson input'):
        p_extpois['t_interval'] = (p_extpois['t_interval'][0], -1)
        event_times = drive_event_times(
            drive_type='extpois',
            target_cell_type='L2_basket',
            params=p_extpois, gid=0)
    with pytest.raises(ValueError, match='The start time for Poisson'):
        p_extpois['t_interval'] = (-1, 5)
        event_times = drive_event_times(
            drive_type='extpois',
            target_cell_type='L2_basket',
            params=p_extpois, gid=0)

    # checks the poisson spike train generation
    prng = np.random.RandomState()
    lamtha = 50.
    event_times = _create_extpois(t0=0, T=100000, lamtha=lamtha, prng=prng)
    event_intervals = np.diff(event_times)
    assert pytest.approx(event_intervals.mean(), abs=1.) == 1000 * 1 / lamtha

    with pytest.raises(ValueError, match='The start time for Poisson'):
        _create_extpois(t0=-5, T=5, lamtha=lamtha, prng=prng)
    with pytest.raises(ValueError, match='The end time for Poisson'):
        _create_extpois(t0=50, T=20, lamtha=lamtha, prng=prng)
    with pytest.raises(ValueError, match='Rate must be > 0'):
        _create_extpois(t0=0, T=1000, lamtha=-5, prng=prng)

    # check bursty/rhythmic input
    t0 = 0
    t0_stdev = 5
    tstop = 100
    f_input = 20.
    events_per_cycle = 3
    cycle_events_isi = 7
    events_jitter_std = 5.
    prng, prng2 = _get_prng(seed=0, gid=5, sync_evinput=False)
    event_times = _create_bursty_input(
        t0=t0, t0_stdev=t0_stdev, tstop=tstop,
        f_input=f_input, events_jitter_std=events_jitter_std,
        events_per_cycle=events_per_cycle, cycle_events_isi=cycle_events_isi,
        prng=prng, prng2=prng2)

    events_per_cycle = 5
    cycle_events_isi = 20
    with pytest.raises(ValueError,
                       match=r'Burst duration (?s).* cannot be greater than'):
        _create_bursty_input(t0=t0, t0_stdev=t0_stdev,
                             tstop=tstop, f_input=f_input,
                             events_jitter_std=events_jitter_std,
                             events_per_cycle=events_per_cycle,
                             cycle_events_isi=cycle_events_isi,
                             prng=prng, prng2=prng2)


def test_add_drives():
    """Test methods for adding drives to a Network."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    net = Network(params, legacy_mode=False)

    # Ensure weights and delays are updated
    weights_ampa = {'L2_basket': 1.0, 'L2_pyramidal': 3.0, 'L5_pyramidal': 4.0}
    syn_delays = {'L2_basket': 1.0, 'L2_pyramidal': 2.0, 'L5_pyramidal': 4.0}

    n_drive_cells = 10
    cell_specific = False  # default for bursty drive
    net.add_bursty_drive(
        'bursty', location='distal', burst_rate=10,
        weights_ampa=weights_ampa, synaptic_delays=syn_delays,
        n_drive_cells=n_drive_cells)

    assert net.external_drives['bursty']['n_drive_cells'] == n_drive_cells
    assert net.external_drives['bursty']['cell_specific'] == cell_specific
    conn_idxs = pick_connection(net, src_gids='bursty')
    for conn_idx in conn_idxs:
        drive_conn = net.connectivity[conn_idx]
        target_type = drive_conn['target_type']
        assert drive_conn['nc_dict']['A_weight'] == weights_ampa[target_type]
        assert drive_conn['nc_dict']['A_delay'] == syn_delays[target_type]

    n_drive_cells = 'n_cells'  # default for evoked drive
    cell_specific = True
    net.add_evoked_drive(
        'evoked_dist', mu=1.0, sigma=1.0, numspikes=1.0,
        weights_ampa=weights_ampa, location='distal',
        synaptic_delays=syn_delays, cell_specific=True)

    n_dist_targets = 235  # 270 with legacy mode
    assert (net.external_drives['evoked_dist']
                               ['n_drive_cells'] == n_dist_targets)
    assert net.external_drives['evoked_dist']['cell_specific'] == cell_specific
    conn_idxs = pick_connection(net, src_gids='evoked_dist')
    for conn_idx in conn_idxs:
        drive_conn = net.connectivity[conn_idx]
        target_type = drive_conn['target_type']
        assert drive_conn['nc_dict']['A_weight'] == weights_ampa[target_type]
        assert drive_conn['nc_dict']['A_delay'] == syn_delays[target_type]

    n_drive_cells = 'n_cells'  # default for poisson drive
    cell_specific = True
    net.add_poisson_drive(
        'poisson', rate_constant=1.0, weights_ampa=weights_ampa,
        location='distal', synaptic_delays=syn_delays,
        cell_specific=cell_specific)

    n_dist_targets = 235  # 270 with non-legacy mode
    assert (net.external_drives['poisson']
                               ['n_drive_cells'] == n_dist_targets)
    assert net.external_drives['poisson']['cell_specific'] == cell_specific
    conn_idxs = pick_connection(net, src_gids='poisson')
    for conn_idx in conn_idxs:
        drive_conn = net.connectivity[conn_idx]
        target_type = drive_conn['target_type']
        assert drive_conn['nc_dict']['A_weight'] == weights_ampa[target_type]
        assert drive_conn['nc_dict']['A_delay'] == syn_delays[target_type]

    # Test probabalistic drive connections.
    # drive with cell_specific=False
    n_drive_cells = 10
    probability = 0.5  # test that only half of possible connections are made
    weights_nmda = {'L2_basket': 1.0, 'L2_pyramidal': 3.0, 'L5_pyramidal': 4.0}
    net.add_bursty_drive(
        'bursty_prob', location='distal', burst_rate=10,
        weights_ampa=weights_ampa, weights_nmda=weights_nmda,
        synaptic_delays=syn_delays, n_drive_cells=n_drive_cells,
        probability=probability)

    for cell_type in weights_ampa.keys():
        conn_idxs = pick_connection(
            net, src_gids='bursty_prob', target_gids=cell_type)
        gid_pairs_comparison = net.connectivity[conn_idxs[0]]['gid_pairs']
        for conn_idx in conn_idxs:
            conn = net.connectivity[conn_idx]
            num_connections = np.sum(
                [len(gids) for gids in conn['gid_pairs'].values()])
            assert gid_pairs_comparison == conn['gid_pairs']
            assert num_connections == \
                np.around(len(net.gid_ranges[cell_type]) * n_drive_cells *
                          probability).astype(int)

    # drives with cell_specific=True
    probability = {'L2_basket': 0.1, 'L2_pyramidal': 0.25, 'L5_pyramidal': 0.5}
    net.add_evoked_drive(
        'evoked_prob', mu=1.0, sigma=1.0, numspikes=1.0,
        weights_ampa=weights_ampa, weights_nmda=weights_nmda,
        location='distal', synaptic_delays=syn_delays, cell_specific=True,
        probability=probability)

    for cell_type in weights_ampa.keys():
        conn_idxs = pick_connection(
            net, src_gids='evoked_prob', target_gids=cell_type)
        gid_pairs_comparison = net.connectivity[conn_idxs[0]]['gid_pairs']
        for conn_idx in conn_idxs:
            conn = net.connectivity[conn_idx]
            num_connections = np.sum(
                [len(gids) for gids in conn['gid_pairs'].values()])
            assert gid_pairs_comparison == conn['gid_pairs']
            assert num_connections == \
                np.around(len(net.gid_ranges[cell_type]) *
                          probability[cell_type]).astype(int)

    # evoked
    with pytest.raises(ValueError,
                       match='Standard deviation cannot be negative'):
        net.add_evoked_drive('evdist1', mu=10, sigma=-1, numspikes=1,
                             location='distal')
    with pytest.raises(ValueError,
                       match='Number of spikes must be greater than zero'):
        net.add_evoked_drive('evdist1', mu=10, sigma=1, numspikes=0,
                             location='distal')

    # Test Network._attach_drive()
    with pytest.raises(ValueError,
                       match=r'Allowed drive target locations are'):
        net.add_evoked_drive('evdist1', mu=10, sigma=1, numspikes=1,
                             location='bogus_location')
    with pytest.raises(ValueError,
                       match='Drive evoked_dist already defined'):
        net.add_evoked_drive('evoked_dist', mu=10, sigma=1, numspikes=1,
                             location='distal')
    with pytest.raises(ValueError,
                       match='No target cell types have been given a synaptic '
                       'weight'):
        net.add_evoked_drive('evdist1', mu=10, sigma=1, numspikes=1,
                             location='distal')
    with pytest.raises(ValueError,
                       match='When adding a distal drive, synaptic weight '
                       'cannot be defined for the L5_basket cell type'):
        net.add_evoked_drive('evdist1', mu=10, sigma=1, numspikes=1,
                             location='distal', weights_ampa={'L5_basket': 1.},
                             synaptic_delays={'L5_basket': .1})
    with pytest.raises(ValueError,
                       match='If cell_specific is True, n_drive_cells'):
        net.add_evoked_drive('evdist1', mu=10, sigma=1, numspikes=1,
                             location='distal', n_drive_cells=10,
                             cell_specific=True, weights_ampa=weights_ampa,
                             synaptic_delays=syn_delays)
    with pytest.raises(ValueError,
                       match='If cell_specific is False, n_drive_cells'):
        net.add_evoked_drive('evdist1', mu=10, sigma=1, numspikes=1,
                             location='distal', n_drive_cells='n_cells',
                             cell_specific=False, weights_ampa=weights_ampa,
                             synaptic_delays=syn_delays)
    with pytest.raises(ValueError,
                       match='Number of drive cells must be greater than 0'):
        net.add_evoked_drive('evdist1', mu=10, sigma=1, numspikes=1,
                             location='distal', n_drive_cells=0,
                             cell_specific=False, weights_ampa=weights_ampa,
                             synaptic_delays=syn_delays)

    # Poisson
    with pytest.raises(ValueError,
                       match='End time of Poisson drive cannot be negative'):
        net.add_poisson_drive('poisson1', tstart=0, tstop=-1,
                              location='distal', rate_constant=10.)
    with pytest.raises(ValueError,
                       match='Start time of Poisson drive cannot be negative'):
        net.add_poisson_drive('poisson1', tstart=-1,
                              location='distal', rate_constant=10.)
    with pytest.raises(ValueError,
                       match='Duration of Poisson drive cannot be negative'):
        net.add_poisson_drive('poisson1', tstart=10, tstop=1,
                              location='distal', rate_constant=10.)
    with pytest.raises(ValueError,
                       match='Rate constant must be positive'):
        net.add_poisson_drive('poisson1', location='distal',
                              rate_constant=0.,
                              weights_ampa=weights_ampa,
                              synaptic_delays=syn_delays)

    with pytest.raises(ValueError,
                       match='Rate constants not provided for all target'):
        net.add_poisson_drive('poisson1', location='distal',
                              rate_constant={'L2_pyramidal': 10.},
                              weights_ampa=weights_ampa,
                              synaptic_delays=syn_delays)
    with pytest.raises(ValueError,
                       match='Rate constant provided for unknown target cell'):
        net.add_poisson_drive('poisson1', location='distal',
                              rate_constant={'L2_pyramidal': 10.,
                                             'bogus_celltype': 20.},
                              weights_ampa={'L2_pyramidal': .01,
                                            'bogus_celltype': .01},
                              synaptic_delays=0.1)

    with pytest.raises(ValueError,
                       match='Drives specific to cell types are only '
                       'possible with cell_specific=True'):
        net.add_poisson_drive('poisson1', location='distal',
                              rate_constant={'L2_basket': 10.,
                                             'L2_pyramidal': 11.,
                                             'L5_basket': 12.,
                                             'L5_pyramidal': 13.},
                              n_drive_cells=1, cell_specific=False,
                              weights_ampa=weights_ampa,
                              synaptic_delays=syn_delays)

    # bursty
    with pytest.raises(ValueError,
                       match='End time of bursty drive cannot be negative'):
        net.add_bursty_drive('bursty_drive', tstop=-1,
                             location='distal', burst_rate=10)
    with pytest.raises(ValueError,
                       match='Start time of bursty drive cannot be negative'):
        net.add_bursty_drive('bursty_drive', tstart=-1,
                             location='distal', burst_rate=10)
    with pytest.raises(ValueError,
                       match='Duration of bursty drive cannot be negative'):
        net.add_bursty_drive('bursty_drive', tstart=10, tstop=1,
                             location='distal', burst_rate=10)

    msg = (r'Burst duration (?s).* cannot be greater than '
           'burst period')
    with pytest.raises(ValueError, match=msg):
        net.add_bursty_drive('bursty_drive', location='distal',
                             burst_rate=10, burst_std=20., numspikes=4,
                             spike_isi=50)

    # attaching drives
    with pytest.raises(ValueError,
                       match='Drive evoked_dist already defined'):
        net.add_poisson_drive('evoked_dist', location='distal',
                              rate_constant=10.,
                              weights_ampa=weights_ampa,
                              synaptic_delays=syn_delays)
    with pytest.raises(ValueError,
                       match='Allowed drive target locations are:'):
        net.add_poisson_drive('weird_poisson', location='inbetween',
                              rate_constant=10.,
                              weights_ampa=weights_ampa,
                              synaptic_delays=syn_delays)
    with pytest.raises(ValueError,
                       match='Allowed drive target cell types are:'):
        net.add_poisson_drive('cell_unknown', location='proximal',
                              rate_constant=10.,
                              weights_ampa={'CA1_pyramidal': 1.},
                              synaptic_delays=.01)
    with pytest.raises(ValueError,
                       match='synaptic_delays is either a common float or '
                       'needs to be specified as a dict for each of the cell'):
        net.add_poisson_drive('cell_unknown', location='proximal',
                              rate_constant=10.,
                              weights_ampa={'L2_pyramidal': 1.},
                              synaptic_delays={'L5_pyramidal': 1.})
    with pytest.raises(ValueError, match='probability must be'):
        net.add_bursty_drive(
            'cell_unknown', location='distal', burst_rate=10,
            weights_ampa={'L2_pyramidal': 1.},
            synaptic_delays={'L2_pyramidal': 1.}, probability=2.0)

    with pytest.raises(TypeError, match='probability must be'):
        net.add_bursty_drive(
            'cell_unknown2', location='distal', burst_rate=10,
            weights_ampa={'L2_pyramidal': 1.},
            synaptic_delays={'L2_pyramidal': 1.}, probability='1.0')

    with pytest.raises(ValueError, match='probability is either'):
        net.add_bursty_drive(
            'cell_unknown2', location='distal', burst_rate=10,
            weights_ampa={'L2_pyramidal': 1.},
            synaptic_delays={'L2_pyramidal': 1.},
            probability={'L5_pyramidal': 1.})

    with pytest.raises(TypeError, match='probability must be'):
        net.add_bursty_drive(
            'cell_unknown2', location='distal', burst_rate=10,
            weights_ampa={'L2_pyramidal': 1.},
            synaptic_delays={'L2_pyramidal': 1.},
            probability={'L2_pyramidal': '1.0'})

    with pytest.raises(ValueError, match='probability must be'):
        net.add_bursty_drive(
            'cell_unknown3', location='distal', burst_rate=10,
            weights_ampa={'L2_pyramidal': 1.},
            synaptic_delays={'L2_pyramidal': 1.},
            probability={'L2_pyramidal': 2.0})
