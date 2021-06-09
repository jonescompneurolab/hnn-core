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

    # check "common" input
    t0 = 0
    t0_stdev = 5
    tstop = 100
    f_input = 20.
    events_per_cycle = 3
    cycle_events_isi = 7
    events_jitter_std = 5.
    repeats = 2
    prng, prng2 = _get_prng(seed=0, gid=5, sync_evinput=False)
    event_times = _create_bursty_input(
        t0=t0, t0_stdev=t0_stdev, tstop=tstop,
        f_input=f_input, events_jitter_std=events_jitter_std,
        events_per_cycle=events_per_cycle, cycle_events_isi=cycle_events_isi,
        repeats=repeats, prng=prng, prng2=prng2)

    events_per_cycle = 5
    cycle_events_isi = 20
    with pytest.raises(ValueError,
                       match=r'Burst duration (?s).* cannot be greater than'):
        _create_bursty_input(t0=t0, t0_stdev=t0_stdev,
                             tstop=tstop, f_input=f_input,
                             events_jitter_std=events_jitter_std,
                             events_per_cycle=events_per_cycle,
                             cycle_events_isi=cycle_events_isi,
                             repeats=repeats, prng=prng, prng2=prng2)


def test_add_drives():
    """Test methods for adding drives to a Network."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    net = Network(params, legacy_mode=False)
    net.add_evoked_drive('early_distal', mu=10, sigma=1, numspikes=1,
                         location='distal')

    # Ensure weights and delays are updated
    weights_ampa = {'L2_basket': 1.0, 'L2_pyramidal': 3.0,
                    'L5_basket': 2.0, 'L5_pyramidal': 4.0}
    syn_delays = {'L2_basket': 1.0, 'L2_pyramidal': 2.0,
                  'L5_basket': 3.0, 'L5_pyramidal': 4.0}
    net.add_bursty_drive(
        'bursty', location='distal', burst_rate=10,
        weights_ampa=weights_ampa, synaptic_delays=syn_delays)

    for type_name, drive in net.external_drives['bursty']['conn'].items():
        assert drive['ampa']['A_weight'] == weights_ampa[type_name]
        assert drive['ampa']['A_delay'] == syn_delays[type_name]

    net.add_evoked_drive(
        'evoked', mu=1.0, sigma=1.0, numspikes=1.0, weights_ampa=weights_ampa,
        location='distal', synaptic_delays=syn_delays)

    for type_name, drive in net.external_drives['evoked']['conn'].items():
        assert drive['ampa']['A_weight'] == weights_ampa[type_name]
        assert drive['ampa']['A_delay'] == syn_delays[type_name]

    net.add_poisson_drive(
        'poisson', rate_constant=1.0, weights_ampa=weights_ampa,
        location='distal', synaptic_delays=syn_delays)

    for type_name, drive in net.external_drives['poisson']['conn'].items():
        assert drive['ampa']['A_weight'] == weights_ampa[type_name]
        assert drive['ampa']['A_delay'] == syn_delays[type_name]

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
                       match='Drive early_distal already defined'):
        net.add_evoked_drive('early_distal', mu=10, sigma=1, numspikes=1,
                             location='distal')

    # Poisson
    with pytest.raises(ValueError,
                       match='End time of Poisson drive cannot be negative'):
        net.add_poisson_drive('tonic_drive', tstart=0, tstop=-1,
                              location='distal', rate_constant=10.)
    with pytest.raises(ValueError,
                       match='Start time of Poisson drive cannot be negative'):
        net.add_poisson_drive('tonic_drive', tstart=-1,
                              location='distal', rate_constant=10.)
    with pytest.raises(ValueError,
                       match='Duration of Poisson drive cannot be negative'):
        net.add_poisson_drive('tonic_drive', tstart=10, tstop=1,
                              location='distal', rate_constant=10.)
    with pytest.raises(ValueError,
                       match='End time of Poisson drive cannot exceed'):
        net.add_poisson_drive('tonic_drive', tstop=params['tstop'] + 1,
                              location='distal', rate_constant=10.)
    with pytest.raises(ValueError,
                       match='Rate constant must be positive'):
        net.add_poisson_drive('tonic_drive', location='distal',
                              rate_constant=0.)

    with pytest.raises(ValueError,
                       match='Rate constants not provided for all target'):
        net.add_poisson_drive('tonic_drive', location='distal',
                              rate_constant={'L2_pyramidal': 10.},
                              weights_ampa={'L5_pyramidal': .01})
    with pytest.raises(ValueError,
                       match='Rate constant provided for unknown target cell'):
        net.add_poisson_drive('tonic_drive', location='distal',
                              rate_constant={'L2_pyramidal': 10.,
                                             'bogus_celltype': 20.})
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
    with pytest.raises(ValueError,
                       match='End time of bursty drive cannot exceed'):
        net.add_bursty_drive('bursty_drive', tstop=params['tstop'] + 1,
                             location='distal', burst_rate=10)

    msg = (r'Burst duration (?s).* cannot be greater than '
           'burst period')
    with pytest.raises(ValueError, match=msg):
        net.add_bursty_drive('bursty_drive', location='distal',
                             burst_rate=10, burst_std=20., numspikes=4,
                             spike_isi=50)

    # attaching drives
    with pytest.raises(ValueError,
                       match='Drive early_distal already defined'):
        net.add_poisson_drive('early_distal', location='distal',
                              rate_constant=10.)
    with pytest.raises(ValueError,
                       match='Allowed drive target locations are:'):
        net.add_poisson_drive('weird_poisson', location='inbetween',
                              rate_constant=10.)
    with pytest.raises(ValueError,
                       match='Allowed drive target cell types are:'):
        net.add_poisson_drive('cell_unknown', location='proximal',
                              rate_constant=10.,
                              weights_ampa={'CA1_pyramidal': 1.})
    with pytest.raises(ValueError,
                       match='synaptic_delays is either a common float or '
                             'needs to be specified as a dict for each cell'):
        net.add_poisson_drive('cell_unknown', location='proximal',
                              rate_constant=10.,
                              weights_ampa={'L2_pyramidal': 1.},
                              synaptic_delays={'L5_pyramidal': 1.})
