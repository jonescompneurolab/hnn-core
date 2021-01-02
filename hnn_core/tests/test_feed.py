# Authors: Mainak Jas <mainakjas@gmail.com>
#          Christopher Bailey <bailey.cj@gmail.com>

import os.path as op
import pytest

import numpy as np

import hnn_core
from hnn_core import Params, read_params
from hnn_core.feed import (feed_event_times, _get_prng, _create_extpois,
                           _create_bursty_input)
from hnn_core.params import create_pext


def test_extfeed():
    """Test the different external feeds."""

    params = Params()
    p_common, p_unique = create_pext(params,
                                     params['tstop'])

    # feed name must be valid and unambiguous
    p_bogus = {'prng_seedcore': 0}
    pytest.raises(ValueError, feed_event_times,
                  'invalid_feed', None, p_bogus, 0)
    pytest.raises(ValueError, feed_event_times,
                  'ev', None, p_bogus, 0)  # ambiguous

    # 'unique' external feeds are always created
    for feed_type in ['extpois', 'extgauss']:
        event_times = feed_event_times(
            feed_type=feed_type,
            target_cell_type='L2_basket',
            params=p_unique[feed_type],
            gid=0)

    # but 'common' (rhythmic) feeds are not
    for ii in range(len(p_common)):  # len == 0 for def. params
        event_times = feed_event_times(
            feed_type='common',
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
        event_times = feed_event_times(
            feed_type='extpois',
            target_cell_type='L2_basket',
            params=p_extpois, gid=0)
    with pytest.raises(ValueError, match='The start time for Poisson'):
        p_extpois['t_interval'] = (-1, 5)
        event_times = feed_event_times(
            feed_type='extpois',
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
    f_input = 100.
    events_jitter_std = 5.
    repeats = 2
    prng, prng2 = _get_prng(seed=0, gid=5, sync_evinput=False)
    with pytest.raises(ValueError, match='distribution not recognized'):
        _create_bursty_input(distribution='blah', t0=t0, t0_stdev=t0_stdev,
                             tstop=tstop, f_input=f_input,
                             events_jitter_std=events_jitter_std,
                             repeats=repeats, prng=prng, prng2=prng2)
    event_times = _create_bursty_input(
        distribution='normal', t0=t0, t0_stdev=t0_stdev, tstop=tstop,
        f_input=f_input, events_jitter_std=events_jitter_std,
        repeats=repeats, prng=prng, prng2=prng2)


def test_tonic_inputs():
    """Test tonic inputs."""
    hnn_core_root = op.dirname(hnn_core.__file__)

    # default params
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)

    net = hnn_core.Network(params, init_drives_from_params=True)
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
        net = hnn_core.Network(params, init_drives_from_params=True)

    params.update({
        'N_pyr_x': 3, 'N_pyr_y': 3,
        'tstop': 25, 'N_trials': 1,
        't_evprox_1': 5,
        't_evdist_1': 10,
        't_evprox_2': 20,
        # tonic inputs
        'Itonic_A_L2Pyr_soma': 1.0,
        'Itonic_t0_L2Pyr_soma': 5.0,
        'Itonic_T_L2Pyr_soma': 15.0
    })
    net = hnn_core.Network(params, init_drives_from_params=True)
    # smoke test for tonic inputs
    hnn_core.simulate_dipole(net)
    assert 'tonic' in net.feed_times
    assert 'L2Pyr' in net.feed_times['tonic']
