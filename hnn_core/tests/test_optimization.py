# Authors: Mainak Jas <mainakjas@gmail.com>

import numpy as np

from hnn_core.optimization import (_consolidate_chunks, _split_by_evinput,
                                   _generate_weights)


def test_consolidate_chunks():
    """Test consolidation of chunks."""
    inputs = {
        'ev1': {
            'start': 5,
            'end': 25,
            'ranges': {'initial': 1e-10, 'minval': 1e-11, 'maxval': 1e-9},
            'opt_end': 90,
            'weights': np.array([5., 10.])
        },
        'ev2': {
            'start': 100,
            'end': 120,
            'ranges': {'initial': 1e-10, 'minval': 1e-11, 'maxval': 1e-9},
            'opt_end': 170,
            'weights': np.array([10., 5.])
        }
    }
    chunks = _consolidate_chunks(inputs)
    assert len(chunks) == len(inputs) + 1  # extra last chunk??
    assert chunks[-1]['opt_end'] == inputs['ev2']['opt_end']
    assert chunks[-1]['inputs'] == ['ev1', 'ev2']
    assert isinstance(chunks, list)

    # overlapping chunks
    inputs['ev1']['end'] = 110
    chunks = _consolidate_chunks(inputs)
    assert len(chunks) == 1
    assert chunks[0]['start'] == inputs['ev1']['start']
    assert chunks[0]['end'] == inputs['ev2']['end']
    assert np.allclose(chunks[0]['weights'],
                       (inputs['ev1']['weights'] +
                        inputs['ev2']['weights']) / 2.)


def test_split_by_evinput():
    """Test splitting evoked input."""
    drive_names = ['ev_drive_1', 'ev_drive_2']
    drive_dynamics = [{'mu': 5., 'sigma': .1}, {'mu': 10., 'sigma': .2}]
    drive_syn_weights = [{'ampa_L2_pyramidal': 1.}, {'nmda_L5_basket': 2.}]
    tstop = 20.
    dt = 0.025

    timing_range_multiplier = 3.0
    sigma_range_multiplier = 50.0
    synweight_range_multiplier = 500.0
    decay_multiplier = 1.6
    evinput_params = _split_by_evinput(drive_names, drive_dynamics,
                                       drive_syn_weights, tstop,
                                       sigma_range_multiplier,
                                       timing_range_multiplier,
                                       synweight_range_multiplier)
    assert list(evinput_params.keys()) == drive_names
    for evinput in evinput_params.values():
        assert list(evinput.keys()) == ['mean', 'sigma', 'ranges',
                                        'start', 'end']

    evinput_params = _generate_weights(evinput_params, tstop, dt,
                                       decay_multiplier)
    for evinput in evinput_params.values():
        assert list(evinput.keys()) == ['ranges', 'start', 'end',
                                        'weights', 'opt_start',
                                        'opt_end']
