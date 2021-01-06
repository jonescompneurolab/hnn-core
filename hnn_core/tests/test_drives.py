# Authors: Mainak Jas <mainakjas@gmail.com>
#          Christopher Bailey <bailey.cj@gmail.com>

import pytest

import os.path as op

import hnn_core
from hnn_core import Network, read_params


def test_add_drives():
    """Test methods for adding drives to a Network."""
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)
    net = Network(params)
    net.add_evoked_drive('early_distal', mu=10, sigma=1, numspikes=1,
                         location='distal')

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
        net.add_poisson_drive('tonic_drive', t0=0, T=-1,
                              location='distal',
                              rate_constants={'L2_pyramidal': 10.})
    with pytest.raises(ValueError,
                       match='Start time of Poisson drive cannot be negative'):
        net.add_poisson_drive('tonic_drive', t0=-1, T=params['tstop'],
                              location='distal',
                              rate_constants={'L2_pyramidal': 10.})
    with pytest.raises(ValueError,
                       match='Duration of Poisson drive cannot be negative'):
        net.add_poisson_drive('tonic_drive', t0=10, T=1,
                              location='distal',
                              rate_constants={'L2_pyramidal': 10.})
    with pytest.raises(ValueError,
                       match='End time of Poisson drive cannot exceed'):
        net.add_poisson_drive('tonic_drive', t0=0, T=params['tstop'] + 1,
                              location='distal',
                              rate_constants={'L2_pyramidal': 10.})
    with pytest.raises(ValueError,
                       match='rate_constants must be a dict of floats'):
        net.add_poisson_drive('tonic_drive', t0=0, T=None, location='distal',
                              rate_constants=10.)
    with pytest.raises(ValueError,
                       match='Rate constants should be defined for'):
        net.add_poisson_drive('tonic_drive', t0=0, T=params['tstop'],
                              location='distal',
                              rate_constants={'L2_pyramidal': 10.,
                                              'bogus_celltype': 20.})
    # bursty
    with pytest.raises(ValueError,
                       match='End time of bursty drive cannot be negative'):
        net.add_bursty_drive('bursty_drive', t0=0, T=-1,
                             location='distal', distribution='normal',
                             burst_f=10, spike_jitter_std=20.,
                             numspikes=2, spike_isi=10, repeats=10)
    with pytest.raises(ValueError,
                       match='Start time of bursty drive cannot be negative'):
        net.add_bursty_drive('bursty_drive', t0=-1, T=params['tstop'],
                             location='distal', distribution='normal',
                             burst_f=10, spike_jitter_std=20.,
                             numspikes=2, spike_isi=10, repeats=10)
    with pytest.raises(ValueError,
                       match='Duration of bursty drive cannot be negative'):
        net.add_bursty_drive('bursty_drive', t0=10, T=1,
                             location='distal', distribution='normal',
                             burst_f=10, spike_jitter_std=20.,
                             numspikes=2, spike_isi=10, repeats=10)
    with pytest.raises(ValueError,
                       match='End time of bursty drive cannot exceed'):
        net.add_bursty_drive('bursty_drive', t0=0, T=params['tstop'] + 1,
                             location='distal', distribution='normal',
                             burst_f=10, spike_jitter_std=20.,
                             numspikes=2, spike_isi=10, repeats=10)

    with pytest.raises(ValueError,
                       match='Burst duration cannot be greater than period'):
        net.add_bursty_drive('bursty_drive', t0=0, T=params['tstop'],
                             location='distal', distribution='normal',
                             burst_f=10, spike_jitter_std=20.,
                             numspikes=4, spike_isi=50, repeats=10)
