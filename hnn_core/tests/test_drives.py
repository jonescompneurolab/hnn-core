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
    net = Network(params, legacy_mode=False)
    net.add_evoked_drive('early_distal', mu=10, sigma=1, numspikes=1,
                         location='distal')

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

    with pytest.raises(ValueError,
                       match='Burst duration cannot be greater than period'):
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
