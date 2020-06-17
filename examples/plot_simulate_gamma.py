"""
======================
Simulate gamma rhythms
======================

This example demonstrates how to simulate gamma rhythms using hnn-core.

Replicates: https://jonescompneurolab.github.io/hnn-tutorials/gamma/gamma
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import os.path as op

###############################################################################
# Let us import hnn_core

import hnn_core
from hnn_core import simulate_dipole, read_params, Network

hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')

###############################################################################
# Then we read the parameters file
params_fname = op.join(hnn_core_root, 'param', 'gamma_L5weak_L2weak.json')
params = read_params(params_fname)
params.update({'prng_seedcore_extpois': -3})
print(params)

###############################################################################
# Now let's simulate the dipole
# You can simulate multiple trials in parallel by using n_jobs > 1

net = Network(params)
dpls = simulate_dipole(net, n_jobs=1, n_trials=1)

###############################################################################
# We can plot the time-frequency response using MNE
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_array_multitaper

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
dpls[0].plot(ax=axes[0], layer='agg', show=False)

sfreq = 1000. / params['dt']
time_bandwidth = 4.0
freqs = np.arange(20., 100., 1.)
n_cycles = freqs / 4.

# MNE expects an array of shape (n_trials, n_channels, n_times)
data = dpls[0].dpl['agg'][None, None, :]
power = tfr_array_multitaper(data, sfreq=sfreq, freqs=freqs,
                             n_cycles=n_cycles,
                             time_bandwidth=time_bandwidth,
                             output='power')
# stop = params['tstop'] + params['dt'] so last point is included
times = np.arange(0, params['tstop'] + params['dt'], params['dt'])
axes[1].pcolormesh(times, freqs, power[0, 0, ...], cmap='RdBu_r')
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Frequency (Hz)')
plt.xlim((0, params['tstop']))
plt.show()
