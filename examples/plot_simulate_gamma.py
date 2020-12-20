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

hnn_core_root = op.dirname(hnn_core.__file__)

###############################################################################
# Then we read the parameters file
params_fname = op.join(hnn_core_root, 'param', 'gamma_L5weak_L2weak.json')
params = read_params(params_fname)
print(params)

###############################################################################
# Now let's simulate the dipole

net = Network(params)
dpls = simulate_dipole(net, n_trials=1)

###############################################################################
# The network requires some time to reach steady state. Hence, we omit the
# first 50 ms in our time-frequency analysis.

tstart = 50
mask = dpls[0].times > tstart
times = dpls[0].times[mask]
data = dpls[0].data['agg'][mask]

###############################################################################
# We can plot the time-frequency response using MNE
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_array_morlet

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
axes[0].plot(times, data)

sfreq = 1000. / params['dt']
freqs = np.arange(20., 100., 1.)
n_cycles = freqs / 8.

# MNE expects an array of shape (n_trials, n_channels, n_times)
data = data[None, None, :]
power = tfr_array_morlet(data, sfreq=sfreq, freqs=freqs,
                         n_cycles=n_cycles, output='power')

im = axes[1].pcolormesh(times, freqs, power[0, 0, ...], cmap='RdBu_r',
                        shading='auto')
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Frequency (Hz)')

# Add colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.12, 0.03, 0.33])
fig.colorbar(im, cax=cbar_ax)

plt.show()
