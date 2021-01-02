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

# add a tonic Poisson-distributed excitation to pyramidal cells
weights_ampa = {'L2_pyramidal': 0.0008, 'L5_pyramidal': 0.0075}
dispersion_time = {'L2_pyramidal': 0.1, 'L5_pyramidal': 1.0}
rate_constants = {'L2_pyramidal': 140.0, 'L5_pyramidal': 40.0}
# XXX online docs had a seed of -3 (!). This worked because extpois was added
# last, the first artificial cell having gid=1352, leading to seeds >= 1349
# With the new API, seeds begin from 270, so init_seed = 1352 - 3 - 270
prng_initial_seed = 1079
net.add_poisson_drive(
    'poisson', rate_constants=rate_constants, weights_ampa=weights_ampa,
    location='proximal', dispersion_time=dispersion_time,
    seedcore=prng_initial_seed)

dpls = simulate_dipole(net, n_trials=1)

###############################################################################
# The network requires some time to reach steady state. Hence, we omit the
# first 50 ms in our time-frequency analysis.

tstart = 50
trial_idx = 0  # pick first trial
mask = dpls[trial_idx].times > tstart
times = dpls[trial_idx].times[mask]
data = dpls[trial_idx].data['agg'][mask]

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

###############################################################################
# As a final exercise, let us try to re-run the simulation with tonic inputs
# to the L5 Pyramidal cells. Notice that the oscillation waveform is now more
# regular with less noise due to the fact that the tonic drive is strong and
# outweighs the influence of the Poisson drive
net.add_tonic_input(cell_type='L5Pyr', amplitude=6., t0=0, T=params['tstop'])
dpls = simulate_dipole(net, n_trials=1)

dpls[0].plot()

###############################################################################
# Notice that the Layer 5 pyramidal neurons are now firing nearly
# synchronously. They in turn synchronously activate the inhibitory basket
# neurons, which then inhibit the pyramidal neurons for ~20 ms. Once the
# tonic drive outweighs the inhibition and the pyramidal neurons firing again
# creating a ~50 Hz PING rhythm. This type of synchronous rhythm is sometimes
# referred to as “strong” PING.
net.cell_response.plot_spikes_raster()
