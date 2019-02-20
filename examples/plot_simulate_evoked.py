"""
===============
Simulate dipole
===============

This example demonstrates how to simulate a dipole for evoked-like
waveforms using MNE-Neuron.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import os.path as op

###############################################################################
# Let us import mne_neuron

import mne_neuron
from mne_neuron import simulate_dipole, Params

from neuron import h

###############################################################################
# Then we setup the directories and Neuron
mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')
h.load_file("stdrun.hoc")

###############################################################################
# Then we read the parameters file
params_fname = op.join(mne_neuron_root, 'param', 'default.json')
params = Params(params_fname)
print(params)

###############################################################################
# This is a lot of parameters! We can also filter the
# parameters using unix-style wildcard characters
print(params['L2Pyr_soma*'])

###############################################################################
# Now let's simulate the dipole and plot it
dpl, net = simulate_dipole(params)
dpl.plot()

import numpy as np
import matplotlib.pyplot as plt
spikes = net.spiketimes.as_numpy()
gids = net.spikegids.as_numpy()
valid_gids = np.r_[net.gid_dict['evprox1'],
                   net.gid_dict['evprox2']]
mask_evprox = np.in1d(gids, valid_gids)
mask_evdist = np.in1d(gids, net.gid_dict['evdist1'])
plt.figure()
plt.hist(spikes[mask_evprox], 50, color='r')
plt.hist(spikes[mask_evdist], 10, color='g')
