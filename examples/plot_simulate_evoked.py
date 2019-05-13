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
from mne_neuron import simulate_dipole, Params, Network

mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')

###############################################################################
# Then we read the parameters file
params_fname = op.join(mne_neuron_root, 'param', 'default.json')
params = Params(params_fname)

# Now let's simulate the dipole
net = Network(params)
dpl = simulate_dipole(net)

###############################################################################
# and then plot it
import matplotlib
from matplotlib import pyplot as plt
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
dpl.plot(ax=axes[0])

###############################################################################
# Finally, we can also plot the spikes.
net.plot_spikes()
