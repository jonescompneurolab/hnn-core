"""
===============
Simulate dipole
===============

This example demonstrates how to simulate a dipole for evoked-like
waveforms using HNN-core.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>

import os.path as op

###############################################################################
# Let us import hnn_core

import hnn_core
from hnn_core import simulate, read_params, Config

hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')

###############################################################################
# Then we read the parameters file
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)

print(params)

###############################################################################
# This is a lot of parameters! We can also filter the
# parameters using unix-style wildcard characters
print(params['L2Pyr_soma*'])

###############################################################################
# Next we build the configuration
config = Config(params)

###############################################################################
# Now let's simulate the dipole
# You can simulate multiple trials in parallel by using n_jobs > 1
dpls, spks = simulate(config, n_trials=1, n_jobs=1)

###############################################################################
# and then plot it
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
for dpl in dpls:
    dpl.plot(ax=axes[0], layer='agg')
spks[0].plot_input_hist(ax=axes[1])

###############################################################################
# Finally, we can also plot the spikes.
spks[0].plot()

###############################################################################
# Now, let us try to make the exogenous driving inputs to the cells
# synchronous and see what happens

config.cfg.sync_evinput = True
config_sync = Config(params)
dpls_sync, spks_sync = simulate(config_sync, n_trials=1, n_jobs=1)
dpls_sync[0].plot()
spks_sync[0].plot_input_hist()
