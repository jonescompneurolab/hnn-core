"""
=====================
08. Record LFP
=====================

This example demonstrates how to record local field potentials (LFPs).
"""

# Authors: Nick Tolley <nicholas_tolley@brown.edu>
#          Mainak Jas <mainakjas@gmail.com>

# sphinx_gallery_thumbnail_number = 3

import os.path as op

###############################################################################
# Let us import ``hnn_core``.

import hnn_core
from hnn_core import read_params, Network, simulate_dipole

hnn_core_root = op.dirname(hnn_core.__file__)

###############################################################################
# Then we read the parameters file
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)

###############################################################################
# We will start with simulating the evoked response from
# :ref:`evoked example <sphx_glr_auto_examples_plot_simulate_evoked.py>`.
# We first instantiate the network.
# (Note: Setting ``add_drives_from_params=True`` loads a set of predefined
# drives without the drives API shown previously).
net = Network(params, add_drives_from_params=True)

###############################################################################
# LFP recordings require specifying the electrode postion. It can be useful
# to visualize the cells of the network to decide on the placement of each
# electrode.
net.plot_cells()

###############################################################################
# Electrode positions are stored under `net.pos_lfp` as a list of tuples. Once
# we have chosen x,y,z coordinates for each electrode, we can add them to the
# simulation.
electrode_pos = [(2, 2, 400), (6, 6, 800)]
net.add_electrode(electrode_pos)
print(net.pos_lfp)
net.plot_cells()

###############################################################################
# Now that our electrodes are specified, we can run the simulation. The LFP
# recordings are stored under `net.lfp`.
import matplotlib.pyplot as plt

dpl = simulate_dipole(net)
times = dpl[0].times[:-1]
plt.figure()
plt.plot(times, net.lfp[0])
plt.plot(times, net.lfp[1])
plt.legend([f'e_pos {electrode_pos[0]}', f'e_pos {electrode_pos[1]}'])
plt.show()
