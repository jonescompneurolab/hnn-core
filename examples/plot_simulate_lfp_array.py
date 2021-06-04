"""
==============
08. Record LFP
==============

This example demonstrates how to record local field potentials (LFPs).
"""

# Authors: Nick Tolley <nicholas_tolley@brown.edu>
#          Mainak Jas <mainakjas@gmail.com>
#          Christopher Bailey <cjb@cfin.au.dk>

# sphinx_gallery_thumbnail_number = 4

import os.path as op

###############################################################################
# Let us import ``hnn_core``.

import hnn_core
from hnn_core import read_params, default_network, simulate_dipole

hnn_core_root = op.dirname(hnn_core.__file__)

###############################################################################
# Then we read the parameters file
# params_fname = op.join(hnn_core_root, 'param', 'default.json')
params_fname = op.join(hnn_core_root, 'param', 'gamma_L5weak_L2weak.json')
params = read_params(params_fname)
params['tstop'] = 250
###############################################################################
# We will start with simulating the evoked response from
# :ref:`evoked example <sphx_glr_auto_examples_plot_simulate_evoked.py>`.
# We first instantiate the network.
# (Note: Setting ``add_drives_from_params=True`` loads a set of predefined
# drives without the drives API shown previously).
net = default_network(params)
weights_ampa = {'L2_pyramidal': 0.0008, 'L5_pyramidal': 0.0075}
synaptic_delays = {'L2_pyramidal': 0.1, 'L5_pyramidal': 1.0}
rate_constant = {'L2_pyramidal': 140.0, 'L5_pyramidal': 40.0}
net.add_poisson_drive(
    'poisson', rate_constant=rate_constant, weights_ampa=weights_ampa,
    location='proximal', synaptic_delays=synaptic_delays, seedcore=1079)

###############################################################################
# LFP recordings require specifying the electrode postion. It can be useful
# to visualize the cells of the network to decide on the placement of each
# electrode.
net.plot_cells()

###############################################################################
# Electrode positions are stored under ``Network.lfp_array`` as a list
# of tuples. Once we have chosen x,y,z coordinates for each electrode, we can
# add them to the simulation.

# XXX coordinates FUBAR, cortical depth direction is Y in NEURON objects!
# electrode_pos = [(2, 400, 2), (6, 800, 6)]
depths = list(range(-412, 1900, 200))
electrode_pos = [(4.5, dep, 4.5) for dep in depths]
net.add_electrode_array('shank1_psa', electrode_pos, method='psa')
net.add_electrode_array('shank1_lsa', electrode_pos, method='lsa')
print(net.rec_array)
net.plot_cells()

dpl = simulate_dipole(net)
###############################################################################
# Now that our electrodes are specified, we can run the simulation. The LFP
# recordings are stored under ``Network.lfp_array``.
import matplotlib.pyplot as plt

trial_idx = 0
net.rec_array['shank1_psa'][trial_idx].plot(contact_no=[1, -2])

###############################################################################
# We can compare the dipole current wave form to that of a single LFP channel.
show_electrodes = [8, 10]
tmin, tmax = 50, 225
window_len = 10  # ms
decimate = [5, 4]  # from 40k to 8k to 2k
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 6))
lfp_ax = ax.twinx()

dpl[trial_idx].copy().smooth(
    window_len=window_len).plot(ax=ax, decim=decimate, tmin=tmin, tmax=tmax,
                                color='k', show=False)

net.rec_array['shank1_psa'][trial_idx].plot(
    contact_no=show_electrodes, window_len=window_len, tmin=tmin, tmax=tmax,
    decim=decimate, ax=lfp_ax, show=False)

ax.set_title('Gamma oscillations in the dipole and LFP waveforms')
ax.legend(['Dipole'], loc='upper center')
lfp_ax.legend([depths[show_electrodes[0]], depths[show_electrodes[1]]],
              loc='lower center')
lfp_ax.yaxis.label.set_color('#1f77b4')

plt.show()

# For comparing PSA and LSA outputs (TEMPORARY!)
fig, axs = plt.subplots(3, 4, sharex=True, sharey=False, figsize=(10, 8))
for idx, ax in enumerate(axs.ravel()):
    net.rec_array['shank1_psa'][trial_idx].plot(contact_no=idx, ax=ax,
                                                tmin=60, tmax=180,
                                                window_len=5,
                                                show=False)
    net.rec_array['shank1_lsa'][trial_idx].plot(contact_no=idx, ax=ax,
                                                tmin=60, tmax=180,
                                                window_len=5,
                                                show=False)
plt.show()
