"""
==============
08. Record LFP
==============

This example demonstrates how to record local field potentials (LFPs).
"""

# Authors: Nick Tolley <nicholas_tolley@brown.edu>
#          Mainak Jas <mainakjas@gmail.com>
#          Christopher Bailey <cjb@cfin.au.dk>

# sphinx_gallery_thumbnail_number = 3

import os.path as op

###############################################################################
# Let us import ``hnn_core``.

import hnn_core
from hnn_core import read_params, Network, simulate_dipole
from hnn_core.parallel_backends import MPIBackend

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
net = Network(params)
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
# Electrode positions are stored under ``Network.lfp`` as a list
# of tuples. Once we have chosen x,y,z coordinates for each electrode, we can
# add them to the simulation.

# XXX coordinates FUBAR, cortical depth direction is Y in NEURON objects!
# electrode_pos = [(2, 400, 2), (6, 800, 6)]
depths = list(range(-400, 1900, 200))
electrode_pos = [(45, dep, 45) for dep in depths]
net.add_electrode(electrode_pos)
print(net.lfp)
net.plot_cells()

with MPIBackend(n_procs=2):
    dpl = simulate_dipole(net)
###############################################################################
# Now that our electrodes are specified, we can run the simulation. The LFP
# recordings are also stored under ``Network.lfp``.
import matplotlib.pyplot as plt

plt.figure()
trial_idx = 0
plt.plot(dpl[trial_idx].times, net.lfp[1]['data'][trial_idx])
plt.plot(dpl[trial_idx].times, net.lfp[-2]['data'][trial_idx])
plt.legend([f'e_pos {electrode_pos[1]}', f'e_pos {electrode_pos[-2]}'])
plt.xlabel('Time (ms)')
plt.ylabel(r'Potential ($\mu V$)')
plt.show()

from hnn_core.viz import _decimate_plot_data
from hnn_core.dipole import _hammfilt
# from matplotlib.colors import SymLogNorm
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
window_len = 10  # ms
decimate = [5, 4]  # from 40k to 8k to 2k
winsz = np.round(1e-3 * window_len * dpl[trial_idx].sfreq)

plot_data = list()
for elec in net.lfp:
    filt_data = _hammfilt(elec['data'][trial_idx], winsz)
    decim_data, decim_time, lfp_sfreq = _decimate_plot_data(
        decimate, filt_data, dpl[trial_idx].times, sfreq=dpl[trial_idx].sfreq)
    plot_data.append(decim_data)
plot_data = np.array(plot_data)

# fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 8),
#                         gridspec_kw={'height_ratios': [2, 1, 2]})
# im = axs[0].pcolormesh(decim_time, depths, plot_data,
#                        norm=SymLogNorm(linthresh=1e-1, linscale=1.,
#                                        vmin=-10e2, vmax=10e2, base=10),
#                        cmap='BrBG_r', shading='auto')
# axs[0].set_ylabel(r'Distance from soma in Y ($\mu m$)')
# axs[0].set_xticks(np.arange(10, 250, 10), minor=True)
# axs[0].grid(True, which='both', axis='x')
# axins = inset_axes(axs[0],
#                    width="2%",  # width = 5% of parent_bbox width
#                    height="80%",  # height : 50%
#                    loc='lower left',
#                    bbox_to_anchor=(1.075, 0.1, 1, 1),
#                    bbox_transform=axs[0].transAxes,
#                    borderpad=0,
#                    )
# cbh = fig.colorbar(im, cax=axins, extend='both')
# cbh.ax.yaxis.set_ticks_position('left')
# cbh.ax.set_ylabel(r'Potential ($\mu V$)')

# dpl[trial_idx].plot(ax=axs[1], show=False)
# axs[1].set_xticks(np.arange(10, 250, 10), minor=True)
# axs[1].grid(True, which='both', axis='x')
# axs[1].set_xlabel('')
# axs[1].set_title('')

# net.cell_response.plot_spikes_raster(ax=axs[2], show=False)
# axs[2].set_xlabel('Time (ms)')
# plt.show()

show_electrode = 6
el_dep = depths[show_electrode]
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 6))
lfp_ax = ax.twinx()
dpl[trial_idx].copy().smooth(window_len=window_len).plot(ax=ax, show=False)
ax.set_title(f'Gamma oscillations in the dipole and LFP waveforms ({el_dep} '
             r'$\mu m$ above soma)')
ax.yaxis.label.set_color('#1f77b4')
lfp_ax.plot(decim_time, plot_data[show_electrode], color='#ff7f0e')
lfp_ax.set_ylabel(r'Electric potential($\mu V$)', color='#ff7f0e')
plt.show()
