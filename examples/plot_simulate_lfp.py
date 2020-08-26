"""
===============
Simulate LFP
===============

This example demonstrates how to simulate a dipole for evoked-like
waveforms using HNN-core.
"""

import os.path as op
import numpy as np

import matplotlib.pyplot as plt
import itertools as it
from glob import glob

from neuron import h

import hnn_core
from hnn_core.tests import test_network, test_dipole
from hnn_core.network import Spikes, read_spikes
from hnn_core import lfp    

import matplotlib.pyplot as plt

from hnn_core import JoblibBackend
from hnn_core import simulate_dipole, read_params, Network, read_spikes, viz

hnn_core_root = op.dirname(hnn_core.__file__)

params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)

net = Network(params)

###############################################################################
# Let us import hnn_core

from hnn_core.pyramidal import L5Pyr
from hnn_core.network_builder import load_custom_mechanisms

load_custom_mechanisms()
cell = L5Pyr()
soma_pos = [cell.get3dinfo()[idx][0] for idx in range(3)]
apical_trunk_pos = [cell.get3dinfo()[idx][1] for idx in range(3)]

num_elec = 5
x_width = [-100,100]
y_width = [-100,100]

z_pos = 0
elec_grid = np.array([[c,r,z_pos] for c in np.linspace(x_width[0], x_width[1], num_elec) 
            for r in np.linspace(y_width[0], y_width[1], num_elec)])

h.load_file("stdgui.hoc")
h.cvode_active(1)

ns = h.NetStim()
ns.number = 1
ns.start = 100
ns.interval = 50.0
ns.noise = 0.  # deterministic

nc = h.NetCon(ns, cell.synapses['apicaltuft_ampa'])
nc.weight[0] = 0.001

h.tstop = 500.0
lfp_rec = {'lsa': ([], []), 'psa': ([], [])}
for method in ['lsa', 'psa']:
    elec_list = []
    for pos in range(elec_grid.shape[0]):
        elec = lfp.LFPElectrode(list(elec_grid[pos,:]), pc=h.ParallelContext(),
                                method=method)
        elec.setup()
        elec.LFPinit()
        elec_list.append(elec)

    h.run()
    for pos in range(elec_grid.shape[0]):
        elec_list[pos].pc.allreduce(elec_list[pos].lfp_v, 1)

        lfp_rec[method][0].append(elec_list[pos].lfp_t.to_python()) 
        lfp_rec[method][1].append(elec_list[pos].lfp_v.to_python())

######
# Plot results  
plt.figure(figsize=(5, 5))
plt.scatter(soma_pos[0], soma_pos[1], s=1000)
plt.plot(elec_grid[:,0], elec_grid[:, 1], 'ko')
plt.plot([soma_pos[0], apical_trunk_pos[0]], [soma_pos[1], apical_trunk_pos[1]])

for elec_idx in range(num_elec**2):
    elec_pos = elec_grid[elec_idx,:]
    elec_t = lfp_rec['lsa'][0][elec_idx]
    elec_v = lfp_rec['lsa'][1][elec_idx]
    elec_v = np.array(elec_v)

    # elec_v_scaled = (elec_v - elec_v[0]) / max(np.abs(np.max(elec_v - elec_v[0])), np.abs(np.min(elec_v - elec_v[0])))
    elec_v_scaled = (elec_v - elec_v[0])/2

    plot_t = np.linspace(elec_pos[0], elec_pos[0] + 25, len(elec_t))
    plot_v = (elec_v_scaled * 5) + elec_pos[1]

    plt.plot(plot_t, plot_v)
