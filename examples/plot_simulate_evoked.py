"""
===============
Simulate dipole
===============

This example demonstrates how to simulate a dipole using the Neuron
simulator.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import os
import os.path as op

###############################################################################
# Let us import mne_neuron

import mne_neuron
import mne_neuron.fileio as fio
import mne_neuron.paramrw as paramrw
from mne_neuron import simulate_dipole

###############################################################################
# Then we setup the directories
mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')
h.load_file("stdrun.hoc")

# data directory - ./data
dproj = op.join(mne_neuron_root, 'data')
f_psim = ''
ntrial = 1
f_psim = op.join(mne_neuron_root, 'param', 'default.param')

simstr = f_psim.split(op.sep)[-1].split('.param')[0]
datdir = op.join(dproj, simstr)
if not op.exists(datdir):
    os.mkdir(op.join(mne_neuron_root, 'data'))

# creates p_exp.sim_prefix and other param structures
p_exp = paramrw.ExpParams(f_psim)

# one directory for all experiments
ddir = fio.SimulationPaths()
ddir.create_new_sim(dproj, p_exp.expmt_groups, p_exp.sim_prefix)
ddir.create_datadir()

# create rotating data files
doutf = {}
doutf['file_dpl'] = op.join(datdir, 'rawdpl.txt')
doutf['file_param'] = op.join(datdir, 'param.txt')

# return the param dict for this simulation
params = p_exp.return_pdict('default', 0)

dpl = simulate_dipole(params)
dpl.plot()
