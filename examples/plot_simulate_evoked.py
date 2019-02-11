"""
===============
Simulate dipole
===============

This example demonstrates how to simulate a dipole using Neurons.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import sys
import os
import os.path as op

import numpy as np
from neuron import h

###############################################################################
# Let us import mne_neuron

import mne_neuron
import mne_neuron.fileio as fio
import mne_neuron.paramrw as paramrw
from mne_neuron.dipole import Dipole
from mne_neuron import network

###############################################################################
# Then we setup the directories
mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')
h.load_file("stdrun.hoc")

# data directory - ./data
dproj = op.join(mne_neuron_root, 'data')
pc = h.ParallelContext(1)
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

# core iterator through experimental groups
expmt_group = p_exp.expmt_groups[0]

# return the param dict for this simulation
simparams = p = p_exp.return_pdict(expmt_group, 0)

pc.barrier()  # get all nodes to this place before continuing
pc.gid_clear()

# global variables, should be node-independent
h("dp_total_L2 = 0.")
h("dp_total_L5 = 0.")

# Set tstop before instantiating any classes
h.tstop = p['tstop']
h.dt = p['dt']  # simulation duration and time-step
h.celsius = p['celsius']  # 37.0 - set temperature
net = network.NetworkOnNode(p)  # create node-specific network

###############################################################################
# We define the arrays (Vector in numpy) for recording the signals
t_vec = h.Vector()
t_vec.record(h._ref_t)  # time recording
dp_rec_L2 = h.Vector()
dp_rec_L2.record(h._ref_dp_total_L2)  # L2 dipole recording
dp_rec_L5 = h.Vector()
dp_rec_L5.record(h._ref_dp_total_L5)  # L5 dipole recording

net.movecellstopos()  # position cells in 2D grid
pc.barrier()

# sets the default max solver step in ms (purposefully large)
pc.set_maxstep(10)

h.finitialize()  # initialize cells to -65 mV, after all the NetCon delays have been specified


###############################################################################
# We define a callback function for printing out time during simulation run
def prsimtime():
    sys.stdout.write('\rSimulation time: {0} ms...'.format(round(h.t, 2)))
    sys.stdout.flush()


printdt = 10
for tt in range(0, int(h.tstop), printdt):
    h.cvode.event(tt, prsimtime)  # print time callbacks

h.fcurrent()
h.frecord_init()  # set state variables if they have been changed since h.finitialize
pc.psolve(h.tstop)  # actual simulation - run the solver
pc.barrier()

# these calls aggregate data across procs/nodes
pc.allreduce(dp_rec_L2, 1)
# combine dp_rec on every node, 1=add contributions together
pc.allreduce(dp_rec_L5, 1)
net.aggregate_currents()  # aggregate the currents independently on each proc
# combine net.current{} variables on each proc
pc.allreduce(net.current['L5Pyr_soma'], 1)
pc.allreduce(net.current['L2Pyr_soma'], 1)

pc.barrier()

# write time and calculated dipole to data file only if on the first proc
# only execute this statement on one proc

# write params to the file
paramrw.write(doutf['file_param'], p, net.gid_dict)
dpl_data = np.c_[t_vec.as_numpy(),
                 dp_rec_L2.as_numpy() + dp_rec_L5.as_numpy(),
                 dp_rec_L2.as_numpy(), dp_rec_L5.as_numpy()]
np.savetxt(doutf['file_dpl'], dpl_data, fmt='%5.4f')

# renormalize the dipole
# fix to allow init from data rather than file
dpl = Dipole(doutf['file_dpl'])
dpl.baseline_renormalize(doutf['file_param'])
dpl.convert_fAm_to_nAm()
dpl.scale(paramrw.find_param(doutf['file_param'], 'dipole_scalefctr'))
dpl.smooth(paramrw.find_param(
    doutf['file_param'], 'dipole_smooth_win') / h.dt)
dpl.plot()

pc.barrier()  # make sure all done in case multiple trials

pc.runworker()
pc.done()
