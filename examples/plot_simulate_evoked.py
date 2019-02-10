"""
===============
Simulate dipole
===============

This example demonstrates how to simulate a dipole using Neurons.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import sys
import os.path as op

from neuron import h

###############################################################################
# Let us import mne_neuron

import mne_neuron
import mne_neuron.fileio as fio
import mne_neuron.paramrw as paramrw
from mne_neuron.dipole import Dipole
from mne_neuron import network
from mne_neuron.conf import readconf

###############################################################################
# Then we setup the directories

mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')
h.load_file("stdrun.hoc")

dconf = readconf(op.join(mne_neuron_root, 'hnn.cfg'))

# data directory - ./data
dproj = dconf['datdir']  # fio.return_data_dir(dconf['datdir'])
debug = dconf['debug']
pc = h.ParallelContext(1)
f_psim = ''
ntrial = 1

f_psim = op.join(mne_neuron_root, 'param', 'default.param')

simstr = f_psim.split(op.sep)[-1].split('.param')[0]
datdir = op.join(dproj, simstr)

# creates p_exp.sim_prefix and other param structures
p_exp = paramrw.ExpParams(f_psim, debug=debug)

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
# write the raw dipole
with open(doutf['file_dpl'], 'w') as f:
    for k in range(int(t_vec.size())):
        f.write("%03.3f\t" % t_vec.x[k])
        f.write("%5.4f\t" % (dp_rec_L2.x[k] + dp_rec_L5.x[k]))
        f.write("%5.4f\t" % dp_rec_L2.x[k])
        f.write("%5.4f\n" % dp_rec_L5.x[k])
# renormalize the dipole and save
# fix to allow init from data rather than file
dpl = Dipole(doutf['file_dpl'])
dpl.baseline_renormalize(doutf['file_param'])
dpl.convert_fAm_to_nAm()
dconf['dipole_scalefctr'] = dpl.scale(
    paramrw.find_param(doutf['file_param'], 'dipole_scalefctr'))
dpl.smooth(paramrw.find_param(
    doutf['file_param'], 'dipole_smooth_win') / h.dt)
dpl.plot()

pc.barrier()  # make sure all done in case multiple trials

pc.runworker()
pc.done()
