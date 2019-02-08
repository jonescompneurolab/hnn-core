"""
===============
Simulate dipole
===============

This example demonstrates how to simulate a dipole using Neurons.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import os
import sys
import time
import shutil
import numpy as np

from neuron import h

# Cells are defined in other files
import mne_neuron.fileio as fio
import mne_neuron.paramrw as paramrw
from mne_neuron.dipole import Dipole

from mne_neuron.pyramidal import L5Pyr, L2Pyr
from mne_neuron.basket import L2Basket, L5Basket

from mne_neuron import network
from mne_neuron.conf import readconf

h.load_file("stdrun.hoc")

dconf = readconf()

# data directory - ./data
dproj = dconf['datdir']  # fio.return_data_dir(dconf['datdir'])
debug = dconf['debug']
pc = h.ParallelContext()
pc_id = int(pc.id())
f_psim = ''
ntrial = 1

f_psim = os.path.join('param', 'default.param')

simstr = f_psim.split(os.path.sep)[-1].split('.param')[0]
datdir = os.path.join(dproj, simstr)


# callback function for printing out time during simulation run
printdt = 10


def prsimtime():
    sys.stdout.write('\rSimulation time: {0} ms...'.format(round(h.t, 2)))
    sys.stdout.flush()


# creates p_exp.sim_prefix and other param structures
p_exp = paramrw.ExpParams(f_psim, debug=debug)

# one directory for all experiments
ddir = fio.SimulationPaths()
ddir.create_new_sim(dproj, p_exp.expmt_groups, p_exp.sim_prefix)
if pc_id == 0:
    ddir.create_datadir()

# create rotating data files
doutf = {}
doutf['file_dpl'] = os.path.join(datdir, 'rawdpl.txt')
doutf['file_param'] = os.path.join(datdir, 'param.txt')

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
h.celsius = p['celsius']  # 37.0 # p['celsius'] # set temperature
net = network.NetworkOnNode(p)  # create node-specific network

t_vec = h.Vector()
t_vec.record(h._ref_t)  # time recording
dp_rec_L2 = h.Vector()
dp_rec_L2.record(h._ref_dp_total_L2)  # L2 dipole recording
dp_rec_L5 = h.Vector()
dp_rec_L5.record(h._ref_dp_total_L5)  # L5 dipole recording

net.movecellstopos()  # position cells in 2D grid
pc.barrier()


def initrands(s=0):  # fix to use s
    # if there are N_trials, then randomize the seed
    # establishes random seed for the seed seeder (yeah.)
    # this creates a prng_tmp on each, but only the value from 0 will be used
    prng_tmp = np.random.RandomState()
    if pc_id == 0:
        r = h.Vector(1, s)  # initialize vector to 1 element, with a 0
        if ntrial == 1:
            prng_base = np.random.RandomState(pc_id + s)
        else:
            # Create a random seed value
            r.x[0] = prng_tmp.randint(1e9)
    else:
        # create the vector 'r' but don't change its init value
        r = h.Vector(1, s)
    pc.broadcast(r, 0)  # broadcast random seed value in r to everyone
    # set object prngbase to random state for the seed value
    # other random seeds here will then be based on the gid
    prng_base = np.random.RandomState(int(r.x[0]))
    # seed list is now a list of seeds to be changed on each run
    # otherwise, its originally set value will remain
    # give a random int seed from [0, 1e9]
    for param in p_exp.prng_seed_list:  # this list empty for single experiment/trial
        p[param] = prng_base.randint(1e9)
    # print('simparams[prng_seedcore]:',simparams['prng_seedcore'])


initrands(0)  # init once


# All units for time: ms

t0 = time.time()  # clock start time

# sets the default max solver step in ms (purposefully large)
pc.set_maxstep(10)

h.finitialize()  # initialize cells to -65 mV, after all the NetCon delays have been specified
if pc_id == 0:
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
if pc_id == 0:
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

    if debug:
        print("Simulation run time: %4.4f s" % (time.time()-t0))
        print("Simulation directory is: %s" % ddir.dsim)

pc.barrier()  # make sure all done in case multiple trials

pc.runworker()
pc.done()
