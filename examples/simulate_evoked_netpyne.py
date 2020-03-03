"""
===============
Simulate dipole
===============

This example demonstrates how to simulate a dipole for evoked-like
waveforms using HNN-core with NetPyNE
"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Salvador Dura-Bernal <salvadordura@gmail.com>

from netpyne import sim

###############################################################################
# Let us import hnn_core
import hnn_core
import os.path as op
from hnn_core import read_params, Config

###############################################################################
# Then we read the parameters file
hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')
params_fname = op.join(hnn_core_root, 'param', 'AlphaAndBeta.param')
params = read_params(params_fname)

###############################################################################
# Next we build the configuration
config = Config(params)
# The netpyne simConfig object is available at config.cfg
# The netpyne netParams object is available at config.net

###############################################################################
# Now let's create, simulate and analyze model
sim.createSimulateAnalyze(simConfig=config.cfg, netParams=config.net)
