"""
===========================
Setting custom cell objects
===========================

This example demonstrates how to make your custom cell objects using
HNN-core.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>

import os.path as op
import tempfile

###############################################################################
# Let us import hnn_core

import hnn_core
from hnn_core import simulate_dipole, read_params, Network, read_spikes

hnn_core_root = op.dirname(hnn_core.__file__)

###############################################################################
# Then we read the parameters file
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)

###############################################################################
# Let us first create our network from the params file.
net = Network(params)


###############################################################################
# The custom cell object we want to create is one with distance-dependent
# ionic dynamics. To do so, first let's create a function for computing
# distance-dependent ionic conductance g. It computes a piecewise linear
# function where the conductance at distance x from soma is linear from gsoma
# at x=0 to gdend at xkink and constant at gdend thereafter.
def get_g_at_dist(x, gsoma, gdend, xkink):
    """Compute distance-dependent ionic conductance."""
    if x > xkink:
        return gdend
    g = gsoma + x * (gdend - gsoma) / xkink
    return g

###############################################################################
# Let us try to visualize this in the case of calcium ion conductances for L5
# Pyramidal cells. At the soma, it is 10 pS/um^2 and at the beginning of
# apical tuft, we want it to be 40 pS/um^2
import numpy as np
import matplotlib.pyplot as plt

params.update({'L5Pyr_soma_gbar_ca': 10, 'L5Pyr_dend_gbar_ca': 40})

xs = np.linspace(0, 2000, 50)
gs = list()
for x in xs:
    g = get_g_at_dist(x, gsoma=params['L5Pyr_soma_gbar_ca'],
                      gdend=params['L5Pyr_dend_gbar_ca'],
                      xkink=1501)
    gs.append(g)

plt.plot(xs, gs)
plt.xlabel('Distance from bottom of soma (um)')
plt.ylabel('Conductance (pS/um^2)')

###############################################################################
# Next, we need to create a subclass of the original HNN class and override
# any methods that we would like to override. The new cell class contains
# two new methods:
#   * ``set_conductance``: sets the conductance of a particular 
#     dendritic segment of the neuron, and
#   * ``set_dends_biophys``: which loops over the segments in a cell and sets
#     their conductance.
import numpy as np

from neuron import h
from hnn_core.pyramidal import L5Pyr


class CustomL5Pyr(L5Pyr):

    def set_conductance(self, seg):
        """Insert distance dependent ionic dynamics."""
        dist_soma = abs(h.distance(seg.x))

        # set the Potassium (gkbar_hh2), Sodium (gnabar_hh2),
        # and Calcium (gbar_ca) dynamics
        seg.gkbar_hh2 = params['L5Pyr_dend_gkbar_hh2'] + \
            params['L5Pyr_soma_gkbar_hh2'] * np.exp(-0.006 * dist_soma)

        seg.gnabar_hh2 = get_g_at_dist(
            x=dist_soma,
            gsoma=params['L5Pyr_soma_gnabar_hh2'],
            gdend=params['L5Pyr_dend_gnabar_hh2'],
            xkink=962)

        seg.gbar_ca = get_g_at_dist(
            x=dist_soma,
            gsoma=params['L5Pyr_soma_gbar_ca'],
            gdend=params['L5Pyr_dend_gbar_ca'],
            xkink=1501)  # beginning of tuft

    def set_dends_biophys(self):
        """Set custom dendritic biophysics"""
        L5Pyr.set_dends_biophys(self)

        # iterate over dendritic segments, compute
        # conductance and set it
        for key in self.dends:
            self.dends[key].push()
            for seg in self.dends[key]:
                self.set_conductance(seg)
            h.pop_section()

###############################################################################
# Now let's set the custom cell object in the NeuronNetwork object
from hnn_core.neuron import NeuronNetwork, _simulate_single_trial

net = Network(params)
neuron_network = NeuronNetwork(net)
neuron_network.set_cell_morphology({'L5Pyr': CustomL5Pyr})

###############################################################################
# Now let's run the simulation
neuron_network._build()
dpl = _simulate_single_trial(neuron_network)
