"""
===========================
Setting custom cell objects
===========================

This is an advanced tutorial that demonstrates how to make your custom cell objects using
HNN-core.
"""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>

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
# distance-dependent ionic conductance *g*. It computes a piecewise linear
# function where the conductance at distance ``x`` from soma is linear from
# ``gsoma``` at x=0 (base of soma) to ``gdend`` at distance ``xkink``
# from base of soma and constant thereafter.
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
#
# * ``set_conductance``: sets the conductance of a particular
#   dendritic segment of the neuron, and
# * ``set_dends_biophys``: which loops over the segments in a cell and sets
#   their conductance.
import numpy as np
from hnn_core.pyramidal import L5Pyr


class CustomL5Pyr(L5Pyr):

    def set_conductance(self, seg):
        """Insert distance dependent ionic dynamics."""
        from neuron import h

        dist_soma = abs(h.distance(seg.x))
        seg.gbar_ca = get_g_at_dist(
            x=dist_soma,
            gsoma=params['L5Pyr_soma_gbar_ca'],
            gdend=params['L5Pyr_dend_gbar_ca'],
            xkink=1501)  # beginning of tuft

    def set_dends_biophys(self):
        """Set custom dendritic biophysics"""
        from neuron import h

        L5Pyr.set_dends_biophys(self)
        # iterate over dendritic segments, compute
        # conductance and set it
        for key in self.dends:
            self.dends[key].push()
            for seg in self.dends[key]:
                self.set_conductance(seg)
            h.pop_section()


###############################################################################
# Now let's set the custom cell object in the NeuronNetwork object. The
# NeuronNetwork object is responsible for creating the cells in Neuron and
# connecting them according to the specification provided by Network object.
from hnn_core.neuron import NeuronNetwork

net = Network(params)
neuron_net = NeuronNetwork(net)
neuron_net.set_cell_morphology({'L5Pyr': CustomL5Pyr})

###############################################################################
# Finally, we will run the simulation in parallel for 2 trials.
from hnn_core import JoblibBackend

with JoblibBackend(n_jobs=1) as parallel:
    dpls = parallel.simulate(net, neuron_net=neuron_net)
