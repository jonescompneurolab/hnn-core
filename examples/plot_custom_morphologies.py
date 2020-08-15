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
print(params)

###############################################################################
# Let us first create our network from the params file.
net = Network(params)

###############################################################################
# Now let's create a Pyramidal cell with custom calcium dynamics. First,
# let us define a function to compute distance-dependent ionic conductance.

def get_g_at_dist(x, gsoma, gdend, xkink):
    """Compute distance-dependent ionic conductance."""
    if x > xkink:
        return gdend
    g = gsoma + x * (gdend - gsoma) / xkink
    return g


###############################################################################
# Next, we need to create a subclass of the original HNN class and override
# any methods that we would like to override.
import numpy as np

from neuron import h
from hnn_core.pyramidal import L5Pyr


class CustomL5Pyr(L5Pyr):
    def insert_almog(self, seg):
        """Insert distance dependent ionic dynamics."""
        # XXX: what does almog stand for?
        dist_soma = abs(h.distance(seg.x))
        seg.gkbar_hh2 = self.p_all['L5Pyr_dend_gkbar_hh2'] + \
            self.p_all['L5Pyr_soma_gkbar_hh2'] * np.exp(-0.006 * dist_soma)

        seg.gnabar_hh2 = get_g_at_dist(
            x=dist_soma,
            gsoma=self.p_all['L5Pyr_soma_gnabar_hh2'],
            gdend=self.p_all['L5Pyr_dend_gnabar_hh2'],
            xkink=962)
        seg.gbar_ca = get_g_at_dist(
            x=dist_soma,
            gsoma=self.p_all['L5Pyr_soma_gbar_ca'],
            gdend=self.p_all['L5Pyr_dend_gbar_ca'],
            xkink=1501)  # beginning of tuft

    def _biophys_dends(self):
        """Override setting dendrites."""
        L5Pyr._biophys_dends(self)
        for key in self.dends:
            self.dends[key].push()
            for seg in self.dends[key]:
                seg.gbar_ar = 1e-6 * np.exp(3e-3 * h.distance(seg.x))
                self.insert_almog(seg)
            h.pop_section()

###############################################################################
# Now let's set the cell morphology of the NeuronNetwork object
from hnn_core.neuron import NeuronNetwork, _simulate_single_trial

net = Network(params)
neuron_network = NeuronNetwork(net)
neuron_network.set_cell_morphology({'L5Pyr': CustomL5Pyr})
neuron_network._build()
dpl = _simulate_single_trial(neuron_network)
