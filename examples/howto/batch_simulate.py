"""
====================
06. Batch Simulation
====================

This example shows how to do batch simulations in HNN-core, allowing users to
efficiently run multiple simulations with different parameters
for comprehensive analysis.
"""

# Authors: Abdul Samad Siddiqui <abdulsamadsid1@gmail.com>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

#############################################################################
# Let us import ``hnn_core``.

import hnn_core
import numpy as np
from hnn_core import BatchSimulate
from hnn_core.network_models import jones_2009_model

# The number of cores may need modifying depending on your current machine.
n_jobs = 10
###########################################################################


def set_params(param_values, net=None):
    """
    Set parameters in the network drives.

    Parameters
    ----------
    param_values : dict
        Dictionary of parameter values.
    net : instance of Network, optional
        If None, a new network is created using the specified model type.
    """
    if net is None:
        net = jones_2009_model()

    weights_ampa = {'L2_basket': param_values['weight_basket'],
                    'L2_pyramidal': param_values['weight_pyr'],
                    'L5_basket': param_values['weight_basket'],
                    'L5_pyramidal': param_values['weight_pyr']}

    synaptic_delays = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                       'L5_basket': 1., 'L5_pyramidal': 1.}

    # Add an evoked drive to the network.
    net.add_evoked_drive('evprox',
                         mu=param_values['mu'],
                         sigma=param_values['sigma'],
                         numspikes=1,
                         location='proximal',
                         weights_ampa=weights_ampa,
                         synaptic_delays=synaptic_delays)

###########################################################################


param_grid = {
    'weight_basket': np.logspace(-4 - 1, 5),
    'weight_pyr': np.logspace(-4, -1, 5),
    'mu': np.linspace(20, 80, 5),
    'sigma': np.linspace(1, 20, 5)
}

batch_simulation = BatchSimulate(set_params=set_params)
simulation_results = batch_simulation.run(param_grid,
                                          n_jobs=n_jobs,
                                          combinations=False)
