"""
====================
07. Batch Simulation
====================

This example shows how to do batch simulations in HNN-core, allowing users to
efficiently run multiple simulations with different parameters
for comprehensive analysis.
"""

# Authors: Abdul Samad Siddiqui <abdulsamadsid1@gmail.com>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

###############################################################################
# Let us import ``hnn_core``.

import matplotlib.pyplot as plt
import numpy as np
from hnn_core.batch_simulate import BatchSimulate
from hnn_core import jones_2009_model

# The number of cores may need modifying depending on your current machine.
n_jobs = 10
###############################################################################


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

###############################################################################
# Define a parameter grid for the batch simulation.


param_grid = {
    'weight_basket': np.logspace(-4 - 1, 5),
    'weight_pyr': np.logspace(-4, -1, 5),
    'mu': np.linspace(20, 80, 5),
    'sigma': np.linspace(1, 20, 5)
}

###############################################################################
# Define a function to calculate summary statistics


def summary_func(results):
    """
    Calculate the min and max dipole peak for each simulation result.

    Parameters
    ----------
    results : list
        List of dictionaries containing simulation results.

    Returns
    -------
    summary_stats : list
        Summary statistics for each simulation result.
    """
    summary_stats = []
    for result in results:
        dpl_data = result['dpl'][0].data['agg']
        min_peak = np.min(dpl_data)
        max_peak = np.max(dpl_data)
        summary_stats.append({'min_peak': min_peak, 'max_peak': max_peak})
    return summary_stats

###############################################################################
# Run the batch simulation and collect the results.

# Comment off this code, if dask and distributed Python packages are installed
# from dask.distributed import Client
# client = Client(threads_per_worker=1, n_workers=5, processes=False)


# Run the batch simulation and collect the results.
batch_simulation = BatchSimulate(set_params=set_params,
                                 summary_func=summary_func)
simulation_results = batch_simulation.run(param_grid,
                                          n_jobs=n_jobs,
                                          combinations=False,
                                          backend='multiprocessing')
# backend='dask' if installed
print("Simulation results:", simulation_results)
###############################################################################
# Extract and Plot Results
###############################################################################

# Extract min and max peaks for plotting
min_peaks, max_peaks = [], []
for res in simulation_results:
    summary_stats = res[0]
    min_peaks.append(summary_stats['min_peak'])
    max_peaks.append(summary_stats['max_peak'])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(min_peaks, label='Min Dipole Peak')
plt.plot(max_peaks, label='Max Dipole Peak')
plt.xlabel('Simulation Index')
plt.ylabel('Dipole Peak Magnitude')
plt.title('Min and Max Dipole Peaks across Simulations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
