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
#
# This project was supported by Google Summer of Code (GSoC) 2024.

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
    weights_ampa = {'L2_basket': param_values['weight_basket'],
                    'L2_pyramidal': param_values['weight_pyr'],
                    'L5_basket': param_values['weight_basket'],
                    'L5_pyramidal': param_values['weight_pyr']}

    synaptic_delays = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                       'L5_basket': 1., 'L5_pyramidal': 1.}

    # Add an evoked drive to the network.
    net.add_evoked_drive('evprox',
                         mu=40,
                         sigma=5,
                         numspikes=1,
                         location='proximal',
                         weights_ampa=weights_ampa,
                         synaptic_delays=synaptic_delays)

###############################################################################
# Define a parameter grid for the batch simulation.


param_grid = {
    'weight_basket': np.logspace(-4 - 1, 10),
    'weight_pyr': np.logspace(-4, -1, 10)
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
        dpl_smooth = result['dpl'][0].copy().smooth(window_len=30)
        dpl_data = dpl_smooth.data['agg']
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
net = jones_2009_model(mesh_shape=(3, 3))
batch_simulation = BatchSimulate(net=net,
                                 set_params=set_params,
                                 summary_func=summary_func)
simulation_results = batch_simulation.run(param_grid,
                                          n_jobs=n_jobs,
                                          combinations=False,
                                          backend='multiprocessing')
# backend='dask' if installed
print("Simulation results:", simulation_results)
###############################################################################
# This plot shows an overlay of all smoothed dipole waveforms from the
# batch simulation. Each line represents a different set of parameters,
# allowing us to visualize the range of responses across the parameter space.

dpl_waveforms = []
for data_list in simulation_results['simulated_data']:
    for data in data_list:
        dpl_smooth = data['dpl'][0].copy().smooth(window_len=30)
        dpl_waveforms.append(dpl_smooth.data['agg'])

plt.figure(figsize=(10, 6))
for waveform in dpl_waveforms:
    plt.plot(waveform, alpha=0.5, linewidth=3)
plt.title('Overlay of Dipole Waveforms')
plt.xlabel('Time (ms)')
plt.ylabel('Dipole Amplitude (nAm)')
plt.grid(True)
plt.tight_layout()
plt.show()
###############################################################################
# This plot displays the minimum and maximum dipole peaks across
# different synaptic strengths. This allows us to see how the range of
# dipole activity changes as we vary the synaptic strength parameter.

min_peaks, max_peaks, param_values = [], [], []
for summary_list, data_list in zip(simulation_results['summary_statistics'],
                                   simulation_results['simulated_data']):
    for summary, data in zip(summary_list, data_list):
        min_peaks.append(summary['min_peak'])
        max_peaks.append(summary['max_peak'])
        param_values.append(data['param_values']['weight_basket'])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(param_values, min_peaks, label='Min Dipole Peak')
plt.plot(param_values, max_peaks, label='Max Dipole Peak')
plt.xlabel('Synaptic Strength (nS)')
plt.ylabel('Dipole Peak Magnitude')
plt.title('Min and Max Dipole Peaks across Simulations')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.tight_layout()
plt.show()
