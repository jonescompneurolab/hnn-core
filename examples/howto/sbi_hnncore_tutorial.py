"""
==========================================
09. SBI with HNN-core: Parameter Inference
==========================================

This tutorial demonstrates how to use Simulation-Based Inference (SBI)
with HNN-core to infer network parameters from simulated data. We'll
simulate neural network activity, then use SBI to infer the parameters
that generated this activity.
"""

# Authors: Abdul Samad Siddiqui <abdulsamadsid1@gmail.com>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

###############################################################################
# First, we need to import the necessary packages. The Simulation-Based
# Inference (SBI) package can be installed with ``pip install sbi``.
# Let us import ``hnn_core`` and all the necessary libraries.

import numpy as np
import torch
from hnn_core.batch_simulate import BatchSimulate
from sbi import utils as sbi_utils
from sbi import inference as sbi_inference
import matplotlib.pyplot as plt
from hnn_core.network_models import jones_2009_model

###############################################################################
# Now, we'll set up our simulation parameters. We're using a small number of
# simulations for this example, but in practice, you might want to use more.

n_simulations = 100
n_jobs = 20

###############################################################################
# This function sets the parameters for our neural network model. It adds an
# evoked drive to the network with specific weights and delays.
#
# The `add_evoked_drive` function simulates external input to the network,
# mimicking sensory stimulation or other external events.
#
# - `evprox` indicates a proximal drive, targeting dendrites near the cell
#   bodies.
# - `mu=40` and `sigma=5` define the timing (mean and spread) of the input.
# - `numspikes=1` means it's a single, brief stimulation.
# - `weights_ampa` and `synaptic_delays` control the strength and
#   timing of the input.
#
# This evoked drive causes the initial positive deflection in the dipole
# signal, triggering a cascade of activity through the network and
# resulting in the complex waveforms observed.


def set_params(param_values, net=None):
    weight_pyr = 10**float(param_values['weight_pyr'])
    weights_ampa = {'L5_pyramidal': weight_pyr}
    synaptic_delays = {'L5_pyramidal': 1.}

    net.add_evoked_drive('evprox',
                         mu=40,
                         sigma=5,
                         numspikes=1,
                         location='proximal',
                         weights_ampa=weights_ampa,
                         synaptic_delays=synaptic_delays)
    return net

###############################################################################
# Here, we generate our parameter grid and run the simulations. We're varying
# the 'weight_pyr' parameter between 10^-4 and 10^-1.


val = np.linspace(-4, -1, n_simulations)
param_grid = {
    'weight_pyr': val.tolist()
}

net = jones_2009_model(mesh_shape=(3, 3))
batch_simulator = BatchSimulate(set_params=set_params,
                                net=net,
                                tstop=170)
simulation_results = batch_simulator.run(
    param_grid, n_jobs=n_jobs, combinations=False)

print(
    f"Number of simulations run: {len(simulation_results['simulated_data'])}")

###############################################################################
# This function extracts the dipole data from our simulation results. Dipole
# data represents the aggregate electrical activity of the neural population.


def extract_dipole_data(sim_results):
    dipole_data_list = []
    for result in sim_results['simulated_data']:
        for sim_data in result:
            dipole_data_list.append(sim_data['dpl'][0].data['agg'])
    return dipole_data_list


dipole_data = extract_dipole_data(simulation_results)

###############################################################################
# Now we prepare our data for the SBI algorithm. 'thetas' are our parameters,
# and 'xs' are our observed data (the dipole activity). These will be used by
# the SBI algorithm to learn the relationship between parameters and
# the resulting neural activity.

thetas = torch.tensor(param_grid['weight_pyr'],
                      dtype=torch.float32).reshape(-1, 1)
xs = torch.stack([torch.tensor(data, dtype=torch.float32)
                 for data in dipole_data])

###############################################################################
# Here we set up our SBI inference. We define a prior distribution for our
# parameter and create our inference object.

prior = sbi_utils.BoxUniform(
    low=torch.tensor([-4]), high=torch.tensor([-1]))
inference = sbi_inference.SNPE(prior=prior)
density_estimator = inference.append_simulations(thetas, xs).train()
posterior = inference.build_posterior(density_estimator)

# The prior distribution represents our initial guess about the range of
# possible values for `weight_pyr`. The SBI algorithm will use this prior,
# along with the simulated data, to build a posterior distribution, which
# represents our updated belief about `weight_pyr` after seeing the data.

###############################################################################
# This function allows us to simulate data for a single parameter value.


def simulator_batch(param):
    param_grid_single = {'weight_pyr': [float(param)]}
    results = batch_simulator.run(
        param_grid_single, n_jobs=1, combinations=False)
    return torch.tensor(extract_dipole_data(results)[0], dtype=torch.float32)

###############################################################################
# Now we'll infer parameters for "unknown" data. We generate this data using
# a parameter value that we pretend we don't know.


unknown_param = torch.tensor([[np.random.choice(np.linspace(-4, -1, 100))]])
x_o = simulator_batch(unknown_param.item())
samples = posterior.sample((1000,), x=x_o)

print(f"True (unknown) parameter: {unknown_param.item()}")
print(f"Inferred parameter (mean): {samples.mean().item()}")
print(f"Inferred parameter (median): {samples.median().item()}")

###############################################################################
# Let's visualize the posterior distribution of our inferred parameters.

plt.figure(figsize=(10, 6))
plt.hist(samples.numpy(), bins=30, density=True, alpha=0.7)
plt.xlabel('Parameter Value')
plt.ylabel('Density')
plt.title('Posterior Distribution of Parameters')
plt.axvline(unknown_param.item(), color='r', linestyle='dashed',
            linewidth=2, label='True Parameter')
plt.legend()
plt.xlim([-4, -1])
plt.show()

# This plot shows the posterior distribution of the inferred parameter values.
# If the inferred posterior distribution is centered around the true parameter
# value, it suggests that the SBI method is accurately capturing the underlying
# parameter. The red dashed line represents the true parameter value.

###############################################################################
# Finally, we'll evaluate the performance of our SBI method on multiple
# unseen parameter values.

unseen_params = np.linspace(-4, -1, 10)
unseen_data = [simulator_batch(param) for param in unseen_params]
unseen_samples = [posterior.sample((100,), x=x) for x in unseen_data]

plt.figure(figsize=(12, 6))
plt.boxplot([samples.numpy().flatten() for samples in unseen_samples],
            positions=unseen_params, widths=0.05, vert=True)
plt.xlabel('True Parameter')
plt.ylabel('Inferred Parameter')
plt.title('SBI Performance on Unseen Data')
plt.xticks(ticks=unseen_params, labels=[f'{param:.2f}'
                                        for param in unseen_params])
plt.xlim(-4.1, -0.9)
plt.show()

# This boxplot visualizes the distribution of inferred parameters for each
# unseen true parameter value. The true parameters are shown on the x-axis,
# and the boxes represent the spread of inferred values. If the inferred
# parameters closely follow the diagonal (where inferred = true), it indicates
# that the SBI method is performing well.
