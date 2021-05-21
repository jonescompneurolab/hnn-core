"""
================================
Calculate simulated dipole error
================================
This example calculates the RMSE between an experimental dipole waveform
and a simulated waveform using MNE-Neuron.
"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import numpy as np
import os.path as op
from collections import OrderedDict
import matplotlib.pyplot as plt

###############################################################################
# Let us import hnn_core

import hnn_core
from hnn_core import simulate_dipole, average_dipoles, rmse
from hnn_core import read_params, Network, Dipole, MPIBackend

hnn_core_root = op.join(op.dirname(hnn_core.__file__))

mpi_cmd = '/autofs/space/meghnn_001/users/mjas/opt/openmpi/bin/mpirun'
n_procs = 10

###############################################################################
# Set some global variables
optiter = 0
minopterr = 1e9

# From HNN, will take hours without parallelism
# default_num_step_sims = 30
# default_num_total_sims = 50
default_num_step_sims = 10
default_num_total_sims = 10

# Assume approx timings in param file
timing_range_multiplier = 3.0

# Estimating sigma is hard with n=small so make range small
sigma_range_multiplier = 50.0

# want to sweep large ranges for weights
synweight_range_multiplier = 500.0

# Chosen beacuse it achieved proper decay behavior
decay_multiplier = 1.6

###############################################################################
# Functions used for optimization


def split_by_evinput(params):
    """ Sorts parameter ranges by evoked inputs into a dictionary
    Parameters
    ----------
    params: an instance of Params
        Full set of simulation parameters

    Returns
    -------
    sorted_evinput_params: Dict
        Dictionary with parameters grouped by evoked inputs.
        Keys for each evoked input are
        'mean', 'sigma', 'ranges', 'num_params', 'start',
        'end', 'cdf', 'weights', 'opt_start', 'opt_end'.
        sorted_evinput_params['evprox1']['ranges'] is a dict
        with keys as the parameters to be optimized and values
        indicating the ranges over which they should be
        optimized. E.g.,
        sorted_evinput['evprox1']['ranges'] = 
        {
            'gbar_evprox_1_L2Pyr_ampa':
                {
                    'initial': 0.05125,
                    'minval': 0,
                    'maxval': 0.0915
                }
        }
        Elements are sorted by their start time.
    """

    evinput_params = {}
    for evinput_t in params['t_*']:
        id_str = evinput_t.lstrip('t_')
        value = float(params[evinput_t])

        evinput_params[id_str] = {}
        try:
            timing_sigma = float(params['sigma_' + evinput_t])
        except KeyError:
            timing_sigma = 3.0
            print("Couldn't fing timing_sigma. Using default %f" %
                  timing_sigma)

        if timing_sigma == 0.0:
            # sigma of 0 will not produce a CDF
            timing_sigma = 0.01

        evinput_params[id_str] = {'mean': value, 'sigma': timing_sigma,
                                  'ranges': {}}

        # calculate range for sigma
        range_min = max(0, timing_sigma -
                        (timing_sigma * sigma_range_multiplier / 100.0))
        range_max = (timing_sigma +
                     (timing_sigma * sigma_range_multiplier / 100.0))
        evinput_params[id_str]['ranges']['sigma_' + evinput_t] = \
            {'initial': timing_sigma, 'minval': range_min, 'maxval': range_max}
        evinput_params[id_str]['num_params'] = 1

        # calculate range for time
        timing_bound = timing_sigma * timing_range_multiplier
        range_min = max(0, value - timing_bound)
        range_max = min(float(params['tstop']), value + timing_bound)

        evinput_params[id_str]['start'] = range_min
        evinput_params[id_str]['end'] = range_max
        evinput_params[id_str]['ranges'][evinput_t] =  \
            {'initial': value, 'minval': range_min, 'maxval': range_max}
        evinput_params[id_str]['num_params'] += 1

        # calculate ranges for syn. weights
        for label in params['gbar_' + id_str + '*']:
            value = float(params[label])
            if value == 0.0:
                range_min = value
                range_max = 1.0
            else:
                range_min = max(0, value -
                                (value * synweight_range_multiplier / 100.0))
                range_max = value + (value *
                                     synweight_range_multiplier / 100.0)
            evinput_params[id_str]['ranges'][label] = \
                {'initial': value, 'minval': range_min, 'maxval': range_max}
            evinput_params[id_str]['num_params'] += 1

        continue

    sorted_evinput_params = OrderedDict(sorted(evinput_params.items(),
                                               key=lambda x: x[1]['start']))
    return sorted_evinput_params


def generate_weights(evinput_params, params, decay_multiplier):
    """Calculation of weight function for wRMSE calcuation

    Returns
    -------
    evinput_params : dict
        Adds the keys 'weights' and 'cdf' to evinput_params[input_name]
        and converts evinput_params['opt_start'] and evinput_params['opt_end']
        to be in multiples of 'dt'.
    """

    import scipy.stats as stats
    from math import ceil, floor

    num_step = ceil(params['tstop'] / params['dt']) + 1
    times = np.linspace(0, params['tstop'], num_step)

    for input_name in evinput_params.keys():
        # calculate cdf using start time (minival of optimization range)
        cdf = stats.norm.cdf(times, evinput_params[input_name]['start'],
                             evinput_params[input_name]['sigma'])
        evinput_params[input_name]['cdf'] = cdf.copy()

    for input_name in evinput_params.keys():
        evinput_params[input_name]['weights'] = \
            evinput_params[input_name]['cdf'].copy()

        for other_input in evinput_params:
            if input_name == other_input:
                # don't subtract our own cdf(s)
                continue
            if evinput_params[other_input]['mean'] < \
               evinput_params[input_name]['mean']:
                # check ordering to only use inputs after us
                continue
            else:
                decay_factor = decay_multiplier * \
                    (evinput_params[other_input]['mean'] -
                     evinput_params[input_name]['mean']) / params['tstop']
                evinput_params[input_name]['weights'] -= \
                    evinput_params[other_input]['cdf'] * decay_factor

        # weights should not drop below 0
        evinput_params[input_name]['weights'] = \
            np.clip(evinput_params[input_name]['weights'], a_min=0, a_max=None)

        # start and stop optimization where the weights are insignificant
        indices = np.where(evinput_params[input_name]['weights'] > 0.01)
        evinput_params[input_name]['opt_start'] = \
            min(evinput_params[input_name]['start'],
                times[indices][0])
        evinput_params[input_name]['opt_end'] = \
            max(evinput_params[input_name]['end'], times[indices][-1])

        # convert to multiples of dt
        evinput_params[input_name]['opt_start'] = \
            floor((evinput_params[input_name]['opt_start'] / params['dt']) *
                  params['dt'])
        evinput_params[input_name]['opt_end'] = \
            ceil((evinput_params[input_name]['opt_end'] / params['dt']) *
                 params['dt'])

    return evinput_params


def create_last_chunk(input_chunks):
    """ This creates a chunk that combines parameters for
    all chunks in input_chunks (final step)

    Parameters
    ----------
    input_chunks: List
        List of ordered chunks for optimization

    Returns
    -------
    combined_chunk: Dict
        Dictionary of with parameters for combined
        chunk (final step)
    """
    global default_num_total_sims

    combined_chunk = {'inputs': [],
                      'ranges': {},
                      'opt_start': 0.0,
                      'opt_end': 0.0}

    for evinput in input_chunks:
        combined_chunk['inputs'].extend(evinput['inputs'])
        combined_chunk['ranges'].update(evinput['ranges'])
        if evinput['opt_end'] > combined_chunk['opt_end']:
            combined_chunk['opt_end'] = evinput['opt_end']

    # wRMSE with weights of 1's is the same as regular RMSE.
    combined_chunk['weights'] = np.ones(len(input_chunks[-1]['weights']))
    combined_chunk['num_sims'] = default_num_total_sims

    return combined_chunk


def consolidate_chunks(inputs):
    """Consolidates inputs into optimization "chunks" defined by
    opt_start and opt_end

    Parameters
    ----------
    inputs: Dict
        Sorted dictionary of inputs with their parameters
        and weight functions

    Returns
    -------
    consolidated_chunks: List
        Combine the evinput_params whenever the end is overlapping
        with the next.
    """
    global default_num_step_sims

    consolidated_chunks = []
    for input_name in inputs.keys():
        input_dict = inputs[input_name].copy()
        input_dict['inputs'] = [input_name]
        input_dict['num_sims'] = default_num_step_sims

        if len(consolidated_chunks) > 0 and \
           (input_dict['start'] <= consolidated_chunks[-1]['end']):
            # update previous chunk
            consolidated_chunks[-1]['inputs'].extend(input_dict['inputs'])
            consolidated_chunks[-1]['num_params'] += input_dict['num_params']
            consolidated_chunks[-1]['end'] = input_dict['end']
            consolidated_chunks[-1]['ranges'].update(input_dict['ranges'])
            consolidated_chunks[-1]['opt_end'] = \
                max(consolidated_chunks[-1]['opt_end'], input_dict['opt_end'])
            # average the weights
            consolidated_chunks[-1]['weights'] = \
                (consolidated_chunks[-1]['weights'] +
                 input_dict['weights']) / 2
        else:
            # new chunk
            consolidated_chunks.append(input_dict)

    # add one last chunk to the end
    if len(consolidated_chunks) > 1:
        last_chunk = create_last_chunk(consolidated_chunks)
        consolidated_chunks.append(last_chunk)

    return consolidated_chunks


def optrun(new_params, grad=0):
    """ This is the function to run a simulation

    Parameters
    ----------
    new_params: Array
        List or numpy array with the parameters chosen by
        optimization engine. Order is consistent with
        opt_params['ranges']
    grad: float | None
        Gradient if provided by optimization algorithm

    Returns
    -------
    avg_rmse: float
        Weighted RMSE between data in dpl and exp_dpl
    """
    global opt_params, params, exp_dpl, best_dpl, cur_step, optiter
    global stepminopterr
    print("Optimization step %d, iteration %d" % (cur_step + 1, optiter + 1))

    # set parameters
    for param_name, test_value in zip(opt_params['ranges'].keys(), new_params):
        if test_value >= opt_params['ranges'][param_name]['minval'] and \
           test_value <= opt_params['ranges'][param_name]['maxval']:
            params[param_name] = test_value
        else:
            # This test is not strictly necessary with COBYLA, but in case the
            # algorithm is changed at some point in the future
            return 1e9  # invalid param value -> large error

    # run the simulation, but stop early if possible
    params['tstop'] = opt_params['opt_end']
    net = Network(params, add_drives_from_params=True)
    with MPIBackend(n_procs=n_procs, mpi_cmd=mpi_cmd):
        dpls = simulate_dipole(net, n_trials=1)
    # avg_dpl = average_dipoles(dpls)
    avg_dpl = dpls[0].copy()
    avg_rmse = rmse(avg_dpl, exp_dpl,
                    tstart=opt_params['opt_start'],
                    tstop=opt_params['opt_end'],
                    weights=opt_params['weights'])

    if avg_rmse < stepminopterr:
        best = "[best] "
        stepminopterr = avg_rmse
        best_dpl = avg_dpl
    else:
        best = ""

    print("%sweighted RMSE: %.2f over range [%3.3f-%3.3f] ms" %
          (best, avg_rmse, opt_params['opt_start'], opt_params['opt_end']))

    optiter += 1

    return avg_rmse  # nlopt expects error


def run_optimization(seed=0):
    """ Start the nlopt optimization
    Parameters
    ----------
    seed: int | None
        Seed for RNG to make optimization results reproducible

    Returns
    -------
    opt_results: List
        List with the values of the best parameters found by
        optimization engine. Order is consistent with
        opt_params['ranges']
    """
    import nlopt
    global opt_params

    nlopt.srand(seed)
    algorithm = nlopt.LN_COBYLA

    num_params = len(opt_params['ranges'])
    opt = nlopt.opt(algorithm, num_params)
    params_arr = np.zeros(num_params)
    lb = np.zeros(num_params)
    ub = np.zeros(num_params)

    for idx, param_name in enumerate(opt_params['ranges'].keys()):
        ub[idx] = opt_params['ranges'][param_name]['maxval']
        lb[idx] = opt_params['ranges'][param_name]['minval']
        params_arr[idx] = opt_params['ranges'][param_name]['initial']

    if algorithm == nlopt.G_MLSL_LDS or algorithm == nlopt.G_MLSL:
        # In case these mixed mode (global + local) algorithms are used in the
        # future
        local_opt = nlopt.opt(nlopt.LN_COBYLA, num_params)
        opt.set_local_optimizer(local_opt)

    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_min_objective(optrun)
    opt.set_xtol_rel(1e-4)
    opt.set_maxeval(opt_params['num_sims'])
    opt_results = opt.optimize(params_arr)

    return opt_results


###############################################################################
# Load experimental data into Dipole object. Data can be retrieved from
# https://raw.githubusercontent.com/jonescompneurolab/hnn/master/data/MEG_detection_data/S1_SupraT.txt
#
# This is a different experiment than the one to which the base parameters were
# tuned. So, the initial RMSE will be large, giving the optimization procedure
# a lot to work with.
extdata = np.loadtxt('S1_SupraT.txt')
exp_dpl = Dipole(extdata[:, 0], np.c_[extdata[:, 1], extdata[:, 1], extdata[:, 1]])

###############################################################################
# Read the base parameters from a file
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)

###############################################################################
# Create a sorted dictionary with the inputs and parameters belonging to each.
# Then calculate the appropriate weight function to be used in RMSE using the
# CDF defined by the input timing's mean and std. deviation parameters. Lastly,
# use the weights to define non-overlapping "chunks" of the simulation
# timeframe to optimize. Chunks are consolidated if more than one input should
# be optimized at a time.
evinput_params = split_by_evinput(params)
evinput_params = generate_weights(evinput_params, params, decay_multiplier)
param_chunks = consolidate_chunks(evinput_params)

sdfdfdf

###############################################################################
# Start the optimization!

print("Running simulation with initial parameters")
net = Network(params, add_drives_from_params=True)
with MPIBackend(n_procs=n_procs, mpi_cmd=mpi_cmd):
    dpls = simulate_dipole(net, n_trials=1)
# initial_dpl = average_dipoles(dpls)
initial_dpl = dpls.copy()
avg_rmse = rmse(initial_dpl[0], exp_dpl, tstop=params['tstop'])
print("Initial RMSE: %.2f" % avg_rmse)

for step in range(len(param_chunks)):
    cur_step = step
    total_steps = len(param_chunks)

    # param_chunks is the optimization information for all steps.
    # opt_params is a pointer to the params for each step
    opt_params = param_chunks[step]

    if opt_params['num_sims'] == 0:
        print("Skipping optimization step %d (0 simulations)" % (step + 1))
        continue

    if cur_step > 0 and cur_step == total_steps - 1:
        # update currently used params
        for var_name in opt_params['ranges']:
            params[var_name] = opt_params['ranges'][var_name]['initial']

        # For the last step (all inputs), recalculate ranges and update
        # param_chunks. If previous optimization steps increased
        # std. dev. this could result in fewer optimization steps as
        # inputs may be deemed too close together and be grouped in a
        # single optimization step.
        #
        # The purpose of the last step (with regular RMSE) is to clean up
        # overfitting introduced by local weighted RMSE optimization.

        evinput_params = split_by_evinput(params)
        evinput_params = generate_weights(evinput_params, params, decay_multiplier)
        param_chunks = consolidate_chunks(evinput_params)

        # reload opt_params for the last step in case the number of
        # steps was changed by updateoptparams()
        opt_params = param_chunks[total_steps - 1]

    print("Starting optimization step %d/%d" % (step + 1, total_steps))

    optiter = 0
    stepminopterr = minopterr
    best_dpl = None

    print('Optimizing from [%3.3f-%3.3f] ms' % (opt_params['opt_start'],
                                                opt_params['opt_end']))
    opt_results = run_optimization(seed=0)

    # update opt_params for the next round
    for var_name, value in zip(opt_params['ranges'], opt_results):
        opt_params['ranges'][var_name]['initial'] = value

# save the optimized params
for var_name in opt_params['ranges']:
    params[var_name] = opt_params['ranges'][var_name]['initial']

# TODO: write the optimized param file

avg_rmse = rmse(best_dpl, exp_dpl, tstop=params['tstop'])
print("Final RMSE: %.2f" % avg_rmse)

###############################################################################
# Now plot the results against experimental data:
# 1. Initial dipole
# 2. Optimized dipole fit
#
# Show the input histograms as well

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

exp_dpl.plot(ax=axes[0], layer='agg', show=False)
initial_dpl.plot(ax=axes[0], layer='agg', show=False)
best_dpl.plot(ax=axes[0], layer='agg', show=False)
net.plot_input(ax=axes[1])
