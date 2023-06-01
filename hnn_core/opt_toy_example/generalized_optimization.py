# Author:  
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import hnn_core
from hnn_core import read_params, jones_2009_model, simulate_dipole
# from hnn_core import JoblibBackend
from hnn_core import MPIBackend

from scipy.optimize import fmin_cobyla
from skopt import gp_minimize 
# from skopt.plots import plot_convergence, plot_objective 
# from hnn_core.optimization import optimize_evoked

def reduced_network():
    # load default params
    hnn_core_root = op.dirname(hnn_core.__file__)
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname) 
    
    # update params
    params.update({'N_pyr_x': 3, 'N_pyr_y': 3})
    net_obj = jones_2009_model(params, add_drives_from_params=True)
    net_obj.clear_drives()
    return net_obj

def sim_dipole(net_obj, params):
    tstop = 100.
    n_trials = 1
    
    # prox input
    synaptic_delays_prox = {'L2_basket': 0.1, 
                       'L2_pyramidal': 0.1,
                       'L5_basket': 1., 
                       'L5_pyramidal': 1.}
   
    weights_ampa_prox = {'L2_basket': params[2],
                    'L2_pyramidal': params[3],
                    'L5_basket': params[4],
                    'L5_pyramidal': params[5]}
    
    net_obj.add_evoked_drive('evprox1', 
                             mu=params[0], 
                             sigma=2, 
                             numspikes=1, 
                             location='proximal', 
                             weights_ampa=weights_ampa_prox,
                             synaptic_delays=synaptic_delays_prox)
    
    # dist input
    synaptic_delays_dist = {'L2_basket': 0.1, 
                       'L2_pyramidal': 0.1,
                       'L5_pyramidal': 1.}
    
    weights_ampa_dist = {'L2_basket': params[6],
                    'L2_pyramidal': params[7],
                    'L5_pyramidal': params[8]}

    net_obj.add_evoked_drive('evdist1', 
                             mu=params[1], 
                             sigma=2, 
                             numspikes=1, 
                             location='distal', 
                             weights_ampa=weights_ampa_dist,
                             synaptic_delays=synaptic_delays_dist)
    with MPIBackend(n_procs=2, mpi_cmd='mpiexec'):
        dpl = simulate_dipole(net_obj, tstop=tstop, n_trials=n_trials)[0] 
    return dpl

error_values_cobyla = list()
def calculate_rmse(params_predicted):
    dpl_estimate = sim_dipole(net.copy(), params_predicted)
    rmse = np.sqrt( ((dpl_estimate.data['agg'] - dpl_true.data['agg'])**2).sum() / len(dpl_estimate.times) ) / (max(dpl_true.data['agg']) - min(dpl_true.data['agg']))
    error_values_cobyla.append(rmse)
    print('New params: ', params_predicted)
    print('NRMSE: ', rmse)
    return rmse

# network
net = reduced_network()

# cell dynamics and synaptic weights 
# params_true = {"prox_mu": [10], "dist_mu": [30], 
#           "prox_ampa": [0.1, 0.4, 0.2, 0.6], "dist_ampa": [0.3, 0.05, 0.7, 0.8]}
# offset parameters 
# params_offset = deepcopy(params_true) 
# params_offset["prox_mu"][0] += 5
# params_offset["dist_mu"][0] -= 5
# params_offset["prox_ampa"][0] += 0.4
# params_offset["prox_ampa"][2] -= 0.1
# params_offset["dist_ampa"][1] += 0.3
# params_offset["dist_ampa"][3] -= 0.5

params_true = [10, 30, 0.12, 0.4, 0.2, 0.6, 0.3, 0.13, 0.8]
params_offset = [15, 25, 0.5, 0.4, 0.15, 0.6, 0.3, 0.35, 0.3]

dpl_true = sim_dipole(net.copy(), params_true)
dpl_offset = sim_dipole(net.copy(), params_offset)  

# constraints 
max_iter = 200
# cons = [(0., 1)] * 4
cons = [(0.0, 20.), (0.0, 50.), (0.0, 1), (0.0, 1), (0.0, 1), (0.0, 1), (0.0, 1), (0.0, 1), (0.0, 1)]
#----------------------------------------------Bayesian optimization------------------------------------     
with MPIBackend(n_procs=2, mpi_cmd='mpiexec'):
    bayesian_optimization_results = gp_minimize(calculate_rmse,
                                            cons,
                                            acq_func='EI',
                                            n_calls=max_iter,
                                            x0=params_offset,
                                            random_state=64)    
params_bayesian = bayesian_optimization_results.x
dpl_bayesian = sim_dipole(net.copy(), params_bayesian)
    
print('Bayesian Optimization Results \n-----------------------------')
print('True params: ', params_true, '\nOffset params: ', params_offset, '\nOptimized params: ', params_bayesian)
# plot_convergence(bayesian_optimization_results)
error_values_bayesian = [np.min(bayesian_optimization_results.func_vals[:i]) for i in range(1, max_iter + 1)]

#--------------------------------------------COBYLA optimization--------------------------------------- 
# constraints 
cons_func = list()
x0 = list()
cons_func.append(lambda x, idx=0: 20. - np.asarray(x[0]))
cons_func.append(lambda x, idx=0: np.asarray(x[0]) - 0.01)
cons_func.append(lambda x, idx=1: 50. - np.asarray(x[0]))
cons_func.append(lambda x, idx=1: np.asarray(x[0]) - 0.01)
for idx in range(2,9):
    cons_func.append(lambda x, idx=idx: 1. - np.asarray(x[0]))
    cons_func.append(lambda x, idx=idx: np.asarray(x[0]) - 0.01)
x0.append(params_offset)

error_values_cobyla.clear()
with MPIBackend(n_procs=2, mpi_cmd='mpiexec'):
    cobyla_optimization_results = fmin_cobyla(calculate_rmse,
                                          cons=cons_func,
                                          rhobeg=0.1,
                                          rhoend=1e-4,
                                          x0=x0,
                                          maxfun=max_iter,
                                          catol=0.0)    
dpl_cobyla = sim_dipole(net.copy(), cobyla_optimization_results[0])

print('COBYLA Optimization Results \n---------------------------')
print('True params: ', params_true, '\nOffset params: ', params_offset, '\nOptimized params: ', cobyla_optimization_results[0])

#--------------------------------------------Current optimization---------------------------------------
# with MPIBackend(n_procs=2):
#     net_opt = optimize_evoked(net, tstop=100, n_trials=1,
#                               target_dpl=dpl_true, initial_dpl=dpl_offset)
# with MPIBackend(n_procs=2):
#     best_dpl = sim_dipole(net_opt, tstop=100, n_trials=1)[0]

#--------------------------------------------Plot-------------------------------------------------------

# fig
fig, axs = plt.subplot_mosaic([['left', 'right'],['bottom', 'bottom']], constrained_layout=False)
ax1 = axs['left']
ax2 = axs['right']
ax3 = axs['bottom']
x_axis = list(range(1, max_iter + 1))
y_max = max([max(error_values_bayesian), max(error_values_cobyla)]) + min([min(error_values_bayesian), min(error_values_cobyla)])

# bayesian 
ax1.plot(dpl_true.times, dpl_true.data['agg'], color = 'black')
ax1.plot(dpl_offset.times, dpl_offset.data['agg'], color = 'tab:blue')
ax1.plot(dpl_bayesian.times, dpl_bayesian.data['agg'], color = 'orange')
ax1.set_title('Bayesian optimization')
ax1.set_ylabel('Dipole moment (nAm)')
ax1.legend(['true', 'offset', 'opt'], frameon=False, loc = 'upper right')

# COBYLA
ax2.plot(dpl_true.times, dpl_true.data['agg'], color = 'black')
ax2.plot(dpl_offset.times, dpl_offset.data['agg'], color = 'tab:blue')
ax2.plot(dpl_cobyla.times, dpl_cobyla.data['agg'], color = 'orange')
ax2.set_title('COBYLA optimization')
ax2.legend(['true', 'offset', 'opt'], frameon=False, loc = 'upper right')

# convergence 
ax3.plot(x_axis, error_values_bayesian, color='tab:green')
ax3.plot(error_values_cobyla, color='tab:purple')
# ax3.plot(len(error_values_cobyla), 0, marker = 'o', markersize = 5, markeredgecolor = 'tab:purple', markerfacecolor = 'tab:purple')
ax3.text(125, 0.1, 'BAYESIAN: ' + str(round(min(error_values_bayesian), 4)), style ='italic', fontsize = 15, color = 'tab:green')
ax3.text(125, 0.15, 'COBYLA: ' + str(round(min(error_values_cobyla), 4)), style = 'italic', fontsize = 15, color = 'tab:purple')
ax3.set_ylim([-.01, y_max])
ax3.set_title('Convergence')
ax3.set_xlabel('Number of calls')
ax3.set_ylabel('NRMSE')
ax3.grid(visible = True)

fig.savefig('/Users/suanypujol/Desktop/development/fig/opt.svg')
