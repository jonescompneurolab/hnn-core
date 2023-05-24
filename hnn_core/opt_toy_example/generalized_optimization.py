# Author:  

import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import hnn_core
from hnn_core import read_params, jones_2009_model, simulate_dipole
from skopt import gp_minimize 
from skopt.plots import plot_convergence, plot_objective 

from scipy.optimize import fmin_cobyla

error_values_cobyla = list()

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

def sim_dipole(net_obj, mu):
    tstop = 10.
    n_trials = 1
    synaptic_delays = {'L2_basket': 0.1, 
                       'L2_pyramidal': 0.1,
                       'L5_basket': 1., 
                       'L5_pyramidal': 1.}
    weights_ampa = {'L2_basket': 0.5,
                    'L2_pyramidal': 0.5,
                    'L5_basket': 0.5,
                    'L5_pyramidal': 0.5}
    net_obj.add_evoked_drive('evprox1', 
                             mu=mu, 
                             sigma=2, 
                             numspikes=1, 
                             location='proximal', 
                             weights_ampa=weights_ampa,
                             synaptic_delays=synaptic_delays)
    dpl = simulate_dipole(net_obj, tstop=tstop, n_trials=n_trials)[0] 
    return dpl
    
def calculate_rmse(mu_predicted):
    dpl_estimate = sim_dipole(net.copy(), mu_predicted)
    rmse = np.sqrt( ((dpl_estimate.data['agg'] - dpl_true.data['agg'])**2).sum() / len(dpl_estimate.times) )
    error_values_cobyla.append(rmse)
    print('New mu: ', mu_predicted, ' RMSE: ', rmse)
    return rmse

    
net = reduced_network()
mu_true = 2
mu_offset = 4
dpl_true = sim_dipole(net.copy(), mu_true)
dpl_offset = sim_dipole(net.copy(), mu_offset)    
max_iter = 15

#----------------------------------------------Bayesian optimization------------------------------------     
bayesian_optimization_results = gp_minimize(calculate_rmse,
                                            [(0.01, 10.)],
                                            acq_func='EI',
                                            n_calls=max_iter,
                                            x0=[mu_offset],
                                            random_state=64)    

mu_bayesian = bayesian_optimization_results.x[0]
dpl_bayesian = sim_dipole(net.copy(), mu_bayesian)
    
print('Bayesian Optimization Results \n-----------------------------')
print('True mu: ', mu_true, '\nOffset mu: ', mu_offset, '\nOptimized mu: ', mu_bayesian)
# plot_convergence(bayesian_optimization_results)
error_values_bayesian = [np.min(bayesian_optimization_results.func_vals[:i]) for i in range(1, max_iter + 1)]

#--------------------------------------------COBYLA optimization--------------------------------------- 
cons = list()
x0 = list()
cons.append(lambda x, idx=0: 10. - x[0])
cons.append(lambda x, ind=0: x[0] - 0.01)
x0.append(mu_offset)
error_values_cobyla.clear()
cobyla_optimization_results = fmin_cobyla(calculate_rmse,
                                          cons=cons,
                                          rhobeg=0.1,
                                          rhoend=1e-4,
                                          x0=x0,
                                          maxfun=max_iter,
                                          catol=0.0)    
dpl_cobyla = sim_dipole(net.copy(), cobyla_optimization_results[0])

print('COBYLA Optimization Results \n---------------------------')
print('True mu: ', mu_true, '\nOffset mu: ', mu_offset, '\nOptimized mu: ', cobyla_optimization_results[0])

#--------------------------------------------Plot-------------------------------------------------------
x_axis = list(range(1, max_iter + 1))
fig, axs = plt.subplots(2, 2, figsize=(10,4))
ax1 = axs[0, 0]
ax2 = axs[0, 1]
ax3 = axs[1, 0]
ax4 = axs[1, 1]
y_max = max([max(error_values_bayesian), max(error_values_cobyla)]) + min([min(error_values_bayesian), min(error_values_cobyla)])

# bayesian 
dpl_true.plot(ax=ax1, color='black')
dpl_offset.plot(ax=ax1, color='blue')
dpl_bayesian.plot(ax=ax1, color='orange')
ax1.set_title('Bayesian optimization')
ax1.legend(['true', 'offset', 'opt'], frameon=False)
ax3.plot(x_axis, error_values_bayesian, color='black')
ax3.set_title('Convergence')
ax3.set_xlabel('Number of calls')
ax3.set_ylabel('RMSE')
ax3.set_ylim([0, y_max])
ax3.text(9, 0.007, 'RMSE: ' + str(round(min(error_values_bayesian), 4)), style ='italic', fontsize = 15, color = 'red')

# COBYLA
dpl_true.plot(ax=ax2, color='black')
dpl_offset.plot(ax=ax2, color='blue')
dpl_cobyla.plot(ax=ax2, color='orange')
ax2.set_title('COBYLA optimization')
ax4.plot(x_axis, error_values_cobyla, color='black')
ax4.set_title('Convergence')
ax4.set_xlabel('Number of calls')
ax4.set_ylabel('RMSE')
ax4.set_ylim([0, y_max])
ax4.text(9, 0.007, 'RMSE: ' + str(round(min(error_values_cobyla), 4)), style = 'italic', fontsize = 15, color = 'red')

plt.tight_layout()

