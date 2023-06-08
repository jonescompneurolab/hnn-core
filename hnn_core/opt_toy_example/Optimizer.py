import os.path as op
import numpy as np
import matplotlib.pyplot as plt 
# import copy.deepcopy as dp 

import hnn_core 
from hnn_core import read_params, jones_2009_model, simulate_dipole 
from hnn_core.network import pick_connection
from hnn_core import MPIBackend

from scipy.optimize import fmin_cobyla
from skopt import gp_minimize 
from skopt.utils import use_named_args
from skopt.space import Real

class Optimizer:
    
    def __init__(self, net, target_dpl, initial_dpl, constraints, res_type, optimizer_type):
        self.net = net
        self.target_dpl = target_dpl
        self.initial_dpl = initial_dpl
        self.constraints = constraints
        self.res_type = res_type 
        self.optimizer_type = optimizer_type
        self.predicted_params = dict()
        self.error_values = list()
        self.b_final_iter = False
        self.max_iter = 200
        
    def _get_drive_params(self, net, drive_names):
        """Get drive parameters (dynamics, weights, and static) from a Network instance."""
        
        drive_dynamics = list() # [{'mu': x, 'sigma': x, 'numspikes': x}]
        drive_syn_weights_ampa = list() # [{'L2_pyramidal': x, 'L2_basket': x, 'L5_pyramidal': x}]
        drive_syn_weights_nmda = list() 
        drive_static_params = dict()  
        # {'evprox1': {'numspikes': 1,
        #             'location': 'proximal',
        #             'n_drive_cells': 'n_cells',
        #             'cell_specific': True,
        #             'space_constant': {'L2_pyramidal': x,
        #                              'L2_basket': x,
        #                              'L5_basket': x,
        #                              'L5_pyramidal': x},
        #             'synaptic_delays': {'L5_basket': x,
        #                                'L5_pyramidal': x,
        #                                'L2_basket': x,
        #                                'L2_pyramidal': x},
        #             'probability': {'L5_basket': 1.0,
        #                            'L5_pyramidal': 1.0,
        #                            'L2_basket': 1.0,
        #                            'L2_pyramidal': 1.0},
        #             'event_seed': 2,
        #             'conn_seed': 3}}

        for drive_name in drive_names:
            drive = net.external_drives[drive_name]
            connection_idxs = pick_connection(net, src_gids = drive_name) # connection (the prox input targets 4 sources [16, 17, 18, 19])
            
            weights_ampa = dict()
            weights_nmda = dict()
            delays = dict()
            probabilities = dict()
            static_params = dict()

            for connection_idx in connection_idxs:
                # source is drive
                target_type = net.connectivity[connection_idx]['target_type']
                target_receptor = net.connectivity[connection_idx]['receptor']
                connection_weight = net.connectivity[connection_idx]['nc_dict']['A_weight'] # A_delay, A_weight, lamtha, threshold
                delay = net.connectivity[connection_idx]['nc_dict']['A_delay']
                space_constant = net.connectivity[connection_idx]['nc_dict']['lamtha']
                probability = net.connectivity[connection_idx]['probability']
                
                # add weight 
                if target_receptor == 'ampa':
                    weights_ampa.update({f'{target_type}': connection_weight})
                elif target_receptor == 'nmda':
                    weights_nmda.update({f'{target_type}': connection_weight})
                # add delay 
                delays.update({f'{target_type}': delay})
                # add probability
                probabilities.update({f'{target_type}': probability})
                
            drive_dynamics.append(drive['dynamics'].copy()) # dynamics
            drive_syn_weights_ampa.append(weights_ampa) # synaptic weights 
            drive_syn_weights_nmda.append(weights_nmda) 

            # static params     
            static_params.update({'numspikes': drive['dynamics']['numspikes']})
            static_params.update({'location': drive['location']})
            # True : each artificial drive cell has 1-to-1 connection parameters (number of drive cells = number of all available cells that this drive can target)
            # False : each artificial drive cell has all-to-all (synchronous input)
            if drive['cell_specific']:
                static_params.update({'n_drive_cells': 'n_cells'})
            else:
                static_params.update({'n_drive_cells': drive['n_drive_cells']})
            static_params.update({'cell_specific': drive['cell_specific']})
            static_params.update({'space_constant': space_constant})
            static_params.update({'synaptic_delays': delays})
            static_params.update({'probability': probabilities})
            static_params.update({'event_seed': drive['event_seed']})
            static_params.update({'conn_seed': drive['conn_seed']})
            drive_static_params.update({drive_name: static_params})

        return drive_dynamics, drive_syn_weights_ampa, drive_syn_weights_nmda, drive_static_params

    def _assemble_predicted_params(self, drive_names, drive_syn_weights_ampa, drive_syn_weights_nmda, **params): 
        """Assembles constraints.
        
            Returns
            -------
            predicted_params : a dictionary ()
                {'evprox1': {'mu': x,
                            'sigma': x,
                            'ampa_weights': {'L2_pyramidal': x,
                                              'L2_basket': x,
                                              'L5_basket': x,
                                              'L5_pyramidal': x},
                            'nmda_weights': {'L2_pyramidal': x,
                                              'L2_basket': x,
                                              'L5_basket': x,
                                              'L5_pyramidal': x}
                            }
            
        """
        
        self.predicted_params = dict()
        _params = dict()
        
        for drive_idx, drive_name in enumerate(drive_names):
            _params.update({'mu': params[f'{drive_name}_mu']})
            _params.update({'sigma': params[f'{drive_name}_sigma']})
            
            ampa_weights = dict()
            nmda_weights = dict()
            for target_key in drive_syn_weights_ampa[drive_idx]: # [{'L2_pyramidal': x, 'L2_basket': x, 'L5_pyramidal': x}]
                ampa_weights.update({target_key: params[f'{drive_name}_ampa_{target_key}']})
            for target_key in drive_syn_weights_nmda[drive_idx]: 
                nmda_weights.update({target_key: params[f'{drive_name}_nmda_{target_key}']})
                
            _params.update({'ampa_weights': ampa_weights})  
            _params.update({'nmda_weights': nmda_weights})   
            self.predicted_params.update({drive_name: _params})
  
        return self.predicted_params
        
    def _simulate_dipole(self, **params): 
        """This function runs a simulation."""
        tstop = 100. 
        n_trials = 1
        
        net = self.net.copy()
                
        # get current dynamics, weights, and static params
        drive_names = [key for key in net.external_drives.keys() if net.external_drives[key]['type'] == 'evoked'] 
        drive_dynamics, drive_syn_weights_ampa, drive_syn_weights_nmda, drive_static_params = self._get_drive_params(net, drive_names) 
        
        if self.b_final_iter: # final params
            predicted_params = params
        else:
            # assemble predicted params
            predicted_params = self._assemble_predicted_params(drive_names, drive_syn_weights_ampa, drive_syn_weights_nmda, **params)

        # remove existing drives, re-add them with updated params 
        for idx, drive_name in enumerate(drive_names):
            # clear drive 
            del net.external_drives[drive_name] 
            # clear connectivity
            conn_idxs = pick_connection(net, src_gids = drive_name)
            net.connectivity = [conn for conn_idx, conn in enumerate(net.connectivity) if conn_idx not in conn_idxs]

            net.add_evoked_drive(name = drive_name,
                                  mu = predicted_params[drive_name]['mu'], 
                                  sigma = predicted_params[drive_name]['sigma'], 
                                  numspikes = drive_dynamics[idx]['numspikes'],
                                  location = drive_static_params[drive_name]['location'],
                                  n_drive_cells = drive_static_params[drive_name]['n_drive_cells'],
                                  cell_specific = drive_static_params[drive_name]['cell_specific'],
                                  weights_ampa = predicted_params[drive_name]['ampa_weights'], 
                                  weights_nmda = predicted_params[drive_name]['nmda_weights'], 
                                  space_constant = drive_static_params[drive_name]['space_constant'],
                                  synaptic_delays = drive_static_params[drive_name]['synaptic_delays'],
                                  probability = drive_static_params[drive_name]['probability'],
                                  event_seed = drive_static_params[drive_name]['event_seed'],
                                  conn_seed = drive_static_params[drive_name]['conn_seed'])
        
        dpl = simulate_dipole(net, tstop=tstop, n_trials=n_trials)[0] 
        return dpl
        
    def _objective_function(self, **params):
        """Objective function depends on response being optimized, whether evoked or rhythmic."""
        
        error_values = list()
        
        # Normalized RMSE
        if self.res_type == 'evoked':
            
            dpl_estimate = self._simulate_dipole(**params) 
            nrmse = np.sqrt( ((dpl_estimate.data['agg'] - self.target_dpl.data['agg'])**2).sum() / len(dpl_estimate.times) ) / (max(self.target_dpl.data['agg']) - min(self.target_dpl.data['agg']))
            error_values.append(nrmse)
            return nrmse   
        
        # ...
        elif self.res_type == 'rhythmic':
            return 
        
    def _assemble_inputs(self, net):
        """Assembles constraints & initial parameters for appropriate optimizer.
        
        Retruns
        -------
        constraints : a (lower_bound, upper_bound, "prior") tuple (if Bayesian Optimizer)

            constraints = [Real(name='evprox1_mu', low=5, high=10),
                           Real(name='evprox1_sigma', low=0.01, high=1),
                           Real(name='evprox1_nmda_weights_target', low=0.01, high=1)]
            
            
        initial_params : a list of single initial input points (if Bayesian Optimizer)
        
            initial_params = [mu, sigma, ampa, nmda, ...]
        """
        
        cons = list() 
        initial_params = list() 
        
        # get drive names 
        drive_names = [key for key in net.external_drives.keys() if net.external_drives[key]['type'] == 'evoked'] # ... rhythmic
        # get current drive params
        drive_dynamics, drive_syn_weights_ampa, drive_syn_weights_nmda, drive_static_params = self._get_drive_params(net, drive_names)
        
        #--------------------------------------------------Bayesian---------------------------------------------------
        if self.optimizer_type == 'bayesian':
            raw_constraints = self.constraints # {'evprox1': {'mu': [5, 10], 'sigma': [1, 5], 'weights': [0.01, 1]}}
            
            for drive_idx, drive_name in enumerate(drive_names):
                for key in raw_constraints[drive_name]: # key is 'mu', 'sigma', 'weights' (in order)
                
                    # add a cons for each target type 
                    if key == 'weights':
                        for target_key in drive_syn_weights_ampa[drive_idx]: # [{'L2_pyramidal': x, 'L2_basket': x, 'L5_pyramidal': x}]
                            cons.append(Real(name=f'{drive_name}_ampa_{target_key}', low = raw_constraints[drive_name][key][0], high = raw_constraints[drive_name][key][1]))
                            initial_params.append(drive_syn_weights_ampa[drive_idx][target_key])

                        for target_key in drive_syn_weights_nmda[drive_idx]: 
                            cons.append(Real(name=f'{drive_name}_nmda_{target_key}', low = raw_constraints[drive_name][key][0], high = raw_constraints[drive_name][key][1]))
                            initial_params.append(drive_syn_weights_nmda[drive_idx][target_key])
                    else:       
                        cons.append(Real(name=f'{drive_name}_{key}', low = raw_constraints[drive_name][key][0], high = raw_constraints[drive_name][key][1])) 
                        initial_params.append(drive_dynamics[drive_idx][key])
                                                                                                                                 
            return cons, initial_params
        
        #--------------------------------------------------COBYLA---------------------------------------------------
        elif self.optimizer_type == 'cobyla':
            return 
    
    def optimize_response(self):
        """Optimize drives to generate evoked or rhythmic response.
        
            Parameters
            ----------
                constraints : {'evprox1': {'mu': [min, max], 'sigma': [min, max], 'weights': [min, max]}}
        """
        
        net = self.net.copy()
        optimization_results = list()
        # assemble constraints & initial_params
        cons, initial_params = self._assemble_inputs(net)
        
        #--------------------------------------------------Bayesian---------------------------------------------------
        if self.optimizer_type == 'bayesian':
            
            @use_named_args(dimensions = cons) 
            def _obj_func(**params):
                return self._objective_function(**params)  
            
            optimization_results = gp_minimize(func = _obj_func,
                                               dimensions = cons, 
                                               acq_func = 'EI',
                                               n_calls = self.max_iter,
                                               x0 = initial_params,
                                               random_state = 64)    
            # params_opt = optimization_results.x
            self.b_final_iter = True
            self.opt_dpl = self._simulate_dipole(**self.predicted_params) 
            self.error_values = [np.min(optimization_results.func_vals[:i]) for i in range(1, self.max_iter + 1)]
            return self.predicted_params, self.opt_dpl, self.error_values
        
        #--------------------------------------------------COBYLA---------------------------------------------------
        elif self.optimizer_type == 'cobyla':
            return
        
    def plot_results(self):
        """Plots target, initial, and optimized dipoles. Plots convergence."""
        # fig
        fig, axs = plt.subplot_mosaic([['left'],['right']], constrained_layout = True, figsize = (8, 5))
        ax1 = axs['left']
        ax2 = axs['right']
        x_axis = list(range(1, self.max_iter + 1))
        y_max = max(self.error_values) + 0.01

        # bayesian 
        ax1.plot(self.target_dpl.times, self.target_dpl.data['agg'], color = 'black')
        ax1.plot(self.initial_dpl.times, self.initial_dpl.data['agg'], color = 'tab:blue')
        ax1.plot(self.opt_dpl.times, self.opt_dpl.data['agg'], color = 'orange')
        ax1.set_title('Dipoles')
        ax1.set_ylabel('Dipole moment (nAm)')
        ax1.legend(['true', 'offset', 'opt'], frameon=False, loc = 'upper right')

        # convergence 
        ax2.plot(x_axis, self.error_values, color='tab:green')
        ax2.set_ylim([-.01, y_max])
        ax2.set_title('Convergence')
        ax2.set_xlabel('Number of calls')
        ax2.set_ylabel('NRMSE')
        ax2.grid(visible = True)

        fig.savefig('opt_bayesian.svg')
         
