import numpy as np
import matplotlib.pyplot as plt 
from collections import namedtuple

from hnn_core import read_params, jones_2009_model, simulate_dipole 
from hnn_core.network import pick_connection

from skopt import gp_minimize 
from skopt.utils import use_named_args
from skopt.space import Real

from scipy.optimize import fmin_cobyla

class Optimizer:
    def __init__(self, net, constraints, solver, metric):
        self.net = net
        self.constraints = constraints
        if solver == 'bayesian':
            self._get_params = _get_params_bayesian
            self._run_opt = _run_opt_bayesian
        elif solver == 'cobyla':
            self._get_params = _get_params_cobyla
            self._run_opt = _run_opt_cobyla
        if metric == 'evoked':
            self.metric = _rmse_evoked
        elif metric == 'rhythmic':
            self.metric = _rmse_rhythmic
        self.net_ = None
        self.error_values = None
        self.opt_params = None
        
    def fit(self, target_statistic, window_len = None):
        """ ...
            Parameters
            ----------
            target_statistic : dpl
            window_len : for smoothing dpl
            """
            
        init_params, cons = self._get_params(self.net, self.constraints)  
        opt_params, error_values = self._run_opt(self.net, init_params, cons, self.metric, target_statistic, window_len)
        
        self.opt_params = opt_params
        self.error_values = error_values
        # self.net_ = _set_params(opt_params)
        return 
        
    def plot_convergence():
        return 
    
    def plot_param_search():
        return 
        
# gets called only once 
def _get_params_bayesian(net, constraints):
    """Assembles constraints & initial parameters as required by gp_minimize. 
    
    Parameters
    ----------
    net : the Network object
    constraints : the user-defined constraints
       
    Returns 
    -------
    init_params : list
        [{drive_name}_mu, {drive_name}_sigma, {drive_name}_ampa_target, {drive_name}_nmda_target] 
    cons : list
        [Real(name='{drive_name}_mu', low=x, high=x),
        Real(name='{drive_name}_sigma', low=x, high=x),
        Real(name='{drive_name}_ampa_target', low=x, high=x)]
        Real(name='{drive_name}_nmda_target', low=x0.01, high=x)]
    """
    
    init_params = list()
    cons = list()
    for drive_name in net.external_drives.keys():
        for cons_key in constraints[drive_name]: 
            if cons_key == 'mu':
                cons.append(Real(name=f'{drive_name}_{cons_key}', low = constraints[drive_name][cons_key][0], high = constraints[drive_name][cons_key][1])) 
                init_params.append(net.external_drives[drive_name]['dynamics']['mu'])
            elif cons_key == 'sigma':
                cons.append(Real(name=f'{drive_name}_{cons_key}', low = constraints[drive_name][cons_key][0], high = constraints[drive_name][cons_key][1])) 
                init_params.append(net.external_drives[drive_name]['dynamics']['sigma'])   
            elif cons_key == 'weights':
                conn_idxs = pick_connection(net, src_gids = drive_name) 
                for conn_idx in conn_idxs: 
                    target_type = net.connectivity[conn_idx]['target_type']
                    target_receptor = net.connectivity[conn_idx]['receptor']
                    cons.append(Real(name=f'{drive_name}_{target_receptor}_{target_type}', low = constraints[drive_name][cons_key][0], high = constraints[drive_name][cons_key][1]))        
                    init_params.append(net.connectivity[conn_idx]['nc_dict']['A_weight'])
    return init_params, cons

# gets called only once 
def _get_params_cobyla(net, constraints):
    """Assembles constraints & initial parameters as required by fmin_cobyla. 
       
        Returns 
        -------
        init_params : list
        cons : list
    """
    
    init_params = list()
    cons = list()
        
    for drive_name in net.external_drives.keys(): 
        for cons_idx, cons_key in enumerate(constraints[drive_name]): 
            if cons_key == 'mu' or cons_key == 'sigma':
                cons.append(lambda x, idx = cons_idx: constraints[drive_name][cons_key][1] - x[idx])
                cons.append(lambda x, idx = cons_idx: constraints[drive_name][cons_key][0] - x[idx])
                init_params.append(net.external_drives[drive_name]['dynamics'][cons_key])
            elif cons_key == 'weights':
                conn_idxs = pick_connection(net, src_gids = drive_name) 
                for conn_idx in conn_idxs: 
                    cons.append(lambda x, idx = cons_idx: constraints[drive_name][cons_key][1] - x[idx])
                    cons.append(lambda x, idx = cons_idx: constraints[drive_name][cons_key][0] - x[idx])
                    init_params.append(net.connectivity[conn_idx]['nc_dict']['A_weight'])
    return init_params, cons

def _run_opt_bayesian(net, init_params, cons, metric, target_statistic, window_len):
    max_iter = 11
    @use_named_args(dimensions = cons) 
    def _obj_func(**params):
        return metric(net, target_statistic, window_len, **params)  
    
    opt_results = gp_minimize(func = _obj_func,
                                       dimensions = cons, 
                                       acq_func = 'EI',
                                       n_calls = max_iter,
                                       x0 = init_params,
                                       random_state = 64)    
    opt_params = opt_results.x 
    # get net_
    # ...
    error_values = [np.min(opt_results.func_vals[:i]) for i in range(1, max_iter + 1)]
    return opt_params, error_values

def _run_opt_cobyla(net, init_params, cons, metric, target_statistic, window_len):
    max_iter = 11
    def _obj_func(params_cobyla):
        return metric(net = net, target_statistic = target_statistic, window_len = window_len, params_cobyla = params_cobyla)  
    
    opt_results = fmin_cobyla(_obj_func,
                              cons = cons,
                              rhobeg = 0.1,
                              rhoend = 1e-4,
                              x0 = init_params,
                              maxfun = max_iter,
                              catol = 0.0)    
    opt_params = opt_results
    # get net_
    # ...
    error_values = list()
    return opt_params, error_values
    
def _get_fixed_params(net):
    """Gets fixed params (we need this function bc we have to remove and reset each drive).
    
    Returns
    -------
    {'drive_name': {'numspikes': x,
                'location': 'proximal',
                'n_drive_cells': 'n_cells',
                'cell_specific': True,
                'space_constant': {'L2_pyramidal': x,
                                  'L2_basket': x,
                                  'L5_basket': x,
                                  'L5_pyramidal': x},
                'synaptic_delays': {'L5_basket': x,
                                    'L5_pyramidal': x,
                                    'L2_basket': x,
                                    'L2_pyramidal': x},
                'probability': {'L5_basket': x,
                                'L5_pyramidal': x,
                                'L2_basket': x,
                                'L2_pyramidal': x},
                'event_seed': x,
                'conn_seed': x}}
    """
    
    fixed_params = dict()  
    for drive_name in net.external_drives.keys():
        drive = net.external_drives[drive_name]
        conn_idxs = pick_connection(net, src_gids = drive_name) 
        delays = dict()
        probabilities = dict()
        conn_fixed_params = dict()
        for conn_idx in conn_idxs:
            target_type = net.connectivity[conn_idx]['target_type']
            delay = net.connectivity[conn_idx]['nc_dict']['A_delay']
            space_constant = net.connectivity[conn_idx]['nc_dict']['lamtha']
            probability = net.connectivity[conn_idx]['probability']
            delays.update({f'{target_type}': delay})
            probabilities.update({f'{target_type}': probability})         
        conn_fixed_params.update({'numspikes': drive['dynamics']['numspikes']})
        conn_fixed_params.update({'location': drive['location']})
        if drive['cell_specific']:
            conn_fixed_params.update({'n_drive_cells': 'n_cells'})
        else:
            conn_fixed_params.update({'n_drive_cells': drive['n_drive_cells']})
        conn_fixed_params.update({'cell_specific': drive['cell_specific']})
        conn_fixed_params.update({'space_constant': space_constant})
        conn_fixed_params.update({'synaptic_delays': delays})
        conn_fixed_params.update({'probability': probabilities})
        conn_fixed_params.update({'event_seed': drive['event_seed']})
        conn_fixed_params.update({'conn_seed': drive['conn_seed']})
        fixed_params.update({drive_name: conn_fixed_params})
    return fixed_params

def _get_predicted_params(net, **params):
    """Assembles the parameters to be passed to the simulation.
       
       Returns
       -------
       predicted_params : a dictionary ()
           {'drive_name': {'mu': x,
                       'sigma': x,
                       'ampa_weights': {'L5_basket': x,
                                         'L2_basket': x,
                                         'L5_pyramidal': x,
                                         'L2_pyramidal': x},
                       'nmda_weights': {'L5_basket': x,
                                         'L2_basket': x,
                                         'L5_pyramidal': x,
                                         'L2_pyramidal': x}
                       }
       """
    
    predicted_params = dict()
    for drive_name in net.external_drives.keys():
        drive_predicted_params = dict()
        drive_predicted_params.update({'mu': params[f'{drive_name}_mu']})
        drive_predicted_params.update({'sigma': params[f'{drive_name}_sigma']})
        ampa_weights = dict()
        nmda_weights = dict()
        conn_idxs = pick_connection(net, src_gids = drive_name) 
        for conn_idx in conn_idxs: 
            target_type = net.connectivity[conn_idx]['target_type']
            target_receptor = net.connectivity[conn_idx]['receptor']
            if target_receptor == 'ampa':
                ampa_weights.update({target_type: params[f'{drive_name}_ampa_{target_type}']})
            elif target_receptor == 'nmda':
                nmda_weights.update({target_type: params[f'{drive_name}_nmda_{target_type}']})
        drive_predicted_params.update({'ampa_weights': ampa_weights})  
        drive_predicted_params.update({'nmda_weights': nmda_weights})   
        predicted_params.update({drive_name: drive_predicted_params})
    return predicted_params

def _set_params(net, fixed_params, predicted_params):
    """Sets the network parameters.
    
       Parameters
       ----------
       net : the Network object 
       fixed_params : unchanging network parameters 
       predicted_params : the parameters predicted by the optimizer
    
       Returns
       -------
       net : Network object
    """
    
    net_new = net.copy()
    # remove existing drives, re-set them with updated parameters 
    for drive_name in net.external_drives.keys():
        # clear drive 
        del net_new.external_drives[drive_name] 
        # clear connectivity
        conn_idxs = pick_connection(net_new, src_gids = drive_name)
        net_new.connectivity = [conn for conn_idx, conn in enumerate(net_new.connectivity) if conn_idx not in conn_idxs]
        net_new.add_evoked_drive(name = drive_name,
                              mu = predicted_params[drive_name]['mu'], 
                              sigma = predicted_params[drive_name]['sigma'], 
                              numspikes = fixed_params[drive_name]['numspikes'],
                              location = fixed_params[drive_name]['location'],
                              n_drive_cells = fixed_params[drive_name]['n_drive_cells'],
                              cell_specific = fixed_params[drive_name]['cell_specific'],
                              weights_ampa = predicted_params[drive_name]['ampa_weights'], 
                              weights_nmda = predicted_params[drive_name]['nmda_weights'], 
                              space_constant = fixed_params[drive_name]['space_constant'],
                              synaptic_delays = fixed_params[drive_name]['synaptic_delays'],
                              probability = fixed_params[drive_name]['probability'],
                              event_seed = fixed_params[drive_name]['event_seed'],
                              conn_seed = fixed_params[drive_name]['conn_seed'])
    return net_new

def _get_opt_net(opt_params, fixed_params):
    """Returns the optimized network net_."""
    return 

# These 2 functions will go in a file called metrics.py
def _rmse_evoked(net, target_statistic, window_len, **params): # params_cobyla = None
    """The objective function for evoked responses.
        
       Parameters
       -----------
       net : the Network object 
       target_dpl : the recorded dipole
       params : the constraints 
       
       Returns
       -------
       rmse : normalized root mean squared error between recorded and simulated dipole 
    """

    fixed_params = _get_fixed_params(net)
    # get predicted params
        # gp_minimize & fmin_cobyla return predicted parameters that are formatted differently (bc we are using gp_minimize w/ named args),
        # we will probably need 2 different _get_predicted_params functions
    predicted_params = _get_predicted_params(net, **params) 
    # get network with predicted params 
    new_net = _set_params(net, fixed_params, predicted_params)
    # simulate dipole
    dpl = simulate_dipole(new_net, tstop = 100, n_trials = 1)[0]
    # smooth & scale
    if window_len:
        dpl.smooth(window_len)
    scaling_factor = target_statistic.scale
    # calculate error
    rmse = np.sqrt( ((dpl.data['agg'] - target_statistic.data['agg'])**2).sum() / len(dpl.times) ) / (max(target_statistic.data['agg']) - min(target_statistic.data['agg']))
    return rmse

def _rmse_rhythmic():
    return 




















