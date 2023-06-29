import numpy as np
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======

<<<<<<< HEAD
>>>>>>> 672ce00 (Address comments for more generalized routine)
from hnn_core import simulate_dipole
=======
>>>>>>> 18ac228 (added methods to remove 1/f and compute psd)
from hnn_core.network import pick_connection

from metrics import _rmse_evoked, _rmse_rhythmic, _rmse_poisson, _compute_welch_psd, _compute_multitaper_psd, _remove_aperiodic_foof, _remove_aperiodic_irasa

from skopt import gp_minimize
from scipy.optimize import fmin_cobyla

=======
import matplotlib.pyplot as plt 
from collections import namedtuple

from hnn_core import read_params, jones_2009_model, simulate_dipole 
=======
from hnn_core import simulate_dipole
>>>>>>> 2f308f8 (added pep8 formatting)
from hnn_core.network import pick_connection
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real
from scipy.optimize import fmin_cobyla
>>>>>>> df86a5d (Draft opt class and functions based on comments)


class Optimizer:
    def __init__(self, net, constraints, solver, obj_fun, psd_method=None,
                 aperiodic_method=None):
        self.net = net
        self.constraints = constraints
        # Optimizer method
        if solver == 'bayesian':
            self._get_params = _get_params_bayesian
            self._run_opt = _run_opt_bayesian
        elif solver == 'cobyla':
            self._get_params = _get_params_cobyla
            self._run_opt = _run_opt_cobyla
        # Response to be optimized
        if obj_fun == 'evoked':
            self.obj_fun = _rmse_evoked
            self.obj_fun_type = 'evoked'
        elif obj_fun == 'rhythmic':
            self.obj_fun = _rmse_rhythmic
            self.obj_fun_type = 'rhythmic'
        elif obj_fun == 'poisson':
            self.obj_fun = _rmse_poisson
        # Methods to compute PSD (for rhythmic responses only)
        self._compute_psd = None  
        if psd_method == 'welch':
            self._compute_psd = _compute_welch_psd
        elif psd_method == 'multitaper':
            self._compute_psd = _compute_multitaper_psd
        # Methods to remove aperiodic component (for rhythmic responses only)
        self._remove_aperiodic = None  
        if aperiodic_method == 'fooof':
            self._remove_aperiodic = _remove_aperiodic_foof
        elif aperiodic_method == 'irasa':
            self._remove_aperiodic = _remove_aperiodic_irasa
        self.net_ = None
        self.obj = list()
        self.opt_params = None
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

    def fit(self, target_statistic, window_len=None):
=======
        
    def fit(self, target_statistic, window_len = None):
>>>>>>> df86a5d (Draft opt class and functions based on comments)
=======

    def fit(self, target_statistic, window_len=None):
>>>>>>> 2f308f8 (added pep8 formatting)
=======
        self.max_iter = 200

    def __repr__(self):
        class_name = self.__class__.__name__
        return '<%s | %s>' % (class_name)

<<<<<<< HEAD
    def fit(self, target_statistic):
>>>>>>> 672ce00 (Address comments for more generalized routine)
=======
    def fit(self, target_statistic, sfreq):
>>>>>>> 18ac228 (added methods to remove 1/f and compute psd)
        """ ...

            Parameters
            ----------
            target_statistic : ndarray
                Recorded dipole (must have the same amount of data points as
                                 the initial, simulated dipole).
            window_len : float
                ...
            sfreq : float
                Sampling frequency of recorded dipole (must be the same as the
                                 sampling frequency of the initial, simulated
                                 dipole).
            """
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 2f308f8 (added pep8 formatting)

<<<<<<< HEAD
        init_params, cons = self._get_params(self.net, self.constraints)
        opt_params, error_values = self._run_opt(self.net,
                                                 init_params,
                                                 cons,
                                                 self.metric,
                                                 target_statistic,
                                                 window_len)
<<<<<<< HEAD
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

=======
            
        init_params, cons = self._get_params(self.net, self.constraints)  
        opt_params, error_values = self._run_opt(self.net, init_params, cons, self.metric, target_statistic, window_len)
        
=======
>>>>>>> 2f308f8 (added pep8 formatting)
=======
        params = self._get_params(self.net, self.constraints)
        opt_params, obj, net_ = self._run_opt(self.net,
                                              params,
                                              self.obj_fun,
                                              target_statistic,
<<<<<<< HEAD
                                              self.max_iter)
>>>>>>> 672ce00 (Address comments for more generalized routine)
=======
                                              self.max_iter,
                                              self.obj_fun_type,
                                              self._remove_aperiodic,
                                              self._compute_psd,
                                              sfreq)
>>>>>>> 18ac228 (added methods to remove 1/f and compute psd)
        self.opt_params = opt_params
        self.obj = obj
        self.net_ = net_
        return

    def plot_convergence(self, ax=None, show=True):
        """Convergence plot.

        Parameters
        ----------
        ax : instance of matplotlib figure | None
            The matplotlib axis.
        show : bool
            If True, show the figure.

        Returns
        -------
        fig : instance of plt.fig
            The matplotlib figure handle.
        """

        import matplotlib as mpl
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(constrained_layout=True)

        axis = ax if isinstance(ax, mpl.axes._axes.Axes) else ax

        x = list(range(1, self.max_iter + 1))
        y_max = max(self.obj) + 0.01

        axis.plot(x, self.obj, color='black')
        axis.set_ylim([-.01, y_max])
        axis.set_title('Convergence')
        axis.set_xlabel('Number of calls')
        axis.set_ylabel('Objective value')
        axis.grid(visible=True)

        fig.show(show)
        return axis.get_figure()

    def plot_param_search():
        return


def _get_params(net, constraints):
    """Gets parameters.

    Parameters
    ----------
    net : Network
    constraints : dictionary
        {'drive_name': {'param_name': [min, max],
                      'param_name': [min, max],
                      'param_name': [min, max]}}

    Returns
    -------
    params : dictionary
        params['initial'] : list
        params['constraints'] : list of tuples (min, max)

    params_to_optim : dictionary
        {drive_name: ['param_name', 'param_name', 'param_name']}
        might use this later to override net params in _set_params
        (might have to go back to weights_ampa instead of ampa)
    """

    params = dict()

    # Get params to optimize
    param_names = dict()
    for drive_name in constraints:
        temp = list()
        for param_name in constraints[drive_name]:
            temp.append(param_name)
        param_names.update({drive_name: temp})
    params.update({'names': param_names})

    # Get initial params (bayesian & cobyla can use the same format)
    initial_params = list()
    cons = list()

    # get net drive names
    drive_names = [key for key in net.external_drives.keys()]

    for drive_name in param_names:
        # get relevant params
        if drive_name in drive_names:
            for param_name in param_names[drive_name]:
                # instead check obj_fun
                if param_name in ('mu', 'sigma', 'tstart', 'burst_rate',
                                  'burst_std'):
                    initial_params.append(net.external_drives[drive_name]
                                          ['dynamics'][param_name])
                    cons.append(constraints[drive_name][param_name])
                elif param_name in ('ampa', 'nmda'):
                    conn_idxs = pick_connection(net, src_gids=drive_name)
                    for conn_idx in conn_idxs:
                        # L5_pyramidal, L2_basket, L5_basket, L2_pyramidal
                        target_receptor = net.connectivity[conn_idx]\
                            ['receptor']
                        if target_receptor == param_name:
                            initial_params.append(net.connectivity[conn_idx]
                                                  ['nc_dict']['A_weight'])
                            cons.append(constraints[drive_name][param_name])

    params.update({'initial': initial_params})
    params.update({'constraints': cons})
    return params


def _get_params_bayesian(net, constraints):
<<<<<<< HEAD
<<<<<<< HEAD
    """Assembles constraints & initial parameters as required by gp_minimize. 
    
>>>>>>> df86a5d (Draft opt class and functions based on comments)
=======
    """Assembles constraints & initial parameters as required by gp_minimize.
=======
    """Assembles constraints in format required by gp_minimize.
>>>>>>> 672ce00 (Address comments for more generalized routine)

>>>>>>> 2f308f8 (added pep8 formatting)
    Parameters
    ----------
<<<<<<< HEAD
    net : the Network object
    constraints : the user-defined constraints
<<<<<<< HEAD
<<<<<<< HEAD

    Returns
    -------
    init_params : list
        [drive_name_mu,
         drive_name_sigma,
         drive_name_ampa_target,
         drive_name_nmda_target]
=======
       
    Returns 
    -------
    init_params : list
        [{drive_name}_mu, {drive_name}_sigma, {drive_name}_ampa_target, {drive_name}_nmda_target] 
>>>>>>> df86a5d (Draft opt class and functions based on comments)
=======

    Returns
    -------
    init_params : list
        [drive_name_mu,
         drive_name_sigma,
         drive_name_ampa_target,
         drive_name_nmda_target]
>>>>>>> 2f308f8 (added pep8 formatting)
    cons : list
        [Real(name='{drive_name}_mu', low=x, high=x),
        Real(name='{drive_name}_sigma', low=x, high=x),
        Real(name='{drive_name}_ampa_target', low=x, high=x)]
        Real(name='{drive_name}_nmda_target', low=x0.01, high=x)]
    """
<<<<<<< HEAD
<<<<<<< HEAD

    init_params = list()
    cons = list()
    for drive_name in net.external_drives.keys():
        for cons_key in constraints[drive_name]:
            if cons_key == 'mu':
                cons.append(Real(name=f'{drive_name}_{cons_key}',
                                 low=constraints[drive_name][cons_key][0],
                                 high=constraints[drive_name][cons_key][1]))
                init_params.append(
                    net.external_drives[drive_name]['dynamics']['mu'])
            elif cons_key == 'sigma':
                cons.append(Real(name=f'{drive_name}_{cons_key}',
                                 low=constraints[drive_name][cons_key][0],
                                 high=constraints[drive_name][cons_key][1]))
                init_params.append(
                    net.external_drives[drive_name]['dynamics']['sigma'])
            elif cons_key == 'weights':
                conn_idxs = pick_connection(net, src_gids=drive_name)
                for conn_idx in conn_idxs:
                    target_type = net.connectivity[conn_idx]['target_type']
                    target_receptor = net.connectivity[conn_idx]['receptor']
                    cons.append(
                        Real(name=f'{drive_name}_{target_receptor}_{target_type}',
                             low=constraints[drive_name][cons_key][0],
                             high=constraints[drive_name][cons_key][1]))
                    init_params.append(
                        net.connectivity[conn_idx]['nc_dict']['A_weight'])
    return init_params, cons


# gets called only once
def _get_params_cobyla(net, constraints):
    """Assembles constraints & initial parameters as required by fmin_cobyla.

        Returns
=======
    
=======

>>>>>>> 2f308f8 (added pep8 formatting)
    init_params = list()
    cons = list()
    for drive_name in net.external_drives.keys():
        for cons_key in constraints[drive_name]:
            if cons_key == 'mu':
                cons.append(Real(name=f'{drive_name}_{cons_key}',
                                 low=constraints[drive_name][cons_key][0],
                                 high=constraints[drive_name][cons_key][1]))
                init_params.append(
                    net.external_drives[drive_name]['dynamics']['mu'])
            elif cons_key == 'sigma':
                cons.append(Real(name=f'{drive_name}_{cons_key}',
                                 low=constraints[drive_name][cons_key][0],
                                 high=constraints[drive_name][cons_key][1]))
                init_params.append(
                    net.external_drives[drive_name]['dynamics']['sigma'])
            elif cons_key == 'weights':
                conn_idxs = pick_connection(net, src_gids=drive_name)
                for conn_idx in conn_idxs:
                    target_type = net.connectivity[conn_idx]['target_type']
                    target_receptor = net.connectivity[conn_idx]['receptor']
                    cons.append(
                        Real(name=f'{drive_name}_{target_receptor}_{target_type}',
                             low=constraints[drive_name][cons_key][0],
                             high=constraints[drive_name][cons_key][1]))
                    init_params.append(
                        net.connectivity[conn_idx]['nc_dict']['A_weight'])
    return init_params, cons


# gets called only once
def _get_params_cobyla(net, constraints):
<<<<<<< HEAD
    """Assembles constraints & initial parameters as required by fmin_cobyla. 
       
        Returns 
>>>>>>> df86a5d (Draft opt class and functions based on comments)
=======
    """Assembles constraints & initial parameters as required by fmin_cobyla.

        Returns
>>>>>>> 2f308f8 (added pep8 formatting)
        -------
        init_params : list
        cons : list
    """
<<<<<<< HEAD
<<<<<<< HEAD

    init_params = list()
    cons = list()

    for drive_name in net.external_drives.keys():
        for cons_idx, cons_key in enumerate(constraints[drive_name]):
            if cons_key == 'mu' or cons_key == 'sigma':
                cons.append(lambda x, idx=cons_idx:
                            constraints[drive_name][cons_key][1] - x[idx])
                cons.append(lambda x, idx=cons_idx:
                            constraints[drive_name][cons_key][0] - x[idx])
                init_params.append(
                    net.external_drives[drive_name]['dynamics'][cons_key])
            elif cons_key == 'weights':
                conn_idxs = pick_connection(net, src_gids=drive_name)
                for conn_idx in conn_idxs:
                    cons.append(lambda x, idx=cons_idx:
                                constraints[drive_name][cons_key][1] - x[idx])
                    cons.append(lambda x, idx=cons_idx:
                                constraints[drive_name][cons_key][0] - x[idx])
                    init_params.append(
                        net.connectivity[conn_idx]['nc_dict']['A_weight'])
    return init_params, cons


def _run_opt_bayesian(net, init_params, cons, metric, target_statistic,
                      window_len):

    @use_named_args(dimensions=cons)
    def _obj_func(**params):
        return metric(net, target_statistic, window_len, **params)

    max_iter = 11
    opt_results = gp_minimize(func=_obj_func,
                              dimensions=cons,
                              acq_func='EI',
                              n_calls=max_iter,
                              x0=init_params,
                              random_state=64)
    opt_params = opt_results.x
    # get net_
    # ...
    error_values = [np.min(opt_results.func_vals[:i])
                    for i in range(1, max_iter + 1)]
    return opt_params, error_values


def _run_opt_cobyla(net, init_params, cons, metric, target_statistic,
                    window_len):

    def _obj_func(params_cobyla):
        return metric(net=net, target_statistic=target_statistic,
                      window_len=window_len, params_cobyla=params_cobyla)

    max_iter = 11
    opt_results = fmin_cobyla(_obj_func,
                              cons=cons,
                              rhobeg=0.1,
                              rhoend=1e-4,
                              x0=init_params,
                              maxfun=max_iter,
                              catol=0.0)
=======
    
=======

>>>>>>> 2f308f8 (added pep8 formatting)
    init_params = list()
    cons = list()
=======
    net : Network
        The network object.
    constraints : list of lists
        Constraints for each parameter to be optimized ([min, max]).

    Returns
    -------
    params : dictionary
        Contains parameter names, initial parameters, and constraints.

    """

    # get initial params
    params = _get_params(net, constraints)

    # assemble constraints in solver-specific format
    cons_bayesian = list()
    for cons in params['constraints']:
        cons_bayesian.append((cons[0], cons[1]))
    params.update({'constraints': cons_bayesian})
    return params


def _get_params_cobyla(net, constraints):
    """Assembles constraints in format required by fmin_cobyla.

    Parameters
    ----------
    net : Network
        The network object.
    constraints : list of lists
        Constraints for each parameter to be optimized ([min, max]).

    Returns
    -------
    params : dictionary
        Contains parameter names, initial parameters, and constraints.

    """

    # get initial params
    params = _get_params(net, constraints)

    # assemble constraints in solver-specific format
    cons_cobyla = list()
    for cons_idx, cons_val in enumerate(params['constraints']):
        cons_cobyla.append(lambda x:
                           params['constraints'][cons_idx][1] - x[cons_idx])
        cons_cobyla.append(lambda x:
                           x[cons_idx] - params['constraints'][cons_idx][0])
    params.update({'constraints': cons_cobyla})
    return params


def _run_opt_bayesian(net, params, obj_fun, target_statistic, max_iter,
                      obj_fun_type, remove_aperiodic, compute_psd, sfreq):
    """Uses gp_minimize optimizer.

       Parameters
       ----------
       net : Network
       params : dictionary
           Contains parameter names, initial parameters, and constraints.
       obj_fun : func
           The objective function.
        target_statistic : ndarray
            The recorded dipole.
        max_iter : int
            Max number of calls.
        obj_fun_type : string
            Evoked or rhythmic.
        remove_aperiodic :
            ...
        compute_psd :
            ...
        sfreq :
            ...

       Returns
       -------
       opt_params : list
           Final parameters.
       obj : list
           Objective values.
       net_ : Network
           Optimized network object.
    """

    # if response type is rhythmic, first remove 1/f
    if obj_fun_type == 'rhythmic':
        target_statistic = remove_aperiodic(target_statistic, compute_psd,
                                            sfreq)  # psd

    def _obj_func(predicted_params):
        return obj_fun(net,
                       params['names'],
                       target_statistic,
<<<<<<< HEAD
                       predicted_params)
>>>>>>> 672ce00 (Address comments for more generalized routine)
=======
                       predicted_params,
                       compute_psd)
>>>>>>> 18ac228 (added methods to remove 1/f and compute psd)

    opt_results = gp_minimize(func=_obj_func,
                              dimensions=params['constraints'],
                              acq_func='EI',
                              n_calls=max_iter,
                              x0=params['initial'],
                              random_state=64)
    opt_params = opt_results.x
    obj = [np.min(opt_results.func_vals[:i]) for i in range(1, max_iter + 1)]
    # get optimized net
    net_ = _set_params(net, params['names'], opt_params)
    return opt_params, obj, net_


def _run_opt_cobyla(net, params, obj_fun, target_statistic, max_iter,
                    obj_fun_type, remove_aperiodic, compute_psd, sfreq):
    """Uses fmin_cobyla optimizer.

       Parameters
       ----------
       net : Network
       params : dictionary
           Contains parameter names, initial parameters, and constraints.
       obj_fun : func
           The objective function.
        target_statistic : ndarray
            The recorded dipole.
        max_iter : int
            Max number of calls.
        obj_fun_type : string
            Evoked or rhythmic.
        remove_aperiodic :
            ...
        compute_psd :
            ...
        sfreq :
            ...

       Returns
       -------
       opt_params : list
           Final parameters.
       obj : list
           Objective values.
       net_ : Network
           Optimized network object.
    """

    # if response type is rhythmic, first remove 1/f
    if obj_fun_type == 'rhythmic':
        target_statistic = remove_aperiodic(target_statistic, compute_psd,
                                            sfreq)

    def _obj_func(predicted_params):
        return obj_fun(net,
                       params['names'],
                       target_statistic,
                       predicted_params,
                       compute_psd)

    opt_results = fmin_cobyla(_obj_func,
<<<<<<< HEAD
<<<<<<< HEAD
                              cons = cons,
                              rhobeg = 0.1,
                              rhoend = 1e-4,
                              x0 = init_params,
                              maxfun = max_iter,
                              catol = 0.0)    
>>>>>>> df86a5d (Draft opt class and functions based on comments)
=======
                              cons=cons,
=======
                              cons=params['constraints'],
>>>>>>> 672ce00 (Address comments for more generalized routine)
                              rhobeg=0.1,
                              rhoend=1e-4,
                              x0=params['initial'],
                              maxfun=max_iter,
                              catol=0.0)
>>>>>>> 2f308f8 (added pep8 formatting)
    opt_params = opt_results
<<<<<<< HEAD
    # get net_
    # ...
    error_values = list()
    return opt_params, error_values
<<<<<<< HEAD
<<<<<<< HEAD

=======
    obj = list()
    # get optimized net
    net_ = _set_params(net, params['names'], opt_params)
    return opt_params, obj, net_
>>>>>>> 672ce00 (Address comments for more generalized routine)


<<<<<<< HEAD
=======
    
def _get_fixed_params(net):
    """Gets fixed params (we need this function bc we have to remove and reset each drive).
    
>>>>>>> df86a5d (Draft opt class and functions based on comments)
=======


def _get_fixed_params(net):
    """Gets fixed params.

>>>>>>> 2f308f8 (added pep8 formatting)
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
<<<<<<< HEAD
<<<<<<< HEAD

    fixed_params = dict()
    for drive_name in net.external_drives.keys():
        drive = net.external_drives[drive_name]
        conn_idxs = pick_connection(net, src_gids=drive_name)
=======
    
    fixed_params = dict()  
    for drive_name in net.external_drives.keys():
        drive = net.external_drives[drive_name]
        conn_idxs = pick_connection(net, src_gids = drive_name) 
>>>>>>> df86a5d (Draft opt class and functions based on comments)
=======

    fixed_params = dict()
    for drive_name in net.external_drives.keys():
        drive = net.external_drives[drive_name]
        conn_idxs = pick_connection(net, src_gids=drive_name)
>>>>>>> 2f308f8 (added pep8 formatting)
        delays = dict()
        probabilities = dict()
        conn_fixed_params = dict()
        for conn_idx in conn_idxs:
            target_type = net.connectivity[conn_idx]['target_type']
            delay = net.connectivity[conn_idx]['nc_dict']['A_delay']
            space_constant = net.connectivity[conn_idx]['nc_dict']['lamtha']
            probability = net.connectivity[conn_idx]['probability']
            delays.update({f'{target_type}': delay})
<<<<<<< HEAD
<<<<<<< HEAD
            probabilities.update({f'{target_type}': probability})
=======
            probabilities.update({f'{target_type}': probability})         
>>>>>>> df86a5d (Draft opt class and functions based on comments)
=======
            probabilities.update({f'{target_type}': probability})
>>>>>>> 2f308f8 (added pep8 formatting)
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

<<<<<<< HEAD
<<<<<<< HEAD

def _get_predicted_params(net, **params):
    """Assembles the parameters to be passed to the simulation.

=======
def _get_predicted_params(net, **params):
    """Assembles the parameters to be passed to the simulation.
       
>>>>>>> df86a5d (Draft opt class and functions based on comments)
=======

def _get_predicted_params(net, **params):
    """Assembles the parameters to be passed to the simulation.

>>>>>>> 2f308f8 (added pep8 formatting)
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
<<<<<<< HEAD
<<<<<<< HEAD

=======
    
>>>>>>> df86a5d (Draft opt class and functions based on comments)
=======

>>>>>>> 2f308f8 (added pep8 formatting)
    predicted_params = dict()
    for drive_name in net.external_drives.keys():
        drive_predicted_params = dict()
        drive_predicted_params.update({'mu': params[f'{drive_name}_mu']})
        drive_predicted_params.update({'sigma': params[f'{drive_name}_sigma']})
        ampa_weights = dict()
        nmda_weights = dict()
<<<<<<< HEAD
<<<<<<< HEAD
        conn_idxs = pick_connection(net, src_gids=drive_name)
        for conn_idx in conn_idxs:
            target_type = net.connectivity[conn_idx]['target_type']
            target_receptor = net.connectivity[conn_idx]['receptor']
            if target_receptor == 'ampa':
                ampa_weights.update(
                    {target_type: params[f'{drive_name}_ampa_{target_type}']})
            elif target_receptor == 'nmda':
                nmda_weights.update(
                    {target_type: params[f'{drive_name}_nmda_{target_type}']})
        drive_predicted_params.update({'ampa_weights': ampa_weights})
        drive_predicted_params.update({'nmda_weights': nmda_weights})
        predicted_params.update({drive_name: drive_predicted_params})
    return predicted_params


def _set_params(net, fixed_params, predicted_params):
=======
def _set_params(net, param_names, predicted_params):
>>>>>>> 672ce00 (Address comments for more generalized routine)
    """Sets the network parameters.

       Parameters
       ----------
       net : Network
       param_names : dictionary
           Parameters to change.
       predicted_params : list
           The parameters selected by the optimizer.

=======
        conn_idxs = pick_connection(net, src_gids = drive_name) 
        for conn_idx in conn_idxs: 
=======
        conn_idxs = pick_connection(net, src_gids=drive_name)
        for conn_idx in conn_idxs:
>>>>>>> 2f308f8 (added pep8 formatting)
            target_type = net.connectivity[conn_idx]['target_type']
            target_receptor = net.connectivity[conn_idx]['receptor']
            if target_receptor == 'ampa':
                ampa_weights.update(
                    {target_type: params[f'{drive_name}_ampa_{target_type}']})
            elif target_receptor == 'nmda':
                nmda_weights.update(
                    {target_type: params[f'{drive_name}_nmda_{target_type}']})
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
<<<<<<< HEAD
    
>>>>>>> df86a5d (Draft opt class and functions based on comments)
=======

>>>>>>> 2f308f8 (added pep8 formatting)
       Returns
       -------
       net : Network
    """
<<<<<<< HEAD
<<<<<<< HEAD

    net_new = net.copy()
    # remove existing drives, re-set them with updated parameters
    for drive_name in net.external_drives.keys():
        # clear drive
        del net_new.external_drives[drive_name]
        # clear connectivity
        conn_idxs = pick_connection(net_new, src_gids=drive_name)
        net_new.connectivity = [conn for conn_idx, conn
                                in enumerate(net_new.connectivity)
                                if conn_idx not in conn_idxs]
        net_new.add_evoked_drive(name=drive_name,
                                 mu=predicted_params[drive_name]['mu'],
                                 sigma=predicted_params[drive_name]['sigma'],
                                 numspikes=fixed_params[drive_name]
                                 ['numspikes'],
                                 location=fixed_params[drive_name]['location'],
                                 n_drive_cells=fixed_params[drive_name]
                                 ['n_drive_cells'],
                                 cell_specific=fixed_params[drive_name]
                                 ['cell_specific'],
                                 weights_ampa=predicted_params[drive_name]
                                 ['ampa_weights'],
                                 weights_nmda=predicted_params[drive_name]
                                 ['nmda_weights'],
                                 space_constant=fixed_params[drive_name]
                                 ['space_constant'],
                                 synaptic_delays=fixed_params[drive_name]
                                 ['synaptic_delays'],
                                 probability=fixed_params[drive_name]
                                 ['probability'],
                                 event_seed=fixed_params[drive_name]
                                 ['event_seed'],
                                 conn_seed=fixed_params[drive_name]
                                 ['conn_seed'])
    return net_new


def _get_opt_net(opt_params, fixed_params):
    """Returns the optimized network net_."""
    return


# These 2 functions will go in a file called metrics.py
def _rmse_evoked(net, target_statistic, window_len, **params):
    # params_cobyla = None
    """The objective function for evoked responses.

       Parameters
       -----------
       net : the Network object
       target_dpl : the recorded dipole
       params : the constraints

       Returns
       -------
       rmse : normalized RMSE between recorded and simulated dipole
=======
    
=======

>>>>>>> 2f308f8 (added pep8 formatting)
    net_new = net.copy()

    # get net drive names
    count = 0
    drive_names = [key for key in net_new.external_drives.keys()]
    for drive_name in param_names:

        # set relevant params
        if drive_name in drive_names:
            for param_name in param_names[drive_name]:
                if param_name in ('mu', 'sigma'):
                    net_new.external_drives[drive_name]['dynamics']\
                        [param_name] = predicted_params[count]
                    count += 1
                elif param_name in ('ampa', 'nmda'):
                    conn_idxs = pick_connection(net_new, src_gids=drive_name)
                    for conn_idx in conn_idxs:
                        target_receptor = net_new.connectivity[conn_idx]\
                            ['receptor']
                        if target_receptor == param_name:
                            net_new.connectivity[conn_idx]['nc_dict']\
                                ['A_weight'] = predicted_params[count]
                            count += 1
    return net_new
<<<<<<< HEAD


# These 2 functions will go in a file called metrics.py
def _rmse_evoked(net, param_names, target_statistic, predicted_params):
    """The objective function for evoked responses.

       Parameters
       -----------
       net : Network
       param_names : dictionary
           Parameters to change.
       target_statistic : Dipole
           The recorded dipole.
       predicted_params : list
           Parameters selected by the optimizer.

       Returns
       -------
<<<<<<< HEAD
       rmse : normalized root mean squared error between recorded and simulated dipole 
>>>>>>> df86a5d (Draft opt class and functions based on comments)
=======
       rmse : normalized RMSE between recorded and simulated dipole
>>>>>>> 2f308f8 (added pep8 formatting)
    """

<<<<<<< HEAD
    fixed_params = _get_fixed_params(net)
    # get predicted params
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 2f308f8 (added pep8 formatting)
    # gp_minimize & fmin_cobyla return params in different formats
    # (bc we are using gp_minimize w/ named args),
    # we will probably need 2 different _get_predicted_params functions
    predicted_params = _get_predicted_params(net, **params)
    # get network with predicted params
<<<<<<< HEAD
    new_net = _set_params(net, fixed_params, predicted_params)
    # simulate dipole
    dpl = simulate_dipole(new_net, tstop=100, n_trials=1)[0]
=======
        # gp_minimize & fmin_cobyla return predicted parameters that are formatted differently (bc we are using gp_minimize w/ named args),
        # we will probably need 2 different _get_predicted_params functions
    predicted_params = _get_predicted_params(net, **params) 
    # get network with predicted params 
    new_net = _set_params(net, fixed_params, predicted_params)
    # simulate dipole
    dpl = simulate_dipole(new_net, tstop = 100, n_trials = 1)[0]
>>>>>>> df86a5d (Draft opt class and functions based on comments)
=======
    new_net = _set_params(net, fixed_params, predicted_params)
    # simulate dipole
    dpl = simulate_dipole(new_net, tstop=100, n_trials=1)[0]
>>>>>>> 2f308f8 (added pep8 formatting)
    # smooth & scale
    if window_len:
        dpl.smooth(window_len)
    scaling_factor = target_statistic.scale
<<<<<<< HEAD
<<<<<<< HEAD
    dpl.scale(scaling_factor)
    # calculate error
    rmse = np.sqrt(((dpl.data['agg'] - target_statistic.data['agg'])**2).sum()
                   / len(dpl.times)) / (max(target_statistic.data['agg'])
                                        - min(target_statistic.data['agg']))
    return rmse


def _rmse_rhythmic():
    return
=======
=======
    dpl.scale(scaling_factor)
>>>>>>> 2f308f8 (added pep8 formatting)
=======
    # get network with predicted params
    new_net = _set_params(net, param_names, predicted_params)
    # simulate dipole
    dpl = simulate_dipole(new_net, tstop=100, n_trials=1)[0]

    # smooth & scale
    # if dpl.smooth(window_len):
    #     dpl_smooth = dpl.copy().smooth(window_len)
    dpl.smooth(30)
    # scaling_factor = get from target_statistic
    # dpl.scale(scaling_factor)

>>>>>>> 672ce00 (Address comments for more generalized routine)
    # calculate error
    rmse = np.sqrt(((dpl.data['agg'] - target_statistic.data['agg'])**2).sum()
                   / len(dpl.times)) / (max(target_statistic.data['agg'])
                                        - min(target_statistic.data['agg']))
    return rmse


<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> df86a5d (Draft opt class and functions based on comments)
=======
def _rmse_rhythmic():
=======
def _rmse_rhythmic(net, param_names, target_statistic, predicted_params):
    """The objective function for evoked responses.

       Parameters
       -----------
       net : Network
       param_names : dictionary
           Parameters to change.
       target_statistic : Dipole
           The recorded dipole.
       predicted_params : list
           Parameters selected by the optimizer.

       Returns
       -------
       rmse : norm
    """

    from scipy import signal

    # expose these
    fmin = 0.0
    fmax = 200.0

    new_net = _set_params(net, param_names, predicted_params)
    dpl = simulate_dipole(new_net, tstop=100, n_trials=1)[0]

    f_target, psd_target = signal.periodogram(target_statistic.data['agg'])
    f_simulated, psd_simulated = signal.periodogram(dpl.data['agg'])

    rmse = np.linalg.norm(psd_target - psd_simulated)
    return rmse


def _rmse_poisson():
>>>>>>> 672ce00 (Address comments for more generalized routine)
    return
>>>>>>> 2f308f8 (added pep8 formatting)
=======
>>>>>>> 18ac228 (added methods to remove 1/f and compute psd)
