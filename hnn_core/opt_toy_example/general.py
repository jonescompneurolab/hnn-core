"""Parameter optimization."""

# Authors: Carolina Fernandez <cxf418@miami.edu>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

import numpy as np
<<<<<<< HEAD
<<<<<<< HEAD
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

from metrics import _rmse_evoked, _rmse_rhythmic, _rmse_poisson
=======

<<<<<<< HEAD
from metrics import _rmse_evoked
>>>>>>> 50186cb (Clean up optimize evoked and example)
=======
from metrics import _rmse_evoked  # change path***
>>>>>>> 46f1268 (Add tests and address comments)

from skopt import gp_minimize
from scipy.optimize import fmin_cobyla

=======
import matplotlib.pyplot as plt 
from collections import namedtuple

from hnn_core import read_params, jones_2009_model, simulate_dipole 
=======
=======

>>>>>>> 1a7e98b (Address comments for more generalized routine)
from hnn_core import simulate_dipole
>>>>>>> 2f308f8 (added pep8 formatting)
from hnn_core.network import pick_connection

from skopt import gp_minimize
from scipy.optimize import fmin_cobyla
>>>>>>> df86a5d (Draft opt class and functions based on comments)


class Optimizer:
<<<<<<< HEAD
    def __init__(self, net, constraints, set_params, solver, obj_fun,
<<<<<<< HEAD
                 scale_factor, smooth_window_len, tstop):
=======
    def __init__(self, net, constraints, solver, obj_fun):
>>>>>>> 1a7e98b (Address comments for more generalized routine)
=======
                 tstop, scale_factor=1., smooth_window_len=None):
<<<<<<< HEAD
>>>>>>> 46f1268 (Add tests and address comments)
=======
        if net.external_drives:
            raise ValueError("The current Network instance has external " +
                             "drives, provide a Network object with no " +
                             "drives.")
>>>>>>> 51112fc (Address comments and fix test script)
        self.net = net
        self.constraints = constraints
        self._set_params = set_params
        # Optimizer method
        if solver == 'bayesian':
            self._assemble_constraints = _assemble_constraints_bayesian
            self._run_opt = _run_opt_bayesian
        elif solver == 'cobyla':
            self._assemble_constraints = _assemble_constraints_cobyla
            self._run_opt = _run_opt_cobyla
<<<<<<< HEAD
<<<<<<< HEAD
=======
        else:
            raise ValueError("solver must be 'bayesian' or 'cobyla'")
>>>>>>> 51112fc (Address comments and fix test script)
        # Response to be optimized
        if obj_fun == 'evoked':
            self.obj_fun = _rmse_evoked
        else:
            raise ValueError("obj_fun must be 'evoked'")
        self.scale_factor = scale_factor
        self.smooth_window_len = smooth_window_len
        self.tstop = tstop
        self.net_ = None
        self.obj = list()
        self.opt_params = None
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
        self.max_iter = 150
>>>>>>> a4f67f6 (add optimize rhythmic function)
=======
        self.max_iter = 200
>>>>>>> f3490e1 (added pep8 formatting)
=======
        self.max_iter = 150
>>>>>>> 9131299 (Address comments for more generalized routine)
=======
        self.max_iter = 200
>>>>>>> 50186cb (Clean up optimize evoked and example)

    def __repr__(self):
        class_name = self.__class__.__name__
        return class_name

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    def fit(self, target_statistic):
>>>>>>> 672ce00 (Address comments for more generalized routine)
=======
    def fit(self, target_statistic, sfreq):
>>>>>>> 18ac228 (added methods to remove 1/f and compute psd)
=======
    def fit(self, target_statistic):
>>>>>>> a4f67f6 (add optimize rhythmic function)
=======
    def fit(self, target_statistic=None):
>>>>>>> e101c24 (Draft opt class and functions based on comments)
=======
        if obj_fun == 'evoked':
            self.obj_fun = _rmse_evoked
        elif obj_fun == 'rhythmic':
            self.obj_fun = _rmse_rhythmic
        elif obj_fun == 'poisson':
            self.obj_fun = _rmse_poisson
        self.net_ = None
        self.obj = list()
        self.opt_params = None
        self.max_iter = 200

    def __repr__(self):
        class_name = self.__class__.__name__
        return '<%s | %s>' % (class_name)

    def fit(self, target_statistic):
>>>>>>> 1a7e98b (Address comments for more generalized routine)
        """ ...

            Parameters
            ----------
<<<<<<< HEAD
            target_statistic : ndarray
                Recorded dipole (must have the same amount of data points as
                                 the initial, simulated dipole).
            scaling_factor : float
                ...
=======
            target_statistic :
                dpl
            window_len : float
                for smoothing dpl
>>>>>>> 1a7e98b (Address comments for more generalized routine)
            """
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 2f308f8 (added pep8 formatting)

<<<<<<< HEAD
<<<<<<< HEAD
        init_params, cons = self._get_params(self.net, self.constraints)
        opt_params, error_values = self._run_opt(self.net,
                                                 init_params,
                                                 cons,
                                                 self.metric,
                                                 target_statistic,
                                                 window_len)
<<<<<<< HEAD
=======
        params = self._get_params(self.net, self.constraints)
        opt_params, obj, net_ = self._run_opt(self.net,
                                              params,
                                              self.obj_fun,
                                              target_statistic,
                                              self.max_iter)
>>>>>>> 1a7e98b (Address comments for more generalized routine)
        self.opt_params = opt_params
        self.obj = obj
        self.net_ = net_
        return

    def plot_convergence(self, ax=None, show=True):
        """Convergence plot.

        Parameters
        ----------
        ax : instance of matplotlib figure | None
            The matplotlib axis
        show : bool
            If True, show the figure

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
    constraints :
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
    """Assembles constraints in format required by gp_minimize.

<<<<<<< HEAD
=======
            
        init_params, cons = self._get_params(self.net, self.constraints)  
        opt_params, error_values = self._run_opt(self.net, init_params, cons, self.metric, target_statistic, window_len)
        
=======
>>>>>>> 2f308f8 (added pep8 formatting)
=======
        params = self._get_params(self.net, self.constraints)
=======
        params = self.__assemble_constraints(self._set_params,
                                             self.constraints)
=======
    def fit(self, target_statistic):
=======
    def fit(self, target):
>>>>>>> 46f1268 (Add tests and address comments)
        """
        Runs optimization routine.

        Parameters
        ----------
        target : ndarray
            The recorded dipole.

        Returns
        -------
        None.

        """

        constraints = self._assemble_constraints(self.constraints)
        initial_params = _get_initial_params(self.constraints)
>>>>>>> 50186cb (Clean up optimize evoked and example)

>>>>>>> e101c24 (Draft opt class and functions based on comments)
        opt_params, obj, net_ = self._run_opt(self.net,
                                              constraints,
                                              initial_params,
                                              self._set_params,
                                              self.obj_fun,
<<<<<<< HEAD
<<<<<<< HEAD
                                              target_statistic,
<<<<<<< HEAD
                                              self.max_iter)
>>>>>>> 672ce00 (Address comments for more generalized routine)
=======
                                              self.max_iter,
<<<<<<< HEAD
                                              self.obj_fun_type,
                                              self._remove_aperiodic,
                                              self._compute_psd,
                                              sfreq)
>>>>>>> 18ac228 (added methods to remove 1/f and compute psd)
=======
                                              self.f_bands,
                                              self.weights)
>>>>>>> a4f67f6 (add optimize rhythmic function)
=======
                                              self.scaling,
=======
                                              self.scale_factor,
                                              self.smooth_window_len,
                                              self.tstop,
>>>>>>> 50186cb (Clean up optimize evoked and example)
                                              self.max_iter,
                                              target)

>>>>>>> e101c24 (Draft opt class and functions based on comments)
        self.opt_params = opt_params
        self.obj = obj
        self.net_ = net_
        return

    def plot_convergence(self, ax=None, show=True):
        """
        Convergence plot.

        Parameters
        ----------
        ax : instance of matplotlib figure, None
            The matplotlib axis. The default is None.
        show : bool
            If True, show the figure. The default is True.

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
        y_min = min(self.obj) - 0.01
        y_max = max(self.obj) + 0.01

        axis.plot(x, self.obj, color='black')
        axis.set_ylim([y_min, y_max])
        axis.set_title('Convergence')
        axis.set_xlabel('Number of calls')
        axis.set_ylabel('Objective value')
        axis.grid(visible=True)

        fig.show(show)
        return axis.get_figure()


def _get_initial_params(constraints):
    """
    Gets initial parameters as midpoints of parameter ranges.

    Parameters
    ----------
    constraints : dict
        The user-defined constraints.

    Returns
    -------
    initial_params : dict
        Keys are parameter names, values are initial parameters.

    """

    initial_params = dict()
    for cons_key in constraints:
        initial_params.update({cons_key: ((constraints[cons_key][0] +
                                          constraints[cons_key][1]))/2})

    return initial_params


def _assemble_constraints_bayesian(constraints):
    """
    Assembles constraints in format required by gp_minimize.

    Parameters
    ----------
<<<<<<< HEAD
    constraints : dict
        The user-defined constraints.

    Returns
    -------
    cons_bayesian : list of tuples
        Lower and higher limit for each parameter.
    """

    # assemble constraints in solver-specific format
    cons_bayesian = list(constraints.values())
    return cons_bayesian


def _assemble_constraints_cobyla(constraints):
    """
    Assembles constraints in format required by fmin_cobyla.

    Parameters
    ----------
    constraints : dict
        The user-defined constraints.

    Returns
    -------
    cons_bayesian : dict
        Set of functions.
    """

    # assemble constraints in solver-specific format
    cons_cobyla = list()
    for idx, cons_key in enumerate(constraints):
        cons_cobyla.append(lambda x, idx=idx:
                           float(constraints[cons_key][1]) - x[idx])
        cons_cobyla.append(lambda x, idx=idx:
                           x[idx] - float(constraints[cons_key][0]))

    return cons_cobyla


<<<<<<< HEAD
<<<<<<< HEAD
def _get_params_bayesian(net, constraints):
<<<<<<< HEAD
<<<<<<< HEAD
    """Assembles constraints & initial parameters as required by gp_minimize. 
    
>>>>>>> df86a5d (Draft opt class and functions based on comments)
=======
    """Assembles constraints & initial parameters as required by gp_minimize.
=======
=======
def _assemble_constraints_bayesian(set_params, constraints):
>>>>>>> e101c24 (Draft opt class and functions based on comments)
    """Assembles constraints in format required by gp_minimize.
>>>>>>> 672ce00 (Address comments for more generalized routine)
=======
def _update_params(initial_params, predicted_params):
    """
    Update param_dict with predicted parameters.
>>>>>>> 50186cb (Clean up optimize evoked and example)

>>>>>>> 2f308f8 (added pep8 formatting)
    Parameters
    ----------
<<<<<<< HEAD
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
>>>>>>> 1a7e98b (Address comments for more generalized routine)

    # assemble constraints in solver-specific format
    cons_cobyla = list()
    for cons_idx, cons_val in enumerate(params['constraints']):
        cons_cobyla.append(lambda x:
                           params['constraints'][cons_idx][1] - x[cons_idx])
        cons_cobyla.append(lambda x:
                           x[cons_idx] - params['constraints'][cons_idx][0])
    params.update({'constraints': cons_cobyla})
    return params


def _run_opt_bayesian(net, params, obj_fun, target_statistic, max_iter):
    """Uses gp_minimize optimizer.

       Parameters
       ----------
       net : Network
       params : dictionary
           Contains parameter names, initial parameters, and constraints.
       obj_fun : func
           The objective function.
        target_statistic : Dipole
            The target statistic.
        max_iter : int
            Max number of calls.

       Returns
       -------
       opt_params : list
           Final parameters.
       obj : list
           Objective values.
       net_ : Network
           Optimized network object.
    """
    def _obj_func(predicted_params):
        return obj_fun(net,
                       params['names'],
                       target_statistic,
                       predicted_params)

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


def _run_opt_cobyla(net, params, obj_fun, target_statistic, max_iter):
    """Uses fmin_cobyla optimizer.

       Parameters
       ----------
       net : Network
       params : dictionary
           Contains parameter names, initial parameters, and constraints.
       obj_fun : func
           The objective function.
        target_statistic : Dipole
            The target statistic.
        max_iter : int
            Max number of calls.

       Returns
       -------
       opt_params : list
           Final parameters.
       obj : list
           Objective values.
       net_ : Network
           Optimized network object.
    """
    def _obj_func(predicted_params):
        return obj_fun(net,
                       params['names'],
                       target_statistic,
                       predicted_params)

    opt_results = fmin_cobyla(_obj_func,
                              cons=params['constraints'],
                              rhobeg=0.1,
                              rhoend=1e-4,
                              x0=params['initial'],
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
=======
    initial_params : dict
        Keys are parameter names, values are initial parameters.
    predicted_params : list
        Parameters selected by the optimizer.
>>>>>>> 50186cb (Clean up optimize evoked and example)

    Returns
    -------
    params_dict : dict
        Keys are parameter names, values are parameters.
    """

    param_dict = dict()
    for param_key, param_name in enumerate(initial_params):
        param_dict.update({param_name: predicted_params[param_key]})

    return param_dict


def _run_opt_bayesian(net, constraints, initial_params, set_params, obj_fun,
                      scale_factor, smooth_window_len, tstop, max_iter,
                      target):
    """
    Runs optimization routine with gp_minimize optimizer.

    Parameters
    ----------
    net : Network
        The network object.
    constraints : list of tuples
        Parameter constraints in solver-specific format.
    initial_params : dict
        Keys are parameter names, values are initial parameters..
    set_params : func
        User-defined function that sets parameters in network drives.
    obj_fun : func
        The objective function.
    scale_factor : float
        The dipole scale factor.
    smooth_window_len : float
        The smooth window length.
    tstop : float
        The simulated dipole's duration.
    max_iter : int
        Number of calls the optimizer makes.
    target : ndarray
        The recorded dipole.

    Returns
    -------
    opt_params : list
        Optimized parameters.
    obj : list
        Objective values.
    net_ : Network
        Optimized network object.
    """

    obj_values = list()

    def _obj_func(predicted_params):
        return obj_fun(net,
<<<<<<< HEAD
                       params['names'],
<<<<<<< HEAD
                       target_statistic,
<<<<<<< HEAD
                       predicted_params)
>>>>>>> 672ce00 (Address comments for more generalized routine)
=======
                       predicted_params,
<<<<<<< HEAD
                       compute_psd)
>>>>>>> 18ac228 (added methods to remove 1/f and compute psd)
=======
                       f_bands,
                       weights,
                       _set_params)
>>>>>>> a4f67f6 (add optimize rhythmic function)
=======
                       set_params,
                       predicted_params,
                       scaling,
                       target_statistic,
                       f_bands,
                       weights)
>>>>>>> e101c24 (Draft opt class and functions based on comments)
=======
                       initial_params,
                       set_params,
                       predicted_params,
                       _update_params,
                       obj_values,
                       scale_factor,
                       smooth_window_len,
                       tstop,
<<<<<<< HEAD
                       target_statistic)
>>>>>>> 50186cb (Clean up optimize evoked and example)
=======
                       target)
>>>>>>> 46f1268 (Add tests and address comments)

    opt_results = gp_minimize(func=_obj_func,
                              dimensions=constraints,
                              acq_func='EI',
                              n_calls=max_iter,
                              x0=list(initial_params.values()))

    # get optimized params
    opt_params = opt_results.x

    # get objective values
    obj = [np.min(obj_values[:i]) for i in range(1, max_iter + 1)]

    # get optimized net
    param_dict = _update_params(initial_params, opt_params)
    net_ = net.copy()
    set_params(net_, param_dict)

    return opt_params, obj, net_


def _run_opt_cobyla(net, constraints, initial_params, set_params, obj_fun,
                    scale_factor, smooth_window_len, tstop, max_iter,
                    target):
    """
    Runs optimization routine with fmin_cobyla optimizer.

    Parameters
    ----------
    net : Network
        The network object.
    constraints : dict
        Parameter constraints in solver-specific format.
    initial_params : dict
        Keys are parameter names, values are initial parameters..
    set_params : func
        User-defined function that sets parameters in network drives.
    obj_fun : func
        The objective function.
    scale_factor : float
        The dipole scale factor.
    smooth_window_len : float
        The smooth window length.
    tstop : float
        The simulated dipole's duration.
    max_iter : int
        Number of calls the optimizer makes.
    target : ndarray, None
        The recorded dipole. The default is None.

    Returns
    -------
    opt_params : list
        Optimized parameters.
    obj : list
        Objective values.
    net_ : Network
        Optimized network object.
    """

    obj_values = list()

    def _obj_func(predicted_params):
        return obj_fun(net,
                       initial_params,
                       set_params,
                       predicted_params,
                       _update_params,
                       obj_values,
                       scale_factor,
                       smooth_window_len,
                       tstop,
                       target)

    opt_results = fmin_cobyla(_obj_func,
<<<<<<< HEAD
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
=======
                              cons=constraints,
>>>>>>> 50186cb (Clean up optimize evoked and example)
                              rhobeg=0.1,
                              rhoend=1e-4,
                              x0=list(initial_params.values()),
                              maxfun=max_iter,
                              catol=0.0)
<<<<<<< HEAD
>>>>>>> 2f308f8 (added pep8 formatting)
    opt_params = opt_results
<<<<<<< HEAD
<<<<<<< HEAD
    # get net_
    # ...
    error_values = list()
    return opt_params, error_values
<<<<<<< HEAD
<<<<<<< HEAD

=======
    obj = list()
=======

    # get optimized params
    opt_params = opt_results

    # get objective values
    obj = [np.min(obj_values[:i]) for i in range(1, max_iter + 1)]

>>>>>>> 50186cb (Clean up optimize evoked and example)
    # get optimized net
    param_dict = _update_params(initial_params, opt_params)
    net_ = net.copy()
    set_params(net_, param_dict)

    return opt_params, obj, net_
<<<<<<< HEAD
>>>>>>> 672ce00 (Address comments for more generalized routine)

<<<<<<< HEAD

<<<<<<< HEAD
=======
    
def _get_fixed_params(net):
    """Gets fixed params (we need this function bc we have to remove and reset each drive).
    
>>>>>>> df86a5d (Draft opt class and functions based on comments)
=======

=======
    obj = list()
    # get optimized net
    net_ = _set_params(net, params['names'], opt_params)
    return opt_params, obj, net_
>>>>>>> 1a7e98b (Address comments for more generalized routine)


<<<<<<< HEAD
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
=======
def _set_params(net, param_names, predicted_params):
>>>>>>> 1a7e98b (Address comments for more generalized routine)
    """Sets the network parameters.

       Parameters
       ----------
<<<<<<< HEAD
       net : the Network object
       fixed_params : unchanging network parameters
       predicted_params : the parameters predicted by the optimizer
<<<<<<< HEAD
    
>>>>>>> df86a5d (Draft opt class and functions based on comments)
=======
=======
       net : Network
       param_names : dictionary
           Parameters to change.
       predicted_params : list
           The parameters selected by the optimizer.
>>>>>>> 1a7e98b (Address comments for more generalized routine)

>>>>>>> 2f308f8 (added pep8 formatting)
       Returns
       -------
       net : Network
    """
<<<<<<< HEAD
<<<<<<< HEAD

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
                if param_name in ('mu', 'sigma', 'tstart', 'burst_rate',
                                  'burst_std'):
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

>>>>>>> 1a7e98b (Address comments for more generalized routine)
    # calculate error
    rmse = np.sqrt(((dpl.data['agg'] - target_statistic.data['agg'])**2).sum()
                   / len(dpl.times)) / (max(target_statistic.data['agg'])
                                        - min(target_statistic.data['agg']))
    return rmse


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
=======
>>>>>>> e101c24 (Draft opt class and functions based on comments)
=======
>>>>>>> 9131299 (Address comments for more generalized routine)
