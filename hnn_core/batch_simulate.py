"""Batch Simulation."""

# Authors: Abdul Samad Siddiqui <abdulsamadsid1@gmail.com>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

from joblib import Parallel, delayed
from .dipole import simulate_dipole
from .network_models import (jones_2009_model,
                             calcium_model, law_2021_model)


class BatchSimulate:
    def __init__(self, set_params, net_name='jones', tstop=170,
                 dt=0.025, n_trials=1, record_vsec=False,
                 record_isec=False, postproc=False):
        """Initialize the BatchSimulate class.

        Parameters
        ----------
        set_params : func
            User-defined function that sets parameters in network drives.

                ``set_params(net, params) -> None``

            where ``net`` is a Network object and ``params`` is a dictionary
            of the parameters that will be set inside the function.
        net_name : str
            The name of the network model to use. Default is `jones`.
        tstop : float, optional
            The stop time for the simulation. Default is 170 ms.
        dt : float, optional
            The time step for the simulation. Default is 0.025 ms.
        n_trials : int, optional
            The number of trials for the simulation. Default is 1.
        record_vsec : 'all' | 'soma' | False
            Option to record voltages from all sections ('all'), or just
            the soma ('soma'). Default: False.
        record_isec : 'all' | 'soma' | False
            Option to record voltages from all sections ('all'), or just
            the soma ('soma'). Default: False.
        postproc : bool
            If True, smoothing (``dipole_smooth_win``) and scaling
            (``dipole_scalefctr``) values are read from the parameter file, and
            applied to the dipole objects before returning.
            Note that this setting
            only affects the dipole waveforms, and not somatic voltages,
            possible extracellular recordings etc.
            The preferred way is to use the
            :meth:`~hnn_core.dipole.Dipole.smooth` and
            :meth:`~hnn_core.dipole.Dipole.scale` methods instead.
            Default: False.
        """
        self.set_params = set_params
        self.net_name = net_name
        self.tstop = tstop
        self.dt = dt
        self.n_trials = n_trials
        self.record_vsec = record_vsec
        self.record_isec = record_isec
        self.postproc = postproc

    def run(self, param_grid, return_output=True, combinations=True, n_jobs=1):
        """Run batch simulations.

        Parameters
        ----------
        param_grid : dict
            Dictionary with parameter names and ranges.
        return_output : bool, optional
            Whether to return the simulation outputs. Default is True.
        combinations : bool, optional
            Whether to generate the Cartesian product of the parameter ranges.
            If False, generate combinations based on corresponding indices.
            Default is True.
        n_jobs : int, optional
            Number of parallel jobs. Default is 1.

        Returns
        -------
        results : list
            List of simulation results if return_output is True.
        """
        param_combinations = self._generate_param_combinations(
            param_grid, combinations)
        results = self.simulate_batch(param_combinations, n_jobs=n_jobs)

        if return_output:
            return results

    def simulate_batch(self, param_combinations, n_jobs=1):
        """Simulate a batch of parameter sets in parallel.

        Parameters
        ----------
        param_combinations : list
            List of parameter combinations.
        n_jobs : int, optional
            Number of parallel jobs. Default is 1.

        Returns
        -------
        res: list
            List of dictionaries containing simulation results.
            Each dictionary contains the following keys:
            - `net`: The network model used for the simulation.
            - `dpl`: The simulated dipole.
            - `param_values`: The parameter values used for the simulation.
        """
        res = Parallel(n_jobs=n_jobs, verbose=50)(
            delayed(self._run_single_sim)(
                params) for params in param_combinations)
        return res

    def _run_single_sim(self, param_values):
        """Run a single simulation.

        Parameters
        ----------
        param_values : dict
            Dictionary of parameter values.

        Returns
        -------
        dict
            Dictionary containing the simulation results.
            The dictionary contains the following keys:
            - `net`: The network model used for the simulation.
            - `dpl`: The simulated dipole.
            - `param_values`: The parameter values used for the simulation.
        """
        if self.net_name not in ['jones', 'law', 'calcium']:
            raise ValueError(
                f"Unknown network model: {self.net_name}. "
                "Valid options are 'jones', 'law', and 'calcium'."
            )
        elif self.net_name == 'jones':
            net = jones_2009_model()
        elif self.net_name == 'law':
            net = law_2021_model()
        elif self.net_name == 'calcium':
            net = calcium_model()
        print(param_values)
        self.set_params(param_values, net)
        dpl = simulate_dipole(net,
                              tstop=self.tstop,
                              dt=self.dt,
                              n_trials=self.n_trials,
                              record_vsec=self.record_vsec,
                              record_isec=self.record_isec,
                              postproc=self.postproc)

        return {'net': net, 'dpl': dpl, 'param_values': param_values}

    def _generate_param_combinations(self, param_grid, combinations=True):
        """Generate combinations of parameters from the grid.

        Parameters
        ----------
        param_grid : dict
            Dictionary with parameter names and ranges.
        combinations : bool, optional
            Whether to generate the Cartesian product of the parameter ranges.
            If False, generate combinations based on corresponding indices.
            Default is True.

        Returns
        -------
        param_combinations: list
            List of parameter combinations.
        """
        from itertools import product

        keys, values = zip(*param_grid.items())
        if combinations:
            param_combinations = [dict(zip(keys, combination))
                                  for combination in product(*values)]
        else:
            param_combinations = [dict(zip(keys, combination))
                                  for combination in zip(*values)]
        return param_combinations
