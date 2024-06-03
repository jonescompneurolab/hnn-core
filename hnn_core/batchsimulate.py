"""Batch simulation."""
from joblib import Parallel, delayed
from hnn_core import simulate_dipole
from hnn_core.network_models import (jones_2009_model,
                                     calcium_model, law_2021_model)


class BatchSimulate:
    def __init__(self, set_params, net_name='jones', tstop=170,
                 dt=0.025, n_trials=1):
        """
        Initialize the BatchSimulate class.

        Parameters
        ----------
        set_params : func
            User-defined function that sets parameters in network drives.
            `set_params(net, params) -> None`
        net_name : str, optional
            The name of the network model to use. Default is 'jones'.
        tstop : float, optional
            The stop time for the simulation. Default is 170 ms.
        dt : float, optional
            The time step for the simulation. Default is 0.025 ms.
        n_trials : int, optional
            The number of trials for the simulation. Default is 1.
        """
        self.set_params = set_params
        self.net_name = net_name
        self.tstop = tstop
        self.dt = dt
        self.n_trials = n_trials

    def run(self, param_grid, max_size=None, return_output=True,
            save_output=False, fpath='./', n_jobs=1):
        """
        Run batch simulations.

        Parameters
        ----------
        param_grid : dict
            Dictionary with parameter names and ranges.
        max_size : int, optional
            Maximum size of the batch. Default is None.
        return_output : bool, optional
            Whether to return the simulation outputs. Default is True.
        save_output : bool, optional
            Whether to save the outputs to disk. Default is False.
        fpath : str, optional
            File path for saving outputs. Default is './'.
        n_jobs : int, optional
            Number of parallel jobs. Default is -1.

        Returns
        -------
        list
            List of simulation results if return_output is True.
        """
        param_combinations = self._generate_param_combinations(
            param_grid, max_size)
        # print("param_combinations-->",param_combinations)
        results = self.simulate_batch(param_combinations, n_jobs=n_jobs)
        print(results)

        # if save_output:
        #     self.save(results, param_combinations, fpath, max_size)

        if return_output:
            return results

    def simulate_batch(self, param_combinations, n_jobs=-1):
        """
        Simulate a batch of parameter sets in parallel.

        Parameters
        ----------
        param_combinations : list
            List of parameter combinations.
        n_jobs : int, optional
            Number of parallel jobs. Default is -1.

        Returns
        -------
        list
            List of simulation results.
        """
        res = Parallel(n_jobs=n_jobs, verbose=50)(
            delayed(self._run_single_sim)(
                params) for params in param_combinations)
        return res

    def _run_single_sim(self, param_values):
        """
        Run a single simulation.

        Parameters
        ----------
        param_values : dict
            Dictionary of parameter values.

        Returns
        -------
        instance of Dipole
            The simulated dipole.
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

        self.set_params(param_values, net)
        dpl = simulate_dipole(net, tstop=self.tstop, dt=self.dt,
                              n_trials=self.n_trials)
        return dpl

    def _generate_param_combinations(self, param_grid, max_size=None):
        """
        Generate combinations of parameters from the grid.

        Parameters
        ----------
        param_grid : dict
            Dictionary with parameter names and ranges.
        max_size : int, optional
            Maximum size of the batch. Default is None.

        Returns
        -------
        list
            List of parameter combinations.
        """
        from itertools import product

        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, combination))
                        for combination in product(*values)]

        if max_size is not None:
            combinations = combinations[:max_size]

        return combinations
