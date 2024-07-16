"""Batch Simulation."""

# Authors: Abdul Samad Siddiqui <abdulsamadsid1@gmail.com>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

import numpy as np
import os
from joblib import Parallel, delayed, parallel_config
from .externals.mne import _validate_type, _check_option
from .dipole import simulate_dipole
from .network_models import (jones_2009_model,
                             calcium_model, law_2021_model)


class BatchSimulate(object):
    def __init__(self, set_params, net_name='jones', tstop=170,
                 dt=0.025, n_trials=1, record_vsec=False,
                 record_isec=False, postproc=False, save_outputs=False,
                 file_path='./sim_results', batch_size=100,
                 overwrite=True, summary_func=None):
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
        save_outputs : bool, optional
            Whether to save the simulation outputs to files. Default is False.
        file_path : str, optional
            The path to save the simulation outputs.
            Default is './sim_results'.
        batch_size : int, optional
            The maximum number of simulations saved in a single file.
            Default is 100.
        overwrite : bool, optional
            Whether to overwrite existing files and create file paths
            if they do not exist. Default is True.
        summary_func : callable, optional
            A function to calculate summary statistics from the simulation
            results. Default is None.
        Notes
        -----
        When `save_output=True`, the saved files will appear as
        `sim_run_{start_idx}-{end_idx}.npy` in the specified `file_path`
        directory. The `start_idx` and `end_idx` indicate the range of
        simulation indices contained in each file. Each file will contain
        a maximum of `batch_size` simulations, split evenly among the
        available files. If `overwrite=True`, existing files with the same name
        will be overwritten.
        """

        _validate_type(net_name, types='str', item_name='net_name')
        _check_option('net_name', net_name, ['jones', 'law', 'calcium'])
        _validate_type(tstop, types='numeric', item_name='tstop')
        _validate_type(dt, types='numeric', item_name='dt')
        _validate_type(n_trials, types='int', item_name='n_trials')
        _check_option('record_vsec', record_vsec, ['all', 'soma', False])
        _check_option('record_isec', record_isec, ['all', 'soma', False])
        _validate_type(file_path, types='path-like', item_name='file_path')
        _validate_type(batch_size, types='int', item_name='batch_size')

        if set_params is not None and not callable(set_params):
            raise TypeError("set_params must be a callable function")

        if summary_func is not None and not callable(summary_func):
            raise TypeError("summary_func must be a callable function")

        self.set_params = set_params
        self.net_name = net_name
        self.tstop = tstop
        self.dt = dt
        self.n_trials = n_trials
        self.record_vsec = record_vsec
        self.record_isec = record_isec
        self.postproc = postproc
        self.save_outputs = save_outputs
        self.file_path = file_path
        self.batch_size = batch_size
        self.overwrite = overwrite
        self.summary_func = summary_func

    def run(self, param_grid, return_output=True,
            combinations=True, n_jobs=1, backend='loky',
            verbose=50, clear_cache=False):
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
        backend : str or joblib.parallel.ParallelBackendBase instance, optional
            The parallel backend to use. Can be one of `loky`, `threading`,
            `multiprocessing`, or `dask`. Default is `loky`
        verbose : int, optional
            The verbosity level for parallel execution. Default is 50.
        clear_cache : bool, optional
            Whether to clear the results cache after saving each batch.
            Default is False

        Returns
        -------
        results : list
            List of simulation results if return_output is True.
        """
        _validate_type(param_grid, types=(dict,), item_name='param_grid')
        _validate_type(n_jobs, types='int', item_name='n_jobs')
        _check_option('backend', backend, ['loky', 'threading',
                                           'multiprocessing', 'dask'])
        _validate_type(verbose, types='int', item_name='verbose')

        param_combinations = self._generate_param_combinations(
            param_grid, combinations)
        total_sims = len(param_combinations)
        num_sims_per_batch = max(total_sims // self.batch_size, 1)
        batch_size = min(self.batch_size, total_sims)

        results = []
        for i in range(batch_size):
            start_idx = i * num_sims_per_batch
            end_idx = start_idx + num_sims_per_batch
            if i == batch_size - 1:
                end_idx = len(param_combinations)
            batch_results = self.simulate_batch(
                param_combinations[start_idx:end_idx],
                n_jobs=n_jobs,
                backend=backend,
                verbose=verbose)

            if self.save_outputs:
                self._save(batch_results, start_idx, end_idx)

            if self.summary_func is not None:
                summary_statistics = self.summary_func(batch_results)
                if return_output and not clear_cache:
                    results.append(summary_statistics)

            elif return_output and not clear_cache:
                results.append(batch_results)

            if clear_cache:
                del batch_results

        if return_output:
            return results

    def simulate_batch(self, param_combinations, n_jobs=1,
                       backend='loky', verbose=50):
        """Simulate a batch of parameter sets in parallel.

        Parameters
        ----------
        param_combinations : list
            List of parameter combinations.
        n_jobs : int, optional
            Number of parallel jobs. Default is 1.
        backend : str or joblib.parallel.ParallelBackendBase instance, optional
            The parallel backend to use. Can be one of `loky`, `threading`,
            `multiprocessing`, or `dask`. Default is `loky`
        verbose : int, optional
            The verbosity level for parallel execution. Default is 50.

        Returns
        -------
        res: list
            List of dictionaries containing simulation results.
            Each dictionary contains the following keys:
            - `net`: The network model used for the simulation.
            - `dpl`: The simulated dipole.
            - `param_values`: The parameter values used for the simulation.
        """
        _validate_type(param_combinations, types=(list,),
                       item_name='param_combinations')
        _validate_type(n_jobs, types='int', item_name='n_jobs')
        _check_option('backend', backend, ['loky', 'threading',
                                           'multiprocessing', 'dask'])
        _validate_type(verbose, types='int', item_name='verbose')

        with parallel_config(backend=backend):
            res = Parallel(n_jobs=n_jobs, verbose=verbose)(
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

    def _save(self, results, start_idx, end_idx):
        """Save simulation results to files.

        Parameters
        ----------
        results : list
            The results to save.
        batch_index : int
            The index of the current batch.
        """
        _validate_type(results, types=(list,), item_name='results')
        _validate_type(start_idx, types='int', item_name='start_idx')
        _validate_type(end_idx, types='int', item_name='end_idx')

        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)

        sim_data = np.stack([dpl['dpl'][0].data['agg'] for dpl in results])

        file_name = os.path.join(self.file_path,
                                 f'sim_run_{start_idx}-{end_idx}.npy')
        if os.path.exists(file_name) and not self.overwrite:
            raise FileExistsError(
                f"File {file_name} already exists and "
                "overwrite is set to False.")

        np.save(file_name, sim_data)
