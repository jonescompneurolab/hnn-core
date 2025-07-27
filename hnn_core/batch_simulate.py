"""Batch Simulation."""

# Authors: Abdul Samad Siddiqui <abdulsamadsid1@gmail.com>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

import numpy as np
import os
from joblib import Parallel, delayed, parallel_config

from .network import Network
from .externals.mne import _validate_type, _check_option
from .dipole import simulate_dipole
from .network_models import jones_2009_model


class BatchSimulate(object):
    """The BatchSimulate class.

    Parameters
    ----------
    set_params : func
        User-defined function that sets parameters in network drives.

            ``set_params(net, params) -> None``

        where ``net`` is a Network object and ``params`` is a dictionary
        of the parameters that will be set inside the function.
    net : Network object, optional
        The network model to use for simulations. Examples include the
        returned value of the following functions:

        - `jones_2009_model`: A network model based on Jones et al. (2009).
        - `law_2021_model`: A network model based on Law et al. (2021).
        - `calcium_model`: A network model incorporating calcium dynamics.

        Default is ``jones_2009_model()``.
    tstop : float, optional
        The stop time for the simulation. Default is 170 ms.
    dt : float, optional
        The time step for the simulation. Default is 0.025 ms.
    n_trials : int, optional
        The number of trials for the simulation. Default is 1.
    save_folder : str, optional
        The path to save the simulation outputs.
        Default is './sim_results'.
    batch_size : int, optional
        The maximum number of simulations saved in a single file.
        Default is 100.
    overwrite : bool, optional
        Whether to overwrite existing files and create file paths
        if they do not exist. Default is True.
    save_outputs : bool, optional
        Whether to save the simulation outputs to files. Default is False.
    save_dpl : bool, optional
        If True, save dipole results. Note, `save_outputs` must be True.
        Default: True.
    save_spiking : bool, optional
        If True, save spiking results. Note, `save_outputs` must be True.
        Default: False.
    save_lfp : bool, optional
        If True, save local field potential (lfp) results.
        Note, `save_outputs` must be True.
        Default: False.
    save_voltages : bool, optional
        If True, save voltages results. Note, `save_outputs` must be True.
        Default: False.
    save_currents : bool, optional
        If True, save currents results. Note, `save_outputs` must be True.
        Default: False.
    save_calcium : bool, optional
        If True, save calcium concentrations.
        Note, `save_outputs` must be True.
        Default: False.
    record_vsec : {False, 'all', 'soma'}
        Option to record voltages from all sections ('all'), or just
        the soma ('soma'). Default: False.
    record_isec : {False, 'all', 'soma'}
        Option to record voltages from all sections ('all'), or just
        the soma ('soma'). Default: False.
    postproc : bool, optional
        If True, smoothing (``dipole_smooth_win``) and scaling
        (``dipole_scalefctr``) values are read from the parameter file, and
        applied to the dipole objects before returning.
        Default: False.
    clear_cache : bool, optional
        Whether to clear the results cache after saving each batch.
        Default is False.
    summary_func : func, optional
        A function to calculate summary statistics from the simulation
        results. Default is None.

    Notes
    -----
    When ``save_output=True``, the saved files will appear as
    ``sim_run_{start_idx}-{end_idx}.npz`` in the specified `save_folder`
    directory. The `start_idx` and `end_idx` indicate the range of
    simulation indices contained in each file. Each file will contain
    a maximum of `batch_size` simulations, split evenly among the
    available files. If ``overwrite=True``, existing files with the same name
    will be overwritten.
    """

    def __init__(
        self,
        set_params,
        net=jones_2009_model(),
        tstop=170,
        dt=0.025,
        n_trials=1,
        save_folder="./sim_results",
        batch_size=100,
        overwrite=True,
        save_outputs=False,
        save_dpl=True,
        save_spiking=False,
        save_lfp=False,
        save_voltages=False,
        save_currents=False,
        save_calcium=False,
        record_vsec=False,
        record_isec=False,
        postproc=False,
        clear_cache=False,
        summary_func=None,
    ):
        _validate_type(net, Network, "net", "Network")
        _validate_type(tstop, types="numeric", item_name="tstop")
        _validate_type(dt, types="numeric", item_name="dt")
        _validate_type(n_trials, types="int", item_name="n_trials")
        _validate_type(save_folder, types="path-like", item_name="save_folder")
        _validate_type(batch_size, types="int", item_name="batch_size")
        _validate_type(save_outputs, types=(bool,), item_name="save_outputs")
        _validate_type(save_dpl, types=(bool,), item_name="save_dpl")
        _validate_type(save_spiking, types=(bool,), item_name="save_spiking")
        _validate_type(save_lfp, types=(bool,), item_name="save_lfp")
        _validate_type(save_voltages, types=(bool,), item_name="save_voltages")
        _validate_type(save_currents, types=(bool,), item_name="save_currents")
        _validate_type(save_calcium, types=(bool,), item_name="save_calcium")
        _check_option("record_vsec", record_vsec, ["all", "soma", False])
        _check_option("record_isec", record_isec, ["all", "soma", False])
        _validate_type(clear_cache, types=(bool,), item_name="clear_cache")

        if set_params is not None and not callable(set_params):
            raise TypeError("set_params must be a callable function")

        if summary_func is not None and not callable(summary_func):
            raise TypeError("summary_func must be a callable function")

        self.set_params = set_params
        self.net = net
        self.tstop = tstop
        self.dt = dt
        self.n_trials = n_trials
        self.save_folder = save_folder
        self.batch_size = batch_size
        self.overwrite = overwrite
        self.save_outputs = save_outputs
        self.save_dpl = save_dpl
        self.save_spiking = save_spiking
        self.save_lfp = save_lfp
        self.save_voltages = save_voltages
        self.save_currents = save_currents
        self.save_calcium = save_calcium
        self.record_vsec = record_vsec
        self.record_isec = record_isec
        self.postproc = postproc
        self.clear_cache = clear_cache
        self.summary_func = summary_func

    def run(
        self,
        param_grid,
        return_output=True,
        combinations=True,
        n_jobs=1,
        backend="loky",
        verbose=50,
    ):
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
            `multiprocessing`, or `dask`. WARNING: currently only `loky` is
            completely operationable; all other backends are in
            development. Default is `loky`.
        verbose : int, optional
            The verbosity level for parallel execution. Default is 50.

        Returns
        -------
        results : dict
            Dictionary containing 'summary_statistics' and optionally
            'simulated_data'. 'simulated_data' may include keys: 'dpl', 'lfp',
            'spikes', 'voltages', 'param_values', 'net', 'times'.

        Notes
        -----
        Return content depends on `summary_func`, `return_output`, and
        `clear_cache` settings.
        """
        _validate_type(param_grid, types=(dict,), item_name="param_grid")
        _validate_type(n_jobs, types="int", item_name="n_jobs")
        _check_option(
            "backend", backend, ["loky", "threading", "multiprocessing", "dask"]
        )
        _validate_type(verbose, types="int", item_name="verbose")

        param_combinations = self._generate_param_combinations(param_grid, combinations)
        total_sims = len(param_combinations)
        batch_size = min(self.batch_size, total_sims)

        results = []
        simulated_data = []
        for i in range(0, total_sims, batch_size):
            start_idx = i
            end_idx = min(i + batch_size, total_sims)
            batch_results = self.simulate_batch(
                param_combinations[start_idx:end_idx],
                n_jobs=n_jobs,
                backend=backend,
                verbose=verbose,
            )

            if self.save_outputs:
                self._save(batch_results, start_idx, end_idx)

            if self.summary_func is not None:
                summary_statistics = self.summary_func(batch_results)
                results.append(summary_statistics)
                if not self.clear_cache:
                    simulated_data.append(batch_results)

            elif return_output and not self.clear_cache:
                simulated_data.append(batch_results)

            if self.clear_cache:
                del batch_results

        if return_output:
            if self.clear_cache:
                return {"summary_statistics": results}
            else:
                return {"summary_statistics": results, "simulated_data": simulated_data}

    def simulate_batch(
        self,
        param_combinations,
        n_jobs=1,
        backend="loky",
        verbose=50,
    ):
        """Simulate a batch of parameter sets in parallel.

        Parameters
        ----------
        param_combinations : list
            List of parameter combinations.
        n_jobs : int, optional
            Number of parallel jobs. Default is 1.
        backend : str or joblib.parallel.ParallelBackendBase instance, optional
            The parallel backend to use. Can be one of `loky`, `threading`,
            `multiprocessing`, or `dask`. WARNING: currently only `loky` is
            completely operationable; all other backends are in
            development. Default is `loky`.
        verbose : int, optional
            The verbosity level for parallel execution. Default is 50.

        Returns
        -------
        res: list
            List of dictionaries containing simulation results.
            Each dictionary contains the following keys along with their
            associated values:

            - `net`: The network model used for the simulation.
            - `dpl`: The simulated dipole.
            - `param_values`: The parameter values used for the simulation.
        """
        _validate_type(
            param_combinations, types=(list,), item_name="param_combinations"
        )
        _validate_type(n_jobs, types="int", item_name="n_jobs")
        _check_option(
            "backend", backend, ["loky", "threading", "multiprocessing", "dask"]
        )
        _validate_type(verbose, types="int", item_name="verbose")

        with parallel_config(backend=backend):
            res = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(self._run_single_sim)(params) for params in param_combinations
            )
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

        net = self.net.copy()
        self.set_params(param_values, net)

        results = {"net": net, "param_values": param_values}

        if self.save_dpl:
            dpl = simulate_dipole(
                net,
                tstop=self.tstop,
                dt=self.dt,
                n_trials=self.n_trials,
                record_vsec=self.record_vsec,
                record_isec=self.record_isec,
                postproc=self.postproc,
            )
            results["dpl"] = dpl

        if self.save_spiking:
            results["spiking"] = {
                "spike_times": net.cell_response.spike_times,
                "spike_types": net.cell_response.spike_types,
                "spike_gids": net.cell_response.spike_gids,
            }

        if self.save_lfp:
            results["lfp"] = net.rec_arrays

        if self.save_voltages:
            results["voltages"] = net.cell_response.vsec

        if self.save_currents:
            results["currents"] = net.cell_response.isec

        if self.save_calcium:
            results["calcium"] = net.cell_response.ca

        return results

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
            param_combinations = [
                dict(zip(keys, combination)) for combination in product(*values)
            ]
        else:
            param_combinations = [
                dict(zip(keys, combination)) for combination in zip(*values)
            ]
        return param_combinations

    def _save(self, results, start_idx, end_idx):
        """Save simulation results to files.

        Parameters
        ----------
        results : list
            The results to save.
        start_idx : int
            The index of the first simulation in this batch.
        end_idx : int
            The index of the last simulation in this batch.
        """
        _validate_type(results, types=(list,), item_name="results")
        _validate_type(start_idx, types="int", item_name="start_idx")
        _validate_type(end_idx, types="int", item_name="end_idx")

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        save_data = {"param_values": [result["param_values"] for result in results]}

        attributes_to_save = [
            "dpl",
            "spiking",
            "lfp",
            "voltages",
            "currents",
            "calcium",
        ]
        for attr in attributes_to_save:
            if getattr(self, f"save_{attr}") and attr in results[0]:
                save_data[attr] = [result[attr] for result in results]

        metadata = {
            "batch_size": self.batch_size,
            "n_trials": self.n_trials,
            "tstop": self.tstop,
            "dt": self.dt,
        }
        save_data["metadata"] = metadata

        file_name = os.path.join(self.save_folder, f"sim_run_{start_idx}-{end_idx}.npz")
        if os.path.exists(file_name) and not self.overwrite:
            raise FileExistsError(
                f"File {file_name} already exists and overwrite is set to False."
            )

        np.savez(file_name, **save_data)

    def load_results(self, file_path, return_data=None):
        """Load simulation results from a file.

        Parameters
        ----------
        file_path : str
            The path to the file containing the simulation results.
        return_data : list of str, optional
            List of data types to return. If None, returns the types specified
            during initialization. Defaults to None.

        Returns
        -------
        results : dict
            Dictionary containing the loaded results and parameter values.
        """
        if return_data is None:
            return_data = []
            if self.save_dpl:
                return_data.append("dpl")
            if self.save_spiking:
                return_data.append("spiking")
            if self.save_lfp:
                return_data.append("lfp")
            if self.save_voltages:
                return_data.append("voltages")
            if self.save_currents:
                return_data.append("currents")
            if self.save_calcium:
                return_data.append("calcium")

        data = np.load(file_path, allow_pickle=True)
        results = {
            key: data[key].tolist()
            for key in data.files
            if key in return_data or key == "param_values"
        }
        return results

    def load_all_results(self):
        """Load all simulation results from the files in `self.save_folder`.

        Returns
        -------
        all_results : list
            List of dictionaries containing all loaded simulation results.
        """
        all_results = []
        for file_name in os.listdir(self.save_folder):
            if file_name.startswith("sim_run_") and file_name.endswith(".npz"):
                file_path = os.path.join(self.save_folder, file_name)
                results = self.load_results(file_path)
                all_results.append(results)
        return all_results
