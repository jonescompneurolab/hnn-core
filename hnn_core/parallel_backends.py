"""Parallel backends"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Mainak Jas <mainakjas@gmail.com>

from warnings import warn

_backend = None
_loaded_dll = None


class Joblib_backend(object):
    """The Joblib_backend class.

    Parameters
    ----------
    n_jobs : int
        The number of jobs to start in parallel

    Attributes
    ----------
    n_jobs : int
        The number of jobs to start in parallel
    """
    def __init__(self, n_jobs=1):
        self.type = 'joblib'

        self.n_jobs = n_jobs
        print("joblib will run over %d jobs" % (self.n_jobs))

    def _parallel_func(self, func):
        if self.n_jobs != 1:
            try:
                from joblib import Parallel, delayed
            except ImportError:
                warn('joblib not installed. Cannot run in parallel.')
                self.n_jobs = 1
        if self.n_jobs == 1:
            my_func = func
            parallel = list
        else:
            parallel = Parallel(self.n_jobs)
            my_func = delayed(func)

        return parallel, my_func

    def __enter__(self):
        global _backend
        self._old_backend = _backend
        _backend = self

        return self

    def __exit__(self, type, value, traceback):
        global _backend
        _backend = self._old_backend

    def _clone_and_simulate(self, net, trial_idx):
        # avoid relative lookups after being forked by joblib
        from hnn_core.neuron import _neuron_network, _simulate_single_trial

        if trial_idx != 0:
            net.params['prng_*'] = trial_idx

        neuron_net = _neuron_network(net)
        dpl = _simulate_single_trial(neuron_net)

        spikedata = neuron_net.get_data_from_neuron()

        return dpl, spikedata

    def simulate(self, net):
        """Simulate the HNN model

        Parameters
        ----------
        net : Network object
            The Network object specifying how cells are
            connected.

        Returns
        -------
        dpl: list of Dipole
            The Dipole results from each simulation trial
        """

        n_trials = net.params['N_trials']
        dpls = []

        parallel, myfunc = self._parallel_func(self._clone_and_simulate)
        data = parallel(myfunc(net, idx) for idx in range(n_trials))

        # the assignments below need to be made after any forking
        for idx in range(n_trials):
            dpls.append(data[idx][0])
            spikedata = data[idx][1]
            net.spikes._times.append(spikedata[0])
            net.spikes._gids.append(spikedata[1])
            net.gid_dict = spikedata[2]  # only have one gid_dict
            net.spikes.update_types(net.gid_dict)

        return dpls


class MPI_backend(object):
    """The MPI_backend class.

    Parameters
    ----------
    n_procs : int | None
        The number of processors (cores) to start MPI job over
    mpi_cmd : str
        The name of the mpi launcher executable. Will use 'mpiexec'
        (openmpi) by default.

    Attributes
    ----------

    n_jobs : int
        The number of jobs to start in parallel (NOT SUPPORTED)
    n_cores : int
        The number of cores used by the backend
    mpi_cmd_str : str
        The string of the mpi command with number of cores and options

    """
    def __init__(self, n_jobs=1, n_procs=None, mpi_cmd='mpiexec'):
        import psutil
        import multiprocessing
        import os
        import hnn_core

        try:
            import mpi4py
            mpi4py.__version__  # for flake8 test
        except ImportError:
            warn('mpi4py not installed. will run on single processor')
            self.n_procs = 1

        self._type = 'mpi'
        self.n_jobs = n_jobs

        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = multiprocessing.cpu_count()

        oversubscribe = False
        # trying to run on more than available logical cores?
        if n_procs is None:
            self.n_cores = logical_cores
        else:
            self.n_cores = n_procs
            if n_procs > logical_cores:
                oversubscribe = True

        # obey limits set by scheduler
        try:
            scheduler_cores = len(os.sched_getaffinity(0))
        except AttributeError:
            scheduler_cores = None

        if scheduler_cores is not None:
            self.n_cores = min(self.n_cores, scheduler_cores)

        # use hwthread-cpus if necessary
        hyperthreading = False
        if self.n_cores > physical_cores:
            hyperthreading = True

        self.mpi_cmd_str = mpi_cmd

        if self.n_cores > 1:
            print("MPI will run over %d cores" % (self.n_cores))
        else:
            print("Only have 1 core available. Running simulation without MPI")
            print("Consider using Joblib_backend with n_jobs > 1 "
                  "for running multiple trials")
            return

        if self.n_jobs > 1:
            raise ValueError("Nested parallelism is not currently supported"
                             " with MPI_backend!\n"
                             "Please use joblib for embarassinly parallel jobs"
                             " (n_jobs > 1)\n"
                             "or multiple cores per simulation with"
                             " MPI_backend\n")

        if hyperthreading:
            self.mpi_cmd_str += ' --use-hwthread-cpus'

        if oversubscribe:
            self.mpi_cmd_str += ' --oversubscribe'

        self.mpi_cmd_str += ' -np ' + str(self.n_cores)

        self.mpi_cmd_str += ' nrniv -python -mpi -nobanner ' + \
            os.path.join(os.path.dirname(hnn_core.__file__), 'mpi_child.py')

    def __enter__(self):
        global _backend
        self._old_backend = _backend
        _backend = self

        return self

    def __exit__(self, type, value, traceback):
        global _backend
        _backend = self._old_backend

    def simulate(self, net):
        """Simulate the HNN model in parallel on all cores

        Parameters
        ----------
        net : Network object
            The Network object specifying how cells are
            connected.

        Returns
        -------
        dpl: list of Dipole
            The Dipole results from each simulation trial
        """
        from subprocess import Popen, PIPE
        import shlex
        import pickle
        import codecs
        import os
        from sys import platform

        # just use the joblib backend for a single core
        if self.n_cores == 1:
            return Joblib_backend(n_jobs=1).simulate(net)

        print("Running %d trials..." % (net.params['N_trials']))

        # Split the command into shell arguments for passing to Popen
        cmdargs = shlex.split(self.mpi_cmd_str, posix="win" not in platform)

        pickled_params = codecs.encode(pickle.dumps(net.params),
                                       "base64").decode()

        # set some MPI environment variables
        my_env = os.environ.copy()
        my_env["OMPI_MCA_btl_base_warn_component_unused"] = '0'
        # Start the simulation in parallel!
        proc = Popen(cmdargs, stdin=PIPE, stdout=PIPE, stderr=PIPE, env=my_env,
                     cwd=os.getcwd(), universal_newlines=True)

        # wait until process completes
        out, err = proc.communicate(pickled_params)

        # print all messages (included error messages)
        print(out)

        # if simulation failed, raise exception
        if proc.returncode != 0:
            # data is padded with "==""
            err_msg = err.split("==")
            if len(err_msg) > 1:
                print(err_msg[1])
            raise RuntimeError("MPI simulation failed")

        data_str = err.rstrip("=")
        if len(data_str) == 0:
            raise RuntimeError("MPI simulation didn't return any data")

        # turn stderr (string) to bytes-like object
        data_bytes = data_str.encode()

        # decode base64 object
        data_pickled = codecs.decode(data_bytes, "base64")

        # unpickle the data
        dpl, spikedata = pickle.loads(data_pickled)

        (spiketimes, spikegids, net.gid_dict) = spikedata
        net.spikes._times.append(spiketimes)
        net.spikes._gids.append(spikegids)
        net.spikes.update_types(net.gid_dict)

        return dpl
