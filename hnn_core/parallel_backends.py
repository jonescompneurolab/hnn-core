"""Parallel backends"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Mainak Jas <mainakjas@gmail.com>

import os
import sys
import multiprocessing
import shlex
import pickle
import codecs
from warnings import warn
from subprocess import Popen
import selectors
import binascii

_BACKEND = None


def _clone_and_simulate(net, trial_idx, prng_seedcore_initial):
    """Run a simulation including building the network

    This is used by both backends. MPIBackend calls this in mpi_child.py, once
    for each trial (blocking), and JoblibBackend calls this for each trial
    (non-blocking)
    """

    # avoid relative lookups after being forked (Joblib)
    from hnn_core.network_builder import NetworkBuilder
    from hnn_core.network_builder import _simulate_single_trial

    # XXX this should be built into NetworkBuilder
    # update prng_seedcore params to provide jitter between trials
    for param_key in prng_seedcore_initial.keys():
        net.params[param_key] = prng_seedcore_initial[param_key] + trial_idx

    neuron_net = NetworkBuilder(net)
    dpl = _simulate_single_trial(neuron_net, trial_idx)

    spikedata = neuron_net.get_data_from_neuron()

    return dpl, spikedata


def _gather_trial_data(sim_data, net, n_trials):
    """Arrange data by trial

    To be called after simulate(). Returns list of Dipoles, one for each trial,
    and saves spiking info in net (instance of Network).
    """
    dpls = []

    for idx in range(n_trials):
        dpls.append(sim_data[idx][0])
        spikedata = sim_data[idx][1]
        net.spikes._times.append(spikedata[0])
        net.spikes._gids.append(spikedata[1])
        net.gid_dict = spikedata[2]  # only have one gid_dict
        net.spikes.update_types(net.gid_dict)

    return dpls


class JoblibBackend(object):
    """The JoblibBackend class.

    Parameters
    ----------
    n_jobs : int | None
        The number of jobs to start in parallel. If None, then 1 trial will be
        started without parallelism

    Attributes
    ----------
    n_jobs : int
        The number of jobs to start in parallel
    """
    def __init__(self, n_jobs=1):
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
        global _BACKEND

        self._old_backend = _BACKEND
        _BACKEND = self

        return self

    def __exit__(self, type, value, traceback):
        global _BACKEND

        _BACKEND = self._old_backend

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

        prng_seedcore_initial = net.params['prng_*'].copy()
        parallel, myfunc = self._parallel_func(_clone_and_simulate)
        sim_data = parallel(myfunc(net, idx, prng_seedcore_initial)
                            for idx in range(n_trials))

        dpls = _gather_trial_data(sim_data, net, n_trials)
        return dpls


class MPIBackend(object):
    """The MPIBackend class.

    Parameters
    ----------
    n_procs : int | None
        The number of MPI processes requested by the user. If None, then will
        attempt to detect number of cores (including hyperthreads) and start
        parallel simulation over all of them.
    mpi_cmd : str
        The name of the mpi launcher executable. Will use 'mpiexec'
        (openmpi) by default.

    Attributes
    ----------

    n_procs : int
        The number of processes MPI will actually use (spread over cores). This
        can be less than the user specified value if limited by the cores on
        the system, the number of cores allowed by the job scheduler, or
        if mpi4py could not be loaded.
    mpi_cmd_str : str
        The string of the mpi command with number of procs and options

    """
    def __init__(self, n_procs=None, mpi_cmd='mpiexec'):
        n_logical_cores = multiprocessing.cpu_count()

        if n_procs is None:
            self.n_procs = n_logical_cores
        else:
            self.n_procs = n_procs

        # obey limits set by scheduler
        if hasattr(os, 'sched_getaffinity'):
            scheduler_cores = len(os.sched_getaffinity(0))
            self.n_procs = min(self.n_procs, scheduler_cores)

        # did user try to force running on more cores than available?
        oversubscribe = False
        if self.n_procs > n_logical_cores:
            oversubscribe = True

        hyperthreading = False

        try:
            import mpi4py
            mpi4py.__version__  # for flake8 test

            try:
                import psutil

                n_physical_cores = psutil.cpu_count(logical=False)

                # detect if we need to use hwthread-cpus with mpiexec
                if self.n_procs > n_physical_cores:
                    hyperthreading = True

            except ImportError:
                warn('psutil not installed, so cannot detect if hyperthreading'
                     'is enabled, assuming yes.')
                hyperthreading = True

        except ImportError:
            warn('mpi4py not installed. will run on single processor')
            self.n_procs = 1

        self.mpi_cmd_str = mpi_cmd

        if self.n_procs == 1:
            print("Backend will use 1 core. Running simulation without MPI")
            return
        else:
            print("MPI will run over %d processes" % (self.n_procs))

        if hyperthreading:
            self.mpi_cmd_str += ' --use-hwthread-cpus'

        if oversubscribe:
            self.mpi_cmd_str += ' --oversubscribe'

        self.mpi_cmd_str += ' -np ' + str(self.n_procs)

        self.mpi_cmd_str += ' nrniv -python -mpi -nobanner ' + \
            sys.executable + ' ' + \
            os.path.join(os.path.dirname(sys.modules[__name__].__file__),
                         'mpi_child.py')

    def __enter__(self):
        global _BACKEND

        self._old_backend = _BACKEND
        _BACKEND = self

        return self

    def __exit__(self, type, value, traceback):
        global _BACKEND

        _BACKEND = self._old_backend

    def _read_data(self, fd, mask):
        """read from fd until data includes padding characters"""
        data = os.read(fd, 4096)
        self.proc_data_bytes += data

        # only _read_stdout gets a signal from the fd. the end of simulation
        # data is signalled by the process terminating
        return None

    def _read_stdout(self, fd, mask):
        """read from fd until receiving the process simulation is complete"""
        data = os.read(fd, 4096)
        if data:
            str_data = data.decode()
            if str_data == 'end_of_sim' or str_data.startswith('end_of_data'):
                return str_data

            # output from process includes newlines
            sys.stdout.write(str_data)

        return None

    def _process_child_data(self, data_bytes, data_len):
        if not data_len == len(data_bytes):
            raise RuntimeError("Failed to receive all data from the child MPI"
                               " process. Expecting %d bytes, got %d" %
                               (data_len, len(data_bytes)))

        if len(data_bytes) == 0:
            raise RuntimeError("MPI simulation didn't return any data")

        # decode base64 byte string
        try:
            data_pickled = codecs.decode(data_bytes, "base64")
        except binascii.Error:
            # This is here for future debugging purposes. Unit tests can't
            # reproduce an incorrectly padded string, but this has been an
            # issue before
            raise ValueError("Incorrect padding for data length %d bytes" %
                             len(data_bytes), "(mod 4 = %d)" %
                             len(data_bytes) % 4)

        # unpickle the data
        return pickle.loads(data_pickled)

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

        # just use the joblib backend for a single core
        if self.n_procs == 1:
            return JoblibBackend(n_jobs=1).simulate(net)

        n_trials = net.params['N_trials']
        print("Running %d trials..." % (n_trials))
        dpls = []

        # Split the command into shell arguments for passing to Popen
        if 'win' in sys.platform:
            use_posix = True
        else:
            use_posix = False
        cmdargs = shlex.split(self.mpi_cmd_str, posix=use_posix)

        pickled_params = codecs.encode(pickle.dumps(net.params),
                                       "base64")

        # set some MPI environment variables
        my_env = os.environ.copy()
        if 'win' not in sys.platform:
            my_env["OMPI_MCA_btl_base_warn_component_unused"] = '0'

        if 'darwin' in sys.platform:
            my_env["PMIX_MCA_gds"] = "^ds12"  # open-mpi/ompi/issues/7516
            my_env["TMPDIR"] = "/tmp"  # open-mpi/ompi/issues/2956

        # set up pairs of pipes to communicate with subprocess
        (pipe_stdin_r, pipe_stdin_w) = os.pipe()
        (pipe_stdout_r, pipe_stdout_w) = os.pipe()
        (pipe_stderr_r, pipe_stderr_w) = os.pipe()

        # Start the simulation in parallel!
        proc = Popen(cmdargs, stdin=pipe_stdin_r, stdout=pipe_stdout_w,
                     stderr=pipe_stderr_w, env=my_env, cwd=os.getcwd(),
                     universal_newlines=True)

        # process will read stdin on startup for params
        os.write(pipe_stdin_w, pickled_params)

        # signal that we are done writing params
        os.close(pipe_stdin_w)
        os.close(pipe_stdin_r)

        # data will be stored here; output will be printed and discarded
        self.proc_data_bytes = b''

        # create the selector instance and register all input events
        # with self.read_stdout which will only echo to stdout
        self.sel = selectors.DefaultSelector()
        self.sel.register(pipe_stdout_r, selectors.EVENT_READ,
                          self._read_stdout)
        self.sel.register(pipe_stderr_r, selectors.EVENT_READ,
                          self._read_stdout)

        # loop while the process is running
        while proc.poll() is None:
            # wait for an event on the selector, timeout after 1s
            events = self.sel.select(timeout=1)
            for key, mask in events:
                callback = key.data
                completion_signal = callback(key.fileobj, mask)
                if completion_signal is not None:
                    if completion_signal == "end_of_sim":
                        # finishied receiving printable output
                        # everything else received is data
                        self.sel.unregister(pipe_stderr_r)
                        self.sel.register(pipe_stderr_r, selectors.EVENT_READ,
                                          self._read_data)
                    elif completion_signal.startswith("end_of_data"):
                        split_string = completion_signal.split(':')
                        if len(split_string) > 1:
                            data_len = int(split_string[1])
                            self.sel.unregister(pipe_stdout_r)
                        else:
                            raise ValueError("Invalid data send completion "
                                             "signal from child MPI process")
                        # there could still be data in stderr, so we return
                        # to waiting until the process ends

        # cleanup the selector
        self.sel.unregister(pipe_stderr_r)
        self.sel.close()

        # done with stdout and stderr
        os.close(pipe_stdout_r)
        os.close(pipe_stdout_w)
        os.close(pipe_stderr_r)
        os.close(pipe_stderr_w)

        # if simulation failed, raise exception
        if proc.returncode != 0:
            raise RuntimeError("MPI simulation failed")

        sim_data = self._process_child_data(self.proc_data_bytes, data_len)

        dpls = _gather_trial_data(sim_data, net, n_trials)
        return dpls
