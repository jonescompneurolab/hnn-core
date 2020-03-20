"""import NEURON module"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Mainak Jas <mainakjas@gmail.com>

from warnings import warn

from neuron import h

# a few globals
pc = None
last_network = None
_backend = Joblib_backend
_loaded_dll = None


def shutdown():
    """Tell nrniv processes to terminate. Will cause calling script
    to also terminate
    """
    pc.done()
    h.quit()


def get_nhosts():
    """Return the number of processors used by ParallelContext
    Returns
    -------
    nhosts: int
        Value from pc.nhost()
    """
    if pc is not None:
        return int(pc.nhost())
    else:
        return 1


def get_rank():
    """Return the MPI rank from ParallelContext
    Returns
    -------
    rank: int
        Value from pc.id()
    """
    if pc is not None:
        return int(pc.id())
    else:
        return 0


def create_parallel_context(n_cores=None):
    """Create parallel context.
    Parameters
    ----------
    n_cores: int | None
        Number of processors to use for a simulation. A value of None will
        allow NEURON to use all available processors.
    """

    global cvode, pc, last_network

    if pc is None:
        if n_cores is None:
            # MPI: Initialize the ParallelContext class
            pc = h.ParallelContext()
        else:
            pc = h.ParallelContext(n_cores)

        cvode = h.CVode()

        # be explicit about using fixed step integration
        cvode.active(0)

        # use cache_efficient mode for allocating elements in contiguous order
        # cvode.cache_efficient(1)
    else:
        # ParallelContext() has already been called. Don't start more workers.
        # Just tell old nrniv workers to quit.
        pc.done()


class Joblib_backend(object):
    """The Joblib_backend class.

    Parameters
    ----------
    n_jobs : int | None
        The number of jobs to start in parallel

    Attributes
    ----------
    """
    def __init__(self, n_jobs=None):
        print("Joblib will up to %d jobs" % (self.n_cores))

    def _parallel_func(self, func, n_jobs):
        if n_jobs != 1:
            try:
                from joblib import Parallel, delayed
            except ImportError:
                warn('joblib not installed. Cannot run in parallel.')
                n_jobs = 1
        if n_jobs == 1:
            n_jobs = 1
            my_func = func
            parallel = list
        else:
            parallel = Parallel(n_jobs)
            my_func = delayed(func)

        return parallel, my_func

    def __enter__(self):
        global _backend
        self._old_backend = _backend
        _backend = self

    def __exit__(self, type, value, traceback):
        global _backend
        _backend = self._old_backend

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


    """
    def __init__(self, n_procs=None, mpi_cmd='mpiexec'):
        import psutil
        import multiprocessing
        import os

        self._type = 'mpi'
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = multiprocessing.cpu_count()

        # can't run on more than available logical cores
        if n_procs is None:
            self.n_cores = logical_cores
        else:
            if n_procs > logical_cores:
                self.n_cores = logical_cores
            else:
                self.n_cores = n_procs

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

        print("MPI will run over %d cores" % (self.n_cores))

        self.mpi_cmd = mpi_cmd

        if hyperthreading:
            self.mpi_cmd += ' --use-hwthread-cpus '

        self.mpi_cmd += '-np ' + str(self.n_cores) + ' '

    def __enter__(self):
        global _backend
        self._old_backend = _backend
        _backend = self

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
        import hnn_core

        cmd = 'nrniv -python -mpi -nobanner ' + \
            os.path.join(os.path.dirname(hnn_core.__file__), 'mpi_child.py')

        if self.n_cores > 1:
            cmd = self.mpi_cmd + cmd

        print("Running %d trials..." % (net.params['N_trials']))
        # Split the command into shell arguments for passing to Popen
        cmdargs = shlex.split(cmd, posix="win" not in platform)

        pickled_params = codecs.encode(pickle.dumps(net.params),
                                       "base64").decode()

        # Start the simulation in parallel!
        proc = Popen(cmdargs, stdin=PIPE, stdout=PIPE, stderr=PIPE,
                     cwd=os.getcwd(), universal_newlines=True)

        # wait until process completes
        out, err = proc.communicate(pickled_params)

        # print all messages (included error messages)
        print(out)

        # if simulation failed, raise exception
        if proc.returncode == 0:
            raise RuntimeError

        # unpickle the data
        dpl, spikedata = pickle.loads(codecs.decode(err.encode(), "base64"))

        (spiketimes, spikegids, net.gid_dict) = spikedata
        net.spiketimes.append(spiketimes)
        net.spikegids.append(spikegids)

        return dpl
