"""import NEURON module"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Mainak Jas <mainakjas@gmail.com>

from warnings import warn

from neuron import h

pc = None
last_network = None


def shutdown():
    """Tell nrniv processes to terminate. Calling script will also terminate"""
    pc.done()
    h.quit()


def get_nhosts():
    """Return the number of processors used by ParallelContext
    Returns
    -------
    nhosts: int
        Value from pc.nhost()
    """
    return nhosts


def create_parallel_context(n_cores=None):
    """Create parallel context.
    Parameters
    ----------
    n_cores: int | None
        Number of processors to use for a simulation. A value of None will
        allow NEURON to use all available processors.
    """

    global rank, nhosts, cvode, pc, last_network

    if pc is None:
        if n_cores is None:
            # MPI: Initialize the ParallelContext class
            pc = h.ParallelContext()
        else:
            pc = h.ParallelContext(n_cores)

        nhosts = int(pc.nhost())  # Find number of hosts
        rank = int(pc.id())     # rank or node number (0 will be the master)
        cvode = h.CVode()

        # be explicit about using fixed step integration
        cvode.active(0)

        # use cache_efficient mode for allocating elements in contiguous order
        # cvode.cache_efficient(1)
    else:
        # ParallelContext() has already been called. Don't start more workers.
        # Just tell old nrniv workers to quit.
        pc.done()


def clear_last_network_objects(net):
    """Clears NEURON objects and saves the current Network instance
    Parameters
    ----------
    net: an instance of Network
        The current Network instance (context) from which NEURON objects will
        be created
    """
    global last_network

    if last_network is not None:
        last_network._clear_neuron_objects()

    net._clear_neuron_objects()
    last_network = net


def _parallel_func(func, n_jobs):
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
