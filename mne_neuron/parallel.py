"""import NEURON module"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Mainak Jas <mainakjas@gmail.com>

from warnings import warn

from neuron import h

rank = 0
nhosts = 1
pc = h.ParallelContext(nhosts)
pc.done()
rank = int(pc.id())
cvode = h.CVode()


def create_parallel_context(n_jobs=1):
    """Create parallel context."""
    rank = int(pc.id())     # rank or node number (0 will be the master)

    if rank == 0:
        pc.gid_clear()


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
