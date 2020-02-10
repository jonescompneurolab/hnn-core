"""Functions for running in parallel"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Mainak Jas <mainakjas@gmail.com>

from warnings import warn


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
