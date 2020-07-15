Parallel backends
=================

Two options are available for making use of multiple CPU cores. The first runs multiple trials in parallel with joblib. Alternatively, you can run each trial across multiple cores to reduce the runtime.

Joblib
------

This is the default backend and will execute multiple trials at the same time, with each trial running on a separate core in "embarrassingly parallel" execution. Note that with only 1 trial, there will be no parallelism.

**Dependencies**::

    $ pip install joblib

**Usage**::

    from hnn_core import JoblibBackend

    # set n_jobs to the number of trials to run in parallel with Joblib (up to number of cores on system)
    with JoblibBackend(n_jobs=2):
        dpls = simulate_dipole(net, n_trials=2)

MPI
---

This backend will use MPI (Message Passing Interface) on the system to split neurons across CPU cores (processors) and reduce the simulation time as more cores are used.

**Linux Dependencies**::

    $ sudo apt-get install libopenmpi-dev openmpi-bin
    $ pip install mpi4py psutil

**MacOS Dependencies**::

    $ conda install yes openmpi mpi4py
    $ pip install psutil

**MacOS Environment**::

    $ export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib

Alternatively, run the commands below will avoid needing to run the export command every time a new shell is opened::

    $ cd ${CONDA_PREFIX}
    $ mkdir -p etc/conda/activate.d etc/conda/deactivate.d
    $ echo "export OLD_LD_LIBRARY_PATH=\$LD_LIBRARY_PATH" >> etc/conda/activate.d/env_vars.sh
    $ echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\${CONDA_PREFIX}/lib" >> etc/conda/activate.d/env_vars.sh
    $ echo "export LD_LIBRARY_PATH=\$OLD_LD_LIBRARY_PATH" >> etc/conda/deactivate.d/env_vars.sh
    $ echo "unset OLD_LD_LIBRARY_PATH" >> etc/conda/deactivate.d/env_vars.sh

**Test MPI**::

    $ mpiexec -np 2 nrniv -mpi -python -c 'from neuron import h; from mpi4py import MPI; \
                                           print("Hello from proc %d" % MPI.COMM_WORLD.Get_rank()); \
                                               h.quit()'
    numprocs=2
    NEURON -- VERSION 7.7.2 7.7 (2b7985ba) 2019-06-20
    Duke, Yale, and the BlueBrain Project -- Copyright 1984-2018
    See http://neuron.yale.edu/neuron/credits

    Hello from proc 0
    Hello from proc 1

Verifies that MPI, NEURON, and Python are all working together.

**Usage**::

    from hnn_core import MPIBackend

    # set n_procs to the number of processors MPI can use (up to number of cores on system)
    with MPIBackend(n_procs=2):
        dpls = simulate_dipole(net, n_trials=1)
