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

    $ conda install -y openmpi mpi4py
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

**Notes for contributors**::

MPI parallelization with NEURON requires that the simulation be launched with the ``nrniv`` binary from the command-line. The ``mpiexec`` command is used to launch multiple ``nrniv`` processes which communicate via MPI. This is done using ``subprocess.Popen()`` in ``MPIBackend.simulate()`` to launch parallel child processes (``MPISimulation``) to carry out the simulation. The communication sequence between ``MPIBackend`` and ``MPISimulation`` is outlined below.

#. In order to pass the parameters from ``MPIBackend`` the child ``MPISimulation`` processes' ``stdin`` is used. Parameters are pickled and base64 encoded before being written to the processes' ``stdin``. Closing ``stdin`` (from the ``MPIBackend`` side) signals to the child processes that it is done sending parameters and the parallel simulation should begin.
#. Output from the simulation (either to ``stdout`` or ``stderr``) is communicated back to ``MPIBackend``, where it will be printed to the console. Typical output at this point would be simulation progress messages as well as any MPI warnings/errors during the simulation.
#. Once the simulation has completed, the child process with rank 0 (in ``MPISimulation.run()``) sends a signal to ``MPIBackend`` that the simulation has completed and simulation data will be written to ``stderr``.  The data is pickled and base64 encoded before it is written to ``stderr`` in ``MPISimulation._write_data_stderr()``. No other output (e.g. raised exceptions) can go to ``stderr`` during this step.
#. At this point, the child process with rank 0 (the only rank with complete simulation results) will send another signal that includes the expected length of the pickled and encoded data (in bytes) to ``stderr`` following the data written in the previous step. ``MPIBackend`` will use this signal to know that data transfer has completed and it will verify the length of data it receives, printing a ``UserWarning`` if the lengths don't match.

It is important that ``MPISimulation`` uses the ``flush()`` method after each signal to ensure that the signal will immediately be available for reading by ``MPIBackend`` and not buffered with other output.

Tests for parallel backends utilize a special ``@pytest.mark.incremental`` decorator (defined in ``conftest.py``) that causes a test failure to skip subsequent tests in the incremental block. For example, if a test running a simple MPI simulation fails, subsequent tests that compare simulation output between different backends will be skipped. These types of failures will be marked as a failure in CI.