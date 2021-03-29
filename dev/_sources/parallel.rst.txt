.. _parallel:

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

    # Set n_procs to the number of processors MPI can use (up to number of cores on system)
    # A different launch command can be specified for MPI distributions other than openmpi
    with MPIBackend(n_procs=2, mpi_cmd='mpiexec'):
        dpls = simulate_dipole(net, n_trials=1)

**Notes for contributors**:

MPI parallelization with NEURON requires that the simulation be launched with the ``nrniv`` binary
from the command-line. The ``mpiexec`` command is used to launch multiple ``nrniv`` processes which
communicate via MPI. This is done using ``subprocess.Popen()`` in ``MPIBackend.simulate()`` to
launch parallel child processes (``MPISimulation``) to carry out the simulation.
The communication sequence between ``MPIBackend`` and ``MPISimulation`` is outlined below.

#. In order to pass the network to simulate from ``MPIBackend``, the child ``MPISimulation``
   processes' ``stdin`` is used. The ready-to-use `Network` object is base64 encoded and pickled
   before being written to the child processes' ``stdin`` by way of a Queue in a non-blocking way.
   See how it is `used in MNE-Python`_. The data is marked by start and end signals that are used
   to extract the pickled net object. After being unpickled, the parallel simulation begins.
#. Output from the simulation (either to ``stdout`` or ``stderr``) is communicated back
   to ``MPIBackend``, where it will be printed to the console. Typical output at this point
   would be simulation progress messages as well as any MPI warnings/errors during the simulation.
#. Once the simulation has completed, the rank 0 of the child process sends back the simulation data
   by base64 encoding and and pickling the data object. It also adds markings for the start and end
   of the encoded data, including the expected length of data (in bytes) in the end of data marking.
   Finally rank 0 writes the whole string with markings and encoded data to ``stderr``.
#. ``MPIBackend`` will look for these markings to know that data is being sent (and will not
   print this). It will verify the length of data it receives, printing a
   ``UserWarning`` if the data length received doesn't match the length part of the marking.
#. To signal that the child process should terminate, ``MPIBackend`` sends a signal to the child
   proccesses' ``stdin``. After sending the simulation data, rank 0 waits for this completion signal
   before continuing and letting all ranks of the MPI process exit successfully.
#. At this point, ``MPIBackend.simulate()`` decodes and unpickles the data, populates the network's
   CellResponse object, and returns the simulation dipoles to the caller.


It is important that ``flush()`` is used whenever data is written to stdin or stderr to ensure that the signal will immediately be available for reading by the other side.

Tests for parallel backends utilize a special ``@pytest.mark.incremental`` decorator (defined in ``conftest.py``) that causes a test failure to skip subsequent tests in the incremental block. For example, if a test running a simple MPI simulation fails, subsequent tests that compare simulation output between different backends will be skipped. These types of failures will be marked as a failure in CI.

.. _used in MNE-Python: https://github.com/mne-tools/mne-python/blob/148de1661d5e43cc88d62e27731ce44e78892951/mne/utils/misc.py#L124-L132
