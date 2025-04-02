(install)=
# Installation Guide

If you have any questions or problems while installing `hnn_core`, feel free to
ask for help [on our GitHub Discussions
page](https://github.com/jonescompneurolab/hnn-core/discussions)!

Please follow the instructions in Steps 1, 2, and 3, in that order.

--------------

# Step 1. Python Environment

We strongly recommend that you install `hnn_core` inside a "virtual
environment" using software like the [Anaconda Python
Distribution](https://www.anaconda.com/download/success). If you
are new to Python or data science in Python, we recommend you review the
resources here: <https://docs.anaconda.com/getting-started/>.

Note that `hnn-core` currently only supports Python 3.8, 3.9, 3.10, 3.11, and 3.12, but *not* 3.13.

--------------

# Step 2. Platform-specific requirements

## Windows

- Before you install `hnn_core`, it is important that you **install** the
  [NEURON][] software system-wide using its [Windows-specific
  binaries](https://nrn.readthedocs.io/en/latest/install/install_instructions.html#windows). You
  can test that NEURON was installed correctly by opening a Command Prompt (or
  similar) via your Anaconda install, and running the following command:

    ```
    $ python -c "import neuron;"
    ```

## MacOS

- If you already installed the [NEURON][] software system-wide using its
  traditional installer package, it is recommended to **remove it** first. We
  will be installing NEURON using its PyPI package alongside `hnn_core`.
- Before you install `hnn_core`, it is important that you install "Xcode
  Command-Line Tools".  This can be done easily by opening a Terminal and
  running the following command (followed by clicking through the prompts):
    ```
    $ xcode-select --install
    ```

## Linux

- If you already installed the [NEURON][] software system-wide using its
  traditional installer package, it is recommended to **remove it** first. We
  will be installing NEURON using its PyPI package alongside `hnn_core`.

--------------

# Step 3. `hnn_core` installation types

Note that Step 1 and Step 2 above are required for **all installation types**. Note that you can install multiple sets of HNN features below during the install process by combining them similar to `pip install "hnn_core[gui,parallel]"`.

## Basic Installation

To install just the `hnn_core` API, open a Terminal or Command Prompt using a fresh virtual
environment, and enter:

    $ pip install hnn_core

This will install only the `hnn_core` API along with its required
dependencies, which include:

- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [NEURON][] (>=7.7)
- [h5io](https://github.com/h5io/h5io)

To check if everything worked fine, you can run:

    $ python -c "import hnn_core"

and it should not give any error messages.

**Basic Usage**: For how to use the `hnn_core` API, which is available for every installation
type, see our extensive {doc}`Examples page <auto_examples/index>`.

## Graphical User Interface (GUI) Installation

To install `hnn_core` with both its API and GUI support, a simple tweak to the above command is
needed (pay attention to the "quotes"):

    $ pip install "hnn_core[gui]"

This will install `hnn_core`, its API dependencies, and the additional GUI dependencies, which
include:

- [ipykernel](https://ipykernel.readthedocs.io/en/stable/)
- [ipympl](https://matplotlib.org/ipympl/)
- [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) >=8.0.0
- [voila](https://github.com/voila-dashboards/voila)

**GUI Usage**: To start the GUI, simply run:

    $ hnn-gui

## Optimization Installation

If you would like to use Bayesian optimization, then
[`scikit-learn`](https://scikit-learn.org/stable/index.html) is required. You
can easily install `hnn_core`, its API dependencies, and its optimization
dependencies using the following command (pay attention to the "quotes"):

    $ pip install "hnn_core[opt]"

If you are planning to use Optimization, it is strongly recommended to also
install [support for Parallelism, which can be found
below](#parallelism).

**Optimization Usage**: An example of how to use our Optimization support can be found {doc}`in the Example
here <auto_examples/howto/optimize_evoked>`.

## Parallelism

`hnn_core` supports two kinds of parallelism:

- [Joblib parallelism](#parallelism-joblib-installation): `hnn_core` can take advantage of [the Joblib
library](https://joblib.readthedocs.io/en/stable/) in order to run multiple
independent simulations simultaneously across multiple CPU processors (also called ["embarrassingly parallel"
jobs](https://en.wikipedia.org/wiki/Embarrassingly_parallel)).

- [MPI parallelism](#parallelism-mpi-installation): `hnn_core` can take
advantage of NEURON's use of [the MPI
protocol](https://en.wikipedia.org/wiki/Message_Passing_Interface) to split
neurons across CPU processors, greatly reducing simulation time as more cores
are used. This is more complex than other installation methods, but offers the
best single-simulation speed. Note that [we do not support MPI on
Windows](#parallelism-mpi-windows).

## Parallelism: Joblib Installation

You can easily install the `hnn_core` API, its API dependencies, and its additional Joblib parallelism
dependencies using the following command (pay attention to the "quotes"):

    $ pip install "hnn_core[parallel]"

This automatically installs the following dependencies:

- [Joblib](https://joblib.readthedocs.io/en/stable/)
- [psutil](https://github.com/giampaolo/psutil)

**Parallelism: Joblib Usage**:
- Once installed, you can run multiple trials simultaneously with only a small modification to your code, like this:
    ```
    from hnn_core import JoblibBackend

    # set n_jobs to the number of trials to run in parallel with
    # Joblib (up to number of cores on system)
    with JoblibBackend(n_jobs=2):
        dpls = simulate_dipole(net, n_trials=2)
    ```
- Some in-depth examples of Joblib usage for trials are available in the Examples {doc}`found here <auto_examples/workflows/plot_simulate_alpha>` and {doc}`also here <auto_examples/workflows/plot_simulate_somato>`.
- Joblib can also be used to simultaneously run multiple simulations across "batches" of simulation parameters, illustrated {doc}`in our Batch Simulation example <auto_examples/howto/plot_batch_simulate>`.

## Parallelism: MPI Installation

For MPI installation, we **strongly** recommend you use the [Anaconda Python
Distribution](https://www.anaconda.com/download/success) specifically, since Anaconda is the easiest way to download [OpenMPI](https://anaconda.org/conda-forge/openmpi) binaries, which are *not* Python code.

### Parallelism: MPI: Windows

Unfortunately, we do not officially support MPI usage on Windows due to the
complexity required. If this is a necessity for you and you do not have access
to Linux/etc.-based HPC resources, please [get in touch via our GitHub
Discussions
page](https://github.com/jonescompneurolab/hnn-core/discussions). We may still
be able to help.

### Parallelism: MPI: MacOS

1. First, create and activate your `conda` environment (but do not install `hnn_core` yet).
2. Inside your `conda` environment, install the `conda-forge` versions of [OpenMPI](https://anaconda.org/conda-forge/openmpi) and [`mpi4py`](https://anaconda.org/conda-forge/mpi4py) using the following command:
    ```
    $ conda install -y conda-forge::openmpi conda-forge::mpi4py
    ```
3. Next, copy, paste, and run the following commands to set some environment variables for your `conda` environment:
    ```
    $ cd ${CONDA_PREFIX}
    $ mkdir -p etc/conda/activate.d etc/conda/deactivate.d
    $ echo "export OLD_DYLD_FALLBACK_LIBRARY_PATH=\$DYLD_FALLBACK_LIBRARY_PATH" >> etc/conda/activate.d/env_vars.sh
    $ echo "export DYLD_FALLBACK_LIBRARY_PATH=\$DYLD_FALLBACK_LIBRARY_PATH:\${CONDA_PREFIX}/lib" >> etc/conda/activate.d/env_vars.sh
    $ echo "export DYLD_FALLBACK_LIBRARY_PATH=\$OLD_DYLD_FALLBACK_LIBRARY_PATH" >> etc/conda/deactivate.d/env_vars.sh
    $ echo "unset OLD_DYLD_FALLBACK_LIBRARY_PATH" >> etc/conda/deactivate.d/env_vars.sh
    $ export OLD_DYLD_FALLBACK_LIBRARY_PATH=$DYLD_FALLBACK_LIBRARY_PATH
    $ export DYLD_FALLBACK_LIBRARY_PATH=$DYLD_FALLBACK_LIBRARY_PATH:${CONDA_PREFIX}/lib
    ```
4. Next, install `hnn_core` with parallel support (pay attention to the "quotes"):
    ```
    $ pip install "hnn_core[parallel]"
    ```
5. Finally, test that the install worked. Run the following command:
    ```
    $ mpiexec -np 2 nrniv -mpi -python -c 'from neuron import h; from mpi4py import MPI; \
                                       print(f"Hello from proc {MPI.COMM_WORLD.Get_rank()}"); \
                                       h.quit()'
    ```
     You should see output the looks like the following; this verifies that MPI, NEURON, and Python are all working together.

    ```
    numprocs=2
    NEURON -- VERSION 7.7.2 7.7 (2b7985ba) 2019-06-20
    Duke, Yale, and the BlueBrain Project -- Copyright 1984-2018
    See http://neuron.yale.edu/neuron/credits

    Hello from proc 0
    Hello from proc 1
    ```


### Parallelism: MPI: Linux

- Note: these instructions are for installing `hnn_core` with MPI support on
a *personal* Linux computer, not a High-Performance Computing (HPC) environment (also
called a computing cluster or supercomputer). If you are on an HPC, then you
should refer to your local HPC environment's documentation on how to load the
necessary MPI libraries/executables before installing `hnn_core`. Feel free
to reach out to us [via our GitHub Discussions
page](https://github.com/jonescompneurolab/hnn-core/discussions) if you are
unsure or run into problems; we should be able to help you get it working.

1. First, create and activate your `conda` environment (but do not install `hnn_core` yet).
2. Inside your `conda` environment, install the `conda-forge` versions of [OpenMPI](https://anaconda.org/conda-forge/openmpi) and [`mpi4py`](https://anaconda.org/conda-forge/mpi4py) using the following command:
    ```
    $ conda install -y conda-forge::openmpi conda-forge::mpi4py
    ```
3. Next, copy, paste, and run the following commands to set some environment variables for your `conda` environment:
    ```
    $ cd ${CONDA_PREFIX}
    $ mkdir -p etc/conda/activate.d etc/conda/deactivate.d
    $ echo "export OLD_LD_LIBRARY_PATH=\$LD_LIBRARY_PATH" >> etc/conda/activate.d/env_vars.sh
    $ echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\${CONDA_PREFIX}/lib" >> etc/conda/activate.d/env_vars.sh
    $ echo "export LD_LIBRARY_PATH=\$OLD_LD_LIBRARY_PATH" >> etc/conda/deactivate.d/env_vars.sh
    $ echo "unset OLD_LD_LIBRARY_PATH" >> etc/conda/deactivate.d/env_vars.sh
    $ export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
    $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CONDA_PREFIX}/lib
    ```
4. Next, install `hnn_core` with parallel support (pay attention to the "quotes"):
    ```
    $ pip install "hnn_core[parallel]"
    ```
5. Finally, test that the install worked. Run the following command:
    ```
    $ mpiexec -np 2 nrniv -mpi -python -c 'from neuron import h; from mpi4py import MPI; \
                                       print(f"Hello from proc {MPI.COMM_WORLD.Get_rank()}"); \
                                       h.quit()'
    ```
     You should see output the looks like the following; this verifies that MPI, NEURON, and Python are all working together.
    ```
    numprocs=2
    NEURON -- VERSION 7.7.2 7.7 (2b7985ba) 2019-06-20
    Duke, Yale, and the BlueBrain Project -- Copyright 1984-2018
    See http://neuron.yale.edu/neuron/credits

    Hello from proc 0
    Hello from proc 1
    ```

### Parallelism: MPI: Usage

- Once installed, you can run any simulation using multiple CPU processors with only a small modification to your code, like this:
    ```
    from hnn_core import MPIBackend

    # Set n_procs to the number of processors MPI can use (up to
    # number of cores on system). A different launch command can be
    # specified for MPI distributions other than openmpi.
    with MPIBackend(n_procs=2, mpi_cmd='mpiexec'):
        dpls = simulate_dipole(net, n_trials=1)
    ```
- An in-depth example of MPI usage is available in the {doc}`Example found here <auto_examples/howto/plot_simulate_mpi_backend>`.

## Installing multiple optional `hnn_core` features

Note that you easily install multiple groups of optional features. For example,
if you want to install both GUI support and Optimization support, then simply
run the following:

    $ pip install "hnn_core[gui,opt]"

[NEURON]: https://nrn.readthedocs.io/

