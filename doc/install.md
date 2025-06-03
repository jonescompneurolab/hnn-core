(install)=
# Installation Guide

If you have any questions or problems while installing HNN-Core, feel free to
ask for help on [our GitHub Discussions page][]!

If you wish to try out HNN-Core in the cloud without installing it locally, see the [Installation page at the HNN Textbook](https://jonescompneurolab.github.io/textbook/content/01_getting_started/installation.html).

Note: If you are at Brown University and trying to install HNN-Core on the OSCAR system, you should **not** follow these instructions, and instead follow the [instructions here](https://github.com/jonescompneurolab/oscar-install).

--------------

# `conda` Installation

We recommend that you install HNN-Core from our Anaconda package instead of from `pip`. This `conda` package includes all features (GUI, Optimization, and both Parallelism features), a working installation of OpenMPI, and is significantly easier to install.

The only exceptions are if:

1. You want to install HNN-Core alongside MNE. In that case, we recommend you use the one of the below [`pip` Installation](#pip-installation) methods.
2. You are using a High-Performance Computing (HPC) environment (also called a computing cluster or supercomputer). The `conda` installation may work on your HPC, but if you want to use your own MPI libraries, you can try our [`pip` MPI method](#pip-mpi-macos-or-linux), and you can always ask us for help at [our Github Discussions page][].

If you want a minimal install of only the HNN-Core API using `conda`, then follow the instructions below, but replace the package name `hnn-core-all` with `hnn-core`.

Follow the below instructions for your operating system.

## `conda` MacOS or Linux

1. Install the [Anaconda Python Distribution](https://www.anaconda.com/download/success).

2. Open a Terminal, then create and activate a new Python 3.12 environment by running the following commands:

```
conda create -y -q -n hnn-core-env python=3.12
conda activate hnn-core-env
```

If desired, you can change the environment name `hnn-core-env` to whatever you wish. If you are new to Python or
data science in Python, we recommend you review the resources here:
<https://docs.anaconda.com/getting-started/>.

3. Install our package using the following command:

```
conda install hnn-core-all -c jonescompneurolab -c conda-forge
```

4. Run the following command, and write down the number that is output. You can use this number as the number of CPU "Cores", which will *greatly* speed up your simulations.

```
python -c "import psutil ; print(psutil.cpu_count(logical=False)-1)"
```

5. That's it! HNN-Core should now be installed. Proceed to [our HNN Textbook website][] to get started using HNN.
6. Note: The next time you need to re-enter the Conda Environment, all you need to do is run `conda activate hnn-core-env`. From there, you can either run Python code using the HNN-Core API that you have written, or start the GUI using the `hnn-gui` command.

## `conda` Windows

For Windows users, there are some extra steps needed since you need to install HNN-Core through "Windows Subsystem for Linux" (WSL).

1. Install WSL: Open the "PowerShell" or "Windows Command Prompt" programs in administrator mode by right-clicking the program icon and selecting "Run as administrator". Then, in the window, run the following command:

```
wsl --install
```

Follow the default options for your install. For more information, see <https://learn.microsoft.com/en-us/windows/wsl/install>.

2. Close the PowerShell or Command Prompt window.
3. You will now have a new App available from the Start Menu called `Ubuntu`. Run that app.
4. The first time you start Ubuntu, it will prompt you to `Create a default Unix user account` and ask for a password. If you provide a password, write it down.
5. You should now see a prompt and a blinking cursor similar to PowerShell/Command Prompt. Copy and paste the following commands. If you entered a password in Step 4, enter that when it prompts you for your password.

```bash
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install wget
```

6. In the same window, copy and paste the following commands, then follow the prompts. We strongly recommend that when you are asked  `Do you wish to update your shell profile to automatically initialize conda?` you enter `yes`. If you do not, then you will have to manually activate Conda whenever you open your Ubuntu app using the command this program provides at the end.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

If you see output similar to `WARNING: Your machine hardware does not appear to be x86_64`, then please contact us via [our Github Discussions page][]. You may be able to install by using <https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh> instead.

7. Close the Ubuntu window, then open a new one. Your prompt should now show `(base)` at the beginning.
8. Inside this window, follow the steps above in the [`conda` MacOS or Linux](#conda-macos-or-linux) section, then return here.
9. HNN-Core should now be installed!
10. Note: On Windows, every time you start the GUI, you will need to navigate to <http://localhost:8866> in your browser (or refresh if you are already on the page).

--------------

# `pip` Installation

Alternatively, you can install HNN-Core through `pip` if you want install it alongside MNE, to contribute to development, only want certain features, or have issues with the `conda` install method. If using the `pip` method, please follow the instructions in Steps 1, 2, and 3, in that order.

## Step 1. `pip` Python Environment

We strongly recommend that you install HNN-Core inside a "virtual environment" using software like the [Anaconda Python Distribution](https://www.anaconda.com/download/success). If you are new to Python or data science in Python, we recommend you review the resources here: <https://docs.anaconda.com/getting-started/>.

Note that if you use a virtual environment, you must first create the environment, and then separately *activate* the environment (see [Conda guidance here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment)) before running any of the commands below. **All of the installation commands below assume you have already activated your environment**. You may need to re-activate your environment every time you restart your computer or open a new terminal, depending on how you installed your virtual environment software.

Note that `hnn_core` currently only supports Python 3.8, 3.9, 3.10, 3.11, and 3.12, but *not* 3.13.

## Step 2. `pip` Platform-specific requirements

### `pip` on MacOS

- If you already installed the [NEURON][] software system-wide using its
  traditional installer package, it is recommended to **remove it** first. We
  will be installing NEURON using its PyPI package.
- Before you install HNN-Core, it is important that you install "Xcode Command-Line Tools". This can be done easily by opening a Terminal, running the following command, and then clicking through the prompts in the window that will pop up. Note that you must restart your computer after Xcode Command-Line Tools has finished installing!

```
xcode-select --install
```

If you run the above command and see output that resembles `xcode-select: note: Command line tools are already installed.`, then you already have it installed, do not need to restart your computer, and can proceed to the next step.

### `pip` on Linux

- If you already installed the [NEURON][] software system-wide using its
  traditional installer package, it is recommended to **remove it** first. We
  will be installing NEURON using its PyPI package.

### `pip` on Windows

- Before you install HNN-Core, it is important that you **install** the [NEURON][] software system-wide using its [Windows-specific binaries](https://nrn.readthedocs.io/en/latest/install/install_instructions.html#windows). You can test that NEURON was installed correctly by opening a Command Prompt (or similar) via your Anaconda install (or virtual environment), and running the following command:

```
python -c "import neuron"
```

## Step 3. `pip` installation types

Note that you can install multiple sets of HNN-Core features below by combining them similar to `pip install "hnn_core[gui,parallel]"`.

### `pip` Basic Installation

To install only the `hnn_core` API, open a Terminal or Command Prompt using a fresh virtual
environment, and enter:

```
pip install hnn_core
```

This will install only the `hnn_core` API along with its required dependencies, which include: [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Matplotlib](https://matplotlib.org/), [NEURON][] (>=7.7), and [h5io](https://github.com/h5io/h5io). To check if everything worked fine, you can run:

```
python -c "from hnn_core import jones_2009_model, simulate_dipole ; simulate_dipole(jones_2009_model(), tstop=20)"
```

This will run a very short test simulation, and should not give any Error messages (Warning messages are fine and expected). For how to use the `hnn_core` API, see [our HNN Textbook website][], including examples like our [API tutorial of ERP simulation](https://jonescompneurolab.github.io/textbook/content/05_erps/hnn_core.html). Our public API documentation {doc}`can be found here <api>`.

### `pip` Graphical User Interface (GUI) Installation

To install `hnn_core` with both its API and GUI support, run the following command (make sure to include the "quotes"):

```
pip install "hnn_core[gui]"
```

This will install `hnn_core`, its API dependencies, and the additional GUI dependencies, which
include: [ipykernel](https://ipykernel.readthedocs.io/en/stable/), [ipympl](https://matplotlib.org/ipympl/), [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) >=8.0.0, and [voila](https://github.com/voila-dashboards/voila).

To start the GUI, simply run:

```
hnn-gui
```

For further guidance on using the GUI, see [our HNN Textbook website][], including examples like our [GUI tutorial of ERPs](https://jonescompneurolab.github.io/textbook/content/05_erps/erps_in_gui.html).

### `pip` Optimization Installation

If you would like to use Bayesian optimization, then
[`scikit-learn`](https://scikit-learn.org/stable/index.html) is required. You
can easily install `hnn_core`, its API dependencies, and its optimization
dependencies using the following command (make sure to include the "quotes"):

```
pip install "hnn_core[opt]"
```

If you are planning to use Optimization, it is recommended to also install Joblib support (see below). Instruction on how to use our Optimization support can be found on [our HNN Textbook website][], including this [optimization tutorial](https://jonescompneurolab.github.io/textbook/content/04_using_hnn/optimize_simulated_evoked_response_parameters.html).

### `pip` Joblib Installation

You can easily install the `hnn_core` API, its API dependencies, and its additional Joblib parallelism
dependencies using the following command (make sure to include the "quotes"):

```
pip install "hnn_core[parallel]"
```

This automatically installs the following dependencies: [Joblib](https://joblib.readthedocs.io/en/stable/) and [psutil](https://github.com/giampaolo/psutil).

Many of the examples on [our HNN Textbook website][] make use of Joblib support, and we provide an explicit [tutorial for it here](https://jonescompneurolab.github.io/textbook/content/04_using_hnn/parallelism_joblib.html).

### `pip` MPI Installation

If you want to use MPI, we recommend you first try to install our `conda` package detailed [at the top of the page](#conda-installation). Otherwise, we **strongly** recommend you use the [Anaconda Python Distribution](https://www.anaconda.com/download/success) specifically, since Anaconda is the easiest way to download [OpenMPI](https://anaconda.org/conda-forge/openmpi) binaries, which are *not* Python code. If you need to use your own specific MPI binaries/libraries (such as if you are installing on an HPC platform) and you need help, then let us know on [our Github Discussions page][].

To install HNN-Core with its MPI dependencies using the `pip` method, follow the below instructions for your operating system.

#### `pip` MPI: Windows

Unfortunately, we do not officially support MPI usage on native Windows due to the complexity required. However, we do support MPI through Windows Subsystem for Linux (WSL). To install `pip` MPI support on WSL, follow the Steps 1 through 7 of the above [`conda` Windows guide](#conda-windows); this will install WSL and Miniconda. Then, inside your `Ubuntu` app window, follow the `pip` steps below for Linux, beginning with step 2.

#### `pip` MPI: MacOS or Linux

1. Install the [Anaconda Python Distribution](https://www.anaconda.com/download/success).

2. Create and activate a new `conda` environment using commands like the following:

```
conda create -y -q -n hnn-core-env python=3.12
conda activate hnn-core-env
```

If desired, you can change the environment name `hnn-core-env` to whatever you wish. Compatible Python versions currently include 3.8, 3.9, 3.10, 3.11, and 3.12, but *not* 3.13.

3. (MacOS only) Copy, paste, and run the following commands to set some environment variables for your `conda` environment:

```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d $CONDA_PREFIX/etc/conda/deactivate.d
echo "export OLD_DYLD_FALLBACK_LIBRARY_PATH=\$DYLD_FALLBACK_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "export DYLD_FALLBACK_LIBRARY_PATH=\$DYLD_FALLBACK_LIBRARY_PATH:\${CONDA_PREFIX}/lib" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "export DYLD_FALLBACK_LIBRARY_PATH=\$OLD_DYLD_FALLBACK_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo "unset OLD_DYLD_FALLBACK_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
export OLD_DYLD_FALLBACK_LIBRARY_PATH=$DYLD_FALLBACK_LIBRARY_PATH
export DYLD_FALLBACK_LIBRARY_PATH=$LD_LIBRARY_PATH:${CONDA_PREFIX}/lib
```

4. (Linux only) Copy, paste, and run the following commands to set some environment variables for your `conda` environment:

```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d $CONDA_PREFIX/etc/conda/deactivate.d
echo "export OLD_LD_LIBRARY_PATH=\$LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\${CONDA_PREFIX}/lib" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "export LD_LIBRARY_PATH=\$OLD_LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo "unset OLD_LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CONDA_PREFIX}/lib
```

5. Run the following command to install your MPI dependencies (if you have your own MPI binaries, change this step accordingly to load/etc. them):

```
conda install -y -q "openmpi>5" mpi4py -c conda-forge
```

6. Finally, install the PyPI package of `hnn_core` using the following command (make sure to include the "quotes"):

```
pip install "hnn_core[parallel]"
```

Note that you can add other HNN-Core features to the install by, for example, changing `[parallel]` in the above command to `[opt,parallel]`, etc.

7. Let's test that the install worked. Run the following command:

```
mpiexec -np 2 nrniv -mpi -python -c 'from neuron import h; from mpi4py import MPI; \
                                   print(f"Hello from proc {MPI.COMM_WORLD.Get_rank()}"); \
                                   h.quit()'
```

You should see output the looks like the following; this verifies that MPI, NEURON, and Python are all working together.

```
numprocs=2
NEURON -- VERSION 8.2.6-1-gb6e6a5fad+ build-osx-wheels-script (b6e6a5fad+) 2024-07-25
, Yale, and the BlueBrain Project -- Copyright 1984-2022
See http://neuron.yale.edu/neuron/credits

Hello from proc 0
Hello from proc 1
```

8. You can find tutorials on how to use MPI parallelism on [our HNN Textbook website][], specifically on [our MPI tutorial here](https://jonescompneurolab.github.io/textbook/content/04_using_hnn/parallelism_mpi.html).

[NEURON]: https://nrn.readthedocs.io/
[our HNN Textbook website]: https://jonescompneurolab.github.io/textbook/content/preface.html
[our GitHub Discussions page]: https://github.com/jonescompneurolab/hnn-core/discussions
