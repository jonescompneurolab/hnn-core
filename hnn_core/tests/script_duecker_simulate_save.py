#!/usr/bin/env python

from hnn_core import (
    MPIBackend,
    duecker_ET_model,
    simulate_dipole,
)
from hnn_core.hnn_io import (
    write_network_configuration,
)


def rerun_and_save_duecker_model(suffix="new", backend="mpi"):
    """Rerun the Duecker ET model and save its network config and output.

    Builds the Duecker ET network in a similar way to the ``handtune_new_syn.ipynb``
    notebook in the ``duecker_ET_model`` branch of the ``hnn-tuning`` repository at
    (as of approximately 2026-07-15):

    https://github.com/jonescompneurolab/hnn-tuning/blob/duecker_ET_model/new_model_HNN/ERP/handtune_new_syn.ipynb

    then runs a single 170 ms simulation with the ``"duecker"`` baseline correction,
    then writes the results to the current working directory depending on ``suffix``.

    If ``suffix="old"`` then this will OVERWRITE existing output data files that are
    used as the "ground truth" for tests involving whether a recent code change has
    affected the output of the Duecker ET model. If ``suffix="new"`` (the default),
    simulation data will be created that can be used to compare against the "old" ground
    truth data, in order to check if any recent code changes have a material effect on
    the output of the Duecker ET model.

    Note that the NEURON mod files must be recompiled (``make``) before
    calling this, otherwise the simulation will use stale mechanisms.

    Parameters
    ----------
    suffix : str, default="new"
        String appended to the spike and dipole output filenames, used to
        distinguish runs. Accepted values are ``"old"`` and ``"new"``.
    backend : str, default="mpi"
        Parallel backend to simulate with, either ``"mpi"`` (runs inside an
        :class:`~hnn_core.MPIBackend` context using ``mpiexec``) or
        ``"joblib"`` (uses whatever backend is currently active).

    Returns
    -------
    None
        Results are written to disk rather than returned:

        - ``net_d_duecker.json`` : the network configuration
        - ``spikes_duecker_output_{suffix}.txt`` : spike times of all cells
        - ``dipole_duecker_output_{suffix}.txt`` : dipole waveform of trial 0
    """
    # Setup network, including drives
    net = duecker_ET_model(add_erp_drives=True)

    # Begin simulation
    if backend == "mpi":
        with MPIBackend(mpi_cmd="mpiexec"):
            dpls = simulate_dipole(net, tstop=170.0, baseline_correction=True)
    elif backend == "joblib":
        dpls = simulate_dipole(net, tstop=170.0, baseline_correction=True)
    else:
        raise ValueError(f"backend must be either 'mpi' or 'joblib', got '{backend}'")

    # Save all output
    write_network_configuration(net, "net_d_duecker.json")
    net.cell_response.write(f"spikes_duecker_output_{suffix}.txt")
    dpls[0].write(f"dipole_duecker_output_{suffix}.txt")


if __name__ == "__main__":
    rerun_and_save_duecker_model(suffix="old")
