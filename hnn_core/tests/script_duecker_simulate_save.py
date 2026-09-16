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
        distinguish runs (e.g. ``"old"`` vs. ``"new"``).
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
    print("---------------------------------\n")
    print("-->REMEMBER to recompile the mods\n")
    print("---------------------------------\n")

    # --------------------------------------------------------------------------------------
    # Begin network setup and drive config
    net = duecker_ET_model()
    net.set_cell_positions(inplane_distance=30.0)
    weights_ampa_p1 = {
        "L2_inhibitory": 0.01,
        "L2_pyramidal": 0.015,
        "L5_inhibitory": 0.0,
        "L5_pyramidal": 0.03,
    }
    weights_nmda_p1 = {
        "L2_inhibitory": 0.01,
        "L2_pyramidal": 0.05,
        "L5_inhibitory": 0.0,
        "L5_pyramidal": 0.025,
    }
    synaptic_delays_prox = {
        "L2_inhibitory": 0.1,
        "L2_pyramidal": 0.1,
        "L5_inhibitory": 1,
        "L5_pyramidal": 1,
    }

    net.add_evoked_drive(
        "prox1",
        mu=18,
        sigma=2.5,
        numspikes=1,
        weights_ampa=weights_ampa_p1,
        weights_nmda=weights_nmda_p1,
        location="proximal",
        synaptic_delays=synaptic_delays_prox,
    )

    weights_ampa_d1 = {
        "L2_inhibitory": 0.005,
        "L2_pyramidal": 0.01,
        "L5_pyramidal": 1.0,
    }
    weights_nmda_d1 = {"L2_inhibitory": 0.0, "L2_pyramidal": 0.01, "L5_pyramidal": 1.0}
    synaptic_delays_dist = {
        "L2_inhibitory": 0.1,
        "L2_pyramidal": 0.1,
        "L5_pyramidal": 0.1,
    }

    net.add_evoked_drive(
        "dist1",
        mu=62,
        sigma=5,
        numspikes=2,
        weights_ampa=weights_ampa_d1,
        weights_nmda=weights_nmda_d1,
        location="distal",
        synaptic_delays=synaptic_delays_dist,
    )

    weights_ampa_p2 = {
        "L2_inhibitory": 0.01,
        "L2_pyramidal": 0.3,
        "L5_inhibitory": 0.001,
        "L5_pyramidal": 0.3,
    }
    weights_nmda_p2 = {
        "L2_inhibitory": 0.01,
        "L2_pyramidal": 0.2,
        "L5_inhibitory": 0.001,
        "L5_pyramidal": 0.2,
    }
    synaptic_delays_prox = {
        "L2_inhibitory": 0.1,
        "L2_pyramidal": 0.1,
        "L5_inhibitory": 1.0,
        "L5_pyramidal": 1.0,
    }
    net.add_evoked_drive(
        "prox2",
        mu=100,
        sigma=15,
        numspikes=1,
        weights_ampa=weights_ampa_p2,
        weights_nmda=weights_nmda_p2,
        location="proximal",
        synaptic_delays=synaptic_delays_prox,
    )
    # End of drive config
    # --------------------------------------------------------------------------------------
    if backend == "mpi":
        with MPIBackend(mpi_cmd="mpiexec"):
            dpls = simulate_dipole(net, tstop=170.0, bsl_cor="duecker")
    elif backend == "joblib":
        dpls = simulate_dipole(net, tstop=170.0, bsl_cor="duecker")
    else:
        raise ValueError(f"backend must be either 'mpi' or 'joblib', got '{backend}'")

    write_network_configuration(net, "net_d_duecker.json")

    net.cell_response.write(f"spikes_duecker_output_{suffix}.txt")
    dpls[0].write(f"dipole_duecker_output_{suffix}.txt")


if __name__ == "__main__":
    rerun_and_save_duecker_model(suffix="old")
