#!/usr/bin/env python

from hnn_core import (
    MPIBackend,
    duecker_ET_model,
    simulate_dipole,
)
from hnn_core.hnn_io import (
    write_network_configuration,
)


def rerun_and_save_duecker_model(suffix=""):
    print("---------------------------------\n")
    print("-->REMEMBER to recompile the mods\n")
    print("---------------------------------\n")

    # net = neymotin_2020_model(add_drives_from_params=True)
    # # keyerror currently
    # net = duecker_ET_model(add_drives_from_params=True)

    # --------------------------------------------------------------------------------------
    # Begin network setup and drive config from
    # https://github.com/jonescompneurolab/hnn-tuning/blob/duecker_ET_model/new_model_HNN/ERP/handtune_new_syn.ipynb
    net = duecker_ET_model()
    net.set_cell_positions(inplane_distance=30.0)
    weights_ampa_p1 = {
        "L2_basket": 0.01,
        "L2_pyramidal": 0.015,
        "L5_basket": 0.0,
        "L5ET": 0.03,
    }
    weights_nmda_p1 = {
        "L2_basket": 0.01,
        "L2_pyramidal": 0.05,
        "L5_basket": 0.0,
        "L5ET": 0.025,
    }
    synaptic_delays_prox = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1,
        "L5ET": 1,
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

    weights_ampa_d1 = {"L2_basket": 0.005, "L2_pyramidal": 0.01, "L5ET": 1.0}
    weights_nmda_d1 = {"L2_basket": 0.0, "L2_pyramidal": 0.01, "L5ET": 1.0}
    synaptic_delays_dist = {"L2_basket": 0.1, "L2_pyramidal": 0.1, "L5ET": 0.1}

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
        "L2_basket": 0.01,
        "L2_pyramidal": 0.3,
        "L5_basket": 0.001,
        "L5ET": 0.3,
    }
    weights_nmda_p2 = {
        "L2_basket": 0.01,
        "L2_pyramidal": 0.2,
        "L5_basket": 0.001,
        "L5ET": 0.2,
    }
    synaptic_delays_prox = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5ET": 1.0,
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
    with MPIBackend(mpi_cmd="mpiexec"):
        dpls = simulate_dipole(net, tstop=170.0, bsl_cor="duecker")
        # dpls = simulate_dipole(net, tstop=170.0, bsl_cor="jones")

    write_network_configuration(net, "net_d_duecker.json")

    net.cell_response.write(f"spikes_duecker_output_{suffix}.txt")
    dpls[0].write(f"dipole_duecker_output_{suffix}.txt")


if __name__ == "__main__":
    rerun_and_save_duecker_model(suffix="old")
