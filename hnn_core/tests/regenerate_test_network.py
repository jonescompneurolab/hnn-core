#!/usr/bin/env python

from pathlib import Path
from shutil import copy

from hnn_core import read_params, jones_2009_model

hnn_core_root = Path(__file__).parents[1]
assets_path = Path(hnn_core_root, "tests", "assets")


def jones_2009_additional_features():
    """Instantiate default network with more features for testing purposes.

    Note: Depending on differences between CPU architectures, OS, and other
    system variables, this may produce a network for which there are small
    floating-point differences in the value "end_pts" of instantiated cells. If
    the differences are small in this case, then regenerating the network can
    probably be done safely. In the event that there are small (decimal place)
    differences in OTHER cell values, such as conductances, and you need to
    regenerate the network, then please discuss with the HNN Development Team
    before pushing your newly-regenerated test network.
    """

    params_path = Path(hnn_core_root, "param", "default.json")
    params = read_params(params_path)

    net = jones_2009_model(
        params=params, add_drives_from_params=True, mesh_shape=(3, 3)
    )

    # Adding bias
    tonic_bias = {
        "L2_pyramidal": 1.0,
        "L5_pyramidal": 0.0,
        "L2_basket": 0.0,
        "L5_basket": 0.0,
    }
    net.add_tonic_bias(amplitude=tonic_bias)

    # Add drives
    location = "proximal"
    burst_std = 20
    weights_ampa_p = {
        "L2_pyramidal": 5.4e-5,
        "L5_pyramidal": 5.4e-5,
        "L2_basket": 0.0,
        "L5_basket": 0.0,
    }
    weights_nmda_p = {
        "L2_pyramidal": 0.0,
        "L5_pyramidal": 0.0,
        "L2_basket": 0.0,
        "L5_basket": 0.0,
    }
    syn_delays_p = {
        "L2_pyramidal": 0.1,
        "L5_pyramidal": 1.0,
        "L2_basket": 0.0,
        "L5_basket": 0.0,
    }
    net.add_bursty_drive(
        "alpha_prox",
        tstart=1.0,
        burst_rate=10,
        burst_std=burst_std,
        numspikes=2,
        spike_isi=10,
        n_drive_cells=10,
        location=location,
        weights_ampa=weights_ampa_p,
        weights_nmda=weights_nmda_p,
        synaptic_delays=syn_delays_p,
        event_seed=284,
    )

    weights_ampa = {
        "L2_pyramidal": 0.0008,
        "L5_pyramidal": 0.0075,
        "L2_basket": 0.0,
        "L5_basket": 0.0,
    }
    synaptic_delays = {
        "L2_pyramidal": 0.1,
        "L5_pyramidal": 1.0,
        "L2_basket": 0.0,
        "L5_basket": 0.0,
    }
    rate_constant = {
        "L2_pyramidal": 140.0,
        "L5_pyramidal": 40.0,
        "L2_basket": 40.0,
        "L5_basket": 40.0,
    }
    net.add_poisson_drive(
        "poisson",
        rate_constant=rate_constant,
        weights_ampa=weights_ampa,
        weights_nmda=weights_nmda_p,
        location="proximal",
        synaptic_delays=synaptic_delays,
        event_seed=1349,
    )

    # Adding electrode arrays
    electrode_pos = (1, 2, 3)
    net.add_electrode_array("el1", electrode_pos)
    electrode_pos = [(1, 2, 3), (-1, -2, -3)]
    net.add_electrode_array("arr1", electrode_pos)

    return net


if __name__ == "__main__":
    new_test_network_path = assets_path.joinpath("jones2009_3x3_drives.json")

    backup_path = Path(
        new_test_network_path.parent,
        (
            new_test_network_path.stem
            + "-pre_regen_BACKUP"
            + new_test_network_path.suffix
        ),
    )

    print(f"""
Note that this will OVERWRITE the current contents of

'{new_test_network_path}'

with a fresh generation of the smaller test network! Only use this if you know
what you are doing. This will make a backup located at

'{backup_path}'
    """)

    copy(new_test_network_path, backup_path)

    net = jones_2009_additional_features()
    net.write_configuration(new_test_network_path)
