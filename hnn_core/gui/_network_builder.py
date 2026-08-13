"""Business logic for constructing a Network from GUI-collected parameter values."""

from copy import deepcopy
from functools import partial

from hnn_core.cells_default import _exp_g_at_dist
from hnn_core.gui._logging import logger
from hnn_core.hnn_io import dict_to_network
from hnn_core.network import pick_connection

global_gain_type_lookup_dict = {
    ("L2_pyramidal", "L2_pyramidal"): "e_e",
    ("L2_pyramidal", "L5_pyramidal"): "e_e",
    ("L5_pyramidal", "L5_pyramidal"): "e_e",
    ("L2_pyramidal", "L2_basket"): "e_i",
    ("L2_pyramidal", "L5_basket"): "e_i",
    ("L5_pyramidal", "L5_basket"): "e_i",
    ("L2_basket", "L2_pyramidal"): "i_e",
    ("L2_basket", "L5_pyramidal"): "i_e",
    ("L5_basket", "L5_pyramidal"): "i_e",
    ("L2_basket", "L2_basket"): "i_i",
    ("L5_basket", "L5_basket"): "i_i",
}


def _drive_widget_to_dict(drive, name):
    """Creates a dict of input widget values
    ... (unchanged body)
    """
    return {k: v.value for k, v in drive[name].items()}


def _init_network_from_widgets(
    params,
    dt,
    tstop,
    single_simulation_data,
    drive_widgets,
    connectivity_textfields,
    cell_params_vboxes,
    global_gain_textfields,
    add_drive=True,
):
    """Construct network and add drives."""
    # ... unchanged body (uses global_gain_type_lookup_dict defined above) ...
    logger.info("init network")
    single_simulation_data["net"] = dict_to_network(
        params, read_drives=False, read_external_biases=False
    )

    # Update with synaptic gains
    global_gain_values = {
        key: widget.value for key, widget in global_gain_textfields.items()
    }

    # adjust connectivity according to the connectivity_tab
    for connectivity_slider in connectivity_textfields:
        for vbox_key in connectivity_slider:
            conn_indices = pick_connection(
                net=single_simulation_data["net"],
                src_gids=vbox_key._belongsto["src_gids"],
                target_gids=vbox_key._belongsto["target_gids"],
                loc=vbox_key._belongsto["location"],
                receptor=vbox_key._belongsto["receptor"],
            )

            if len(conn_indices) > 0:
                assert len(conn_indices) == 1
                conn_idx = conn_indices[0]
                single_simulation_data["net"].connectivity[conn_idx]["nc_dict"][
                    "A_weight"
                ] = vbox_key.children[1].children[0].value

                # 1. identify which case of global_gain_textfield applies to this
                #    src/target
                global_gain_type = global_gain_type_lookup_dict[
                    (
                        vbox_key._belongsto["src_gids"],
                        vbox_key._belongsto["target_gids"],
                    )
                ]
                applied_global_gain_value = global_gain_values[global_gain_type]

                # 2. Multiply global by single synapse gain to get total
                single_simulation_data["net"].connectivity[conn_idx]["nc_dict"][
                    "gain"
                ] = (
                    1
                    + (applied_global_gain_value - 1)
                    + (vbox_key.children[2].children[0].value - 1)
                )

    # Update cell params
    update_functions = {
        "L2 Geometry": _update_L2_geometry_cell_params,
        "L5 Geometry": _update_L5_geometry_cell_params,
        "Synapses": _update_synapse_cell_params,
        "L2 Pyramidal_Biophysics": _update_L2_biophysics_cell_params,
        "L5 Pyramidal_Biophysics": _update_L5_biophysics_cell_params,
    }

    # Update cell params
    for vbox_key, cell_param_list in cell_params_vboxes.items():
        for key, update_function in update_functions.items():
            if key in vbox_key:
                cell_type = vbox_key.split()[0]
                update_function(
                    single_simulation_data["net"], cell_type, cell_param_list.children
                )
                break  # update needed only once per vbox_key

    for cell_type in single_simulation_data["net"].cell_types.keys():
        single_simulation_data["net"].cell_types[cell_type][
            "cell_object"
        ]._update_end_pts()
        single_simulation_data["net"].cell_types[cell_type][
            "cell_object"
        ]._compute_section_mechs()

    if add_drive is False:
        return
    # add drives to network
    for drive in drive_widgets:
        if drive["type"] in ("Tonic"):
            weights_amplitudes = _drive_widget_to_dict(drive, "amplitude")
            single_simulation_data["net"].add_tonic_bias(
                bias_name=drive["name"],
                amplitude=weights_amplitudes,
                t0=drive["t0"].value,
                tstop=drive["tstop"].value,
            )
        else:
            sync_inputs_kwargs = dict(
                n_drive_cells=(
                    "n_cells"
                    if drive["is_cell_specific"].value
                    else drive["n_drive_cells"].value
                ),
                cell_specific=drive["is_cell_specific"].value,
            )

            weights_ampa = _drive_widget_to_dict(drive, "weights_ampa")
            weights_nmda = _drive_widget_to_dict(drive, "weights_nmda")
            synaptic_delays = _drive_widget_to_dict(drive, "delays")
            logger.info(f"drive type is {drive['type']}, location={drive['location']}")
            if drive["type"] == "Poisson":
                rate_constant = _drive_widget_to_dict(drive, "rate_constant")

                single_simulation_data["net"].add_poisson_drive(
                    name=drive["name"],
                    tstart=drive["tstart"].value,
                    tstop=drive["tstop"].value,
                    rate_constant=rate_constant,
                    location=drive["location"],
                    weights_ampa=weights_ampa,
                    weights_nmda=weights_nmda,
                    synaptic_delays=synaptic_delays,
                    space_constant=100.0,
                    event_seed=drive["seedcore"].value,
                    **sync_inputs_kwargs,
                )
            elif drive["type"] in ("Evoked", "Gaussian"):
                single_simulation_data["net"].add_evoked_drive(
                    name=drive["name"],
                    mu=drive["mu"].value,
                    sigma=drive["sigma"].value,
                    numspikes=drive["numspikes"].value,
                    location=drive["location"],
                    weights_ampa=weights_ampa,
                    weights_nmda=weights_nmda,
                    synaptic_delays=synaptic_delays,
                    space_constant=3.0,
                    event_seed=drive["seedcore"].value,
                    **sync_inputs_kwargs,
                )
            elif drive["type"] in ("Rhythmic", "Bursty"):
                single_simulation_data["net"].add_bursty_drive(
                    name=drive["name"],
                    tstart=drive["tstart"].value,
                    tstart_std=drive["tstart_std"].value,
                    tstop=drive["tstop"].value,
                    location=drive["location"],
                    burst_rate=drive["burst_rate"].value,
                    burst_std=drive["burst_std"].value,
                    numspikes=drive["numspikes"].value,
                    weights_ampa=weights_ampa,
                    weights_nmda=weights_nmda,
                    synaptic_delays=synaptic_delays,
                    event_seed=drive["seedcore"].value,
                    **sync_inputs_kwargs,
                )



def _update_L2_geometry_cell_params(net, cell_param_key, param_list):
    # ... unchanged body ...
    cell_params = param_list
    cell_type = f"{cell_param_key.split('_')[0]}_pyramidal"

    sections = net.cell_types[cell_type]["cell_object"].sections
    # Soma
    sections["soma"]._L = cell_params[0].value
    sections["soma"]._diam = cell_params[1].value
    sections["soma"]._cm = cell_params[2].value
    sections["soma"]._Ra = cell_params[3].value

    # Dendrite common parameters
    dendrite_cm = cell_params[4].value
    dendrite_Ra = cell_params[5].value

    dendrite_sections = [name for name in sections.keys() if name != "soma"]

    param_indices = [(6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19)]

    # Dendrite
    for section, indices in zip(dendrite_sections, param_indices):
        sections[section]._L = cell_params[indices[0]].value
        sections[section]._diam = cell_params[indices[1]].value
        sections[section]._cm = dendrite_cm
        sections[section]._Ra = dendrite_Ra

def _update_L5_geometry_cell_params(net, cell_param_key, param_list):
    # ... unchanged body ...
    cell_params = param_list
    cell_type = f"{cell_param_key.split('_')[0]}_pyramidal"

    sections = net.cell_types[cell_type]["cell_object"].sections
    # Soma
    sections["soma"]._L = cell_params[0].value
    sections["soma"]._diam = cell_params[1].value
    sections["soma"]._cm = cell_params[2].value
    sections["soma"]._Ra = cell_params[3].value

    # Dendrite common parameters
    dendrite_cm = cell_params[4].value
    dendrite_Ra = cell_params[5].value

    dendrite_sections = [name for name in sections.keys() if name != "soma"]

    param_indices = [
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (14, 15),
        (16, 17),
        (18, 19),
        (20, 21),
    ]

    # Dentrite
    for section, indices in zip(dendrite_sections, param_indices):
        sections[section]._L = cell_params[indices[0]].value
        sections[section]._diam = cell_params[indices[1]].value
        sections[section]._cm = dendrite_cm
        sections[section]._Ra = dendrite_Ra

def _update_synapse_cell_params(net, cell_param_key, param_list):
    # ... unchanged body ...
    cell_params = param_list
    cell_type = f"{cell_param_key.split('_')[0]}_pyramidal"
    network_synapses = net.cell_types[cell_type]["cell_object"].synapses
    synapse_sections = ["ampa", "nmda", "gabaa", "gabab"]

    param_indices = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]

    # Update Dendrite
    for section, indices in zip(synapse_sections, param_indices):
        network_synapses[section]["e"] = cell_params[indices[0]].value
        network_synapses[section]["tau1"] = cell_params[indices[1]].value
        network_synapses[section]["tau2"] = cell_params[indices[2]].value


def _update_L2_biophysics_cell_params(net, cell_param_key, param_list):
    # ... unchanged body ...
    cell_type = f"{cell_param_key.split('_')[0]}_pyramidal"
    sections = net.cell_types[cell_type]["cell_object"].sections
    # Soma
    mechs_params = {
        "hh2": {
            "gkbar_hh2": param_list[0].value,
            "gnabar_hh2": param_list[1].value,
            "el_hh2": param_list[2].value,
            "gl_hh2": param_list[3].value,
        },
        "km": {"gbar_km": param_list[4].value},
    }

    sections["soma"].mechs.update(mechs_params)

    # dendrites
    mechs_params["hh2"] = {
        "gkbar_hh2": param_list[5].value,
        "gnabar_hh2": param_list[6].value,
        "el_hh2": param_list[7].value,
        "gl_hh2": param_list[8].value,
    }
    mechs_params["km"] = {"gbar_km": param_list[9].value}

    update_common_dendrite_sections(sections, mechs_params)


def _update_L5_biophysics_cell_params(net, cell_param_key, param_list):
    # ... unchanged body (uses partial(_exp_g_at_dist, ...)) ...
    cell_type = f"{cell_param_key.split('_')[0]}_pyramidal"
    sections = net.cell_types[cell_type]["cell_object"].sections
    # Soma
    mechs_params = {
        "hh2": {
            "gkbar_hh2": param_list[0].value,
            "gnabar_hh2": param_list[1].value,
            "el_hh2": param_list[2].value,
            "gl_hh2": param_list[3].value,
        },
        "ca": {"gbar_ca": param_list[4].value},
        "cad": {"taur_cad": param_list[5].value},
        "kca": {"gbar_kca": param_list[6].value},
        "km": {"gbar_km": param_list[7].value},
        "cat": {"gbar_cat": param_list[8].value},
        "ar": {"gbar_ar": param_list[9].value},
    }

    sections["soma"].mechs.update(mechs_params)

    # dendrites
    mechs_params["hh2"] = {
        "gkbar_hh2": param_list[10].value,
        "gnabar_hh2": param_list[11].value,
        "el_hh2": param_list[12].value,
        "gl_hh2": param_list[13].value,
    }

    mechs_params["ca"] = {"gbar_ca": param_list[14].value}
    mechs_params["cad"] = {"taur_cad": param_list[15].value}
    mechs_params["kca"] = {"gbar_kca": param_list[16].value}
    mechs_params["km"] = {"gbar_km": param_list[17].value}
    mechs_params["cat"] = {"gbar_cat": param_list[18].value}
    mechs_params["ar"] = {
        "gbar_ar": partial(
            _exp_g_at_dist, gbar_at_zero=param_list[19].value, exp_term=3e-3, offset=0.0
        )
    }

    update_common_dendrite_sections(sections, mechs_params)


def update_common_dendrite_sections(sections, mechs_params):
    dendrite_sections = [name for name in sections.keys() if name != "soma"]
    for section in dendrite_sections:
        sections[section].mechs.update(deepcopy(mechs_params))