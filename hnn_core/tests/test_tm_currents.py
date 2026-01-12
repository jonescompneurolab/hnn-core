# HEREEEEEEEEEEE

# %% [markdown] ###########################################################
## Disclaimer
# Note that this is **not** an actual test file and will not stay in the final
# version. I'm using this file to draft and test some features that will
# probably turn into a demo notebook for the new Textbook
# %% ######################################################################

# %% [markdown] ###########################################################
## Setup
# %% ######################################################################

import matplotlib.pyplot as plt
import numpy as np

from hnn_core import (
    JoblibBackend,
    jones_2009_model,
    simulate_dipole,
)
from hnn_core.cells_default import pyramidal
from hnn_core.network_builder import load_custom_mechanisms
from hnn_core.network_models import add_erp_drives_to_jones_model

# %% [markdown] ###########################################################
## Simulation
# %% ######################################################################

net = jones_2009_model()
add_erp_drives_to_jones_model(net)

n_trials = 1

if "dpls" not in locals():
    with JoblibBackend(8):
        dpls = simulate_dipole(
            net,
            tstop=170.0,
            n_trials=n_trials,
            record_agg_i_mem="all",
            # record_agg_ina="all",
            # record_agg_ik="all",
            record_agg_i_cap="all",
            record_ina_hh2="all",
            record_ik_hh2="all",
            record_ik_kca="all",
            record_ik_km="all",
            record_ica_ca="all",
            record_ica_cat="all",
            record_il_hh2="all",
            record_i_ar="all",
            record_isec="all",
        )

scaling_factor = 3000
for dpl in dpls:
    dpl.scale(scaling_factor)

dpl = dpls[0]
dpl_plot = dpl.plot(
    layer=["L5"],
)

# %% [markdown] ###########################################################
## Function to recreate dipole from transmembrane currents
# %% ######################################################################


def postproc_tm_currents(
    net,
    trial=0,
    cell_type="L5_pyramidal",
    scaling_factor=3000,
    from_components=False,
):
    """
    Function for processing transmembrane currents to recreate the dipole moment
    calculated from the axial currents in hnn_core. This can be done from either the
    total recorded transmembrane current, or from the constituent components.

    Note: isec (the transmembrane synaptic current) is part of the total transmembrane
    current, but is *not* aggregated with the other component channel currents
    in this function. This is due to the fact that isec contains *section-specific*
    currents (since synapses are placed at the section midpoint), as opposed to
    *segment-specfic* currents, which are required for this method of recreating
    the dipole.

    Parameters
    ----------
    net : Network object
        The network object containing the simulation data.
    trial : int
        The index of the trial to use
    cell_type : str
        The cell type to process
    scaling_factor : float
        The scaling factor to apply to the dipole
    from_components : bool
        if True, use agg_i_mem to reproduce the dipole. if False, use the component
        currents for either L5_pyramidal or L2_pyramidal

    Returns
    -------
    dipole : np.ndarray
        The reconstructed dipole moment from transmembrane currents.
    """

    load_custom_mechanisms()

    # initialize variable to hold dipole data
    dipole = None

    # build a template cell to get "metadata" for sections
    template_cell = pyramidal(cell_name=cell_type)
    template_cell.build(sec_name_apical="apical_trunk")

    # get the relative endpoints for each section from the template cell
    rel_endpoints = {}
    for sec_name, sec in template_cell._nrn_sections.items():
        start = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
        # sec.n3d() returns the number of 3D points along a section; essentially len()
        # so "sec.n3d() - 1" is the index of the last 3D point
        end = np.array(
            [sec.x3d(sec.n3d() - 1), sec.y3d(sec.n3d() - 1), sec.z3d(sec.n3d() - 1)]
        )
        rel_endpoints[sec_name] = (start, end)

    if not from_components:
        all_tm_channels = ["agg_i_mem"]
    else:
        if cell_type == "L5_pyramidal":
            all_tm_channels = [
                "agg_i_cap",
                "ina_hh2",
                "ik_hh2",
                "ik_kca",
                "ik_km",
                "ica_ca",
                "ica_cat",
                "il_hh2",
                "i_ar",
            ]
        elif cell_type == "L2_pyramidal":
            all_tm_channels = [
                "agg_i_cap",
                "ina_hh2",
                "ik_hh2",
                "ik_km",
                "il_hh2",
            ]
        else:
            raise ValueError(
                f"Valid channels types for {cell_type} are not known.\n"
                "Please pass the channels types as a list of str to tm_channels"
            )

    # loop through GIDs for the cell_type of interest
    for gid in net.gid_ranges[cell_type]:
        # get the updated soma position for this instantiation of the cell
        # index of the first cell: e.g., 170 for the first L5Pyr cell
        start_index = net.gid_ranges[cell_type][0]
        # get soma position from position dictionary, which uses its own indexing
        # that does not match the GID, hence the "- start_index"
        soma_pos = np.array(net.pos_dict[cell_type][gid - start_index])

        # create a dictionary of all channel data for the cell
        cell_channels = {
            ch: net.cell_response.transmembrane_currents[ch][trial][gid]
            for ch in all_tm_channels
        }

        # get the cell sections to loop over
        # the key used shouldn't matter, but we don't want to hard code it since
        # we can pass different channels to this function, so we get it dynamically
        first_key = list(cell_channels.keys())[0]
        cell_sections = list(cell_channels[first_key].keys())

        for sec_name in cell_sections:
            # offset the start/end positions by the realized soma position for this
            # cell instantiation
            start_rel, end_rel = rel_endpoints[sec_name]
            start = start_rel + soma_pos
            end = end_rel + soma_pos

            # get the normalized segment positions along the cell section
            nseg = len(cell_channels[first_key][sec_name])
            seg_positions = [(i - 0.5) / nseg for i in range(1, nseg + 1)]

            for pos, seg_key in zip(
                seg_positions,
                cell_channels[first_key][sec_name].keys(),
            ):
                # convert the normalized position to the absolute position
                # via linear interpolation
                abs_pos = start + pos * (end - start)
                # simplification: we are using the z position only here we only
                # need the vertical component of the dipole momen
                # we do *not* need to do geometric projection (via cos_theta)
                # as we do for the dipole calculation from axial currents
                z_i = abs_pos[2]

                # sum all currents for this segment
                I_t = np.zeros_like(
                    np.array(cell_channels[first_key][sec_name][seg_key])
                )
                for ch in all_tm_channels:
                    # get channel data
                    vec = np.array(cell_channels[ch][sec_name][seg_key])

                    # get segment area and convert from µm^2 to cm^2
                    seg = template_cell._nrn_sections[sec_name](pos)
                    area_um2 = seg.area()  # µm^2
                    area_cm2 = area_um2 * 1e-8  # cm^2

                    if ch == "agg_i_mem":
                        # agg_i_mem is not recorded continuously as a density; it is
                        # recorded after each timestep. Ergo, the units conversion
                        # here is not necessary as the units are already in nA
                        #
                        # multiplying the contribution by zi in um will give us fAm,
                        # so we will later need to divide by 1e6 to convert to nAm
                        I_abs = vec
                    # convert densities (mA/cm^2) to absolute currents (mA)]
                    else:
                        I_abs = vec * area_cm2  # keep as mA

                        # [WIP]
                        # Should I flip sign for the capacitive currents? I *think*
                        # so, but I haven't found direct confirmation of this ...
                        #
                        # I ideally would want to test the sign flip empirically,
                        # and confirm that we can reproduce i_mem by summing up all
                        # of its constituent components. However, we can't get the
                        # per-segment synaptic currents since we model them as point
                        # processes at the midpoint of the section (not the segment);
                        #
                        # Ergo, we are missing the synaptic piece of the total
                        # transmembrane current needed to reproduce i_mem exactly
                        #
                        # Note: isec is the *per-synapse* current, and not the
                        # *per-segment* current that we need
                        if ch == "agg_i_cap":
                            I_abs = I_abs * -1  # flip sign (?)
                        # [end WIP]

                    I_t += I_abs

                # multiple by r_i per Naess 2015 Ch 2 (simplified to zi in this case)
                # for ionic currents, we have 1 mA*um = 1 nAm (correct units)
                # for i_mem, we have nA rather than mA. and 1 nA*um = 1 fAm
                contrib = I_t * z_i

                # for agg_i_mem, divide by 1e6 to convert fAm to nAm
                if not from_components:
                    contrib = contrib / 1e6 * scaling_factor
                else:
                    contrib = contrib * scaling_factor

                if dipole is None:
                    dipole = contrib.copy()
                else:
                    dipole += contrib

    return dipole


# %% [markdown] ###########################################################
## Compare dipoles calculated from axial vs transmembrane currents
# %% ######################################################################


# %% [markdown] ----------------------------------------
### Layer 5
# %% ---------------------------------------------------

fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=(8, 15),
)

test_imem_L5 = postproc_tm_currents(
    net=net,
    from_components=False,
)

ax[1].plot(
    dpl.times,
    test_imem_L5,
)

ax[1].set_ylim(-200, 100)

_ = dpl.plot(
    layer=["L5"],
    ax=ax[0],
)


# %% [markdown] ----------------------------------------
### Layers 2/3
# %% ---------------------------------------------------
test_imem_L2 = postproc_tm_currents(
    net=net,
    cell_type="L2_pyramidal",
    from_components=False,
)

fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=(8, 10),
)

ax[0].plot(
    dpl.times,
    test_imem_L2,
)

ax[0].set_ylim(-30, 50)

_ = dpl.plot(
    layer=["L2"],
    ax=ax[1],
)


# %% [markdown] ###########################################################
## [DEV] Visualize transmembrane currents
# %% ######################################################################

# %% ####################
# for a single trial and current, get recordings by section, segment for each cell


def agg_transmembrane_current_across_celltype(
    trial_number=0,
    cell_type="L5_pyramidal",
    target_channel="i_mem",
):
    # get recordings for the target channel and trial number
    channel_cell_recordings = (
        net.cell_response.transmembrane_currents[target_channel][trial_number]  # noqa: E203,W503
    )

    gids = list(net.gid_ranges[cell_type])

    # limit scope to gids for the specified cell_type
    channel_cell_recordings = {gid: channel_cell_recordings[gid] for gid in gids}

    aggregate = {}

    for gid_data_dict in channel_cell_recordings.values():
        for section_key, section_data_dict in gid_data_dict.items():
            for segment_key, segment_data in section_data_dict.items():
                values = np.array(segment_data)
                if section_key not in aggregate:
                    aggregate[section_key] = {}
                if segment_key not in aggregate[section_key]:
                    aggregate[section_key][segment_key] = np.zeros_like(values)
                aggregate[section_key][segment_key] += values

    return aggregate


ina_hh2 = agg_transmembrane_current_across_celltype(
    trial_number=0,
    target_channel="ina_hh2",
)


def plot_segment_recordings_by_section(
    section_name,
    channel_data_dict,
    channel_name="[Channel name not specified]",
):
    segment_data_dict = channel_data_dict[section_name]
    fig, ax = plt.subplots(
        nrows=len(segment_data_dict),
        ncols=1,
        sharex=True,
        figsize=(8, 3 * len(segment_data_dict)),
    )

    # np.inf guarantees the values will be replaced on the first iteration
    y_min = np.inf
    y_max = -np.inf
    for i, (segment_key, segment_data) in enumerate(segment_data_dict.items()):
        ax[i].plot(segment_data)
        ax[i].set_title(f"{segment_key.replace('seg_', 'Segment ')}")
        y_min = min(y_min, segment_data.min())
        y_max = max(y_max, segment_data.max())

    section_title = section_name.replace("_", " ")
    section_title = section_title.title()

    # set consistent y axes, with extra padding
    padding = 0.05 * (y_max - y_min)
    y_limits = (y_min - padding, y_max + padding)

    for axis in ax:
        axis.set_ylim(y_limits)

    fig.suptitle(
        f"Transmembrane Recordings for {section_title} of {channel_name}",
        fontsize=16,
    )
    plt.xlabel("Time (ms)")
    plt.tight_layout()
    plt.show()

    return


plot_segment_recordings_by_section(
    section_name="apical_trunk",
    channel_data_dict=ina_hh2,
    channel_name="Na+ HH2",
)


# %% [markdown] ###########################################################
## [WIP] Testing and Feature Development
# %% ######################################################################

# %% [markdown] ----------------------------------------
### Recreating dipole for the soma only
# %% ---------------------------------------------------


def postproc_soma_dipole(
    net,
    trial=0,
    cell_type="L5_pyramidal",
    scaling_factor=3000,
    from_components=False,
):
    """ """

    # this function will only handle the "soma", as it's composed of exactly one
    # segment where pos = 0.5
    sec_name = "soma"
    seg_key = "seg_1"
    pos = 0.5

    # load custom mechanisms
    load_custom_mechanisms()

    # initialize variable to hold dipole data
    dipole = None

    # build a template cell to get "metadata" for sections
    template_cell = pyramidal(cell_name=cell_type)
    template_cell.build(sec_name_apical="apical_trunk")

    # get the relative endpoints for the soma
    rel_endpoints = {}
    sec = template_cell._nrn_sections["soma"]
    start = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
    end = np.array(
        [sec.x3d(sec.n3d() - 1), sec.y3d(sec.n3d() - 1), sec.z3d(sec.n3d() - 1)]
    )
    rel_endpoints[sec_name] = (start, end)

    if not from_components:
        all_tm_channels = ["agg_i_mem"]
    else:
        if cell_type == "L5_pyramidal":
            all_tm_channels = [
                "agg_i_cap",
                "ina_hh2",
                "ik_hh2",
                "ik_kca",
                "ik_km",
                "ica_ca",
                "ica_cat",
                "il_hh2",
                "i_ar",
            ]
        elif cell_type == "L2_pyramidal":
            all_tm_channels = [
                "agg_i_cap",
                "ina_hh2",
                "ik_hh2",
                "ik_km",
                "il_hh2",
            ]
        else:
            raise ValueError(
                f"Valid channels types for {cell_type} are not known.\n"
                "Please pass the channels types as a list of str to tm_channels"
            )

    # loop through GIDs for the cell_type of interest
    for gid in net.gid_ranges[cell_type]:
        # get the updated soma position for this instantiation of the cell
        # index of the first cell: e.g., 170 for the first L5Pyr cell
        start_index = net.gid_ranges[cell_type][0]
        # get soma position from position dictionary, which uses its own indexing
        # that does not match the GID, hence the "- start_index"
        soma_pos = np.array(net.pos_dict[cell_type][gid - start_index])

        # create a dictionary of all channel data for the cell
        cell_channels = {
            ch: net.cell_response.transmembrane_currents[ch][trial][gid]
            for ch in all_tm_channels
        }

        # get the cell sections to loop over
        # the key used shouldn't matter, but we don't want to hard code it since
        # we can pass different channels to this function, so we get it dynamically
        first_key = list(cell_channels.keys())[0]

        print(first_key)

        start_rel, end_rel = rel_endpoints[sec_name]
        start = start_rel + soma_pos
        end = end_rel + soma_pos

        abs_pos = start + pos * (end - start)
        z_i = abs_pos[2]

        # sum all currents for this segment
        I_t = np.zeros_like(
            np.array(cell_channels[first_key][sec_name][seg_key]),
        )

        for ch in all_tm_channels:
            # get channel data
            vec = np.array(cell_channels[ch][sec_name][seg_key])

            # get segment area and convert from µm^2 to cm^2
            seg = template_cell._nrn_sections[sec_name](pos)
            area_um2 = seg.area()  # µm^2
            area_cm2 = area_um2 * 1e-8  # cm^2

            if ch == "agg_i_mem":
                # agg_i_mem is not recorded continuously as a density; it is
                # recorded after each timestep. Ergo, the units conversion
                # here is not necessary as the units are already in nA
                #
                # multiplying the contribution by zi in um will give us fAm,
                # so we will later need to divide by 1e6 to convert to nAm
                I_abs = vec
            # convert densities (mA/cm^2) to absolute currents (mA)]
            else:
                I_abs = vec * area_cm2  # keep as mA
                I_abs = I_abs * -1

                if ch == "agg_i_cap":
                    I_abs = I_abs * -1  # flip sign (?)

            I_t += I_abs

        # multiple by r_i per Naess 2015 Ch 2 (simplified to zi in this case)
        # for ionic currents, we have 1 mA*um = 1 nAm (correct units)
        # for i_mem, we have nA rather than mA. and 1 nA*um = 1 fAm
        contrib = I_t * z_i

        # for agg_i_mem, divide by 1e6 to convert fAm to nAm
        if not from_components:
            contrib = contrib / 1e6 * scaling_factor
        else:
            contrib = contrib * scaling_factor

        if dipole is None:
            dipole = contrib.copy()
        else:
            dipole += contrib

    return dipole


# %% ---------------------------------------------------

fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=(8, 15),
)


test_imem_L5 = postproc_soma_dipole(
    net=net,
    from_components=False,
)

ax[0].plot(
    dpl.times[1:],
    test_imem_L5[1:],
)

test_imem_L5 = postproc_soma_dipole(
    net=net,
    from_components=True,
)

ax[1].plot(
    dpl.times[1:],
    test_imem_L5[1:],
)

ax[0].set_ylim(-200, 100)
ax[1].set_ylim(-200, 100)

# %% [markdown] ----------------------------------------
### Next steps
# %% ---------------------------------------------------

"""
For feature example (code contribution):
- Show recording of transmembrane currents individually; there is some baseline code
  for this that I need to migrate to this repository
- Likely remove the dipole reconstruction from_components for this PR; it's not
  strictly necessary and not complete
- Dipole reconstruction from i_mem (total transmembrane current) can be kept in the
  example notebook for this PR, as it is complete and shows off new use cases. The
  function herein should be adapted to remove the from_components section, which is
  still a work in progress

For dipole reconstruction (science contribution):
- add isec to soma calculation function, which should be allowed since soma has only
  one segment, and isec should be the transmembrane synaptic current component that
  is missing
- run function with isec, see if dipole reproduction matches
- potentially try reconstruction for one cell only to interrogate discrepancies
-
"""

# %%
