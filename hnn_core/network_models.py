"""Network model functions."""

# Authors: Nick Tolley <nicholas_tolley@brown.edu>

from pathlib import Path
from copy import deepcopy
import warnings

import hnn_core
from hnn_core import read_params
from .network import Network, _create_cell_coords
from .params import _short_name
from .cells_default import (
    basket,
    pyramidal,
    pyramidal_ca,
    pyramidal_humanL5ET,
    pyramidal_humanL23,
    human_gen_interneuron,
)
from .externals.mne import _validate_type

# Default cell metadata for the standard Jones 2009 network cell types.
# Defined here at module level so that other code (e.g. JSON
# serialisation / deserialisation) can import it without instantiating
# a full Network object.

default_cell_metadata = {
    "L2_basket": {
        "morpho_type": "basket",
        "electro_type": "inhibitory",
        "layer": "2",
        "zdist_origin": 0.8,  # distance to origin in percent of layer_separation
        "measure_dipole": False,
        "reference": "https://doi.org/10.7554/eLife.51214",
        "color": "m",
        "marker": "x",  # shape from prev viz.py line:926
    },
    "L2_pyramidal": {
        "morpho_type": "pyramidal",
        "electro_type": "excitatory",
        "layer": "2",
        "zdist_origin": 1,
        "measure_dipole": True,
        "reference": "https://doi.org/10.7554/eLife.51214",
        "color": "c",
        "marker": "^",
    },
    "L5_basket": {
        "morpho_type": "basket",
        "electro_type": "inhibitory",
        "layer": "5",
        "zdist_origin": 0.2,
        "measure_dipole": False,
        "reference": "https://doi.org/10.7554/eLife.51214",
        "color": "r",
        "marker": "x",
    },
    "L5_pyramidal": {
        "morpho_type": "pyramidal",
        "electro_type": "excitatory",
        "layer": "5",
        "zdist_origin": 0,
        "measure_dipole": True,
        "reference": "https://doi.org/10.7554/eLife.51214",
        "color": "b",
        "marker": "^",
    },
}

default_drive_colors = {
    "proximal": "r",
    "distal": "g",
    "default": "#8B4513",
}


def _validate_params_for_model(
    net,
    params,
    model_variant,
    alt_variants=[],
    require_variant=False,
    excluded_cells=[],
):
    """Check that a param file matches the network model it is used for.

    Parameters
    ----------
    net : Instance of Network object
        The network the parameters are used for.
    params : dict
        The parameters the network was built from.
    model_variant : str
        Name of the network model, e.g. 'duecker_ET_model'. The
        'model_variant' entry of `params` must be this name (or of one of
        `alt_variants`).
    alt_variants : list of str, default=[]
        Further model names that are accepted in the 'model_variant' entry of `params`,
        e.g. the deprecated name of a model. If `params` defines a local 'model_variant'
        that is in 'alt_variants', the returned value will be the value of
        'model_variant' that is passed to this function instead.
    require_variant : bool, default=False
        If True, raise if `params` does not define 'model_variant'. Used for
        models that share no parameters with the default model, and would
        otherwise silently fall back to default values.
    excluded_cells : list of str, default=[]
        Short names of cells that are *not* part of this network, e.g.
        ('L2Basket', 'L5Basket') for a model in which basket cells are
        replaced. Parameters for these cells are rejected.

    Returns
    -------
    model_variant : str
        The official model variant name.
    """
    check_var = params.get("model_variant", None)
    if check_var is None:
        if require_variant:
            raise ValueError(
                f"'model_variant' is required for simulations with "
                f"{model_variant}. If you are sure that you are using the "
                f"correct parameters, add 'model_variant': '{model_variant}', "
                "to the first line of the param .json file."
            )
    elif check_var not in [model_variant] + alt_variants:
        raise ValueError(
            f"Parameters for {check_var} used for {model_variant}."
            " Ensure that your param .json file matches the network "
            f"and that your model variant is one of {[model_variant] + alt_variants}. "
        )

    # check that the params define the cell types of this network
    missing_cells = [
        cell_name
        for cell_name in net.cell_types
        if not any(_short_name(cell_name) in key for key in params)
    ]
    if missing_cells:
        raise ValueError(
            f"No parameters found for {', '.join(missing_cells)}."
            " Ensure that your param .json file matches the network. "
            " Reach out to us if this doesn't solve your problem.  "
            " https://github.com/jonescompneurolab/hnn-core/discussions"
        )

    # check that the params don't define cell types this network replaced
    unexpected_cells = [
        cell_name
        for cell_name in excluded_cells
        if any(cell_name in key for key in params)
    ]
    if unexpected_cells:
        raise ValueError(
            f"Parameters found for {', '.join(unexpected_cells)}, which"
            f" are not part of {model_variant}. Ensure that your"
            " param .json file matches the network."
            " Reach out to us if this doesn't solve your problem.  "
            " https://github.com/jonescompneurolab/hnn-core/discussions"
        )
    return model_variant


def neymotin_2020_model(
    params=None,
    add_drives_from_params=False,
    legacy_mode=False,
    mesh_shape=(10, 10),
):
    """Instantiate the network model described in Neymotin et al. 2020

    Parameters
    ----------
    params : str | Path | dict | None, default=None
        Custom Network parameters to use, if any. If string or Path, it is assumed to be
        a path to a legacy "flat" JSON file containing the parameters in the style of
        `hnn_core/param/default.json` (NOT a modern "hierarchical" JSON file like
        `hnn_core/param/neymotin2020_base.json`). If dict, it is assumed to be a
        dictionary of parameters in the "flat" JSON style. If None (the default), the
        default parameters are used from `hnn_core/param/default.json`. Note that if you
        have drive parameters included, but you ALSO set the deprecated
        `add_drives_from_params` to True, the drives will be added twice if the drive
        names are the same.
    add_drives_from_params : bool, default=False
        If True, add drives as defined in the params-dict. NB this is mainly
        for backward-compatibility with HNN GUI, and will be deprecated in a
        future release. Default: False
    legacy_mode : bool, default=False
        Set to False by default. Enables matching HNN GUI output when drives
        are added suitably. Will be deprecated in a future release.
    mesh_shape : tuple of int (default: (10, 10))
        Defines the (n_x, n_y) shape of the grid of pyramidal cells.

    Returns
    -------
    net : Instance of Network object
        Network object used to store

    Notes
    -----
    The network is composed of a square grid of pyramidal cells, arranged in
    two layers (L5 and L2). The default in-plane separation of the grid points
    is 1.0 um, and the layer separation 1307.4 um. These can be adjusted after
    the net is created using the set_cell_positions-method. An all-to-all
    connectivity pattern is applied between cells. Inhibitory basket cells are
    present at a 1:3-ratio.

    This network was first described in Jones et al. 2009 [1]_ , and this code provides
    the implementation used in Neymotin et al. 2020 [2]_ .

    References
    ----------
    .. [1] Jones, Stephanie R., et al. "Quantitative Analysis and
           Biophysically Realistic Neural Modeling of the MEG Mu Rhythm:
           Rhythmogenesis and Modulation of Sensory-Evoked Responses."
           Journal of Neurophysiology 102, 3554–3572 (2009).
           https://doi.org/10.1152/jn.00535.2009

    .. [2] Neymotin, Samuel A, et al. 2020. "Human Neocortical Neurosolver (HNN), a New
           Software Tool for Interpreting the Cellular and Network Origin of Human
           MEG/EEG Data." eLife 9 (January):e51214. https://doi.org/10.7554/eLife.51214

    """
    hnn_core_root = Path(hnn_core.__file__).parent
    if params is None:
        params_fname = hnn_core_root / "param" / "default.json"
        params = read_params(params_fname)
    elif isinstance(params, str) or isinstance(params, Path):
        params = read_params(params)

    # Define cell types for Jones 2009 model
    # data is here in metaData format
    cell_types = {
        "L2_basket": {
            "cell_object": basket(cell_name="L2_basket"),
            "cell_metadata": deepcopy(default_cell_metadata["L2_basket"]),
        },
        "L2_pyramidal": {
            "cell_object": pyramidal(cell_name="L2_pyramidal"),
            "cell_metadata": deepcopy(default_cell_metadata["L2_pyramidal"]),
        },
        "L5_basket": {
            "cell_object": basket(cell_name="L5_basket"),
            "cell_metadata": deepcopy(default_cell_metadata["L5_basket"]),
        },
        "L5_pyramidal": {
            "cell_object": pyramidal(cell_name="L5_pyramidal"),
            "cell_metadata": deepcopy(default_cell_metadata["L5_pyramidal"]),
        },
    }

    # Create layer positions
    layer_dict = _create_cell_coords(
        n_pyr_x=mesh_shape[0],
        n_pyr_y=mesh_shape[1],
        z_coord=1307.4,  # Default layer separation
        inplane_distance=1.0,  # Default in-plane distance
    )

    # Map cell types to layer positions
    pos_dict = {
        "L5_pyramidal": layer_dict["L5_bottom"],
        "L2_pyramidal": layer_dict["L2_bottom"],
        "L5_basket": layer_dict["L5_mid"],
        "L2_basket": layer_dict["L2_mid"],
        "origin": layer_dict["origin"],
    }

    # Create network with cell types and positions
    net = Network(
        params,
        add_drives_from_params=add_drives_from_params,
        legacy_mode=legacy_mode,
        mesh_shape=mesh_shape,
        pos_dict=pos_dict,
        cell_types=cell_types,
    )

    delay = net.delay

    # Ensure model_variant and params' cell types match current model
    net._model_variant = _validate_params_for_model(net, params, "neymotin_2020_model")

    # source of synapse is always at soma

    # layer2 Pyr -> layer2 Pyr
    # layer5 Pyr -> layer5 Pyr
    lamtha = 3.0
    loc = "proximal"
    for target_cell in ["L2_pyramidal", "L5_pyramidal"]:
        for receptor in ["nmda", "ampa"]:
            key = (
                f"gbar_{_short_name(target_cell)}_{_short_name(target_cell)}_{receptor}"
            )
            weight = net._params[key]
            net.add_connection(
                target_cell,
                target_cell,
                loc,
                receptor,
                weight,
                delay,
                lamtha,
                allow_autapses=False,
            )

    # layer2 Basket -> layer2 Pyr
    src_cell = "L2_basket"
    target_cell = "L2_pyramidal"
    lamtha = 50.0
    loc = "soma"
    for receptor in ["gabaa", "gabab"]:
        key = f"gbar_L2Basket_L2Pyr_{receptor}"
        weight = net._params[key]
        net.add_connection(src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # layer5 Basket -> layer5 Pyr
    src_cell = "L5_basket"
    target_cell = "L5_pyramidal"
    lamtha = 70.0
    loc = "soma"
    for receptor in ["gabaa", "gabab"]:
        key = f"gbar_L5Basket_{_short_name(target_cell)}_{receptor}"
        weight = net._params[key]
        net.add_connection(src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # layer2 Pyr -> layer5 Pyr
    src_cell = "L2_pyramidal"
    lamtha = 3.0
    receptor = "ampa"
    for loc in ["proximal", "distal"]:
        key = f"gbar_L2Pyr_{_short_name(target_cell)}"
        weight = net._params[key]
        net.add_connection(src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # layer2 Basket -> layer5 Pyr
    src_cell = "L2_basket"
    lamtha = 50.0
    key = f"gbar_L2Basket_{_short_name(target_cell)}"
    weight = net._params[key]
    loc = "distal"
    receptor = "gabaa"
    net.add_connection(src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # xx -> layer2 Basket
    src_cell = "L2_pyramidal"
    target_cell = "L2_basket"
    lamtha = 3.0
    key = f"gbar_L2Pyr_{_short_name(target_cell)}"
    weight = net._params[key]
    loc = "soma"
    receptor = "ampa"
    net.add_connection(src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    src_cell = "L2_basket"
    lamtha = 20.0
    key = f"gbar_L2Basket_{_short_name(target_cell)}"
    weight = net._params[key]
    loc = "soma"
    receptor = "gabaa"
    net.add_connection(src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # xx -> layer5 Basket
    src_cell = "L5_basket"
    target_cell = "L5_basket"
    lamtha = 20.0
    loc = "soma"
    receptor = "gabaa"
    key = f"gbar_L5Basket_{_short_name(target_cell)}"
    weight = net._params[key]
    net.add_connection(
        src_cell,
        target_cell,
        loc,
        receptor,
        weight,
        delay,
        lamtha,
        allow_autapses=False,
    )

    src_cell = "L5_pyramidal"
    lamtha = 3.0
    key = f"gbar_L5Pyr_{_short_name(target_cell)}"
    weight = net._params[key]
    loc = "soma"
    receptor = "ampa"
    net.add_connection(src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    src_cell = "L2_pyramidal"
    lamtha = 3.0
    key = f"gbar_L2Pyr_{_short_name(target_cell)}"
    weight = net._params[key]
    loc = "soma"
    receptor = "ampa"
    net.add_connection(src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    return net


def jones_2009_model(
    params=None,
    add_drives_from_params=False,
    legacy_mode=False,
    mesh_shape=(10, 10),
):
    """Instantiate the network model described in Jones et al., 2009.

    DEPRECATED: This function is now deprecated in favor of using `neymotin_2020_model`
    which is a more accurate name for the model being used. This function is still kept
    in HNN-Core for backwards-compatibility reasons, and is not currently planned to be
    removed.

    Parameters
    ----------
    params : str | Path | dict | None, default=None
        Custom Network parameters to use, if any. If string or Path, it is assumed to be
        a path to a legacy "flat" JSON file containing the parameters in the style of
        `hnn_core/param/default.json` (NOT a modern "hierarchical" JSON file like
        `hnn_core/param/neymotin2020_base.json`). If dict, it is assumed to be a
        dictionary of parameters in the "flat" JSON style. If None (the default), the
        default parameters are used from `hnn_core/param/default.json`. Note that if you
        have drive parameters included, but you ALSO set the deprecated
        `add_drives_from_params` to True, the drives will be added twice if the drive
        names are the same.
    add_drives_from_params : bool, default=False
        If True, add drives as defined in the params-dict. NB this is mainly
        for backward-compatibility with HNN GUI, and will be deprecated in a
        future release. Default: False
    legacy_mode : bool, default=False
        Set to False by default. Enables matching HNN GUI output when drives
        are added suitably. Will be deprecated in a future release.
    mesh_shape : tuple of int (default: (10, 10))
        Defines the (n_x, n_y) shape of the grid of pyramidal cells.

    Returns
    -------
    net : Instance of Network object
        Network object used to store

    Notes
    -----
    The network is composed of a square grid of pyramidal cells, arranged in two layers
    (L5 and L2). The default in-plane separation of the grid points is 1.0 um, and the
    layer separation 1307.4 um. These can be adjusted after the net is created using the
    set_cell_positions-method. An all-to-all connectivity pattern is applied between
    cells. Inhibitory basket cells are present at a 1:3-ratio.

    This network was first described in Jones et al. 2009 [1]_ , and this code provides
    the implementation used in Neymotin et al. 2020 [2]_ .

    References
    ----------
    .. [1] Jones, Stephanie R., et al. "Quantitative Analysis and Biophysically
           Realistic Neural Modeling of the MEG Mu Rhythm: Rhythmogenesis and Modulation
           of Sensory-Evoked Responses." Journal of Neurophysiology 102, 3554–3572
           (2009). https://doi.org/10.1152/jn.00535.2009

    .. [2] Neymotin, Samuel A, et al. 2020. "Human Neocortical Neurosolver (HNN), a New
           Software Tool for Interpreting the Cellular and Network Origin of Human
           MEG/EEG Data." eLife 9 (January):e51214. https://doi.org/10.7554/eLife.51214

    """

    warnings.warn(
        """
        Calling the default model with `jones_2009_model` is now deprecated. Please
        update your scripts to use `neymotin_2020_model`, which is a more accurate name
        for the model. `jones_2009_model` will still be made available for
        backwards-compatilibity purposes.
        """,
        FutureWarning,
    )

    net = neymotin_2020_model(params, add_drives_from_params, legacy_mode, mesh_shape)
    return net


def law_2021_model(
    params=None,
    add_drives_from_params=False,
    legacy_mode=False,
    mesh_shape=(10, 10),
):
    """Instantiate the network model described in Law et al., 2021.

    This creates a network that is the expansion of Jones 2009 model to study beta
    modulated ERPs as described in Law et al. Cereb. Cortex 2021 [1]_ .

    Parameters
    ----------
    params : str | Path | dict | None, default=None
        Custom Network parameters to use, if any. If string or Path, it is assumed to be
        a path to a legacy "flat" JSON file containing the parameters in the style of
        `hnn_core/param/default.json` (NOT a modern "hierarchical" JSON file like
        `hnn_core/param/neymotin2020_base.json`). If dict, it is assumed to be a
        dictionary of parameters in the "flat" JSON style. If None (the default), the
        default parameters are used from `hnn_core/param/default.json`. Note that if you
        have drive parameters included, but you ALSO set the deprecated
        `add_drives_from_params` to True, the drives will be added twice if the drive
        names are the same.
    add_drives_from_params : bool, default=False
        If True, add drives as defined in the params-dict. NB this is mainly
        for backward-compatibility with HNN GUI, and will be deprecated in a
        future release. Default: False
    legacy_mode : bool, default=False
        Set to False by default. Enables matching HNN GUI output when drives
        are added suitably. Will be deprecated in a future release.
    mesh_shape : tuple of int (default: (10, 10))
        Defines the (n_x, n_y) shape of the grid of pyramidal cells.

    Returns
    -------
    net : Instance of Network object
        Network object used to store the model used in
        Law et al. 2021.

    See Also
    --------
    jones_2009_model

    Notes
    -----
    Model reproduces results from Law et al. 2021
    This model differs from the default network model in several
    parameters including
    1) Increased GABAb time constants on L2/L5 pyramidal cells
    2) Decrease L5_pyramidal -> L5_pyramidal nmda weight
    3) Modified L5_basket -> L5_pyramidal inhibition weights
    4) Removal of L5 pyramidal somatic and basal dendrite calcium channels
    5) Replace L2_basket -> L5_pyramidal GABAa connection with GABAb
    6) Addition of L5_basket -> L5_pyramidal distal connection

    References
    ----------
    .. [1] Law, Robert G., et al. "Thalamocortical Mechanisms Regulating the
           Relationship between Transient Beta Events and Human Tactile
           Perception." Cerebral Cortex, 32, 668–688 (2022).
    """

    hnn_core_root = Path(hnn_core.__file__).parent
    if params is None:
        params_fname = hnn_core_root / "param" / "default.json"
        params = read_params(params_fname)
    elif isinstance(params, str) or isinstance(params, Path):
        params = read_params(params)

    net = neymotin_2020_model(
        params,
        add_drives_from_params,
        legacy_mode,
        mesh_shape=mesh_shape,
    )
    # Ensure model_variant and params' cell types match current model (same cell types
    # as 'neymotin_2020_model')
    net._model_variant = _validate_params_for_model(net, params, "law_2021_model")

    # Update biophysics (increase gabab duration of inhibition)
    net.cell_types["L2_pyramidal"]["cell_object"].synapses["gabab"]["tau1"] = 45.0
    net.cell_types["L2_pyramidal"]["cell_object"].synapses["gabab"]["tau2"] = 200.0
    net.cell_types["L5_pyramidal"]["cell_object"].synapses["gabab"]["tau1"] = 45.0
    net.cell_types["L5_pyramidal"]["cell_object"].synapses["gabab"]["tau2"] = 200.0

    # Decrease L5_pyramidal -> L5_pyramidal nmda weight
    net.connectivity[2]["nc_dict"]["A_weight"] = 0.0004

    # Modify L5_basket -> L5_pyramidal inhibition
    net.connectivity[6]["nc_dict"]["A_weight"] = 0.02  # gabaa
    net.connectivity[7]["nc_dict"]["A_weight"] = 0.005  # gabab

    # Remove L5 pyramidal somatic and basal dendrite calcium channels
    for sec in ["soma", "basal_1", "basal_2", "basal_3"]:
        del net.cell_types["L5_pyramidal"]["cell_object"].sections[sec].mechs["ca"]

    # Remove L2_basket -> L5_pyramidal gabaa connection
    del net.connectivity[10]  # Original paper simply sets gbar to 0.0

    # Add L2_basket -> L5_pyramidal gabab connection
    delay = net.delay
    src_cell = "L2_basket"
    target_cell = "L5_pyramidal"
    lamtha = 50.0
    weight = 0.0002
    loc = "distal"
    receptor = "gabab"
    net.add_connection(src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # Add L5_basket -> L5_pyramidal distal connection
    # ("Martinotti-like recurrent tuft connection")
    src_cell = "L5_basket"
    target_cell = "L5_pyramidal"
    lamtha = 70.0
    loc = "distal"
    receptor = "gabaa"
    key = f"gbar_L5Basket_L5Pyr_{receptor}"
    weight = net._params[key]
    net.add_connection(src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    return net


# Remove params argument after updating examples
# (only relevant for Jones 2009 model)
def calcium_model(
    params=None,
    add_drives_from_params=False,
    legacy_mode=False,
    mesh_shape=(10, 10),
):
    """Instantiate the Jones 2009 model with improved calcium dynamics in
    L5 pyramidal neurons. For more details on changes to calcium dynamics
    see Kohl et al. Brain Topragr 2022 [1]_

    Parameters
    ----------
    params : str | Path | dict | None, default=None
        Custom Network parameters to use, if any. If string or Path, it is assumed to be
        a path to a legacy "flat" JSON file containing the parameters in the style of
        `hnn_core/param/default.json` (NOT a modern "hierarchical" JSON file like
        `hnn_core/param/neymotin2020_base.json`). If dict, it is assumed to be a
        dictionary of parameters in the "flat" JSON style. If None (the default), the
        default parameters are used from `hnn_core/param/default.json`. Note that if you
        have drive parameters included, but you ALSO set the deprecated
        `add_drives_from_params` to True, the drives will be added twice if the drive
        names are the same.
    add_drives_from_params : bool, default=False
        If True, add drives as defined in the params-dict. NB this is mainly
        for backward-compatibility with HNN GUI, and will be deprecated in a
        future release. Default: False
    legacy_mode : bool, default=False
        Set to False by default. Enables matching HNN GUI output when drives
        are added suitably. Will be deprecated in a future release.
    mesh_shape : tuple of int (default: (10, 10))
        Defines the (n_x, n_y) shape of the grid of pyramidal cells.

    Returns
    -------
    net : Instance of Network object
        Network object used to store the Jones 2009 model with an improved
        calcium channel distribution.

    See Also
    --------
    jones_2009_model

    Notes
    -----
    This model builds on the Jones 2009 model by using a more biologically
    accurate distribution of calcium channels on L5 pyramidal cells.
    Specifically, this model introduces a distance dependent maximum
    conductance (gbar) on calcium channels such that the gbar linearly
    decreases along the dendrites in the direction of the soma.

    References
    ----------
    .. [1] Kohl, Carmen, et al. "Neural Mechanisms Underlying Human Auditory
           Evoked Responses Revealed By Human Neocortical Neurosolver."
           Brain Topography, 35, 19–35 (2022).
    """
    hnn_core_root = Path(hnn_core.__file__).parent
    if params is None:
        params_fname = hnn_core_root / "param" / "default.json"
        params = read_params(params_fname)
    elif isinstance(params, str) or isinstance(params, Path):
        params = read_params(params)

    net = jones_2009_model(
        params,
        add_drives_from_params,
        legacy_mode,
        mesh_shape=mesh_shape,
    )

    # Ensure model_variant and params' cell types match current model (same cell types
    # as 'neymotin_2020_model')
    net._model_variant = _validate_params_for_model(net, params, "calcium_model")

    # Replace L5 pyramidal cell template with updated calcium
    cell_name = "L5_pyramidal"
    pos = net.cell_types[cell_name]["cell_object"].pos
    net.cell_types[cell_name]["cell_object"] = pyramidal_ca(
        cell_name=cell_name, pos=pos
    )

    return net


def duecker_ET_model(
    params=None,
    add_drives_from_params=False,
    legacy_mode=False,
    mesh_shape=(10, 10),
):
    """Initiate the Duecker model (like old calcium model and then replace with new cells)

    Parameters
    ----------
    params : str | Path | dict | None, default=None
        Custom Network parameters to use, if any. If string or Path, it is assumed to be
        a path to a legacy "flat" JSON file containing the parameters in the style of
        `hnn_core/param/default.json` (NOT a modern "hierarchical" JSON file like
        `hnn_core/param/neymotin2020_base.json`). If dict, it is assumed to be a
        dictionary of parameters in the "flat" JSON style. If None (the default), the
        default parameters are used from `hnn_core/param/default.json`. Note that if you
        have drive parameters included, but you ALSO set the deprecated
        `add_drives_from_params` to True, the drives will be added twice if the drive
        names are the same.
    add_drives_from_params : bool, default=False
        If True, add drives as defined in the params-dict. NB this is mainly
        for backward-compatibility with HNN GUI, and will be deprecated in a
        future release. Default: False
    legacy_mode : bool, default=False
        Set to False by default. Enables matching HNN GUI output when drives
        are added suitably. Will be deprecated in a future release.
    mesh_shape : tuple of int (default: (10, 10))
        Defines the (n_x, n_y) shape of the grid of pyramidal cells.

    Returns
    -------
    net : Instance of Network object
        Network object used to store
    """

    hnn_core_root = Path(hnn_core.__file__).parent
    if params is None:
        params_fname = hnn_core_root / "param" / "default_duecker_ET.json"
        params = read_params(params_fname)
    elif isinstance(params, str) or isinstance(params, Path):
        params = read_params(params)

    cell_types = {
        "L2_inhibitory": {
            "cell_object": human_gen_interneuron(
                cell_name=_short_name("L2_inhibitory"), layer=2
            ),
            "cell_metadata": {
                "morpho_type": "interneuron",
                "electro_type": "inhibitory",
                "layer": "2",
                "zdist_origin": 0.8,
                "measure_dipole": False,
                "reference": "",
                "color": "#daa69c",
                "marker": "o",
            },
        },
        "L2_pyramidal": {
            "cell_object": pyramidal_humanL23(cell_name=_short_name("L2_pyramidal")),
            "cell_metadata": {
                "morpho_type": "pyramidal",
                "electro_type": "excitatory",
                "layer": "2",
                "zdist_origin": 1,
                "measure_dipole": True,
                "reference": "",
                "color": "#a41e4f",
                "marker": "^",
            },
        },
        "L5_inhibitory": {
            "cell_object": human_gen_interneuron(
                cell_name=_short_name("L5_inhibitory"), layer=5
            ),
            "cell_metadata": {
                "morpho_type": "interneuron",
                "electro_type": "inhibitory",
                "layer": "5",
                "zdist_origin": 0.2,
                "measure_dipole": False,
                "reference": "",
                "color": "#77a1bb",
                "marker": "o",
            },
        },
        "L5_pyramidal": {
            "cell_object": pyramidal_humanL5ET(cell_name=_short_name("L5_pyramidal")),
            "cell_metadata": {
                "morpho_type": "pyramidal",
                "electro_type": "excitatory",
                "layer": "5",
                "zdist_origin": 0,
                "measure_dipole": True,
                "reference": "",
                "color": "#5c73b7",
                "marker": "^",
            },
        },
    }

    # Create layer positions
    layer_dict = _create_cell_coords(
        n_pyr_x=mesh_shape[0],
        n_pyr_y=mesh_shape[1],
        z_coord=1307.4,  # Default layer separation
        inplane_distance=1.0,  # in-plane distance appropriate for LFP recordings
    )

    # Map cell types to layer positions
    pos_dict = {
        "L5_pyramidal": layer_dict["L5_bottom"],
        "L2_pyramidal": layer_dict["L2_bottom"],
        "L5_inhibitory": layer_dict["L5_mid"],
        "L2_inhibitory": layer_dict["L2_mid"],
        "origin": layer_dict["origin"],
    }

    # Create network with cell types and positions
    net = Network(
        params,
        add_drives_from_params=add_drives_from_params,
        legacy_mode=legacy_mode,
        mesh_shape=mesh_shape,
        pos_dict=pos_dict,
        cell_types=cell_types,
    )

    # check variant and cell types. Basket cells are replaced by
    # interneurons in duecker_ET_model, so their parameters are rejected
    net._model_variant = _validate_params_for_model(
        net,
        params,
        "duecker_ET_model",
        require_variant=True,
        excluded_cells=("L2Basket", "L5Basket"),
    )

    delay = net.delay

    # layer2 Pyr -> layer2 Pyr
    lamtha = 6.125  # calculated from human data Campganola et al. 2022
    loc = "proximal"
    target_cell = "L2_pyramidal"
    for receptor in ["nmda", "ampa"]:
        key = f"gbar_{_short_name(target_cell)}_{_short_name(target_cell)}_{receptor}"
        weight = params[key]
        net.add_connection(
            target_cell,
            target_cell,
            loc,
            receptor,
            weight,
            delay,
            lamtha,
            allow_autapses=False,
        )
    # layer5 Pyr -> layer5 Pyr
    target_cell = "L5_pyramidal"
    for receptor in ["nmda", "ampa"]:
        key = f"gbar_{_short_name(target_cell)}_{_short_name(target_cell)}_{receptor}"
        weight = params[key]

        net.add_connection(
            target_cell,
            target_cell,
            loc,
            receptor,
            weight,
            delay,
            lamtha,
            allow_autapses=False,
        )

    # layer2 inhibitory -> layer2 Pyr
    src_cell = "L2_inhibitory"
    target_cell = "L2_pyramidal"
    lamtha = 6.125  # *0.8  # shorter space constant (Campagnola, 2022, mice data)
    loc = "soma"
    for receptor in ["gabaa", "gabab"]:
        key = f"gbar_{_short_name(src_cell)}_{_short_name(target_cell)}_{receptor}"
        weight = params[key]
        loc = "soma"
        net.add_connection(
            src_cell,
            target_cell,
            loc,
            receptor,
            weight,
            delay,
            lamtha,
            allow_autapses=False,
        )

    # layer5 inhibitory -> layer5 Pyr
    src_cell = "L5_inhibitory"
    target_cell = "L5_pyramidal"
    lamtha = 6.125  # *0.8  # shorter space constant (Campagnola, 2022, mice data)
    loc = "soma"
    for receptor in ["gabaa", "gabab"]:
        key = f"gbar_{_short_name(src_cell)}_{_short_name(target_cell)}_{receptor}"
        weight = params[key]
        net.add_connection(
            src_cell,
            target_cell,
            loc,
            receptor,
            weight,
            delay,
            lamtha,
            allow_autapses=False,
        )

    # layer2 Pyr -> layer5 Pyr
    src_cell = "L2_pyramidal"
    lamtha = 6.125
    for receptor in ["ampa", "nmda"]:
        for loc in ["proximal", "apical_2"]:
            key = f"gbar_{_short_name(src_cell)}_{_short_name(target_cell)}_{receptor}"
            weight = params[key]
            net.add_connection(
                src_cell, target_cell, loc, receptor, weight, delay, lamtha
            )

    # layer2 inhibitory -> layer5 Pyr
    src_cell = "L2_inhibitory"
    receptor = "gabaa_slow"
    lamtha = 6.125
    key = f"gbar_{_short_name(src_cell)}_{_short_name(target_cell)}_{receptor}"
    weight = params[key]

    # add GABAA connection to apical_2 as Martinotti-like inhibition (SST cells)
    loc = "apical_2"
    net.add_connection(src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # this connection is set to 0 as we're not simulating NGF cells.
    loc = "apical_tuft"
    receptor = "gabab"
    key = f"gbar_{_short_name(src_cell)}_{_short_name(target_cell)}_{receptor}"
    weight = params[key]
    net.add_connection(src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    # xx -> layer2 inhibitory
    src_cell = "L2_pyramidal"
    target_cell = "L2_inhibitory"
    lamtha = 6.125 * 0.8  # shorter space constant (Campagnola, 2022, mice data)
    key = f"gbar_{_short_name(src_cell)}_{_short_name(target_cell)}"
    weight = params[key]
    loc = "soma"
    receptor = "ampa"
    net.add_connection(src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    weight = params[key] * 0.18  # see Koh 1995; Kriener 2022
    receptor = "nmda"
    net.add_connection(src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    src_cell = "L2_inhibitory"
    lamtha = 6.125
    receptor = "gabaa"
    key = f"gbar_{_short_name(src_cell)}_{_short_name(target_cell)}_{receptor}"
    weight = params[key]
    loc = "soma"
    net.add_connection(
        src_cell,
        target_cell,
        loc,
        receptor,
        weight,
        delay,
        lamtha,
        allow_autapses=False,
    )

    receptor = "gabab"
    key = f"gbar_{_short_name(src_cell)}_{_short_name(target_cell)}_{receptor}"
    weight = params[key]
    loc = "soma"
    net.add_connection(
        src_cell,
        target_cell,
        loc,
        receptor,
        weight,
        delay,
        lamtha,
        allow_autapses=False,
    )

    # xx -> layer5 Basket
    src_cell = "L5_inhibitory"
    target_cell = "L5_inhibitory"
    lamtha = 6.125
    loc = "soma"
    receptor = "gabaa"
    key = f"gbar_{_short_name(src_cell)}_{_short_name(target_cell)}_{receptor}"
    weight = params[key]
    net.add_connection(
        src_cell,
        target_cell,
        loc,
        receptor,
        weight,
        delay,
        lamtha,
        allow_autapses=False,
    )

    receptor = "gabab"
    key = f"gbar_{_short_name(src_cell)}_{_short_name(target_cell)}_{receptor}"
    weight = params[key]
    net.add_connection(
        src_cell,
        target_cell,
        loc,
        receptor,
        weight,
        delay,
        lamtha,
        allow_autapses=False,
    )

    src_cell = "L5_pyramidal"
    lamtha = 6.125 * 0.8  # shorter space constant (Campagnola, 2022, mice data)
    key = f"gbar_{_short_name(src_cell)}_{_short_name(target_cell)}"
    weight = params[key]
    loc = "soma"
    receptor = "ampa"
    net.add_connection(src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    weight = params[key] * 0.2  # see Koh 1995; Kriener 2022
    receptor = "nmda"
    net.add_connection(src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    src_cell = "L2_pyramidal"
    lamtha = 6.125 * 0.8  # shorter space constant (Campagnola, 2022, mice data)
    key = f"gbar_{_short_name(src_cell)}_{_short_name(target_cell)}"
    weight = params[key]
    loc = "soma"
    receptor = "ampa"
    net.add_connection(src_cell, target_cell, loc, receptor, weight, delay, lamtha)

    return net


def add_erp_drives_to_jones_model(net, tstart=0.0):
    """Add drives necessary for an event related potential (ERP)

    Parameters
    ----------
    net : Instance of Network object
        Network object that will be updated with ERP drives.
        Drives are updated in place.
    tstart : float | int
        Start time of sensory input in ms. (Default 0.0 ms)

    Notes
    -----
    The first proximal input arrives at cortex ~20 ms after sensory
    stimulus. The exact delay depends random number generator due to
    random sampling of times from a gaussian.
    """
    _validate_type(net, Network, "net", "Network")
    _validate_type(tstart, (float, int), "tstart", "float or int")

    # Add distal drive
    weights_ampa_d1 = {
        "L2_basket": 0.006562,
        "L2_pyramidal": 7e-6,
        "L5_pyramidal": 0.142300,
    }
    weights_nmda_d1 = {
        "L2_basket": 0.019482,
        "L2_pyramidal": 0.004317,
        "L5_pyramidal": 0.080074,
    }
    synaptic_delays_d1 = {"L2_basket": 0.1, "L2_pyramidal": 0.1, "L5_pyramidal": 0.1}
    net.add_evoked_drive(
        "evdist1",
        mu=63.53 + tstart,
        sigma=3.85,
        numspikes=1,
        weights_ampa=weights_ampa_d1,
        weights_nmda=weights_nmda_d1,
        location="distal",
        synaptic_delays=synaptic_delays_d1,
        event_seed=274,
    )

    # Add proximal drives
    weights_ampa_p1 = {
        "L2_basket": 0.08831,
        "L2_pyramidal": 0.01525,
        "L5_basket": 0.19934,
        "L5_pyramidal": 0.00865,
    }
    synaptic_delays_prox = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }
    net.add_evoked_drive(
        "evprox1",
        mu=26.61 + tstart,
        sigma=2.47,
        numspikes=1,
        weights_ampa=weights_ampa_p1,
        weights_nmda=None,
        location="proximal",
        synaptic_delays=synaptic_delays_prox,
        event_seed=544,
    )

    weights_ampa_p2 = {
        "L2_basket": 0.000003,
        "L2_pyramidal": 1.438840,
        "L5_basket": 0.008958,
        "L5_pyramidal": 0.684013,
    }
    net.add_evoked_drive(
        "evprox2",
        mu=137.12 + tstart,
        sigma=8.33,
        numspikes=1,
        weights_ampa=weights_ampa_p2,
        location="proximal",
        synaptic_delays=synaptic_delays_prox,
        event_seed=814,
    )
