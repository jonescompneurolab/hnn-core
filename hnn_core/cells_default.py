"""Default cell models."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Sam Neymotin <samnemo@gmail.com>

import numpy as np
from functools import partial
from .cell import Cell, Section

from .params import compare_dictionaries
from .params_default import get_L2Pyr_params_default, get_L5Pyr_params_default

# Units for e: mV
# Units for gbar: S/cm^2 unless otherwise noted
# units for taur: ms


def _get_dends(
    params,
    cell_type,
    section_names,
    v_init={"all": -65},
    is_basal_specific=False,
):
    """Create dendritic Section objects from flat parameter dictionary.

    Extracts geometric and electrical properties (length, diameter, axial resistance,
    membrane capacitance) from a flat parameter dictionary, takes initial membrane
    voltage from an argument, and constructs Section objects for each dendritic
    compartment. Handles parameter key name transformations (e.g., 'apical_trunk' ->
    'apicaltrunk') required for lookup in the parameter dictionary.

    *Importantly*, "Section objects" in this context are objects of the class
    `hnn_core/cell.py::Section`, NOT the "true" NEURON sections. The "true" NEURON
    sections are only created later, immediately before a simulation is run, using
    `NetworkBuilder._build`.

    Parameters
    ----------
    params : dict
        Flat dictionary containing cell parameters with keys formatted as
        '{cell_type}_{section}_{property}' (e.g., 'L5Pyr_apicaltrunk_L'). 'Ra' and 'cm'
        use "dend" as the middle component rather than specific section names. This
        'params' dictionary is expected to be constructed using
        functions like `params_default.py::get_L2Pyr_params_default`.
    cell_type : {'L2Pyr', 'L5Pyr'}
        Cell type identifier used as prefix in parameter key lookups.
    section_names : list of str
        Names of dendritic sections to create (e.g., ['apical_trunk', 'apical_1',
        'basal_2']). Underscores are removed for parameter lookups except for 'Ra' and
        'cm'.
    v_init : dict, default={"all": -65}
        Initial membrane potential in mV. If dict contains single key "all", that value
        is applied to all sections. Otherwise, keys must match 'section_names' for
        section-specific initialization.
    is_basal_specific : bool, default=False
        Flag indicating whether or not to use the (Duecker 2025) model's custom basal
        dendrite parameters. If True, this will read the 'Ra' and 'cm' parameters from
        'params' using '{cell_type}_basal_{property}' instead of the default
        '{cell_type}_dend_{property}' naming scheme.

    Returns
    -------
    sections : dict
        Dictionary mapping section names (str) to Section objects with attributes 'L',
        'diam', 'Ra', and 'cm' set from 'params', and 'v0' set from argument.

    Notes
    -----
    - KD: This function is where the initial voltages for the dendritic sections are
      set; these voltages are not overridden by `h.finitialize` unless called with a
      value, e.g. `h.finitialize(-65)`.
    - The 'v0' (initial voltage) parameter is handled separately from other properties
      as it is a newer addition not found in legacy parameter files.
    - In the (Jones et al., 2009) model, this is used to construct both apical and basal
      dendrite sections. In the newer (Duecker 2025) model, this is only used for the
      apical dendrite sections.
    """
    prop_names = ["L", "diam", "Ra", "cm"]
    sections = dict()
    for section_name in section_names:
        dend_prop = dict()
        for key in prop_names:
            if key in ["Ra", "cm"]:
                if is_basal_specific:
                    middle = "basal"
                else:
                    middle = "dend"
            else:
                # map apicaltrunk -> apical_trunk etc.
                middle = section_name.replace("_", "")
            dend_prop[key] = params[f"{cell_type}_{middle}_{key}"]
        # v0 is handled separately since it is "newer", and will never be found in the
        # `params` input.
        if len(v_init) == 1:
            dend_prop["v0"] = v_init["all"]
        else:
            dend_prop["v0"] = v_init[section_name]

        sections[section_name] = Section(
            L=dend_prop["L"],
            diam=dend_prop["diam"],
            Ra=dend_prop["Ra"],
            cm=dend_prop["cm"],
            v0=dend_prop["v0"],
        )
    return sections


def _get_pyr_soma(p_all, cell_type, v_init=-65):
    """Create Pyramidal somatic Section objects from flat parameter dictionary.

    Extracts geometric and electrical properties (length, diameter, axial resistance,
    membrane capacitance) from a flat parameter dictionary, takes initial membrane
    voltage from an argument, and constructs a Section object for each Pyramidal soma
    compartment.

    *Importantly*, "Section objects" in this context are objects of the class
    `hnn_core/cell.py::Section`, NOT the "true" NEURON sections. The "true" NEURON
    sections are only created later, immediately before a simulation is run, using
    `NetworkBuilder._build`.

    Parameters
    ----------
    p_all : dict
        Flat dictionary containing cell parameters with keys formatted as
        '{cell_type}_soma_{property}' (e.g., 'L5Pyr_soma_L'). This 'p_all' dictionary
        is expected to be constructed using functions like
        `params_default.py::get_L2Pyr_params_default`.
    cell_type : {'L2Pyr', 'L5Pyr'}
        Cell type identifier used as prefix in parameter key lookups.
    v_init : dict, default={"all": -65}
        Initial membrane potential in mV. If dict contains single key "all", that value
        is applied to all sections. Otherwise, keys must match 'section_names' for
        section-specific initialization.

    Returns
    -------
    Section
        A Section object with attributes 'L', 'diam', 'Ra', and 'cm' set from 'p_all',
        and 'v0' set from argument.

    Notes
    -----
    - KD: This function is where the initial voltages for the somata are set; these
      voltages are not overridden by `h.finitialize` unless called with a value,
      e.g. `h.finitialize(-65)`.
    - The 'v0' (initial voltage) parameter is handled separately from other properties
      as it is a newer addition not found in legacy parameter files.
    """
    return Section(
        L=p_all[f"{cell_type}_soma_L"],
        diam=p_all[f"{cell_type}_soma_diam"],
        cm=p_all[f"{cell_type}_soma_cm"],
        Ra=p_all[f"{cell_type}_soma_Ra"],
        v0=v_init,
    )


def _cell_L2Pyr(override_params, pos=(0.0, 0.0, 0), gid=0):
    """Create a Cell object of the Layer 2/3 Pyramidal cell type.

    This constructs a Layer 2/3 Pyramidal cell type (i.e. 'L2Pyr') using the following
    steps:
    1. "Loads" the default parameters for this celltype using
      `params_default.py::get_L2Pyr_params_default`.
    2. Overrides the default parameters based on the 'override_params' argument.
    3. Creates all dendrite Section compartment objects, including initializing their
      voltages.
    4. Creates the soma Section compartment object, including initializing its voltage.
    5. Programs the 'end_pts' of each Section using hard-coded values.
    6. Sets the mechanisms for each Section. In this celltype, all Sections contain the
      same set of mechanisms.
    7. Sets the receiving synapse types for each Section. Somata only receive inhibitory
      synapses, while dendrites receive all types.
    8. Constructs a map of the cell tree, connecting each Section appropriately.
    9. Assigns different dendritic Sections to either the 'proximal' or 'distal' groups.
    10. Sets parameters for all synaptic types.
    11. Finally, creates the Cell object with all of the above information, including
      'pos' cell position and 'gid' identifier that are set by arguments.

    *Importantly*, "Cell objects" in this context are objects of the class
    `hnn_core/cell.py::Cell`, not "true" NEURON cells. Similarly, "Section objects"
    in this context are objects of the class `hnn_core/cell.py::Section`, NOT the "true"
    Section objects as created and used by NEURON. The "true" NEURON sections and cells
    are only created later, immediately before a simulation is run, using
    `NetworkBuilder._build`.

    Parameters
    ----------
    override_params : dict
        Flat dictionary containing cell parameters with keys formatted as
        '{cell_type}_{section}_{property}' (e.g., 'L2Pyr_apicaltrunk_L'), where
        key-value pairs are only provided for those values where the user wants to use
        custom, non-default parameters. The default parameters can be found in
        `params_default.py::get_L2Pyr_params_default`. If no overrides are desired, then
        this argument should be None.
    pos : tuple of (float, float, int), default=(0.0, 0.0, 0)
        3-dimensional position to place the cell at.
    gid : int, default=0
        The unique, "global ID" (GID) of the cell.

    Returns
    -------
    Cell
        A Cell object of the Layer 2/3 Pyramidal cell type.
    """
    p_all = get_L2Pyr_params_default()
    if override_params is not None:
        assert isinstance(override_params, dict)
        p_all = compare_dictionaries(p_all, override_params)

    # All sections of this cell type use the same initial membrane voltage:
    all_v_init = -71.46

    section_names = [
        "apical_trunk",
        "apical_1",
        "apical_tuft",
        "apical_oblique",
        "basal_1",
        "basal_2",
        "basal_3",
    ]

    sections = _get_dends(
        p_all,
        cell_type="L2Pyr",
        section_names=section_names,
        v_init={
            "all": all_v_init,
        },
    )
    sections["soma"] = _get_pyr_soma(
        p_all,
        "L2Pyr",
        v_init=all_v_init,
    )

    end_pts = {
        "soma": [[-50, 0, 765], [-50, 0, 778]],
        "apical_trunk": [[-50, 0, 778], [-50, 0, 813]],
        "apical_oblique": [[-50, 0, 813], [-250, 0, 813]],
        "apical_1": [[-50, 0, 813], [-50, 0, 993]],
        "apical_tuft": [[-50, 0, 993], [-50, 0, 1133]],
        "basal_1": [[-50, 0, 765], [-50, 0, 715]],
        "basal_2": [[-50, 0, 715], [-156, 0, 609]],
        "basal_3": [[-50, 0, 715], [56, 0, 609]],
    }

    mechanisms = {
        "km": ["gbar_km"],
        "hh2": ["gkbar_hh2", "gnabar_hh2", "gl_hh2", "el_hh2"],
    }
    p_mech = _get_mechanisms(p_all, "L2Pyr", ["soma"] + section_names, mechanisms)

    for sec_name, section in sections.items():
        section._end_pts = end_pts[sec_name]

        if sec_name == "soma":
            section.syns = ["gabaa", "gabab"]
        else:
            section.syns = ["ampa", "nmda", "gabaa", "gabab"]

        section.mechs = p_mech[sec_name]

    # Node description - (section_name, end_point)
    cell_tree = {
        ("apical_trunk", 0): [("apical_trunk", 1)],
        ("apical_1", 0): [("apical_1", 1)],
        ("apical_tuft", 0): [("apical_tuft", 1)],
        ("apical_oblique", 0): [("apical_oblique", 1)],
        ("basal_1", 0): [("basal_1", 1)],
        ("basal_2", 0): [("basal_2", 1)],
        ("basal_3", 0): [("basal_3", 1)],
        # Different sections connected
        ("soma", 0): [("soma", 1), ("basal_1", 0)],
        ("soma", 1): [("apical_trunk", 0)],
        ("apical_trunk", 1): [("apical_1", 0), ("apical_oblique", 0)],
        ("apical_1", 1): [("apical_tuft", 0)],
        ("basal_1", 1): [("basal_2", 0), ("basal_3", 0)],
    }

    sect_loc = {
        "proximal": ["apical_oblique", "basal_2", "basal_3"],
        "distal": ["apical_tuft"],
    }

    synapses = _get_pyr_syn_props(p_all, "L2Pyr")
    return Cell(
        "L2Pyr",
        pos,
        sections=sections,
        synapses=synapses,
        sect_loc=sect_loc,
        cell_tree=cell_tree,
        gid=gid,
    )


def _cell_L5Pyr(override_params, pos=(0.0, 0.0, 0), gid=0):
    """Create a Cell object of the Layer 5 Pyramidal cell type.

    This constructs a Layer 5 Pyramidal cell type (i.e. 'L5Pyr') using the following
    steps:
    1. "Loads" the default parameters for this celltype using
      `params_default.py::get_L5Pyr_params_default`.
    2. Overrides the default parameters based on the 'override_params' argument.
    3. Creates all dendrite Section compartment objects, including initializing their
      voltages.
    4. Creates the soma Section compartment object, including initializing its voltage.
    5. Programs the 'end_pts' of each Section using hard-coded values.
    6. Sets the mechanisms for each Section. In this celltype, all Sections contain the
      same set of mechanisms.
    7. Sets the receiving synapse types for each Section. Somata only receive inhibitory
      synapses, while dendrites receive all types.
    8. Sets the AR current maximal conductance according to a spatial algorithm.
    9. Constructs a map of the cell tree, connecting each Section appropriately.
    10. Assigns different dendritic Sections to either the 'proximal' or 'distal' groups.
    11. Sets parameters for all synaptic types.
    12. Finally, creates the Cell object with all of the above information, including
      'pos' cell position and 'gid' identifier that are set by arguments.

    *Importantly*, "Cell objects" in this context are objects of the class
    `hnn_core/cell.py::Cell`, not "true" NEURON cells. Similarly, "Section objects"
    in this context are objects of the class `hnn_core/cell.py::Section`, NOT the "true"
    Section objects as created and used by NEURON. The "true" NEURON sections and cells
    are only created later, immediately before a simulation is run, using
    `NetworkBuilder._build`.

    Parameters
    ----------
    override_params : dict
        Flat dictionary containing cell parameters with keys formatted as
        '{cell_type}_{section}_{property}' (e.g., 'L5Pyr_apicaltrunk_L'), where
        key-value pairs are only provided for those values where the user wants to use
        custom, non-default parameters. The default parameters can be found in
        `params_default.py::get_L5Pyr_params_default`. If no overrides are desired, then
        this argument should be None.
    pos : tuple of (float, float, int), default=(0.0, 0.0, 0)
        3-dimensional position to place the cell at.
    gid : int, default=0
        The unique, "global ID" (GID) of the cell.

    Returns
    -------
    Cell
        A Cell object of the Layer 5 Pyramidal cell type.
    """

    p_all = get_L5Pyr_params_default()
    if override_params is not None:
        assert isinstance(override_params, dict)
        p_all = compare_dictionaries(p_all, override_params)

    section_names = [
        "apical_trunk",
        "apical_1",
        "apical_2",
        "apical_tuft",
        "apical_oblique",
        "basal_1",
        "basal_2",
        "basal_3",
    ]

    # Different sections of this cell type use different initial membrane voltages:
    v_init = {
        "apical_1": -71.32,
        "apical_2": -69.08,
        "apical_tuft": -67.30,
        "apical_trunk": -72,
        "soma": -72.0,
        "basal_1": -72,
        "basal_2": -72,
        "basal_3": -72,
        "apical_oblique": -72,
    }

    sections = _get_dends(
        p_all,
        cell_type="L5Pyr",
        section_names=section_names,
        v_init=v_init,
    )
    sections["soma"] = _get_pyr_soma(
        p_all,
        "L5Pyr",
        v_init=-72,
    )

    end_pts = {
        "soma": [[0, 0, 0], [0, 0, 23]],
        "apical_trunk": [[0, 0, 23], [0, 0, 83]],
        "apical_oblique": [[0, 0, 83], [-150, 0, 83]],
        "apical_1": [[0, 0, 83], [0, 0, 483]],
        "apical_2": [[0, 0, 483], [0, 0, 883]],
        "apical_tuft": [[0, 0, 883], [0, 0, 1133]],
        "basal_1": [[0, 0, 0], [0, 0, -50]],
        "basal_2": [[0, 0, -50], [-106, 0, -156]],
        "basal_3": [[0, 0, -50], [106, 0, -156]],
    }

    # AES TODO: what's up with this units comment?
    # units = ['pS/um^2', 'S/cm^2', 'pS/um^2', '??', 'tau', '??']
    mechanisms = {
        "hh2": ["gkbar_hh2", "gnabar_hh2", "gl_hh2", "el_hh2"],
        "ca": ["gbar_ca"],
        "cad": ["taur_cad"],
        "kca": ["gbar_kca"],
        "km": ["gbar_km"],
        "cat": ["gbar_cat"],
        "ar": ["gbar_ar"],
    }
    p_mech = _get_mechanisms(p_all, "L5Pyr", ["soma"] + section_names, mechanisms)

    for sec_name, section in sections.items():
        section._end_pts = end_pts[sec_name]

        if sec_name == "soma":
            section.syns = ["gabaa", "gabab"]
        else:
            section.syns = ["ampa", "nmda", "gabaa", "gabab"]

        section.mechs = p_mech[sec_name]

        if sec_name != "soma":
            sections[sec_name].mechs["ar"]["gbar_ar"] = partial(
                _exp_g_at_dist, gbar_at_zero=1e-6, exp_term=3e-3, offset=0.0
            )

    cell_tree = {
        ("apical_trunk", 0): [("apical_trunk", 1)],
        ("apical_1", 0): [("apical_1", 1)],
        ("apical_2", 0): [("apical_2", 1)],
        ("apical_tuft", 0): [("apical_tuft", 1)],
        ("apical_oblique", 0): [("apical_oblique", 1)],
        ("basal_1", 0): [("basal_1", 1)],
        ("basal_2", 0): [("basal_2", 1)],
        ("basal_3", 0): [("basal_3", 1)],
        # Different sections connected
        ("soma", 0): [("soma", 1), ("basal_1", 0)],
        ("soma", 1): [("apical_trunk", 0)],
        ("apical_trunk", 1): [("apical_1", 0), ("apical_oblique", 0)],
        ("apical_1", 1): [("apical_2", 0)],
        ("apical_2", 1): [("apical_tuft", 0)],
        ("basal_1", 1): [("basal_2", 0), ("basal_3", 0)],
    }

    sect_loc = {
        "proximal": ["apical_oblique", "basal_2", "basal_3"],
        "distal": ["apical_tuft"],
    }

    synapses = _get_pyr_syn_props(p_all, "L5Pyr")
    return Cell(
        "L5Pyr",
        pos,
        sections=sections,
        synapses=synapses,
        sect_loc=sect_loc,
        cell_tree=cell_tree,
        gid=gid,
    )


def _get_basket_soma(cell_name, v_init=-64.9737):
    end_pts = [[0, 0, 0], [0, 0, 39.0]]
    return Section(
        L=39.0,
        diam=20.0,
        cm=0.85,
        Ra=200.0,
        v0=v_init,
        end_pts=end_pts,
    )


def _get_pyr_syn_props(p_all, cell_type):
    return {
        "ampa": {
            "e": p_all["%s_ampa_e" % cell_type],
            "tau1": p_all["%s_ampa_tau1" % cell_type],
            "tau2": p_all["%s_ampa_tau2" % cell_type],
        },
        "nmda": {
            "e": p_all["%s_nmda_e" % cell_type],
            "tau1": p_all["%s_nmda_tau1" % cell_type],
            "tau2": p_all["%s_nmda_tau2" % cell_type],
        },
        "gabaa": {
            "e": p_all["%s_gabaa_e" % cell_type],
            "tau1": p_all["%s_gabaa_tau1" % cell_type],
            "tau2": p_all["%s_gabaa_tau2" % cell_type],
        },
        "gabab": {
            "e": p_all["%s_gabab_e" % cell_type],
            "tau1": p_all["%s_gabab_tau1" % cell_type],
            "tau2": p_all["%s_gabab_tau2" % cell_type],
        },
    }


def _get_basket_syn_props():
    return {
        "ampa": {"e": 0, "tau1": 0.5, "tau2": 5.0},
        "gabaa": {"e": -80, "tau1": 0.5, "tau2": 5.0},
        "nmda": {"e": 0, "tau1": 1.0, "tau2": 20.0},
    }


def _get_mechanisms(p_all, cell_type, section_names, mechanisms):
    """Get mechanism

    Parameters
    ----------
    cell_type : str
        The cell type
    section_names : str
        The section_names
    mechanisms : dict of list
        The mechanism properties to extract

    Returns
    -------
    mech_props : dict of dict of dict
        Nested dictionary of the form
        sections -> mechanism -> mechanism properties
        used to instantiate the mechanism in Neuron
    """
    mech_props = dict()
    for sec_name in section_names:
        this_sec_prop = dict()
        for mech_name in mechanisms:
            this_mech_prop = dict()
            for mech_attr in mechanisms[mech_name]:
                if sec_name == "soma":
                    key = f"{cell_type}_soma_{mech_attr}"
                else:
                    key = f"{cell_type}_dend_{mech_attr}"
                this_mech_prop[mech_attr] = p_all[key]
            this_sec_prop[mech_name] = this_mech_prop
        mech_props[sec_name] = this_sec_prop
    return mech_props


def _exp_g_at_dist(x, gbar_at_zero, exp_term, offset, slope=1):
    """Compute exponential distance-dependent ionic conductance.

    Parameters
    ----------
    x : float | int
        Distance from soma
    gbar_at_zero : float | int
        Value of function when x = 0
    exp_term : float | int
        Multiplier of x in the exponent
    offset : float | int
        Offset value added to output
    slope : int | float, default=1
        Slope of the exponential component
    """
    gbar = gbar_at_zero * (slope * np.exp(exp_term * x) + offset)
    return gbar


def basket(cell_name, pos=(0, 0, 0), gid=None):
    """Get layer 2 / layer 5 basket cells.

    Parameters
    ----------
    cell_name : str
        The name of the cell.
    pos : tuple
        Coordinates of cell soma in xyz-space
    gid : int or None (optional)
        Each cell in a network is uniquely identified by it's "global ID": GID.
        The GID is an integer from 0 to n_cells, or None if the cell is not
        yet attached to a network. Once the GID is set, it cannot be changed.

    Returns
    -------
    cell : instance of BasketSingle
        The basket cell.
    """
    if cell_name == "L2_basket":
        sect_loc = dict(proximal=["soma"], distal=["soma"])
    elif cell_name == "L5_basket":
        sect_loc = dict(proximal=["soma"], distal=[])
    else:
        raise ValueError(f"Unknown basket cell type: {cell_name}")

    sections = dict()
    sections["soma"] = _get_basket_soma(cell_name)
    synapses = _get_basket_syn_props()
    sections["soma"].syns = list(synapses.keys())
    sections["soma"].mechs = {"hh2": dict()}

    cell_tree = None
    return Cell(
        cell_name,
        pos,
        sections=sections,
        synapses=synapses,
        sect_loc=sect_loc,
        cell_tree=cell_tree,
        gid=gid,
    )


def pyramidal(cell_name, pos=(0, 0, 0), override_params=None, gid=None):
    """Pyramidal neuron.

    Parameters
    ----------
    cell_name : str
        'L5Pyr' or 'L2Pyr'. The pyramidal cell type.
    pos : tuple
        Coordinates of cell soma in xyz-space
    override_params : dict or None (optional)
        Parameters specific to L2 pyramidal neurons to override the default set
    gid : int or None (optional)
        Each cell in a network is uniquely identified by it's "global ID": GID.
        The GID is an integer from 0 to n_cells, or None if the cell is not
        yet attached to a network. Once the GID is set, it cannot be changed.
    """
    if cell_name == "L2_pyramidal":
        return _cell_L2Pyr(override_params, pos=pos, gid=gid)
    elif cell_name == "L5_pyramidal":
        return _cell_L5Pyr(override_params, pos=pos, gid=gid)
    else:
        raise ValueError(f"Unknown pyramidal cell type: {cell_name}")


def _linear_g_at_dist(
    x, gsoma, gdend, xkink, hotzone_factor=1, hotzone_boundaries=[0, 0]
):
    """Compute linear distance-dependent ionic conductance.

    Parameters
    ----------
    x : float | int
        Distance from soma
    gsoma : float | int
        Somatic conductance.
    gdend : float | int
        Dendritic conductance
    xkink : float | int
        Plateau value where conductance is fixed at gdend.
    hotzone_factor: int | float, default=1
        Increase in conducivity that creates a hotzone.
    hotzone_boundaries : [float, float]
        Start and end of hotzone if hotzone_factor > 1. Units are the same as that of
        `x`.

    Notes
    -----
    Linearly scales conductance along dendrite.
    Returns gdend when x > xkink.
    """
    gbar = gsoma + np.min([xkink, x]) * (gdend - gsoma) / xkink
    if hotzone_boundaries[0] < x < hotzone_boundaries[1]:
        gbar *= hotzone_factor

    return gbar


def pyramidal_ca(cell_name, pos, override_params=None, gid=None):
    """Calcium dynamics."""

    if override_params is None:
        override_params = dict()

    override_params["L5Pyr_soma_gkbar_hh2"] = 0.06
    override_params["L5Pyr_soma_gnabar_hh2"] = 0.32

    gbar_ca = partial(_linear_g_at_dist, gsoma=10.0, gdend=40.0, xkink=1501)
    gbar_na = partial(
        _linear_g_at_dist,
        gsoma=override_params["L5Pyr_soma_gnabar_hh2"],
        gdend=28e-4,
        xkink=962,
    )
    gbar_k = partial(
        _exp_g_at_dist,
        gbar_at_zero=override_params["L5Pyr_soma_gkbar_hh2"],
        exp_term=-0.006,
        offset=1e-4,
    )

    override_params["L5Pyr_dend_gbar_ca"] = gbar_ca
    override_params["L5Pyr_dend_gnabar_hh2"] = gbar_na
    override_params["L5Pyr_dend_gkbar_hh2"] = gbar_k

    cell = pyramidal(cell_name, pos, override_params=override_params, gid=gid)

    return cell
