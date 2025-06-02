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


def _get_dends(params, cell_type, section_names):
    """Convert a flat dictionary to a nested dictionary.

    Returns
    -------
    sections : dict
        Dictionary of sections. Keys are section names
    """
    prop_names = ["L", "diam", "Ra", "cm"]
    sections = dict()
    for section_name in section_names:
        dend_prop = dict()
        for key in prop_names:
            if key in ["Ra", "cm"]:
                middle = "dend"
            else:
                # map apicaltrunk -> apical_trunk etc.
                middle = section_name.replace("_", "")
            dend_prop[key] = params[f"{cell_type}_{middle}_{key}"]
        sections[section_name] = Section(
            L=dend_prop["L"],
            diam=dend_prop["diam"],
            Ra=dend_prop["Ra"],
            cm=dend_prop["cm"],
        )
    return sections


def _get_pyr_soma(p_all, cell_type):
    """Get somatic properties."""
    return Section(
        L=p_all[f"{cell_type}_soma_L"],
        diam=p_all[f"{cell_type}_soma_diam"],
        cm=p_all[f"{cell_type}_soma_cm"],
        Ra=p_all[f"{cell_type}_soma_Ra"],
    )


def _cell_L2Pyr(override_params, pos=(0.0, 0.0, 0), gid=0.0):
    """The geometry of the default sections in L2Pyr neuron."""
    p_all = get_L2Pyr_params_default()
    if override_params is not None:
        assert isinstance(override_params, dict)
        p_all = compare_dictionaries(p_all, override_params)

    section_names = [
        "apical_trunk",
        "apical_1",
        "apical_tuft",
        "apical_oblique",
        "basal_1",
        "basal_2",
        "basal_3",
    ]

    sections = _get_dends(p_all, cell_type="L2Pyr", section_names=section_names)
    sections["soma"] = _get_pyr_soma(p_all, "L2Pyr")

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


def _cell_L5Pyr(override_params, pos=(0.0, 0.0, 0), gid=0.0):
    """The geometry of the default sections in L5Pyr Neuron."""

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

    sections = _get_dends(p_all, cell_type="L5Pyr", section_names=section_names)
    sections["soma"] = _get_pyr_soma(p_all, "L5Pyr")

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
                _exp_g_at_dist, zero_val=1e-6, exp_term=3e-3, offset=0.0
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


def _get_basket_soma(cell_name):
    end_pts = [[0, 0, 0], [0, 0, 39.0]]
    return Section(L=39.0, diam=20.0, cm=0.85, Ra=200.0, end_pts=end_pts)


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


def _exp_g_at_dist(x, zero_val, exp_term, offset):
    """Compute exponential distance-dependent ionic conductance.

    Parameters
    ----------
    x : float | int
        Distance from soma
    zero_val : float | int
        Value of function when x = 0
    exp_term : float | int
        Multiplier of x in the exponent
    offset: float |int
        Offset value added to output

    """

    return zero_val * np.exp(exp_term * x) + offset


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
    if cell_name == "L2Basket":
        sect_loc = dict(proximal=["soma"], distal=["soma"])
    elif cell_name == "L5Basket":
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
    if cell_name == "L2Pyr":
        return _cell_L2Pyr(override_params, pos=pos, gid=gid)
    elif cell_name == "L5Pyr":
        return _cell_L5Pyr(override_params, pos=pos, gid=gid)
    else:
        raise ValueError(f"Unknown pyramidal cell type: {cell_name}")


def _linear_g_at_dist(x, gsoma, gdend, xkink):
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

    Notes
    -----
    Linearly scales conductance along dendrite.
    Returns gdend when x > xkink.
    """
    return gsoma + np.min([xkink, x]) * (gdend - gsoma) / xkink


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
        zero_val=override_params["L5Pyr_soma_gkbar_hh2"],
        exp_term=-0.006,
        offset=1e-4,
    )

    override_params["L5Pyr_dend_gbar_ca"] = gbar_ca
    override_params["L5Pyr_dend_gnabar_hh2"] = gbar_na
    override_params["L5Pyr_dend_gkbar_hh2"] = gbar_k

    cell = pyramidal(cell_name, pos, override_params=override_params, gid=gid)

    return cell
