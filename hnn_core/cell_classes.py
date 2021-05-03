"""Model for Pyramidal cell class."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import numpy as np

from neuron import h

from .cell import Cell

from .params import compare_dictionaries
from .params_default import (get_L2Pyr_params_default,
                             get_L5Pyr_params_default,
                             _secs_L2Pyr, _secs_L5Pyr,
                             _secs_Basket)

# Units for e: mV
# Units for gbar: S/cm^2 unless otherwise noted
# units for taur: ms


def _get_dend_props(params, cell_type, section_names, prop_names):
    """Convert a flat dictionary to a nested dictionary.

    Returns
    -------
    dend_props : dict
        Nested dictionary. The outer dictionary has keys
        with names of dendrites and the inner dictionary
        specifies the geometry of these sections.

        * L: length of a section in microns
        * diam: diameter of a section in microns
        * cm: membrane capacitance in micro-Farads
        * Ra: axial resistivity in ohm-cm
    """
    dend_props = dict()
    for section_name in section_names:
        dend_prop = dict()
        for key in prop_names:
            if key in ['Ra', 'cm']:
                middle = 'dend'
            else:
                # map apicaltrunk -> apical_trunk etc.
                middle = section_name.replace('_', '')
            dend_prop[key] = params[f'{cell_type}_{middle}_{key}']
        dend_props[section_name] = dend_prop
    return dend_props


def _get_pyr_soma_props(p_all, cell_type):
    """Get somatic properties."""
    return {
        'L': p_all[f'{cell_type}_soma_L'],
        'diam': p_all[f'{cell_type}_soma_diam'],
        'cm': p_all[f'{cell_type}_soma_cm'],
        'Ra': p_all[f'{cell_type}_soma_Ra']
    }


def _get_basket_soma_props(cell_name):
    return {
        'L': 39.,
        'diam': 20.,
        'cm': 0.85,
        'Ra': 200.
    }


def _get_pyr_syn_props(p_all, cell_type):
    return {
        'ampa': {
            'e': p_all['%s_ampa_e' % cell_type],
            'tau1': p_all['%s_ampa_tau1' % cell_type],
            'tau2': p_all['%s_ampa_tau2' % cell_type],
        },
        'nmda': {
            'e': p_all['%s_nmda_e' % cell_type],
            'tau1': p_all['%s_nmda_tau1' % cell_type],
            'tau2': p_all['%s_nmda_tau2' % cell_type],
        },
        'gabaa': {
            'e': p_all['%s_gabaa_e' % cell_type],
            'tau1': p_all['%s_gabaa_tau1' % cell_type],
            'tau2': p_all['%s_gabaa_tau2' % cell_type],
        },
        'gabab': {
            'e': p_all['%s_gabab_e' % cell_type],
            'tau1': p_all['%s_gabab_tau1' % cell_type],
            'tau2': p_all['%s_gabab_tau2' % cell_type],
        }
    }


def _get_basket_syn_props():
    return {
        'ampa': {
            'e': 0,
            'tau1': 0.5,
            'tau2': 5.
        },
        'gabaa': {
            'e': -80,
            'tau1': 0.5,
            'tau2': 5.
        },
        'nmda': {
            'e': 0,
            'tau1': 1.,
            'tau2': 20.
        }
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
                if sec_name == 'soma':
                    key = f'{cell_type}_soma_{mech_attr}'
                else:
                    key = f'{cell_type}_dend_{mech_attr}'
                this_mech_prop[mech_attr] = p_all[key]
            this_sec_prop[mech_name] = this_mech_prop
        mech_props[sec_name] = this_sec_prop
    return mech_props


def basket(pos, cell_name='L2Basket', gid=None):
    """Get layer 2 / layer 5 basket cells.

    Parameters
    ----------
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
    cell = Cell(pos=pos, gid=gid)
    cell.name = cell_name

    p_secs = dict()
    p_secs['soma'] = _get_basket_soma_props(cell_name)
    p_syn = _get_basket_syn_props()
    p_secs['soma']['syns'] = list(p_syn.keys())
    p_secs['soma']['mechs'] = {'hh2': dict()}

    cell.secs = _secs_Basket()

    if cell_name == 'L2Basket':
        cell.sect_loc = dict(proximal=['soma'], distal=['soma'])
    elif cell_name == 'L5Basket':
        cell.sect_loc = dict(proximal=['soma'], distal=[])

    cell.build(p_secs, p_syn)

    return cell


def pyramidal(pos, celltype, override_params=None, gid=None):
    """Pyramidal neuron.

    Parameters
    ----------
    pos : tuple
        Coordinates of cell soma in xyz-space
    celltype : str
        'L5_pyramidal' or 'L2_pyramidal'. The pyramidal cell type.
    override_params : dict or None (optional)
        Parameters specific to L2 pyramidal neurons to override the default set
    gid : int or None (optional)
        Each cell in a network is uniquely identified by it's "global ID": GID.
        The GID is an integer from 0 to n_cells, or None if the cell is not
        yet attached to a network. Once the GID is set, it cannot be changed.
    """

    cell = Cell(pos=pos, gid=gid)
    if celltype == 'L5_pyramidal':
        p_all_default = get_L5Pyr_params_default()
        cell.name = 'L5Pyr'
        # units = ['pS/um^2', 'S/cm^2', 'pS/um^2', '??', 'tau', '??']
        mechanisms = {
            'hh2': ['gkbar_hh2', 'gnabar_hh2',
                    'gl_hh2', 'el_hh2'],
            'ca': ['gbar_ca'],
            'cad': ['taur_cad'],
            'kca': ['gbar_kca'],
            'km': ['gbar_km'],
            'cat': ['gbar_cat'],
            'ar': ['gbar_ar']
        }
        section_names = ['apical_trunk', 'apical_1',
                         'apical_2', 'apical_tuft',
                         'apical_oblique', 'basal_1', 'basal_2', 'basal_3']
        cell.secs = _secs_L5Pyr()
    elif celltype == 'L2_pyramidal':
        p_all_default = get_L2Pyr_params_default()
        cell.name = 'L2Pyr'
        mechanisms = {
            'km': ['gbar_km'],
            'hh2': ['gkbar_hh2', 'gnabar_hh2',
                    'gl_hh2', 'el_hh2']}
        section_names = ['apical_trunk', 'apical_1', 'apical_tuft',
                         'apical_oblique', 'basal_1', 'basal_2', 'basal_3']
        cell.secs = _secs_L2Pyr()
    else:
        raise ValueError(f'Unknown pyramidal cell type: {celltype}')

    p_all = p_all_default
    if override_params is not None:
        assert isinstance(override_params, dict)
        p_all = compare_dictionaries(p_all_default, override_params)

    prop_names = ['L', 'diam', 'Ra', 'cm']
    # Get somatic, dendritic, and synapse properties
    p_soma = _get_pyr_soma_props(p_all, cell.name)
    p_dend = _get_dend_props(p_all, cell_type=cell.name,
                             section_names=section_names,
                             prop_names=prop_names)
    p_syn = _get_pyr_syn_props(p_all, cell.name)
    p_secs = p_dend.copy()
    p_secs['soma'] = p_soma
    p_mech = _get_mechanisms(p_all, cell.name, ['soma'] + section_names,
                             mechanisms)
    for key in p_secs:
        p_secs[key]['mechs'] = p_mech[key]
    for key in p_secs:
        if key == 'soma':
            syns = ['gabaa', 'gabab']
        else:
            syns = list(p_syn.keys())
        p_secs[key]['syns'] = syns

    cell.sect_loc['proximal'] = ['apicaloblique', 'basal2', 'basal3']
    cell.sect_loc['distal'] = ['apicaltuft']

    cell.build(p_secs, p_syn)

    if celltype == 'L5_pyramidal':

        # set gbar_ar
        # Value depends on distance from the soma. Soma is set as
        # origin by passing cell.soma as a sec argument to h.distance()
        # Then iterate over segment nodes of dendritic sections
        # and set gbar_ar depending on h.distance(seg.x), which returns
        # distance from the soma to this point on the CURRENTLY ACCESSED
        # SECTION!!!
        h.distance(sec=cell.soma)

        for key in cell.dends:
            cell.dends[key].push()
            for seg in cell.dends[key]:
                seg.gbar_ar = 1e-6 * np.exp(3e-3 * h.distance(seg.x))

            h.pop_section()

    # insert dipole
    yscale = cell.secs[3]
    cell.insert_dipole(yscale)

    return cell
