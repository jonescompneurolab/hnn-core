"""Default cell models."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Sam Neymotin <samnemo@gmail.com>

import numpy as np
from functools import partial
from .cell import Cell, Section

from .params import compare_dictionaries
from .params_default import (get_L2Pyr_params_default,
                             get_L5Pyr_params_default,
                             get_L2Pyr_params_new,
                             get_L5PyrET_params,
                             get_Int_params)
# Units for e: mV
# Units for gbar: S/cm^2 unless otherwise noted
# units for taur: ms


def _get_dends(params, cell_type, section_names, v_init = {'all': -65}):
    """Convert a flat dictionary to a nested dictionary.

    Returns
    -------
    sections : dict
        Dictionary of sections. Keys are section names
    """
    prop_names = ['L', 'diam', 'Ra', 'cm']
    sections = dict()
    for section_name in section_names:
        dend_prop = dict()
        for key in prop_names:
            if key in ['Ra', 'cm']:
                middle = 'dend'
            else:
                # map apicaltrunk -> apical_trunk etc.
                middle = section_name.replace('_', '')
            dend_prop[key] = params[f'{cell_type}_{middle}_{key}']
            if len(v_init) == 1:
                v = v_init['all']
            else:
                v = v_init[section_name]
        sections[section_name] = Section(L=dend_prop['L'],
                                         diam=dend_prop['diam'],
                                         Ra=dend_prop['Ra'],
                                         cm=dend_prop['cm'],
                                         v = v)
    return sections

def _get_basal(params, cell_type, section_names, v_init = {'all': -65}):
    """Convert a flat dictionary to a nested dictionary.

    Returns
    -------
    sections : dict
        Dictionary of sections. Keys are section names
    """
    prop_names = ['L', 'diam', 'Ra', 'cm']
    sections = dict()
    for section_name in section_names:
        dend_prop = dict()
        middle = section_name.replace('_', '')
        for key in prop_names:
            if key == 'Ra':
                dend_prop[key] = params[f'{cell_type}_dend_{key}']*2
            elif key=='cm':
                dend_prop[key] = params[f'{cell_type}_dend_{key}']
            else:
                dend_prop[key] = params[f'{cell_type}_{middle}_{key}']
            if len(v_init) == 1:
                v = v_init['all']
            else:
                v = v_init[section_name]
        sections[section_name] = Section(L=dend_prop['L'],
                                         diam=dend_prop['diam'],
                                         Ra=dend_prop['Ra'],
                                         cm=dend_prop['cm'],
                                         v = v)
    return sections


def _get_pyr_soma(p_all, cell_type, v_init = -65):
    """Get somatic properties."""
    return Section(
        L=p_all[f'{cell_type}_soma_L'],
        diam=p_all[f'{cell_type}_soma_diam'],
        cm=p_all[f'{cell_type}_soma_cm'],
        Ra=p_all[f'{cell_type}_soma_Ra'],
        v = v_init
    )


def _cell_L2Pyr(override_params, pos=(0., 0., 0), gid=0.):
    """The geometry of the default sections in L2Pyr neuron."""
    p_all = get_L2Pyr_params_default()
    if override_params is not None:
        assert isinstance(override_params, dict)
        p_all = compare_dictionaries(p_all, override_params)

    section_names = ['apical_trunk', 'apical_1', 'apical_tuft',
                     'apical_oblique', 'basal_1', 'basal_2', 'basal_3']

    sections = _get_dends(p_all, cell_type='L2Pyr',
                          section_names=section_names, v_init={'all': -71.46})
    sections['soma'] = _get_pyr_soma(p_all, 'L2Pyr')

    end_pts = {
        'soma': [[-50, 0, 765], [-50, 0, 778]],
        'apical_trunk': [[-50, 0, 778], [-50, 0, 813]],
        'apical_oblique': [[-50, 0, 813], [-250, 0, 813]],
        'apical_1': [[-50, 0, 813], [-50, 0, 993]],
        'apical_tuft': [[-50, 0, 993], [-50, 0, 1133]],
        'basal_1': [[-50, 0, 765], [-50, 0, 715]],
        'basal_2': [[-50, 0, 715], [-156, 0, 609]],
        'basal_3': [[-50, 0, 715], [56, 0, 609]],
    }

    mechanisms = {
        'km': ['gbar_km'],
        'hh2': ['gkbar_hh2', 'gnabar_hh2',
                'gl_hh2', 'el_hh2']
    }
    p_mech = _get_mechanisms(p_all, 'L2Pyr', ['soma'] + section_names,
                             mechanisms)

    for sec_name, section in sections.items():
        section._end_pts = end_pts[sec_name]

        if sec_name == 'soma':
            section.syns = ['gabaa', 'gabab']
        else:
            section.syns = ['ampa', 'nmda', 'gabaa', 'gabab']

        section.mechs = p_mech[sec_name]

    # Node description - (section_name, end_point)
    cell_tree = {
        ('apical_trunk', 0): [('apical_trunk', 1)],
        ('apical_1', 0): [('apical_1', 1)],
        ('apical_tuft', 0): [('apical_tuft', 1)],
        ('apical_oblique', 0): [('apical_oblique', 1)],
        ('basal_1', 0): [('basal_1', 1)],
        ('basal_2', 0): [('basal_2', 1)],
        ('basal_3', 0): [('basal_3', 1)],
        # Different sections connected
        ('soma', 0): [('soma', 1), ('basal_1', 0)],
        ('soma', 1): [('apical_trunk', 0)],
        ('apical_trunk', 1): [('apical_1', 0), ('apical_oblique', 0)],
        ('apical_1', 1): [('apical_tuft', 0)],
        ('basal_1', 1): [('basal_2', 0), ('basal_3', 0)]
    }

    sect_loc = {'proximal': ['apical_oblique', 'basal_2', 'basal_3'],
                'distal': ['apical_tuft']}

    synapses = _get_pyr_syn_props(p_all, 'L2Pyr')
    return Cell('L2Pyr', pos,
                sections=sections,
                synapses=synapses,
                sect_loc=sect_loc,
                cell_tree=cell_tree,
                gid=gid)


def _cell_L5Pyr(override_params, pos=(0., 0., 0), gid=0.):
    """The geometry of the default sections in L5Pyr Neuron."""

    p_all = get_L5Pyr_params_default()
    if override_params is not None:
        assert isinstance(override_params, dict)
        p_all = compare_dictionaries(p_all, override_params)

    section_names = ['apical_trunk', 'apical_1',
                     'apical_2', 'apical_tuft',
                     'apical_oblique', 'basal_1', 'basal_2', 'basal_3']
    
    v_init = {'apical_1': -71.32,
              'apical_2': -69.08,
              'apical_tuft': -67.30,
              'apical_trunk': -72,
              'soma': -72.0,
              'basal_1': -72,
              'basal_2': -72,
              'basal_3': -72,
              'apical_oblique': -72}

    sections = _get_dends(p_all, cell_type='L5Pyr',
                          section_names=section_names, v_init=v_init)
    sections['soma'] = _get_pyr_soma(p_all, 'L5Pyr', v_init=-72)

    end_pts = {
        'soma': [[0, 0, 0], [0, 0, 23]],
        'apical_trunk': [[0, 0, 23], [0, 0, 83]],
        'apical_oblique': [[0, 0, 83], [-150, 0, 83]],
        'apical_1': [[0, 0, 83], [0, 0, 483]],
        'apical_2': [[0, 0, 483], [0, 0, 883]],
        'apical_tuft': [[0, 0, 883], [0, 0, 1133]],
        'basal_1': [[0, 0, 0], [0, 0, -50]],
        'basal_2': [[0, 0, -50], [-106, 0, -156]],
        'basal_3': [[0, 0, -50], [106, 0, -156]]
    }

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
    p_mech = _get_mechanisms(p_all, 'L5Pyr', ['soma'] + section_names,
                             mechanisms)

    for sec_name, section in sections.items():
        section._end_pts = end_pts[sec_name]

        if sec_name == 'soma':
            section.syns = ['gabaa', 'gabab']
        else:
            section.syns = ['ampa', 'nmda', 'gabaa', 'gabab']

        section.mechs = p_mech[sec_name]

        if sec_name != 'soma':
            sections[sec_name].mechs['ar']['gbar_ar'] = \
                partial(_exp_g_at_dist, zero_val=1e-6,
                        exp_term=3e-3, offset=0.0)

    cell_tree = {
        ('apical_trunk', 0): [('apical_trunk', 1)],
        ('apical_1', 0): [('apical_1', 1)],
        ('apical_2', 0): [('apical_2', 1)],
        ('apical_tuft', 0): [('apical_tuft', 1)],
        ('apical_oblique', 0): [('apical_oblique', 1)],
        ('basal_1', 0): [('basal_1', 1)],
        ('basal_2', 0): [('basal_2', 1)],
        ('basal_3', 0): [('basal_3', 1)],
        # Different sections connected
        ('soma', 0): [('soma', 1), ('basal_1', 0)],
        ('soma', 1): [('apical_trunk', 0)],
        ('apical_trunk', 1): [('apical_1', 0), ('apical_oblique', 0)],
        ('apical_1', 1): [('apical_2', 0)],
        ('apical_2', 1): [('apical_tuft', 0)],
        ('basal_1', 1): [('basal_2', 0), ('basal_3', 0)]
    }

    sect_loc = {'proximal': ['apical_oblique', 'basal_2', 'basal_3'],
                'distal': ['apical_tuft']}

    synapses = _get_pyr_syn_props(p_all, 'L5Pyr')
    return Cell('L5Pyr', pos,
                sections=sections,
                synapses=synapses,
                sect_loc=sect_loc,
                cell_tree=cell_tree,
                gid=gid)


def _get_basket_soma(cell_name, v_init=-64.9737):
    end_pts = [[0, 0, 0], [0, 0, 39.]]
    return Section(
        L=39.,
        diam=20.,
        cm=0.85,
        Ra=200.,
        end_pts=end_pts
    )


# values from Chamberland et al 2023
def _get_interneuron_soma(cell_name, v_init=-75):
    end_pts = [[0, 0, 0], [0, 0, 20.]]
    return Section(
        L=20.,
        diam=20.,
        cm=1,
        Ra=200.,
        end_pts=end_pts
    )


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


def _exp_g_at_dist(x, zero_val, exp_term, offset, slope=1):
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

    return zero_val * (slope * np.exp(exp_term * x)  + offset)


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
    if cell_name == 'L2Basket':
        sect_loc = dict(proximal=['soma'], distal=['soma'])
    elif cell_name == 'L5Basket':
        sect_loc = dict(proximal=['soma'], distal=[])
    else:
        raise ValueError(f'Unknown basket cell type: {cell_name}')

    sections = dict()
    sections['soma'] = _get_basket_soma(cell_name)
    synapses = _get_basket_syn_props()
    sections['soma'].syns = list(synapses.keys())
    sections['soma'].mechs = {'hh2': dict()}

    cell_tree = None
    return Cell(cell_name, pos,
                sections=sections,
                synapses=synapses,
                sect_loc=sect_loc,
                cell_tree=cell_tree,
                gid=gid)


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
    if cell_name == 'L2Pyr':
        return _cell_L2Pyr(override_params, pos=pos, gid=gid)
    elif cell_name == 'L5Pyr':
        return _cell_L5Pyr(override_params, pos=pos, gid=gid)
    else:
        raise ValueError(f'Unknown pyramidal cell type: {cell_name}')


def _linear_g_at_dist(x, gsoma, gdend, xkink, hotzone_factor=1):
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
        Start of hotzone if hotzone_factor > 1.
    hotzone_factor: float | int 
        Increase in conducivity that creates a hotzone.

    Notes
    -----
    Linearly scales conductance along dendrite.
    Returns gdend when x > xkink.
    """
    gbar = gsoma + np.min([xkink, x]) * (gdend - gsoma) / xkink
    if x > xkink:
        gbar *= hotzone_factor

    return gbar


def pyramidal_ca(cell_name, pos, override_params=None, gid=None):
    """Calcium dynamics."""

    if override_params is None:
        override_params = dict()

    override_params['L5Pyr_soma_gkbar_hh2'] = 0.06
    override_params['L5Pyr_soma_gnabar_hh2'] = 0.32

    gbar_ca = partial(
        _linear_g_at_dist, gsoma=10., gdend=40., xkink=1501)
    gbar_na = partial(
        _linear_g_at_dist, gsoma=override_params['L5Pyr_soma_gnabar_hh2'],
        gdend=28e-4, xkink=962)
    gbar_k = partial(
        _exp_g_at_dist, zero_val=override_params['L5Pyr_soma_gkbar_hh2'],
        exp_term=-0.006, offset=1e-4/override_params['L5Pyr_soma_gkbar_hh2'])  # quick fix because I changed the function

    override_params['L5Pyr_dend_gbar_ca'] = gbar_ca
    override_params['L5Pyr_dend_gnabar_hh2'] = gbar_na
    override_params['L5Pyr_dend_gkbar_hh2'] = gbar_k

    cell = pyramidal(cell_name, pos, override_params=override_params,
                     gid=gid)

    return cell

def pyramidal_l5ET(cell_name,pos, gid=None):
        
    p_all = get_L5PyrET_params()

    # override params according to function
    gbar_Ca_HVA = partial(_linear_g_at_dist, gsoma=p_all['L5Pyr_dend_gbar_Ca_HVA']*1., gdend=p_all['L5Pyr_dend_gbar_Ca_HVA']*6.6, xkink=1500, hotzone_factor=3.9)
    gbar_Ca_LVA = partial(_linear_g_at_dist, gsoma=p_all['L5Pyr_dend_gbar_Ca_LVAst'], gdend=p_all['L5Pyr_dend_gbar_Ca_LVAst']*2.75, xkink=1500, hotzone_factor=3)
    gbar_Ih = partial(_exp_g_at_dist, zero_val=p_all['L5Pyr_dend_gbar_Ih'],exp_term = 1./323, slope=2.087, offset=-.8696)
    
    # basal dendrites
    gbar_NaTs2_t = partial(_linear_g_at_dist, gsoma=p_all['L5Pyr_basal_gbar_NaTs2_t']*0.01, gdend=0, xkink=255)
    gbar_SKv3_1 = partial(_linear_g_at_dist, gsoma=0, gdend=p_all['L5Pyr_basal_gbar_SKv3_1'], xkink=255)


    override_params = dict()
    override_params['L5Pyr_dend_gbar_Ca_HVA'] = gbar_Ca_HVA
    override_params['L5Pyr_dend_gbar_Ca_LVAst'] = gbar_Ca_LVA
    override_params['L5Pyr_dend_gbar_Ih'] = gbar_Ih
    override_params['L5Pyr_basal_gbar_NaTs2_t'] = gbar_NaTs2_t
    override_params['L5Pyr_basal_gbar_SKv3_1'] = gbar_SKv3_1



    p_all = compare_dictionaries(p_all, override_params)

    end_pts = {
            'soma': [[0, 0, 0], [0, 0, 23]],
            'apical_trunk': [[0, 0, 23], [0, 0, 83]],
            'apical_oblique': [[0, 0, 83], [-150, 0, 83]],
            'apical_1': [[0, 0, 83], [0, 0, 483]],
            'apical_2': [[0, 0, 483], [0, 0, 883]],
            'apical_tuft': [[0, 0, 883], [0, 0, 1133]],
            'basal_1': [[0, 0, 0], [0, 0, -50]],
            'basal_2': [[0, 0, -50], [-106, 0, -156]],
            'basal_3': [[0, 0, -50], [106, 0, -156]]
        }

    cell_tree = {
            ('apical_trunk', 0): [('apical_trunk', 1)],
            ('apical_1', 0): [('apical_1', 1)],
            ('apical_2', 0): [('apical_2', 1)],
            ('apical_tuft', 0): [('apical_tuft', 1)],
            ('apical_oblique', 0): [('apical_oblique', 1)],
            ('basal_1', 0): [('basal_1', 1)],
            ('basal_2', 0): [('basal_2', 1)],
            ('basal_3', 0): [('basal_3', 1)],
            # Different sections connected
            ('soma', 0): [('soma', 1), ('basal_1', 0)],
            ('soma', 1): [('apical_trunk', 0)],
            ('apical_trunk', 1): [('apical_1', 0), ('apical_oblique', 0)],
            ('apical_1', 1): [('apical_2', 0)],
            ('apical_2', 1): [('apical_tuft', 0)],
            ('basal_1', 1): [('basal_2', 0), ('basal_3', 0)]
            }

    # build sections
    section_names = list(end_pts.keys())

    # initialize section voltage
    v_init = {'soma': -64,
                'basal_1': -64,
                'basal_2': -64,
                'basal_3': -64,
                'apical_oblique': -64,
                'apical_trunk': -64,
                'apical_1': -64,
                'apical_2': -60,
                'apical_tuft': -58}

    sections_apcl = _get_dends(p_all, 'L5Pyr', section_names=['apical_trunk', 'apical_1', 'apical_2', 'apical_tuft'], v_init=v_init)
    sections_basal = _get_basal(p_all, 'L5Pyr', section_names=['basal_1', 'basal_2', 'basal_3', 'apical_oblique'], v_init=v_init)

    sections = {**sections_apcl, **sections_basal}

    sections['soma'] = _get_pyr_soma(p_all, 'L5Pyr', v_init=v_init['soma'])

    # Soma and apical mechanisms
    mechanisms = {'NaTs2_t':['gbar_NaTs2_t'], 
                    'SKv3_1': ['gbar_SKv3_1'],
                    'Nap_Et2': ['gbar_Nap_Et2'],
                    'Ca_HVA': ['gbar_Ca_HVA'],
                    'Ca_LVAst': ['gbar_Ca_LVAst'],
                    'SK_E2': ['gbar_SK_E2'],
                    'pas': ['g_pas', 'e_pas'],
                    'Ih': ['gbar_Ih'],
                    'Im': ['gbar_Im'],
                    'K_Pst': ['gbar_K_Pst'],
                    'K_Tst': ['gbar_K_Tst'],
                    'CaDynamics_E2': ['decay_CaDynamics_E2', 'gamma_CaDynamics_E2']}

    p_mech_soma = _get_mechanisms(p_all, 'L5Pyr', ['soma'], mechanisms)

    section_names = ['apical_trunk', 'apical_1', 'apical_2', 'apical_tuft']

    mechanisms = {'NaTa_t':['gbar_NaTa_t'], 
                    'SKv3_1': ['gbar_SKv3_1'],
                    'Ca_HVA': ['gbar_Ca_HVA'],
                    'Ca_LVAst': ['gbar_Ca_LVAst'],
                    'SK_E2': ['gbar_SK_E2'],
                    'pas': ['g_pas', 'e_pas'],
                    'Ih': ['gbar_Ih'],
                    'Im': ['gbar_Im'],
                    'K_Pst': ['gbar_K_Pst'],
                    'K_Tst': ['gbar_K_Tst'],
                    'CaDynamics_E2': ['decay_CaDynamics_E2', 'gamma_CaDynamics_E2']}
    
    p_mech_apical = _get_mechanisms(p_all, 'L5Pyr', section_names, mechanisms)


    # basal sections - super hacky because I can't mess with _get_mechanisms
    mechanisms = {'NaTs2_t': ['gbar_NaTs2_t'], 
                    'SKv3_1': ['gbar_SKv3_1'],
                    'pas': ['g_pas', 'e_pas'],
                    'Ih': ['gbar_Ih']}

    section_names = ['basal_1', 'basal_2', 'basal_3', 'apical_oblique']

    p_mech_basal = dict()
    for sec_name in section_names:
        this_sec_prop = dict()
        
        for mech_name in mechanisms:
            this_mech_prop = dict()
            for mech_attr in mechanisms[mech_name]:
                key = f'{cell_name}_basal_{mech_attr}'
                this_mech_prop[mech_attr] = p_all[key]
            this_sec_prop[mech_name] = this_mech_prop
        p_mech_basal[sec_name] = this_sec_prop


    p_mech = {**p_mech_soma, **p_mech_basal, **p_mech_apical}


    for sec_name, section in sections.items():
            section._end_pts = end_pts[sec_name]

            if sec_name == 'soma':
                section.syns = ['gabaa', 'gabab']
            else:
                section.syns = ['ampa', 'nmda', 'gabaa', 'gabab']

            section.mechs = p_mech[sec_name]

    sect_loc = {'proximal': ['apical_oblique', 'basal_2', 'basal_3'],
                    'distal': ['apical_tuft']}

    synapses = _get_pyr_syn_props(p_all, 'L5Pyr')

    cell = Cell(cell_name, pos,
                    sections=sections,
                    synapses=synapses,
                    sect_loc=sect_loc,
                    cell_tree=cell_tree,
                    gid=gid)
    
    return cell

def pyramidal_l5IT(cell_name,pos, gid=None):
        
    p_all = get_L5PyrIT_params()

   
    # basal dendrites
    gbar_NaTs2_t = partial(_linear_g_at_dist, gsoma=p_all['L5Pyr_basal_gbar_NaTs2_t']*0.5, gdend=0, xkink=255)
    gbar_SKv3_1 = partial(_linear_g_at_dist, gsoma=0, gdend=p_all['L5Pyr_basal_gbar_SKv3_1']*0.5, xkink=255)


    override_params = dict()

    override_params['L5Pyr_basal_gbar_NaTs2_t'] = gbar_NaTs2_t
    override_params['L5Pyr_basal_gbar_SKv3_1'] = gbar_SKv3_1



    p_all = compare_dictionaries(p_all, override_params)

    end_pts = {
            'soma': [[0, 0, 0], [0, 0, 23]],
            'apical_trunk': [[0, 0, 23], [0, 0, 83]],
            'apical_oblique': [[0, 0, 83], [-150, 0, 83]],
            'apical_1': [[0, 0, 83], [0, 0, 483]],
            'apical_2': [[0, 0, 483], [0, 0, 883]],
            'apical_tuft': [[0, 0, 883], [0, 0, 1133]],
            'basal_1': [[0, 0, 0], [0, 0, -50]],
            'basal_2': [[0, 0, -50], [-106, 0, -156]],
            'basal_3': [[0, 0, -50], [106, 0, -156]]
        }

    cell_tree = {
            ('apical_trunk', 0): [('apical_trunk', 1)],
            ('apical_1', 0): [('apical_1', 1)],
            ('apical_2', 0): [('apical_2', 1)],
            ('apical_tuft', 0): [('apical_tuft', 1)],
            ('apical_oblique', 0): [('apical_oblique', 1)],
            ('basal_1', 0): [('basal_1', 1)],
            ('basal_2', 0): [('basal_2', 1)],
            ('basal_3', 0): [('basal_3', 1)],
            # Different sections connected
            ('soma', 0): [('soma', 1), ('basal_1', 0)],
            ('soma', 1): [('apical_trunk', 0)],
            ('apical_trunk', 1): [('apical_1', 0), ('apical_oblique', 0)],
            ('apical_1', 1): [('apical_2', 0)],
            ('apical_2', 1): [('apical_tuft', 0)],
            ('basal_1', 1): [('basal_2', 0), ('basal_3', 0)]
            }

    # build sections
    section_names = list(end_pts.keys())

    # initialize section voltage
    v_init = {'soma': -64,
                'basal_1': -64,
                'basal_2': -64,
                'basal_3': -64,
                'apical_oblique': -64,
                'apical_trunk': -64,
                'apical_1': -64,
                'apical_2': -60,
                'apical_tuft': -58}

    sections_apcl = _get_dends(p_all, 'L5Pyr', section_names=['apical_trunk', 'apical_1', 'apical_2', 'apical_tuft'], v_init={'all':-68})
    sections_basal = _get_basal(p_all, 'L5Pyr', section_names=['basal_1', 'basal_2', 'basal_3', 'apical_oblique'], v_init={'all':-68})

    sections = {**sections_apcl, **sections_basal}

    sections['soma'] = _get_pyr_soma(p_all, 'L5Pyr', v_init=-68)

    # Soma and apical mechanisms
    mechanisms = {'NaTs2_t':['gbar_NaTs2_t'], 
                    'SKv3_1': ['gbar_SKv3_1'],
                    'Nap_Et2': ['gbar_Nap_Et2'],
                    'Ca_HVA': ['gbar_Ca_HVA'],
                    'Ca_LVAst': ['gbar_Ca_LVAst'],
                    'SK_E2': ['gbar_SK_E2'],
                    'pas': ['g_pas', 'e_pas'],
                    'Ih': ['gbar_Ih'],
                    'Im': ['gbar_Im'],
                    'K_Pst': ['gbar_K_Pst'],
                    'K_Tst': ['gbar_K_Tst'],
                    'CaDynamics_E2': ['decay_CaDynamics_E2', 'gamma_CaDynamics_E2']}

    p_mech_soma = _get_mechanisms(p_all, 'L5Pyr', ['soma'], mechanisms)

    section_names = ['apical_trunk', 'apical_1', 'apical_2', 'apical_tuft']

    mechanisms = {'NaTa_t':['gbar_NaTa_t'], 
                    'SKv3_1': ['gbar_SKv3_1'],
                    'Ca_HVA': ['gbar_Ca_HVA'],
                    'Ca_LVAst': ['gbar_Ca_LVAst'],
                    'SK_E2': ['gbar_SK_E2'],
                    'pas': ['g_pas', 'e_pas'],
                    'Ih': ['gbar_Ih'],
                    'Im': ['gbar_Im'],
                    'K_Pst': ['gbar_K_Pst'],
                    'K_Tst': ['gbar_K_Tst'],
                    'CaDynamics_E2': ['decay_CaDynamics_E2', 'gamma_CaDynamics_E2']}
    
    p_mech_apical = _get_mechanisms(p_all, 'L5Pyr', section_names, mechanisms)


    # basal sections - super hacky because I can't mess with _get_mechanisms
    mechanisms = {'NaTs2_t': ['gbar_NaTs2_t'], 
                    'SKv3_1': ['gbar_SKv3_1'],
                    'pas': ['g_pas', 'e_pas'],
                    'Ih': ['gbar_Ih']}

    section_names = ['basal_1', 'basal_2', 'basal_3', 'apical_oblique']

    p_mech_basal = dict()
    for sec_name in section_names:
        this_sec_prop = dict()
        
        for mech_name in mechanisms:
            this_mech_prop = dict()
            for mech_attr in mechanisms[mech_name]:
                key = f'{cell_name}_basal_{mech_attr}'
                this_mech_prop[mech_attr] = p_all[key]
            this_sec_prop[mech_name] = this_mech_prop
        p_mech_basal[sec_name] = this_sec_prop


    p_mech = {**p_mech_soma, **p_mech_basal, **p_mech_apical}


    for sec_name, section in sections.items():
            section._end_pts = end_pts[sec_name]

            if sec_name == 'soma':
                section.syns = ['gabaa', 'gabab']
            else:
                section.syns = ['ampa', 'nmda', 'gabaa', 'gabab']

            section.mechs = p_mech[sec_name]

    sect_loc = {'proximal': ['apical_oblique', 'basal_2', 'basal_3'],
                    'distal': ['apical_tuft']}

    synapses = _get_pyr_syn_props(p_all, 'L5Pyr')

    cell = Cell(cell_name, pos,
                    sections=sections,
                    synapses=synapses,
                    sect_loc=sect_loc,
                    cell_tree=cell_tree,
                    gid=gid)
    
    return cell

def pyramidal_l23(cell_name,pos, gid=None):

    p_all = get_L2Pyr_params_new()

    gbar_Ih = partial(_exp_g_at_dist, zero_val=p_all['L2Pyr_dend_gbar_Ih'],exp_term = 1/110, slope=2, offset=-0.869600)
    # gbar_NaTs2_t_32d = partial(_linear_g_at_dist, gsoma=p_all['L2Pyr_soma_gbar_NaTs2_t_32d'], gdend=p_all['L2Pyr_soma_gbar_NaTs2_t_32d']*0.1, xkink=1500)
    # gbar_SKv3_1 = partial(_linear_g_at_dist, gsoma=p_all['L2Pyr_soma_gbar_SKv3_1']*0.5, gdend=p_all['L2Pyr_soma_gbar_SKv3_1'], xkink=1500)

    override_params = dict()
    override_params['L2Pyr_dend_gbar_Ih'] = gbar_Ih
    # override_params['L2Pyr_basal_gbar_NaTs2_t_32d'] = gbar_NaTs2_t_32d
    # override_params['L2Pyr_basal_gbar_SKv3_1'] = gbar_SKv3_1

    p_all = compare_dictionaries(p_all, override_params)

    end_pts = {
            'soma': [[-50, 0, 765], [-50, 0, 778]],
            'apical_trunk': [[-50, 0, 778], [-50, 0, 813]],
            'apical_oblique': [[-50, 0, 813], [-250, 0, 813]],
            'apical_1': [[-50, 0, 813], [-50, 0, 993]],
            'apical_tuft': [[-50, 0, 993], [-50, 0, 1133]],
            'basal_1': [[-50, 0, 765], [-50, 0, 715]],
            'basal_2': [[-50, 0, 715], [-156, 0, 609]],
            'basal_3': [[-50, 0, 715], [56, 0, 609]],
        }

    cell_tree = {
    ('apical_trunk', 0): [('apical_trunk', 1)],
    ('apical_1', 0): [('apical_1', 1)],
    ('apical_tuft', 0): [('apical_tuft', 1)],
    ('apical_oblique', 0): [('apical_oblique', 1)],
    ('basal_1', 0): [('basal_1', 1)],
    ('basal_2', 0): [('basal_2', 1)],
    ('basal_3', 0): [('basal_3', 1)],
    # Different sections connected
    ('soma', 0): [('soma', 1), ('basal_1', 0)],
    ('soma', 1): [('apical_trunk', 0)],
    ('apical_trunk', 1): [('apical_1', 0), ('apical_oblique', 0)],
    ('apical_1', 1): [('apical_tuft', 0)],
    ('basal_1', 1): [('basal_2', 0), ('basal_3', 0)]}

    # build sections
    section_names = list(end_pts.keys())

    v_init = {'apical_1': -71.32,
              'apical_tuft': -66.5,
              'apical_trunk': -72.5,
              'apical_oblique': -72.5,
              'basal_1': -72.5,
              'basal_2': -72.5,
              'basal_3': -72.5,
              'soma': -72.5}
    
    sections = _get_dends(p_all, 'L2Pyr', section_names, v_init=v_init)
    sections['soma'] = _get_pyr_soma(p_all, 'L2Pyr', v_init=-72.5)


    mechanisms = {'NaTs2_t_32d': ['gbar_NaTs2_t_32d'], 
                    'SKv3_1': ['gbar_SKv3_1'],
                    'Nap_Et2': ['gbar_Nap_Et2'],
                    'Ca_HVA': ['gbar_Ca_HVA'],
                    'Ca_LVAst': ['gbar_Ca_LVAst'],
                    'SK_E2': ['gbar_SK_E2'],
                    'pas': ['g_pas', 'e_pas'],
                    'Ih': ['gbar_Ih'],
                    'Im': ['gbar_Im'],
                    'CaDynamics_E2': ['decay_CaDynamics_E2', 'gamma_CaDynamics_E2']}


    p_mech_soma = _get_mechanisms(p_all, 'L2Pyr', ['soma'], mechanisms)

    # apical sections
    mechanisms = {'NaTa_t_32d': ['gbar_NaTa_t_32d'],  
                    'SKv3_1': ['gbar_SKv3_1'],
                    'Ca_HVA': ['gbar_Ca_HVA'],
                    'Ca_LVAst': ['gbar_Ca_LVAst'],
                    'pas': ['g_pas', 'e_pas'],
                    'Ih': ['gbar_Ih'],
                    'Im': ['gbar_Im']}

    section_names = ['apical_trunk', 'apical_1', 'apical_tuft']
    p_mech_apical = _get_mechanisms(p_all, 'L2Pyr', section_names, mechanisms)

    # basal sections - super hacky because I can't mess with _get_mechanisms
    mechanisms = {'NaTs2_t_32d': ['gbar_NaTs2_t_32d'], 
                    'SKv3_1': ['gbar_SKv3_1'],
                    'pas': ['g_pas', 'e_pas'],
                    'Ih': ['gbar_Ih']}

    section_names = ['basal_1', 'basal_2', 'basal_3', 'apical_oblique']

    p_mech_basal = dict()
    for sec_name in section_names:
        this_sec_prop = dict()
        for mech_name in mechanisms:
            this_mech_prop = dict()
            for mech_attr in mechanisms[mech_name]:
                key = f'{cell_name}_basal_{mech_attr}'
                this_mech_prop[mech_attr] = p_all[key]
            this_sec_prop[mech_name] = this_mech_prop
        p_mech_basal[sec_name] = this_sec_prop

    p_mech = {**p_mech_soma, **p_mech_basal, **p_mech_apical}

    for sec_name, section in sections.items():
            section._end_pts = end_pts[sec_name]

            if sec_name == 'soma':
                section.syns = ['gabaa', 'gabab']
            else:
                section.syns = ['ampa', 'nmda', 'gabaa', 'gabab']

            section.mechs = p_mech[sec_name]

    sect_loc = {'proximal': ['apical_oblique', 'basal_2', 'basal_3'],
                'distal': ['apical_tuft']}

    synapses = _get_pyr_syn_props(p_all, 'L2Pyr')

    cell = Cell(cell_name, pos,
                    sections=sections,
                    synapses=synapses,
                    sect_loc=sect_loc,
                    cell_tree=cell_tree,
                    gid=gid)

    return cell

def interneuron(cell_name, pos, gid=None):

    p_all = get_Int_params()
    sections = dict()
    sections['soma'] = _get_interneuron_soma(cell_name, v_init=-65)
    synapses = _get_basket_syn_props()
    sections['soma'].syns = list(synapses.keys())

    if cell_name == 'L2Basket':
        sect_loc = dict(proximal=['soma'], distal=['soma'])
    elif cell_name == 'L5Basket':
        sect_loc = dict(proximal=['soma'], distal=[])

    cell_tree = None

    mechanisms = {'nas': ['gbar_nas'],
                'kdr': ['gbar_kdr'],
                'kd': ['gbar_kd'],
                'Ih': ['gbar_Ih'],
                'pas': ['g_pas','e_pas']}

    sections['soma'].mechs = dict()

    for mech_name in mechanisms:
        this_mech_prop = dict()
        for mech_attr in mechanisms[mech_name]:    
            key = f'Int_{mech_attr}'
            this_mech_prop[mech_attr] = p_all[key]
        sections['soma'].mechs[mech_name] = this_mech_prop


    cell = Cell(cell_name, pos,
                sections=sections,
                synapses=synapses,
                sect_loc=sect_loc,
                cell_tree=cell_tree,
                gid=gid)


    return cell
