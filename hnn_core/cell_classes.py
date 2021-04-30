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


def _flat_to_nested(params, cell_type, level1_keys, level2_keys):
    """Convert a flat dictionary to a nested dictionary."""
    nested_dict = dict()
    for level1_key in level1_keys:
        level2_dict = dict()
        for key in level2_keys:
            if key in ['Ra', 'cm']:
                middle = 'dend'
            else:
                # map apicaltrunk -> apical_trunk etc.
                middle = level1_key.replace('_', '')
            level2_dict[key] = params[f'{cell_type}_{middle}_{key}']
        nested_dict[level1_key] = level2_dict
    return nested_dict


def _get_soma_props(p_all, cell_type):
    """Hardcoded somatic properties."""
    return {
        'L': p_all[f'{cell_type}_soma_L'],
        'diam': p_all[f'{cell_type}_soma_diam'],
        'cm': p_all[f'{cell_type}_soma_cm'],
        'Ra': p_all[f'{cell_type}_soma_Ra'],
        'name': cell_type,
    }


def _get_basket_soma_props(cell_name):
    return {
        'L': 39.,
        'diam': 20.,
        'cm': 0.85,
        'Ra': 200.,
        'name': cell_name,
    }


def _get_syn_props(p_all, cell_type):
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


def basket(pos, cell_name='L2Basket', gid=None):
    """Get layer 2 basket cells.

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
    props = _get_basket_soma_props(cell_name)
    cell = Cell(props, pos=pos, gid=gid)

    cell.create_soma()
    cell.sections = [cell.soma]   # XXX: needed?
    cell.name = cell_name
    cell.secs = _secs_Basket()

    cell.set_geometry()

    cell.synapses = dict()
    # cell._synapse_create()
    cell.synapses['soma_ampa'] = cell.syn_create(
        cell.soma(0.5), e=0., tau1=0.5, tau2=5.)
    cell.synapses['soma_gabaa'] = cell.syn_create(
        cell.soma(0.5), e=-80, tau1=0.5, tau2=5.)
    cell.synapses['soma_nmda'] = cell.syn_create(
        cell.soma(0.5), e=0., tau1=1., tau2=20.)

    cell.soma.insert('hh2')

    if cell_name == 'L2Basket':
        cell.sect_loc = dict(proximal=['soma'], distal=['soma'])
    elif cell_name == 'L5Basket':
        cell.sect_loc = dict(proximal=['soma'], distal=[])
    return cell


class Pyr(Cell):
    """Pyramidal neuron.

    Parameters
    ----------
    pos : tuple
        Coordinates of cell soma in xyz-space
    override_params : dict or None (optional)
        Parameters specific to L2 pyramidal neurons to override the default set
    gid : int or None (optional)
        Each cell in a network is uniquely identified by it's "global ID": GID.
        The GID is an integer from 0 to n_cells, or None if the cell is not
        yet attached to a network. Once the GID is set, it cannot be changed..

    Attributes
    ----------
    name : str
        The name of the cell, 'L5Pyr' or 'L2Pyr'
    sect_loc : dict of list
        Can have keys 'proximal' or 'distal' each containing
        names of section locations that are proximal or distal.
    synapses : dict
        The synapses that the cell can use for connections.
    """

    def __init__(self, pos, celltype, override_params=None, gid=None):

        if celltype == 'L5_pyramidal':
            p_all_default = get_L5Pyr_params_default()
            self.name = 'L5Pyr'
            self.secs = _secs_L5Pyr()
            self.mechanisms = {
                'hh2': ['gkbar_hh2', 'gnabar_hh2',
                        'gl_hh2', 'el_hh2'],
                'ca': ['gbar_ca'],
                'cad': ['taur_cad'],
                'kca': ['gbar_kca'],
                'km': ['gbar_km'],
                'cat': ['gbar_cat']
            }
        elif celltype == 'L2_pyramidal':
            p_all_default = get_L2Pyr_params_default()
            self.name = 'L2Pyr'
            self.secs = _secs_L2Pyr()
            self.mechanisms = {
                'km': ['gbar_km'],
                'hh2': ['gkbar_hh2', 'gnabar_hh2',
                        'gl_hh2', 'el_hh2']}
        else:
            raise ValueError(f'Unknown pyramidal cell type: {celltype}')

        p_all = p_all_default
        if override_params is not None:
            assert isinstance(override_params, dict)
            p_all = compare_dictionaries(p_all_default, override_params)

        # Get somatic, dendritic, and synapse properties
        soma_props = _get_soma_props(p_all, self.name)

        Cell.__init__(self, soma_props, pos=pos, gid=gid)

        level2_keys = ['L', 'diam', 'Ra', 'cm']
        p_dend = _flat_to_nested(p_all, cell_type=self.name,
                                 level1_keys=self.section_names(),
                                 level2_keys=level2_keys)
        p_syn = _get_syn_props(p_all, self.name)

        # Geometry
        # dend Cm and dend Ra set using soma Cm and soma Ra
        self.create_soma()
        self.create_dends(p_dend)  # just creates the sections
        self.sections = [self.soma] + list(self.dends.values())

        self.sect_loc['proximal'] = ['apicaloblique', 'basal2', 'basal3']
        self.sect_loc['distal'] = ['apicaltuft']

        # sets geom properties; adjusted after translation from
        # hoc (2009 model)
        self.set_geometry(p_dend)

        # biophysics
        self.set_biophysics(p_all)

        if celltype == 'L5_pyramidal':
            self.soma.insert('ar')
            self.soma.gbar_ar = p_all['L5Pyr_soma_gbar_ar']

            # set dend biophysics not specified in Pyr()
            for key in self.dends:
                # insert 'ar' mechanism
                self.dends[key].insert('ar')

            # set gbar_ar
            # Value depends on distance from the soma. Soma is set as
            # origin by passing self.soma as a sec argument to h.distance()
            # Then iterate over segment nodes of dendritic sections
            # and set gbar_ar depending on h.distance(seg.x), which returns
            # distance from the soma to this point on the CURRENTLY ACCESSED
            # SECTION!!!
            h.distance(sec=self.soma)

            for key in self.dends:
                self.dends[key].push()
                for seg in self.dends[key]:
                    seg.gbar_ar = 1e-6 * np.exp(3e-3 * h.distance(seg.x))

                h.pop_section()

        # insert dipole
        yscale = self.secs[3]
        self.insert_dipole(yscale)

        # create synapses
        self._synapse_create(p_syn)

    def set_geometry(self, p_dend):
        """Define shape of the neuron and connect sections.

        Parameters
        ----------
        p_dend : dict | None
            Nested dictionary. The outer dictionary has keys
            with names of dendrites and the inner dictionary
            specifies the geometry of these sections.

            * L: length of a section in microns
            * diam: diameter of a section in microns
            * cm: membrane capacitance in micro-Farads
            * Ra: axial resistivity in ohm-cm
        """
        Cell.set_geometry(self)
        # resets length,diam,etc. based on param specification
        for key in p_dend:
            # set dend props
            self.dends[key].L = p_dend[key]['L']
            self.dends[key].diam = p_dend[key]['diam']
            self.dends[key].Ra = p_dend[key]['Ra']
            self.dends[key].cm = p_dend[key]['cm']
            # set dend nseg
            if p_dend[key]['L'] > 100.:
                self.dends[key].nseg = int(p_dend[key]['L'] / 50.)
                # make dend.nseg odd for all sections
                if not self.dends[key].nseg % 2:
                    self.dends[key].nseg += 1

    def section_names(self):
        if self.name == 'L2Pyr':
            return ['apical_trunk', 'apical_1', 'apical_tuft',
                    'apical_oblique', 'basal_1', 'basal_2', 'basal_3']
        else:
            return ['apical_trunk', 'apical_1', 'apical_2', 'apical_tuft',
                    'apical_oblique', 'basal_1', 'basal_2', 'basal_3']

    def create_dends(self, p_dend_props):
        """Create dendrites."""
        # XXX: name should be unique even across cell types?
        # otherwise Neuron cannot disambiguate, hence
        # self.name + '_' + key
        for key in p_dend_props:
            self.dends[key] = h.Section(
                name=self.name + '_' + key)  # create dend

    def _synapse_create(self, p_syn):
        """Creates synapses onto this cell."""
        # Somatic synapses
        self.synapses['soma_gabaa'] = self.syn_create(self.soma(0.5),
                                                      **p_syn['gabaa'])
        self.synapses['soma_gabab'] = self.syn_create(self.soma(0.5),
                                                      **p_syn['gabab'])

        # Dendritic synapses
        for sec in self.section_names():
            for receptor in p_syn:
                syn_key = sec.replace('_', '') + '_' + receptor
                self.synapses[syn_key] = self.syn_create(
                    self.dends[sec](0.5), **p_syn[receptor])
