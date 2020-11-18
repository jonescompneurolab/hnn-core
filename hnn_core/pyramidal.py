"""Model for Pyramidal cell class."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import numpy as np

from neuron import h

from .cell import _Cell

from .params import compare_dictionaries
from .params_default import (get_L2Pyr_params_default,
                             get_L5Pyr_params_default)

# Units for e: mV
# Units for gbar: S/cm^2 unless otherwise noted


class Pyr(_Cell):
    """Pyramidal neuron.

    Parameters
    ----------
    pos : tuple
        Coordinates of cell soma in xyz-space
    celltype : str
        Either 'L2_Pyramidal' or 'L5_Pyramidal'
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
    list_dend : list of str
        List of dendrites.
    sect_loc : dict of list
        Can have keys 'proximal' or 'distal' each containing
        names of section locations that are proximal or distal.
    celltype : str
        The cell type, 'L5_Pyramidal' or 'L2_Pyramidal'
    dends : dict
        The dendrites. The key is the name of the dendrite
        and the value is an instance of h.Section.
    synapses : dict
        The synapses that the cell can use for connections.
    """

    def __init__(self, pos, celltype, override_params=None, gid=None):

        if celltype == 'L5_pyramidal':
            p_all_default = get_L5Pyr_params_default()
        elif celltype == 'L2_pyramidal':
            p_all_default = get_L2Pyr_params_default()
        else:
            raise ValueError(f'Unknown pyramidal cell type: {celltype}')

        p_all = p_all_default
        if override_params is not None:
            assert isinstance(override_params, dict)
            p_all = compare_dictionaries(p_all_default, override_params)

        # Get somatic, dendritic, and synapse properties
        soma_props = self._get_soma_props(pos, p_all)

        _Cell.__init__(self, soma_props, gid=gid)
        self.create_soma()
        # store cell_name as self variable for later use
        self.name = soma_props['name']
        # preallocate dict to store dends
        self.dends = {}
        self.synapses = dict()
        self.sect_loc = dict()
        # for legacy use with L5Pyr
        self.list_dend = []
        self.celltype = celltype

        p_dend = self._get_dend_props(p_all)
        p_syn = self._get_syn_props(p_all)

        # Geometry
        # dend Cm and dend Ra set using soma Cm and soma Ra
        self.create_dends(p_dend)  # just creates the sections
        # sets geom properties; adjusted after translation from
        # hoc (2009 model)
        self.set_geometry(p_dend)

        # biophysics
        self.set_biophysics(p_all)

        # insert dipole
        yscale = self.secs()[3]
        self.insert_dipole(yscale)

        # create synapses
        self._synapse_create(p_syn)

        # insert iclamp
        self.list_IClamp = []

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
        sec_pts, sec_lens, sec_diams, _, topology = self.secs()

        # Connects sections of THIS cell together.
        for connection in topology:
            # XXX: risky to use self.soma as default. Unfortunately there isn't
            # a dictionary with all the sections (including soma)
            parent_sec = self.dends.get(connection[0], self.soma)
            parent_loc = connection[1]
            child_sec = self.dends.get(connection[2], self.soma)
            child_loc = connection[3]
            child_sec.connect(parent_sec, parent_loc, child_loc)

        # Neuron shape based on Jones et al., 2009
        for sec in [self.soma] + self.list_dend:
            h.pt3dclear(sec=sec)
            sec_name = sec.name().split('_', 1)[1]
            for pt in sec_pts[sec_name]:
                h.pt3dadd(pt[0], pt[1], pt[2], 1, sec=sec)
            sec.L = sec_lens[sec_name]
            sec.diam = sec_diams[sec_name]

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

    def create_dends(self, p_dend_props):
        """Create dendrites."""
        # XXX: name should be unique even across cell types?
        # otherwise Neuron cannot disambiguate, hence
        # self.name + '_' + key
        for key in p_dend_props:
            self.dends[key] = h.Section(
                name=self.name + '_' + key)  # create dend
        # apical: 0--4; basal: 5--7
        self.list_dend = [self.dends[key] for key in
                          ['apical_trunk', 'apical_oblique', 'apical_1',
                           'apical_2', 'apical_tuft', 'basal_1', 'basal_2',
                           'basal_3'] if key in self.dends]
        self.sect_loc['proximal'] = ['apicaloblique', 'basal2', 'basal3']
        self.sect_loc['distal'] = ['apicaltuft']

    def get_sections(self):
        ls = [self.soma]
        for key in ['apical_trunk', 'apical_1', 'apical_2', 'apical_tuft',
                    'apical_oblique', 'basal_1', 'basal_2', 'basal_3']:
            if key in self.dends:
                ls.append(self.dends[key])
        return ls

    def _get_dend_props(self, p_all):
        """Returns hardcoded dendritic properties."""
        props = {
            'apical_trunk': {
                'L': p_all['%s_apicaltrunk_L' % self.name],
                'diam': p_all['%s_apicaltrunk_diam' % self.name],
                'cm': p_all['%s_dend_cm' % self.name],
                'Ra': p_all['%s_dend_Ra' % self.name],
            },
            'apical_1': {
                'L': p_all['%s_apical1_L' % self.name],
                'diam': p_all['%s_apical1_diam' % self.name],
                'cm': p_all['%s_dend_cm' % self.name],
                'Ra': p_all['%s_dend_Ra' % self.name],
            },
            'apical_tuft': {
                'L': p_all['%s_apicaltuft_L' % self.name],
                'diam': p_all['%s_apicaltuft_diam' % self.name],
                'cm': p_all['%s_dend_cm' % self.name],
                'Ra': p_all['%s_dend_Ra' % self.name],
            },
            'apical_oblique': {
                'L': p_all['%s_apicaloblique_L' % self.name],
                'diam': p_all['%s_apicaloblique_diam' % self.name],
                'cm': p_all['%s_dend_cm' % self.name],
                'Ra': p_all['%s_dend_Ra' % self.name],
            },
            'basal_1': {
                'L': p_all['%s_basal1_L' % self.name],
                'diam': p_all['%s_basal1_diam' % self.name],
                'cm': p_all['%s_dend_cm' % self.name],
                'Ra': p_all['%s_dend_Ra' % self.name],
            },
            'basal_2': {
                'L': p_all['%s_basal2_L' % self.name],
                'diam': p_all['%s_basal2_diam' % self.name],
                'cm': p_all['%s_dend_cm' % self.name],
                'Ra': p_all['%s_dend_Ra' % self.name],
            },
            'basal_3': {
                'L': p_all['%s_basal3_L' % self.name],
                'diam': p_all['%s_basal3_diam' % self.name],
                'cm': p_all['%s_dend_cm' % self.name],
                'Ra': p_all['%s_dend_Ra' % self.name],
            },
        }
        if self.name == 'L5Pyr':
            props.update({
                'apical_2': {
                    'L': p_all['L5Pyr_apical2_L'],
                    'diam': p_all['L5Pyr_apical2_diam'],
                    'cm': p_all['L5Pyr_dend_cm'],
                    'Ra': p_all['L5Pyr_dend_Ra'],
                },
            })
        return props

    def _get_syn_props(self, p_all):
        return {
            'ampa': {
                'e': p_all['%s_ampa_e' % self.name],
                'tau1': p_all['%s_ampa_tau1' % self.name],
                'tau2': p_all['%s_ampa_tau2' % self.name],
            },
            'nmda': {
                'e': p_all['%s_nmda_e' % self.name],
                'tau1': p_all['%s_nmda_tau1' % self.name],
                'tau2': p_all['%s_nmda_tau2' % self.name],
            },
            'gabaa': {
                'e': p_all['%s_gabaa_e' % self.name],
                'tau1': p_all['%s_gabaa_tau1' % self.name],
                'tau2': p_all['%s_gabaa_tau2' % self.name],
            },
            'gabab': {
                'e': p_all['%s_gabab_e' % self.name],
                'tau1': p_all['%s_gabab_tau1' % self.name],
                'tau2': p_all['%s_gabab_tau2' % self.name],
            }
        }

    def _synapse_create(self, p_syn):
        """Creates synapses onto this cell."""
        # Somatic synapses
        self.synapses['soma_gabaa'] = self.syn_create(self.soma(0.5),
                                                      **p_syn['gabaa'])
        self.synapses['soma_gabab'] = self.syn_create(self.soma(0.5),
                                                      **p_syn['gabab'])

        # Dendritic synapses
        self.synapses['apicaloblique_ampa'] = self.syn_create(
            self.dends['apical_oblique'](0.5), **p_syn['ampa'])
        self.synapses['apicaloblique_nmda'] = self.syn_create(
            self.dends['apical_oblique'](0.5), **p_syn['nmda'])

        self.synapses['basal2_ampa'] = self.syn_create(
            self.dends['basal_2'](0.5), **p_syn['ampa'])
        self.synapses['basal2_nmda'] = self.syn_create(
            self.dends['basal_2'](0.5), **p_syn['nmda'])

        self.synapses['basal3_ampa'] = self.syn_create(
            self.dends['basal_3'](0.5), **p_syn['ampa'])
        self.synapses['basal3_nmda'] = self.syn_create(
            self.dends['basal_3'](0.5), **p_syn['nmda'])

        self.synapses['apicaltuft_ampa'] = self.syn_create(
            self.dends['apical_tuft'](0.5), **p_syn['ampa'])
        self.synapses['apicaltuft_nmda'] = self.syn_create(
            self.dends['apical_tuft'](0.5), **p_syn['nmda'])

        if self.name == 'L5Pyr':
            self.synapses['apicaltuft_gabaa'] = self.syn_create(
                self.dends['apical_tuft'](0.5), **p_syn['gabaa'])


class L2Pyr(Pyr):
    """Layer 2 pyramidal cell class.

    Parameters
    ----------
    pos : tuple
        Coordinates of cell soma in xyz-space
    override_params : dict or None (optional)
        Parameters specific to L2 pyramidal neurons to override the default set
    gid : int or None (optional)
        Each cell in a network is uniquely identified by it's "global ID": GID.
        The GID is an integer from 0 to n_cells, or None if the cell is not
        yet attached to a network. Once the GID is set, it cannot be changed.

    Attributes
    ----------
    name : str
        The name of the cell
    list_dend : list of str
        List of dendrites.
    dends : dict
        The dendrites. The key is the name of the dendrite
        and the value is an instance of h.Section.
    synapses : dict
        The synapses that the cell can use for connections.
    """

    def __init__(self, pos=None, override_params=None, gid=None):
        Pyr.__init__(self, pos, 'L2_pyramidal', override_params, gid=gid)

    def _get_soma_props(self, pos, p_all):
        """Hardcoded somatic properties."""
        return {
            'pos': pos,
            'L': p_all['L2Pyr_soma_L'],
            'diam': p_all['L2Pyr_soma_diam'],
            'cm': p_all['L2Pyr_soma_cm'],
            'Ra': p_all['L2Pyr_soma_Ra'],
            'name': 'L2Pyr',
        }

    def secs(self):
        """The geometry of the default sections in the neuron."""
        sec_pts = {
            'soma': [[-50, 765, 0], [-50, 778, 0]],
            'apical_trunk': [[-50, 778, 0], [-50, 813, 0]],
            'apical_oblique': [[-50, 813, 0], [-250, 813, 0]],
            'apical_1': [[-50, 813, 0], [-50, 993, 0]],
            'apical_tuft': [[-50, 993, 0], [-50, 1133, 0]],
            'basal_1': [[-50, 765, 0], [-50, 715, 0]],
            'basal_2': [[-50, 715, 0], [-156, 609, 0]],
            'basal_3': [[-50, 715, 0], [56, 609, 0]],
        }
        # increased by 70% for human
        sec_lens = {  # microns
            'soma': 22.1,
            'apical_trunk': 59.5,
            'apical_oblique': 340,
            'apical_1': 306,
            'apical_tuft': 238,
            'basal_1': 85,
            'basal_2': 255,
            'basal_3': 255
        }
        sec_diams = {  # microns
            'soma': 23.4,
            'apical_trunk': 4.25,
            'apical_oblique': 3.91,
            'apical_1': 4.08,
            'apical_tuft': 3.4,
            'basal_1': 4.25,
            'basal_2': 2.72,
            'basal_3': 2.72
        }
        sec_scales = {  # factor to scale the dipole by
            'soma': 1.,
            'apical_trunk': 1.,
            'apical_oblique': 0.,
            'apical_1': 1.,
            'apical_tuft': 1.,
            'basal_1': -1.,
            'basal_2': -np.sqrt(2.) / 2.,
            'basal_3': -np.sqrt(2.) / 2.
        }
        # parent, parent_end, child, {child_start=0}
        topology = [
            # Distal (Apical)
            ['soma', 1, 'apical_trunk', 0],
            ['apical_trunk', 1, 'apical_1', 0],
            ['apical_1', 1, 'apical_tuft', 0],
            # apical_oblique comes off distal end of apical_trunk
            ['apical_trunk', 1, 'apical_oblique', 0],
            # Proximal (basal)
            ['soma', 0, 'basal_1', 0],
            ['basal_1', 1, 'basal_2', 0],
            ['basal_1', 1, 'basal_3', 0]
        ]
        return sec_pts, sec_lens, sec_diams, sec_scales, topology

    def set_biophysics(self, p_all):
        """Adds biophysics to soma."""

        # Insert 'hh2' mechanism
        self.soma.insert('hh2')
        self.soma.gkbar_hh2 = p_all['L2Pyr_soma_gkbar_hh2']
        self.soma.gl_hh2 = p_all['L2Pyr_soma_gl_hh2']
        self.soma.el_hh2 = p_all['L2Pyr_soma_el_hh2']
        self.soma.gnabar_hh2 = p_all['L2Pyr_soma_gnabar_hh2']

        # Insert 'km' mechanism
        # Units: pS/um^2
        self.soma.insert('km')
        self.soma.gbar_km = p_all['L2Pyr_soma_gbar_km']

        # set dend biophysics
        # iterate over keys in self.dends and set biophysics for each dend
        for key in self.dends:
            # neuron syntax is used to set values for mechanisms
            # sec.gbar_mech = x sets value of gbar for mech to x for all segs
            # in a section. This method is significantly faster than using
            # a for loop to iterate over all segments to set mech values

            # Insert 'hh' mechanism
            self.dends[key].insert('hh2')
            self.dends[key].gkbar_hh2 = p_all['L2Pyr_dend_gkbar_hh2']
            self.dends[key].gl_hh2 = p_all['L2Pyr_dend_gl_hh2']
            self.dends[key].gnabar_hh2 = p_all['L2Pyr_dend_gnabar_hh2']
            self.dends[key].el_hh2 = p_all['L2Pyr_dend_el_hh2']

            # Insert 'km' mechanism
            # Units: pS/um^2
            self.dends[key].insert('km')
            self.dends[key].gbar_km = p_all['L2Pyr_dend_gbar_km']


# Units for e: mV
# Units for gbar: S/cm^2 unless otherwise noted
# units for taur: ms

class L5Pyr(Pyr):
    """Layer 5 Pyramidal class.

    Parameters
    ----------
    pos : tuple
        Coordinates of cell soma in xyz-space
    override_params : dict or None (optional)
        Parameters specific to L2 pyramidal neurons to override the default set
    gid : int or None (optional)
        Each cell in a network is uniquely identified by it's "global ID": GID.
        The GID is an integer from 0 to n_cells, or None if the cell is not
        yet attached to a network. Once the GID is set, it cannot be changed.

    Attributes
    ----------
    name : str
        The name of the cell
    list_dend : list of str
        List of dendrites.
    dends : dict
        The dendrites. The key is the name of the dendrite
        and the value is an instance of h.Section.
    synapses : dict
        The synapses that the cell can use for connections.
    """

    def __init__(self, pos=None, override_params=None, gid=None):
        """Get default L5Pyr params and update them with
            corresponding params in p."""
        Pyr.__init__(self, pos, 'L5_pyramidal', override_params, gid=gid)

    def secs(self):
        """The geometry of the default sections in the Neuron."""
        sec_pts = {
            'soma': [[0, 0, 0], [0, 23, 0]],
            'apical_trunk': [[0, 23, 0], [0, 83, 0]],
            'apical_oblique': [[0, 83, 0], [-150, 83, 0]],
            'apical_1': [[0, 83, 0], [0, 483, 0]],
            'apical_2': [[0, 483, 0], [0, 883, 0]],
            'apical_tuft': [[0, 883, 0], [0, 1133, 0]],
            'basal_1': [[0, 0, 0], [0, -50, 0]],
            'basal_2': [[0, -50, 0], [-106, -156, 0]],
            'basal_3': [[0, -50, 0], [106, -156, 0]]
        }
        sec_lens = {  # microns
            'soma': 39,
            'apical_trunk': 102,
            'apical_oblique': 255,
            'apical_1': 680,
            'apical_2': 680,
            'apical_tuft': 425,
            'basal_1': 85,
            'basal_2': 255,
            'basal_3': 255
        }
        sec_diams = {  # microns
            'soma': 28.9,
            'apical_trunk': 10.2,
            'apical_oblique': 5.1,
            'apical_1': 7.48,
            'apical_2': 4.93,
            'apical_tuft': 3.4,
            'basal_1': 6.8,
            'basal_2': 8.5,
            'basal_3': 8.5
        }
        sec_scales = {  # factor to scale the dipole by
            'soma': 1.,
            'apical_trunk': 1.,
            'apical_oblique': 0.,
            'apical_1': 1.,
            'apical_2': 1.,
            'apical_tuft': 1.,
            'basal_1': -1.,
            'basal_2': -np.sqrt(2.) / 2.,
            'basal_3': -np.sqrt(2.) / 2.
        }
        topology = [
            # Distal (Apical)
            ['soma', 1, 'apical_trunk', 0],
            ['apical_trunk', 1, 'apical_1', 0],
            ['apical_1', 1, 'apical_2', 0],
            ['apical_2', 1, 'apical_tuft', 0],
            # apical_oblique comes off distal end of apical_trunk
            ['apical_trunk', 1, 'apical_oblique', 0],
            # Proximal (basal)
            ['soma', 0, 'basal_1', 0],
            ['basal_1', 1, 'basal_2', 0],
            ['basal_1', 1, 'basal_3', 0]
        ]
        return sec_pts, sec_lens, sec_diams, sec_scales, topology

    def _get_soma_props(self, pos, p_all):
        """Sets somatic properties. Returns dictionary."""
        return {
            'pos': pos,
            'L': p_all['L5Pyr_soma_L'],
            'diam': p_all['L5Pyr_soma_diam'],
            'cm': p_all['L5Pyr_soma_cm'],
            'Ra': p_all['L5Pyr_soma_Ra'],
            'name': 'L5Pyr',
        }

    def set_biophysics(self, p_all):
        "Set the biophysics for the default Pyramidal cell."

        # Insert 'hh2' mechanism
        self.soma.insert('hh2')
        self.soma.gkbar_hh2 = p_all['L5Pyr_soma_gkbar_hh2']
        self.soma.gnabar_hh2 = p_all['L5Pyr_soma_gnabar_hh2']
        self.soma.gl_hh2 = p_all['L5Pyr_soma_gl_hh2']
        self.soma.el_hh2 = p_all['L5Pyr_soma_el_hh2']

        # insert 'ca' mechanism
        # Units: pS/um^2
        self.soma.insert('ca')
        self.soma.gbar_ca = p_all['L5Pyr_soma_gbar_ca']

        # insert 'cad' mechanism
        # units of tau are ms
        self.soma.insert('cad')
        self.soma.taur_cad = p_all['L5Pyr_soma_taur_cad']

        # insert 'kca' mechanism
        # units are S/cm^2?
        self.soma.insert('kca')
        self.soma.gbar_kca = p_all['L5Pyr_soma_gbar_kca']

        # Insert 'km' mechanism
        # Units: pS/um^2
        self.soma.insert('km')
        self.soma.gbar_km = p_all['L5Pyr_soma_gbar_km']

        # insert 'cat' mechanism
        self.soma.insert('cat')
        self.soma.gbar_cat = p_all['L5Pyr_soma_gbar_cat']

        # insert 'ar' mechanism
        self.soma.insert('ar')
        self.soma.gbar_ar = p_all['L5Pyr_soma_gbar_ar']

        # set dend biophysics not specified in Pyr()
        for key in self.dends:
            # Insert 'hh2' mechanism
            self.dends[key].insert('hh2')
            self.dends[key].gkbar_hh2 = p_all['L5Pyr_dend_gkbar_hh2']
            self.dends[key].gl_hh2 = p_all['L5Pyr_dend_gl_hh2']
            self.dends[key].gnabar_hh2 = p_all['L5Pyr_dend_gnabar_hh2']
            self.dends[key].el_hh2 = p_all['L5Pyr_dend_el_hh2']

            # Insert 'ca' mechanims
            # Units: pS/um^2
            self.dends[key].insert('ca')
            self.dends[key].gbar_ca = p_all['L5Pyr_dend_gbar_ca']

            # Insert 'cad' mechanism
            self.dends[key].insert('cad')
            self.dends[key].taur_cad = p_all['L5Pyr_dend_taur_cad']

            # Insert 'kca' mechanism
            self.dends[key].insert('kca')
            self.dends[key].gbar_kca = p_all['L5Pyr_dend_gbar_kca']

            # Insert 'km' mechansim
            # Units: pS/um^2
            self.dends[key].insert('km')
            self.dends[key].gbar_km = p_all['L5Pyr_dend_gbar_km']

            # insert 'cat' mechanism
            self.dends[key].insert('cat')
            self.dends[key].gbar_cat = p_all['L5Pyr_dend_gbar_cat']

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
