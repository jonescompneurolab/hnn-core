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

    Attributes
    ----------
    name : str
        The name of the cell, 'L5Pyr' or 'L2Pyr'
    dends : dict
        The dendrites. The key is the name of the dendrite
        and the value is an instance of h.Section.
    synapses : dict
        The synapses that the cell can use for connections.
    list_dend : list of str
        List of dendrites.
    sect_loc : dict of list
        Can have keys 'proximal' or 'distal' each containing
        names of section locations that are proximal or distal.
    celltype : str
        The cell type, 'L5_Pyramidal' or 'L2_Pyramidal'
    """

    def __init__(self, gid, soma_props):
        _Cell.__init__(self, gid, soma_props)
        self.create_soma()
        # store cell_name as self variable for later use
        self.name = soma_props['name']
        # preallocate dict to store dends
        self.dends = {}
        self.synapses = dict()
        self.sect_loc = dict()
        # for legacy use with L5Pyr
        self.list_dend = []
        self.celltype = 'Pyramidal'

    def get_sectnames(self):
        """Create dictionary of section names with entries
           to scale section lengths to length along z-axis."""
        seclist = h.SectionList()
        seclist.wholetree(sec=self.soma)
        d = dict((sect.name(), 1.) for sect in seclist)
        for key in d.keys():
            # basal_2 and basal_3 at 45 degree angle to z-axis.
            if 'basal_2' in key:
                d[key] = np.sqrt(2) / 2.
            elif 'basal_3' in key:
                d[key] = np.sqrt(2) / 2.
            # apical_oblique at 90 perpendicular to z-axis
            elif 'apical_oblique' in key:
                d[key] = 0.
            # All basalar dendrites extend along negative z-axis
            if 'basal' in key:
                d[key] = -d[key]
        return d

    def basic_shape(self):
        """Define shape of the neuron."""
        # THESE AND LENGHTHS MUST CHANGE TOGETHER!!!
        for sec in [self.soma] + self.list_dend:
            h.pt3dclear(sec=sec)
            sec_name = sec.name().split('_', 1)[1]
            for pt in self.sec_pts()[sec_name]:
                h.pt3dadd(pt[0], pt[1], pt[2], 1, sec=sec)

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

    def set_dend_props(self, p_dend_props):
        """"Iterate over keys in p_dend_props. Create dend for each key."""
        for key in p_dend_props:
            # set dend props
            self.dends[key].L = p_dend_props[key]['L']
            self.dends[key].diam = p_dend_props[key]['diam']
            self.dends[key].Ra = p_dend_props[key]['Ra']
            self.dends[key].cm = p_dend_props[key]['cm']
            # set dend nseg
            if p_dend_props[key]['L'] > 100.:
                self.dends[key].nseg = int(p_dend_props[key]['L'] / 50.)
                # make dend.nseg odd for all sections
                if not self.dends[key].nseg % 2:
                    self.dends[key].nseg += 1

    def _get_dend_props(self):
        """Returns hardcoded dendritic properties."""
        props = {
            'apical_trunk': {
                'L': self.p_all['%s_apicaltrunk_L' % self.name],
                'diam': self.p_all['%s_apicaltrunk_diam' % self.name],
                'cm': self.p_all['%s_dend_cm' % self.name],
                'Ra': self.p_all['%s_dend_Ra' % self.name],
            },
            'apical_1': {
                'L': self.p_all['%s_apical1_L' % self.name],
                'diam': self.p_all['%s_apical1_diam' % self.name],
                'cm': self.p_all['%s_dend_cm' % self.name],
                'Ra': self.p_all['%s_dend_Ra' % self.name],
            },
            'apical_tuft': {
                'L': self.p_all['%s_apicaltuft_L' % self.name],
                'diam': self.p_all['%s_apicaltuft_diam' % self.name],
                'cm': self.p_all['%s_dend_cm' % self.name],
                'Ra': self.p_all['%s_dend_Ra' % self.name],
            },
            'apical_oblique': {
                'L': self.p_all['%s_apicaloblique_L' % self.name],
                'diam': self.p_all['%s_apicaloblique_diam' % self.name],
                'cm': self.p_all['%s_dend_cm' % self.name],
                'Ra': self.p_all['%s_dend_Ra' % self.name],
            },
            'basal_1': {
                'L': self.p_all['%s_basal1_L' % self.name],
                'diam': self.p_all['%s_basal1_diam' % self.name],
                'cm': self.p_all['%s_dend_cm' % self.name],
                'Ra': self.p_all['%s_dend_Ra' % self.name],
            },
            'basal_2': {
                'L': self.p_all['%s_basal2_L' % self.name],
                'diam': self.p_all['%s_basal2_diam' % self.name],
                'cm': self.p_all['%s_dend_cm' % self.name],
                'Ra': self.p_all['%s_dend_Ra' % self.name],
            },
            'basal_3': {
                'L': self.p_all['%s_basal3_L' % self.name],
                'diam': self.p_all['%s_basal3_diam' % self.name],
                'cm': self.p_all['%s_dend_cm' % self.name],
                'Ra': self.p_all['%s_dend_Ra' % self.name],
            },
        }
        if self.name == 'L5Pyr':
            props.update({
                'apical_2': {
                    'L': self.p_all['L5Pyr_apical2_L'],
                    'diam': self.p_all['L5Pyr_apical2_diam'],
                    'cm': self.p_all['L5Pyr_dend_cm'],
                    'Ra': self.p_all['L5Pyr_dend_Ra'],
                },
            })
        return props

    def _get_syn_props(self):
        return {
            'ampa': {
                'e': self.p_all['%s_ampa_e' % self.name],
                'tau1': self.p_all['%s_ampa_tau1' % self.name],
                'tau2': self.p_all['%s_ampa_tau2' % self.name],
            },
            'nmda': {
                'e': self.p_all['%s_nmda_e' % self.name],
                'tau1': self.p_all['%s_nmda_tau1' % self.name],
                'tau2': self.p_all['%s_nmda_tau2' % self.name],
            },
            'gabaa': {
                'e': self.p_all['%s_gabaa_e' % self.name],
                'tau1': self.p_all['%s_gabaa_tau1' % self.name],
                'tau2': self.p_all['%s_gabaa_tau2' % self.name],
            },
            'gabab': {
                'e': self.p_all['%s_gabab_e' % self.name],
                'tau1': self.p_all['%s_gabab_tau1' % self.name],
                'tau2': self.p_all['%s_gabab_tau2' % self.name],
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
    gid : int
        The cell id.
    p : dict
        The parameters dictionary.

    Attributes
    ----------
    name : str
        The name of the cell
    dends : dict
        The dendrites. The key is the name of the dendrite
        and the value is an instance of h.Section.
    list_dend : list of h.Section
        List of dendrites.
    """

    def __init__(self, gid=-1, pos=-1, p={}):
        # Get default L2Pyr params and update them with any
        # corresponding params in p
        p_all_default = get_L2Pyr_params_default()
        self.p_all = compare_dictionaries(p_all_default, p)

        # Get somatic, dendritic, and synapse properties
        p_soma = self._get_soma_props(pos)

        # usage: Pyr.__init__(self, soma_props)
        Pyr.__init__(self, gid, p_soma)

        p_dend = self._get_dend_props()
        p_syn = self._get_syn_props()

        self.celltype = 'L2_pyramidal'

        # geometry
        # creates dict of dends: self.dends
        self.create_dends(p_dend)
        self.topol()  # sets the connectivity between sections
        # sets geom properties;
        # adjusted after translation from hoc (2009 model)
        self.geom(p_dend)

        # biophysics
        self._biophys_soma()
        self._biophys_dends()

        # dipole_insert() comes from Cell()
        self.yscale = self.get_sectnames()
        self.dipole_insert(self.yscale)

        # create synapses
        self._synapse_create(p_syn)
        # self.__synapse_create()

        # run record_current_soma(), defined in Cell()
        self.record_current_soma()

    # Returns hardcoded somatic properties
    def _get_soma_props(self, pos):
        return {
            'pos': pos,
            'L': self.p_all['L2Pyr_soma_L'],
            'diam': self.p_all['L2Pyr_soma_diam'],
            'cm': self.p_all['L2Pyr_soma_cm'],
            'Ra': self.p_all['L2Pyr_soma_Ra'],
            'name': 'L2Pyr',
        }

    def geom(self, p_dend):
        """The geometry."""
        # increased by 70% for human
        soma = self.soma
        dend = self.list_dend
        sec_lens = [59.5, 340, 306, 238, 85, 255, 255]
        sec_diams = [4.25, 3.91, 4.08, 3.4, 4.25, 2.72, 2.72]

        soma.L = 22.1
        soma.diam = 23.4

        for idx, dend in enumerate(self.list_dend):
            dend.L = sec_lens[idx]
            dend.diam = sec_diams[idx]

        # resets length,diam,etc. based on param specification
        self.set_dend_props(p_dend)

    def topol(self):
        """Connects sections of THIS cell together."""
        # child.connect(parent, parent_end, {child_start=0})
        # Distal (Apical)
        self.dends['apical_trunk'].connect(self.soma, 1, 0)
        self.dends['apical_1'].connect(self.dends['apical_trunk'], 1, 0)
        self.dends['apical_tuft'].connect(self.dends['apical_1'], 1, 0)

        # apical_oblique comes off distal end of apical_trunk
        self.dends['apical_oblique'].connect(self.dends['apical_trunk'], 1, 0)

        # Proximal (basal)
        self.dends['basal_1'].connect(self.soma, 0, 0)
        self.dends['basal_2'].connect(self.dends['basal_1'], 1, 0)
        self.dends['basal_3'].connect(self.dends['basal_1'], 1, 0)

        self.basic_shape()  # translated from original hoc (2009 model)

    def sec_pts(self):
        return {
            'soma': [[-50, 765, 0], [-50, 778, 0]],
            'apical_trunk': [[-50, 778, 0], [-50, 813, 0]],
            'apical_oblique': [[-50, 813, 0], [-250, 813, 0]],
            'apical_1': [[-50, 813, 0], [-50, 993, 0]],
            'apical_tuft': [[-50, 993, 0], [-50, 1133, 0]],
            'basal_1': [[-50, 765, 0], [-50, 715, 0]],
            'basal_2': [[-50, 715, 0], [-156, 609, 0]],
            'basal_3': [[-50, 715, 0], [56, 609, 0]],
        }

    def _biophys_soma(self):
        """Adds biophysics to soma."""
        # set soma biophysics specified in Pyr
        # self.pyr_biophys_soma()

        # Insert 'hh2' mechanism
        self.soma.insert('hh2')
        self.soma.gkbar_hh2 = self.p_all['L2Pyr_soma_gkbar_hh2']
        self.soma.gl_hh2 = self.p_all['L2Pyr_soma_gl_hh2']
        self.soma.el_hh2 = self.p_all['L2Pyr_soma_el_hh2']
        self.soma.gnabar_hh2 = self.p_all['L2Pyr_soma_gnabar_hh2']

        # Insert 'km' mechanism
        # Units: pS/um^2
        self.soma.insert('km')
        self.soma.gbar_km = self.p_all['L2Pyr_soma_gbar_km']

    def _biophys_dends(self):
        """Defining biophysics for dendrites."""
        # set dend biophysics
        # iterate over keys in self.dends and set biophysics for each dend
        for key in self.dends:
            # neuron syntax is used to set values for mechanisms
            # sec.gbar_mech = x sets value of gbar for mech to x for all segs
            # in a section. This method is significantly faster than using
            # a for loop to iterate over all segments to set mech values

            # Insert 'hh' mechanism
            self.dends[key].insert('hh2')
            self.dends[key].gkbar_hh2 = self.p_all['L2Pyr_dend_gkbar_hh2']
            self.dends[key].gl_hh2 = self.p_all['L2Pyr_dend_gl_hh2']
            self.dends[key].gnabar_hh2 = self.p_all['L2Pyr_dend_gnabar_hh2']
            self.dends[key].el_hh2 = self.p_all['L2Pyr_dend_el_hh2']

            # Insert 'km' mechanism
            # Units: pS/um^2
            self.dends[key].insert('km')
            self.dends[key].gbar_km = self.p_all['L2Pyr_dend_gbar_km']


# Units for e: mV
# Units for gbar: S/cm^2 unless otherwise noted
# units for taur: ms

class L5Pyr(Pyr):
    """Layer 5 Pyramidal class.

    Attributes
    ----------
    name : str
        The name of the cell
    dends : dict
        The dendrites. The key is the name of the dendrite
        and the value is an instance of h.Section.
    list_dend : list of h.Section
        List of dendrites.
    """

    def __init__(self, gid=-1, pos=-1, p={}):
        """Get default L5Pyr params and update them with
            corresponding params in p."""
        p_all_default = get_L5Pyr_params_default()
        self.p_all = compare_dictionaries(p_all_default, p)

        # Get somatic, dendirtic, and synapse properties
        p_soma = self.__get_soma_props(pos)

        Pyr.__init__(self, gid, p_soma)
        p_dend = self._get_dend_props()
        p_syn = self._get_syn_props()

        self.celltype = 'L5_pyramidal'

        # Geometry
        # dend Cm and dend Ra set using soma Cm and soma Ra
        self.create_dends(p_dend)  # just creates the sections
        self.topol()  # sets the connectivity between sections
        # sets geom properties; adjusted after translation from
        # hoc (2009 model)
        self.geom(p_dend)

        # biophysics
        self.__biophys_soma()
        self.__biophys_dends()

        # Dictionary of length scales to calculate dipole without
        # 3d shape. Comes from Pyr().
        # dipole_insert() comes from Cell()
        self.yscale = self.get_sectnames()
        self.dipole_insert(self.yscale)

        # create synapses
        self._synapse_create(p_syn)

        # insert iclamp
        self.list_IClamp = []

        # run record current soma, defined in Cell()
        self.record_current_soma()

    def sec_pts(self):
        return {
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

    def geom(self, p_dend):
        """The geometry."""
        soma = self.soma
        dend = self.list_dend
        # soma.L = 13 # BUSH 1999 spike amp smaller
        sec_lens = [102, 255, 680, 680, 425, 85, 255, 255]
        sec_diams = [10.2, 5.1, 7.48, 4.93, 3.4, 6.8, 8.5, 8.5]

        soma.L = 39  # Bush 1993
        # soma.diam = 18.95 # Bush 1999
        soma.diam = 28.9  # Bush 1993

        for idx, dend in enumerate(self.list_dend):
            dend.L = sec_lens[idx]
            dend.diam = sec_diams[idx]

        # resets length,diam,etc. based on param specification
        self.set_dend_props(p_dend)

    def __get_soma_props(self, pos):
        """Sets somatic properties. Returns dictionary."""
        return {
            'pos': pos,
            'L': self.p_all['L5Pyr_soma_L'],
            'diam': self.p_all['L5Pyr_soma_diam'],
            'cm': self.p_all['L5Pyr_soma_cm'],
            'Ra': self.p_all['L5Pyr_soma_Ra'],
            'name': 'L5Pyr',
        }

    def topol(self):
        """Connects sections of this cell together."""

        # child.connect(parent, parent_end, {child_start=0})
        # Distal (apical)
        self.dends['apical_trunk'].connect(self.soma, 1, 0)
        self.dends['apical_1'].connect(self.dends['apical_trunk'], 1, 0)
        self.dends['apical_2'].connect(self.dends['apical_1'], 1, 0)
        self.dends['apical_tuft'].connect(self.dends['apical_2'], 1, 0)

        # apical_oblique comes off distal end of apical_trunk
        self.dends['apical_oblique'].connect(self.dends['apical_trunk'], 1, 0)

        # Proximal (basal)
        self.dends['basal_1'].connect(self.soma, 0, 0)
        self.dends['basal_2'].connect(self.dends['basal_1'], 1, 0)
        self.dends['basal_3'].connect(self.dends['basal_1'], 1, 0)

        self.basic_shape()  # translated from original hoc (2009 model)

    # adds biophysics to soma
    def __biophys_soma(self):
        # set soma biophysics specified in Pyr
        # self.pyr_biophys_soma()

        # Insert 'hh2' mechanism
        self.soma.insert('hh2')
        self.soma.gkbar_hh2 = self.p_all['L5Pyr_soma_gkbar_hh2']
        self.soma.gnabar_hh2 = self.p_all['L5Pyr_soma_gnabar_hh2']
        self.soma.gl_hh2 = self.p_all['L5Pyr_soma_gl_hh2']
        self.soma.el_hh2 = self.p_all['L5Pyr_soma_el_hh2']

        # insert 'ca' mechanism
        # Units: pS/um^2
        self.soma.insert('ca')
        self.soma.gbar_ca = self.p_all['L5Pyr_soma_gbar_ca']

        # insert 'cad' mechanism
        # units of tau are ms
        self.soma.insert('cad')
        self.soma.taur_cad = self.p_all['L5Pyr_soma_taur_cad']

        # insert 'kca' mechanism
        # units are S/cm^2?
        self.soma.insert('kca')
        self.soma.gbar_kca = self.p_all['L5Pyr_soma_gbar_kca']

        # Insert 'km' mechanism
        # Units: pS/um^2
        self.soma.insert('km')
        self.soma.gbar_km = self.p_all['L5Pyr_soma_gbar_km']

        # insert 'cat' mechanism
        self.soma.insert('cat')
        self.soma.gbar_cat = self.p_all['L5Pyr_soma_gbar_cat']

        # insert 'ar' mechanism
        self.soma.insert('ar')
        self.soma.gbar_ar = self.p_all['L5Pyr_soma_gbar_ar']

    def __biophys_dends(self):
        # set dend biophysics specified in Pyr()
        # self.pyr_biophys_dends()

        # set dend biophysics not specified in Pyr()
        for key in self.dends:
            # Insert 'hh2' mechanism
            self.dends[key].insert('hh2')
            self.dends[key].gkbar_hh2 = self.p_all['L5Pyr_dend_gkbar_hh2']
            self.dends[key].gl_hh2 = self.p_all['L5Pyr_dend_gl_hh2']
            self.dends[key].gnabar_hh2 = self.p_all['L5Pyr_dend_gnabar_hh2']
            self.dends[key].el_hh2 = self.p_all['L5Pyr_dend_el_hh2']

            # Insert 'ca' mechanims
            # Units: pS/um^2
            self.dends[key].insert('ca')
            self.dends[key].gbar_ca = self.p_all['L5Pyr_dend_gbar_ca']

            # Insert 'cad' mechanism
            self.dends[key].insert('cad')
            self.dends[key].taur_cad = self.p_all['L5Pyr_dend_taur_cad']

            # Insert 'kca' mechanism
            self.dends[key].insert('kca')
            self.dends[key].gbar_kca = self.p_all['L5Pyr_dend_gbar_kca']

            # Insert 'km' mechansim
            # Units: pS/um^2
            self.dends[key].insert('km')
            self.dends[key].gbar_km = self.p_all['L5Pyr_dend_gbar_km']

            # insert 'cat' mechanism
            self.dends[key].insert('cat')
            self.dends[key].gbar_cat = self.p_all['L5Pyr_dend_gbar_cat']

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
