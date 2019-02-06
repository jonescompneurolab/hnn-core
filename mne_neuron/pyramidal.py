"""Model for Pyramidal cell class."""

import numpy as np

from neuron import h

from .cell import Cell

from .paramrw import compare_dictionaries
from .params_default import (get_L2Pyr_params_default,
                             get_L5Pyr_params_default)

# Units for e: mV
# Units for gbar: S/cm^2 unless otherwise noted


class Pyr(Cell):
    def __init__(self, gid, soma_props):
        Cell.__init__(self, gid, soma_props)
        # store cell_name as self variable for later use
        self.name = soma_props['name']
        # preallocate dict to store dends
        self.dends = {}
        # for legacy use with L5Pyr
        self.list_dend = []

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

    def create_dends(self, p_dend_props):
        for key in p_dend_props:
            self.dends[key] = h.Section(
                name=self.name + '_' + key)  # create dend
        # apical: 0--4; basal: 5--7
        self.list_dend = [self.dends[key] for key in
                          ['apical_trunk', 'apical_oblique', 'apical_1',
                           'apical_2', 'apical_tuft', 'basal_1', 'basal_2',
                           'basal_3'] if key in self.dends]

    def set_dend_props(self, p_dend_props):
        # iterate over keys in p_dend_props. Create dend for each key.
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

    # Creates dendritic sections based only on dictionary of dendrite props
    def create_dends_new(self, p_dend_props):
        # iterate over keys in p_dend_props. Create dend for each key.
        for key in p_dend_props:
            # create dend
            self.dends[key] = h.Section(name=self.name + '_' + key)

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

        # apical: 0--4
        # basal: 5--7
        self.list_dend = [self.dends[key] for key in
                          ['apical_trunk', 'apical_oblique', 'apical_1',
                           'apical_2', 'apical_tuft', 'basal_1', 'basal_2',
                           'basal_3'] if key in self.dends]

    def get_sections(self):
        ls = [self.soma]
        for key in ['apical_trunk', 'apical_1', 'apical_2', 'apical_tuft',
                    'apical_oblique', 'basal_1', 'basal_2', 'basal_3']:
            if key in self.dends:
                ls.append(self.dends[key])
        return ls

    def get_section_names(self):
        ls = ['soma']
        for key in ['apical_trunk', 'apical_1', 'apical_2', 'apical_tuft',
                    'apical_oblique', 'basal_1', 'basal_2', 'basal_3']:
            if key in self.dends:
                ls.append(key)
        return ls

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
        self.synapses = {
            'soma_gabaa': self.syn_create(self.soma(0.5), p_syn['gabaa']),
            'soma_gabab': self.syn_create(self.soma(0.5), p_syn['gabab']),
        }

        # Dendritic synapses
        self.apicaloblique_ampa = self.syn_create(
            self.dends['apical_oblique'](0.5), p_syn['ampa'])
        self.apicaloblique_nmda = self.syn_create(
            self.dends['apical_oblique'](0.5), p_syn['nmda'])

        self.basal2_ampa = self.syn_create(
            self.dends['basal_2'](0.5), p_syn['ampa'])
        self.basal2_nmda = self.syn_create(
            self.dends['basal_2'](0.5), p_syn['nmda'])

        self.basal3_ampa = self.syn_create(
            self.dends['basal_3'](0.5), p_syn['ampa'])
        self.basal3_nmda = self.syn_create(
            self.dends['basal_3'](0.5), p_syn['nmda'])

        self.apicaltuft_ampa = self.syn_create(
            self.dends['apical_tuft'](0.5), p_syn['ampa'])
        self.apicaltuft_nmda = self.syn_create(
            self.dends['apical_tuft'](0.5), p_syn['nmda'])

        if self.name == 'L5Pyr':
            self.apicaltuft_gabaa = self.syn_create(
                self.dends['apical_tuft'](0.5), p_syn['gabaa'])


class L2Pyr(Pyr):
    """Layer 2 pyramidal cell class.

    Parameters
    ----------
    gid : int
        The cell id.
    p : dict
        The parameters dictionary.
    """

    def __init__(self, gid=-1, pos=-1, p={}):
        # Get default L2Pyr params and update them with any
        # corresponding params in p
        p_all_default = get_L2Pyr_params_default()
        self.p_all = compare_dictionaries(p_all_default, p)

        # Get somatic, dendritic, and synapse properties
        p_soma = self.__get_soma_props(pos)

        # usage: Pyr.__init__(self, soma_props)
        Pyr.__init__(self, gid, p_soma)

        p_dend = self._get_dend_props()
        p_syn = self._get_syn_props()
        # p_dend_props, dend_names = self.__get_dend_props()

        self.celltype = 'L2_pyramidal'

        # geometry
        # creates dict of dends: self.dends
        self.create_dends(p_dend)
        self.topol()  # sets the connectivity between sections
        # sets geom properties;
        # adjusted after translation from hoc (2009 model)
        self.geom(p_dend)

        # biophysics
        self.__biophys_soma()
        self.__biophys_dends()

        # dipole_insert() comes from Cell()
        self.yscale = self.get_sectnames()
        self.dipole_insert(self.yscale)

        # create synapses
        self._synapse_create(p_syn)
        # self.__synapse_create()

        # run record_current_soma(), defined in Cell()
        self.record_current_soma()

    # Returns hardcoded somatic properties
    def __get_soma_props(self, pos):
        return {
            'pos': pos,
            'L': self.p_all['L2Pyr_soma_L'],
            'diam': self.p_all['L2Pyr_soma_diam'],
            'cm': self.p_all['L2Pyr_soma_cm'],
            'Ra': self.p_all['L2Pyr_soma_Ra'],
            'name': 'L2Pyr',
        }

    def geom(self, p_dend):
        soma = self.soma
        dend = self.list_dend
        # increased by 70% for human
        soma.L = 22.1
        dend[0].L = 59.5
        dend[1].L = 340
        dend[2].L = 306
        dend[3].L = 238
        dend[4].L = 85
        dend[5].L = 255
        dend[6].L = 255
        soma.diam = 23.4
        dend[0].diam = 4.25
        dend[1].diam = 3.91
        dend[2].diam = 4.08
        dend[3].diam = 3.4
        dend[4].diam = 4.25
        dend[5].diam = 2.72
        dend[6].diam = 2.72
        # resets length,diam,etc. based on param specification
        self.set_dend_props(p_dend)

    # Connects sections of THIS cell together
    def topol(self):
        """ original topol
        connect dend(0), soma(1)
        for i = 1, 2 connect dend[i](0), dend(1)
        connect dend[3](0), dend[2](1)
        connect dend[4](0), soma(0) //was soma(1), 0 is correct!
        for i = 5, 6 connect dend[i](0), dend[4](1)

        """

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

    def basic_shape(self):
        # THESE AND LENGHTHS MUST CHANGE TOGETHER!!!
        pt3dclear = h.pt3dclear
        pt3dadd = h.pt3dadd
        soma = self.soma
        dend = self.list_dend
        pt3dclear(sec=soma)
        pt3dadd(-50, 765, 0, 1, sec=soma)
        pt3dadd(-50, 778, 0, 1, sec=soma)
        pt3dclear(sec=dend[0])
        pt3dadd(-50, 778, 0, 1, sec=dend[0])
        pt3dadd(-50, 813, 0, 1, sec=dend[0])
        pt3dclear(sec=dend[1])
        pt3dadd(-50, 813, 0, 1, sec=dend[1])
        pt3dadd(-250, 813, 0, 1, sec=dend[1])
        pt3dclear(sec=dend[2])
        pt3dadd(-50, 813, 0, 1, sec=dend[2])
        pt3dadd(-50, 993, 0, 1, sec=dend[2])
        pt3dclear(sec=dend[3])
        pt3dadd(-50, 993, 0, 1, sec=dend[3])
        pt3dadd(-50, 1133, 0, 1, sec=dend[3])
        pt3dclear(sec=dend[4])
        pt3dadd(-50, 765, 0, 1, sec=dend[4])
        pt3dadd(-50, 715, 0, 1, sec=dend[4])
        pt3dclear(sec=dend[5])
        pt3dadd(-50, 715, 0, 1, sec=dend[5])
        pt3dadd(-156, 609, 0, 1, sec=dend[5])
        pt3dclear(sec=dend[6])
        pt3dadd(-50, 715, 0, 1, sec=dend[6])
        pt3dadd(56, 609, 0, 1, sec=dend[6])

    # Adds biophysics to soma
    def __biophys_soma(self):
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

    # Defining biophysics for dendrites
    def __biophys_dends(self):
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

    # collect receptor-type-based connections here
    def parconnect(self, gid, gid_dict, pos_dict, p):
        # init dict of dicts
        # nc_dict for ampa and nmda may be the same for this cell type
        nc_dict = {
            'ampa': None,
            'nmda': None,
        }

        # Connections FROM all other L2 Pyramidal cells to this one
        for gid_src, pos in zip(gid_dict['L2_pyramidal'],
                                pos_dict['L2_pyramidal']):
            # don't be redundant, this is only possible for LIKE cells,
            # but it might not hurt to check
            if gid_src != gid:
                nc_dict['ampa'] = {
                    'pos_src': pos,
                    'A_weight': p['gbar_L2Pyr_L2Pyr_ampa'],
                    'A_delay': 1.,
                    'lamtha': 3.,
                    'threshold': p['threshold'],
                    'type_src': 'L2_pyramidal'
                }

                # parconnect_from_src(gid_presyn, nc_dict, postsyn)
                # ampa connections
                self.ncfrom_L2Pyr.append(self.parconnect_from_src(
                    gid_src, nc_dict['ampa'], self.apicaloblique_ampa))
                self.ncfrom_L2Pyr.append(self.parconnect_from_src(
                    gid_src, nc_dict['ampa'], self.basal2_ampa))
                self.ncfrom_L2Pyr.append(self.parconnect_from_src(
                    gid_src, nc_dict['ampa'], self.basal3_ampa))

                nc_dict['nmda'] = {
                    'pos_src': pos,
                    'A_weight': p['gbar_L2Pyr_L2Pyr_nmda'],
                    'A_delay': 1.,
                    'lamtha': 3.,
                    'threshold': p['threshold'],
                    'type_src': 'L2_pyramidal'
                }

                # parconnect_from_src(gid_presyn, nc_dict, postsyn)
                # nmda connections
                self.ncfrom_L2Pyr.append(self.parconnect_from_src(
                    gid_src, nc_dict['nmda'], self.apicaloblique_nmda))
                self.ncfrom_L2Pyr.append(self.parconnect_from_src(
                    gid_src, nc_dict['nmda'], self.basal2_nmda))
                self.ncfrom_L2Pyr.append(self.parconnect_from_src(
                    gid_src, nc_dict['nmda'], self.basal3_nmda))

        # connections FROM L2 basket cells TO this L2Pyr cell
        for gid_src, pos in zip(gid_dict['L2_basket'], pos_dict['L2_basket']):
            nc_dict['gabaa'] = {
                'pos_src': pos,
                'A_weight': p['gbar_L2Basket_L2Pyr_gabaa'],
                'A_delay': 1.,
                'lamtha': 50.,
                'threshold': p['threshold'],
                'type_src': 'L2_basket'
            }

            nc_dict['gabab'] = {
                'pos_src': pos,
                'A_weight': p['gbar_L2Basket_L2Pyr_gabab'],
                'A_delay': 1.,
                'lamtha': 50.,
                'threshold': p['threshold'],
                'type_src': 'L2_basket'
            }

            self.ncfrom_L2Basket.append(self.parconnect_from_src(
                gid_src, nc_dict['gabaa'], self.synapses['soma_gabaa']))
            self.ncfrom_L2Basket.append(self.parconnect_from_src(
                gid_src, nc_dict['gabab'], self.synapses['soma_gabab']))

    # may be reorganizable
    def parreceive(self, gid, gid_dict, pos_dict, p_ext):
        for gid_src, p_src, pos in zip(gid_dict['extinput'], p_ext,
                                       pos_dict['extinput']):
            # Check if AMPA params defined in p_src
            if 'L2Pyr_ampa' in p_src.keys():
                nc_dict_ampa = {
                    'pos_src': pos,
                    'A_weight': p_src['L2Pyr_ampa'][0],
                    'A_delay': p_src['L2Pyr_ampa'][1],
                    'lamtha': p_src['lamtha'],
                    'threshold': p_src['threshold'],
                    'type_src': 'ext'
                }

                # Proximal feed AMPA synapses
                if p_src['loc'] is 'proximal':
                    self.ncfrom_extinput.append(self.parconnect_from_src(
                        gid_src, nc_dict_ampa, self.basal2_ampa))
                    self.ncfrom_extinput.append(self.parconnect_from_src(
                        gid_src, nc_dict_ampa, self.basal3_ampa))
                    self.ncfrom_extinput.append(self.parconnect_from_src(
                        gid_src, nc_dict_ampa, self.apicaloblique_ampa))
                # Distal feed AMPA synapses
                elif p_src['loc'] is 'distal':
                    self.ncfrom_extinput.append(self.parconnect_from_src(
                        gid_src, nc_dict_ampa, self.apicaltuft_ampa))

            # Check is NMDA params defined in p_src
            if 'L2Pyr_nmda' in p_src.keys():
                nc_dict_nmda = {
                    'pos_src': pos,
                    'A_weight': p_src['L2Pyr_nmda'][0],
                    'A_delay': p_src['L2Pyr_nmda'][1],
                    'lamtha': p_src['lamtha'],
                    'threshold': p_src['threshold'],
                    'type_src': 'ext'
                }

                # Proximal feed NMDA synapses
                if p_src['loc'] is 'proximal':
                    self.ncfrom_extinput.append(self.parconnect_from_src(
                        gid_src, nc_dict_nmda, self.basal2_nmda))
                    self.ncfrom_extinput.append(self.parconnect_from_src(
                        gid_src, nc_dict_nmda, self.basal3_nmda))
                    self.ncfrom_extinput.append(self.parconnect_from_src(
                        gid_src, nc_dict_nmda, self.apicaloblique_nmda))
                # Distal feed NMDA synapses
                elif p_src['loc'] is 'distal':
                    self.ncfrom_extinput.append(self.parconnect_from_src(
                        gid_src, nc_dict_nmda, self.apicaltuft_nmda))

    # one parreceive function to handle all types of external parreceives
    # types must be defined explicitly here
    # this function handles evoked, gaussian, and poisson inputs
    def parreceive_ext(self, type, gid, gid_dict, pos_dict, p_ext):
        if type.startswith(('evprox', 'evdist')):
            if self.celltype in p_ext.keys():
                gid_ev = gid + gid_dict[type][0]

                # separate dictionaries for ampa and nmda evoked inputs
                nc_dict_ampa = {
                    'pos_src': pos_dict[type][gid],
                    # index 0 for ampa weight
                    'A_weight': p_ext[self.celltype][0],
                    'A_delay': p_ext[self.celltype][2],  # index 2 for delay
                    'lamtha': p_ext['lamtha_space'],
                    'threshold': p_ext['threshold'],
                    'type_src': type
                }

                nc_dict_nmda = {
                    'pos_src': pos_dict[type][gid],
                    # index 1 for nmda weight
                    'A_weight': p_ext[self.celltype][1],
                    'A_delay': p_ext[self.celltype][2],  # index 2 for delay
                    'lamtha': p_ext['lamtha_space'],
                    'threshold': p_ext['threshold'],
                    'type_src': type
                }

                if p_ext['loc'] is 'proximal':
                    self.ncfrom_ev.append(self.parconnect_from_src(
                        gid_ev, nc_dict_ampa, self.basal2_ampa))
                    self.ncfrom_ev.append(self.parconnect_from_src(
                        gid_ev, nc_dict_ampa, self.basal3_ampa))
                    self.ncfrom_ev.append(self.parconnect_from_src(
                        gid_ev, nc_dict_ampa, self.apicaloblique_ampa))

                    # NEW: note that default/original is 0 nmda weight
                    # for these proximal dends
                    self.ncfrom_ev.append(self.parconnect_from_src(
                        gid_ev, nc_dict_nmda, self.basal2_nmda))
                    self.ncfrom_ev.append(self.parconnect_from_src(
                        gid_ev, nc_dict_nmda, self.basal3_nmda))
                    self.ncfrom_ev.append(self.parconnect_from_src(
                        gid_ev, nc_dict_nmda, self.apicaloblique_nmda))

                elif p_ext['loc'] is 'distal':
                    self.ncfrom_ev.append(self.parconnect_from_src(
                        gid_ev, nc_dict_ampa, self.apicaltuft_ampa))
                    self.ncfrom_ev.append(self.parconnect_from_src(
                        gid_ev, nc_dict_nmda, self.apicaltuft_nmda))

        elif type == 'extgauss':
            # gid is this cell's gid
            # gid_dict is the whole dictionary, including the gids of
            # the extgauss
            # pos_list is also the pos of the extgauss (net origin)
            # p_ext_gauss are the params (strength, etc.)

            # gid shift is based on L2_pyramidal cells NOT L5
            # I recognize this is ugly (hack)
            # gid_shift = gid_dict['extgauss'][0] - gid_dict['L2_pyramidal'][0]
            if 'L2_pyramidal' in p_ext.keys():
                gid_extgauss = gid + gid_dict['extgauss'][0]

                nc_dict = {
                    'pos_src': pos_dict['extgauss'][gid],
                    # index 0 for ampa weight (nmda not yet used in Gauss)
                    'A_weight': p_ext['L2_pyramidal'][0],
                    'A_delay': p_ext['L2_pyramidal'][2],  # index 2 for delay
                    'lamtha': p_ext['lamtha'],
                    'threshold': p_ext['threshold'],
                    'type_src': type
                }

                self.ncfrom_extgauss.append(self.parconnect_from_src(
                    gid_extgauss, nc_dict, self.basal2_ampa))
                self.ncfrom_extgauss.append(self.parconnect_from_src(
                    gid_extgauss, nc_dict, self.basal3_ampa))
                self.ncfrom_extgauss.append(self.parconnect_from_src(
                    gid_extgauss, nc_dict, self.apicaloblique_ampa))

        elif type == 'extpois':
            if self.celltype in p_ext.keys():
                gid_extpois = gid + gid_dict['extpois'][0]

                nc_dict = {
                    'pos_src': pos_dict['extpois'][gid],
                    # index 0 for ampa weight
                    'A_weight': p_ext[self.celltype][0],
                    'A_delay': p_ext[self.celltype][2],  # index 2 for delay
                    'lamtha': p_ext['lamtha_space'],
                    'threshold': p_ext['threshold'],
                    'type_src': type
                }

                self.ncfrom_extpois.append(self.parconnect_from_src(
                    gid_extpois, nc_dict, self.basal2_ampa))
                self.ncfrom_extpois.append(self.parconnect_from_src(
                    gid_extpois, nc_dict, self.basal3_ampa))
                self.ncfrom_extpois.append(self.parconnect_from_src(
                    gid_extpois, nc_dict, self.apicaloblique_ampa))

                if p_ext[self.celltype][1] > 0.0:
                    # index 1 for nmda weight
                    nc_dict['A_weight'] = p_ext[self.celltype][1]
                    self.ncfrom_extpois.append(self.parconnect_from_src(
                        gid_extpois, nc_dict, self.basal2_nmda))
                    self.ncfrom_extpois.append(self.parconnect_from_src(
                        gid_extpois, nc_dict, self.basal3_nmda))
                    self.ncfrom_extpois.append(self.parconnect_from_src(
                        gid_extpois, nc_dict, self.apicaloblique_nmda))

        else:
            print("Warning, ext type def does not exist in L2Pyr")


# Units for e: mV
# Units for gbar: S/cm^2 unless otherwise noted
# units for taur: ms

class L5Pyr(Pyr):

    def basic_shape(self):
        # THESE AND LENGHTHS MUST CHANGE TOGETHER!!!
        pt3dclear = h.pt3dclear
        pt3dadd = h.pt3dadd
        dend = self.list_dend
        pt3dclear(sec=self.soma)
        pt3dadd(0, 0, 0, 1, sec=self.soma)
        pt3dadd(0, 23, 0, 1, sec=self.soma)
        pt3dclear(sec=dend[0])
        pt3dadd(0, 23, 0, 1, sec=dend[0])
        pt3dadd(0, 83, 0, 1, sec=dend[0])
        pt3dclear(sec=dend[1])
        pt3dadd(0, 83, 0, 1, sec=dend[1])
        pt3dadd(-150, 83, 0, 1, sec=dend[1])
        pt3dclear(sec=dend[2])
        pt3dadd(0, 83, 0, 1, sec=dend[2])
        pt3dadd(0, 483, 0, 1, sec=dend[2])
        pt3dclear(sec=dend[3])
        pt3dadd(0, 483, 0, 1, sec=dend[3])
        pt3dadd(0, 883, 0, 1, sec=dend[3])
        pt3dclear(sec=dend[4])
        pt3dadd(0, 883, 0, 1, sec=dend[4])
        pt3dadd(0, 1133, 0, 1, sec=dend[4])
        pt3dclear(sec=dend[5])
        pt3dadd(0, 0, 0, 1, sec=dend[5])
        pt3dadd(0, -50, 0, 1, sec=dend[5])
        pt3dclear(sec=dend[6])
        pt3dadd(0, -50, 0, 1, sec=dend[6])
        pt3dadd(-106, -156, 0, 1, sec=dend[6])
        pt3dclear(sec=dend[7])
        pt3dadd(0, -50, 0, 1, sec=dend[7])
        pt3dadd(106, -156, 0, 1, sec=dend[7])

    def geom(self, p_dend):
        soma = self.soma
        dend = self.list_dend
        # soma.L = 13 # BUSH 1999 spike amp smaller
        soma.L = 39  # Bush 1993
        dend[0].L = 102
        dend[1].L = 255
        dend[2].L = 680  # default 400
        dend[3].L = 680  # default 400
        dend[4].L = 425
        dend[5].L = 85
        dend[6].L = 255  # default 150
        dend[7].L = 255  # default 150
        # soma.diam = 18.95 # Bush 1999
        soma.diam = 28.9  # Bush 1993
        dend[0].diam = 10.2
        dend[1].diam = 5.1
        dend[2].diam = 7.48  # default 4.4
        dend[3].diam = 4.93  # default 2.9
        dend[4].diam = 3.4
        dend[5].diam = 6.8
        dend[6].diam = 8.5
        dend[7].diam = 8.5
        self.set_dend_props(p_dend) # resets length,diam,etc. based on param specification

    def __init__(self, gid=-1, pos=-1, p={}):
        """.Get default L5Pyr params and update them with
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
        self.geom(p_dend)  # sets geom properties; adjusted after translation from hoc (2009 model)

        # biophysics
        self.__biophys_soma()
        self.__biophys_dends()

        # Dictionary of length scales to calculate dipole without 3d shape. Comes from Pyr().
        # dipole_insert() comes from Cell()
        self.yscale = self.get_sectnames()
        self.dipole_insert(self.yscale)

        # create synapses
        self._synapse_create(p_syn)

        # insert iclamp
        self.list_IClamp = []

        # run record current soma, defined in Cell()
        self.record_current_soma()

    # Sets somatic properties. Returns dictionary.
    def __get_soma_props(self, pos):
        return {
            'pos': pos,
            'L': self.p_all['L5Pyr_soma_L'],
            'diam': self.p_all['L5Pyr_soma_diam'],
            'cm': self.p_all['L5Pyr_soma_cm'],
            'Ra': self.p_all['L5Pyr_soma_Ra'],
            'name': 'L5Pyr',
        }

    # connects sections of this cell together
    def topol (self):

        """ original topol
        connect dend(0), soma(1) // dend[0] is apical trunk
        for i = 1, 2 connect dend[i](0), dend(1) // dend[1] is oblique, dend[2] is apic1
        for i = 3, 4 connect dend[i](0), dend[i-1](1) // dend[3],dend[4] are apic2,apic tuft
        connect dend[5](0), soma(0) //was soma(1)this is correct! 
        for i = 6, 7 connect dend[i](0), dend[5](1)
        """

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

    # parallel connection function FROM all cell types TO here
    def parconnect(self, gid, gid_dict, pos_dict, p):
        # init dict of dicts
        # nc_dict for ampa and nmda may be the same for this cell type
        nc_dict = {
            'ampa': None,
            'nmda': None,
        }

        # connections FROM L5Pyr TO here
        for gid_src, pos in zip(gid_dict['L5_pyramidal'],
                                pos_dict['L5_pyramidal']):
            # no autapses
            if gid_src != gid:
                nc_dict['ampa'] = {
                    'pos_src': pos,
                    'A_weight': p['gbar_L5Pyr_L5Pyr_ampa'],
                    'A_delay': 1.,
                    'lamtha': 3.,
                    'threshold': p['threshold'],
                    'type_src': 'L5_pyramidal'
                }

                # ampa connections
                self.ncfrom_L5Pyr.append(
                    self.parconnect_from_src(
                        gid_src, nc_dict['ampa'], self.apicaloblique_ampa))
                self.ncfrom_L5Pyr.append(
                    self.parconnect_from_src(
                        gid_src, nc_dict['ampa'], self.basal2_ampa))
                self.ncfrom_L5Pyr.append(
                    self.parconnect_from_src(
                        gid_src, nc_dict['ampa'], self.basal3_ampa))

                nc_dict['nmda'] = {
                    'pos_src': pos,
                    'A_weight': p['gbar_L5Pyr_L5Pyr_nmda'],
                    'A_delay': 1.,
                    'lamtha': 3.,
                    'threshold': p['threshold'],
                    'type_src': 'L5_pyramidal'
                }

                # nmda connections
                self.ncfrom_L5Pyr.append(
                    self.parconnect_from_src(gid_src, nc_dict['nmda'],
                                             self.apicaloblique_nmda))
                self.ncfrom_L5Pyr.append(
                    self.parconnect_from_src(gid_src, nc_dict['nmda'],
                                             self.basal2_nmda))
                self.ncfrom_L5Pyr.append(
                    self.parconnect_from_src(
                        gid_src, nc_dict['nmda'], self.basal3_nmda))

        # connections FROM L5Basket TO here
        for gid_src, pos in zip(gid_dict['L5_basket'], pos_dict['L5_basket']):
            nc_dict['gabaa'] = {
                'pos_src': pos,
                'A_weight': p['gbar_L5Basket_L5Pyr_gabaa'],
                'A_delay': 1.,
                'lamtha': 70.,
                'threshold': p['threshold'],
                'type_src': 'L5_basket'
            }

            nc_dict['gabab'] = {
                'pos_src': pos,
                'A_weight': p['gbar_L5Basket_L5Pyr_gabab'],
                'A_delay': 1.,
                'lamtha': 70.,
                'threshold': p['threshold'],
                'type_src': 'L5_basket'
            }

            # soma synapses are defined in Pyr()
            self.ncfrom_L5Basket.append(
                self.parconnect_from_src(gid_src, nc_dict['gabaa'],
                                         self.synapses['soma_gabaa']))
            self.ncfrom_L5Basket.append(
                self.parconnect_from_src(gid_src, nc_dict['gabab'],
                                         self.synapses['soma_gabab']))

        # connections FROM L2Pyr TO here
        for gid_src, pos in zip(gid_dict['L2_pyramidal'],
                                pos_dict['L2_pyramidal']):
            # this delay is longer than most
            nc_dict = {
                'pos_src': pos,
                'A_weight': p['gbar_L2Pyr_L5Pyr'],
                'A_delay': 1.,
                'lamtha': 3.,
                'threshold': p['threshold'],
                'type_src': 'L2_pyramidal'
            }

            self.ncfrom_L2Pyr.append(
                self.parconnect_from_src(
                    gid_src, nc_dict, self.basal2_ampa))
            self.ncfrom_L2Pyr.append(
                self.parconnect_from_src(
                    gid_src, nc_dict, self.basal3_ampa))
            self.ncfrom_L2Pyr.append(
                self.parconnect_from_src(
                    gid_src, nc_dict, self.apicaltuft_ampa))
            self.ncfrom_L2Pyr.append(
                self.parconnect_from_src(
                    gid_src, nc_dict, self.apicaloblique_ampa))

        # connections FROM L2Basket TO here
        for gid_src, pos in zip(gid_dict['L2_basket'], pos_dict['L2_basket']):
            nc_dict = {
                'pos_src': pos,
                'A_weight': p['gbar_L2Basket_L5Pyr'],
                'A_delay': 1.,
                'lamtha': 50.,
                'threshold': p['threshold'],
                'type_src': 'L2_basket'
            }

            self.ncfrom_L2Basket.append(
                self.parconnect_from_src(
                    gid_src, nc_dict, self.apicaltuft_gabaa))

    # receive from external inputs
    def parreceive(self, gid, gid_dict, pos_dict, p_ext):
        for gid_src, p_src, pos in zip(gid_dict['extinput'],
                                       p_ext, pos_dict['extinput']):
            # Check if AMPA params defined in p_src
            if 'L5Pyr_ampa' in p_src.keys():
                nc_dict_ampa = {
                    'pos_src': pos,
                    'A_weight': p_src['L5Pyr_ampa'][0],
                    'A_delay': p_src['L5Pyr_ampa'][1],
                    'lamtha': p_src['lamtha'],
                    'threshold': p_src['threshold'],
                    'type_src': 'ext'
                }

                # Proximal feed AMPA synapses
                if p_src['loc'] is 'proximal':
                    # basal2_ampa, basal3_ampa, apicaloblique_ampa
                    self.ncfrom_extinput.append(
                        self.parconnect_from_src(gid_src, nc_dict_ampa,
                                                 self.basal2_ampa))
                    self.ncfrom_extinput.append(
                        self.parconnect_from_src(gid_src, nc_dict_ampa,
                                                 self.basal3_ampa))
                    self.ncfrom_extinput.append(
                        self.parconnect_from_src(gid_src, nc_dict_ampa,
                                                 self.apicaloblique_ampa))
                # Distal feed AMPA synsapes
                elif p_src['loc'] is 'distal':
                    # apical tuft
                    self.ncfrom_extinput.append(
                        self.parconnect_from_src(gid_src, nc_dict_ampa,
                                                 self.apicaltuft_ampa))

            # Check if NMDA params defined in p_src
            if 'L5Pyr_nmda' in p_src.keys():
                nc_dict_nmda = {
                    'pos_src': pos,
                    'A_weight': p_src['L5Pyr_nmda'][0],
                    'A_delay': p_src['L5Pyr_nmda'][1],
                    'lamtha': p_src['lamtha'],
                    'threshold': p_src['threshold'],
                    'type_src': 'ext'
                }

                # Proximal feed NMDA synapses
                if p_src['loc'] is 'proximal':
                    # basal2_nmda, basal3_nmda, apicaloblique_nmda
                    self.ncfrom_extinput.append(
                        self.parconnect_from_src(
                            gid_src, nc_dict_nmda, self.basal2_nmda))
                    self.ncfrom_extinput.append(
                        self.parconnect_from_src(
                            gid_src, nc_dict_nmda, self.basal3_nmda))
                    self.ncfrom_extinput.append(
                        self.parconnect_from_src(
                            gid_src, nc_dict_nmda, self.apicaloblique_nmda))
                # Distal feed NMDA synsapes
                elif p_src['loc'] is 'distal':
                    # apical tuft
                    self.ncfrom_extinput.append(
                        self.parconnect_from_src(
                            gid_src, nc_dict_nmda, self.apicaltuft_nmda))

    # one parreceive function to handle all types of external parreceives
    # types must be defined explicitly here
    def parreceive_ext(self, type, gid, gid_dict, pos_dict, p_ext):
        if type.startswith(('evprox', 'evdist')):
            if self.celltype in p_ext.keys():
                gid_ev = gid + gid_dict[type][0]

                nc_dict_ampa = {
                    'pos_src': pos_dict[type][gid],
                    'A_weight': p_ext[self.celltype][0],  # index 0 for ampa weight
                    'A_delay': p_ext[self.celltype][2],  # index 2 for delay
                    'lamtha': p_ext['lamtha_space'],
                    'threshold': p_ext['threshold'],
                    'type_src': type
                }

                nc_dict_nmda = {
                    'pos_src': pos_dict[type][gid],
                    'A_weight': p_ext[self.celltype][1],  # index 1 for nmda weight
                    'A_delay': p_ext[self.celltype][2],  # index 2 for delay
                    'lamtha': p_ext['lamtha_space'],
                    'threshold': p_ext['threshold'],
                    'type_src': type
                }

                if p_ext['loc'] is 'proximal':
                    self.ncfrom_ev.append(
                        self.parconnect_from_src(
                            gid_ev, nc_dict_ampa, self.basal2_ampa))
                    self.ncfrom_ev.append(
                        self.parconnect_from_src(
                            gid_ev, nc_dict_ampa, self.basal3_ampa))
                    self.ncfrom_ev.append(
                        self.parconnect_from_src(
                            gid_ev, nc_dict_ampa, self.apicaloblique_ampa))

                    # NEW: note that default/original is 0 nmda weight for these proximal dends
                    self.ncfrom_ev.append(
                        self.parconnect_from_src(
                            gid_ev, nc_dict_nmda, self.basal2_nmda))
                    self.ncfrom_ev.append(
                        self.parconnect_from_src(
                            gid_ev, nc_dict_nmda, self.basal3_nmda))
                    self.ncfrom_ev.append(
                        self.parconnect_from_src(
                            gid_ev, nc_dict_nmda, self.apicaloblique_nmda))

                elif p_ext['loc'] is 'distal':
                    # apical tuft
                    self.ncfrom_ev.append(
                        self.parconnect_from_src(
                            gid_ev, nc_dict_ampa, self.apicaltuft_ampa))
                    self.ncfrom_ev.append(
                        self.parconnect_from_src(
                            gid_ev, nc_dict_nmda, self.apicaltuft_nmda))

        elif type == 'extgauss':
            # gid is this cell's gid
            # gid_dict is the whole dictionary, including the gids of the extgauss
            # pos_dict is also the pos of the extgauss (net origin)
            # p_ext_gauss are the params (strength, etc.)
            # doesn't matter if this doesn't do anything

            # gid shift is based on L2_pyramidal cells NOT L5
            # I recognize this is ugly (hack)
            # gid_shift = gid_dict['extgauss'][0] - gid_dict['L2_pyramidal'][0]
            if 'L5_pyramidal' in p_ext.keys():
                gid_extgauss = gid + gid_dict['extgauss'][0]

                nc_dict = {
                    'pos_src': pos_dict['extgauss'][gid],
                    'A_weight': p_ext['L5_pyramidal'][0], # index 0 for ampa weight
                    'A_delay': p_ext['L5_pyramidal'][2], # index 2 for delay
                    'lamtha': p_ext['lamtha'],
                    'threshold': p_ext['threshold'],
                    'type_src' : type
                }

                self.ncfrom_extgauss.append(
                    self.parconnect_from_src(
                        gid_extgauss, nc_dict, self.basal2_ampa))
                self.ncfrom_extgauss.append(
                    self.parconnect_from_src(
                        gid_extgauss, nc_dict, self.basal3_ampa))
                self.ncfrom_extgauss.append(
                    self.parconnect_from_src(
                        gid_extgauss, nc_dict, self.apicaloblique_ampa))

        elif type == 'extpois':
            if self.celltype in p_ext.keys():
                gid_extpois = gid + gid_dict['extpois'][0]

                nc_dict = {
                    'pos_src': pos_dict['extpois'][gid],
                    'A_weight': p_ext[self.celltype][0], # index 0 for ampa weight
                    'A_delay': p_ext[self.celltype][2], # index 2 for delay
                    'lamtha': p_ext['lamtha_space'],
                    'threshold': p_ext['threshold'],
                    'type_src' : type
                }

                self.ncfrom_extpois.append(
                    self.parconnect_from_src(
                        gid_extpois, nc_dict, self.basal2_ampa))
                self.ncfrom_extpois.append(
                    self.parconnect_from_src(
                        gid_extpois, nc_dict, self.basal3_ampa))
                self.ncfrom_extpois.append(
                    self.parconnect_from_src(
                        gid_extpois, nc_dict, self.apicaloblique_ampa))

                if p_ext[self.celltype][1] > 0.0:
                    nc_dict['A_weight'] = p_ext[self.celltype][1] # index 1 for nmda weight
                    self.ncfrom_extpois.append(
                        self.parconnect_from_src(
                            gid_extpois, nc_dict, self.basal2_nmda))
                    self.ncfrom_extpois.append(
                        self.parconnect_from_src(
                            gid_extpois, nc_dict, self.basal3_nmda))
                    self.ncfrom_extpois.append(
                          self.parconnect_from_src(
                              gid_extpois, nc_dict, self.apicaloblique_nmda))
