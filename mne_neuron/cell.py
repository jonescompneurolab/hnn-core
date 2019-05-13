"""Establish class def for general cell features."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import numpy as np
from neuron import h

# global variables, should be node-independent
h("dp_total_L2 = 0.")
h("dp_total_L5 = 0.")  # put here since these variables used in cells

# Units for e: mV
# Units for gbar: S/cm^2


class _Cell(object):
    """Create a cell object.

    Parameters
    ----------
    gid : int
        The cell ID
    soma_props : dict
        The properties of the soma. Must contain
        keys 'L', 'diam', and 'pos'
    """

    def __init__(self, gid, soma_props):
        self.gid = gid
        # make L_soma and diam_soma elements of self
        # Used in shape_change() b/c func clobbers self.soma.L, self.soma.diam
        self.L = soma_props['L']
        self.diam = soma_props['diam']
        self.pos = soma_props['pos']
        # create soma and set geometry
        self.soma = h.Section(cell=self, name=soma_props['name'] + '_soma')
        self.soma.L = soma_props['L']
        self.soma.diam = soma_props['diam']
        self.soma.Ra = soma_props['Ra']
        self.soma.cm = soma_props['cm']
        # variable for the list_IClamp
        self.list_IClamp = None
        # par: create arbitrary lists of connections FROM other cells
        # TO this cell instantiation
        # these lists are allowed to be empty
        # this should be a dict
        self.ncfrom_L2Pyr = []
        self.ncfrom_L2Basket = []
        self.ncfrom_L5Pyr = []
        self.ncfrom_L5Basket = []
        self.ncfrom_extinput = []
        self.ncfrom_extgauss = []
        self.ncfrom_extpois = []
        self.ncfrom_ev = []

    def record_volt_soma(self):
        self.vsoma = h.Vector()
        self.vsoma.record(self.soma(0.5)._ref_v)

    def get_sections(self):
        return [self.soma]

    def get3dinfo(self):
        ls = self.get_sections()
        lx, ly, lz, ldiam = [], [], [], []
        for s in ls:
            for i in range(s.n3d()):
                lx.append(s.x3d(i))
                ly.append(s.y3d(i))
                lz.append(s.z3d(i))
                ldiam.append(s.diam3d(i))
        return lx, ly, lz, ldiam

    # get cell's bounding box
    def getbbox(self):
        lx, ly, lz, ldiam = self.get3dinfo()
        minx, miny, minz = 1e9, 1e9, 1e9
        maxx, maxy, maxz = -1e9, -1e9, -1e9
        for x, y, z in zip(lx, ly, lz):
            minx = min(x, minx)
            miny = min(y, miny)
            minz = min(z, minz)
            maxx = max(x, maxx)
            maxy = max(y, maxy)
            maxz = max(z, maxz)
        return ((minx, maxx), (miny, maxy), (minz, maxz))

    def translate3d(self, dx, dy, dz):
        for s in self.get_sections():
            for i in range(s.n3d()):
                h.pt3dchange(i, s.x3d(i) + dx, s.y3d(i) + dy,
                             s.z3d(i) + dz, s.diam3d(i), sec=s)

    def translateto(self, x, y, z):
        x0 = self.soma.x3d(0)
        y0 = self.soma.y3d(0)
        z0 = self.soma.z3d(0)
        dx = x - x0
        dy = y - y0
        dz = z - z0
        # print('dx:',dx,'dy:',dy,'dz:',dz)
        self.translate3d(dx, dy, dz)

    def movetopos(self):
        self.translateto(self.pos[0] * 100, self.pos[2], self.pos[1] * 100)

    # two things need to happen here for h:
    # 1. dipole needs to be inserted into each section
    # 2. a list needs to be created with a Dipole (Point Process) in each
    #    section at position 1
    # In Cell() and not Pyr() for future possibilities
    def dipole_insert(self, yscale):
        # insert dipole into each section of this cell
        # dends must have already been created!!
        # it's easier to use wholetree here, this includes soma
        seclist = h.SectionList()
        seclist.wholetree(sec=self.soma)
        # create a python section list list_all
        self.list_all = [sec for sec in seclist]
        for sect in self.list_all:
            sect.insert('dipole')
        # Dipole is defined in dipole_pp.mod
        self.dipole_pp = [h.Dipole(1, sec=sect) for sect in self.list_all]
        # setting pointers and ztan values
        for sect, dpp in zip(self.list_all, self.dipole_pp):
            # assign internal resistance values to dipole point process (dpp)
            dpp.ri = h.ri(1, sec=sect)
            # sets pointers in dipole mod file to the correct locations
            # h.setpointer(ref, ptr, obj)
            h.setpointer(sect(0.99)._ref_v, 'pv', dpp)
            if self.celltype.startswith('L2'):
                h.setpointer(h._ref_dp_total_L2, 'Qtotal', dpp)
            elif self.celltype.startswith('L5'):
                h.setpointer(h._ref_dp_total_L5, 'Qtotal', dpp)
            # gives INTERNAL segments of the section, non-endpoints
            # creating this because need multiple values simultaneously
            loc = np.array([seg.x for seg in sect])
            # these are the positions, including 0 but not L
            pos = np.array([seg.x for seg in sect.allseg()])
            # diff in yvals, scaled against the pos np.array. y_long as
            # in longitudinal
            y_scale = (yscale[sect.name()] * sect.L) * pos
            # y_long = (h.y3d(1, sec=sect) - h.y3d(0, sec=sect)) * pos
            # diff values calculate length between successive section points
            y_diff = np.diff(y_scale)
            # y_diff = np.diff(y_long)
            # doing range to index multiple values of the same
            # np.array simultaneously
            for i in range(len(loc)):
                # assign the ri value to the dipole
                sect(loc[i]).dipole.ri = h.ri(loc[i], sec=sect)
                # range variable 'dipole'
                # set pointers to previous segment's voltage, with
                # boundary condition
                if i:
                    h.setpointer(sect(loc[i - 1])._ref_v,
                                 'pv', sect(loc[i]).dipole)
                else:
                    h.setpointer(sect(0)._ref_v, 'pv', sect(loc[i]).dipole)
                # set aggregate pointers
                h.setpointer(dpp._ref_Qsum, 'Qsum', sect(loc[i]).dipole)
                if self.celltype.startswith('L2'):
                    h.setpointer(h._ref_dp_total_L2, 'Qtotal',
                                 sect(loc[i]).dipole)
                elif self.celltype.startswith('L5'):
                    h.setpointer(h._ref_dp_total_L5, 'Qtotal',
                                 sect(loc[i]).dipole)
                # add ztan values
                sect(loc[i]).dipole.ztan = y_diff[i]
            # set the pp dipole's ztan value to the last value from y_diff
            dpp.ztan = y_diff[-1]

    # Add IClamp to a segment
    def insert_IClamp(self, sect_name, props_IClamp):
        # def insert_iclamp(self, sect_name, seg_loc, tstart, tstop, weight):
        # gather list of all sections
        seclist = h.SectionList()
        seclist.wholetree(sec=self.soma)
        # find specified sect in section list, insert IClamp, set props
        for sect in seclist:
            if sect_name in sect.name():
                stim = h.IClamp(sect(props_IClamp['loc']))
                stim.delay = props_IClamp['delay']
                stim.dur = props_IClamp['dur']
                stim.amp = props_IClamp['amp']
                # stim.dur = tstop - tstart
                # stim = h.IClamp(sect(seg_loc))
        # object must exist for NEURON somewhere and needs to be saved
        return stim

    # simple function to record current
    # for now only at the soma
    def record_current_soma(self):
        # a soma exists at self.soma
        self.rec_i = h.Vector()
        try:
            # assumes that self.synapses is a dict that exists
            list_syn_soma = [key for key in self.synapses.keys()
                             if key.startswith('soma_')]
            # matching dict from the list_syn_soma keys
            self.dict_currents = dict.fromkeys(list_syn_soma)
            # iterate through keys and record currents appropriately
            for key in self.dict_currents:
                self.dict_currents[key] = h.Vector()
                self.dict_currents[key].record(self.synapses[key]._ref_i)
        except:
            print(
                "Warning in Cell(): record_current_soma() was called,"
                " but no self.synapses dict was found")
            pass

    # General fn that creates any Exp2Syn synapse type
    # requires dictionary of synapse properties
    def syn_create(self, secloc, p):
        """Create an h.Exp2Syn synapse.

        Parameters
        ----------
        p : dict
            Should contain keys
                - 'e' (reverse potential)
                - 'tau1' (rise time)
                - 'tau2' (decay time)

        Returns
        -------
        syn : instance of h.Exp2Syn
            A two state kinetic scheme synapse.
        """
        syn = h.Exp2Syn(secloc)
        syn.e = p['e']
        syn.tau1 = p['tau1']
        syn.tau2 = p['tau2']
        return syn

    # For all synapses, section location 'secloc' is being explicitly supplied
    # for clarity, even though they are (right now) always 0.5.
    # Might change in future
    # creates a RECEIVING inhibitory synapse at secloc
    def syn_gabaa_create(self, secloc):
        syn_gabaa = h.Exp2Syn(secloc)
        syn_gabaa.e = -80
        syn_gabaa.tau1 = 0.5
        syn_gabaa.tau2 = 5.
        return syn_gabaa

    # creates a RECEIVING slow inhibitory synapse at secloc
    # called: self.soma_gabab = syn_gabab_create(self.soma(0.5))
    def syn_gabab_create(self, secloc):
        syn_gabab = h.Exp2Syn(secloc)
        syn_gabab.e = -80
        syn_gabab.tau1 = 1
        syn_gabab.tau2 = 20.
        return syn_gabab

    # creates a RECEIVING excitatory synapse at secloc
    # def syn_ampa_create(self, secloc, tau_decay, prng_obj):
    def syn_ampa_create(self, secloc):
        syn_ampa = h.Exp2Syn(secloc)
        syn_ampa.e = 0.
        syn_ampa.tau1 = 0.5
        syn_ampa.tau2 = 5.
        return syn_ampa

    # creates a RECEIVING nmda synapse at secloc
    # this is a pretty fast NMDA, no?
    def syn_nmda_create(self, secloc):
        syn_nmda = h.Exp2Syn(secloc)
        syn_nmda.e = 0.
        syn_nmda.tau1 = 1.
        syn_nmda.tau2 = 20.
        return syn_nmda

    # connect_to_target created for pc, used in Network()
    # these are SOURCES of spikes
    def connect_to_target(self, target, threshold):
        nc = h.NetCon(self.soma(0.5)._ref_v, target, sec=self.soma)
        nc.threshold = threshold
        return nc

    def parconnect_from_src(self, gid_presyn, nc_dict, postsyn):
        """Parallel receptor-centric connect FROM presyn TO this cell,
           based on GID.

        Parameters
        ----------
        gid_presyn : int
            The cell ID of the presynaptic neuron
        nc_dict : dict
            Dictionary with keys: pos_src, A_weight, A_delay, lamtha
            Defines the connection parameters
        postsyn : str
            The postsynaptic cell object.

        Returns
        -------
        nc : instance of h.NetCon
            A network connection object.
        """
        from .parallel import pc

        nc = pc.gid_connect(gid_presyn, postsyn)
        # calculate distance between cell positions with pardistance()
        d = self.__pardistance(nc_dict['pos_src'])
        # set props here
        nc.threshold = nc_dict['threshold']
        nc.weight[0] = nc_dict['A_weight'] * \
            np.exp(-(d**2) / (nc_dict['lamtha']**2))
        nc.delay = nc_dict['A_delay'] / \
            (np.exp(-(d**2) / (nc_dict['lamtha']**2)))

        return nc

    # pardistance function requires pre position, since it is
    # calculated on POST cell
    def __pardistance(self, pos_pre):
        dx = self.pos[0] - pos_pre[0]
        dy = self.pos[1] - pos_pre[1]
        return np.sqrt(dx**2 + dy**2)

    # Define 3D shape of soma -- is needed for gui representation of cell
    # DO NOT need to call h.define_shape() explicitly!!
    def shape_soma(self):
        h.pt3dclear(sec=self.soma)
        # h.ptdadd(x, y, z, diam) -- if this function is run, clobbers
        # self.soma.diam set above
        h.pt3dadd(0, 0, 0, self.diam, sec=self.soma)
        h.pt3dadd(0, self.L, 0, self.diam, sec=self.soma)

    # insert IClamps in all situations
    def create_all_IClamp(self, p):
        """ temporarily an external function taking the p dict
        """
        # list of sections for this celltype
        sect_list_IClamp = [
            'soma',
        ]

        name_key = self.name
        if 'Pyr' in self.name:
            name_key = '%s_soma' % self.name
        # some parameters
        t_delay = p['Itonic_t0_%s' % name_key]

        # T = -1 means use h.tstop
        if p['Itonic_T_%s' % name_key] == -1:
            t_dur = p['tstop'] - t_delay

        else:
            t_dur = p['Itonic_T_%s' % name_key] - t_delay

        # t_dur must be nonnegative, I imagine
        if t_dur < 0.:
            t_dur = 0.

        # properties of the IClamp
        props_IClamp = {
            'loc': 0.5,
            'delay': t_delay,
            'dur': t_dur,
            'amp': p['Itonic_A_%s' % name_key]
        }

        # iterate through list of sect_list_IClamp to create a persistent
        # IClamp object
        # the insert_IClamp procedure is in Cell() and checks on names
        # so names must be actual section names, or else it will fail silently
        self.list_IClamp = [self.insert_IClamp(
            sect_name, props_IClamp) for sect_name in sect_list_IClamp]
