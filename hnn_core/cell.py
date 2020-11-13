"""Establish class def for general cell features."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

from abc import ABC, abstractmethod

import numpy as np
from neuron import h, nrn

# Units for e: mV
# Units for gbar: S/cm^2


class _ArtificialCell:
    """The ArtificialCell class for initializing a NEURON feed source.

    Parameters
    ----------
    event_times : list
        Spike times associated with a single feed source (i.e.,
        associated with a unique gid).
    threshold : float
        Membrane potential threshold that demarks a spike.

    Attributes
    ----------
    nrn_eventvec : instance of h.Vector()
        NEURON h.Vector() object of event times.
    nrn_vecstim : instance of h.VecStim()
        NEURON h.VecStim() object of spike sources created
        from nrn_eventvec.
    nrn_netcon : instance of h.NetCon()
        NEURON h.NetCon() object that creates the spike
        source-to-target references for nrn_vecstim.
    """
    def __init__(self, event_times, threshold):
        # Convert event times into nrn vector
        self.nrn_eventvec = h.Vector()
        self.nrn_eventvec.from_python(event_times)

        # load eventvec into VecStim object
        self.nrn_vecstim = h.VecStim()
        self.nrn_vecstim.play(self.nrn_eventvec)

        # create the cell and artificial NetCon
        self.nrn_netcon = h.NetCon(self.nrn_vecstim, None)
        self.nrn_netcon.threshold = threshold


class _Cell(ABC):
    """Create a cell object.

    Parameters
    ----------
    gid : int
        The cell ID
    soma_props : dict
        The properties of the soma. Must contain
        keys 'L', 'diam', and 'pos'
    record_vsoma : bool
        Option to record somatic voltages from cells

    Attributes
    ----------
    pos : list of length 3
        The position of the cell.
    dipole_pp : list of h.Dipole()
        The Dipole objects (see dipole.mod).
    dict_currents : dict of h.Vector()
        The soma currents (keys are soma_gabaa, soma_gabab etc.)
    """

    def __init__(self, gid, soma_props, record_vsoma=False):
        self.gid = gid
        # variable for the list_IClamp
        self.list_IClamp = None
        self.soma_props = soma_props
        self.create_soma()
        self.record_vsoma = record_vsoma

    def __repr__(self):
        class_name = self.__class__.__name__
        soma_props = self.soma_props
        s = ('soma: L %f, diam %f, Ra %f, cm %f' %
             (soma_props['L'], soma_props['diam'],
              soma_props['Ra'], soma_props['cm']))
        return '<%s | %s>' % (class_name, s)

    @abstractmethod
    def get_sections(self):
        """Get sections in a cell."""
        pass

    def create_soma(self):
        """Create soma and set geometry."""
        # make L_soma and diam_soma elements of self
        # Used in shape_change() b/c func clobbers self.soma.L, self.soma.diam
        soma_props = self.soma_props

        self.L = soma_props['L']
        self.diam = soma_props['diam']
        self.pos = soma_props['pos']

        self.soma = h.Section(cell=self, name=soma_props['name'] + '_soma')
        self.soma.L = soma_props['L']
        self.soma.diam = soma_props['diam']
        self.soma.Ra = soma_props['Ra']
        self.soma.cm = soma_props['cm']

    def move_to_pos(self):
        """Move cell to position."""
        x0 = self.soma.x3d(0)
        y0 = self.soma.y3d(0)
        z0 = self.soma.z3d(0)
        dx = self.pos[0] * 100 - x0
        dy = self.pos[2] - y0
        dz = self.pos[1] * 100 - z0

        for s in self.get_sections():
            for i in range(s.n3d()):
                h.pt3dchange(i, s.x3d(i) + dx, s.y3d(i) + dy,
                             s.z3d(i) + dz, s.diam3d(i), sec=s)

    # two things need to happen here for h:
    # 1. dipole needs to be inserted into each section
    # 2. a list needs to be created with a Dipole (Point Process) in each
    #    section at position 1
    # In Cell() and not Pyr() for future possibilities
    def insert_dipole(self, yscale):
        """Insert dipole into each section of this cell.

        Parameters
        ----------
        yscale : dict
            Dictionary of length scales to calculate dipole without
            3d shape.
        """
        self.dpl_vec = h.Vector(1)
        self.dpl_ref = self.dpl_vec._ref_x[0]

        # dends must have already been created!!
        # it's easier to use wholetree here, this includes soma
        seclist = h.SectionList()
        seclist.wholetree(sec=self.soma)
        # create a python section list list_all
        list_all = [sec for sec in seclist]
        for sect in list_all:
            sect.insert('dipole')
        # Dipole is defined in dipole_pp.mod
        self.dipole_pp = [h.Dipole(1, sec=sect) for sect in list_all]
        # setting pointers and ztan values
        for sect, dpp in zip(list_all, self.dipole_pp):
            # assign internal resistance values to dipole point process (dpp)
            dpp.ri = h.ri(1, sec=sect)
            # sets pointers in dipole mod file to the correct locations
            # h.setpointer(ref, ptr, obj)
            h.setpointer(sect(0.99)._ref_v, 'pv', dpp)
            h.setpointer(self.dpl_ref, 'Qtotal', dpp)
            # gives INTERNAL segments of the section, non-endpoints
            # creating this because need multiple values simultaneously
            loc = np.array([seg.x for seg in sect])
            # these are the positions, including 0 but not L
            pos = np.array([seg.x for seg in sect.allseg()])
            # diff in yvals, scaled against the pos np.array. y_long as
            # in longitudinal
            y_scale = (yscale[sect.name().split('_', 1)[1]] * sect.L) * pos
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
                if i > 0:
                    h.setpointer(sect(loc[i - 1])._ref_v,
                                 'pv', sect(loc[i]).dipole)
                else:
                    h.setpointer(sect(0)._ref_v, 'pv', sect(loc[i]).dipole)
                # set aggregate pointers
                h.setpointer(dpp._ref_Qsum, 'Qsum', sect(loc[i]).dipole)
                h.setpointer(self.dpl_ref, 'Qtotal', sect(loc[i]).dipole)
                # add ztan values
                sect(loc[i]).dipole.ztan = y_diff[i]
            # set the pp dipole's ztan value to the last value from y_diff
            dpp.ztan = y_diff[-1]
        self.dipole = h.Vector().record(self.dpl_ref)

    def record_current_soma(self):
        """Record current at soma."""
        # a soma exists at self.soma
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

    def record_voltage_soma(self):
        """Record current at soma."""
        if self.record_vsoma:
            self.rec_v = h.Vector().record(self.soma(0.5)._ref_v)
        else:
            self.rec_v = h.Vector()

    def syn_create(self, secloc, e, tau1, tau2):
        """Create an h.Exp2Syn synapse.

        Parameters
        ----------
        secloc : instance of nrn.Segment
            The section location. E.g., soma(0.5).
        e: float
            Reverse potential (in mV)
        tau1: float
            Rise time (in ms)
        tau2: float
            Decay time (in ms)

        Returns
        -------
        syn : instance of h.Exp2Syn
            A two state kinetic scheme synapse.
        """
        if not isinstance(secloc, nrn.Segment):
            raise TypeError(f'secloc must be instance of'
                            f'nrn.Segment. Got {type(secloc)}')
        syn = h.Exp2Syn(secloc)
        syn.e = e
        syn.tau1 = tau1
        syn.tau2 = tau2
        return syn

    def setup_source_netcon(self, threshold):
        """Created for _PC.cell and specifies SOURCES of spikes.

        Parameters
        ----------
        threshold : float
            The voltage threshold for action potential.
        """
        nc = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
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
        postsyn : instance of h.Exp2Syn
            The postsynaptic cell object.

        Returns
        -------
        nc : instance of h.NetCon
            A network connection object.
        """
        from .network_builder import _PC

        nc = _PC.gid_connect(gid_presyn, postsyn)
        # calculate distance between cell positions with pardistance()
        d = self._pardistance(nc_dict['pos_src'])
        # set props here
        nc.threshold = nc_dict['threshold']
        nc.weight[0] = nc_dict['A_weight'] * \
            np.exp(-(d**2) / (nc_dict['lamtha']**2))
        nc.delay = nc_dict['A_delay'] / \
            (np.exp(-(d**2) / (nc_dict['lamtha']**2)))

        return nc

    # pardistance function requires pre position, since it is
    # calculated on POST cell
    def _pardistance(self, pos_pre):
        dx = self.pos[0] - pos_pre[0]
        dy = self.pos[1] - pos_pre[1]
        return np.sqrt(dx**2 + dy**2)

    def shape_soma(self):
        """Define 3D shape of soma.

        .. warning:: needed for gui representation of cell
                     DO NOT need to call h.define_shape() explicitly!
        """
        h.pt3dclear(sec=self.soma)
        # h.ptdadd(x, y, z, diam) -- if this function is run, clobbers
        # self.soma.diam set above
        h.pt3dadd(0, 0, 0, self.diam, sec=self.soma)
        h.pt3dadd(0, self.L, 0, self.diam, sec=self.soma)
