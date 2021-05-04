"""Establish class def for general cell features."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

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
    gid : int or None (optional)
        Each cell in a network is uniquely identified by it's "global ID": GID.
        The GID is an integer from 0 to n_cells, or None if the cell is not
        yet attached to a network. Once the GID is set, it cannot be changed.

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
    gid : int
        GID of the cell in a network (or None if not yet assigned)
    """
    def __init__(self, event_times, threshold, gid=None):
        # Convert event times into nrn vector
        self.nrn_eventvec = h.Vector()
        self.nrn_eventvec.from_python(event_times)

        # load eventvec into VecStim object
        self.nrn_vecstim = h.VecStim()
        self.nrn_vecstim.play(self.nrn_eventvec)

        # create the cell and artificial NetCon
        self.nrn_netcon = h.NetCon(self.nrn_vecstim, None)
        self.nrn_netcon.threshold = threshold

        self._gid = None
        if gid is not None:
            self.gid = gid  # use setter method to check input argument gid

    @property
    def gid(self):
        return self._gid

    @gid.setter
    def gid(self, gid):
        if not isinstance(gid, int):
            raise ValueError('gid must be an integer')
        if self._gid is None:
            self._gid = gid
        else:
            raise RuntimeError('Global ID for this cell already assigned!')


class Cell:
    """Create a cell object.

    Parameters
    ----------
    name : str
        The name of the cell.
    pos : tuple
        The (x, y, z) coordinates.
    gid : int or None (optional)
        Each cell in a network is uniquely identified by it's "global ID": GID.
        The GID is an integer from 0 to n_cells, or None if the cell is not
        yet attached to a network. Once the GID is set, it cannot be changed.

    Attributes
    ----------
    pos : list of length 3
        The position of the cell.
    soma : instance of h.Section | None
        The soma of the cell instantiated in Neuron. If None,
        the cell has not yet been built in Neuron.
    dends : dict
        The dendrites. The key is the name of the dendrite
        and the value is an instance of h.Section.
    sections : list of h.Section
        All the sections including soma and dendrites.
    synapses : dict
        The synapses that the cell can use for connections.
    dipole_pp : list of h.Dipole()
        The Dipole objects (see dipole.mod).
    rec_v : h.Vector()
        Recording of somatic voltage. Must be enabled
        by running simulate_dipole(net, record_vsoma=True)
    rec_i : dict
        Contains recording of somatic currents indexed
        by synapse type. (keys are soma_gabaa, soma_gabab etc.)
        Must be enabled by running simulate_dipole(net, record_isoma=True)
    tonic_biases : list of h.IClamp
        The current clamps inserted at each section of the cell
        for tonic biasing inputs.
    gid : int
        GID of the cell in a network (or None if not yet assigned)
    sect_loc : dict of list
        Can have keys 'proximal' or 'distal' each containing
        names of section locations that are proximal or distal.
    """

    def __init__(self, name, pos, gid=None):
        self.name = name
        self.pos = pos
        self.soma = None
        self.dends = dict()
        self.synapses = dict()
        self.sect_loc = dict()
        self.rec_v = h.Vector()
        self.rec_i = dict()
        # insert iclamp
        self.list_IClamp = list()
        self._gid = None
        self.tonic_biases = list()
        if gid is not None:
            self.gid = gid  # use setter method to check input argument gid

    def __repr__(self):
        class_name = self.__class__.__name__
        return f'<{class_name} | gid={self._gid}>'

    @property
    def gid(self):
        return self._gid

    @property
    def sections(self):
        if self.soma is None:
            raise ValueError('Cell has not yet been built in Neuron')
        return [self.soma] + list(self.dends.values())

    @gid.setter
    def gid(self, gid):
        if not isinstance(gid, int):
            raise ValueError('gid must be an integer')
        if self._gid is None:
            self._gid = gid
        else:
            raise RuntimeError('Global ID for this cell already assigned!')

    def _set_biophysics(self, p_secs):
        "Set the biophysics for the default Pyramidal cell."

        # neuron syntax is used to set values for mechanisms
        # sec.gbar_mech = x sets value of gbar for mech to x for all segs
        # in a section. This method is significantly faster than using
        # a for loop to iterate over all segments to set mech values

        # Value depends on distance from the soma. Soma is set as
        # origin by passing cell.soma as a sec argument to h.distance()
        # Then iterate over segment nodes of dendritic sections
        # and set gbar_ar depending on h.distance(seg.x), which returns
        # distance from the soma to this point on the CURRENTLY ACCESSED
        # SECTION!!!
        h.distance(sec=self.soma)
        for sec in self.sections:
            sec_name = sec.name().split('_', 1)[1]
            for mech_name, mech in p_secs[sec_name]['mechs'].items():
                sec.insert(mech_name)
                for attr, val in mech.items():
                    if hasattr(val, '__call__'):
                        sec.push()
                        for seg in sec:
                            setattr(seg, attr, val(h.distance(seg.x)))
                        h.pop_section()
                    else:
                        setattr(sec, attr, val)

    def _create_synapses(self, p_secs, p_syn):
        """Create synapses."""
        for sec_name in p_secs:
            for receptor in p_secs[sec_name]['syns']:
                sec_name_sanitized = sec_name.replace('_', '')
                syn_key = f'{sec_name_sanitized}_{receptor}'
                if sec_name == 'soma':
                    seg = self.soma(0.5)
                else:
                    seg = self.dends[sec_name](0.5)
                self.synapses[syn_key] = self.syn_create(
                    seg, **p_syn[receptor])

    def _create_sections(self, p_secs, topology):
        """Create soma and set geometry.

        Notes
        -----
        By default neuron uses xy plane
        for height and xz plane for depth. This is opposite for model as a
        whole, but convention is followed in this function ease use of gui.
        """
        for sec_name in p_secs:
            if sec_name == 'soma':
                self.soma = h.Section(cell=self, name=self.name + '_soma')
                sec = self.soma
            else:
                self.dends[sec_name] = h.Section(
                    name=self.name + '_' + sec_name)
                sec = self.dends[sec_name]

            h.pt3dclear(sec=sec)
            for pt in p_secs[sec_name]['sec_pts']:
                h.pt3dadd(pt[0], pt[1], pt[2], 1, sec=sec)
            sec.L = p_secs[sec_name]['L']
            sec.diam = p_secs[sec_name]['diam']
            sec.Ra = p_secs[sec_name]['Ra']
            sec.cm = p_secs[sec_name]['cm']

            if sec.L > 100.:  # 100 um
                sec.nseg = int(sec.L / 50.)
                # make dend.nseg odd for all sections
                if not sec.nseg % 2:
                    sec.nseg += 1

        if topology is None:
            topology = list()

        # Connects sections of THIS cell together.
        for connection in topology:
            if connection[0] == 'soma':
                parent_sec = self.soma
            else:
                parent_sec = self.dends[connection[0]]
            if connection[2] == 'soma':
                child_sec = self.soma
            else:
                child_sec = self.dends[connection[2]]
            parent_loc = connection[1]
            child_loc = connection[3]
            child_sec.connect(parent_sec, parent_loc, child_loc)

    def build(self, p_secs, p_syn, topology, sect_loc):
        """Build cell in Neuron.

        Parameters
        ----------
        p_secs : dict
            Dictionary with keys as section name.
            p_secs[sec_name] is a dictionary with keys
            L, diam, Ra, cm, syns and mech.
            syns is a list specifying the synapses at that section.
            The properties of syn are specified in p_syn.
            mech is a dict with keys as the mechanism names. The
            values are dictionaries with properties of the mechanism.
        p_syn : dict of dict
            Keys are name of synaptic mechanism. Each synaptic mechanism
            has keys for parameters of the mechanism, e.g., 'e', 'tau1',
            'tau2'.
        topology : list of list
            The topology of cell sections. Each element is a list of
            4 items in the format
            [parent_sec, parent_loc, child_sec, child_loc] where
            parent_sec and parent_loc are float between 0 and 1
            specifying the location in the section to connect and
            parent_sec and child_sec are names of the connecting
            sections.
        sect_loc : dict of list
            Can have keys 'proximal' or 'distal' each containing
            names of section locations that are proximal or distal.

        Examples
        --------
        p_secs = {
            'soma':
            {
                'L': 39,
                'diam': 20,
                'cm': 0.85,
                'Ra': 200.,
                'sec_pts': [[0, 0, 0], [0, 39., 0]],
                'syns': ['ampa', 'gabaa', 'nmda'],
                'mechs' : {
                    'ca': {
                        'gbar_ca': 60
                    }
                }
            }
        }
        """
        self._create_sections(p_secs, topology)
        self._create_synapses(p_secs, p_syn)
        self._set_biophysics(p_secs)
        self.sect_loc = sect_loc

    def move_to_pos(self):
        """Move cell to position."""
        x0 = self.soma.x3d(0)
        y0 = self.soma.y3d(0)
        z0 = self.soma.z3d(0)
        dx = self.pos[0] * 100 - x0
        dy = self.pos[2] - y0
        dz = self.pos[1] * 100 - z0

        for s in self.sections:
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
        sec_list = h.SectionList()
        sec_list.wholetree(sec=self.soma)
        sec_list = [sec for sec in sec_list]
        for sect in sec_list:
            sect.insert('dipole')
        # Dipole is defined in dipole_pp.mod
        self.dipole_pp = [h.Dipole(1, sec=sect) for sect in sec_list]
        # setting pointers and ztan values
        for sect, dpp in zip(sec_list, self.dipole_pp):
            dpp.ri = h.ri(1, sec=sect)  # assign internal resistance
            # sets pointers in dipole mod file to the correct locations
            dpp._ref_pv = sect(0.99)._ref_v
            dpp._ref_Qtotal = self.dpl_ref
            # gives INTERNAL segments of the section, non-endpoints
            # creating this because need multiple values simultaneously
            pos_all = np.array([seg.x for seg in sect.allseg()])
            # diff in yvals, scaled against the pos np.array. y_long as
            # in longitudinal
            sect_name = sect.name().split('_', 1)[1]
            y_scale = (yscale[sect_name] * sect.L) * pos_all
            # y_long = (h.y3d(1, sec=sect) - h.y3d(0, sec=sect)) * pos
            # diff values calculate length between successive section points
            y_diff = np.diff(y_scale)
            # y_diff = np.diff(y_long)
            # doing range to index multiple values of the same
            # np.array simultaneously
            for idx, pos in enumerate(pos_all[1:-1]):
                # assign the ri value to the dipole
                # ri not defined at 0 and L
                sect(pos).dipole.ri = h.ri(pos, sec=sect)
                # range variable 'dipole'
                # set pointers to previous segment's voltage, with
                # boundary condition
                sect(pos).dipole._ref_pv = sect(pos_all[idx])._ref_v

                # set aggregate pointers
                sect(pos).dipole._ref_Qsum = dpp._ref_Qsum
                sect(pos).dipole._ref_Qtotal = self.dpl_ref
                # add ztan values
                sect(pos).dipole.ztan = y_diff[idx]
            # set the pp dipole's ztan value to the last value from y_diff
            dpp.ztan = y_diff[-1]
        self.dipole = h.Vector().record(self.dpl_ref)

    def create_tonic_bias(self, amplitude, t0, T, loc=0.5):
        """Create tonic bias at the soma.

        Parameters
        ----------
        amplitude : float
            The amplitude of the input.
        t0 : float
            The start time of tonic input (in ms).
        T : float
            The end time of tonic input (in ms).
        loc : float (0 to 1)
            The location of the input in the soma section.
        """
        stim = h.IClamp(self.soma(loc))
        stim.delay = t0
        stim.dur = T - t0
        stim.amp = amplitude
        self.tonic_biases.append(stim)

    def record_soma(self, record_vsoma=False, record_isoma=False):
        """Record current and voltage at soma.

        Parameters
        ----------
        record_vsoma : bool
            Option to record somatic voltages from cells
        record_isoma : bool
            Option to record somatic currents from cells

        """
        # a soma exists at self.soma
        if record_isoma:
            # assumes that self.synapses is a dict that exists
            list_syn_soma = [key for key in self.synapses.keys()
                             if key.startswith('soma_')]
            # matching dict from the list_syn_soma keys
            self.rec_i = dict.fromkeys(list_syn_soma)
            # iterate through keys and record currents appropriately
            for key in self.rec_i:
                self.rec_i[key] = h.Vector()
                self.rec_i[key].record(self.synapses[key]._ref_i)

        if record_vsoma:
            self.rec_v.record(self.soma(0.5)._ref_v)

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
