"""Establish class def for general cell features."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Sam Neymotin <samnemo@gmail.com>

import numpy as np
from numpy.linalg import norm

from neuron import h, nrn

from .viz import plot_cell_morphology

# Units for e: mV
# Units for gbar: S/cm^2


def _get_cos_theta(p_secs, sec_name_apical):
    """Get cos(theta) to compute dipole along the apical dendrite."""
    a = (np.array(p_secs[sec_name_apical]['sec_pts'][1]) -
         np.array(p_secs[sec_name_apical]['sec_pts'][0]))
    cos_thetas = dict()
    for sec_name, p_sec in p_secs.items():
        b = np.array(p_sec['sec_pts'][1]) - np.array(p_sec['sec_pts'][0])
        cos_thetas[sec_name] = np.dot(a, b) / (norm(a) * norm(b))
    return cos_thetas


def _calculate_gaussian(x_val, height, lamtha):
    """Return height of gaussian at x_val.

    Parameters
    ----------
    x_val : float
        Value on x-axis to query height of gaussian curve.
    height : float
        Height of the gaussian curve at zero.
    lamtha : float
        Space constant.

    Returns
    -------
    x_height : float
        Height of gaussian at x_val.

    Notes
    -----
    Gaussian curve is centered at zero and has a fixed peak height
    such the _calculate_gaussian(0, lamtha) returns 1 for all lamtha.
    """
    x_height = height * np.exp(-(x_val**2) / (lamtha**2))

    return x_height


def _get_gaussian_connection(src_pos, target_pos, nc_dict):
    """Calculate distance dependent connection properties.

    Parameters
    ----------
    src_pos : float
        Position of source cell.
    target_pos : float
        Position of target cell.
    nc_dict : dict
        Dictionary with keys: pos_src, A_weight, A_delay, lamtha
        Defines the connection parameters

    Returns
    -------
    weight : float
        Weight of the synaptic connection.
    delay : float
        Delay of synaptic connection.

    Notes
    -----
    Distance in xy plane is used for gaussian decay.
    """
    x_dist = target_pos[0] - src_pos[0]
    y_dist = target_pos[1] - src_pos[1]
    cell_dist = np.sqrt(x_dist**2 + y_dist**2)

    weight = _calculate_gaussian(
        cell_dist, nc_dict['A_weight'], nc_dict['lamtha'])
    delay = nc_dict['A_delay'] / _calculate_gaussian(
        cell_dist, 1, nc_dict['lamtha'])
    return weight, delay


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
    p_secs : nested dict
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
    gid : int or None (optional)
        Each cell in a network is uniquely identified by it's "global ID": GID.
        The GID is an integer from 0 to n_cells, or None if the cell is not
        yet attached to a network. Once the GID is set, it cannot be changed.

    Attributes
    ----------
    pos : list of length 3
        The position of the cell.
    sections : dict
        The sections. The key is the name of the section
        and the value is an instance of h.Section.
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

    Examples
    --------
    >>> p_secs = {
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

    def __init__(self, name, pos, p_secs, p_syn, topology, sect_loc, gid=None):
        self.name = name
        self.pos = pos
        self.p_secs = p_secs
        self.p_syn = p_syn
        self.topology = topology
        self.sect_loc = sect_loc
        self.sections = dict()
        self.synapses = dict()
        self.dipole_pp = list()
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

    @gid.setter
    def gid(self, gid):
        if not isinstance(gid, int):
            raise ValueError('gid must be an integer')
        if self._gid is None:
            self._gid = gid
        else:
            raise RuntimeError('Global ID for this cell already assigned!')

    def _set_biophysics(self, p_secs):
        "Set the biophysics for the cell."

        # neuron syntax is used to set values for mechanisms
        # sec.gbar_mech = x sets value of gbar for mech to x for all segs
        # in a section. This method is significantly faster than using
        # a for loop to iterate over all segments to set mech values

        # If value depends on distance from the soma. Soma is set as
        # origin by passing cell.soma as a sec argument to h.distance()
        # Then iterate over segment nodes of dendritic sections
        # and set attribute depending on h.distance(seg.x), which returns
        # distance from the soma to this point on the CURRENTLY ACCESSED
        # SECTION!!!
        h.distance(sec=self.sections['soma'])
        for sec_name, p_sec in p_secs.items():
            sec = self.sections[sec_name]
            for mech_name, p_mech in p_sec['mechs'].items():
                sec.insert(mech_name)
                for attr, val in p_mech.items():
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
                syn_key = f'{sec_name}_{receptor}'
                seg = self.sections[sec_name](0.5)
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
        # shift cell to self.pos and reorient apical dendrite
        # along z direction of self.pos
        dx = self.pos[0] - self.p_secs['soma']['sec_pts'][0][0]
        dy = self.pos[1] - self.p_secs['soma']['sec_pts'][0][1]
        dz = self.pos[2] - self.p_secs['soma']['sec_pts'][0][2]

        for sec_name in p_secs:
            sec = h.Section(name=f'{self.name}_{sec_name}')
            self.sections[sec_name] = sec

            h.pt3dclear(sec=sec)
            for pt in p_secs[sec_name]['sec_pts']:
                h.pt3dadd(pt[0] + dx,
                          pt[1] + dy,
                          pt[2] + dz, 1, sec=sec)
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
            parent_sec = self.sections[connection[0]]
            child_sec = self.sections[connection[2]]
            parent_loc = connection[1]
            child_loc = connection[3]
            child_sec.connect(parent_sec, parent_loc, child_loc)

    def build(self, sec_name_apical=None):
        """Build cell in Neuron and insert dipole if applicable.

        Parameters
        ----------
        sec_name_apical : str | None
            If not None, a dipole will be inserted in this cell in alignment
            with this section. The section should belong to the apical dendrite
            of a pyramidal neuron.
        """
        if 'soma' not in self.p_secs:
            raise KeyError('soma must be defined for cell')
        self._create_sections(self.p_secs, self.topology)
        self._create_synapses(self.p_secs, self.p_syn)
        self._set_biophysics(self.p_secs)
        if sec_name_apical in self.sections:
            self._insert_dipole(sec_name_apical)
        elif sec_name_apical is not None:
            raise ValueError(f'sec_name_apical must be an existing '
                             f'section of the current cell or None. '
                             f'Got {sec_name_apical}.')

    # two things need to happen here for h:
    # 1. dipole needs to be inserted into each section
    # 2. a list needs to be created with a Dipole (Point Process) in each
    #    section at position 1
    # In Cell() and not Pyr() for future possibilities
    def _insert_dipole(self, sec_name_apical):
        """Insert dipole into each section of this cell.

        Parameters
        ----------
        sec_name_apical : str
            The name of the section along which dipole moment is calculated.
        """
        self.dpl_vec = h.Vector(1)
        self.dpl_ref = self.dpl_vec._ref_x[0]
        cos_thetas = _get_cos_theta(self.p_secs, 'apical_trunk')

        # setting pointers and ztan values
        for sect_name in self.p_secs:
            sect = self.sections[sect_name]
            sect.insert('dipole')

            dpp = h.Dipole(1, sec=sect)  # defined in dipole_pp.mod
            self.dipole_pp.append(dpp)
            dpp.ri = h.ri(1, sec=sect)  # assign internal resistance
            # sets pointers in dipole mod file to the correct locations
            dpp._ref_pv = sect(0.99)._ref_v
            dpp._ref_Qtotal = self.dpl_ref
            # gives INTERNAL segments of the section, non-endpoints
            # creating this because need multiple values simultaneously
            pos_all = np.array([seg.x for seg in sect.allseg()])
            seg_lens = np.diff(pos_all) * sect.L
            seg_lens_z = seg_lens * cos_thetas[sect_name]

            # alternative procedure below with y_long(itudinal)
            # y_long = (h.y3d(1, sec=sect) - h.y3d(0, sec=sect)) * pos
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
                sect(pos).dipole.ztan = seg_lens_z[idx]
            # set the pp dipole's ztan value to the last value from seg_lens_z
            dpp.ztan = seg_lens_z[-1]
        self.dipole = h.Vector().record(self.dpl_ref)

    def create_tonic_bias(self, amplitude, t0, tstop, loc=0.5):
        """Create tonic bias at the soma.

        Parameters
        ----------
        amplitude : float
            The amplitude of the input.
        t0 : float
            The start time of tonic input (in ms).
        tstop : float
            The end time of tonic input (in ms).
        loc : float (0 to 1)
            The location of the input in the soma section.
        """
        stim = h.IClamp(self.sections['soma'](loc))
        stim.delay = t0
        stim.dur = tstop - t0
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
        # a soma exists at self.sections['soma']
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
            self.rec_v.record(self.sections['soma'](0.5)._ref_v)

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
        nc = h.NetCon(self.sections['soma'](0.5)._ref_v, None,
                      sec=self.sections['soma'])
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

        # set props here.
        nc.threshold = nc_dict['threshold']
        nc.weight[0], nc.delay = _get_gaussian_connection(
            nc_dict['pos_src'], self.pos, nc_dict)

        return nc

    def plot_morphology(self, ax=None, cell_types=None, show=True):
        """Plot the cell morphology.

        Parameters
        ----------
        ax : instance of Axes3D
            Matplotlib 3D axis
        show : bool
            If True, show the plot

        Returns
        -------
        axes : instance of Axes3D
            The matplotlib 3D axis handle.
        """
        return plot_cell_morphology(self, ax=ax, show=show)
