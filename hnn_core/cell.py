"""Establish class def for general cell features."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Sam Neymotin <samnemo@gmail.com>

from copy import deepcopy

import numpy as np
from numpy.linalg import norm

from neuron import h, nrn

from .viz import plot_cell_morphology
from .externals.mne import _validate_type, _check_option

# Units for e: mV
# Units for gbar: S/cm^2


def _get_cos_theta(sections, sec_name_apical):
    """Get cos(theta) to compute dipole along the apical dendrite."""
    a = (np.array(sections[sec_name_apical].end_pts[1]) -
         np.array(sections[sec_name_apical].end_pts[0]))
    cos_thetas = dict()
    for sec_name, section in sections.items():
        b = np.array(section.end_pts[1]) - np.array(section.end_pts[0])
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


def _get_gaussian_connection(src_pos, target_pos, nc_dict,
                             inplane_distance=1.):
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
    inplane_distance : float
        The in plane-distance (in um) between pyramidal cell somas in the
        square grid. Default: 1.0 um.

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
    scaled_lamtha = nc_dict['lamtha'] * inplane_distance

    weight = _calculate_gaussian(
        cell_dist, nc_dict['A_weight'], scaled_lamtha)
    delay = nc_dict['A_delay'] / _calculate_gaussian(
        cell_dist, 1, scaled_lamtha)
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


class Section:
    """Section class.

    Parameters
    ----------
    L : float
        length of a section in microns.
    diam : float
        diameter of a section in microns.
    cm : float
        membrane capacitance in micro-Farads.
    Ra : float
        axial resistivity in ohm-cm
    end_pts : list of [x, y, z]
        The start and stop points of the section.

    Attributes
    ----------
    mechs : dict
        Mechanisms to insert in this section. The keys
        are the names of the mechanisms and values
        are the properties. For e.g., {'ca': {'gbar_ca': 60}}
    syns : list of str
        The synaptic mechanisms to add in this section
    end_pts : list of [x, y, z]
        The start and stop points of the section. Cannot be changed.
    L : float
        length of a section in microns.
    diam : float
        diameter of a section in microns.
    cm : float
        membrane capacitance in micro-Farads.
    Ra : float
        axial resistivity in ohm-cm.
    """
    def __init__(self, L, diam, Ra, cm, end_pts=None):

        self._L = L
        self._diam = diam
        self._Ra = Ra
        self._cm = cm
        if end_pts is None:
            end_pts = list()
        self._end_pts = end_pts

        self.mechs = dict()
        self.syns = list()

    def __repr__(self):
        return f'L={self.L}, diam={self.diam}, cm={self.cm}, Ra={self.Ra}'

    @property
    def L(self):
        return self._L

    @property
    def diam(self):
        return self._diam

    @property
    def cm(self):
        return self._cm

    @property
    def Ra(self):
        return self._Ra

    @property
    def end_pts(self):
        return self._end_pts


class Cell:
    """Create a cell object.

    Parameters
    ----------
    name : str
        The name of the cell.
    pos : tuple
        The (x, y, z) coordinates.
    sections : dict of Section
        Dictionary with keys as section name.
    synapses : dict of dict
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
    sections : nested dict
        The section parameters. The key is the name of the section
        and the value is a dictionary parametrizing the morphology
        of the section and the mechanisms inserted.
    synapses : dict
        The synapses that the cell can use for connections.
    dipole_pp : list of h.Dipole()
        The Dipole objects (see dipole.mod).
    rec_vsec : dict
        Recording of somatic voltage. Must be enabled
        by running simulate_dipole(net, record_vsoma=True)
    rec_isec : dict
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
    >>> section_soma = Section(
            L=39,
            diam=20,
            cm=0.85,
            Ra=200.,
            end_pts=[[0, 0, 0], [0, 39., 0]]
        )
    """

    def __init__(self, name, pos, sections, synapses, topology, sect_loc,
                 gid=None):
        self.name = name
        self.pos = pos
        for section in sections.values():
            if not isinstance(section, Section):
                raise ValueError(f'Items in section must be instances'
                                 f' of Section. Got {type(section)}')
        self.sections = sections
        self.synapses = synapses
        self.topology = topology
        self.sect_loc = sect_loc
        self._nrn_sections = dict()
        self._nrn_synapses = dict()
        self.dipole_pp = list()
        self.rec_vsec = dict
        self.rec_isec = dict()
        # insert iclamp
        self.list_IClamp = list()
        self._gid = None
        self.tonic_biases = list()
        if gid is not None:
            self.gid = gid  # use setter method to check input argument gid

        self._update_end_pts()

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

    def _set_biophysics(self, sections):
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
        h.distance(sec=self._nrn_sections['soma'])
        for sec_name, section in sections.items():
            sec = self._nrn_sections[sec_name]
            for mech_name, p_mech in section.mechs.items():
                sec.insert(mech_name)
                for attr, val in p_mech.items():
                    if hasattr(val, '__call__'):
                        sec.push()
                        for seg in sec:
                            setattr(seg, attr, val(h.distance(seg.x)))
                        h.pop_section()
                    else:
                        setattr(sec, attr, val)

    def _create_synapses(self, sections, synapses):
        """Create synapses."""
        for sec_name in sections:
            for receptor in sections[sec_name].syns:
                syn_key = f'{sec_name}_{receptor}'
                seg = self._nrn_sections[sec_name](0.5)
                self._nrn_synapses[syn_key] = self.syn_create(
                    seg, **synapses[receptor])

    def _create_sections(self, sections, topology):
        """Create soma and set geometry.

        Notes
        -----
        By default neuron uses xy plane
        for height and xz plane for depth. This is opposite for model as a
        whole, but convention is followed in this function ease use of gui.
        """
        if 'soma' not in self.sections:
            raise KeyError('soma must be defined for cell')
        # shift cell to self.pos and reorient apical dendrite
        # along z direction of self.pos
        dx = self.pos[0] - self.sections['soma'].end_pts[0][0]
        dy = self.pos[1] - self.sections['soma'].end_pts[0][1]
        dz = self.pos[2] - self.sections['soma'].end_pts[0][2]

        for sec_name in sections:
            sec = h.Section(name=f'{self.name}_{sec_name}')
            self._nrn_sections[sec_name] = sec

            h.pt3dclear(sec=sec)
            h.pt3dconst(0, sec=sec)  # be explicit, see documentation
            for pt in sections[sec_name].end_pts:
                h.pt3dadd(pt[0] + dx,
                          pt[1] + dy,
                          pt[2] + dz, 1, sec=sec)
            # with pt3dconst==0, these will alter the 3d points defined above!
            sec.L = sections[sec_name].L
            sec.diam = sections[sec_name].diam
            sec.Ra = sections[sec_name].Ra
            sec.cm = sections[sec_name].cm

            if sec.L > 100.:  # 100 um
                sec.nseg = int(sec.L / 50.)
                # make dend.nseg odd for all sections
                if not sec.nseg % 2:
                    sec.nseg += 1

        if topology is None:
            topology = list()

        # Connects sections of THIS cell together.
        for connection in topology:
            parent_sec = self._nrn_sections[connection[0]]
            child_sec = self._nrn_sections[connection[2]]
            parent_loc = connection[1]
            child_loc = connection[3]
            child_sec.connect(parent_sec, parent_loc, child_loc)

        # be explicit about letting sec.L dominate over the 3d points used by
        # h.pt3dadd(); see
        # https://nrn.readthedocs.io/en/latest/python/modelspec/programmatic/topology/geometry.html?highlight=pt3dadd#pt3dadd  # noqa
        h.define_shape()

    def build(self, sec_name_apical=None):
        """Build cell in Neuron and insert dipole if applicable.

        Parameters
        ----------
        sec_name_apical : str | None
            If not None, a dipole will be inserted in this cell in alignment
            with this section. The section should belong to the apical dendrite
            of a pyramidal neuron.
        """
        self._create_sections(self.sections, self.topology)
        self._create_synapses(self.sections, self.synapses)
        self._set_biophysics(self.sections)
        if sec_name_apical in self._nrn_sections:
            self._insert_dipole(sec_name_apical)
        elif sec_name_apical is not None:
            raise ValueError(f'sec_name_apical must be an existing '
                             f'section of the current cell or None. '
                             f'Got {sec_name_apical}.')

    def copy(self):
        """Return copy of instance."""
        return deepcopy(self)

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
        cos_thetas = _get_cos_theta(self.sections, 'apical_trunk')

        # setting pointers and ztan values
        for sect_name in self.sections:
            sect = self._nrn_sections[sect_name]
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
        stim = h.IClamp(self._nrn_sections['soma'](loc))
        stim.delay = t0
        stim.dur = tstop - t0
        stim.amp = amplitude
        self.tonic_biases.append(stim)

    def record(self, record_vsoma=False, record_isoma=False,
               record_vsec=False, record_isec=False):
        """ Record current and voltage from all sections

        Parameters
        ----------
        record_vsoma : bool
            Option to record somatic voltages from cells. Default: False.
        record_isoma : bool
            Option to record somatic currents from cells. Default: False.
        record_vsec : bool
            Option to record voltages from all sections. Default: False.
        record_isec : bool
            Option to record currents from all sections. Default: False.
        """

        section_names = list(self.sections.keys())

        # Logic checks if just recording soma, sections, or both
        if record_vsoma and not record_vsec:
            self.rec_vsec = dict(['soma'])
        elif record_vsec:
            self.rec_vsec = dict.fromkeys(section_names)

        if record_vsoma or record_vsec:
            for sec_name in self.rec_vsec:
                self.rec_vsec[sec_name] = h.Vector()
                self.rec_vsec[sec_name].record(
                    self._nrn_sections[sec_name](0.5)._ref_v)

        if record_isoma and not record_isec:
            self.rec_isec = dict(['soma'])
        elif record_isec:
            self.rec_isec = dict.fromkeys(section_names)

        if record_isoma or record_isec:
            self.rec_isec = dict.fromkeys(section_names)
            for sec_name in self.rec_isec:
                list_syn = [key for key in self._nrn_synapses.keys()
                            if key.startswith(f'{sec_name}_')]
                self.rec_isec[sec_name] = dict.fromkeys(list_syn)

                for syn_name in self.rec_isec[sec_name]:
                    self.rec_isec[sec_name][syn_name] = h.Vector()

                    self.rec_isec[sec_name][syn_name].record(
                        self._nrn_synapses[syn_name]._ref_i)

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
        nc = h.NetCon(self._nrn_sections['soma'](0.5)._ref_v, None,
                      sec=self._nrn_sections['soma'])
        nc.threshold = threshold
        return nc

    def parconnect_from_src(self, gid_presyn, nc_dict, postsyn,
                            inplane_distance):
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
        inplane_distance : float
            The in plane-distance (in um) between pyramidal cell somas in the
            square grid.

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
            nc_dict['pos_src'], self.pos, nc_dict,
            inplane_distance=inplane_distance)

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

    def _update_end_pts(self):
        """"Create cell and copy coordinates to Section.end_pts"""
        self._create_sections(self.sections, self.topology)
        section_names = list(self.sections.keys())

        for name in section_names:
            nrn_pts = self._nrn_sections[name].psection()['morphology'][
                'pts3d']

            del self._nrn_sections[name]

            x0, y0, z0 = nrn_pts[0][0], nrn_pts[0][1], nrn_pts[0][2]
            x1, y1, z1 = nrn_pts[1][0], nrn_pts[1][1], nrn_pts[1][2]
            self.sections[name]._end_pts = [[x0, y0, z0], [x1, y1, z1]]

        self._nrn_sections = dict()

    def modify_section(self, sec_name, L=None, diam=None, cm=None, Ra=None):
        """Change attributes of section specified by `sec_name`

        Parameters
        ----------
        sec_name : str
            Name of section to be modified. Must be a key of Cell.sections
        L : float | int | None
            length of a section in microns. Default None.
        diam : float | int | None
            diameter of a section in microns.
        cm : float | int | None
            membrane capacitance in micro-Farads.
        Ra : float | int | None
            axial resistivity in ohm-cm.

        Notes
        -----
        Leaving default of None produces no change.
        """
        valid_sec_names = list(self.sections.keys())
        _check_option('sec_name', sec_name, valid_sec_names)

        if L is not None:
            _validate_type(L, (float, int), 'L')
            self.sections[sec_name]._L = L

        if diam is not None:
            _validate_type(diam, (float, int), 'diam')
            self.sections[sec_name]._diam = diam

        if cm is not None:
            _validate_type(cm, (float, int), 'cm')
            self.sections[sec_name]._cm = cm

        if Ra is not None:
            _validate_type(Ra, (float, int), 'Ra')
            self.sections[sec_name]._Ra = Ra

        self._update_end_pts()
