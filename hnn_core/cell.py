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


def node_to_str(node):
    return node[0] + "," + str(node[1])


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


def _get_nseg(L):
    nseg = 1
    if L > 100.:  # 100 um
        nseg = int(L / 50.)
        # make dend.nseg odd for all sections
        if not nseg % 2:
            nseg += 1
    return nseg


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
    nseg : int
        Number of segments in the section
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

        # For distance functionality
        self.nseg = _get_nseg(self.L)

    def __repr__(self):
        return f'L={self.L}, diam={self.diam}, cm={self.cm}, Ra={self.Ra}'

    def __eq__(self, other):
        if not isinstance(other, Section):
            return NotImplemented

        # Check equality for mechs
        for mech_name in self.mechs.keys():
            self_mech = self.mechs[mech_name]
            other_mech = other.mechs[mech_name]
            for attr in self_mech.keys():
                if self_mech[attr] != other_mech[attr]:
                    return False

        # Check end_pts
        for self_end_pt, other_end_pt in zip(self.end_pts, other.end_pts):
            if np.testing.assert_almost_equal(self_end_pt,
                                              other_end_pt, 5) is not None:
                return False

        all_attrs = dir(self)
        attrs_to_ignore = [x for x in all_attrs if x.startswith('_')]
        attrs_to_ignore.extend(['end_pts', 'mechs', 'to_dict'])
        attrs_to_check = [x for x in all_attrs if x not in attrs_to_ignore]

        # Check all other attributes
        for attr in attrs_to_check:
            if getattr(self, attr) != getattr(other, attr):
                return False

        return True

    def to_dict(self):
        """Converts an object of Section class to a dictionary.

        Returns
        -------
        dictionary form of an object of Section class.
        """
        section_data = dict()
        section_data['L'] = self.L
        section_data['diam'] = self.diam
        section_data['cm'] = self.cm
        section_data['Ra'] = self.Ra
        section_data['end_pts'] = self.end_pts
        section_data['nseg'] = self.nseg
        # Need to solve the partial function problem
        # in mechs
        section_data['mechs'] = self.mechs
        section_data['syns'] = self.syns
        return section_data

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
        sections.
    sect_loc : dict of list
        Can have keys 'proximal' or 'distal' each containing
        names of section locations that are proximal or distal.
    gid : int or None (optional)
        Each cell in a network is uniquely identified by it's "global ID": GID.
        The GID is an integer from 0 to n_cells, or None if the cell is not
        yet attached to a network. Once the GID is set, it cannot be changed.
    cell_tree : dict of list
        Stores the tree representation of a cell.
        Root is the 0 end of 'soma'. Nodes are a tuple (sec_name, node_pos)
        where sec_name is the name of the section and node_pos is the 0 end
        or 1 end. The data structure is the adjacency list representation of a
        tree. The keys of the dict are the parent nodes. The value is the
        list of nodes (children nodes) connected to the parent node.


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
    vsec : dict
        Recording of section specific voltage. Must be enabled
        by running simulate_dipole(net, record_vsec=True) or
        simulate_dipole(net, record_vsoma=True)
    isec : dict
        Contains recording of section specific currents indexed
        by synapse type (keys can be soma_gabaa, soma_gabab etc.).
        Must be enabled by running simulate_dipole(net, record_isec=True)
        or simulate_dipole(net, record_isoma=True)
    ca : dict
        Contains recording of section speicifc calcium concentration.
        Must be enabled by running simulate_dipole(net, record_ca=True).
    tonic_biases : list of h.IClamp
        The current clamps inserted at each section of the cell
        for tonic biasing inputs.
    gid : int
        GID of the cell in a network (or None if not yet assigned)
    sect_loc : dict of list
        Can have keys 'proximal' or 'distal' each containing
        names of section locations that are proximal or distal.
    cell_tree : dict of list
        Stores the tree representation of a cell.
        Root is the 0 end of 'soma'. Nodes are a tuple (sec_name, node_pos)
        where sec_name is the name of the section and node_pos is the 0 end
        or 1 end. The data structure is the adjacency list representation of a
        tree. The keys of the dict are the parent nodes. The value is the
        list of nodes (children nodes) connected to the parent node.

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

    def __init__(self, name, pos, sections, synapses, sect_loc, cell_tree,
                 gid=None):
        self.name = name
        self.pos = pos
        for section in sections.values():
            if not isinstance(section, Section):
                raise ValueError(f'Items in section must be instances'
                                 f' of Section. Got {type(section)}')
        self.sections = sections
        self.synapses = synapses
        self.sect_loc = sect_loc
        self._nrn_sections = dict()
        self._nrn_synapses = dict()
        self.dipole_pp = list()
        self.vsec = dict()
        self.isec = dict()
        self.ca = dict()
        # insert iclamp
        self.list_IClamp = list()
        self._gid = None
        self.tonic_biases = list()
        if gid is not None:
            self.gid = gid  # use setter method to check input argument gid

        # Store the tree representation of the cell
        self.cell_tree = cell_tree

        self._update_end_pts()  # New implementation

        self._compute_section_mechs()  # Set mech values of all sections

    def __repr__(self):
        class_name = self.__class__.__name__
        return f'<{class_name} | gid={self._gid}>'

    def __eq__(self, other):
        if not isinstance(other, Cell):
            return NotImplemented

        all_attrs = dir(self)
        attrs_to_ignore = [x for x in all_attrs if x.startswith('_')]
        attrs_to_ignore.extend(['build', 'copy', 'create_tonic_bias',
                                'define_shape', 'distance_section', 'gid',
                                'list_IClamp', 'modify_section',
                                'parconnect_from_src', 'plot_morphology',
                                'record', 'sections', 'setup_source_netcon',
                                'syn_create', 'to_dict'])
        attrs_to_check = [x for x in all_attrs if x not in attrs_to_ignore]

        # Check all other attributes
        for attr in attrs_to_check:
            if getattr(self, attr) != getattr(other, attr):
                return False

        if not (self.sections.keys() == other.sections.keys()):
            return False

        for key in self.sections.keys():
            if self.sections[key] != other.sections[key]:
                return False

        return True

    def to_dict(self):
        """Converts an object of Cell class to a dictionary.

        Returns
        -------
        dictionary form of an object of Cell class.
        """
        cell_data = dict()
        cell_data['name'] = self.name
        cell_data['pos'] = self.pos
        cell_data['sections'] = dict()
        for key in self.sections:
            cell_data['sections'][key] = self.sections[key].to_dict()
        cell_data['synapses'] = self.synapses
        # cell_data['cell_tree'] = self.cell_tree
        if self.cell_tree is None:
            cell_data['cell_tree'] = None
        else:
            cell_tree_dict = dict()
            for parent, children in self.cell_tree.items():
                key = node_to_str(parent)
                value = list()
                for child in children:
                    value.append(node_to_str(child))
                cell_tree_dict[key] = value
            cell_data['cell_tree'] = cell_tree_dict
        cell_data['sect_loc'] = self.sect_loc
        cell_data['gid'] = self.gid
        cell_data['dipole_pp'] = self.dipole_pp
        cell_data['vsec'] = self.vsec
        cell_data['isec'] = self.isec
        cell_data['ca'] = self.ca
        cell_data['tonic_biases'] = self.tonic_biases
        return cell_data

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

    def distance_section(self, target_sec_name, curr_node):
        """Find distance between the current node and the target section.

        Parameters
        ----------
        target_sec_name : string
            Name of the target section
        curr_node : tuple
            Source node from where search begins.
            It is of the the form (sec_name, end_pt).

        Returns
        -------
        distance : float
            Path distance between source node and mid of the target section.
        """
        # Python version of the Neuron distance function
        # https://nrn.readthedocs.io/en/latest/python/modelspec/programmatic/topology/geometry.html#distance  # noqa
        if self.cell_tree is None:
            raise TypeError("distance_section() "
                            "cannot work with cell_tree as None.")
        if curr_node not in self.cell_tree:
            return np.nan

        # Children of the current section
        curr_sec_children = self.cell_tree[curr_node]
        # All sections have 0 and 1 ends
        end_pts = (0, 1)

        # Base condition
        # If target section is connected to current section
        # Return (target section length / 2)
        # As distances are measured till the centre of the target section
        for end_pt in end_pts:
            if (target_sec_name, end_pt) in curr_sec_children:
                return self.sections[target_sec_name].L / 2

        dist = np.nan  # Return nan

        # Recursion to find distance
        for node in self.cell_tree[curr_node]:
            if (node[0] == curr_node[0]):
                dist_temp = (self.distance_section(target_sec_name, node) +
                             self.sections[node[0]].L)
            else:
                dist_temp = self.distance_section(target_sec_name, node)
            if np.isnan(dist) and np.isnan(dist_temp):
                dist = np.nan
            else:
                dist = np.nanmin([dist, dist_temp])

        return dist

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
                    if isinstance(val, list):
                        seg_xs, seg_vals = val[0], val[1]
                        for seg, seg_x, seg_val in zip(sec, seg_xs, seg_vals):
                            setattr(seg, attr, seg_val)
                    else:
                        setattr(sec, attr, val)

    def _compute_section_mechs(self):
        sections = self.sections
        for sec_name, section in sections.items():
            for mech_name, p_mech in section.mechs.items():
                for attr, val in p_mech.items():
                    if hasattr(val, '__call__'):
                        seg_xs, seg_vals = list(), list()
                        section_distance = self.distance_section(sec_name,
                                                                 ('soma', 0))
                        seg_centers = (np.linspace(0, 1, section.nseg * 2 + 1)
                                       [1::2])

                        for seg_x in seg_centers:
                            # sec_end_dist is distance between 0 end of soma to
                            # the 0 or 1 end of section (whichever is closer)
                            sec_end_dist = section_distance - (section.L / 2)
                            seg_xs.append(seg_x)
                            seg_vals.append(val(sec_end_dist +
                                                (seg_x * section.L)))
                        p_mech[attr] = [seg_xs, seg_vals]
        return self.sections

    def _create_synapses(self, sections, synapses):
        """Create synapses."""
        for sec_name in sections:
            for receptor in sections[sec_name].syns:
                syn_key = f'{sec_name}_{receptor}'
                seg = self._nrn_sections[sec_name](0.5)
                self._nrn_synapses[syn_key] = self.syn_create(
                    seg, **synapses[receptor])

    def _create_sections(self, sections, cell_tree):
        """Create soma and set geometry.

        Notes
        -----
        By default neuron uses xy plane
        for height and xz plane for depth. This is opposite for model as a
        whole, but convention is followed in this function ease use of gui.
        """
        if 'soma' not in self.sections:
            raise KeyError('soma must be defined for cell')

        for sec_name in sections:
            sec = h.Section(name=f'{self.name}_{sec_name}')
            self._nrn_sections[sec_name] = sec

            h.pt3dclear(sec=sec)
            h.pt3dconst(0, sec=sec)  # be explicit, see documentation
            for pt in sections[sec_name].end_pts:
                h.pt3dadd(pt[0], pt[1], pt[2], 1, sec=sec)
            # with pt3dconst==0, these will alter the 3d points defined above!
            sec.L = sections[sec_name].L
            sec.diam = sections[sec_name].diam
            sec.Ra = sections[sec_name].Ra
            sec.cm = sections[sec_name].cm
            sec.nseg = sections[sec_name].nseg

        if cell_tree is None:
            cell_tree = dict()

        # Connects sections of THIS cell together.
        for parent_node in cell_tree:
            for child_node in cell_tree[parent_node]:
                parent_sec = self._nrn_sections[parent_node[0]]
                child_sec = self._nrn_sections[child_node[0]]
                if parent_sec == child_sec:
                    continue
                parent_loc = parent_node[1]
                child_loc = child_node[1]
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
        self._create_sections(self.sections, self.cell_tree)
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

    def record(self, record_vsec=False, record_isec=False, record_ca=False):
        """ Record current and voltage from all sections

        Parameters
        ----------
        record_vsec : 'all' | 'soma' | False
            Option to record voltages from all sections ('all'), or just
            the soma ('soma'). Default: False.
        record_isec : 'all' | 'soma' | False
            Option to record voltages from all sections ('all'), or just
            the soma ('soma'). Default: False.
        record_ca : 'all' | 'soma' | False
            Option to record calcium concentration from all sections ('all'),
            or just the soma ('soma'). Default: False.
        """

        section_names = list(self.sections.keys())

        # Logic checks if just recording soma, sections, or both
        if record_vsec == 'soma':
            self.vsec = dict.fromkeys(['soma'])
        elif record_vsec == 'all':
            self.vsec = dict.fromkeys(section_names)

        if record_vsec:
            for sec_name in self.vsec:
                self.vsec[sec_name] = h.Vector()
                self.vsec[sec_name].record(
                    self._nrn_sections[sec_name](0.5)._ref_v)

        if record_isec == 'soma':
            self.isec = dict.fromkeys(['soma'])
        elif record_isec == 'all':
            self.isec = dict.fromkeys(section_names)

        if record_isec:
            for sec_name in self.isec:
                list_syn = [key for key in self._nrn_synapses.keys()
                            if key.startswith(f'{sec_name}_')]
                self.isec[sec_name] = dict.fromkeys(list_syn)

                for syn_name in self.isec[sec_name]:
                    self.isec[sec_name][syn_name] = h.Vector()
                    self.isec[sec_name][syn_name].record(
                        self._nrn_synapses[syn_name]._ref_i)

        # calcium concentration
        if record_ca == 'soma':
            self.ca = dict.fromkeys(['soma'])
        elif record_ca == 'all':
            self.ca = dict.fromkeys(section_names)

        if record_ca:
            for sec_name in self.ca:
                if hasattr(self._nrn_sections[sec_name](0.5), '_ref_cai'):
                    self.ca[sec_name] = h.Vector()
                    self.ca[sec_name].record(
                        self._nrn_sections[sec_name](0.5)._ref_cai)

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

    def plot_morphology(self, ax=None, color=None, pos=(0, 0, 0),
                        xlim=(-250, 150), ylim=(-100, 100), zlim=(-100, 1200),
                        show=True):
        """Plot the cell morphology.

        Parameters
        ----------
        ax : instance of Axes3D
            Matplotlib 3D axis
        color : str | dict | None
            Color of cell. If str, entire cell plotted with
            color indicated by str. If dict, colors of individual sections
            can be specified. Must have a key for every section in cell as
            defined in the `Cell.sections` attribute.

        | Ex: ``{'apical_trunk': 'r', 'soma': 'b', ...}``
        pos : tuple of int or float | None
            Position of cell soma. Must be a tuple of 3 elements for the
            (x, y, z) position of the soma in 3D space. Default: (0, 0, 0)
        xlim : tuple of int | tuple of float
            x limits of plot window. Default (-250, 150)
        ylim : tuple of int | tuple of float
            y limits of plot window. Default (-100, 100)
        zlim : tuple of int | tuple of float
            z limits of plot window. Default (-100, 1200)
        show : bool
            If True, show the plot

        Returns
        -------
        axes : instance of Axes3D
            The matplotlib 3D axis handle.
        """
        return plot_cell_morphology(self, ax=ax, color=color, pos=pos,
                                    xlim=xlim, ylim=ylim, zlim=zlim, show=show)

    def _update_section_end_pts_L(self, node, dpt):
        if self.cell_tree is None:
            return
        x = self.sections[node[0]].end_pts[node[1]][0]
        y = self.sections[node[0]].end_pts[node[1]][1]
        z = self.sections[node[0]].end_pts[node[1]][2]
        self.sections[node[0]].end_pts[node[1]][0] = x + dpt[0]
        self.sections[node[0]].end_pts[node[1]][1] = y + dpt[1]
        self.sections[node[0]].end_pts[node[1]][2] = z + dpt[2]

        # If current node is a leaf node
        if node not in self.cell_tree:
            return

        # If current node is an internal node
        for child_node in self.cell_tree[node]:
            self._update_section_end_pts_L(child_node, dpt)

    def define_shape(self, node):
        """Redefines end_pts according to section lengths.

        Detects change in section lengths of the sections in the
        subtree of the input node.

        Parameters
        ----------
        node : tuple of size 2
            The first element is the section name
            The second element is the node end used (0 or 1)

        Note
        ----
        Using sec_name as 'soma' and node end as 0 checks for changes
        in any section length of the cell as (soma, 0) is the root node
        of the cell.
        """
        # Python version of Neuron define_shape function
        # https://nrn.readthedocs.io/en/latest/python/modelspec/programmatic/topology/geometry.html?highlight=pt3dadd#pt3dadd  # noqa
        # cell tree is None therefore cannot define shape
        if self.cell_tree is None:
            return
        # Find the end pts of the section
        node_opp_end = 1
        if node[1] == 1:
            node_opp_end = 0
        pts = self.sections[node[0]].end_pts
        x0, y0, z0 = pts[node[1]][0], pts[node[1]][1], pts[node[1]][2]
        x1, y1, z1 = (pts[node_opp_end][0], pts[node_opp_end][1],
                      pts[node_opp_end][2])

        # Find the factor by which length is changed
        end_1 = np.array((x0, y0, z0))
        end_2 = np.array((x1, y1, z1))
        old_len = np.linalg.norm(end_1 - end_2)
        new_len = self.sections[node[0]].L
        fac = new_len / old_len
        x_new = x0 + (x1 - x0) * fac
        y_new = y0 + (y1 - y0) * fac
        z_new = z0 + (z1 - z0) * fac

        # Find the change in coordinates
        dx = x_new - x1
        dy = y_new - y1
        dz = z_new - z1
        dpt = [dx, dy, dz]

        # Update all coordinates in the subtree
        self._update_section_end_pts_L((node[0], node_opp_end), dpt)

        # Check for change in section lengths in the subtree
        if node in self.cell_tree:
            for child_node in self.cell_tree[node]:
                self.define_shape(child_node)

    def _update_end_pts(self):
        """Update all end pts according to the length of the sections.

        Can be used whenever length of any section is updated

        Returns
        -------
        Updated end pts for the cell
        """
        if 'soma' not in self.sections:
            raise KeyError('soma must be defined for cell')
        # cell tree is None therefore no end_pts to update
        if self.cell_tree is None:
            return

        # shift cell to self.pos and reorient apical dendrite
        # along z direction of self.pos
        dx = self.pos[0] - self.sections['soma'].end_pts[0][0]
        dy = self.pos[1] - self.sections['soma'].end_pts[0][1]
        dz = self.pos[2] - self.sections['soma'].end_pts[0][2]
        for sec_name in self.sections:
            end_pts = self.sections[sec_name].end_pts
            updated_end_pts = list()
            for pt in end_pts:
                updated_end_pts.append(
                    [
                        pt[0] + dx,
                        pt[1] + dy,
                        pt[2] + dz
                    ]
                )
            self.sections[sec_name]._end_pts = updated_end_pts

        # Check and update all end pts starting from root according to length
        # of sections.
        self.define_shape(('soma', 0))

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
