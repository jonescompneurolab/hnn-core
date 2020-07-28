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


class _Cell(object):
    """Create a cell object.

    Parameters
    ----------
    gid : int
        The cell ID
    soma_props : dict
        The properties of the soma. Must contain
        keys 'L', 'diam', and 'pos'

    Attributes
    ----------
    pos : list of length 3
        The position of the cell.
    """

    def __init__(self, gid, soma_props):
        self.gid = gid
        # variable for the list_IClamp
        self.list_IClamp = None
        self.soma_props = soma_props
        self.create_soma()
        # par: create arbitrary lists of connections FROM other cells
        # TO this cell instantiation
        # these lists are allowed to be empty
        # this should be a dict
        self.ncfrom_L2Pyr = []
        self.ncfrom_L2Basket = []
        self.ncfrom_L5Pyr = []
        self.ncfrom_L5Basket = []
        self.ncfrom_common = []
        self.ncfrom_extgauss = []
        self.ncfrom_extpois = []
        self.ncfrom_ev = []

    def __repr__(self):
        class_name = self.__class__.__name__
        soma_props = self.soma_props
        s = ('soma: L %f, diam %f, Ra %f, cm %f' %
             (soma_props['L'], soma_props['diam'],
              soma_props['Ra'], soma_props['cm']))
        return '<%s | %s>' % (class_name, s)

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

    def get_sections(self):
        """Get sections."""
        return [self.soma]

    def get3dinfo(self):
        """Get 3d info."""
        ls = self.get_sections()
        lx, ly, lz, ldiam = [], [], [], []
        for s in ls:
            for i in range(s.n3d()):
                lx.append(s.x3d(i))
                ly.append(s.y3d(i))
                lz.append(s.z3d(i))
                ldiam.append(s.diam3d(i))
        return lx, ly, lz, ldiam

    def getbbox(self):
        """Get cell's bounding box."""
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
        """Translate 3d."""
        for s in self.get_sections():
            for i in range(s.n3d()):
                h.pt3dchange(i, s.x3d(i) + dx, s.y3d(i) + dy,
                             s.z3d(i) + dz, s.diam3d(i), sec=s)

    def translate_to(self, x, y, z):
        """Translate to position."""
        x0 = self.soma.x3d(0)
        y0 = self.soma.y3d(0)
        z0 = self.soma.z3d(0)
        dx = x - x0
        dy = y - y0
        dz = z - z0
        # print('dx:',dx,'dy:',dy,'dz:',dz)
        self.translate3d(dx, dy, dz)

    def move_to_pos(self):
        """Move cell to position."""
        self.translate_to(self.pos[0] * 100, self.pos[2], self.pos[1] * 100)

    def _connect(self, gid, gid_dict, pos_dict, p, type_src, name_src,
                 lamtha=3., receptor=None, postsyns=None, autapses=True):
        for gid_src, pos in zip(gid_dict[type_src],
                                pos_dict[type_src]):
            if not autapses and gid_src == gid:
                continue
            if receptor is not None:
                A_weight = p['gbar_%s_%s_%s' %
                             (name_src, self.name, receptor)]
            else:
                A_weight = p['gbar_%s_%s' % (name_src, self.name)]
            nc_dict = {
                'pos_src': pos,
                'A_weight': A_weight,
                'A_delay': 1.,
                'lamtha': lamtha,
                'threshold': p['threshold'],
                'type_src': type_src
            }

            for postsyn in postsyns:
                getattr(self, 'ncfrom_%s' % name_src).append(
                    self.parconnect_from_src(
                        gid_src, nc_dict, postsyn))

    # two things need to happen here for h:
    # 1. dipole needs to be inserted into each section
    # 2. a list needs to be created with a Dipole (Point Process) in each
    #    section at position 1
    # In Cell() and not Pyr() for future possibilities
    def dipole_insert(self, yscale):
        """Insert dipole into each section of this cell."""
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

    def record_current_soma(self):
        """Record current at soma."""
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
        """Create gabaa receiving synapse.

        Parameters
        ----------
        secloc : float (0 to 1.0)
            The section location
        """
        syn_gabaa = h.Exp2Syn(secloc)
        syn_gabaa.e = -80
        syn_gabaa.tau1 = 0.5
        syn_gabaa.tau2 = 5.
        return syn_gabaa

    # creates a RECEIVING slow inhibitory synapse at secloc
    # called: self.soma_gabab = syn_gabab_create(self.soma(0.5))
    def syn_gabab_create(self, secloc):
        """Create gabab receiving synapse.

        Parameters
        ----------
        secloc : float (0 to 1.0)
            The section location.
        """
        syn_gabab = h.Exp2Syn(secloc)
        syn_gabab.e = -80
        syn_gabab.tau1 = 1
        syn_gabab.tau2 = 20.
        return syn_gabab

    # creates a RECEIVING excitatory synapse at secloc
    # def syn_ampa_create(self, secloc, tau_decay, prng_obj):
    def syn_ampa_create(self, secloc):
        """Create ampa receiving synapse.

        Parameters
        ----------
        secloc : float (0 to 1.0)
            The section location.
        """
        syn_ampa = h.Exp2Syn(secloc)
        syn_ampa.e = 0.
        syn_ampa.tau1 = 0.5
        syn_ampa.tau2 = 5.
        return syn_ampa

    # creates a RECEIVING nmda synapse at secloc
    # this is a pretty fast NMDA, no?
    def syn_nmda_create(self, secloc):
        """Create nmda receiving synapse.

        Parameters
        ----------
        secloc : float (0 to 1.0)
            The section location.
        """
        syn_nmda = h.Exp2Syn(secloc)
        syn_nmda.e = 0.
        syn_nmda.tau1 = 1.
        syn_nmda.tau2 = 20.
        return syn_nmda

    def setup_netcon(self, threshold):
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
        postsyn : str
            The postsynaptic cell object.

        Returns
        -------
        nc : instance of h.NetCon
            A network connection object.
        """
        from .neuron import _PC

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

    def plot_voltage(self, ax=None, delay=2, duration=100, dt=0.2,
                     amplitude=1, show=True):
        """Plot voltage on soma for an injected current

        Parameters
        ----------
        ax : instance of matplotlib axis | None
            An axis object from matplotlib. If None,
            a new figure is created.
        delay : float (in ms)
            The start time of the injection current.
        duration : float (in ms)
            The duration of the injection current.
        dt : float (in ms)
            The integration time step
        amplitude : float (in nA)
            The amplitude of the injection current.
        show : bool
            Call plt.show() if True. Set to False if working in
            headless mode (e.g., over a remote connection).
        """
        import matplotlib.pyplot as plt
        from neuron import h
        h.load_file('stdrun.hoc')

        soma = self.soma

        h.tstop = duration
        h.dt = dt
        h.celsius = 37

        iclamp = h.IClamp(soma(0.5))
        iclamp.delay = delay
        iclamp.dur = duration
        iclamp.amp = amplitude

        v_membrane = h.Vector().record(self.soma(0.5)._ref_v)
        times = h.Vector().record(h._ref_t)

        print('Simulating soma voltage')
        h.finitialize()

        def simulation_time():
            print('Simulation time: {0} ms...'.format(round(h.t, 2)))

        for tt in range(0, int(h.tstop), 10):
            h.CVode().event(tt, simulation_time)

        h.continuerun(duration)
        print('[Done]')

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.plot(times, v_membrane)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (mV)')
        if show:
            plt.show()
