"""Neuron simulation functions and NetworkBuilder class."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>

import numpy as np
from neuron import h

from .cell import _ArtificialCell
from .pyramidal import L2Pyr, L5Pyr
from .basket import L2Basket, L5Basket

# a few globals
_PC = None
_CVODE = None

# We need to maintain a reference to the last
# NetworkBuilder instance that ran pc.gid_clear(). Even if
# pc is global, if pc.gid_clear() is called within a new
# NetworkBuilder, it will seg fault.
_LAST_NETWORK = None


def _simulate_single_trial(neuron_net, trial_idx):
    """Simulate one trial."""

    from .dipole import Dipole

    global _PC, _CVODE

    h.load_file("stdrun.hoc")

    rank = _get_rank()
    nhosts = _get_nhosts()

    # Now let's simulate the dipole

    _PC.barrier()  # sync for output to screen
    if rank == 0:
        print("running trial %d on %d cores" %
              (trial_idx + 1, nhosts))

    # Set tstop before instantiating any classes
    h.tstop = neuron_net.net.params['tstop']
    h.dt = neuron_net.net.params['dt']  # simulation duration and time-step
    h.celsius = neuron_net.net.params['celsius']  # 37.0 - set temperature

    times = neuron_net.net.spikes.times

    # sets the default max solver step in ms (purposefully large)
    _PC.set_maxstep(10)

    # initialize cells to -65 mV, after all the NetCon
    # delays have been specified
    h.finitialize()

    def simulation_time():
        print('Simulation time: {0} ms...'.format(round(h.t, 2)))

    if rank == 0:
        for tt in range(0, int(h.tstop), 10):
            _CVODE.event(tt, simulation_time)

    h.fcurrent()

    # initialization complete, but wait for all procs to start the solver
    _PC.barrier()

    # actual simulation - run the solver
    _PC.psolve(h.tstop)

    _PC.barrier()

    # these calls aggregate data across procs/nodes
    neuron_net.aggregate_dipoles()
    _PC.allreduce(neuron_net.dipoles['L5_pyramidal'], 1)
    _PC.allreduce(neuron_net.dipoles['L2_pyramidal'], 1)

    # aggregate the currents independently on each proc
    neuron_net.aggregate_currents()
    # combine neuron_net.current{} variables from each proc
    _PC.allreduce(neuron_net.current['L5_pyramidal_soma'], 1)
    _PC.allreduce(neuron_net.current['L2_pyramidal_soma'], 1)

    neuron_net.aggregate_voltages()
    _PC.py_gather(neuron_net._vsoma, 0)

    # combine spiking data from each proc
    spike_times_list = _PC.py_gather(neuron_net._spike_times, 0)
    spike_gids_list = _PC.py_gather(neuron_net._spike_gids, 0)

    # only rank 0's lists are complete

    if rank == 0:
        for spike_vec in spike_times_list:
            neuron_net._all_spike_times.append(spike_vec)
        for spike_vec in spike_gids_list:
            neuron_net._all_spike_gids.append(spike_vec)

    _PC.barrier()  # get all nodes to this place before continuing

    dpl_data = np.c_[np.array(neuron_net.dipoles['L2_pyramidal'].to_python()) +
                     np.array(neuron_net.dipoles['L5_pyramidal'].to_python()),
                     np.array(neuron_net.dipoles['L2_pyramidal'].to_python()),
                     np.array(neuron_net.dipoles['L5_pyramidal'].to_python())]

    dpl = Dipole(times, dpl_data)
    if rank == 0:
        if neuron_net.net.params['save_dpl']:
            dpl.write('rawdpl.txt')

        dpl.baseline_renormalize(neuron_net.net.params)
        dpl.convert_fAm_to_nAm()
        dpl.scale(neuron_net.net.params['dipole_scalefctr'])
        dpl.smooth(neuron_net.net.params['dipole_smooth_win'] / h.dt)

    return dpl


def _is_loaded_mechanisms():
    # copied from:
    # https://www.neuron.yale.edu/neuron/static/py_doc/modelspec/programmatic/mechtype.html
    mt = h.MechanismType(0)
    mname = h.ref('')
    mnames = list()
    for i in range(mt.count()):
        mt.select(i)
        mt.selected(mname)
        mnames.append(mname[0])
    if 'hh2' not in mnames:
        return False
    else:
        return True


def load_custom_mechanisms():
    import platform
    import os.path as op

    if _is_loaded_mechanisms():
        return

    if platform.system() == 'Windows':
        mech_fname = op.join(op.dirname(__file__), 'mod', 'nrnmech.dll')
    else:
        mech_fname = op.join(op.dirname(__file__), 'mod', 'x86_64',
                             '.libs', 'libnrnmech.so')
    if not op.exists(mech_fname):
        raise FileNotFoundError(f'The file {mech_fname} could not be found')

    h.nrn_load_dll(mech_fname)
    print('Loading custom mechanism files from %s' % mech_fname)
    if not _is_loaded_mechanisms():
        raise ValueError('The custom mechanisms could not be loaded')


def _get_nhosts():
    """Return the number of processors used by ParallelContext

    Returns
    -------
    nhosts: int
        Value from pc.nhost()
    """
    if _PC is not None:
        return int(_PC.nhost())

    return 1


def _get_rank():
    """Return the MPI rank from ParallelContext

    Returns
    -------
    rank: int
        Value from pc.id()
    """
    if _PC is not None:
        return int(_PC.id())

    return 0


def _create_parallel_context(n_cores=None):
    """Create parallel context.

    Parameters
    ----------
    n_cores: int | None
        Number of processors to use for a simulation. A value of None will
        allow NEURON to use all available processors.
    """

    global _CVODE, _PC

    if _PC is None:
        if n_cores is None:
            # MPI: Initialize the ParallelContext class
            _PC = h.ParallelContext()
        else:
            _PC = h.ParallelContext(n_cores)

        _CVODE = h.CVode()

        # be explicit about using fixed step integration
        _CVODE.active(0)

        # use cache_efficient mode for allocating elements in contiguous order
        # cvode.cache_efficient(1)
    else:
        # ParallelContext() has already been called. Don't start more workers.
        # Just tell old nrniv workers to quit.
        _PC.done()


def _long_name(short_name):
    long_name = dict(L2Basket='L2_basket', L5Basket='L5_basket',
                     L2Pyr='L2_pyramidal', L5Pyr='L5_pyramidal')
    if short_name in long_name:
        return long_name[short_name]
    return short_name


class NetworkBuilder(object):
    """The NetworkBuilder class.

    Parameters
    ----------
    net : Network object
        The instance of Network to instantiate in NEURON-Python
    trial_idx : int (optional)
        Index number of the trial being processed (different event statistics).
        Defaults to 0.

    Attributes
    ----------
    trial_idx : int
        The index number of the current trial of a simulation.
    cells : list of Cell objects.
        The list of cells containing features used in a NEURON simulation.
    ncs : dict of list
        A dictionary with key describing the types of cell objects connected
        and contains a list of NetCon objects.
    dipoles : dict of h.Vector()
        A dictionary containing total magnetic dipole moment over cell types.
        Keys are L2_pyramidal and L5_pyramidal.
    current : dict of h.Vector()
        A dictionary containing total somatic currents over cell types.
        Keys are L2_pyramidal and L5_pyramidal.

    Notes
    -----
    NetworkBuilder is not a pickleable class because it contains many NEURON
    objects once it has been instantiated. This is important for the Joblib
    backend that passes a pickled Network object to each forked process (job)
    and only instantiates NetworkBuilder after the fork.

    The `_build` routine can be called again to run more simulations without
    creating new `nrniv` processes. Instead, the NERUON objects are recreated
    and gids are reassigned according to the specifications in
    `self.net.params` and the network is ready for another simulation.
    """

    def __init__(self, net, trial_idx=0):
        self.net = net
        self.trial_idx = trial_idx

        # When computing the network dynamics in parallel, the nodes of the
        # network (real and artificial cells) potentially get distributed
        # on different host machines/threads. NetworkBuilder._gid_assign
        # assigns each node, identified by its unique GID, to one of the
        # possible hosts/threads for computations. _gid_list here contains
        # the GIDs of all the nodes assigned to the current host/thread.
        self._gid_list = []
        # Note that GIDs are already defined in Network.gid_ranges
        # All that's left for NetworkBuilder is then to:
        # - _PC.set_gid2node(gid, rank)
        # - _PC.cell(gid, nrn_netcon) (or _PC.cell(feed_cell.gid, nrn_netcon))

        # create cells (and create self.origin in create_cells_pyr())
        self.cells = []

        # artificial cells must be appended to a list in order to preserve
        # the NEURON hoc objects and the corresonding python references
        # initialized by _ArtificialCell()
        self._feed_cells = []

        self.ncs = dict()

        self._build()

    def _build(self):
        """Building the network in NEURON."""

        _create_parallel_context()

        # load mechanisms needs ParallelContext for get_rank
        load_custom_mechanisms()

        if _get_rank() == 0:
            print('Building the NEURON model')

        self._clear_last_network_objects()

        # Create a h.Vector() with size 1xself.N_t, zero'd
        self.current = {
            'L5_pyramidal_soma': h.Vector(self.net.spikes.times.size, 0),
            'L2_pyramidal_soma': h.Vector(self.net.spikes.times.size, 0),
        }

        self.dipoles = {
            'L5_pyramidal': h.Vector(self.net.spikes.times.size, 0),
            'L2_pyramidal': h.Vector(self.net.spikes.times.size, 0),
        }

        self._gid_assign()

        record_vsoma = self.net.params['record_vsoma']
        self._create_cells_and_feeds(threshold=self.net.params['threshold'],
                                     record_vsoma=record_vsoma)

        self.state_init()
        self._parnet_connect()

        # set to record spikes and somatic voltages
        self._spike_times = h.Vector()
        self._spike_gids = h.Vector()
        self._vsoma = dict()

        # used by rank 0 for spikes across all procs (MPI)
        self._all_spike_times = h.Vector()
        self._all_spike_gids = h.Vector()

        self._record_spikes()

        self.move_cells_to_pos()  # position cells in 2D grid

        if _get_rank() == 0:
            print('[Done]')

    def __enter__(self):
        """Context manager to cleanly build NetworkBuilder objects"""
        return self

    def __exit__(self, cell_type, value, traceback):
        """Clear up NEURON internal gid information."""

        self._clear_neuron_objects()
        if _LAST_NETWORK is not None:
            _LAST_NETWORK._clear_neuron_objects()

    # this happens on EACH node
    # creates self._gid_list for THIS node
    def _gid_assign(self):

        rank = _get_rank()
        nhosts = _get_nhosts()

        # round robin assignment of gids
        for gid in range(rank, self.net.n_cells, nhosts):
            # set the cell gid
            _PC.set_gid2node(gid, rank)
            self._gid_list.append(gid)
            # now to do the cell-specific external input gids on the same proc
            # these are guaranteed to exist because all of
            # these inputs were created for each cell
            # get list of all NetworkDrives that contact this cell, and
            # make sure the corresponding _ArtificialCell gids are associated
            # with the current node/rank
            for key in self.net._p_unique.keys():
                gid_input = gid + self.net.gid_ranges[key][0]
                _PC.set_gid2node(gid_input, rank)
                self._gid_list.append(gid_input)

        for gid_base in range(rank, self.net._n_common_feeds, nhosts):
            # shift the gid_base to the common gid
            gid = gid_base + self.net.gid_ranges['common'][0]
            # set as usual
            _PC.set_gid2node(gid, rank)
            self._gid_list.append(gid)
        # extremely important to get the gids in the right order
        self._gid_list.sort()

    def _create_cells_and_feeds(self, threshold, record_vsoma=False):
        """Parallel create cells AND external inputs (feeds)

        NB: _Cell.__init__ calls h.Section -> non-picklable!
        NB: _ArtificialCell.__init__ calls h.*** -> non-picklable!

        These feeds are spike SOURCES but cells are also targets.
        External inputs are not targets.
        """
        type2class = {'L2_pyramidal': L2Pyr, 'L5_pyramidal': L5Pyr,
                      'L2_basket': L2Basket, 'L5_basket': L5Basket}
        # loop through ALL gids
        # have to loop over self._gid_list, since this is what we got
        # on this rank (MPI)

        # mechanism for Builder to keep track of which trial it's on
        this_trial_event_times = self.net.trial_event_times[self.trial_idx]

        for gid in self._gid_list:
            src_type, src_pos, is_cell = self.net._get_src_type_and_pos(gid)

            if is_cell:  # not a feed
                # figure out which cell type is assoc with the gid
                # create cells based on loc property
                if src_type in ('L2_pyramidal', 'L5_pyramidal'):
                    PyramidalCell = type2class[src_type]
                    # XXX Why doesn't a _Cell have a .threshold? Would make a
                    # lot of sense to include it, as _ArtificialCells do.
                    cell = PyramidalCell(src_pos, override_params=None,
                                         gid=gid)
                else:
                    BasketCell = type2class[src_type]
                    cell = BasketCell(src_pos, gid=gid)
                if record_vsoma:
                    cell.record_voltage_soma()

                # this call could belong in init of a _Cell (with threshold)?
                nrn_netcon = cell.setup_source_netcon(threshold)
                _PC.cell(cell.gid, nrn_netcon)
                self.cells.append(cell)

            # external inputs are special types of artificial-cells
            # 'common': all cells impacted with identical TIMING of spike
            # events. NB: cell types can still have different weights for
            # how such 'common' spikes influence them
            else:
                gid_idx = gid - self.net.gid_ranges[src_type][0]
                et = this_trial_event_times[src_type][gid_idx]
                feed_cell = _ArtificialCell(et, threshold, gid=gid)
                _PC.cell(feed_cell.gid, feed_cell.nrn_netcon)
                self._feed_cells.append(feed_cell)

    def _connect_celltypes(self, src_type, target_type, loc,
                           receptor, nc_dict, unique=False,
                           allow_autapses=True):
        """Connect two cell types for a particular receptor.

        Parameters
        ----------
        src_type : str
            The source cell type
        target_type : str
            The target cell type
        loc : str
            If 'proximal' or 'distal', the corresponding
            dendritic sections from Cell.sect_loc['proximal']
            or Cell.Sect_loc['distal'] are used
        receptor : str
            The receptor.
        nc_dict : dict
            The connection dictionary containing keys
            A_delay, A_weight, lamtha, and threshold.
        unique : bool
            If True, each target cell gets one "unique" feed.
            If False, all src_type cells are connected to
            all target_type cells.
        allow_autapses : bool
            If True, allow connecting neuron to itself.
        """
        net = self.net
        connection_name = f'{src_type}_{target_type}_{receptor}'
        if connection_name not in self.ncs:
            self.ncs[connection_name] = list()

        assert len(self.cells) == len(self._gid_list) - len(self._feed_cells)
        # NB this assumes that REAL cells are first in the _gid_list
        for gid_target, target_cell in zip(self._gid_list, self.cells):
            is_target_gid = (gid_target in
                             self.net.gid_ranges[_long_name(target_type)])
            if _PC.gid_exists(gid_target) and is_target_gid:
                gid_srcs = net.gid_ranges[_long_name(src_type)]
                if unique:
                    gid_srcs = [gid_target + net.gid_ranges[src_type][0]]
                for gid_src in gid_srcs:

                    if not allow_autapses and gid_src == gid_target:
                        continue

                    pos_idx = gid_src - net.gid_ranges[_long_name(src_type)][0]
                    nc_dict['pos_src'] = net.pos_dict[
                        _long_name(src_type)][pos_idx]

                    # get synapse locations
                    syn_keys = list()
                    if loc in ['proximal', 'distal']:
                        for sect in target_cell.sect_loc[loc]:
                            syn_keys.append(f'{sect}_{receptor}')
                    else:
                        syn_keys = [f'{loc}_{receptor}']

                    for syn_key in syn_keys:
                        nc = target_cell.parconnect_from_src(
                            gid_src, nc_dict, target_cell.synapses[syn_key])
                        self.ncs[connection_name].append(nc)

    # connections:
    # this NODE is aware of its cells as targets
    # for each syn, return list of source GIDs.
    # for each item in the list, do a:
    # nc = pc.gid_connect(source_gid, target_syn), weight,delay
    # Both for synapses AND for external inputs
    def _parnet_connect(self):
        params = self.net.params
        nc_dict = {
            'A_delay': 1.,
            'threshold': params['threshold'],
        }

        # source of synapse is always at soma

        # layer2 Pyr -> layer2 Pyr
        # layer5 Pyr -> layer5 Pyr
        nc_dict['lamtha'] = 3.
        for target_cell in ['L2Pyr', 'L5Pyr']:
            for receptor in ['nmda', 'ampa']:
                key = f'gbar_{target_cell}_{target_cell}_{receptor}'
                nc_dict['A_weight'] = params[key]
                self._connect_celltypes(target_cell, target_cell, 'proximal',
                                        receptor, nc_dict,
                                        allow_autapses=False)

        # layer2 Basket -> layer2 Pyr
        target_cell = 'L2Pyr'
        nc_dict['lamtha'] = 50.
        for receptor in ['gabaa', 'gabab']:
            nc_dict['A_weight'] = params[f'gbar_L2Basket_L2Pyr_{receptor}']
            self._connect_celltypes('L2Basket', target_cell, 'soma', receptor,
                                    nc_dict)

        # layer5 Basket -> layer5 Pyr
        target_cell = 'L5Pyr'
        nc_dict['lamtha'] = 70.
        for receptor in ['gabaa', 'gabab']:
            key = f'gbar_L5Basket_{target_cell}_{receptor}'
            nc_dict['A_weight'] = params[key]
            self._connect_celltypes('L5Basket', target_cell, 'soma', receptor,
                                    nc_dict)

        # layer2 Pyr -> layer5 Pyr
        nc_dict['lamtha'] = 3.
        for loc in ['proximal', 'distal']:
            nc_dict['A_weight'] = params[f'gbar_L2Pyr_{target_cell}']
            self._connect_celltypes('L2Pyr', target_cell, loc, 'ampa',
                                    nc_dict)
        # layer2 Basket -> layer5 Pyr
        nc_dict['lamtha'] = 50.
        nc_dict['A_weight'] = params[f'gbar_L2Basket_{target_cell}']
        self._connect_celltypes('L2Basket', target_cell, 'distal', 'gabaa',
                                nc_dict)

        # xx -> layer2 Basket
        target_cell = 'L2Basket'
        nc_dict['lamtha'] = 3.
        nc_dict['A_weight'] = params[f'gbar_L2Pyr_{target_cell}']
        self._connect_celltypes('L2Pyr', target_cell, 'soma', 'ampa',
                                nc_dict)
        nc_dict['lamtha'] = 20.
        nc_dict['A_weight'] = params[f'gbar_L2Basket_{target_cell}']
        self._connect_celltypes('L2Basket', target_cell, 'soma', 'gabaa',
                                nc_dict)

        # xx -> layer5 Basket
        target_cell = 'L5Basket'
        nc_dict['lamtha'] = 20.
        nc_dict['A_weight'] = params[f'gbar_L5Basket_{target_cell}']
        self._connect_celltypes('L5Basket', target_cell, 'soma', 'gabaa',
                                nc_dict, allow_autapses=False)
        nc_dict['lamtha'] = 3.
        nc_dict['A_weight'] = params[f'gbar_L5Pyr_{target_cell}']
        self._connect_celltypes('L5Pyr', target_cell, 'soma', 'ampa',
                                nc_dict)
        nc_dict['A_weight'] = params[f'gbar_L2Pyr_{target_cell}']
        self._connect_celltypes('L2Pyr', target_cell, 'soma', 'ampa',
                                nc_dict)

        # common feed -> xx
        for p_common in self.net._p_common:
            for target_cell_type in ['L2Basket', 'L5Basket', 'L5Pyr', 'L2Pyr']:
                if (target_cell_type == 'L5Basket' and
                        p_common['loc'] == 'distal'):
                    continue
                for receptor in ['ampa', 'nmda']:
                    if f'{target_cell_type}_{receptor}' in p_common.keys():
                        nc_dict['lamtha'] = p_common['lamtha']
                        nc_dict['A_weight'] = \
                            p_common[f'{target_cell_type}_{receptor}'][0]
                        nc_dict['A_delay'] = \
                            p_common[f'{target_cell_type}_{receptor}'][1]
                        self._connect_celltypes('common', target_cell_type,
                                                p_common['loc'], receptor,
                                                nc_dict)

        # unique feed -> xx
        p_unique = self.net._p_unique
        for src_cell_type in p_unique:

            p_src = p_unique[src_cell_type]
            receptors = ['ampa', 'nmda']
            if src_cell_type == 'extgauss':
                receptors = ['ampa']

            for target_cell_type in ['L2Basket', 'L5Basket', 'L5Pyr', 'L2Pyr']:
                # XXX: hack for distal connection
                if target_cell_type == 'L5Basket' and p_src['loc'] == 'distal':
                    continue
                for receptor in receptors:
                    target_cell_long = _long_name(target_cell_type)
                    nc_dict['lamtha'] = p_src['lamtha']
                    nc_dict['A_delay'] = p_src[target_cell_long][2]
                    if receptor == 'ampa':
                        nc_dict['A_weight'] = p_src[target_cell_long][0]
                    elif receptor == 'nmda':
                        nc_dict['A_weight'] = p_src[target_cell_long][1]
                    self._connect_celltypes(src_cell_type, target_cell_type,
                                            p_src['loc'], receptor, nc_dict,
                                            unique=True)

    # setup spike recording for this node
    def _record_spikes(self):

        # iterate through gids on this node and
        # set to record spikes in spike time vec and id vec
        # agnostic to type of source, will sort that out later
        for gid in self._gid_list:
            if _PC.gid_exists(gid):
                _PC.spike_record(gid, self._spike_times, self._spike_gids)

    # aggregate recording all the somatic voltages for pyr
    def aggregate_currents(self):
        """Aggregate somatic currents for Pyramidal cells."""
        for cell in self.cells:
            if cell.celltype in ('L5_pyramidal', 'L2_pyramidal'):
                # iterate over dict_currents created in
                # cell.record_current_soma()
                for _, I_soma in cell.dict_currents.items():
                    # self.current['L5_pyramidal_soma'] was created upon
                    # in parallel, each node has its own NetworkBuilder()
                    self.current['%s_soma' % cell.celltype].add(I_soma)

    def aggregate_dipoles(self):
        """Aggregate dipoles."""
        for cell in self.cells:
            if cell.celltype in ('L5_pyramidal', 'L2_pyramidal'):
                self.dipoles[cell.celltype].add(cell.dipole)

    def aggregate_voltages(self):
        """Organize voltage recordings into dictionary indexed by gid"""
        for cell in self.cells:
            self._vsoma[cell.gid] = cell.rec_v

    def state_init(self):
        """Initializes the state closer to baseline."""

        for cell in self.cells:
            seclist = h.SectionList()
            seclist.wholetree(sec=cell.soma)
            for sect in seclist:
                for seg in sect:
                    if cell.celltype == 'L2_pyramidal':
                        seg.v = -71.46
                    elif cell.celltype == 'L5_pyramidal':
                        if sect.name() == 'L5Pyr_apical_1':
                            seg.v = -71.32
                        elif sect.name() == 'L5Pyr_apical_2':
                            seg.v = -69.08
                        elif sect.name() == 'L5Pyr_apical_tuft':
                            seg.v = -67.30
                        else:
                            seg.v = -72.
                    elif cell.celltype == 'L2_basket':
                        seg.v = -64.9737
                    elif cell.celltype == 'L5_basket':
                        seg.v = -64.9737

    def move_cells_to_pos(self):
        """Move cells 3d positions to positions used for wiring."""
        for cell in self.cells:
            cell.move_to_pos()

    def _clear_neuron_objects(self):
        """Clear up NEURON internal gid information.

        Note: This function must be called from the context of the
        Network instance that ran build_in_neuron. This is a bug or
        peculiarity of NEURON. If this function is called from a different
        context, then the next simulation will run very slow because nrniv
        workers are still going for the old simulation. If pc.gid_clear is
        called from the right context, then those workers can exit.
        """

        _PC.gid_clear()

        # dereference cell and NetConn objects
        for gid, cell in zip(self._gid_list, self.cells):
            # only work on cells on this node
            if _PC.gid_exists(gid):
                for nc_key in self.ncs:
                    for nc in self.ncs[nc_key]:
                        if nc.valid():
                            # delete NEURON cell object
                            cell_obj1 = nc.precell(gid)
                            if cell_obj1 is not None:
                                del cell_obj1
                            cell_obj2 = nc.postcell(gid)
                            if cell_obj2 is not None:
                                del cell_obj2
                            del nc

        self._gid_list = []
        self.cells = []

    def get_data_from_neuron(self):
        """Get copies of spike data that are pickleable"""

        vsoma_py = {}
        for gid, rec_v in self._vsoma.items():
            vsoma_py[gid] = rec_v.to_python()

        from copy import deepcopy
        data = (self._all_spike_times.to_python(),
                self._all_spike_gids.to_python(),
                deepcopy(self.net.gid_ranges),
                deepcopy(vsoma_py))
        return data

    def _clear_last_network_objects(self):
        """Clears NEURON objects and saves the current Network instance"""

        global _LAST_NETWORK

        if _LAST_NETWORK is not None:
            _LAST_NETWORK._clear_neuron_objects()

        self._clear_neuron_objects()
        _LAST_NETWORK = self
