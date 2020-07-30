"""Neuron simulation functions and NeuronNetwork class."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>

import numpy as np
from neuron import h

from .feed import ExtFeed
from .cell import _ArtificialCell
from .pyramidal import L2Pyr, L5Pyr
from .basket import L2Basket, L5Basket

# a few globals
_PC = None
_CVODE = None

# We need to maintain a reference to the last
# NeuronNetwork instance that ran pc.gid_clear(). Even if
# pc is global, if pc.gid_clear() is called within a new
# NeuronNetwork, it will seg fault.
_LAST_NETWORK = None

# NEURON only allows mechanisms to be loaded once (per Python interpreter)
_LOADED_DLL = None


def _simulate_single_trial(neuron_net):
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
              (neuron_net.net.trial_idx + 1, nhosts))

    # create or reinitialize scalars in NEURON (hoc) context
    h("dp_total_L2 = 0.")
    h("dp_total_L5 = 0.")

    # Set tstop before instantiating any classes
    h.tstop = neuron_net.net.params['tstop']
    h.dt = neuron_net.net.params['dt']  # simulation duration and time-step
    h.celsius = neuron_net.net.params['celsius']  # 37.0 - set temperature

    # We define the arrays (Vector in numpy) for recording the signals
    t_vec = h.Vector()
    t_vec.record(h._ref_t)  # time recording
    dp_rec_L2 = h.Vector()
    dp_rec_L2.record(h._ref_dp_total_L2)  # L2 dipole recording
    dp_rec_L5 = h.Vector()
    dp_rec_L5.record(h._ref_dp_total_L5)  # L5 dipole recording

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
    _PC.allreduce(dp_rec_L2, 1)
    # combine dp_rec on every node, 1=add contributions together
    _PC.allreduce(dp_rec_L5, 1)
    # aggregate the currents independently on each proc
    neuron_net.aggregate_currents()
    # combine neuron_net.current{} variables from each proc
    _PC.allreduce(neuron_net.current['L5Pyr_soma'], 1)
    _PC.allreduce(neuron_net.current['L2Pyr_soma'], 1)

    # combine spiking data from each proc
    spiketimes_list = _PC.py_gather(neuron_net._spiketimes, 0)
    spikegids_list = _PC.py_gather(neuron_net._spikegids, 0)
    # only rank 0's lists are complete

    if rank == 0:
        for spike_vec in spiketimes_list:
            neuron_net._all_spiketimes.append(spike_vec)
        for spike_vec in spikegids_list:
            neuron_net._all_spikegids.append(spike_vec)

    _PC.barrier()  # get all nodes to this place before continuing

    dpl_data = np.c_[np.array(dp_rec_L2.to_python()) +
                     np.array(dp_rec_L5.to_python()),
                     np.array(dp_rec_L2.to_python()),
                     np.array(dp_rec_L5.to_python())]

    dpl = Dipole(np.array(t_vec.to_python()), dpl_data)
    if rank == 0:
        if neuron_net.net.params['save_dpl']:
            dpl.write('rawdpl.txt')

        dpl.baseline_renormalize(neuron_net.net.params)
        dpl.convert_fAm_to_nAm()
        dpl.scale(neuron_net.net.params['dipole_scalefctr'])
        dpl.smooth(neuron_net.net.params['dipole_smooth_win'] / h.dt)

    neuron_net.net.trial_idx += 1

    return dpl


def load_custom_mechanisms():
    import platform
    import os.path as op

    global _LOADED_DLL

    if _LOADED_DLL is not None:
        return

    if platform.system() == 'Windows':
        mech_fname = op.join(op.dirname(__file__), 'mod', 'nrnmech.dll')
    else:
        mech_fname = op.join(op.dirname(__file__), 'mod', 'x86_64',
                             '.libs', 'libnrnmech.so')
    h.nrn_load_dll(mech_fname)
    _LOADED_DLL = mech_fname

    if _get_rank() == 0:
        print('Loading custom mechanism files from %s' % mech_fname)

    return


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


class NeuronNetwork(object):
    """The NeuronNetwork class.

    Parameters
    ----------
    net : Network object
        The instance of Network to instantiate in NEURON-Python

    Attributes
    ----------
    params : dict
        The parameters
    cells : list of Cell objects.
        The list of cells
    gid_dict : dict
        Dictionary with keys 'evprox1', 'evdist1' etc.
        containing the range of Cell IDs of different cell
        (or input) types.

    Notes
    -----
    NeuronNetwork is not a pickleable class because it contains many NEURON
    objects once it has been instantiated. This is important for the Joblib
    backend that passes a pickled Network object to each forked process (job)
    and only instantiates NeuronNetwork after the fork.

    The `_build` routine can be called again to run more simulations without
    creating new `nrniv` processes. Instead, the NERUON objects are recreated
    and gids are reassigned according to the specifications in
    `self.net.params` and the network is ready for another simulation.
    """

    def __init__(self, net):
        self.net = net

        # create cells (and create self.origin in create_cells_pyr())
        self.cells = []

        # artificial cells must be appended to a list in order to preserve
        # the NEURON hoc objects and the corresonding python references
        # initialized by _ArtificialCell()
        self._feed_cells = []
        self._build()

    def _build(self):
        """Building the network in NEURON."""

        _create_parallel_context()

        # load mechanisms needs ParallelContext for get_rank
        load_custom_mechanisms()

        if _get_rank() == 0:
            print('Building the NEURON model')

        self._clear_last_network_objects()

        self._gid_assign()

        # Create a h.Vector() with size 1xself.N_t, zero'd
        self.current = {
            'L5Pyr_soma': h.Vector(self.net.n_times, 0),
            'L2Pyr_soma': h.Vector(self.net.n_times, 0),
        }

        self._create_cells_and_feeds()
        self.state_init()
        self._parnet_connect()

        # set to record spikes
        self._spiketimes = h.Vector()
        self._spikegids = h.Vector()

        # used by rank 0 for spikes across all procs (MPI)
        self._all_spiketimes = h.Vector()
        self._all_spikegids = h.Vector()

        self._record_spikes()
        self.move_cells_to_pos()  # position cells in 2D grid

        if _get_rank() == 0:
            print('[Done]')

    def __enter__(self):
        """Context manager to cleanly build NeuronNetwork objects"""
        return self

    def __exit__(self, cell_type, value, traceback):
        """Clear up NEURON internal gid information."""

        self._clear_neuron_objects()
        if _LAST_NETWORK is not None:
            _LAST_NETWORK._clear_neuron_objects()

    # this happens on EACH node
    # creates self.net._gid_list for THIS node
    def _gid_assign(self):

        rank = _get_rank()
        nhosts = _get_nhosts()

        # round robin assignment of gids
        for gid in range(rank, self.net.n_cells, nhosts):
            # set the cell gid
            _PC.set_gid2node(gid, rank)
            self.net._gid_list.append(gid)
            # now to do the cell-specific external input gids on the same proc
            # these are guaranteed to exist because all of
            # these inputs were created for each cell
            for key in self.net.p_unique.keys():
                gid_input = gid + self.net.gid_dict[key][0]
                _PC.set_gid2node(gid_input, rank)
                self.net._gid_list.append(gid_input)

        for gid_base in range(rank, self.net.n_common_feeds, nhosts):
            # shift the gid_base to the common gid
            gid = gid_base + self.net.gid_dict['common'][0]
            # set as usual
            _PC.set_gid2node(gid, rank)
            self.net._gid_list.append(gid)
        # extremely important to get the gids in the right order
        self.net._gid_list.sort()

    def _create_cells_and_feeds(self):
        """Parallel create cells AND external inputs (feeds)

        These feeds are spike SOURCES but cells are also targets.
        External inputs are not targets.
        """
        params = self.net.params

        # loop through gids on this node
        for gid in self.net._gid_list:

            src_type, src_pos, is_cell = self.net._get_src_type_and_pos(gid)

            # check existence of gid with Neuron
            if not _PC.gid_exists(gid):
                msg = ('Source of type %s with ID %d does not exists in '
                       'Network' % (src_type, gid))
                raise RuntimeError(msg)

            if is_cell:  # not a feed
                # figure out which cell type is assoc with the gid
                # create cells based on loc property
                # creates a NetCon object internally to Neuron
                type2class = {'L2_pyramidal': L2Pyr, 'L5_pyramidal': L5Pyr,
                              'L2_basket': L2Basket, 'L5_basket': L5Basket}
                Cell = type2class[src_type]
                if src_type in ('L2_pyramidal', 'L5_pyramidal'):
                    cell = Cell(gid, src_pos, params)
                else:
                    cell = Cell(gid, src_pos)
                self.cells.append(cell)

            # external inputs are special types of artificial-cells
            # 'common': all cells impacted with identical TIMING of spike
            # events. NB: cell types can still have different weights for how
            # such 'common' spikes influence them
            elif src_type == 'common':
                # print('cell_type',cell_type)
                # to find param index, take difference between REAL gid
                # here and gid start point of the items
                p_ind = gid - self.net.gid_dict['common'][0]

                # new ExtFeed: target cell type irrelevant (None) since input
                # timing will be identical for all cells
                feed = ExtFeed(feed_type=src_type,
                               target_cell_type=None,
                               params=self.net.p_common[p_ind],
                               gid=gid)
                feed_cell = _ArtificialCell(feed.event_times,
                                            params['threshold'])
                self._feed_cells.append(feed_cell)

            # external inputs can also be Poisson- or Gaussian-
            # distributed, or 'evoked' inputs (proximal or distal)
            # these are cell-specific ('unique')
            elif src_type in self.net.p_unique.keys():
                gid_target = gid - self.net.gid_dict[src_type][0]
                target_cell_type = self.net.gid_to_type(gid_target)

                # new ExtFeed, where now both feed type and target cell type
                # specified because these feeds have cell-specific parameters
                feed = ExtFeed(feed_type=src_type,
                               target_cell_type=target_cell_type,
                               params=self.net.p_unique[src_type],
                               gid=gid)
                feed_cell = _ArtificialCell(feed.event_times,
                                            params['threshold'])
                self._feed_cells.append(feed_cell)
            else:
                raise ValueError('No parameters specified for external feed '
                                 'type: %s' % src_type)

            # Now let's associate the cell objects with a netcon
            if is_cell:
                nrn_netcon = cell.setup_source_netcon(params['threshold'])
                _PC.cell(gid, nrn_netcon)
            else:
                _PC.cell(gid, feed_cell.nrn_netcon)

    # connections:
    # this NODE is aware of its cells as targets
    # for each syn, return list of source GIDs.
    # for each item in the list, do a:
    # nc = pc.gid_connect(source_gid, target_syn), weight,delay
    # Both for synapses AND for external inputs
    def _parnet_connect(self):

        # loop over target zipped gids and cells
        for gid, cell in zip(self.net._gid_list, self.cells):
            # ignore iteration over inputs, since they are NOT targets
            if _PC.gid_exists(gid) and self.net.gid_to_type(gid) != 'common':
                # for each gid, find all the other cells connected to it,
                # based on gid
                # this MUST be defined in EACH class of cell in self.cells
                # parconnect receives connections from other cells
                # parreceive receives connections from common external inputs
                cell.parconnect(gid, self.net.gid_dict, self.net.pos_dict,
                                self.net.params)
                cell.parreceive(gid, self.net.gid_dict,
                                self.net.pos_dict, self.net.p_common)
                # now do the unique external feeds specific to these cells
                # parreceive_ext receives connections from UNIQUE
                # external inputs
                for cell_type in self.net.p_unique.keys():
                    p_type = self.net.p_unique[cell_type]
                    cell.parreceive_ext(
                        cell_type, gid, self.net.gid_dict, self.net.pos_dict,
                        p_type)

    # setup spike recording for this node
    def _record_spikes(self):

        # iterate through gids on this node and
        # set to record spikes in spike time vec and id vec
        # agnostic to type of source, will sort that out later
        for gid in self.net._gid_list:
            if _PC.gid_exists(gid):
                _PC.spike_record(gid, self._spiketimes, self._spikegids)

    # aggregate recording all the somatic voltages for pyr
    def aggregate_currents(self):
        """This method must be run post-integration."""
        # this is quite ugly
        for cell in self.cells:
            # check for celltype
            if cell.celltype in ('L5_pyramidal', 'L2_pyramidal'):
                # iterate over somatic currents, assumes this list exists
                # is guaranteed in L5Pyr()
                for key, I_soma in cell.dict_currents.items():
                    # self.current_L5Pyr_soma was created upon
                    # in parallel, each node has its own Net()
                    self.current['%s_soma' % cell.name].add(I_soma)

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
        for gid, cell in zip(self.net._gid_list, self.cells):
            # only work on cells on this node
            if _PC.gid_exists(gid):
                for name_src in ['L2Pyr', 'L2Basket', 'L5Pyr', 'L5Basket',
                                 'extinput', 'extgauss', 'extpois', 'ev']:
                    for nc in getattr(cell, 'ncfrom_%s' % name_src):
                        if nc.valid():
                            # delete NEURON cell object
                            cell_obj1 = nc.precell(gid)
                            if cell_obj1 is not None:
                                del cell_obj1
                            cell_obj2 = nc.postcell(gid)
                            if cell_obj2 is not None:
                                del cell_obj2
                            del nc

        self.net._gid_list = []
        self.cells = []

    def get_data_from_neuron(self):
        """Get copies of spike data that are pickleable"""

        from copy import deepcopy
        data = (self._all_spiketimes.to_python(),
                self._all_spikegids.to_python(),
                deepcopy(self.net.gid_dict))
        return data

    def _clear_last_network_objects(self):
        """Clears NEURON objects and saves the current Network instance"""

        global _LAST_NETWORK

        if _LAST_NETWORK is not None:
            _LAST_NETWORK._clear_neuron_objects()

        self._clear_neuron_objects()
        _LAST_NETWORK = self
