"""Neuron simulation functions and _neuron_network class."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>

import itertools as it
import numpy as np
from neuron import h

from .feed import ExtFeed
from .pyramidal import L2Pyr, L5Pyr
from .basket import L2Basket, L5Basket
from .params import create_pext
from .network import Spikes

# a few globals
_pc = None
_cvode = None

# We need to maintain a reference to the last
# _neuron_network instance that ran pc.gid_clear(). Even if
# pc is global, if pc.gid_clear() is called within a new
# _neuron_network, it will seg fault.
_last_network = None

# NEURON only allows mechanisms to be loaded once (per Python interpreter)
_loaded_dll = None


def _simulate_single_trial(neuron_net):
    """Simulate one trial."""

    from .dipole import Dipole

    global _pc, _cvode

    h.load_file("stdrun.hoc")

    rank = _get_rank()
    nhosts = _get_nhosts()

    # Now let's simulate the dipole

    _pc.barrier()  # sync for output to screen
    if rank == 0:
        print("running trial %d on %d cores" %
              (neuron_net.trial_idx, nhosts))

    # create or reinitialize scalars in NEURON (hoc) context
    h("dp_total_L2 = 0.")
    h("dp_total_L5 = 0.")

    # Set tstop before instantiating any classes
    h.tstop = neuron_net.params['tstop']
    h.dt = neuron_net.params['dt']  # simulation duration and time-step
    h.celsius = neuron_net.params['celsius']  # 37.0 - set temperature

    # We define the arrays (Vector in numpy) for recording the signals
    t_vec = h.Vector()
    t_vec.record(h._ref_t)  # time recording
    dp_rec_L2 = h.Vector()
    dp_rec_L2.record(h._ref_dp_total_L2)  # L2 dipole recording
    dp_rec_L5 = h.Vector()
    dp_rec_L5.record(h._ref_dp_total_L5)  # L5 dipole recording

    # sets the default max solver step in ms (purposefully large)
    _pc.set_maxstep(10)

    # initialize cells to -65 mV, after all the NetCon
    # delays have been specified
    h.finitialize()

    def simulation_time():
        print('Simulation time: {0} ms...'.format(round(h.t, 2)))

    if rank == 0:
        for tt in range(0, int(h.tstop), 10):
            _cvode.event(tt, simulation_time)

    h.fcurrent()

    # initialization complete, but wait for all procs to start the solver
    _pc.barrier()

    # actual simulation - run the solver
    _pc.psolve(h.tstop)

    _pc.barrier()

    # these calls aggregate data across procs/nodes
    _pc.allreduce(dp_rec_L2, 1)
    # combine dp_rec on every node, 1=add contributions together
    _pc.allreduce(dp_rec_L5, 1)
    # aggregate the currents independently on each proc
    neuron_net.aggregate_currents()
    # combine net.current{} variables on each proc
    _pc.allreduce(neuron_net.current['L5Pyr_soma'], 1)
    _pc.allreduce(neuron_net.current['L2Pyr_soma'], 1)

    _pc.barrier()  # get all nodes to this place before continuing

    dpl_data = np.c_[np.array(dp_rec_L2.to_python()) +
                     np.array(dp_rec_L5.to_python()),
                     np.array(dp_rec_L2.to_python()),
                     np.array(dp_rec_L5.to_python())]

    dpl = Dipole(np.array(t_vec.to_python()), dpl_data)
    if rank == 0:
        if neuron_net.params['save_dpl']:
            dpl.write('rawdpl.txt')

        dpl.baseline_renormalize(neuron_net.params)
        dpl.convert_fAm_to_nAm()
        dpl.scale(neuron_net.params['dipole_scalefctr'])
        dpl.smooth(neuron_net.params['dipole_smooth_win'] / h.dt)

    neuron_net.trial_idx += 1

    return dpl


def load_custom_mechanisms():
    import platform
    import os.path as op

    global _loaded_dll

    if _loaded_dll is not None:
        return

    if platform.system() == 'Windows':
        mech_fname = op.join(op.dirname(__file__), '..', 'mod', 'nrnmech.dll')
    else:
        mech_fname = op.join(op.dirname(__file__), '..', 'mod', 'x86_64',
                             '.libs', 'libnrnmech.so')
    h.nrn_load_dll(mech_fname)
    _loaded_dll = mech_fname

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
    if _pc is not None:
        return int(_pc.nhost())
    else:
        return 1


def _get_rank():
    """Return the MPI rank from ParallelContext

    Returns
    -------
    rank: int
        Value from pc.id()
    """
    if _pc is not None:
        return int(_pc.id())
    else:
        return 0


def _create_parallel_context(n_cores=None):
    """Create parallel context.

    Parameters
    ----------
    n_cores: int | None
        Number of processors to use for a simulation. A value of None will
        allow NEURON to use all available processors.
    """

    global _cvode, _pc

    if _pc is None:
        if n_cores is None:
            # MPI: Initialize the ParallelContext class
            _pc = h.ParallelContext()
        else:
            _pc = h.ParallelContext(n_cores)

        _cvode = h.CVode()

        # be explicit about using fixed step integration
        _cvode.active(0)

        # use cache_efficient mode for allocating elements in contiguous order
        # cvode.cache_efficient(1)
    else:
        # ParallelContext() has already been called. Don't start more workers.
        # Just tell old nrniv workers to quit.
        _pc.done()


class _neuron_network(object):
    """The Neuron Network class.

    Parameters
    ----------
    params : dict
        The parameters

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
    extfeed_list : dictionary of list of ExtFeed.
        Keys are:
            'evprox1', 'evprox2', etc.
            'evdist1', etc.
            'extgauss', 'extpois'
    spikes : Spikes
        An instance of the Spikes object.
    """

    def __init__(self, params):

        # set the params internally for this net
        # better than passing it around like ...
        self.params = params
        # Number of time points
        # Originally used to create the empty vec for synaptic currents,
        # ensuring that they exist on this node irrespective of whether
        # or not cells of relevant type actually do

        self.n_times = np.arange(0., self.params['tstop'],
                                 self.params['dt']).size + 1

        # int variables for grid of pyramidal cells (for now in both L2 and L5)
        self.gridpyr = {
            'x': self.params['N_pyr_x'],
            'y': self.params['N_pyr_y'],
        }
        self.n_src = 0
        self.n_of_type = {}  # numbers of sources
        self.n_cells = 0  # init self.n_cells
        # zdiff is expressed as a positive DEPTH of L5 relative to L2
        # this is a deviation from the original, where L5 was defined at 0
        # this should not change interlaminar weight/delay calculations
        self.zdiff = 1307.4
        # params of common external feeds inputs in p_common
        # Global number of external inputs ... automatic counting
        # makes more sense
        # p_unique represent ext inputs that are going to go to each cell
        self.p_common, self.p_unique = create_pext(self.params,
                                                   self.params['tstop'])
        self.n_common_feeds = len(self.p_common)
        # Source list of names
        # in particular order (cells, common, names of unique inputs)
        self.src_list_new = self._create_src_list()
        # cell position lists, also will give counts: must be known
        # by ALL nodes
        # common positions are all located at origin.
        # sort of a hack bc of redundancy
        self.pos_dict = dict.fromkeys(self.src_list_new)
        # create coords in pos_dict for all cells first
        self._create_coords_pyr()
        self._create_coords_basket()
        self._count_cells()
        # create coords for all other sources
        self._create_coords_common_feeds()
        # count external sources
        self._count_extsrcs()
        # create dictionary of GIDs according to cell type
        # global dictionary of gid and cell type
        self.gid_dict = {}
        self._create_gid_dict()
        # Create empty spikes object
        self.spikes = Spikes()
        # assign gid to hosts, creates list of gids for this node in _gid_list
        # _gid_list length is number of cells assigned to this id()
        self._gid_list = []
        # create cells (and create self.origin in create_cells_pyr())
        self.cells = []
        self.common_feeds = []
        # external unique input list dictionary
        self.unique_feeds = dict.fromkeys(self.p_unique)
        # initialize the lists in the dict
        for key in self.unique_feeds.keys():
            self.unique_feeds[key] = []

        self.trial_idx = 0
        self._build()

    def __repr__(self):
        class_name = self.__class__.__name__
        s = ("%d x %d Pyramidal cells (L2, L5)"
             % (self.gridpyr['x'], self.gridpyr['y']))
        s += ("\n%d L2 basket cells\n%d L5 basket cells"
              % (self.n_of_type['L2_basket'], self.n_of_type['L5_basket']))
        return '<%s | %s>' % (class_name, s)

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
            'L5Pyr_soma': h.Vector(self.n_times, 0),
            'L2Pyr_soma': h.Vector(self.n_times, 0),
        }

        self._create_all_spike_sources()
        self.state_init()
        self._parnet_connect()

        # set to record spikes
        self.spikes._times = h.Vector()
        self.spikes._gids = h.Vector()
        self._record_spikes()
        self.move_cells_to_pos()  # position cells in 2D grid

        if _get_rank() == 0:
            print('[Done]')

    def __enter__(self):
        """Context manager to cleanly build _neuron_network objects"""
        return self

    def __exit__(self, cell_type, value, traceback):
        """Clear up NEURON internal gid information."""

        self._clear_neuron_objects()
        if _last_network is not None:
            _last_network._clear_neuron_objects()

    # creates the immutable source list along with corresponding numbers
    # of cells
    def _create_src_list(self):
        # base source list of tuples, name and number, in this order
        self.cellname_list = [
            'L2_basket',
            'L2_pyramidal',
            'L5_basket',
            'L5_pyramidal',
        ]
        self.extname_list = []
        self.extname_list.append('common')
        # grab the keys for the unique set of inputs and sort the names
        # append them to the src list along with the number of cells
        unique_keys = sorted(self.p_unique.keys())
        self.extname_list += unique_keys
        # return one final source list
        src_list = self.cellname_list + self.extname_list
        return src_list

    # Creates cells and grid
    def _create_coords_pyr(self):
        """ pyr grid is the immutable grid, origin now calculated in relation to feed
        """
        xrange = np.arange(self.gridpyr['x'])
        yrange = np.arange(self.gridpyr['y'])
        # create list of tuples/coords, (x, y, z)
        self.pos_dict['L2_pyramidal'] = [
            pos for pos in it.product(xrange, yrange, [0])]
        self.pos_dict['L5_pyramidal'] = [
            pos for pos in it.product(xrange, yrange, [self.zdiff])]

    def _create_coords_basket(self):
        """Create basket cell coords based on pyr grid."""
        # define relevant x spacings for basket cells
        xzero = np.arange(0, self.gridpyr['x'], 3)
        xone = np.arange(1, self.gridpyr['x'], 3)
        # split even and odd y vals
        yeven = np.arange(0, self.gridpyr['y'], 2)
        yodd = np.arange(1, self.gridpyr['y'], 2)
        # create general list of x,y coords and sort it
        coords = [pos for pos in it.product(
            xzero, yeven)] + [pos for pos in it.product(xone, yodd)]
        coords_sorted = sorted(coords, key=lambda pos: pos[1])
        # append the z value for position for L2 and L5
        # print(len(coords_sorted))
        self.pos_dict['L2_basket'] = [pos_xy + (0,) for
                                      pos_xy in coords_sorted]
        self.pos_dict['L5_basket'] = [
            pos_xy + (self.zdiff,) for pos_xy in coords_sorted]

    # creates origin AND creates common feed input coords
    def _create_coords_common_feeds(self):
        """ (same thing for now but won't fix because could change)
        """
        xrange = np.arange(self.gridpyr['x'])
        yrange = np.arange(self.gridpyr['y'])
        # origin's z component isn't really used in
        # calculating distance functions from origin
        # these will be forced as ints!
        origin_x = xrange[int((len(xrange) - 1) // 2)]
        origin_y = yrange[int((len(yrange) - 1) // 2)]
        origin_z = np.floor(self.zdiff / 2)
        self.origin = (origin_x, origin_y, origin_z)
        self.pos_dict['common'] = [self.origin for i in
                                   range(self.n_common_feeds)]
        # at this time, each of the unique inputs is per cell
        for key in self.p_unique.keys():
            # create the pos_dict for all the sources
            self.pos_dict[key] = [self.origin for i in range(self.n_cells)]

    def _count_cells(self):
        """Cell counting routine."""
        # cellname list is used *only* for this purpose for now
        for src in self.cellname_list:
            # if it's a cell, then add the number to total number of cells
            self.n_of_type[src] = len(self.pos_dict[src])
            self.n_cells += self.n_of_type[src]

    # general counting method requires pos_dict is correct for each source
    # and that all sources are represented
    def _count_extsrcs(self):
        # all src numbers are based off of length of pos_dict entry
        # generally done here in lieu of upstream changes
        for src in self.extname_list:
            self.n_of_type[src] = len(self.pos_dict[src])

    def _create_gid_dict(self):
        """Creates gid dicts and pos_lists."""
        # initialize gid index gid_ind to start at 0
        gid_ind = [0]
        # append a new gid_ind based on previous and next cell count
        # order is guaranteed by self.src_list_new
        for i in range(len(self.src_list_new)):
            # N = self.src_list_new[i][1]
            # grab the src name in ordered list src_list_new
            src = self.src_list_new[i]
            # query the N dict for that number and append here
            # to gid_ind, based on previous entry
            gid_ind.append(gid_ind[i] + self.n_of_type[src])
            # accumulate total source count
            self.n_src += self.n_of_type[src]
        # now actually assign the ranges
        for i in range(len(self.src_list_new)):
            src = self.src_list_new[i]
            self.gid_dict[src] = range(gid_ind[i], gid_ind[i + 1])

    # this happens on EACH node
    # creates self._gid_list for THIS node
    def _gid_assign(self):

        rank = _get_rank()
        nhosts = _get_nhosts()

        # round robin assignment of gids
        for gid in range(rank, self.n_cells, nhosts):
            # set the cell gid
            _pc.set_gid2node(gid, rank)
            self._gid_list.append(gid)
            # now to do the cell-specific external input gids on the same proc
            # these are guaranteed to exist because all of
            # these inputs were created for each cell
            for key in self.p_unique.keys():
                gid_input = gid + self.gid_dict[key][0]
                _pc.set_gid2node(gid_input, rank)
                self._gid_list.append(gid_input)

        for gid_base in range(rank, self.n_common_feeds, nhosts):
            # shift the gid_base to the common gid
            gid = gid_base + self.gid_dict['common'][0]
            # set as usual
            _pc.set_gid2node(gid, rank)
            self._gid_list.append(gid)
        # extremely important to get the gids in the right order
        self._gid_list.sort()

    def gid_to_type(self, gid):
        """Reverse lookup of gid to type."""
        for gidtype, gids in self.gid_dict.items():
            if gid in gids:
                return gidtype

    def _get_src_type_and_pos(self, gid):
        """Source type, position and whether it's a cell or artificial feed"""

        # get type of cell and pos via gid
        src_type = self.gid_to_type(gid)
        type_pos_ind = gid - self.gid_dict[src_type][0]
        src_pos = self.pos_dict[src_type][type_pos_ind]

        real_cell_types = ['L2_pyramidal', 'L5_pyramidal',
                           'L2_basket', 'L5_basket']

        return src_type, src_pos, src_type in real_cell_types

    def _create_all_spike_sources(self):
        """Parallel create cells AND external inputs (feeds)
           these are spike SOURCES but cells are also targets
           external inputs are not targets.
        """

        # loop through gids on this node
        for gid in self._gid_list:

            src_type, src_pos, is_cell = self._get_src_type_and_pos(gid)

            # check existence of gid with Neuron
            if not _pc.gid_exists(gid):
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
                    self.cells.append(Cell(gid, src_pos, self.params))
                else:
                    self.cells.append(Cell(gid, src_pos))

                _pc.cell(gid, self.cells[-1].connect_to_target(
                         None, self.params['threshold']))

            # external inputs are special types of artificial-cells
            # 'common': all cells impacted with identical TIMING of spike
            # events. NB: cell types can still have different weights for how
            # such 'common' spikes influence them
            elif src_type == 'common':
                # print('cell_type',cell_type)
                # to find param index, take difference between REAL gid
                # here and gid start point of the items
                p_ind = gid - self.gid_dict['common'][0]

                # new ExtFeed: target cell type irrelevant (None) since input
                # timing will be identical for all cells
                # XXX common_feeds is a list of dict
                self.common_feeds.append(
                    ExtFeed(feed_type=src_type,
                            target_cell_type=None,
                            params=self.p_common[p_ind],
                            gid=gid))

                # create the cell and artificial NetCon
                _pc.cell(gid, self.common_feeds[-1].connect_to_target(
                         self.params['threshold']))

            # external inputs can also be Poisson- or Gaussian-
            # distributed, or 'evoked' inputs (proximal or distal)
            # these are cell-specific ('unique')
            elif src_type in self.p_unique.keys():
                gid_target = gid - self.gid_dict[src_type][0]
                target_cell_type = self.gid_to_type(gid_target)

                # new ExtFeed, where now both feed type and target cell type
                # specified because these feeds have cell-specific parameters
                # XXX unique_feeds is a dict of dict
                self.unique_feeds[src_type].append(
                    ExtFeed(feed_type=src_type,
                            target_cell_type=target_cell_type,
                            params=self.p_unique[src_type],
                            gid=gid))
                _pc.cell(gid,
                         self.unique_feeds[src_type][-1].connect_to_target(
                             self.params['threshold']))
            else:
                raise ValueError('No parameters specified for external feed '
                                 'type: %s' % src_type)

    # connections:
    # this NODE is aware of its cells as targets
    # for each syn, return list of source GIDs.
    # for each item in the list, do a:
    # nc = pc.gid_connect(source_gid, target_syn), weight,delay
    # Both for synapses AND for external inputs
    def _parnet_connect(self):

        # loop over target zipped gids and cells
        for gid, cell in zip(self._gid_list, self.cells):
            # ignore iteration over inputs, since they are NOT targets
            if _pc.gid_exists(gid) and self.gid_to_type(gid) != 'common':
                # for each gid, find all the other cells connected to it,
                # based on gid
                # this MUST be defined in EACH class of cell in self.cells
                # parconnect receives connections from other cells
                # parreceive receives connections from common external inputs
                cell.parconnect(gid, self.gid_dict, self.pos_dict, self.params)
                cell.parreceive(gid, self.gid_dict,
                                self.pos_dict, self.p_common)
                # now do the unique external feeds specific to these cells
                # parreceive_ext receives connections from UNIQUE
                # external inputs
                for cell_type in self.p_unique.keys():
                    p_type = self.p_unique[cell_type]
                    cell.parreceive_ext(
                        cell_type, gid, self.gid_dict, self.pos_dict, p_type)

    # setup spike recording for this node
    def _record_spikes(self):

        # iterate through gids on this node and
        # set to record spikes in spike time vec and id vec
        # agnostic to type of source, will sort that out later
        for gid in self._gid_list:
            if _pc.gid_exists(gid):
                _pc.spike_record(gid, self.spikes._times, self.spikes._gids)

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
        """
        Clear up NEURON internal gid information.

        Note: This function must be called from the context of the
        Network instance that ran build_in_neuron. This is a bug or
        peculiarity of NEURON. If this function is called from a different
        context, then the next simulation will run very slow because nrniv
        workers are still going for the old simulation. If pc.gid_clear is
        called from the right context, then those workers can exit.
        """

        _pc.gid_clear()

        # dereference cell and NetConn objects
        for gid, cell in zip(self._gid_list, self.cells):
            # only work on cells on this node
            if _pc.gid_exists(gid):
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

        self._gid_list = []
        self.cells = []

    def get_data_from_neuron(self):
        """Get copies of the data that are pickleable"""

        from copy import deepcopy
        data = (self.spikes._times.to_python(),
                self.spikes._gids.to_python(),
                deepcopy(self.gid_dict))
        return data

    def _clear_last_network_objects(self):
        """Clears NEURON objects and saves the current Network instance"""

        global _last_network

        if _last_network is not None:
            _last_network._clear_neuron_objects()

        self._clear_neuron_objects()
        _last_network = self
