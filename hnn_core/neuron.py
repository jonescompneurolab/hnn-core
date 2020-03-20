"""Neuron Network class."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>

import itertools as it
import numpy as np

from .feed import ExtFeed
from .pyramidal import L2Pyr, L5Pyr
from .basket import L2Basket, L5Basket
from .params import create_pext

# We need to maintain a reference to the last
# _neuron_network instance that ran pc.gid_clear(). Even if
# pc is global, if pc.gid_clear() is called within a new
# _neuron_network, it will seg fault.
_last_network = None


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
        containing the range of Cell IDs of different cell types.
    ext_list : dictionary of list of ExtFeed.
        Keys are:
            'evprox1', 'evprox2', etc.
            'evdist1', etc.
            'extgauss', 'extpois'
    spiketimes : tuple (n_trials, ) of list of float
        Each element of the tuple is a trial.
        The list contains the time stamps of spikes.
    spikegids : tuple (n_trials, ) of list of float
        Each element of the tuple is a trial.
        The list contains the cell IDs of neurons that spiked.
    """

    def __init__(self, params):

        # set the params internally for this net
        # better than passing it around like ...
        self.params = params
        # Number of time points
        # Originally used to create the empty vec for synaptic currents,
        # ensuring that they exist on this node irrespective of whether
        # or not cells of relevant type actually do

        self.N_t = np.arange(0., self.params['tstop'],
                             self.params['dt']).size + 1

        # int variables for grid of pyramidal cells (for now in both L2 and L5)
        self.gridpyr = {
            'x': self.params['N_pyr_x'],
            'y': self.params['N_pyr_y'],
        }
        self.N_src = 0
        self.N = {}  # numbers of sources
        self.N_cells = 0  # init self.N_cells
        # zdiff is expressed as a positive DEPTH of L5 relative to L2
        # this is a deviation from the original, where L5 was defined at 0
        # this should not change interlaminar weight/delay calculations
        self.zdiff = 1307.4
        # params of external inputs in p_ext
        # Global number of external inputs ... automatic counting
        # makes more sense
        # p_unique represent ext inputs that are going to go to each cell
        self.p_ext, self.p_unique = create_pext(self.params,
                                                self.params['tstop'])
        self.N_extinput = len(self.p_ext)
        # Source list of names
        # in particular order (cells, extinput, alpha names of unique inputs)
        self.src_list_new = self._create_src_list()
        # cell position lists, also will give counts: must be known
        # by ALL nodes
        # extinput positions are all located at origin.
        # sort of a hack bc of redundancy
        self.pos_dict = dict.fromkeys(self.src_list_new)
        # create coords in pos_dict for all cells first
        self._create_coords_pyr()
        self._create_coords_basket()
        self._count_cells()
        # create coords for all other sources
        self._create_coords_extinput()
        # count external sources
        self._count_extsrcs()
        # create dictionary of GIDs according to cell type
        # global dictionary of gid and cell type
        self.gid_dict = {}
        self._create_gid_dict()
        # assign gid to hosts, creates list of gids for this node in _gid_list
        # _gid_list length is number of cells assigned to this id()
        self._gid_list = []
        # create cells (and create self.origin in create_cells_pyr())
        self.cells = []
        self.extinput_list = []
        # external unique input list dictionary
        self.ext_list = dict.fromkeys(self.p_unique)
        # initialize the lists in the dict
        for key in self.ext_list.keys():
            self.ext_list[key] = []

        self.trial_idx = 0
        self.build()

    def build(self):
        """Building the network in NEURON."""

        from neuron import h
        from .parallel import create_parallel_context, get_rank
        from .utils import load_custom_mechanisms

        create_parallel_context()

        # load mechanisms needs ParallelContext for get_rank
        load_custom_mechanisms()

        if get_rank() == 0:
            print('Building the NEURON model')

        self._clear_last_network_objects()

        self._gid_assign()
        # Create a h.Vector() with size 1xself.N_t, zero'd
        self.current = {
            'L5Pyr_soma': h.Vector(self.N_t, 0),
            'L2Pyr_soma': h.Vector(self.N_t, 0),
        }

        self._create_all_src()
        self.state_init()
        self._parnet_connect()

        # set to record spikes
        self.spiketimes = h.Vector()
        self.spikegids = h.Vector()
        self._record_spikes()
        self.move_cells_to_pos()  # position cells in 2D grid

        if get_rank() == 0:
            print('[Done]')

    def __enter__(self):
        """Context manager to cleanly build Network objects"""
        return self

    def __exit__(self, type, value, traceback):
        """Clear up NEURON internal gid information.

        Notes
        -----
        This function must be called from the context of the
        Network instance that ran __enter__(). This is a bug or
        peculiarity of NEURON. If this function is called from a different
        context, then the next simulation will run very slow because nrniv
        workers are still going for the old simulation. If pc.gid_clear() is
        called from the right context, then those workers can exit.
        """
        from .parallel import pc
        pc.gid_clear()

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
        # add the legacy extinput here
        self.extname_list = []
        self.extname_list.append('extinput')
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

    # creates origin AND creates external input coords
    def _create_coords_extinput(self):
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
        self.pos_dict['extinput'] = [self.origin for i in
                                     range(self.N_extinput)]
        # at this time, each of the unique inputs is per cell
        for key in self.p_unique.keys():
            # create the pos_dict for all the sources
            self.pos_dict[key] = [self.origin for i in range(self.N_cells)]

    def _count_cells(self):
        """Cell counting routine."""
        # cellname list is used *only* for this purpose for now
        for src in self.cellname_list:
            # if it's a cell, then add the number to total number of cells
            self.N[src] = len(self.pos_dict[src])
            self.N_cells += self.N[src]

    # general counting method requires pos_dict is correct for each source
    # and that all sources are represented
    def _count_extsrcs(self):
        # all src numbers are based off of length of pos_dict entry
        # generally done here in lieu of upstream changes
        for src in self.extname_list:
            self.N[src] = len(self.pos_dict[src])

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
            gid_ind.append(gid_ind[i] + self.N[src])
            # accumulate total source count
            self.N_src += self.N[src]
        # now actually assign the ranges
        for i in range(len(self.src_list_new)):
            src = self.src_list_new[i]
            self.gid_dict[src] = range(gid_ind[i], gid_ind[i + 1])

    # this happens on EACH node
    # creates self._gid_list for THIS node
    def _gid_assign(self):

        from .parallel import get_rank, get_nhosts, pc

        rank = get_rank()
        nhosts = get_nhosts()

        # round robin assignment of gids
        for gid in range(rank, self.N_cells, nhosts):
            # set the cell gid
            pc.set_gid2node(gid, rank)
            self._gid_list.append(gid)
            # now to do the cell-specific external input gids on the same proc
            # these are guaranteed to exist because all of
            # these inputs were created for each cell
            for key in self.p_unique.keys():
                gid_input = gid + self.gid_dict[key][0]
                pc.set_gid2node(gid_input, rank)
                self._gid_list.append(gid_input)
        # legacy handling of the external inputs
        # NOT perfectly balanced for now
        for gid_base in range(rank, self.N_extinput, nhosts):
            # shift the gid_base to the extinput gid
            gid = gid_base + self.gid_dict['extinput'][0]
            # set as usual
            pc.set_gid2node(gid, rank)
            self._gid_list.append(gid)
        # extremely important to get the gids in the right order
        self._gid_list.sort()

    def gid_to_type(self, gid):
        """Reverse lookup of gid to type."""
        for gidtype, gids in self.gid_dict.items():
            if gid in gids:
                return gidtype

    def _create_all_src(self):
        """Parallel create cells AND external inputs (feeds)
           these are spike SOURCES but cells are also targets
           external inputs are not targets.
        """

        from .parallel import pc

        # loop through gids on this node
        for gid in self._gid_list:
            # check existence of gid with Neuron
            if pc.gid_exists(gid):
                # get type of cell and pos via gid
                # now should be valid for ext inputs
                type = self.gid_to_type(gid)
                type_pos_ind = gid - self.gid_dict[type][0]
                pos = self.pos_dict[type][type_pos_ind]
                # figure out which cell type is assoc with the gid
                # create cells based on loc property
                # creates a NetCon object internally to Neuron
                type2class = {'L2_pyramidal': L2Pyr, 'L5_pyramidal': L5Pyr,
                              'L2_basket': L2Basket, 'L5_basket': L5Basket}
                if type in ('L2_pyramidal', 'L5_pyramidal', 'L2_basket',
                            'L5_basket'):
                    Cell = type2class[type]
                    if type in ('L2_pyramidal', 'L5_pyramidal'):
                        self.cells.append(Cell(gid, pos, self.params))
                    else:
                        self.cells.append(Cell(gid, pos))
                    pc.cell(
                        gid, self.cells[-1].connect_to_target(
                            None, self.params['threshold']))
                elif type == 'extinput':
                    # print('type',type)
                    # to find param index, take difference between REAL gid
                    # here and gid start point of the items
                    p_ind = gid - self.gid_dict['extinput'][0]
                    # now use the param index in the params and create
                    # the cell and artificial NetCon
                    self.extinput_list.append(ExtFeed(
                        type, None, self.p_ext[p_ind], gid))
                    pc.cell(
                        gid, self.extinput_list[-1].connect_to_target(
                            self.params['threshold']))
                elif type in self.p_unique.keys():
                    gid_post = gid - self.gid_dict[type][0]
                    cell_type = self.gid_to_type(gid_post)
                    # create dictionary entry, append to list
                    self.ext_list[type].append(ExtFeed(
                        type, cell_type, self.p_unique[type], gid))
                    pc.cell(
                        gid, self.ext_list[type][-1].connect_to_target(
                            self.params['threshold']))
                else:
                    print("None of these types in Net()")
                    exit()
            else:
                print("None of these types in Net()")
                exit()

    # connections:
    # this NODE is aware of its cells as targets
    # for each syn, return list of source GIDs.
    # for each item in the list, do a:
    # nc = pc.gid_connect(source_gid, target_syn), weight,delay
    # Both for synapses AND for external inputs
    def _parnet_connect(self):
        from .parallel import pc

        # loop over target zipped gids and cells
        # cells has NO extinputs anyway. also no extgausses
        for gid, cell in zip(self._gid_list, self.cells):
            # ignore iteration over inputs, since they are NOT targets
            if pc.gid_exists(gid) and self.gid_to_type(gid) \
                    != 'extinput':
                # for each gid, find all the other cells connected to it,
                # based on gid
                # this MUST be defined in EACH class of cell in self.cells
                # parconnect receives connections from other cells
                # parreceive receives connections from external inputs
                cell.parconnect(gid, self.gid_dict, self.pos_dict, self.params)
                cell.parreceive(gid, self.gid_dict, self.pos_dict, self.p_ext)
                # now do the unique inputs specific to these cells
                # parreceive_ext receives connections from UNIQUE
                # external inputs
                for type in self.p_unique.keys():
                    p_type = self.p_unique[type]
                    cell.parreceive_ext(
                        type, gid, self.gid_dict, self.pos_dict, p_type)

    # setup spike recording for this node
    def _record_spikes(self):
        from .parallel import pc

        # iterate through gids on this node and
        # set to record spikes in spike time vec and id vec
        # agnostic to type of source, will sort that out later
        for gid in self._gid_list:
            if pc.gid_exists(gid):
                pc.spike_record(gid, self.spiketimes, self.spikegids)

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
        from neuron import h

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
        from .parallel import pc

        pc.gid_clear()

        # dereference cell and NetConn objects
        for gid, cell in zip(self._gid_list, self.cells):
            # only work on cells on this node
            if pc.gid_exists(gid):
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
        data = (self.spiketimes.to_python(), self.spikegids.to_python(),
                deepcopy(self.gid_dict))
        return data

    def _clear_last_network_objects(self):
        """Clears NEURON objects and saves the current Network instance"""

        global _last_network

        if _last_network is not None:
            _last_network._clear_neuron_objects()

        self._clear_neuron_objects()
        _last_network = self
