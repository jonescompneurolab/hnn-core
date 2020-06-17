"""Network class."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>

import itertools as it
import numpy as np
from glob import glob

from .params import create_pext


def read_spikes(fname, gid_dict=None):
    """Read spiking activity from a collection of spike trial files.

    Parameters
    ----------
    fname : str
        Wildcard expression (e.g., '<pathname>/spk_*.txt') of the
        path to the spike file(s).
    gid_dict : dict of lists or range objects | None
        Dictionary with keys 'evprox1', 'evdist1' etc.
        containing the range of Cell or input IDs of different
        cell or input types. If None, each spike file must contain
        a 3rd column for spike type.

    Returns
    ----------
    spikes : Spikes
        An instance of the Spikes object.
    """

    spike_times = []
    spike_gids = []
    spike_types = []
    for file in sorted(glob(fname)):
        spike_trial = np.loadtxt(file, dtype=str)
        spike_times += [list(spike_trial[:, 0].astype(float))]
        spike_gids += [list(spike_trial[:, 1].astype(int))]

        # Note that legacy HNN 'spk.txt' files don't contain a 3rd column for
        # spike type. If reading a legacy version, validate that a gid_dict is
        # provided.
        if spike_trial.shape[1] == 3:
            spike_types += [list(spike_trial[:, 2].astype(str))]
        else:
            if gid_dict is None:
                raise ValueError("gid_dict must be provided if spike types "
                                 "are unspecified in the file %s" % (file,))
            spike_types += [[]]

    spikes = Spikes(times=spike_times, gids=spike_gids, types=spike_types)
    if gid_dict is not None:
        spikes.update_types(gid_dict)

    return Spikes(times=spike_times, gids=spike_gids, types=spike_types)


class Network(object):
    """The Network class.

    Parameters
    ----------
    params : dict
        The parameters

    Attributes
    ----------
    params : dict
        The parameters
    gid_dict : dict
        Dictionary with keys 'evprox1', 'evdist1' etc.
        containing the range of Cell IDs of different cell
        (or input) types.
    spikes : Spikes
        An instance of the Spikes object.
    trial_idx : int
        Current trial number (starting from 0)
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
        self.trial_idx = 0

    def __repr__(self):
        class_name = self.__class__.__name__
        s = ("%d x %d Pyramidal cells (L2, L5)"
             % (self.gridpyr['x'], self.gridpyr['y']))
        s += ("\n%d L2 basket cells\n%d L5 basket cells"
              % (self.n_of_type['L2_basket'], self.n_of_type['L5_basket']))
        return '<%s | %s>' % (class_name, s)

    def _create_src_list(self):
        """ creates the immutable source list along with corresponding numbers of cells
        """
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

    def plot_input(self, ax=None, show=True):
        """Plot the histogram of input.

        Parameters
        ----------
        ax : instance of matplotlib axis | None
            An axis object from matplotlib. If None,
            a new figure is created.
        show : bool
            If True, show the figure.

        Returns
        -------
        fig : instance of matplotlib Figure
            The matplotlib figure handle.
        """
        import matplotlib.pyplot as plt
        spikes = np.array(sum(self.spikes.times, []))
        gids = np.array(sum(self.spikes.gids, []))
        valid_gids = np.r_[[v for (k, v) in self.gid_dict.items()
                            if k.startswith('evprox')]]
        mask_evprox = np.in1d(gids, valid_gids)
        valid_gids = np.r_[[v for (k, v) in self.gid_dict.items()
                            if k.startswith('evdist')]]
        mask_evdist = np.in1d(gids, valid_gids)
        bins = np.linspace(0, self.params['tstop'], 50)

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.hist(spikes[mask_evprox], bins, color='r', label='Proximal')
        ax.hist(spikes[mask_evdist], bins, color='g', label='Distal')
        plt.legend()
        if show:
            plt.show()
        return ax.get_figure()


class Spikes(object):
    """The Spikes class.

    Parameters
    ----------
    times : list (n_trials,) of list (n_spikes,) of float, shape | None
        Each element of the outer list is a trial.
        The inner list contains the time stamps of spikes.
    gids : list (n_trials,) of list (n_spikes,) of float, shape | None
        Each element of the outer list is a trial.
        The inner list contains the cell IDs of neurons that
        spiked.
    types : list (n_trials,) of list (n_spikes,) of float, shape | None
        Each element of the outer list is a trial.
        The inner list contains the type of spike (e.g., evprox1
        or L2_pyramidal) that occured at the corresonding time stamp.
        Each gid corresponds to a type via Network().gid_dict.

    Attributes
    ----------
    times : list (n_trials,) of list (n_spikes,) of float, shape
        Each element of the outer list is a trial.
        The inner list contains the time stamps of spikes.
    gids : list (n_trials,) of list (n_spikes,) of float, shape
        Each element of the outer list is a trial.
        The inner list contains the cell IDs of neurons that
        spiked.
    types : list (n_trials,) of list (n_spikes,) of float, shape
        Each element of the outer list is a trial.
        The inner list contains the type of spike (e.g., evprox1
        or L2_pyramidal) that occured at the corresonding time stamp.
        Each gid corresponds to a type via Network::gid_dict.

    Methods
    -------
    update_types(gid_dict)
        Update spike types in the current instance of Spikes.
    plot(ax=None, show=True)
        Plot and return a matplotlib Figure object showing the
        aggregate network spiking activity according to cell type.
    write(fname)
        Write spiking activity to a collection of spike trial files.
    """

    def __init__(self, times=None, gids=None, types=None):
        if times is None:
            times = list()
        if gids is None:
            gids = list()
        if types is None:
            types = list()

        # Validate arguments
        arg_names = ['times', 'gids', 'types']
        for arg_idx, arg in enumerate([times, gids, types]):
            # Validate outer list
            if not isinstance(arg, list):
                raise TypeError('%s should be a list of lists'
                                % (arg_names[arg_idx],))
            # If arg is not an empty list, validate inner list
            for trial_list in arg:
                if not isinstance(trial_list, list):
                    raise TypeError('%s should be a list of lists'
                                    % (arg_names[arg_idx],))
            # Set the length of 'times' as a references and validate
            # uniform length
            if arg == times:
                n_trials = len(times)
            if len(arg) != n_trials:
                raise ValueError('times, gids, and types should be lists of '
                                 'the same length')
        self._times = times
        self._gids = gids
        self._types = types

    def __repr__(self):
        class_name = self.__class__.__name__
        n_trials = len(self._times)
        return '<%s | %d simulation trials>' % (class_name, n_trials)

    def __eq__(self, other):
        if not isinstance(other, Spikes):
            return NotImplemented
        # Round each time element
        times_self = [[round(time, 3) for time in trial]
                      for trial in self._times]
        times_other = [[round(time, 3) for time in trial]
                       for trial in other._times]
        return (times_self == times_other and
                self._gids == other._gids and
                self._types == other._types)

    @property
    def times(self):
        return self._times

    @property
    def gids(self):
        return self._gids

    @property
    def types(self):
        return self._types

    def update_types(self, gid_dict):
        """Update spike types in the current instance of Spikes.

        Parameters
        ----------
        gid_dict : dict of lists or range objects
            Dictionary with keys 'evprox1', 'evdist1' etc.
            containing the range of Cell or input IDs of different
            cell or input types.
        """

        # Validate gid_dict
        gid_dict_ranges = list(gid_dict.values())
        for item_idx_1 in range(len(gid_dict_ranges)):
            for item_idx_2 in range(item_idx_1 + 1, len(gid_dict_ranges)):
                gid_set_1 = set(gid_dict_ranges[item_idx_1])
                gid_set_2 = set(gid_dict_ranges[item_idx_2])
                if not gid_set_1.isdisjoint(gid_set_2):
                    raise ValueError('gid_dict should contain only disjoint '
                                     'sets of gid values')

        spike_types = list()
        for trial_idx in range(len(self._times)):
            spike_types_trial = np.empty_like(self._times[trial_idx],
                                              dtype='<U36')
            for gidtype, gids in gid_dict.items():
                spike_gids_mask = np.in1d(self._gids[trial_idx], gids)
                spike_types_trial[spike_gids_mask] = gidtype
            spike_types += [list(spike_types_trial)]
        self._types = spike_types

    def plot(self, ax=None, show=True):
        """Plot the aggregate spiking activity according to cell type.

        Parameters
        ----------
        ax : instance of matplotlib axis | None
            An axis object from matplotlib. If None,
            a new figure is created.
        show : bool
            If True, show the figure.

        Returns
        -------
        fig : instance of matplotlib Figure
            The matplotlib figure object.
        """

        import matplotlib.pyplot as plt
        spike_times = np.array(sum(self._times, []))
        spike_types = np.array(sum(self._types, []))
        cell_types = ['L5_pyramidal', 'L5_basket', 'L2_pyramidal', 'L2_basket']
        spike_times_cell = [spike_times[spike_types == cell_type]
                            for cell_type in cell_types]

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.eventplot(spike_times_cell, colors=['r', 'b', 'g', 'w'])
        ax.legend(cell_types, ncol=2)
        ax.set_facecolor('k')
        ax.set_xlabel('Time (ms)')
        ax.get_yaxis().set_visible(False)
        ax.set_ylim((-1, 4.5))
        ax.set_xlim(left=0)

        if show:
            plt.show()
        return ax.get_figure()

    def write(self, fname):
        """Write spiking activity per trial to a collection of files.

        Parameters
        ----------
        fname : str
            String format (e.g., '<pathname>/spk_%d.txt') of the
            path to the output spike file(s).

        Outputs
        -------
        A tab separated txt file for each trial where rows
            correspond to spikes, and columns correspond to
            1) spike time (s),
            2) spike gid, and
            3) gid type
        """

        for trial_idx in range(len(self._times)):
            with open(fname % (trial_idx,), 'w') as f:
                for spike_idx in range(len(self._times[trial_idx])):
                    f.write('{:.3f}\t{}\t{}\n'.format(
                        self._times[trial_idx][spike_idx],
                        int(self._gids[trial_idx][spike_idx]),
                        self._types[trial_idx][spike_idx]))
