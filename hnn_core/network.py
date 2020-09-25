"""Network class."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>

import itertools as it
import numpy as np
from glob import glob

from .params import create_pext
from .viz import plot_spikes_hist, plot_spikes_raster, plot_cells


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
    for file in sorted(glob(str(fname))):
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


def _create_cell_coords(n_pyr_x, n_pyr_y, zdiff=1307.4):
    """Creates coordinate grid and place cells in it.

    Parameters
    ----------
    n_pyr_x : int
        The number of Pyramidal cells in x direction.
    n_pyr_y : int
        The number of Pyramidal cells in y direction.

    zdiff : float
        Expressed as a positive DEPTH of L5 relative to L2
        This is a deviation from the original, where L5 was defined at 0
        This should not change interlaminar weight/delay calculations.

    Returns
    -------
    pos_dict : dict of list of tuple (x, y, z)
        Dictionary containing coordinate positions.
        Keys are 'L2_pyramidal', 'L5_pyramidal', 'L2_basket', 'L5_basket',
        'common', or any of the elements of the list p_unique_keys

    Notes
    -----
    Common positions are all located at origin.
    Sort of a hack bc of redundancy
    """
    pos_dict = dict()

    # PYRAMIDAL CELLS
    xxrange = np.arange(n_pyr_x)
    yyrange = np.arange(n_pyr_y)

    pos_dict['L2_pyramidal'] = [
        pos for pos in it.product(xxrange, yyrange, [0])]
    pos_dict['L5_pyramidal'] = [
        pos for pos in it.product(xxrange, yyrange, [zdiff])]

    # BASKET CELLS
    xzero = np.arange(0, n_pyr_x, 3)
    xone = np.arange(1, n_pyr_x, 3)
    # split even and odd y vals
    yeven = np.arange(0, n_pyr_y, 2)
    yodd = np.arange(1, n_pyr_y, 2)
    # create general list of x,y coords and sort it
    coords = [pos for pos in it.product(
        xzero, yeven)] + [pos for pos in it.product(xone, yodd)]
    coords_sorted = sorted(coords, key=lambda pos: pos[1])
    # append the z value for position for L2 and L5
    # print(len(coords_sorted))

    pos_dict['L2_basket'] = [pos_xy + (0,) for
                             pos_xy in coords_sorted]
    pos_dict['L5_basket'] = [
        pos_xy + (zdiff,) for pos_xy in coords_sorted]

    # ORIGIN
    # origin's z component isn't really used in
    # calculating distance functions from origin
    # these will be forced as ints!
    origin_x = xxrange[int((len(xxrange) - 1) // 2)]
    origin_y = yyrange[int((len(yyrange) - 1) // 2)]
    origin_z = np.floor(zdiff / 2)
    origin = (origin_x, origin_y, origin_z)

    # save the origin for adding external feeds later
    pos_dict['origin'] = origin

    return pos_dict


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
    cellname_list : list
        The names of real cell types in the network (e.g. 'L2_basket')
    feedname_list : list
        The names of external drivers ('feeds') to the network (e.g. 'evdist1')
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
        # Save the parameters used to create the Network
        self.params = params

        # Number of time points
        # Originally used to create the empty vec for synaptic currents,
        # ensuring that they exist on this node irrespective of whether
        # or not cells of relevant type actually do
        self.n_times = np.arange(0., self.params['tstop'],
                                 self.params['dt']).size + 1

        self.n_src = 0
        self.n_of_type = {}  # numbers of sources
        self.n_cells = 0  # init self.n_cells
        # Source list of names, first real ones only!
        self.cellname_list = [
            'L2_basket',
            'L2_pyramidal',
            'L5_basket',
            'L5_pyramidal',
        ]
        self.feedname_list = []  # no feeds defined yet
        # this will contain a list of all cell names, real and artificial
        self._all_cell_names = []

        # cell position lists, also will give counts: must be known
        # by ALL nodes
        # XXX structure of pos_dict determines all downstream inferences of
        # cell counts, real and artificial
        self.pos_dict = _create_cell_coords(n_pyr_x=self.params['N_pyr_x'],
                                            n_pyr_y=self.params['N_pyr_y'],
                                            zdiff=1307.4)

        # set n_cells, EXCLUDING Artificial ones
        self._count_cells()

        # Initialise a dictionary of cell ID's, which get assigned when the
        # network is constructed ('built') in NetworkBuilder
        # We want it to remain in each Network object, so that the user can
        # interrogate a built and simulated net. In addition, Spikes are
        # attached to a Network during simulation---Network is the natural
        # place to keep this information
        # XXX rename gid_dict in future
        self.gid_dict = dict()

        # When computing the network dynamics in parallel, the nodes of the
        # network (real and artificial cells) potentially get distributed
        # on different host machines/threads. NetworkBuilder._gid_assign
        # assigns each node, identified by its unique GID, to one of the
        # possible hosts/threads for computations. _gid_list here contains
        # the GIDs of all the nodes assigned to the current host/thread.
        self._gid_list = []

        # Create empty spikes object
        self.spikes = Spikes()

        # XXX The legacy code in HNN-GUI _always_ defines 2 'common' and 5
        # 'unique' feeds. They are added here for backwards compatibility
        # until a new handling of external NetworkDrives's is completed.
        self._add_external_feeds()

        # in particular order (cells, common, names of unique inputs)
        self._update_all_cell_names()

    def __repr__(self):
        class_name = self.__class__.__name__
        s = ("%d x %d Pyramidal cells (L2, L5)"
             % (self.params['N_pyr_x'], self.params['N_pyr_y']))
        s += ("\n%d L2 basket cells\n%d L5 basket cells"
              % (self.n_of_type['L2_basket'], self.n_of_type['L5_basket']))
        return '<%s | %s>' % (class_name, s)

    def _add_external_feeds(self):
        """Legacy function, for backward compatibility with original HNN-GUI

        This function is called exactly once during initialisation
        """
        # params of common external feeds inputs in p_common
        # Global number of external inputs ... automatic counting
        # makes more sense
        # p_unique represent ext inputs that are going to go to each cell
        self.p_common, self.p_unique = create_pext(self.params,
                                                   self.params['tstop'])
        self.n_common_feeds = len(self.p_common)

        # 'position' the artificial cells arbitrarily in the origin of the
        # network grid. Non-biophysical cell placement is irrelevant
        origin = self.pos_dict['origin']

        # COMMON FEEDS
        self.pos_dict['common'] = [origin for i in range(self.n_common_feeds)]

        # UNIQUE FEEDS
        for key in self.p_unique.keys():
            # create the pos_dict for all the sources
            self.pos_dict[key] = [origin for i in range(self.n_cells)]

        # Now add the names of the feeds to a list
        self.feedname_list.append('common')
        # grab the keys for the unique set of inputs and sort the names
        # append them to the src list along with the number of cells
        unique_keys = sorted(self.p_unique.keys())
        self.feedname_list += unique_keys

        # update external feed/source/drive count
        # all src numbers are based off of length of pos_dict entry
        # generally done here in lieu of upstream changes
        for src in self.feedname_list:
            self.n_of_type[src] = len(self.pos_dict[src])

    def _update_all_cell_names(self):
        """Updates the list of cell names (real ones and feeds)"""
        # return one final source list
        self._all_cell_names = self.cellname_list + self.feedname_list

    def _count_cells(self):
        """Cell counting routine."""
        # cellname list is used *only* for this purpose for now
        for src in self.cellname_list:
            # if it's a cell, then add the number to total number of cells
            self.n_of_type[src] = len(self.pos_dict[src])
            self.n_cells += self.n_of_type[src]

    def _create_gid_dict(self):
        """Creates gid dicts and pos_lists."""
        # initialize gid index gid_ind to start at 0
        gid_ind = [0]
        # append a new gid_ind based on previous and next cell count
        # order is guaranteed by self.src_list_new
        for i in range(len(self._all_cell_names)):
            # N = self.src_list_new[i][1]
            # grab the src name in ordered list src_list_new
            src = self._all_cell_names[i]
            # query the N dict for that number and append here
            # to gid_ind, based on previous entry
            gid_ind.append(gid_ind[i] + self.n_of_type[src])
            # accumulate total source count
            self.n_src += self.n_of_type[src]
        # now actually assign the ranges
        for i in range(len(self._all_cell_names)):
            src = self._all_cell_names[i]
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

        return src_type, src_pos, src_type in self.cellname_list

    def plot_cells(self, ax=None, show=True):
        """Plot the cells using Network.pos_dict.

        Parameters
        ----------
        ax : instance of matplotlib Axes3D | None
            An axis object from matplotlib. If None,
            a new figure is created.
        show : bool
            If True, show the figure.

        Returns
        -------
        fig : instance of matplotlib Figure
            The matplotlib figure handle.
        """
        return plot_cells(net=self, ax=ax, show=show)


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
        return plot_spikes_raster(spikes=self, ax=ax, show=show)

    def plot_hist(self, ax=None, spike_types=None, show=True):
        """Plot the histogram of spiking activity across trials.

        Parameters
        ----------
        ax : instance of matplotlib axis | None
            An axis object from matplotlib. If None,
            a new figure is created.
        spike_types: string | list | dictionary | None
            String input of a valid spike type is plotted individually.
                Ex: 'common', 'evdist', 'evprox', 'extgauss', 'extpois'
            List of valid string inputs will plot each spike type individually.
                Ex: ['common', 'evdist']
            Dictionary of valid lists will plot list elements as a group.
                Ex: {'Evoked': ['evdist', 'evprox'], 'External': ['extpois']}
            If None, all input spike types are plotted individually if any
            are present. Otherwise spikes from all cells are plotted.
            Valid strings also include leading characters of spike types
                Example: 'ext' is equivalent to ['extgauss', 'extpois']
        show : bool
            If True, show the figure.

        Returns
        -------
        fig : instance of matplotlib Figure
            The matplotlib figure handle.
        """
        return plot_spikes_hist(
            self, ax=ax, spike_types=spike_types, show=show)

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
            with open(str(fname) % (trial_idx,), 'w') as f:
                for spike_idx in range(len(self._times[trial_idx])):
                    f.write('{:.3f}\t{}\t{}\n'.format(
                        self._times[trial_idx][spike_idx],
                        int(self._gids[trial_idx][spike_idx]),
                        self._types[trial_idx][spike_idx]))
