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

        # Source list of names, first real ones only!
        self.cellname_list = [
            'L2_basket',
            'L2_pyramidal',
            'L5_basket',
            'L5_pyramidal',
        ]
        self.feedname_list = []  # no feeds defined yet

        # cell position lists, also will give counts: must be known
        # by ALL nodes
        # XXX structure of pos_dict determines all downstream inferences of
        # cell counts, real and artificial
        self.pos_dict = _create_cell_coords(n_pyr_x=self.params['N_pyr_x'],
                                            n_pyr_y=self.params['N_pyr_y'],
                                            zdiff=1307.4)

        # set n_cells, EXCLUDING Artificial ones
        self.n_cells = sum(len(self.pos_dict[src]) for src in
                           self.cellname_list)

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

    def __repr__(self):
        class_name = self.__class__.__name__
        s = ("%d x %d Pyramidal cells (L2, L5)"
             % (self.params['N_pyr_x'], self.params['N_pyr_y']))
        s += ("\n%d L2 basket cells\n%d L5 basket cells"
              % (len(self.pos_dict['L2_basket']),
                 len(self.pos_dict['L5_basket'])))
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

    def _create_gid_dict(self):
        """Creates gid dicts."""
        gid_lims = [0]  # start and end gids per cell type
        src_types = self.cellname_list + self.feedname_list
        for idx, src_type in enumerate(src_types):
            n_srcs = len(self.pos_dict[src_type])
            gid_lims.append(gid_lims[idx] + n_srcs)
            self.gid_dict[src_type] = range(gid_lims[idx],
                                            gid_lims[idx + 1])

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
    vsoma : dict
        Dictionary indexed by gids containing somatic voltages

    Methods
    -------
    update_types(gid_dict)
        Update spike types in the current instance of Spikes.
    plot(ax=None, show=True)
        Plot and return a matplotlib Figure object showing the
        aggregate network spiking activity according to cell type.
    mean_rates(tstart, tstop, gid_dict, mean_type='all')
        Calculate mean firing rate for each cell type. Specify
        averaging method with mean_type argument.
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
        self._vsoma = list()

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

    def __getitem__(self, gid_item):
        """Returns a Spikes object with a copied subset filtered by gid.

        Parameters
        ----------
        gid_item : int | slice
            Subset of gids .

        Returns
        -------
        spikes : instance of Spikes
            See below for use cases.
        """

        if isinstance(gid_item, slice):
            gid_item = np.arange(gid_item.stop)[gid_item]
        elif isinstance(gid_item, int):
            gid_item = np.array([gid_item])
        else:
            raise TypeError("indices must be integers or slices, "
                            f"not {type(gid_item).__name__}")

        n_trials = len(self._times)
        times_slice = []
        gids_slice = []
        types_slice = []
        vsoma_slice = []
        for trial_idx in range(n_trials):
            gid_mask = np.in1d(self._gids[trial_idx], gid_item)
            times_trial = np.array(self._times[trial_idx])[gid_mask].tolist()
            gids_trial = np.array(self._gids[trial_idx])[gid_mask].tolist()
            types_trial = np.array(self._types[trial_idx])[gid_mask].tolist()

            vsoma_trial = {gid: self._vsoma[trial_idx][gid] for gid in gid_item
                           if gid in self._vsoma[trial_idx].keys()}

            times_slice.append(times_trial)
            gids_slice.append(gids_trial)
            types_slice.append(types_trial)
            vsoma_slice.append(vsoma_trial)

        spikes_slice = Spikes(times=times_slice, gids=gids_slice,
                              types=types_slice)
        spikes_slice._vsoma = vsoma_slice

        return spikes_slice

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

    def mean_rates(self, tstart, tstop, gid_dict, mean_type='all'):
        """Mean spike rates (Hz) by cell type.

        Parameters
        ----------
        tstart : int | float | None
            Value defining the start time of all trials.
        tstop : int | float | None
            Value defining the stop time of all trials.
        gid_dict : dict of lists or range objects
            Dictionary with keys 'evprox1', 'evdist1' etc.
            containing the range of Cell or input IDs of different
            cell or input types.
        mean_type : str
            'all' : Average over trials and cells
                Returns mean rate for cell types
            'trial' : Average over cell types
                Returns trial mean rate for cell types
            'cell' : Average over individual cells
                Returns trial mean rate for individual cells

        Returns
        -------
        spike_rate : dict
            Dictionary with keys 'L5_pyramidal', 'L5_basket', etc.
        """
        cell_types = ['L5_pyramidal', 'L5_basket', 'L2_pyramidal', 'L2_basket']
        spike_rates = dict()

        if mean_type not in ['all', 'trial', 'cell']:
            raise ValueError("Invalid mean_type. Valid arguments include "
                             f"'all', 'trial', or 'cell'. Got {mean_type}")

        # Validate tstart, tstop
        if not isinstance(tstart, (int, float)) or not isinstance(
                tstop, (int, float)):
            raise ValueError('tstart and tstop must be of type int or float')
        elif tstop <= tstart:
            raise ValueError('tstop must be greater than tstart')

        for cell_type in cell_types:
            cell_type_gids = np.array(gid_dict[cell_type])
            n_trials, n_cells = len(self._times), len(cell_type_gids)
            gid_spike_rate = np.zeros((n_trials, n_cells))

            trial_data = zip(self._types, self._gids)
            for trial_idx, (spike_types, spike_gids) in enumerate(trial_data):
                trial_type_mask = np.in1d(spike_types, cell_type)
                gids, gid_counts = np.unique(np.array(
                    spike_gids)[trial_type_mask], return_counts=True)

                gid_spike_rate[trial_idx, np.in1d(cell_type_gids, gids)] = (
                    gid_counts / (tstop - tstart)) * 1000

            if mean_type == 'all':
                spike_rates[cell_type] = np.mean(
                    gid_spike_rate.mean(axis=1))
            if mean_type == 'trial':
                spike_rates[cell_type] = np.mean(
                    gid_spike_rate, axis=1).tolist()
            if mean_type == 'cell':
                spike_rates[cell_type] = [gid_trial_rate.tolist()
                                          for gid_trial_rate in gid_spike_rate]

        return spike_rates

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
