"""Network class."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>

import itertools as it
import numpy as np
from glob import glob
from .feed import ExtFeed

from .params import create_pext
from .viz import plot_spikes_hist, plot_spikes_raster, plot_cells


def read_spikes(fname, gid_ranges=None):
    """Read spiking activity from a collection of spike trial files.

    Parameters
    ----------
    fname : str
        Wildcard expression (e.g., '<pathname>/spk_*.txt') of the
        path to the spike file(s).
    gid_ranges : dict of lists or range objects | None
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
        # spike type. If reading a legacy version, validate that a gid_ranges
        # is provided.
        if spike_trial.shape[1] == 3:
            spike_types += [list(spike_trial[:, 2].astype(str))]
        else:
            if gid_ranges is None:
                raise ValueError("gid_ranges must be provided if spike types "
                                 "are unspecified in the file %s" % (file,))
            spike_types += [[]]

    spikes = Spikes(spike_times=spike_times, spike_gids=spike_gids,
                    spike_types=spike_types)
    if gid_ranges is not None:
        spikes.update_types(gid_ranges)

    return Spikes(spike_times=spike_times, spike_gids=spike_gids,
                  spike_types=spike_types)


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
        The parameters to use for constructing the network.

    Attributes
    ----------
    params : dict
        The parameters of the network
    cellname_list : list
        The names of real cell types in the network (e.g. 'L2_basket')
    feedname_list : list
        The names of external drivers ('feeds') to the network (e.g. 'evdist1')
    gid_ranges : dict
        A dictionary of unique identifiers of each real and artificial cell
        in the network. Every cell type is represented by a key read from
        cellname_list, followed by entries read from feedname_list. The value
        of each key is a range of ints, one for each cell in given category.
        Examples: 'L2_basket': range(0, 270), 'evdist1': range(272, 542), etc
    pos_dict : dict
        Dictionary containing the coordinate positions of all cells.
        Keys are 'L2_pyramidal', 'L5_pyramidal', 'L2_basket', 'L5_basket',
        'common', or any of the elements of the cellname or feedname lists.
    spikes : Spikes
        An instance of the Spikes object.
    """

    def __init__(self, params):
        # Save the parameters used to create the Network
        self.params = params
        # Initialise a dictionary of cell ID's, which get used when the
        # network is constructed ('built') in NetworkBuilder
        # We want it to remain in each Network object, so that the user can
        # interrogate a built and simulated net. In addition, Spikes are
        # attached to a Network during simulation---Network is the natural
        # place to keep this information
        self.gid_ranges = dict()

        # Create array of equally sampled time points for simulating currents
        # NB (only) used to initialise self.spikes._times
        times = np.arange(0., params['tstop'] + params['dt'], params['dt'])
        # Create spikes object, initialised with simulation time points
        self.spikes = Spikes(times=times)

        # Source list of names, first real ones only!
        self.cellname_list = [
            'L2_basket',
            'L2_pyramidal',
            'L5_basket',
            'L5_pyramidal',
        ]
        self.feedname_list = []  # no feeds defined yet
        self.trial_event_times = []  # list of len == n_trials

        # contents of pos_dict determines all downstream inferences of
        # cell counts, real and artificial
        self.pos_dict = _create_cell_coords(n_pyr_x=self.params['N_pyr_x'],
                                            n_pyr_y=self.params['N_pyr_y'],
                                            zdiff=1307.4)
        # Every time pos_dict is updated, gid_ranges must be updated too
        self._update_gid_ranges()

        # set n_cells, EXCLUDING Artificial ones
        self.n_cells = sum(len(self.pos_dict[src]) for src in
                           self.cellname_list)

        # The legacy code in HNN-GUI _always_ defines 2 'common' and 5
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
        self._p_common, self._p_unique = create_pext(self.params,
                                                     self.params['tstop'])
        self._n_common_feeds = len(self._p_common)

        # 'position' the artificial cells arbitrarily in the origin of the
        # network grid. Non-biophysical cell placement is irrelevant
        # However, they must be present to be included in gid_ranges!
        origin = self.pos_dict['origin']

        # COMMON FEEDS
        self.pos_dict['common'] = [origin for i in range(self._n_common_feeds)]

        # UNIQUE FEEDS
        for key in self._p_unique.keys():
            # create the pos_dict for all the sources
            self.pos_dict[key] = [origin for i in range(self.n_cells)]

        # Now add the names of the feeds to a list
        self.feedname_list.append('common')
        # grab the keys for the unique set of inputs and sort the names
        # append them to the src list along with the number of cells
        unique_keys = sorted(self._p_unique.keys())
        self.feedname_list += unique_keys

        # Every time pos_dict is updated, gid_ranges must be updated too
        self._update_gid_ranges()

        # Create the feed dynamics (event_times)
        self._instantiate_feeds(n_trials=self.params['N_trials'])

    def _instantiate_feeds(self, n_trials=1):
        """Creates event_time vectors for all feeds and all trials

        Parameters
        ----------
        n_trials : int
            Number of trials to create events for (default: 1)

        NB this must be a separate method because dipole.py:simulate_dipole
        accepts an n_trials-argument, which overrides the N_trials-parameter
        used at intialisation time. The good news is that only the event_times
        need to be recalculated, all the GIDs etc remain the same.
        """
        # each trial needs unique event time vectors

        self.trial_event_times = []  # reset if called again from dipole.py

        cur_params = self.params.copy()  # these get mangled below!
        for trial_idx in range(n_trials):

            prng_seedcore_initial = cur_params['prng_*'].copy()
            for param_key in prng_seedcore_initial.keys():
                cur_params[param_key] =\
                    prng_seedcore_initial[param_key] + trial_idx
            # needs to be re-run to create the dicts going into ExtFeed
            # the only thing changing is the initial seed
            p_common, p_unique = create_pext(cur_params,
                                             cur_params['tstop'])

            src_types = self.feedname_list
            event_times_per_source = {}
            for src_type in src_types:
                event_times = []
                if src_type == 'common':
                    for idx, gid in enumerate(self.gid_ranges['common']):
                        gid_target = None  # 'common' attaches to all
                        feed = ExtFeed(feed_type=src_type,
                                       target_cell_type=gid_target,
                                       params=p_common[idx],
                                       gid=gid)
                        event_times.append(feed.event_times)

                elif src_type in p_unique.keys():
                    for gid in self.gid_ranges[src_type]:
                        gid_target = gid - self.gid_ranges[src_type][0]
                        target_cell_type = self.gid_to_type(gid_target)
                        feed = ExtFeed(feed_type=src_type,
                                       target_cell_type=target_cell_type,
                                       params=p_unique[src_type],
                                       gid=gid)
                        event_times.append(feed.event_times)
                event_times_per_source.update({src_type: event_times})

            # list of dict of list of list
            self.trial_event_times.append(event_times_per_source.copy())

    def _update_gid_ranges(self):
        """Creates gid ranges from scratch every time called.

        Any method that adds real or artificial cells to the network must
        call this to update the list of GIDs. Note that it's based on the
        content of pos_dict and the lists of real and artificial cell names.
        """
        gid_lims = [0]  # start and end gids per cell type
        src_types = self.cellname_list + self.feedname_list
        for idx, src_type in enumerate(src_types):
            n_srcs = len(self.pos_dict[src_type])
            gid_lims.append(gid_lims[idx] + n_srcs)
            self.gid_ranges[src_type] = range(gid_lims[idx],
                                              gid_lims[idx + 1])

    def gid_to_type(self, gid):
        """Reverse lookup of gid to type."""
        for gidtype, gids in self.gid_ranges.items():
            if gid in gids:
                return gidtype

    def _get_src_type_and_pos(self, gid):
        """Source type, position and whether it's a cell or artificial feed"""

        # get type of cell and pos via gid
        src_type = self.gid_to_type(gid)
        type_pos_ind = gid - self.gid_ranges[src_type][0]
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
    spike_times : list (n_trials,) of list (n_spikes,) of float, shape | None
        Each element of the outer list is a trial.
        The inner list contains the time stamps of spikes.
    spike_gids : list (n_trials,) of list (n_spikes,) of float, shape | None
        Each element of the outer list is a trial.
        The inner list contains the cell IDs of neurons that
        spiked.
    spike_types : list (n_trials,) of list (n_spikes,) of float, shape | None
        Each element of the outer list is a trial.
        The inner list contains the type of spike (e.g., evprox1
        or L2_pyramidal) that occured at the corresonding time stamp.
        Each gid corresponds to a type via Network().gid_ranges.

    Attributes
    ----------
    spike_times : list (n_trials,) of list (n_spikes,) of float, shape
        Each element of the outer list is a trial.
        The inner list contains the time stamps of spikes.
    spike_gids : list (n_trials,) of list (n_spikes,) of float, shape
        Each element of the outer list is a trial.
        The inner list contains the cell IDs of neurons that
        spiked.
    spike_types : list (n_trials,) of list (n_spikes,) of float, shape
        Each element of the outer list is a trial.
        The inner list contains the type of spike (e.g., evprox1
        or L2_pyramidal) that occured at the corresonding time stamp.
        Each gid corresponds to a type via Network::gid_ranges.
    vsoma : dict
        Dictionary indexed by gids containing somatic voltages
    times : numpy array
        Array of time points for samples in continuous data.
        This includes vsoma.

    Methods
    -------
    update_types(gid_ranges)
        Update spike types in the current instance of Spikes.
    plot(ax=None, show=True)
        Plot and return a matplotlib Figure object showing the
        aggregate network spiking activity according to cell type.
    mean_rates(tstart, tstop, gid_ranges, mean_type='all')
        Calculate mean firing rate for each cell type. Specify
        averaging method with mean_type argument.
    write(fname)
        Write spiking activity to a collection of spike trial files.
    """

    def __init__(self, spike_times=None, spike_gids=None, spike_types=None,
                 times=None):
        if spike_times is None:
            spike_times = list()
        if spike_gids is None:
            spike_gids = list()
        if spike_types is None:
            spike_types = list()

        # Validate arguments
        arg_names = ['spike_times', 'spike_gids', 'spike_types']
        for arg_idx, arg in enumerate([spike_times, spike_gids, spike_types]):
            # Validate outer list
            if not isinstance(arg, list):
                raise TypeError('%s should be a list of lists'
                                % (arg_names[arg_idx],))
            # If arg is not an empty list, validate inner list
            for trial_list in arg:
                if not isinstance(trial_list, list):
                    raise TypeError('%s should be a list of lists'
                                    % (arg_names[arg_idx],))
            # Set the length of 'spike_times' as a references and validate
            # uniform length
            if arg == spike_times:
                n_trials = len(spike_times)
            if len(arg) != n_trials:
                raise ValueError('spike times, gids, and types should be '
                                 'lists of the same length')
        self._spike_times = spike_times
        self._spike_gids = spike_gids
        self._spike_types = spike_types
        self._vsoma = list()
        if times is not None:
            if not isinstance(times, np.ndarray):
                raise TypeError("'times' is an np.ndarray of simulation times")
        self._times = times

    def __repr__(self):
        class_name = self.__class__.__name__
        n_trials = len(self._spike_times)
        return '<%s | %d simulation trials>' % (class_name, n_trials)

    def __eq__(self, other):
        if not isinstance(other, Spikes):
            return NotImplemented
        # Round each time element
        times_self = [[round(time, 3) for time in trial]
                      for trial in self._spike_times]
        times_other = [[round(time, 3) for time in trial]
                       for trial in other._spike_times]
        return (times_self == times_other and
                self._spike_gids == other._spike_gids and
                self._spike_types == other._spike_types)

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
        elif isinstance(gid_item, list):
            gid_item = np.array(gid_item)
        elif isinstance(gid_item, np.ndarray):
            if gid_item.ndim > 1:
                raise ValueError("ndarray cannot exceed 1 dimension")
            else:
                pass
        elif isinstance(gid_item, int):
            gid_item = np.array([gid_item])
        else:
            raise TypeError("indices must be int, slice, or array-like, "
                            f"not {type(gid_item).__name__}")

        if not np.issubdtype(gid_item.dtype, np.integer):
            raise TypeError("gids must be of dtype int, "
                            f"not {gid_item.dtype.name}")

        n_trials = len(self._spike_times)
        times_slice = []
        gids_slice = []
        types_slice = []
        vsoma_slice = []
        for trial_idx in range(n_trials):
            gid_mask = np.in1d(self._spike_gids[trial_idx], gid_item)
            times_trial = np.array(
                self._spike_times[trial_idx])[gid_mask].tolist()
            gids_trial = np.array(
                self._spike_gids[trial_idx])[gid_mask].tolist()
            types_trial = np.array(
                self._spike_types[trial_idx])[gid_mask].tolist()

            vsoma_trial = {gid: self._vsoma[trial_idx][gid] for gid in gid_item
                           if gid in self._vsoma[trial_idx].keys()}

            times_slice.append(times_trial)
            gids_slice.append(gids_trial)
            types_slice.append(types_trial)
            vsoma_slice.append(vsoma_trial)

        spikes_slice = Spikes(spike_times=times_slice, spike_gids=gids_slice,
                              spike_types=types_slice)
        spikes_slice._vsoma = vsoma_slice

        return spikes_slice

    @property
    def spike_times(self):
        return self._spike_times

    @property
    def spike_gids(self):
        return self._spike_gids

    @property
    def spike_types(self):
        return self._spike_types

    @property
    def vsoma(self):
        return self._vsoma

    @property
    def times(self):
        return self._times

    def update_types(self, gid_ranges):
        """Update spike types in the current instance of Spikes.

        Parameters
        ----------
        gid_ranges : dict of lists or range objects
            Dictionary with keys 'evprox1', 'evdist1' etc.
            containing the range of Cell or input IDs of different
            cell or input types.
        """

        # Validate gid_ranges
        all_gid_ranges = list(gid_ranges.values())
        for item_idx_1 in range(len(all_gid_ranges)):
            for item_idx_2 in range(item_idx_1 + 1, len(all_gid_ranges)):
                gid_set_1 = set(all_gid_ranges[item_idx_1])
                gid_set_2 = set(all_gid_ranges[item_idx_2])
                if not gid_set_1.isdisjoint(gid_set_2):
                    raise ValueError('gid_ranges should contain only disjoint '
                                     'sets of gid values')

        spike_types = list()
        for trial_idx in range(len(self._spike_times)):
            spike_types_trial = np.empty_like(self._spike_times[trial_idx],
                                              dtype='<U36')
            for gidtype, gids in gid_ranges.items():
                spike_gids_mask = np.in1d(self._spike_gids[trial_idx], gids)
                spike_types_trial[spike_gids_mask] = gidtype
            spike_types += [list(spike_types_trial)]
        self._spike_types = spike_types

    def mean_rates(self, tstart, tstop, gid_ranges, mean_type='all'):
        """Mean spike rates (Hz) by cell type.

        Parameters
        ----------
        tstart : int | float | None
            Value defining the start time of all trials.
        tstop : int | float | None
            Value defining the stop time of all trials.
        gid_ranges : dict of lists or range objects
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
            cell_type_gids = np.array(gid_ranges[cell_type])
            n_trials, n_cells = len(self._spike_times), len(cell_type_gids)
            gid_spike_rate = np.zeros((n_trials, n_cells))

            trial_data = zip(self._spike_types, self._spike_gids)
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

        for trial_idx in range(len(self._spike_times)):
            with open(str(fname) % (trial_idx,), 'w') as f:
                for spike_idx in range(len(self._spike_times[trial_idx])):
                    f.write('{:.3f}\t{}\t{}\n'.format(
                        self._spike_times[trial_idx][spike_idx],
                        int(self._spike_gids[trial_idx][spike_idx]),
                        self._spike_types[trial_idx][spike_idx]))
