"""CellResponse class."""

# Authors: Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

from glob import glob
import numpy as np

from .viz import plot_spikes_hist, plot_spikes_raster


class CellResponse(object):
    """The CellResponse class.

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
    times : numpy array | None
        Array of time points for samples in continuous data.
        This includes vsoma and isoma.
    cell_type_names : list
        List of unique cell type names that are explicitly modeled in the
        network

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
    vsoma : list (n_trials,) of dict, shape
        Each element of the outer list is a trial.
        Dictionary indexed by gids containing somatic voltages.
    isoma : list (n_trials,) of dict, shape
        Each element of the outer list is a trial.
        Dictionary indexed by gids containing somatic currents.
    times : numpy array
        Array of time points for samples in continuous data.
        This includes vsoma and isoma.

    Methods
    -------
    reset()
        Reset all recorded attributes to empty lists.
    update_types(gid_ranges)
        Update spike types in the current instance of CellResponse.
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
                 times=None, cell_type_names=['L2_basket',
                                              'L2_pyramidal',
                                              'L5_basket',
                                              'L5_pyramidal']):
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
        self._isoma = list()
        if times is not None:
            if not isinstance(times, np.ndarray):
                raise TypeError("'times' is an np.ndarray of simulation times")
        self._times = times
        self._cell_type_names = cell_type_names

    def __repr__(self):
        class_name = self.__class__.__name__
        n_trials = len(self._spike_times)
        return '<%s | %d simulation trials>' % (class_name, n_trials)

    def __eq__(self, other):
        if not isinstance(other, CellResponse):
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
        """Returns a CellResponse object with a copied subset filtered by gid.

        Parameters
        ----------
        gid_item : int | slice
            Subset of gids .

        Returns
        -------
        cell_response : instance of CellResponse
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
        times_slice = list()
        gids_slice = list()
        types_slice = list()
        vsoma_slice = list()
        isoma_slice = list()
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

            isoma_trial = {gid: self._isoma[trial_idx][gid] for gid in gid_item
                           if gid in self._isoma[trial_idx].keys()}

            times_slice.append(times_trial)
            gids_slice.append(gids_trial)
            types_slice.append(types_trial)
            vsoma_slice.append(vsoma_trial)
            isoma_slice.append(isoma_trial)

        cell_response_slice = CellResponse(spike_times=times_slice,
                                           spike_gids=gids_slice,
                                           spike_types=types_slice)
        cell_response_slice._vsoma = vsoma_slice
        cell_response_slice._isoma = isoma_slice

        return cell_response_slice

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
    def isoma(self):
        return self._isoma

    @property
    def times(self):
        return self._times

    def reset(self):
        """Reset all recorded attributes to empty lists."""
        self._spike_times = list()
        self._spike_gids = list()
        self._spike_types = list()
        self._vsoma = list()
        self._isoma = list()

    def update_types(self, gid_ranges):
        """Update spike types in the current instance of CellResponse.

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

        for cell_type in self._cell_type_names:
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

    def plot_spikes_raster(self, ax=None, show=True):
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
        return plot_spikes_raster(cell_response=self, ax=ax, show=show)

    def plot_spikes_hist(self, ax=None, spike_types=None, show=True):
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
    cell_response : CellResponse
        An instance of the CellResponse object.
    """

    spike_times = list()
    spike_gids = list()
    spike_types = list()
    for file in sorted(glob(str(fname))):
        spike_trial = np.loadtxt(file, dtype=str)
        if spike_trial.shape[0] > 0:
            spike_times += [list(spike_trial[:, 0].astype(float))]
            spike_gids += [list(spike_trial[:, 1].astype(int))]

            # Note that legacy HNN 'spk.txt' files don't contain a 3rd column
            # for spike type. If reading a legacy version, ensure that a
            # gid_dict is provided.
            if spike_trial.shape[1] == 3:
                spike_types += [list(spike_trial[:, 2].astype(str))]
            else:
                if gid_ranges is None:
                    raise ValueError("gid_ranges must be provided if spike "
                                     "types are unspecified in the "
                                     "file %s" % (file,))
                spike_types += [list()]
        else:
            spike_times += [list()]
            spike_gids += [list()]
            spike_types += [list()]

    cell_response = CellResponse(spike_times=spike_times,
                                 spike_gids=spike_gids,
                                 spike_types=spike_types)
    if gid_ranges is not None:
        cell_response.update_types(gid_ranges)

    return CellResponse(spike_times=spike_times, spike_gids=spike_gids,
                        spike_types=spike_types)
