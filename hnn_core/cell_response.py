"""CellResponse class."""

# Authors: Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

from glob import glob
from warnings import warn

import numpy as np

from .viz import plot_spikes_hist, plot_spikes_raster, plot_firing_rate_time
from .externals.mne import _validate_type


class CellResponse(object):
    """The CellResponse class.

    Parameters
    ----------
    cell_type_names : list
        List of unique cell type names that are explicitly modeled in the
        network.
    cell_type_metadata : dict, optional
        Corresponding metadata of the cell types in the Network that spawned this
        CellResponse, in the dictionary form of metadata available at the top of
        `network_models.py`.
    spike_times : list (n_trials,) of list (n_spikes,) of float | None
        Each element of the outer list is a trial.
        The inner list contains the time stamps of spikes.
    spike_gids : list (n_trials,) of list (n_spikes,) of float | None
        Each element of the outer list is a trial.
        The inner list contains the cell IDs of neurons that
        spiked.
    spike_types : list (n_trials,) of list (n_spikes,) of float | None
        Each element of the outer list is a trial. The inner list contains the
        type of spike (e.g., evprox1 or L2_pyramidal) that occurred at the
        corresponding time stamp.  Each gid corresponds to a type via
        Network().gid_ranges. Note that the type of spike can be from a
        cell type or a drive.
    times : numpy array | None
        Array of time points for samples in continuous data.
        This includes vsoma and isoma.

    Attributes
    ----------
    spike_times : list (n_trials,) of list (n_spikes,) of float
        Each element of the outer list is a trial.
        The inner list contains the time stamps of spikes.
    spike_gids : list (n_trials,) of list (n_spikes,) of float
        Each element of the outer list is a trial.
        The inner list contains the cell IDs of neurons that
        spiked.
    spike_types : list (n_trials,) of list (n_spikes,) of float
        Each element of the outer list is a trial. The inner list contains the
        type of spike (e.g., evprox1 or L2_pyramidal) that occurred at the
        corresponding time stamp.  Each gid corresponds to a type via
        Network().gid_ranges. Note that the type of spike can be from a
        cell type or a drive.
    vsec : list (n_trials,) of dict
        Each element of the outer list is a trial.
        Dictionary indexed by gids containing voltages for cell sections.
    isec : list (n_trials,) of dict
        Each element of the outer list is a trial.
        Dictionary indexed by gids containing currents for cell sections.
    ca : list (n_trials,) of dict, shape
        Each element of the outer list is a trial.
        Dictionary indexed by gids containing calcium concentration
        for cell sections.
    times : array-like, shape (n_times,)
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
    mean_rates(tstart, tstop, gid_ranges=None, mean_type='all')
        Calculate mean firing rate for each cell type. Specify
        averaging method with mean_type argument.
    write(fname)
        Write spiking activity to a collection of spike trial files.
    """

    def __init__(
        self,
        cell_type_names,
        cell_type_metadata=None,
        spike_times=None,
        spike_gids=None,
        spike_types=None,
        times=None,
    ):
        if spike_times is None:
            spike_times = list()
        if spike_gids is None:
            spike_gids = list()
        if spike_types is None:
            spike_types = list()
        if times is None:
            times = list()

        # Validate arguments
        arg_names = ["spike_times", "spike_gids", "spike_types"]
        for arg_idx, arg in enumerate([spike_times, spike_gids, spike_types]):
            # Validate outer list
            if not isinstance(arg, list):
                raise TypeError("%s should be a list of lists" % (arg_names[arg_idx],))
            # If arg is not an empty list, validate inner list
            for trial_list in arg:
                if not isinstance(trial_list, list):
                    raise TypeError(
                        "%s should be a list of lists" % (arg_names[arg_idx],)
                    )
            # Set the length of 'spike_times' as a references and validate
            # uniform length
            if arg == spike_times:
                n_trials = len(spike_times)
            if len(arg) != n_trials:
                raise ValueError(
                    "spike times, gids, and types should be lists of the same length"
                )
        self._spike_times = spike_times
        self._spike_gids = spike_gids
        self._spike_types = spike_types
        self._vsec = list()
        self._isec = list()
        self._ca = list()
        if times is not None:
            if not isinstance(times, (list, np.ndarray)):
                raise TypeError("'times' is an np.ndarray of simulation times")
        self._times = np.array(times)
        self._cell_type_metadata = cell_type_metadata
        self._cell_type_names = cell_type_names

    def __repr__(self):
        class_name = self.__class__.__name__
        n_trials = len(self._spike_times)
        return "<%s | %d simulation trials>" % (class_name, n_trials)

    def __eq__(self, other):
        if not isinstance(other, CellResponse):
            return NotImplemented
        # Round each time element
        times_self = [[round(time, 3) for time in trial] for trial in self._spike_times]
        times_other = [
            [round(time, 3) for time in trial] for trial in other._spike_times
        ]
        return (
            times_self == times_other
            and self._spike_gids == other._spike_gids
            and self._spike_types == other._spike_types
            and self._vsec == other._vsec
            and self._isec == other._isec
            and self._ca == other._ca
            and self.vsec == other.vsec
            and self.isec == other.isec
            and self.ca == other.ca
        )

    @property
    def spike_times(self):
        return self._spike_times

    @property
    def cell_types(self):
        """Get unique cell types."""
        spike_types_data = np.concatenate(np.array(self.spike_types, dtype=object))
        return np.unique(spike_types_data).tolist()

    @property
    def spike_times_by_type(self):
        """Get a dictionary of spike times by cell type"""
        spike_times = dict()
        for cell_type in self.cell_types:
            spike_times[cell_type] = list()
            for trial_spike_times, trial_spike_types in zip(
                self.spike_times, self.spike_types
            ):
                mask = np.isin(trial_spike_types, cell_type)
                cell_spike_times = np.array(trial_spike_times)[mask].tolist()
                spike_times[cell_type].append(cell_spike_times)
        return spike_times

    @property
    def spike_gids(self):
        return self._spike_gids

    @property
    def spike_types(self):
        return self._spike_types

    @property
    def vsec(self):
        return self._vsec

    @property
    def isec(self):
        return self._isec

    @property
    def ca(self):
        return self._ca

    @property
    def times(self):
        return self._times

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
                    raise ValueError(
                        "gid_ranges should contain only disjoint sets of gid values"
                    )

        spike_types = list()
        for trial_idx in range(len(self._spike_times)):
            spike_types_trial = np.empty_like(
                self._spike_times[trial_idx], dtype="<U36"
            )
            for gidtype, gids in gid_ranges.items():
                spike_gids_mask = np.isin(self._spike_gids[trial_idx], gids)
                spike_types_trial[spike_gids_mask] = gidtype
            spike_types += [list(spike_types_trial)]
        self._spike_types = spike_types

    def _gids_from_spikes(self, cell_type, trial_idx=None):
        """Count how many gids per cell type spiked over all trials.
        This function allows mean_rates to be used with gid_ranges=None.

        Parameters
        ----------
        cell_type : str
            Cell type name, e.g. 'L5_pyramidal'.
        trial_idx : None | int | list of int, default=None
            Trial index, if None (default) return gids over all trials,
            if int, return gids that spiked in individual trial,
            if list, return gids that spiked over trials in list.

        Returns
        -------
        gids : np.ndarray
            Sorted unique gids of ``cell_type`` observed in the spike record.
        """
        gids, types = [], []

        if trial_idx is None:
            for trial_gids, trial_types in zip(self._spike_gids, self._spike_types):
                gids.extend(trial_gids)
                types.extend(trial_types)
        elif isinstance(trial_idx, list):
            for tidx in trial_idx:
                trial_gids = self._spike_gids[tidx]
                trial_types = self._spike_types[tidx]
                gids.extend(trial_gids)
                types.extend(trial_types)
        elif isinstance(trial_idx, int):
            trial_gids = self._spike_gids[trial_idx]
            trial_types = self._spike_types[trial_idx]
            gids.extend(trial_gids)
            types.extend(trial_types)

        gids = np.array(gids)
        types = np.array(types)
        if len(gids) == 0:
            return np.array([], dtype=int)
        return np.unique(gids[types == cell_type])

    def mean_rates(self, tstart, tstop, gid_ranges=None, mean_type="all"):
        """Mean spike rates (Hz) by cell type.

        Parameters
        ----------
        tstart : int | float | None
            Value defining the start time of all trials.
        tstop : int | float | None
            Value defining the stop time of all trials.
        gid_ranges : dict of lists or range objects | None
            Dictionary with keys, e.g. net.gid_ranges containing the range of Cell or
            input GIDs of different cell or input types. If None (default), the number
            of cells per type is inferred from the recorded spikes.
        mean_type : str
            'all' : Average over trials and cells
                Returns mean firing rate over time, trials, and gids, per cell type.
                Returns: `spike_rates[cell_type] = <scalar>`
            'trial' : Average over cell types
                Returns mean firing rate over time and gids, per trials and cell type.
                Returns: `spike_rates[cell_type] = <list of <scalar>, one per trial>`
            'cell' : Average over individual cells
                Returns mean firing rate averaged over time, per gid, trial and cell type.
                Returns: `spike_rates[cell_type] = <list of <list of <scalar>, one per cell>, one per trial>`

        Returns
        -------
        spike_rate : dict
            Dictionary with keys 'L5_pyramidal', 'L5_basket', etc.
        """
        spike_rates = dict()

        if mean_type not in ["all", "trial", "cell"]:
            raise ValueError(
                "Invalid mean_type. Valid arguments include "
                f"'all', 'trial', or 'cell'. Got {mean_type}"
            )

        # Validate tstart, tstop
        if not isinstance(tstart, (int, float)) or not isinstance(tstop, (int, float)):
            raise ValueError("tstart and tstop must be of type int or float")
        elif tstop <= tstart:
            raise ValueError("tstop must be greater than tstart")

        for cell_type in self._cell_type_names:
            # if gid_ranges defined by user
            if gid_ranges is not None:
                cell_type_gids = np.array(gid_ranges[cell_type])
            # if gid_ranges not defined by user, calculate number of gids
            # per type as number of gids that fired at least once.
            else:
                cell_type_gids = self._gids_from_spikes(cell_type)
            n_trials, n_cells = len(self._spike_times), len(cell_type_gids)

            # if this cell type had no spikes
            if n_cells == 0:
                if mean_type == "all":
                    spike_rates[cell_type] = 0.0
                elif mean_type == "trial":
                    spike_rates[cell_type] = [0.0] * n_trials
                elif mean_type == "cell":
                    spike_rates[cell_type] = [[0.0] for _ in range(n_trials)]

            # if there are spikes, calculate respective averages
            else:
                gid_spike_rate = np.zeros((n_trials, n_cells))

                trial_data = zip(self._spike_types, self._spike_gids)
                for trial_idx, (spike_types, spike_gids) in enumerate(trial_data):
                    trial_type_mask = np.isin(spike_types, cell_type)
                    gids, gid_counts = np.unique(
                        np.array(spike_gids)[trial_type_mask], return_counts=True
                    )

                    gid_spike_rate[trial_idx, np.isin(cell_type_gids, gids)] = (
                        (gid_counts / (tstop - tstart)) * 1000
                    )  # conversion to go from simulation time in ms to rates of Hz

                if mean_type == "all":
                    spike_rates[cell_type] = np.mean(gid_spike_rate.mean(axis=1))
                elif mean_type == "trial":
                    spike_rates[cell_type] = np.mean(gid_spike_rate, axis=1).tolist()
                elif mean_type == "cell":
                    spike_rates[cell_type] = [
                        gid_trial_rate.tolist() for gid_trial_rate in gid_spike_rate
                    ]

        return spike_rates

    def _preprocess_rate_over_time_args(self, window_length, cell_types, trial_idx):
        """Validation and preprocessing of arguments for rate_over_time methods.

        Specifically intended for arguments to `CellResponse.rate_over_time` and
        `CellResponse.plot_firing_rate_time`
        """

        # This checks that the CellResponse.times attribute is valid BEFORE checking any
        # arguments, since `.times` is NOT guaranteed to be nonzero or nonempty.
        times = self.times
        if len(times) < 2 or np.isclose(np.diff(times)[0], 0):
            raise ValueError(
                "'times' must contain at least two entries with non-zero "
                f"spacing to compute a sampling rate. Got {self.times}"
            )
        dt = np.diff(times)[0]

        # Validate window_length
        _validate_type(window_length, (int, float), "window_length", "int, float")
        total_duration = times[-1] - times[0]
        if window_length <= dt:
            raise ValueError(
                f"'window_length' must be greater than the 'dt' simulation timestep. "
                f"Got {window_length}"
            )
        elif window_length > total_duration:
            raise ValueError(
                f"'window_length' must not be greater than the total duration of "
                f"the simulation ({total_duration}). Got {window_length}"
            )

        # Validate trial_idx
        _validate_type(
            trial_idx, (int, list, None), "trial_idx", "int, list of int, or None"
        )
        n_max_trials = len(self.spike_gids)
        if trial_idx is None:
            trial_list = list(range(n_max_trials))
        elif isinstance(trial_idx, int):
            if trial_idx < 0 or trial_idx >= n_max_trials:
                raise ValueError(
                    f"'trial_idx' must be a non-negative integer less "
                    f"than the number of trials ({n_max_trials}). Got {trial_idx}"
                )
            trial_list = [trial_idx]
        elif isinstance(trial_idx, list):
            if len(trial_idx) == 0:
                raise ValueError("'trial_idx' must not be an empty list.")
            for trial in trial_idx:
                if not isinstance(trial, int):
                    raise ValueError(
                        f"Trial indices in 'trial_idx' must be integers. Got {type(trial)}"
                    )
                elif trial < 0 or trial >= n_max_trials:
                    raise ValueError(
                        f"Each element of 'trial_idx' must be a non-negative integer "
                        f"less than the number of trials ({n_max_trials}). Got {trial}"
                    )
            trial_list = trial_idx

        # Validate cell_types
        _validate_type(
            cell_types, (str, list, None), "cell_types", "str, list of str, or None"
        )
        if cell_types is None:
            cell_types = self._cell_type_names
        elif isinstance(cell_types, str):
            if cell_types not in self._cell_type_names:
                raise ValueError(
                    f"Invalid cell type provided. Must be present in set "
                    f"{self._cell_type_names}. Got {cell_types}"
                )
            else:
                cell_types = [cell_types]

        elif isinstance(cell_types, list):
            if len(cell_types) == 0:
                raise ValueError("'cell_types' must not be an empty list.")
            if not set(cell_types).issubset(set(self._cell_type_names)):
                raise ValueError(
                    f"Invalid cell type provided. Must be present in set "
                    f"{self._cell_type_names}. Got {cell_types}"
                )

        return times, dt, trial_list, cell_types

    def rate_over_time(
        self, window_length, gid_ranges=None, cell_types=None, trial_idx=None
    ):
        """Mean spike rates (Hz) by cell type over time.

        Parameters
        ----------
        window_length : int | float
            Length of the sliding window over which firing rates are calculated, in
            ms. Must be greater than the 'dt' of the simulation.
        gid_ranges : None | dict of lists or range objects, default=None
            Dictionary with keys, e.g. net.gid_ranges containing the range of Cell or
            input IDs of different cell or input types. If None (default), the number of
            cells per type is inferred from the recorded spikes.
        cell_types : None| str | list of str, default=None
            Cell types for which firing rates are calculated. If None (the default),
            firing rates are calculated for all cell types in the network.
        trial_idx : None | int | list of int, default=None
            Trial index (or list of indices) for which firing rate is calculated. If
            None (the default), firing rates are calculated for all trials.

        Returns
        -------
        rates : dict
            Dictionary one key per cell type (e.g. 'L5_pyramidal)
            containing np.array of size <n_trials, timepoints>
        """
        # Validation and preprocessing of arguments
        times, dt, trial_list, cell_types = self._preprocess_rate_over_time_args(
            window_length, cell_types, trial_idx
        )

        # Main body of function
        window_length_samples = int(window_length / dt)
        edges = np.concatenate(
            [times - dt / 2, [times[-1] + dt / 2]]
        )  # len(times)+1 edges

        taper = np.hanning(int(window_length_samples))
        taper /= taper.sum()
        rates_time = dict()

        for cell_type in cell_types:
            # if gid_ranges defined by user, n_cells is number of cells
            # of that type in network
            if gid_ranges is not None:
                n_cells = len(gid_ranges[cell_type])

            rates_time[cell_type] = np.zeros((len(trial_list), len(times)))

            if cell_type in self.spike_times_by_type:
                for row, trial in enumerate(trial_list):
                    # if gid_ranges not defined by user, calculate number of gids
                    # per type as number of gids that fired at least once.
                    if gid_ranges is None:
                        n_cells = len(
                            self._gids_from_spikes(cell_type, trial_idx=trial)
                        )
                    if n_cells == 0:
                        n_cells = 1

                    spike_type_time = np.histogram(
                        self.spike_times_by_type[cell_type][trial], bins=edges
                    )[0]
                    rates_time[cell_type][row] = (
                        np.convolve(spike_type_time, taper, "same")
                        / n_cells
                        / (dt / 1e3)  # conversion of dt from ms to s (to get 1/s = Hz)
                    )

        return rates_time

    def plot_spikes_raster(
        self,
        trial_idx=None,
        ax=None,
        show=True,
        cell_types=None,
        colors=None,
        show_legend=True,
        marker_size=5.0,
        dpl=None,
        overlay_dipoles=False,
        xticks=None,
        yticks=None,
        xlabel="Time (ms)",
        ylabel="Neuron index",
        title=None,
    ):
        """Plot the aggregate spiking activity according to cell type.

        Parameters
        ----------
        trial_idx : int | list of int | None
            Index of trials to be plotted. If None, all trials plotted.
        ax : instance of matplotlib axis | None
            An axis object from matplotlib. If None, a new figure is created.
        show : bool
            If True, show the figure.
        cell_types : list of str
            List of cell types to plot
        colors : list of str | None
            Optional custom colors to plot. Default will use the colors defined in cell metadata.
        show_legend : bool
            If True, show the legend with colors for cell types
        marker_size : float
            Optional marker size to use when plotting spikes. Uses
            "linelengths" argument of ax.eventplot, which accepts positive
            numeric values only
        dpl : instance of Dipole | list
            The Dipole object containing layer-specific dipole data
            to overlay on the raster plot
        overlay_dipoles : bool
            If True, overlay the layer-specific dipole data on the
            raster plot
        xticks : list | np.array | None
            Ticks on x-axis. If None, ticks are created by matplotlib.
        yticks : list | np.array | None
            Ticks on y-axis,  If None, ticks are created by matplotlib.
        xlabel : str, default: "Time (ms)"
            The matplotlib x-axis label
        ylabel : str, default: "Neuron index"
            The matplotlib y-axis label
        title : str | None
            The matplotlib figure title

        Returns
        -------
        fig : instance of matplotlib Figure
            The matplotlib figure object.
        """
        return plot_spikes_raster(
            cell_response=self,
            trial_idx=trial_idx,
            ax=ax,
            show=show,
            cell_types=cell_types,
            colors=colors,
            show_legend=show_legend,
            marker_size=marker_size,
            dpl=dpl,
            overlay_dipoles=overlay_dipoles,
            xticks=xticks,
            yticks=yticks,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

    def plot_firing_rate_time(
        self,
        window_length,
        trial_idx=None,
        ax=None,
        show=True,
        cell_types=None,
        colors=None,
        show_legend=True,
        sharex=False,
        sharey=False,
        xticks=None,
        yticks=None,
        xlim=None,
        ylim=None,
        xlabel="time (ms)",
        ylabel="firing rate (Hz)",
    ):
        """Plot time course of firing rates

        Parameters
        ----------
        window_length : int | float
            Length of the sliding window over which mean rates are calculated, in ms.
        trial_idx : int | list of int | None
            Trial index (or list of indices) to be plotted. If None (the default), mean
            rate over all trials is plotted, and standard deviation is indicated by
            shading.
        ax : instance of matplotlib axis | None
            An axis object from matplotlib. If None, a new figure is created.
        show : bool
            If True, show the figure.
        cell_types : list of str | None
            List of cell types to plot. If None, all cell types are plotted.
        colors : list of str | None
            Optional custom colors to plot. Default will use the colors defined in cell metadata.
        show_legend : bool
            If True, show the legend with colors for cell types
        sharex : bool
            If True, subplot x-axes will be shared. Only used when creating a new
            figure (i.e., when `ax` is None).
        sharey : bool
            If True, subplot y-axes will be shared. Only used when creating a new
            figure (i.e., when `ax` is None).
        xticks : array-like | None
            Custom x-axis tick locations. If None, matplotlib's default is used.
        yticks : array-like | None
            Custom y-axis tick locations. If None, matplotlib's default is used.
        xlim : tuple of (float, float) | None
            Custom x-axis limits. If None, defaults to the full time range.
        ylim : tuple of (float, float) | None
            Custom y-axis limits. If None, matplotlib's default is used.
        xlabel : str
            Label for the x-axis.
        ylabel : str
            Label for the y-axis.

        Returns
        -------
        fig : instance of matplotlib Figure
            The matplotlib figure object.
        """
        return plot_firing_rate_time(
            self,
            window_length=window_length,
            trial_idx=trial_idx,
            ax=ax,
            show=show,
            cell_types=cell_types,
            colors=colors,
            show_legend=show_legend,
            sharex=sharex,
            sharey=sharey,
            xticks=xticks,
            yticks=yticks,
            xlim=xlim,
            ylim=ylim,
            xlabel=xlabel,
            ylabel=ylabel,
        )

    def plot_spikes_hist(
        self,
        trial_idx=None,
        ax=None,
        spike_types=None,
        color=None,
        invert_spike_types=None,
        show=True,
        **kwargs_hist,
    ):
        """Plot the histogram of spiking activity across trials.

        Parameters
        ----------
        trial_idx : int | list of int | None
            Index of trials to be plotted. If None, all trials plotted.
        ax : instance of matplotlib axis | None
            An axis object from matplotlib. If None, a new figure is created.
        spike_types: string | list | dictionary | None
            String input of a valid spike type is plotted individually.

            | Ex: ``'poisson'``, ``'evdist'``, ``'evprox'``, ...

            List of valid string inputs will plot each spike type individually.

            | Ex: ``['poisson', 'evdist']``

            Dictionary of valid lists will plot list elements as a group.

            | Ex: ``{'Evoked': ['evdist', 'evprox'], 'Tonic': ['poisson']}``

            If None, all input spike types are plotted individually if any
            are present. Otherwise spikes from all cells are plotted.
            Valid strings also include leading characters of spike types

            | Ex: ``'ev'`` is equivalent to ``['evdist', 'evprox']``
        color : str | list of str | dict | None
            Input defining colors of plotted histograms. If str, all
            histograms plotted with same color. If list of str provided,
            histograms for each spike type will be plotted by cycling
            through colors in the list.

            If dict, colors must be specified for all spike_types as a key.
            If a group of spike types is defined by the `spike_types`
            parameter (see dictionary example for `spike_types`),
            the name of this group must be used to specify the colors.

            | Ex: ``{'evdist': 'g', 'evprox': 'r'}``, ``{'Tonic': 'b'}``

            If None, default color cycle used.
        show : bool
            If True, show the figure.
        **kwargs_hist : dict
            Additional keyword arguments to pass to ax.hist.

        Returns
        -------
        fig : instance of matplotlib Figure
            The matplotlib figure handle.
        """
        return plot_spikes_hist(
            self,
            trial_idx=trial_idx,
            ax=ax,
            spike_types=spike_types,
            invert_spike_types=invert_spike_types,
            color=color,
            show=show,
            **kwargs_hist,
        )

    def to_dict(self):
        """Return cell response as a dict object.

        Returns
        -------
        dict object containing the cell response
        """
        cell_response_data = dict()
        cell_response_data["spike_times"] = self.spike_times
        cell_response_data["spike_gids"] = self.spike_gids
        cell_response_data["spike_types"] = self.spike_types
        vsec_data = self.vsec
        cell_response_data["vsec"] = list()
        for trial in vsec_data:
            # Turn `int` gid keys into string values for hdf5 format
            trial = dict((str(key), val) for key, val in trial.items())
            cell_response_data["vsec"].append(trial)
        isec_data = self.isec
        cell_response_data["isec"] = list()
        for trial in isec_data:
            # Turn `int` gid keys into string values for hdf5 format
            trial = dict((str(key), val) for key, val in trial.items())
            cell_response_data["isec"].append(trial)
        ca_data = self.ca
        cell_response_data["ca"] = list()
        for trial in ca_data:
            # Turn `int` gid keys into string values for hdf5 format
            trial = dict((str(key), val) for key, val in trial.items())
            cell_response_data["ca"].append(trial)
        cell_response_data["times"] = self.times
        return cell_response_data

    def write(self, fname):
        """Write spiking activity per trial to a collection of files.

        Parameters
        ----------
        fname : str
            String format (e.g., 'spk_%d.txt' or 'spk_{0}.txt') of the
            path to the output spike file(s). If no string format
            is provided, the trial index will be automatically
            appended to the file name.

        Outputs
        -------
        A tab separated txt file for each trial where rows
            correspond to spikes, and columns correspond to
            1) spike time (ms),
            2) spike gid, and
            3) gid type
        """
        fname = str(fname)
        old_style = True
        try:
            fname % 0
        except TypeError:
            fname.format(0)
            old_style = False
        except TypeError:
            fname.replace(".txt", "_%d.txt")

        for trial_idx in range(len(self._spike_times)):
            if old_style:
                this_fname = fname % (trial_idx,)
            else:
                this_fname = fname.format(trial_idx)
            print(f"Writing file {this_fname}")
            with open(this_fname, "w") as f:
                for spike_idx in range(len(self._spike_times[trial_idx])):
                    f.write(
                        "{:.3f}\t{}\t{}\n".format(
                            self._spike_times[trial_idx][spike_idx],
                            int(self._spike_gids[trial_idx][spike_idx]),
                            self._spike_types[trial_idx][spike_idx],
                        )
                    )


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
    -------
    cell_response : CellResponse
        An instance of the CellResponse object.
    """
    warn(
        "Reading cell response from txt files is deprecated "
        "and will be removed in future versions. Please load "
        "cell response along with simulated network",
        FutureWarning,
        stacklevel=2,
    )
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
                    raise ValueError(
                        "gid_ranges must be provided if spike "
                        "types are unspecified in the "
                        "file %s" % (file,)
                    )
                spike_types += [list()]
        else:
            spike_times += [list()]
            spike_gids += [list()]
            spike_types += [list()]

    network_cell_names = ["L2_basket", "L2_pyramidal", "L5_basket", "L5_pyramidal"]
    # Need to cast "np.str_" to str type, since types are loaded above as np.str_, not
    # regular strings
    cell_type_names = list(
        cell_name for cell_name in network_cell_names if cell_name in str(spike_types)
    )
    cell_response = CellResponse(
        cell_type_names=cell_type_names,
        cell_type_metadata=None,
        spike_times=spike_times,
        spike_gids=spike_gids,
        spike_types=spike_types,
    )
    if gid_ranges is not None:
        cell_response.update_types(gid_ranges)

    return cell_response
