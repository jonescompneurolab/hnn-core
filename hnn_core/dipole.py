"""Class to handle the dipoles."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Sam Neymotin <samnemo@gmail.com>

import warnings
import numpy as np
from copy import deepcopy
from .externals.mne import _check_option

from .viz import plot_dipole, plot_psd, plot_tfr_morlet


def simulate_dipole(net, tstop, dt=0.025, n_trials=None, record_vsec=False,
                    record_isec=False, postproc=False):
    """Simulate a dipole given the experiment parameters.

    Parameters
    ----------
    net : Network object
        The Network object specifying how cells are
        connected.
    tstop : float
        The simulation stop time (ms).
    dt : float
        The integration time step of h.CVode (ms)
    n_trials : int | None
        The number of trials to simulate. If None, the 'N_trials' value
        of the ``params`` used to create ``net`` is used (must be >0)
    record_vsec : 'all' | 'soma' | False
        Option to record voltages from all sections ('all'), or just
        the soma ('soma'). Default: False.
    record_isec : 'all' | 'soma' | False
        Option to record voltages from all sections ('all'), or just
        the soma ('soma'). Default: False.
    postproc : bool
        If True, smoothing (``dipole_smooth_win``) and scaling
        (``dipole_scalefctr``) values are read from the parameter file, and
        applied to the dipole objects before returning. Note that this setting
        only affects the dipole waveforms, and not somatic voltages, possible
        extracellular recordings etc. The preferred way is to use the
        :meth:`~hnn_core.dipole.Dipole.smooth` and
        :meth:`~hnn_core.dipole.Dipole.scale` methods instead. Default: False.

    Returns
    -------
    dpls: list
        List of dipole objects for each trials
    """

    from .parallel_backends import _BACKEND, JoblibBackend

    if _BACKEND is None:
        _BACKEND = JoblibBackend(n_jobs=1)

    if n_trials is None:
        n_trials = net._params['N_trials']
    if n_trials < 1:
        raise ValueError("Invalid number of simulations: %d" % n_trials)

    if not net.connectivity:
        warnings.warn('No connections instantiated in network. Consider using '
                      'net = jones_2009_model() or net = law_2021_model() to '
                      'create a predefined network from published models.',
                      UserWarning)
    # ADD DRIVE WARNINGS HERE
    if not net.external_drives and not net.external_biases:
        warnings.warn('No external drives or biases loaded', UserWarning)

    for drive_name, drive in net.external_drives.items():
        if 'tstop' in drive['dynamics']:
            if drive['dynamics']['tstop'] is None:
                drive['dynamics']['tstop'] = tstop
    for bias_name, bias in net.external_biases.items():
        for cell_type, bias_cell_type in bias.items():
            if bias_cell_type['tstop'] is None:
                bias_cell_type['tstop'] = tstop
            if bias_cell_type['tstop'] < 0.:
                raise ValueError('End time of tonic input cannot be negative')
            duration = bias_cell_type['tstop'] - bias_cell_type['t0']
            if duration < 0.:
                raise ValueError('Duration of tonic input cannot be negative')

    net._instantiate_drives(n_trials=n_trials, tstop=tstop)
    net._reset_rec_arrays()

    _check_option('record_vsec', record_vsec, ['all', 'soma', False])

    net._params['record_vsec'] = record_vsec

    _check_option('record_isec', record_isec, ['all', 'soma', False])

    net._params['record_isec'] = record_isec

    if postproc:
        warnings.warn('The postproc-argument is deprecated and will be removed'
                      ' in a future release of hnn-core. Please define '
                      'smoothing and scaling explicitly using Dipole methods.',
                      DeprecationWarning)
    dpls = _BACKEND.simulate(net, tstop, dt, n_trials, postproc)

    return dpls


def read_dipole(fname):
    """Read dipole values from a file and create a Dipole instance.

    Parameters
    ----------
    fname : str
        Full path to the input file (.txt)

    Returns
    -------
    dpl : Dipole
        The instance of Dipole class
    """
    dpl_data = np.loadtxt(fname, dtype=float)
    ncols = dpl_data.shape[1]
    if ncols not in (2, 4):
        raise ValueError(
            f'Data are supposed to have 2 or 4 columns while we have {ncols}.')
    dpl = Dipole(dpl_data[:, 0], dpl_data[:, 1:])
    return dpl


def average_dipoles(dpls):
    """Compute dipole averages over a list of Dipole objects.

    Parameters
    ----------
    dpls : list of Dipole objects
        Contains list of dipole objects, each with a `data` member containing
        'L2', 'L5' and 'agg' components

    Returns
    -------
    dpl : instance of Dipole
        A new dipole object with each component of `dpl.data` representing the
        average over the same components in the input list
    """
    scale_applied = dpls[0].scale_applied
    for dpl_idx, dpl in enumerate(dpls):
        if dpl.scale_applied != scale_applied:
            raise RuntimeError('All dipoles must be scaled equally!')
        if not isinstance(dpl, Dipole):
            raise ValueError(
                f"All elements in the list should be instances of "
                f"Dipole. Got {type(dpl)}")
        if dpl.nave > 1:
            raise ValueError("Dipole at index %d was already an average of %d"
                             " trials. Cannot reaverage" %
                             (dpl_idx, dpl.nave))

    avg_data = list()
    layers = dpl.data.keys()
    for layer in layers:
        avg_data.append(
            np.mean(np.array([dpl.data[layer] for dpl in dpls]), axis=0)
        )
    avg_data = np.c_[avg_data].T
    avg_dpl = Dipole(dpls[0].times, avg_data)
    # The averaged scale should equal all scals in the input dpl list.
    avg_dpl.scale_applied = scale_applied

    # set nave to the number of trials averaged in this dipole
    avg_dpl.nave = len(dpls)

    return avg_dpl


def _rmse(dpl, exp_dpl, tstart=0.0, tstop=0.0, weights=None):
    """ Calculates RMSE between data in dpl and exp_dpl
    Parameters
    ----------
    dpl : instance of Dipole
        A dipole object with simulated data
    exp_dpl : instance of Dipole
        A dipole object with experimental data
    tstart : None | float
        Time at beginning of range over which to calculate RMSE
    tstop : None | float
        Time at end of range over which to calculate RMSE
    weights : None | array
        An array of weights to be applied to each point in
        simulated dpl. Must have length >= dpl.data
        If None, weights will be replaced with 1's for typical RMSE
        calculation.

    Returns
    -------
    err : float
        Weighted RMSE between data in dpl and exp_dpl
    """
    from scipy import signal

    exp_times = exp_dpl.times
    sim_times = dpl.times

    # do tstart and tstop fall within both datasets?
    # if not, use the closest data point as the new tstop/tstart
    for tseries in [exp_times, sim_times]:
        if tstart < tseries[0]:
            tstart = tseries[0]
        if tstop > tseries[-1]:
            tstop = tseries[-1]

    # make sure start and end times are valid for both dipoles
    exp_start_index = (np.abs(exp_times - tstart)).argmin()
    exp_end_index = (np.abs(exp_times - tstop)).argmin()
    exp_length = exp_end_index - exp_start_index

    sim_start_index = (np.abs(sim_times - tstart)).argmin()
    sim_end_index = (np.abs(sim_times - tstop)).argmin()
    sim_length = sim_end_index - sim_start_index

    if weights is None:
        # weighted RMSE with weights of all 1's is equivalent to
        # normal RMSE
        weights = np.ones(len(sim_times[0:sim_end_index]))
    weights = weights[sim_start_index:sim_end_index]

    dpl1 = dpl.data['agg'][sim_start_index:sim_end_index]
    dpl2 = exp_dpl.data['agg'][exp_start_index:exp_end_index]

    if (sim_length > exp_length):
        # downsample simulation timeseries to match exp data
        dpl1 = signal.resample(dpl1, exp_length)
        weights = signal.resample(weights, exp_length)
        indices = np.where(weights < 1e-4)
        weights[indices] = 0
    elif (sim_length < exp_length):
        # downsample exp timeseries to match simulation data
        dpl2 = signal.resample(dpl2, sim_length)

    return np.sqrt((weights * ((dpl1 - dpl2) ** 2)).sum() / weights.sum())


class Dipole(object):
    """Dipole class.

    An instance of the ``Dipole``-class contains the simulated dipole moment
    timecourses for L2 and L5 pyramidal cells, as well as their aggregate
    (``'agg'``). The units of the dipole moment are in ``nAm``
    (1e-9 Ampere-meters).

    Parameters
    ----------
    times : array (n_times,)
        The time vector (in ms)
    data : array, shape (n_times x n_layers)
        The data. The first column represents 'agg' (the total diple),
        the second 'L2' layer and the last one 'L5' layer. For experimental
        data, it can contain only one column.
    nave : int
        Number of trials that were averaged to produce this Dipole. Defaults
        to 1

    Attributes
    ----------
    times : array-like
        The time vector (in ms)
    sfreq : float
        The sampling frequency (in Hz)
    data : dict of array
        Dipole moment timecourse arrays with keys 'agg', 'L2' and 'L5'
    nave : int
        Number of trials that were averaged to produce this Dipole
    scale_applied : int or float
        The total factor by which the dipole has been scaled (using
        :meth:`~hnn_core.dipole.Dipole.scale`).
    """

    def __init__(self, times, data, nave=1):  # noqa: D102
        self.times = np.array(times)

        if data.ndim == 1:
            data = data[:, None]

        if data.shape[1] == 3:
            self.data = {'agg': data[:, 0], 'L2': data[:, 1], 'L5': data[:, 2]}
        elif data.shape[1] == 1:
            self.data = {'agg': data[:, 0]}

        self.nave = nave
        self.sfreq = 1000. / (times[1] - times[0])  # NB assumes len > 1
        self.scale_applied = 1  # for visualisation

    def copy(self):
        """Return a copy of the Dipole instance

        Returns
        -------
        dpl_copy : instance of Dipole
            A copy of the Dipole instance.
        """
        return deepcopy(self)

    def _post_proc(self, window_len, fctr):
        """Apply scaling and smoothing from param-files (DEPRECATE)

        Parameters
        ----------
        window_len : int
            Smoothing window in ms
        fctr : int
            Scaling factor
        """
        self.scale(fctr)

        if window_len > 0:  # this is to allow param-files with len==0
            self.smooth(window_len)

    def _convert_fAm_to_nAm(self):
        """The NEURON simulator output is in fAm, convert to nAm

        NB! Must be run `after` :meth:`Dipole.baseline_renormalization`
        """
        for key in self.data.keys():
            self.data[key] *= 1e-6

    def scale(self, factor):
        """Scale (multiply) the dipole moment by a fixed factor

        The attribute ``Dipole.scale_applied`` is updated to reflect factors
        applied and displayed in plots.

        Parameters
        ----------
        factor : int
            Scaling factor, applied to the data in-place.
        """
        for key in self.data.keys():
            self.data[key] *= factor
        self.scale_applied *= factor
        return self

    def smooth(self, window_len):
        """Smooth the dipole waveform using Hamming-windowed convolution

        Note that this method operates in-place, i.e., it will alter the data.
        If you prefer a filtered copy, consider using the
        :meth:`~hnn_core.dipole.Dipole.copy`-method.

        Parameters
        ----------
        window_len : float
            The length (in ms) of a `~numpy.hamming` window to convolve the
            data with.

        Returns
        -------
        dpl_copy : instance of Dipole
            A copy of the modified Dipole instance.
        """
        from .utils import smooth_waveform

        for key in self.data.keys():
            self.data[key] = smooth_waveform(self.data[key], window_len,
                                             self.sfreq)

        return self

    def savgol_filter(self, h_freq):
        """Smooth the dipole waveform using Savitzky-Golay filtering

        Note that this method operates in-place, i.e., it will alter the data.
        If you prefer a filtered copy, consider using the
        :meth:`~hnn_core.dipole.Dipole.copy`-method. The high-frequency cutoff
        value of a Savitzky-Golay filter is approximate; see the SciPy
        reference: :func:`~scipy.signal.savgol_filter`.

        Parameters
        ----------
        h_freq : float or None
            Approximate high cutoff frequency in Hz. Note that this
            is not an exact cutoff, since Savitzky-Golay filtering
            is done using polynomial fits
            instead of FIR/IIR filtering. This parameter is thus used to
            determine the length of the window over which a 5th-order
            polynomial smoothing is applied.

        Returns
        -------
        dpl_copy : instance of Dipole
            A copy of the modified Dipole instance.
        """
        from .utils import _savgol_filter
        if h_freq < 0:
            raise ValueError('h_freq cannot be negative')
        elif h_freq > 0.5 * self.sfreq:
            raise ValueError(
                'h_freq must be less than half the sample rate')
        for key in self.data.keys():
            self.data[key] = _savgol_filter(self.data[key],
                                            h_freq,
                                            self.sfreq)
        return self

    def plot(self, tmin=None, tmax=None, layer='agg', decim=None, ax=None,
             color='k', show=True):
        """Simple layer-specific plot function.

        Parameters
        ----------
        tmin : float or None
            Start time of plot (in ms). If None, plot entire simulation.
        tmax : float or None
            End time of plot (in ms). If None, plot entire simulation.
        layer : str
            The layer to plot. Can be one of 'agg', 'L2', and 'L5'
        decimate : int
            Factor by which to decimate the raw dipole traces (optional)
        ax : instance of matplotlib figure | None
            The matplotlib axis
        color : tuple of float
            RGBA value to use for plotting. By default, 'k' (black)
        show : bool
            If True, show the figure

        Returns
        -------
        fig : instance of plt.fig
            The matplotlib figure handle.
        """
        return plot_dipole(self, tmin=tmin, tmax=tmax, ax=ax, layer=layer,
                           decim=decim, color=color, show=show)

    def plot_psd(self, fmin=0, fmax=None, tmin=None, tmax=None, layer='agg',
                 color=None, label=None, ax=None, show=True):
        """Plot power spectral density (PSD) of dipole time course

        Applies `~scipy.signal.periodogram` from SciPy with
        ``window='hamming'``.
        Note that no spectral averaging is applied across time, as most
        ``hnn_core`` simulations are short-duration. However, passing a list of
        `Dipole` instances will plot their average (Hamming-windowed) power,
        which resembles the `Welch`-method applied over time.

        Parameters
        ----------
        dpl : instance of Dipole | list of Dipole instances
            The Dipole object.
        fmin : float
            Minimum frequency to plot (in Hz). Default: 0 Hz
        fmax : float
            Maximum frequency to plot (in Hz). Default: None (plot up to
            Nyquist)
        tmin : float or None
            Start time of data to include (in ms). If None, use entire
            simulation.
        tmax : float or None
            End time of data to include (in ms). If None, use entire
            simulation.
        layer : str, default 'agg'
            The layer to plot. Can be one of 'agg', 'L2', and 'L5'
        color : str | tuple | None
            The line color of PSD
        label : str | None
            Line label for PSD
        ax : instance of matplotlib figure | None
            The matplotlib axis.
        show : bool
            If True, show the figure

        Returns
        -------
        fig : instance of matplotlib Figure
            The matplotlib figure handle.
        """
        return plot_psd(self, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,
                        layer=layer, color=color, label=label, ax=ax,
                        show=show)

    def plot_tfr_morlet(self, freqs, n_cycles=7., tmin=None, tmax=None,
                        layer='agg', decim=None, padding='zeros', ax=None,
                        colormap='inferno', colorbar=True,
                        colorbar_inside=False, show=True):
        """Plot Morlet time-frequency representation of dipole time course

        NB: Calls `~mne.time_frequency.tfr_array_morlet`, so ``mne`` must be
        installed.

        Parameters
        ----------
        dpl : instance of Dipole | list of Dipole instances
            The Dipole object. If a list of dipoles is given, the power is
            calculated separately for each trial, then averaged.
        freqs : array
            Frequency range of interest.
        n_cycles : float or array of float, default 7.0
            Number of cycles. Fixed number or one per frequency.
        tmin : float or None
            Start time of plot in milliseconds. If None, plot entire
            simulation.
        tmax : float or None
            End time of plot in milliseconds. If None, plot entire simulation.
        layer : str, default 'agg'
            The layer to plot. Can be one of 'agg', 'L2', and 'L5'
        decim : int or list of int or None (default)
            Optional (integer) factor by which to decimate the raw dipole
            traces. The SciPy function :func:`~scipy.signal.decimate` is used,
            which recommends values <13. To achieve higher decimation factors,
            a list of ints can be provided. These are applied successively.
        padding : str or None
            Optional padding of the dipole time course beyond the plotting
            limits. Possible values are: 'zeros' for padding with 0's
            (default), 'mirror' for mirror-image padding.
        ax : instance of matplotlib figure | None
            The matplotlib axis
        colormap : str
            The name of a matplotlib colormap, e.g., 'viridis'. Default:
            'inferno'
        colorbar : bool
            If True (default), adjust figure to include colorbar.
        colorbar_inside: bool, default False
            Put the color inside the heatmap if True.
        show : bool
            If True, show the figure

        Returns
        -------
        fig : instance of matplotlib Figure
            The matplotlib figure handle.
        """
        return plot_tfr_morlet(
            self, freqs, n_cycles=n_cycles, tmin=tmin, tmax=tmax,
            layer=layer, decim=decim, padding=padding, ax=ax,
            colormap=colormap, colorbar=colorbar,
            colorbar_inside=colorbar_inside, show=show)

    def _baseline_renormalize(self, N_pyr_x, N_pyr_y):
        """Only baseline renormalize if the units are fAm.

        Parameters
        ----------
        N_pyr_x : int
            Nr of cells (x)
        N_pyr_y : int
            Nr of cells (y)
        """
        # N_pyr cells in grid. This is PER LAYER
        N_pyr = N_pyr_x * N_pyr_y
        # dipole offset calculation: increasing number of pyr
        # cells (L2 and L5, simultaneously)
        # with no inputs resulted in an aggregate dipole over the
        # interval [50., 1000.] ms that
        # eventually plateaus at -48 fAm. The range over this interval
        # is something like 3 fAm
        # so the resultant correction is here, per dipole
        # dpl_offset = N_pyr * 50.207
        dpl_offset = {
            # these values will be subtracted
            'L2': N_pyr * 0.0443,
            'L5': N_pyr * -49.0502
            # 'L5': N_pyr * -48.3642,
            # will be calculated next, this is a placeholder
            # 'agg': None,
        }
        # L2 dipole offset can be roughly baseline shifted over
        # the entire range of t
        self.data['L2'] -= dpl_offset['L2']
        # L5 dipole offset should be different for interval [50., 500.]
        # and then it can be offset
        # slope (m) and intercept (b) params for L5 dipole offset
        # uncorrected for N_cells
        # these values were fit over the range [37., 750.)
        m = 3.4770508e-3
        b = -51.231085
        # these values were fit over the range [750., 5000]
        t1 = 750.
        m1 = 1.01e-4
        b1 = -48.412078
        # piecewise normalization
        self.data['L5'][self.times <= 37.] -= dpl_offset['L5']
        self.data['L5'][(self.times > 37.) & (self.times < t1)] -= N_pyr * \
            (m * self.times[(self.times > 37.) & (self.times < t1)] + b)
        self.data['L5'][self.times >= t1] -= N_pyr * \
            (m1 * self.times[self.times >= t1] + b1)
        # recalculate the aggregate dipole based on the baseline
        # normalized ones
        self.data['agg'] = self.data['L2'] + self.data['L5']

    def write(self, fname):
        """Write dipole values to a file.

        Parameters
        ----------
        fname : str
            Full path to the output file (.txt)

        Outputs
        -------
        A tab separatd txt file where rows correspond
            to samples and columns correspond to
            1) time (s),
            2) aggregate current dipole (scaled nAm),
            3) L2/3 current dipole (scaled nAm), and
            4) L5 current dipole (scaled nAm)
        """

        if self.nave > 1:
            warnings.warn("Saving Dipole to file that is an average of %d"
                          " trials" % self.nave)

        X = [self.times]
        fmt = ['%3.3f']
        for data in self.data.values():
            X.append(data)
            fmt.append('%5.4f')
        X = np.r_[X].T

        np.savetxt(fname, X, fmt=fmt, delimiter='\t')
