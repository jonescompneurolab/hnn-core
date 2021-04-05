"""Class to handle the dipoles."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import warnings
import numpy as np
from copy import deepcopy
from numpy import convolve, hamming

from .viz import plot_dipole, plot_psd, plot_tfr_morlet


def _hammfilt(x, winsz):
    """Convolve with a hamming window."""
    win = hamming(winsz)
    win /= sum(win)
    return convolve(x, win, 'same')


# Savitzky-Golay filtering, lifted and adapted from mne-python (0.22)
def _savgol_filter(data, h_freq, sfreq):
    """Filter the data using Savitzky-Golay polynomial method.

    Parameters
    ----------
    data : array-like
        The data to filter (1D)
    h_freq : float
        Approximate high cutoff frequency in Hz. Note that this
        is not an exact cutoff, since Savitzky-Golay filtering
        is done using polynomial fits
        instead of FIR/IIR filtering. This parameter is thus used to
        determine the length of the window over which a 5th-order
        polynomial smoothing is applied.
    sfreq : float
        The sampling frequency (in Hz)

    Returns
    -------
    filt_data : array-like
        The filtered data
    """  # noqa: E501
    from scipy.signal import savgol_filter

    h_freq = float(h_freq)
    if h_freq >= sfreq / 2.:
        raise ValueError('h_freq must be less than half the sample rate')

    # savitzky-golay filtering
    window_length = (int(np.round(sfreq / h_freq)) // 2) * 2 + 1
    # loop over 'agg', 'L2', and 'L5'
    filt_data = savgol_filter(data, axis=-1, polyorder=5,
                              window_length=window_length)
    return filt_data


def simulate_dipole(net, n_trials=None, record_vsoma=False,
                    record_isoma=False, postproc=True):
    """Simulate a dipole given the experiment parameters.

    Parameters
    ----------
    net : Network object
        The Network object specifying how cells are
        connected.
    n_trials : int | None
        The number of trials to simulate. If None, the 'N_trials' value
        of the ``params`` used to create ``net`` is used (must be >0)
    record_vsoma : bool
        Option to record somatic voltages from cells
    record_isoma : bool
        Option to record somatic currents from cells
    postproc : bool
        If False, no postprocessing applied to the dipole

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

    # XXX needed in mpi_child.py:run()#L103; include fix in #211 or later PR
    net._params['N_trials'] = n_trials
    net._instantiate_drives(n_trials=n_trials)
    net.cell_response.reset()  # see #290 for context; relevant for MPI

    if isinstance(record_vsoma, bool):
        net._params['record_vsoma'] = record_vsoma
    else:
        raise TypeError("record_vsoma must be bool, got %s"
                        % type(record_vsoma).__name__)

    if isinstance(record_isoma, bool):
        net._params['record_isoma'] = record_isoma
    else:
        raise TypeError("record_isoma must be bool, got %s"
                        % type(record_isoma).__name__)

    dpls = _BACKEND.simulate(net, n_trials, postproc)

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
    dpl = Dipole(dpl_data[:, 0], dpl_data[:, 1:4])
    return dpl


def average_dipoles(dpls):
    """Compute dipole averages over a list of Dipole objects.

    Parameters
    ----------
    dpls: list of Dipole objects
        Contains list of dipole objects, each with a `data` member containing
        'L2', 'L5' and 'agg' components

    Returns
    -------
    dpl: instance of Dipole
        A new dipole object with each component of `dpl.data` representing the
        average over the same components in the input list
    """
    # need at least one Dipole to get times
    if len(dpls) < 2:
        raise ValueError("Need at least two dipole object to compute an"
                         " average")

    for dpl_idx, dpl in enumerate(dpls):
        if dpl.nave > 1:
            raise ValueError("Dipole at index %d was already an average of %d"
                             " trials. Cannot reaverage" %
                             (dpl_idx, dpl.nave))

    agg_avg = np.mean(np.array([dpl.data['agg'] for dpl in dpls]), axis=0)
    L2_avg = np.mean(np.array([dpl.data['L2'] for dpl in dpls]), axis=0)
    L5_avg = np.mean(np.array([dpl.data['L5'] for dpl in dpls]), axis=0)

    avg_dpl_data = np.c_[agg_avg,
                         L2_avg,
                         L5_avg]

    avg_dpl = Dipole(dpls[0].times, avg_dpl_data)

    # set nave to the number of trials averaged in this dipole
    avg_dpl.nave = len(dpls)

    return avg_dpl


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
    data : array (n_times x 3)
        The data. The first column represents 'agg',
        the second 'L2' and the last one 'L5'
    nave : int
        Number of trials that were averaged to produce this Dipole. Defaults
        to 1

    Attributes
    ----------
    times : array
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
        self.times = times
        self.data = {'agg': data[:, 0], 'L2': data[:, 1], 'L5': data[:, 2]}
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
        if not isinstance(window_len, (float, int)) or window_len < 0:
            raise ValueError('Window length must be a non-negative number')
        elif 0 < window_len < 1:
            raise ValueError('Window length less than 1 ms is not supported')

        # convolutional filter length is given in samples
        winsz = np.round(1e-3 * window_len * self.sfreq)
        if winsz > len(self.times):
            raise ValueError(
                f'Window length too long: {winsz} samples; data length is '
                f'{len(self.times)} samples')

        for key in self.data.keys():
            self.data[key] = _hammfilt(self.data[key], winsz)

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
             show=True):
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
        show : bool
            If True, show the figure

        Returns
        -------
        fig : instance of plt.fig
            The matplotlib figure handle.
        """
        return plot_dipole(self, tmin=tmin, tmax=tmax, ax=ax, layer=layer,
                           decim=decim, show=show)

    def plot_psd(self, fmin=0, fmax=None, tmin=None, tmax=None, layer='agg',
                 ax=None, show=True):
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
                        layer=layer, ax=ax, show=show)

    def plot_tfr_morlet(self, freqs, n_cycles=7., tmin=None, tmax=None,
                        layer='agg', decim=None, padding='zeros', ax=None,
                        colormap='inferno', colorbar=True, show=True):
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
            colormap=colormap, colorbar=colorbar, show=show)

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

        X = np.r_[[self.times, self.data['agg'], self.data['L2'],
                   self.data['L5']]].T
        np.savetxt(fname, X, fmt=['%3.3f', '%5.4f', '%5.4f', '%5.4f'],
                   delimiter='\t')
