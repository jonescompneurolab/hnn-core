"""Class to handle the dipoles."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import warnings
import numpy as np
from numpy import convolve, hamming

from .viz import plot_dipole


def _hammfilt(x, winsz):
    """Convolve with a hamming window."""
    win = hamming(winsz)
    win /= sum(win)
    return convolve(x, win, 'same')


def simulate_dipole(net, n_trials=None, record_vsoma=False,
                    record_isoma=False, postproc=True):
    """Simulate a dipole given the experiment parameters.

    Parameters
    ----------
    net : Network object
        The Network object specifying how cells are
        connected.
    n_trials : int | None
        The number of trials to simulate. If None, the value in
        net.params['N_trials'] is used (must be >0)
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
        n_trials = net.params['N_trials']
    if n_trials < 1:
        raise ValueError("Invalid number of simulations: %d" % n_trials)

    # XXX needed in mpi_child.py:run()#L103; include fix in #211 or later PR
    net.params['N_trials'] = n_trials
    net._instantiate_drives(n_trials=n_trials)

    if isinstance(record_vsoma, bool):
        net.params['record_vsoma'] = record_vsoma
    else:
        raise TypeError("record_vsoma must be bool, got %s"
                        % type(record_vsoma).__name__)

    if isinstance(record_isoma, bool):
        net.params['record_isoma'] = record_isoma
    else:
        raise TypeError("record_isoma must be bool, got %s"
                        % type(record_isoma).__name__)

    dpls = _BACKEND.simulate(net, n_trials, postproc)

    return dpls


def read_dipole(fname, units='nAm'):
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
    if units == 'nAm':
        dpl.units = units
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
        The time vector
    sfreq : float
        The sampling frequency (in Hz)
    data : dict of array
        The dipole with keys 'agg', 'L2' and 'L5'
    nave : int
        Number of trials that were averaged to produce this Dipole
    """

    def __init__(self, times, data, nave=1):  # noqa: D102
        self.units = 'fAm'
        self.N = data.shape[0]
        self.times = times
        self.data = {'agg': data[:, 0], 'L2': data[:, 1], 'L5': data[:, 2]}
        self.nave = nave
        self.sfreq = 1000. / (times[1] - times[0])  # NB assumes len > 1

    def post_proc(self, N_pyr_x, N_pyr_y, winsz, fctr):
        """ Apply baseline, unit conversion, scaling and smoothing

       Parameters
        ----------
        N_pyr_x : int
            Number of Pyramidal cells in x direction
        N_pyr_y : int
            Number of Pyramidal cells in y direction
        winsz : int
            Smoothing window
        fctr : int
            Scaling factor
        """
        self.baseline_renormalize(N_pyr_x, N_pyr_y)
        self.convert_fAm_to_nAm()
        self.scale(fctr)
        self.smooth(winsz)

    def convert_fAm_to_nAm(self):
        """ must be run after baseline_renormalization()
        """
        for key in self.data.keys():
            self.data[key] *= 1e-6
        self.units = 'nAm'

    def scale(self, fctr):
        for key in self.data.keys():
            self.data[key] *= fctr
        return fctr

    def smooth(self, winsz):
        # XXX: add check to make sure self.times is
        # not smaller than winsz
        if winsz <= 1:
            return
        for key in self.data.keys():
            self.data[key] = _hammfilt(self.data[key], winsz)

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
        return plot_dipole(dpl=self, tmin=tmin, tmax=tmax, ax=ax, layer=layer,
                           decim=decim, show=show)

    def baseline_renormalize(self, N_pyr_x, N_pyr_y):
        """Only baseline renormalize if the units are fAm.

        Parameters
        ----------
        N_pyr_x : int
            Nr of cells (x)
        N_pyr_y : int
            Nr of cells (y)
        """
        if self.units != 'fAm':
            print("Warning, no dipole renormalization done because units"
                  " were in %s" % (self.units))
            return
        # L5: -3.6498e+00 * np.exp(1.9647e-03 * t) + -4.8023e+01
        # L2: 2.8063e-03 * np.exp(1.1149e-02 * t) + 4.4301e-02
        L5_exp = -3.6498e+00 * np.exp(1.9647e-03 * self.times) - 4.8023e+01
        L2_exp = 2.8063e-03 * np.exp(1.1149e-02 * self.times) + 4.4301e-02

        self.data['L5'] -= L5_exp * N_pyr_x * N_pyr_y
        self.data['L2'] -= L2_exp * N_pyr_x * N_pyr_y
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
