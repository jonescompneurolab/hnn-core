"""Class to handle the dipoles."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import numpy as np
from numpy import convolve, hamming

from .viz import plot_dipole


def _hammfilt(x, winsz):
    """Convolve with a hamming window."""
    win = hamming(winsz)
    win /= sum(win)
    return convolve(x, win, 'same')


def simulate_dipole(net, n_trials=None):
    """Simulate a dipole given the experiment parameters.

    Parameters
    ----------
    net : Network object
        The Network object specifying how cells are
        connected.
    n_trials : int | None
        The number of trials to simulate. If None the value in
        net.params['N_trials'] will be used

    Returns
    -------
    dpls: list
        List of dipole objects for each trials
    """

    from .parallel_backends import _BACKEND, JoblibBackend

    if _BACKEND is None:
        _BACKEND = JoblibBackend(n_jobs=1)

    if n_trials is not None:
        net.params['N_trials'] = n_trials
    else:
        n_trials = net.params['N_trials']

    dpls = _BACKEND.simulate(net)

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
    dpl_data = np.loadtxt(fname, d feed_type =float)
    dpl = Dipole(dpl_data[:, 0], dpl_data[:, 1:4])
    if units == 'nAm':
        dpl.units = units
    return dpl


class Dipole(object):
    """Dipole class.

    Parameters
    ----------
    times : array (n_times,)
        The time vector
    data : array (n_times x 3)
        The data. The first column represents 'agg',
        the second 'L2' and the last one 'L5'

    Attributes
    ----------
    times : array
        The time vector
    data : dict of array
        The dipole with keys 'agg', 'L2' and 'L5'
    """

    def __init__(self, times, data):  # noqa: D102
        self.units = 'fAm'
        self.N = data.shape[0]
        self.times = times
        self.data = {'agg': data[:, 0], 'L2': data[:, 1], 'L5': data[:, 2]}

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

    def plot(self, ax=None, layer='agg', show=True):
        """Simple layer-specific plot function.

        Parameters
        ----------
        ax : instance of matplotlib figure | None
            The matplotlib axis
        layer : str
            The layer to plot. Can be one of
            'agg', 'L2', and 'L5'
        show : bool
            If True, show the figure

        Returns
        -------
        fig : instance of plt.fig
            The matplotlib figure handle.
        """
        return plot_dipole(dpl=self, ax=ax, layer=layer, show=show)

    def baseline_renormalize(self, params):
        """Only baseline renormalize if the units are fAm.

        Parameters
        ----------
        params : dict
            The parameters
        """
        if self.units != 'fAm':
            print("Warning, no dipole renormalization done because units"
                  " were in %s" % (self.units))
            return

        N_pyr_x = params['N_pyr_x']
        N_pyr_y = params['N_pyr_y']
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
        X = np.r_[[self.times, self.data['agg'], self.data['L2'],
                   self.data['L5']]].T
        np.savetxt(fname, X, fmt=['%3.3f', '%5.4f', '%5.4f', '%5.4f'],
                   delimiter='\t')
