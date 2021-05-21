"""
Handler classes to calculate Local Field Potentials (LFP) at ideal (point-like)
electrodes based on net transmembrane currents of all neurons in the network.

The code is inspired by [1], but important modifications were made to comply
with the original derivation of the 'line source approximation method'.
LFPsim - Simulation scripts to compute Local Field Potentials (LFP) fro

References
----------
1 . Parasuram H, Nair B, D'Angelo E, Hines M, Naldi G, Diwakar S (2016)
Computational Modeling of Single Neuron Extracellular Electric Potentials and
Network Local Field Potentials using LFPsim. Front Comput Neurosci 10:65.
2. Holt, G. R. (1998) A critical reexamination of some assumptions and
implications of cable theory in neurobiology. CalTech, PhD Thesis.
"""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Sam Neymotin <samnemo@gmail.com>
#          Christopher Bailey <cjb@cfin.au.dk>

from neuron import h

import numpy as np
from numpy.linalg import norm


def _get_sections_on_this_rank(sec_type='Pyr'):
    ls = h.allsec()
    ls = [s for s in ls if sec_type in s.name()]
    return ls


def _get_segment_counts(all_sections):
    """The segment count of a section excludes the endpoints (0, 1)"""
    seg_counts = list()
    for sec in all_sections:
        seg_counts.append(sec.nseg)
    return seg_counts


def _transfer_resistance(section, electrode_pos, sigma, method):
    """Transfer resistance between section and electrode position.

    To arrive at the extracellular potential, the value returned by this
    function is multiplied by the net transmembrane current flowing through all
    segments of the section. Hence the term "resistance" (voltage equals
    current times resistance).

    Parameters
    ----------
    section : h.Section()
        The NEURON section.
    ele_pos : list (x, y, z)
        The x, y, z coordinates of the electrode (in um)
    sigma : float
        Extracellular conductivity (in S/m)
    method : str
        Approximation to use. 'psa' assigns all transmembrane currents to the
        center point (0.5) (point source approximation). 'lsa' treats the
        section as a line source, but a single multiplier is calculated for
        each section (line source approximation).

    Returns
    -------
    vres : list
        The resistance at each section.
    """
    electrode_pos = np.array(electrode_pos)  # electrode position to Numpy

    start = np.array([section.x3d(0), section.y3d(0), section.z3d(0)])
    end = np.array([section.x3d(1), section.y3d(1), section.z3d(1)])

    if method == 'psa':

        mid = (start + end) / 2.

        # distance from section midpoint to electrode
        dis = norm(electrode_pos - mid)

        # setting radius limit
        if dis < section.diam / 2.0:
            dis = section.diam / 2.0 + 0.1

        phi = 1. / dis

    elif method == 'lsa':
        # From: Appendix C (pp. 137) in Holt, G. R. A critical reexamination of
        # some assumptions and implications of cable theory in neurobiology.
        # CalTech, PhD Thesis (1998).
        #
        #                      Electrode position
        #   |------ L --------*
        #                 b / | R
        #                 /   |
        #   0==== a ====1- H -+
        #
        # a: vector oriented along the section
        # b: position vector of electrode with respect to section end (1)
        # H: parallel distance from section end to electrode
        # R: radial distance from section end to electrode
        # L: parallel distance from section start to electrode
        # Note that there are three distinct regimes to this approximation,
        # depending on the electrode position along the section axis.
        a = end - start
        norm_a = norm(a)
        b = electrode_pos - end
        # projection: H = a.cos(theta) = a.dot(b) / |a|
        H = np.dot(b, a) / norm_a  # NB can be negative
        L = H + norm_a
        R2 = np.dot(b, b) - H ** 2  # NB squares
        # To avoid numerical errors when electrode is placed (anywhere) on the
        # section axis, enforce minimal axial distance
        R2 = max(R2, (section.diam / 2.0 + 0.1) ** 2)

        if L < 0 and H < 0:  # electrode is "behind" section
            num = np.sqrt(H ** 2 + R2) - H  # == norm(b) - H
            denom = np.sqrt(L ** 2 + R2) - L
        elif L > 0 and H < 0:  # electrode is "on top of" section
            num = (np.sqrt(H ** 2 + R2) - H) * (L + np.sqrt(L ** 2 + R2))
            denom = R2
        else:  # electrode is "ahead of" section
            num = np.sqrt(L ** 2 + R2) + L
            denom = np.sqrt(H ** 2 + R2) + H  # == norm(b) + H

        phi = np.log(num / denom) / norm_a

    # [dis]: um; [sigma]: S / m
    # [phi / sigma] = [1/dis] / [sigma] = 1 / [dis] x [sigma]
    # [dis] x [sigma] = um x (S / m) = 1e-6 S
    # transmembrane current returned by _ref_i_membrane_ is in [nA]
    # ==> 1e-9 A x (1 / 1e-6 S) = 1e-3 V = mV
    # ===> multiply by 1e3 to get uV
    return 1000.0 * phi / (4.0 * np.pi * sigma)


def _get_lfp_plot_data(times, data, tmin, tmax):
    plot_tmin = times[0]
    if tmin is not None:
        plot_tmin = max(tmin, plot_tmin)
    plot_tmax = times[-1]
    if tmax is not None:
        plot_tmax = min(tmax, plot_tmax)

    times = np.array(times)
    mask = np.logical_and(times >= plot_tmin, times < plot_tmax)
    times = times[mask]

    data = np.array(data)[mask]

    return data, times


class LFPArray:
    """Local field potential (LFP) electrode array class.

    Parameters
    ----------
    positions : tuple | list of tuple
        The (x, y, z) coordinates (in um) of the LFP electrodes.
    sigma : float
        Extracellular conductivity, in S/m, of the assumed infinite,
        homogeneous volume conductor that the cell and electrode are in.
    method : str
        Approximation to use. 'psa' (default) assigns all transmembrane
        currents to the center point (0.5) (point source approximation).
        'lsa' treats the section as a line source, and a single multiplier is
        calculated for each section (line source approximation).

    Attributes
    ----------
    voltages : list of float
        The LFP voltages at the electrode positions (in uV).
    times : list of float
        The time points the LFP is sampled at (ms)
    sfreq : float
        Sampling rate of the LFP data (Hz).

    Notes
    -----
    See Table 5 in http://jn.physiology.org/content/104/6/3388.long for
    measured values of sigma in rat cortex (note units there are mS/cm)
    """

    def __init__(self, positions, sigma=0.3, method='psa', times=list(),
                 voltages=list()):
        self.positions = positions
        self.sigma = sigma
        self.method = method
        self.times = times
        self._data = voltages

    def __getitem__(self, trial_no):
        if isinstance(trial_no, int):
            return_data = [self._data[trial_no]]
        elif isinstance(trial_no, slice):
            return_data = self._data[trial_no]
        elif not isinstance(trial_no, (list, tuple)):
            return_data = [self._data[trial] for trial in trial_no]

        return LFPArray(self.positions, sigma=self.sigma, method=self.method,
                        times=self.times, voltages=return_data)

    def __repr__(self):
        class_name = self.__class__.__name__
        msg = (f'{len(self.positions)} electrodes, sigma={self.sigma}, '
               f'method={self.method}')
        if len(self._data) > 0:
            msg += f' | {len(self._data)} trials, {len(self.times)} times'
        return f'<{class_name} | {msg}>'

    def __len__(self):
        return len(self.positions)

    @property
    def sfreq(self):
        """Return the sampling rate of the LFP data."""
        if len(self.times) > 1:
            return 1000. / (self.times[1] - self.times[0])  # times in ms
        else:
            return None

    def reset(self):
        self._data = list()
        self.times = list()

    def get_data(self):
        return np.array(self._data)

    def plot(self, *, trial_no=None, contact_no=None, window_len=None,
             tmin=None, tmax=None, ax=None, decim=None, color=None, show=True):

        if trial_no is None:
            plot_data = self.get_data()
        elif isinstance(trial_no, (list, tuple, int, slice)):
            plot_data = self.get_data()[trial_no, ]

        if isinstance(contact_no, (list, tuple, int, slice)):
            plot_data = plot_data[:, contact_no, ]
        elif contact_no is not None:
            raise ValueError('invalid')

        for trial_data in plot_data:
            _plot_lfp(self.times, trial_data, self.sfreq,
                      window_len=window_len, tmin=tmin, tmax=tmax, ax=ax,
                      decim=decim, color=color, show=show)


def _plot_lfp(times, data, sfreq, window_len=None, tmin=None,
              tmax=None, ax=None, decim=None, color=None, show=True):
    """Plot LFP traces

    Parameters
    ----------
    trial_idx : int | list of int
        Trial number(s) to plot
    contact_idx : int | list of int
        Electrode contact number(s) to plot
    window_len : float
        If set, apply a Hamming-windowed convolution kernel of specified
        length (in ms) to the data before plotting. Default: None
    tmin : float or None
        Start time of plot in milliseconds. If None, plot entire
        simulation.
    tmax : float or None
        End time of plot in milliseconds. If None, plot entire simulation.
    ax : instance of matplotlib figure | None
        The matplotlib axis
    decim : int or list of int or None (default)
        Optional (integer) factor by which to decimate the raw dipole
        traces. The SciPy function :func:`~scipy.signal.decimate` is used,
        which recommends values <13. To achieve higher decimation factors,
        a list of ints can be provided. These are applied successively.
    color : tuple of float
        RGBA value to use for plotting (optional)
    show : bool
        If True, show the figure

    Returns
    -------
    fig : instance of plt.fig
        The matplotlib figure handle.
    """
    import matplotlib.pyplot as plt
    from .dipole import _hammfilt
    from .viz import _decimate_plot_data, plt_show

    if ax is None:
        _, ax = plt.subplots(1, 1)

    for contact_no, trace in enumerate(np.atleast_2d(data)):
        plot_data, plot_times = _get_lfp_plot_data(times, trace, tmin, tmax)
        if window_len is not None:
            winsz = np.round(1e-3 * window_len * sfreq)
            plot_data = _hammfilt(plot_data, winsz)
        if decim is not None:
            plot_data, plot_times = _decimate_plot_data(decim, plot_data,
                                                        plot_times)

        ax.plot(plot_times, plot_data, label=f'C{contact_no}', color=color)

    ax.ticklabel_format(axis='both', scilimits=(-2, 3))
    ax.set_xlabel('Time (ms)')
    ylabel = r'Electric potential ($\mu V$)'
    ax.set_ylabel(ylabel, multialignment='center')

    plt_show(show)
    return ax.get_figure()


class _LFPArray:
    """Class for electrode arrays containing NEURON objects

    The handler is set up to maintain a vector of membrane currents at at every
    inner segment of every section of every cell on each CVODE integration
    step. In addition, it records a time vector of sample times. This class
    must be instantiated and attached to the network during the building
    process. It is used in conjunction with the calculation of extracellular
    potentials.

    Parameters
    ----------
    array : instance of LFPArray
        Initialised, e.g., by the Network.add_electrode_array()-method
    cvode : instance of h.CVode
        Multi order variable time step integration method.
    """
    def __init__(self, array, cvode=None):
        self.positions = array.positions
        self.n_contacts = len(self.positions)
        self.sigma = array.sigma
        self.method = array.method

        # Attach a callback for calculating the potentials at each time step.
        # Enables fast calculation of transmembrane current (nA) at each
        # segment. Note that this will run on each rank, so it is safe to use
        # the extra_scatter_gather-method, which docs say doesn't support
        # "multiple threads".
        cvode.extra_scatter_gather(0, self.calc_potential_callback)

    @property
    def n_samples(self):
        """Return the length (in samples) of the LFP data."""
        return int(self._lfp_v.size() / self.n_contacts)

    @property
    def voltages(self):
        """The LFP data (n_contacts x n_samples)."""
        if len(self._lfp_v) > 0:
            # return as a Neuron Matrix object for efficiency
            lfpmat = h.Matrix(self.n_contacts, self.n_samples)
            lfpmat.from_vector(self._lfp_v)
            return lfpmat
        else:
            return None  # simulation not yet run

    @property
    def times(self):
        """The sampling time points."""
        if self._lfp_t.size() > 0:
            # NB _lfp_t is one sample longer than _lfp_v
            return self._lfp_t.to_python()[:self.n_samples]
        else:
            return None  # simulation not yet run

    def build(self):
        """Create the Neuron objects needed to record LFPs

        """
        # ordered list of h.Sections on this rank (if running in parallel)
        secs_on_rank = _get_sections_on_this_rank()
        # np.array of number of segments for each section, ordered as above
        segment_counts = np.array(_get_segment_counts(secs_on_rank))

        # pointers assigned to _ref_i_membrane_ at each EACH internal segment
        self.imem_ptrvec = h.PtrVector(segment_counts.sum())
        # placeholder into which pointer values are read on each sim step
        self.imem_vec_len = int(self.imem_ptrvec.size())
        self.imem_vec = h.Vector(self.imem_vec_len)

        ptr_count = 0
        for sec in secs_on_rank:
            for seg in sec:  # section end points (0, 1) not included
                # set Nth pointer to the net membrane current at this segment
                self.imem_ptrvec.pset(
                    ptr_count, sec(seg.x)._ref_i_membrane_)
                ptr_count += 1
        if ptr_count != self.imem_vec_len:
            raise RuntimeError(f'Expected {self.imem_vec_len} imem pointers, '
                               f'got {ptr_count}.')

        # transfer resistances for each segment (keep in Neuron Matrix object)
        self.r_transfer = h.Matrix(self.n_contacts, self.imem_vec_len)

        for row, pos in enumerate(self.positions):
            transfer_resistance = list()
            for sec, n_segs in zip(secs_on_rank, segment_counts):
                this_xfer_r = _transfer_resistance(sec, pos,
                                                   sigma=self.sigma,
                                                   method=self.method)
                # the n_segs of this section get assigned the same value (e.g.
                # in PSA, the distance is calculated for the section mid point
                # only)
                transfer_resistance.extend([this_xfer_r] * n_segs)

            self.r_transfer.setrow(row, h.Vector(transfer_resistance))

        # record time for each array
        self._lfp_t = h.Vector().record(h._ref_t)

        # contributions of all segments on this rank to total calculated
        # potential at electrode (_PC.allreduce called in _simulate_dipole)
        self.reset()

    def reset(self):
        self._lfp_v = h.Vector()
        self.imem_vec = h.Vector(self.imem_vec_len)

    def calc_potential_callback(self):
        # keep all data in Neuron objects for efficiency

        # 'gather' the values of seg.i_membrane_ into self.imem_vec
        self.imem_ptrvec.gather(self.imem_vec)

        # r_transfer is now a Matrix. Calculate potentials by multiplying the
        # imem_vec by the matrix. This is equivalent to a row-by-row dot-
        # product: V_i = SUM_j (R_i,j x I_j)

        # electrode_potentials = self.r_transfer.mulv(self.imem_vec)

        # append all values at current time step (must be reshaped later)
        self._lfp_v.append(self.r_transfer.mulv(self.imem_vec))
