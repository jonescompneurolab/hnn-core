"""
Handler classes to calculate extracellular electric potentials, such as the
Local Field Potential (LFP), at ideal (point-like) electrodes based on net
transmembrane currents of all neurons in the network.

The code is inspired by [1], but important modifications were made to comply
with the original derivation [2] of the 'line source approximation method'.

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

import numpy as np
from copy import deepcopy
from numpy.linalg import norm
from neuron import h

from .externals.mne import _validate_type, _check_option


def calculate_csd2d(lfp_data, delta=1):
    """Current source density (CSD) estimation

    Parameters
    ----------
    lfp_data : array, shape (n_channels, n_times)
        LFP data.
    delta : int
        Spacing between channels (um), scales the CSD.

    Returns
    -------
    csd2d : array, shape (n_channels, n_times)
        The 2nd derivative current source density estimate (csd2d)

    Notes
    -----
    The three-point finite-difference approximation of the
    second spatial derivative for computing 1-dimensional CSD
    with border electrode interpolation
    csd[electrode] = -(LFP[electrode - 1] - 2*LFP[electrode] +
                       LFP[electrode + 1]) / spacing ** 2
    """
    csd2d = -np.diff(lfp_data, n=2, axis=0) / delta ** 2
    bottom_border = csd2d[-1, :] * 2 - csd2d[-2, :]
    top_border = csd2d[0, :] * 2 - csd2d[1, :]
    csd2d = np.concatenate((top_border[None, ...], csd2d,
                            bottom_border[None, ...]), axis=0)
    return csd2d


def _get_laminar_z_coords(electrode_positions):
    """Get equispaced, colinear electrode coordinates along z-axis.

    Parameters
    ----------
    electrode_positions : list of tuple
        The (x, y, z) coordinates (in um) of the extracellular electrodes.

    Returns
    -------
    z_coordinates : array, shape (n_contacts,)
        Z-coordinates of the electrode contacts.
    z_delta : float
        Magnitude of change in the z-direction.
    """
    n_contacts = np.array(electrode_positions).shape[0]
    if n_contacts < 2:
        raise ValueError(
            'Electrode array positions must contain more than 1 contact to be '
            'compatible with laminar profiling in a neocortical column. Got '
            f'{n_contacts} electrode contact positions.')
    displacements = np.diff(electrode_positions, axis=0)
    z_delta = np.abs(displacements[0, 2])
    magnitudes = np.linalg.norm(displacements, axis=1)
    cross_prods = np.cross(displacements[:-1], displacements[1:])
    if not (np.allclose(magnitudes, magnitudes[0]) and  # equally spaced
            z_delta > 0 and  # changes in z-direction
            np.allclose(cross_prods, 0)):  # colinear
        raise ValueError(
            'Electrode contacts are incompatible with laminar profiling '
            'in a neocortical column. Make sure the '
            'electrode postions are equispaced, colinear, and projecting '
            'along the z-axis.')
    else:
        return np.array(electrode_positions)[:, 2], z_delta


def _transfer_resistance(section, electrode_pos, conductivity, method,
                         min_distance=0.5):
    """Transfer resistance between section and electrode position.

    To arrive at the extracellular potential, the value returned by this
    function is multiplied by the net transmembrane current flowing through all
    segments of the section. Hence the term "resistance" (voltage equals
    current times resistance).

    Parameters
    ----------
    section : h.Section()
        The NEURON section.
    electrode_pos : list (x, y, z)
        The x, y, z coordinates of the electrode (in um)
    conductivity : float
        Extracellular conductivity (in S/m)
    method : str
        Approximation to use. ``'psa'`` (point source approximation) treats
        each segment junction as a point extracellular current source.
        ``'lsa'`` (line source approximation) treats each segment as a line
        source of current, which extends from the previous to the next segment
        center point: |---x---|, where x is the current segment flanked by |.
    min_distance : float (default: 0.5)
        To avoid numerical errors in the 1/R calculation, we'll by default
        limit the distance to 0.5 um, corresponding to 1 um diameter dendrites.
        NB: LFPy uses section.diam / 2.0, i.e., whatever the closest section
        radius happens to be. This may not make sense for HNN model neurons, in
        which dendrite diameters have been adjusted to represent the entire
        tree (Bush & Sejnowski, 1993).

    Returns
    -------
    vres : list
        The transfer resistance at each segment of the section.
    """
    electrode_pos = np.array(electrode_pos)  # electrode position to Numpy

    sec_start = np.array([section.x3d(0), section.y3d(0), section.z3d(0)])
    sec_end = np.array([section.x3d(1), section.y3d(1), section.z3d(1)])
    sec_vec = sec_end - sec_start

    # NB segment lengths aren't equal! First/last segment center point is
    # closer to respective end point than to next/previous segment!
    # for nseg == 5, the segment centers are: [0.1, 0.3, 0.5, 0.7, 0.9]
    seg_ctr = np.zeros((section.nseg, 3))
    line_lens = np.zeros((section.nseg + 2))
    for ii, seg in enumerate(section):
        seg_ctr[ii, :] = sec_start + seg.x * sec_vec
        line_lens[ii + 1] = seg.x * section.L
    line_lens[-1] = section.L
    line_lens = np.diff(line_lens)
    first_len = line_lens[0]
    line_lens = np.array([first_len] + list(line_lens[2:]))

    if method == 'psa':

        # distance from segment midpoints to electrode
        dis = norm(np.tile(electrode_pos, (section.nseg, 1)) - seg_ctr,
                   axis=1)

        # To avoid very large values when electrode is placed close to a
        # segment junction, enforce minimal radial distance
        dis = np.maximum(dis, min_distance)

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
        phi = np.zeros(len(seg_ctr))
        for idx, (ctr, line_len) in enumerate(zip(seg_ctr, line_lens)):
            start = ctr - line_len * sec_vec / norm(sec_vec)
            end = ctr + line_len * sec_vec / norm(sec_vec)
            a = end - start
            norm_a = norm(a)
            b = electrode_pos - end
            # projection: H = a.cos(theta) = a.dot(b) / |a|
            H = np.dot(b, a) / norm_a  # NB can be negative
            L = H + norm_a
            R2 = np.dot(b, b) - H ** 2  # NB squares

            # To avoid very large values when electrode is placed (anywhere) on
            # the section axis, enforce minimal perpendicular distance
            R2 = np.maximum(R2, min_distance ** 2)

            if L < 0 and H < 0:  # electrode is "behind" line segment
                num = np.sqrt(H ** 2 + R2) - H  # == norm(b) - H
                denom = np.sqrt(L ** 2 + R2) - L
            elif L > 0 and H < 0:  # electrode is "on top of" line segment
                num = (np.sqrt(H ** 2 + R2) - H) * (L + np.sqrt(L ** 2 + R2))
                denom = R2
            else:  # electrode is "ahead of" line segment
                num = np.sqrt(L ** 2 + R2) + L
                denom = np.sqrt(H ** 2 + R2) + H  # == norm(b) + H

            phi[idx] = np.log(num / denom) / norm_a

    # [dis]: um; [conductivity]: S / m
    # [phi / conductivity] = [1/dis] / [conductivity] = 1 / [dis] x [conduct'y]
    # [dis] x [conductivity] = um x (S / m) = 1e-6 S
    # transmembrane current returned by _ref_i_membrane_ is in [nA]
    # ==> 1e-9 A x (1 / 1e-6 S) = 1e-3 V = mV
    # ===> multiply by 1e3 to get uV
    return 1000.0 * phi / (4.0 * np.pi * conductivity)


class ExtracellularArray:
    """Class for recording extracellular potential fields with electrode array

    Note that to add an electrode array to a simulation, you should use the
    :meth:`hnn_core.Network.add_electrode_array`-method. After simulation,
    the network will contain a dictionary of `ExtracellularArray`-objects
    in ``net.rec_arrays`` (each array must be added with a unique name). An
    `ExtracellularArray` contains the voltages at each electrode contact,
    along with the time points at which the voltages were sampled.

    Parameters
    ----------
    positions : tuple | list of tuple
        The (x, y, z) coordinates (in um) of the extracellular electrodes.
    conductivity : float
        Extracellular conductivity, in S/m, of the assumed infinite,
        homogeneous volume conductor that the cell and electrode are in.
    method : str
        Approximation to use. ``'psa'`` (point source approximation) treats
        each segment junction as a point extracellular current source.
        ``'lsa'`` (line source approximation) treats each segment as a line
        source of current, which extends from the previous to the next segment
        center point: /---x---/, where x is the current segment flanked by /.
    min_distance : float (default: 0.5; unit: um)
        To avoid numerical errors in calculating potentials, apply a minimum
        distance limit between the electrode contacts and the active neuronal
        membrane elements that act as sources of current. The default value of
        0.5 um corresponds to 1 um diameter dendrites.
    times : array-like, shape (n_times,) | None
        Optionally, provide precomputed voltage sampling times for electrodes
        at `positions`.
    voltages : array-like, shape (n_trials, n_electrodes, n_times) | None
        Optionally, provide precomputed voltages for electrodes at
        ``positions``.

    Attributes
    ----------
    times : array, shape (n_times,)
        The time points the extracellular voltages are sampled at (ms)
    voltages : array, shape (n_trials, n_electrodes, n_times)
        The measured extracellular voltages
    sfreq : float
        Sampling rate of the extracellular data (Hz).

    Notes
    -----
    The length of an ``ExtracellularArray`` is equal to the number of trials of
    data it contains. Slicing an ``ExtracellularArray`` returns a `copy` of the
    corresponding trials: ``array[:5]`` returns a new array of length 5, etc.

    See Table 5 in https://doi.org/10.1152/jn.00122.2010 for
    measured values of conductivity in rat cortex (note units there are mS/cm)
    """

    def __init__(self, positions, *, conductivity=0.3, method='psa',
                 min_distance=0.5, times=None, voltages=None):

        _validate_type(positions, (tuple, list), 'positions')
        if np.array(positions).shape == (3,):  # a single coordinate given
            positions = [positions]
        for pos in positions:
            _validate_type(pos, (tuple, list), 'positions')
            if len(pos) != 3:
                raise ValueError('positions should be provided as xyz '
                                 f'coordinate triplets, got: {positions}')

        _validate_type(conductivity, float, 'conductivity')
        if not conductivity > 0.:
            raise ValueError('conductivity must be a positive number')
        _validate_type(min_distance, float, 'min_distance')
        if not min_distance > 0.:
            raise ValueError('min_distance must be a positive number')
        if method is not None:  # method allowed to be None for testing
            _validate_type(method, str, 'method')
            _check_option('method', method, ['psa', 'lsa'])

        if times is None:
            times = np.array([])
        if voltages is None:
            voltages = np.array([])

        times = np.array(times, dtype='float')
        voltages = np.array(voltages, dtype='float')

        if voltages.size != 0:  # voltages is not None
            n_trials, n_electrodes, n_times = voltages.shape
            if len(positions) != n_electrodes:
                raise ValueError(f'number of voltage traces must match number'
                                 f' of channels, got {n_electrodes} and '
                                 f'{len(positions)}')
            if len(times) != n_times:
                raise ValueError('length of times and voltages must match,'
                                 f' got {len(times)} and {n_times} ')

        self.positions = positions
        self.n_contacts = len(self.positions)
        self.conductivity = conductivity
        self.method = method
        self.min_distance = min_distance

        self._times = times
        self._data = voltages

    def __getitem__(self, trial_no):
        try:
            if isinstance(trial_no, int):
                return_data = [self._data[trial_no]]
            elif isinstance(trial_no, slice):
                return_data = self._data[trial_no]
            elif isinstance(trial_no, (list, tuple)):
                return_data = [self._data[trial] for trial in trial_no]
            else:
                raise TypeError(f'trial index must be int, slice or list-like,'
                                f' got: {trial_no} which is {type(trial_no)}')
        except IndexError:
            raise IndexError(f'the data contain {len(self)} trials, the '
                             f'indices provided are out of range: {trial_no}')
        return ExtracellularArray(self.positions,
                                  conductivity=self.conductivity,
                                  method=self.method,
                                  times=self.times,
                                  voltages=return_data)

    def __repr__(self):
        class_name = self.__class__.__name__
        msg = (f'{self.n_contacts} electrodes, '
               f'conductivity={self.conductivity}, method={self.method}')
        if len(self._data) > 0:
            msg += f' | {len(self._data)} trials, {len(self.times)} times'
        else:
            msg += ' (no data recorded yet)'
        return f'<{class_name} | {msg}>'

    def __len__(self):
        return len(self._data)  # length == number of trials

    def copy(self):
        """Return a copy of the ExtracellularArray instance

        Returns
        -------
        array_copy : instance of ExtracellularArray
            A copy of the array instance.
        """
        return deepcopy(self)

    @property
    def times(self):
        return np.array(self._times)

    @property
    def voltages(self):
        return np.array(self._data)

    @property
    def sfreq(self):
        """Return the sampling rate of the extracellular data."""
        if len(self.times) == 0:
            return None
        elif len(self.times) == 1:
            raise RuntimeError('Sampling rate is not defined for one sample')

        dT = np.diff(self.times)
        Tsamp = np.median(dT)
        if np.abs(dT.max() - Tsamp) > 1e-3 or np.abs(dT.min() - Tsamp) > 1e-3:
            raise RuntimeError(
                'Extracellular sampling times vary by more than 1 us. Check '
                'times-attribute for errors.')

        return 1000. / Tsamp  # times are in in ms

    def _reset(self):
        self._data = list()
        self._times = list()

    def smooth(self, window_len):
        """Smooth extracellular waveforms using Hamming-windowed convolution

        Note that this method operates in-place, i.e., it will alter the data.
        If you prefer a filtered copy, consider using the
        :meth:`~hnn_core.extracellular.ExtracellularArray.copy`-method.

        Parameters
        ----------
        window_len : float
            The length (in ms) of a `~numpy.hamming` window to convolve the
            data with.

        Returns
        -------
        extracellular_copy : instance of ExtraCellularArray
            The modified ExtraCellularArray instance.
        """
        from .utils import smooth_waveform

        for n_trial in range(len(self)):
            for n_contact in range(self.n_contacts):
                self._data[n_trial][n_contact] = smooth_waveform(
                    self._data[n_trial][n_contact], window_len,
                    self.sfreq)  # XXX smooth_waveform returns ndarray

        return self

    def plot_lfp(self, *, trial_no=None, contact_no=None, tmin=None, tmax=None,
                 ax=None, decim=None, color='cividis', voltage_offset=50,
                 voltage_scalebar=200, show=True):
        """Plot laminar local field potential time series.

        One plot is created for each trial. Multiple trials can be overlaid
        with or without (default) and offset.

        Parameters
        ----------
        trial_no : int | list of int | slice
            Trial number(s) to plot
        contact_no : int | list of int | slice
            Electrode contact number(s) to plot
        tmin : float | None
            Start time of plot in milliseconds. If None, plot entire
            simulation.
        tmax : float | None
            End time of plot in milliseconds. If None, plot entire simulation.
        ax : instance of matplotlib figure | None
            The matplotlib axis
        decim : int | list of int | None (default)
            Optional (integer) factor by which to decimate the raw dipole
            traces. The SciPy function :func:`~scipy.signal.decimate` is used,
            which recommends values <13. To achieve higher decimation factors,
            a list of ints can be provided. These are applied successively.
        color : string | array of floats | matplotlib.colors.ListedColormap
            The color to use for plotting (optional). The usual Matplotlib
            standard color strings may be used (e.g., 'b' for blue). A color
            can also be defined as an RGBA-quadruplet, or an array of
            RGBA-values (one for each electrode contact trace to plot). An
            instance of :class:`~matplotlib.colors.ListedColormap` may also be
            provided.
        voltage_offset : float | None (optional)
            Amount to offset traces by on the voltage-axis. Useful for plotting
            laminar arrays.
        voltage_scalebar : float | None (optional)
            Height, in units of uV, of a scale bar to plot in the top-left
            corner of the plot.
        show : bool
            If True, show the figure

        Returns
        -------
        fig : instance of plt.fig
            The matplotlib figure handle into which time series were plotted.
        """
        from .viz import plot_laminar_lfp

        if trial_no is None:
            plot_data = self.voltages
        elif isinstance(trial_no, (list, tuple, int, slice)):
            plot_data = self.voltages[trial_no, ]
        else:
            raise ValueError(f'unknown trial number type, got {trial_no}')

        if isinstance(contact_no, (list, tuple, int, slice)):
            plot_data = plot_data[:, contact_no, ]
        elif contact_no is not None:
            raise ValueError(f'unknown contact number type, got {contact_no}')

        contact_labels, _ = _get_laminar_z_coords(self.positions)

        for trial_data in plot_data:
            fig = plot_laminar_lfp(
                self.times, trial_data, tmin=tmin, tmax=tmax, ax=ax,
                decim=decim, color=color,
                voltage_offset=voltage_offset,
                voltage_scalebar=voltage_scalebar,
                contact_labels=contact_labels,
                show=show)
        return fig

    def plot_csd(self, colorbar=True, ax=None, show=True):
        """Plot laminar current source density (CSD) estimation

        Parameters
        ----------
        colorbar : bool
            If the colorbar is presented.
        ax : instance of matplotlib figure | None
            The matplotlib axis.
        show : bool
            If True, show the plot.

        Returns
        -------
        fig : instance of matplotlib Figure
            The matplotlib figure handle.
        """
        from .viz import plot_laminar_csd
        lfp = self.voltages[0]
        contact_labels, delta = _get_laminar_z_coords(self.positions)

        csd_data = calculate_csd2d(lfp_data=lfp,
                                   delta=delta)

        fig = plot_laminar_csd(self.times, csd_data,
                               contact_labels=contact_labels, ax=ax,
                               colorbar=colorbar, show=show)

        return fig


class _ExtracellularArrayBuilder(object):
    """The _ExtracellularArrayBuilder class

    Parameters
    ----------
    array : ExtracellularArray object
        The instance of :class:`hnn_core.extracellular.ExtracellularArray` to
        build in NEURON-Python
    """
    def __init__(self, array):
        self.array = array
        self.n_contacts = array.n_contacts
        self._nrn_imem_ptrvec = None
        self._nrn_imem_vec = None
        self._nrn_r_transfer = None
        self._nrn_times = None
        self._nrn_voltages = None
        self._recording_callback = None

    def _build(self, cvode=None, include_celltypes='all'):
        """Assemble NEURON objects for calculating extracellular potentials.

        The handler is set up to maintain a vector of membrane currents at at
        every inner segment of every section of every cell on each CVODE
        integration step. In addition, it records a time vector of sample
        times.

        Parameters
        ----------
        cvode : instance of h.CVode
            Multi order variable time step integration method.
        include_celltypes : str
            String to match against the cell type of each section. Defaults to
            ``'all'``: calculate extracellular potential generated by all
            cells. To restrict this to include only pyramidal cells, use
            ``'Pyr'``. For basket cells, use ``'Basket'``. NB This argument is
            currently not exposed in the API.
        """
        secs_on_rank = h.allsec()  # get all h.Sections known to this MPI rank
        _validate_type(include_celltypes, str)
        _check_option('include_celltypes', include_celltypes, ['all', 'Pyr',
                                                               'Basket'])
        if include_celltypes.lower() != 'all':
            secs_on_rank = [s for s in secs_on_rank if
                            include_celltypes in s.name()]

        segment_counts = [sec.nseg for sec in secs_on_rank]
        n_total_segments = np.sum(segment_counts)

        # pointers assigned to _ref_i_membrane_ at each EACH internal segment
        self._nrn_imem_ptrvec = h.PtrVector(n_total_segments)
        # placeholder into which pointer values are read on each sim time step
        self._nrn_imem_vec = h.Vector(n_total_segments)

        ptr_idx = 0
        for sec in secs_on_rank:
            for seg in sec:  # section end points (0, 1) not included
                # set Nth pointer to the net membrane current at this segment
                self._nrn_imem_ptrvec.pset(
                    ptr_idx, sec(seg.x)._ref_i_membrane_)
                ptr_idx += 1
        if ptr_idx != n_total_segments:
            raise RuntimeError(f'Expected {n_total_segments} imem pointers, '
                               f'got {ptr_idx}.')

        # transfer resistances for each segment (keep in Neuron Matrix object)
        self._nrn_r_transfer = h.Matrix(self.n_contacts, n_total_segments)

        for row, pos in enumerate(self.array.positions):
            if self.array.method is not None:
                transfer_resistance = list()
                for sec in secs_on_rank:
                    this_xfer_r = _transfer_resistance(
                        sec, pos, conductivity=self.array.conductivity,
                        method=self.array.method,
                        min_distance=self.array.min_distance)
                    transfer_resistance.extend(this_xfer_r)

                self._nrn_r_transfer.setrow(row, h.Vector(transfer_resistance))
            else:
                # for testing, make a matrix of ones
                self._nrn_r_transfer.setrow(row,
                                            h.Vector(n_total_segments, 1.))

        # record time for each array
        self._nrn_times = h.Vector().record(h._ref_t)

        # contributions of all segments on this rank to total calculated
        # potential at electrode (_PC.allreduce called in _simulate_dipole)
        # NB voltages of all contacts are initialised to 0 mV, i.e., the
        # potential at time 0.0 ms is defined to be zero.
        self._nrn_voltages = h.Vector(self.n_contacts, 0.)

        # NB we must make a copy of the function reference, and keep it for
        # later decoupling using extra_scatter_gather_remove
        # (instead of a new function reference)
        self._recording_callback = self._gather_nrn_voltages
        # Nb extra_scatter_gather is called _after_ the solver takes a step,
        # so the initial state is not recorded (initialised to zero above)
        cvode.extra_scatter_gather(0, self._recording_callback)

    def _gather_nrn_voltages(self):
        """Callback function for _CVODE.extra_scatter_gather

        Enables fast calculation of transmembrane current (nA) at each
        segment. Note that this will run on each rank, so it is safe to use
        the extra_scatter_gather-method, which docs say doesn't support
        'multiple threads'.
        """
        # keep all data in Neuron objects for efficiency

        # 'gather' the values of seg.i_membrane_ into self.imem_vec
        self._nrn_imem_ptrvec.gather(self._nrn_imem_vec)

        # Calculate potentials by multiplying the _nrn_imem_vec by the matrix
        # _nrn_r_transfer. This is equivalent to a row-by-row dot-product:
        # V_i(t) = SUM_j ( R_i,j x I_j (t) )
        self._nrn_voltages.append(
            self._nrn_r_transfer.mulv(self._nrn_imem_vec))
        # NB all values appended to the h.Vector _nrn_voltages at current time
        # step. The vector will have size (n_contacts x n_samples, 1), which
        # will be reshaped later to (n_contacts, n_samples).

    @property
    def _nrn_n_samples(self):
        """Return the length (in samples) of the extracellular data."""
        if self._nrn_voltages.size() % self.n_contacts != 0:
            raise RuntimeError(f'Something went wrong: have {self.n_contacts}'
                               f', but {self._nrn_voltages.size()} samples')
        return int(self._nrn_voltages.size() / self.n_contacts)

    def _get_nrn_voltages(self):
        """The extracellular data (n_contacts x n_samples)."""
        if len(self._nrn_voltages) > 0:
            assert (self._nrn_voltages.size() ==
                    self.n_contacts * self._nrn_n_samples)

            # first reshape to a Neuron Matrix object
            extmat = h.Matrix(self.n_contacts, self._nrn_n_samples)
            extmat.from_vector(self._nrn_voltages)

            # then unpack into 2D python list and return
            return [extmat.getrow(ii).to_python() for
                    ii in range(extmat.nrow())]
        else:
            raise RuntimeError('Simulation not yet run!')

    def _get_nrn_times(self):
        """The sampling time points."""
        if self._nrn_times.size() > 0:
            return self._nrn_times.to_python()
        else:
            raise RuntimeError('Simulation not yet run!')
