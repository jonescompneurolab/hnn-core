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


class _LFPElectrode:
    """Local field potential (LFP) electrode class.

    The handler is set up to maintain a vector of membrane currents at at every
    inner segment of every section of every cell on each CVODE integration
    step. In addition, it records a time vector of sample times. This class
    must be instantiated and attached to the network during the building
    process. It is used in conjunction with the calculation of extracellular
    potentials.

    Parameters
    ----------
    coord : tuple
        The (x, y, z) coordinates (in um) of the LFP electrode.
    sigma : float
        Extracellular conductivity, in S/m, of the assumed infinite,
        homogeneous volume conductor that the cell and electrode are in.
    method : str
        Approximation to use. 'psa' (default) assigns all transmembrane
        currents to the center point (0.5) (point source approximation).
        'lsa' treats the section as a line source, and a single multiplier is
        calculated for each section (line source approximation).
    cvode : instance of h.CVode
        Multi order variable time step integration method.

    Attributes
    ----------
    lfp_v : instance of h.Vector
        The LFP voltage (in uV).

    Notes
    -----
    See Table 5 in http://jn.physiology.org/content/104/6/3388.long for
    measured values of sigma in rat cortex (note units there are mS/cm)
    """

    def __init__(self, coord, sigma=0.3, method='psa', cvode=None):

        # ordered list of h.Sections on this rank (if running in parallel)
        secs_on_rank = _get_sections_on_this_rank()
        # np.array of number of segments for each section, ordered as above
        segment_counts = np.array(_get_segment_counts(secs_on_rank))

        # pointers assigned to _ref_i_membrane_ at each EACH internal segment
        self.imem_ptrvec = h.PtrVector(segment_counts.sum())
        # placeholder into which pointer values are read on each sim step
        imem_vec_len = int(self.imem_ptrvec.size())
        self.imem_vec = h.Vector(imem_vec_len)

        transfer_resistance = list()
        ptr_count = 0
        for sec, n_segs in zip(secs_on_rank, segment_counts):
            this_xfer_r = _transfer_resistance(sec, coord, sigma=sigma,
                                               method=method)
            # the n_segs of this section get assigned the same value (e.g. in
            # PSA, the distance is calculated for the section mid point only)
            transfer_resistance.extend([this_xfer_r] * n_segs)
            for seg in sec:  # section end points (0, 1) not included
                # set Nth pointer to the net membrane current at this segment
                self.imem_ptrvec.pset(ptr_count, sec(seg.x)._ref_i_membrane_)
                ptr_count += 1

        if ptr_count != imem_vec_len:
            raise RuntimeError(f'Expected {imem_vec_len} imem pointers, '
                               'got {count}.')

        # transfer resistances for each segment (keep in NEURON object)
        self.r_transfer = h.Vector(transfer_resistance)

        # contributions of all segments on this rank to total calculated
        # potential at electrode (_PC.allreduce called in _simulate_dipole)
        self.lfp_v = h.Vector()

        # Attach a callback for calculating the potentials at each time step.
        # Enables fast calculation of transmembrane current (nA) at each
        # segment. Note that this will run on each rank, so it is safe to use
        # the extra_scatter_gather-method, which docs say doesn't support
        # "multiple threads".
        cvode.extra_scatter_gather(0, self.calc_potential_callback)

    def reset(self):
        self.lfp_v = h.Vector()
        self.imem_vec = h.Vector(int(self.imem_ptrvec.size()))

    def calc_potential_callback(self):

        # 'gather' the values of seg.i_membrane_ into self.imem_vec
        self.imem_ptrvec.gather(self.imem_vec)

        # multiply elementwise, then sum (dot-product): V = SUM (R_i x I_i)
        potential = self.r_transfer.dot(self.imem_vec)

        # append to Vector
        self.lfp_v.append(potential)
