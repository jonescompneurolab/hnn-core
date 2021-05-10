"""
LFPsim - Simulation scripts to compute Local Field Potentials (LFP) from cable
compartmental models of neurons and networks implemented in NEURON simulation
environment.

LFPsim works reliably on biophysically detailed multi-compartmental neurons
with ion channels in some or all compartments.

Last updated 12-March-2016
Developed by :
Harilal Parasuram & Shyam Diwakar
Computational Neuroscience & Neurophysiology Lab, School of Biotechnology,
Amrita University, India.

Email: harilalp@am.amrita.edu; shyam@amrita.edu
www.amrita.edu/compneuro

translated to Python and modified to use use_fast_imem by Sam Neymotin
based on mhines code

References
----------
1 . Parasuram H, Nair B, D'Angelo E, Hines M, Naldi G, Diwakar S (2016)
Computational Modeling of Single Neuron Extracellular Electric Potentials
and Network Local Field Potentials using LFPsim. Front Comput Neurosci 10:65
[PubMed].
"""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Sam Neymotin <samnemo@gmail.com>

from neuron import h

import numpy as np
from numpy.linalg import norm


def _get_all_sections(sec_type='Pyr'):
    ls = h.allsec()
    ls = [s for s in ls if sec_type in s.name()]
    return ls


def _get_segment_counts(all_sections):
    seg_counts = np.zeros((len(all_sections), ), dtype=np.int)
    for ii, sec in enumerate(all_sections):
        seg_counts[ii] = np.int(sec.nseg)
    return seg_counts


def _transfer_resistance(section, ele_pos, sigma, method):
    """Transfer resistance between section and electrode position.

    To arrive at the extracellular potential, the value returned by this
    function is multiplied by the net transmembrane current flowing through all
    segments of the section. Hence the term "resistance" (voltage equals
    current times resistance).

    Parameters
    ----------
    section : h.Section() The NEURON section. ele_pos : list (x, y, z) The x,
        y, z coordinates of the electrode (in um) sigma : float Extracellular
        conductivity (in S/m) method : str Approximation to use. 'psa' assigns
        all transmembrane currents to the center point (0.5). 'lsa' treats the
        section as a line source, but a single multiplier is calculated for
        each section. Returns
    -------
    vres : list The resistance at each section.
    """
    ele_pos = np.array(ele_pos)  # electrode position to Numpy

    start = np.array([section.x3d(0), section.y3d(0), section.z3d(0)])
    end = np.array([section.x3d(1), section.y3d(1), section.z3d(1)])
    mid = (start + end) / 2.

    if method == 'psa':

        # distance from compartment to electrode
        dis = norm(ele_pos - mid)

        # setting radius limit
        if dis < section.diam / 2.0:
            dis = section.diam / 2.0 + 0.1

        phi = 1. / dis

    # XXX the 'lsa' method implementation is unverified, proceed with caution!
    elif method == 'lsa':
        # calculate length of the compartment
        a = end - start
        dis = norm(a)

        # setting radius limit
        if dis < section.diam / 2.0:
            dis = section.diam / 2.0 + 0.1

        # if a = position vector of end with respect to start
        #    b = position vector of electrode with respect to end
        #
        # we want to compute the length of projection of "a"
        # "b".
        # H = a.cos(theta) = a.dot(b) / |a|
        a = end - start
        b = ele_pos - end
        H = a.dot(b) / dis

        # total longitudinal distance from start of compartment
        L = H + dis

        # if a.dot(b) < 0, projection will fall on the
        # compartment.
        if H < 0:
            H = -H

        # distance ^ 2 of electrode to end
        r_sq = np.linalg.norm(ele_pos - end) ** 2

        # phi
        num = np.sqrt(H ** 2 + r_sq) - H
        denom = np.sqrt(H ** 2 + r_sq) - L
        phi = 1. / dis * np.log(num / denom)

    # [dis]: um; [sigma]: S / m
    # [phi / sigma] = [1/dis] / [sigma] = 1 / [dis] x [sigma]
    # [dis] x [sigma] = um x (S / m) = 1e-6 S
    # transmembrane current returned by _ref_i_membrane_ is in [nA]
    # ==> 1e-9 A x (1 / 1e-6 S) = 1e-3 V = mV
    # ===> multiply by 1e3 to get uV
    return 1000.0 * phi / (4.0 * np.pi * sigma)


class _LFPElectrode:
    """LFP electrode class.

    Parameters
    ----------
    coord : tuple
        The (x, y, z) coordinates (in um) of the LFP electrode.
    sigma : float
        Extracellular conductivity, in S/m, of the assumed infinite,
        homogeneous volume conductor that the cell and electrode are in.
    pc : instance of h.ParallelContext()
        ParallelContext instance for running in parallel
    cvode : instanse of h.CVode
        Multi order variable time step integration method.
    method : str
        'psa' (default), i.e., point source approximation or line source
        approximation, i.e., 'lsa'

    Attributes
    ----------
    lfp_t : instance of h.Vector
        The LFP time instances.
    lfp_v : instance of h.Vector
        The LFP voltage.
    imem_vec : instance of h.Vector
        The transmembrane ionic current.

    Notes
    -----
    See Table 5 in http://jn.physiology.org/content/104/6/3388.long for
    measured values of sigma in rat cortex (note units there are mS / cm)
    """

    def __init__(self, coord, sigma=0.3, pc=None, cvode=None, method='psa'):

        secs_in_network = _get_all_sections()  # ordered list of h.Sections
        # np.array of number of segments for each section, ordered as above
        self.segment_counts = _get_segment_counts(secs_in_network)

        # pointers assigned to _ref_i_membrane_ at each EACH segment below
        self.imem_ptrvec = h.PtrVector(self.segment_counts.sum())
        # placeholder into which pointer values are read on each sim step
        self.imem_vec = h.Vector(int(self.imem_ptrvec.size()))
        # transfer resistances, same length as membrane current pointers
        self.r_transfer = np.empty((int(self.imem_ptrvec.size()), ),
                                   dtype=np.int)
        self.pc = pc
        if self.pc is None:
            self.pc = h.ParallelContext()

        # Enables fast calculation of transmembrane current (nA) at each
        # segment
        self.bscallback = cvode.extra_scatter_gather(0, self.callback)

        xfer_resistance_list = list()
        count = 0
        for sec, n_segs in zip(secs_in_network, self.segment_counts):
            this_xfer_r = _transfer_resistance(sec, coord, sigma=sigma,
                                               method=method)
            # the n_segs of this section get assigned the same value (e.g. in
            # PSA, the distance is calculated for the section mid point only)
            xfer_resistance_list.extend([this_xfer_r] * n_segs)
            for seg in sec:  # section end points (0, 1) not included
                # set Nth pointer to the net membrane current at this segment
                self.imem_ptrvec.pset(count, sec(seg.x)._ref_i_membrane_)
                count += 1
        # convert to numpy array for speedy calculation in callback
        self.r_transfer = np.array(xfer_resistance_list)
        assert count == int(self.imem_ptrvec.size())  # smoke test

        self.lfp_t = h.Vector()
        self.lfp_v = h.Vector()

    def callback(self):
        self.imem_ptrvec.gather(self.imem_vec)

        # verify sum i_membrane_ == stimulus
        # s = pc.allreduce(imem_vec.sum(), 1)
        # if rank == 0: print pc.t(0), s

        # sum up the weighted i_membrane_. Result in vx
        # rx.mulv(imem_vec, vx)

        potential = np.dot(np.array(self.imem_vec.to_python()),
                           self.r_transfer)

        # append to Vector
        self.lfp_t.append(self.pc.t(0))
        self.lfp_v.append(potential)
