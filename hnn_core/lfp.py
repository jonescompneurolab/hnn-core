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

from neuron import h

import matplotlib.pyplot as plt


def get_all_sections(sec_type='Pyr'):
    ls = h.allsec()
    ls = [s for s in ls if sec_type in s.name()]
    return ls


class LFPElectrode:
    """LFP electrode class.

    Parameters
    ----------
    coord : tuple
        The (x, y, z) coordinates of the LFP electrode.
    sigma : float
        Extracellular conductivity in mS/cm (uniform for simplicity)
    pc : instance of h.ParallelContext()
        ParallelContext instance for running in parallel
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
    """

    def __init__(self, coord, sigma=3.0, pc=None, cvode=None, method='psa'):

        # see http://jn.physiology.org/content/104/6/3388.long shows table of
        # values with conductivity
        self.sigma = sigma
        self.coord = coord
        self.vres = None

        self.imem_ptrvec = None
        self.imem_vec = None
        self.bscallback = None
        self.fih = None
        self.cvode = cvode
        self.method = method

        if pc is None:
            self.pc = h.ParallelContext()
        else:
            self.pc = pc

    def setup(self):
        """Enables fast calculation of transmembrane current (nA) at
           each segment."""
        # h.cvode.use_fast_imem(1)
        self.bscallback = self.cvode.extra_scatter_gather(0, self.callback)
        fih = h.FInitializeHandler(1, self.LFPinit)

    def transfer_resistance(self, exyz, method):
        """Transfer resistance.

        Parameters
        ----------
        exyz : list (x, y, z)
            The x, y, z coordinates of the electrode.
        use_point : bool
            Whether to do a point source approximation
            for extracellular currents.

        Returns
        -------
        vres : instance of h.Vector
            The resistance.
        """
        import numpy as np
        from numpy.linalg import norm

        vres = h.Vector()
        lsec = get_all_sections()
        sigma = self.sigma

        exyz = np.array(exyz)  # electrode position

        for s in lsec:

            start = np.array([s.x3d(0), s.y3d(0), s.z3d(0)])
            end = np.array([s.x3d(1), s.y3d(1), s.z3d(1)])
            mid = (start + end) / 2.

            if method == 'psa':

                # distance from compartment to electrode
                dis = norm(exyz - mid)

                # setting radius limit
                if dis < s.diam / 2.0:
                    dis = s.diam / 2.0 + 0.1

                phi = 1. / dis

            elif method == 'lsa':
                # calculate length of the compartment
                a = end - start
                dis = norm(a)

                # setting radius limit
                if dis < s.diam / 2.0:
                    dis = s.diam / 2.0 + 0.1

                # if a = position vector of end with respect to start
                #    b = position vector of electrode with respect to end
                #
                # we want to compute the length of projection of "a"
                # "b".
                # H = a.cos(theta) = a.dot(b) / |a|
                a = end - start
                b = exyz - end
                H = a.dot(b) / dis

                # total longitudinal distance from start of compartment
                L = H + dis

                # if a.dot(b) < 0, projection will fall on the
                # compartment.
                if H < 0:
                    H = -H

                # distance ^ 2 of electrode to end
                r_sq = np.linalg.norm(exyz - end) ** 2

                # phi
                num = np.sqrt(H ** 2 + r_sq) - H
                denom = np.sqrt(H ** 2 + r_sq) - L
                phi = 1. / dis * np.log(num / denom)

            # x10000 for units of microV : nA/(microm*(mS/cm)) -> microV
            vres.append(10000.0 * phi / (4.0 * np.pi * sigma))
        return vres

    def LFPinit(self):
        lsec = get_all_sections()
        n_sections = len(lsec)

        self.imem_ptrvec = h.PtrVector(n_sections)
        self.imem_vec = h.Vector(n_sections)
        for i, s in enumerate(lsec):
            seg = s(0.5)
            # for seg in s # so do not need to use segments...?
            # more accurate to use segments and their neighbors
            self.imem_ptrvec.pset(i, seg._ref_i_membrane_)

        self.vres = self.transfer_resistance(self.coord, method=self.method)
        self.lfp_t = h.Vector()
        self.lfp_v = h.Vector()

    def callback(self):
        # print('In lfp callback - pc.id = ',self.pc.id(),' t=',self.pc.t(0))
        self.imem_ptrvec.gather(self.imem_vec)

        # verify sum i_membrane_ == stimulus
        # s = pc.allreduce(imem_vec.sum(), 1)
        # if rank == 0: print pc.t(0), s

        # sum up the weighted i_membrane_. Result in vx
        # rx.mulv(imem_vec, vx)

        val = 0.0
        for j in range(len(self.vres)):
            val += self.imem_vec.x[j] * self.vres.x[j]

        # append to Vector
        self.lfp_t.append(self.pc.t(0))
        self.lfp_v.append(val)


if __name__ == '__main__':
    from hnn_core.pyramidal import L5Pyr
    from hnn_core.network_builder import load_custom_mechanisms

    load_custom_mechanisms()
    cell = L5Pyr()

    h.load_file("stdgui.hoc")
    h.cvode_active(1)

    ns = h.NetStim()
    ns.number = 10
    ns.start = 100
    ns.interval = 50.0
    ns.noise = 0.  # deterministic

    nc = h.NetCon(ns, cell.synapses['apicaltuft_ampa'])
    nc.weight[0] = 0.001

    h.tstop = 2000.0

    for method in ['lsa', 'psa']:
        elec = LFPElectrode([0, 100.0, 100.0], pc=h.ParallelContext(),
                            method='psa')
        elec.setup()
        elec.LFPinit()
        h.run()
        elec.pc.allreduce(elec.lfp_v, 1)

        plt.plot(elec.lfp_t.to_python(), elec.lfp_v.to_python(),
                 label=method)
    plt.legend()
    plt.show()
