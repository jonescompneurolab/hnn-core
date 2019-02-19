"""Class to handle the dipoles."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import numpy as np
from numpy import convolve, hamming

from neuron import h

from .paramrw import find_param
from .network import NetworkOnNode


def _hammfilt(x, winsz):
    """Convolve with a hamming window."""
    win = hamming(winsz)
    win /= sum(win)
    return convolve(x, win, 'same')


def simulate_dipole(params):
    """Simulate a dipole given the experiment parameters.

    Parameters
    ----------
    params : dict
        The experiment parameters

    Returns
    -------
    dpl: instance of Dipole
        The dipole object
    """
    pc = h.ParallelContext(1)

    # global variables, should be node-independent
    h("dp_total_L2 = 0.")
    h("dp_total_L5 = 0.")

    # Set tstop before instantiating any classes
    h.tstop = params['tstop']
    h.dt = params['dt']  # simulation duration and time-step
    h.celsius = params['celsius']  # 37.0 - set temperature
    net = NetworkOnNode(params)  # create node-specific network

    # We define the arrays (Vector in numpy) for recording the signals
    t_vec = h.Vector()
    t_vec.record(h._ref_t)  # time recording
    dp_rec_L2 = h.Vector()
    dp_rec_L2.record(h._ref_dp_total_L2)  # L2 dipole recording
    dp_rec_L5 = h.Vector()
    dp_rec_L5.record(h._ref_dp_total_L5)  # L5 dipole recording

    net.movecellstopos()  # position cells in 2D grid

    # sets the default max solver step in ms (purposefully large)
    pc.set_maxstep(10)

    # initialize cells to -65 mV, after all the NetCon
    # delays have been specified
    h.finitialize()

    def prsimtime():
        print('Simulation time: {0} ms...'.format(round(h.t, 2)))

    printdt = 10
    for tt in range(0, int(h.tstop), printdt):
        h.cvode.event(tt, prsimtime)  # print time callbacks

    h.fcurrent()
    # set state variables if they have been changed since h.finitialize
    h.frecord_init()
    # actual simulation - run the solver
    pc.psolve(h.tstop)

    # these calls aggregate data across procs/nodes
    pc.allreduce(dp_rec_L2, 1)
    # combine dp_rec on every node, 1=add contributions together
    pc.allreduce(dp_rec_L5, 1)
    # aggregate the currents independently on each proc
    net.aggregate_currents()
    # combine net.current{} variables on each proc
    pc.allreduce(net.current['L5Pyr_soma'], 1)
    pc.allreduce(net.current['L2Pyr_soma'], 1)

    dpl_data = np.c_[t_vec.as_numpy(),
                     dp_rec_L2.as_numpy() + dp_rec_L5.as_numpy(),
                     dp_rec_L2.as_numpy(), dp_rec_L5.as_numpy()]

    pc.barrier()  # get all nodes to this place before continuing
    pc.gid_clear()

    pc.runworker()
    pc.done()

    np.savetxt(doutf['file_dpl'], dpl_data, fmt='%5.4f')

    dpl = Dipole(doutf['file_dpl'])
    dpl.baseline_renormalize(doutf['file_param'])
    dpl.convert_fAm_to_nAm()
    dpl.scale(paramrw.find_param(doutf['file_param'], 'dipole_scalefctr'))
    dpl.smooth(paramrw.find_param(
        doutf['file_param'], 'dipole_smooth_win') / h.dt)
    return dpl


class Dipole():
    """Dipole class."""

    # fix to allow init from data in memory (not disk)
    def __init__(self, f_dpl):
        """ some usage: dpl = Dipole(file_dipole, file_param)
            this gives dpl.t and dpl.dpl
        """
        self.units = None
        self.N = None
        self._parse_f(f_dpl)

    def _parse_f(self, f_dpl):
        """Opens the file and sets units."""
        x = np.loadtxt(open(f_dpl, 'r'))
        # better implemented as a dict
        self.t = x[:, 0]
        self.dpl = {
            'agg': x[:, 1],
            'L2': x[:, 2],
            'L5': x[:, 3],
        }
        self.N = self.dpl['agg'].shape[-1]
        # string that holds the units
        self.units = 'fAm'

    # conversion from fAm to nAm
    def convert_fAm_to_nAm(self):
        """ must be run after baseline_renormalization()
        """
        for key in self.dpl.keys():
            self.dpl[key] *= 1e-6
        # change the units string
        self.units = 'nAm'

    def scale(self, fctr):
        for key in self.dpl.keys():
            self.dpl[key] *= fctr
        return fctr

    def smooth(self, winsz):
        if winsz <= 1:
            return
        for key in self.dpl.keys():
            self.dpl[key] = _hammfilt(self.dpl[key], winsz)

    def plot(self, layer='agg'):
        """Simple layer-specific plot function.

        Parameters
        ----------
        layer : str
            The layer to plot

        Returns
        -------
        fig : instance of plt.fig
            The matplotlib figure handle.
        """
        import matplotlib.pyplot as plt
        if layer in self.dpl.keys():
            fig = plt.plot(self.t, self.dpl[layer])
            plt.xlabel('Time (ms)')
        plt.show()
        return fig

    # ext function to renormalize
    # this function changes in place but does NOT write
    # the new values to the file
    def baseline_renormalize(self, f_param):
        # only baseline renormalize if the units are fAm
        if self.units == 'fAm':
            N_pyr_x = find_param(f_param, 'N_pyr_x')
            N_pyr_y = find_param(f_param, 'N_pyr_y')
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
            self.dpl['L2'] -= dpl_offset['L2']
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
            self.dpl['L5'][self.t <= 37.] -= dpl_offset['L5']
            self.dpl['L5'][(self.t > 37.) & (self.t < t1)] -= N_pyr * \
                (m * self.t[(self.t > 37.) & (self.t < t1)] + b)
            self.dpl['L5'][self.t >= t1] -= N_pyr * \
                (m1 * self.t[self.t >= t1] + b1)
            # recalculate the aggregate dipole based on the baseline
            # normalized ones
            self.dpl['agg'] = self.dpl['L2'] + self.dpl['L5']
        else:
            print("Warning, no dipole renormalization done because units"
                  " were in %s" % (self.units))

    # function to write to a file!
    # f_dpl must be fully specified
    def write(self, f_dpl):
        with open(f_dpl, 'w') as f:
            for t, x_agg, x_L2, x_L5 in zip(self.t, self.dpl['agg'],
                                            self.dpl['L2'], self.dpl['L5']):
                f.write("%03.3f\t" % t)
                f.write("%5.4f\t" % x_agg)
                f.write("%5.4f\t" % x_L2)
                f.write("%5.4f\n" % x_L5)
