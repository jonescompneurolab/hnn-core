"""Class to handle the dipoles."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import numpy as np
from numpy import convolve, hamming


def _hammfilt(x, winsz):
    """Convolve with a hamming window."""
    win = hamming(winsz)
    win /= sum(win)
    return convolve(x, win, 'same')


def rmse (a1, a2):
    from numpy import sqrt

    # return root mean squared error between a1, a2; assumes same lengths, sampling rates
    len1,len2 = len(a1),len(a2)
    sz = min(len1,len2)
    return sqrt(((a1[0:sz] - a2[0:sz]) ** 2).mean())


def calcerr (ddat):
    from scipy import signal

    # calculates RMSE error from ddat
    # first downsample simulation timeseries to 600 Hz (assumes same time length as data)
    dpldown = signal.resample(ddat['dpl']['agg'], len(ddat['dextdata']))
    err0 = rmse(ddat['dextdata'][:,1], dpldown)
    return err0


def initialize_sim_once(net):
    """
    Initialize NEURON simulation variables

    Parameters
    ----------
    net : Network object
        The Network object with parameter values
    """

    from .parallel import pc
    from neuron import h
    h.load_file("stdrun.hoc")

    global t_vec, dp_rec_L2, dp_rec_L5

    t_vec = h.Vector()
    dp_rec_L2 = h.Vector()
    dp_rec_L5 = h.Vector()

    # global variables, should be node-independent
    h("dp_total_L2 = 0.")
    h("dp_total_L5 = 0.")

    t_vec.record(h._ref_t)  # time recording
    dp_rec_L2.record(h._ref_dp_total_L2)  # L2 dipole recording
    dp_rec_L5.record(h._ref_dp_total_L5)  # L5 dipole recording

    # Set tstop before instantiating any classes
    h.tstop = net.params['tstop']
    h.dt = net.params['dt']  # simulation duration and time-step
    h.celsius = net.params['celsius']  # 37.0 - set temperature


def simulate_dipole(net, trial=0, inc_evinput=0.0, print_progress=True, extdata=None):
    """Simulate a dipole given the experiment parameters.

    Parameters
    ----------
    net : Network object
        The Network object specifying how cells are
        connected.

    trial : int
        Current trial number

    evinputinc : float
        An increment (in milliseconds) that gets added
        to the evoked inputs on each successive trial.
        The default value is 0.0.

    print_progress : bool
        False will turn off "Simulation time" messages

    extdata : np.Array | None
        Array with preloaded data to compare simulation
        results against

    Returns
    -------
    dpl: instance of Dipole
        The dipole object
    """
    from .parallel import rank, nhosts, pc, cvode
    from .network import Network

    from neuron import h
    h.load_file("stdrun.hoc")

    # maintain vectors across trials
    global t_vec, dp_rec_L2, dp_rec_L5

    if trial == 0:
        initialize_sim_once(net)
    else:
        net.state_init()
        # adjusts the rng seeds and then the feed/event input times
        net.reset_src_event_times(inc_evinput = inc_evinput * (trial + 1))

    # Now let's simulate the dipole

    pc.barrier() # sync for output to screen
    if rank == 0:
        print("Running trial %d (on %d cores)" % (trial+1, nhosts))

    # initialize cells to -65 mV, after all the NetCon
    # delays have been specified
    h.finitialize()

    def prsimtime():
        print('Simulation time: {0} ms...'.format(round(h.t, 2)))

    printdt = 10
    if print_progress and rank == 0:
        for tt in range(0, int(h.tstop), printdt):
            cvode.event(tt, prsimtime)  # print time callbacks

    h.fcurrent()

    pc.barrier()  # get all nodes to this place before continuing

    # actual simulation - run the solver
    pc.psolve(h.tstop)

    pc.barrier()

    # these calls aggregate data across procs/nodes
    pc.allreduce(dp_rec_L2, 1)
    # combine dp_rec on every node, 1=add contributions together
    pc.allreduce(dp_rec_L5, 1)
    # aggregate the currents independently on each proc
    net.aggregate_currents()
    # combine net.current{} variables on each proc
    pc.allreduce(net.current['L5Pyr_soma'], 1)
    pc.allreduce(net.current['L2Pyr_soma'], 1)

    pc.barrier()  # get all nodes to this place before continuing

    dpl_data = np.c_[np.array(dp_rec_L2.to_python()) +
                     np.array(dp_rec_L5.to_python()),
                     np.array(dp_rec_L2.to_python()),
                     np.array(dp_rec_L5.to_python())]


    dpl = Dipole(np.array(t_vec.to_python()), dpl_data)

    err = None
    if rank == 0:
        if net.params['save_dpl']:
            dpl.write('rawdpl_%d.txt' % trial)

        dpl.baseline_renormalize(net.params)
        dpl.convert_fAm_to_nAm()
        dpl.scale(net.params['dipole_scalefctr'])
        dpl.smooth(net.params['dipole_smooth_win'] / h.dt)

        try:
            if extdata.any():
                ddat = {'dpl' : dpl.dpl, 'dextdata' : extdata}
                err = calcerr(ddat)
                print("RMSE:", err)
        except AttributeError:
            # extdata is not an array
            pass

    return dpl, err


def average_dipoles(dpls):
    """Compute average over a list of Dipole objects.

    Parameters
    ----------
    dpls: list of Dipole objects
        Contains list of dipole results to be averaged

    Returns
    -------
    dpl: instance of Dipole
        A dipole object with averages of the dipole data
    """

    agg_avg = np.mean(np.array([dpl.dpl['agg'] for dpl in dpls]), axis=0)
    L5_avg = np.mean(np.array([dpl.dpl['L5'] for dpl in dpls]), axis=0)
    L2_avg = np.mean(np.array([dpl.dpl['L2'] for dpl in dpls]), axis=0)

    avg_dpl_data = np.c_[agg_avg,
                         L2_avg,
                         L5_avg]

    avg_dpl = Dipole(np.array(t_vec.to_python()), avg_dpl_data)

    return avg_dpl


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
    t : array
        The time vector
    dpl : dict of array
        The dipole with keys 'agg', 'L2' and 'L5'
    """

    def __init__(self, times, data):  # noqa: D102
        self.units = 'fAm'
        self.N = data.shape[0]
        self.t = times
        self.dpl = {'agg': data[:, 0], 'L2': data[:, 1], 'L5': data[:, 2]}

    # conversion from fAm to nAm
    def convert_fAm_to_nAm(self):
        """ must be run after baseline_renormalization()
        """
        for key in self.dpl.keys():
            self.dpl[key] *= 1e-6
        self.units = 'nAm'

    def scale(self, fctr):
        for key in self.dpl.keys():
            self.dpl[key] *= fctr
        return fctr

    def smooth(self, winsz):
        # XXX: add check to make sure self.t is
        # not smaller than winsz
        if winsz <= 1:
            return
        for key in self.dpl.keys():
            self.dpl[key] = _hammfilt(self.dpl[key], winsz)

    def plot(self, ax=None, layer='agg'):
        """Simple layer-specific plot function.

        Parameters
        ----------
        ax : instance of matplotlib figure | None
            The matplotlib axis
        layer : str
            The layer to plot
        show : bool
            If True, show the figure

        Returns
        -------
        fig : instance of plt.fig
            The matplotlib figure handle.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if layer in self.dpl.keys():
            ax.plot(self.t, self.dpl[layer])
            ax.set_xlabel('Time (ms)')
        if True:
            plt.show()
        return ax.get_figure()

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

    def write(self, fname = 'dpl.txt'):
        """Write dipole values to a file.

        Parameters
        ----------
        fname : str
            Output filename. Must be fully specified
        """

        with open(fname, 'w') as f:
            for t, x_agg, x_L2, x_L5 in zip(self.t, self.dpl['agg'], self.dpl['L2'], self.dpl['L5']):
                f.write("%03.3f\t" % t)
                f.write("%5.8f\t" % x_agg)
                f.write("%5.8f\t" % x_L2)
                f.write("%5.8f\n" % x_L5)
