# dipole.py - dipole-based analysis functions
#
# v 1.10.0-py35
# rev 2016-05-01 (SL: itertools and return data dir)
# last major: (SL: toward python3)
import numpy as np

from .paramrw import find_param
from .filt import hammfilt

# class Dipole() is for a single set of f_dpl and f_param


class Dipole():
    def __init__(self, f_dpl):  # fix to allow init from data in memory (not disk)
        """ some usage: dpl = Dipole(file_dipole, file_param)
            this gives dpl.t and dpl.dpl
        """
        self.units = None
        self.N = None
        self.__parse_f(f_dpl)

    # opens the file and sets units
    def __parse_f(self, f_dpl):
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

    # truncate to a length and save here
    def truncate(self, t0, T):
        """ this is independent of the other stuff
            moved to an external function so as to not disturb the delicate genius of this object
        """
        self.t, self.dpl = self.truncate_ext(t0, T)

    # just return the values, do not modify the class internally
    def truncate_ext(self, t0, T):
        # only do this if the limits make sense
        if (t0 >= self.t[0]) & (T <= self.t[-1]):
            dpl_truncated = dict.fromkeys(self.dpl)
            # do this for each dpl
            for key in self.dpl.keys():
                dpl_truncated[key] = self.dpl[key][(
                    self.t >= t0) & (self.t <= T)]
            t_truncated = self.t[(self.t >= t0) & (self.t <= T)]
        return t_truncated, dpl_truncated

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
        #for key in self.dpl.keys(): self.dpl[key] = boxfilt(self.dpl[key],winsz)
        for key in self.dpl.keys():
            self.dpl[key] = hammfilt(self.dpl[key], winsz)

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
        plt.show()
        return fig

    # ext function to renormalize
    # this function changes in place but does NOT write the new values to the file
    def baseline_renormalize(self, f_param):
        # only baseline renormalize if the units are fAm
        if self.units == 'fAm':
            N_pyr_x = find_param(f_param, 'N_pyr_x')
            N_pyr_y = find_param(f_param, 'N_pyr_y')
            # N_pyr cells in grid. This is PER LAYER
            N_pyr = N_pyr_x * N_pyr_y
            # dipole offset calculation: increasing number of pyr cells (L2 and L5, simultaneously)
            # with no inputs resulted in an aggregate dipole over the interval [50., 1000.] ms that
            # eventually plateaus at -48 fAm. The range over this interval is something like 3 fAm
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
            # L2 dipole offset can be roughly baseline shifted over the entire range of t
            self.dpl['L2'] -= dpl_offset['L2']
            # L5 dipole offset should be different for interval [50., 500.] and then it can be offset
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
            # recalculate the aggregate dipole based on the baseline normalized ones
            self.dpl['agg'] = self.dpl['L2'] + self.dpl['L5']
        else:
            print("Warning, no dipole renormalization done because units were in %s" % (
                self.units))

    # function to write to a file!
    # f_dpl must be fully specified
    def write(self, f_dpl):
        with open(f_dpl, 'w') as f:
            for t, x_agg, x_L2, x_L5 in zip(self.t, self.dpl['agg'], self.dpl['L2'], self.dpl['L5']):
                f.write("%03.3f\t" % t)
                f.write("%5.4f\t" % x_agg)
                f.write("%5.4f\t" % x_L2)
                f.write("%5.4f\n" % x_L5)
