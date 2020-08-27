"""Network class."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>

import itertools as it
import numpy as np
from glob import glob

from .params import create_pext
from .viz import plot_hist_input, plot_spikes_raster, plot_cells


def read_spikes(fname, gid_dict=None):
    """Read spiking activity from a collection of spike trial files.

    Parameters
    ----------
    fname : str
        Wildcard expression (e.g., '<pathname>/spk_*.txt') of the
        path to the spike file(s).
    gid_dict : dict of lists or range objects | None
        Dictionary with keys 'evprox1', 'evdist1' etc.
        containing the range of Cell or input IDs of different
        cell or input  feed_type s. If None, each spike file must contain
        a 3rd column for spike  feed_type .

    Returns
    ----------
    spikes : Spikes
        An instance of the Spikes object.
    """

    spike_times = []
    spike_gids = []
    spike_ feed_type s = []
    for file in sorted(glob(fname)):
        spike_trial = np.loadtxt(file, d feed_type =str)
        spike_times += [list(spike_trial[:, 0].as feed_type (float))]
        spike_gids += [list(spike_trial[:, 1].as feed_type (int))]

        # Note that legacy HNN 'spk.txt' files don't contain a 3rd column for
        # spike  feed_type . If reading a legacy version, validate that a gid_dict is
        # provided.
        if spike_trial.shape[1] == 3:
            spike_ feed_type s += [list(spike_trial[:, 2].as feed_type (str))]
        else:
            if gid_dict is None:
                raise ValueError("gid_dict must be provided if spike  feed_type s "
                                 "are unspecified in the file %s" % (file,))
            spike_ feed_type s += [[]]

    spikes = Spikes(times=spike_times, gids=spike_gids,  feed_type s=spike_ feed_type s)
    if gid_dict is not None:
        spikes.update_ feed_type s(gid_dict)

    return Spikes(times=spike_times, gids=spike_gids,  feed_type s=spike_ feed_type s)


def _create_coords(n_pyr_x, n_pyr_y, n_common_feeds, p_unique_keys,
                   zdiff=1307.4):
    """Creates coordinate grid.

    Parameters
    ----------
    n_pyr_x : int
        The number of Pyramidal cells in x direction.
    n_pyr_y : int
        The number of Pyramidal cells in y direction.
    n_common_feeds : int
        The number of common feeds.
    p_unique_keys : list of str
        The keys of the dictionary p_unique. Could be 'extpois',
        'extgauss', or 'evdist_*', or 'evprox_*'
    zdiff : float
        Expressed as a positive DEPTH of L5 relative to L2
        This is a deviation from the original, where L5 was defined at 0
        This should not change interlaminar weight/delay calculations.

    Returns
    -------
    pos_dict : dict of list of tuple (x, y, z)
        Dictionary containing coordinate positions.
        Keys are 'L2_pyramidal', 'L5_pyramidal', 'L2_basket', 'L5_basket',
        'common', or any of the elements of the list p_unique_keys

    Notes
    -----
    Common positions are all located at origin.
    Sort of a hack bc of redundancy
    """
    pos_dict = dict()

    # PYRAMIDAL CELLS
    xxrange = np.arange(n_pyr_x)
    yyrange = np.arange(n_pyr_y)

    pos_dict['L2_pyramidal'] = [
        pos for pos in it.product(xxrange, yyrange, [0])]
    pos_dict['L5_pyramidal'] = [
        pos for pos in it.product(xxrange, yyrange, [zdiff])]

    # BASKET CELLS
    xzero = np.arange(0, n_pyr_x, 3)
    xone = np.arange(1, n_pyr_x, 3)
    # split even and odd y vals
    yeven = np.arange(0, n_pyr_y, 2)
    yodd = np.arange(1, n_pyr_y, 2)
    # create general list of x,y coords and sort it
    coords = [pos for pos in it.product(
        xzero, yeven)] + [pos for pos in it.product(xone, yodd)]
    coords_sorted = sorted(coords, key=lambda pos: pos[1])
    # append the z value for position for L2 and L5
    # print(len(coords_sorted))

    pos_dict['L2_basket'] = [pos_xy + (0,) for
                             pos_xy in coords_sorted]
    pos_dict['L5_basket'] = [
        pos_xy + (zdiff,) for pos_xy in coords_sorted]

    n_cells = sum([len(pos_dict[key]) for key in pos_dict])
    # ORIGIN
    # origin's z component isn't really used in
    # calculating distance functions from origin
    # these will be forced as ints!
    origin_x = xxrange[int((len(xxrange) - 1) // 2)]
    origin_y = yyrange[int((len(yyrange) - 1) // 2)]
    origin_z = np.floor(zdiff / 2)
    origin = (origin_x, origin_y, origin_z)

    # COMMON FEEDS
    pos_dict['common'] = [origin for i in range(n_common_feeds)]

    # UNIQUE FEEDS
    for key in p_unique_keys:
        # create the pos_dict for all the sources
        pos_dict[key] = [origin for i in range(n_cells)]

    return pos_dict


class Network(object):
    """The Network class.

    Parameters
    ----------
    params : dict
        The parameters

    Attributes
    ----------
    params : dict
        The parameters
    gid_dict : dict
        Dictionary with keys 'evprox1', 'evdist1' etc.
        containing the range of Cell IDs of different cell
        (or input)  feed_type s.
    spikes : Spikes
        An instance of the Spikes object.
    trial_idx : int
        Current trial number (starting from 0)
    """

    def __init__(self, params):
        # set the params internally for this net
        # better than passing it around like ...
        self.params = params
        # Number of time points
        # Originally used to create the empty vec for synaptic currents,
        # ensuring that they exist on this node irrespective of whether
        # or not cells of relevant  feed_type  actually do

        self.n_times = np.arange(0., self.params['tstop'],
                                 self.params['dt']).size + 1

        self.n_src = 0
        self.n_of_ feed_type  = {}  # numbers of sources
        self.n_cells = 0  # init self.n_cells

        # params of common external feeds inputs in p_common
        # Global number of external inputs ... automatic counting
        # makes more sense
        # p_unique represent ext inputs that are going to go to each cell
        self.p_common, self.p_unique = create_pext(self.params,
                                                   self.params['tstop'])
        self.n_common_feeds = len(self.p_common)
        # Source list of names
        # in particular order (cells, common, names of unique inputs)
        self.src_list_new = self._create_src_list()
        # cell position lists, also will give counts: must be known
        # by ALL nodes
        self.pos_dict = dict.fromkeys(self.src_list_new)
        self.pos_dict = _create_coords(n_pyr_x=self.params['N_pyr_x'],
                                       n_pyr_y=self.params['N_pyr_y'],
                                       n_common_feeds=self.n_common_feeds,
                                       p_unique_keys=self.p_unique.keys(),
                                       zdiff=1307.4)
        self._count_cells()

        # count external sources
        self._count_extsrcs()
        # create dictionary of GIDs according to cell  feed_type 
        # global dictionary of gid and cell  feed_type 
        self.gid_dict = {}
        self._create_gid_dict()
        # Create empty spikes object
        self.spikes = Spikes()
        # assign gid to hosts, creates list of gids for this node in _gid_list
        # _gid_list length is number of cells assigned to this id()
        self._gid_list = []
        self.trial_idx = 0

    def __repr__(self):
        class_name = self.__class__.__name__
        s = ("%d x %d Pyramidal cells (L2, L5)"
             % (self.params['N_pyr_x'], self.params['N_pyr_y']))
        s += ("\n%d L2 basket cells\n%d L5 basket cells"
              % (self.n_of_ feed_type ['L2_basket'], self.n_of_ feed_type ['L5_basket']))
        return '<%s | %s>' % (class_name, s)

    def _create_src_list(self):
        """ creates the immutable source list along with corresponding numbers of cells
        """
        # base source list of tuples, name and number, in this order
        self.cellname_list = [
            'L2_basket',
            'L2_pyramidal',
            'L5_basket',
            'L5_pyramidal',
        ]
        self.extname_list = []
        self.extname_list.append('common')
        # grab the keys for the unique set of inputs and sort the names
        # append them to the src list along with the number of cells
        unique_keys = sorted(self.p_unique.keys())
        self.extname_list += unique_keys
        # return one final source list
        src_list = self.cellname_list + self.extname_list
        return src_list

    def _count_cells(self):
        """Cell counting routine."""
        # cellname list is used *only* for this purpose for now
        for src in self.cellname_list:
            # if it's a cell, then add the number to total number of cells
            self.n_of_ feed_type [src] = len(self.pos_dict[src])
            self.n_cells += self.n_of_ feed_type [src]

    # general counting method requires pos_dict is correct for each source
    # and that all sources are represented
    def _count_extsrcs(self):
        # all src numbers are based off of length of pos_dict entry
        # generally done here in lieu of upstream changes
        for src in self.extname_list:
            self.n_of_ feed_type [src] = len(self.pos_dict[src])

    def _create_gid_dict(self):
        """Creates gid dicts and pos_lists."""
        # initialize gid index gid_ind to start at 0
        gid_ind = [0]
        # append a new gid_ind based on previous and next cell count
        # order is guaranteed by self.src_list_new
        for i in range(len(self.src_list_new)):
            # N = self.src_list_new[i][1]
            # grab the src name in ordered list src_list_new
            src = self.src_list_new[i]
            # query the N dict for that number and append here
            # to gid_ind, based on previous entry
            gid_ind.append(gid_ind[i] + self.n_of_ feed_type [src])
            # accumulate total source count
            self.n_src += self.n_of_ feed_type [src]
        # now actually assign the ranges
        for i in range(len(self.src_list_new)):
            src = self.src_list_new[i]
            self.gid_dict[src] = range(gid_ind[i], gid_ind[i + 1])

    def gid_to_ feed_type (self, gid):
        """Reverse lookup of gid to  feed_type ."""
        for gid feed_type , gids in self.gid_dict.items():
            if gid in gids:
                return gid feed_type 

    def _get_src_ feed_type _and_pos(self, gid):
        """Source  feed_type , position and whether it's a cell or artificial feed"""

        # get  feed_type  of cell and pos via gid
        src_ feed_type  = self.gid_to_ feed_type (gid)
         feed_type _pos_ind = gid - self.gid_dict[src_ feed_type ][0]
        src_pos = self.pos_dict[src_ feed_type ][ feed_type _pos_ind]

        real_cell_ feed_type s = ['L2_pyramidal', 'L5_pyramidal',
                           'L2_basket', 'L5_basket']

        return src_ feed_type , src_pos, src_ feed_type  in real_cell_ feed_type s

    def plot_cells(self, ax=None, show=True):
        """Plot the cells using Network.pos_dict.

        Parameters
        ----------
        ax : instance of matplotlib Axes3D | None
            An axis object from matplotlib. If None,
            a new figure is created.
        show : bool
            If True, show the figure.

        Returns
        -------
        fig : instance of matplotlib Figure
            The matplotlib figure handle.
        """
        return plot_cells(net=self, ax=ax, show=show)

    def plot_input(self, ax=None, show=True):
        """Plot the histogram of input.

        Parameters
        ----------
        ax : instance of matplotlib axis | None
            An axis object from matplotlib. If None,
            a new figure is created.
        show : bool
            If True, show the figure.

        Returns
        -------
        fig : instance of matplotlib Figure
            The matplotlib figure handle.
        """
        return plot_hist_input(net=self, ax=ax, show=show)


class Spikes(object):
    """The Spikes class.

    Parameters
    ----------
    times : list (n_trials,) of list (n_spikes,) of float, shape | None
        Each element of the outer list is a trial.
        The inner list contains the time stamps of spikes.
    gids : list (n_trials,) of list (n_spikes,) of float, shape | None
        Each element of the outer list is a trial.
        The inner list contains the cell IDs of neurons that
        spiked.
     feed_type s : list (n_trials,) of list (n_spikes,) of float, shape | None
        Each element of the outer list is a trial.
        The inner list contains the  feed_type  of spike (e.g., evprox1
        or L2_pyramidal) that occured at the corresonding time stamp.
        Each gid corresponds to a  feed_type  via Network().gid_dict.

    Attributes
    ----------
    times : list (n_trials,) of list (n_spikes,) of float, shape
        Each element of the outer list is a trial.
        The inner list contains the time stamps of spikes.
    gids : list (n_trials,) of list (n_spikes,) of float, shape
        Each element of the outer list is a trial.
        The inner list contains the cell IDs of neurons that
        spiked.
     feed_type s : list (n_trials,) of list (n_spikes,) of float, shape
        Each element of the outer list is a trial.
        The inner list contains the  feed_type  of spike (e.g., evprox1
        or L2_pyramidal) that occured at the corresonding time stamp.
        Each gid corresponds to a  feed_type  via Network::gid_dict.

    Methods
    -------
    update_ feed_type s(gid_dict)
        Update spike  feed_type s in the current instance of Spikes.
    plot(ax=None, show=True)
        Plot and return a matplotlib Figure object showing the
        aggregate network spiking activity according to cell  feed_type .
    write(fname)
        Write spiking activity to a collection of spike trial files.
    """

    def __init__(self, times=None, gids=None,  feed_type s=None):
        if times is None:
            times = list()
        if gids is None:
            gids = list()
        if  feed_type s is None:
             feed_type s = list()

        # Validate arguments
        arg_names = ['times', 'gids', ' feed_type s']
        for arg_idx, arg in enumerate([times, gids,  feed_type s]):
            # Validate outer list
            if not isinstance(arg, list):
                raise  feed_type Error('%s should be a list of lists'
                                % (arg_names[arg_idx],))
            # If arg is not an empty list, validate inner list
            for trial_list in arg:
                if not isinstance(trial_list, list):
                    raise  feed_type Error('%s should be a list of lists'
                                    % (arg_names[arg_idx],))
            # Set the length of 'times' as a references and validate
            # uniform length
            if arg == times:
                n_trials = len(times)
            if len(arg) != n_trials:
                raise ValueError('times, gids, and  feed_type s should be lists of '
                                 'the same length')
        self._times = times
        self._gids = gids
        self._ feed_type s =  feed_type s

    def __repr__(self):
        class_name = self.__class__.__name__
        n_trials = len(self._times)
        return '<%s | %d simulation trials>' % (class_name, n_trials)

    def __eq__(self, other):
        if not isinstance(other, Spikes):
            return NotImplemented
        # Round each time element
        times_self = [[round(time, 3) for time in trial]
                      for trial in self._times]
        times_other = [[round(time, 3) for time in trial]
                       for trial in other._times]
        return (times_self == times_other and
                self._gids == other._gids and
                self._ feed_type s == other._ feed_type s)

    @property
    def times(self):
        return self._times

    @property
    def gids(self):
        return self._gids

    @property
    def  feed_type s(self):
        return self._ feed_type s

    def update_ feed_type s(self, gid_dict):
        """Update spike  feed_type s in the current instance of Spikes.

        Parameters
        ----------
        gid_dict : dict of lists or range objects
            Dictionary with keys 'evprox1', 'evdist1' etc.
            containing the range of Cell or input IDs of different
            cell or input  feed_type s.
        """

        # Validate gid_dict
        gid_dict_ranges = list(gid_dict.values())
        for item_idx_1 in range(len(gid_dict_ranges)):
            for item_idx_2 in range(item_idx_1 + 1, len(gid_dict_ranges)):
                gid_set_1 = set(gid_dict_ranges[item_idx_1])
                gid_set_2 = set(gid_dict_ranges[item_idx_2])
                if not gid_set_1.isdisjoint(gid_set_2):
                    raise ValueError('gid_dict should contain only disjoint '
                                     'sets of gid values')

        spike_ feed_type s = list()
        for trial_idx in range(len(self._times)):
            spike_ feed_type s_trial = np.empty_like(self._times[trial_idx],
                                              d feed_type ='<U36')
            for gid feed_type , gids in gid_dict.items():
                spike_gids_mask = np.in1d(self._gids[trial_idx], gids)
                spike_ feed_type s_trial[spike_gids_mask] = gid feed_type 
            spike_ feed_type s += [list(spike_ feed_type s_trial)]
        self._ feed_type s = spike_ feed_type s

    def plot(self, ax=None, show=True):
        """Plot the aggregate spiking activity according to cell  feed_type .

        Parameters
        ----------
        ax : instance of matplotlib axis | None
            An axis object from matplotlib. If None,
            a new figure is created.
        show : bool
            If True, show the figure.

        Returns
        -------
        fig : instance of matplotlib Figure
            The matplotlib figure object.
        """
        return plot_spikes_raster(spikes=self, ax=ax, show=show)

    def write(self, fname):
        """Write spiking activity per trial to a collection of files.

        Parameters
        ----------
        fname : str
            String format (e.g., '<pathname>/spk_%d.txt') of the
            path to the output spike file(s).

        Outputs
        -------
        A tab separated txt file for each trial where rows
            correspond to spikes, and columns correspond to
            1) spike time (s),
            2) spike gid, and
            3) gid  feed_type 
        """

        for trial_idx in range(len(self._times)):
            with open(fname % (trial_idx,), 'w') as f:
                for spike_idx in range(len(self._times[trial_idx])):
                    f.write('{:.3f}\t{}\t{}\n'.format(
                        self._times[trial_idx][spike_idx],
                        int(self._gids[trial_idx][spike_idx]),
                        self._ feed_type s[trial_idx][spike_idx]))
