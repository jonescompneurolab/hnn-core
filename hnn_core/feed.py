"""External feed to network."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Christopher Bailey <bailey.cj@gmail.com>

import numpy as np


# based on cdf for exp wait time distribution from unif [0, 1)
# returns in ms based on lamtha in Hz
def _t_wait(prng, lamtha):
    return -1000. * np.log(1. - prng.rand()) / lamtha


def _get_prng(seed, gid, sync_evinput=False):
    """Random generator for this instance.

    Parameters
    ----------
    seed : int
        The seed for random state generator.
    gid : int
        The cell ID
    sync_evinput : bool
        If True, all cells get the same prng

    Returns
    -------
    prng : instance of RandomState
        The seed for events assuming a given start time.
    prng2 : instance of RandomState
        The seed for start times.
    """
    # XXX: some param files use seed < 0 but numpy
    # does not allow this.
    if seed > 0:
        prng2 = np.random.RandomState(seed)
    else:
        prng2 = None

    if not sync_evinput:
        seed = seed + gid

    prng = np.random.RandomState(seed)
    return prng, prng2


class ExtFeed(object):
    """The ExtFeed class of external spike input times.

    An external input "feed" to the network, i.e., one that is independent of
    the spiking output of cells in the network.

    Parameters
    ----------
    feed_type : str
        The feed type, which is one of
        'extpois' : Poisson-distributed input to proximal dendrites
        'extgauss' : Gaussian-distributed input to proximal dendrites
        'evprox' : Proximal input at specified time (or Gaussian spread)
        'evdist' : Distal input at specified time (or Gaussian spread)

        'common' : As opposed to other feed types, these have timing that is
        identical (synchronous) for all real cells in the network. Proximal
        and distal dendrites have separate parameter sets, and need not be
        synchronous. Note that not all cells classes (types) are required to
        receive 'common' input---separate conductivity values can be assigned
        to basket vs. pyramidal cells and AMPA vs. NMDA synapses
    target_cell_type : str | None
        The target cell type of the feed, e.g., 'L2_basket', 'L5_pyramidal',
        etc., or None for 'common' inputs
    params : dict
        Parameters of the external input feed, arranged into a dictionary.
    gid : int
        The cell ID.

    Attributes
    ----------
    event_times : list
        A list of event times
    feed_type : str
        The feed type corresponding to the given gid (e.g., 'extpois',
        'extgauss', 'common', 'evprox', 'evdist')
    params : dict
        Parameters of the given feed type
    seed : int
        The seed
    gid : int
        The cell ID
    """

    def __init__(self, feed_type, target_cell_type, params, gid):
        self.params = params
        # used to determine cell type-specific parameters for
        # (not used for 'common', such as for rhythmic alpha/beta input)
        self.cell_type = target_cell_type
        self.feed_type = feed_type
        self.gid = gid
        prng, prng2 = _get_prng(
            seed=self.params['prng_seedcore'],
            gid=self.gid,
            sync_evinput=self.params.get('sync_evinput', False))
        self.prng, self.prng2 = prng, prng2
        self.event_times = list()
        self.set_event_times()

    def __repr__(self):
        class_name = self.__class__.__name__
        repr_str = "<%s of type '%s' " % (class_name, self.feed_type)
        repr_str += 'with %d events ' % len(self.event_times)
        repr_str += '| gid %d>' % (self.gid)
        return repr_str

    def set_event_times(self):

        # check feed name validity, allowing substring matches ('evprox1' etc)
        valid_feeds = ['extpois', 'extgauss', 'common', 'evprox', 'evdist']
        # NB check if self.feed_type has a valid substring, not vice versa
        matches = [f for f in valid_feeds if f in self.feed_type]
        if len(matches) == 0:
            raise ValueError('Invalid external feed: %s' % self.feed_type)
        elif len(matches) > 1:
            raise ValueError('Ambiguous external feed: %s' % self.feed_type)

        # Return values not checked: False if all weights for given feed type
        # are zero. Designed to be silent so that zeroing input weights
        # effectively disables each.
        zero_ampa_nmda = False
        if self.cell_type in self.params:
            zero_ampa_nmda = (self.params[self.cell_type][0] <= 0.0 and
                              self.params[self.cell_type][1] <= 0.0)

        all_syn_weights_zero = True
        for key in self.params.keys():
            if key.startswith(('L2Pyr', 'L5Pyr', 'L2Bask', 'L5Bask')):
                if self.params[key][0] > 0.0:
                    all_syn_weights_zero = False

        event_times = list()
        if self.feed_type == 'extpois' and not zero_ampa_nmda:
            event_times = self._create_extpois(
                t0=self.params['t_interval'][0],
                T=self.params['t_interval'][1],
                # ind 3 is frequency (lamtha))
                lamtha=self.params[self.cell_type][3],
                prng=self.prng)
        elif self.feed_type.startswith(('evprox', 'evdist')) and \
                self.cell_type in self.params:
            event_times = self._create_evoked(
                mu=self.params['t0'],
                # ind 3 is sigma_t (stdev))
                sigma=self.params[self.cell_type][3],
                numspikes=int(self.params['numspikes']),
                prng=self.prng)
        elif self.feed_type == 'extgauss' and not zero_ampa_nmda:
            event_times = self._create_extgauss(
                mu=self.params[self.cell_type][3],
                sigma=self.params[self.cell_type][4],
                prng=self.prng)
        elif self.feed_type == 'common' and not all_syn_weights_zero:
            event_times = self._create_common_input(
                distribution=self.params['distribution'],
                t0=self.params['t0'],
                t0_stdev=self.params['t0_stdev'],
                tstop=self.params['tstop'],
                f_input=self.params['f_input'],
                stdev=self.params['stdev'],
                repeats=self.params['repeats'],
                events_per_cycle=self.params['events_per_cycle'],
                prng=self.prng,
                prng2=self.prng2)

        self.event_times = event_times

    def _create_extpois(self, t0, T, lamtha, prng):
        """Create poisson inputs.

        Parameters
        ----------
        t0 : float
            The start time.
        T : float
            The end time.
        lamtha : float
            The spatial decay lambda.
        prng : instance of RandomState
            The random state.

        Returns
        -------
        event_times : array
            The event times.
        """
        if t0 < 0:
            raise ValueError('The start time for Poisson inputs must be'
                             f'greater than 0. Got {t0}')
        if T < t0:
            raise ValueError('The end time for Poisson inputs must be'
                             f'greater than start time. Got ({t0}, {T})')
        # values MUST be sorted for VecStim()!
        # start the initial value
        val_pois = np.array([])
        if lamtha > 0.:
            t_gen = t0 + _t_wait(prng, lamtha)
            if t_gen < T:
                np.append(val_pois, t_gen)
            # vals are guaranteed to be monotonically increasing, no need to
            # sort
            while t_gen < T:
                # so as to not clobber confusingly base off of t_gen ...
                t_gen += _t_wait(prng, lamtha)
                if t_gen < T:
                    val_pois = np.append(val_pois, t_gen)
        return val_pois.tolist()

    def _create_evoked(self, mu, sigma, numspikes, prng):
        """Create evoked inputs.

        Parameters
        ----------
        mu : float
            The mean time of spikes.
        sigma : float
            The standard deviation.
        numspikes : float
            The number of spikes.
        prng : instance of RandomState
            The random state.

        Returns
        -------
        event_times : array
            The event times.
        """
        val_evoked = np.array([])
        if sigma > 0:
            val_evoked = prng.normal(mu, sigma, numspikes)
        else:
            # if sigma is specified at 0
            val_evoked = np.array([mu] * numspikes)
        val_evoked = val_evoked[val_evoked > 0]
        # vals must be sorted
        val_evoked.sort()
        return val_evoked.tolist()

    def _create_extgauss(self, mu, sigma, prng):
        """Create gaussian input.

        Parameters
        ----------
        mu : float
            The mean time of spikes.
        sigma : float
            The standard deviation.

        Returns
        -------
        event_times : array
            The event times.

        Notes
        -----
        non-zero values are removed (why?)
        """
        # one single value from Gaussian dist.
        # values MUST be sorted for VecStim()!
        val_gauss = prng.normal(mu, sigma, 50)
        val_gauss = val_gauss[val_gauss > 0]
        val_gauss.sort()
        return val_gauss.tolist()

    def _create_common_input(self, distribution, t0, t0_stdev, tstop, f_input,
                             stdev, repeats, events_per_cycle, prng, prng2):
        """Creates the common ongoing external inputs.

        Used for, e.g., for rhythmic inputs in alpha/beta generation.

        Parameters
        ----------
        distribution : str
            The distribution for each burst. One of 'normal' or 'uniform'.
        t0 : float
            The start times. If -1, then randomize the start time
            of inputs.
        t0_stdev : float
            Standard deviation of jitter to start time.
        tstop : float
            The stop time.
        f_input : float
            The frequency of input bursts.
        stdev : float
            The standard deviation.
        repeats : int
            The number of repeats.
        events_per_cycle : float
            The events per cycle. Must be 1 or 2.
        prng : instance of RandomState
            The random state.
        prng : instance of RandomState
            The random state used for jitter to start time (see t0_stdev).
        """
        if distribution not in ('normal', 'uniform'):
            raise ValueError("Indicated distribution not recognized. "
                             "Not making any common feeds.")

        # store f_input as self variable for later use if it exists in p
        if t0 == -1:
            t0 = prng.uniform(25., 125.)
        elif t0_stdev > 0.0:
            t0 = prng2.normal(t0, t0_stdev)

        if events_per_cycle != 1:
            print("events_per_cycle should be either 1 or 2, trying 2")
            events_per_cycle = 2

        if distribution == 'normal':
            # array of mean stimulus times, starts at t0
            isi_array = np.arange(t0, tstop, 1000. / f_input)
            # array of single stimulus times -- no doublets
            if stdev > 0:
                t_array = prng.normal(np.repeat(isi_array, repeats), stdev)
            else:
                t_array = isi_array
        elif distribution == 'uniform':
            n_inputs = repeats * f_input * (tstop - t0) / 1000.
            t_array = prng.uniform(t0, tstop, n_inputs)

        t_input = np.array([])
        if events_per_cycle == 2:
            # Two arrays store doublet times
            t_input_low = t_array - 5
            t_input_high = t_array + 5
            # Array with ALL stimulus times for input
            # np.append concatenates two np arrays
            t_input = np.append(t_input_low, t_input_high)
        elif events_per_cycle == 1:
            t_input = t_array
        # brute force remove non-zero times. Might result in fewer vals
        # than desired
        t_input = t_input[t_input > 0]
        t_input.sort()

        return t_input.tolist()
