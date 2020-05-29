"""External feed to network."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Christopher Bailey <bailey.cj@gmail.com>

import numpy as np
from neuron import h


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
        'common' : Common inputs to all cells (locations parameter-dependent)
    cell_type : str | None
        The cell type, e.g., 'L2_basket', 'L5_pyramidal', etc., or None for
        common inputs (cell population-level weights for common inputs
        are defined separately in the parameters)
    p_ext : dict
        Parameters of the external input feed, arranged into a dictionary.
    gid : int
        The cell ID.

    Attributes
    ----------
    nrn_eventvec : instance of NEURON Vector
        A vector of event times
    nrn_vecstim : instance of NEURON VecStim
        A VecStim is an artificial spiking cell that generates events at
        times that are specified in a (NEURON) Vector (see vecevent.mod)
    seed : int
        The seed
    gid : int
        The cell ID
    """

    def __init__(self, feed_type, cell_type, p_ext, gid):
        # VecStim setup
        self.nrn_eventvec = h.Vector()
        self.nrn_vecstim = h.VecStim()
        self.p_ext = p_ext
        # used to determine cell type-specific parameters for
        # (not used for 'common', such as for rhythmic alpha/beta input)
        self.cell_type = cell_type  # XXX rename for all cell types?
        self.feed_type = feed_type
        self.gid = gid
        self.set_prng()  # sets seeds for random num generator
        # sets event times into self.nrn_eventvec (Vector)
        # and plays into self.nrn_vecstim (VecStim)
        self.set_event_times()

    def __repr__(self):
        class_name = self.__class__.__name__
        repr_str = "<%s of type '%s' " % (class_name, self.feed_type)
        repr_str += 'with %d events ' % len(self.nrn_eventvec)
        repr_str += '| seed %d, gid %d>' % (self.seed, self.gid)
        return repr_str

    def set_prng(self, seed=None):
        if seed is None:  # no seed specified then use p_ext to determine seed
            # random generator for this instance
            # qnd hack to make the seeds the same across all gids
            # for just evoked
            if self.feed_type.startswith(('evprox', 'evdist')):
                if self.p_ext['sync_evinput']:
                    self.seed = self.p_ext['prng_seedcore']
                else:
                    self.seed = self.p_ext['prng_seedcore'] + self.gid - 2
            elif self.feed_type.startswith('common'):
                # seed for events assuming a given start time
                self.seed = self.p_ext['prng_seedcore'] + self.gid
                # separate seed for start times
                self.seed2 = self.p_ext['prng_seedcore']
            else:
                self.seed = self.p_ext['prng_seedcore'] + self.gid
        else:  # if seed explicitly specified use it
            self.seed = seed
            if hasattr(self, 'seed2'):
                self.seed2 = seed
        self.prng = np.random.RandomState(self.seed)
        if hasattr(self, 'seed2'):
            self.prng2 = np.random.RandomState(self.seed2)
        # print('feed_type,seed:',self.feed_type,self.seed)

    def set_event_times(self, inc_evinput=0.0):

        # check feed name validity, allowing substring matches ('evprox1' etc)
        valid_feeds = ['extpois', 'extgauss', 'common', 'evprox', 'evdist']
        # NB check if self.feed_type has a valid substring, not vice versa
        matches = [f for f in valid_feeds if f in self.feed_type]
        if len(matches) == 0:
            raise ValueError('Invalid external feed: %s' % self.feed_type)
        elif len(matches) > 1:
            raise ValueError('Ambiguous external feed: %s' % self.feed_type)

        # Each of these methods creates self.nrn_eventvec for playback
        # Return values not checked: False if all weights for given feed type
        # are zero. Designed to be silent so that zeroing input weights
        # effectively disables each.
        if self.feed_type == 'extpois':
            self._create_extpois()
        elif self.feed_type.startswith(('evprox', 'evdist')):
            self._create_evoked(inc_evinput)
        elif self.feed_type == 'extgauss':
            self._create_extgauss()
        elif self.feed_type == 'common':
            self._create_common_input()
        # load eventvec into VecStim object
        self.nrn_vecstim.play(self.nrn_eventvec)

    # based on cdf for exp wait time distribution from unif [0, 1)
    # returns in ms based on lamtha in Hz
    def _t_wait(self, lamtha):
        return -1000. * np.log(1. - self.prng.rand()) / lamtha

    # new external pois designation
    def _create_extpois(self):
        # print("_create_extpois")
        if self.p_ext[self.cell_type][0] <= 0.0 and \
                self.p_ext[self.cell_type][1] <= 0.0:
            return False  # 0 ampa and 0 nmda weight
        # check the t interval
        t0 = self.p_ext['t_interval'][0]
        T = self.p_ext['t_interval'][1]
        lamtha = self.p_ext[self.cell_type][3]  # index 3 is frequency (lamtha)
        # values MUST be sorted for VecStim()!
        # start the initial value
        if lamtha > 0.:
            t_gen = t0 + self._t_wait(lamtha)
            val_pois = np.array([])
            if t_gen < T:
                np.append(val_pois, t_gen)
            # vals are guaranteed to be monotonically increasing, no need to
            # sort
            while t_gen < T:
                # so as to not clobber confusingly base off of t_gen ...
                t_gen += self._t_wait(lamtha)
                if t_gen < T:
                    val_pois = np.append(val_pois, t_gen)
        else:
            val_pois = np.array([])
        # checks the distribution stats
        # if len(val_pois):
        #     xdiff = np.diff(val_pois/1000)
        #     print(lamtha, np.mean(xdiff), np.var(xdiff), 1/lamtha**2)
        # Convert array into nrn vector
        # if len(val_pois)>0: print('val_pois:',val_pois)
        self.nrn_eventvec.from_python(val_pois)
        return self.nrn_eventvec.size() > 0

    # mu and sigma vals come from p
    def _create_evoked(self, inc=0.0):
        if self.cell_type in self.p_ext.keys():
            # assign the params
            mu = self.p_ext['t0'] + inc
            sigma = self.p_ext[self.cell_type][3]  # index 3 is sigma_t (stdev)
            numspikes = int(self.p_ext['numspikes'])
            # print('mu:',mu,'sigma:',sigma,'inc:',inc)
            # if a non-zero sigma is specified
            if sigma:
                val_evoked = self.prng.normal(mu, sigma, numspikes)
            else:
                # if sigma is specified at 0
                val_evoked = np.array([mu] * numspikes)
            val_evoked = val_evoked[val_evoked > 0]
            # vals must be sorted
            val_evoked.sort()
            # print('_create_evoked val_evoked:',val_evoked)
            self.nrn_eventvec.from_python(val_evoked)
        else:
            # return an empty eventvec list
            self.nrn_eventvec.from_python([])
        return self.nrn_eventvec.size() > 0

    def _create_extgauss(self):
        # assign the params
        if self.p_ext[self.cell_type][0] <= 0.0 and \
                self.p_ext[self.cell_type][1] <= 0.0:
            return False  # 0 ampa and 0 nmda weight
        # print('gauss params:',self.p_ext[self.cell_type])
        mu = self.p_ext[self.cell_type][3]
        sigma = self.p_ext[self.cell_type][4]
        # mu and sigma values come from p
        # one single value from Gaussian dist.
        # values MUST be sorted for VecStim()!
        val_gauss = self.prng.normal(mu, sigma, 50)
        # val_gauss = np.random.normal(mu, sigma, 50)
        # remove non-zero values brute force-ly
        val_gauss = val_gauss[val_gauss > 0]
        # sort values - critical for nrn
        val_gauss.sort()
        # if len(val_gauss)>0: print('val_gauss:',val_gauss)
        # Convert array into nrn vector
        self.nrn_eventvec.from_python(val_gauss)
        return self.nrn_eventvec.size() > 0

    def _create_common_input(self):
        """Creates the common ongoing external inputs.

        Used for, e.g., for rhythmic inputs in alpha/beta generation
        """
        # print("_create_common_input")
        # Return if all synaptic weights are 0
        all_syn_weights_zero = True
        for key in self.p_ext.keys():
            if key.startswith('L2Pyr') or \
                    key.startswith('L5Pyr') or \
                    key.startswith('L2Bask') or \
                    key.startswith('L5Bask'):
                if self.p_ext[key][0] > 0.0:
                    all_syn_weights_zero = False
        if all_syn_weights_zero:
            return False

        # store f_input as self variable for later use if it exists in p
        # t0 is always defined
        t0 = self.p_ext['t0']
        # If t0 is -1, randomize start time of inputs
        if t0 == -1:
            t0 = self.prng.uniform(25., 125.)
        # randomize start time based on t0_stdev
        elif self.p_ext['t0_stdev'] > 0.0:
            # start time uses different prng
            t0 = self.prng2.normal(t0, self.p_ext['t0_stdev'])
        f_input = self.p_ext['f_input']
        stdev = self.p_ext['stdev']
        events_per_cycle = self.p_ext['events_per_cycle']
        distribution = self.p_ext['distribution']
        # events_per_cycle = 1
        if events_per_cycle > 2 or events_per_cycle <= 0:
            print("events_per_cycle should be either 1 or 2, trying 2")
            events_per_cycle = 2
        # If frequency is 0, create empty vector if input times
        if not f_input:
            t_input = []
        elif distribution == 'normal':
            # array of mean stimulus times, starts at t0
            isi_array = np.arange(t0, self.p_ext['tstop'], 1000. / f_input)
            # array of single stimulus times -- no doublets
            if stdev:
                t_array = self.prng.normal(
                    np.repeat(isi_array, self.p_ext['repeats']), stdev)
            else:
                t_array = isi_array
            if events_per_cycle == 2:  # spikes/burst in GUI
                # Two arrays store doublet times
                t_array_low = t_array - 5
                t_array_high = t_array + 5
                # Array with ALL stimulus times for input
                # np.append concatenates two np arrays
                t_input = np.append(t_array_low, t_array_high)
            elif events_per_cycle == 1:
                t_input = t_array
            # brute force remove zero times. Might result in fewer vals than
            # desired
            t_input = t_input[t_input > 0]
            t_input.sort()
        # Uniform Distribution
        elif distribution == 'uniform':
            n_inputs = self.p_ext['repeats'] * \
                f_input * (self.p_ext['tstop'] - t0) / 1000.
            t_array = self.prng.uniform(t0, self.p_ext['tstop'], n_inputs)
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
        else:
            print("Indicated distribution not recognized. "
                  "Not making any common feeds.")
            t_input = []
        # Convert array into nrn vector
        self.nrn_eventvec.from_python(t_input)
        return self.nrn_eventvec.size() > 0

    def connect_to_target(self, threshold):
        # print("connect_to_target")
        nc = h.NetCon(self.nrn_vecstim, None)  # why is target always None??
        nc.threshold = threshold
        return nc
