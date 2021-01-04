"""Network class."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>
#          Christopher Bailey <cjb@cfin.au.dk>

import itertools as it
import numpy as np
from glob import glob

from .feed import drive_cell_event_times
from .params import create_pext
from .network_builder import _short_name  # XXX placement?!
from .viz import plot_spikes_hist, plot_spikes_raster, plot_cells


def read_spikes(fname, gid_ranges=None):
    """Read spiking activity from a collection of spike trial files.

    Parameters
    ----------
    fname : str
        Wildcard expression (e.g., '<pathname>/spk_*.txt') of the
        path to the spike file(s).
    gid_ranges : dict of lists or range objects | None
        Dictionary with keys 'evprox1', 'evdist1' etc.
        containing the range of Cell or input IDs of different
        cell or input types. If None, each spike file must contain
        a 3rd column for spike type.

    Returns
    ----------
    cell_response : CellResponse
        An instance of the CellResponse object.
    """

    spike_times = []
    spike_gids = []
    spike_types = []
    for file in sorted(glob(str(fname))):
        spike_trial = np.loadtxt(file, dtype=str)
        spike_times += [list(spike_trial[:, 0].astype(float))]
        spike_gids += [list(spike_trial[:, 1].astype(int))]

        # Note that legacy HNN 'spk.txt' files don't contain a 3rd column for
        # spike type. If reading a legacy version, validate that a gid_ranges
        # is provided.
        if spike_trial.shape[1] == 3:
            spike_types += [list(spike_trial[:, 2].astype(str))]
        else:
            if gid_ranges is None:
                raise ValueError("gid_ranges must be provided if spike types "
                                 "are unspecified in the file %s" % (file,))
            spike_types += [[]]

    cell_response = CellResponse(spike_times=spike_times,
                                 spike_gids=spike_gids,
                                 spike_types=spike_types)
    if gid_ranges is not None:
        cell_response.update_types(gid_ranges)

    return CellResponse(spike_times=spike_times, spike_gids=spike_gids,
                        spike_types=spike_types)


def _create_cell_coords(n_pyr_x, n_pyr_y, zdiff=1307.4):
    """Creates coordinate grid and place cells in it.

    Parameters
    ----------
    n_pyr_x : int
        The number of Pyramidal cells in x direction.
    n_pyr_y : int
        The number of Pyramidal cells in y direction.

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

    # ORIGIN
    # origin's z component isn't really used in
    # calculating distance functions from origin
    # these will be forced as ints!
    origin_x = xxrange[int((len(xxrange) - 1) // 2)]
    origin_y = yyrange[int((len(yyrange) - 1) // 2)]
    origin_z = np.floor(zdiff / 2)
    origin = (origin_x, origin_y, origin_z)

    # save the origin for adding external drives later
    pos_dict['origin'] = origin

    return pos_dict


class Network(object):
    """The Network class.

    Parameters
    ----------
    params : dict
        The parameters to use for constructing the network.
    initialise_hnn_drives : bool
        If True (default), attach and instantiate drives as in HNN-GUI

    Attributes
    ----------
    params : dict
        The parameters of the network
    cellname_list : list
        The names of real cell types in the network (e.g. 'L2_basket')
    gid_ranges : dict
        A dictionary of unique identifiers of each real and artificial cell
        in the network. Every cell type is represented by a key read from
        cellname_list, followed by keys read from external_drives. The value
        of each key is a range of ints, one for each cell in given category.
        Examples: 'L2_basket': range(0, 270), 'evdist1': range(272, 542), etc
    pos_dict : dict
        Dictionary containing the coordinate positions of all cells.
        Keys are 'L2_pyramidal', 'L5_pyramidal', 'L2_basket', 'L5_basket',
        or any external drive name
    cell_response : CellResponse
        An instance of the CellResponse object.
    external_drives : dict of list
        The parameters of external driving inputs to the network.
    external_biases : dict of list
        The parameters of bias inputs to cell somata, e.g., tonic current clamp
    drive_event_times : dict of list (trials) of list (cells) of list (times)
        The event times of input drives (empty before initialised)
    """

    def __init__(self, params, initialise_hnn_drives=True):
        # Save the parameters used to create the Network
        self.params = params
        # Initialise a dictionary of cell ID's, which get used when the
        # network is constructed ('built') in NetworkBuilder
        # We want it to remain in each Network object, so that the user can
        # interrogate a built and simulated net. In addition, CellResponse is
        # attached to a Network during simulation---Network is the natural
        # place to keep this information
        self.gid_ranges = dict()
        self._n_gids = 0  # utility: keep track of last GID
        self._legacy_mode = initialise_hnn_drives  # for GID seeding

        # Create array of equally sampled time points for simulating currents
        # NB (only) used to initialise self.cell_response._times
        times = np.arange(0., params['tstop'] + params['dt'], params['dt'])
        # Create CellResponse object, initialised with simulation time points
        self.cell_response = CellResponse(times=times)

        # Source list of names, first real ones only!
        self.cellname_list = [
            'L2_basket',
            'L2_pyramidal',
            'L5_basket',
            'L5_pyramidal',
        ]

        # external drives and biases
        self.external_drives = dict()
        self.external_biases = dict()
        self._cell_gid_to_drive = dict()  # reverse-lookup table for Builder

        # this is needed network_builder.py:_gid_assign() because there's no
        # way to keep track of which gids are already assigned to other nodes;
        # without it, we can't round-robin assign the non-cell specfiic drives
        self._global_drive_gids = []

        # contents of pos_dict determines all downstream inferences of
        # cell counts, real and artificial
        self.pos_dict = _create_cell_coords(n_pyr_x=self.params['N_pyr_x'],
                                            n_pyr_y=self.params['N_pyr_y'],
                                            zdiff=1307.4)
        # Every time pos_dict is updated, gid_ranges must be updated too
        self._update_gid_ranges()

        # set n_cells, EXCLUDING Artificial ones
        self.n_cells = sum(len(self.pos_dict[src]) for src in
                           self.cellname_list)

        if initialise_hnn_drives:
            drive_specs = self._extract_drive_specs_from_hnn_params()
            bias_specs = self._extract_bias_specs_from_hnn_params()

            for drive_name in sorted(drive_specs.keys()):  # order matters
                specs = drive_specs[drive_name]
                if specs['type'] == 'evoked':
                    self.add_evoked_drive(
                        drive_name, specs['dynamics']['mu'],
                        specs['dynamics']['sigma'],
                        specs['dynamics']['numspikes'],
                        specs['weights_ampa'], specs['weights_nmda'],
                        specs['location'], seedcore=specs['seedcore'],
                        space_constant=specs['space_constant'])
                elif specs['type'] == 'poisson':
                    self.add_poisson_drive(
                        drive_name, specs['dynamics']['t0'],
                        specs['dynamics']['T'],
                        specs['dynamics']['rate_constants'],
                        specs['weights_ampa'], specs['weights_nmda'],
                        specs['location'], seedcore=specs['seedcore'],
                        space_constant=specs['space_constant'])
                elif specs['type'] == 'gaussian':
                    self.add_gaussian_drive(
                        drive_name, specs['dynamics']['mu'],
                        specs['dynamics']['sigma'],
                        specs['dynamics']['numspikes'],
                        specs['weights_ampa'], specs['weights_nmda'],
                        specs['location'], seedcore=specs['seedcore'],
                        space_constant=specs['space_constant'])
                elif specs['type'] == 'bursty':
                    self.add_bursty_drive(
                        drive_name, specs['dynamics']['distribution'],
                        specs['dynamics']['t0'], specs['dynamics']['sigma_t0'],
                        specs['dynamics']['T'], specs['dynamics']['burst_f'],
                        specs['dynamics']['burst_sigma_f'],
                        specs['dynamics']['numspikes'],
                        specs['dynamics']['repeats'],
                        specs['weights_ampa'], specs['weights_nmda'],
                        specs['synaptic_delays'], specs['location'],
                        space_constant=specs['space_constant'],
                        seedcore=specs['seedcore'])

            # add tonic biases if present in params
            for cellname in bias_specs['tonic']:
                self.add_tonic_bias(
                    cell_type=cellname,
                    amplitude=bias_specs['tonic'][cellname]['amplitude'],
                    t0=bias_specs['tonic'][cellname]['t0'],
                    T=bias_specs['tonic'][cellname]['T'])

            self.instantiate_drives(n_trials=self.params['N_trials'])

    def __repr__(self):
        class_name = self.__class__.__name__
        s = ("%d x %d Pyramidal cells (L2, L5)"
             % (self.params['N_pyr_x'], self.params['N_pyr_y']))
        s += ("\n%d L2 basket cells\n%d L5 basket cells"
              % (len(self.pos_dict['L2_basket']),
                 len(self.pos_dict['L5_basket'])))
        return '<%s | %s>' % (class_name, s)

    def _create_drive_cells(self, target_populations, weights_by_receptor,
                            location, space_constant, synaptic_delays,
                            cell_specific=True):
        if isinstance(synaptic_delays, dict):
            for receptor in ['ampa', 'nmda']:
                if not (
                    set(list(weights_by_receptor[receptor].keys())).issubset(
                        set(list(synaptic_delays[receptor].keys())))):
                    raise ValueError(
                        'synaptic_delays is either common (float), or needs '
                        'to be specified for each non-zero AMPA/NMDA weight')

        next_gid = self._n_gids
        drive_gids = []
        drive_cells = []
        if cell_specific:
            for cellname in target_populations:
                for target_gid in self.gid_ranges[cellname]:
                    drive_cell = dict()

                    drive_cell['target_gid'] = target_gid
                    # NB list! This is used later in _parnet_connect
                    drive_cell['target_types'] = [cellname]
                    drive_cell['gid'] = next_gid
                    next_gid += 1
                    drive_cell['location'] = location

                    # reverse-lookup: real_cell-to-drive_cell
                    if f'{target_gid}' not in self._cell_gid_to_drive:
                        self._cell_gid_to_drive[f'{target_gid}'] = []
                    self._cell_gid_to_drive[
                        f'{target_gid}'].append(drive_cell['gid'])

                    for receptor, weights in weights_by_receptor.items():
                        drive_cell[receptor] = dict()
                        if cellname in weights:
                            drive_cell[receptor]['lamtha'] = space_constant
                            drive_cell[receptor]['A_delay'] = synaptic_delays
                            drive_cell[receptor][
                                'A_weight'] = weights[cellname]

                    drive_gids += [drive_cell['gid']]  # for convenience
                    drive_cells.append(drive_cell)
        else:
            drive_cell = dict()

            # NB list! This is used later in _parnet_connect
            drive_cell['target_types'] = target_populations  # all cells
            drive_cell['gid'] = next_gid
            next_gid += 1
            drive_cell['location'] = location
            for cellname in target_populations:
                for target_gid in self.gid_ranges[cellname]:
                    # reverse-lookup: real_cell-to-drive_cell
                    if f'{target_gid}' not in self._cell_gid_to_drive:
                        self._cell_gid_to_drive[f'{target_gid}'] = []
                    self._cell_gid_to_drive[
                        f'{target_gid}'].append(drive_cell['gid'])
                    for receptor, weights in weights_by_receptor.items():
                        drive_cell[receptor] = dict()

                        if cellname in weights:
                            drive_cell[receptor]['lamtha'] = space_constant
                            drive_cell[receptor]['A_delay'] = synaptic_delays[
                                receptor][cellname]
                            drive_cell[receptor][
                                'A_weight'] = weights[cellname]
            drive_gids += [drive_cell['gid']]  # for convenience
            drive_cells.append(drive_cell)

        return(drive_cells, drive_gids)

    def add_evoked_drive(self, name, mu, sigma, numspikes, weights_ampa,
                         weights_nmda, location, space_constant=3.,
                         seedcore=None):
        """
        """
        # CHECK name not exists!
        drive = NetworkDrive()
        drive['type'] = 'evoked'
        drive['cell_specific'] = True
        drive['seedcore'] = seedcore

        drive['dynamics'] = dict(mu=mu, sigma=sigma, numspikes=numspikes)
        drive['events'] = []

        self._attach_drive(name, drive, weights_ampa, weights_nmda, location,
                           space_constant)

    def add_poisson_drive(self, name, t0, T, rate_constants, weights_ampa,
                          weights_nmda, location, space_constant=100.,
                          seedcore=None):
        """
        """
        # CHECK name not exists!
        # CHECK rate_constants == dict

        drive = NetworkDrive()
        drive['type'] = 'poisson'
        drive['cell_specific'] = True
        drive['seedcore'] = seedcore

        drive['dynamics'] = dict(t0=t0, T=T, rate_constants=rate_constants)
        drive['events'] = []
        self._attach_drive(name, drive, weights_ampa, weights_nmda, location,
                           space_constant)

    def add_gaussian_drive(self, name, mu, sigma, numspikes, weights_ampa,
                           weights_nmda, location, space_constant=100.,
                           seedcore=None):
        """
        """
        # CHECK name not exists!

        drive = NetworkDrive()
        drive['type'] = 'gaussian'
        drive['cell_specific'] = True
        drive['seedcore'] = seedcore

        drive['dynamics'] = dict(mu=mu, sigma=sigma, numspikes=numspikes)
        drive['events'] = []

        self._attach_drive(name, drive, weights_ampa, weights_nmda, location,
                           space_constant)

    def add_bursty_drive(self, name, distribution, t0, sigma_t0, T, burst_f,
                         burst_sigma_f, numspikes, repeats, weights_ampa,
                         weights_nmda, synaptic_delays, location,
                         space_constant=100., seedcore=None):
        """
        """
        # CHECK name not exists!

        drive = NetworkDrive()
        drive['type'] = 'bursty'
        drive['cell_specific'] = False
        drive['seedcore'] = seedcore

        drive['dynamics'] = dict(distribution=distribution, t0=t0,
                                 sigma_t0=sigma_t0, T=T, burst_f=burst_f,
                                 burst_sigma_f=burst_sigma_f,
                                 numspikes=numspikes, repeats=repeats
                                 )
        drive['events'] = []

        self._attach_drive(name, drive, weights_ampa, weights_nmda, location,
                           space_constant, cell_specific=False,
                           synaptic_delays=synaptic_delays)

    def _attach_drive(self, name, drive, weights_ampa, weights_nmda, location,
                      space_constant, synaptic_delays=0.1, cell_specific=True):
        # CHECKS

        target_populations = (set(weights_ampa.keys()) |
                              set(weights_nmda.keys()))
        if not target_populations.issubset(set(self.cellname_list)):
            raise ValueError('Allowed target cell types are: ',
                             f'{self.cellname_list}')
        if self._legacy_mode:
            target_populations = self.cellname_list

        weights_by_receptor = {'ampa': weights_ampa, 'nmda': weights_nmda}

        drive['cells'], drive['gids'] = self._create_drive_cells(
            target_populations, weights_by_receptor, location,
            space_constant, synaptic_delays, cell_specific=cell_specific)

        self.external_drives[name] = drive
        if not cell_specific:
            self._global_drive_gids.extend(drive['gids'])  # for Builder

        # Must remember to update the GID ranges based on pos_dict!
        self.pos_dict[name] = [self.pos_dict['origin'] for dg in drive['gids']]
        # Every time pos_dict is updated, gid_ranges must be updated too
        self._update_gid_ranges()

    def instantiate_drives(self, n_trials=1):
        """Creates drive_event_times vectors for all drives and all trials

        Parameters
        ----------
        n_trials : int
            Number of trials to create events for (default: 1)

        NB this must be a separate method because dipole.py:simulate_dipole
        accepts an n_trials-argument, which overrides the N_trials-parameter
        used at intialisation time. The good news is that only the event_times
        need to be recalculated, all the GIDs etc remain the same.
        """
        # reset every time called again, e.g., from dipole.py
        for drive_name in self.external_drives.keys():
            self.external_drives[drive_name]['events'] = []

        # each trial needs unique event time vectors
        for trial_idx in range(n_trials):
            for drive_name in self.external_drives.keys():
                event_times = list()  # new list for each trial and drive

                dyn_specs = self.external_drives[drive_name]['dynamics']
                drive_type = self.external_drives[drive_name]['type']
                seedcore = self.external_drives[drive_name]['seedcore']

                # loop over drive 'cells' and create event times for each
                for drive_cell in self.external_drives[drive_name]['cells']:
                    event_times.append(
                        drive_cell_event_times(drive_type, drive_cell,
                                               dyn_specs, trial_idx=trial_idx,
                                               seedcore=seedcore)
                    )
                # 'events': list (trials) of list (cells) of list (events)
                self.external_drives[
                    drive_name]['events'].append(event_times)

    def add_tonic_bias(self, cell_type, amplitude, t0, T):
        """Attach parameters of tonic biasing input for a given cell type.

        Parameters
        ----------
        cell_type : str
            The cell type whose cells will get the tonic input.
            Valid inputs are those in `net.cellname_list`.
        amplitude : float
            The amplitude of the input.
        t0 : float
            The start time of tonic input (in ms).
        T : float
            The end time of tonic input (in ms).
        """
        if T < 0.:
            raise ValueError('End time of tonic input cannot be negative')
        tstop = self.cell_response.times[-1]
        if T > tstop:
            raise ValueError(f'End time of tonic input cannot exceed '
                             f'simulation end time {tstop}. Got {T}.')
        if cell_type not in self.cellname_list:
            raise ValueError(f'cell_type must be one of {self.cellname_list}'
                             f'. Got {cell_type}')
        duration = T - t0
        if duration < 0.:
            raise ValueError('Duration of tonic input cannot be negative')

        self.external_biases['tonic'][cell_type] = {
            'amplitude': amplitude,
            't0': t0,
            'T': T
        }

    def _update_gid_ranges(self):
        """Creates gid ranges from scratch every time called.

        Any method that adds real or artificial cells to the network must
        call this to update the list of GIDs. Note that it's based on the
        content of pos_dict and the lists of real and artificial cell names.
        """
        # if external drives dict is empty, list will also be empty
        ext_drive_names = list(self.external_drives.keys())
        gid_lims = [0]  # start and end gids per cell type
        src_types = self.cellname_list + ext_drive_names
        for idx, src_type in enumerate(src_types):
            n_srcs = len(self.pos_dict[src_type])
            gid_lims.append(gid_lims[idx] + n_srcs)
            self.gid_ranges[src_type] = range(gid_lims[idx],
                                              gid_lims[idx + 1])
        self._n_gids = gid_lims[idx + 1]

    def gid_to_type(self, gid):
        """Reverse lookup of gid to type."""
        for gidtype, gids in self.gid_ranges.items():
            if gid in gids:
                return gidtype

    def _get_src_type_and_pos(self, gid):
        """Source type, position and whether it's a cell or artificial feed"""

        # get type of cell and pos via gid
        src_type = self.gid_to_type(gid)
        type_pos_ind = gid - self.gid_ranges[src_type][0]
        src_pos = self.pos_dict[src_type][type_pos_ind]

        return src_type, src_pos, src_type in self.cellname_list

    def _extract_bias_specs_from_hnn_params(self):
        """Create 'bias specification' dicts from saved parameters
        """
        bias_specs = {'tonic': {}}  # currently only 'tonic' biases known
        for cellname in self.cellname_list:
            short_name = _short_name(cellname)
            is_tonic_present = [f'Itonic_{p}_{short_name}_soma' in
                                self.params for p in ['A', 't0', 'T']]
            if any(is_tonic_present):
                if not all(is_tonic_present):
                    raise ValueError(
                        f'Tonic input must have the amplitude, '
                        f'start time and end time specified. One '
                        f'or more parameter may be missing for '
                        f'cell type {cellname}')
                bias_specs['tonic'][cellname] = {
                    'amplitude': self.params[f'Itonic_A_{short_name}_soma'],
                    't0': self.params[f'Itonic_t0_{short_name}_soma'],
                    'T': self.params[f'Itonic_T_{short_name}_soma']
                }
        return(bias_specs)

    def _extract_drive_specs_from_hnn_params(self):
        """Create 'drive specification' dicts from saved parameters
        """
        # convert legacy params-dict to legacy "feeds" dicts
        p_common, p_unique = create_pext(self.params, self.params['tstop'])

        # Using 'feed' for legacy compatibility, 'drives' for new API
        drive_specs = {}
        for ic, par in enumerate(p_common):
            feed_name = f'bursty{ic + 1}'
            drive = dict()
            drive['type'] = 'bursty'
            drive['cell_specific'] = False
            drive['dynamics'] = {'distribution': par['distribution'],
                                 't0': par['t0'],
                                 'sigma_t0': par['t0_stdev'],
                                 'T': par['tstop'],
                                 'burst_f': par['f_input'],
                                 'burst_sigma_f': par['stdev'],
                                 'numspikes': par['events_per_cycle'],
                                 'repeats': par['repeats']}
            drive['location'] = par['loc']
            drive['space_constant'] = par['lamtha']
            drive['seedcore'] = par['prng_seedcore']
            drive['weights_ampa'] = {}
            drive['weights_nmda'] = {}
            drive['synaptic_delays'] = {'ampa': {}, 'nmda': {}}

            for cellname in self.cellname_list:
                cname_ampa = _short_name(cellname) + '_ampa'
                cname_nmda = _short_name(cellname) + '_nmda'
                if cname_ampa in par:
                    ampa_w = par[cname_ampa][0]
                    ampa_d = par[cname_ampa][1]
                    if ampa_w > 0.:
                        drive['weights_ampa'][cellname] = ampa_w
                    if ampa_d > 0.:
                        drive['synaptic_delays']['ampa'][cellname] = ampa_d
                if cname_nmda in par:
                    nmda_w = par[cname_nmda][0]
                    nmda_d = par[cname_nmda][1]
                    if nmda_w > 0.:
                        drive['weights_nmda'][cellname] = nmda_w
                    if nmda_d > 0.:
                        drive['synaptic_delays']['nmda'][cellname] = nmda_d

            drive_specs[feed_name] = drive

        for feed_name, par in p_unique.items():
            drive = dict()
            drive['cell_specific'] = True
            if (feed_name.startswith('evprox') or
                    feed_name.startswith('evdist')):
                drive['type'] = 'evoked'
                if feed_name.startswith('evprox'):
                    drive['location'] = 'proximal'
                else:
                    drive['location'] = 'distal'

                if par['sync_evinput']:
                    sigma = 0.
                else:
                    sigma = par['L2_basket'][3]  # NB IID for all cells!

                drive['dynamics'] = {'mu': par['t0'],
                                     'sigma': sigma,
                                     'numspikes': par['numspikes']}
                drive['space_constant'] = par['lamtha']
                drive['seedcore'] = par['prng_seedcore']
                drive['weights_ampa'] = {}
                drive['weights_nmda'] = {}
                for cellname in self.cellname_list:
                    if cellname in par:
                        ampa_w = par[cellname][0]
                        nmda_w = par[cellname][1]
                        if ampa_w > 0.:
                            drive['weights_ampa'][cellname] = ampa_w
                        if nmda_w > 0.:
                            drive['weights_nmda'][cellname] = nmda_w
            elif feed_name.startswith('extgauss'):
                drive['type'] = 'gaussian'
                drive['location'] = par['loc']

                drive['dynamics'] = {'mu': par['L2_basket'][3],  # NB IID
                                     'sigma': par['L2_basket'][4],
                                     'numspikes': 50}  # NB hard-coded in GUI!
                drive['space_constant'] = par['lamtha']
                drive['seedcore'] = par['prng_seedcore']
                drive['weights_ampa'] = {}
                drive['weights_nmda'] = {}
                for cellname in self.cellname_list:
                    if cellname in par:
                        ampa_w = par[cellname][0]
                        if ampa_w > 0.:
                            drive['weights_ampa'][cellname] = ampa_w
                drive['weights_nmda'] = {}  # no NMDA weights for Gaussians
            elif feed_name.startswith('extpois'):
                drive['type'] = 'poisson'
                drive['location'] = par['loc']

                rate_params = {}
                for cellname in self.cellname_list:
                    if cellname in par:
                        rate_params[cellname] = par[cellname][3]
                        ampa_w = par[cellname][0]
                        nmda_w = par[cellname][1]
                        if ampa_w > 0.:
                            drive['weights_ampa'][cellname] = ampa_w
                        if nmda_w > 0.:
                            drive['weights_nmda'][cellname] = nmda_w

                drive['dynamics'] = {'t0': par['t_interval'][0],
                                     'T': par['t_interval'][1],
                                     'rate_constants': rate_params}
                drive['space_constant'] = par['lamtha']
                drive['seedcore'] = par['prng_seedcore']
                drive['weights_ampa'] = {}
                drive['weights_nmda'] = {}
                drive['weights_nmda'] = {}  # no NMDA weights for Gaussians
            drive_specs[feed_name] = drive
        return(drive_specs)

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


class NetworkDrive(dict):
    """Foo
    """

    def __repr__(self):
        entr = '<NetworkDrive'
        if 'type' in self.keys():
            entr += f" of type {self['type']}"
        entr += '>'
        return(entr)


class CellResponse(object):
    """The CellResponse class.

    Parameters
    ----------
    spike_times : list (n_trials,) of list (n_spikes,) of float, shape | None
        Each element of the outer list is a trial.
        The inner list contains the time stamps of spikes.
    spike_gids : list (n_trials,) of list (n_spikes,) of float, shape | None
        Each element of the outer list is a trial.
        The inner list contains the cell IDs of neurons that
        spiked.
    spike_types : list (n_trials,) of list (n_spikes,) of float, shape | None
        Each element of the outer list is a trial.
        The inner list contains the type of spike (e.g., evprox1
        or L2_pyramidal) that occured at the corresonding time stamp.
        Each gid corresponds to a type via Network().gid_ranges.

    Attributes
    ----------
    spike_times : list (n_trials,) of list (n_spikes,) of float, shape
        Each element of the outer list is a trial.
        The inner list contains the time stamps of spikes.
    spike_gids : list (n_trials,) of list (n_spikes,) of float, shape
        Each element of the outer list is a trial.
        The inner list contains the cell IDs of neurons that
        spiked.
    spike_types : list (n_trials,) of list (n_spikes,) of float, shape
        Each element of the outer list is a trial.
        The inner list contains the type of spike (e.g., evprox1
        or L2_pyramidal) that occured at the corresonding time stamp.
        Each gid corresponds to a type via Network::gid_ranges.
    vsoma : list (n_trials,) of dict, shape
        Each element of the outer list is a trial.
        Dictionary indexed by gids containing somatic voltages.
    isoma : list (n_trials,) of dict, shape
        Each element of the outer list is a trial.
        Dictionary indexed by gids containing somatic currents.

    times : numpy array
        Array of time points for samples in continuous data.
        This includes vsoma and isoma.

    Methods
    -------
    update_types(gid_ranges)
        Update spike types in the current instance of CellResponse.
    plot(ax=None, show=True)
        Plot and return a matplotlib Figure object showing the
        aggregate network spiking activity according to cell type.
    mean_rates(tstart, tstop, gid_ranges, mean_type='all')
        Calculate mean firing rate for each cell type. Specify
        averaging method with mean_type argument.
    write(fname)
        Write spiking activity to a collection of spike trial files.
    """

    def __init__(self, spike_times=None, spike_gids=None, spike_types=None,
                 times=None):
        if spike_times is None:
            spike_times = list()
        if spike_gids is None:
            spike_gids = list()
        if spike_types is None:
            spike_types = list()

        # Validate arguments
        arg_names = ['spike_times', 'spike_gids', 'spike_types']
        for arg_idx, arg in enumerate([spike_times, spike_gids, spike_types]):
            # Validate outer list
            if not isinstance(arg, list):
                raise TypeError('%s should be a list of lists'
                                % (arg_names[arg_idx],))
            # If arg is not an empty list, validate inner list
            for trial_list in arg:
                if not isinstance(trial_list, list):
                    raise TypeError('%s should be a list of lists'
                                    % (arg_names[arg_idx],))
            # Set the length of 'spike_times' as a references and validate
            # uniform length
            if arg == spike_times:
                n_trials = len(spike_times)
            if len(arg) != n_trials:
                raise ValueError('spike times, gids, and types should be '
                                 'lists of the same length')
        self._spike_times = spike_times
        self._spike_gids = spike_gids
        self._spike_types = spike_types
        self._vsoma = list()
        self._isoma = list()
        if times is not None:
            if not isinstance(times, np.ndarray):
                raise TypeError("'times' is an np.ndarray of simulation times")
        self._times = times

    def __repr__(self):
        class_name = self.__class__.__name__
        n_trials = len(self._spike_times)
        return '<%s | %d simulation trials>' % (class_name, n_trials)

    def __eq__(self, other):
        if not isinstance(other, CellResponse):
            return NotImplemented
        # Round each time element
        times_self = [[round(time, 3) for time in trial]
                      for trial in self._spike_times]
        times_other = [[round(time, 3) for time in trial]
                       for trial in other._spike_times]
        return (times_self == times_other and
                self._spike_gids == other._spike_gids and
                self._spike_types == other._spike_types)

    def __getitem__(self, gid_item):
        """Returns a CellResponse object with a copied subset filtered by gid.

        Parameters
        ----------
        gid_item : int | slice
            Subset of gids .

        Returns
        -------
        cell_response : instance of CellResponse
            See below for use cases.
        """

        if isinstance(gid_item, slice):
            gid_item = np.arange(gid_item.stop)[gid_item]
        elif isinstance(gid_item, list):
            gid_item = np.array(gid_item)
        elif isinstance(gid_item, np.ndarray):
            if gid_item.ndim > 1:
                raise ValueError("ndarray cannot exceed 1 dimension")
            else:
                pass
        elif isinstance(gid_item, int):
            gid_item = np.array([gid_item])
        else:
            raise TypeError("indices must be int, slice, or array-like, "
                            f"not {type(gid_item).__name__}")

        if not np.issubdtype(gid_item.dtype, np.integer):
            raise TypeError("gids must be of dtype int, "
                            f"not {gid_item.dtype.name}")

        n_trials = len(self._spike_times)
        times_slice = []
        gids_slice = []
        types_slice = []
        vsoma_slice = []
        isoma_slice = []
        for trial_idx in range(n_trials):
            gid_mask = np.in1d(self._spike_gids[trial_idx], gid_item)
            times_trial = np.array(
                self._spike_times[trial_idx])[gid_mask].tolist()
            gids_trial = np.array(
                self._spike_gids[trial_idx])[gid_mask].tolist()
            types_trial = np.array(
                self._spike_types[trial_idx])[gid_mask].tolist()

            vsoma_trial = {gid: self._vsoma[trial_idx][gid] for gid in gid_item
                           if gid in self._vsoma[trial_idx].keys()}

            isoma_trial = {gid: self._isoma[trial_idx][gid] for gid in gid_item
                           if gid in self._isoma[trial_idx].keys()}

            times_slice.append(times_trial)
            gids_slice.append(gids_trial)
            types_slice.append(types_trial)
            vsoma_slice.append(vsoma_trial)
            isoma_slice.append(isoma_trial)

        cell_response_slice = CellResponse(spike_times=times_slice,
                                           spike_gids=gids_slice,
                                           spike_types=types_slice)
        cell_response_slice._vsoma = vsoma_slice
        cell_response_slice._isoma = isoma_slice

        return cell_response_slice

    @property
    def spike_times(self):
        return self._spike_times

    @property
    def spike_gids(self):
        return self._spike_gids

    @property
    def spike_types(self):
        return self._spike_types

    @property
    def vsoma(self):
        return self._vsoma

    @property
    def isoma(self):
        return self._isoma

    @property
    def times(self):
        return self._times

    def update_types(self, gid_ranges):
        """Update spike types in the current instance of CellResponse.

        Parameters
        ----------
        gid_ranges : dict of lists or range objects
            Dictionary with keys 'evprox1', 'evdist1' etc.
            containing the range of Cell or input IDs of different
            cell or input types.
        """

        # Validate gid_ranges
        all_gid_ranges = list(gid_ranges.values())
        for item_idx_1 in range(len(all_gid_ranges)):
            for item_idx_2 in range(item_idx_1 + 1, len(all_gid_ranges)):
                gid_set_1 = set(all_gid_ranges[item_idx_1])
                gid_set_2 = set(all_gid_ranges[item_idx_2])
                if not gid_set_1.isdisjoint(gid_set_2):
                    raise ValueError('gid_ranges should contain only disjoint '
                                     'sets of gid values')

        spike_types = list()
        for trial_idx in range(len(self._spike_times)):
            spike_types_trial = np.empty_like(self._spike_times[trial_idx],
                                              dtype='<U36')
            for gidtype, gids in gid_ranges.items():
                spike_gids_mask = np.in1d(self._spike_gids[trial_idx], gids)
                spike_types_trial[spike_gids_mask] = gidtype
            spike_types += [list(spike_types_trial)]
        self._spike_types = spike_types

    def mean_rates(self, tstart, tstop, gid_ranges, mean_type='all'):
        """Mean spike rates (Hz) by cell type.

        Parameters
        ----------
        tstart : int | float | None
            Value defining the start time of all trials.
        tstop : int | float | None
            Value defining the stop time of all trials.
        gid_ranges : dict of lists or range objects
            Dictionary with keys 'evprox1', 'evdist1' etc.
            containing the range of Cell or input IDs of different
            cell or input types.
        mean_type : str
            'all' : Average over trials and cells
                Returns mean rate for cell types
            'trial' : Average over cell types
                Returns trial mean rate for cell types
            'cell' : Average over individual cells
                Returns trial mean rate for individual cells

        Returns
        -------
        spike_rate : dict
            Dictionary with keys 'L5_pyramidal', 'L5_basket', etc.
        """
        cell_types = ['L5_pyramidal', 'L5_basket', 'L2_pyramidal', 'L2_basket']
        spike_rates = dict()

        if mean_type not in ['all', 'trial', 'cell']:
            raise ValueError("Invalid mean_type. Valid arguments include "
                             f"'all', 'trial', or 'cell'. Got {mean_type}")

        # Validate tstart, tstop
        if not isinstance(tstart, (int, float)) or not isinstance(
                tstop, (int, float)):
            raise ValueError('tstart and tstop must be of type int or float')
        elif tstop <= tstart:
            raise ValueError('tstop must be greater than tstart')

        for cell_type in cell_types:
            cell_type_gids = np.array(gid_ranges[cell_type])
            n_trials, n_cells = len(self._spike_times), len(cell_type_gids)
            gid_spike_rate = np.zeros((n_trials, n_cells))

            trial_data = zip(self._spike_types, self._spike_gids)
            for trial_idx, (spike_types, spike_gids) in enumerate(trial_data):
                trial_type_mask = np.in1d(spike_types, cell_type)
                gids, gid_counts = np.unique(np.array(
                    spike_gids)[trial_type_mask], return_counts=True)

                gid_spike_rate[trial_idx, np.in1d(cell_type_gids, gids)] = (
                    gid_counts / (tstop - tstart)) * 1000

            if mean_type == 'all':
                spike_rates[cell_type] = np.mean(
                    gid_spike_rate.mean(axis=1))
            if mean_type == 'trial':
                spike_rates[cell_type] = np.mean(
                    gid_spike_rate, axis=1).tolist()
            if mean_type == 'cell':
                spike_rates[cell_type] = [gid_trial_rate.tolist()
                                          for gid_trial_rate in gid_spike_rate]

        return spike_rates

    def plot_spikes_raster(self, ax=None, show=True):
        """Plot the aggregate spiking activity according to cell type.

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
        return plot_spikes_raster(cell_response=self, ax=ax, show=show)

    def plot_spikes_hist(self, ax=None, spike_types=None, show=True):
        """Plot the histogram of spiking activity across trials.

        Parameters
        ----------
        ax : instance of matplotlib axis | None
            An axis object from matplotlib. If None,
            a new figure is created.
        spike_types: string | list | dictionary | None
            String input of a valid spike type is plotted individually.
                Ex: 'common', 'evdist', 'evprox', 'extgauss', 'extpois'
            List of valid string inputs will plot each spike type individually.
                Ex: ['common', 'evdist']
            Dictionary of valid lists will plot list elements as a group.
                Ex: {'Evoked': ['evdist', 'evprox'], 'External': ['extpois']}
            If None, all input spike types are plotted individually if any
            are present. Otherwise spikes from all cells are plotted.
            Valid strings also include leading characters of spike types
                Example: 'ext' is equivalent to ['extgauss', 'extpois']
        show : bool
            If True, show the figure.

        Returns
        -------
        fig : instance of matplotlib Figure
            The matplotlib figure handle.
        """
        return plot_spikes_hist(
            self, ax=ax, spike_types=spike_types, show=show)

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
            3) gid type
        """

        for trial_idx in range(len(self._spike_times)):
            with open(str(fname) % (trial_idx,), 'w') as f:
                for spike_idx in range(len(self._spike_times[trial_idx])):
                    f.write('{:.3f}\t{}\t{}\n'.format(
                        self._spike_times[trial_idx][spike_idx],
                        int(self._spike_gids[trial_idx][spike_idx]),
                        self._spike_types[trial_idx][spike_idx]))
