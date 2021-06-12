"""Network class."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>
#          Christopher Bailey <cjb@cfin.au.dk>
#          Nick Tolley <nicholas_tolley@brown.edu>

import itertools as it
import numpy as np
from copy import deepcopy
from warnings import warn

from .drives import _drive_cell_event_times
from .drives import _get_target_populations, _add_drives_from_params
from .drives import _check_drive_parameter_values, _check_poisson_rates
from .cells_default import pyramidal, basket
from .cell_response import CellResponse
from .params import _long_name, _short_name
from .viz import plot_cells
from .externals.mne import _validate_type, _check_option


def _create_cell_coords(n_pyr_x, n_pyr_y, zdiff=1307.4):
    """Creates coordinate grid and place cells in it.

    Parameters
    ----------
    n_pyr_x : int
        The number of Pyramidal cells in x direction.
    n_pyr_y : int
        The number of Pyramidal cells in y direction.

    zdiff : float
        Expressed as a positive DEPTH of L2 relative to L5, where L5 is defined
        to lie at z==0. Interlaminar weight/delay calculations (lamtha) are not
        affected.

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

    pos_dict['L5_pyramidal'] = [
        pos for pos in it.product(xxrange, yyrange, [0])]
    pos_dict['L2_pyramidal'] = [
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

    pos_dict['L5_basket'] = [pos_xy + (0,) for
                             pos_xy in coords_sorted]
    pos_dict['L2_basket'] = [
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


def default_network(params, add_drives_from_params=False):
    """Instantiate the default all-to-all connected network.

    Parameters
    ----------
    params : dict
        The parameters to use for constructing the network.
    add_drives_from_params : bool
        If True, add drives as defined in the params-dict. NB this is mainly
        for backward-compatibility with HNN GUI, and will be deprecated in a
        future release. Default: False

    Returns
    -------
    net : Instance of Network object
        Network object used to store the default network.
        All connections defining the default network will be
        appeneded to net.connectivity.
    """

    net = Network(params, add_drives_from_params=add_drives_from_params)

    nc_dict = {
        'A_delay': net.delay,
        'threshold': net.threshold,
    }

    # source of synapse is always at soma

    # layer2 Pyr -> layer2 Pyr
    # layer5 Pyr -> layer5 Pyr
    nc_dict['lamtha'] = 3.
    loc = 'proximal'
    for target_cell in ['L2Pyr', 'L5Pyr']:
        for receptor in ['nmda', 'ampa']:
            key = f'gbar_{target_cell}_{target_cell}_{receptor}'
            nc_dict['A_weight'] = net._params[key]
            net._all_to_all_connect(
                target_cell, target_cell, loc, receptor,
                nc_dict, allow_autapses=False)

    # layer2 Basket -> layer2 Pyr
    src_cell = 'L2Basket'
    target_cell = 'L2Pyr'
    nc_dict['lamtha'] = 50.
    loc = 'soma'
    for receptor in ['gabaa', 'gabab']:
        key = f'gbar_L2Basket_L2Pyr_{receptor}'
        nc_dict['A_weight'] = net._params[key]
        net._all_to_all_connect(
            src_cell, target_cell, loc, receptor, nc_dict)

    # layer5 Basket -> layer5 Pyr
    src_cell = 'L5Basket'
    target_cell = 'L5Pyr'
    nc_dict['lamtha'] = 70.
    loc = 'soma'
    for receptor in ['gabaa', 'gabab']:
        key = f'gbar_L5Basket_{target_cell}_{receptor}'
        nc_dict['A_weight'] = net._params[key]
        net._all_to_all_connect(
            src_cell, target_cell, loc, receptor, nc_dict)

    # layer2 Pyr -> layer5 Pyr
    src_cell = 'L2Pyr'
    nc_dict['lamtha'] = 3.
    receptor = 'ampa'
    for loc in ['proximal', 'distal']:
        key = f'gbar_L2Pyr_{target_cell}'
        nc_dict['A_weight'] = net._params[key]
        net._all_to_all_connect(
            src_cell, target_cell, loc, receptor, nc_dict)

    # layer2 Basket -> layer5 Pyr
    src_cell = 'L2Basket'
    nc_dict['lamtha'] = 50.
    key = f'gbar_L2Basket_{target_cell}'
    nc_dict['A_weight'] = net._params[key]
    loc = 'distal'
    receptor = 'gabaa'
    net._all_to_all_connect(
        src_cell, target_cell, loc, receptor, nc_dict)

    # xx -> layer2 Basket
    src_cell = 'L2Pyr'
    target_cell = 'L2Basket'
    nc_dict['lamtha'] = 3.
    key = f'gbar_L2Pyr_{target_cell}'
    nc_dict['A_weight'] = net._params[key]
    loc = 'soma'
    receptor = 'ampa'
    net._all_to_all_connect(
        src_cell, target_cell, loc, receptor, nc_dict)

    src_cell = 'L2Basket'
    nc_dict['lamtha'] = 20.
    key = f'gbar_L2Basket_{target_cell}'
    nc_dict['A_weight'] = net._params[key]
    loc = 'soma'
    receptor = 'gabaa'
    net._all_to_all_connect(
        src_cell, target_cell, loc, receptor, nc_dict)

    # xx -> layer5 Basket
    src_cell = 'L5Basket'
    target_cell = 'L5Basket'
    nc_dict['lamtha'] = 20.
    loc = 'soma'
    receptor = 'gabaa'
    key = f'gbar_L5Basket_{target_cell}'
    nc_dict['A_weight'] = net._params[key]
    net._all_to_all_connect(
        src_cell, target_cell, loc, receptor, nc_dict,
        allow_autapses=False)

    src_cell = 'L5Pyr'
    nc_dict['lamtha'] = 3.
    key = f'gbar_L5Pyr_{target_cell}'
    nc_dict['A_weight'] = net._params[key]
    loc = 'soma'
    receptor = 'ampa'
    net._all_to_all_connect(
        src_cell, target_cell, loc, receptor, nc_dict)

    src_cell = 'L2Pyr'
    key = f'gbar_L2Pyr_{target_cell}'
    nc_dict['A_weight'] = net._params[key]
    loc = 'soma'
    receptor = 'ampa'
    net._all_to_all_connect(
        src_cell, target_cell, loc, receptor, nc_dict)

    return net


def _connection_probability(conn, probability, seed=0):
    """Remove/keep a random subset of connections.

    Parameters
    ----------
    conn : Instance of _Connectivity object
        Object specifying the biophysical parameters and src target pairs
        of a specific connection class. Function modifies conn in place.
    probability : float
        Probability of connection between any src-target pair.
        Defaults to 1.0 producing an all-to-all pattern.
    seed : int
        Seed for the numpy random number generator.

    Notes
    -----
    num_srcs and num_targets are not updated after pruning connections.
    These variables are meant to describe the set of original connections
    before they are randomly removed.

    The probability attribute will store the most recent value passed to
    this function. As such, this number does not accurately describe the
    connections probability of the original set after successive calls.
    """
    # Random number generator for random connection selection
    rng = np.random.default_rng(seed)
    _validate_type(probability, float, 'probability')
    if probability <= 0.0 or probability >= 1.0:
        raise ValueError('probability must be in the range (0,1)')
    # Flatten connections into a list of targets.
    all_connections = np.concatenate(
        [target_src_pair for
            target_src_pair in conn['gid_pairs'].values()])
    n_connections = np.round(
        len(all_connections) * probability).astype(int)

    # Select a random subset of connections to retain.
    new_connections = rng.choice(
        range(len(all_connections)), n_connections, replace=False)
    remove_srcs = list()
    connection_idx = 0
    for src_gid, target_src_pair in conn['gid_pairs'].items():
        target_new = list()
        for target_gid in target_src_pair:
            if connection_idx in new_connections:
                target_new.append(target_gid)
            connection_idx += 1

        # Update targets for src_gid
        if target_new:
            conn['gid_pairs'][src_gid] = target_new
        else:
            remove_srcs.append(src_gid)
    # Remove src_gids with no targets
    for src_gid in remove_srcs:
        conn['gid_pairs'].pop(src_gid)


class Network(object):
    """The Network class.

    Parameters
    ----------
    params : dict
        The parameters to use for constructing the network.
    add_drives_from_params : bool
        If True, add drives as defined in the params-dict. NB this is mainly
        for backward-compatibility with HNN GUI, and will be deprecated in a
        future release. Default: False
    legacy_mode : bool
        Set to True by default to enable matching HNN GUI output when drives
        are added suitably. Will be deprecated in a future release.

    Attributes
    ----------
    cell_types : dict
        Dictionary containing names of real cell types in the network
        (e.g. 'L2_basket') as keys and corresponding Cell instances as values.
        The Cell instance associated with a given key is used as a template
        for the other cells of its type in the population.
    gid_ranges : dict
        A dictionary of unique identifiers of each real and artificial cell
        in the network. Every cell type is represented by a key read from
        cell_types, followed by keys read from external_drives. The value
        of each key is a range of ints, one for each cell in given category.
        Examples: 'L2_basket': range(0, 270), 'evdist1': range(272, 542), etc
    pos_dict : dict
        Dictionary containing the coordinate positions of all cells.
        Keys are 'L2_pyramidal', 'L5_pyramidal', 'L2_basket', 'L5_basket',
        or any external drive name
    cells : list of Cell objects.
        The list of cells of the network, each containing features used in a
        NEURON simulation.
    cell_response : CellResponse
        An instance of the CellResponse object.
    external_drives : dict (keys: drive names) of dict (keys: parameters)
        The external driving inputs to the network. Drives are added by
        defining their spike-time dynamics, and their connectivity to the real
        cells of the network. Event times are instantiated before simulation,
        and are stored under the ``'events'``-key (list of list; first
        index for trials, second for event time lists for each drive cell).
    external_biases : dict of dict (bias parameters for each cell type)
        The parameters of bias inputs to cell somata, e.g., tonic current clamp
    connectivity : list of dict
        List of dictionaries specifying each cell-cell and drive-cell
        connection
    threshold : float
        Firing threshold of all cells.
    delay : float
        Synaptic delay in ms.

    Notes
    ----
    `net = default_network(params)` is the reccomended path for creating a
    network. Instantiating the network as `net = Network(params)` will
    produce a network with no cell to cell connections. As such,
    connectivity information contained in `params` will be ignored.
    """

    def __init__(self, params, add_drives_from_params=False,
                 legacy_mode=True):
        # Save the parameters used to create the Network
        self._params = params
        # Initialise a dictionary of cell ID's, which get used when the
        # network is constructed ('built') in NetworkBuilder
        # We want it to remain in each Network object, so that the user can
        # interrogate a built and simulated net. In addition, CellResponse is
        # attached to a Network during simulation---Network is the natural
        # place to keep this information
        self.gid_ranges = dict()
        self._n_gids = 0  # utility: keep track of last GID

        # XXX this can be removed once tests are made independent of HNN GUI
        # creates nc_dict-entries for ALL cell types
        self._legacy_mode = legacy_mode

        # Source dict of names, first real ones only!
        self.cell_types = {
            'L2_basket': basket(cell_name=_short_name('L2_basket')),
            'L2_pyramidal': pyramidal(cell_name=_short_name('L2_pyramidal')),
            'L5_basket': basket(cell_name=_short_name('L5_basket')),
            'L5_pyramidal': pyramidal(cell_name=_short_name('L5_pyramidal'))
        }

        # Create array of equally sampled time points for simulating currents
        # NB (only) used to initialise self.cell_response._times
        times = np.arange(0., params['tstop'] + params['dt'], params['dt'])
        cell_type_names = list(self.cell_types.keys())
        # Create CellResponse object, initialised with simulation time points
        self.cell_response = CellResponse(times=times,
                                          cell_type_names=cell_type_names)

        # external drives and biases
        self.external_drives = dict()
        self.external_biases = dict()

        # network connectivity
        self.connectivity = list()
        self.threshold = self._params['threshold']
        self.delay = 1.0

        # contents of pos_dict determines all downstream inferences of
        # cell counts, real and artificial
        self.pos_dict = dict()
        pos = _create_cell_coords(n_pyr_x=self._params['N_pyr_x'],
                                  n_pyr_y=self._params['N_pyr_y'],
                                  zdiff=1307.4)
        self.pos_dict['origin'] = pos['origin']
        # Every time pos_dict is updated, gid_ranges must be updated too
        for cell_name in self.cell_types:
            self._add_cell_type(cell_name, pos[cell_name])

        self.cells = dict()

        # set n_cells, EXCLUDING Artificial ones
        self.n_cells = sum(len(self.pos_dict[src]) for src in
                           self.cell_types)

        if add_drives_from_params:
            _add_drives_from_params(self)

    def __repr__(self):
        class_name = self.__class__.__name__
        s = ("%d x %d Pyramidal cells (L2, L5)"
             % (self._params['N_pyr_x'], self._params['N_pyr_y']))
        s += ("\n%d L2 basket cells\n%d L5 basket cells"
              % (len(self.pos_dict['L2_basket']),
                 len(self.pos_dict['L5_basket'])))
        return '<%s | %s>' % (class_name, s)

    def copy(self):
        """Return a copy of the Network instance

        The returned copy retains the intrinsic connectivity between cells, as
        well as those of any external drives or biases added to the network.
        The parameters of drive dynamics are also retained, but the
        instantiated ``events`` of the drives are cleared. This allows
        iterating over the values defining drive dynamics, without the need to
        re-define connectivity.

        Returns
        -------
        net_copy : instance of Network
            A copy of the instance with previous simulation results and
            ``events`` of external drives removed.
        """
        # clear cells containing Neuron objects to avoid pickling error
        self.cells = dict()
        net_copy = deepcopy(self)
        net_copy.cell_response = CellResponse(times=self.cell_response._times)
        net_copy._reset_drives()
        return net_copy

    def _update_cells(self):
        """Populate the network with cell objects"""

        self.n_cells = 0
        for cell_type in self.pos_dict.keys():
            if cell_type in self.cell_types:
                cells = list()
                for cell_idx, pos in enumerate(self.pos_dict[cell_type]):
                    cell = deepcopy(self.cell_types[cell_type])
                    cell.gid = self.gid_ranges[cell_type][cell_idx]
                    cell.pos = pos
                    cells.append(cell)
                    self.cells[cell_type] = cells
                self.n_cells += len(cells)

    def add_evoked_drive(self, name, *, mu, sigma, numspikes,
                         sync_within_trial=False, location,
                         weights_ampa=None, weights_nmda=None,
                         space_constant=3., synaptic_delays=0.1, seedcore=2):
        """Add an 'evoked' external drive to the network

        Parameters
        ----------
        name : str
            Unique name for the drive
        mu : float
            Mean of Gaussian event time distribution
        sigma : float
            Standard deviation of event time distribution
        numspikes : int
            Number of spikes at each target cell
        sync_within_trial : bool
            If True, the target cells receive each numspike synchronously. By
            default (False), spike times arriving at each target cell are
            sampled independently using the Gaussian parameteres (mu, sigma).
        location : str
            Target location of synapses ('distal' or 'proximal')
        weights_ampa : dict or None
            Synaptic weights (in uS) of AMPA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        weights_nmda : dict or None
            Synaptic weights (in uS) of NMDA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        synaptic_delays : float or dict
            Synaptic delay (in ms) at the column origin, dispersed laterally as
            a function of the space_constant. If float, applies to all target
            cell types. Use dict to create delay->cell mapping.
        space_constant : float
            Describes lateral dispersion (from column origin) of synaptic
            weights and delays within the simulated column
        seedcore : int
            Optional initial seed for random number generator (default: 2).
        """
        if not self._legacy_mode:
            _check_drive_parameter_values('evoked', sigma=sigma,
                                          numspikes=numspikes)

        drive = _NetworkDrive()
        drive['type'] = 'evoked'
        if name == 'extgauss':
            drive['type'] = 'gaussian'  # XXX needed to pass legacy tests!
        drive['cell_specific'] = True
        drive['seedcore'] = seedcore

        drive['dynamics'] = dict(mu=mu, sigma=sigma, numspikes=numspikes,
                                 sync_within_trial=sync_within_trial)
        drive['events'] = list()

        self._attach_drive(name, drive, weights_ampa, weights_nmda, location,
                           space_constant, synaptic_delays)

    def add_poisson_drive(self, name, *, tstart=0, tstop=None, rate_constant,
                          location, weights_ampa=None, weights_nmda=None,
                          space_constant=100., synaptic_delays=0.1,
                          seedcore=2):
        """Add a Poisson-distributed external drive to the network

        Parameters
        ----------
        name : str
            Unique name for the drive
        tstart : float
            Start time of Poisson-distributed spike train (default: 0)
        tstop : float
            End time of the spike train (defaults to None: tstop is set to the
            end of the simulation)
        rate_constant : float or dict of floats
            Rate constant (lambda) of the renewal-process generating the
            samples. If a float is provided, the same rate constant is applied
            to each target cell type. Cell type-specific values may be
            provided as a dictionary, in which a key must be present for each
            cell type with non-zero AMPA or NMDA weights.
        location : str
            Target location of synapses ('distal' or 'proximal')
        weights_ampa : dict or None
            Synaptic weights (in uS) of AMPA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        weights_nmda : dict or None
            Synaptic weights (in uS) of NMDA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        synaptic_delays : float or dict
            Synaptic delay (in ms) at the column origin, dispersed laterally as
            a function of the space_constant. If float, applies to all target
            cell types. Use dict to create delay->cell mapping.
        space_constant : float
            Describes lateral dispersion (from column origin) of synaptic
            weigths and delays within the simulated column
        seedcore : int
            Optional initial seed for random number generator (default: 2).
        """
        sim_end_time = self.cell_response.times[-1]
        if tstop is None:
            tstop = sim_end_time

        if not self._legacy_mode:
            _check_drive_parameter_values('Poisson', tstart=tstart,
                                          tstop=tstop,
                                          sim_end_time=sim_end_time)
            target_populations = _get_target_populations(weights_ampa,
                                                         weights_nmda)[0]
            _check_poisson_rates(rate_constant, target_populations,
                                 self.cell_types.keys())

        drive = _NetworkDrive()
        drive['type'] = 'poisson'
        drive['cell_specific'] = True
        drive['seedcore'] = seedcore

        drive['dynamics'] = dict(tstart=tstart, tstop=tstop,
                                 rate_constant=rate_constant)
        drive['events'] = list()
        self._attach_drive(name, drive, weights_ampa, weights_nmda, location,
                           space_constant, synaptic_delays)

    def add_bursty_drive(self, name, *, tstart=0, tstart_std=0, tstop=None,
                         location, burst_rate, burst_std=0, numspikes=2,
                         spike_isi=10, repeats=1, weights_ampa=None,
                         weights_nmda=None, synaptic_delays=0.1,
                         space_constant=100., seedcore=2):
        """Add a bursty (rhythmic) external drive to all cells of the network

        Parameters
        ----------
        name : str
            Unique name for the drive
        tstart : float
            Start time of the burst trains (default: 0)
        tstart_std : float
            If greater than 0, randomize start time with standard deviation
            tstart_std (unit: ms). Effectively jitters start time across
            multiple trials.
        tstop : float
            End time of burst trains (defaults to None: tstop is set to the
            end of the simulation)
        burst_rate : float
            The mean rate at which cyclic bursts occur (unit: Hz)
        burst_std : float
            The standard deviation of the burst occurrence on each cycle
            (unit: ms). Default: 0 ms
        numspikes : int
            The number of spikes in a burst. This is the spikes/burst parameter
            in the GUI. Default: 2 (doublet)
        spike_isi : float
            Time between spike events within a cycle (ISI). Default: 10 ms
        repeats : int
            The number of bursts per cycle. Default: 1
        location : str
            Target location of synapses ('distal' or 'proximal')
        weights_ampa : dict or None
            Synaptic weights (in uS) of AMPA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        weights_nmda : dict or None
            Synaptic weights (in uS) of NMDA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        synaptic_delays : float or dict
            Synaptic delay (in ms) at the column origin, dispersed laterally as
            a function of the space_constant. If float, applies to all target
            cell types. Use dict to create delay->cell mapping.
        space_constant : float
            Describes lateral dispersion (from column origin) of synaptic
            weights and delays within the simulated column
        seedcore : int
            Optional initial seed for random number generator (default: 2).
        """
        sim_end_time = self.cell_response.times[-1]
        if tstop is None:
            tstop = sim_end_time
        if not self._legacy_mode:
            _check_drive_parameter_values('bursty', tstart=tstart, tstop=tstop,
                                          sim_end_time=sim_end_time,
                                          sigma=tstart_std, location=location)
            _check_drive_parameter_values('bursty', sigma=burst_std,
                                          numspikes=numspikes,
                                          spike_isi=spike_isi,
                                          burst_rate=burst_rate)

        drive = _NetworkDrive()
        drive['type'] = 'bursty'
        drive['cell_specific'] = False
        drive['seedcore'] = seedcore

        drive['dynamics'] = dict(tstart=tstart,
                                 tstart_std=tstart_std, tstop=tstop,
                                 burst_rate=burst_rate, burst_std=burst_std,
                                 numspikes=numspikes, spike_isi=spike_isi,
                                 repeats=repeats)
        drive['events'] = list()

        self._attach_drive(name, drive, weights_ampa, weights_nmda, location,
                           space_constant, synaptic_delays,
                           cell_specific=False)

    def _attach_drive(self, name, drive, weights_ampa, weights_nmda, location,
                      space_constant, synaptic_delays, cell_specific=True):
        """Attach a drive to network based on connectivity information

        Parameters
        ----------
        name : str
            Name of drive (must be unique)
        drive : instance of _NetworkDrive
            Collection of parameters defining the dynamics of the drive
        weights_ampa : dict or None
            Synaptic weights (in uS) of AMPA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        weights_nmda : dict or None
            Synaptic weights (in uS) of NMDA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        location : str
            Target location of synapses ('distal' or 'proximal')
        space_constant : float
            Describes lateral dispersion (from column origin) of synaptic
            weights and delays within the simulated column
        synaptic_delays : dict
            Synaptic delay (in ms) at the column origin, dispersed laterally as
            a function of the space_constant
        cell_specific : bool
            Whether each cell has unique connection parameters (default: True)
            or all cells have common connections to a global (single) drive.

        Attached drive is stored in self.external_drives[name]
        self.pos_dict is updated, and self._update_gid_ranges() called
        """
        if name in self.external_drives:
            raise ValueError(f"Drive {name} already defined")
        if location not in ['distal', 'proximal']:
            raise ValueError("Allowed drive target locations are: 'distal', "
                             f"and 'proximal', got {location}")
        # allow passing weights as None, convert to dict here
        target_populations, weights_ampa, weights_nmda = \
            _get_target_populations(weights_ampa, weights_nmda)

        # weights passed must correspond to cells in the network
        if not target_populations.issubset(set(self.cell_types.keys())):
            raise ValueError('Allowed drive target cell types are: ',
                             f'{self.cell_types.keys()}')

        weights_by_receptor = {'ampa': weights_ampa, 'nmda': weights_nmda}
        if isinstance(synaptic_delays, dict):
            for receptor in ['ampa', 'nmda']:
                # synaptic_delays must be defined for all cell types for which
                # either AMPA or NMDA weights are non-zero
                if not (set(weights_by_receptor[receptor].keys()).issubset(
                        set(synaptic_delays.keys()))):
                    raise ValueError(
                        'synaptic_delays is either a common float or needs '
                        'to be specified as a dict for each cell type')

        # this is needed to keep the drive GIDs identical to those in HNN,
        # e.g., 'evdist1': range(272, 542), even when no L5_basket cells
        # are targeted (event times lists are empty). The logic in
        # _connect_celltypes is hard-coded to use these implict gid ranges
        if self._legacy_mode:
            # XXX tests must match HNN GUI output
            target_populations = list(self.cell_types.keys())
        elif len(target_populations) == 0:
            warn('No AMPA or NMDA weights > 0')

        drive['name'] = name  # for easier for-looping later
        drive['target_types'] = target_populations  # for _connect_celltypes

        drive['conn'], src_gid_ran = self._create_drive_conns(
            target_populations, weights_by_receptor, location,
            space_constant, synaptic_delays, cell_specific=cell_specific)

        self.external_drives[name] = drive

        pos = [self.pos_dict['origin'] for _ in src_gid_ran]
        self._add_cell_type(name, pos)

        # Update connectivity_list
        nc_dict = {
            'A_delay': self.delay,
            'threshold': self.threshold,
        }
        receptors = ['ampa', 'nmda']
        if drive['type'] == 'gaussian':
            receptors = ['ampa']
        # conn-parameters are for each target cell type
        for target_cell, drive_conn in drive['conn'].items():
            for receptor in receptors:
                if len(drive_conn[receptor]) > 0:
                    nc_dict['lamtha'] = drive_conn[
                        receptor]['lamtha']
                    nc_dict['A_delay'] = drive_conn[
                        receptor]['A_delay']
                    nc_dict['A_weight'] = drive_conn[
                        receptor]['A_weight']
                    loc = drive_conn['location']
                    self._all_to_all_connect(
                        drive['name'], target_cell, loc, receptor,
                        deepcopy(nc_dict), unique=drive['cell_specific'])

    def _create_drive_conns(self, target_populations, weights_by_receptor,
                            location, space_constant, synaptic_delays,
                            cell_specific=True):
        """Create parameter dictionary defining how drive connects to network

        Parameters
        ----------
        target_populations : list
            Cell names/types to attach the drive to
        weights_by_receptor : dict (keys: 'ampa' and 'nmda')
            Synaptic weights (in uS) for each receptor type
        location : str
            Target location of synapses ('distal' or 'proximal')
        space_constant : float
            Describes lateral dispersion (from column origin) of synaptic
            weights and delays within the simulated column
        synaptic_delays : dict or float
            Synaptic delay (in ms) at the column origin, dispersed laterally as
            a function of the space_constant
        cell_specific : bool
            Whether each cell has unique connection parameters (default: True)
            or all cells have common connections to a global (single) drive.

        Returns
        -------
        drive_conn_by_cell : dict
            Keys are target_populations, each item is itself a dict with keys
            relevant to how the drive connects to the rest of the network.
        src_gid_ran : range
            For convenience, return back the range of GIDs associated with
            all the driving units (they become _ArtificialCells later)

        Note
        ----
        drive_conn : dict
            'target_gids': range of target cell GIDs
            'target_type': target cell type (e.g. 'L2_basket')
            'src_gids': range of source (artificial) cell GIDs
            'location': 'distal' or 'proximal'
            'ampa' and 'nmda': dict
                'A_weight': synaptic weight
                'A_delay':  synaptic delay (used with space constant)
                'lamtha': space constant
        """
        drive_conn_by_cell = dict()
        src_gid_curr = self._n_gids

        if cell_specific:
            for cellname in target_populations:
                drive_conn = dict()  # NB created inside target_pop-loop
                drive_conn['location'] = location

                drive_conn['target_gids'] = self.gid_ranges[cellname]
                # NB list! This is used later in _parnet_connect
                drive_conn['target_type'] = cellname
                drive_conn['src_gids'] = range(
                    src_gid_curr,
                    src_gid_curr + len(drive_conn['target_gids']))
                src_gid_curr += len(drive_conn['target_gids'])
                drive_conn_by_cell[cellname] = drive_conn
        else:
            drive_conn = dict()
            drive_conn['location'] = location

            # NB list! This is used later in _parnet_connect
            drive_conn['src_gids'] = [src_gid_curr]
            src_gid_curr += 1

            drive_conn['target_gids'] = list()  # fill in below
            for cellname in target_populations:
                drive_conn['target_gids'] = self.gid_ranges[cellname]
                drive_conn['target_type'] = cellname
                drive_conn_by_cell[cellname] = drive_conn.copy()

        for cellname in target_populations:
            for receptor, weights in weights_by_receptor.items():
                drive_conn_by_cell[cellname][receptor] = dict()
                if cellname in weights:
                    drive_conn_by_cell[cellname][receptor][
                        'lamtha'] = space_constant
                    if isinstance(synaptic_delays, float):
                        drive_conn_by_cell[cellname][receptor][
                            'A_delay'] = synaptic_delays
                    elif isinstance(synaptic_delays, dict):
                        drive_conn_by_cell[cellname][receptor][
                            'A_delay'] = synaptic_delays[cellname]
                    drive_conn_by_cell[cellname][receptor][
                        'A_weight'] = weights[cellname]

        return drive_conn_by_cell, range(self._n_gids, src_gid_curr)

    def _reset_drives(self):
        # reset every time called again, e.g., from dipole.py or in self.copy()
        for drive_name in self.external_drives.keys():
            self.external_drives[drive_name]['events'] = list()

    def _instantiate_drives(self, n_trials=1):
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
        self._reset_drives()

        # each trial needs unique event time vectors
        for trial_idx in range(n_trials):
            for drive in self.external_drives.values():
                event_times = list()  # new list for each trial and drive

                # loop over drives (one for each target cell population)
                # and create event times
                for this_cell_drive_conn in drive['conn'].values():
                    if drive['cell_specific']:
                        for drive_cell_gid in this_cell_drive_conn['src_gids']:
                            event_times.append(_drive_cell_event_times(
                                drive['type'], this_cell_drive_conn,
                                drive['dynamics'], trial_idx=trial_idx,
                                drive_cell_gid=drive_cell_gid,
                                seedcore=drive['seedcore'])
                            )
                    else:
                        # cell_specific=False should only have one src_gid
                        assert len(this_cell_drive_conn['src_gids']) == 1

                        # Only return empty event times if all cells have
                        # no events
                        drive_cell_gid = this_cell_drive_conn[
                            'src_gids'][0]
                        event_times = [_drive_cell_event_times(
                            drive['type'], this_cell_drive_conn,
                            drive['dynamics'], trial_idx=trial_idx,
                            drive_cell_gid=drive_cell_gid,
                            seedcore=drive['seedcore'])]
                        # only one event times list for one src_gid
                        if len(event_times[0]) > 0:
                            break

                # 'events': nested list (n_trials x n_drive_cells x n_events)
                self.external_drives[
                    drive['name']]['events'].append(event_times)

    def add_tonic_bias(self, *, cell_type=None, amplitude=None,
                       t0=None, tstop=None):
        """Attach parameters of tonic biasing input for a given cell type.

        Parameters
        ----------
        cell_type : str
            The cell type whose cells will get the tonic input.
            Valid inputs are those in `net.cell_types`.
        amplitude : float
            The amplitude of the input.
        t0 : float
            The start time of tonic input (in ms). Default: 0 (beginning of
            simulation).
        tstop : float
            The end time of tonic input (in ms). Default: end of simulation.
        """
        if (cell_type is None or amplitude is None):
            raise ValueError('cell_type and amplitude must be defined'
                             f', got {cell_type}, {amplitude}')
        if 'tonic' not in self.external_biases:
            self.external_biases['tonic'] = dict()
        if cell_type in self.external_biases['tonic']:
            raise ValueError(f'Tonic bias already defined for {cell_type}')
        if t0 is None:
            t0 = 0
        if tstop is None:
            tstop = self.cell_response.times[-1]
        if tstop < 0.:
            raise ValueError('End time of tonic input cannot be negative')
        if tstop > self.cell_response.times[-1]:
            raise ValueError(
                f'End time of tonic input cannot exceed '
                f'simulation end time {self.cell_response.times[-1]}. '
                f'Got {tstop}.')
        if cell_type not in self.cell_types:
            raise ValueError(f'cell_type must be one of '
                             f'{list(self.cell_types.keys())}. '
                             f'Got {cell_type}')
        duration = tstop - t0
        if duration < 0.:
            raise ValueError('Duration of tonic input cannot be negative')

        self.external_biases['tonic'][cell_type] = {
            'amplitude': amplitude,
            't0': t0,
            'tstop': tstop
        }

    def _add_cell_type(self, cell_name, pos):
        """Add cell type by updating pos_dict and gid_ranges."""
        ll = self._n_gids
        self._n_gids = ll + len(pos)
        self.gid_ranges[cell_name] = range(ll, self._n_gids)
        self.pos_dict[cell_name] = pos

    def gid_to_type(self, gid):
        """Reverse lookup of gid to type."""
        for gidtype, gids in self.gid_ranges.items():
            if gid in gids:
                return gidtype

    def _gid_to_cell(self, gid):
        """Reverse lookup of gid to cell.

        Returns None if not a cell; should only be called after self.cells is
        populated.
        """
        src_type = self.gid_to_type(gid)
        if src_type not in self.cell_types:
            cell = None
        else:
            type_pos_ind = gid - self.gid_ranges[src_type][0]
            cell = self.cells[src_type][type_pos_ind]
        return cell

    def _all_to_all_connect(self, src_cell, target_cell,
                            loc, receptor, nc_dict,
                            allow_autapses=True, unique=False):
        """Generate connectivity list given lists of sources and targets.

        Parameters
        ----------
        src_cell : str
            Source cell type.
        target_cell : str
            Target cell type.
        loc : str
            Location of synapse on target cell. Must be
            'proximal', 'distal', or 'soma'.
        receptor : str
            Synaptic receptor of connection. Must be one of:
            'ampa', 'nmda', 'gabaa', or 'gabab'.
        nc_dict : dict
            The connection dictionary containing keys
            A_delay, A_weight, lamtha, and threshold.
        allow_autapses : bool
            If True, allow connecting neuron to itself.
        unique : bool
            If True, each target cell gets one "unique" feed.
            If False, all src_type cells are connected to
            all target_type cells.
        """
        src_gids = self.gid_ranges[_long_name(src_cell)]
        target_range = self.gid_ranges[_long_name(target_cell)]

        src_start = src_gids[0]  # Necessary for unique feeds

        if unique:
            src_gids = [src_gid + src_start for src_gid in target_range]
            target_gids = [[target_gid] for target_gid in target_range]
        else:
            target_gids = list()
            for src_gid in src_gids:
                target_src_pair = list()
                for target_gid in target_range:
                    if not allow_autapses and src_gid == target_gid:
                        continue
                    target_src_pair.append(target_gid)
                target_gids.append(target_src_pair)

        self.add_connection(
            src_gids, target_gids, loc, receptor,
            nc_dict['A_weight'], nc_dict['A_delay'], nc_dict['lamtha'])

    def add_connection(self, src_gids, target_gids, loc, receptor,
                       weight, delay, lamtha, probability=1.0, seed=0):
        """Appends connections to connectivity list

        Parameters
        ----------
        src_gids : str | int | range | list of int
            Identifier for source cells. Passing str arguments
            ('L2_pyramidal', 'L2_basket', 'L5_pyramidal', 'L5_basket') is
            equivalent to passing a list of gids for the relvant cell type.
            source - target connections are made in an all-to-all pattern.
        target_gids : str | int | range | list of int
            Identifer for targets of source cells. Passing str arguments
            ('L2_pyramidal', 'L2_basket', 'L5_pyramidal', 'L5_basket') is
            equivalent to passing a list of gids for the relvant cell type.
            source - target connections are made in an all-to-all pattern.
        loc : str
            Location of synapse on target cell. Must be
            'proximal', 'distal', or 'soma'. Note that inhibitory synapses
            (receptor='gabaa' or 'gabab') of L2 pyramidal neurons are only
            valid loc='soma'.
        receptor : str
            Synaptic receptor of connection. Must be one of:
            'ampa', 'nmda', 'gabaa', or 'gabab'.
        weight : float
            Synaptic weight on target cell.
        delay : float
            Synaptic delay in ms.
        lamtha : float
            Space constant.
        probability : float
            Probability of connection between any src-target pair.
            Defaults to 1.0 producing an all-to-all pattern.
        seed : int
            Seed for the numpy random number generator.

        Notes
        -----
        Connections are stored in:
        net.connectivity[idx]['gid_pairs'] : dict
            dict indexed by src gids with the format:
            {src_gid: [target_gids, ...], ...}
            where each src_gid indexes a list of all its targets.
        """
        conn = _Connectivity()
        threshold = self.threshold
        _validate_type(src_gids, (int, list, range, str), 'src_gids',
                       'int list, range, or str')
        _validate_type(target_gids, (int, list, range, str), 'target_gids',
                       'int list, range or str')
        valid_cells = [
            'L2_pyramidal', 'L2_basket', 'L5_pyramidal', 'L5_basket']
        # Convert src_gids to list
        if isinstance(src_gids, int):
            src_gids = [src_gids]
        elif isinstance(src_gids, str):
            _check_option('src_gids', src_gids, valid_cells)
            src_gids = self.gid_ranges[_long_name(src_gids)]

        # Convert target_gids to list of list, one element for each src_gid
        if isinstance(target_gids, int):
            target_gids = [[target_gids] for _ in range(len(src_gids))]
        elif isinstance(target_gids, str):
            _check_option('target_gids', target_gids, valid_cells)
            target_gids = [list(self.gid_ranges[_long_name(target_gids)])
                           for _ in range(len(src_gids))]
        elif isinstance(target_gids, range):
            target_gids = [list(target_gids) for _ in range(len(src_gids))]
        elif isinstance(target_gids, list) and all(isinstance(t_gid, int)
                                                   for t_gid in target_gids):
            target_gids = [target_gids for _ in range(len(src_gids))]

        # Validate each target list - src pairs.
        # set() used to avoid redundant checks.
        target_set = set()
        for target_src_pair in target_gids:
            _validate_type(target_src_pair, list, 'target_gids[idx]',
                           'list or range')
            for target_gid in target_src_pair:
                target_set.add(target_gid)
        target_type = self.gid_to_type(target_gids[0][0])
        for target_gid in target_set:
            _validate_type(target_gid, int, 'target_gid', 'int')
            # Ensure gids in range of Network.gid_ranges
            gid_type = self.gid_to_type(target_gid)
            if gid_type is None:
                raise AssertionError(
                    f'target_gid {target_gid}''not in net.gid_ranges')
            elif gid_type != target_type:
                raise AssertionError(
                    'All target_gids must be of the same type')
        conn['target_type'] = target_type
        conn['target_range'] = self.gid_ranges[_long_name(target_type)]
        conn['num_targets'] = len(target_set)

        if len(target_gids) != len(src_gids):
            raise AssertionError('target_gids must have a list for each src.')

        # Format gid_pairs and add to conn dictionary
        gid_pairs = dict()
        src_type = self.gid_to_type(src_gids[0])
        for src_gid, target_src_pair in zip(src_gids, target_gids):
            _validate_type(src_gid, int, 'src_gid', 'int')
            gid_type = self.gid_to_type(src_gid)
            if gid_type is None:
                raise AssertionError(
                    f'src_gid {src_gid} not in net.gid_ranges')
            elif gid_type != src_type:
                raise AssertionError('All src_gids must be of the same type')
            gid_pairs[src_gid] = target_src_pair
        conn['src_type'] = src_type
        conn['src_range'] = self.gid_ranges[_long_name(src_type)]
        conn['num_srcs'] = len(src_gids)

        conn['gid_pairs'] = gid_pairs

        # Validate string inputs
        _validate_type(loc, str, 'loc')
        _validate_type(receptor, str, 'receptor')

        valid_loc = ['proximal', 'distal', 'soma']
        _check_option('loc', loc, valid_loc)
        conn['loc'] = loc

        valid_receptor = ['ampa', 'nmda', 'gabaa', 'gabab']
        _check_option('receptor', receptor, valid_receptor)
        conn['receptor'] = receptor

        # Create and validate nc_dict
        conn['nc_dict'] = dict()
        arg_names = ['delay', 'weight', 'lamtha', 'threshold']
        nc_dict_keys = ['A_delay', 'A_weight', 'lamtha', 'threshold']
        nc_conn_items = [delay, weight, lamtha, threshold]
        for key, arg_name, item in zip(nc_dict_keys, arg_names, nc_conn_items):
            _validate_type(item, (int, float), arg_name, 'int or float')
            conn['nc_dict'][key] = item

        # Probabilistically define connections
        if probability != 1.0:
            _connection_probability(conn, probability, seed)

        conn['probability'] = probability

        self.connectivity.append(deepcopy(conn))

    def clear_connectivity(self):
        """Remove all connections defined in Network.connectivity
        """
        connectivity = list()
        for conn in self.connectivity:
            if conn['src_type'] in self.external_drives.keys():
                connectivity.append(conn)
        self.connectivity = connectivity

    def clear_drives(self):
        """Remove all drives defined in Network.connectivity"""
        connectivity = list()
        for conn in self.connectivity:
            if conn['src_type'] not in self.external_drives.keys():
                connectivity.append(conn)
        self.external_drives = dict()
        self.connectivity = connectivity

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


class _Connectivity(dict):
    """A class for containing the connectivity details of the network

    Class instances are essentially dictionaries, with the keys described below
    as 'attributes'.

    Attributes
    ----------
    src_type : str
        Cell type of source gids.
    target_type : str
        Cell type of target gids.
    gid_pairs : dict
        dict indexed by src gids with the format:
        {src_gid: [target_gids, ...], ...}
        where each src_gid indexes a list of all its targets.
    num_srcs : int
        Number of unique source gids.
    num_targets : int
        Number of unique target gids.
    src_range : range
        Range of gids identified by src_type.
    target_range : range
        Range of gids identified by target_type.
    loc : str
        Location of synapse on target cell. Must be
        'proximal', 'distal', or 'soma'. Note that inhibitory synapses
        (receptor='gabaa' or 'gabab') of L2 pyramidal neurons are only
        valid loc='soma'.
    receptor : str
        Synaptic receptor of connection. Must be one of:
        'ampa', 'nmda', 'gabaa', or 'gabab'.
    nc_dict : dict
        Dictionary containing details of synaptic connection.
        Elements include:
        A_weight : float
            Synaptic weight on target cell.
        A_delay : float
            Synaptic delay in ms.
        lamtha : float
            Space constant.
    probability : float
        Probability of connection between any src-target pair.
        Defaults to 1.0 producing an all-to-all pattern.

    Notes
    -----
    The len() of src_range or target_range will not match
    num_srcs and num_targets for probability < 1.0.
    """

    def __repr__(self):
        entr = f"{self['src_type']} -> {self['target_type']}"
        entr += f"\ncell counts: {self['num_srcs']} srcs, "
        entr += f"{self['num_targets']} targets"
        entr += f"\nconnection probability: {self['probability']} "
        entr += f"\nloc: '{self['loc']}'; receptor: '{self['receptor']}'"
        entr += f"\nweight: {self['nc_dict']['A_weight']}; "
        entr += f"delay: {self['nc_dict']['A_delay']}; "
        entr += f"lamtha: {self['nc_dict']['lamtha']}"
        entr += "\n "

        return entr


class _NetworkDrive(dict):
    """A class for containing the parameters of external drives

    Class instances are essentially dictionaries, with keys described below
    as 'attributes'. For example, drive['events'] contains the spike times of
    exogeneous inputs.

    Attributes
    ----------
    name : str
        Name of drive (must be unique)
    type : str
        Examples: 'evoked', 'gaussian', 'poisson', 'bursty'
    events : list of lists
        List of spike time lists. First index is of length n_trials. Second
        index is over the 'artificial' cells associated with this drive.
    cell_specific : bool
        Whether each cell has unique connection parameters (default: True)
        or all cells have common connections to a global (single) drive.
    seedcore : int
        Optional initial seed for random number generator
        Each artificial drive cell has seed = seedcore + gid
    target_types : set or list of str
        Names of cell types targeted by this drive (must be subset of
        net.cell_types.keys()).
    dynamics : dict
        Parameters describing how the temporal dynamics of spike trains in the
        drive. The keys are specific to the type of drive ('evoked', 'bursty',
        etc.). See the drive add-methods in Network for details.
    conn : dict
        Parameters describing how the drive connects to the network.
        Valid keys are 'L2_basket', 'L2_pyramidal', 'L5_basket', 'L5_pyramidal'
        conn['L2_basket'] is a dict with the following keys:
            'target_gids' : range
                Range of target cell GIDs;
            'target_type' : str
                Target cell type (e.g. 'L2_basket');
            'src_gids' : range
                Source (artificial) cell GIDs;
            'location' : str
                Valid values are 'distal' or 'proximal'
            'nmda' or 'ampa' : dict
                Connectivity parameters for each receptor type
                specifying the synaptic weights. Valid keys are:
                    'A_weight': float
                        Synaptic weight
                    'A_delay': float
                        Synaptic delay at d=0 (used with space constant)
                    'lamtha': float
                        Space constant
    """

    def __repr__(self):
        entr = f"<External drive '{self['name']}'"
        if 'type' in self.keys():
            entr += f"\ndrive class: {self['type']}"
            entr += f"\ntarget cell types: {self['target_types']}"
            entr += f"\ncell-specific: {self['cell_specific']}"
            entr += "\ndynamic parameters:"
            for key, val in self['dynamics'].items():
                entr += f"\n\t{key}: {val}"
        if len(self['events']) > 0:
            plurl = 's' if len(self['events']) > 1 else ''
            entr += ("\nevent times instantiated for "
                     f"{len(self['events'])} trial{plurl}")
        entr += '>'
        return entr
