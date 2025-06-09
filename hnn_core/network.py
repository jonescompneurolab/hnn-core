"""Network class."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Sam Neymotin <samnemo@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>
#          Christopher Bailey <cjb@cfin.au.dk>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>

import itertools as it
from copy import deepcopy
from collections import OrderedDict, defaultdict
from typing import Dict

import numpy as np
import warnings

from .drives import _drive_cell_event_times
from .drives import _get_target_properties, _add_drives_from_params
from .drives import _check_drive_parameter_values, _check_poisson_rates
from .cells_default import pyramidal, basket
from .params import _long_name, _short_name
from .viz import plot_cells
from .externals.mne import _validate_type, _check_option
from .extracellular import ExtracellularArray
from .check import _check_gids, _gid_to_type, _string_input_to_list
from .hnn_io import write_network_configuration, network_to_dict
from .externals.mne import copy_doc
from .utils import _replace_dict_identifier


def _create_cell_coords(n_pyr_x, n_pyr_y, zdiff, inplane_distance):
    """Creates coordinate grid and place cells in it.

    Parameters
    ----------
    n_pyr_x : int
        The number of Pyramidal cells in x direction.
    n_pyr_y : int
        The number of Pyramidal cells in y direction.
    zdiff : float
        Expressed as a positive DEPTH of L2 relative to L5 pyramidal cell
        somas, where L5 is defined to lie at z==0. Interlaminar weight/delay
        calculations (lamtha) are not affected. The basket cells are
        arbitrarily placed slightly above (L5) and slightly below (L2) their
        respective pyramidal cell layers.
    inplane_distance : float
        The grid spacing of pyramidal cells (in um). Note that basket cells are
        placed in an uneven formation. Each one of them lies on a grid point
        together with a pyramidal cell, though (overlapping).

    Returns
    -------
    pos_dict : dict of list of tuple (x, y, z)
        Dictionary containing coordinate positions.
        Keys are 'L2_pyramidal', 'L5_pyramidal', 'L2_basket', 'L5_basket',
        'common', or any of the elements of the list p_unique_keys

    Notes
    -----
    Common positions are all located at origin.
    Sort of a hack because of redundancy.
    """

    def _calc_pyramidal_coord(xxrange, yyrange, zdiff):
        list_coords = [pos for pos in it.product(xxrange, yyrange, [zdiff])]
        return list_coords

    def _calc_basket_coord(n_pyr_x, n_pyr_y, zdiff, inplane_distance, weight):
        xzero = np.arange(0, n_pyr_x, 3) * inplane_distance
        xone = np.arange(1, n_pyr_x, 3) * inplane_distance
        # split even and odd y vals
        yeven = np.arange(0, n_pyr_y, 2) * inplane_distance
        yodd = np.arange(1, n_pyr_y, 2) * inplane_distance
        # create general list of x,y coords and sort it
        coords = [pos for pos in it.product(xzero, yeven)] + [
            pos for pos in it.product(xone, yodd)
        ]
        coords_sorted = sorted(coords, key=lambda pos: pos[1])

        # append the z value for position
        list_coords = [
            (pos_xy[0], pos_xy[1], weight * zdiff) for pos_xy in coords_sorted
        ]

        return list_coords

    def _calc_origin(xxrange, yyrange, zdiff):
        # origin's z component isn't used in calculating distance functions.
        # will be used for adding external drives.
        origin_x = xxrange[int((len(xxrange) - 1) // 2)]
        origin_y = yyrange[int((len(yyrange) - 1) // 2)]
        origin_z = np.floor(zdiff / 2)
        origin = (origin_x, origin_y, origin_z)
        return origin

    # Calculate distances
    xxrange = np.arange(n_pyr_x) * inplane_distance
    yyrange = np.arange(n_pyr_y) * inplane_distance

    pos_dict = {
        "L5_pyramidal": _calc_pyramidal_coord(xxrange, yyrange, zdiff=0),
        "L2_pyramidal": _calc_pyramidal_coord(xxrange, yyrange, zdiff=zdiff),
        "L5_basket": _calc_basket_coord(
            n_pyr_x, n_pyr_y, zdiff, inplane_distance, weight=0.2
        ),
        "L2_basket": _calc_basket_coord(
            n_pyr_x, n_pyr_y, zdiff, inplane_distance, weight=0.8
        ),
        "origin": _calc_origin(xxrange, yyrange, zdiff),
    }

    return pos_dict


def _compare_lists(s, t):
    """
    Compares lists for equality

    From https://stackoverflow.com/a/7829388
    """
    t = list(t)  # make a mutable copy
    try:
        for elem in s:
            t.remove(elem)
    except ValueError:
        return False
    return not t


def _connection_probability(conn, probability, conn_seed=None):
    """Remove/keep a random subset of connections.

    Parameters
    ----------
    conn : Instance of _Connectivity object
        Object specifying the biophysical parameters and src target pairs
        of a specific connection class. Function modifies conn in place.
    probability : float
        Probability of connection between any src-target pair.
        Defaults to 1.0 producing an all-to-all pattern.
    conn_seed : int
        Optional initial seed for random number generator (default: None).
        Used to randomly remove connections when probability < 1.0.

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
    rng = np.random.default_rng(conn_seed)
    _validate_type(probability, float, "probability")
    if probability <= 0.0 or probability >= 1.0:
        raise ValueError("probability must be in the range (0,1)")
    # Flatten connections into a list of targets.
    all_connections = np.concatenate(
        [target_src_pair for target_src_pair in conn["gid_pairs"].values()]
    )
    n_connections = np.round(len(all_connections) * probability).astype(int)

    # Select a random subset of connections to retain.
    new_connections = rng.choice(
        range(len(all_connections)), n_connections, replace=False
    )
    remove_srcs = list()
    connection_idx = 0
    for src_gid, target_src_pair in conn["gid_pairs"].items():
        target_new = list()
        for target_gid in target_src_pair:
            if connection_idx in new_connections:
                target_new.append(target_gid)
            connection_idx += 1

        # Update targets for src_gid
        if target_new:
            conn["gid_pairs"][src_gid] = target_new
        else:
            remove_srcs.append(src_gid)
    # Remove src_gids with no targets
    for src_gid in remove_srcs:
        conn["gid_pairs"].pop(src_gid)


def pick_connection(net, src_gids=None, target_gids=None, loc=None, receptor=None):
    """Returns indices of connections that match search parameters.

    Parameters
    ----------
    net : Instance of Network object
        The Network object
    src_gids : str | int | range | list of int | None
        Identifier for source cells. Passing str arguments
        ('L2_pyramidal', 'L2_basket', 'L5_pyramidal', 'L5_basket') is
        equivalent to passing a list of gids for the relevant cell type.
        source - target connections are made in an all-to-all pattern.
    target_gids : str | int | range | list of int | None
        Identifier for targets of source cells. Passing str arguments
        ('L2_pyramidal', 'L2_basket', 'L5_pyramidal', 'L5_basket') is
        equivalent to passing a list of gids for the relevant cell type.
        source - target connections are made in an all-to-all pattern.
    loc : str | list of str | None
        Location of synapse on target cell. Must be
        'proximal', 'distal', or 'soma'. Note that inhibitory synapses
        (receptor='gabaa' or 'gabab') of L2 pyramidal neurons are only
        valid loc='soma'.
    receptor : str | list of str | None
        Synaptic receptor of connection. Must be one of:
        'ampa', 'nmda', 'gabaa', or 'gabab'.

    Returns
    -------
    conn_indices : list of int
        List of indices corresponding to items in net.connectivity.
        Connection indices are included if any of the provided parameter
        values are present in a connection.

    Notes
    -----
    Passing a list of values to a single parameter corresponds to a
    logical OR operation across indices. For example,
    loc=['distal', 'proximal'] returns all connections that target
    distal or proximal dendrites.

    Passing  multiple parameters corresponds to a logical AND operation.
    For example, net.pick_connection(loc='distal', receptor='ampa')
    returns only the indices of connections that target the distal
    dendrites and have ampa receptors.
    """

    # Convert src and target gids to lists
    valid_srcs = list(net.gid_ranges.keys())  # includes drives as srcs
    valid_targets = list(net.cell_types.keys())
    src_gids_checked = _check_gids(
        src_gids, net.gid_ranges, valid_srcs, "src_gids", same_type=False
    )
    target_gids_checked = _check_gids(
        target_gids, net.gid_ranges, valid_targets, "target_gids", same_type=False
    )

    _validate_type(loc, (str, list, None), "loc", "str, list, or None")
    _validate_type(receptor, (str, list, None), "receptor", "str, list, or None")

    valid_loc = ["proximal", "distal", "soma"]
    valid_receptor = ["ampa", "nmda", "gabaa", "gabab"]

    # Convert receptor and loc to list
    loc_list = _string_input_to_list(loc, valid_loc, "loc")
    receptor_list = _string_input_to_list(receptor, valid_receptor, "receptor")

    # Create lookup dictionaries
    src_dict, target_dict = defaultdict(list), defaultdict(list)
    loc_dict, receptor_dict = defaultdict(list), defaultdict(list)
    for conn_idx, conn in enumerate(net.connectivity):
        # Store connections matching each src_gid
        for src_gid in conn["src_gids"]:
            src_dict[src_gid].append(conn_idx)

        # Store connections matching each target_gid
        for target_gid in conn["target_gids"]:
            target_dict[target_gid].append(conn_idx)

        # Store connections matching each location
        loc_dict[conn["loc"]].append(conn_idx)

        # Store connections matching each receptor
        receptor_dict[conn["receptor"]].append(conn_idx)

    # Look up conn indices that match search terms and add to set.
    conn_set = set()
    search_pairs = [
        (src_gids_checked, src_dict),
        (target_gids_checked, target_dict),
        (loc_list, loc_dict),
        (receptor_list, receptor_dict),
    ]
    for search_terms, search_dict in search_pairs:
        if search_terms:
            inner_set = set()
            # Union of indices which match inputs for single parameter
            for term in search_terms:
                inner_set = inner_set.union(search_dict.get(term, list()))

            # Empty search
            if not inner_set:
                return list()
            # Initial search has results
            elif inner_set and not conn_set:
                conn_set = inner_set.copy()
            # Subsequent searches have results
            elif inner_set and conn_set:
                # Intersect across parameters
                conn_set = conn_set.intersection(inner_set)
                # If at any point there's no matching elements, return empty
                if not conn_set:
                    return list()

    return sorted(conn_set)


class Network:
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
    mesh_shape : tuple of int (default: (10, 10))
        Defines the (n_x, n_y) shape of the grid of pyramidal cells.


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
    rec_arrays : dict
        Stores electrode position information and voltages recorded by them
        for extracellular potential measurements. Multiple electrode arrays
        may be defined as unique keys. The values of the dictionary are
        instances of :class:`hnn_core.extracellular.ExtracellularArray`.
    threshold : float
        Firing threshold of all cells.
    delay : float
        Synaptic delay in ms.

    Notes
    -----
    ``net = jones_2009_model(params)`` is the recommended path for creating a
    network. Instantiating the network as ``net = Network(params)`` will
    produce a network with no cell-to-cell connections. As such,
    connectivity information contained in ``params`` will be ignored.
    """

    def __init__(
        self,
        params,
        add_drives_from_params=False,
        legacy_mode=False,
        mesh_shape=(10, 10),
    ):
        # Save the parameters used to create the Network
        _validate_type(params, dict, "params")
        self._params = params
        # Initialise a dictionary of cell ID's, which get used when the
        # network is constructed ('built') in NetworkBuilder
        # We want it to remain in each Network object, so that the user can
        # interrogate a built and simulated net. In addition, CellResponse is
        # attached to a Network during simulation---Network is the natural
        # place to keep this information. Order matters: cell gids first, then
        # artificial drive cells
        self.gid_ranges = OrderedDict()
        self._n_gids = 0  # utility: keep track of last GID

        # XXX this can be removed once tests are made independent of HNN GUI
        # creates nc_dict-entries for ALL cell types
        self._legacy_mode = legacy_mode
        if self._legacy_mode:
            warnings.warn(
                "Legacy mode is used solely to maintain compatibility with"
                ".param files of the old HNN GUI. This feature will be "
                "deprecrated in future releases.",
                DeprecationWarning,
                stacklevel=1,
            )

        # Source dict of names, first real ones only!
        cell_types = {
            "L2_basket": basket(cell_name=_short_name("L2_basket")),
            "L2_pyramidal": pyramidal(cell_name=_short_name("L2_pyramidal")),
            "L5_basket": basket(cell_name=_short_name("L5_basket")),
            "L5_pyramidal": pyramidal(cell_name=_short_name("L5_pyramidal")),
        }

        self.cell_response = None
        # external drives and biases
        self.external_drives = dict()
        self.external_biases = dict()

        # network connectivity
        self.connectivity = list()
        self.threshold = self._params["threshold"]
        self.delay = 1.0

        # extracellular recordings (if applicable)
        self.rec_arrays = dict()

        # contents of pos_dict determines all downstream inferences of
        # cell counts, real and artificial
        self._n_cells = 0  # used in tests and MPIBackend checks
        self.pos_dict = dict()
        self.cell_types = dict()

        # set the mesh shape
        _validate_type(mesh_shape, tuple, "mesh_shape")
        _validate_type(mesh_shape[0], int, "mesh_shape[0]")
        _validate_type(mesh_shape[1], int, "mesh_shape[1]")

        if mesh_shape[0] < 1 or mesh_shape[1] < 1:
            raise ValueError(
                f"mesh_shape must be a tuple of positive integers, got: {mesh_shape}"
            )

        self._N_pyr_x = mesh_shape[0]
        self._N_pyr_y = mesh_shape[1]

        self._inplane_distance = 1.0  # XXX hard-coded default
        self._layer_separation = 1307.4  # XXX hard-coded default
        self.set_cell_positions(
            inplane_distance=self._inplane_distance,
            layer_separation=self._layer_separation,
        )

        # populates self.gid_ranges for the 1st time: order matters for
        # NetworkBuilder!
        for cell_name in cell_types:
            self._add_cell_type(
                cell_name, self.pos_dict[cell_name], cell_template=cell_types[cell_name]
            )

        if add_drives_from_params:
            _add_drives_from_params(self)

        self._tstop = None
        self._dt = None

    def __repr__(self):
        class_name = self.__class__.__name__
        # Dynamically create the description based on the current cell types
        descriptions = list()
        for cell_name in self.cell_types:
            count = len(self.pos_dict.get(cell_name, []))
            descriptions.append(f"{count} {cell_name} cells")

        # Combine all descriptions into a single string
        description_str = "\n".join(descriptions)
        return f"<{class_name} | {description_str}>"

    def __eq__(self, other):
        if not isinstance(other, Network):
            return NotImplemented

        # Check connectivity
        if (len(self.connectivity) != len(other.connectivity)) or not (
            _compare_lists(self.connectivity, other.connectivity)
        ):
            return False

        # Check all other attributes
        attrs_to_ignore = ["connectivity"]
        for attr in vars(self).keys():
            if attr.startswith("_") or attr in attrs_to_ignore:
                continue

            if hasattr(self, attr) and hasattr(other, attr):
                if getattr(self, attr) != getattr(other, attr):
                    return False
            else:
                # Does not have the same set of attributes
                return False

        return True

    def set_cell_positions(self, *, inplane_distance=None, layer_separation=None):
        """Set relative positions of cells arranged in a square grid

        Note that it is possible to change only a subset of the parameters
        (the default value of each is None, which implies no change).

        Parameters
        ----------
        inplane_distance : float
            The in plane-distance (in um) between pyramidal cell somas in the
            square grid. Note that this parameter does not affect the amplitude
            of the dipole moment.
        layer_separation : float
            The separation of pyramidal cell soma layers 2/3 and 5. Note that
            this parameter does not affect the amplitude of the dipole moment.
        """
        if inplane_distance is None:
            inplane_distance = self._inplane_distance
        _validate_type(inplane_distance, (float, int), "inplane_distance")
        if not inplane_distance > 0.0:
            raise ValueError(
                f"In-plane distance must be positive, got: {inplane_distance}"
            )

        if layer_separation is None:
            layer_separation = self._layer_separation
        _validate_type(layer_separation, (float, int), "layer_separation")
        if not layer_separation > 0.0:
            raise ValueError(
                f"Layer separation must be positive, got: {layer_separation}"
            )

        pos = _create_cell_coords(
            n_pyr_x=self._N_pyr_x,
            n_pyr_y=self._N_pyr_y,
            zdiff=layer_separation,
            inplane_distance=inplane_distance,
        )
        # update positions of the real cells
        for key in pos.keys():
            self.pos_dict[key] = pos[key]

        # update drives to be positioned at network origin
        for drive_name, drive in self.external_drives.items():
            pos = [self.pos_dict["origin"]] * drive["n_drive_cells"]
            self.pos_dict[drive_name] = pos

        self._inplane_distance = inplane_distance
        self._layer_separation = layer_separation

    def copy(self):
        """Return a copy of the Network instance

        The returned copy retains the intrinsic connectivity between cells, as
        well as those of any external drives or biases added to the network.
        The parameters of drive dynamics are also retained, but the
        instantiated ``events`` of the drives are cleared. This allows
        iterating over the values defining drive dynamics, without the need to
        re-define connectivity. Extracellular recording arrays are retained in
        the network, but cleared of existing data.

        Returns
        -------
        net_copy : instance of Network
            A copy of the instance with previous simulation results and
            ``events`` of external drives removed.
        """
        net_copy = deepcopy(self)
        net_copy._reset_drives()
        net_copy._reset_rec_arrays()
        return net_copy

    def add_evoked_drive(
        self,
        name,
        *,
        mu,
        sigma,
        numspikes,
        location,
        n_drive_cells="n_cells",
        cell_specific=True,
        weights_ampa=None,
        weights_nmda=None,
        space_constant=3.0,
        synaptic_delays=0.1,
        probability=1.0,
        event_seed=2,
        conn_seed=3,
    ):
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
        location : str
            Target location of synapses. Must be an element of
            `Cell.sect_loc` such as 'proximal' or 'distal', which defines a
            group of sections, or an existing section such as 'soma' or
            'apical_tuft' (defined in `Cell.sections` for all targeted cells).
            The parameter `legacy_mode` of the `Network` must be set to `False`
            to target specific sections.
        n_drive_cells : int | 'n_cells'
            The number of drive cells that each contribute an independently
            sampled synaptic spike to the network according to the Gaussian
            time distribution (mu, sigma). If n_drive_cells='n_cells'
            (default) and cell_specific=True, a drive cell gets assigned to
            each available simulated cell in the network with 1-to-1
            connectivity. Otherwise, drive cells are assigned with
            all-to-all connectivity. If you wish to synchronize the timing of
            this evoked drive across the network in a given trial with one
            spike, set n_drive_cells=1 and cell_specific=False.
        cell_specific : bool
            Whether each artificial drive cell has 1-to-1 (True, default) or
            all-to-all (False) connection parameters. Note that 1-to-1
            connectivity requires that n_drive_cells='n_cells', where 'n_cells'
            denotes the number of all available cells that this drive can
            target in the network.
        weights_ampa : dict or None
            Synaptic weights (in uS) of AMPA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        weights_nmda : dict or None
            Synaptic weights (in uS) of NMDA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        synaptic_delays : dict or float
            Synaptic delay (in ms) at the column origin, dispersed laterally as
            a function of the space_constant. If float, applies to all target
            cell types. Use dict to create delay->cell mapping.
        space_constant : float
            Describes lateral dispersion (from the column origin) of synaptic
            weights and delays within the simulated column. The constant is
            measured in the units of ``inplane_distance`` of
            :class:`~hnn_core.Network`. For example, for ``space_constant=3``,
            the weights are modulated by the factor
            ``exp(-(x / (3 * inplane_distance)) ** 2)``, where x is the
            physical distance (in um) between the connected cells in the xy
            plane (delays are modulated by the inverse of this factor).
        probability : dict or float (default: 1.0)
            Probability of connection between any src-target pair.
            Use dict to create probability->cell mapping. If float, applies to
            all target cell types
        event_seed : int
            Optional initial seed for random number generator (default: 2).
            Used to generate event times for drive cells.
            Not fixed across trials (see Notes)
        conn_seed : int
            Optional initial seed for random number generator (default: 3).
            Used to randomly remove connections when probability < 1.0.
            Fixed across trials (see Notes)

        Notes
        -----
        Random seeding behavior across trials is different for event_seed
        and conn_seed (n_trials > 1 in simulate_dipole(..., n_trials):

        - event_seed
            Across trials, the random seed is incremented such that
            the exact spike times are different
        - conn_seed
            The random seed does not change across trials. This means for
            probability < 1.0, the random subset of gids targeted is the same.
        """
        if not self._legacy_mode:
            _check_drive_parameter_values("evoked", sigma=sigma, numspikes=numspikes)
        drive = _NetworkDrive()
        drive["type"] = "evoked"
        drive["location"] = location
        if name == "extgauss":
            drive["type"] = "gaussian"  # XXX needed to pass legacy tests!
        drive["n_drive_cells"] = n_drive_cells
        drive["event_seed"] = event_seed
        drive["conn_seed"] = conn_seed
        drive["dynamics"] = dict(mu=mu, sigma=sigma, numspikes=numspikes)
        drive["events"] = list()
        # Need to save this information
        drive["weights_ampa"] = weights_ampa
        drive["weights_nmda"] = weights_nmda
        drive["synaptic_delays"] = synaptic_delays
        drive["probability"] = probability

        self._attach_drive(
            name,
            drive,
            weights_ampa,
            weights_nmda,
            location,
            space_constant,
            synaptic_delays,
            n_drive_cells,
            cell_specific,
            probability,
        )

    def add_poisson_drive(
        self,
        name,
        *,
        tstart=0,
        tstop=None,
        rate_constant,
        location,
        n_drive_cells="n_cells",
        cell_specific=True,
        weights_ampa=None,
        weights_nmda=None,
        space_constant=100.0,
        synaptic_delays=0.1,
        probability=1.0,
        event_seed=2,
        conn_seed=3,
    ):
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
            Rate constant (lambda > 0) of the renewal-process generating the
            samples. If a float is provided, the same rate constant is applied
            to each target cell type. Cell type-specific values may be
            provided as a dictionary, in which a key must be present for each
            cell type with non-zero AMPA or NMDA weights.
        location : str
            Target location of synapses. Must be an element of
            `Cell.sect_loc` such as 'proximal' or 'distal', which defines a
            group of sections, or an existing section such as 'soma' or
            'apical_tuft' (defined in `Cell.sections` for all targeted cells).
            The parameter `legacy_mode` of the `Network` must be set to `False`
            to target specific sections.
        n_drive_cells : int | 'n_cells'
            The number of drive cells that each contribute an independently
            sampled synaptic spike to the network according to a Poisson
            process. If n_drive_cells='n_cells' (default) and
            cell_specific=True, a drive cell gets assigned to each available
            simulated cell in the network with 1-to-1 connectivity. Otherwise,
            drive cells are assigned with all-to-all connectivity. If you wish
            to synchronize the timing of Poisson drive across the network in a
            given trial, set n_drive_cells=1 and cell_specific=False.
        cell_specific : bool
            Whether each artificial drive cell has 1-to-1 (True, default) or
            all-to-all (False) connection parameters. Note that 1-to-1
            connectivity requires that n_drive_cells='n_cells', where 'n_cells'
            denotes the number of all available cells that this drive can
            target in the network.
        weights_ampa : dict or None
            Synaptic weights (in uS) of AMPA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        weights_nmda : dict or None
            Synaptic weights (in uS) of NMDA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        synaptic_delays : dict or float
            Synaptic delay (in ms) at the column origin, dispersed laterally as
            a function of the space_constant. If float, applies to all target
            cell types. Use dict to create delay->cell mapping.
        space_constant : float
            Describes lateral dispersion (from the column origin) of synaptic
            weights and delays within the simulated column. The constant is
            measured in the units of ``inplane_distance`` of
            :class:`~hnn_core.Network`. For example, for ``space_constant=3``,
            the weights and delays are modulated by the factor
            ``exp(-(x / (3 * inplane_distance)) ** 2)``, where ``x`` is the
            physical distance (in um) between the connected cells in the xy
            plane.
        probability : dict or float (default: 1.0)
            Probability of connection between any src-target pair.
            Use dict to create probability->cell mapping. If float, applies to
            all target cell types.
        event_seed : int
            Optional initial seed for random number generator (default: 2).
            Used to generate event times for drive cells.
        conn_seed : int
            Optional initial seed for random number generator (default: 3).
            Used to randomly remove connections when probability < 1.0.
        """

        _check_drive_parameter_values("Poisson", tstart=tstart, tstop=tstop)
        target_populations = _get_target_properties(
            weights_ampa, weights_nmda, synaptic_delays, location
        )[0]
        _check_poisson_rates(rate_constant, target_populations, self.cell_types.keys())
        if isinstance(rate_constant, dict):
            if not cell_specific:
                raise ValueError(
                    f"Drives specific to cell types are only "
                    f"possible with cell_specific=True and "
                    f"n_drive_cells='n_cells'. Got cell_specific"
                    f" cell_specific={cell_specific} and "
                    f"n_drive_cells={n_drive_cells}."
                )
        elif isinstance(rate_constant, (float, int)):
            if cell_specific:
                rate_constant = {
                    cell_type: rate_constant for cell_type in target_populations
                }

        drive = _NetworkDrive()
        drive["type"] = "poisson"
        drive["location"] = location
        drive["n_drive_cells"] = n_drive_cells
        drive["event_seed"] = event_seed
        drive["conn_seed"] = conn_seed
        drive["dynamics"] = dict(
            tstart=tstart, tstop=tstop, rate_constant=rate_constant
        )
        drive["events"] = list()
        # Need to save this information
        drive["weights_ampa"] = weights_ampa
        drive["weights_nmda"] = weights_nmda
        drive["synaptic_delays"] = synaptic_delays
        drive["probability"] = probability

        self._attach_drive(
            name,
            drive,
            weights_ampa,
            weights_nmda,
            location,
            space_constant,
            synaptic_delays,
            n_drive_cells,
            cell_specific,
            probability,
        )

    def add_bursty_drive(
        self,
        name,
        *,
        tstart=0,
        tstart_std=0,
        tstop=None,
        location,
        burst_rate,
        burst_std=0,
        numspikes=2,
        spike_isi=10,
        n_drive_cells=1,
        cell_specific=False,
        weights_ampa=None,
        weights_nmda=None,
        synaptic_delays=0.1,
        space_constant=100.0,
        probability=1.0,
        event_seed=2,
        conn_seed=3,
    ):
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
        location : str
            Target location of synapses. Must be an element of
            `Cell.sect_loc` such as 'proximal' or 'distal', which defines a
            group of sections, or an existing section such as 'soma' or
            'apical_tuft' (defined in `Cell.sections` for all targeted cells).
            The parameter `legacy_mode` of the `Network` must be set to `False`
            to target specific sections.
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
        n_drive_cells : int | 'n_cells'
            The number of drive cells that contribute an independently sampled
            burst at each cycle. If n_drive_cells='n_cells' and
            cell_specific=True, a drive cell gets assigned to
            each available simulated cell in the network with 1-to-1
            connectivity. Otherwise (default: 1), drive cells are assigned with
            all-to-all connectivity and provide synchronous input to cells in
            the network.
        cell_specific : bool
            Whether each artificial drive cell has 1-to-1 (True) or all-to-all
            (False, default) connection parameters. Note that 1-to-1
            connectivity requires that n_drive_cells='n_cells', where 'n_cells'
            denotes the number of all available cells that this drive can
            target in the network.
        weights_ampa : dict or None
            Synaptic weights (in uS) of AMPA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        weights_nmda : dict or None
            Synaptic weights (in uS) of NMDA receptors on each targeted cell
            type (dict keys). Cell types omitted from the dict are set to zero.
        synaptic_delays : dict or float
            Synaptic delay (in ms) at the column origin, dispersed laterally as
            a function of the space_constant. If float, applies to all target
            cell types. Use dict to create delay->cell mapping.
        space_constant : float
            Describes lateral dispersion (from the column origin) of synaptic
            weights and delays within the simulated column. The constant is
            measured in the units of ``inplane_distance`` of
            :class:`~hnn_core.Network`. For example, for ``space_constant=3``,
            the weights and delays are modulated by the factor
            ``exp(-(x / (3 * inplane_distance)) ** 2)``, where ``x`` is the
            physical distance (in um) between the connected cells in the xy
            plane.
        probability : dict or float (default: 1.0)
            Probability of connection between any src-target pair.
            Use dict to create probability->cell mapping. If float, applies to
            all target cell types.
        event_seed : int
            Optional initial seed for random number generator (default: 2).
            Used to generate event times for drive cells.
        conn_seed : int
            Optional initial seed for random number generator (default: 3).
            Used to randomly remove connections when probability < 1.0.
        """
        if not self._legacy_mode:
            _check_drive_parameter_values(
                "bursty",
                tstart=tstart,
                tstop=tstop,
                sigma=tstart_std,
                location=location,
            )
            _check_drive_parameter_values(
                "bursty",
                sigma=burst_std,
                numspikes=numspikes,
                spike_isi=spike_isi,
                burst_rate=burst_rate,
            )

        drive = _NetworkDrive()
        drive["type"] = "bursty"
        drive["location"] = location
        drive["n_drive_cells"] = n_drive_cells
        drive["event_seed"] = event_seed
        drive["conn_seed"] = conn_seed
        drive["dynamics"] = dict(
            tstart=tstart,
            tstart_std=tstart_std,
            tstop=tstop,
            burst_rate=burst_rate,
            burst_std=burst_std,
            numspikes=numspikes,
            spike_isi=spike_isi,
        )
        drive["events"] = list()
        # Need to save this information
        drive["weights_ampa"] = weights_ampa
        drive["weights_nmda"] = weights_nmda
        drive["synaptic_delays"] = synaptic_delays
        drive["probability"] = probability

        self._attach_drive(
            name,
            drive,
            weights_ampa,
            weights_nmda,
            location,
            space_constant,
            synaptic_delays,
            n_drive_cells,
            cell_specific,
            probability,
        )

    def _attach_drive(
        self,
        name,
        drive,
        weights_ampa,
        weights_nmda,
        location,
        space_constant,
        synaptic_delays,
        n_drive_cells,
        cell_specific,
        probability,
    ):
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
            Target location of synapses. Must be an element of
            `Cell.sect_loc` such as 'proximal' or 'distal', which defines a
            group of sections, or an existing section such as 'soma' or
            'apical_tuft' (defined in `Cell.sections` for all targeted cells).
            The parameter `legacy_mode` of the `Network` must be set to `False`
            to target specific sections.
        space_constant : float
            Describes lateral dispersion (from the column origin) of synaptic
            weights and delays within the simulated column. The constant is
            measured in the units of ``inplane_distance`` of
            :class:`~hnn_core.Network`. For example, for ``space_constant=3``,
            the weights and delays are modulated by the factor
            ``exp(-(x / (3 * inplane_distance)) ** 2)``, where ``x`` is the
            physical distance (in um) between the connected cells in the xy
            plane.
        synaptic_delays : dict or float
            Synaptic delay (in ms) at the column origin, dispersed laterally as
            a function of the space_constant
        n_drive_cells : int | 'n_cells'
            The number of drive cells (i.e., ArtificialCell objects) that
            contribute to this drive. If n_drive_cells='n_cells' and
            cell_specific=True, an artificial drive cell gets assigned to each
            available cell in the network with 1-to-1 connectivity (completely
            unsynchronous). Otherwise, drive cells get assigned with all-to-all
            connectivity. If you wish to synchronize the timing of this evoked
            drive across the network in a given trial with one spike, set
            n_drive_cells=1 and cell_specific=False.
        cell_specific : bool
            Whether each artificial drive cell has 1-to-1 (True) or all-to-all
            (False) connection parameters. Note that 1-to-1
            connectivity requires that n_drive_cells='n_cells', where 'n_cells'
            denotes the number of all available cells that this drive can
            target in the network.
        probability : dict or float (default: 1.0)
            Probability of connection between any src-target pair.
            Use dict to create probability->cell mapping. If float, applies to
            all target cell types

        Attached drive is stored in self.external_drives[name]
        self.pos_dict is updated, and self._update_gid_ranges() called
        """
        if name in self.external_drives:
            raise ValueError(f"Drive {name} already defined")

        _validate_type(probability, (float, dict), "probability", "float or dict")
        # allow passing weights as None, convert to dict here
        (target_populations, weights_by_type, delays_by_type, probability_by_type) = (
            _get_target_properties(
                weights_ampa, weights_nmda, synaptic_delays, location, probability
            )
        )

        # weights passed must correspond to cells in the network
        if not target_populations.issubset(set(self.cell_types.keys())):
            raise ValueError(
                "Allowed drive target cell types are: ", f"{self.cell_types.keys()}"
            )

        # enforce the same order as in self.cell_types - necessary for
        # consistent source gid assignment
        target_populations = [
            cell_type
            for cell_type in self.cell_types.keys()
            if cell_type in target_populations
        ]

        # Ensure location exists for all target cells
        cell_sections = [
            set(self.cell_types[cell_type].sections.keys())
            for cell_type in target_populations
        ]
        sect_locs = [
            set(self.cell_types[cell_type].sect_loc.keys())
            for cell_type in target_populations
        ]

        valid_cell_sections = set.intersection(*cell_sections)
        valid_sect_locs = set.intersection(*sect_locs)
        valid_loc = list(valid_cell_sections) + list(valid_sect_locs)

        _check_option(
            "location",
            location,
            valid_loc,
            extra=(
                f" (the location '{location}' is not defined "
                "for one of the targeted cells)"
            ),
        )

        if self._legacy_mode:
            # allows tests must match HNN GUI output by preserving original
            # gid assignment convention
            target_populations = list(self.cell_types.keys())
            for target_type in target_populations:
                if target_type not in weights_by_type:
                    weights_by_type.update({target_type: {"ampa": 0.0}})
                if target_type not in delays_by_type:
                    delays_by_type.update({target_type: 0.1})
                if target_type not in probability_by_type:
                    probability_by_type.update({target_type: 1.0})
        elif len(target_populations) == 0:
            raise ValueError(
                "No target populations have been specified for this drive."
            )

        if cell_specific and n_drive_cells != "n_cells":
            raise ValueError(
                f"If cell_specific is True, n_drive_cells must"
                f" equal 'n_cells'. Got {n_drive_cells}."
            )
        elif not cell_specific:
            if not isinstance(n_drive_cells, int):
                raise ValueError(
                    f"If cell_specific is False, n_drive_cells "
                    f"must be of type int. Got "
                    f"{type(n_drive_cells)}."
                )
            if not n_drive_cells > 0:
                raise ValueError(
                    "Number of drive cells must be greater than "
                    f"0. Got {n_drive_cells}."
                )

        drive["name"] = name  # for easier for-looping later
        drive["target_types"] = target_populations  # for _connect_celltypes
        drive["cell_specific"] = cell_specific

        if n_drive_cells == "n_cells":
            n_drive_cells = 0
            for cell_type in target_populations:
                n_drive_cells += len(self.gid_ranges[cell_type])

        drive["n_drive_cells"] = n_drive_cells
        self.external_drives[name] = drive

        pos = [self.pos_dict["origin"]] * n_drive_cells
        self._add_cell_type(name, pos)

        # Set the starting index for cell-specific source gids
        # This will be updated depending on the number of target cells
        # of each cell type
        src_idx = 0

        # seed_increment increased by 1 for each target cell type,
        # added to conn_seed to ensure statistical independence of random
        # connections when probability < 1.0
        for seed_increment, target_cell_type in enumerate(target_populations):
            target_gids = list(self.gid_ranges[target_cell_type])
            delays = delays_by_type[target_cell_type]
            probability = probability_by_type[target_cell_type]
            if cell_specific:
                target_gids_nested = [[target_gid] for target_gid in target_gids]
                src_idx_end = src_idx + len(target_gids)
                src_gids = list(self.gid_ranges[name])[src_idx:src_idx_end]
                src_idx = src_idx_end
                for receptor_idx, receptor in enumerate(
                    weights_by_type[target_cell_type]
                ):
                    weights = weights_by_type[target_cell_type][receptor]
                    self.add_connection(
                        src_gids=src_gids,
                        target_gids=target_gids_nested,
                        loc=location,
                        receptor=receptor,
                        weight=weights,
                        delay=delays,
                        lamtha=space_constant,
                        probability=probability,
                        conn_seed=drive["conn_seed"] + seed_increment,
                    )
                    # Ensure that AMPA/NMDA connections target the same gids
                    if receptor_idx > 0:
                        self.connectivity[-1]["src_gids"] = self.connectivity[-2][
                            "src_gids"
                        ]

            else:
                for receptor_idx, receptor in enumerate(
                    weights_by_type[target_cell_type]
                ):
                    weights = weights_by_type[target_cell_type][receptor]
                    self.add_connection(
                        src_gids=name,
                        target_gids=target_gids,
                        loc=location,
                        receptor=receptor,
                        weight=weights,
                        delay=delays,
                        lamtha=space_constant,
                        probability=probability,
                        conn_seed=drive["conn_seed"] + seed_increment,
                    )
                    # Ensure that AMPA/NMDA connections target the same gids
                    # when probability < 1
                    if receptor_idx > 0:
                        self.connectivity[-1]["src_gids"] = self.connectivity[-2][
                            "src_gids"
                        ]

    def _reset_drives(self):
        # reset every time called again, e.g., from dipole.py or in self.copy()
        for drive_name in self.external_drives.keys():
            self.external_drives[drive_name]["events"] = list()

    def _reset_rec_arrays(self):
        # clear the data in rec_arrays
        for arr in self.rec_arrays.values():
            arr._reset()

    def _instantiate_drives(self, tstop, n_trials=1):
        """Creates event time vectors for all drives across trials

        Parameters
        ----------
        tstop : float
            The simulation stop time (ms)
        n_trials : int
            Number of trials to create events for (default: 1)

        NB this must be a separate method because dipole.py:simulate_dipole
        accepts an n_trials-argument, which overrides the N_trials-parameter
        used at initialisation time. The good news is that only the event_times
        need to be recalculated, all the GIDs etc remain the same.
        """
        self._reset_drives()

        # each trial needs unique event time vectors
        for trial_idx in range(n_trials):
            for drive in self.external_drives.values():
                event_times = list()  # new list for each trial and drive
                for drive_cell_gid in self.gid_ranges[drive["name"]]:
                    drive_cell_gid_offset = (
                        drive_cell_gid - self.gid_ranges[drive["name"]][0]
                    )
                    trial_seed_offset = self._n_gids
                    if drive["cell_specific"]:
                        # loop over drives (one for each target cell
                        # population) and create event times
                        conn_idxs = pick_connection(self, src_gids=drive_cell_gid)
                        target_types = set(
                            [
                                self.connectivity[conn_idx]["target_type"]
                                for conn_idx in conn_idxs
                            ]
                        )
                        for target_type in target_types:
                            event_times.append(
                                _drive_cell_event_times(
                                    drive["type"],
                                    drive["dynamics"],
                                    target_type=target_type,
                                    trial_idx=trial_idx,
                                    drive_cell_gid=drive_cell_gid_offset,
                                    event_seed=drive["event_seed"],
                                    tstop=tstop,
                                    trial_seed_offset=trial_seed_offset,
                                )
                            )
                    else:
                        src_event_times = _drive_cell_event_times(
                            drive["type"],
                            drive["dynamics"],
                            tstop=tstop,
                            target_type="any",
                            trial_idx=trial_idx,
                            drive_cell_gid=drive_cell_gid_offset,
                            event_seed=drive["event_seed"],
                            trial_seed_offset=trial_seed_offset,
                        )
                        event_times.append(src_event_times)
                # 'events': nested list (n_trials x n_drive_cells x n_events)
                self.external_drives[drive["name"]]["events"].append(event_times)

    def add_tonic_bias(
        self,
        *,
        cell_type=None,
        section="soma",
        bias_name="tonic",
        amplitude,
        t0=0,
        tstop=None,
    ):
        """Attaches parameters of tonic bias input for given cell types

        Parameters
        ----------
        cell_types : str | None
            The name of the cell type to add a tonic input. When supplied,
            a float value must be provided with the `amplitude` keyword.
            Valid inputs are those listed in  `net.cell_types`.
        section : str
            name of cell section the bias should be applied to.
            See net.cell_types[cell_type].sections.keys()
        bias_name : str
            The name of the bias.
        amplitude: dict | float
            A dictionary of cell type keys (str) to amplitude values (float).
            Valid inputs for cell types are those listed in `net.cell_types`.
            If `cell_types` is not None, `amplitude` should be
            a float indicating the amplitude of the tonic input
            for the specified cell type.
        t0 : float
            The start time of tonic input (in ms). Default: 0 (beginning of
            simulation). This value will be applied to all the  tonic biases if
            multiple are specified with the `amplitude` keyword.
        tstop : float
            The end time of tonic input (in ms). Default: end of simulation.
            This value will be applied to all the  tonic biases if
            multiple are specified with the `amplitude` keyword.
        """

        # old functionality single cell type - amplitude
        if cell_type is not None:
            warnings.warn(
                "cell_type argument will be deprecated and "
                "removed in future releases. Use amplitude as a "
                "cell_type:str,amplitude:float dictionary."
                "Read the function docustring for more information",
                DeprecationWarning,
                stacklevel=1,
            )
            _validate_type(amplitude, (float, int), "amplitude")

            _add_cell_type_bias(
                network=self,
                cell_type=cell_type,
                section=section,
                bias_name=bias_name,
                amplitude=float(amplitude),
                t_0=t0,
                t_stop=tstop,
            )
        else:
            _validate_type(amplitude, dict, "amplitude")
            if len(amplitude) == 0:
                warnings.warn(
                    "No bias have been defined, no action taken",
                    UserWarning,
                    stacklevel=1,
                )
                return

            for _cell_type, _amplitude in amplitude.items():
                _add_cell_type_bias(
                    network=self,
                    cell_type=_cell_type,
                    section=section,
                    bias_name=bias_name,
                    amplitude=_amplitude,
                    t_0=t0,
                    t_stop=tstop,
                )

    def _add_cell_type(self, cell_name, pos, cell_template=None):
        """Add cell type by updating pos_dict and gid_ranges."""
        ll = self._n_gids
        self._n_gids += len(pos)
        self.gid_ranges[cell_name] = range(ll, self._n_gids)
        self.pos_dict[cell_name] = pos
        if cell_template is not None:
            self.cell_types.update({cell_name: cell_template})
            self._n_cells += len(pos)

    def _rename_cell_types(self, name_mapping: Dict[str, str]):
        """Renames cell types in the network.

        XXX: All HNN functionality is not supported, such as Dipole calculation

        Parameters
        ----------
        name_mapping: dict[str, str]
            Dictionary of what cell type names to change, and what to change
            them to. Keys are existing cell type name strings, and values are
            what string to change each key to. Note that both elements must be
            strings.
        """
        _validate_type(name_mapping, dict, "name_mapping")
        for original_name, new_name in name_mapping.items():
            if original_name not in self.cell_types.keys():
                raise ValueError(f"'{original_name}' is not in cell_types!")
            elif new_name in self.cell_types.keys():
                raise ValueError(f"'{new_name}' is already in cell_types!")
            elif original_name in self.cell_types.keys():
                for attr_name in [
                    "cell_types",
                    "pos_dict",
                    "external_biases",
                    "external_drives",
                    "gid_ranges",
                ]:
                    attr = getattr(self, attr_name)
                    if isinstance(attr, dict):
                        updated_attr = _replace_dict_identifier(
                            attr, original_name, new_name
                        )
                        setattr(self, attr_name, updated_attr)

                # Update Network.connectivity
                for connection in self.connectivity:
                    if connection["src_type"] == original_name:
                        connection["src_type"] = new_name
                    if connection["target_type"] == original_name:
                        connection["target_type"] = new_name

    def gid_to_type(self, gid):
        """Reverse lookup of gid to type."""
        return _gid_to_type(gid, self.gid_ranges)

    def add_connection(
        self,
        src_gids,
        target_gids,
        loc,
        receptor,
        weight,
        delay,
        lamtha,
        allow_autapses=True,
        probability=1.0,
        conn_seed=None,
    ):
        """Appends connections to connectivity list

        Parameters
        ----------
        src_gids : str | int | range | list of int
            Identifier for source cells. Passing str arguments ('evdist1',
            'L2_pyramidal', 'L2_basket', 'L5_pyramidal', 'L5_basket', etc.) is
            equivalent to passing a list of gids for the relevant cell type.
            source - target connections are made in an all-to-all pattern.
        target_gids : str | int | range | list of int
            Identifier for targets of source cells. Passing str arguments
            ('L2_pyramidal', 'L2_basket', 'L5_pyramidal', 'L5_basket') is
            equivalent to passing a list of gids for the relevant cell type.
            source - target connections are made in an all-to-all pattern.
        loc : str
            Target location of synapses. Must be an element of
            `Cell.sect_loc` such as 'proximal' or 'distal', which defines a
            group of sections, or an existing section such as 'soma' or
            'apical_tuft' (defined in `Cell.sections` for all targeted cells).
            The parameter `legacy_mode` of the `Network` must be set to `False`
            to target specific sections.
        receptor : str
            Synaptic receptor of connection. Must be one of:
            'ampa', 'nmda', 'gabaa', or 'gabab'.
        weight : float
            Synaptic weight on target cell.
        delay : float
            Synaptic delay in ms.
        lamtha : float
            Space constant.
        allow_autapses : bool
            If True, allow connecting neuron to itself.
        probability : float
            Probability of connection between any src-target pair.
            Defaults to 1.0 producing an all-to-all pattern.
        conn_seed : int
            Optional initial seed for random number generator (default: None).
            Used to randomly remove connections when probability < 1.0.

        Notes
        -----
        Connections are stored in ``net.connectivity[idx]['gid_pairs']``, a
        dictionary indexed by src gids with the format:
        {src_gid: [target_gids, ...], ...} where each src_gid indexes a list of
        all its targets.
        """
        conn = _Connectivity()
        threshold = self.threshold

        _validate_type(
            target_gids,
            (int, list, range, str),
            "target_gids",
            "int list, range or str",
        )
        _validate_type(allow_autapses, bool, "target_gids", "bool")
        valid_source_cells = list(self.gid_ranges.keys())

        # Convert src_gids to list
        src_gids = _check_gids(
            src_gids, self.gid_ranges, valid_source_cells, "src_gids"
        )

        # Convert target_gids to list of list, one element for each src_gid
        valid_target_cells = list(self.cell_types.keys())
        if isinstance(target_gids, int):
            target_gids = [[target_gids] for _ in range(len(src_gids))]
        elif isinstance(target_gids, str):
            _check_option("target_gids", target_gids, valid_target_cells)
            target_gids = [
                list(self.gid_ranges[_long_name(target_gids)])
                for _ in range(len(src_gids))
            ]
        elif isinstance(target_gids, range):
            target_gids = [list(target_gids) for _ in range(len(src_gids))]
        elif isinstance(target_gids, list) and all(
            isinstance(t_gid, int) for t_gid in target_gids
        ):
            target_gids = [target_gids for _ in range(len(src_gids))]

        # Validate each target list - src pairs.
        # set() used to avoid redundant checks.
        target_set = set()
        for target_src_pair in target_gids:
            _validate_type(target_src_pair, list, "target_gids[idx]", "list or range")
            for target_gid in target_src_pair:
                target_set.add(target_gid)
        target_type = self.gid_to_type(target_gids[0][0])
        for target_gid in target_set:
            _validate_type(target_gid, int, "target_gid", "int")
            # Ensure gids in range of Network.gid_ranges
            gid_type = self.gid_to_type(target_gid)
            if gid_type is None:
                raise AssertionError(f"target_gid {target_gid}not in net.gid_ranges")
            elif gid_type != target_type:
                raise AssertionError("All target_gids must be of the same type")
        conn["target_type"] = target_type
        conn["target_gids"] = target_set
        conn["num_targets"] = len(target_set)

        if len(target_gids) != len(src_gids):
            raise AssertionError("target_gids must have a list for each src.")

        # Format gid_pairs and add to conn dictionary
        gid_pairs = dict()
        for src_gid, target_src_pair in zip(src_gids, target_gids):
            if not allow_autapses:
                mask = np.isin(target_src_pair, src_gid, invert=True)
                target_src_pair = np.array(target_src_pair)[mask].tolist()
            gid_pairs[src_gid] = target_src_pair

        conn["src_type"] = self.gid_to_type(src_gids[0])
        conn["src_gids"] = set(src_gids)
        conn["num_srcs"] = len(src_gids)

        conn["gid_pairs"] = gid_pairs

        # Validate string inputs
        _validate_type(loc, str, "loc")
        _validate_type(receptor, str, "receptor")

        target_sect_loc = self.cell_types[target_type].sect_loc
        target_sections = self.cell_types[target_type].sections
        valid_loc = list(target_sect_loc.keys()) + list(target_sections.keys())

        _check_option(
            "loc",
            loc,
            valid_loc,
            extra=(f" (the loc '{loc}' is not defined for '{target_type}' cells)"),
        )
        conn["loc"] = loc

        # `loc` specifies a group of sections, all must contain the synapse
        # specified by `receptor`
        if loc in target_sect_loc:
            for sec_name in target_sect_loc[loc]:
                valid_receptor = target_sections[sec_name].syns
                _check_option(
                    "receptor",
                    receptor,
                    valid_receptor,
                    extra=f" (the '{receptor}' receptor is not "
                    f"defined for the '{sec_name}' of"
                    f"'{target_type}' cells)",
                )
        # `loc` specifies an individual section
        else:
            valid_receptor = target_sections[loc].syns
            _check_option(
                "receptor",
                receptor,
                valid_receptor,
                extra=f"(the '{receptor}' receptor is not "
                f"defined for the '{loc}' of"
                f"'{target_type}' cells)",
            )

        conn["receptor"] = receptor

        # Create and validate nc_dict
        conn["nc_dict"] = dict()
        arg_names = ["delay", "weight", "lamtha", "threshold"]
        nc_dict_keys = ["A_delay", "A_weight", "lamtha", "threshold"]
        nc_conn_items = [delay, weight, lamtha, threshold]
        for key, arg_name, item in zip(nc_dict_keys, arg_names, nc_conn_items):
            _validate_type(item, (int, float), arg_name, "int or float")
            conn["nc_dict"][key] = item

        conn["nc_dict"]["gain"] = 1.0

        # Probabilistically define connections
        if probability != 1.0:
            _connection_probability(conn, probability, conn_seed)

        conn["probability"] = probability
        conn["allow_autapses"] = allow_autapses

        self.connectivity.append(deepcopy(conn))

    def clear_connectivity(self):
        """Remove all connections defined in Network.connectivity"""
        connectivity = list()
        for conn in self.connectivity:
            if conn["src_type"] in self.external_drives.keys():
                connectivity.append(conn)
        self.connectivity = connectivity

    def clear_drives(self):
        """Remove all drives defined in Network.connectivity"""
        self.connectivity = [
            conn
            for conn in self.connectivity
            if conn["src_type"] not in self.external_drives.keys()
        ]

        for cell_name in list(self.gid_ranges.keys()):
            if cell_name in self.external_drives:
                self._n_gids -= len(self.gid_ranges[cell_name])
                del self.gid_ranges[cell_name]
                del self.pos_dict[cell_name]

        self.external_drives = dict()

    def add_electrode_array(
        self, name, electrode_pos, *, conductivity=0.3, method="psa", min_distance=0.5
    ):
        """Specify coordinates of electrode array for extracellular recording.

        Parameters
        ----------
        name : str
            Unique name of the array.
        electrode_pos : tuple | list of tuple
            Coordinates specifying the position for extracellular electrodes in
            the form of (x, y, z) (in um).
        conductivity : float
            Extracellular conductivity, in S/m, of the assumed infinite,
            homogeneous volume conductor that the cell and electrode are in.
        method : str
            Approximation to use. ``'psa'`` (point source approximation) treats
            each segment junction as a point extracellular current source.
            ``'lsa'`` (line source approximation) treats each segment as a line
            source of current, which extends from the previous to the next
            segment center point: /---x---/, where x is the current segment
            flanked by /.
        min_distance : float (default: 0.5; unit: um)
            To avoid numerical errors in calculating potentials, apply a
            minimum distance limit between the electrode contacts and the
            active neuronal membrane elements that act as sources of current.
            The default value of 0.5 um corresponds to 1 um diameter dendrites.
        """
        _validate_type(name, str, "name")
        if name in self.rec_arrays.keys():
            raise ValueError(f"{name} already exists, use another name!")

        # let ExtracellularArray perform all remaining argument checks
        self.rec_arrays.update(
            {
                name: ExtracellularArray(
                    electrode_pos,
                    conductivity=conductivity,
                    method=method,
                    min_distance=min_distance,
                )
            }
        )

    def update_weights(self, e_e=None, e_i=None, i_e=None, i_i=None, copy=False):
        """Update synaptic weights of the network.

        Parameters
        ----------
        e_e : float
            Synaptic gain of excitatory to excitatory connections
            (default None)
        e_i : float
            Synaptic gain of excitatory to inhibitory connections
            (default None)
        i_e : float
            Synaptic gain of inhibitory to excitatory connections
            (default None)
        i_i : float
            Synaptic gain of inhibitory to inhibitory connections
            (default None)
        copy : bool
            If True, returns a copy of the network. If False,
            the network is updated in place with a return of None.
            (default False)

        Returns
        -------
        net : instance of Network
            A copy of the instance with updated synaptic gains if copy=True.

        Notes
        -----
        Synaptic gains must be non-negative. The synaptic gains will only be
        updated if a float value is provided. If None is provided
        (the default), the synapticgain will remain unchanged.

        """
        _validate_type(copy, bool, "copy")

        net = self.copy() if copy else self

        e_conns = pick_connection(self, receptor=["ampa", "nmda"])
        e_cells = np.concatenate(
            [list(net.connectivity[conn_idx]["src_gids"]) for conn_idx in e_conns]
        ).tolist()

        i_conns = pick_connection(self, receptor=["gabaa", "gabab"])
        i_cells = np.concatenate(
            [list(net.connectivity[conn_idx]["src_gids"]) for conn_idx in i_conns]
        ).tolist()
        conn_types = {
            "e_e": (e_e, e_cells, e_cells),
            "e_i": (e_i, e_cells, i_cells),
            "i_e": (i_e, i_cells, e_cells),
            "i_i": (i_i, i_cells, i_cells),
        }

        for conn_type, (gain, e_vals, i_vals) in conn_types.items():
            if gain is None:
                continue

            _validate_type(gain, (int, float), conn_type, "int or float")
            if gain < 0.0:
                raise ValueError(
                    f"Synaptic gains must be non-negative.Got {gain} for '{conn_type}'."
                )

            conn_indices = pick_connection(net, src_gids=e_vals, target_gids=i_vals)
            for conn_idx in conn_indices:
                net.connectivity[conn_idx]["nc_dict"]["gain"] = gain

        if copy:
            return net

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

    def to_dict(self, write_output=False):
        return network_to_dict(self, write_output=write_output)

    @copy_doc(write_network_configuration)
    def write_configuration(self, fname, overwrite=True):
        write_network_configuration(self, fname, overwrite)


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
    src_gids : set of int
        Set of unique source gids in connection.
    target_gids : set of int
        Set of unique target gids in connection.
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
        gain : float
            Multiplicative factor for synaptic weight.
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
    location : str
        Target location of synapses ('distal' or 'proximal').
    type : str
        Examples: 'evoked', 'gaussian', 'poisson', 'bursty'
    events : list of lists
        List of spike time lists. First index is of length n_trials. Second
        index is over the 'artificial' cells associated with this drive.
    n_drive_cells : int
        The number of drive cells that contribute to this drive.
    cell_specific : bool
        Whether each cell has unique connection parameters (default: True)
        or all cells have common connections to a global (single) drive.
    event_seed : int
        Optional initial seed for random number generator used for event times.
        Each artificial drive cell has seed = event_seed + gid
    conn_seed : int
        Optional initial seed for random number generator.
        Used to randomly remove connections when probability < 1.0.
    target_types : set or list of str
        Names of cell types targeted by this drive (must be subset of
        net.cell_types.keys()).
    dynamics : dict
        Parameters describing how the temporal dynamics of spike trains in the
        drive. The keys are specific to the type of drive ('evoked', 'bursty',
        etc.). See the drive add-methods in Network for details.
    """

    def __repr__(self):
        entr = f"<External drive '{self['name']}'"
        if "type" in self.keys():
            entr += f"\ndrive class: {self['type']}"
            entr += f"\ntarget location: {self['location']}"
            entr += f"\ntarget cell types: {self['target_types']}"
            entr += f"\nnumber of drive cells: {self['n_drive_cells']}"
            entr += f"\ncell-specific: {self['cell_specific']}"
            entr += "\ndynamic parameters:"
            for key, val in self["dynamics"].items():
                entr += f"\n\t{key}: {val}"
        if len(self["events"]) > 0:
            plurl = "s" if len(self["events"]) > 1 else ""
            entr += f"\nevent times instantiated for {len(self['events'])} trial{plurl}"
        entr += ">"
        return entr


def _add_cell_type_bias(
    network: Network,
    amplitude: float,
    cell_type: str,
    section="soma",
    bias_name="tonic",
    t_0=0,
    t_stop=None,
):
    """Add a tonic bias to a specific cell type in the network.

    Parameters
    ----------
    network : Network
        The network to which the tonic bias is added.
    amplitude : float
        The amplitude of the tonic input (in nA) applied to the specified
        `cell_type`.
    cell_type : str
        The cell type to which the bias is applied.
    section : str, default 'soma'
        The section of the cell where the bias is applied (e.g., 'soma',
        'apical_tuft').
    bias_name : str, default 'tonic'
        A name identifier for the bias configuration, allowing multiple biases
        to be applied.
    t_0 : float, default 0
        The start time of the tonic input in milliseconds.
    t_stop : float, optional
        The end time of the tonic input in milliseconds. If None, the bias
        continues until the end of the simulation.
    """
    # Validate cell_type value
    if cell_type not in network.cell_types:
        raise ValueError(
            f"cell_type must be one of "
            f"{list(network.cell_types.keys())}. "
            f"Got {cell_type}"
        )

    if bias_name not in network.external_biases:
        network.external_biases[bias_name] = dict()

    if cell_type in network.external_biases[bias_name]:
        raise ValueError(f"Bias named {bias_name} already defined for {cell_type}")

    cell_type_bias = {
        "amplitude": amplitude,
        "t0": t_0,
        "tstop": t_stop,
        "section": section,
    }

    sections = list(network.cell_types[cell_type].sections.keys())

    # error when section is defined that doesn't exist.
    if section not in sections:
        raise ValueError(f"section must be one of {sections}. Got {section}.")
    else:
        cell_type_bias["section"] = section

    network.external_biases[bias_name][cell_type] = cell_type_bias
