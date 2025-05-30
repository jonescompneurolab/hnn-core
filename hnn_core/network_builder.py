"""Neuron simulation functions and NetworkBuilder class."""

# Authors: Mainak Jas <mjas@mgh.harvard.edu>
#          Sam Neymotin <samnemo@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>

import os
import os.path as op
from copy import deepcopy

import numpy as np
from neuron import h

# This is due to: https://github.com/neuronsimulator/nrn/pull/746
from neuron import __version__

if int(__version__[0]) >= 8:
    h.nrnunit_use_legacy(1)

from .cell import _ArtificialCell
from .params import _long_name, _short_name
from .extracellular import _ExtracellularArrayBuilder
from .network import pick_connection

# a few globals
_PC = None
_CVODE = None

# We need to maintain a reference to the last
# NetworkBuilder instance that ran pc.gid_clear(). Even if
# pc is global, if pc.gid_clear() is called within a new
# NetworkBuilder, it will seg fault.
_LAST_NETWORK = None


def _simulate_single_trial(net, tstop, dt, trial_idx):
    """Simulate one trial including building the network

    This is used by both backends. MPIBackend calls this in mpi_child.py, once
    for each trial (blocking), and JoblibBackend calls this for each trial
    (non-blocking)
    """

    neuron_net = NetworkBuilder(net, trial_idx=trial_idx)

    global _PC, _CVODE

    h.load_file("stdrun.hoc")

    rank = _get_rank()

    # Now let's simulate the dipole

    _PC.barrier()  # sync for output to screen

    # Set tstop before instantiating any classes
    h.tstop = tstop
    h.dt = dt  # simulation duration and time-step
    h.celsius = net._params["celsius"]  # 37.0 - set temperature

    times = h.Vector().record(h._ref_t)

    # sets the default max solver step in ms (purposefully large)
    _PC.set_maxstep(10)

    # initialize cells to -65 mV, after all the NetCon
    # delays have been specified
    h.finitialize()

    def simulation_time():
        print(f"Trial {trial_idx + 1}: {round(h.t, 2)} ms...")

    if rank == 0:
        for tt in range(0, int(h.tstop), 10):
            _CVODE.event(tt, simulation_time)

    h.fcurrent()

    # initialization complete, but wait for all procs to start the solver
    _PC.barrier()

    # actual simulation - run the solver
    _PC.psolve(h.tstop)

    _PC.barrier()

    # these calls aggregate data across procs/nodes
    neuron_net.aggregate_data(n_samples=times.size())

    # now convert data from Neuron into Python
    vsec_py = dict()
    for gid, vsec_dict in neuron_net._vsec.items():
        vsec_py[gid] = dict()
        for sec_name, vsec in vsec_dict.items():
            vsec_py[gid][sec_name] = vsec.to_python()

    isec_py = dict()
    for gid, isec_dict in neuron_net._isec.items():
        isec_py[gid] = dict()
        for sec_name, isec in isec_dict.items():
            isec_py[gid][sec_name] = {
                key: isec.to_python() for key, isec in isec.items()
            }

    ca_py = dict()
    for gid, ca_dict in neuron_net._ca.items():
        ca_py[gid] = dict()
        for sec_name, ca in ca_dict.items():
            if ca is not None:
                ca_py[gid][sec_name] = ca.to_python()

    dpl_data = np.c_[
        neuron_net._nrn_dipoles["L2_pyramidal"].as_numpy()
        + neuron_net._nrn_dipoles["L5_pyramidal"].as_numpy(),
        neuron_net._nrn_dipoles["L2_pyramidal"].as_numpy(),
        neuron_net._nrn_dipoles["L5_pyramidal"].as_numpy(),
    ]

    rec_arr_py = dict()
    rec_times_py = dict()
    for arr_name, nrn_arr in neuron_net._nrn_rec_arrays.items():
        rec_arr_py.update({arr_name: nrn_arr._get_nrn_voltages()})
        rec_times_py.update({arr_name: nrn_arr._get_nrn_times()})

    data = {
        "dpl_data": dpl_data,
        "spike_times": neuron_net._all_spike_times.to_python(),
        "spike_gids": neuron_net._all_spike_gids.to_python(),
        "gid_ranges": net.gid_ranges,
        "vsec": vsec_py,
        "isec": isec_py,
        "ca": ca_py,
        "rec_data": rec_arr_py,
        "rec_times": rec_times_py,
        "times": times.to_python(),
    }

    return data


def _is_loaded_mechanisms():
    # copied from:
    # https://www.neuron.yale.edu/neuron/static/py_doc/modelspec/programmatic/mechtype.html
    mt = h.MechanismType(0)
    mname = h.ref("")
    mnames = list()
    for i in range(mt.count()):
        mt.select(i)
        mt.selected(mname)
        mnames.append(mname[0])
    if "hh2" not in mnames:
        return False
    else:
        return True


def load_custom_mechanisms():
    if _is_loaded_mechanisms():
        return

    # recursively find the .so / .dll library
    mech_fname = list()
    mod_dir = op.join(op.dirname(__file__), "mod")
    for root, dirnames, filenames in os.walk(mod_dir):
        for filename in filenames:
            if filename.endswith((".so", ".dll")):
                mech_fname.append(os.path.join(root, filename))
                break

    if len(mech_fname) == 0:
        raise FileNotFoundError(f"No .so or .dll file found in {mod_dir}")

    h.nrn_load_dll(mech_fname[0])
    print("Loading custom mechanism files from %s" % mech_fname[0])
    if not _is_loaded_mechanisms():
        raise ValueError("The custom mechanisms could not be loaded")


def _get_nhosts():
    """Return the number of processors used by ParallelContext

    Returns
    -------
    nhosts: int
        Value from pc.nhost()
    """
    if _PC is not None:
        return int(_PC.nhost())

    return 1


def _get_rank():
    """Return the MPI rank from ParallelContext

    Returns
    -------
    rank: int
        Value from pc.id()
    """
    if _PC is not None:
        return int(_PC.id())

    return 0


def _create_parallel_context(n_cores=None, expose_imem=False):
    """Create parallel context.

    Parameters
    ----------
    n_cores: int | None
        Number of processors to use for a simulation. A value of None will
        allow NEURON to use all available processors.
    expose_imem : bool
        If True, sets _CVODE.use_fast_imem(1) (default: False)
    """

    global _CVODE, _PC

    if _PC is None:
        if n_cores is None:
            # MPI: Initialize the ParallelContext class
            _PC = h.ParallelContext()
        else:
            _PC = h.ParallelContext(n_cores)

        _CVODE = h.CVode()

        # use cache_efficient mode for allocating elements in contiguous order
        # cvode.cache_efficient(1)
    else:
        # ParallelContext() has already been called. Don't start more workers.
        # Just tell old nrniv workers to quit.
        _PC.done()

    # be explicit about using fixed step integration
    _CVODE.active(0)
    # note that CVode seems to forget this setting in either parallel backend
    if expose_imem:
        _CVODE.use_fast_imem(1)


class NetworkBuilder(object):
    """The NetworkBuilder class.

    Parameters
    ----------
    net : Network object
        The instance of Network to instantiate in NEURON-Python
    trial_idx : int (optional)
        Index number of the trial being processed (different event statistics).
        Defaults to 0.

    Attributes
    ----------
    trial_idx : int
        The index number of the current trial of a simulation.
    ncs : dict of list
        A dictionary with key describing the types of cell objects connected
        and contains a list of NetCon objects.

    Notes
    -----
    NetworkBuilder is not a pickleable class because it contains many NEURON
    objects once it has been instantiated. This is important for the Joblib
    backend that passes a pickled Network object to each forked process (job)
    and only instantiates NetworkBuilder after the fork.

    The `_build` routine can be called again to run more simulations without
    creating new `nrniv` processes. Instead, the NERUON objects are recreated
    and gids are reassigned according to the specifications in
    `self.net._params` and the network is ready for another simulation.
    """

    def __init__(self, net, trial_idx=0):
        self.net = net
        self.trial_idx = trial_idx

        # When computing the network dynamics in parallel, the nodes of the
        # network (real and artificial cells) potentially get distributed
        # on different host machines/threads. NetworkBuilder._gid_assign
        # assigns each node, identified by its unique GID, to one of the
        # possible hosts/threads for computations. _gid_list here contains
        # the GIDs of all the nodes assigned to the current host/thread.
        self._gid_list = list()
        # Note that GIDs are already defined in Network.gid_ranges
        # All that's left for NetworkBuilder is then to:
        # - _PC.set_gid2node(gid, rank)
        # - _PC.cell(gid, nrn_netcon) (or _PC.cell(drive_cell.gid, nrn_netcon))

        # cells from the network assigned to the current host/thread
        self._cells = list()

        # artificial cells must be appended to a list in order to preserve
        # the NEURON hoc objects and the corresponding python references
        # initialized by _ArtificialCell()
        self._drive_cells = list()

        self.ncs = dict()
        self._nrn_dipoles = dict()

        self._vsec = dict()
        self._isec = dict()
        self._ca = dict()
        self._nrn_rec_arrays = dict()
        self._nrn_rec_callbacks = list()

        # if extracellular electrodes have been included, we need to calculate
        # transmembrane currents at each integration step
        self._expose_imem = False
        if len(self.net.rec_arrays) > 0:
            self._expose_imem = True

        self._rank = 0

        self._build()

    def _build(self):
        """Building the network in NEURON."""

        global _CVODE, _PC
        _create_parallel_context(expose_imem=self._expose_imem)

        self._rank = _get_rank()

        # load mechanisms needs ParallelContext for get_rank
        load_custom_mechanisms()

        if self._rank == 0:
            print("Building the NEURON model")

        self._clear_last_network_objects()

        self._nrn_dipoles["L5_pyramidal"] = h.Vector()
        self._nrn_dipoles["L2_pyramidal"] = h.Vector()

        self._gid_assign()

        record_vsec = self.net._params["record_vsec"]
        record_isec = self.net._params["record_isec"]
        record_ca = self.net._params["record_ca"]
        self._create_cells_and_drives(
            threshold=self.net._params["threshold"],
            record_vsec=record_vsec,
            record_isec=record_isec,
            record_ca=record_ca,
        )

        self.state_init()

        # set to record spikes, somatic voltages, and extracellular potentials
        self._spike_times = h.Vector()
        self._spike_gids = h.Vector()

        # used by rank 0 for spikes across all procs (MPI)
        self._all_spike_times = h.Vector()
        self._all_spike_gids = h.Vector()

        self._record_spikes()
        self._connect_celltypes()

        if len(self.net.rec_arrays) > 0:
            self._record_extracellular()

        if self._rank == 0:
            print("[Done]")

    def _gid_assign(self, rank=None, n_hosts=None):
        """Assign cell IDs to this node

        Parameters
        ----------
        rank : int | None
            If not None, override the rank set
            automatically using Neuron. Used for testing.
        n_hosts : int | None
            If not None, override the number of hosts set
            automatically using Neuron. Used for testing.
        """
        if rank is not None:
            self._rank = rank
        if n_hosts is None:
            n_hosts = _get_nhosts()

        # round robin assignment of cell gids
        for gid in range(self._rank, self.net._n_cells, n_hosts):
            self._gid_list.append(gid)

        for drive in self.net.external_drives.values():
            if drive["cell_specific"]:
                # only assign drive gids that have a target cell gid already
                # assigned to this rank
                for src_gid in self.net.gid_ranges[drive["name"]]:
                    conn_idxs = pick_connection(self.net, src_gids=src_gid)
                    target_gids = set()
                    for conn_idx in conn_idxs:
                        gid_pairs = self.net.connectivity[conn_idx]["gid_pairs"]
                        if src_gid in gid_pairs:
                            target_gids.update(
                                self.net.connectivity[conn_idx]["gid_pairs"][src_gid]
                            )

                    for target_gid in target_gids:
                        if (
                            target_gid in self._gid_list
                            and src_gid not in self._gid_list
                        ):
                            self._gid_list.append(src_gid)
            else:
                # round robin assignment of drive gids
                src_gids = list(self.net.gid_ranges[drive["name"]])
                for gid_idx in range(self._rank, len(src_gids), n_hosts):
                    self._gid_list.append(src_gids[gid_idx])

        # extremely important to get the gids in the right order
        self._gid_list.sort()

    def _create_cells_and_drives(
        self, threshold, record_vsec=False, record_isec=False, record_ca=False
    ):
        """Parallel create cells AND external drives

        NB: _Cell.__init__ calls h.Section -> non-picklable!
        NB: _ArtificialCell.__init__ calls h.*** -> non-picklable!

        These drives are spike SOURCES but cells are also targets.
        External inputs are not targets.
        """

        for gid in self._gid_list:
            _PC.set_gid2node(gid, self._rank)

        # loop through ALL gids
        # have to loop over self._gid_list, since this is what we got
        # on this rank (MPI)
        for gid in self._gid_list:
            src_type = self.net.gid_to_type(gid)
            gid_idx = gid - self.net.gid_ranges[src_type][0]
            if src_type in self.net.cell_types:
                # copy cell object from template cell type in Network
                cell = self.net.cell_types[src_type].copy()
                cell.gid = gid
                cell.pos = self.net.pos_dict[src_type][gid_idx]

                # instantiate NEURON object
                if src_type in ("L2_pyramidal", "L5_pyramidal"):
                    cell.build(sec_name_apical="apical_trunk")
                else:
                    cell.build()
                # add tonic biases
                for bias in self.net.external_biases:
                    if src_type in self.net.external_biases[bias]:
                        cell.create_tonic_bias(
                            **self.net.external_biases[bias][src_type]
                        )
                cell.record(record_vsec, record_isec, record_ca)

                # this call could belong in init of a _Cell (with threshold)?
                nrn_netcon = cell.setup_source_netcon(threshold)
                assert cell.gid in self._gid_list
                _PC.cell(cell.gid, nrn_netcon)
                self._cells.append(cell)

            # external driving inputs are special types of artificial-cells
            else:
                event_times = self.net.external_drives[src_type]["events"][
                    self.trial_idx
                ][gid_idx]
                drive_cell = _ArtificialCell(event_times, threshold, gid=gid)
                _PC.cell(drive_cell.gid, drive_cell.nrn_netcon)
                self._drive_cells.append(drive_cell)

    # connections:
    # this NODE is aware of its cells as targets
    # for each syn, return list of source GIDs.
    # for each item in the list, do a:
    # nc = pc.gid_connect(source_gid, target_syn), weight,delay
    # Both for synapses AND for external inputs
    def _connect_celltypes(self):
        """Connect two cell types for a particular receptor."""
        net = self.net
        connectivity = self.net.connectivity

        assert len(self._cells) == len(self._gid_list) - len(self._drive_cells)

        for conn in connectivity:
            loc, receptor = conn["loc"], conn["receptor"]
            nc_dict = deepcopy(conn["nc_dict"])
            nc_dict["A_weight"] *= nc_dict["gain"]
            # Gather indices of targets on current node
            valid_targets = set()
            for src_gid, target_gids in conn["gid_pairs"].items():
                filtered_targets = list()
                for target_gid in target_gids:
                    if _PC.gid_exists(target_gid):
                        filtered_targets.append(target_gid)
                        valid_targets.add(target_gid)
                conn["gid_pairs"][src_gid] = filtered_targets

            target_filter = dict()
            for idx in range(len(self._cells)):
                gid = self._gid_list[idx]
                if gid in valid_targets:
                    target_filter[gid] = idx

            # Iterate over src/target pairs and connect cells
            for src_gid, target_gids in conn["gid_pairs"].items():
                for target_gid in target_gids:
                    src_type = self.net.gid_to_type(src_gid)
                    target_type = self.net.gid_to_type(target_gid)
                    target_cell = self._cells[target_filter[target_gid]]
                    connection_name = (
                        f"{_short_name(src_type)}_{_short_name(target_type)}_{receptor}"
                    )
                    if connection_name not in self.ncs:
                        self.ncs[connection_name] = list()
                    pos_idx = src_gid - net.gid_ranges[_long_name(src_type)][0]
                    # NB pos_dict for this drive must include ALL cell types!
                    nc_dict["pos_src"] = net.pos_dict[_long_name(src_type)][pos_idx]

                    # get synapse locations
                    syn_keys = list()
                    # Targeting group of sections like proximal or distal
                    if loc in target_cell.sect_loc:
                        for sect in target_cell.sect_loc[loc]:
                            syn_keys.append(f"{sect}_{receptor}")
                    # Targeting individual section like soma or apical_tuft
                    else:
                        syn_keys = [f"{loc}_{receptor}"]

                    for syn_key in syn_keys:
                        nc = target_cell.parconnect_from_src(
                            src_gid,
                            deepcopy(nc_dict),
                            target_cell._nrn_synapses[syn_key],
                            net._inplane_distance,
                        )
                        self.ncs[connection_name].append(nc)

    def _record_extracellular(self):
        for arr_name, arr in self.net.rec_arrays.items():
            nrn_arr = _ExtracellularArrayBuilder(arr)
            nrn_arr._build(cvode=_CVODE)
            self._nrn_rec_arrays.update({arr_name: nrn_arr})

    def _record_spikes(self):
        """Setup spike recording for this node"""
        # iterate through gids on this node and
        # set to record spikes in spike time vec and id vec
        # agnostic to type of source, will sort that out later
        for gid in self._gid_list:
            if _PC.gid_exists(gid):
                _PC.spike_record(gid, self._spike_times, self._spike_gids)

    def aggregate_data(self, n_samples):
        """Aggregate somatic currents, voltages, and dipoles.

        Parameters
        ----------
        n_samples : int
            Number of samples contained in continuous data types (e.g.,
            current dipole and somatic voltage).

        Notes
        -----
        Specifying ``n_samples`` ensures that certain NEURON data objects
        (e.g., h.Vector()) are congruent in shape and can thus be reduced
        across all MPI ranks when using ``MPIBackend``.
        """
        # ensure that the shape of this rank's nrn_dpl h.Vector() object is
        # initialized consistently across all MPI ranks regardless of whether
        # this rank contains cells contributing to the net dipole calculation
        for nrn_dpl in self._nrn_dipoles.values():
            if nrn_dpl.size() != n_samples:
                nrn_dpl.append(h.Vector(n_samples, 0))

        for cell in self._cells:
            # add dipoles across neurons on the current thread
            if hasattr(cell, "dipole"):
                if cell.dipole.size() != n_samples:
                    raise ValueError(
                        f"n_samples does not match the size "
                        f"of at least one cell's dipole vector. "
                        f"Got n_samples={n_samples}, {cell.name}."
                        f"dipole.size()={cell.dipole.size()}."
                    )
                nrn_dpl = self._nrn_dipoles[_long_name(cell.name)]
                nrn_dpl.add(cell.dipole)

            self._vsec[cell.gid] = cell.vsec
            self._isec[cell.gid] = cell.isec
            self._ca[cell.gid] = cell.ca

        # reduce across threads
        for nrn_dpl in self._nrn_dipoles.values():
            _PC.allreduce(nrn_dpl, 1)
        for nrn_arr in self._nrn_rec_arrays.values():
            _PC.allreduce(nrn_arr._nrn_voltages, 1)

        # aggregate the currents and voltages independently on each proc
        vsec_list = _PC.py_gather(self._vsec, 0)
        isec_list = _PC.py_gather(self._isec, 0)
        ca_list = _PC.py_gather(self._ca, 0)

        # combine spiking data from each proc
        spike_times_list = _PC.py_gather(self._spike_times, 0)
        spike_gids_list = _PC.py_gather(self._spike_gids, 0)

        # only rank 0's lists are complete
        if _get_rank() == 0:
            for spike_vec in spike_times_list:
                self._all_spike_times.append(spike_vec)
            for spike_vec in spike_gids_list:
                self._all_spike_gids.append(spike_vec)
            for vsec in vsec_list:
                self._vsec.update(vsec)
            for isec in isec_list:
                self._isec.update(isec)
            for ca in ca_list:
                self._ca.update(ca)

        _PC.barrier()  # get all nodes to this place before continuing

    def state_init(self):
        """Initializes the state closer to baseline."""

        for cell in self._cells:
            seclist = h.SectionList()
            seclist.wholetree(sec=cell._nrn_sections["soma"])
            for sect in seclist:
                for seg in sect:
                    if cell.name == "L2Pyr":
                        seg.v = -71.46
                    elif cell.name == "L5Pyr":
                        if sect.name() == "L5Pyr_apical_1":
                            seg.v = -71.32
                        elif sect.name() == "L5Pyr_apical_2":
                            seg.v = -69.08
                        elif sect.name() == "L5Pyr_apical_tuft":
                            seg.v = -67.30
                        else:
                            seg.v = -72.0
                    elif cell.name == "L2Basket":
                        seg.v = -64.9737
                    elif cell.name == "L5Basket":
                        seg.v = -64.9737

    def _clear_neuron_objects(self):
        """Clear up NEURON internal gid and reference information.

        Note: This function must be called from the context of the
        Network instance that ran `_build`. This is a bug or
        peculiarity of NEURON. If this function is called from a different
        context, then the next simulation will run very slow because nrniv
        workers are still going for the old simulation. If pc.gid_clear is
        called from the right context, then those workers can exit.
        """

        _PC.gid_clear()

        # dereference cell and NetConn objects
        for gid, cell in zip(self._gid_list, self._cells):
            # only work on cells on this node
            if _PC.gid_exists(gid):
                for nc_key in self.ncs:
                    for nc in self.ncs[nc_key]:
                        if nc.valid():
                            # delete NEURON cell object
                            cell_obj1 = nc.precell(gid)
                            if cell_obj1 is not None:
                                del cell_obj1
                            cell_obj2 = nc.postcell(gid)
                            if cell_obj2 is not None:
                                del cell_obj2
                            del nc

        self._gid_list = list()
        self._cells = list()
        self._drive_cells = list()

        # NB needed if multiple simulations are run in same python proc.
        # removes callbacks used to gather transmembrane currents
        for nrn_arr in self._nrn_rec_arrays.values():
            if nrn_arr._recording_callback is not None:
                _CVODE.extra_scatter_gather_remove(nrn_arr._recording_callback)

    def _clear_last_network_objects(self):
        """Clears NEURON objects and saves the current Network instance"""

        global _LAST_NETWORK

        if _LAST_NETWORK is not None:
            _LAST_NETWORK._clear_neuron_objects()

        self._clear_neuron_objects()
        _LAST_NETWORK = self
