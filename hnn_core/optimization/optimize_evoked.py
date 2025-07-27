"""Parameter optimization functions."""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

from math import ceil, floor
from collections import OrderedDict
import fnmatch
from functools import partial

import numpy as np
import scipy.stats as stats
from scipy.optimize import fmin_cobyla

from ..dipole import simulate_dipole, _rmse, average_dipoles
from ..network import pick_connection


def _get_range(val, multiplier):
    """Get range of values to sweep over."""
    range_min = max(0, val - val * multiplier / 100.0)
    range_max = val + val * multiplier / 100.0
    ranges = {"initial": val, "minval": range_min, "maxval": range_max}
    return ranges


def _split_by_evinput(
    drive_names,
    drive_dynamics,
    drive_syn_weights,
    tstop,
    sigma_range_multiplier,
    timing_range_multiplier,
    synweight_range_multiplier,
):
    """Sorts parameter ranges by evoked inputs into a dictionary

    Parameters
    ----------
    drive_names : list of str, shape (n_drives, )
        Names corresponding to n Network drives.
    drive_dynamics : list of dict, shape (n_drives, )
        Dynamics parameters for each drive specified in drive_names.
    drive_syn_weights : list of dict, shape (n_drives, )
        Synaptic weight parameters for each drive specified in drive_names.
    tstop : float
        The simulation stop time (ms).
    sigma_range_multiplier : float
        The scale of sigma values to sweep over.
    timing_range_multiplier : float
        The scale of timing values to sweep over.
    synweight_range_multiplier : float
        The scale of input synaptic weights to sweep over.

    Returns
    -------
    sorted_evinput_params: dict
        Dictionary with parameters grouped by evoked inputs.
        Keys for each evoked input are
        'mean', 'sigma', 'ranges', 'start', and 'end'.
        sorted_evinput_params['evprox1']['ranges'] is a dict
        with keys as the parameters to be optimized and values
        indicating the ranges over which they should be
        optimized. E.g.,
        sorted_evinput['evprox1']['ranges'] =
        {
            'gbar_evprox_1_L2Pyr_ampa':
                {
                    'initial': 0.05125,
                    'minval': 0,
                    'maxval': 0.0915
                }
        }
        Elements are sorted by their start time.
    """

    evinput_params = {}
    for drive_idx, drive_name in enumerate(drive_names):
        timing_mean = drive_dynamics[drive_idx]["mu"]
        timing_sigma = drive_dynamics[drive_idx]["sigma"]

        if timing_sigma == 0.0:
            # sigma of 0 will not produce a CDF
            timing_sigma = 0.01

        evinput_params[drive_name] = {
            "mean": timing_mean,
            "sigma": timing_sigma,
            "ranges": {},
        }

        evinput_params[drive_name]["ranges"][f"{drive_name}_sigma"] = _get_range(
            timing_sigma, sigma_range_multiplier
        )

        # calculate range for time
        timing_bound = timing_sigma * timing_range_multiplier
        range_min = max(0, timing_mean - timing_bound)
        range_max = min(tstop, timing_mean + timing_bound)

        evinput_params[drive_name]["start"] = range_min
        evinput_params[drive_name]["end"] = range_max
        evinput_params[drive_name]["ranges"][f"{drive_name}_mu"] = {
            "initial": timing_mean,
            "minval": range_min,
            "maxval": range_max,
        }

        # calculate ranges for syn. weights
        for syn_weight_key in drive_syn_weights[drive_idx]:
            new_key = f"{drive_name}_gbar_{syn_weight_key}"
            weight = drive_syn_weights[drive_idx][syn_weight_key]
            ranges = _get_range(weight, synweight_range_multiplier)
            if weight == 0.0:
                ranges["minval"] = weight
                ranges["maxval"] = 1.0
            evinput_params[drive_name]["ranges"][new_key] = ranges

    sorted_evinput_params = OrderedDict(
        sorted(evinput_params.items(), key=lambda x: x[1]["start"])
    )
    return sorted_evinput_params


def _generate_weights(evinput_params, tstop, dt, decay_multiplier):
    """Calculation of weight function for wRMSE calculation

    Returns
    -------
    evinput_params : dict
        Adds the keys 'weights', 'opt_start', 'opt_end'
        to evinput_params[input_name] and removes 'mean'
        and 'sigma' which were needed to compute 'weights'.
    """
    num_step = ceil(tstop / dt) + 1
    times = np.linspace(0, tstop, num_step)

    for evinput_this in evinput_params.values():
        # calculate cdf using start time (minival of optimization range)
        evinput_this["cdf"] = stats.norm.cdf(
            times, evinput_this["start"], evinput_this["sigma"]
        )

    for input_name, evinput_this in evinput_params.items():
        evinput_this["weights"] = evinput_this["cdf"].copy()

        for other_input, evinput_other in evinput_params.items():
            # check ordering to only use inputs after us
            # and don't subtract our own cdf(s)
            if (
                evinput_other["mean"] < evinput_this["mean"]
                or input_name == other_input
            ):
                continue

            decay_factor = (
                decay_multiplier
                * (evinput_other["mean"] - evinput_this["mean"])
                / tstop
            )
            evinput_this["weights"] -= evinput_other["cdf"] * decay_factor

        # weights should not drop below 0
        np.clip(
            evinput_this["weights"], a_min=0, a_max=None, out=evinput_this["weights"]
        )

        # start and stop optimization where the weights are insignificant
        indices = np.where(evinput_this["weights"] > 0.01)
        evinput_this["opt_start"] = min(evinput_this["start"], times[indices][0])
        evinput_this["opt_end"] = max(evinput_this["end"], times[indices][-1])

        # convert to multiples of dt
        evinput_this["opt_start"] = floor(evinput_this["opt_start"] / dt) * dt
        evinput_params[input_name]["opt_end"] = ceil(evinput_this["opt_end"] / dt) * dt

    for evinput_this in evinput_params.values():
        del evinput_this["mean"], evinput_this["sigma"], evinput_this["cdf"]

    return evinput_params


def _create_last_chunk(input_chunks):
    """This creates a chunk that combines parameters for
    all chunks in input_chunks (final step)

    Parameters
    ----------
    input_chunks: List
        List of ordered chunks for optimization

    Returns
    -------
    chunk: dict
        Dictionary of with parameters for combined
        chunk (final step)
    """
    chunk = {"inputs": [], "ranges": {}, "opt_start": 0.0, "opt_end": 0.0}

    for evinput in input_chunks:
        chunk["inputs"].extend(evinput["inputs"])
        chunk["ranges"].update(evinput["ranges"])
        if evinput["opt_end"] > chunk["opt_end"]:
            chunk["opt_end"] = evinput["opt_end"]

    # wRMSE with weights of 1's is the same as regular RMSE.
    chunk["weights"] = np.ones_like(input_chunks[-1]["weights"])

    return chunk


def _consolidate_chunks(inputs):
    """Consolidates inputs into optimization "chunks" defined by
    opt_start and opt_end

    Parameters
    ----------
    inputs: dict
        Sorted dictionary of inputs with their parameters
        and weight functions

    Returns
    -------
    chunks: list
        Combine the evinput_params whenever the end is overlapping
        with the next.
    """
    chunks = list()
    for input_name in inputs:
        input_dict = inputs[input_name].copy()
        input_dict["inputs"] = [input_name]

        if len(chunks) > 0 and input_dict["start"] <= chunks[-1]["end"]:
            # update previous chunk
            chunks[-1]["inputs"].extend(input_dict["inputs"])
            chunks[-1]["end"] = input_dict["end"]
            chunks[-1]["ranges"].update(input_dict["ranges"])
            chunks[-1]["opt_end"] = max(chunks[-1]["opt_end"], input_dict["opt_end"])
            # average the weights
            chunks[-1]["weights"] = (chunks[-1]["weights"] + input_dict["weights"]) / 2
        else:
            # new chunk
            chunks.append(input_dict)

    # add one last chunk to the end
    if len(chunks) > 1:
        last_chunk = _create_last_chunk(chunks)
        chunks.append(last_chunk)

    return chunks


def _optrun(
    drive_params_updated,
    drive_params_static,
    net,
    tstop,
    dt,
    n_trials,
    opt_params,
    opt_dpls,
    scale_factor,
    smooth_window_len,
    return_rmse,
):
    """This is the function to run a simulation

    Parameters
    ----------
    drive_params_updated : array-like, shape (n_params, )
        List or numpy array with the parameters chosen by
        optimization engine. Order is consistent with
        opt_params['ranges'].
    drive_params_static : dict
        Drive parameters that remain constant throughout optimization. Keys
        correspond to the drive names that are being optimized in this chunk.
    net : Network instance
        Network instance with attached drives. This object will be modified
        in-place.
    tstop : float
        The simulation stop time (ms).
    dt : float
        The integration time step (ms) of h.CVode during simulation.
    n_trials : int
        The number of trials to simulate.
    opt_params : dict
        The optimization parameters.
    opt_dpls : dict
        Dictionary with keys 'target_dpl' and 'best' for
        the experimental dipole and best dipole.
    scale_factor : float
        Scales the simulated dipoles by scale_factor to match
        exp_dpl.
    smooth_window_len : int
        The length of the hamming window (in samples) to smooth the
        simulated dipole waveform in each optimization step.
    return_rmse : bool
        Returns list of unweighted RMSEs between data in dpl and exp_dpl
        for each optimization step

    Returns
    -------
    avg_rmse: float
        Weighted RMSE between data in dpl and exp_dpl
    """
    print(
        "Optimization step %d, iteration %d"
        % (opt_params["cur_step"] + 1, opt_params["optiter"] + 1)
    )

    # match parameter values contained in list to their respective key names
    params_dict = dict()
    for param_name, test_value in zip(
        opt_params["ranges"].keys(), drive_params_updated
    ):
        # tiny negative weights are possible. Clip them to 0.
        if test_value < 0:
            test_value = 0
        params_dict[param_name] = test_value

    # modify drives according to the drive names in the current chunk
    for drive_name in opt_params["inputs"]:
        # clear drive and its connectivity
        del net.external_drives[drive_name]
        conn_idxs = pick_connection(net, src_gids=drive_name)
        net.connectivity = [
            conn
            for conn_idx, conn in enumerate(net.connectivity)
            if conn_idx not in conn_idxs
        ]

        # extract syn weights: final weights dicts should have keys that
        # correspond to cell types
        keys_ampa = fnmatch.filter(params_dict.keys(), f"{drive_name}_gbar_ampa_*")
        keys_nmda = fnmatch.filter(params_dict.keys(), f"{drive_name}_gbar_nmda_*")
        weights_ampa = {
            key.lstrip(f"{drive_name}_gbar_ampa_"): params_dict[key]
            for key in keys_ampa
        }
        weights_nmda = {
            key.lstrip(f"{drive_name}_gbar_nmda_"): params_dict[key]
            for key in keys_nmda
        }

        net.add_evoked_drive(
            name=drive_name,
            mu=params_dict[drive_name + "_mu"],
            sigma=params_dict[drive_name + "_sigma"],
            numspikes=drive_params_static[drive_name]["numspikes"],
            location=drive_params_static[drive_name]["location"],
            n_drive_cells=drive_params_static[drive_name]["n_drive_cells"],
            cell_specific=drive_params_static[drive_name]["cell_specific"],
            weights_ampa=weights_ampa,
            weights_nmda=weights_nmda,
            space_constant=drive_params_static[drive_name]["space_constant"],
            synaptic_delays=drive_params_static[drive_name]["synaptic_delays"],
            probability=drive_params_static[drive_name]["probability"],
            event_seed=drive_params_static[drive_name]["event_seed"],
            conn_seed=drive_params_static[drive_name]["conn_seed"],
        )

    # run the simulation
    dpls = simulate_dipole(net, tstop=tstop, dt=dt, n_trials=n_trials)
    # order of operations: scale, smooth, then average
    dpls = [dpl.scale(scale_factor) for dpl in dpls]
    if smooth_window_len is not None:
        dpls = [dpl.smooth(smooth_window_len) for dpl in dpls]
    avg_dpl = average_dipoles(dpls)

    avg_rmse = _rmse(
        avg_dpl,
        opt_dpls["target_dpl"],
        tstart=opt_params["opt_start"],
        tstop=opt_params["opt_end"],
        weights=opt_params["weights"],
    )
    avg_rmse_unweighted = _rmse(
        avg_dpl,
        opt_dpls["target_dpl"],
        tstart=opt_params["opt_start"],
        tstop=tstop,
        weights=None,
    )

    if return_rmse:
        opt_params["iter_avg_rmse"].append(avg_rmse_unweighted)
    opt_params["stepminopterr"] = avg_rmse
    opt_dpls["best_dpl"] = avg_dpl

    print(
        "weighted RMSE: %.2e over range [%3.3f-%3.3f] ms"
        % (avg_rmse, opt_params["opt_start"], opt_params["opt_end"])
    )

    opt_params["optiter"] += 1

    return avg_rmse


def _run_optimization(maxiter, param_ranges, optrun):
    cons = list()
    x0 = list()
    for idx, param_name in enumerate(param_ranges):
        x0.append(param_ranges[param_name]["initial"])
        cons.append(lambda x, idx=idx: param_ranges[param_name]["maxval"] - x[idx])
        cons.append(lambda x, idx=idx: x[idx] - param_ranges[param_name]["minval"])
    result = fmin_cobyla(
        func=optrun,
        cons=cons,
        rhobeg=0.1,
        rhoend=1e-4,
        x0=x0,
        maxfun=maxiter,
        catol=0.0,
    )
    return result


def _get_drive_params(net, drive_names):
    """Get evoked drive parameters from a Network instance."""

    drive_dynamics = list()
    drive_syn_weights = list()
    drive_static_params = dict()
    for drive_name in drive_names:
        drive = net.external_drives[drive_name]
        drive_dynamics.append(drive["dynamics"].copy())
        conn_idxs = pick_connection(net, src_gids=drive_name)
        weights = dict()
        delays = dict()
        probabilities = dict()
        for conn_idx in conn_idxs:
            target_type = net.connectivity[conn_idx]["target_type"]
            target_receptor = net.connectivity[conn_idx]["receptor"]
            weight = net.connectivity[conn_idx]["nc_dict"]["A_weight"]
            # note that for each drive, the weights dict should be unnested
            # across target cell types and receptors for ease-of-use when
            # these values get restructured into a list downstream

            # legacy_mode hack: don't include invalid connections that have
            # been added in Network when legacy_mode=True
            if not (drive["location"] == "distal" and target_type == "L5_basket"):
                if target_receptor == "ampa":
                    weights.update({f"ampa_{target_type}": weight})
                if target_receptor == "nmda":
                    weights.update({f"nmda_{target_type}": weight})
                # delay should be constant across AMPA and NMDA receptor types
                delay = net.connectivity[conn_idx]["nc_dict"]["A_delay"]
                delays.update({target_type: delay})
                # space constant should be constant across drive connections
                space_const = net.connectivity[conn_idx]["nc_dict"]["lamtha"]
                # probability should be constant across AMPA and NMDA receptor
                # types
                probability = net.connectivity[conn_idx]["probability"]
                probabilities.update({target_type: probability})

        drive_syn_weights.append(weights)

        static_params = dict()
        static_params["numspikes"] = drive["dynamics"]["numspikes"]
        static_params["location"] = drive["location"]
        if drive["cell_specific"]:
            static_params["n_drive_cells"] = "n_cells"
        else:
            static_params["n_drive_cells"] = drive["n_drive_cells"]
        static_params["cell_specific"] = drive["cell_specific"]
        static_params["space_constant"] = space_const
        static_params["synaptic_delays"] = delays
        static_params["probability"] = probabilities
        static_params["event_seed"] = drive["event_seed"]
        static_params["conn_seed"] = drive["conn_seed"]
        drive_static_params.update({drive_name: static_params})

    return drive_dynamics, drive_syn_weights, drive_static_params


def optimize_evoked(
    net,
    tstop,
    n_trials,
    target_dpl,
    initial_dpl,
    maxiter=50,
    timing_range_multiplier=3.0,
    sigma_range_multiplier=50.0,
    synweight_range_multiplier=500.0,
    decay_multiplier=1.6,
    scale_factor=1.0,
    smooth_window_len=None,
    dt=0.025,
    which_drives="all",
    return_rmse=False,
):
    """Optimize drives to generate evoked response.

    Parameters
    ----------
    net : Network instance
        An instance of the Network object with attached evoked drives. Timing
        and synaptic weight parameters will be optimized for each attached
        evoked drive. Note that no new drives will be created or old drives
        destroyed.
    tstop : float
        The simulation stop time (ms).
    n_trials : int
        The number of trials to simulate.
    target_dpl : instance of Dipole
        The target experimental dipole.
    initial_dpl : instance of Dipole
        The initial dipole to start the optimization.
    maxiter : int
        The maximum number of simulations to run for optimizing
        one "chunk". Must be at least 12 or greater.
    timing_range_multiplier : float
        The scale of timing values to sweep over.
    sigma_range_multiplier : float
        The scale of sigma values to sweep over.
    synweight_range_multiplier : float
        The scale of input synaptic weights to sweep over.
    decay_multiplier : float
        The decay multiplier.
    scale_factor : float
        Scales the simulated dipoles by scale_factor to match
        target_dpl.
    smooth_window_len : int
        The length of the hamming window (in samples) to smooth the
        simulated dipole waveform in each optimization step.
    dt : float
        The integration time step (ms) of h.CVode during simulation.
    which_drives: 'all' or list
        Evoked drives to optimize. If 'all', will optimize all evoked drives.
        If a subset list of evoked drives, will optimize only the evoked drives in the list.
    return_rmse : bool
        Returns list of unweighted RMSEs between the simulated and experimental dipole
        waveforms for each optimization step

    Returns
    -------
    net : Network instance
        An instance of the Network object with the optimized configuration of
        attached drives.
    iter_avg_rmse : list of float
        Unweighted RMSE between data in dpl and exp_dpl for each iteration. Returned only
        if return_rmse is True

    Notes
    -----
    This optimization protocol utilizes the Constrained Optimization
    By Linear Approximation (COBYLA) method:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_cobyla.html  # noqa
    """

    net = net.copy()

    evoked_drive_names = [
        key
        for key in net.external_drives.keys()
        if net.external_drives[key]["type"] == "evoked"
    ]

    if len(evoked_drive_names) == 0:
        raise ValueError(
            "The current Network instance lacks any evoked "
            "drives. Consider adding drives using "
            "net.add_evoked_drive"
        )
    elif which_drives == "all":
        drive_names = evoked_drive_names
    else:
        drive_names = [
            mydrive
            for mydrive in np.unique(which_drives)
            if mydrive in evoked_drive_names
        ]
    if len(drive_names) == 0:
        raise ValueError(
            "The drives selected to be optimized are not evoked "
            "drives. Optimization works only evoked drives."
        )

    drive_dynamics, drive_syn_weights, drive_static_params = _get_drive_params(
        net, drive_names
    )

    # Create a sorted dictionary with the inputs and parameters
    # belonging to each.
    # Then, calculate the appropriate weight function to be used
    # in RMSE using the CDF defined by the input timing's mean and
    # std. deviation parameters.
    # Lastly, use the weights to define non-overlapping "chunks" of
    # the simulation timeframe to optimize. Chunks are consolidated if
    # more than one input should
    # be optimized at a time.
    evinput_params = _split_by_evinput(
        drive_names,
        drive_dynamics,
        drive_syn_weights,
        tstop,
        sigma_range_multiplier,
        timing_range_multiplier,
        synweight_range_multiplier,
    )
    evinput_params = _generate_weights(evinput_params, tstop, dt, decay_multiplier)
    param_chunks = _consolidate_chunks(evinput_params)

    best_rmse = _rmse(initial_dpl, target_dpl, tstop=tstop)
    opt_dpls = dict(best_dpl=initial_dpl, target_dpl=target_dpl)
    print("Initial RMSE: %.2e" % best_rmse)

    opt_params = dict()

    if return_rmse is True:
        opt_params["iter_avg_rmse"] = list()

    for step in range(len(param_chunks)):
        opt_params["cur_step"] = step
        total_steps = len(param_chunks)

        # param_chunks is the optimization information for all steps.
        # opt_params is a pointer to the params for each step
        opt_params.update(param_chunks[step])

        if maxiter == 0:
            print("Skipping optimization step %d (0 simulations)" % (step + 1))
            continue
        elif maxiter < 12:
            print(
                "'maxiter' must be at least 12 for optimization to run. Increasing 'maxiter' to 12."
            )
            maxiter = 12

        if opt_params["cur_step"] > 0 and opt_params["cur_step"] == total_steps - 1:
            # For the last step (all inputs), recalculate ranges and update
            # param_chunks. If previous optimization steps increased
            # std. dev. this could result in fewer optimization steps as
            # inputs may be deemed too close together and be grouped in a
            # single optimization step.
            #
            # The purpose of the last step (with regular RMSE) is to clean up
            # overfitting introduced by local weighted RMSE optimization.

            evinput_params = _split_by_evinput(
                drive_names,
                drive_dynamics,
                drive_syn_weights,
                tstop,
                sigma_range_multiplier,
                timing_range_multiplier,
                synweight_range_multiplier,
            )
            evinput_params = _generate_weights(
                evinput_params, tstop, dt, decay_multiplier
            )
            param_chunks = _consolidate_chunks(evinput_params)

            # reload opt_params for the last step in case the number of
            # steps was changed by updateoptparams()
            opt_params.update(param_chunks[total_steps - 1])

        print("Starting optimization step %d/%d" % (step + 1, total_steps))

        opt_params["optiter"] = 0
        opt_params["stepminopterr"] = _rmse(
            opt_dpls["best_dpl"],
            opt_dpls["target_dpl"],
            tstart=opt_params["opt_start"],
            tstop=opt_params["opt_end"],
            weights=opt_params["weights"],
        )

        net_opt = net.copy()
        # drive_params_updated must be a list for compatibility with the args
        # in the optimization engine, scipy.optimize.fmin_cobyla
        _myoptrun = partial(
            _optrun,
            drive_params_static=drive_static_params,
            net=net_opt,
            tstop=tstop,
            dt=dt,
            n_trials=n_trials,
            opt_params=opt_params,
            opt_dpls=opt_dpls,
            scale_factor=scale_factor,
            smooth_window_len=smooth_window_len,
            return_rmse=return_rmse,
        )

        print(
            "Optimizing from [%3.3f-%3.3f] ms"
            % (opt_params["opt_start"], opt_params["opt_end"])
        )
        opt_results = _run_optimization(
            maxiter=maxiter, param_ranges=opt_params["ranges"], optrun=_myoptrun
        )
        # tiny negative weights are possible. Clip them to 0.
        opt_results[opt_results < 0] = 0

        # update opt_params for the next round if total rmse decreased
        avg_rmse = _rmse(
            opt_dpls["best_dpl"], opt_dpls["target_dpl"], tstop=tstop, weights=None
        )
        if avg_rmse <= best_rmse:
            best_rmse = avg_rmse
            for var_name, value in zip(opt_params["ranges"], opt_results):
                opt_params["ranges"][var_name]["initial"] = value

            net = net_opt

    print("Final RMSE: %.2e" % best_rmse)

    if return_rmse is True:
        return net, opt_params["iter_avg_rmse"]
    return net
