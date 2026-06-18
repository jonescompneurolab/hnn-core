"""Objective functions for parameter optimization."""

# Authors: Carolina Fernandez <cxf418@miami.edu>
#          Nick Tolley <nicholas_tolley@brown.edu>
#          Ryan Thorpe <ryan_thorpe@brown.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

import numpy as np
from scipy.signal import periodogram

from hnn_core import simulate_dipole
from ..dipole import _rmse, _anticorr, average_dipoles
from ..batch_simulate import BatchSimulate


def _check_is_batch(predicted_params):
    """Check predicted params dimensionality, dim=2 indicates batch simulation"""
    if len(np.array(predicted_params).shape) == 1:
        is_batch = False
    elif len(np.array(predicted_params).shape) == 2:
        is_batch = True
    else:
        raise ValueError(
            f"Incorrect shape for predicted params. Got {np.array(predicted_params).shape}"
        )
    return is_batch


def _preprocess_dipole(dpls, obj_fun_kwargs):
    """Apply smooth and scale preprocessing"""
    if "scale_factor" in obj_fun_kwargs:
        [dpl.scale(obj_fun_kwargs["scale_factor"]) for dpl in dpls]
    if "smooth_window_len" in obj_fun_kwargs:
        [dpl.smooth(obj_fun_kwargs["smooth_window_len"]) for dpl in dpls]


def _get_relative_power(dpl, obj_fun_kwargs):
    # get psd of simulated dpl
    freqs_simulated, psd_simulated = periodogram(
        dpl.data["agg"], dpl.sfreq, window="hamming"
    )

    # for each f band
    f_bands_psds = list()
    relative_bandpower = obj_fun_kwargs["relative_bandpower"]

    # Handle float and list inputs for relative_bandpower
    if isinstance(relative_bandpower, float):
        relative_bandpower = [relative_bandpower] * len(obj_fun_kwargs["f_bands"])
    elif len(relative_bandpower) != len(obj_fun_kwargs["f_bands"]):
        raise ValueError("Length of relative_bandpower must match length of f_bands.")

    for idx, f_band in enumerate(obj_fun_kwargs["f_bands"]):
        f_band_idx = np.where(
            np.logical_and(freqs_simulated >= f_band[0], freqs_simulated <= f_band[1])
        )[0]
        f_bands_psds.append(relative_bandpower[idx] * sum(psd_simulated[f_band_idx]))

    # The optimizer is designed to minimize the objective function.
    # Maximizing the relative band power is equivalent to minimizing its negative.
    obj = -sum(f_bands_psds) / sum(psd_simulated)
    return obj


def _calculate_obj_fun(
    obj_fun_lambda,
    initial_net,
    initial_params,
    set_params,
    predicted_params,
    update_params,
    tstop,
    obj_values,
    obj_fun_kwargs,
    best=None,
):
    """Run simulations, calculate your objective function, and return its value(s).

    Parameters
    ----------
    obj_fun_lambda : callable
        The objective function that you want to apply. It must take a single ``Dipole``
        object as its argument. It can be a lambda that includes usage of other functions (see the lambdas of the functions that call this for examples).
    initial_net : instance of Network
        The network object.
    initial_params : dict
        Keys are parameter names, values are initial parameters.
    set_params : func
        User-defined function that sets network drives and parameters.
    predicted_params : list
        Parameters selected by the optimizer.
    update_params : func
        Function to update params.
    obj_values : list
        List to store objective function values.
    tstop : float
        The simulated dipole's duration.
    obj_fun_kwargs : dict
        A kwargs-style dictionary that contains additional arguments for this particular
        objective function and/or a particular solver. See `Optimizer.fit` for more
        details. The key-value pairs specific to this objective function are:

        target : instance of Dipole
            Required. A dipole object with experimental data.
        n_trials : int, default=1
            Number of trials to simulate and average.

    best : dict, optional
        Dictionary with keys "obj" and "params" to store the best objective value and
        corresponding parameters. Note that `best` will be updated as a "side-effect"
        (similar to "pass-by-reference"), and is not returned by the function; this is
        necessary because the optimization routines in `scipy.optimize` require the
        objective functions to return a single scalar value. Only used if the solver is
        set to "cobyla" or "cma".

    Returns
    -------
    obj : float
        Normalized RMSE between recorded and simulated dipole.
    """
    is_batch = _check_is_batch(predicted_params)

    if is_batch:
        # The "batch" case only occurs if the solver is set to "cma"
        predicted_params = np.array(predicted_params).reshape(-1, len(initial_params))
        print(predicted_params.shape)
        params_batch = {
            name: predicted_params[:, idx]
            for idx, name in enumerate(initial_params.keys())
        }

        # simulate dpl with predicted params
        new_net = initial_net.copy()

        batch_simulation = BatchSimulate(
            net=new_net,
            set_params=set_params,
            save_outputs=False,
            save_dpl=True,
            dt=obj_fun_kwargs.get("dt", 0.025),
            n_trials=obj_fun_kwargs.get("n_trials", 1),
            tstop=tstop,
            overwrite=False,
            clear_cache=False,
        )

        res = batch_simulation.run(
            params_batch,
            n_jobs=obj_fun_kwargs.get("n_jobs", 1),
            combinations=False,
            backend="loky",
            verbose=obj_fun_kwargs.get("verbose", True),
        )

        dpls = list()
        for batch_res in res["simulated_data"]:
            for data in batch_res:
                # smooth & scale all dipoles
                _preprocess_dipole(data["dpl"], obj_fun_kwargs)
                # average dipoles per population
                dpls.append(average_dipoles(data["dpl"]))

        obj = [obj_fun_lambda(dpl) for dpl in dpls]

    else:
        # The non-"batch" case occurs if the solver is set to "cobyla" or "bayesian"
        params = update_params(initial_params, predicted_params)

        # simulate dpl with predicted params
        new_net = initial_net.copy()
        set_params(new_net, params)

        dpls = simulate_dipole(
            new_net,
            tstop=tstop,
            dt=obj_fun_kwargs.get("dt", 0.025),
            n_trials=obj_fun_kwargs.get("n_trials", 1),
        )

        # smooth & scale all dipoles
        _preprocess_dipole(dpls, obj_fun_kwargs)

        dpl = average_dipoles(dpls)
        obj = obj_fun_lambda(dpl)

        # Update best params; this is a "side-effect" that changes the `best` dictionary
        # in-place in the parent scope
        if best is not None and obj < best["obj"]:
            best["obj"] = obj
            best["params"] = predicted_params.copy()

    print(f"Mean Loss: {np.mean(obj):.2f}; Min Loss: {np.min(obj):.2f}")
    # Update the store of objective function values via a "side-effect" in-place in the
    # parent scope
    obj_values.append(obj)

    return obj


def _rmse_evoked(
    initial_net,
    initial_params,
    set_params,
    predicted_params,
    update_params,
    obj_values,
    tstop,
    obj_fun_kwargs,
    best=None,
):
    """The objective function for evoked responses.

    Parameters
    ----------
    initial_net : instance of Network
        The network object.
    initial_params : dict
        Keys are parameter names, values are initial parameters.
    set_params : func
        User-defined function that sets network drives and parameters.
    predicted_params : list
        Parameters selected by the optimizer.
    update_params : func
        Function to update params.
    obj_values : list
        List of objective values for each epoch (aka iteration) during optimization.
        Updated as a side effect of evaluating the objective function.
    tstop : float
        The simulated dipole's duration.
    obj_fun_kwargs : dict
        Additional arguments along with their respective values to be passed
        to the objective function (see ``Optimizer.fit`` for more details):

        target : instance of Dipole (Required)
            A dipole object with experimental data.
        n_trials : int, optional
            Number of trials to simulate and average.
        verbose : bool, optional
            If True, print build steps and simulation progress to console. Default: True.

    Returns
    -------
    obj : float
        Normalized RMSE between recorded and simulated dipole.
    """
    obj = _calculate_obj_fun(
        lambda dpl: _rmse(dpl, obj_fun_kwargs["target"], tstop=tstop),
        initial_net,
        initial_params,
        set_params,
        predicted_params,
        update_params,
        tstop,
        obj_values,
        obj_fun_kwargs,
        best,
    )
    obj_values.append(obj)
    return obj


def _maximize_psd(
    initial_net,
    initial_params,
    set_params,
    predicted_params,
    update_params,
    obj_values,
    tstop,
    obj_fun_kwargs,
    best=None,
):
    """The objective function for PSDs.

    Parameters
    ----------
    initial_net : instance of Network
        The network object.
    initial_params : dict
        Keys are parameter names, values are initial parameters.
    set_params : func
        User-defined function that sets network drives and parameters.
    predicted_params : list
        Parameters selected by the optimizer.
    update_params : func
        Function to update params.
    obj_values : list
        List of objective values for each epoch (aka iteration) during optimization.
        Updated as a side effect of evaluating the objective function.
    tstop : float
        The simulated dipole's duration.
    obj_fun_kwargs : dict
        A kwargs-style dictionary that contains additional arguments for this particular
        objective function and/or a particular solver. See `Optimizer.fit` for more
        details. The key-value pairs specific to this objective function are:

        f_bands : list of tuples (Required)
            Lower and higher limit for each frequency band.
        relative_bandpower : list of float | float (Required)
            Weight for each frequency band in f_bands. If a single float is provided,
            the same weight is applied to all frequency bands.
        verbose : bool, optional
            If True, print build steps and simulation progress to console. Default: True.

    best : dict, optional
        Dictionary with keys "obj" and "params" to store the best objective value and
        corresponding parameters. Note that `best` will be updated as a "side-effect"
        (similar to "pass-by-reference"), and is not returned by the function; this is
        necessary because the optimization routines in `scipy.optimize` require the
        objective functions to return a single scalar value. Only used if the solver is
        set to "cobyla" or "cma".

    Returns
    -------
    obj : float
        Sum of the weighted frequency band PSDs relative to total signal PSD.

    Notes
    -----
    The objective function minimizes the sum of the weighted (user-defined) frequency
    band PSDs (user-defined) relative to the total PSD of the signal. The objective
    function can be represented as -Σc[ΣPSD(i)/ΣPSD(j)] where c is the weight for each
    frequency band, PSD(i) is the PSD for each frequency band, and PSD(j) is the total
    PSD of the signal.
    """
    obj = _calculate_obj_fun(
        lambda dpl: _get_relative_power(dpl, obj_fun_kwargs),
        initial_net,
        initial_params,
        set_params,
        predicted_params,
        update_params,
        tstop,
        obj_values,
        obj_fun_kwargs,
        best,
    )
    obj_values.append(obj)
    return obj


def _anticorr_evoked(
    initial_net,
    initial_params,
    set_params,
    predicted_params,
    update_params,
    obj_values,
    tstop,
    obj_fun_kwargs,
    best=None,
):
    """The objective function for evoked responses.

    Parameters
    ----------
    initial_net : instance of Network
        The network object.
    initial_params : dict
        Keys are parameter names, values are initial parameters.
    set_params : func
        User-defined function that sets network drives and parameters.
    predicted_params : list
        Parameters selected by the optimizer.
    update_params : func
        Function to update params.
    obj_values : list
        List of objective values for each epoch (aka iteration) during optimization.
        Updated as a side effect of evaluating the objective function.
    tstop : float
        The simulated dipole's duration.
    obj_fun_kwargs : dict
        A kwargs-style dictionary that contains additional arguments for this particular
        objective function and/or a particular solver. See `Optimizer.fit` for more
        details. The key-value pairs specific to this objective function are:

        target : instance of Dipole (Required)
            A dipole object with experimental data.
        n_trials : int, default=1
            Number of trials to simulate and average.
        verbose : bool, optional
            If True, print build steps and simulation progress to console. Default: True.

    best : dict, optional
        Dictionary with keys "obj" and "params" to store the best objective value and
        corresponding parameters. Note that `best` will be updated as a "side-effect"
        (similar to "pass-by-reference"), and is not returned by the function; this is
        necessary because the optimization routines in `scipy.optimize` require the
        objective functions to return a single scalar value. Only used if the solver is
        set to "cobyla" or "cma".

    Returns
    -------
    obj : float
        Anticorrelation between recorded and simulated dipole.
    """
    obj = _calculate_obj_fun(
        lambda dpl: _anticorr(dpl, obj_fun_kwargs["target"], tstop=tstop),
        initial_net,
        initial_params,
        set_params,
        predicted_params,
        update_params,
        tstop,
        obj_values,
        obj_fun_kwargs,
        best,
    )
    obj_values.append(obj)
    return obj


def _custom_objective_function(
    initial_net,
    initial_params,
    set_params,
    predicted_params,
    update_params,
    obj_values,
    tstop,
    obj_fun_kwargs,
    best=None,
):
    """The generic objective function for user-defined loss functions.

    Parameters
    ----------
    initial_net : instance of Network
        The network object.
    initial_params : dict
        Keys are parameter names, values are initial parameters.
    set_params : func
        User-defined function that sets network drives and parameters.
    predicted_params : list
        Parameters selected by the optimizer.
    update_params : func
        Function to update params.
    obj_values : list
        List of objective values for each epoch (aka iteration) during optimization.
        Updated as a side effect of evaluating the objective function.
    tstop : float
        The simulated dipole's duration.
    obj_fun_kwargs : dict
        Additional arguments to pass to the objective function, must contain
        'loss_fun', a callable.

    Returns
    -------
    obj : float
        The loss value returned by the user-defined loss function.
    """
    obj = _calculate_obj_fun(
        lambda dpl: obj_fun_kwargs["loss_fun"](dpl, obj_fun_kwargs),
        initial_net,
        initial_params,
        set_params,
        predicted_params,
        update_params,
        tstop,
        obj_values,
        obj_fun_kwargs,
        best,
    )
    obj_values.append(obj)
    return obj
