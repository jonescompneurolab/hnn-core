"""Simulations from the HNN textbook notebooks, for comparing dipoles between
NEURON versions.

Each ``nb_*`` function below replays the simulation steps of one notebook in
~/rep/brn/textbook (the two optimization notebooks are excluded, and
``meta_data_tutorial_documented`` and ``tests/test`` run no simulations),
calling ``run`` wherever the notebook calls ``simulate_dipole``. Plotting,
printing, and anything else that cannot affect a simulation are omitted. All
simulations use ``MPIBackend``, whatever backend the notebook uses, since the
backend does not change the output.

``script_dpl_old.py`` (env ``hc11``, NEURON 8) and ``script_dpl_new.py`` (env
``hc11-nrn9``, NEURON 9) both call ``run_all``, which saves each simulation's
dipoles to ``textbook_dpls/<label>/<notebook>__<simulation>.npz``, along with a
``meta.json`` describing the environment. ``script_dpl_new.py`` then calls
``compare``, which prints a table of differences and saves plots to
``textbook_dpls/plots/``.

The two envs import hnn-core from different checkouts, so ``meta.json`` also
records a hash of every hnn-core source file, and ``compare`` lists any that
differ between the two runs.

One deviation from the notebooks: hnn-data's ``tutorial_networks/gui_defaults.json``
predates the ``zdist_origin`` cell metadata that hnn-core ``master`` requires,
so ``read_network_configuration`` fails on it (and so does the ERP notebook).
``_read_network_configuration`` fills in such missing metadata from hnn-core's
defaults first, which does not change the simulation.
"""

import hashlib
import json
import os.path as op
import platform
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np

import hnn_core
from hnn_core import (
    law_2021_model,
    neymotin_2020_model,
    read_network_configuration,
    read_spikes,
    simulate_dipole,
)
from hnn_core.network import pick_connection
from hnn_core.network_models import add_erp_drives_to_jones_model, default_cell_metadata
from hnn_core.parallel_backends import MPIBackend

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "textbook_data"
OUT_DIR = HERE / "textbook_dpls"
HNN_DATA_URL = "https://raw.githubusercontent.com/jonescompneurolab/hnn-data/main/"
ALPHA_DIR = "workshops/2025-04-09-HNN-online_workshop/alpha_beta_gui_walkthrough/"
GAMMA_DIR = "workshops/2025-04-09-HNN-online_workshop/gamma_gui_walkthrough/"
LAYERS = ("agg", "L2", "L5")
LEGACY_FARADAY = 96485.309


def _data(rel_path):
    """Path to a cached copy of an hnn-data file, downloaded on first use.

    Caching guarantees that the "old" and "new" runs read identical files.
    """
    fname = DATA_DIR / rel_path
    if not fname.exists():
        fname.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(HNN_DATA_URL + rel_path, fname)
    return fname


def _read_network_configuration(fname):
    """``read_network_configuration``, after adding any ``cell_metadata`` keys
    that ``fname`` lacks (e.g. ``zdist_origin``) from ``default_cell_metadata``.
    """
    net_data = json.loads(Path(fname).read_text())
    added = set()
    for cell_type, cell_data in net_data["cell_types"].items():
        cell_metadata = cell_data.get("cell_metadata")
        if cell_metadata is None:  # older format, which hnn-core converts itself
            continue
        for key, value in default_cell_metadata[cell_type].items():
            if key not in cell_metadata:
                cell_metadata[key] = value
                added.add(key)
    if not added:
        return read_network_configuration(fname)

    print(f"{Path(fname).name}: added missing cell metadata {sorted(added)}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        patched_fname = Path(tmp_dir) / Path(fname).name
        patched_fname.write_text(json.dumps(net_data))
        return read_network_configuration(patched_fname)


# ----------------------------------------------------------------------------
# Notebook simulations
# ----------------------------------------------------------------------------
def nb_ch05_erp_api_walkthrough(run):
    """content/05_erps/erp_api_walkthrough.ipynb"""
    sim_kwargs = {"tstop": 170.0, "n_trials": 1, "dt": 0.025}

    net_default = neymotin_2020_model()
    add_erp_drives_to_jones_model(net_default)
    run("default", net_default, **sim_kwargs)

    gui_default = _data("tutorial_networks/gui_defaults.json")
    net_default_gui = _read_network_configuration(gui_default)
    run("default_gui", net_default_gui, **sim_kwargs)

    # net_01: the GUI drives with scaled-up AMPA weights
    net_01 = neymotin_2020_model()
    new_ampa_weights = {
        "evprox1": {"L2_pyramidal": 0.100, "L5_pyramidal": 0.045},
        "evdist1": {"L2_pyramidal": 0.5, "L5_pyramidal": 0.5},
        "evprox2": {"L2_pyramidal": 7.0, "L5_pyramidal": 3.5},
    }
    for name, location in [
        ("evprox1", "proximal"),
        ("evdist1", "distal"),
        ("evprox2", "proximal"),
    ]:
        drive = net_default_gui.external_drives[name]
        weights_ampa = drive["weights_ampa"].copy()
        weights_ampa.update(new_ampa_weights[name])
        net_01.add_evoked_drive(
            name=name,
            mu=drive["dynamics"]["mu"],
            sigma=drive["dynamics"]["sigma"],
            numspikes=1,
            location=location,
            n_drive_cells="n_cells",
            cell_specific=True,
            weights_ampa=weights_ampa,
            weights_nmda=drive["weights_nmda"],
            synaptic_delays=drive["synaptic_delays"],
            event_seed=drive["event_seed"],
        )
    run("net_01_sim_01", net_01, **sim_kwargs)

    update_arrival_times = {"evprox1": 18, "evdist1": 63, "evprox2": 130}
    for name, drive in net_01.external_drives.items():
        drive["dynamics"]["mu"] = update_arrival_times[name]
    run("net_01_sim_02", net_01, **sim_kwargs)

    net_01.external_drives["evdist1"]["dynamics"]["sigma"] = 1.5
    run("net_01_sim_03", net_01, **sim_kwargs)

    net_01.external_drives["evdist1"]["dynamics"]["mu"] = 65
    run("net_01_sim_04", net_01, **sim_kwargs)

    # net_02: the GUI network with altered global synaptic gains
    net_02 = _read_network_configuration(gui_default)
    net_02.set_global_synaptic_gains(e_e=2, e_i=2)
    run("net_02_sim_01", net_02, **sim_kwargs)

    net_02.set_global_synaptic_gains(e_e=1, e_i=1, i_e=0.5)
    run("net_02_sim_02", net_02, **sim_kwargs)


def _set_drive_ampa_weight(net, drive_name, weights_ampa):
    """``set_drive_ampa_weight`` from api_alpha_beta.ipynb."""
    for target_type, weight in weights_ampa.items():
        conn_idxs = pick_connection(
            net, src_gids=drive_name, target_gids=target_type, receptor="ampa"
        )
        for conn_idx in conn_idxs:
            net.connectivity[conn_idx]["nc_dict"]["A_weight"] = weight
        net.external_drives[drive_name]["weights_ampa"][target_type] = weight


def nb_ch06_api_alpha_beta(run):
    """content/06_alpha_beta/api_alpha_beta.ipynb"""
    tstop = 700.0

    def load(fname):
        return _read_network_configuration(_data(ALPHA_DIR + fname))

    run("only_rhythmic_prox", load("OnlyRhythmicProx.json"), tstop=tstop, n_trials=1)
    run("only_rhythmic_dist", load("OnlyRhythmicDist.json"), tstop=tstop, n_trials=1)
    run("alpha_and_beta", load("AlphaAndBeta.json"), tstop=tstop, n_trials=1)
    run(
        "alpha_and_beta_jitter50",
        load("AlphaAndBetaJitter50.json"),
        tstop=tstop,
        n_trials=10,
    )

    net_incr_beta = load("AlphaAndBeta.json")
    net_incr_beta.external_drives["bursty2"]["dynamics"]["burst_std"] = 10.0
    _set_drive_ampa_weight(
        net_incr_beta, "bursty2", {"L2_pyramidal": 6e-5, "L5_pyramidal": 6e-5}
    )
    run("incr_beta", net_incr_beta, tstop=tstop, n_trials=1)

    net_alpha_only = load("AlphaAndBeta.json")
    net_alpha_only.external_drives["bursty1"]["dynamics"]["tstart"] = 100.0
    run("alpha_only", net_alpha_only, tstop=tstop, n_trials=1)

    net_high_freq = load("AlphaAndBeta.json")
    _set_drive_ampa_weight(
        net_high_freq, "bursty2", {"L2_pyramidal": 4e-4, "L5_pyramidal": 4e-4}
    )
    run("high_freq", net_high_freq, tstop=tstop, n_trials=1)


def nb_ch07_plot_simulate_gamma(run):
    """content/07_gamma/plot_simulate_gamma.ipynb"""
    tstop = 300.0
    tstop_rhythmic = 550.0

    def load(fname):
        return _read_network_configuration(_data(GAMMA_DIR + fname))

    run("L5weak_L2weak", load("gamma_L5weak_L2weak.json"), tstop=tstop)
    run("L5weak_only", load("gamma_L5weak_only.json"), tstop=tstop)

    net_tonic_weak = load("gamma_L5weak_only.json")
    net_tonic_weak.add_tonic_bias(amplitude={"L5_pyramidal": 2.0})
    run("L5weak_tonic_01", net_tonic_weak, tstop=tstop)

    net_tonic_strong = load("gamma_L5weak_only.json")
    net_tonic_strong.add_tonic_bias(amplitude={"L5_pyramidal": 6.0})
    run("L5weak_tonic_02", net_tonic_strong, tstop=tstop)

    net_weak_conn = load("gamma_L5weak_only.json")
    net_weak_conn.add_tonic_bias(amplitude={"L5_pyramidal": 6.0})
    for idx in pick_connection(
        net_weak_conn, src_gids="L5_pyramidal", target_gids="L5_basket", receptor="ampa"
    ):
        net_weak_conn.connectivity[idx]["nc_dict"]["A_weight"] *= 0.1
    run("L5weak_tonic_03", net_weak_conn, tstop=tstop)

    net_no_inh = load("gamma_L5weak_only.json")
    for idx in pick_connection(
        net_no_inh, src_gids="L5_basket", target_gids="L5_basket", receptor="gabaa"
    ):
        net_no_inh.connectivity[idx]["nc_dict"]["A_weight"] = 0.0
    run("L5weak_only_noinh", net_no_inh, tstop=tstop)

    net_gabaa = load("gamma_L5weak_only.json")
    net_gabaa.cell_types["L5_pyramidal"]["cell_object"].synapses["gabaa"]["tau2"] = 2.0
    run("L5weak_only_fasterinh", net_gabaa, tstop=tstop)

    net_ping = load("gamma_L5ping_L2ping.json")
    net_ping.add_tonic_bias(amplitude={"L2_pyramidal": 4.0, "L5_pyramidal": 6.0})
    run("L5ping_L2ping", net_ping, tstop=tstop)

    run("rhythmic_drive", load("gamma_rhythmic_drive.json"), tstop=tstop_rhythmic)

    net_noisy = load("gamma_rhythmic_drive.json")
    for drive_name in ["bursty1", "bursty2"]:
        net_noisy.external_drives[drive_name]["dynamics"]["burst_std"] = 5.0
    run("rhythmic_more_noise", net_noisy, tstop=tstop_rhythmic)


def nb_ch08_full_api_howto(run):
    """content/08_using_hnn_api/full_api_howto_notebook.ipynb

    Only ``net`` is simulated; the notebook's ``net_sparse`` and ``net_replay``
    are copies that are never simulated, so they are omitted.
    """
    net = neymotin_2020_model()
    net.update_cell_positions(inplane_distance=2.0, layer_separation=1500.0)

    net._params["celsius"] = 36.0
    new_threshold = -10.0
    net._params["threshold"] = new_threshold
    net.threshold = new_threshold
    for conn in net.connectivity:
        conn["nc_dict"]["threshold"] = new_threshold

    l5_pyr = net.cell_types["L5_pyramidal"]["cell_object"]
    l2_basket = net.cell_types["L2_basket"]["cell_object"]

    l5_pyr.modify_section("soma", L=40.0, diam=30.0)
    l5_pyr.modify_section("apical_tuft", cm=0.9, Ra=180.0, v0=-68.0)

    l5_pyr.sections["soma"].mechs["hh2"]["gnabar_hh2"] = 0.17
    l2_basket.sections["soma"].mechs["hh2"]["gkbar_hh2"] = 0.04
    for sec_name, section in l5_pyr.sections.items():
        if sec_name != "soma":
            section.mechs["km"]["gbar_km"] = 220.0

    for sec_name, section in l5_pyr.sections.items():
        if sec_name != "soma":
            section.mechs["ar"]["gbar_ar"] = lambda x: 2e-6 * np.exp(3e-3 * x)
    l5_pyr._compute_section_mechs()

    for cell_type in ("L2_pyramidal", "L5_pyramidal"):
        synapses = net.cell_types[cell_type]["cell_object"].synapses
        synapses["gabab"]["tau1"] = 45.0
        synapses["gabab"]["tau2"] = 200.0

    for idx in pick_connection(
        net, src_gids="L5_pyramidal", target_gids="L5_pyramidal", receptor="nmda"
    ):
        net.connectivity[idx]["nc_dict"]["A_weight"] = 0.0004
    for idx in pick_connection(net, src_gids="L2_pyramidal", target_gids="L5_pyramidal"):
        net.connectivity[idx]["nc_dict"]["A_delay"] = 1.5
        net.connectivity[idx]["nc_dict"]["lamtha"] = 4.0

    net.set_global_synaptic_gains(e_e=1.0, e_i=1.2, i_e=0.9, i_i=1.0)

    net.add_evoked_drive(
        "evprox1",
        mu=26.61,
        sigma=2.47,
        numspikes=1,
        location="proximal",
        n_drive_cells="n_cells",
        cell_specific=True,
        weights_ampa={
            "L2_basket": 0.08831,
            "L2_pyramidal": 0.01525,
            "L5_basket": 0.19934,
            "L5_pyramidal": 0.00865,
        },
        synaptic_delays={
            "L2_basket": 0.1,
            "L2_pyramidal": 0.1,
            "L5_basket": 1.0,
            "L5_pyramidal": 1.0,
        },
        space_constant=3.0,
        probability=1.0,
        event_seed=274,
        conn_seed=3,
    )
    net.add_bursty_drive(
        "alpha_dist",
        tstart=50.0,
        tstart_std=0.0,
        tstop=None,
        burst_rate=10.0,
        burst_std=20.0,
        numspikes=2,
        spike_isi=10.0,
        location="distal",
        n_drive_cells=10,
        cell_specific=False,
        weights_ampa={"L2_pyramidal": 5.4e-5, "L5_pyramidal": 5.4e-5},
        synaptic_delays=0.1,
        event_seed=278,
    )
    net.add_poisson_drive(
        "poisson_prox",
        tstart=0.0,
        tstop=None,
        rate_constant={"L2_pyramidal": 10.0, "L5_pyramidal": 10.0},
        location="proximal",
        weights_ampa={"L2_pyramidal": 5e-4, "L5_pyramidal": 5e-4},
        synaptic_delays={"L2_pyramidal": 0.1, "L5_pyramidal": 1.0},
        event_seed=1079,
    )

    net.external_drives["evprox1"]["dynamics"]["mu"] = 30.0
    net.external_drives["evprox1"]["event_seed"] = 275
    net.external_drives["alpha_dist"]["dynamics"]["burst_rate"] = 12.0
    net.external_drives["poisson_prox"]["dynamics"]["rate_constant"]["L5_pyramidal"] = 15.0

    for idx in pick_connection(net, src_gids="alpha_dist", target_gids="L5_pyramidal"):
        net.connectivity[idx]["nc_dict"]["A_weight"] = 6e-5

    net.add_tonic_bias(
        amplitude={"L5_pyramidal": 0.01}, section="soma", t0=0.0, tstop=None
    )
    net.external_biases["tonic"]["L5_pyramidal"]["amplitude"] = 0.02

    run("edited_net", net, tstop=170.0, dt=0.025, n_trials=1)


def nb_ch08_simulate_beta_modulated_erp(run):
    """content/08_using_hnn_api/simulate_beta_modulated_erp_notebook.ipynb"""

    def add_erp_drives(net, stimulus_start):
        syn_delays_prox = {
            "L2_basket": 0.1,
            "L2_pyramidal": 0.1,
            "L5_basket": 1.0,
            "L5_pyramidal": 1.0,
        }
        net.add_evoked_drive(
            "evdist1",
            mu=70.0 + stimulus_start,
            sigma=0.0,
            numspikes=1,
            weights_ampa={
                "L2_basket": 0.0005,
                "L2_pyramidal": 0.004,
                "L5_pyramidal": 0.0005,
            },
            weights_nmda={
                "L2_basket": 0.0005,
                "L2_pyramidal": 0.004,
                "L5_pyramidal": 0.0005,
            },
            location="distal",
            synaptic_delays={"L2_basket": 0.1, "L2_pyramidal": 0.1, "L5_pyramidal": 0.1},
            event_seed=274,
        )
        net.add_evoked_drive(
            "evprox1",
            mu=25.0 + stimulus_start,
            sigma=0.0,
            numspikes=1,
            weights_ampa={
                "L2_basket": 0.002,
                "L2_pyramidal": 0.0011,
                "L5_basket": 0.001,
                "L5_pyramidal": 0.001,
            },
            weights_nmda=None,
            location="proximal",
            synaptic_delays=syn_delays_prox,
            event_seed=544,
        )
        net.add_evoked_drive(
            "evprox2",
            mu=135.0 + stimulus_start,
            sigma=0.0,
            numspikes=1,
            weights_ampa={
                "L2_basket": 0.005,
                "L2_pyramidal": 0.005,
                "L5_basket": 0.01,
                "L5_pyramidal": 0.01,
            },
            location="proximal",
            synaptic_delays=syn_delays_prox,
            event_seed=814,
        )
        return net

    def add_beta_drives(net, beta_start):
        net.add_bursty_drive(
            "beta_dist",
            tstart=beta_start,
            tstart_std=0.0,
            tstop=beta_start + 50.0,
            burst_rate=1.0,
            burst_std=10.0,
            numspikes=2,
            spike_isi=10,
            n_drive_cells=10,
            location="distal",
            weights_ampa={
                "L2_basket": 0.00032,
                "L2_pyramidal": 0.00008,
                "L5_pyramidal": 0.00004,
            },
            synaptic_delays={"L2_basket": 0.5, "L2_pyramidal": 0.5, "L5_pyramidal": 0.5},
            event_seed=290,
        )
        net.add_bursty_drive(
            "beta_prox",
            tstart=beta_start,
            tstart_std=0.0,
            tstop=beta_start + 50.0,
            burst_rate=1.0,
            burst_std=20.0,
            numspikes=2,
            spike_isi=10,
            n_drive_cells=10,
            location="proximal",
            weights_ampa={
                "L2_basket": 0.00004,
                "L2_pyramidal": 0.00002,
                "L5_basket": 0.00002,
                "L5_pyramidal": 0.00002,
            },
            synaptic_delays={
                "L2_basket": 0.1,
                "L2_pyramidal": 0.1,
                "L5_basket": 1.0,
                "L5_pyramidal": 1.0,
            },
            event_seed=300,
        )
        return net

    net = law_2021_model()
    beta_start, stimulus_start = 50.0, 125.0
    net_beta = add_beta_drives(net.copy(), beta_start)
    net_erp = add_erp_drives(net.copy(), stimulus_start)
    net_beta_erp = add_erp_drives(net_beta.copy(), stimulus_start)

    run("beta", net_beta, tstop=400)
    run("erp", net_erp, tstop=400)
    run("beta_erp", net_beta_erp, tstop=400)


def nb_ch08_batch_simulation(run):
    """content/08_using_hnn_api/batch_simulation_notebook.ipynb

    Replays what ``BatchSimulate.run(param_grid, combinations=False)`` does for
    each parameter set (copy the net, apply ``set_params``, simulate with the
    ``BatchSimulate`` defaults), rather than calling ``BatchSimulate`` itself:
    it calls ``set_params(net, params)``, but the notebook's ``set_params``
    takes ``(param_values, net)``, so the notebook fails as written.
    """
    net = neymotin_2020_model(mesh_shape=(3, 3))
    weights_basket = np.logspace(-4, -1, 20)
    weights_pyr = np.logspace(-4, -1, 20)
    for idx, (weight_basket, weight_pyr) in enumerate(zip(weights_basket, weights_pyr)):
        net_sim = net.copy()
        net_sim.add_evoked_drive(
            "evprox",
            mu=40,
            sigma=5,
            numspikes=1,
            location="proximal",
            weights_ampa={
                "L2_basket": weight_basket,
                "L2_pyramidal": weight_pyr,
                "L5_basket": weight_basket,
                "L5_pyramidal": weight_pyr,
            },
            synaptic_delays={
                "L2_basket": 0.1,
                "L2_pyramidal": 0.1,
                "L5_basket": 1.0,
                "L5_pyramidal": 1.0,
            },
        )
        run(f"sim_{idx:02d}", net_sim, tstop=170, dt=0.025, n_trials=1)


def nb_ch08_modifying_local_connectivity(run):
    """content/08_using_hnn_api/modifying_local_connectivity_notebook.ipynb"""
    net_erp = neymotin_2020_model(add_drives_from_params=True)
    run("erp", net_erp, tstop=170.0, n_trials=1)

    def get_network(probability=1.0):
        net = neymotin_2020_model(add_drives_from_params=True)
        net.clear_connectivity()
        conn_seed = 3
        for src, location, receptor, targets in [
            ("L5_pyramidal", "distal", "ampa", ["L5_pyramidal", "L2_basket"]),
            ("L2_basket", "soma", "gabaa", ["L5_pyramidal", "L2_basket"]),
        ]:
            weight, delay, lamtha = 1.0, 1.0, 70
            for target in targets:
                # positional order as in the notebook
                net.add_connection(
                    src,
                    target,
                    location,
                    receptor,
                    delay,
                    weight,
                    lamtha,
                    probability=probability,
                    conn_seed=conn_seed,
                )
        return net

    run("all", get_network(), tstop=170.0, n_trials=1)
    run("sparse", get_network(probability=0.1), tstop=170.0, n_trials=1)


def _bursty_net():
    """Network used by both parallelism notebooks."""
    net = neymotin_2020_model()
    net.add_bursty_drive(
        "bursty",
        tstart=50.0,
        burst_rate=10,
        burst_std=20.0,
        numspikes=2,
        spike_isi=10,
        n_drive_cells=10,
        location="distal",
        weights_ampa={"L2_pyramidal": 5.4e-5, "L5_pyramidal": 5.4e-5},
        event_seed=278,
    )
    return net


def nb_ch08_parallelism_joblib(run):
    """content/08_using_hnn_api/parallelism_joblib_notebook.ipynb"""
    run("bursty", _bursty_net(), tstop=210.0, n_trials=6)


def nb_ch08_parallelism_mpi(run):
    """content/08_using_hnn_api/parallelism_mpi_notebook.ipynb"""
    run("bursty", _bursty_net(), tstop=210.0, n_trials=1)


def nb_ch08_animating_hnn_simulations(run):
    """content/08_using_hnn_api/animating_hnn_simulations_notebook.ipynb"""
    net = neymotin_2020_model(mesh_shape=(3, 3))
    net.set_cell_positions(inplane_distance=300)
    add_erp_drives_to_jones_model(net)
    run("erp_3x3", net, tstop=170, record_vsec="all")


def _add_evoked_example_drives(net, n_drive_cells="n_cells", cell_specific=True):
    """The three evoked drives shared by plot_firing_pattern and the archived
    plot_simulate_evoked notebook."""
    synaptic_delays_prox = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }
    net.add_evoked_drive(
        "evdist1",
        mu=63.53,
        sigma=3.85,
        numspikes=1,
        weights_ampa={"L2_basket": 0.006562, "L2_pyramidal": 7e-6, "L5_pyramidal": 0.142300},
        weights_nmda={
            "L2_basket": 0.019482,
            "L2_pyramidal": 0.004317,
            "L5_pyramidal": 0.080074,
        },
        location="distal",
        n_drive_cells=n_drive_cells,
        cell_specific=cell_specific,
        synaptic_delays={"L2_basket": 0.1, "L2_pyramidal": 0.1, "L5_pyramidal": 0.1},
        event_seed=274,
    )
    net.add_evoked_drive(
        "evprox1",
        mu=26.61,
        sigma=2.47,
        numspikes=1,
        weights_ampa={
            "L2_basket": 0.08831,
            "L2_pyramidal": 0.01525,
            "L5_basket": 0.19934,
            "L5_pyramidal": 0.00865,
        },
        weights_nmda=None,
        location="proximal",
        n_drive_cells=n_drive_cells,
        cell_specific=cell_specific,
        synaptic_delays=synaptic_delays_prox,
        event_seed=544,
    )
    net.add_evoked_drive(
        "evprox2",
        mu=137.12,
        sigma=8.33,
        numspikes=1,
        weights_ampa={
            "L2_basket": 0.000003,
            "L2_pyramidal": 1.438840,
            "L5_basket": 0.008958,
            "L5_pyramidal": 0.684013,
        },
        location="proximal",
        n_drive_cells=n_drive_cells,
        cell_specific=cell_specific,
        synaptic_delays=synaptic_delays_prox,
        event_seed=814,
    )
    return net


def nb_ch08_plot_firing_pattern(run):
    """content/08_using_hnn_api/plot_firing_pattern_notebook.ipynb"""
    net = _add_evoked_example_drives(neymotin_2020_model())
    run("evoked", net, tstop=170.0, record_vsec="soma")


def nb_ch08_record_extracellular_potentials(run):
    """content/08_using_hnn_api/record_and_plot_extracellular_potentials_notebook.ipynb"""
    net = neymotin_2020_model()
    add_erp_drives_to_jones_model(net)
    net.set_cell_positions(inplane_distance=30.0)
    depths = list(range(-325, 2150, 100))
    net.add_electrode_array("shank1", [(135, 135, dep) for dep in depths])
    run("erp_with_electrodes", net, tstop=170)


def nb_ch08_replaying_spike_data_as_input(run):
    """content/08_using_hnn_api/replaying_spike_data_as_input_notebook.ipynb"""
    net_A = neymotin_2020_model()
    net_A._params.update({"tstop": 170.0})
    net_A.add_evoked_drive(
        name="evdist1",
        mu=63.5,
        sigma=3.8,
        numspikes=1,
        location="distal",
        event_seed=274,
        weights_ampa={"L2_basket": 0.006, "L2_pyramidal": 0.0005, "L5_pyramidal": 0.14},
        weights_nmda={"L2_basket": 0.019, "L2_pyramidal": 0.004, "L5_pyramidal": 0.08},
        synaptic_delays={"L2_basket": 0.1, "L2_pyramidal": 0.1, "L5_pyramidal": 0.1},
    )
    run("net_A", net_A, tstop=170.0, n_trials=1)

    # the notebook round-trips the spikes through text files, so do the same
    with tempfile.TemporaryDirectory() as tmp_dir:
        net_A.cell_response.write(op.join(tmp_dir, "spk_%d.txt"))
        cell_response = read_spikes(op.join(tmp_dir, "spk_*.txt"))

    net_B = neymotin_2020_model()
    net_B._params.update({"tstop": 225.0})

    trial_idx = 0
    spike_times = cell_response.spike_times[trial_idx]
    spike_gids = cell_response.spike_gids[trial_idx]
    spike_types = cell_response.spike_types[trial_idx]
    pyramidal_mask = np.array([t in ["L2_pyramidal", "L5_pyramidal"] for t in spike_types])
    filtered_times = np.array(spike_times)[pyramidal_mask]
    filtered_gids = np.array(spike_gids)[pyramidal_mask]
    filtered_types = np.array(spike_types)[pyramidal_mask]
    spike_data = {f"NetA_{t}_GID{g}": [] for t, g in zip(filtered_types, filtered_gids)}
    for t, g, spike_time in zip(filtered_types, filtered_gids, filtered_times):
        spike_data[f"NetA_{t}_GID{g}"].append(spike_time)

    conn_properties = {
        "L5_pyramidal": {"weights_ampa": 0.005},
        "L2_pyramidal": {"weights_ampa": 0.003},
    }
    target_config = {}
    for i, src_id in enumerate(spike_data):
        target_type = "L5_pyramidal" if i % 2 == 0 else "L2_pyramidal"
        target_config[src_id] = {
            "weights_ampa": {target_type: conn_properties[target_type]["weights_ampa"]}
        }
    all_weights_ampa = {
        k: cfg["weights_ampa"][k]
        for cfg in target_config.values()
        for k in cfg["weights_ampa"]
    }

    net_B.add_spike_train_drive(
        name="drive_from_NetA",
        spike_data=spike_data,
        location="distal",
        weights_ampa=all_weights_ampa,
        weights_nmda=None,
        synaptic_delays=0.1,
        conn_seed=42,
    )
    run("net_B", net_B, tstop=225.0, n_trials=1)


def nb_ch09_from_meg_to_hnn(run):
    """content/09_data_to_simulation/from_meg_to_hnn_notebook.ipynb

    Only the HNN simulation; the MNE source-estimation steps are omitted.
    """
    params_fname = op.join(op.dirname(hnn_core.__file__), "param", "N20.json")
    net = neymotin_2020_model(params_fname)

    prox_delays = {"L2_basket": 0.1, "L2_pyramidal": 0.1, "L5_basket": 1.0, "L5_pyramidal": 1.0}
    dist_delays = {"L2_basket": 0.1, "L2_pyramidal": 0.1, "L5_pyramidal": 0.1}
    drives = [
        # name, mu, sigma, location, weights_ampa, weights_nmda, delays, seed
        (
            "evprox1", 21.0, 4.0, "proximal",
            {"L2_basket": 0.0036, "L2_pyramidal": 0.0039, "L5_basket": 0.0019, "L5_pyramidal": 0.0020},
            {"L2_basket": 0.0029, "L2_pyramidal": 0.0005, "L5_basket": 0.0030, "L5_pyramidal": 0.0019},
            prox_delays, 276,
        ),
        (
            "evprox2", 134.0, 4.5, "proximal",
            {"L2_basket": 0.003, "L2_pyramidal": 0.0039, "L5_basket": 0.004, "L5_pyramidal": 0.0020},
            {"L2_basket": 0.001, "L2_pyramidal": 0.0005, "L5_basket": 0.002, "L5_pyramidal": 0.0020},
            prox_delays, 276,
        ),
        (
            "evdist1", 32.0, 2.5, "distal",
            {"L2_basket": 0.0043, "L2_pyramidal": 0.0032, "L5_pyramidal": 0.0009},
            {"L2_basket": 0.0029, "L2_pyramidal": 0.0051, "L5_pyramidal": 0.0010},
            dist_delays, 277,
        ),
        (
            "evdist2", 84.0, 4.5, "distal",
            {"L2_basket": 0.0041, "L2_pyramidal": 0.0019, "L5_pyramidal": 0.0018},
            {"L2_basket": 0.0032, "L2_pyramidal": 0.0018, "L5_pyramidal": 0.0017},
            dist_delays, 275,
        ),
    ]  # fmt: skip
    for name, mu, sigma, location, w_ampa, w_nmda, delays, seed in drives:
        net.add_evoked_drive(
            name,
            mu=mu,
            sigma=sigma,
            numspikes=1,
            location=location,
            n_drive_cells=1,
            cell_specific=False,
            weights_ampa=w_ampa,
            weights_nmda=w_nmda,
            synaptic_delays=delays,
            event_seed=seed,
        )
    run("N20", net, tstop=170.0, n_trials=2)


def nb_archive_plot_simulate_evoked(run):
    """resources/archive/05_erps/plot_simulate_evoked.ipynb"""
    net = _add_evoked_example_drives(neymotin_2020_model())
    run("evoked", net, tstop=170.0, n_trials=2)

    net_sync = _add_evoked_example_drives(
        neymotin_2020_model(), n_drive_cells=1, cell_specific=False
    )
    run("evoked_sync", net_sync, tstop=170.0, n_trials=1)


NOTEBOOKS = {
    name[len("nb_") :]: func
    for name, func in globals().items()
    if name.startswith("nb_") and callable(func)
}


# ----------------------------------------------------------------------------
# Running and saving
# ----------------------------------------------------------------------------
class _Runner:
    """The ``run`` callable handed to each notebook function."""

    def __init__(self, nb_name, out_dir):
        self.nb_name = nb_name
        self.out_dir = out_dir
        self.sims = []

    def __call__(self, sim_name, net, **simulate_kwargs):
        print(f"\n===== {self.nb_name} / {sim_name} =====")
        with MPIBackend():
            dpls = simulate_dipole(net, **simulate_kwargs)
        np.savez(
            self.out_dir / f"{self.nb_name}__{sim_name}.npz",
            times=np.asarray(dpls[0].times),
            **{layer: np.array([dpl.data[layer] for dpl in dpls]) for layer in LAYERS},
        )
        self.sims.append(sim_name)
        return dpls


def _git(path, *args):
    return subprocess.run(
        ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
    ).stdout


def _git_describe(path):
    """Short commit hash of the git checkout containing ``path``, if any."""
    try:
        commit = _git(path, "rev-parse", "--short", "HEAD").strip()
        dirty = _git(path, "status", "--porcelain", "--", ".").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return commit + ("-dirty" if dirty else "")


def _source_hashes(hnn_core_dir):
    """SHA-1 of every hnn-core source file except the tests.

    Uses the git-tracked files of a checkout (hashing their working-tree
    contents), so that untracked files and build products are ignored.
    """
    try:
        fnames = _git(hnn_core_dir, "ls-files", "-z").split("\0")
    except (subprocess.CalledProcessError, FileNotFoundError):  # not a checkout
        fnames = [
            str(fname.relative_to(hnn_core_dir))
            for fname in hnn_core_dir.rglob("*")
            if fname.suffix in (".py", ".mod", ".json")
        ]
    return {
        fname: hashlib.sha1((hnn_core_dir / fname).read_bytes()).hexdigest()
        for fname in sorted(fnames)
        if not fname.startswith("tests/") and (hnn_core_dir / fname).is_file()
    }


def _environment_info():
    import neuron
    from neuron import h

    # hnn-core turns on NEURON's legacy units (if it does) when this is imported
    import hnn_core.network_builder

    hnn_core_dir = Path(hnn_core.__file__).resolve().parent
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "neuron": neuron.__version__,
        "hnn_core": hnn_core.__version__,
        "hnn_core_path": str(hnn_core_dir),
        "hnn_core_git": _git_describe(hnn_core_dir),
        # legacy and modern FARADAY differ by only 2e-7 relative
        "units": "legacy" if abs(h.FARADAY - LEGACY_FARADAY) < 1e-4 else "modern",
        "hnn_core_sources": _source_hashes(hnn_core_dir),
    }


def _matches(nb_name, patterns):
    return not patterns or any(pattern in nb_name for pattern in patterns)


def run_all(label, patterns=()):
    """Run every notebook whose name contains one of ``patterns`` (all if empty)."""
    out_dir = OUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_fname = out_dir / "meta.json"
    meta = json.loads(meta_fname.read_text()) if meta_fname.exists() else {}
    meta["environment"] = _environment_info()
    meta.setdefault("notebooks", {})
    print(json.dumps(
        {k: v for k, v in meta["environment"].items() if k != "hnn_core_sources"},
        indent=2,
    ))

    for nb_name, nb_func in NOTEBOOKS.items():
        if not _matches(nb_name, patterns):
            continue
        for stale in out_dir.glob(f"{nb_name}__*.npz"):
            stale.unlink()
        run = _Runner(nb_name, out_dir)
        start = time.time()
        try:
            nb_func(run)
            status = "ok"
        except Exception:  # noqa: BLE001
            status = traceback.format_exc()
            print(f"\n!!!!! {nb_name} FAILED:\n{status}")
        meta["notebooks"][nb_name] = {
            "status": status,
            "sims": run.sims,
            "seconds": round(time.time() - start, 1),
        }
        # rewrite after every notebook so that an interrupted run keeps its results
        meta_fname.write_text(json.dumps(meta, indent=2))


# ----------------------------------------------------------------------------
# Comparing
# ----------------------------------------------------------------------------
def _load(fname):
    with np.load(fname) as data:
        return {key: data[key] for key in data.files}


def compare(label_a="old", label_b="new", patterns=()):
    """Print and plot differences between the dipoles of two runs."""
    dir_a, dir_b = OUT_DIR / label_a, OUT_DIR / label_b
    meta_a = json.loads((dir_a / "meta.json").read_text())
    meta_b = json.loads((dir_b / "meta.json").read_text())
    env_a, env_b = meta_a["environment"], meta_b["environment"]
    src_a, src_b = env_a.pop("hnn_core_sources"), env_b.pop("hnn_core_sources")

    print(f"\n{'':15s} {label_a:<60s} {label_b}")
    for key in env_a:
        flag = "" if env_a[key] == env_b.get(key) else "   <-- differs"
        print(f"{key:15s} {env_a[key]!s:<60s} {env_b.get(key)}{flag}")

    src_diffs = sorted(
        fname for fname in src_a.keys() | src_b.keys() if src_a.get(fname) != src_b.get(fname)
    )
    if src_diffs:
        print(
            "\nWARNING: these hnn-core source files differ between the two runs, so "
            "differences below may come from them rather than from NEURON:"
        )
        for fname in src_diffs:
            if fname not in src_a or fname not in src_b:
                print(f"    {fname} (only in '{label_a if fname in src_a else label_b}')")
            else:
                print(f"    {fname}")
    else:
        print("\nhnn-core source files are identical in the two runs.")
    if env_a["units"] != env_b["units"]:
        print(
            "\nWARNING: the two runs used different NEURON unit systems "
            "(legacy vs modern); this alone changes the output."
        )

    header = (
        f"\n{'simulation':58s} {'trials':>6s} {'max|agg|':>10s} {'max|Δagg|':>10s} "
        f"{'rel':>9s} {'max|ΔL2|':>10s} {'max|ΔL5|':>10s}  result"
    )
    print(header)
    print("-" * len(header))
    counts = {"identical": 0, "differs": 0, "missing": 0}
    plot_dir = OUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for nb_name in NOTEBOOKS:
        if not _matches(nb_name, patterns):
            continue
        nb_a = meta_a["notebooks"].get(nb_name, {})
        nb_b = meta_b["notebooks"].get(nb_name, {})
        for label, nb in [(label_a, nb_a), (label_b, nb_b)]:
            if nb.get("status", "ok") != "ok":
                print(f"{nb_name}: FAILED in '{label}' run: {nb['status'].splitlines()[-1]}")
        sim_names = list(dict.fromkeys(nb_a.get("sims", []) + nb_b.get("sims", [])))
        if not sim_names:
            continue

        fig, axes = plt.subplots(
            len(sim_names), 2, figsize=(12, 2.2 * len(sim_names)), squeeze=False
        )
        for row, sim_name in enumerate(sim_names):
            fname = f"{nb_name}__{sim_name}.npz"
            full_name = f"{nb_name}/{sim_name}"
            if not ((dir_a / fname).exists() and (dir_b / fname).exists()):
                print(f"{full_name:58s} missing from one run")
                counts["missing"] += 1
                continue
            A, B = _load(dir_a / fname), _load(dir_b / fname)
            if A["agg"].shape != B["agg"].shape or not np.allclose(A["times"], B["times"]):
                print(f"{full_name:58s} shapes differ: {A['agg'].shape} vs {B['agg'].shape}")
                counts["differs"] += 1
                continue
            diff = {layer: np.abs(B[layer] - A[layer]).max() for layer in LAYERS}
            scale = np.abs(A["agg"]).max()
            identical = all(np.array_equal(A[layer], B[layer]) for layer in LAYERS)
            counts["identical" if identical else "differs"] += 1
            print(
                f"{full_name:58s} {A['agg'].shape[0]:6d} {scale:10.3e} {diff['agg']:10.3e} "
                f"{diff['agg'] / scale if scale else 0.0:9.2e} {diff['L2']:10.3e} "
                f"{diff['L5']:10.3e}  {'IDENTICAL' if identical else 'differs'}"
            )

            times = A["times"]
            axes[row, 0].plot(times, A["agg"][0], label=label_a)
            axes[row, 0].plot(times, B["agg"][0], "--", label=label_b)
            axes[row, 0].set_ylabel(sim_name, fontsize=8)
            axes[row, 1].plot(times, (B["agg"] - A["agg"]).T, lw=0.8)
        axes[0, 0].set_title("agg dipole, trial 0")
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].set_title(f"agg difference ({label_b} - {label_a}), all trials")
        for ax in axes[-1]:
            ax.set_xlabel("Time (ms)")
        fig.suptitle(nb_name)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{nb_name}.png", dpi=100)
        plt.close(fig)

    print(
        f"\n{counts['identical']} identical, {counts['differs']} differ, "
        f"{counts['missing']} missing. Plots saved to {plot_dir}"
    )
