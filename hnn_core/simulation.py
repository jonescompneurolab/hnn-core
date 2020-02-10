"""Functions for running simulations"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>
#          Blake Caldwell <blake_caldwell@brown.edu>

import numpy as np

from .parallel import _parallel_func
from .dipole import _process_dipole
from .spike import Spike


def _hammfilt(x, winsz):
    """Convolve with a hamming window."""
    win = np.hamming(winsz)
    win /= sum(win)
    return np.convolve(x, win, 'same')


def _clone_and_simulate(config, trial_idx):

    # if trial_idx != 0:
    #     params['prng_*'] = trial_idx

    return _simulate_single_trial(config, trial_idx)


def _simulate_single_trial(config, trial_idx):
    """Simulate one trial.

    Parameters
    ----------
    config : Config object
        The specification for running the simulation
    trial_idx : int
        The current trial index

    Returns
    -------
    out: tuple of Dipole and Spike objects
        The dipole and spiking output from the simulation
    """

    from netpyne import sim

    # only create sim if parallel context hasn't been initialized
    if hasattr(sim, 'nhosts'):
        sim.runSim(reRun=True)
    else:
        sim.create(simConfig=config.cfg, netParams=config.net)
        sim.runSim()
    sim.gatherData()

    dpl = spk = None
    if sim.rank == 0:
        dpl = _process_dipole(sim.allSimData['dipole'],
                              sim.simData['t'], config.cfg, trial_idx)

        cells = {}
        for c in sim.net.allCells:
            cells[c['gid']] = c['tags']['pop']
        spk = Spike(sim.allSimData['spkt'], sim.allSimData['spkid'],
                    cells, config.cfg.duration)

    return dpl, spk


def simulate(config, n_trials=1, n_jobs=1):
    """Simulate the HNN model.

    Parameters
    ----------
    config : Config object
        The specification for running the simulation(s)
    n_trials : int
        The number of trials to simulate.
    n_jobs : int
        The number of jobs to run in parallel.

    Returns
    -------
    out: list of tuples
        The Dipole and Spike output from each simulation trial
    """

    parallel, myfunc = _parallel_func(_clone_and_simulate, n_jobs=n_jobs)
    out = parallel(myfunc(config, idx) for idx in range(n_trials))
    return zip(*out)
