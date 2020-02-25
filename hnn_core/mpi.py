"""Class to handle the dipoles."""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>

from .simulation import simulate

import sys
import os


def isWindows():
    # are we on windows? or linux/mac ?
    return sys.platform.startswith('win')


def simulate_mpi(config, n_procs=None, n_trials=None):
    """Simulate the HNN model in parallel on all cores.

    Parameters
    ----------
    config : Config object
        The specification for running the simulation(s)
    n_procs: int | None
        The number of cores to run simulations on.
    n_trials : int | None
        The number of trials to simulate.

    Returns
    -------
    sim_data: list of tuples
        The Dipole and Spike output from each simulation trial
    """
    from mpi4py import MPI
    import multiprocessing
    from psutil import cpu_count

    if n_procs is None:
        try:
            n_procs = len(os.sched_getaffinity(0))
        except AttributeError:
            physical_cores = cpu_count(logical=False)
            logical_cores = multiprocessing.cpu_count()

        if logical_cores is not None and logical_cores > physical_cores:
            n_procs = logical_cores
        else:
            n_procs = physical_cores

    # Start clock
    start = MPI.Wtime()
    mpiinfo = MPI.Info().Create()
    if not isWindows():
        mpiinfo.Set('ompi_param', 'mpi_yield_when_idle=false')

    # spawn NEURON sim
    subcomm = MPI.COMM_SELF.Spawn('nrniv',
                                  args=['-python', '-mpi', '-nobanner',
                                        'python',
                                        '-c', 'from hnn_core import mpi; ' +
                                        'mpi.run_simulate_mpi(); exit()'],
                                  info=mpiinfo, maxprocs=n_procs)

    # send params and extdata to spawned nrniv procs
    subcomm.bcast(config, root=MPI.ROOT)

    # wait to receive results from child rank 0
    sim_data = subcomm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)

    finish = MPI.Wtime() - start
    print('Simulation in %.2f secs' % finish)

    MPI.Finalize()

    return sim_data


def run_simulate_mpi():
    from mpi4py import MPI

    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()
    config = comm.bcast(rank, root=0)
    sim_data = simulate(config, n_trials=config.cfg.N_trials)

    if rank == 0:
        # send results back to parent
        comm.send(sim_data, dest=0)

    MPI.Finalize()


def shutdown():
    from netpyne import sim
    from neuron import h

    sim.pc.done()
    h.quit()
