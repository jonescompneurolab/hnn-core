"""Script for running parallel simulations with MPI when called with mpiexec.
This script is called directly from MPIBackend.simulate()
"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Ryan Thorpe <ryvthorpe@gmail.com>

from hnn_core.network_builder import _simulate_single_trial


class MPISimulation(object):
    """The MPISimulation class.

    Parameters
    ----------
    skip_mpi_import : bool | None
        Skip importing MPI. Only useful for testing with pytest.

    Attributes
    ----------
    comm : mpi4py.Comm object
        The handle used for communicating among MPI processes
    rank : int
        The rank for each processor part of the MPI communicator
    """
    def __init__(self, skip_mpi_import=False):
        self.skip_mpi_import = skip_mpi_import
        if skip_mpi_import:
            self.rank = 0
        else:
            from mpi4py import MPI

            self.intercomm = MPI.Comm.Get_parent()
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # skip Finalize() if we didn't import MPI on __init__
        if hasattr(self, 'comm'):
            self.intercomm.Disconnect()

    def _read_net(self):
        """Read net and associated objects broadcasted to all ranks on stdin"""

        return self.intercomm.bcast(None, root=0)

    def run(self, net, tstop, dt, n_trials):
        """Run MPI simulation(s) and write results to stderr"""

        sim_data = []
        for trial_idx in range(n_trials):
            single_sim_data = _simulate_single_trial(net, tstop, dt, trial_idx)

            # go ahead and append trial data for each rank, though
            # only rank 0 has data that should be sent back to MPIBackend
            sim_data.append(single_sim_data)

        return sim_data


if __name__ == '__main__':
    """This file is called on command-line from nrniv"""

    with MPISimulation() as mpi_sim:
        net, tstop, dt, n_trials = mpi_sim._read_net()
        sim_data = mpi_sim.run(net, tstop, dt, n_trials)
        if mpi_sim.rank == 0:
            mpi_sim.intercomm.send(sim_data, dest=0)
