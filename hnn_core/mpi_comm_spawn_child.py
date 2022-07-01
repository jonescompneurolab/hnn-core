


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

            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # skip Finalize() if we didn't import MPI on __init__
        if hasattr(self, 'comm'):
            from mpi4py import MPI
            MPI.Finalize()