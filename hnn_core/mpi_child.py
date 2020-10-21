"""Script for running parallel simulations with MPI when called with mpiexec.
This script is called directly from MPIBackend.simulate()
"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>

import sys
import pickle
import base64


def _read_all_bytes(stream_in, chunk_size=4096):
    all_data = b""
    while True:
        data = stream_in.read(chunk_size)
        all_data += data
        if len(data) < chunk_size:
            break

    return all_data


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

    def _read_params(self):
        """Read params broadcasted to all ranks on stdin"""

        # get parameters from stdin
        if self.rank == 0:
            input_bytes = _read_all_bytes(sys.stdin.buffer)
            sys.stdin.close()

            params = pickle.loads(base64.b64decode(input_bytes, validate=True))
        else:
            params = None

        params = self.comm.bcast(params, root=0)
        return params

    def _pickle_data(self, sim_data):
        # pickle the data and encode as base64 before sending to stderr
        pickled_str = pickle.dumps(sim_data)
        pickled_bytes = base64.b64encode(pickled_str)

        return pickled_bytes

    def _write_data_stderr(self, sim_data):
        """write base64 encoded data to stderr"""

        # only have rank 0 write to stdout/stderr
        if self.rank > 0:
            return

        pickled_bytes = self._pickle_data(sim_data)

        sys.stderr.write(pickled_bytes.decode())
        sys.stderr.flush()

        # the parent process is waiting for "end_of_sim:[#bytes]" with the
        # length of data
        sys.stderr.write('end_of_data:%d' % len(pickled_bytes))
        sys.stderr.flush()  # flush to ensure signal is not buffered

    def run(self, params):
        """Run MPI simulation(s) and write results to stderr"""

        from hnn_core import Network
        from hnn_core.parallel_backends import _clone_and_simulate

        prng_seedcore_initial = params['prng_*']

        net = Network(params)
        sim_data = []
        for trial_idx in range(params['N_trials']):
            single_sim_data = _clone_and_simulate(net, trial_idx,
                                                  prng_seedcore_initial)

            # go ahead and append trial data for each rank, though
            # only rank 0 has data that should be sent back to MPIBackend
            sim_data.append(single_sim_data)

        # flush output buffers from all ranks (any errors or status mesages)
        sys.stdout.flush()
        sys.stderr.flush()

        if self.rank == 0:
            # the parent process is waiting for "end_of_sim" to signal that
            # the following stderr will only contain sim_data
            sys.stdout.write('end_of_sim')
            sys.stdout.flush()  # flush to ensure signal is not buffered

        return sim_data


if __name__ == '__main__':
    """This file is called on command-line from nrniv"""

    import traceback
    rc = 0

    try:
        with MPISimulation() as mpi_sim:
            params = mpi_sim._read_params()
            sim_data = mpi_sim.run(params)
            mpi_sim._write_data_stderr(sim_data)
    except Exception:
        # This can be useful to indicate the problem to the
        # caller (in parallel_backends.py)
        traceback.print_exc(file=sys.stdout)
        rc = 2

    sys.exit(rc)
