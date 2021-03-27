"""Script for running parallel simulations with MPI when called with mpiexec.
This script is called directly from MPIBackend.simulate()
"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>

import sys
import pickle
import base64
from warnings import warn
from queue import Queue, Empty
from threading import Thread

from hnn_core.parallel_backends import _extract_data, _extract_data_length


def _enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        # different from MNE version in that newlines are removed
        line = line.rstrip('\n')
        queue.put(line)


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
        self.input_bytes = b''

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

    def _process_input(self, in_q):
        self.input_bytes = b''
        try:
            input_str = in_q.get(timeout=0.01)
        except Empty:
            return None

        data = _extract_data(input_str, 'net')
        if len(data) > 0:
            # start the search after data
            data_length = _extract_data_length(input_str[len(data):],
                                               'net')
            self.input_bytes += data.encode()
            if len(data) == data_length:
                return data_length
            else:
                # This is a weird "bug" for which there isn't a good fix.
                # An eextra 4082 bytes are read from stdin that weren't
                # put there by the parent process. They extra bytes are
                # inserted at byte 65535, so it's in the middle of the Network
                # object. Simply slicing out the 4082 seems to give the correct
                # object too, but resending is less opaque.
                warn("Got incorrect network size: %d bytes " % len(data) +
                     "expected length: %d" % data_length)

                # signal to parent that there was an error with the network
                sys.stderr.write("@net_receive_error@\n")
                sys.stderr.flush()

        return None

    def _read_net(self):
        """Read net broadcasted to all ranks on stdin"""

        # get parameters from stdin
        if self.rank == 0:
            in_q = Queue()
            in_t = Thread(target=_enqueue_output,
                          args=(sys.stdin, in_q))
            in_t.daemon = True
            in_t.start()

            while True:
                data_len = self._process_input(in_q)
                # data is in self.input_bytes
                if isinstance(data_len, int):
                    break

            net = pickle.loads(base64.b64decode(self.input_bytes,
                                                validate=True))
        else:
            net = None

        net = self.comm.bcast(net, root=0)
        return net

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

        sys.stderr.write('@start_of_data@')
        pickled_bytes = self._pickle_data(sim_data)
        sys.stderr.write(pickled_bytes.decode())

        # the parent process is waiting for "@end_of_data:[#bytes]@" with the
        # length of data. The '@' is not found in base64 encoding, so we can
        # be certain it is the border of the signal
        sys.stderr.write('@end_of_data:%d@' % len(pickled_bytes))
        sys.stderr.flush()  # flush to ensure signal is not buffered

    def run(self, net):
        """Run MPI simulation(s) and write results to stderr"""

        from hnn_core.parallel_backends import _clone_and_simulate

        sim_data = []
        for trial_idx in range(net.params['N_trials']):
            single_sim_data = _clone_and_simulate(net, trial_idx)

            # go ahead and append trial data for each rank, though
            # only rank 0 has data that should be sent back to MPIBackend
            sim_data.append(single_sim_data)

        # flush output buffers from all ranks (any errors or status mesages)
        sys.stdout.flush()
        sys.stderr.flush()

        return sim_data


if __name__ == '__main__':
    """This file is called on command-line from nrniv"""

    import traceback
    rc = 0

    try:
        with MPISimulation() as mpi_sim:
            net = mpi_sim._read_net()
            sim_data = mpi_sim.run(net)
            mpi_sim._write_data_stderr(sim_data)
    except Exception:
        # This can be useful to indicate the problem to the
        # caller (in parallel_backends.py)
        traceback.print_exc(file=sys.stdout)
        rc = 2

    sys.exit(rc)
