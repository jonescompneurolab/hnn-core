"""Script for running parallel simulations with MPI when called with mpiexec.
This script is called directly from MPIBackend.simulate()
"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>

import sys
import pickle
import base64
import re
from hnn_core.parallel_backends import _extract_data, _extract_data_length
import logging

import os
import tempfile


def _str_to_net(input_str):
    net = None

    data_str = _extract_data(input_str, "net")
    if len(data_str) > 0:
        # get the size, but start the search after data
        net_size = _extract_data_length(input_str[len(data_str) :], "net")
        # check the size
        if len(data_str) != net_size:
            raise ValueError(
                "Got incorrect network size: %d bytes " % len(data_str)
                + "expected length: %d" % net_size
            )
        # unpickle the net
        net = pickle.loads(base64.b64decode(data_str.encode(), validate=True))
    return net


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
    verbose_subprocess: boolean
        If True, prints progress messages and status updates to log file.
    """

    def __init__(self, skip_mpi_import=False, verbose_subprocess=False):
        self.skip_mpi_import = skip_mpi_import
        self.logger = None
        if skip_mpi_import:
            self.rank = 0
        else:
            from mpi4py import MPI

            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            size = self.comm.Get_size()

            if verbose_subprocess:
                log_filename = f"process_{self.rank}.log"
                logging.basicConfig(
                    filename=log_filename,
                    filemode="w",
                    level=logging.INFO,
                    format=f"[Rank {self.rank}/{size}] %(asctime)s - %(levelname)s - %(message)s",
                )
                self.logger = logging.getLogger(f"mpi_child.rank{self.rank}")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # skip Finalize() if we didn't import MPI on __init__
        if hasattr(self, "comm"):
            from mpi4py import MPI

            MPI.Finalize()

    def _read_net(self):
        """Read net broadcasted to all ranks on stdin"""

        # read Network from stdin
        if self.rank == 0:
            input_str = ""
            while True:
                line = sys.stdin.readline()
                line = line.rstrip("\n")
                input_str += line
                end_match = re.search(r"@end_of_net:\d+@", input_str)
                if end_match is not None:
                    break

            net = _str_to_net(input_str)
        else:
            net = None

        net = self.comm.bcast(net, root=0)
        if self.logger:
            self.logger.info(f"Net has been loaded on rank {self.rank}")
        return net

    def _wait_for_exit_signal(self):
        """Wait for the parent to acknowledge data receipt before exiting.

        Rank 0 blocks on stdin until it receives "@data_received@", sent by
        the parent after it has read the result file path from stderr."""

        # read from stdin
        if self.rank == 0:
            input_str = ""
            while True:
                line = sys.stdin.readline()
                line = line.rstrip("\n")
                input_str += line
                if "@data_received@" in input_str:
                    break

    def _write_data_tempfile(self, sim_data):
        """Pickle sim_data to a temp file and return file path
        and byte count."""

        pickled_bytes = pickle.dumps(sim_data)
        fd, tmp_path = tempfile.mkstemp(prefix="hnn_mpi_data_", suffix=".pkl")
        if self.logger:
            self.logger.info(f"Rank 0 begins writing data to temp file {tmp_path}")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(pickled_bytes)
        except Exception:
            # This exception was necessary in HPC environments to make the runtime error more verbose
            os.unlink(tmp_path)
            raise

        return tmp_path, len(pickled_bytes)

    def _signal_data_file_stderr(self, tmp_path, pickled_size):
        """Signal parent process with temp file path
        and byte count on stderr."""

        # Signal the parent process the file path @data_file:/path/to/file:SIZE@
        # solves pipe buffering problems
        # TODO make sure that this string is under 64kb right?
        sys.stderr.write("@data_file:%s:%d@\n" % (tmp_path, pickled_size))
        sys.stderr.flush()  # flush to ensure signal is not buffered

    def _write_data_stderr(self, sim_data):
        """Pickle sim_data to a temp file and signal the parent via stderr.

        Rank 0 writes the file path and byte count as "@data_file:PATH:SIZE@\n";
        all other ranks return immediately."""

        # only have rank 0 write to stdout/stderr
        if self.rank > 0:
            if self.logger:
                self.logger.info("Child process beginning to wait for exit signal")
            return

        if self.logger:
            self.logger.info(
                "Rank 0 beginning to write data to temp file and signal parent process"
            )
        tmp_path, pickled_size = self._write_data_tempfile(sim_data)
        self._signal_data_file_stderr(tmp_path, pickled_size)
        if self.logger:
            self.logger.info(f"Rank 0 finished writing data to temp file {tmp_path}")

    def run(self, net, tstop, dt, n_trials):
        """Run MPI simulation(s) and write results to stderr"""

        from hnn_core.network_builder import _simulate_single_trial

        sim_data = list()
        for trial_idx in range(n_trials):
            if self.logger:
                self.logger.info(
                    f"Beginning simulation of trial {trial_idx} on rank {self.rank}"
                )
            single_sim_data = _simulate_single_trial(net, tstop, dt, trial_idx)

            # go ahead and append trial data for each rank, though
            # only rank 0 has data that should be sent back to MPIBackend
            sim_data.append(single_sim_data)

            if self.logger:
                # AES: Is this always successful?
                self.logger.info(
                    f"Successfully finished simulation of trial {trial_idx} on rank {self.rank}"
                )

        # flush output buffers from all ranks (any errors or status messages)
        sys.stdout.flush()
        sys.stderr.flush()
        if self.logger:
            self.logger.info(
                f"Successfully finished all simulations on rank {self.rank}"
            )
        return sim_data


if __name__ == "__main__":
    """This file is called on command-line from nrniv"""

    import traceback

    rc = 0
    verbose_subprocess = "--verbose-subprocess" in sys.argv

    try:
        with MPISimulation(verbose_subprocess=verbose_subprocess) as mpi_sim:
            net, tstop, dt, n_trials = mpi_sim._read_net()
            sim_data = mpi_sim.run(net, tstop, dt, n_trials)
            mpi_sim._write_data_stderr(sim_data)
            mpi_sim._wait_for_exit_signal()
    except Exception:
        # This can be useful to indicate the problem to the
        # caller (in parallel_backends.py)
        traceback.print_exc(file=sys.stdout)
        rc = 2

    sys.exit(rc)
