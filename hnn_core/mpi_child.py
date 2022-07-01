"""Script for running parallel simulations with MPI when called with mpiexec.
This script is called directly from MPIBackend.simulate()
"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Ryan Thorpe <ryvthorpe@gmail.com>

import sys
import pickle
import base64
import re
import shlex
from os import environ
from mpi4py import MPI

from hnn_core.parallel_backends import _extract_data, _extract_data_length


def _pickle_data(sim_data):
    # pickle the data and encode as base64 before sending to stderr
    pickled_str = pickle.dumps(sim_data)
    pickled_bytes = base64.b64encode(pickled_str)

    return pickled_bytes


def _str_to_net(input_str):
    net = None

    data_str = _extract_data(input_str, 'net')
    if len(data_str) > 0:
        # get the size, but start the search after data
        net_size = _extract_data_length(input_str[len(data_str):],
                                        'net')
        # check the size
        if len(data_str) != net_size:
            raise ValueError("Got incorrect network size: %d bytes " %
                             len(data_str) + "expected length: %d" % net_size)
        # unpickle the net
        net = pickle.loads(base64.b64decode(data_str.encode(),
                                            validate=True))
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

    def _read_net(self):
        """Read net and associated objects broadcasted to all ranks on stdin"""

        # read Network from stdin
        if self.rank == 0:
            input_str = ''
            while True:
                line = sys.stdin.readline()
                line = line.rstrip('\n')
                input_str += line
                end_match = re.search(r'@end_of_net:\d+@', input_str)
                if end_match is not None:
                    break

            net = _str_to_net(input_str)
        else:
            net = None

        net = self.comm.bcast(net, root=0)
        return net

    def _wait_for_exit_signal(self):
        # read from stdin
        if self.rank == 0:
            input_str = ''
            while True:
                line = sys.stdin.readline()
                line = line.rstrip('\n')
                input_str += line
                if '@data_received@' in input_str:
                    break

    def _write_data_stderr(self, sim_data):
        """write base64 encoded data to stderr"""

        # only have rank 0 write to stdout/stderr
        if self.rank > 0:
            return

        sys.stderr.write('@start_of_data@')
        pickled_bytes = _pickle_data(sim_data)
        sys.stderr.write(pickled_bytes.decode())

        # the parent process is waiting for "@end_of_data:[#bytes]@" with the
        # length of data. The '@' is not found in base64 encoding, so we can
        # be certain it is the border of the signal
        sys.stderr.write('@end_of_data:%d@\n' % len(pickled_bytes))
        sys.stderr.flush()  # flush to ensure signal is not buffered

    def run(self, net, tstop, dt, n_trials):
        """Run MPI simulation(s) and write results to stderr"""

        from hnn_core.network_builder import _simulate_single_trial

        sim_data = []
        for trial_idx in range(n_trials):
            single_sim_data = _simulate_single_trial(net, tstop, dt, trial_idx)

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
        try:
            if bool(environ['HNN_CORE_MPI_COMM_SPAWN']):
                cmd = environ['HNN_CORE_SPAWN_CMD']
                # Split the command into shell arguments for passing to Popen
                if 'win' in sys.platform:
                    use_posix = True
                else:
                    use_posix = False
                cmd = shlex.split(cmd, posix=use_posix)

                n_procs = int(environ['HNN_CORE_SPAWN_N_PROCS'])
                info = environ['HNN_CORE_SPAWN_INFO']

                # important: update MPI_COMM_SPAWN env var so that it can call
                # mpi_child.py again without spawning its own child MPI process
                environ['HNN_CORE_MPI_COMM_SPAWN'] = '0'
                
                if not info:
                    subcomm = MPI.COMM_SELF.Spawn('nrniv', args=cmd,
                                                  maxprocs=n_procs)
                else:
                    subcomm = MPI.COMM_SELF.Spawn('nrniv', args=cmd,
                                                  info=info,
                                                  maxprocs=n_procs)
            else:
                raise KeyError  # trigger exception where the simulation is run

        except KeyError:
            with MPISimulation() as mpi_sim:
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
