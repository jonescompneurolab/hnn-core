"""Script for running parallel simulations with MPI when called with mpiexec.
This script is called directly from MPIBackend.simulate()
"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>

import sys


def _read_all_bytes(stream_in, chunk_size=4096):
    all_data = b""
    while True:
        data = stream_in.read(chunk_size)
        all_data += data
        if len(data) < chunk_size:
            break

    return all_data


def run_mpi_simulation():
    from mpi4py import MPI

    import pickle
    import codecs
    import os
    import io

    # suppress output to stderr
    stderr_fileno = sys.stderr.fileno()
    null_fd = os.open(os.devnull, os.O_RDWR)
    old_err_fd = os.dup(stderr_fileno)
    os.dup2(null_fd, stderr_fileno)

    # temporarily use a StringIO object to capture stderr
    str_err = io.StringIO()
    sys.stderr = str_err

    from hnn_core import Network
    from hnn_core.network_builder import NetworkBuilder, _simulate_single_trial

    # using template for reading stdin from:
    # https://github.com/cloudpipe/cloudpickle/blob/master/tests/testutils.py

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # get parameters from stdin
    if rank == 0:
        stream_in = sys.stdin
        # Force the use of bytes streams under Python 3
        if hasattr(sys.stdin, 'buffer'):
            stream_in = sys.stdin.buffer
        input_bytes = _read_all_bytes(stream_in)
        stream_in.close()

        params = pickle.loads(codecs.decode(input_bytes, "base64"))
    else:
        params = None

    params = comm.bcast(params, root=0)
    net = Network(params)
    # XXX store the initial prng_seedcore params to be referenced in each trial
    prng_seedcore_initial = net.params['prng_*'].copy()

    sim_data = []
    for trial_idx in range(params['N_trials']):
        # XXX this should be built into NetworkBuilder
        # update prng_seedcore params to provide jitter between trials
        for param_key in prng_seedcore_initial.keys():
            net.params[param_key] = (prng_seedcore_initial[param_key] +
                                     trial_idx)
        neuron_net = NetworkBuilder(net)
        dpl = _simulate_single_trial(neuron_net, trial_idx)
        if rank == 0:
            spikedata = neuron_net.get_data_from_neuron()
            sim_data.append((dpl, spikedata))

    # send results to stderr
    if rank == 0:
        data_iostream = io.BytesIO()

        # Force the use of bytes streams under Python 3
        if hasattr(data_iostream, 'buffer'):
            data_iostream = data_iostream.buffer

        # send back dpls and spikedata
        pickled_string = pickle.dumps(sim_data)

        # encode as base64 before sending to stderr
        repickled_bytes = codecs.encode(pickled_string,
                                        'base64')

        data_iostream.write(repickled_bytes + b"===")

    # flush anything in stderr (still points to str_err) to stdout
    sys.stderr.flush()
    sys.stdout.write(sys.stderr.getvalue())

    # restore the old stderr
    os.dup2(old_err_fd, stderr_fileno)
    sys.stderr = open(old_err_fd, 'w')
    os.close(null_fd)

    if rank == 0:
        data_str = data_iostream.getvalue().decode()
        sys.stderr.write(data_str)

    # close the StringIO object
    str_err.close()

    MPI.Finalize()
    return 0


if __name__ == '__main__':
    try:
        rc = run_mpi_simulation()
    except Exception as e:
        # This can be useful to indicate the problem to the
        # caller (in parallel_backends.py)
        print("Exception: %s" % e)
        rc = 2
    sys.exit(rc)
