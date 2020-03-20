"""Script for running parallel simulations with MPI"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>


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
    import sys

    old_stderr = sys.stderr
    sys.stderr = sys.stdout

    from hnn_core.neuron import _neuron_network
    from hnn_core.dipole import _simulate_single_trial

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
    neuron_net = _neuron_network(params)

    dpls = []
    for trial in range(params['N_trials']):
        dpls.append(_simulate_single_trial(neuron_net))

    spikedata = neuron_net.get_data_from_neuron()

    sys.stderr = old_stderr
    # send results to stderr
    if rank == 0:
        stream_err = sys.stderr
        # Force the use of bytes streams under Python 3
        if hasattr(sys.stderr, 'buffer'):
            stream_err = sys.stderr.buffer

        repickled_bytes = codecs.encode(pickle.dumps((dpls, spikedata)),
                                        'base64')
        stream_err.write(repickled_bytes)
        stream_err.close()
    MPI.Finalize()


if __name__ == '__main__':
    rc = run_mpi_simulation()
