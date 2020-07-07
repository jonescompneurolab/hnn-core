"""Script for running parallel simulations with MPI"""

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

    # temporarily use a StringIO object to capture stderr
    old_err_fd = os.dup(sys.stderr.fileno())
    str_err = io.StringIO()
    sys.stderr = str_err

    from hnn_core import Network
    from hnn_core.neuron import _neuron_network, _simulate_single_trial

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
    neuron_net = _neuron_network(net)

    dpls = []
    for trial in range(params['N_trials']):
        dpls.append(_simulate_single_trial(neuron_net))

    # send results to stderr
    if rank == 0:
        spikedata = neuron_net.get_data_from_neuron()

        # send back dpls and spikedata
        return_data = (dpls, spikedata)

        # pickle data
        pickled_string = pickle.dumps(return_data)

        # pad data before encoding, always add at least 4 "=" to mark end
        padding = len(pickled_string) % 4
        pickled_string += b"=" * padding
        pickled_string += b"=" * 4

        # encode as base64 before sending to stderr
        repickled_bytes = codecs.encode(pickled_string,
                                        'base64')

        data_iostream = io.BytesIO()

        # Force the use of bytes streams under Python 3
        if hasattr(data_iostream, 'buffer'):
            data_iostream = data_iostream.buffer

        data_iostream.write(repickled_bytes)

    if rank == 0:
        try:
            # flush anything in stderr (still points to str_err) to stdout
            sys.stderr.flush()
            sys.stdout.write(sys.stderr.getvalue())

            # restore the old stderr and write data to it
            sys.stderr = open(old_err_fd, 'w')
            data_str = data_iostream.getvalue().decode()
            sys.stderr.write(data_str)
        except Exception as e:
            print("Exception: %s" % e)

    else:
        sys.stderr.flush()
        sys.stdout.write(sys.stderr.getvalue())

    # close the StringIO object
    str_err.close()

    MPI.Finalize()
    return 0


if __name__ == '__main__':
    rc = run_mpi_simulation()
    sys.exit(rc)
