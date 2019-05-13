"""import NEURON module"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>

from neuron import h


def create_parallel_context(n_jobs=1):
    """Create parallel context."""
    global rank, nhosts, cvode, pc
    nhosts = n_jobs
    rank = 0
    pc = h.ParallelContext(nhosts)  # MPI: Initialize the ParallelContext class
    pc.done()
    nhosts = int(pc.nhost())  # Find number of hosts
    rank = int(pc.id())     # rank or node number (0 will be the master)
    cvode = h.CVode()

    if rank == 0:
        pc.gid_clear()
