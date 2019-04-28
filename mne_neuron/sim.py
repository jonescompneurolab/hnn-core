# import NEURON module
from neuron import h

#------------------------------------------------------------------------------
# Create parallel context
#------------------------------------------------------------------------------
def createParallelContext ():
    global rank, nhosts, cvode, pc
    rank = 0
    pc = h.ParallelContext() # MPI: Initialize the ParallelContext class
    pc.done()
    nhosts = int(pc.nhost()) # Find number of hosts
    rank = int(pc.id())     # rank or node number (0 will be the master)
    cvode = h.CVode()

    if rank==0:
        pc.gid_clear()
