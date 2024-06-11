from mpi4py import MPI
import logging
from neuron import h
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

logger.info(f"Process {rank} out of {size} processors")
# Initialize NEURON with MPI support
h('''
objref pc
pc = new ParallelContext()
''')

# Check if NEURON MPI is enabled
is_mpi_enabled = int(h.pc.nhost() > 1)

if is_mpi_enabled:
    logger.info(f"NEURON MPI is enabled. Running on {int(h.pc.nhost())} processes")
else:
    logger.info("NEURON MPI is not enabled.")

if rank == 0:
    print("NEURON MPI test completed")