from mpi4py import MPI
import logging
from neuron import h
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

logger.info(f"Process {rank} out of {size} processors")
# Test NEURON MPI support
try:
    h('''
    if (nrnmpi_use) {
        printf("NEURON MPI is enabled. Running on %d processes\\n", nrnmpi_numprocs_world)
    } else {
        printf("NEURON MPI is not enabled.\\n")
    }
    ''')
except Exception as e:
    logger.error(f"Error testing NEURON MPI support: {e}")

if rank == 0:
    print("NEURON MPI test completed")