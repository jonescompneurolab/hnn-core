from mpi4py import MPI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

logger.info(f"Process {rank} out of {size} processors")
