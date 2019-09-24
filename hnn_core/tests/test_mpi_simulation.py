import os.path as op

from numpy import loadtxt
from numpy.testing import assert_array_equal

from mne.utils import _fetch_file
import hnn_core
from hnn_core import Params


def test_simulate_dipole_mpi():
    """Test to check if simulate dipole can be called with MPI."""
    from mpi4py import MPI
    from psutil import cpu_count

    # need to get physical CPU cores using psutil and leave one CPU
    # for the current script (bug in OpenMPI < 4.0) or MPI will fail
    # to spawn nrniv workers
    n_core = cpu_count(logical=False) - 1
    if n_core < 1:
        n_core = 1

    # spawn NEURON sim
    args = ['-python', '-mpi', '-nobanner', 'python',
            op.join(op.dirname(hnn_core.__file__), 'tests',
                    'test_compare_hnn.py')]

    mpiinfo = MPI.Info().Create()
    mpiinfo.Set('env', 'OMPI_MCA_btl=^openib')
    mpiinfo.Set('env', 'OMPI_MCA_btl_base_warn_component_unused=0')
    mpiinfo.Set('env', 'OMPI_MCA_rmaps_base_oversubscribe=1')
    child = MPI.COMM_SELF.Spawn('nrniv', args=args, maxprocs=n_core,
                                info=mpiinfo)

    # get dipole data and params
    data_url = ('https://raw.githubusercontent.com/hnnsolver/'
                'hnn-core/test_data/dpl.txt')
    if not op.exists('dpl.txt'):
        _fetch_file(data_url, 'dpl.txt')
    dpl_master = loadtxt('dpl.txt')

    hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')

    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = Params(params_fname)

    # send params and dipole data to spawned nrniv procs to start sim
    simdata = (params, dpl_master)
    child.bcast(simdata, root=MPI.ROOT)

    # wait to recevie results from child rank 0
    dpl = child.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
    child.Barrier()
    child.Disconnect()

    fname = './dpl2.txt'
    dpl.write(fname)

    dpl_pr = loadtxt(fname)
    assert_array_equal(dpl_pr[:, 2], dpl_master[:, 2])  # L2
    assert_array_equal(dpl_pr[:, 3], dpl_master[:, 3])  # L5
