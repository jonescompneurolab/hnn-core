import os.path as op
import sys

from numpy import loadtxt
from numpy.testing import assert_array_equal
import pytest

from mne.utils import _fetch_file
import hnn_core


def run_simulation(n_jobs):
    from hnn_core import simulate_dipole, Params, Network, get_rank

    hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')

    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = Params(params_fname)
    net = Network(params)
    dpl = simulate_dipole(net, n_jobs=n_jobs, n_trials=1)[0]

    if get_rank() == 0:
        # write the dipole to a file and compare
        fname = './dpl2.txt'
        dpl.write(fname)

        data_url = ('https://raw.githubusercontent.com/hnnsolver/'
                    'hnn-core/test_data/dpl.txt')
        if not op.exists('dpl.txt'):
            _fetch_file(data_url, 'dpl.txt')
        dpl_master = loadtxt('dpl.txt')

        dpl_pr = loadtxt(fname)
        assert_array_equal(dpl_pr[:, 2], dpl_master[:, 2])  # L2
        assert_array_equal(dpl_pr[:, 3], dpl_master[:, 3])  # L5


def spawn_mpi_simulation(n_jobs):
    from subprocess import Popen, PIPE
    import shlex
    from os import getcwd
    from sys import platform

    import psutil
    n_cores = psutil.cpu_count(logical=False)

    mpicmd = 'mpiexec -np ' + str(n_cores) + ' '
    nrnivcmd = 'nrniv -python -mpi -nobanner ' + \
               op.join(op.dirname(hnn_core.__file__),
                       'tests',
                       'test_compare_hnn.py')
    cmd = mpicmd + nrnivcmd

    ###########################################################################
    # Now we need to split the command into shell arguments for passing to
    # subprocess.Popen
    cmdargs = shlex.split(cmd, posix="win" not in platform)

    ###########################################################################
    # Start the simulation in parallel!
    proc = Popen(cmdargs, stdout=PIPE, stderr=PIPE, cwd=getcwd(),
                 universal_newlines=True)

    ###########################################################################
    # Read the output while waiting for job to finish

    failed = False
    while True:
        status = proc.poll()
        if status is not None:
            if not status == 0:
                print("Simulation exited with return code %d. Stderr:"
                      % status)
                failed = True
            else:
                # success
                break

        for line in iter(proc.stdout.readline, ""):
            print(line.strip())
        for line in iter(proc.stderr.readline, ""):
            print(line.strip())

        if failed:
            break

    # make sure spawn process is dead
    proc.kill()

    if failed:
        raise ValueError('Running MPI simulation failed')


def test_mpi_simulation():
    spawn_mpi_simulation(1)


def test_mpi_joblibs_incompat():
    pytest.raises(ValueError, spawn_mpi_simulation, 2)


if __name__ == '__main__':
    # read n_jobs, very long list of arguments with nrniv..
    try:
        n_jobs = int(sys.argv[-1])
    except ValueError:
        raise ValueError('Received bad argument for n_jobs: %s' % sys.argv[-1])

    run_simulation(n_jobs)
