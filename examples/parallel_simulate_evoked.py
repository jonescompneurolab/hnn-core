"""
========================
Parallel Simulate dipole
========================

This example demonstrates how to simulate a dipole can be called by
MPI via nrniv for running the simulation in parallel.
"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

from subprocess import Popen, PIPE
import shlex
import os.path as op
from os import getcwd
from sys import platform

###############################################################################
# Let us import hnn_core for the path

import hnn_core
hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')

###############################################################################
# Get the number of cores on this system and prepare the command to run MPI.
# Note that if we don't use psutil, we might get twice the number of cores
# that MPI can acutally run on. We can run on those cores with the MPI option
# `--use-hwthread-cpus`, which will be slightly slower than using the number
# of that MPI can actually run on.

try:
    import psutil
    n_cores = psutil.cpu_count(logical=False)
    mpicmd = 'mpiexec -np ' + str(n_cores) + ' '
except ImportError:
    import multiprocessing
    n_cores = multiprocessing.cpu_count()
    mpicmd = 'mpiexec -np ' + str(n_cores) + ' --use-hwthread-cpus '

nrnivcmd = 'nrniv -python -mpi -nobanner ' + op.join(hnn_core_root,
                                                     'examples',
                                                     'simulate_evoked.py')
cmd = mpicmd + nrnivcmd

###############################################################################
# The full command to run. Now we need to split into shell arguments for
# passing to subprocess.Popen
print(cmd)
cmdargs = shlex.split(cmd, posix="win" not in platform)

###############################################################################
# Run the simulation in parallel!

proc = Popen(cmdargs, stdout=PIPE, stderr=PIPE, cwd=getcwd(),
             universal_newlines=True)

###############################################################################
# Read the output while waiting for job to finish

failed = False
while True:
    status = proc.poll()
    if status is not None:
        if not status == 0:
            print("Simulation exited with return code %d. Stderr:" % status)
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
    exit(1)
