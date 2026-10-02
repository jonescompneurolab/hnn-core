"""Simulate a simple dipole and export it to dpl_old.txt, then run the
simulations of the HNN textbook notebooks (see textbook_sims.py) and save their
dipoles to textbook_dpls/old/.

Any command-line arguments restrict the textbook notebooks run to those whose
names contain one of them, e.g. ``python script_dpl_old.py gamma alpha``.
"""

import os
import sys
from pathlib import Path

import textbook_sims

from hnn_core import jones_2009_model, simulate_dipole
from hnn_core.parallel_backends import MPIBackend

# MPIBackend's child processes import hnn_core from the current directory
# first, so run from here rather than from an hnn-core checkout
os.chdir(Path(__file__).resolve().parent)

net = jones_2009_model()
net.add_evoked_drive(
    "evprox1",
    mu=40,
    sigma=4,
    numspikes=1,
    weights_ampa={"L2_pyramidal": 0.001, "L5_pyramidal": 0.001},
    location="proximal",
)

with MPIBackend():
    dpls = simulate_dipole(net, tstop=170, n_trials=1)
dpls[0].write("dpl_old.txt")

textbook_sims.run_all("old", sys.argv[1:])
