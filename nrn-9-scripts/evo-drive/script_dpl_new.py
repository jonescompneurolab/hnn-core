"""Re-run the same simulation as script_dpl_old.py, save to dpl_new.txt,
then plot and print both dipoles along with their difference.

Then re-run the simulations of the HNN textbook notebooks (see
textbook_sims.py), save their dipoles to textbook_dpls/new/, and compare them
with those saved by script_dpl_old.py in textbook_dpls/old/.

Any command-line arguments restrict the textbook notebooks run and compared to
those whose names contain one of them, e.g.
``python script_dpl_new.py gamma alpha``.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import textbook_sims

from hnn_core import jones_2009_model, read_dipole, simulate_dipole
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
dpls[0].write("dpl_new.txt")

dpl_old = read_dipole("dpl_old.txt")
dpl_new = read_dipole("dpl_new.txt")

print("dpl_old data:")
print(dpl_old.data["agg"])
print("dpl_new data:")
print(dpl_new.data["agg"])

diff = dpl_new.data["agg"] - dpl_old.data["agg"]
print("difference (new - old):")
print(diff)

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
dpl_old.plot(ax=ax[0], show=False)
ax[0].set_title("dpl_old")
dpl_new.plot(ax=ax[1], show=False)
ax[1].set_title("dpl_new")
ax[2].plot(dpl_old.times, diff)
ax[2].set_title("difference (new - old)")
ax[2].set_xlabel("Time (ms)")
plt.tight_layout()

patterns = sys.argv[1:]
textbook_sims.run_all("new", patterns)
textbook_sims.compare("old", "new", patterns)

plt.show()
