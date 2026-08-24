"""Re-run the same simulation as script_dpl_old.py, save to dpl_new.txt,
then plot and print both dipoles along with their difference.

Also records the somatic membrane potential of the first cell of each
cell type, compares it against the values saved by script_dpl_old.py in
vsec_old.npz, and plots the per-cell-type differences.
"""

import numpy as np
import matplotlib.pyplot as plt

from hnn_core import simulate_dipole, jones_2009_model, read_dipole
from hnn_core.parallel_backends import MPIBackend

net = jones_2009_model()
# net.add_evoked_drive(
#     "evprox1",
#     mu=40,
#     sigma=4,
#     numspikes=1,
#     weights_ampa={"L2_pyramidal": 0.001, "L5_pyramidal": 0.001},
#     location="proximal",
# )
net.add_tonic_bias(cell_type="L5_pyramidal", amplitude=1.0, tstop=170)

with MPIBackend():
    dpls = simulate_dipole(net, tstop=170, n_trials=1, record_vsec="soma")
dpls[0].write("dpl_new.txt")

vsec_new = dict()
for cell_type in net.cell_types:
    first_gid = net.gid_ranges[cell_type][0]
    vsec_new[cell_type] = np.array(net.cell_response.vsec[0][first_gid]["soma"])

np.savez("vsec_new.npz", **vsec_new)

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
plt.show()

vsec_old = np.load("vsec_old.npz")
cell_types = list(net.cell_types)
times = dpl_old.times

fig, ax = plt.subplots(len(cell_types), 3, sharex=True, figsize=(12, 3 * len(cell_types)))
for row, cell_type in enumerate(cell_types):
    v_old = vsec_old[cell_type]
    v_new = vsec_new[cell_type]
    v_diff = v_new - v_old

    print(f"{cell_type} soma Vm (old):")
    print(v_old)
    print(f"{cell_type} soma Vm (new):")
    print(v_new)
    print(f"{cell_type} soma Vm difference (new - old):")
    print(v_diff)

    ax[row, 0].plot(times, v_old)
    ax[row, 0].set_ylabel(cell_type)
    ax[row, 1].plot(times, v_new)
    ax[row, 2].plot(times, v_diff)

    if row == 0:
        ax[row, 0].set_title("soma Vm (old)")
        ax[row, 1].set_title("soma Vm (new)")
        ax[row, 2].set_title("difference (new - old)")

for a in ax[-1, :]:
    a.set_xlabel("Time (ms)")
plt.tight_layout()
plt.show()
