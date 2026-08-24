"""Simulate a simple dipole and export it to dpl_old.txt.

Also records the somatic membrane potential of the first cell of each
cell type and saves it to vsec_old.npz.
"""

import numpy as np

import hnn_core
from hnn_core import simulate_dipole, jones_2009_model
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
dpls[0].write("dpl_old.txt")

vsec_old = dict()
for cell_type in net.cell_types:
    first_gid = net.gid_ranges[cell_type][0]
    vsec_old[cell_type] = np.array(net.cell_response.vsec[0][first_gid]["soma"])

np.savez("vsec_old.npz", **vsec_old)
