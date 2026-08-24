"""Simulate a simple dipole and export it to dpl_old.txt."""

import hnn_core
from hnn_core import simulate_dipole, jones_2009_model
from hnn_core.parallel_backends import MPIBackend

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
