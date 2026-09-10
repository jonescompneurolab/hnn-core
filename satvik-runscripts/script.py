from hnn_core import neymotin_2020_model, simulate_dipole
net_a = neymotin_2020_model(use_dataframe=True)
net_a.add_evoked_drive(
    "evprox",
    mu=40,
    sigma=5,
    numspikes=1,
    location="proximal",
    weights_ampa={
        "L2_pyramidal": 0.01,
        "L5_pyramidal": 0.01,
    },
    synaptic_delays={
        "L2_pyramidal": 0.1,
        "L5_pyramidal": 0.1,
    },
)
dpls_a = simulate_dipole(net_a, tstop=170, dt=0.025)
net_b = neymotin_2020_model(use_dataframe=False)

net_b.add_evoked_drive(
    "evprox",
    mu=40,
    sigma=5,
    numspikes=1,
    location="proximal",
    weights_ampa={
        "L2_pyramidal": 0.01,
        "L5_pyramidal": 0.01,
    },
    synaptic_delays={
        "L2_pyramidal": 0.1,
        "L5_pyramidal": 0.1,
    },
)
dpls_b = simulate_dipole(net_b, tstop=170, dt=0.025)

import numpy as np

print(np.allclose(
    dpls_a[0].data['agg'],
    dpls_b[0].data['agg']
))



