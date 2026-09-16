from hnn_core import neymotin_2020_model, simulate_dipole , simple
net_a = neymotin_2020_model(use_dataframe=True)
net_a.add_evoked_drive(
    "evprox",
    mu=40,
    sigma=5,
    numspikes=1,
    location="distal",
    weights_ampa={
        "L2_pyramidal": 0.01,
        "L5_pyramidal": 0.01,
    },
    synaptic_delays={
        "L2_pyramidal": 0.1,
        "L5_pyramidal": 0.1,
    },
)

dpls=simulate_dipole(net_a,tstop=30)


print(simple.total)