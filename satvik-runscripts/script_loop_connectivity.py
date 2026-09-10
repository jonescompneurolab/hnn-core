import numpy as np
from hnn_core import neymotin_2020_model, simulate_dipole
import pandas as pd
net_ney = neymotin_2020_model(use_data_frame=False)

for conn_idx, conn in enumerate(net_ney.connectivity):

    # for each iteration create a new network
    net1 = neymotin_2020_model(use_data_frame=True)
    net1.clear_connectivity()
    net1.conn_dataframe = pd.DataFrame()
    # this creates the connections old style 
    for c in net_ney.connectivity[:conn_idx + 1]:
        net1.add_connection(
            c["src_type"], c["target_type"], c["loc"], c["receptor"],
            c["nc_dict"]["A_weight"], c["nc_dict"]["A_delay"], c["nc_dict"]["lamtha"],
        )

    net2 = neymotin_2020_model(use_data_frame=False)
    net2.conn_dataframe = pd.DataFrame()
    net2.clear_connectivity()
    # now create with dataframe
    for c in net_ney.connectivity[:conn_idx + 1]:
        net2.add_connection(
            c["src_type"], c["target_type"], c["loc"], c["receptor"],
            c["nc_dict"]["A_weight"], c["nc_dict"]["A_delay"], c["nc_dict"]["lamtha"],
        )

    # if just testing intra-network connections, add the same tonic bias
    net1.add_evoked_drive('evprox', mu=40, sigma=5, numspikes=1,
        location='proximal',
        weights_ampa={'L2_pyramidal': 0.01, 'L5_pyramidal': 0.01},
        synaptic_delays={'L2_pyramidal': 0.1, 'L5_pyramidal': 0.1},)
    net2.add_evoked_drive('evprox',mu=40, sigma=5, numspikes=1,
        location='proximal',
        weights_ampa={'L2_pyramidal': 0.01, 'L5_pyramidal': 0.01},
        synaptic_delays={'L2_pyramidal': 0.1, 'L5_pyramidal': 0.1},)

    dpl1 = simulate_dipole(net1, tstop=170, record_vsec='all')
    dpl2 = simulate_dipole(net2, tstop=170, record_vsec='all')

    # compare dipoles, spike times, spike gids -> any difference for any connection? if so, which one
    diff = np.max(np.abs(dpl1[0].data['agg'] - dpl2[0].data['agg']))
    print(f"conn {conn_idx}: {conn['src_type']} -> {conn['target_type']} "
          f"({conn['receptor']}, {conn['loc']})  max_diff={diff:.3e}  "
          f"allclose={np.allclose(dpl1[0].data['agg'], dpl2[0].data['agg'])}")

    del net1, net2, dpl1, dpl2