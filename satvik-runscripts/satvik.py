from hnn_core import neymotin_2020_model, simulate_dipole
import pandas as pd
net_a = neymotin_2020_model(use_dataframe=True)
a = net_a.connectivity_df
print(a.to_string)
