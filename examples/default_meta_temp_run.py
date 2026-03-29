# direct import and usage
from hnn_core.network_models import default_cell_metadata

print("default_cell_metadata, no network needed:")
for cell_name, meta in default_cell_metadata.items():
    print(f"\n  {cell_name}:")
    for key, value in meta.items():
        print(f"    {key:20s} = {value}")

# look up a single cell type's metadata directly
print("\nL5_pyramidal defaults:")
print(f"  {default_cell_metadata['L5_pyramidal']}")

# fill missing keys from defaults
partial = {"layer": "5", "morpho_type": "pyramidal"}
filled = {**default_cell_metadata["L5_pyramidal"], **partial}
print(f"\nPartial metadata   : {partial}")
print(f"Filled from defaults: {filled}")

# verifing the consistency
from hnn_core import jones_2009_model

net = jones_2009_model()

print("\n check-check time: constant == og network's metadata")
for cell_name in default_cell_metadata:
    live = net.cell_types[cell_name]["cell_metadata"]
    print(f"  {cell_name:20s} match = {live == default_cell_metadata[cell_name]}")
