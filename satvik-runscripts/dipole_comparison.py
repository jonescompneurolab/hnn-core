import pickle
import numpy as np

with open("dpl1_orignal.pkl", "rb") as f:
    dpl1 = pickle.load(f)

with open("dpl2_syanpse_tree.pkl", "rb") as f:
    dpl2 = pickle.load(f)

with open("net1_orignal.pkl", "rb") as f:
    net1 = pickle.load(f)

with open("net2_syanpse_tree.pkl", "rb") as f:
    net2 = pickle.load(f)

print("=" * 60)
print("DIPOLE COMPARISON")
print("=" * 60)

for key in dpl1[0].data.keys():
    arr1 = np.asarray(dpl1[0].data[key])
    arr2 = np.asarray(dpl2[0].data[key])
    print(f"\n{key}")
    print("Equal      :", np.array_equal(arr1, arr2))
    print("Allclose   :", np.allclose(arr1, arr2))
    print("Max diff   :", np.max(np.abs(arr1 - arr2)))
    print("Mean diff  :", np.mean(np.abs(arr1 - arr2)))

print("\nTime vector:", np.array_equal(dpl1[0].times, dpl2[0].times))

print("\n" + "=" * 60)
print("NETWORK COMPARISON")
print("=" * 60)

print("Connections:", len(net1.connectivity), len(net2.connectivity))
print("Connectivity equal:", net1.connectivity == net2.connectivity)
print("GID ranges equal :", net1.gid_ranges == net2.gid_ranges)
print("Cell types equal :", net1.cell_types.keys() == net2.cell_types.keys())
print("Drives equal     :", net1.external_drives == net2.external_drives)
print("Biases equal     :", net1.external_biases == net2.external_biases)

for i, (c1, c2) in enumerate(zip(net1.connectivity, net2.connectivity)):
    if c1 != c2:
        print(f"\nDifference at connection {i}")
        print("Original:", c1)
        print("Modified:", c2)