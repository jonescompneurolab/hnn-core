from hnn_core import jones_2009_model
alltotal=0
net = jones_2009_model()
cell = net.cell_types['L5_pyramidal']['cell_object']
total=0
for sec_name, sec in cell.sections.items():
    print(f"  {sec_name} nseg={sec.nseg}  syns={sec.syns}")
    total+=sec.nseg*len(sec.syns)
print(f'so in each L5_pyramidal, total number of synapse woudkl be {total}')
print(f'so in whole network the total number of synapse would be{total*100} ')
alltotal+=total*100
cell = net.cell_types['L2_pyramidal']['cell_object']
total=0
for sec_name, sec in cell.sections.items():
    print(f"  {sec_name} nseg={sec.nseg}  syns={sec.syns}")
    total+=sec.nseg*len(sec.syns)
print(f'so in each L2_pyramidal, total number of synapse woudkl be {total}')
print(f'so in whole network the total number of synapse would be{total*100} ')
alltotal+=total*100
cell = net.cell_types['L2_basket']['cell_object']
total=0
for sec_name, sec in cell.sections.items():
    print(f"  {sec_name} nseg={sec.nseg}  syns={sec.syns}")
    total+=sec.nseg*len(sec.syns)
print(f'so in each L2_basket, total number of synapse woudkl be {total}')
print(f'so in whole network the total number of synapse would be{total*35} ')
alltotal+=total*35
cell = net.cell_types['L5_basket']['cell_object']
total=0
for sec_name, sec in cell.sections.items():
    print(f"  {sec_name} nseg={sec.nseg}  syns={sec.syns}")
    total+=sec.nseg*len(sec.syns)
print(f'so in each L5_basket, total number of synapse woudkl be {total}')
print(f'so in whole network the total number of synapse would be{total*35}')
alltotal+=total*35

print(f"total synapses in full network: {alltotal}")