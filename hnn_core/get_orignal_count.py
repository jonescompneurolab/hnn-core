from hnn_core import jones_2009_model
net = jones_2009_model()
alltotal = 0
cell = net.cell_types['L5_pyramidal']['cell_object']
total = 0
for sec_name, sec in cell.sections.items():
    total += len(sec.syns)
print(f' L5_pyramidal per cell={total} and forwhole network {total*100}')
alltotal += total*100
cell = net.cell_types['L2_pyramidal']['cell_object']
total = 0
for sec_name, sec in cell.sections.items():
    total += len(sec.syns)
print(f'L2_pyramidal per cell={total} and for whole netowrk = {total*100}')
alltotal += total*100
cell = net.cell_types['L2_basket']['cell_object']
total = 0
for sec_name, sec in cell.sections.items():
    total += len(sec.syns)
print(f'L2_basketper cell={total}and for whole network={total*35}')
alltotal += total*35
cell = net.cell_types['L5_basket']['cell_object']
total = 0
for sec_name, sec in cell.sections.items():
    total += len(sec.syns)
print(f'L5_basket per cell={total}and for whole net{total*35}')
alltotal += total*35
print(f'ORIGINAL total = {alltotal}')