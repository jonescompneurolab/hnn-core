from hnn_core import jones_2009_model

net = jones_2009_model()
cell = net.cell_types['L5_pyramidal']['cell_object']

for sec_name, sec in cell.sections.items():
    print(f"  {sec_name} nseg={sec.nseg}  syns={sec.syns}")