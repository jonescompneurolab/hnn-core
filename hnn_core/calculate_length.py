from hnn_core import jones_2009_model
net = jones_2009_model()
for cell_type_name, cell_type in net.cell_types.items():
    abc = cell_type['cell_object']#can improve namning
    print(f"{cell_type_name}")
    for sec_name, sec in abc.sections.items():
        nseg = sec.nseg
        positions = []
        for i in range(nseg):
            positions.append((i + 0.5) / nseg)
        print(f"{sec_name} L={sec.L} nseg={nseg} positions={positions}")