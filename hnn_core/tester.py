from neuron import h
from hnn_core import jones_2009_model
from hnn_core.network_builder import NetworkBuilder

net = jones_2009_model()
nb = NetworkBuilder(net)

cell = nb._cells[35]  
for syn_key, syn in cell._nrn_synapses.items():
    seg = syn.get_segment()
    print(f"{syn_key}  nseg={seg.sec.nseg}  x={seg.x}")