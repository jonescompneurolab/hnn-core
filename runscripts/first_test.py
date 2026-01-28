

from hnn_core import jones_2009_model, MPIBackend, simulate_dipole

net = jones_2009_model()

# (for now) manually attach a new attribute to net
net._verbose = False

#breakpoint()
#print(net)

#with MPIBackend(n_procs=8):
dpl = simulate_dipole(net, tstop=50.0, n_trials=1)


'''
simulate_dipole(net, tstop=170, dt=0.025, verbose=False)

Not trivial:
Two code design choices

1) pass verbose to every function
2) have a private variable attached to the network net._verbose <- you have initialize a hidden variable “verbose" in net
If net._verbose==false
….


An additional idea: augmenting and adding to various “rear” functions
Expand and update functions to have all the arguments (because they are not currently up to date)
'''