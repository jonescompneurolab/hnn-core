from hnn_core import jones_2009_model , simulate_dipole
import time 
import tracemalloc

#maybe for validation or gettign a better average -> we can run loop and take average
net=jones_2009_model()
net.add_evoked_drive(
        'ev', mu=50, sigma=5, numspikes=1,
        location='proximal',
        weights_ampa={
            'L2_pyramidal': 0.001, 'L5_pyramidal': 0.001,
            'L2_basket': 0.001, 'L5_basket': 0.001
        },
        synaptic_delays={
            'L2_pyramidal': 0.1, 'L5_pyramidal': 0.1,
            'L2_basket': 0.1, 'L5_basket': 0.1
        }
    )


tracemalloc.start()

timestart=time.perf_counter()
simulate_dipole(net,tstop=170,record_isec=False);
timetaken=time.perf_counter() - timestart
total ,peak =tracemalloc.get_traced_memory() 
tracemalloc.stop()
net2=jones_2009_model()
net2.add_evoked_drive(
        'ev', mu=50, sigma=5, numspikes=1,
        location='proximal',
        weights_ampa={
            'L2_pyramidal': 0.001, 'L5_pyramidal': 0.001,
            'L2_basket': 0.001, 'L5_basket': 0.001
        },
        synaptic_delays={
            'L2_pyramidal': 0.1, 'L5_pyramidal': 0.1,
            'L2_basket': 0.1, 'L5_basket': 0.1
        }
    )
tracemalloc.start()

timestart2=time.perf_counter()
simulate_dipole(net2,tstop=170,record_isec='all');
timetaken2=time.perf_counter()-timestart2
total2 ,peak2 =tracemalloc.get_traced_memory() 
tracemalloc.stop()

print(f"elapsed time in record_isec=None is {timetaken} seconds and total memory is{float(total/1024/1024)}MB and peak memory is {float(peak/1024/1024)}MB")
print(f"elapsed time in record_isec=all is {timetaken2} seconds and total memory is {float(total2/1024/1024/1024)}GB and peak memory is{float(peak2/1024/1024/1024)}GB ")