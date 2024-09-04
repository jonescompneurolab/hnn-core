"""Parallel backends"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Mainak Jas <mainakjas@gmail.com>

import os
import sys
import re
import multiprocessing
import shlex
import pickle
import base64
from warnings import warn
from subprocess import Popen, PIPE, TimeoutExpired
import binascii
from queue import Queue, Empty
from threading import Thread, Event

from .cell_response import CellResponse
from .dipole import Dipole
from .network_builder import _simulate_single_trial

_BACKEND = None


def _thread_handler(event, out, queue):
    while not event.is_set():
        line = out.readline()
        if line == '':
            break
        queue.put(line)


def _gather_trial_data(sim_data, net, n_trials, postproc):
    """Arrange data by trial

    To be called after simulate(). Returns list of Dipoles, one for each trial,
    and saves spiking info in net (instance of Network).
    """
    dpls = list()

    # Create array of equally sampled time points for simulating currents
    cell_type_names = list(net.cell_types.keys())
    cell_response = CellResponse(times=sim_data[0]['times'],
                                 cell_type_names=cell_type_names)
    net.cell_response = cell_response

    for idx in range(n_trials):

        # cell response
        net.cell_response._spike_times.append(sim_data[idx]['spike_times'])
        net.cell_response._spike_gids.append(sim_data[idx]['spike_gids'])
        net.cell_response.update_types(net.gid_ranges)
        net.cell_response._vsec.append(sim_data[idx]['vsec'])
        net.cell_response._isec.append(sim_data[idx]['isec'])
        net.cell_response._ca.append(sim_data[idx]['ca'])

        # extracellular array
        for arr_name, arr in net.rec_arrays.items():
            # voltages is a n_trials x n_contacts x n_samples array
            arr._data.append(sim_data[idx]['rec_data'][arr_name])
            arr._times = sim_data[idx]['rec_times'][arr_name]

        # dipole
        dpl = Dipole(times=sim_data[idx]['times'],
                     data=sim_data[idx]['dpl_data'])

        N_pyr_x = net._N_pyr_x
        N_pyr_y = net._N_pyr_y
        dpl._baseline_renormalize(N_pyr_x, N_pyr_y)  # XXX cf. #270
        dpl._convert_fAm_to_nAm()  # always applied, cf. #264
        if postproc:
            window_len = net._params['dipole_smooth_win']  # specified in ms
            fctr = net._params['dipole_scalefctr']
            if window_len > 0:  # param files set this to zero for no smoothing
                dpl.smooth(window_len=window_len)
            if fctr > 0:
                dpl.scale(fctr)
        dpls.append(dpl)

    return dpls


def _get_mpi_env():
    """Set some MPI environment variables."""
    my_env = os.environ.copy()
    # For Linux systems
    if sys.platform != 'win32':
        my_env["OMPI_MCA_btl_base_warn_component_unused"] = '0'

    if 'darwin' in sys.platform:
        my_env["PMIX_MCA_gds"] = "^ds12"  # open-mpi/ompi/issues/7516
        my_env["TMPDIR"] = "/tmp"  # open-mpi/ompi/issues/2956
    return my_env


def run_subprocess(command, obj, timeout, proc_queue=None, *args, **kwargs):
    """Run process and communicate with it.
    Parameters
    ----------
    command : list of str | str
        Command to run as subprocess (see subprocess.Popen documentation).
    obj : object
        The object to write to stdin after starting child process
        with MPI command.
    timeout : float
        The number of seconds to wait for a process without output.
    *args, **kwargs : arguments
        Additional arguments to pass to subprocess.Popen.
    Returns
    -------
    child_data : object
        The data returned by the child process.
    """
    proc_data_bytes = b''
    # each loop while waiting will involve two Queue.get() timeouts, each
    # 0.01s. This calculation will error on the side of a longer timeout
    # than is specified because more is done each loop that just Queue.get()
    timeout_cycles = timeout / 0.02

    pickled_obj = base64.b64encode(pickle.dumps(obj))

    # non-blocking adapted from https://stackoverflow.com/questions/375427/non-blocking-read-on-a-subprocess-pipe-in-python#4896288  # noqa: E501
    out_q = Queue()
    err_q = Queue()

    threads_started = False

    try:
        proc = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE, *args,
                     **kwargs)

        # now that the process has started, add it to the queue
        # used by MPIBackend.terminate()
        if proc_queue is not None:
            proc_queue.put(proc)

        # set up polling first so all of child's stdout/stderr
        # gets captured
        event = Event()
        out_t = Thread(target=_thread_handler,
                       args=(event, proc.stdout, out_q))
        err_t = Thread(target=_thread_handler,
                       args=(event, proc.stderr, err_q))
        out_t.start()
        err_t.start()
        threads_started = True
        data_received = False
        sent_network = False
        count_since_last_output = 0

        # loop while the process is running the simulation
        while True:
            child_terminated = proc.poll() is not None

            if not data_received:
                if _echo_child_output(out_q):
                    count_since_last_output = 0
                else:
                    count_since_last_output += 1
                # look for data in stderr and print child stdout
                data_len, proc_data_bytes = _get_data_from_child_err(err_q)
                if data_len > 0:
                    data_received = True
                    _write_child_exit_signal(proc.stdin)
                elif child_terminated:
                    # child terminated early, and we already
                    # captured output left in queues
                    warn("Child process failed unexpectedly")
                    kill_proc_name('nrniv')
                    break

            if not sent_network:
                # Send network object to child so it can start
                try:
                    _write_net(proc.stdin, pickled_obj)
                except BrokenPipeError:
                    # child failed during _write_net(). get the
                    # output and break out of loop on the next
                    # iteration
                    warn("Received BrokenPipeError exception. "
                         "Child process failed unexpectedly")
                    continue
                else:
                    sent_network = True
                    # This is not the same as "network received", but we
                    # assume it was successful and move on to waiting for
                    # data in the next loop iteration.

            if child_terminated and data_received:
                # both exit conditions have been met (also we know that
                # the network has been sent)
                break

            if not child_terminated and \
                    count_since_last_output > timeout_cycles:
                warn("Timeout exceeded while waiting for child process output"
                     ". Terminating...")
                kill_proc_name('nrniv')
                break
    except KeyboardInterrupt:
        warn("Received KeyboardInterrupt. Stopping simulation process...")

    if threads_started:
        # stop the threads
        event.set()  # close signal
        out_t.join()
        err_t.join()

    # wait for the process to terminate. we need use proc.communicate to
    # read any output at its end of life.
    try:
        outs, errs = proc.communicate(timeout=1)
    except TimeoutExpired:
        proc.kill()
        # wait for output again after kill signal
        outs, errs = proc.communicate(timeout=1)

    sys.stdout.write(outs)
    sys.stdout.write(errs)

    if proc.returncode is None:
        # It's theoretically possible that we have received data
        # and exited the loop above, but the child process has not
        # yet terminated. This is unexpected unless KeyboarInterrupt
        # is caught
        proc.terminate()
        try:
            proc.wait(1)  # wait maximum of 1s
        except TimeoutExpired:
            warn("Could not kill python subprocess: PID %d" % proc.pid)

    if not proc.returncode == 0:
        # simulation failed with a numeric return code
        raise RuntimeError("MPI simulation failed. Return code: %d" %
                           proc.returncode)

    child_data = _process_child_data(proc_data_bytes, data_len)

    # clean up the queue
    try:
        proc_queue.get_nowait()
    except Empty:
        pass

    return proc, child_data


def _process_child_data(data_bytes, data_len):
    """Process the data returned by child process.

    Parameters
    ----------
    data_bytes : str
        The data bytes

    Returns
    -------
    data_unpickled : object
        The unpickled data.
    """
    if not data_len == len(data_bytes):
        # This is indicative of a failure. For debugging purposes.
        warn("Length of received data unexpected. Expecting %d bytes, "
             "got %d" % (data_len, len(data_bytes)))

    if len(data_bytes) == 0:
        raise RuntimeError("MPI simulation didn't return any data")

    # decode base64 byte string
    try:
        data_pickled = base64.b64decode(data_bytes, validate=True)
    except binascii.Error:
        # This is here for future debugging purposes. Unit tests can't
        # reproduce an incorrectly padded string, but this has been an
        # issue before
        raise ValueError("Incorrect padding for data length %d bytes" %
                         len(data_len) + " (mod 4 = %d)" %
                         (len(data_len) % 4))

    # unpickle the data
    return pickle.loads(data_pickled)


def _echo_child_output(out_q):
    out = ''
    while True:
        try:
            out += out_q.get(timeout=0.01)
        except Empty:
            break

    if len(out) > 0:
        sys.stdout.write(out)
        return True
    return False


def _get_data_from_child_err(err_q):
    err = ''
    data_length = 0
    data_bytes = b''

    while True:
        try:
            err += err_q.get(timeout=0.01)
        except Empty:
            break

    # check for data signal
    extracted_data = _extract_data(err, 'data')
    if len(extracted_data) > 0:
        # _extract_data only returns data when signals on
        # both sides were seen

        err = err.replace('@start_of_data@', '')
        err = err.replace(extracted_data, '')
        data_length = _extract_data_length(err, 'data')
        err = err.replace('@end_of_data:%d@\n' % data_length, '')
        data_bytes = extracted_data.encode()

    # print the rest of the child's stderr to our stdout
    sys.stdout.write(err)

    return data_length, data_bytes


def _has_mpi4py():
    """Determine if mpi4py is present."""
    try:
        import mpi4py  # noqa
    except ImportError:
        return False
    else:
        return True


def _has_psutil():
    """Determine if psutil is present."""
    try:
        import psutil  # noqa
    except ImportError:
        return False
    else:
        return True


def requires_mpi4py(function):
    """Decorator for testing functions that require MPI."""
    import pytest

    try:
        import mpi4py
        assert hasattr(mpi4py, '__version__')
        skip = False
    except (ImportError, ModuleNotFoundError) as err:
        if "TRAVIS_OS_NAME" not in os.environ:
            skip = True
        else:
            raise ImportError(err)
    reason = 'mpi4py not available'
    return pytest.mark.skipif(skip, reason=reason)(function)


def requires_psutil(function):
    """Decorator for testing functions that require psutil."""
    import pytest

    try:
        import psutil
        assert hasattr(psutil, '__version__')
        skip = False
    except (ImportError, ModuleNotFoundError) as err:
        if "TRAVIS_OS_NAME" not in os.environ:
            skip = True
        else:
            raise ImportError(err)
    reason = 'psutil not available'
    return pytest.mark.skipif(skip, reason=reason)(function)


def _extract_data_length(data_str, object_name):
    data_len_match = re.search('@end_of_%s:' % object_name + r'(\d+)@',
                               data_str)
    if data_len_match is not None:
        return int(data_len_match.group(1))
    else:
        raise ValueError("Couldn't find data length in string")


def _extract_data(data_str, object_name):
    start_idx = 0
    end_idx = 0
    start_match = re.search('@start_of_%s@' % object_name, data_str)
    if start_match is not None:
        start_idx = start_match.end()
    else:
        # need start signal
        return ''

    end_match = re.search('@end_of_%s:' % object_name + r'\d+@', data_str)
    if end_match is not None:
        end_idx = end_match.start()

    return data_str[start_idx:end_idx]


# Next 3 functions are from HNN. Will move here. They require psutil
def _kill_procs(procs):
    """Tries to terminate processes in a list before sending kill signal"""
    from psutil import wait_procs, NoSuchProcess

    # try terminate first
    for p in procs:
        try:
            p.terminate()
        except NoSuchProcess:
            pass
    _, alive = wait_procs(procs, timeout=3)

    # now try kill
    for p in alive:
        p.kill()
    _, alive = wait_procs(procs, timeout=3)

    return alive


def _get_procs_running(proc_name):
    """Return a list of processes currently running"""
    from psutil import process_iter

    process_list = []
    for p in process_iter(attrs=["name", "exe", "cmdline"]):
        if proc_name == p.info['name'] or \
                (p.info['exe'] is not None and
                 os.path.basename(p.info['exe']) == proc_name) or \
                (p.info['cmdline'] and
                 p.info['cmdline'][0] == proc_name):
            process_list.append(p)
    return process_list


def kill_proc_name(proc_name):
    """Make best effort to kill processes

    Parameters
    ----------
    proc_name : str
        A string to match process names against and kill all matches

    Returns
    -------
    killed_procs : bool
        True if any processes were killed
    """

    killed_procs = False
    procs = _get_procs_running(proc_name)
    if len(procs) > 0:
        running = _kill_procs(procs)
        if len(running) > 0:
            if len(running) < len(procs):
                killed_procs = True
            pids = [str(proc.pid) for proc in running]
            warn("Failed to kill nrniv process(es) %s" %
                 ','.join(pids))
        else:
            killed_procs = True

    return killed_procs


def _write_net(stream, pickled_net):
    stream.flush()
    stream.write('@start_of_net@')
    stream.write(pickled_net.decode())
    stream.write('@end_of_net:%d@\n' % len(pickled_net))
    stream.flush()


def _write_child_exit_signal(stream):
    stream.flush()
    stream.write('@data_received@\n')
    stream.flush()


class JoblibBackend(object):
    """The JoblibBackend class.

    Parameters
    ----------
    n_jobs : int | None
        The number of jobs to start in parallel. If None, then 1 trial will be
        started without parallelism

    Attributes
    ----------
    n_jobs : int
        The number of jobs to start in parallel
    """
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def _parallel_func(self, func):
        if self.n_jobs != 1:
            try:
                from joblib import Parallel, delayed
            except ImportError:
                warn('joblib not installed. Cannot run in parallel.')
                self.n_jobs = 1
        if self.n_jobs == 1:
            my_func = func
            parallel = list
        else:
            parallel = Parallel(self.n_jobs)
            my_func = delayed(func)

        return parallel, my_func

    def __enter__(self):
        global _BACKEND

        self._old_backend = _BACKEND
        _BACKEND = self

        return self

    def __exit__(self, type, value, traceback):
        global _BACKEND

        _BACKEND = self._old_backend

    def simulate(self, net, tstop, dt, n_trials, postproc=False):
        """Simulate the HNN model

        Parameters
        ----------
        net : Network object
            The Network object specifying how cells are
            connected.
        n_trials : int
            Number of trials to simulate.
        tstop : float
            The simulation stop time (ms).
        dt : float
            The integration time step of h.CVode (ms)
        postproc : bool
            If False, no postprocessing applied to the dipole

        Returns
        -------
        dpl: list of Dipole
            The Dipole results from each simulation trial
        """

        print(f"Joblib will run {n_trials} trial(s) in parallel by "
              f"distributing trials over {self.n_jobs} jobs.")
        parallel, myfunc = self._parallel_func(_simulate_single_trial)
        sim_data = parallel(myfunc(net, tstop, dt, trial_idx) for
                            trial_idx in range(n_trials))

        dpls = _gather_trial_data(sim_data, net=net, n_trials=n_trials,
                                  postproc=postproc)

        return dpls


class MPIBackend(object):
    """The MPIBackend class.

    Parameters
    ----------
    n_procs : int | None
        The number of MPI processes requested by the user. If None, then will
        attempt to detect number of cores (including hyperthreads) and start
        parallel simulation over all of them.
    mpi_cmd : str
        The name of the mpi launcher executable. Will use 'mpiexec'
        (openmpi) by default.

    Attributes
    ----------

    n_procs : int
        The number of processes MPI will actually use (spread over cores). If 1
        is specified or mpi4py could not be loaded, the simulation will be run
        with the JoblibBackend
    mpi_cmd : list of str
        The mpi command with number of procs and options to be passed to Popen
    expected_data_length : int
        Used to check consistency between data that was sent and what
        MPIBackend received.
    proc_queue : threading.Queue
        A Queue object to hold process handles from Popen in a thread-safe way.
        There will be a valid process handle present the queue when a MPI
        åsimulation is running.
    """
    def __init__(self, n_procs=None, mpi_cmd='mpiexec'):
        self.expected_data_length = 0
        self.proc = None
        self.proc_queue = Queue()

        n_logical_cores = multiprocessing.cpu_count()
        if n_procs is None:
            self.n_procs = n_logical_cores
        else:
            self.n_procs = n_procs

        # did user try to force running on more cores than available?
        oversubscribe = False
        if self.n_procs > n_logical_cores:
            oversubscribe = True

        hyperthreading = False

        if _has_mpi4py() and _has_psutil():
            import psutil

            n_physical_cores = psutil.cpu_count(logical=False)

            # detect if we need to use hwthread-cpus with mpiexec
            if self.n_procs > n_physical_cores:
                hyperthreading = True

        else:
            packages = list()
            if not _has_mpi4py():
                packages += ['mpi4py']
            if not _has_psutil():
                packages += ['psutil']
            packages = ' and '.join(packages)
            warn(f'{packages} not installed. Will run on single processor')
            self.n_procs = 1

        self.mpi_cmd = mpi_cmd

        if hyperthreading:
            self.mpi_cmd += ' --use-hwthread-cpus'

        if oversubscribe:
            self.mpi_cmd += ' --oversubscribe'

        self.mpi_cmd += ' -np ' + str(self.n_procs)

        self.mpi_cmd += ' nrniv -python -mpi -nobanner ' + \
            sys.executable + ' ' + \
            os.path.join(os.path.dirname(sys.modules[__name__].__file__),
                         'mpi_child.py')

        # Split the command into shell arguments for passing to Popen
        use_posix = True if sys.platform != 'win32' else False
        self.mpi_cmd = shlex.split(self.mpi_cmd, posix=use_posix)

    def __enter__(self):
        global _BACKEND

        self._old_backend = _BACKEND
        _BACKEND = self

        return self

    def __exit__(self, type, value, traceback):
        global _BACKEND

        _BACKEND = self._old_backend

        # always kill nrniv processes for good measure
        if self.n_procs > 1:
            kill_proc_name('nrniv')

    def simulate(self, net, tstop, dt, n_trials, postproc=False):
        """Simulate the HNN model in parallel on all cores

        Parameters
        ----------
        net : Network object
            The Network object specifying how cells are
            connected.
        tstop : float
            The simulation stop time (ms).
        dt : float
            The integration time step of h.CVode (ms)
        n_trials : int
            Number of trials to simulate.
        postproc : bool
            If False, no postprocessing applied to the dipole

        Returns
        -------
        dpl : list of Dipole
            The Dipole results from each simulation trial
        """

        # just use the joblib backend for a single core
        if self.n_procs == 1:
            print("MPIBackend is set to use 1 core: transferring the "
                  "simulation to JoblibBackend....")
            return JoblibBackend(n_jobs=1).simulate(net, tstop=tstop,
                                                    dt=dt,
                                                    n_trials=n_trials,
                                                    postproc=postproc)

        if self.n_procs > net._n_cells:
            raise ValueError(f'More MPI processes were assigned than there '
                             f'are cells in the network. Please decrease '
                             f'the number of parallel processes (got n_procs='
                             f'{self.n_procs}) over which you will '
                             f'distribute the {net._n_cells} network neurons.')

        print(f"MPI will run {n_trials} trial(s) sequentially by "
              f"distributing network neurons over {self.n_procs} processes.")

        env = _get_mpi_env()

        self.proc, sim_data = run_subprocess(
            command=self.mpi_cmd, obj=[net, tstop, dt, n_trials], timeout=30,
            proc_queue=self.proc_queue, env=env, cwd=os.getcwd(),
            universal_newlines=True)

        dpls = _gather_trial_data(sim_data, net, n_trials, postproc)
        return dpls

    def terminate(self):
        """Terminate running simulation on this MPIBackend

        Safe to call from another thread from the one `simulate_dipole`
        was called from.
        """
        proc = None
        try:
            proc = self.proc_queue.get(timeout=1)
        except Empty:
            warn("No currently running process to terminate")

        if proc is not None:
            proc.terminate()
            try:
                proc.wait(5)  # wait maximum of 5s
            except TimeoutExpired:
                warn("Could not kill python subprocess: PID %d" %
                     proc.pid)
