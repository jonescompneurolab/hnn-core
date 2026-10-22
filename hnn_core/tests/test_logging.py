"""Tests for the centralized logging configuration."""

import contextlib
import importlib
import io
import logging
import subprocess
import sys
from pathlib import Path

import pytest

import hnn_core
from hnn_core._logging import _OWNED_ATTR, configure_logging, set_log_level
from hnn_core.parallel_backends import requires_mpi4py, requires_psutil

pkg_logger = logging.getLogger("hnn_core")

# NEURON itself can print a banner on stderr when it is imported on a machine
# without a display (e.g. CI runners). It is unrelated to hnn_core's logging.
_NEURON_STDERR_NOISE = (
    "Warning: no DISPLAY environment variable.",
    "--No graphics will be displayed.",
)


def _hnn_stderr(stderr):
    """Return ``stderr`` without NEURON's own startup banner lines."""
    lines = [ln for ln in stderr.splitlines() if ln.strip() not in _NEURON_STDERR_NOISE]
    return "\n".join(lines)


@pytest.fixture(autouse=True)
def restore_logger():
    """Restore the ``hnn_core`` logger state after each test."""
    handlers = list(pkg_logger.handlers)
    level, propagate = pkg_logger.level, pkg_logger.propagate
    yield
    for handler in list(pkg_logger.handlers):
        if handler not in handlers:
            pkg_logger.removeHandler(handler)
            handler.close()
    pkg_logger.handlers[:] = handlers
    pkg_logger.setLevel(level)
    pkg_logger.propagate = propagate


def _owned():
    return [h for h in pkg_logger.handlers if getattr(h, _OWNED_ATTR, False)]


def test_import_shows_progress_by_default():
    """Importing hnn_core shows INFO messages on stdout and leaves the root alone."""
    code = (
        "import logging, hnn_core;"
        "l = logging.getLogger('hnn_core');"
        "print(len(logging.getLogger().handlers),"
        " [type(h).__name__ for h in l.handlers], l.level);"
        "logging.getLogger('hnn_core.x').info('progress');"
        "logging.getLogger('hnn_core.x').debug('hidden')"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert _hnn_stderr(res.stderr) == ""
    assert res.stdout.splitlines() == [
        "0 ['NullHandler', '_StdoutHandler'] 20",
        "progress",
    ]


def test_default_handler_follows_stdout_redirection(capsys):
    """Like ``print``, the default handler writes to the *current* ``sys.stdout``."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        logging.getLogger("hnn_core.x").info("redirected")
    assert buf.getvalue() == "redirected\n"
    assert capsys.readouterr().out == ""


def test_stream_false_silences_default_output(capsys):
    configure_logging(stream=False)
    logging.getLogger("hnn_core.x").info("silent")
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_public_api():
    assert hnn_core.configure_logging is configure_logging


def test_levels_and_stream(capsys):
    configure_logging(level="WARNING")
    child = logging.getLogger("hnn_core.some_module")
    child.info("hidden info")
    child.warning("shown warning")
    child.error("shown error")
    out = capsys.readouterr().out
    assert "hidden info" not in out
    assert "shown warning" in out and "shown error" in out

    configure_logging(level=logging.DEBUG)
    child.debug("shown debug")
    assert "shown debug" in capsys.readouterr().out


def test_invalid_level():
    with pytest.raises(ValueError, match="Unknown logging level"):
        configure_logging(level="LOUD")
    with pytest.raises(TypeError, match="level must be"):
        configure_logging(level=1.5)


def test_idempotent_and_host_handlers_kept(tmp_path):
    host = logging.NullHandler()
    pkg_logger.addHandler(host)
    for _ in range(3):
        configure_logging(log_file=tmp_path / "a.log")
    assert len(_owned()) == 2  # stream + file, no accumulation
    assert host in pkg_logger.handlers
    # NullHandler installed at import is also still there
    assert any(type(h) is logging.NullHandler for h in pkg_logger.handlers)


def test_no_duplicate_with_root_handler(caplog, capsys):
    root_stream = logging.StreamHandler(sys.stderr)
    logging.getLogger().addHandler(root_stream)
    try:
        configure_logging()
        logging.getLogger("hnn_core.x").info("only-once")
        captured = capsys.readouterr()
        assert captured.out.count("only-once") == 1
        assert "only-once" not in captured.err

        configure_logging(propagate=True, stream=False)
        logging.getLogger("hnn_core.x").info("via-root")
        captured = capsys.readouterr()
        assert captured.err.count("via-root") == 1
        assert "via-root" not in captured.out
    finally:
        logging.getLogger().removeHandler(root_stream)


def test_file_logging(tmp_path):
    log_file = tmp_path / "hnn.log"
    configure_logging(level="INFO", log_file=log_file, stream=False)
    logging.getLogger("hnn_core.mod").info("hello file")
    logging.getLogger("hnn_core.mod").debug("not written")
    for handler in _owned():
        handler.flush()
    text = log_file.read_text()
    assert "hello file" in text and "[INFO]" in text and "hnn_core.mod" in text
    assert "not written" not in text


def test_file_logging_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("HNN_CORE_LOG_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    configure_logging()
    logging.getLogger("hnn_core").info("x")
    assert not any(isinstance(h, logging.FileHandler) for h in _owned())
    assert list(tmp_path.iterdir()) == []


def test_file_logging_env_var(tmp_path, monkeypatch):
    log_file = tmp_path / "env.log"
    monkeypatch.setenv("HNN_CORE_LOG_FILE", str(log_file))
    configure_logging(stream=False)
    logging.getLogger("hnn_core").info("from env")
    for handler in _owned():
        handler.flush()
    assert "from env" in log_file.read_text()


def test_reconfigure_drops_file_handler(tmp_path):
    configure_logging(log_file=tmp_path / "a.log")
    configure_logging()
    assert not any(isinstance(h, logging.FileHandler) for h in _owned())


def test_failed_configure_keeps_previous(tmp_path):
    configure_logging()
    before = _owned()
    with pytest.raises(OSError):
        configure_logging(log_file=tmp_path / "missing_dir" / "a.log")
    assert _owned() == before


def test_set_log_level():
    before = _owned()
    set_log_level("ERROR")
    assert pkg_logger.level == logging.ERROR
    assert _owned() == before


def test_stream_accepts_file_like(capsys):
    buf = io.StringIO()
    configure_logging(stream=buf)
    logging.getLogger("hnn_core.x").info("to buffer")
    captured = capsys.readouterr()
    assert buf.getvalue() == "to buffer\n"
    assert captured.out == "" and captured.err == ""


def test_child_stdout_logging_isolated_and_idempotent():
    """Child helper writes INFO to stdout, once, without touching the root logger."""
    code = (
        "import logging, sys\n"
        "from hnn_core._logging import _enable_child_stdout_logging\n"
        "_enable_child_stdout_logging(); _enable_child_stdout_logging()\n"
        "logging.getLogger('hnn_core.network_builder').info('child progress')\n"
        "logging.getLogger('hnn_core.network_builder').debug('hidden')\n"
        "print(len(logging.getLogger().handlers), file=sys.stderr)\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert res.stdout == "child progress\n"
    assert _hnn_stderr(res.stderr).strip() == "0"


# ---------------------------------------------------------------------------
# Simulation-dependent behavior (requires NEURON)
# ---------------------------------------------------------------------------
_JOBLIB_SCRIPT = """
import sys
from hnn_core import neymotin_2020_model, simulate_dipole, JoblibBackend
if {silence}:
    from hnn_core import configure_logging
    configure_logging(level="WARNING")
net = neymotin_2020_model(mesh_shape=(3, 3), add_drives_from_params=True)
with JoblibBackend(n_jobs=2):
    simulate_dipole(net, tstop=10, n_trials=2, dt=0.5, verbose=True)
"""


@pytest.mark.parametrize("silence", [False, True])
def test_joblib_workers_progress_follows_parent_logging(silence):
    """Worker-process progress appears (on stdout) unless the parent silenced INFO."""
    res = subprocess.run(
        [sys.executable, "-c", _JOBLIB_SCRIPT.format(silence=silence)],
        capture_output=True,
        text=True,
        check=True,
    )
    if silence:
        assert "Trial" not in res.stdout + res.stderr
        assert "Building" not in res.stdout + res.stderr
    else:
        assert "Trial 1" in res.stdout and "Trial 2" in res.stdout
        assert "Building the NEURON model" in res.stdout


def test_joblib_worker_helper_ignored_in_parent_process():
    """In the parent process (n_jobs=1 / threads) no extra handler is installed."""
    from hnn_core import neymotin_2020_model
    from hnn_core.parallel_backends import _simulate_single_trial_in_worker

    net = neymotin_2020_model(mesh_shape=(3, 3), add_drives_from_params=True)
    net._instantiate_drives(tstop=5, n_trials=1)
    n_handlers = len(pkg_logger.handlers)
    propagate = pkg_logger.propagate
    _simulate_single_trial_in_worker(net, 5, 0.5, 0, True)
    assert len(pkg_logger.handlers) == n_handlers
    assert pkg_logger.propagate == propagate


def test_joblib_worker_helper_silent_when_parent_logging_is_off(
    monkeypatch, capsys
):
    """A worker process stays silent if INFO messages are disabled in the parent."""
    import multiprocessing

    from hnn_core import neymotin_2020_model
    from hnn_core.parallel_backends import _simulate_single_trial_in_worker

    monkeypatch.setattr(multiprocessing, "parent_process", lambda: object())
    net = neymotin_2020_model(mesh_shape=(3, 3), add_drives_from_params=True)
    net._instantiate_drives(tstop=5, n_trials=1)
    _simulate_single_trial_in_worker(net, 5, 0.5, 0, False)
    assert "Building the NEURON model" not in capsys.readouterr().out
    assert not _owned()


def test_maximize_psd_trials_warning():
    from hnn_core import neymotin_2020_model
    from hnn_core.optimization import Optimizer

    net = neymotin_2020_model(mesh_shape=(3, 3), add_drives_from_params=True)
    optim = Optimizer(
        net,
        tstop=10,
        constraints={"evprox1_gauss_weight_L2_basket_ampa": (1e-5, 1e-3)},
        set_params=lambda net, params: None,
        solver="bayesian",
        obj_fun="maximize_psd",
        max_iter=11,
    )
    seen = {}

    def fake_run_opt(*args):
        seen["n_trials"] = args[-1]["n_trials"]
        return [1e-4], [0.0], net

    optim._run_opt = fake_run_opt  # skip the actual optimization
    with pytest.warns(UserWarning, match="only 1 trial is supported"):
        optim.fit(f_bands=[(8, 12)], relative_bandpower=[1], n_trials=3)
    assert seen["n_trials"] == 1  # behavior preserved: overwritten to 1


def test_optimize_evoked_maxiter_warning(monkeypatch):
    from hnn_core import neymotin_2020_model, simulate_dipole

    params = hnn_core.read_params(
        Path(hnn_core.__file__).parent / "param" / "default.json"
    )
    params.update(
        {
            "t_evprox_1": 6.0,
            "sigma_t_evprox_1": 2.0,
            "t_evdist_1": 8.0,
            "sigma_t_evdist_1": 2.0,
        }
    )
    net = neymotin_2020_model(params, add_drives_from_params=True, mesh_shape=(3, 3))
    del net.external_drives["evprox2"]
    dpl = simulate_dipole(net, tstop=10.0, n_trials=1)[0]
    oe_module = importlib.import_module("hnn_core.optimization.optimize_evoked")

    class _Stop(Exception):
        pass

    def stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(oe_module, "_run_optimization", stop)  # after the warning
    with pytest.warns(UserWarning, match="'maxiter' must be at least 12"):
        with pytest.raises(_Stop):
            oe_module.optimize_evoked(
                net,
                tstop=10.0,
                n_trials=1,
                target_dpl=dpl,
                initial_dpl=dpl,
                maxiter=5,
                which_drives=["evprox1"],
            )


@requires_mpi4py
@requires_psutil
@pytest.mark.uses_mpi
def test_mpi_child_silent_without_progress_when_not_verbose(capsys):
    from hnn_core import MPIBackend, neymotin_2020_model, simulate_dipole

    net = neymotin_2020_model(mesh_shape=(3, 3), add_drives_from_params=True)
    with MPIBackend(n_procs=2):
        simulate_dipole(net, tstop=10, dt=0.5, n_trials=1, verbose=False)
    assert "Trial" not in capsys.readouterr().out


def test_child_stdout_logging_in_process(capsys):
    """In-process twin of the subprocess test (so that coverage measures it)."""
    from hnn_core._logging import _CHILD_ATTR, _enable_child_stdout_logging

    _enable_child_stdout_logging()
    _enable_child_stdout_logging()  # idempotent
    logging.getLogger("hnn_core.network_builder").info("child progress")
    logging.getLogger("hnn_core.network_builder").debug("hidden")
    assert capsys.readouterr().out == "child progress\n"
    assert sum(getattr(h, _CHILD_ATTR, False) for h in pkg_logger.handlers) == 1
    assert pkg_logger.propagate is False


def test_joblib_worker_helper_enables_logging_in_worker_process(monkeypatch, capsys):
    """In a worker *process* the helper sends INFO progress to the worker's stdout."""
    import multiprocessing

    from hnn_core import neymotin_2020_model
    from hnn_core._logging import _CHILD_ATTR
    from hnn_core.parallel_backends import _simulate_single_trial_in_worker

    monkeypatch.setattr(multiprocessing, "parent_process", lambda: object())
    net = neymotin_2020_model(mesh_shape=(3, 3), add_drives_from_params=True)
    net._instantiate_drives(tstop=5, n_trials=1)
    _simulate_single_trial_in_worker(net, 5, 0.5, 0, True)
    assert "Building the NEURON model" in capsys.readouterr().out
    assert sum(getattr(h, _CHILD_ATTR, False) for h in pkg_logger.handlers) == 1


@requires_mpi4py
@requires_psutil
@pytest.mark.uses_mpi
def test_mpi_backend_verbose_logs_parent_diagnostics(caplog):
    """The parent-side diagnostics of ``MPIBackend(verbose=True)`` are logged."""
    from hnn_core import MPIBackend, neymotin_2020_model, simulate_dipole

    net = neymotin_2020_model(mesh_shape=(3, 3), add_drives_from_params=True)
    with caplog.at_level(logging.INFO, logger="hnn_core"):
        with MPIBackend(n_procs=2, verbose=True):
            simulate_dipole(net, tstop=10, dt=0.5, n_trials=1, verbose=False)
    assert "Network size" in caplog.text
    assert "process terminated after" in caplog.text


def test_optimize_evoked_skips_step_when_maxiter_is_zero(caplog):
    from hnn_core import neymotin_2020_model, simulate_dipole
    from hnn_core.optimization.optimize_evoked import optimize_evoked

    params = hnn_core.read_params(
        Path(hnn_core.__file__).parent / "param" / "default.json"
    )
    params.update(
        {
            "t_evprox_1": 6.0,
            "sigma_t_evprox_1": 2.0,
            "t_evdist_1": 8.0,
            "sigma_t_evdist_1": 2.0,
        }
    )
    net = neymotin_2020_model(params, add_drives_from_params=True, mesh_shape=(3, 3))
    del net.external_drives["evprox2"]
    dpl = simulate_dipole(net, tstop=10.0, n_trials=1)[0]
    with caplog.at_level(logging.INFO, logger="hnn_core"):
        optimize_evoked(
            net,
            tstop=10.0,
            n_trials=1,
            target_dpl=dpl,
            initial_dpl=dpl,
            maxiter=0,
            which_drives=["evprox1"],
        )
    assert "Skipping optimization step" in caplog.text
    assert "Initial RMSE" in caplog.text and "Final RMSE" in caplog.text


_BATCH_SCRIPT = """
import sys, tempfile
from hnn_core import neymotin_2020_model
from hnn_core.batch_simulate import BatchSimulate
if {silence}:
    from hnn_core import configure_logging
    configure_logging(level="WARNING")

def set_params(net, p):
    net.add_evoked_drive(
        "evprox", mu=p["mu"], sigma=2.0, numspikes=1, location="proximal",
        weights_ampa={{"L2_basket": 1e-3, "L2_pyramidal": 1e-3,
                      "L5_basket": 1e-3, "L5_pyramidal": 1e-3}},
        synaptic_delays={{"L2_basket": 0.1, "L2_pyramidal": 0.1,
                         "L5_basket": 1.0, "L5_pyramidal": 1.0}},
    )

net = neymotin_2020_model(mesh_shape=(3, 3))
batch = BatchSimulate(net=net, set_params=set_params, tstop=20,
                      save_folder=tempfile.mkdtemp(), batch_size=2, n_trials=1)
batch.run({{"mu": [10, 20]}}, n_jobs=2, backend="loky", verbose=True,
          return_output=False)
"""


@pytest.mark.parametrize("silence", [False, True])
def test_batch_simulate_workers_progress_follows_parent_logging(silence):
    """BatchSimulate worker progress appears (on stdout) unless INFO was silenced."""
    res = subprocess.run(
        [sys.executable, "-c", _BATCH_SCRIPT.format(silence=silence)],
        capture_output=True,
        text=True,
        check=True,
    )
    if silence:
        assert "Building the NEURON model" not in res.stdout + res.stderr
    else:
        assert "Building the NEURON model" in res.stdout


def test_batch_simulate_worker_enables_logging_in_worker_process(monkeypatch, capsys):
    """In a worker *process*, a batch simulation sends its progress to stdout."""
    import multiprocessing

    from hnn_core import neymotin_2020_model
    from hnn_core._logging import _CHILD_ATTR
    from hnn_core.batch_simulate import BatchSimulate

    def set_params(net, param_values):
        pass

    net = neymotin_2020_model(mesh_shape=(3, 3), add_drives_from_params=True)
    batch = BatchSimulate(net=net, set_params=set_params, tstop=5, n_trials=1)
    batch._worker_stdout_logging = True
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: object())
    batch._run_single_sim({})
    assert "Building the NEURON model" in capsys.readouterr().out
    assert sum(getattr(h, _CHILD_ATTR, False) for h in pkg_logger.handlers) == 1
