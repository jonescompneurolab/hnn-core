"""Tests for the centralized logging configuration."""

import contextlib
import importlib
import io
import logging
import os
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


def test_import_honors_log_file_env_var(tmp_path):
    """Setting HNN_CORE_LOG_FILE before import writes a log file with no extra call."""
    log_file = tmp_path / "import.log"
    env = os.environ.copy()
    env["HNN_CORE_LOG_FILE"] = str(log_file)
    code = (
        "import logging, hnn_core;"
        "logging.getLogger('hnn_core.x').info('from env at import')"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert "from env at import" in res.stdout.splitlines()
    assert "from env at import" in log_file.read_text()


def test_import_warns_on_unwritable_log_file_env_var(tmp_path):
    """A bad HNN_CORE_LOG_FILE path must not prevent importing hnn_core."""
    missing = tmp_path / "no_such_dir" / "hnn.log"
    env = os.environ.copy()
    env["HNN_CORE_LOG_FILE"] = str(missing)
    code = (
        "import logging, hnn_core;logging.getLogger('hnn_core.x').info('still works')"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert res.returncode == 0
    assert "still works" in res.stdout
    assert "HNN_CORE_LOG_FILE" in res.stderr
    assert not missing.exists()


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


# ---------------------------------------------------------------------------
# Simulation-dependent behavior (requires NEURON)
# ---------------------------------------------------------------------------
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
