"""Tests for the centralized logging configuration."""

import importlib
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


def test_import_is_silent():
    """Importing hnn_core must not configure logging or write output."""
    code = (
        "import logging, hnn_core;"
        "l = logging.getLogger('hnn_core');"
        "print(len(logging.getLogger().handlers),"
        " [type(h).__name__ for h in l.handlers], l.level)"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert _hnn_stderr(res.stderr) == ""
    assert res.stdout.strip() == "0 ['NullHandler'] 0"


def test_public_api():
    assert hnn_core.configure_logging is configure_logging


def test_levels_and_stream(capsys):
    configure_logging(level="WARNING")
    child = logging.getLogger("hnn_core.some_module")
    child.info("hidden info")
    child.warning("shown warning")
    child.error("shown error")
    err = capsys.readouterr().err
    assert "hidden info" not in err
    assert "shown warning" in err and "shown error" in err

    configure_logging(level=logging.DEBUG)
    child.debug("shown debug")
    assert "shown debug" in capsys.readouterr().err


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
    root_stream = logging.StreamHandler(sys.stdout)
    logging.getLogger().addHandler(root_stream)
    try:
        configure_logging()
        logging.getLogger("hnn_core.x").info("only-once")
        captured = capsys.readouterr()
        assert captured.err.count("only-once") == 1
        assert "only-once" not in captured.out

        configure_logging(propagate=True, stream=False)
        logging.getLogger("hnn_core.x").info("via-root")
        captured = capsys.readouterr()
        assert captured.out.count("via-root") == 1
        assert "via-root" not in captured.err
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


def test_set_log_level(caplog):
    set_log_level("ERROR")
    assert pkg_logger.level == logging.ERROR
    assert not _owned()


# ---------------------------------------------------------------------------
# GUI integration
# ---------------------------------------------------------------------------
def test_gui_import_does_not_replace_stdout():
    """Importing the GUI must not hijack ``sys.stdout``."""
    pytest.importorskip("ipywidgets")
    original = sys.stdout
    import hnn_core.gui.gui as gui_module  # noqa: F401

    assert sys.stdout is original
    assert not hasattr(gui_module, "_GUI_PrintToLogger")


def test_gui_logger_compat_alias():
    pytest.importorskip("ipywidgets")
    from hnn_core.gui._logging import logger as gui_logger

    assert gui_logger is logging.getLogger("hnn_core.gui")
    assert gui_logger.parent is pkg_logger


def test_gui_log_window_integration(capsys):
    """GUI window shows GUI and core messages, once, without duplicates."""
    pytest.importorskip("ipywidgets")
    from ipywidgets import Output
    from hnn_core.gui.gui import HNNGUI, _OutputWidgetHandler
    from hnn_core.gui._data_store import data_store
    from hnn_core.gui._logging import logger as gui_logger

    def widget_handlers():
        return [h for h in pkg_logger.handlers if isinstance(h, _OutputWidgetHandler)]

    # constructing the handler must not touch the data store
    data_store.simulated_data["sentinel"] = {}
    _OutputWidgetHandler(Output())
    assert "sentinel" in data_store.simulated_data

    gui = HNNGUI()
    # creating the GUI is the point where the data store is reset
    assert "sentinel" not in data_store.simulated_data
    assert len(widget_handlers()) == 1

    gui_logger.info("gui message")
    logging.getLogger("hnn_core.network_builder").info("core message")
    logging.getLogger("hnn_core.network_builder").debug("debug message")
    text = "\n".join(o["text"] for o in gui._log_out.outputs)
    assert text.count("gui message") == 1
    assert text.count("core message") == 1
    assert "debug message" not in text
    assert "[INFO]" in text

    # a second GUI replaces the first handler instead of adding another one
    gui2 = HNNGUI()
    assert len(widget_handlers()) == 1
    gui_logger.warning("second window")
    assert "second window" not in "\n".join(o["text"] for o in gui._log_out.outputs)
    texts2 = [o["text"] for o in gui2._log_out.outputs]
    assert sum("second window" in t for t in texts2) == 1

    # nothing leaks to stdout; no duplicates when configure_logging is also used
    configure_logging(level="INFO")
    gui_logger.info("with stream handler")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("with stream handler") == 1
    texts2 = [o["text"] for o in gui2._log_out.outputs]
    assert sum("with stream handler" in t for t in texts2) == 1
    assert len(widget_handlers()) == 1


def test_gui_shows_nonuniform_gain_warning(tmp_path):
    """The non-uniform gain ``UserWarning`` is mirrored in the GUI log window."""
    pytest.importorskip("ipywidgets")
    from hnn_core import neymotin_2020_model, write_network_configuration
    from hnn_core.gui.gui import HNNGUI

    net = neymotin_2020_model(mesh_shape=(3, 3))
    picks = hnn_core.pick_connection(
        net, src_gids="L2_pyramidal", target_gids="L2_pyramidal"
    )
    net.connectivity[picks[0]]["nc_dict"]["gain"] = 1.0
    net.connectivity[picks[1]]["nc_dict"]["gain"] = 2.0
    fname = tmp_path / "net.json"
    write_network_configuration(net, fname)

    gui = HNNGUI()
    gui.compose()
    gui._simulate_upload_connectivity(fname)
    text = "\n".join(o["text"] for o in gui._log_out.outputs)
    assert "custom synaptic gain values" in text
    assert "[WARNING]" in text


# ---------------------------------------------------------------------------
# Forwarded child-process output
# ---------------------------------------------------------------------------
def test_forward_child_output(capsys, caplog, tmp_path):
    """Child output goes to stdout once, and is logged without terminal duplicates."""
    from hnn_core._logging import FORWARDED_ATTR
    from hnn_core.parallel_backends import _forward_child_output

    log_file = tmp_path / "fwd.log"
    configure_logging(level="INFO", log_file=log_file, propagate=True)

    _forward_child_output("Trial 1: 0.5 ms...\n")
    _forward_child_output("")  # nothing to do
    _forward_child_output("\n")  # whitespace only: written, but not logged

    captured = capsys.readouterr()
    assert captured.out == "Trial 1: 0.5 ms...\n\n"  # terminal forwarding unchanged
    assert captured.err == ""  # no second copy through the stream handler

    records = [r for r in caplog.records if "Trial 1" in r.getMessage()]
    assert len(records) == 1
    assert getattr(records[0], FORWARDED_ATTR) is True
    assert records[0].levelno == logging.INFO
    for handler in _owned():
        handler.flush()
    assert log_file.read_text().count("Trial 1: 0.5 ms...") == 1


def test_echo_child_output_is_logged(capsys, caplog):
    from queue import Queue
    from hnn_core.parallel_backends import _echo_child_output

    q = Queue()
    q.put("Building the NEURON model\n")
    with caplog.at_level(logging.INFO, logger="hnn_core"):
        assert _echo_child_output(q) == "Building the NEURON model\n"
    assert capsys.readouterr().out == "Building the NEURON model\n"
    assert [r.getMessage() for r in caplog.records] == ["Building the NEURON model"]


def test_gui_window_receives_forwarded_output_once(capsys):
    pytest.importorskip("ipywidgets")
    from hnn_core.gui.gui import HNNGUI
    from hnn_core.parallel_backends import _forward_child_output

    gui = HNNGUI()
    configure_logging()  # also enable the terminal stream handler
    _forward_child_output("MPI child says hi\n")
    texts = [o["text"] for o in gui._log_out.outputs]
    assert sum("MPI child says hi" in t for t in texts) == 1
    captured = capsys.readouterr()
    assert captured.out.count("MPI child says hi") == 1
    assert captured.err == ""


def test_stream_accepts_file_like(capsys):
    configure_logging(stream=sys.stdout)
    logging.getLogger("hnn_core.x").info("to stdout")
    captured = capsys.readouterr()
    assert "to stdout" in captured.out and captured.err == ""


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
# Warnings in the GUI
# ---------------------------------------------------------------------------
def _gui_window_text(gui):
    return "\n".join(o["text"] for o in gui._log_out.outputs)


def test_warnings_to_log_window():
    """GUI path: shown in the log window and nowhere else; hook is restored."""
    pytest.importorskip("ipywidgets")
    import warnings
    from hnn_core.gui.gui import HNNGUI, _warnings_to_log_window

    gui = HNNGUI()
    original_hook = warnings.showwarning
    shown = []
    warnings.showwarning = lambda *args, **kwargs: shown.append(args)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            with _warnings_to_log_window():
                warnings.warn("gui-path warning", UserWarning)
            assert shown == []  # the default display hook was not used
            # outside the context manager the previous hook is active again
            warnings.warn("outside gui", UserWarning)
            assert len(shown) == 1
    finally:
        warnings.showwarning = original_hook
    text = _gui_window_text(gui)
    assert text.count("gui-path warning") == 1 and "[WARNING]" in text
    assert "UserWarning" in text
    assert "outside gui" not in text


def test_warnings_to_log_window_respects_filters():
    pytest.importorskip("ipywidgets")
    import warnings
    from hnn_core.gui.gui import HNNGUI, _warnings_to_log_window

    gui = HNNGUI()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with _warnings_to_log_window():
            warnings.warn("ignored warning", UserWarning)
    assert "ignored warning" not in _gui_window_text(gui)


# ---------------------------------------------------------------------------
# Simulation-dependent behavior (requires NEURON)
# ---------------------------------------------------------------------------
_JOBLIB_SCRIPT = """
import sys
from hnn_core import neymotin_2020_model, simulate_dipole, JoblibBackend
if {configure}:
    from hnn_core import configure_logging
    configure_logging()
net = neymotin_2020_model(mesh_shape=(3, 3), add_drives_from_params=True)
with JoblibBackend(n_jobs=2):
    simulate_dipole(net, tstop=10, n_trials=2, dt=0.5, verbose=True)
"""


@pytest.mark.parametrize("configure", [False, True])
def test_joblib_workers_progress_follows_parent_logging(configure):
    """Worker-process progress appears (on stdout) only if INFO is enabled."""
    res = subprocess.run(
        [sys.executable, "-c", _JOBLIB_SCRIPT.format(configure=configure)],
        capture_output=True,
        text=True,
        check=True,
    )
    if configure:
        assert "Trial 1" in res.stdout and "Trial 2" in res.stdout
        assert "Building the NEURON model" in res.stdout
    else:
        assert "Trial" not in res.stdout + res.stderr
        assert "Building" not in res.stdout + res.stderr


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


def test_gains_warning_nongui_and_gui(recwarn):
    """Network check warns (non-GUI); the GUI window shows it instead of stderr."""
    pytest.importorskip("ipywidgets")
    from hnn_core import neymotin_2020_model
    from hnn_core.network import _check_global_synaptic_gains_uniformity
    from hnn_core.gui.gui import HNNGUI, _warnings_to_log_window

    net = neymotin_2020_model(mesh_shape=(3, 3))
    picks = hnn_core.pick_connection(
        net, src_gids="L2_pyramidal", target_gids="L2_pyramidal"
    )
    net.connectivity[picks[1]]["nc_dict"]["gain"] = 2.0

    with pytest.warns(UserWarning, match="custom synaptic gain values") as record:
        assert _check_global_synaptic_gains_uniformity(net) is False
    assert len(record) == 1

    gui = HNNGUI()
    recwarn.clear()
    with _warnings_to_log_window():
        _check_global_synaptic_gains_uniformity(net)
    assert len(recwarn.list) == 0  # not emitted a second time
    assert "custom synaptic gain values" in _gui_window_text(gui)


def test_maximize_psd_trials_warning_nongui_and_gui(recwarn):
    pytest.importorskip("ipywidgets")
    from hnn_core import neymotin_2020_model
    from hnn_core.optimization import Optimizer
    from hnn_core.gui.gui import HNNGUI, _warnings_to_log_window

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
    fit_kwargs = dict(f_bands=[(8, 12)], relative_bandpower=[1], n_trials=3)

    with pytest.warns(UserWarning, match="only 1 trial is supported"):
        optim.fit(**fit_kwargs)
    assert seen["n_trials"] == 1  # behavior preserved: overwritten to 1

    gui = HNNGUI()
    recwarn.clear()
    with _warnings_to_log_window():
        optim.fit(**fit_kwargs)
    assert len(recwarn.list) == 0
    assert "only 1 trial is supported" in _gui_window_text(gui)


def test_optimize_evoked_maxiter_warning_nongui_and_gui(monkeypatch, recwarn):
    pytest.importorskip("ipywidgets")
    from hnn_core import neymotin_2020_model, simulate_dipole
    from hnn_core.gui.gui import HNNGUI, _warnings_to_log_window

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

    def run():
        oe_module.optimize_evoked(
            net,
            tstop=10.0,
            n_trials=1,
            target_dpl=dpl,
            initial_dpl=dpl,
            maxiter=5,
            which_drives=["evprox1"],
        )

    with pytest.warns(UserWarning, match="'maxiter' must be at least 12"):
        with pytest.raises(_Stop):
            run()

    gui = HNNGUI()
    recwarn.clear()
    with _warnings_to_log_window(), pytest.raises(_Stop):
        run()
    assert not [w for w in recwarn.list if "maxiter" in str(w.message)]
    assert "'maxiter' must be at least 12" in _gui_window_text(gui)


@requires_mpi4py
@requires_psutil
@pytest.mark.uses_mpi
def test_mpi_child_output_forwarded_to_stdout_and_logger(capsys, caplog, tmp_path):
    """MPI child progress: on stdout exactly once, and visible to logging/GUI."""
    from hnn_core import MPIBackend, neymotin_2020_model, simulate_dipole
    from hnn_core._logging import FORWARDED_ATTR

    pytest.importorskip("ipywidgets")
    from hnn_core.gui.gui import HNNGUI

    gui = HNNGUI()
    log_file = tmp_path / "mpi_parent.log"
    # stream handler on: forwarded text must not be duplicated on the terminal
    configure_logging(level="INFO", log_file=log_file, propagate=True)

    net = neymotin_2020_model(mesh_shape=(3, 3), add_drives_from_params=True)
    with MPIBackend(n_procs=2):
        simulate_dipole(net, tstop=10, dt=0.5, n_trials=1, verbose=True)

    # Rank 0 prints this exactly once. (The ``Trial 1: ... ms...`` progress line is
    # not used here: it is occasionally lost on its way back from the MPI child, also
    # without any logging changes, so asserting on it would make this test flaky.)
    msg = "Building the NEURON model"
    captured = capsys.readouterr()
    assert captured.out.count(msg) == 1  # forwarded child stdout
    assert msg not in captured.err  # no duplicate via the stream handler

    forwarded = [r for r in caplog.records if getattr(r, FORWARDED_ATTR, False)]
    assert any(msg in r.getMessage() for r in forwarded)
    assert _gui_window_text(gui).count(msg) == 1
    for handler in _owned():
        handler.flush()
    assert msg in log_file.read_text()


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
if {configure}:
    from hnn_core import configure_logging
    configure_logging()

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


@pytest.mark.parametrize("configure", [False, True])
def test_batch_simulate_workers_progress_follows_parent_logging(configure):
    """BatchSimulate worker progress appears (on stdout) only if INFO is enabled."""
    res = subprocess.run(
        [sys.executable, "-c", _BATCH_SCRIPT.format(configure=configure)],
        capture_output=True,
        text=True,
        check=True,
    )
    if configure:
        assert "Building the NEURON model" in res.stdout
    else:
        assert "Building the NEURON model" not in res.stdout + res.stderr


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


def test_legacy_hnn_gui_logger_forwards_once(capsys):
    """Messages emitted on the old ``hnn_gui`` logger reach the GUI window once."""
    pytest.importorskip("ipywidgets")
    from hnn_core.gui.gui import HNNGUI

    gui = HNNGUI()
    configure_logging()
    logging.getLogger("hnn_gui").info("legacy message")
    logging.getLogger("hnn_gui").debug("legacy debug")
    assert _gui_window_text(gui).count("legacy message") == 1
    assert "legacy debug" not in _gui_window_text(gui)
    assert capsys.readouterr().err.count("legacy message") == 1
    # the legacy logger must not also reach the root logger (no duplicates)
    assert logging.getLogger("hnn_gui").propagate is False
