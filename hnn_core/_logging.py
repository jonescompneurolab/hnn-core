"""Centralized logging configuration for HNN-Core.

Library modules obtain loggers with ``logging.getLogger(__name__)`` (all of them
live under the ``hnn_core`` namespace). Progress messages are shown by default, as
they were when they were written with ``print``: importing HNN-Core attaches a
handler to the ``hnn_core`` logger that writes INFO messages to ``sys.stdout``.
If ``HNN_CORE_LOG_FILE`` is set at import time, the same messages are also
appended to that file. Users change this with :func:`configure_logging`, and the
``verbose`` options of the simulation functions decide which messages are emitted
at all.
"""

import logging
import os
import sys
import warnings

_LOGGER_NAME = "hnn_core"
LOG_FILE_ENV_VAR = "HNN_CORE_LOG_FILE"

_STREAM_FORMAT = "%(message)s"
_FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Marks handlers installed by ``configure_logging`` so that repeated calls
# only ever replace our own handlers, never ones owned by the host application.
_OWNED_ATTR = "_hnn_core_owned"
_CHILD_ATTR = "_hnn_core_child_stdout"

# MPI children cannot inherit the parent's logger. ``MPIBackend.simulate``
# appends this flag to the ``nrniv`` command when INFO is disabled in the
# parent, and ``mpi_child`` reads it to stay silent the same way Joblib workers
# do.
_CHILD_STDOUT_SILENCE_FLAG = "--silence-hnn-logging"

logger = logging.getLogger(_LOGGER_NAME)
logger.addHandler(logging.NullHandler())


class _StdoutHandler(logging.StreamHandler):
    """Handler writing to whatever ``sys.stdout`` is at the time of each message.

    Looking ``sys.stdout`` up when emitting (like ``print`` does), instead of
    keeping the stream that existed at import time, means that output redirection
    (e.g. by the GUI or by ``contextlib.redirect_stdout``) keeps working.
    """

    def __init__(self, level=logging.NOTSET):
        super().__init__(stream=None)
        self.setLevel(level)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        # ``StreamHandler.__init__`` assigns ``self.stream``; we always follow
        # the current ``sys.stdout`` instead of a snapshot taken at init.
        return


def _resolve_level(level):
    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        if not isinstance(resolved, int):
            raise ValueError(f"Unknown logging level: {level!r}")
        return resolved
    if isinstance(level, int) and not isinstance(level, bool):
        return level
    raise TypeError(f"level must be an int or str, got {type(level).__name__}")


def _remove_owned_handlers():
    for handler in list(logger.handlers):
        if getattr(handler, _OWNED_ATTR, False):
            logger.removeHandler(handler)
            handler.close()


def _mark_owned(handler):
    setattr(handler, _OWNED_ATTR, True)
    return handler


def _env_log_file():
    return os.environ.get(LOG_FILE_ENV_VAR) or None


def _owned_file_handler(log_file):
    handler = logging.FileHandler(os.fspath(log_file), encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    return _mark_owned(handler)


def configure_logging(level="INFO", log_file=None, stream=True, propagate=False):
    """Configure HNN-Core logging output.

    By default (when this function is never called), INFO messages are written to
    ``sys.stdout``. If ``HNN_CORE_LOG_FILE`` is set when HNN-Core is imported, they
    are also appended to that file without calling this function. Calling this
    function replaces the handlers installed by HNN-Core itself, i.e. the default
    handler or those of a previous call; handlers added to the ``hnn_core`` logger
    by other code are left untouched.

    Parameters
    ----------
    level : int | str, default="INFO"
        Logging level for the ``hnn_core`` logger, e.g. ``"DEBUG"``, ``"INFO"``,
        ``"WARNING"``, ``"ERROR"`` or ``"CRITICAL"``.
    log_file : str | path-like | None
        If given, messages are also appended to this file together with
        timestamps, levels and logger names. If None, the path in the
        ``HNN_CORE_LOG_FILE`` environment variable is used when it is set
        (that variable is also read at import). File logging is otherwise
        disabled.
    stream : bool | file-like, default=True
        Where to write messages (without decoration). True writes to
        ``sys.stdout`` (as by default), False disables terminal output, and a
        file-like object (e.g. ``sys.stderr``) writes there. ``stream=False``
        also keeps ``JoblibBackend``, ``BatchSimulate`` and MPI worker
        processes quiet: they inherit this process' stdout, so they only write
        progress there when this process would itself display INFO on a stream.
    propagate : bool, default=False
        Whether records should also reach the root logger's handlers. This is
        False by default so that messages are not printed twice when the host
        application has configured the root logger (e.g. ``logging.basicConfig``).

    Returns
    -------
    logger : logging.Logger
        The ``hnn_core`` logger.
    """
    level = _resolve_level(level)
    if log_file is None:
        log_file = _env_log_file()

    # Build new handlers before removing the old ones so that a failure
    # (e.g. unwritable log file) leaves the existing configuration intact.
    new_handlers = []
    if stream:
        stream_handler = (
            _StdoutHandler() if stream is True else logging.StreamHandler(stream)
        )
        stream_handler.setFormatter(logging.Formatter(_STREAM_FORMAT))
        new_handlers.append(_mark_owned(stream_handler))
    if log_file is not None:
        new_handlers.append(_owned_file_handler(log_file))

    _remove_owned_handlers()
    for handler in new_handlers:
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = propagate
    return logger


def set_log_level(level):
    """Set the level of the ``hnn_core`` logger without touching handlers."""
    logger.setLevel(_resolve_level(level))


def _enable_child_stdout_logging():
    """Send messages of this (sub)process to its stdout at INFO level.

    Worker and MPI child processes do not inherit the logging configuration of
    the parent. Their stdout is forwarded to the user by the parent process
    (MPI) or inherited from it (joblib workers), so progress messages are
    written there. This replaces the default handler (which a freshly started
    process also has), so that messages are not written twice. Calling this more
    than once has no further effect.
    """
    if any(getattr(h, _CHILD_ATTR, False) for h in logger.handlers):
        return
    _remove_owned_handlers()
    handler = _StdoutHandler()
    handler.setFormatter(logging.Formatter(_STREAM_FORMAT))
    setattr(handler, _CHILD_ATTR, True)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _remove_child_handlers():
    for handler in list(logger.handlers):
        if getattr(handler, _CHILD_ATTR, False):
            logger.removeHandler(handler)
            handler.close()


def _configure_child_logging(enabled):
    """Make a worker or MPI child follow the logging choice of its parent process.

    Parameters
    ----------
    enabled : bool
        Whether INFO messages are enabled in the parent. If True, the process writes
        its progress to its own stdout; if False, handlers installed by HNN-Core
        (the default import handler or a previous child handler) are removed so
        that it stays silent.
    """
    if enabled:
        _enable_child_stdout_logging()
    else:
        _remove_owned_handlers()
        _remove_child_handlers()


def _apply_parent_logging_from_argv(argv):
    """Configure this MPI child from the flag sent by ``MPIBackend.simulate``."""
    _configure_child_logging(_CHILD_STDOUT_SILENCE_FLAG not in argv)


def _parent_displays_info():
    """Whether this process would write INFO messages to a stream (not a file).

    Joblib, BatchSimulate and MPI workers inherit stdout from the parent. They
    should stay silent not only when the logger level is above INFO, but also
    when the parent silenced the terminal with ``configure_logging(stream=False)``.
    File handlers do not count: workers cannot write to the parent's log file.
    """
    if not logger.isEnabledFor(logging.INFO):
        return False
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            continue
        if not isinstance(handler, logging.StreamHandler):
            continue
        if handler.level == logging.NOTSET or handler.level <= logging.INFO:
            return True
    return False


def _install_default_handler():
    """Show INFO on stdout, and to ``HNN_CORE_LOG_FILE`` when that is set."""
    handler = _StdoutHandler()
    handler.setFormatter(logging.Formatter(_STREAM_FORMAT))
    logger.addHandler(_mark_owned(handler))
    log_file = _env_log_file()
    if log_file is not None:
        try:
            logger.addHandler(_owned_file_handler(log_file))
        except OSError as err:
            # Do not fail ``import hnn_core`` if the path is not writable.
            warnings.warn(
                f"Could not open {LOG_FILE_ENV_VAR}={log_file!r} for logging: {err}",
                RuntimeWarning,
                stacklevel=2,
            )
    logger.setLevel(logging.INFO)
    logger.propagate = False


_install_default_handler()
