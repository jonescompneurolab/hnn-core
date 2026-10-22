"""Centralized logging configuration for HNN-Core.

Library modules obtain loggers with ``logging.getLogger(__name__)`` (all of them
live under the ``hnn_core`` namespace). Progress messages are shown by default, as
they were when they were written with ``print``: importing HNN-Core attaches a
handler to the ``hnn_core`` logger that writes INFO messages to ``sys.stdout``.
Users change this with :func:`configure_logging`, and the ``verbose`` options of
the simulation functions decide which messages are emitted at all.
"""

import logging
import os
import sys

_LOGGER_NAME = "hnn_core"
LOG_FILE_ENV_VAR = "HNN_CORE_LOG_FILE"

_STREAM_FORMAT = "%(message)s"
_FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Marks handlers installed by ``configure_logging`` so that repeated calls
# only ever replace our own handlers, never ones owned by the host application.
_OWNED_ATTR = "_hnn_core_owned"
_CHILD_ATTR = "_hnn_core_child_stdout"

logger = logging.getLogger(_LOGGER_NAME)
logger.addHandler(logging.NullHandler())


class _StdoutHandler(logging.StreamHandler):
    """Handler writing to whatever ``sys.stdout`` is at the time of each message.

    Looking ``sys.stdout`` up when emitting (like ``print`` does), instead of
    keeping the stream that existed at import time, means that output redirection
    (e.g. by the GUI or by ``contextlib.redirect_stdout``) keeps working.
    """

    def __init__(self, level=logging.NOTSET):
        logging.Handler.__init__(self, level)

    @property
    def stream(self):
        return sys.stdout


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


def configure_logging(level="INFO", log_file=None, stream=True, propagate=False):
    """Configure HNN-Core logging output.

    By default (when this function is never called), INFO messages are written to
    ``sys.stdout``. Calling this function replaces the handlers installed by
    HNN-Core itself, i.e. the default handler or those of a previous call; handlers
    added to the ``hnn_core`` logger by other code are left untouched.

    Parameters
    ----------
    level : int | str, default="INFO"
        Logging level for the ``hnn_core`` logger, e.g. ``"DEBUG"``, ``"INFO"``,
        ``"WARNING"``, ``"ERROR"`` or ``"CRITICAL"``.
    log_file : str | path-like | None
        If given, messages are also appended to this file together with
        timestamps, levels and logger names. If None, the path in the
        ``HNN_CORE_LOG_FILE`` environment variable is used when it is set.
        File logging is otherwise disabled.
    stream : bool | file-like, default=True
        Where to write messages (without decoration). True writes to
        ``sys.stdout`` (as by default), False disables terminal output, and a
        file-like object (e.g. ``sys.stderr``) writes there.
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
        log_file = os.environ.get(LOG_FILE_ENV_VAR) or None

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
        file_handler = logging.FileHandler(os.fspath(log_file), encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        new_handlers.append(_mark_owned(file_handler))

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


def _configure_child_logging(enabled):
    """Make a worker process follow the logging choice of its parent process.

    Parameters
    ----------
    enabled : bool
        Whether INFO messages are enabled in the parent. If True, the worker writes
        its progress to its own stdout; if False, the default handler that the
        worker created when importing HNN-Core is removed so that it stays silent.
    """
    if enabled:
        _enable_child_stdout_logging()
    else:
        _remove_owned_handlers()


def _install_default_handler():
    """Show INFO messages on stdout until the user configures logging."""
    handler = _StdoutHandler()
    handler.setFormatter(logging.Formatter(_STREAM_FORMAT))
    logger.addHandler(_mark_owned(handler))
    logger.setLevel(logging.INFO)
    logger.propagate = False


_install_default_handler()
