import logging

# The GUI logger is a child of the ``hnn_core`` package logger, so that GUI and
# library messages share one configuration (see ``hnn_core._logging``). Kept as
# ``hnn_core.gui._logging.logger`` for backward compatibility.
_logger_name = "hnn_core.gui"
logger = logging.getLogger(_logger_name)

# Backward compatibility: the GUI logger used to be called "hnn_gui". Code that
# emits messages through ``logging.getLogger("hnn_gui")`` keeps working: they are
# re-dispatched to the logger above, so they reach the GUI log window and any
# ``hnn_core`` handler exactly once. The legacy logger does not propagate to the
# root logger, which prevents duplicates. Handlers attached directly to
# "hnn_gui" only see messages emitted on that logger, not those of the GUI itself.
_legacy_logger_name = "hnn_gui"


class _ForwardToGUILogger(logging.Handler):
    def emit(self, record):
        logger.handle(record)


_legacy_logger = logging.getLogger(_legacy_logger_name)
_legacy_logger.setLevel(logging.INFO)
_legacy_logger.propagate = False
if not any(isinstance(h, _ForwardToGUILogger) for h in _legacy_logger.handlers):
    _legacy_logger.addHandler(_ForwardToGUILogger())
