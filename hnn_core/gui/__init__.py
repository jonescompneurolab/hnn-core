from .gui import HNNGUI, launch

def _maybe_check_first_run():
    try:
        from ..first_run import check_first_run
        check_first_run()
    except ImportError as e:
        print("Warning: could not run first-run check:", e)

_maybe_check_first_run()