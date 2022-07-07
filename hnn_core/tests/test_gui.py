# Authors: Huzi Cheng <hzcheng15@icloud.com>
from hnn_core.gui import run_hnn_gui
import os


def test_run_gui():
    """Test if main gui function gives proper ipywidget"""
    os.environ["DEBUG_HNNGUI"] = "0"
    gui = run_hnn_gui()
    assert gui is not None
