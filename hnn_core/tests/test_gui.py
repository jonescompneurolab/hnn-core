# Authors: Huzi Cheng <hzcheng15@icloud.com>
from hnn_core.gui.gui import HNNGUI


def test_run_gui():
    """Test if main gui function gives proper ipywidget"""
    gui = HNNGUI()
    app_layout = gui.run()
    assert app_layout is not None
