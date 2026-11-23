"""Browser tests for HNN GUI inputs.

Install the browser tools with `pip install pytest-playwright "playwright>=1.45"`
and `playwright install chromium`, then run `pytest hnn_core/tests/test_gui_browser.py`.
"""

import pytest

from hnn_core.gui import HNNGUI

pytest.importorskip("pytest_playwright")


def test_typing_while_scrolling_number_input(page):
    """Scroll the panel while keeping a focused number field editable."""
    page.clock.install(time="2025-01-01T00:00:00Z")
    gui_assets = HNNGUI().load_custom_gui_styling()
    page.set_content(
        f'<div class="hnn-gui" style="height: 120px; overflow: auto">{gui_assets}'
        '<input type="number" value="10"><div style="height: 500px"></div></div>'
    )
    field = page.get_by_role("spinbutton")
    field.click()
    page.keyboard.press("End")

    # Deliver the wheel and key in the same frame, regardless of machine speed.
    page.clock.pause_at("2025-01-01T00:01:00Z")
    field.evaluate(
        "el => el.addEventListener('wheel', () => window.wheelReceived = true)"
    )
    page.mouse.wheel(0, 60)
    page.wait_for_function("window.wheelReceived", timeout=5000)
    assert field.input_value() == "10"
    page.keyboard.press("5")
    page.keyboard.insert_text("6")  # Text insertion without a keydown event.
    page.keyboard.press("ArrowUp")
    assert field.input_value() == "1057"

    page.clock.resume()
    page.wait_for_function("document.querySelector('.hnn-gui').scrollTop > 0")
