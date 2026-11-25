"""Render the whole Streamlit app headlessly and fail on any exception."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_renders_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 6
    assert len(at.metric) >= 6


def test_app_filter_to_single_task():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.multiselect[0].set_value(["text-generation"]).run()
    assert not at.exception, at.exception


def test_app_search_with_no_matches_shows_info():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.text_input[0].set_value("zzzz-no-such-model").run()
    assert not at.exception
    assert any("No models match" in i.value for i in at.info)
