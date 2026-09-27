"""The local site entry (app/site.py) and its home page render without exceptions."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"


def test_site_entry_renders_home():
    at = AppTest.from_file(str(APP / "site.py"), default_timeout=180).run()
    assert not at.exception
    assert any("Pavilion Cerebro" in t.value for t in at.title)


def test_home_page_standalone():
    at = AppTest.from_file(str(APP / "site_pages" / "home.py"), default_timeout=180).run()
    assert not at.exception
    # Every result line carries a data label badge.
    lines = [m.value for m in at.markdown if "-badge[" in m.value]
    assert lines, "home page shows no labelled results"
