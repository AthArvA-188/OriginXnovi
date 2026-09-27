"""AppTest smoke check for the LA rain-exposure page (reads the committed eval/rain artifacts; no network)."""

from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "app" / "site_pages" / "rain_la.py"
ART = Path(__file__).resolve().parents[1] / "eval" / "rain" / "station_roses.json"


@pytest.mark.skipif(not ART.exists(), reason="eval/rain artifacts not built")
def test_rain_la_page_runs_without_exception():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PAGE), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "rain" in at.title[0].value.lower()
    text = " ".join(m.value for m in at.markdown)
    assert "Sources & licences" in text
    # the 'check your building' control reacts to a different wall direction and station
    at.slider(key="rain_b_az").set_value(270).run()
    at.selectbox(key="rain_b_station").set_value("BUR").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("of 8" in str(m.value) for m in at.metric)
    # coordinate mode picks the nearest measured station (default coordinates are downtown LA)
    at.radio(key="rain_b_mode").set_value("Find it from my building's coordinates").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Nearest measured station" in c.value and "CQT" in c.value for c in at.caption)
