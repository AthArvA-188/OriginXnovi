"""The Numeric AI site page renders from the committed artifacts with no exception (offline, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "app" / "site_pages" / "numeric_ai.py"
EVAL = ROOT / "eval" / "numeric"


@pytest.mark.skipif(not (EVAL / "backtest_bdg2.json").exists(), reason="numeric artifacts not generated")
def test_numeric_ai_page_runs():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PAGE), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title and "Numeric AI" in at.title[0].value and "honest" not in at.title[0].value
    text = " ".join(m.value for m in at.markdown)
    assert "REAL" in text and "SYNTHETIC" in text
    assert "Paired WAPE differences" in text and "1.4826 x MAD" in text
    assert "building layer's threshold" not in text  # K is on the same scale but the baseline differs
    assert len(at.dataframe) >= 4  # backtest, paired differences, anomaly and catalogue tables


@pytest.mark.skipif(not (ROOT / "models" / "numeric" / "hgb_bdg2.joblib").exists(), reason="HGB bundle missing")
def test_live_hgb_rerun_matches_stored_forecast():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PAGE), default_timeout=120).run()
    hgb = [o for o in at.radio[0].options if "HGB" in o][0]
    at = at.radio[0].set_value(hgb).run()
    at = at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    msg = at.success[0].value
    diff = float(msg.split("stored forecast: ")[1].split(" kWh")[0])
    assert diff < 1e-6


@pytest.mark.skipif(not (EVAL / "backtest_bdg2.json").exists(), reason="numeric artifacts not generated")
def test_numeric_ai_page_other_model_choices():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PAGE), default_timeout=120).run()
    radio = at.radio[0]
    for opt in radio.options:
        at = radio.set_value(opt).run()
        assert not at.exception, (opt, [e.value for e in at.exception])
        radio = at.radio[0]
