"""The energy site page renders from the committed artifacts without exceptions (streamlit AppTest)."""

from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "app" / "site_pages" / "energy.py"
ARTIFACT = Path(__file__).resolve().parents[1] / "eval" / "energy" / "summary.json"


@pytest.mark.skipif(not ARTIFACT.exists(), reason="run scripts/train_energy_models.py first")
def test_energy_page_renders():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PAGE), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value.startswith("Common-area energy")
    assert len(at.metric) >= 6
    labels = " ".join(m.label for m in at.metric)
    assert "Exit-route" in labels and "0 by design" in labels  # the 1 fc egress tile says it cannot fail
    assert "Stair minutes below" in labels  # the stair in-use shortfall is shown as its own tile
    assert "motion sensors alone" in labels  # the ML extension's cost sits next to the headline
    text = " ".join(m.value for m in at.markdown)
    assert "never cut" not in text  # the old over-claim is gone
    # flip the rule profile radio and pick another month: page must still render
    at.radio[0].set_value("corridor_rule").run()
    assert not at.exception
    sb = [s for s in at.selectbox if s.label == "Month detail"][0]
    sb.set_value("Jul").run()
    assert not at.exception
    # a named reviewer approves a proposal; status is kept in the session
    at.text_input[0].set_value("Test Reviewer").run()
    [b for b in at.button if b.label == "Approve"][0].click().run()
    assert not at.exception


def test_energy_page_has_no_set_page_config():
    assert "set_page_config" not in PAGE.read_text(encoding="utf-8")
