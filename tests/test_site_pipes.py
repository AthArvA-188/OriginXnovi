"""AppTest smoke check for the Pipes page (reads precomputed artifacts in eval/clog/)."""
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "app" / "site_pages" / "pipes.py"
EVAL = Path(__file__).resolve().parents[1] / "eval" / "clog"


@pytest.mark.skipif(not (EVAL / "riser_metrics.json").exists(), reason="clog artifacts not generated")
def test_pipes_page_runs_without_exception():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PAGE), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Pipe clogs" in t.value for t in at.title)
    text = " ".join(m.value for m in at.markdown)
    for label in ("REAL", "SIMULATED", "SYNTHETIC", "INJECTED"):
        assert label in text
    assert "requires human approval" in " ".join(str(d.value.to_dict()) for d in at.dataframe)
