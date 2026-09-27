"""AppTest smoke check for the exterior inspection page (runs offline from committed artifacts)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "app" / "site_pages" / "exterior_inspection.py"


def _artifacts_or_skip():
    if not (ROOT / "eval" / "facade" / "tilecls_v1.json").exists() or not (ROOT / "models" / "facade" / "tilecls_resnet18_v1.onnx").exists():
        pytest.skip("facade artifacts not built")


def test_exterior_page_renders_without_exceptions():
    _artifacts_or_skip()
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PAGE), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Exterior facade screening" in t.value for t in at.title)
    text = " ".join(m.value for m in at.markdown)
    assert "What this means" in text
    assert "REAL" in text
    assert "SB 721/326 are out of scope" in text


def test_grader_is_labelled_unmeasured_and_needs_confirmation():
    _artifacts_or_skip()
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PAGE), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    captions = " ".join(c.value for c in at.caption)
    assert "Grader accuracy on facade photos is unmeasured" in captions
    # never sends crops to the API before the person ticks the confirmation box (key or no key)
    assert at.button(key="fx_grade").disabled
    assert at.checkbox(key="fx_grade_ok").value is False
    assert "Provenance:" in captions


def test_side_by_side_review_list_can_list_all_nine_squares_of_a_sample_frame():
    _artifacts_or_skip()
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PAGE), default_timeout=120).run()
    at.slider(key="fx_k").set_value(12)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    captions = " ".join(c.value for c in at.caption)
    assert "9 distinct squares available in this photo; listing 9." in captions
