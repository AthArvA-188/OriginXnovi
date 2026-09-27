"""AppTest smoke check for the interior walls page (runs offline from committed artifacts)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "app" / "site_pages" / "interior_walls.py"


def test_interior_page_renders_without_exceptions():
    if not (ROOT / "eval" / "interior" / "ndt_strength_v1.json").exists():
        pytest.skip("interior artifacts not built")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PAGE), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Interior wall testing" in t.value for t in at.title)
    text = " ".join(m.value for m in at.markdown)
    assert "What this means" in text and "REAL" in text


def test_interior_form_gives_estimate_with_interval_and_core_warning():
    if not (ROOT / "models" / "interior" / "ndt_strength_v1.json").exists():
        pytest.skip("interior model not built")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PAGE), default_timeout=120).run()
    at.number_input(key="iw_rn").set_value(35.0)
    at.number_input(key="iw_vp").set_value(4000.0)
    at.number_input(key="iw_design").set_value(140.0)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Estimated strength" in labels
    warn = " ".join(w.value for w in at.warning)
    assert "Calibrate with cores before any decision" in warn
    assert any("below the design strength" in e.value for e in at.error)


def test_single_instrument_estimate_shows_its_own_fallback_error():
    if not (ROOT / "eval" / "interior" / "ndt_strength_v1.json").exists():
        pytest.skip("interior artifacts not built")
    import json

    from streamlit.testing.v1 import AppTest

    fb = json.loads((ROOT / "eval" / "interior" / "ndt_strength_v1.json").read_text(encoding="utf-8"))["fallbacks"]["rn_only_on_rebound_db"]
    at = AppTest.from_file(str(PAGE), default_timeout=120).run()
    at.checkbox(key="iw_use_vp").uncheck()
    at.number_input(key="iw_rn").set_value(35.0)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    info = " ".join(i.value for i in at.info)
    assert "rebound-only fallback" in info
    assert f"MAE {fb['loso_powerlaw']['mae']:.1f} MPa" in info
    assert f"{fb['loso_powerlaw_in_situ']['mae']:.1f} MPa" in info


def test_moisture_promise_does_not_overreach():
    if not (ROOT / "eval" / "interior" / "ndt_strength_v1.json").exists():
        pytest.skip("interior artifacts not built")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PAGE), default_timeout=120).run()
    text = " ".join(m.value for m in at.markdown)
    assert "other meter readings show U" in text
    assert "Moisture readings are graded by EPA-based rules" not in text
    k3 = next(m for m in at.metric if m.label == "With 3 same-study specimens")
    assert "random draws each" in (getattr(k3.proto, "help", "") or "")
