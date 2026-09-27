"""AppTest check for the Fire plan site page: it runs offline from the committed artifacts with no exception, and its
controls re-plan for another floor and an escalation alarm."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = Path(__file__).resolve().parents[1] / "app" / "site_pages" / "fire_plan.py"


def _all_text(at) -> str:
    parts = [m.value for m in at.markdown] + [e.value for e in at.error] + [i.value for i in at.info] + [c.value for c in at.caption]
    return "\n".join(str(p) for p in parts)


def test_fire_plan_page_runs():
    at = AppTest.from_file(str(PAGE), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    text = _all_text(at)
    assert "never controls alarms, sprinklers, smoke control or elevators" in text
    assert "What this means" in text
    assert "SYNTHETIC" in text and "REAL" in text
    assert "Sources" in "\n".join(s.value for s in at.subheader)
    assert any("F18-F20" == m.value for m in at.metric)


def test_fire_plan_page_replans_and_escalates():
    at = AppTest.from_file(str(PAGE), default_timeout=120).run()
    at.selectbox(key="fire_floor").set_value("F05").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.value == "F04-F06" for m in at.metric)
    at.multiselect(key="fire_extra").set_value(["F08"]).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("E1" in w.value for w in at.warning)


def test_fire_plan_page_numbers_match_artifacts():
    """The headline sentence and the planner banner must show the counts stored in eval/fire (nothing typed)."""
    import json

    import pytest

    ev = PAGE.parents[2] / "eval" / "fire"
    if not (ev / "sensor_eval.json").exists() or not (ev / "planner_checks.json").exists():
        pytest.skip("eval/fire artifacts not generated")
    res = json.loads((ev / "sensor_eval.json").read_text(encoding="utf-8"))
    chk = json.loads((ev / "planner_checks.json").read_text(encoding="utf-8"))
    at = AppTest.from_file(str(PAGE), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    md = "\n".join(str(m.value) for m in at.markdown)
    head = res["frozen_config"]["headline_features"]
    hall = res["hall"][head]["metrics"]
    s1, ts = hall["stage1_pm"], hall["two_stage"]
    assert f"{s1['fire']['alarmed']} of {s1['fire']['n']} fires" in md
    assert f"{s1['nuisance']['alarmed']} of {s1['nuisance']['n']} nuisances" in md
    assert f"{ts['nuisance']['alarmed']} of {ts['nuisance']['n']} nuisances fire-like" in md
    assert f"AUROC {hall['stage2_auroc']['auroc']:.2f}" in md
    # the verdict sentence follows the artifact, not a typed conclusion
    no_transfer = ts["nuisance"]["alarmed"] >= s1["nuisance"]["alarmed"] and hall["stage2_auroc"]["auroc"] < 0.75
    assert ("did not transfer to a new room" in md) == no_transfer
    ok = "\n".join(str(s.value) for s in at.success)
    assert f"{chk['cards_passing']} of {chk['cards_checked']}" in ok
    assert f"{chk['escalated_cards_passing']} of {chk['escalated_cards_checked']}" in ok
    # the promise no longer claims every rule is sourced
    assert "Every rule has a source" not in md and "[team-proposed, validate] tag" in md
    # results table: the displayed cells equal the artifact
    df = next(d.value for d in at.dataframe if "Method" in d.value.columns and "Bg count" in d.value.columns)  # headline tab
    row = df[df["Method"].str.startswith("Two-stage")].iloc[0]
    assert row["Fires"].startswith(f"{ts['fire']['alarmed']}/{ts['fire']['n']} ")
    assert row["Bg count"] == f"{ts['background']['alarms']} ({ts['background']['sensor_hours']:.0f} h)"
