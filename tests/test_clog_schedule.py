"""Scheduler rules: regulatory maximums, event triggers, approval flag."""
from datetime import date

from cascade.building.clog import schedule as S

AS_OF = date(2026, 9, 26)


def test_backflow_never_later_than_365_days():
    row = S.plan(S.Asset("BF", "backflow", "RP assembly", "2025-11-01"), AS_OF)
    assert row["next_check"] == "2026-11-01"
    assert "BACKFLOW_ANNUAL" in row["rule_keys"] and "licensed" in row["performed_by"]


def test_fog_projection_and_limit():
    readings = [("2026-09-01", 10.0), ("2026-09-08", 13.0), ("2026-09-15", 16.0)]
    proj, slope = S.fog_projection(readings, 25.0)
    assert abs(slope - 3.0 / 7) < 1e-9 and proj == date(2026, 10, 6)
    row = S.plan(S.Asset("GI", "grease_interceptor", "GI", "2026-08-30", fse_permit=True, fog_readings=readings),
                 AS_OF)
    assert row["next_check"] == "2026-10-03"
    over = S.plan(S.Asset("GI", "grease_interceptor", "GI", "2026-08-30", fse_permit=True,
                          fog_readings=readings + [("2026-09-22", 26.0)]), AS_OF)
    assert over["next_check"] == AS_OF.isoformat()


def test_backup_triggers_camera_within_48h_and_alarm_triggers_strainer():
    bd = S.plan(S.Asset("BD", "building_drain", "drain", "2026-01-01", floors_served=32, backups_12m=1,
                        last_backup="2026-09-25"), AS_OF)
    assert bd["next_check"] == "2026-09-27" and "CCTV_48H_AFTER_OVERFLOW" in bd["rule_keys"]
    st = S.plan(S.Asset("STR", "prv_strainer", "strainer", "2026-09-01", nightly_alarm_persistent=True), AS_OF)
    assert st["next_check"] == "2026-10-03"


def test_strainer_reason_reports_the_measured_alarm_nights():
    st = S.plan(S.Asset("STR", "prv_strainer", "strainer", "2026-09-01", nightly_alarm_persistent=True,
                        nightly_alarm_nights=4, nightly_nights=7, extra_headloss_m=1.08), AS_OF)
    assert "alarmed on 4 of 7 nights" in st["reason"] and "2 alarms in 3 consecutive nights" in st["reason"]
    calm = S.plan(S.Asset("STR", "prv_strainer", "strainer", "2026-09-01", nightly_alarm_nights=1,
                          nightly_nights=7), AS_OF)
    assert calm["reason"].startswith("No persistent nightly-test alarm (alarmed on 1 of 7 nights)")


def test_risk_shortens_camera_interval_and_every_row_needs_approval():
    calm = S.plan(S.Asset("A", "stack_base", "A", "2026-01-01", floors_served=11), AS_OF)
    risky = S.plan(S.Asset("B", "stack_base", "B", "2026-01-01", floors_served=11, backups_12m=2,
                           drain_down_ratio=1.5), AS_OF)
    assert risky["next_check"] < calm["next_check"]
    for r in (calm, risky):
        assert r["requires_human_approval"] is True
