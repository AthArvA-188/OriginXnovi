"""LA calendar: NOAA normals parsing, GML day length, DST, DR season days, seasonal plan; Open-Meteo GMT conversion."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cascade.building.energy import calendar_la as C
from cascade.building.energy import data as D

FX = Path(__file__).parent / "fixtures" / "energy"


@pytest.fixture(scope="module")
def noaa():
    return D.read_noaa_normals(FX / "noaa_normals_monthly_1991_2020.csv")


def test_noaa_parsing_strips_padding(noaa):
    assert noaa["MLY-CLDD-NORMAL"].dtype.kind == "f"
    assert set(noaa["STATION"]) == {C.USC, C.LAX}


def test_calendar_has_12_rows_and_cdd_matches_annual_normals(noaa):
    cal = C.build_calendar(2027, noaa)
    assert [r["month"] for r in cal] == list(range(1, 13))
    ann = D.read_noaa_normals(FX / "noaa_normals_annual_1991_2020.csv").set_index("STATION")
    assert sum(r["cdd65_usc"] for r in cal) == pytest.approx(ann.loc[C.USC, "ANN-CLDD-NORMAL"], abs=0.2)
    assert sum(r["cdd65_lax"] for r in cal) == pytest.approx(ann.loc[C.LAX, "ANN-CLDD-NORMAL"], abs=0.2)
    aug = cal[7]
    assert aug["cdd_rank_usc"] == 1 and aug["ladwp_season"] == "high"
    assert cal[5]["ladwp_dr_days"] == 16 and cal[6]["ladwp_dr_days"] == 31 and cal[9]["ladwp_dr_days"] == 15
    assert cal[0]["ladwp_dr_days"] == 0
    assert all(r["actions"] for r in cal)


def test_gml_day_length_la():
    assert C.solar_day(date(2026, 6, 21))["day_length_h"] == pytest.approx(14.43, abs=0.03)
    assert C.solar_day(date(2026, 12, 21))["day_length_h"] == pytest.approx(9.88, abs=0.03)


def test_dst_and_holidays():
    assert [d["date"] for d in C.dst_changes(2026)] == ["2026-03-08", "2026-11-01"]
    assert [d["date"] for d in C.dst_changes(2027)] == ["2027-03-14", "2027-11-07"]
    h = C.holidays(2026)
    assert {"2026-01-01", "2026-01-19"} <= {x["date"] for x in h}


def test_modes_and_plan(noaa):
    cal = C.build_calendar(2027, noaa)
    modes = C.month_hour_modes(cal)
    assert len(modes) == 288
    jul14 = next(m for m in modes if m["month_num"] == 7 and m["hour"] == 14)
    assert jul14["mode"] == "Peak cap + DR ready"
    jan14 = next(m for m in modes if m["month_num"] == 1 and m["hour"] == 14)
    assert jan14["mode"] != "Peak cap + DR ready"
    plan = C.seasonal_plan(cal)
    assert len(plan) == 4 and sorted(m for s in plan for m in s["months"]) == list(range(1, 13))


def test_openmeteo_gmt_conversion():
    pay = json.loads((FX / "openmeteo_la_2025_gmt_slice.json").read_text(encoding="utf-8"))
    df = D.openmeteo_to_local(pay)
    assert str(df.index.tz) == D.LA_TZ
    assert df.index.is_unique and df.index.is_monotonic_increasing
    # local midnight Jan 1 (PST, UTC-8) is the 08:00 UTC value
    i = pay["hourly"]["time"].index("2025-01-01T08:00")
    assert df.loc[pd.Timestamp("2025-01-01 00:00", tz=D.LA_TZ), "temperature_2m"] == pay["hourly"]["temperature_2m"][i]
    # spring forward on 2025-03-09: local 02:00 does not exist, 01:00 PST is followed by 03:00 PDT
    day = df[df.index.date == date(2025, 3, 9)]
    assert 2 not in set(day.index.hour)
    bad = dict(pay, utc_offset_seconds=-25200)
    with pytest.raises(ValueError):
        D.openmeteo_to_local(bad)


def test_action_text_and_modes_follow_the_tariff_table(noaa, monkeypatch):
    from cascade.building.energy import tariff as T

    monkeypatch.setitem(T.LADWP_A2B["periods_weekday"], "high_peak", [[14, 18]])
    cal = C.build_calendar(2027, noaa)
    assert any("weekdays 14:00-18:00" in a for a in cal[6]["actions"])
    modes = {(m["month_num"], m["hour"]): m["mode"] for m in C.month_hour_modes(cal)}
    assert modes[(7, 17)] == "Peak cap + DR ready" and modes[(7, 13)] != "Peak cap + DR ready"
    do = C.seasonal_plan(cal)[2]["do"]
    assert do[0].startswith("Weekdays 14:00-18:00") and "before 14:00" in do[1]
