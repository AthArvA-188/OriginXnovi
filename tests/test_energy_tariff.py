"""LADWP A-2 Rate B period mapping and charges (base charges as posted 2026-09-26)."""

import pandas as pd
import pytest

from cascade.building.energy import tariff as T


@pytest.mark.parametrize("ts,period,season,rate", [
    ("2026-07-15 14:00", "high_peak", "high", 0.06322),  # Wednesday, summer
    ("2026-07-18 14:00", "base", "high", 0.03522),       # Saturday
    ("2026-01-14 14:00", "high_peak", "low", 0.05688),   # Wednesday, winter
    ("2026-07-15 10:30", "low_peak", "high", 0.05595),
    ("2026-07-15 18:00", "low_peak", "high", 0.05595),
    ("2026-07-15 20:00", "base", "high", 0.03522),
    ("2026-07-15 12:59", "low_peak", "high", 0.05595),
    ("2026-07-15 16:59", "high_peak", "high", 0.06322),
    ("2026-07-03 14:00", "high_peak", "high", 0.06322),  # observed holiday (Fri): NOT mapped to base for LADWP
])
def test_ladwp_periods(ts, period, season, rate):
    t = [pd.Timestamp(ts)]
    assert T.ladwp_period(t)[0] == period
    assert T.ladwp_season(t)[0] == season
    assert T.energy_rate(t)[0] == pytest.approx(rate)


def test_demand_charges_constant_load_one_summer_week():
    idx = pd.date_range("2026-07-13", "2026-07-19 23:55", freq="5min")
    kw = pd.Series(10.0, index=idx)
    d = T.demand_charges(kw)
    assert d["demand_usd"] == pytest.approx(10 * 10.00 + 10 * 3.75)
    assert d["facilities_usd"] == pytest.approx(10 * 5.36)


def test_energy_cost_and_flags():
    idx = pd.date_range("2026-07-15 00:00", periods=24, freq="h")
    kwh = pd.Series(1.0, index=idx)
    exp = 4 * 0.06322 + 6 * 0.05595 + 14 * 0.03522
    assert T.energy_cost(kwh) == pytest.approx(exp)
    assert T.LADWP_A2B["holidays_as_base"] is False
    assert T.LADWP_DR["min_curtail_kw"] == 100
    tab = T.tou_hour_table()
    assert len(tab) == 24 and tab.loc[tab.hour == 17, "sce_summer_on_peak"].item()
    assert tab.loc[tab.hour == 14, "ladwp_summer_period"].item() == "high_peak"
