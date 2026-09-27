"""Unit tests for cascade.building.rainexposure (ISO 15927-3 WDR physics, loading rules, QC, circular statistics).

All inputs are tiny SYNTHETIC fixtures written for the tests (tests/fixtures/rain/synthetic_asos_sample.csv and inline
frames); no network and no data/raw access.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cascade.building import rainexposure as rx

FIX = Path(__file__).parent / "fixtures" / "rain"


# ----------------------------------------------------------------------------------------- ISO physics

def test_iso_hourly_pins_blocken_carmeliet_eq5():
    # east wind 5 m/s, 2 mm/h: E wall = (2/9) * 5 * 2^(8/9) (wind speed to the power 1), W and N walls = 0
    h = rx.iso_hourly([5.0], [2.0], [90.0], [rx.AZ["E"], rx.AZ["W"], rx.AZ["N"]])[0]
    assert h[0] == pytest.approx((2 / 9) * 5 * 2 ** (8 / 9), rel=1e-12)
    assert h[0] == pytest.approx(2.0575, abs=5e-4)   # 0.2222 * 5 * 1.8517 (the verifier note's "2.069" is an arithmetic slip)
    assert h[1] == 0.0
    assert abs(h[2]) < 1e-12


def test_iso_exponents_wind_linear_rain_eight_ninths():
    a = rx.iso_hourly([4.0, 8.0, 4.0], [1.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0])[:, 0]
    assert a[1] / a[0] == pytest.approx(2.0)                 # v enters linearly (not v^(8/9))
    assert a[2] / a[0] == pytest.approx(2 ** (8 / 9))        # r enters as r^(8/9)


def test_iso_any_azimuth_and_cosine():
    h = rx.iso_hourly([6.0], [3.0], [100.0], [100.0, 40.0, 190.0, 280.0])[0]
    full = (2 / 9) * 6 * 3 ** (8 / 9)
    assert h[0] == pytest.approx(full)
    assert h[1] == pytest.approx(full * 0.5)                 # cos 60
    assert abs(h[2]) < 1e-12                                 # cos 90
    assert h[3] == 0.0                                       # leeward


def test_iso_calm_missing_and_dry_give_zero():
    h = rx.iso_hourly([0.0, 5.0, 5.0, np.nan], [2.0, 0.0, 2.0, 2.0], [90.0, 90.0, np.nan, 90.0], [90.0])[:, 0]
    assert list(h) == [0.0, 0.0, 0.0, 0.0]


def test_annual_index_divides_by_years():
    idx = pd.date_range("2020-01-01", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame({"r": [1.0, 1.0, 0.0, 1.0], "v": [3.0, 3.0, 3.0, 3.0], "D": [90.0, 90.0, 90.0, 270.0]}, index=idx)
    ia = rx.annual_index(rx.iso_frame(df), 2)
    assert ia["E"] == pytest.approx(2 * (2 / 9) * 3 / 2)
    assert ia["W"] == pytest.approx((2 / 9) * 3 / 2)


def test_c_r_table_values():
    assert rx.c_r(100, "IV") == pytest.approx(0.24 * math.log(100 / 1.0))
    assert rx.c_r(10, "IV") == pytest.approx(0.24 * math.log(16))          # below z_min uses z_min
    assert rx.c_r(100, "IV") / rx.c_r(16, "IV") == pytest.approx(1.661, abs=1e-3)
    assert rx.c_r(10, "II") == pytest.approx(0.19 * math.log(10 / 0.05))


def test_obstruction_factor_table():
    assert rx.obstruction_factor(None) == 1.0
    assert rx.obstruction_factor(5) == 0.2
    assert rx.obstruction_factor(50) == 0.6
    assert rx.obstruction_factor(119.9) == 0.9
    assert rx.obstruction_factor(500) == 1.0
    assert rx.obstruction_factor(2) == 0.2


def test_wall_index_combines_factors():
    assert rx.wall_index(100.0, 100, "IV", 50) == pytest.approx(100 * 0.24 * math.log(100) * 0.6)


def test_spells_split_on_96_hours_of_zero_index():
    idx = pd.date_range("2020-01-01", periods=400, freq="h", tz="UTC")
    s = pd.Series(0.0, index=idx)
    s.iloc[0] = 1.0
    s.iloc[96] = 2.0            # 95 zero hours in between -> same spell
    s.iloc[193] = 3.0           # 96 zero hours in between -> new spell
    sp = rx.spells(s)
    assert len(sp) == 2
    assert list(sp["total"]) == [3.0, 3.0]


# ----------------------------------------------------------------------------------------- loading and QC

def test_load_asos_rules_on_synthetic_fixture():
    d = rx.load_asos_csv(FIX / "synthetic_asos_sample.csv", "CQT")
    assert len(d) == 7                                            # minute-10 special and post-move row dropped
    ts = lambda s: pd.Timestamp(s, tz="UTC")                      # noqa: E731
    assert d.loc[ts("2011-12-01 01:00"), "r"] == 0.0              # 'M' before the 2012 switch -> 0
    assert d.loc[ts("2011-12-01 02:00"), "r"] == 0.0              # 'T' -> 0
    assert d.loc[ts("2011-12-01 03:00"), "r"] == pytest.approx(2.54)
    assert d.loc[ts("2011-12-01 03:00"), "v"] == pytest.approx(10 * 0.514444)
    assert np.isnan(d.loc[ts("2013-03-01 01:00"), "r"])           # 'M' after the switch -> missing
    assert bool(d.loc[ts("2013-03-01 02:00"), "calm"])
    assert np.isnan(d.loc[ts("2013-03-01 02:00"), "D"])           # calm has no direction
    assert np.isnan(d.loc[ts("2013-03-01 03:00"), "v"])
    assert d.index.max() < pd.Timestamp(rx.rule("CQT_END"))       # CQT cut at the 2024-05-20 site move


def test_qc_daily_zeroes_spike_days_and_ratio():
    idx = pd.date_range("2008-10-20 09:00", periods=48, freq="h", tz="UTC")  # two local-standard days
    a = pd.DataFrame({"r": 0.0, "v": 3.0, "D": 90.0}, index=idx)
    a.iloc[5, 0] = 24.6                                           # spurious spike on day 1 (GHCN says 0)
    a.iloc[30, 0] = 4.0                                           # real rain on day 2
    g = pd.Series([0.0, 4.0], index=pd.to_datetime(["2008-10-20", "2008-10-21"]))
    clean, tab = rx.qc_daily(a, g)
    assert tab["spike"].sum() == 1
    assert clean["r"].sum() == pytest.approx(4.0)
    wy = rx.qc_water_years(tab, min_ghcnd_mm=1.0)
    assert wy.loc[0, "ratio"] == pytest.approx(1.0)
    assert bool(wy.loc[0, "passed"])


def test_lst_day_uses_local_standard_time():
    # hour ending 08:00 UTC = 00:00 PST belongs to the previous local day; hour ending 09:00 UTC to the new day
    d = rx.lst_day(pd.DatetimeIndex(["2020-01-02 08:00", "2020-01-02 09:00"], tz="UTC"))
    assert str(d[0].date()) == "2020-01-01" and str(d[1].date()) == "2020-01-02"


# ----------------------------------------------------------------------------------------- events and circular stats

def test_segment_events_gap_and_minimum():
    idx = pd.date_range("2020-01-01", periods=40, freq="h", tz="UTC")
    r = pd.Series(0.0, index=idx)
    r.iloc[[0, 1, 2]] = 2.0             # 6 mm
    r.iloc[8] = 1.0                     # 5 dry hours after hour 2 -> same event (gap 6)
    r.iloc[20] = 3.0                    # 11 dry hours -> new event, only 3 mm -> dropped at min 5
    ev = rx.segment_events(r, gap_h=6, min_mm=5)
    assert len(ev) == 1
    assert ev.loc[0, "total_mm"] == pytest.approx(7.0)
    assert ev.loc[0, "wet_hours"] == 4


def test_weighted_circmean_wraps_north():
    m, rbar = rx.weighted_circmean([350.0, 10.0])
    assert min(m, 360 - m) < 1e-9
    assert rbar == pytest.approx(math.cos(math.radians(10)))


def test_rayleigh_uniform_vs_concentrated():
    uni = rx.rayleigh([0, 90, 180, 270] * 10)
    rng = np.random.default_rng(0)
    conc = rx.rayleigh(120 + rng.normal(0, 15, 60))
    assert uni["p"] > 0.5
    assert conc["p"] < 1e-10


def test_block_permutation_detects_direction_difference():
    rng = np.random.default_rng(1)
    n = 60
    same = rng.uniform(0, 360, size=(2 * n, 10))
    sx, sy = np.sin(np.deg2rad(same)).sum(1), np.cos(np.deg2rad(same)).sum(1)
    lab = np.arange(2 * n) < n
    p_same = rx.block_permutation_test(sx, sy, np.full(2 * n, 10), lab, n_perm=300)["p"]
    rain = 100 + rng.normal(0, 20, size=(n, 10))
    dry = 260 + rng.normal(0, 20, size=(n, 10))
    ang = np.deg2rad(np.vstack([rain, dry]))
    res = rx.block_permutation_test(np.sin(ang).sum(1), np.cos(ang).sum(1), np.full(2 * n, 10), lab, n_perm=300)
    assert p_same > 0.05
    assert res["p"] <= 1 / 301 + 1e-12


def test_sector_shares_centered_on_compass_points():
    s = rx.sector_shares([0.0, 22.0, 23.0, 90.0])
    assert s["N"] == pytest.approx(0.5) and s["NE"] == pytest.approx(0.25) and s["E"] == pytest.approx(0.25)


# ----------------------------------------------------------------------------------------- building-layer API

def _fake_roses():
    """SYNTHETIC station entry: exposure peaks at 100 deg."""
    az = np.arange(360)
    curve = np.clip(np.cos(np.deg2rad(az - 100.0)), 0, None) * 50 + 5
    ia = {f: float(curve[int(rx.AZ[f])]) for f in rx.FACADES}
    return {"stations": {"LAX": {"IA_curve_1deg": curve.tolist(), "IA_L_m2_yr": ia, "years_used": 20}}}


def test_exposure_rank_and_facade_exposure():
    ro = _fake_roses()
    rk = rx.exposure_rank(ro["stations"]["LAX"], 90.0)
    assert rk["rank_of_8"] == 1 and rk["max_azimuth"] == 100
    assert rx.exposure_rank(ro["stations"]["LAX"], 270.0)["rank_of_8"] == 5   # ties with the other leeward sides share rank 5
    res = rx.facade_exposure(33.94, -118.40, [90.0, 270.0], [10.0, 100.0], terrain="IV", obstruction_m=50, roses=ro)
    assert res["station"] == "LAX" and len(res["rows"]) == 4
    r100 = [r for r in res["rows"] if r["azimuth"] == 90.0 and r["height_m"] == 100.0][0]
    assert r100["I_WA"] == pytest.approx(r100["I_A"] * 0.24 * math.log(100) * 0.6)
    sh = rx.exposure_shares(33.94, -118.40, ["N", "E", "S", "W"], roses=ro)
    assert sum(sh.values()) == pytest.approx(1.0) and max(sh, key=sh.get) == "E"


def test_wdr_iso_series_on_building_weather_contract():
    # SYNTHETIC weather rows in the building weather.csv contract
    idx = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    w = pd.DataFrame({"rain_mm": [2.0, 2.0, np.nan, 1.0], "wind_speed_ms": [5.0, 5.0, 5.0, 0.0],
                      "wind_dir_deg": [90.0, 270.0, 90.0, np.nan]}, index=idx)
    s = rx.wdr_iso_series(w, "E")
    assert s.iloc[0] == pytest.approx((2 / 9) * 5 * 2 ** (8 / 9))
    assert s.iloc[1] == 0.0                      # leeward
    assert np.isnan(s.iloc[2])                   # missing rain stays missing
    assert s.iloc[3] == 0.0                      # calm: no direction needed, index 0
    assert rx.wdr_iso_series(w, 90.0).iloc[0] == pytest.approx(s.iloc[0])


@pytest.mark.skipif(not rx.ROSES_PATH.exists(), reason="eval/rain artifacts not built")
def test_built_roses_are_self_consistent():
    """The 1-degree curve in the REAL artifact must agree with the 8-facade indices, and shares must sum to 1."""
    ro = rx.load_roses()
    assert len(ro["stations"]) == 12
    for sid, e in ro["stations"].items():
        for f in rx.FACADES:
            assert e["IA_curve_1deg"][int(rx.AZ[f])] == pytest.approx(e["IA_L_m2_yr"][f], rel=1e-3, abs=0.02), (sid, f)
        assert sum(e["IA_share"].values()) == pytest.approx(1.0, abs=5e-3)
        assert e["top_facade"] == max(e["IA_L_m2_yr"], key=e["IA_L_m2_yr"].get)
    res = rx.facade_exposure(34.05, -118.25, [0, 90, 180, 270], [10, 100])
    assert res["station"] == "CQT" and len(res["rows"]) == 8


def test_nearest_station():
    sid, km = rx.nearest_station(34.05, -118.25)       # downtown LA
    assert sid == "CQT" and km < 5
