"""Unit tests for cascade.building.rainmodel on tiny SYNTHETIC (labelled) fixtures: event table construction from
gridded precipitation, model fit/predict shapes, baselines, metrics and the storm-cluster bootstrap.
No network, no data/raw."""

import numpy as np
import pandas as pd
import pytest

from cascade.building import rainexposure as rx
from cascade.building import rainmodel as rm


def _synthetic_hourly(seed=0, hours=24 * 60):
    """SYNTHETIC gridded weather + SYNTHETIC ASOS frame with a few easterly storms."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-10-01", periods=hours, freq="h", tz="UTC")
    p = np.zeros(hours)
    for k in range(6):
        s = 100 + k * 200
        p[s:s + 10] = rng.uniform(0.5, 3.0, 10)
    src = pd.DataFrame({"precip": p, "ws10": 5.0, "wd10": 100.0, "ws100": 8.0, "wd100": 110.0}, index=idx)
    asos = pd.DataFrame({"r": p * 1.1, "v": 4.0, "D": 95.0}, index=idx)
    return src, asos


def test_build_event_table_targets_and_features():
    src, asos = _synthetic_hourly()
    df, c = rm.build_event_table("TST", src, asos, 34.0, -118.0, 50.0, passed_wy=[2022], gap_h=6, min_mm=5)
    assert c["events_detected"] == 6 and len(df) == 6
    assert set(rm.FEATS_FULL) <= set(df.columns) and set(rm.YCOLS) <= set(df.columns)
    # easterly storms: the E facade target dominates and the W facade is dry
    assert (df["y_E"] > df["y_W"]).all() and (df["y_W"] == 0).all()
    # target equals the ISO index of the synthetic ASOS series over the padded window
    e0 = df.iloc[0]
    w = asos[(asos.index >= e0["start"] - pd.Timedelta(hours=3)) & (asos.index <= e0["end"] + pd.Timedelta(hours=3))]
    assert e0["y_E"] == pytest.approx(rx.iso_frame(w)["E"].sum())
    assert e0["b1_E"] > 0 and e0["p_tot"] >= 5


def test_build_event_table_drops_failed_water_years():
    src, asos = _synthetic_hourly()
    df, c = rm.build_event_table("TST", src, asos, 34.0, -118.0, 50.0, passed_wy=[2019], gap_h=6, min_mm=5)
    assert len(df) == 0 and c["dropped_qc_wy"] == 6


def _synthetic_table(n=80, seed=0, stations=("A", "B", "C")):
    """SYNTHETIC event table: targets proportional to the physics features plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        st = stations[i % len(stations)]
        row = {"station": st, "start": pd.Timestamp("2016-10-01", tz="UTC") + pd.Timedelta(days=5 * i),
               "wy": 2017 + (i * len(stations) // n) * 3}
        for f in rm.FEATS_FULL:
            row[f] = float(rng.uniform(0, 10))
        for f in rx.FACADES:
            row[f"y_{f}"] = max(0.0, 0.8 * row[f"b1_{f}"] + rng.normal(0, 0.5))
        rows.append(row)
    return pd.DataFrame(rows)


def test_model_fit_predict_and_baselines_shapes():
    df = _synthetic_table()
    tr, te = df.iloc[:60], df.iloc[60:]
    preds = rm.predict_all(tr, te, rm.FEATS_10M)
    assert set(preds) == {"B0_station_climatology", "B1_iso_physics_on_grid", "B2_station_coef_x_grid_rain", "HGB_model"}
    for P in preds.values():
        assert P.shape == (len(te), 8) and np.isfinite(P).all() and (P >= 0).all()


def test_metrics_perfect_and_wrong():
    Y = np.array([[0, 0, 5, 1, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0, 0, 1.0]])
    m = rm.metrics(Y, Y)
    assert m["MAE_L_m2"] == 0 and m["top1_acc"] == 1 and m["within45_acc"] == 1 and m["median_spearman"] == pytest.approx(1)
    P = np.roll(Y, 1, axis=1)                    # most-exposed facade shifted by one sector (45 deg)
    m2 = rm.metrics(Y, P)
    assert m2["top1_acc"] == 0 and m2["within45_acc"] == 1
    P4 = np.roll(Y, 4, axis=1)                   # opposite facade
    assert rm.metrics(Y, P4)["within45_acc"] == 0


def test_clusters_and_bootstrap():
    starts = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 10:00", "2020-01-05 00:00", "2020-01-05 20:00", "2020-02-01 00:00"], utc=True)
    cl = rm.clusters_by_start(starts)
    assert len(np.unique(cl)) == 3 and cl[0] == cl[1] and cl[2] == cl[3]
    eb, em = np.array([2.0, 2, 2, 2, 2]), np.array([1.0, 1, 1, 1, 1])
    b = rm.cluster_bootstrap_diff(eb, em, cl, n_boot=200)
    assert b["diff"] == pytest.approx(1.0) and b["lo95"] == pytest.approx(1.0) and b["n_clusters"] == 3


def test_evaluate_split_and_grouped_scores():
    df = _synthetic_table(n=90)
    res = rm.evaluate_split(df[df["wy"] <= 2020], df[df["wy"] > 2020], rm.FEATS_10M, n_boot=50)
    assert res["n_test"] > 0 and "HGB_model" in res["methods"] and "B1_iso_physics_on_grid" in res["bootstrap_mae_diff_vs_hgb"]
    folds = [((df["station"] != s).to_numpy(), (df["station"] == s).to_numpy()) for s in ("A", "B", "C")]
    g, meta = rm.evaluate_grouped(df, rm.FEATS_10M, folds, n_boot=50)
    assert g["n_events"] == len(df) and set(g["per_station"]) == {"A", "B", "C"}


def test_open_meteo_frame_suffix():
    j = {"hourly": {"time": ["2024-02-04T00:00", "2024-02-04T01:00"], "precipitation_previous_day1": [0.0, 1.8],
                    "wind_speed_10m_previous_day1": [3.0, 4.0], "wind_direction_10m_previous_day1": [120, 130]}}
    f = rm.open_meteo_frame(j, suffix="_previous_day1")
    assert list(f.columns) == ["precip", "ws10", "wd10"] and f["precip"].iloc[1] == 1.8
    assert str(f.index.tz) == "UTC"
