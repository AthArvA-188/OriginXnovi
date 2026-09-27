"""Tests for cascade.building.firesense on tiny SYNTHETIC fixtures (no network, no data/raw).

They check the verifier's corrections: episodes split on a gap OR a label change, the Hall 13 -> 14 merge, causal
features, k-consecutive alarm runs, attribution with a 5-minute pre-start window and guard bands, thresholds from
training background only, and the metric arithmetic. The last test checks the published artifact, when present.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cascade.building import firesense as fs

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "fire"


@pytest.fixture(scope="module")
def en54():
    df = fs.load_site(FIX / "en54_tiny.csv", "en54")
    eps = fs.episodes_en54(df)
    return df, eps


@pytest.fixture(scope="module")
def hall():
    df = fs.load_site(FIX / "hall_tiny.csv", "hall")
    eps = fs.episodes_hall(df)
    return df, eps


def test_en54_episodes_split_on_gap_or_label_change(en54):
    _, eps = en54
    assert list(eps["kind"]) == ["fire", "nuisance", "nuisance", "nuisance"]
    assert list(eps["scenario"]) == ["Wood", "Deodorant", "Ethanol", "Ethanol"]
    # the gap-only rule would merge Wood and Deodorant (no gap between them)
    assert eps["start"].is_monotonic_increasing


def test_hall_merges_fragment_13_into_14_and_reports_other(hall):
    _, eps = hall
    assert list(eps["ep"]) == ["hall-03", "hall-05", "hall-14"]
    assert dict(zip(eps["ep"], eps["kind"])) == {"hall-03": "nuisance", "hall-05": "other", "hall-14": "fire"}
    frag_t = pd.Timestamp("2023-07-10 10:39:57", tz="UTC")
    start14 = eps.set_index("ep").loc["hall-14", "start"]
    assert abs((start14 - frag_t).total_seconds()) <= 10  # the fragment row is now the start of episode 14


def test_hall_timestamps_are_utc(hall):
    df, _ = hall
    assert str(df["t"].dt.tz) == "UTC"
    assert df["t"].min() == pd.Timestamp("2023-07-10 08:30:00", tz="UTC")


def test_features_are_causal(en54):
    df, eps = en54
    full = fs.add_features(df)
    cut = pd.Timestamp("2022-07-04 01:10:00", tz="UTC")
    part = fs.add_features(df[df["t"] <= cut].reset_index(drop=True))
    cols = list(fs.FEATURE_SETS["full"])
    a = full[full["t"] <= cut].sort_values(["sensor", "t"])[cols].to_numpy(dtype="float64")
    b = part.sort_values(["sensor", "t"])[cols].to_numpy(dtype="float64")
    assert a.shape == b.shape
    assert np.allclose(a, b, equal_nan=True)


def test_delta_excludes_the_current_sample(en54):
    df, _ = en54
    f = fs.add_features(df)
    g = f[f["sensor"] == f["sensor"].iloc[0]].reset_index(drop=True)
    i = 400
    win = g[(g["t"] > g["t"][i] - pd.Timedelta("30min")) & (g["t"] < g["t"][i])]["PM_Total_Room"]
    assert g["PM_Total_Room_d"][i] == pytest.approx(g["PM_Total_Room"][i] - win.median(), abs=1e-3)


def test_run_length_and_rising_edges():
    c = np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
    seg = np.zeros(10, dtype=bool)
    seg[0] = True
    assert fs.run_length(c, seg).tolist() == [1, 2, 3, 4, 0, 1, 2, 3, 0, 1]
    seg[2] = True  # a gap restarts the run
    assert fs.run_length(c, seg).tolist() == [1, 2, 1, 2, 0, 1, 2, 3, 0, 1]
    df = pd.DataFrame({"sensor": ["a"] * 10, "t": pd.date_range("2022-01-01", periods=10, freq="10s", tz="UTC"), "warm": False})
    df["ts"] = fs.secs(df["t"])
    assert fs.alarm_events(df, c).tolist() == [2, 7]


def test_guard_bands_and_attribution(en54):
    df, eps = en54
    p = fs.prepare(df, eps)
    st = eps["start_s"].iloc[0]
    en = eps["end_s"].iloc[-1]
    # warm-up: first 30 min never clean background
    assert not p.loc[p["ts"] < p["ts"].min() + 1790, "clean_bg"].any()
    # 20 min before the first episode is inside the 30-min guard band
    assert not p.loc[(p["ts"] > st - 1200) & (p["ts"] < st), "clean_bg"].any()
    # 45 min after the last episode end is inside the 60-min guard band
    assert not p.loc[(p["ts"] > en) & (p["ts"] < en + 2700), "clean_bg"].any()
    assert p.loc[p["ts"] > en + 3700, "clean_bg"].all()
    ts = np.array([st - 240, st - 900, st - 7200, st + 60])
    cb = np.array([False, False, True, False])
    assert fs.attribute(ts, eps, cb).tolist() == ["en54-01", "near", "bg", "en54-01"]


def test_stage1_threshold_uses_only_clean_background(en54):
    df, eps = en54
    p = fs.prepare(df, eps)
    thr = fs.stage1_threshold(p, "PM_Total_Room_d", 0.999)
    bg = p.loc[p["clean_bg"], "PM_Total_Room_d"].dropna()
    assert thr == pytest.approx(float(bg.quantile(0.999)))
    assert thr < 50  # the 300-count fire rise in the fixture never leaks into the threshold


def _toy_run(en54, pfire_fire=0.9):
    df, eps = en54
    p = fs.prepare(df, eps).sort_values(["sensor", "t"], kind="stable").reset_index(drop=True)
    p["pfire"] = np.where(p["kind"] == "fire", pfire_fire, 0.05).astype("float32")
    p["thr_pm"] = fs.stage1_threshold(p, fs.FROZEN["stage1_channel"], fs.FROZEN["stage1_quantile"])
    p["thr_co"] = fs.stage1_threshold(p, fs.FROZEN["co_channel"], fs.FROZEN["stage1_quantile"])
    return fs.SiteRun(df=p, eps=eps, thr_pm=None, thr_co=None)


def test_two_stage_only_labels_existing_triggers(en54):
    run = _toy_run(en54)
    s1 = fs.events_for_method(run, "stage1_pm")
    two = fs.events_for_method(run, "two_stage")
    allv = fs.events_for_method(run, "two_stage_all")
    assert set(two["row"]) <= set(s1["row"])
    assert len(allv) == len(s1)  # every trigger stays, with a label
    assert (two["t_alarm"] >= two["t"]).all()
    assert (two["t_alarm"] <= two["t"] + fs.FROZEN["verify_window_s"]).all()


def test_metric_arithmetic_on_fixture(en54):
    run = _toy_run(en54)
    ev = fs.events_for_method(run, "stage1_pm")
    r = fs.summarize(run, "stage1_pm", ev)
    # fixture: the Wood fire (+300 counts) raises PM; Ethanol does not. Deodorant follows the fire directly, so its
    # rise sits below the still-elevated 30-min rolling median and does not trigger either.
    assert r["fire"] == {"n": 1, "alarmed": 1, "rate": 1.0, "ci95": [1.0, 1.0]}
    assert r["nuisance"]["n"] == 3
    tab = fs.episode_table(run, {"stage1_pm": ev}).set_index("ep")
    assert tab.loc["en54-03", "stage1_pm_first_min"] is None or np.isnan(tab.loc["en54-03", "stage1_pm_first_min"])
    assert r["nuisance"]["alarmed"] == int(tab.loc[tab["kind"] == "nuisance", "stage1_pm_first_min"].notna().sum())
    lat = r["latency_min"]["median"]
    assert lat is not None and -5.0 <= lat <= 5.0
    two = fs.summarize(run, "two_stage", fs.events_for_method(run, "two_stage"))
    assert two["fire"]["alarmed"] == 1 and two["nuisance"]["alarmed"] == 0
    bg = r["background"]
    assert bg["sensor_hours"] == pytest.approx(run.df["clean_bg"].sum() * 10 / 3600)
    if bg["sensor_hours"] > 0:
        assert bg["per_24_sensor_h"] == pytest.approx(bg["alarms"] / bg["sensor_hours"] * 24)
    again = fs.summarize(run, "stage1_pm", ev)
    assert again == r  # seeded bootstrap is deterministic


def test_stage2_trains_and_scores(en54):
    run = _toy_run(en54)
    clf, info = fs.train_stage2(run.df, fs.FEATURE_SETS["deltas"], {**fs.FROZEN, "hgb": {**fs.FROZEN["hgb"], "max_iter": 20}})
    p = fs.p_fire(clf, run.df, fs.FEATURE_SETS["deltas"])
    assert info["n_fire_rows"] > 0 and info["n_background_rows"] > 0
    assert np.nanmin(p) >= 0 and np.nanmax(p) <= 1


def test_frozen_hash_stable():
    assert fs.frozen_hash() == fs.frozen_hash(dict(fs.FROZEN))
    assert fs.frozen_hash({**fs.FROZEN, "stage2_p_fire": 0.6}) != fs.frozen_hash()


def test_published_artifact_consistent():
    p = ROOT / "eval" / "fire" / "sensor_eval.json"
    if not p.exists():
        pytest.skip("eval/fire/sensor_eval.json not generated yet")
    r = json.loads(p.read_text(encoding="utf-8"))
    assert r["frozen_hash"] == fs.frozen_hash(r["frozen_config"])
    for block in [r["cv_en54"][v]["metrics"] for v in r["cv_en54"]] + ([r["hall"][v]["metrics"] for v in ("full", "deltas")] if "hall" in r else []):
        for m in fs.METHODS:
            for kind in ("fire", "nuisance", "other"):
                assert 0 <= block[m][kind]["alarmed"] <= block[m][kind]["n"]
    if "hall" in r:
        assert r["hall"]["meta"]["frozen_hash"] == r["frozen_hash"]
        assert r["episodes"]["hall"]["by_kind"] == {"fire": 17, "nuisance": 11, "other": 3}
    assert r["episodes"]["en54"]["by_kind"] == {"fire": 13, "nuisance": 7}


def test_posthoc_summary_matches_episode_table():
    """sensitivity_hall.json counts are recomputable from the saved per-episode table (no typed numbers)."""
    import pandas as pd

    p = ROOT / "eval" / "fire" / "sensitivity_hall.json"
    e_path = ROOT / "eval" / "fire" / "episodes_hall_full.csv"
    if not (p.exists() and e_path.exists()):
        pytest.skip("post-hoc summary not generated yet")
    s = json.loads(p.read_text(encoding="utf-8"))
    e = pd.read_csv(e_path)
    for m in fs.METHODS:
        c = e[f"{m}_first_min"]
        got = s["first_alarm_before_label_start"][m]
        assert got["alarmed"] == int(c.notna().sum())
        assert got["first_alarm_early"] == int((c < 0).sum())
    g = s["episode_gaps_min"]
    assert 0 < g["min"] <= g["median"] <= g["max"] and g["n"] == 31 - 4  # 31 Hall episodes over 4 recording days
