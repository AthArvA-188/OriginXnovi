"""Paired up-rise / down-drop rule on a tiny synthetic gauge pair (no Bellinge download needed)."""
import numpy as np
import pandas as pd

from cascade.building.clog import bellinge as B

BLOCK = pd.Timestamp("2020-07-16 12:00")


def _pair():
    idx = pd.date_range("2020-07-01", periods=20 * 288, freq="5min")
    h = idx.hour + idx.minute / 60
    rng = np.random.default_rng(0)
    dn = 0.05 + 0.02 * np.sin(2 * np.pi * h / 24) + rng.normal(0, 0.002, len(idx))
    up = 0.03 + 1.6 * dn + rng.normal(0, 0.002, len(idx))
    blk = idx >= BLOCK
    return pd.Series(np.where(blk, 1.1, up), idx), pd.Series(np.where(blk, 0.2 * dn, dn), idx)


def test_episodes_merge_gap():
    idx = pd.date_range("2020-01-01", periods=10, freq="1h")
    m = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 1], idx).astype(bool)
    assert len(B.episodes(m, gap="6h")) == 2
    assert len(B.episodes(m, gap="12h")) == 1


def test_paired_rule_fires_only_after_blockage():
    up, dn = _pair()
    X = B.make_features(up, dn)
    res = X["up"] - (0.03 + 1.6 * X["dn"])  # stand-in for a normal-behaviour model
    res_s = res.rolling(6, min_periods=4).median()
    alarm = B.paired_mask(res_s, X["dn"], thr=0.05, ratio=0.7, win_days=3)
    first = alarm[alarm].index.min()
    assert BLOCK <= first <= BLOCK + pd.Timedelta("1h")
    assert not alarm[alarm.index < BLOCK].any()


def test_onset_detects_rise():
    up, _dn = _pair()
    ons = B.onset(up, rise_m=0.10, ev=(pd.Timestamp("2020-07-16"), pd.Timestamp("2020-07-18")))
    assert ons == BLOCK


def test_summarise_counts_hits_and_other_episodes():
    idx = pd.date_range("2020-07-01", "2020-10-11 23:55", freq="5min")
    m = pd.Series(False, idx)
    m[(idx >= "2020-07-23 19:00") & (idx < "2020-07-24")] = True
    m[(idx >= "2020-08-10 10:00") & (idx < "2020-08-10 11:00")] = True
    s = B.summarise(m, pd.Timestamp("2020-07-23 18:30"))
    assert s["detected"] and s["first_alarm"] == "2020-07-23 19:00:00"
    assert s["delay_min_from_onset"] == 30.0 and s["other_test_episodes"] == 1
    assert s["pre_onset_episodes"] == 0


def test_summarise_does_not_credit_alarms_long_before_onset():
    idx = pd.date_range("2020-07-01", "2020-10-11 23:55", freq="5min")
    m = pd.Series(False, idx)
    m[(idx >= "2020-07-23 02:15") & (idx < "2020-07-23 05:00")] = True  # nightly-style episode, 16 h early
    s = B.summarise(m, pd.Timestamp("2020-07-23 18:30"))
    assert not s["detected"] and s["first_alarm"] is None and s["delay_min_from_onset"] is None
    assert s["pre_onset_episodes"] == 1 and s["other_test_episodes"] == 1
    m[(idx >= "2020-07-23 18:00") & (idx < "2020-07-23 20:00")] = True  # within the margin: attributable
    s = B.summarise(m, pd.Timestamp("2020-07-23 18:30"))
    assert s["detected"] and s["delay_min_from_onset"] == -30.0 and s["pre_onset_episodes"] == 1
