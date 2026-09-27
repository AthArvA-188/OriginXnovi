"""REAL case study: the documented Bellinge throttle-pipe blockage, 23-28 July 2020 (Pedersen et al. 2021, CC BY 4.0).

Bellinge is a 1.7 km2 MUNICIPAL COMBINED SEWER in Denmark; the blocked element is a throttle pipe between two basins
(upstream gauge G71F04R Level 1, downstream gauge G71F06R Level inlet). It checks detector LOGIC on real data; it is
not a building drain and it is ONE event, so results are a case study, never an accuracy or recall.

Detectors (5-min data, local time UTC+01 with DST as delivered):
  fixed level   upstream depth above a fixed threshold (several thresholds, all shown)
  residual      a normal-behaviour model predicts upstream depth from downstream depth, its lags and time of day;
                alarm when the 30-min median residual exceeds a threshold set on TIME-BLOCKED OUT-OF-FOLD residuals
  paired        residual alarm AND downstream 30-min median below `ratio` x its trailing `win`-day median
                (a blockage holds water upstream and starves the pipe downstream)
Train Jan-Jun 2020 (no documented blockage), test 2020-07-01 to 2020-10-11.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

EVENT = (pd.Timestamp("2020-07-23"), pd.Timestamp("2020-07-29"))  # documented 23-07-2020 - 28-07-2020, inclusive
TRAIN = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-07-01"))
TEST = (pd.Timestamp("2020-07-01"), pd.Timestamp("2020-10-12"))
STEP = "5min"
PER_DAY = 288


def episodes(mask: pd.Series, gap: str = "6h") -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Merge alarm timestamps closer than `gap` into (start, end) episodes."""
    t = mask[mask.fillna(False).astype(bool)].index
    if len(t) == 0:
        return []
    eps, s, prev = [], t[0], t[0]
    for x in t[1:]:
        if x - prev > pd.Timedelta(gap):
            eps.append((s, prev))
            s = x
        prev = x
    eps.append((s, prev))
    return eps


def split_hits(eps, ev=EVENT):
    hit = [e for e in eps if e[1] >= ev[0] and e[0] < ev[1]]
    other = [e for e in eps if not (e[1] >= ev[0] and e[0] < ev[1])]
    return hit, other


def make_features(up: pd.Series, dn: pd.Series) -> pd.DataFrame:
    """5-min regular grid; features use only downstream depth and time of day (never upstream)."""
    idx = pd.date_range(min(up.index.min(), dn.index.min()), max(up.index.max(), dn.index.max()), freq=STEP)
    X = pd.DataFrame({"up": up.reindex(idx), "dn": dn.reindex(idx)})
    for lag in (1, 3, 6, 12):
        X[f"dn_l{lag}"] = X["dn"].shift(lag)
    X["dn_r12"] = X["dn"].rolling(12, min_periods=6).mean()
    X["dn_r72"] = X["dn"].rolling(72, min_periods=36).mean()
    h = X.index.hour + X.index.minute / 60.0
    X["hs"], X["hc"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    return X


FEATS = ["dn", "dn_l1", "dn_l3", "dn_l6", "dn_l12", "dn_r12", "dn_r72", "hs", "hc"]


def _model():
    return HistGradientBoostingRegressor(max_iter=300, random_state=0)


def fit_residuals(X: pd.DataFrame) -> Tuple[pd.Series, pd.Series, object]:
    """Returns (residual for every row: out-of-fold by calendar month inside TRAIN, model-on-all-train outside),
    the 30-min median of it, and the final model."""
    ok = X[["up"] + FEATS].notna().all(axis=1)
    tr = ok & (X.index >= TRAIN[0]) & (X.index < TRAIN[1])
    res = pd.Series(np.nan, index=X.index)
    months = X.index.to_period("M")
    for m in sorted(set(months[tr])):
        hold = tr & (months == m)
        fit = tr & (months != m)
        mdl = _model().fit(X.loc[fit, FEATS], X.loc[fit, "up"])
        res[hold] = X.loc[hold, "up"] - mdl.predict(X.loc[hold, FEATS])
    final = _model().fit(X.loc[tr, FEATS], X.loc[tr, "up"])
    rest = ok & ~tr
    res[rest] = X.loc[rest, "up"] - final.predict(X.loc[rest, FEATS])
    res_s = res.rolling(6, min_periods=4).median()
    return res, res_s, final


def paired_mask(res_s: pd.Series, dn: pd.Series, thr: float, ratio: float, win_days: int) -> pd.Series:
    med = dn.rolling(PER_DAY * win_days, min_periods=PER_DAY).median()
    dn_s = dn.rolling(6, min_periods=4).median()
    return (res_s > thr) & (dn_s < ratio * med)


def onset(up: pd.Series, rise_m: float = 0.10, ev=EVENT) -> pd.Timestamp:
    """First 5-min step inside the documented window where upstream depth exceeds the median of the previous 24 h
    by more than rise_m (the visible onset; the log gives dates only)."""
    base = up.rolling(PER_DAY, min_periods=PER_DAY // 2).median().shift(1)
    m = (up > base + rise_m) & (up.index >= ev[0]) & (up.index < ev[1])
    hits = m[m].index
    return hits[0] if len(hits) else pd.NaT


PRE_ONSET_MARGIN_MIN = 60  # team-proposed: an episode starting more than 60 min before the visible onset is not
#                             attributed to the blockage (the log gives dates only, so the window starts at midnight)


def summarise(mask: pd.Series, ons: pd.Timestamp, margin_min: float = PRE_ONSET_MARGIN_MIN) -> Dict[str, object]:
    """First attributable alarm in the event window, delay from onset, other episodes in test and train periods.

    Episodes inside the documented date window that START more than `margin_min` before the visible onset are not
    counted as detections: they are reported as `pre_onset_episodes` and added to the other (unattributable)
    held-out episodes, so a detector that fires every night is never credited with an early detection."""
    test = mask[(mask.index >= TEST[0]) & (mask.index < TEST[1])]
    train = mask[(mask.index >= TRAIN[0]) & (mask.index < TRAIN[1])]
    hit, other = split_hits(episodes(test))
    pre = []
    if not pd.isna(ons):
        cut = ons - pd.Timedelta(minutes=margin_min)
        pre = [e for e in hit if e[0] < cut]
        hit = [e for e in hit if e[0] >= cut]
    months_test = (TEST[1] - TEST[0]).days / 30.44
    first = hit[0][0] if hit else None
    delay = None if first is None or pd.isna(ons) else (first - ons).total_seconds() / 60.0
    n_other = len(other) + len(pre)
    return {"detected": bool(hit), "first_alarm": None if first is None else str(first),
            "delay_min_from_onset": delay, "other_test_episodes": n_other,
            "other_test_episodes_per_month": n_other / months_test,
            "pre_onset_episodes": len(pre), "pre_onset_first": str(pre[0][0]) if pre else None,
            "pre_onset_margin_min": margin_min,
            "train_episodes": len(episodes(train)),
            "other_episode_starts": [str(e[0]) for e in sorted(other + pre)[:20]]}
