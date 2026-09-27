"""M3 day-ahead hourly garage load forecaster on REAL BDG2 parking meters (train 2016, test 2017).

Forecast issued at 00:00 local for the next 24 h, so every lag is >= 24 h. Baselines: lag-24 (yesterday same hour)
and lag-168 (last week same hour). HGB variants: observed same-hour temperature (ORACLE weather, not available at
issue time), 24 h-lagged temperature (deployable) and no weather (deployable). Metrics: CV(RMSE) and NMBE (ASHRAE
Guideline 14 style, p = 0). Optional zero-shot comparator: amazon/chronos-bolt-small (Apache-2.0); its torch code lives
only in scripts/train_energy_models.py (step `chronos`, separate torch env). This module imports no torch.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

VARIANTS = {
    "hgb_oracle_weather": ["t", "how", "lag168", "lag24", "month"],
    "hgb_lagged_weather": ["t_lag24", "how", "lag168", "lag24", "month"],
    "hgb_no_weather": ["how", "lag168", "lag24", "month"],
}
DEPLOYABLE_HEADLINE = "hgb_no_weather"


def cvrmse(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)) / np.mean(y))


def nmbe(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sum(y - p) / (len(y) * np.mean(y)))


def meter_frame(s: pd.Series, temp: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": s.astype(float), "t": temp})
    df["how"] = df.index.dayofweek * 24 + df.index.hour
    df["lag168"] = df["y"].shift(168)
    df["lag24"] = df["y"].shift(24)
    df["month"] = df.index.month
    df["t_lag24"] = df["t"].shift(24)
    return df


def night_day_ratio(s: pd.Series) -> float:
    h = s.index.hour
    return float(s[(h >= 1) & (h < 5)].mean() / s[(h >= 10) & (h < 16)].mean())


def eval_meter(s: pd.Series, temp: pd.Series, max_iter: int = 150) -> Dict:
    df = meter_frame(s, temp)
    tr = df[df.index.year == 2016].dropna()
    te = df[df.index.year == 2017].dropna()
    y = te["y"].to_numpy()
    res = {"n_train_h": int(len(tr)), "n_test_h": int(len(te)),
           "lag168": {"cvrmse": cvrmse(y, te["lag168"]), "nmbe": nmbe(y, te["lag168"])},
           "lag24": {"cvrmse": cvrmse(y, te["lag24"]), "nmbe": nmbe(y, te["lag24"])}}
    preds = {}
    for name, X in VARIANTS.items():
        m = HistGradientBoostingRegressor(random_state=0, max_iter=max_iter).fit(tr[X], tr["y"])
        p = m.predict(te[X])
        preds[name] = p
        res[name] = {"cvrmse": cvrmse(y, p), "nmbe": nmbe(y, p)}
    res["best_naive"] = "lag24" if res["lag24"]["cvrmse"] <= res["lag168"]["cvrmse"] else "lag168"
    res["test_index"] = te.index
    res["preds"] = preds
    res["y"] = y
    return res


def garage_eval(el: pd.DataFrame, meta: pd.DataFrame, wx: pd.DataFrame, min_coverage: float = 0.8) -> Dict:
    rows, skipped, profiles, keep = [], [], {}, {}
    for b in el.columns:
        s = el[b].astype(float)
        cov = float(s.notna().mean())
        if cov < min_coverage or not (s.mean() > 0):
            skipped.append({"meter": b, "coverage": round(cov, 3), "reason": "coverage < 0.8" if cov < min_coverage else "mean <= 0"})
            continue
        site = b.split("_")[0]
        temp = wx[wx["site_id"] == site].set_index("timestamp")["airTemperature"].reindex(s.index).interpolate(limit=6)
        r = eval_meter(s, temp)
        row = {"meter": b, "site": site, "timezone": str(meta.loc[b, "timezone"]), "sqft": float(meta.loc[b, "sqft"]),
               "coverage": cov, "mean_kwh_per_h": float(s.mean()), "night_day_ratio": night_day_ratio(s),
               "n_train_h": r["n_train_h"], "n_test_h": r["n_test_h"]}
        for k in ("lag168", "lag24", *VARIANTS):
            row[f"cvrmse_{k}"] = r[k]["cvrmse"]
            row[f"nmbe_{k}"] = r[k]["nmbe"]
        best = min(row["cvrmse_lag168"], row["cvrmse_lag24"])
        for k in VARIANTS:
            row[f"{k}_beats_best_naive"] = bool(row[f"cvrmse_{k}"] < best)
        rows.append(row)
        keep[b] = {"index": r["test_index"], "y": r["y"], "hgb": r["preds"][DEPLOYABLE_HEADLINE]}
        wk = s[s.index.dayofweek < 5]
        prof = (wk.groupby(wk.index.hour).mean() / s.mean()).round(4)
        profiles[b] = [float(v) for v in prof.to_numpy()]
    f = pd.DataFrame(rows)
    summary = {"n_meters": int(len(f)), "n_skipped": len(skipped),
               "median_night_day_ratio": float(f["night_day_ratio"].median()),
               "median_cvrmse": {k: float(f[f"cvrmse_{k}"].median()) for k in ("lag168", "lag24", *VARIANTS)},
               "median_nmbe": {k: float(f[f"nmbe_{k}"].median()) for k in ("lag168", "lag24", *VARIANTS)},
               "beats_lag168": {k: int((f[f"cvrmse_{k}"] < f["cvrmse_lag168"]).sum()) for k in VARIANTS},
               "beats_lag24": {k: int((f[f"cvrmse_{k}"] < f["cvrmse_lag24"]).sum()) for k in VARIANTS},
               "beats_best_naive": {k: int(f[f"{k}_beats_best_naive"].sum()) for k in VARIANTS},
               "n_pacific": int((f["timezone"] == "US/Pacific").sum())}
    return {"summary": summary, "meters": rows, "skipped": skipped, "profiles_weekday_norm": profiles, "_keep": keep}


def history_contexts(s: pd.Series, days: pd.DatetimeIndex, context_h: int = 512, min_h: int = 168):
    """Context windows for a day-ahead forecast issued at 00:00 of each day, built from history ONLY.

    Gaps are filled inside the history window (linear between two past values, then carried forward from the last past
    value), so no value at or after the issue time can leak into the context. Returns (issue_days, [np.ndarray]).
    Pure pandas/numpy so it runs in the runtime env; the Chronos call itself lives in scripts/train_energy_models.py.
    """
    s = s.astype(float)
    starts, ctxs = [], []
    for d in days:
        pos = int(s.index.searchsorted(d, side="left"))  # rows strictly before the issue time
        hist = s.iloc[max(0, pos - (context_h + 48)):pos]  # small margin so gaps near the window start have a left anchor
        hist =hist.interpolate(limit_area="inside").ffill().bfill().dropna()
        arr = hist.to_numpy()[-context_h:]
        if len(arr) < min_h:
            continue
        starts.append(d)
        ctxs.append(arr)
    return starts, ctxs
