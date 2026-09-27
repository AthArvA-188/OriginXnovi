"""Per-facade storm wetting model: predicts the ASOS-measured ISO 15927-3 storm index on 8 facades from gridded or
forecast weather (ERA5 reanalysis via Open-Meteo, or GFS day-1 forecasts from the Open-Meteo Previous Runs API).

Events are segmented on the GRIDDED/FORECAST precipitation (the same way in training and at inference), so false
alarms count: the target is the ASOS ISO index summed over the event window +/- EVENT_PAD_H hours, whatever the
station measured. Model: one sklearn HistGradientBoostingRegressor per facade on log1p(target). Baselines:
  B0 station climatology (mean training target per facade), B1 ISO physics on the gridded 10 m wind (no learning),
  B2 per-station, per-facade least-squares coefficient x gridded event precipitation (pooled for unseen stations).
Metrics: MAE per facade-event (L/m2), MAE on each event's measured most-exposed facade, top-1 and within +/-45 deg
accuracy of the most-exposed facade, median per-event Spearman across facades, and a storm-cluster bootstrap CI of the
MAE difference (clusters = event starts within 36 h across stations).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .rainexposure import AZ, FACADES, iso_frame, iso_hourly, rule, segment_events

HGB_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=1.0,
                  random_state=0)
BASE_FEATS_10M = ["p_tot", "p_max", "wet_hours", "span_h", "u10", "v10", "s10", "msin", "mcos", "lat", "lon", "elev"]
FEATS_10M = BASE_FEATS_10M + [f"b1_{f}" for f in FACADES]
FEATS_FULL = FEATS_10M + ["u100", "v100", "s100"] + [f"b1h_{f}" for f in FACADES]
YCOLS = [f"y_{f}" for f in FACADES]
B1COLS = [f"b1_{f}" for f in FACADES]


# ----------------------------------------------------------------------------------------- inputs

def open_meteo_frame(j: dict, suffix: str = "") -> pd.DataFrame:
    """Open-Meteo hourly JSON -> frame indexed by UTC hour (precipitation is the sum over the preceding hour).
    Columns: precip, ws10, wd10 and, when present, ws100, wd100. suffix handles '_previous_day1' variables."""
    h = j["hourly"]
    idx = pd.to_datetime(h["time"], utc=True)
    out = pd.DataFrame(index=idx)
    names = {"precip": "precipitation", "ws10": "wind_speed_10m", "wd10": "wind_direction_10m",
             "ws100": "wind_speed_100m", "wd100": "wind_direction_100m"}
    for k, v in names.items():
        key = v + suffix
        if key in h:
            out[k] = pd.to_numeric(pd.Series(h[key], index=idx), errors="coerce")
    return out


def _wmean_vec(ws: np.ndarray, wd: np.ndarray, w: np.ndarray) -> Tuple[float, float, float]:
    """Precipitation-weighted from-vector components (east, north) and mean speed."""
    ok = np.isfinite(ws) & np.isfinite(wd)
    if not ok.any():
        return float("nan"), float("nan"), float("nan")
    ww = w[ok] if w[ok].sum() > 0 else np.ones(ok.sum())
    a = np.deg2rad(wd[ok])
    u = float((ww * ws[ok] * np.sin(a)).sum() / ww.sum())
    v = float((ww * ws[ok] * np.cos(a)).sum() / ww.sum())
    s = float((ww * ws[ok]).sum() / ww.sum())
    return u, v, s


def event_features(src: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, lat: float, lon: float, elev: float,
                   pad_h: int) -> Optional[dict]:
    """Features of one event from gridded/forecast hourly data (precip, ws10, wd10[, ws100, wd100])."""
    t0, t1 = start - pd.Timedelta(hours=pad_h), end + pd.Timedelta(hours=pad_h)
    w = src.loc[(src.index >= t0) & (src.index <= t1)]
    if len(w) == 0 or w["precip"].isna().all():
        return None
    P = w["precip"].fillna(0.0).to_numpy()
    ws10, wd10 = w["ws10"].to_numpy(dtype=float), w["wd10"].to_numpy(dtype=float)
    u10, v10, s10 = _wmean_vec(ws10, wd10, P)
    b1 = iso_hourly(ws10, P, wd10, [AZ[f] for f in FACADES]).sum(0)
    m = start.month
    row = {"p_tot": float(P.sum()), "p_max": float(P.max()), "wet_hours": int((P >= rule("WET_HOUR_MM")).sum()),
           "span_h": float((end - start) / pd.Timedelta(hours=1) + 1), "u10": u10, "v10": v10, "s10": s10,
           "msin": math.sin(2 * math.pi * m / 12), "mcos": math.cos(2 * math.pi * m / 12),
           "lat": float(lat), "lon": float(lon), "elev": float(elev),
           **{f"b1_{f}": float(b1[i]) for i, f in enumerate(FACADES)}}
    if "ws100" in w.columns and "wd100" in w.columns:
        ws100, wd100 = w["ws100"].to_numpy(dtype=float), w["wd100"].to_numpy(dtype=float)
        u100, v100, s100 = _wmean_vec(ws100, wd100, P)
        b1h = iso_hourly(ws100, P, wd100, [AZ[f] for f in FACADES]).sum(0)
        row.update({"u100": u100, "v100": v100, "s100": s100, **{f"b1h_{f}": float(b1h[i]) for i, f in enumerate(FACADES)}})
    return row


def build_event_table(station: str, src: pd.DataFrame, asos: Optional[pd.DataFrame], lat: float, lon: float,
                      elev: float, passed_wy: Optional[Sequence[int]] = None, gap_h: float = 6.0, min_mm: float = 5.0,
                      pad_h: Optional[int] = None, min_coverage: float = 0.8, source: str = "era5") -> Tuple[pd.DataFrame, dict]:
    """Events segmented on src precipitation; features from src; targets = ASOS ISO index per facade summed over
    the padded window. Events are dropped when the ASOS window coverage (hours with a rain value) is below
    min_coverage or their water year failed QC. Returns (table, counts)."""
    pad = int(rule("EVENT_PAD_H") if pad_h is None else pad_h)
    ev = segment_events(src["precip"], gap_h=gap_h, min_mm=min_mm, wet_mm=float(rule("WET_HOUR_MM")))
    counts = {"events_detected": int(len(ev)), "dropped_coverage": 0, "dropped_qc_wy": 0, "dropped_no_features": 0}
    rows: List[dict] = []
    if asos is not None:
        full = pd.date_range(asos.index.min(), asos.index.max(), freq="h")
        a = asos.reindex(full)
        iso = iso_frame(a.fillna({"r": 0.0}))
        cs_iso = np.vstack([np.zeros((1, len(FACADES))), np.cumsum(iso.to_numpy(), axis=0)])
        rr = a["r"].to_numpy()
        cs_r = np.concatenate([[0.0], np.cumsum(np.nan_to_num(rr))])
        cs_ok = np.concatenate([[0], np.cumsum(np.isfinite(rr))])
    for _, e in ev.iterrows():
        start, end = e["start"], e["end"]
        wy = int(start.year + (start.month >= 10))
        feat = event_features(src, start, end, lat, lon, elev, pad)
        if feat is None:
            counts["dropped_no_features"] += 1
            continue
        row = {"station": station, "source": source, "start": start, "end": end, "wy": wy, **feat}
        if asos is not None:
            if passed_wy is not None and wy not in set(passed_wy):
                counts["dropped_qc_wy"] += 1
                continue
            t0 = start - pd.Timedelta(hours=pad)
            t1 = end + pd.Timedelta(hours=pad)
            i0 = int(full.searchsorted(t0, side="left"))
            i1 = int(full.searchsorted(t1, side="right"))
            n_exp = int(round((t1 - t0) / pd.Timedelta(hours=1))) + 1
            cov = (cs_ok[i1] - cs_ok[i0]) / n_exp if n_exp > 0 else 0.0
            if cov < min_coverage:
                counts["dropped_coverage"] += 1
                continue
            y = cs_iso[i1] - cs_iso[i0]
            row.update({"asos_mm": float(cs_r[i1] - cs_r[i0]), "asos_coverage": float(cov),
                        **{f"y_{f}": float(y[k]) for k, f in enumerate(FACADES)}})
        rows.append(row)
    df = pd.DataFrame(rows)
    counts["events_used"] = int(len(df))
    return df, counts


# ----------------------------------------------------------------------------------------- models and baselines

class FacadeModel:
    """Eight HistGradientBoostingRegressor models (one per facade), trained on log1p(target)."""

    def __init__(self, features: Sequence[str], params: Optional[dict] = None):
        self.features = list(features)
        self.params = dict(HGB_PARAMS if params is None else params)
        self.models: Dict[str, object] = {}

    def fit(self, df: pd.DataFrame) -> "FacadeModel":
        from sklearn.ensemble import HistGradientBoostingRegressor
        X = df[self.features].to_numpy(dtype=float)
        for f in FACADES:
            m = HistGradientBoostingRegressor(**self.params)
            m.fit(X, np.log1p(df[f"y_{f}"].to_numpy(dtype=float).clip(0)))
            self.models[f] = m
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].to_numpy(dtype=float)
        return np.column_stack([np.expm1(self.models[f].predict(X)).clip(0) for f in FACADES])


def baseline_b0(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    means = train.groupby("station")[YCOLS].mean()
    pooled = train[YCOLS].mean()
    return np.vstack([means.loc[s].to_numpy() if s in means.index else pooled.to_numpy() for s in test["station"]])


def baseline_b1(test: pd.DataFrame) -> np.ndarray:
    return test[B1COLS].to_numpy(dtype=float)


def baseline_b2(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    P = np.zeros((len(test), len(FACADES)))
    x_all = train["p_tot"].to_numpy(dtype=float)
    for j, f in enumerate(FACADES):
        y_all = train[f"y_{f}"].to_numpy(dtype=float)
        pooled = float((x_all * y_all).sum() / max((x_all ** 2).sum(), 1e-9))
        ks = {s: float((g["p_tot"] * g[f"y_{f}"]).sum() / max((g["p_tot"] ** 2).sum(), 1e-9)) for s, g in train.groupby("station")}
        P[:, j] = [ks.get(s, pooled) * x for s, x in zip(test["station"], test["p_tot"])]
    return P


def predict_all(train: pd.DataFrame, test: pd.DataFrame, features: Sequence[str]) -> Dict[str, np.ndarray]:
    return {"B0_station_climatology": baseline_b0(train, test), "B1_iso_physics_on_grid": baseline_b1(test),
            "B2_station_coef_x_grid_rain": baseline_b2(train, test),
            "HGB_model": FacadeModel(features).fit(train).predict(test)}


# ----------------------------------------------------------------------------------------- metrics

def metrics(Y: np.ndarray, P: np.ndarray) -> dict:
    """Y, P: (n_events, 8) in L/m2. Direction metrics use events whose measured max > 0."""
    from scipy.stats import spearmanr
    Y, P = np.asarray(Y, dtype=float), np.asarray(P, dtype=float)
    n = len(Y)
    out = {"n_events": int(n), "MAE_L_m2": float(np.mean(np.abs(Y - P))) if n else float("nan"),
           "bias_L_m2": float(np.mean(P - Y)) if n else float("nan"),
           "mean_target_L_m2": float(Y.mean()) if n else float("nan")}
    s = Y.max(1) > 0 if n else np.zeros(0, bool)
    out["n_scored"] = int(s.sum())
    if s.sum() == 0:
        out.update({"peak_MAE_L_m2": float("nan"), "top1_acc": float("nan"), "within45_acc": float("nan"),
                    "median_spearman": float("nan")})
        return out
    Ys, Ps = Y[s], P[s]
    iy, ip = Ys.argmax(1), Ps.argmax(1)
    r = np.arange(len(Ys))
    out["peak_MAE_L_m2"] = float(np.mean(np.abs(Ys[r, iy] - Ps[r, iy])))
    out["mean_peak_target_L_m2"] = float(Ys[r, iy].mean())
    out["top1_acc"] = float(np.mean(iy == ip))
    out["within45_acc"] = float(np.mean(np.abs(((iy - ip) + 4) % 8 - 4) <= 1))
    rhos = [spearmanr(Ys[i], Ps[i]).statistic for i in range(len(Ys)) if np.std(Ys[i]) > 0 and np.std(Ps[i]) > 0]
    out["median_spearman"] = float(np.nanmedian(rhos)) if rhos else float("nan")
    out["n_spearman"] = int(len(rhos))
    return out


def clusters_by_start(starts: Sequence, gap_h: float = 36.0) -> np.ndarray:
    """Cluster ids for events whose starts (across stations) fall within gap_h hours of the previous one."""
    t = pd.to_datetime(pd.Series(list(starts)), utc=True)
    order = np.argsort(t.to_numpy())
    ts = t.to_numpy()[order]
    d = np.diff(ts).astype("timedelta64[s]").astype(float) / 3600.0
    cl_sorted = np.concatenate([[0], np.cumsum(d > gap_h)])
    out = np.empty(len(t), dtype=int)
    out[order] = cl_sorted
    return out


def cluster_bootstrap_diff(err_base: np.ndarray, err_model: np.ndarray, clusters: np.ndarray, n_boot: int = 2000,
                           seed: int = 0) -> dict:
    """MAE(base) - MAE(model) with a 95 % CI from resampling whole storm clusters. Positive = model better."""
    eb, em, cl = np.asarray(err_base, float), np.asarray(err_model, float), np.asarray(clusters)
    u, inv = np.unique(cl, return_inverse=True)
    K = len(u)
    sb = np.bincount(inv, weights=eb, minlength=K)
    sm = np.bincount(inv, weights=em, minlength=K)
    cnt = np.bincount(inv, minlength=K).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, K, size=(n_boot, K))
    diffs = (sb[idx].sum(1) - sm[idx].sum(1)) / cnt[idx].sum(1)
    return {"diff": float(eb.mean() - em.mean()), "lo95": float(np.percentile(diffs, 2.5)),
            "hi95": float(np.percentile(diffs, 97.5)), "n_clusters": int(K), "n_boot": int(n_boot)}


def score(Y: np.ndarray, preds: Dict[str, np.ndarray], starts: Sequence, n_boot: int = 2000) -> dict:
    """Metrics for every method and the cluster bootstrap of each baseline against HGB_model."""
    cl = clusters_by_start(starts)
    res = {"methods": {k: metrics(Y, P) for k, P in preds.items()}, "n_events": int(len(Y)),
           "n_clusters": int(len(np.unique(cl)))}
    if "HGB_model" in preds:
        em = np.abs(Y - preds["HGB_model"]).mean(1)
        res["bootstrap_mae_diff_vs_hgb"] = {k: cluster_bootstrap_diff(np.abs(Y - P).mean(1), em, cl, n_boot)
                                            for k, P in preds.items() if k != "HGB_model"}
    return res


def evaluate_split(train: pd.DataFrame, test: pd.DataFrame, features: Sequence[str], n_boot: int = 2000) -> dict:
    preds = predict_all(train, test, features)
    out = score(test[YCOLS].to_numpy(dtype=float), preds, test["start"], n_boot)
    out["n_train"] = int(len(train))
    out["n_test"] = int(len(test))
    return out


def evaluate_grouped(df: pd.DataFrame, features: Sequence[str], folds: Sequence[Tuple[np.ndarray, np.ndarray]],
                     n_boot: int = 2000) -> Tuple[dict, pd.DataFrame]:
    """Pool out-of-fold predictions over folds (train_mask, test_mask) and score once. Returns (scores, per-row
    predictions with station and fold)."""
    Ys, parts, starts, meta = [], {}, [], []
    for k, (trm, tem) in enumerate(folds):
        tr, te = df[trm], df[tem]
        if len(te) == 0 or len(tr) == 0:
            continue
        p = predict_all(tr, te, features)
        for name, P in p.items():
            parts.setdefault(name, []).append(P)
        Ys.append(te[YCOLS].to_numpy(dtype=float))
        starts.extend(list(te["start"]))
        meta.append(pd.DataFrame({"station": te["station"].to_numpy(), "fold": k}))
    Y = np.vstack(Ys)
    preds = {k: np.vstack(v) for k, v in parts.items()}
    res = score(Y, preds, starts, n_boot)
    m = pd.concat(meta, ignore_index=True)
    per = {}
    for s in m["station"].unique():
        ix = (m["station"] == s).to_numpy()
        per[s] = {k: metrics(Y[ix], P[ix]) for k, P in preds.items()}
    res["per_station"] = per
    return res, m
