"""Rolling-origin day-ahead backtest: daily origins at 00:00, 24 h horizon, no fitting on the test window.

Long frame layout (one row per meter, origin and forecast hour): meter, origin, ts, h, y (the RAW target, NaN when
the meter did not report), then per model `<model>_q50` and optionally `<model>_q10` / `<model>_q90`.
Only hours with a real target and a prediction from every compared model are scored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from . import metrics as M
from .base import causal_fill, hourly
from .tabular import HGBForecaster


@dataclass(frozen=True)
class BacktestConfig:
    test_start: str = "2017-11-01"
    last_origin: str = "2017-12-24"
    horizon: int = 24
    context: int = 1024
    hgb_train_end: str = "2017-10-01"  # HGB trains before this ...
    hgb_calib_end: str = "2017-11-01"  # ... and calibrates its band on October 2017
    mase_period: int = 168
    alpha: float = 0.2

    def origins(self) -> pd.DatetimeIndex:
        return pd.date_range(self.test_start, self.last_origin, freq="D")

    def to_dict(self) -> dict:
        return asdict(self)


def target_index(origins: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    rows = [(o, o + pd.Timedelta(hours=h - 1), h) for o in origins for h in range(1, horizon + 1)]
    return pd.DataFrame(rows, columns=["origin", "ts", "h"])


def baseline_frame(y_raw: pd.Series, meter: str, cfg: BacktestConfig) -> pd.DataFrame:
    """Seasonal naive (lag 168) and naive (lag 24) from the causally filled series; target stays raw."""
    y_raw = hourly(y_raw)
    ctx = causal_fill(y_raw)
    t = target_index(cfg.origins(), cfg.horizon)
    t.insert(0, "meter", meter)
    t["y"] = y_raw.reindex(t.ts).to_numpy()
    # lag 168 is always >= 144 h before the origin and lag 24 >= 1 h before it for h <= 24
    t["snaive168_q50"] = ctx.shift(168).reindex(t.ts).to_numpy()
    t["naive24_q50"] = ctx.shift(24).reindex(t.ts).to_numpy()
    return t


def hgb_frame(y_raw: pd.Series, meter: str, cfg: BacktestConfig, max_iter: int = 300,
              point_only_refit: bool = False) -> tuple:
    """HGB + split-conformal band from ONE model (train < hgb_train_end, calibrate until hgb_calib_end).
    With point_only_refit, a second model trained through the day before test_start is added as `hgb_oct`
    (point forecast only, no band)."""
    y_raw = hourly(y_raw)
    t = target_index(cfg.origins(), cfg.horizon)
    idx = pd.DatetimeIndex(t.ts)
    m = HGBForecaster(max_iter=max_iter, alpha=cfg.alpha).fit(y_raw, cfg.hgb_train_end, cfg.hgb_calib_end)
    f = m.predict_at(y_raw, idx)
    out = pd.DataFrame({"meter": meter, "ts": idx, "hgb_q10": f.q10.to_numpy(), "hgb_q50": f.q50.to_numpy(),
                        "hgb_q90": f.q90.to_numpy()})
    models = {"hgb": m}
    if point_only_refit:
        m2 = HGBForecaster(max_iter=max_iter, alpha=cfg.alpha).fit(y_raw, cfg.test_start, cfg.test_start)
        out["hgb_oct_q50"] = m2.predict_at(y_raw, idx).q50.to_numpy()
        models["hgb_oct"] = m2
    return out, models


def chronos_contexts(y_raw: pd.Series, cfg: BacktestConfig) -> Dict[pd.Timestamp, np.ndarray]:
    """Context windows for the zero-shot models: the last `context` hours before each origin, causally filled,
    with NaN left in place for longer gaps (Chronos accepts NaN in the context)."""
    ctx = causal_fill(hourly(y_raw))
    out = {}
    for o in cfg.origins():
        w = ctx[ctx.index < o].to_numpy()[-cfg.context:]
        out[o] = w.astype("float32")
    return out


def _model_names(frame: pd.DataFrame) -> List[str]:
    return [c[:-4] for c in frame.columns if c.endswith("_q50")]


def score(frame: pd.DataFrame, scales: Dict[str, float], baseline: str = "snaive168",
          models: Optional[Iterable[str]] = None, n_boot: int = 1000, seed: int = 0,
          primary: Optional[str] = None, site_of: Optional[Dict[str, str]] = None) -> dict:
    """Metrics per model on the common scored hours, per-meter MASE, wins vs the baseline, a meter-level
    bootstrap 95% interval for WAPE, PAIRED meter-bootstrap intervals for WAPE differences (every model minus the
    baseline, and the primary model minus every other model; the same resampled meters for both sides of a
    difference), and WAPE per site when site_of is given. Meter resampling ignores site clustering."""
    models = list(models or _model_names(frame))
    need = ["y"] + [f"{m}_q50" for m in models]
    ok = frame[need].notna().all(axis=1)
    f = frame[ok]
    y = f.y.to_numpy()
    meters = sorted(f.meter.unique())
    per_meter: Dict[str, Dict[str, float]] = {m: {} for m in models}
    abs_err = {m: (f[f"{m}_q50"] - f.y).abs() for m in models}
    for m in models:
        g = abs_err[m].groupby(f.meter).mean()
        per_meter[m] = {b: float(g[b] / scales[b]) for b in meters}
    # meter-level bootstrap for WAPE
    rng = np.random.default_rng(seed)
    num = {m: abs_err[m].groupby(f.meter).sum().reindex(meters).to_numpy() for m in models}
    den = f.y.abs().groupby(f.meter).sum().reindex(meters).to_numpy()
    boots = rng.integers(0, len(meters), size=(n_boot, len(meters)))
    out: Dict[str, dict] = {}
    for m in models:
        p = f[f"{m}_q50"].to_numpy()
        w = 100.0 * num[m][boots].sum(axis=1) / den[boots].sum(axis=1)
        d = {"MAE_kWh": M.mae(y, p), "WAPE_pct": M.wape(y, p),
             "WAPE_pct_ci95": [float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))],
             "CVRMSE_pct": M.cvrmse(y, p), "NMBE_pct": M.nmbe(y, p),
             "MASE_mean_over_meters": float(np.mean(list(per_meter[m].values()))),
             "pinball_q50": M.pinball(y, p, 0.5)}
        if m != baseline:
            d["meters_better_than_baseline"] = int(sum(per_meter[m][b] < per_meter[baseline][b] for b in meters))
            d["meters_total"] = len(meters)
        lo_c, hi_c = f"{m}_q10", f"{m}_q90"
        if lo_c in f and hi_c in f and f[lo_c].notna().all() and f[hi_c].notna().all():
            lo, hi = f[lo_c].to_numpy(), f[hi_c].to_numpy()
            d.update({"coverage_80": M.coverage(y, lo, hi), "mean_band_width_kWh": M.mean_width(lo, hi),
                      "pinball_q10": M.pinball(y, lo, 0.1), "pinball_q90": M.pinball(y, hi, 0.9),
                      "mean_pinball_q10_q50_q90": float(np.mean([M.pinball(y, lo, 0.1), M.pinball(y, p, 0.5),
                                                                 M.pinball(y, hi, 0.9)]))})
        else:
            d["coverage_80"] = None
        out[m] = d

    # paired differences: WAPE(a) - WAPE(b) on the SAME bootstrap draw of meters (percentage points)
    wb = {m: 100.0 * num[m][boots].sum(axis=1) / den[boots].sum(axis=1) for m in models}
    pairs = [(m, baseline) for m in models if m != baseline]
    if primary in models:
        pairs += [(primary, m) for m in models if m not in (primary, baseline)]
    paired = []
    for a, b in pairs:
        diff = wb[a] - wb[b]
        paired.append({"a": a, "b": b, "diff_pp": float(out[a]["WAPE_pct"] - out[b]["WAPE_pct"]),
                       "ci95_pp": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                       "share_of_resamples_a_lower": float((diff < 0).mean())})
    res = {"models": out, "per_meter_mase": per_meter, "n_scored_hours": int(ok.sum()),
           "n_dropped_hours": int((~ok).sum()), "n_meters": len(meters), "baseline": baseline,
           "paired_wape_diff": {"method": f"paired meter bootstrap, {n_boot} resamples, seed {seed}; diff = WAPE(a) - "
                                          "WAPE(b) in percentage points; negative = a has lower error. Meters are "
                                          "resampled independently, so clustering of meters within sites is ignored.",
                                "primary": primary, "pairs": paired}}
    if site_of:
        site = f.meter.map(site_of)
        den_s = f.y.abs().groupby(site).sum()
        res["per_site_WAPE_pct"] = {m: {k: float(100.0 * v / den_s[k]) for k, v in abs_err[m].groupby(site).sum().items()}
                                    for m in models}
        res["meters_per_site"] = {k: int(v) for k, v in pd.Series(site_of).reindex(meters).value_counts().sort_index().items()}
    return res
