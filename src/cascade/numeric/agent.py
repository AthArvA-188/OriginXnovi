"""NumericAgent: one interface (fit / forecast / anomaly_score) over several backends, each result with provenance.

Backends
- "seasonal_naive": always available; the baseline every other model is shown against.
- "hgb": scikit-learn HistGradientBoosting + split-conformal 80% band. Runs in the app process (no torch).
- "chronos-2", "chronos-bolt-small": Hugging Face models run OFFLINE in a separate env
  (scripts/numeric_chronos_precompute.py); the agent only reads their precomputed forecasts, and only when the
  provenance file shows the pinned revision. Otherwise it falls back to "hgb", then to "seasonal_naive".

Outputs are indicators. to_observations() emits building Observations with level None and the provenance in the
text; nothing here switches equipment or touches a life-safety system. A person reviews and approves any action.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import ADVISORY, EVAL_DIR, MODELS_DIR, AnomalyResult, DataCard, ForecastResult, Provenance, hourly
from .baselines import RobustZ, SeasonalNaive


@dataclass(frozen=True)
class HFModel:
    model_id: str
    revision: str
    licence: str
    params: int
    weights_bytes: int
    tag: str  # file tag: forecasts_<tag>.parquet


# Revisions and sizes from the Hugging Face API (accessed 2026-09-26); see eval/numeric/model_catalogue.json.
HF_MODELS: Dict[str, HFModel] = {
    "chronos-2": HFModel("amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c", "Apache-2.0",
                         119_477_664, 477_930_472, "chronos_2"),
    "chronos-bolt-small": HFModel("amazon/chronos-bolt-small", "772f3d25d38aec6d914c8949dab4462e2d46f5d8",
                                  "Apache-2.0", 47_718_016, 190_888_824, "chronos_bolt_small"),
}
PREFERENCE = ("chronos-2", "chronos-bolt-small", "hgb", "seasonal_naive")


# Where the robust-z threshold comes from, stated wherever the threshold is reported.
THRESHOLD_SOURCE = ("K = cascade.building.rules ANOMALY_MAD_K (team-proposed, validate), applied to 1.4826 x MAD as in "
                    "electrical.baseline_anomalies. Same K and scale; different baseline (trailing 4-week same-hour "
                    "median and trailing 28-day MAD here, a fixed first-28-day hour-of-week median and pooled MAD there).")


def _anomaly_threshold() -> float:
    """K from the building layer (rules.ANOMALY_MAD_K). RobustZ applies it to 1.4826 x MAD, the same scale as
    electrical.baseline_anomalies; the baselines themselves differ (see THRESHOLD_SOURCE)."""
    try:
        from ..building import rules

        return float(rules.t("ANOMALY_MAD_K"))
    except Exception:  # the building layer is optional for this package
        return 5.0


class NumericAgent:
    def __init__(self, artifacts_dir: Path = EVAL_DIR, models_dir: Path = MODELS_DIR,
                 data: Optional[DataCard] = None):
        self.artifacts_dir = Path(artifacts_dir)
        self.models_dir = Path(models_dir)
        self.data = data or DataCard("unspecified", "SYNTHETIC")
        self._hgb: Dict[str, object] = {}
        self._pre: Dict[str, pd.DataFrame] = {}
        self._bundle: Optional[Dict[str, object]] = None

    # ---- precomputed Hugging Face forecasts -------------------------------------------------------------------
    def provenance_file(self, backend: str) -> Path:
        return self.artifacts_dir / f"forecasts_{HF_MODELS[backend].tag}.provenance.json"

    def precomputed_ok(self, backend: str) -> bool:
        """True only when the parquet exists and its provenance names the pinned model id and revision."""
        spec = HF_MODELS[backend]
        pq, pv = self.artifacts_dir / f"forecasts_{spec.tag}.parquet", self.provenance_file(backend)
        if not (pq.exists() and pv.exists()):
            return False
        try:
            meta = json.loads(pv.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return meta.get("model_id") == spec.model_id and meta.get("revision") == spec.revision

    def _precomputed(self, backend: str) -> pd.DataFrame:
        if backend not in self._pre:
            spec = HF_MODELS[backend]
            df = pd.read_parquet(self.artifacts_dir / f"forecasts_{spec.tag}.parquet")
            df["origin"] = pd.to_datetime(df["origin"])
            df["ts"] = pd.to_datetime(df["ts"])
            self._pre[backend] = df
        return self._pre[backend]

    def hf_provenance(self, backend: str) -> Provenance:
        spec = HF_MODELS[backend]
        meta = json.loads(self.provenance_file(backend).read_text(encoding="utf-8"))
        return Provenance(spec.model_id, "hf-precomputed", spec.revision, spec.licence, self.data,
                          {k: meta.get(k) for k in ("context", "horizon", "quantiles", "device", "created_utc",
                                                    "library_versions")})

    # ---- selection ---------------------------------------------------------------------------------------------
    def available(self) -> List[str]:
        out = [b for b in HF_MODELS if self.precomputed_ok(b)]
        out.append("hgb")
        out.append("seasonal_naive")
        return out

    def select(self, task: str = "forecast", series_id: Optional[str] = None) -> str:
        if task == "anomaly":
            return "robust_z"
        key = series_id or "series"
        for b in PREFERENCE:
            if b in HF_MODELS:  # precomputed forecasts are looked up by series id, so one is required
                if (series_id is not None and self.precomputed_ok(b)
                        and series_id in set(self._precomputed(b)["meter"].unique())):
                    return b
            elif b == "hgb":
                if key in self._hgb or self._hgb_bundle_has(key):
                    return b
            else:
                return b
        return "seasonal_naive"

    # ---- hgb models -------------------------------------------------------------------------------------------
    def _hgb_bundle(self) -> Dict[str, object]:
        """Stored per-meter HGB models (written by scripts/numeric_backtest.py), loaded once."""
        if self._bundle is None:
            p = self.models_dir / "hgb_bdg2.joblib"
            if not p.exists():
                return {}
            import joblib

            self._bundle = joblib.load(p)
        return self._bundle

    def _hgb_bundle_has(self, series_id: Optional[str]) -> bool:
        return series_id is not None and series_id in self._hgb_bundle().get("models", {})

    def load_hgb(self, series_id: str):
        if series_id not in self._hgb:
            models = self._hgb_bundle().get("models", {})
            if series_id not in models:
                raise KeyError(f"no stored HGB model for {series_id}")
            self._hgb[series_id] = models[series_id]
        return self._hgb[series_id]

    # ---- the interface ----------------------------------------------------------------------------------------
    def fit(self, y: pd.Series, backend: str = "hgb", series_id: str = "series", train_end=None, calib_end=None,
            max_iter: int = 300) -> Provenance:
        if backend == "seasonal_naive":
            return SeasonalNaive(data=self.data).forecast(y, 24).provenance
        if backend != "hgb":
            raise ValueError(f"{backend} is precomputed offline; it is not fitted in the app process")
        from .tabular import HGBForecaster

        m = HGBForecaster(max_iter=max_iter, data=self.data).fit(y, train_end, calib_end)
        self._hgb[series_id] = m
        return m.provenance()

    def forecast(self, context: pd.Series, horizon: int = 24, backend: Optional[str] = None,
                 series_id: Optional[str] = None) -> ForecastResult:
        """Forecast the `horizon` hours after the last context timestamp. backend None = select()."""
        backend = backend or self.select("forecast", series_id)
        ctx = hourly(context)
        origin = ctx.index[-1] + pd.Timedelta(hours=1)
        if backend in HF_MODELS:
            if not self.precomputed_ok(backend):
                key = series_id or "series"
                nxt = "hgb" if key in self._hgb or self._hgb_bundle_has(key) else "seasonal_naive"
                return self.forecast(context, horizon, nxt, series_id)
            df = self._precomputed(backend)
            rows = df[(df.meter == series_id) & (df.origin == origin)].sort_values("h").head(horizon)
            if len(rows) < horizon:
                raise KeyError(f"no precomputed {backend} forecast for {series_id} at {origin}")
            return ForecastResult(pd.DatetimeIndex(rows.ts), rows.q10.to_numpy(float), rows.q50.to_numpy(float),
                                  rows.q90.to_numpy(float), self.hf_provenance(backend))
        if backend == "hgb":
            key = series_id or "series"
            m = self._hgb.get(key) or self.load_hgb(key)
            return m.forecast(ctx, horizon)
        if backend == "seasonal_naive":
            return SeasonalNaive(data=self.data).forecast(ctx, horizon)
        raise ValueError(f"unknown backend {backend}")

    def anomaly_score(self, y: pd.Series, method: str = "robust_z", model=None) -> AnomalyResult:
        if method == "robust_z":
            return RobustZ(threshold=_anomaly_threshold(), data=self.data).anomaly_score(y)
        if method == "isolation_forest":
            from .tabular import IsoForestScorer

            return (model or IsoForestScorer(data=self.data)).anomaly_score(y)
        if method == "hgb_supervised":
            if model is None:
                raise ValueError("hgb_supervised needs a classifier fitted on labelled series")
            return model.anomaly_score(y)
        raise ValueError(f"unknown anomaly method {method}")

    # ---- building-layer hand-off ------------------------------------------------------------------------------
    @staticmethod
    def to_observations(result: AnomalyResult, *, source: str, zone_id: Optional[str] = None,
                        element_id: Optional[str] = None, kind: str = "load_event", unit: Optional[str] = None,
                        max_obs: int = 50) -> list:
        """One indicator Observation per flagged hour (highest scores first, at most max_obs). level is None."""
        from ..building.model import Observation

        p = result.provenance
        synthetic = p.data.label != "REAL"
        idx = np.flatnonzero(result.flags)
        order = idx[np.argsort(-np.asarray(result.scores)[idx])][:max_obs]
        out = []
        for i in sorted(order):
            ts = pd.Timestamp(result.index[i])
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            unit = p.params.get("threshold_unit")
            thr = f"{result.threshold:g}" + (f" ({unit})" if unit else "")
            text = (f"{result.method} score {float(result.scores[i]):.2f} >= {thr} | model {p.model_id}"
                    f" rev {p.revision or 'n/a'} | licence {p.licence} | data {p.data.label}: {p.data.name} | "
                    f"{ADVISORY}")
            out.append(Observation(obs_id=f"NUM-{source}-{ts:%Y%m%dT%H}", kind=kind, ts=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                   zone_id=zone_id, element_id=element_id, source=f"numeric:{source}", level=None,
                                   value=float(result.scores[i]), unit=unit or "score", text=text,
                                   synthetic=synthetic))
        return out
