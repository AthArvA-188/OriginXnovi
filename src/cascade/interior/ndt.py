"""Concrete compressive strength from rebound hammer (RN) and ultrasonic pulse velocity (Vp).

An ESTIMATE for an engineer, never a structural verdict. The model is the SonReb power law
    ln fc = a + b ln RN + c ln Vp
fitted by least squares on the REAL Matthews et al. (2025) NDT databases (Zenodo 15392443,
CC BY 4.0). Single-instrument fallbacks: ln fc = a + b ln RN (rebound database) and
ln fc = a + c ln Vp (UPV database). Per-building calibration shifts the prediction by the mean
log residual of k cores taken at tested locations. Intervals are split-conformal quantiles of
absolute log residuals measured on held-out STUDIES (never on the rows being scored); with k
cores a separate quantile table (q_by_k) applies.

Numpy + pandas only at runtime; the fitted coefficients and quantile tables live in
models/interior/ndt_strength_v1.json (written by scripts/interior/ndt_strength.py).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "models" / "interior" / "ndt_strength_v1.json"

# Cleaning rules (build spec step 8): physically plausible ranges, applied before any fit.
VP_MIN_MS = 500.0
RN_MIN = 5.0
FC_MIN_MPA = 0.0
COVERAGE = 0.90

COL = {
    "study": "Literature",
    "rn": "Median Rebound Number, RN",
    "vp": "Average Velocity, Vp (m/s)",
    "fc": "fc,cyl (MPa)",
    "age": "Specimen Age (days)",
    "type": "NDT Specimen Type",
    "test": "Test Type",
    "comp": "Compression Specimen",
    "h": "Height (mm)",
    "d": "Width/Diameter (mm)",
    "fc_core": "fc,core (MPa)",
    "design": "28-day Design Strength (MPa)",
}


def _clean(df: pd.DataFrame, need: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame({"study": df[COL["study"]].astype(str).str.strip()})
    for k in ("rn", "vp", "fc", "age", "h", "d", "fc_core", "design"):
        if COL[k] in df:
            out[k] = pd.to_numeric(df[COL[k]], errors="coerce")
    for k in ("type", "test", "comp"):
        if COL[k] in df:
            out[k] = df[COL[k]].astype(str).str.strip()
    out["in_situ"] = df[COL["type"]].astype(str).str.contains("In-situ", case=False).values
    out = out.dropna(subset=list(need))
    if "vp" in need:
        out = out[out.vp > VP_MIN_MS]
    if "rn" in need:
        out = out[out.rn > RN_MIN]
    out = out[out.fc > FC_MIN_MPA]
    return out.reset_index(drop=True)


def load_sonreb(path: Path) -> pd.DataFrame:
    return _clean(pd.read_csv(path), ("rn", "vp", "fc"))


def load_rebound(path: Path) -> pd.DataFrame:
    """Rebound-only database. Its 'NDT Specimen Type' labels are inconsistent ('In-situ' vs
    'Element - In-situ'); in_situ is matched on the substring, which covers both spellings."""
    return _clean(pd.read_csv(path), ("rn", "fc"))


def load_upv(path: Path) -> pd.DataFrame:
    return _clean(pd.read_csv(path), ("vp", "fc"))


# --- power law ---------------------------------------------------------------------------------

FEATURES = {"sonreb": ("rn", "vp"), "rn_only": ("rn",), "vp_only": ("vp",)}


@dataclass
class PowerLaw:
    """ln fc = intercept + sum_i coef_i * ln x_i. Fitted with numpy least squares."""

    features: Tuple[str, ...]
    intercept: float = float("nan")
    coef: Tuple[float, ...] = ()

    def fit(self, df: pd.DataFrame) -> "PowerLaw":
        X = np.column_stack([np.ones(len(df))] + [np.log(df[f].to_numpy(float)) for f in self.features])
        beta, *_ = np.linalg.lstsq(X, np.log(df["fc"].to_numpy(float)), rcond=None)
        self.intercept = float(beta[0])
        self.coef = tuple(float(b) for b in beta[1:])
        return self

    def predict_log(self, df_or_dict) -> np.ndarray:
        cols = [np.log(np.asarray(df_or_dict[f], dtype=float)) for f in self.features]
        return self.intercept + sum(c * x for c, x in zip(self.coef, cols))

    def predict(self, df_or_dict) -> np.ndarray:
        return np.exp(self.predict_log(df_or_dict))

    def as_dict(self) -> dict:
        return {"features": list(self.features), "intercept": self.intercept, "coef": list(self.coef),
                "form": "ln fc = intercept + " + " + ".join(f"coef[{i}]*ln({f})" for i, f in enumerate(self.features))}

    @classmethod
    def from_dict(cls, d: dict) -> "PowerLaw":
        return cls(features=tuple(d["features"]), intercept=float(d["intercept"]), coef=tuple(float(c) for c in d["coef"]))


def conformal_q(abs_log_resid: np.ndarray, coverage: float = COVERAGE) -> float:
    """Split-conformal quantile: the ceil((n+1)*coverage)-th smallest absolute residual."""
    r = np.sort(np.asarray(abs_log_resid, dtype=float))
    n = len(r)
    if n == 0:
        return float("nan")
    k = min(n, int(math.ceil((n + 1) * coverage)))
    return float(r[k - 1])


def calibration_shift(log_pred_at_cores: Sequence[float], fc_cores: Sequence[float]) -> float:
    """Mean log residual of k cores: added to ln(prediction) at every other location of the building."""
    lp = np.asarray(log_pred_at_cores, dtype=float)
    fc = np.asarray(fc_cores, dtype=float)
    if len(lp) == 0:
        return 0.0
    if np.any(fc <= 0):
        raise ValueError("core strengths must be positive")
    return float(np.mean(np.log(fc) - lp))


def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    e = p - y
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mape_pct": float(np.mean(np.abs(e) / y) * 100), "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")}


# --- shipped estimator -------------------------------------------------------------------------


@dataclass
class Estimate:
    model: str
    fc_mpa: float
    lo_mpa: float
    hi_mpa: float
    coverage: float
    k_cores: int
    shift_log: float
    notes: List[str] = field(default_factory=list)
    below_design: Optional[bool] = None
    action: str = "calibrate_with_cores"


class StrengthModel:
    """Loads models/interior/ndt_strength_v1.json and turns field readings into an estimate."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.laws = {k: PowerLaw.from_dict(v["law"]) for k, v in spec["models"].items()}

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "StrengthModel":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def pick(self, rn: Optional[float], vp: Optional[float]) -> str:
        if rn is not None and vp is not None:
            return "sonreb"
        if rn is not None:
            return "rn_only"
        if vp is not None:
            return "vp_only"
        raise ValueError("need a rebound number, a pulse velocity, or both")

    def q_for(self, name: str, k: int) -> float:
        m = self.spec["models"][name]
        if k <= 0:
            return float(m["q_log_k0"])
        table = {int(kk): float(v) for kk, v in m["q_log_by_k"].items()}
        kk = max(kk for kk in table if kk <= k) if any(kk <= k for kk in table) else min(table)
        return table[kk]

    def estimate(self, rn: Optional[float] = None, vp: Optional[float] = None,
                 cores: Sequence[Tuple[Optional[float], Optional[float], float]] = (),
                 design_fc_mpa: Optional[float] = None) -> Estimate:
        """rn/vp at the location of interest; cores = [(rn, vp, fc_core_mpa)] measured at tested
        locations of the SAME building (fc as 150x300 cylinder-equivalent MPa)."""
        rn = None if rn is None or (isinstance(rn, float) and math.isnan(rn)) else float(rn)
        vp = None if vp is None or (isinstance(vp, float) and math.isnan(vp)) else float(vp)
        if rn is not None and rn <= RN_MIN:
            raise ValueError(f"rebound number must be > {RN_MIN:g}")
        if vp is not None and vp <= VP_MIN_MS:
            raise ValueError(f"pulse velocity must be > {VP_MIN_MS:g} m/s")
        name = self.pick(rn, vp)
        law = self.laws[name]
        x = {"rn": [rn if rn is not None else 1.0], "vp": [vp if vp is not None else 1.0]}
        lp = float(law.predict_log(x)[0])
        notes = [f"model: {name} power law fitted on the Matthews et al. NDT database (REAL, CC BY 4.0)"]
        usable = []
        for c in cores:
            crn, cvp, cfc = c
            if cfc is None or cfc <= 0:
                continue
            if any(f == "rn" for f in law.features) and crn is None:
                continue
            if any(f == "vp" for f in law.features) and cvp is None:
                continue
            usable.append((float(law.predict_log({"rn": [crn or 1.0], "vp": [cvp or 1.0]})[0]), float(cfc)))
        k = len(usable)
        shift = calibration_shift([u[0] for u in usable], [u[1] for u in usable]) if k else 0.0
        q = self.q_for(name, k)
        fc = math.exp(lp + shift)
        est = Estimate(model=name, fc_mpa=fc, lo_mpa=math.exp(lp + shift - q), hi_mpa=math.exp(lp + shift + q),
                       coverage=float(self.spec.get("coverage", COVERAGE)), k_cores=k, shift_log=shift, notes=notes)
        if k == 0:
            notes.append("no cores: interval comes from leave-study-out residuals and is wide; calibrate with cores before any decision")
        else:
            notes.append(f"calibrated with {k} core(s): prediction multiplied by {math.exp(shift):.3f}")
        if design_fc_mpa is not None and design_fc_mpa > 0:
            est.below_design = est.hi_mpa < design_fc_mpa
            if est.below_design:
                est.action = "schedule"  # schedule cores / engineer review; never a structural verdict
                notes.append("upper end of the interval is below the design strength: schedule cores and engineer review")
            elif est.lo_mpa < design_fc_mpa:
                est.action = "calibrate_with_cores"
                notes.append("interval straddles the design strength: cannot tell without cores")
            else:
                est.action = "record"
                notes.append("interval is above the design strength; still an estimate, not an acceptance test")
        return est
