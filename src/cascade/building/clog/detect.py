"""Detectors and evaluation helpers for riser clogs (numpy / pandas / scikit-learn; runtime-safe).

* z-score rule (no ML): per-feature z against clean training nights, alarm on the max z, locate by argmax,
  grade by two cut points on max z chosen on out-of-fold training nights.
* gradient boosting: HistGradientBoostingClassifier for severity (4 classes) and for location (8 classes).
* thresholds always come from OUT-OF-FOLD clean training nights (GroupKFold by week), never from the test weeks.
* evaluation per scenario (one week with one clog). 95% confidence intervals use a two-stage cluster bootstrap when
  the table has a ``week`` column: resample the demand weeks with replacement, then scenarios within each drawn week
  (scenarios that share a demand week are not independent). Without a ``week`` column they resample scenarios.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

HGB_PARAMS = dict(max_iter=300, learning_rate=0.08, random_state=0)


def hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**HGB_PARAMS)


# ------------------------------------------------------------------------------------------------ z-score rule
class ZRule:
    """Transparent baseline: z of each feature vs clean training nights; max z alarms and locates."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ZRule":
        clean = X[y == 0]
        self.mu = np.nanmean(clean, axis=0)
        self.sd = np.nanstd(clean, axis=0) + 1e-9
        return self

    def z(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd

    def maxz(self, X: np.ndarray) -> np.ndarray:
        return np.nanmax(self.z(X), axis=1)

    def locate(self, X: np.ndarray) -> np.ndarray:
        return np.nanargmax(self.z(X), axis=1)


def oof_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, kind: str, n_splits: int = 4) -> np.ndarray:
    """Out-of-fold clog score per training night: P(clog) for 'hgb', max z for 'z'."""
    out = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if kind == "hgb":
            m = hgb().fit(X[tr], y[tr])
            out[te] = 1.0 - m.predict_proba(X[te])[:, list(m.classes_).index(0)]
        else:
            out[te] = ZRule().fit(X[tr], y[tr]).maxz(X[te])
    return out


def threshold_at(scores: np.ndarray, y: np.ndarray, fa_target: float = 0.05) -> float:
    return float(np.nanquantile(scores[y == 0], 1.0 - fa_target))


def z_cutpoints(maxz: np.ndarray, y: np.ndarray, thr: float) -> List[float]:
    """Two cut points c1 < c2 (both above the alarm threshold) that maximise training macro-F1 of
    sev = 0 below thr, 1 in [thr, c1), 2 in [c1, c2), 3 above c2."""
    grid = np.unique(np.nanquantile(maxz[maxz > thr], np.linspace(0.02, 0.98, 49))) if (maxz > thr).sum() > 5 \
        else np.array([thr + 1, thr + 2])
    best, bc = -1.0, [float(grid[0]), float(grid[-1])]
    for i, c1 in enumerate(grid):
        for c2 in grid[i + 1:]:
            pred = z_grade(maxz, thr, [c1, c2])
            f = f1_score(y, pred, average="macro", labels=[0, 1, 2, 3], zero_division=0)
            if f > best:
                best, bc = f, [float(c1), float(c2)]
    return bc


def z_grade(maxz: np.ndarray, thr: float, cuts: Sequence[float]) -> np.ndarray:
    return np.where(maxz <= thr, 0, np.where(maxz < cuts[0], 1, np.where(maxz < cuts[1], 2, 3)))


# ------------------------------------------------------------------------------------------------ evaluation
def persistent(flags: np.ndarray, k: int = 2, n: int = 3) -> bool:
    """True if any window of n consecutive nights has at least k alarms."""
    f = np.asarray(flags, int)
    if len(f) < n:
        return bool(f.sum() >= k)
    return bool(np.any(np.convolve(f, np.ones(n, int), mode="valid") >= k))


def _groups(labels: np.ndarray) -> List[np.ndarray]:
    labels = np.asarray(labels)
    return [np.nonzero(labels == u)[0] for u in np.unique(labels)]


def cluster_resample(clusters: Optional[np.ndarray], n: int, rng: np.random.Generator,
                     strata: Optional[np.ndarray] = None) -> np.ndarray:
    """Indices of one bootstrap sample of n units.

    clusters None: units drawn with replacement (within each stratum if strata is given).
    clusters given: two-stage cluster bootstrap. Clusters (demand weeks) are drawn with replacement, then units are
    drawn with replacement inside each drawn cluster (within each stratum of that cluster if strata is given)."""
    if clusters is None:
        groups = _groups(strata) if strata is not None else [np.arange(n)]
        return np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
    cgroups = _groups(np.asarray(clusters))
    idx = []
    for k in rng.integers(0, len(cgroups), len(cgroups)):
        members = cgroups[k]
        sub = [members] if strata is None else [members[m] for m in _groups(np.asarray(strata)[members])]
        idx += [g[rng.integers(0, len(g), len(g))] for g in sub]
    return np.concatenate(idx)


def boot_ci(per_scenario: np.ndarray, n_boot: int = 1000, seed: int = 0,
            clusters: Optional[np.ndarray] = None) -> List[float]:
    """95% CI of the mean of per-scenario values (each scenario = one week with one clog state).
    With ``clusters`` (the demand week of each scenario) the resampling is two-stage: weeks, then scenarios."""
    v = np.asarray(per_scenario, float)
    if len(v) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    if clusters is None:
        means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    else:
        means = np.array([v[cluster_resample(clusters, len(v), rng)].mean() for _ in range(n_boot)])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _scenario_weeks(d: pd.DataFrame, per_scen: pd.Series) -> Optional[np.ndarray]:
    """Demand week of each scenario in per_scen's index order, or None when the table has no week column."""
    if "week" not in d.columns:
        return None
    return d.groupby("scenario")["week"].first().reindex(per_scen.index).to_numpy()


def macro_f1_from_cm(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(axis=0), out=np.zeros_like(tp), where=cm.sum(axis=0) > 0)
    rec = np.divide(tp, cm.sum(axis=1), out=np.zeros_like(tp), where=cm.sum(axis=1) > 0)
    f = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f.mean())


def alarm_rates(df: pd.DataFrame, flag_col: str, n_boot: int = 1000, seed: int = 0) -> Dict[str, dict]:
    """Per true severity: mean nightly alarm rate over scenarios, bootstrap CI (week-cluster when a week column
    exists), n scenarios / nights / weeks, and the share of scenarios with a persistent (2 of 3 nights) alarm."""
    out = {}
    for s, d in df.groupby("sev"):
        per_scen = d.groupby("scenario")[flag_col].mean()
        pers = d.sort_values(["scenario", "day"]).groupby("scenario")[flag_col].apply(lambda f: persistent(f.values))
        wk = _scenario_weeks(d, per_scen)
        out[str(int(s))] = {"rate": float(per_scen.mean()), "ci95": boot_ci(per_scen.to_numpy(), n_boot, seed, wk),
                            "n_scenarios": int(len(per_scen)), "n_nights": int(len(d)),
                            "n_weeks": None if wk is None else int(len(np.unique(wk))),
                            "persistent_2of3": float(pers.mean()),
                            "persistent_ci95": boot_ci(pers.reindex(per_scen.index).to_numpy(float), n_boot, seed,
                                                       wk)}
    return out


def severity_metrics(df: pd.DataFrame, pred_col: str, n_boot: int = 1000, seed: int = 0) -> dict:
    y, p = df["sev"].to_numpy(int), df[pred_col].to_numpy(int)
    cm = confusion_matrix(y, p, labels=[0, 1, 2, 3])
    scen = df.groupby("scenario")
    cms = np.stack([confusion_matrix(g["sev"], g[pred_col], labels=[0, 1, 2, 3]) for _, g in scen])
    strata = scen["sev"].first().to_numpy()
    weeks = scen["week"].first().to_numpy() if "week" in df.columns else None
    rng = np.random.default_rng(seed)
    f1s = [macro_f1_from_cm(cms[cluster_resample(weeks, len(strata), rng, strata)].sum(axis=0))
           for _ in range(n_boot)]
    qwk = cohen_kappa_score(y, p, weights="quadratic", labels=[0, 1, 2, 3])
    return {"macro_f1": macro_f1_from_cm(cm), "macro_f1_ci95": [float(np.quantile(f1s, 0.025)),
                                                                float(np.quantile(f1s, 0.975))],
            "qwk": float(qwk), "confusion": cm.tolist(), "n_nights": int(len(y)),
            "n_scenarios": int(df["scenario"].nunique())}


def localisation(df: pd.DataFrame, pred_col: str, n_boot: int = 1000, seed: int = 0) -> Dict[str, dict]:
    """Accuracy of the predicted location on clog nights, by true severity (chance = 1/8)."""
    out = {}
    c = df[df.sev > 0]
    for s, d in c.groupby("sev"):
        hit = (d[pred_col].to_numpy() == d["loc_id"].to_numpy()).astype(float)
        per_scen = pd.Series(hit, index=d.index).groupby(d["scenario"]).mean()
        out[str(int(s))] = {"accuracy": float(hit.mean()),
                            "ci95": boot_ci(per_scen.to_numpy(), n_boot, seed, _scenario_weeks(d, per_scen)),
                            "n_scenarios": int(len(per_scen)), "n_nights": int(len(d))}
    return out


def majority_f1(y_train: np.ndarray, y_test: np.ndarray) -> float:
    maj = int(pd.Series(y_train).mode()[0])
    return float(f1_score(y_test, np.full(len(y_test), maj), average="macro", labels=[0, 1, 2, 3], zero_division=0))


def baseline_f1s(y_train: np.ndarray, y_test: np.ndarray, n_draws: int = 1000, seed: int = 0) -> Dict[str, object]:
    """Naive grading baselines on the test labels (macro-F1 over 4 classes):
    majority = always the most common TRAINING class; constant[k] = always class k;
    stratified_random = a random class drawn with the training class frequencies (mean and 95% range over draws)."""
    y_train, y_test = np.asarray(y_train, int), np.asarray(y_test, int)
    maj = int(pd.Series(y_train).mode()[0])
    const = {str(k): float(f1_score(y_test, np.full(len(y_test), k), average="macro", labels=[0, 1, 2, 3],
                                    zero_division=0)) for k in (0, 1, 2, 3)}
    freq = np.bincount(y_train, minlength=4) / len(y_train)
    rng = np.random.default_rng(seed)
    rand = np.array([f1_score(y_test, rng.choice(4, len(y_test), p=freq), average="macro", labels=[0, 1, 2, 3],
                              zero_division=0) for _ in range(n_draws)])
    return {"majority_class": maj, "majority_macro_f1": const[str(maj)], "constant_macro_f1": const,
            "train_class_freq": [float(x) for x in freq],
            "stratified_random": {"mean": float(rand.mean()), "range95": [float(np.quantile(rand, 0.025)),
                                                                           float(np.quantile(rand, 0.975))],
                                  "n_draws": n_draws}}


def fit_predict(Xtr: np.ndarray, ytr: np.ndarray, gtr: np.ndarray, Xte: np.ndarray, kind: str,
                fa_target: float = 0.05) -> Dict[str, object]:
    """Fit a detector on train, set its alarm threshold out of fold, return test scores, flags, grades."""
    oof = oof_scores(Xtr, ytr, gtr, kind)
    thr = threshold_at(oof, ytr, fa_target)
    if kind == "hgb":
        m = hgb().fit(Xtr, ytr)
        score = 1.0 - m.predict_proba(Xte)[:, list(m.classes_).index(0)]
        grade = m.predict(Xte)
        return {"model": m, "thr": thr, "score": score, "flag": score > thr, "grade": grade, "oof": oof}
    z = ZRule().fit(Xtr, ytr)
    cuts = z_cutpoints(oof, ytr, thr)
    score = z.maxz(Xte)
    return {"model": z, "thr": thr, "cuts": cuts, "score": score, "flag": score > thr,
            "grade": z_grade(score, thr, cuts), "loc": z.locate(Xte), "oof": oof}


def as_float(x: Optional[float]) -> Optional[float]:
    return None if x is None or not np.isfinite(x) else float(x)
