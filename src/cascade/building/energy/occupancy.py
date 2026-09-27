"""M1 presence classifier and M2 1-hour presence forecaster on REAL ROBOD, plus UCI checks.

M1  HistGradientBoostingClassifier on environment + Wi-Fi + time features. Lighting energy, illuminance, plug load,
    fans and HVAC signals are EXCLUDED because they respond to the lights/equipment being controlled (leakage).
    Splits: per room, first 70% of days train / last 30% test (day blocks), and leave-one-room-out.
    Baselines fitted on train only: majority class, 48-bin (weekday/weekend x hour) and 168-bin (day x hour) schedules.
M2  presence 12 steps (1 h) ahead within the same day. Oracle variant uses ground-truth presence at t (labelled);
    deployable variant uses the M1 probability at t instead (train-day values are out-of-fold by day groups).
    Baselines: persistence (oracle), M1 persistence (deployable), schedules at the target time, majority.
UCI 357  official train / test2 (forward in time) / test1 (NOT forward: precedes training) with and without Light.
UCI 864  PIR miss / false-trigger rates per 5-min window, used to INJECT an imperfect motion sensor into replays.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import Y

CO2 = "indoor_co2 [ppm]"
SPL = "sound_pressure_level [dba]"
WIFI = "wifi_connected_devices [number]"
M1_FEATURES = [CO2, "co2_d15", "co2_d30", SPL, "spl_max30", "voc [ppb]", "indoor_relative_humidity [%]",
               "air_temperature [Celsius]", WIFI, "hsin", "hcos", "dow", "wkend"]
LEAKY = ["lighting_energy [kWh]", "illuminance [lux]", "plug_load_energy [kWh]", "ceiling_fan_energy [kWh]",
         "chilled_water_energy [kWh]", "ahu_fan_energy [kWh]", "occupant_count [number]"]
M2_ORACLE = [CO2, "co2_d15", WIFI, "pres_now", "pres_1h_mean", "hod_t", "wkend", "dow"]
M2_DEPLOY = [CO2, "co2_d15", WIFI, "m1_now", "m1_1h_mean", "hod_t", "wkend", "dow"]
H_AHEAD = 12
HGB_KW = dict(random_state=0, max_iter=300)


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["day"] = d["ts"].dt.normalize()
    d["hod"] = d["ts"].dt.hour + d["ts"].dt.minute / 60.0
    d["hour"] = d["ts"].dt.hour
    d["hsin"] = np.sin(2 * np.pi * d["hod"] / 24)
    d["hcos"] = np.cos(2 * np.pi * d["hod"] / 24)
    d["dow"] = d["ts"].dt.dayofweek
    d["wkend"] = (d["dow"] >= 5).astype(int)
    d["how"] = d["dow"] * 24 + d["hour"]
    co2 = d[CO2]
    d["co2_d15"] = co2.diff(3)
    d["co2_d30"] = co2.diff(6)
    d["spl_max30"] = d[SPL].rolling(6, min_periods=1).max()
    d = d.dropna(subset=[Y])
    d[Y] = d[Y].astype(int)
    return d


def day_split(d: pd.DataFrame, frac: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    days = sorted(d["day"].unique())
    cut = days[int(len(days) * frac)]
    return d[d["day"] < cut], d[d["day"] >= cut], str(pd.Timestamp(cut).date())


def schedule_fit(tr: pd.DataFrame, bins: int, y: str = Y) -> Tuple[pd.Series, float]:
    key = ["wkend", "hour"] if bins == 48 else ["how"]
    return tr.groupby(key)[y].mean(), float(tr[y].mean())


def schedule_prob(table: pd.Series, prior: float, frame: pd.DataFrame, bins: int,
                  hour_col: str = "hour", how_col: str = "how") -> np.ndarray:
    if bins == 48:
        keys = list(zip(frame["wkend"], frame[hour_col]))
    else:
        keys = list(frame[how_col])
    return np.array([table.get(k, prior) for k in keys], dtype=float)


def clf_metrics(y, pred, prob=None) -> Dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()) if len(y) else None,
           "acc": float(accuracy_score(y, pred)), "f1": float(f1_score(y, pred, zero_division=0)),
           "bal_acc": float(balanced_accuracy_score(y, pred)) if len(set(y)) > 1 else None,
           "false_vacancy_rate": float(fn / (tp + fn)) if (tp + fn) else None,
           "false_occupied_rate": float(fp / (fp + tn)) if (fp + tn) else None, "auc": None}
    if prob is not None and len(set(y)) > 1 and len(set(np.round(prob, 12))) > 1:
        out["auc"] = float(roc_auc_score(y, prob))
    return out


def fit_hgb(X: pd.DataFrame, y) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**HGB_KW).fit(X, np.asarray(y).astype(int))


def day_bootstrap_f1_diff(y, pred_a, pred_b, days, n_boot: int = 1000, seed: int = 0) -> Dict:
    """F1(a) - F1(b) with a 95% percentile interval from resampling whole test days (cluster bootstrap)."""
    y, pa, pb = np.asarray(y).astype(int), np.asarray(pred_a).astype(int), np.asarray(pred_b).astype(int)
    days = np.asarray(days)
    uniq = np.unique(days)
    idx = {d: np.flatnonzero(days == d) for d in uniq}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[d] for d in rng.choice(uniq, len(uniq), replace=True)])
        diffs.append(f1_score(y[rows], pa[rows], zero_division=0) - f1_score(y[rows], pb[rows], zero_division=0))
    point = f1_score(y, pa, zero_division=0) - f1_score(y, pb, zero_division=0)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    verdict = "model better" if lo > 0 else ("baseline better" if hi < 0 else "no clear difference")
    return {"f1_diff": float(point), "ci95": [float(lo), float(hi)], "n_boot": n_boot, "n_days": int(len(uniq)),
            "unit": "test days", "verdict": verdict}


def _baselines(tr: pd.DataFrame, te: pd.DataFrame, y_te, *, hour_col="hour", how_col="how", y_tr_col=Y,
               preds: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Dict]:
    out = {}
    maj = int(tr[y_tr_col].mean() >= 0.5)
    out["majority"] = clf_metrics(y_te, np.full(len(y_te), maj))
    if preds is not None:
        preds["majority"] = np.full(len(y_te), maj)
    for bins in (48, 168):
        tab, prior = schedule_fit(tr, bins, y_tr_col)
        p = schedule_prob(tab, prior, te, bins, hour_col, how_col)
        out[f"schedule_{bins}"] = clf_metrics(y_te, (p >= 0.5).astype(int), p)
        if preds is not None:
            preds[f"schedule_{bins}"] = (p >= 0.5).astype(int)
    return out


def _vs_best(models: Dict, preds: Dict, y, model_key: str, cands, days) -> Dict:
    best = max(cands, key=lambda k: models[k]["f1"])
    return {"model": model_key, "baseline": best, **day_bootstrap_f1_diff(y, preds[model_key], preds[best], days)}


def m1_room_eval(rooms: Dict[int, pd.DataFrame]) -> Tuple[Dict, Dict[int, pd.DataFrame]]:
    """Day-blocked evaluation per room. Returns (results, per-room test frames with m1_prob)."""
    res, test_frames = {}, {}
    for i, raw in rooms.items():
        d = add_features(raw)
        tr, te, cut = day_split(d)
        y = te[Y].to_numpy()
        preds: Dict[str, np.ndarray] = {}
        r = {"split": f"day-blocked: train days < {cut} ({tr['day'].nunique()} d), test days >= {cut} ({te['day'].nunique()} d)",
             "n_train": int(len(tr)), "models": _baselines(tr, te, y, preds=preds)}
        m = fit_hgb(tr[M1_FEATURES], tr[Y])
        prob = m.predict_proba(te[M1_FEATURES])[:, 1]
        r["models"]["m1_hgb"] = clf_metrics(y, (prob >= 0.5).astype(int), prob)
        preds["m1_hgb"] = (prob >= 0.5).astype(int)
        r["vs_best_baseline"] = _vs_best(r["models"], preds, y, "m1_hgb", ("majority", "schedule_48", "schedule_168"),
                                         te["day"].to_numpy())
        env = [c for c in M1_FEATURES if c != WIFI]
        m_env = fit_hgb(tr[env], tr[Y])
        pe = m_env.predict_proba(te[env])[:, 1]
        r["models"]["m1_hgb_no_wifi"] = clf_metrics(y, (pe >= 0.5).astype(int), pe)
        res[i] = r
        tf = te.copy()
        tf["m1_prob"] = prob
        test_frames[i] = tf
    return res, test_frames


def m1_train_oof_frames(rooms: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
    """Training days of each room with out-of-fold M1 probabilities (folds grouped by day). Used to choose the switching
    policy without looking at the held-out test days."""
    out = {}
    for i, raw in rooms.items():
        tr, _, _ = day_split(add_features(raw))
        f = tr.copy()
        f["m1_prob"] = oof_by_day(tr, M1_FEATURES)
        out[i] = f
    return out


def m1_loro(rooms: Dict[int, pd.DataFrame]) -> Dict:
    feats = {i: add_features(d) for i, d in rooms.items()}
    res = {}
    for i in feats:
        tr = pd.concat([feats[j] for j in feats if j != i], ignore_index=True)
        te = feats[i]
        y = te[Y].to_numpy()
        r = {"split": f"leave-one-room-out: train rooms {[j for j in feats if j != i]}, test room {i} (all days)",
             "n_train": int(len(tr)), "models": _baselines(tr, te, y)}
        m = fit_hgb(tr[M1_FEATURES], tr[Y])
        prob = m.predict_proba(te[M1_FEATURES])[:, 1]
        r["models"]["m1_hgb"] = clf_metrics(y, (prob >= 0.5).astype(int), prob)
        res[i] = r
    return res


def oof_by_day(tr: pd.DataFrame, features: List[str], n_splits: int = 5) -> np.ndarray:
    """Out-of-fold M1 probabilities on training days, folds grouped by day (no row of a day sees its own day)."""
    out = np.full(len(tr), np.nan)
    groups = tr["day"].astype("int64").to_numpy()
    k = min(n_splits, len(np.unique(groups)))
    for a, b in GroupKFold(n_splits=k).split(tr, groups=groups):
        m = fit_hgb(tr.iloc[a][features], tr.iloc[a][Y])
        out[b] = m.predict_proba(tr.iloc[b][features])[:, 1]
    return out


def _m2_frame(d: pd.DataFrame, m1: np.ndarray) -> pd.DataFrame:
    d = d.copy()
    d["pres_now"] = d[Y]
    d["pres_1h_mean"] = d.groupby("day")[Y].transform(lambda s: s.rolling(12, min_periods=1).mean())
    d["m1_now"] = m1
    d["m1_1h_mean"] = d.groupby("day")["m1_now"].transform(lambda s: s.rolling(12, min_periods=1).mean())
    d["target"] = d.groupby("day")[Y].shift(-H_AHEAD)
    tgt_ts = d["ts"] + pd.Timedelta(minutes=5 * H_AHEAD)
    d["hod_t"] = tgt_ts.dt.hour + tgt_ts.dt.minute / 60.0
    d["hour_t"] = tgt_ts.dt.hour
    d["how_t"] = tgt_ts.dt.dayofweek * 24 + tgt_ts.dt.hour
    return d.dropna(subset=["target"])


def m2_eval(rooms: Dict[int, pd.DataFrame]) -> Tuple[Dict, Dict[int, pd.DataFrame]]:
    res, frames = {}, {}
    for i, raw in rooms.items():
        d = add_features(raw)
        tr, te, cut = day_split(d)
        m1 = fit_hgb(tr[M1_FEATURES], tr[Y])
        m1_tr = oof_by_day(tr, M1_FEATURES)
        m1_te = m1.predict_proba(te[M1_FEATURES])[:, 1]
        ftr, fte = _m2_frame(tr, m1_tr), _m2_frame(te, m1_te)
        y = fte["target"].astype(int).to_numpy()
        models, preds = {}, {}
        maj = int(ftr["target"].mean() >= 0.5)
        models["majority"] = clf_metrics(y, np.full(len(y), maj))
        preds["majority"] = np.full(len(y), maj)
        for bins in (48, 168):
            tab, prior = schedule_fit(tr, bins)
            p = schedule_prob(tab, prior, fte, bins, "hour_t", "how_t")
            models[f"schedule_{bins}"] = clf_metrics(y, (p >= 0.5).astype(int), p)
            preds[f"schedule_{bins}"] = (p >= 0.5).astype(int)
        models["persistence_oracle"] = clf_metrics(y, fte["pres_now"].to_numpy())
        preds["persistence_oracle"] = fte["pres_now"].to_numpy().astype(int)
        models["persistence_m1_deployable"] = clf_metrics(y, (fte["m1_now"] >= 0.5).astype(int), fte["m1_now"].to_numpy())
        preds["persistence_m1_deployable"] = (fte["m1_now"] >= 0.5).astype(int).to_numpy()
        mo = fit_hgb(ftr[M2_ORACLE], ftr["target"])
        po = mo.predict_proba(fte[M2_ORACLE])[:, 1]
        models["m2_hgb_oracle"] = clf_metrics(y, (po >= 0.5).astype(int), po)
        preds["m2_hgb_oracle"] = (po >= 0.5).astype(int)
        md = fit_hgb(ftr[M2_DEPLOY], ftr["target"])
        pdp = md.predict_proba(fte[M2_DEPLOY])[:, 1]
        models["m2_hgb_deployable"] = clf_metrics(y, (pdp >= 0.5).astype(int), pdp)
        preds["m2_hgb_deployable"] = (pdp >= 0.5).astype(int)
        dd = fte["day"].to_numpy()
        res[i] = {"split": f"day-blocked: test days >= {cut}; target = presence 1 h ahead within the same day",
                  "models": models,
                  "deployable_vs_best_baseline": _vs_best(models, preds, y, "m2_hgb_deployable",
                                                          ("majority", "schedule_48", "schedule_168", "persistence_m1_deployable"), dd),
                  "oracle_vs_persistence": {"model": "m2_hgb_oracle", "baseline": "persistence_oracle",
                                            **day_bootstrap_f1_diff(y, preds["m2_hgb_oracle"], preds["persistence_oracle"], dd)}}
        f = fte[["ts", "day", Y, "target"]].copy()
        f["m2_prob"] = pdp
        frames[i] = f
    return res, frames


# ------------------------------------------------------------------------------------------------------ UCI 357
UCI_SETS = {
    "with_light (leaky)": ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio", "hsin", "hcos", "wkend"],
    "no_light": ["Temperature", "Humidity", "CO2", "HumidityRatio", "co2_d10", "hsin", "hcos", "wkend"],
}


def _uci_feats(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    hod = d["date"].dt.hour + d["date"].dt.minute / 60.0
    d["hour"] = d["date"].dt.hour
    d["hsin"], d["hcos"] = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)
    d["wkend"] = (d["date"].dt.dayofweek >= 5).astype(int)
    d["co2_d10"] = d["CO2"].diff(10).fillna(0)
    return d


def uci357_eval(sets: Dict[str, pd.DataFrame]) -> Dict:
    tr = _uci_feats(sets["datatraining"])
    out = {"train": {"n": int(len(tr)), "from": str(tr["date"].min()), "to": str(tr["date"].max())}, "tests": {}}
    tab = tr.groupby(["wkend", "hour"])["Occupancy"].mean()
    prior = float(tr["Occupancy"].mean())
    for name, note in [("datatest2", "forward in time (after training)"), ("datatest", "NOT forward: precedes the training days")]:
        te = _uci_feats(sets[name])
        y = te["Occupancy"].to_numpy()
        rows = {"majority": clf_metrics(y, np.full(len(y), int(prior >= 0.5)))}
        p = np.array([tab.get(k, prior) for k in zip(te["wkend"], te["hour"])])
        rows["schedule_48"] = clf_metrics(y, (p >= 0.5).astype(int), p)
        for sn, cols in UCI_SETS.items():
            for mn, m in [("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                          ("hgb", HistGradientBoostingClassifier(random_state=0))]:
                m.fit(tr[cols], tr["Occupancy"])
                pr = m.predict_proba(te[cols])[:, 1]
                rows[f"{mn}:{sn}"] = clf_metrics(y, (pr >= 0.5).astype(int), pr)
        out["tests"][name] = {"note": note, "n": int(len(te)), "from": str(te["date"].min()), "to": str(te["date"].max()),
                              "models": rows}
    return out


# ------------------------------------------------------------------------------------------------------ UCI 864
def pir_rates(d: pd.DataFrame, window: str = "5min", min_samples: int = 8) -> Dict:
    """Per-window PIR miss rate (fully occupied window with no PIR trigger) and false-trigger rate."""
    g = d.assign(w=d["ts"].dt.floor(window), pir=((d["S6_PIR"] > 0) | (d["S7_PIR"] > 0)).astype(int))
    agg = g.groupby("w").agg(n=("pir", "size"), pir_any=("pir", "max"), occ_min=("Room_Occupancy_Count", "min"),
                             occ_max=("Room_Occupancy_Count", "max"))
    agg = agg[agg["n"] >= min_samples]
    occ = agg[agg["occ_min"] > 0]
    vac = agg[agg["occ_max"] == 0]
    return {"window": window, "min_samples_per_window": min_samples,
            "n_occupied_windows": int(len(occ)), "n_vacant_windows": int(len(vac)),
            "miss_rate": float((occ["pir_any"] == 0).mean()) if len(occ) else None,
            "false_trigger_rate": float((vac["pir_any"] == 1).mean()) if len(vac) else None,
            "sensors": "S6_PIR or S7_PIR", "sample_interval_s": 30}


def inject_detection(occ: np.ndarray, miss_rate: float, false_rate: float, rng: np.random.Generator) -> np.ndarray:
    """INJECTED motion-sensor signal: independent per-window misses / false triggers at the UCI 864 rates."""
    occ = np.asarray(occ, dtype=bool)
    u = rng.random(len(occ))
    return np.where(occ, u >= miss_rate, u < false_rate)
