"""Real-data early-fire alarm verifier (ask 5 Part B). REAL lab data; decision support only.

Data: two Mendeley Data records by the same contributor (listed as "Pascal V", Otto von Guericke Universitat
Magdeburg), both CC BY 4.0: the EN54 test room (v1 file, doi 10.17632/npk2zcm85h.1) for training, and the
Industrial Hall (doi 10.17632/yghykzm4km.1) as an unseen room for the held-out test. 10 s multi-sensor samples
(CO2, CO, H2, humidity, particulate counts, temperature, VOC) from 9 nodes per room. Changes made by us: features
derived, episodes segmented, Hall id 13 (a 1-row fragment) merged into id 14.

Two stages:
- Stage 1 (conventional single-signal trigger, a team-defined proxy, NOT a listed detector): PM_Total_Room rise over
  its 30-min rolling median (shifted one sample, so causal) above the q99.9 of TRAINING clean background, for k=3
  consecutive samples of one sensor node.
- Stage 2: HistGradientBoosting (3 classes: background / fire / nuisance) gives P(fire). It only LABELS a stage-1
  trigger: 'fire-like evidence' when P(fire) >= 0.5 within 60 s of the trigger, otherwise 'nuisance-like, FSD to
  check'. It never suppresses, delays or downgrades an alarm; in a building the listed fire alarm system decides.

Every setting is frozen in FROZEN before the Hall is touched. Metric definitions follow the verifier's corrections:
latency from the label start with triggers up to 5 min early counted; clean background = at least 30 min before any
episode start and at least 60 min after any episode end, after a 30-min warm-up per sensor series and after gaps
over 30 min; sensor-hours = samples x 10 s; bootstrap over episodes and over recording days.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

CHANNELS: Tuple[str, ...] = ("CO2_Room", "CO_Room", "H2_Room", "Humidity_Room", "PM05_Room", "PM10_Room", "PM25_Room", "PM40_Room",
                             "PM100_Room", "PM_Total_Room", "PM_Room_Typical_Size", "Temperature_Room", "VOC_Room_RAW")
FEATURE_SETS: Dict[str, Tuple[str, ...]] = {
    "full": tuple(CHANNELS) + tuple(c + "_d" for c in CHANNELS) + tuple(c + "_s" for c in CHANNELS),
    "deltas": tuple(c + "_d" for c in CHANNELS) + tuple(c + "_s" for c in CHANNELS),
}
KIND_CODE = {"background": 0, "fire": 1, "nuisance": 2}

DATASETS: Dict[str, Dict[str, Any]] = {
    "en54": {
        "file": "en54_room.csv",
        "title": "Indoor Fire Dataset with Distributed Multi-Sensor Nodes (EN54 Test Room)",
        "doi": "10.17632/npk2zcm85h.1", "version": 1, "published": "2023-06-07",
        "page": "https://data.mendeley.com/datasets/npk2zcm85h/1",
        "url": "https://data.mendeley.com/public-files/datasets/npk2zcm85h/files/0c5ed6b5-d59a-4e09-ada2-6f58e34b7666/file_downloaded",
        "sha256": "65a568cf84bf17c0e7d25fe89f3547c93a097a18c65b9f357eab3c7104c97c1a", "bytes": 44149547,
        "citation": "V, Pascal (2023), 'Indoor Fire Dataset with Distributed Multi-Sensor Nodes (EN54 Test Room)', Mendeley Data, V1, "
                    "doi: 10.17632/npk2zcm85h.1",
        "note": "A version 2 (doi 10.17632/npk2zcm85h.2, 2025-06-20) with revised labels exists (no Nuisance class in ternary_label, "
                "one Cable run removed, later fire starts). This build uses the V1 file.",
    },
    "hall": {
        "file": "industrial_hall.csv",
        "title": "Indoor Fire Dataset with Distributed Multi-Sensor Nodes (Industrial Hall)",
        "doi": "10.17632/yghykzm4km.1", "version": 1, "published": "2024-09-16",
        "page": "https://data.mendeley.com/datasets/yghykzm4km/1",
        "url": "https://data.mendeley.com/public-files/datasets/yghykzm4km/files/ac22049a-6c73-4276-8583-05b2cf735601/file_downloaded",
        "sha256": "b431b611a44e4282510bd53158aaa71512554481433d8f3a90f9475c17134839", "bytes": 35800222,
        "citation": "V, Pascal (2024), 'Indoor Fire Dataset with Distributed Multi-Sensor Nodes (Industrial Hall)', Mendeley Data, V1, "
                    "doi: 10.17632/yghykzm4km.1",
        "note": "Episode ids from anomaly_scenario; id 13 (1 row) merged into id 14; Ethanol (5) and CO release (6, 17) carry neither "
                "the fire nor the nuisance flag and are reported as 'other'; anomaly_label is ignored.",
    },
}
LICENCE = {"name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/",
           "contributor": "Pascal V, Otto von Guericke Universitat Magdeburg (as listed on the Mendeley records)",
           "changes": "features derived (rolling-median deltas, 6-sample differences); episodes segmented; Hall id 13 merged into 14"}

FROZEN: Dict[str, Any] = {
    "k_consecutive": 3,
    "verify_window_s": 60,
    "median_window": "30min",
    "median_min_periods": 5,
    "slope_lag_samples": 6,
    "stage1_channel": "PM_Total_Room_d",
    "co_channel": "CO_Room_d",
    "stage1_quantile": 0.999,
    "stage2_p_fire": 0.5,
    "hgb": {"max_iter": 150, "learning_rate": 0.1, "class_weight": "balanced", "random_state": 0},
    "train_bg_subsample": 3,
    "pre_start_s": 300,
    "bg_before_s": 1800,
    "bg_after_s": 3600,
    "warmup_s": 1800,
    "warmup_gap_s": 1800,
    "run_gap_s": 60,
    "sample_s": 10,
    "en54_split_gap_s": 600,
    "hall_merge": {"13": 14},
    "n_boot": 1000,
    "seed": 0,
    "headline_features": "full",
    "ablation_features": "deltas",
    "within_site_cv": "leave-one-recording-day-out on EN54",
    "cross_site": "train on all EN54, test once on the Industrial Hall",
}
METHODS: Tuple[str, ...] = ("stage1_pm", "co_delta", "ml_alone", "two_stage")
METHOD_LABEL = {"stage1_pm": "PM rise trigger alone (naive baseline, every trigger called fire)",
                "co_delta": "CO rise trigger alone (naive baseline)",
                "ml_alone": "ML alone, no gate (P(fire) >= 0.5 for 3 samples)",
                "two_stage": "Two-stage: PM trigger raises, ML labels (ours)"}


def frozen_hash(cfg: Optional[Dict[str, Any]] = None) -> str:
    return hashlib.sha256(json.dumps(cfg or FROZEN, sort_keys=True).encode()).hexdigest()[:16]


_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def secs(t: pd.Series) -> np.ndarray:
    """UTC seconds since the epoch as float64 (independent of the pandas datetime unit)."""
    return ((pd.to_datetime(t, utc=True) - _EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype="float64")


def with_secs(eps: pd.DataFrame) -> pd.DataFrame:
    out = eps.copy()
    out["start_s"] = secs(out["start"]) if len(out) else np.array([], dtype=float)
    out["end_s"] = secs(out["end"]) if len(out) else np.array([], dtype=float)
    return out


# ------------------------------------------------------------------------------------------ fetch and load

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(raw_dir: Path, download: bool = True) -> Dict[str, Any]:
    """Make sure both CSVs are in raw_dir (download with requests when missing), check sha256 against the Mendeley API
    hashes and return a provenance manifest."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Any] = {"licence": LICENCE, "datasets": {}}
    for key, d in DATASETS.items():
        p = raw_dir / d["file"]
        if not p.exists():
            if not download:
                raise FileNotFoundError(p)
            import requests

            with requests.get(d["url"], stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = p.with_suffix(".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.replace(p)
        sha = sha256_file(p)
        out["datasets"][key] = {**{k: v for k, v in d.items() if k != "file"}, "local_file": p.name, "sha256_measured": sha,
                                "sha256_ok": sha == d["sha256"], "bytes_measured": p.stat().st_size}
    return out


def load_site(path: Path, site: str) -> pd.DataFrame:
    """One tidy frame per site: t (UTC), sensor, the 13 common channels (float32), scen (scenario label) and the site's
    label columns. Sorted by sensor then time."""
    raw = pd.read_csv(path)
    df = pd.DataFrame({"t": pd.to_datetime(raw["Date"], utc=True, format="mixed"), "sensor": raw["Sensor_ID"].astype(str)})
    for c in CHANNELS:
        df[c] = pd.to_numeric(raw[c], errors="coerce").astype("float32")
    df["scen"] = raw["scenario_label"].astype(str)
    if site == "en54":
        df["ternary"] = raw["ternary_label"].astype(str)
    else:
        df["anomaly_scenario"] = pd.to_numeric(raw["anomaly_scenario"], errors="coerce")
        df["fire_flag"] = pd.to_numeric(raw["fire"], errors="coerce").fillna(0).astype(int)
        df["nuisance_flag"] = pd.to_numeric(raw["nuisance"], errors="coerce").fillna(0).astype(int)
    df["site"] = site
    return df.sort_values(["sensor", "t"], kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------------------------------ episodes

def episodes_en54(df: pd.DataFrame, gap_s: float = FROZEN["en54_split_gap_s"]) -> pd.DataFrame:
    """Runs of non-Background scenario_label rows (all sensors, time order), split on a gap over gap_s OR a
    scenario_label change (the verifier's correction; the gap-only rule merges three nuisance runs)."""
    a = df.loc[df["scen"] != "Background", ["t", "scen", "ternary"]].sort_values("t", kind="stable")
    if a.empty:
        return pd.DataFrame(columns=["ep", "site", "kind", "scenario", "start", "end", "n_rows"])
    gap = a["t"].diff().dt.total_seconds().fillna(np.inf)
    new = (a["scen"] != a["scen"].shift()) | (gap > gap_s)
    a = a.assign(ep=new.cumsum().astype(int))
    rows = []
    for ep, g in a.groupby("ep", sort=True):
        tern = g["ternary"].mode().iloc[0].lower()
        kind = tern if tern in ("fire", "nuisance") else "other"
        rows.append({"ep": f"en54-{ep:02d}", "site": "en54", "kind": kind, "scenario": g["scen"].iloc[0], "start": g["t"].min(),
                     "end": g["t"].max(), "n_rows": len(g)})
    return with_secs(pd.DataFrame(rows))


def episodes_hall(df: pd.DataFrame, merge: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """Episodes from anomaly_scenario ids. Ids in `merge` are folded into their target (13 -> 14). kind = fire when the
    fire flag is set, nuisance when the nuisance flag is set, otherwise 'other' (Ethanol 5, CO release 6 and 17)."""
    merge = {int(k): int(v) for k, v in (merge if merge is not None else FROZEN["hall_merge"]).items()}
    a = df.loc[df["anomaly_scenario"].notna(), ["t", "scen", "anomaly_scenario", "fire_flag", "nuisance_flag"]].copy()
    a["aid"] = a["anomaly_scenario"].astype(int).map(lambda x: merge.get(x, x))
    rows = []
    for aid, g in a.groupby("aid", sort=True):
        kind = "fire" if g["fire_flag"].max() == 1 else ("nuisance" if g["nuisance_flag"].max() == 1 else "other")
        rows.append({"ep": f"hall-{aid:02d}", "site": "hall", "kind": kind, "scenario": ",".join(sorted(set(g["scen"]))),
                     "start": g["t"].min(), "end": g["t"].max(), "n_rows": len(g)})
    return with_secs(pd.DataFrame(rows))


# ------------------------------------------------------------------------------------------ features and masks

def add_features(df: pd.DataFrame, cfg: Dict[str, Any] = FROZEN) -> pd.DataFrame:
    """Per sensor node: <ch>_d = value minus the 30-min rolling median shifted by one sample; <ch>_s = value minus the
    value slope_lag samples earlier. Both use only data up to the current sample (causal)."""
    out = df.copy()
    lag = int(cfg["slope_lag_samples"])
    d_cols = {c + "_d": np.full(len(df), np.nan, dtype="float32") for c in CHANNELS}
    s_cols = {c + "_s": np.full(len(df), np.nan, dtype="float32") for c in CHANNELS}
    for _, idx in df.groupby("sensor", sort=False).indices.items():
        g = df.iloc[idx]
        gi = g.set_index("t")[list(CHANNELS)].astype("float64")
        med = gi.rolling(cfg["median_window"], min_periods=int(cfg["median_min_periods"])).median().shift(1)
        vals = gi.to_numpy()
        dv = vals - med.to_numpy()
        sv = vals - gi.shift(lag).to_numpy()
        for j, c in enumerate(CHANNELS):
            d_cols[c + "_d"][idx] = dv[:, j]
            s_cols[c + "_s"][idx] = sv[:, j]
    for k, v in {**d_cols, **s_cols}.items():
        out[k] = v
    return out


def warmup_mask(df: pd.DataFrame, cfg: Dict[str, Any] = FROZEN) -> np.ndarray:
    """True where a row is inside the median warm-up: the first warmup_s of each sensor series and warmup_s after any
    gap over warmup_gap_s. Expects rows sorted by sensor then time."""
    warm = np.zeros(len(df), dtype=bool)
    tall = df["ts"].to_numpy() if "ts" in df else secs(df["t"])
    for _, idx in df.groupby("sensor", sort=False).indices.items():
        t = tall[idx]
        gap = np.diff(t, prepend=-np.inf)
        seg_start_t = np.where(gap > cfg["warmup_gap_s"], t, np.nan)
        seg_start_t = pd.Series(seg_start_t).ffill().to_numpy()
        warm[idx] = (t - seg_start_t) < cfg["warmup_s"]
    return warm


def episode_of_rows(ts: np.ndarray, eps: pd.DataFrame) -> np.ndarray:
    """Episode id per row when the row time (epoch seconds) is inside [start, end] of an episode, else ''."""
    out = np.array([""] * len(ts), dtype=object)
    for e in eps.itertuples():
        out[(ts >= e.start_s) & (ts <= e.end_s)] = e.ep
    return out


def clean_background_mask(ts: np.ndarray, eps: pd.DataFrame, warm: np.ndarray, cfg: Dict[str, Any] = FROZEN) -> np.ndarray:
    """True for rows at least bg_before_s before every episode start and at least bg_after_s after every episode end,
    outside warm-up."""
    ok = ~np.asarray(warm, dtype=bool).copy()
    for e in eps.itertuples():
        ok &= ~((ts >= e.start_s - cfg["bg_before_s"]) & (ts <= e.end_s + cfg["bg_after_s"]))
    return ok


def prepare(df: pd.DataFrame, eps: pd.DataFrame, cfg: Dict[str, Any] = FROZEN) -> pd.DataFrame:
    """Features plus bookkeeping columns: warm, ep (episode id or ''), kind (episode kind or 'background'/'guard'),
    clean_bg, day."""
    out = add_features(df, cfg)
    out["ts"] = secs(out["t"])
    out["warm"] = warmup_mask(out, cfg)
    out["ep"] = episode_of_rows(out["ts"].to_numpy(), eps)
    kind_by_ep = dict(zip(eps["ep"], eps["kind"]))
    out["clean_bg"] = clean_background_mask(out["ts"].to_numpy(), eps, out["warm"].to_numpy(), cfg)
    out["kind"] = np.where(out["ep"] != "", out["ep"].map(kind_by_ep).fillna("other"), np.where(out["clean_bg"], "background", "guard"))
    out["day"] = out["t"].dt.strftime("%Y-%m-%d")
    return out


# ------------------------------------------------------------------------------------------ stage 1 and stage 2

def stage1_threshold(train: pd.DataFrame, col: str, q: float) -> float:
    """q-quantile of a delta channel over TRAINING clean background rows only."""
    v = train.loc[train["clean_bg"], col].dropna()
    if v.empty:
        raise ValueError("no clean background rows to set a threshold")
    return float(v.quantile(q))


def run_length(cond: np.ndarray, seg_start: np.ndarray) -> np.ndarray:
    """Length of the current run of True values, restarting at every segment start."""
    c = cond.astype(bool)
    cs = np.cumsum(c.astype(np.int64))
    base = np.where(~c, cs, np.where(seg_start, cs - 1, 0))
    base = np.maximum.accumulate(base)
    return cs - base


def segment_starts(df: pd.DataFrame, gap_s: float) -> np.ndarray:
    """True at the first row of each sensor and after any time gap over gap_s (rows sorted by sensor, time)."""
    sensor = df["sensor"].to_numpy()
    t = df["ts"].to_numpy() if "ts" in df else secs(df["t"])
    start = np.ones(len(df), dtype=bool)
    if len(df) > 1:
        start[1:] = (sensor[1:] != sensor[:-1]) | (np.diff(t) > gap_s)
    return start


def alarm_events(df: pd.DataFrame, cond: np.ndarray, cfg: Dict[str, Any] = FROZEN) -> np.ndarray:
    """Row indices where a condition first holds for k consecutive samples of one sensor (rising edges). Warm-up rows
    never alarm; a time gap over run_gap_s breaks a run."""
    c = np.asarray(cond, dtype=bool) & ~df["warm"].to_numpy()
    rl = run_length(c, segment_starts(df, cfg["run_gap_s"]))
    return np.flatnonzero(rl == int(cfg["k_consecutive"]))


def verify_triggers(df: pd.DataFrame, trig_idx: np.ndarray, pfire: np.ndarray, cfg: Dict[str, Any] = FROZEN) -> pd.DataFrame:
    """For each stage-1 trigger: max P(fire) of the same sensor within verify_window_s after the trigger, and the time
    of the first sample in that window with P(fire) >= stage2_p_fire (NaT when never)."""
    t = df["ts"].to_numpy()
    sensor = df["sensor"].to_numpy()
    win = float(cfg["verify_window_s"])
    rows = []
    n = len(df)
    for i in trig_idx:
        j = i
        pmax, tver = -1.0, np.nan
        while j < n and sensor[j] == sensor[i] and t[j] <= t[i] + win:
            p = float(pfire[j]) if np.isfinite(pfire[j]) else 0.0
            pmax = max(pmax, p)
            if np.isnan(tver) and p >= cfg["stage2_p_fire"]:
                tver = t[j]
            j += 1
        rows.append({"row": int(i), "t": t[i], "sensor": sensor[i], "pmax": pmax, "t_verified": tver, "verified": not np.isnan(tver)})
    return pd.DataFrame(rows, columns=["row", "t", "sensor", "pmax", "t_verified", "verified"])


def attribute(ts: np.ndarray, eps: pd.DataFrame, clean_bg_at: np.ndarray, cfg: Dict[str, Any] = FROZEN) -> np.ndarray:
    """Episode id for an event time inside [start - pre_start_s, end] (an episode containing t in [start, end] wins),
    'bg' when the event row is clean background, else 'near' (guard band)."""
    out = []
    pre = float(cfg["pre_start_s"])
    st, en, ids = eps["start_s"].to_numpy(), eps["end_s"].to_numpy(), eps["ep"].to_numpy()
    for k, tt in enumerate(np.asarray(ts, dtype=float)):
        inside = np.flatnonzero((st <= tt) & (en >= tt))
        if len(inside):
            out.append(ids[inside[-1]])
            continue
        early = np.flatnonzero((st - pre <= tt) & (en >= tt))
        if len(early):
            out.append(ids[early[np.argmin(st[early])]])
            continue
        out.append("bg" if clean_bg_at[k] else "near")
    return np.array(out, dtype=object)


def train_stage2(train: pd.DataFrame, features: Sequence[str], cfg: Dict[str, Any] = FROZEN):
    """3-class HistGradientBoosting on episode rows (fire, nuisance) and clean background rows (every
    train_bg_subsample-th). Guard-band, warm-up and 'other' rows are not used for training."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    use_ep = (train["kind"].isin(["fire", "nuisance"])) & (~train["warm"])
    bg_idx = np.flatnonzero(train["clean_bg"].to_numpy())
    keep_bg = np.zeros(len(train), dtype=bool)
    keep_bg[bg_idx[:: int(cfg["train_bg_subsample"])]] = True
    m = use_ep.to_numpy() | keep_bg
    X = train.loc[m, list(features)].to_numpy(dtype="float32")
    y = train.loc[m, "kind"].map(KIND_CODE).to_numpy()
    h = cfg["hgb"]
    clf = HistGradientBoostingClassifier(max_iter=h["max_iter"], learning_rate=h["learning_rate"], class_weight=h["class_weight"],
                                         random_state=h["random_state"])
    clf.fit(X, y)
    return clf, {"n_train_rows": int(m.sum()), "n_fire_rows": int((y == 1).sum()), "n_nuisance_rows": int((y == 2).sum()),
                 "n_background_rows": int((y == 0).sum())}


def p_fire(clf, df: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    proba = clf.predict_proba(df[list(features)].to_numpy(dtype="float32"))
    col = list(clf.classes_).index(1)
    return proba[:, col].astype("float32")


# ------------------------------------------------------------------------------------------ evaluation

@dataclass
class SiteRun:
    """Per-row scores and thresholds for one evaluated site (pooled over folds for the within-site CV)."""
    df: pd.DataFrame  # prepared rows, sorted by sensor and time, with pfire
    eps: pd.DataFrame
    thr_pm: Any  # float, or dict fold -> float
    thr_co: Any


def events_for_method(run: SiteRun, method: str, cfg: Dict[str, Any] = FROZEN) -> pd.DataFrame:
    """Alarm events of one method: columns t (event time used for attribution), t_alarm (time the alarm or the
    fire-like label is shown), sensor, attr (episode id, 'bg' or 'near'), pmax (two-stage only)."""
    df = run.df
    pm_thr = df["thr_pm"].to_numpy()
    co_thr = df["thr_co"].to_numpy()
    if method == "stage1_pm":
        idx = alarm_events(df, df[cfg["stage1_channel"]].to_numpy() > pm_thr, cfg)
        ev = pd.DataFrame({"row": idx, "t": df["ts"].to_numpy()[idx], "sensor": df["sensor"].to_numpy()[idx]})
        ev["t_alarm"] = ev["t"]
        ev["pmax"] = np.nan
    elif method == "co_delta":
        idx = alarm_events(df, df[cfg["co_channel"]].to_numpy() > co_thr, cfg)
        ev = pd.DataFrame({"row": idx, "t": df["ts"].to_numpy()[idx], "sensor": df["sensor"].to_numpy()[idx]})
        ev["t_alarm"] = ev["t"]
        ev["pmax"] = np.nan
    elif method == "ml_alone":
        idx = alarm_events(df, df["pfire"].to_numpy() >= cfg["stage2_p_fire"], cfg)
        ev = pd.DataFrame({"row": idx, "t": df["ts"].to_numpy()[idx], "sensor": df["sensor"].to_numpy()[idx]})
        ev["t_alarm"] = ev["t"]
        ev["pmax"] = df["pfire"].to_numpy()[idx]
    elif method in ("two_stage", "two_stage_all"):
        idx = alarm_events(df, df[cfg["stage1_channel"]].to_numpy() > pm_thr, cfg)
        v = verify_triggers(df, idx, df["pfire"].to_numpy(), cfg)
        if method == "two_stage":
            v = v[v["verified"]]
        ev = pd.DataFrame({"row": v["row"].to_numpy(dtype=int), "t": v["t"].to_numpy(), "sensor": v["sensor"].to_numpy(),
                           "t_alarm": v["t_verified"].to_numpy(), "pmax": v["pmax"].to_numpy(), "verified": v["verified"].to_numpy()})
    else:
        raise KeyError(method)
    cb = df["clean_bg"].to_numpy()[ev["row"].to_numpy(dtype=int)] if len(ev) else np.array([], dtype=bool)
    ev["attr"] = attribute(ev["t"].to_numpy(), run.eps, cb, cfg) if len(ev) else np.array([], dtype=object)
    return ev


def _boot_rate(flags: np.ndarray, rng: np.random.Generator, n_boot: int) -> Tuple[Optional[float], Optional[float]]:
    if len(flags) == 0:
        return None, None
    vals = [flags[rng.integers(0, len(flags), len(flags))].mean() for _ in range(n_boot)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _boot_median(lat: np.ndarray, rng: np.random.Generator, n_boot: int) -> Tuple[Optional[float], Optional[float]]:
    """Bootstrap over fire episodes (NaN = missed); median of the detected ones in each resample."""
    if len(lat) == 0 or np.all(np.isnan(lat)):
        return None, None
    vals = []
    for _ in range(n_boot):
        s = lat[rng.integers(0, len(lat), len(lat))]
        s = s[~np.isnan(s)]
        if len(s):
            vals.append(np.median(s))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (None, None)


def background_hours(df: pd.DataFrame, cfg: Dict[str, Any] = FROZEN) -> pd.Series:
    """Clean-background sensor-hours per recording day (samples x sample_s)."""
    return df.loc[df["clean_bg"]].groupby("day").size() * cfg["sample_s"] / 3600.0


def summarize(run: SiteRun, method: str, ev: pd.DataFrame, cfg: Dict[str, Any] = FROZEN) -> Dict[str, Any]:
    """Episode counts with bootstrap intervals, latency, background alarms per 24 sensor-hours (day-block bootstrap)."""
    rng = np.random.default_rng(cfg["seed"])
    nb = int(cfg["n_boot"])
    eps = run.eps
    first = ev.groupby("attr")["t_alarm"].min() if len(ev) else pd.Series(dtype="float64")
    out: Dict[str, Any] = {"method": method, "label": METHOD_LABEL.get(method, method)}
    lat_rows = []
    for kind in ("fire", "nuisance", "other"):
        e = eps[eps["kind"] == kind]
        flags = np.array([ep in first.index for ep in e["ep"]], dtype=float)
        lo, hi = _boot_rate(flags, rng, nb)
        out[kind] = {"n": int(len(e)), "alarmed": int(flags.sum()), "rate": (float(flags.mean()) if len(e) else None),
                     "ci95": [lo, hi]}
        if kind == "fire":
            for ep, st in zip(e["ep"], e["start_s"]):
                lat_rows.append((float(first[ep]) - st) / 60.0 if ep in first.index else np.nan)
    lat = np.array(lat_rows, dtype=float)
    det = lat[~np.isnan(lat)]
    lo, hi = _boot_median(lat, rng, nb)
    out["latency_min"] = {"median": (float(np.median(det)) if len(det) else None), "max": (float(np.max(det)) if len(det) else None),
                          "min": (float(np.min(det)) if len(det) else None), "n_detected": int(len(det)), "median_ci95": [lo, hi]}
    hours = background_hours(run.df, cfg)
    bg = ev[ev["attr"] == "bg"] if len(ev) else ev
    per_day = (bg.assign(day=run.df["day"].to_numpy()[bg["row"].to_numpy(dtype=int)]).groupby("day").size()
               if len(bg) else pd.Series(dtype=int))
    days = list(hours.index)
    cnt = np.array([int(per_day.get(d, 0)) for d in days], dtype=float)
    hrs = np.array([float(hours[d]) for d in days], dtype=float)
    rate = float(cnt.sum() / hrs.sum() * 24.0) if hrs.sum() > 0 else None
    boots = []
    for _ in range(nb):
        k = rng.integers(0, len(days), len(days)) if days else np.array([], dtype=int)
        h = hrs[k].sum()
        if h > 0:
            boots.append(cnt[k].sum() / h * 24.0)
    out["background"] = {"alarms": int(cnt.sum()), "sensor_hours": float(hrs.sum()), "per_24_sensor_h": rate,
                         "ci95": ([float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else [None, None]),
                         "n_days": len(days), "near_episode_alarms": int((ev["attr"] == "near").sum()) if len(ev) else 0}
    out["n_events"] = int(len(ev))
    return out


def stage2_auroc(run: SiteRun, cfg: Dict[str, Any] = FROZEN) -> Dict[str, Any]:
    """AUROC of the stage-2 score over every stage-1 trigger: positives = triggers attributed to fire episodes,
    negatives = triggers attributed to nuisance or other episodes or to clean background; near-episode triggers are
    excluded. Cluster bootstrap over episodes and background days."""
    from sklearn.metrics import roc_auc_score

    ev = events_for_method(run, "two_stage_all", cfg)
    kind_by_ep = dict(zip(run.eps["ep"], run.eps["kind"]))
    ev = ev[ev["attr"] != "near"].copy()
    ev["y"] = ev["attr"].map(lambda a: 1 if kind_by_ep.get(a) == "fire" else 0)
    ev["cluster"] = np.where(ev["attr"] == "bg", "bg-" + run.df["day"].to_numpy()[ev["row"].to_numpy(dtype=int)].astype(str), ev["attr"])
    npos, nneg = int(ev["y"].sum()), int((1 - ev["y"]).sum())
    res: Dict[str, Any] = {"n_triggers": int(len(ev)), "n_pos": npos, "n_neg": nneg, "auroc": None, "ci95": [None, None]}
    if npos == 0 or nneg == 0:
        return res
    res["auroc"] = float(roc_auc_score(ev["y"], ev["pmax"]))
    rng = np.random.default_rng(cfg["seed"])
    groups = {c: g for c, g in ev.groupby("cluster")}
    keys = list(groups)
    vals = []
    for _ in range(int(cfg["n_boot"])):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        s = pd.concat([groups[k] for k in pick])
        if s["y"].nunique() == 2:
            vals.append(roc_auc_score(s["y"], s["pmax"]))
    if vals:
        res["ci95"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    res["label_counts"] = {"fire_triggers_labelled_fire": int(((ev["y"] == 1) & ev["verified"]).sum()),
                           "nonfire_triggers_labelled_fire": int(((ev["y"] == 0) & ev["verified"]).sum())}
    return res


def episode_table(run: SiteRun, events: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per episode: kind, scenario, start, end and minutes from start to the first alarm per method."""
    rows = []
    for e in run.eps.itertuples():
        r = {"ep": e.ep, "site": e.site, "kind": e.kind, "scenario": e.scenario, "start": e.start.isoformat(), "end": e.end.isoformat(),
             "duration_min": round((e.end - e.start).total_seconds() / 60, 2)}
        for m, ev in events.items():
            hit = ev[ev["attr"] == e.ep] if len(ev) else ev
            r[f"{m}_first_min"] = (round((float(hit["t_alarm"].min()) - e.start_s) / 60, 2) if len(hit) else None)
        rows.append(r)
    return pd.DataFrame(rows)


def evaluate_run(run: SiteRun, cfg: Dict[str, Any] = FROZEN) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    events = {m: events_for_method(run, m, cfg) for m in METHODS}
    res = {m: summarize(run, m, events[m], cfg) for m in METHODS}
    res["stage2_auroc"] = stage2_auroc(run, cfg)
    return res, events


def cv_en54(df: pd.DataFrame, eps: pd.DataFrame, features: Sequence[str], cfg: Dict[str, Any] = FROZEN,
            log=print) -> Tuple[SiteRun, List[Dict[str, Any]]]:
    """Leave-one-recording-day-out on EN54: thresholds and the model come from the other days only."""
    parts, folds = [], []
    for day in sorted(df["day"].unique()):
        tr = df[df["day"] != day]
        te = df[df["day"] == day].copy()
        thr_pm = stage1_threshold(tr, cfg["stage1_channel"], cfg["stage1_quantile"])
        thr_co = stage1_threshold(tr, cfg["co_channel"], cfg["stage1_quantile"])
        clf, info = train_stage2(tr, features, cfg)
        te["pfire"] = p_fire(clf, te, features)
        te["thr_pm"], te["thr_co"] = thr_pm, thr_co
        te["fold"] = day
        parts.append(te)
        folds.append({"test_day": day, "thr_pm": thr_pm, "thr_co": thr_co, **info, "n_test_rows": int(len(te))})
        log(f"  fold {day}: thr_pm {thr_pm:.3f} thr_co {thr_co:.3f} train rows {info['n_train_rows']}")
    pooled = pd.concat(parts).sort_values(["sensor", "t"], kind="stable").reset_index(drop=True)
    return SiteRun(df=pooled, eps=eps, thr_pm={f["test_day"]: f["thr_pm"] for f in folds},
                   thr_co={f["test_day"]: f["thr_co"] for f in folds}), folds


def replay_frames(run: SiteRun, events: Dict[str, pd.DataFrame], pad_min: float = 10.0) -> pd.DataFrame:
    """Per episode, the series of the sensor node that triggered first (or with the largest PM rise): minutes from the
    label start, PM and CO deltas, P(fire), stage-1 trigger and two-stage fire-like label markers."""
    df = run.df
    s1 = events["stage1_pm"]
    ts = events["two_stage"]
    out = []
    for e in run.eps.itertuples():
        hit = s1[s1["attr"] == e.ep]
        if len(hit):
            sensor = hit.sort_values("t")["sensor"].iloc[0]
        else:
            inside = df[df["ep"] == e.ep]
            sensor = inside.loc[inside["PM_Total_Room_d"].idxmax(), "sensor"] if inside["PM_Total_Room_d"].notna().any() else inside["sensor"].iloc[0]
        lo = e.start_s - pad_min * 60
        hi = e.end_s + pad_min * 60
        g = df[(df["sensor"] == sensor) & (df["ts"] >= lo) & (df["ts"] <= hi)]
        trig_t = set(s1.loc[s1["sensor"] == sensor, "t"].astype(float))
        ver_t = set(ts.loc[ts["sensor"] == sensor, "t_alarm"].astype(float))
        out.append(pd.DataFrame({
            "ep": e.ep, "kind": e.kind, "scenario": e.scenario, "sensor": sensor,
            "min_from_start": ((g["ts"] - e.start_s) / 60).round(3).to_numpy(),
            "pm_total_delta": g["PM_Total_Room_d"].round(2).to_numpy(), "co_delta": g["CO_Room_d"].round(3).to_numpy(),
            "p_fire": g["pfire"].round(3).to_numpy(),
            "stage1_trigger": g["ts"].isin(trig_t).to_numpy(), "fire_like_label": g["ts"].isin(ver_t).to_numpy(),
            "end_min": round((e.end_s - e.start_s) / 60, 2)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def json_safe(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, np.bool_):
        return bool(o)
    return o
