"""Build every LA rain-exposure artifact from the cached REAL data in data/raw/rain (see fetch_la_rain_data.py).

Parts (run all by default; each writes to eval/rain/ or models/rain/):
  qc      IEM hourly vs GHCN-Daily per local-standard day; spike days zeroed; per-water-year ratios -> qa_station_wy.json
  roses   per-station rain roses, all-hours wind roses, ISO 15927-3 airfield index per facade and per 1-degree azimuth,
          circular statistics, Rayleigh test on storm directions, block permutation test rain vs dry wind,
          most-exposed facade per water year, sensitivity runs                     -> station_roses.json
  events  events segmented on ERA5 (and GFS day-1) precipitation, features, ASOS targets
                                                                                    -> events_era5.parquet, events_gfs.parquet
  model   temporal holdout, leave-one-station-out, leave-station-and-period-out, forecast-lead test; final models
                                                                                    -> metrics.json, models/rain/*.joblib
  map     ERA5 0.25 deg cells: modelled annual per-facade index                     -> grid_exposure.json
  normals NCEI 1991-2020 annual/seasonal precipitation normals and elevation correlation -> normals.json
  summary headline numbers for the page                                             -> summary.json

Usage: python scripts/build_la_rain.py [--parts qc,roses,events,model,map,normals,summary] [--n-boot 2000]
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cascade.building import rainexposure as rx  # noqa: E402
from cascade.building import rainmodel as rm  # noqa: E402

RAW = ROOT / "data" / "raw" / "rain"
CACHE = RAW / "derived"
EVAL = ROOT / "eval" / "rain"
MODELS = ROOT / "models" / "rain"
STATIONS = list(rx.STATIONS)
WY_ALL = list(range(2006, 2026))
STORM_MIN_MM, STORM_GAP_H = 5.0, 6.0
try:  # read (not edit) the shared registry so the storm rule stays in one place
    from cascade.building.rules import t as _t
    STORM_MIN_MM, STORM_GAP_H = float(_t("STORM_MIN_MM")), float(_t("STORM_GAP_H"))
except Exception:  # noqa: BLE001
    pass


def read_json(path: Path):
    """JSON from disk; pilot caches were written in the Windows code page (degree sign), so fall back to latin-1."""
    b = path.read_bytes()
    try:
        return json.loads(b.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(b.decode("latin-1"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
            return o.isoformat()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    def clean(o):
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o

    path.write_text(json.dumps(clean(json.loads(json.dumps(obj, default=conv))), indent=1), encoding="utf-8")


def r3(x, k=3):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), k)


# ----------------------------------------------------------------------------------------- QC

def clean_asos(sid: str) -> pd.DataFrame:
    p = CACHE / f"asos_{sid}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    raise FileNotFoundError(p)


def part_qc():
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {"rule": {k: rx.RULES[k] for k in ("P01I_M_SWITCH", "SPIKE_DAY_MM", "QC_RATIO_BAND", "QC_MIN_GHCND_MM", "CQT_END")},
           "stations": {}}
    for sid in STATIONS:
        d = rx.load_asos_csv(RAW / "iem" / f"{sid}.csv", sid)
        g = rx.load_ghcnd_csv(RAW / "ghcnd" / f"{sid}.csv")
        c, tab = rx.qc_daily(d, g)
        c.to_parquet(CACHE / f"asos_{sid}.parquet")
        wy = rx.qc_water_years(tab)
        wyi = rx.water_year(c.index)
        hours = pd.Series(1, index=c.index).groupby(wyi).size()
        wy["hours"] = wy["wy"].map(hours).fillna(0).astype(int)
        wy["m_as_zero_share"] = wy["wy"].map(pd.Series(d["p_missing"].to_numpy(), index=d.index).groupby(wyi).mean()).round(3)
        agg_i, agg_g = float(wy["iem_mm"].sum()), float(wy["ghcnd_mm"].sum())
        spikes = tab[tab["spike"]]
        out["stations"][sid] = {
            "ghcnd": rx.STATIONS[sid]["ghcnd"], "n_spike_days": int(len(spikes)),
            "spike_days": [str(x.date()) for x in spikes.index[:50]],
            "aggregate_ratio": r3(agg_i / agg_g if agg_g > 0 else float("nan")),
            "wy": [{"wy": int(r.wy), "iem_mm": r3(r.iem_mm, 1), "ghcnd_mm": r3(r.ghcnd_mm, 1), "ratio": r3(r.ratio),
                    "spikes": int(r.spikes), "hours": int(r.hours), "m_as_zero_share": r3(r.m_as_zero_share),
                    "passed": bool(r.passed), "reason": r.reason} for r in wy.itertuples()],
            "passed_wy": [int(x) for x in wy.loc[wy["passed"], "wy"]],
            "failed_wy": [int(x) for x in wy.loc[~wy["passed"], "wy"]],
        }
        log("qc", sid, "spikes", len(spikes), "failed WY", out["stations"][sid]["failed_wy"])
    jdump(out, EVAL / "qa_station_wy.json")
    return out


def passed_wys(sid: str) -> list:
    q = json.loads((EVAL / "qa_station_wy.json").read_text(encoding="utf-8"))
    return q["stations"][sid]["passed_wy"]


# ----------------------------------------------------------------------------------------- roses

def station_rose(sid: str, a: pd.DataFrame, wys: list, n_perm: int) -> dict:
    wyi = rx.water_year(a.index)
    a = a[np.isin(wyi, wys)]
    wyi = rx.water_year(a.index)
    ny = len(np.unique(wyi))
    iso = rx.iso_frame(a)
    base = rx.ISO_COEF * a["v"].fillna(0).clip(lower=0) * np.power(a["r"].fillna(0).clip(lower=0), rx.ISO_R_EXP)
    rain = a["r"] > 0
    nc = a["D"].notna()
    good = rain & nc
    ia = rx.annual_index(iso, ny)
    share = ia / ia.sum()
    top, least = str(ia.idxmax()), str(ia.idxmin())
    top3 = list(ia.sort_values(ascending=False).index[:3])
    # 1-degree curve
    wr = (a["r"] > 0).to_numpy() & a["D"].notna().to_numpy()
    curve = rx.iso_hourly(a["v"].to_numpy()[wr], a["r"].to_numpy()[wr], a["D"].to_numpy()[wr], np.arange(360)).sum(0) / ny
    # directions
    md, rb = rx.weighted_circmean(a.loc[good, "D"], a.loc[good, "r"])
    wd, wb = rx.weighted_circmean(a.loc[good, "D"], base[good])
    ad, ab = rx.weighted_circmean(a.loc[nc, "D"])
    dd, db = rx.weighted_circmean(a.loc[nc & ~rain, "D"])
    # sector shares
    rain_share = rx.sector_shares(a.loc[good, "D"], a.loc[good, "r"])
    all_share = rx.sector_shares(a.loc[nc, "D"])
    wdr_share = rx.sector_shares(a.loc[good, "D"], base[good])
    calm_rain_frac = float(a.loc[rain & a["calm"], "r"].sum() / max(a.loc[rain, "r"].sum(), 1e-9))
    calm_all_frac = float(a["calm"].mean())
    # per water year
    by_wy = iso.groupby(wyi).sum()
    top_wy = {int(k): str(v) for k, v in by_wy.idxmax(axis=1).items() if by_wy.loc[k].sum() > 0}
    # season and intensity
    mon = a.index.month
    novmar = float(base[np.isin(mon, [11, 12, 1, 2, 3])].sum() / max(base.sum(), 1e-9))
    lo, hi = rx.rule("HEAVY_MM_H")
    by_int = {}
    for name, m in {"<2.5 mm/h": good & (a["r"] < lo), "2.5-6 mm/h": good & (a["r"] >= lo) & (a["r"] < hi),
                    ">=6 mm/h": good & (a["r"] >= hi)}.items():
        dd_, rr_ = rx.weighted_circmean(a.loc[m, "D"], a.loc[m, "r"])
        by_int[name] = {"hours": int(m.sum()), "mean_dir": r3(dd_, 1), "rbar": r3(rr_),
                        "share_of_driving_rain": r3(float(base[m].sum() / max(base.sum(), 1e-9)))}
    # storms (ASOS, >= STORM_MIN_MM, STORM_GAP_H)
    ev = rx.segment_events(a["r"], gap_h=STORM_GAP_H, min_mm=STORM_MIN_MM)
    sdirs, stops, stot = [], [], []
    cs = np.vstack([np.zeros((1, 8)), np.cumsum(iso.to_numpy(), 0)])
    for e in ev.itertuples():
        i0 = a.index.searchsorted(e.start, "left")
        i1 = a.index.searchsorted(e.end, "right")
        g = a.iloc[i0:i1]
        gg = g[(g["r"] > 0) & g["D"].notna()]
        if len(gg):
            sdirs.append(rx.weighted_circmean(gg["D"], gg["r"])[0])
        tot = cs[i1] - cs[i0]
        stot.append(float(tot.sum()))
        stops.append(rx.FACADES[int(np.argmax(tot))] if tot.sum() > 0 else None)
    ray = rx.rayleigh(sdirs)
    sdm, sdb = rx.weighted_circmean(sdirs)
    stot_s = np.sort(np.asarray(stot))[::-1]
    k10 = max(1, int(round(0.1 * len(stot_s)))) if len(stot_s) else 0
    top10 = float(stot_s[:k10].sum() / stot_s.sum()) if len(stot_s) and stot_s.sum() > 0 else float("nan")
    # block permutation: rain episodes (any rain, STORM_GAP_H) vs fully dry local days
    epi = rx.segment_events(a["r"], gap_h=STORM_GAP_H, min_mm=0.0)
    block = np.full(len(a), -1, dtype=np.int64)
    for e in epi.itertuples():
        i0 = a.index.searchsorted(e.start, "left")
        i1 = a.index.searchsorted(e.end, "right")
        seg = np.arange(i0, i1)
        seg = seg[(a["r"].to_numpy()[seg] > 0)]
        block[seg] = e.event
    nrain = int(len(epi))
    day = rx.lst_day(a.index)
    dfd = pd.DataFrame({"day": day, "r": a["r"].to_numpy()})
    dfd["wet_or_missing"] = dfd["r"].fillna(1.0) > 0
    daywet = dfd.groupby("day")["wet_or_missing"].any()
    dmask = (~dfd["day"].map(daywet).to_numpy(dtype=bool)) & (block < 0)
    dcode = pd.factorize(pd.Index(day)[dmask])[0] + nrain
    block[np.flatnonzero(dmask)] = dcode
    use = (block >= 0) & a["D"].notna().to_numpy()
    bs = rx.block_sums(a["D"][use], block[use])
    bs = bs[bs["n"] > 0]
    perm = rx.block_permutation_test(bs["sx"], bs["sy"], bs["n"], bs.index.to_numpy() < nrain, n_perm=n_perm, seed=0)
    # spells for the most-exposed facade
    sp = rx.spells(iso[top])
    sp_wy = sp.groupby(rx.water_year(pd.DatetimeIndex(sp["start"])))["total"].max() if len(sp) else pd.Series(dtype=float)
    # sensitivity
    def top_for(mask_wy):
        m = np.isin(rx.water_year(iso.index), mask_wy)
        s = iso[m].sum()
        return str(s.idxmax()) if s.sum() > 0 else None
    meta = rx.STATIONS[sid]
    return {
        "station": sid, **{k: meta[k] for k in ("name", "lat", "lon", "elev", "region")},
        "label": "REAL (measured, IEM ASOS routine hourly, QC'd against GHCN-Daily)",
        "years_used": int(ny), "wy_used": [int(x) for x in sorted(np.unique(wyi))], "hours": int(len(a)),
        "hours_rain": int(rain.sum()), "mean_annual_mm": r3(float(a["r"].sum()) / ny, 1),
        "rain_w_mean_dir": r3(md, 1), "rain_w_rbar": r3(rb), "wdr_w_mean_dir": r3(wd, 1), "wdr_w_rbar": r3(wb),
        "allhours_mean_dir": r3(ad, 1), "allhours_rbar": r3(ab), "dry_mean_dir": r3(dd, 1), "dry_rbar": r3(db),
        "v_rain_ms": r3(a.loc[rain, "v"].mean(), 2), "v_dry_ms": r3(a.loc[~rain, "v"].mean(), 2),
        "calm_share_of_rain": r3(calm_rain_frac), "calm_share_of_hours": r3(calm_all_frac),
        "sector_rain_share": {k: r3(v) for k, v in rain_share.items()},
        "sector_allhours_share": {k: r3(v) for k, v in all_share.items()},
        "sector_wdr_share": {k: r3(v) for k, v in wdr_share.items()},
        "IA_L_m2_yr": {k: r3(v, 2) for k, v in ia.items()}, "IA_share": {k: r3(v) for k, v in share.items()},
        "top_facade": top, "least_facade": least, "max_min_ratio": r3(ia.max() / max(ia.min(), 1e-9), 1),
        "top3_facades": top3, "top3_share": r3(float(ia.sort_values().iloc[-3:].sum() / ia.sum())),
        "IA_curve_1deg": [round(float(x), 3) for x in curve], "curve_max_azimuth": int(np.argmax(curve)),
        "top_facade_by_wy": top_wy, "top_facade_wy_counts": pd.Series(list(top_wy.values())).value_counts().to_dict(),
        "novmar_share_of_driving_rain": r3(novmar), "by_intensity": by_int,
        "storms": {"n": int(len(ev)), "rule": f">= {STORM_MIN_MM} mm, split by >= {STORM_GAP_H} dry hours",
                   "mean_dir": r3(sdm, 1), "rayleigh": {k: (r3(v, 4) if k != "p" else v) for k, v in ray.items()},
                   "top_facade_counts": pd.Series([s for s in stops if s]).value_counts().to_dict(),
                   "top10pct_share_of_storm_index": r3(top10)},
        "rain_vs_dry_permutation": {**{k: (r3(v, 4) if isinstance(v, float) else v) for k, v in perm.items()},
                                    "blocks": "rain episodes (any rain, split by >= 6 dry h) vs fully dry local days",
                                    "statistic": "distance between mean unit wind vectors (non-calm hours)"},
        "spells_top_facade": {"n_spells": int(len(sp)), "max_spell_L_m2": r3(sp["total"].max() if len(sp) else float("nan"), 2),
                              "median_wy_max_spell_L_m2": r3(sp_wy.median() if len(sp_wy) else float("nan"), 2),
                              "note": "ISO 3-year-return spell index not computed; per-water-year maxima shown"},
        "sensitivity": {"top_facade_all_20wy": top_for(WY_ALL), "top_facade_wy2013_2025": top_for(list(range(2013, 2026)))},
    }


def part_roses(n_perm: int):
    out = {"method": "ISO 15927-3 method 1: I_A = (2/9) sum U10 Rh^(8/9) cos(D - theta) / N over hours with cos > 0 "
                     "(Blocken & Carmeliet 2010 eq. 5); C_T and W not applied",
           "labels": rx.LABELS, "facades": list(rx.FACADES), "stations": {}}
    for sid in STATIONS:
        t = time.time()
        ent = station_rose(sid, clean_asos(sid), passed_wys(sid), n_perm)
        out["stations"][sid] = ent
        log("rose", sid, ent["top_facade"], "years", ent["years_used"], "rayleigh p", ent["storms"]["rayleigh"]["p"],
            "perm p", ent["rain_vs_dry_permutation"]["p"], f"{time.time() - t:.0f}s")
    jdump(out, EVAL / "station_roses.json")
    return out


# ----------------------------------------------------------------------------------------- events

def era5_station(sid: str) -> pd.DataFrame:
    return rm.open_meteo_frame(read_json(RAW / "era5" / f"{sid}.json"))


def part_events():
    parts, counts = [], {}
    gparts = []
    for sid in STATIONS:
        m = rx.STATIONS[sid]
        a = clean_asos(sid)
        e5 = era5_station(sid)
        df, c = rm.build_event_table(sid, e5, a, m["lat"], m["lon"], m["elev"], passed_wys(sid),
                                     gap_h=STORM_GAP_H, min_mm=STORM_MIN_MM, source="era5")
        parts.append(df)
        counts[sid] = {"era5": c}
        gp = RAW / "prev_gfs" / f"{sid}.json"
        if gp.exists():
            gf = rm.open_meteo_frame(read_json(gp), suffix="_previous_day1")
            dg, cg = rm.build_event_table(sid, gf, a, m["lat"], m["lon"], m["elev"], passed_wys(sid),
                                          gap_h=STORM_GAP_H, min_mm=STORM_MIN_MM, source="gfs_day1")
            gparts.append(dg)
            counts[sid]["gfs_day1"] = cg
        log("events", sid, counts[sid])
    ev = pd.concat(parts, ignore_index=True)
    ev.to_parquet(EVAL / "events_era5.parquet", index=False)
    if gparts:
        pd.concat(gparts, ignore_index=True).to_parquet(EVAL / "events_gfs.parquet", index=False)
    # ERA5 vs ASOS agreement per station
    comp = {}
    for sid, g in ev.groupby("station"):
        d10 = np.degrees(np.arctan2(g["u10"], g["v10"])) % 360
        d100 = np.degrees(np.arctan2(g["u100"], g["v100"])) % 360
        wet = g["asos_mm"] >= 1.0
        comp[sid] = {"n_events": int(len(g)), "corr_era5_asos_mm": r3(np.corrcoef(g["p_tot"], g["asos_mm"])[0, 1]),
                     "sum_ratio_era5_over_asos": r3(g["p_tot"].sum() / max(g["asos_mm"].sum(), 1e-9)),
                     "false_alarm_share_asos_lt_1mm": r3(float((~wet).mean())),
                     "era5_10m_mean_dir_rainweighted": r3(rx.weighted_circmean(d10, g["p_tot"])[0], 1),
                     "era5_100m_mean_dir_rainweighted": r3(rx.weighted_circmean(d100, g["p_tot"])[0], 1),
                     "median_abs_dir_diff_10m_100m": r3(float(np.median(np.abs(rx.angdiff(d10, d100)))), 1)}
    jdump({"counts": counts, "era5_vs_asos": comp,
           "rule": f"events on ERA5/GFS precipitation: wet hour >= {rx.rule('WET_HOUR_MM')} mm, split by >= {STORM_GAP_H} h, "
                   f"total >= {STORM_MIN_MM} mm; target window +/- {rx.rule('EVENT_PAD_H')} h; ASOS coverage >= 0.8"},
          EVAL / "events_summary.json")
    return ev


# ----------------------------------------------------------------------------------------- model

def rounded(d):
    if isinstance(d, dict):
        return {k: rounded(v) for k, v in d.items()}
    if isinstance(d, float):
        return r3(d, 4)
    return d


def part_model(n_boot: int):
    import joblib
    import sklearn
    ev = pd.read_parquet(EVAL / "events_era5.parquet")
    ev["start"] = pd.to_datetime(ev["start"], utc=True)
    res = {"unit": "one event at one station (events segmented on ERA5 precipitation, so false alarms count)",
           "target": "ASOS-measured ISO 15927-3 storm index per facade (L/m2 at 10 m, airfield terrain), summed over the "
                     "event window +/- 3 h",
           "label": "REAL held-out evaluation on measured data; ERA5 is reanalysis (perfect-prognosis stand-in for a forecast)",
           "features_full": rm.FEATS_FULL, "features_10m": rm.FEATS_10M, "hgb_params": rm.HGB_PARAMS,
           "n_events_total": int(len(ev)), "stations": sorted(ev["station"].unique().tolist()),
           "wy_range": [int(ev["wy"].min()), int(ev["wy"].max())]}
    t = time.time()
    tr, te = ev[ev["wy"] <= 2022], ev[ev["wy"] >= 2023]
    res["temporal"] = {"split": "train WY2016-2022, test WY2023-2025", **rounded(rm.evaluate_split(tr, te, rm.FEATS_FULL, n_boot))}
    log("temporal", f"{time.time() - t:.0f}s", res["temporal"]["methods"]["HGB_model"]["MAE_L_m2"])
    res["temporal_10m"] = {"split": "train WY2016-2022, test WY2023-2025 (10 m features only)",
                           **rounded(rm.evaluate_split(tr, te, rm.FEATS_10M, n_boot))}
    folds = [((ev["station"] != s).to_numpy(), (ev["station"] == s).to_numpy()) for s in STATIONS]
    lo, _ = rm.evaluate_grouped(ev, rm.FEATS_FULL, folds, n_boot)
    res["loso"] = {"split": "leave one station out (12 folds, all water years)", **rounded(lo)}
    log("loso", f"{time.time() - t:.0f}s", res["loso"]["methods"]["HGB_model"]["MAE_L_m2"])
    folds2 = [(((ev["station"] != s) & (ev["wy"] <= 2022)).to_numpy(), ((ev["station"] == s) & (ev["wy"] >= 2023)).to_numpy())
              for s in STATIONS]
    lspo, _ = rm.evaluate_grouped(ev, rm.FEATS_FULL, folds2, n_boot)
    res["lspo"] = {"split": "leave station AND period out: train other stations WY2016-2022, test held-out station WY2023-2025",
                   **rounded(lspo)}
    log("lspo", f"{time.time() - t:.0f}s")
    gp = EVAL / "events_gfs.parquet"
    if gp.exists():
        eg = pd.read_parquet(gp)
        eg["start"] = pd.to_datetime(eg["start"], utc=True)
        tr10 = ev[ev["start"] < pd.Timestamp("2023-10-01", tz="UTC")]
        teg = eg[eg["start"] >= pd.Timestamp("2024-01-01", tz="UTC")]
        fl = rm.evaluate_split(tr10, teg, rm.FEATS_10M, n_boot)
        # residual band of the 10 m model on the forecast-lead test (for live-panel uncertainty)
        P = rm.FacadeModel(rm.FEATS_10M).fit(tr10).predict(teg)
        resid = (teg[rm.YCOLS].to_numpy() - P).ravel()
        fl["residual_quantiles_L_m2"] = {q: float(np.quantile(resid, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
        res["forecast_lead"] = {"split": "10 m model trained on ERA5 events WY2016-2023; tested on events segmented on real "
                                         "GFS day-1 forecasts (Open-Meteo Previous Runs), 2024-01-01 to 2025-09-30",
                                **rounded(fl)}
        te_ref = ev[ev["start"] >= pd.Timestamp("2024-01-01", tz="UTC")]
        res["forecast_period_era5_reference"] = {
            "split": "same 10 m model and training set, tested on ERA5 events in the same period (perfect-prognosis reference)",
            **rounded(rm.evaluate_split(tr10, te_ref, rm.FEATS_10M, n_boot))}
        log("forecast", f"{time.time() - t:.0f}s")
    # final models on all events
    MODELS.mkdir(parents=True, exist_ok=True)
    for name, feats in (("hgb_era5_full", rm.FEATS_FULL), ("hgb_10m", rm.FEATS_10M)):
        mdl = rm.FacadeModel(feats).fit(ev)
        joblib.dump({"model": mdl.models, "features": feats, "facades": list(rx.FACADES), "sklearn": sklearn.__version__,
                     "trained_on": f"{len(ev)} ERA5 events at 12 LA ASOS stations, WY{res['wy_range'][0]}-{res['wy_range'][1]}",
                     "target": res["target"]}, MODELS / f"{name}.joblib", compress=3)
    res["model_files"] = {n: f"models/rain/{n}.joblib" for n in ("hgb_era5_full", "hgb_10m")}
    res["sklearn_version"] = sklearn.__version__
    res["seconds"] = round(time.time() - t, 1)
    jdump(res, EVAL / "metrics.json")
    return res


# ----------------------------------------------------------------------------------------- map

def _cell(x):
    return round(round(float(x) * 4) / 4, 2)


def part_map():
    import joblib
    bundle = joblib.load(MODELS / "hgb_10m.joblib")
    mdl = rm.FacadeModel(bundle["features"])
    mdl.models = bundle["model"]
    cells = {}
    for p in sorted((RAW / "era5").glob("*.json")):
        j = read_json(p)
        key = (_cell(j["latitude"]), _cell(j["longitude"]))
        cells.setdefault(key, {"json": j, "stations": []})["stations"].append(p.stem)
    for p in sorted((RAW / "era5_grid").glob("*.json")) if (RAW / "era5_grid").exists() else []:
        j = read_json(p)
        key = (_cell(j["latitude"]), _cell(j["longitude"]))
        cells.setdefault(key, {"json": j, "stations": []})
    max_elev = max(v["elev"] for v in rx.STATIONS.values())
    rows = []
    for (la, lo), c in sorted(cells.items()):
        j = c["json"]
        src = rm.open_meteo_frame(j)
        ny = len(np.unique(rx.water_year(src.index)))
        elev = float(j.get("elevation", float("nan")))
        ev, _ = rm.build_event_table("cell", src, None, la, lo, elev, gap_h=STORM_GAP_H, min_mm=STORM_MIN_MM)
        if len(ev) == 0:
            continue
        P = mdl.predict(ev).sum(0) / ny
        B1 = ev[rm.B1COLS].to_numpy().sum(0) / ny
        top = rx.FACADES[int(np.argmax(P))]
        rows.append({"lat": la, "lon": lo, "elev_m": r3(elev, 0), "stations_in_cell": c["stations"], "n_events": int(len(ev)),
                     "years": int(ny), "model_IA": {f: r3(P[i], 2) for i, f in enumerate(rx.FACADES)},
                     "physics_IA": {f: r3(B1[i], 2) for i, f in enumerate(rx.FACADES)},
                     "top_facade_model": top, "top_facade_physics": rx.FACADES[int(np.argmax(B1))],
                     "top_share_model": r3(P.max() / max(P.sum(), 1e-9)),
                     "outside_training_elevation": bool(elev > max_elev + 50),
                     "label": "MODELLED (ERA5 0.25 deg + HGB 10 m model); expected skill = leave-one-station-out metrics"})
    jdump({"cells": rows, "n_cells": len(rows), "grid_step_deg": 0.25, "max_training_elev_m": max_elev,
           "shared_cells": [r["stations_in_cell"] for r in rows if len(r["stations_in_cell"]) > 1],
           "note": f"Cells more than 50 m above the highest training station ({max_elev:.0f} m) are mountain terrain outside "
                   "the training range and may be outside ISO 15927-3 scope (sheer cliffs, gorges)."},
          EVAL / "grid_exposure.json")
    log("map cells", len(rows))
    return rows


# ----------------------------------------------------------------------------------------- normals

def part_normals():
    from scipy.stats import spearmanr
    raw = json.loads((RAW / "ncei_normals_la.json").read_text(encoding="utf-8"))

    def f(x):
        try:
            v = float(str(x).strip())
            return v if v > -999 else None  # NCEI sentinels are -9999, -7777 etc.
        except Exception:  # noqa: BLE001
            return None
    pts = []
    for r in raw:
        ann = f(r.get("ANN-PRCP-NORMAL"))
        if ann is None:
            continue
        pts.append({"station": r["STATION"], "name": str(r.get("NAME", "")).strip(), "lat": f(r.get("LATITUDE")),
                    "lon": f(r.get("LONGITUDE")), "elev_m": f(r.get("ELEVATION")), "ann_in": ann,
                    "djf_in": f(r.get("DJF-PRCP-NORMAL")), "mam_in": f(r.get("MAM-PRCP-NORMAL")),
                    "jja_in": f(r.get("JJA-PRCP-NORMAL")), "son_in": f(r.get("SON-PRCP-NORMAL"))})
    df = pd.DataFrame(pts).dropna(subset=["lat", "lon", "elev_m"])
    df["side"] = np.where(df["lat"] >= 34.45, "north of 34.45 N (Antelope Valley side, approx.)", "south of 34.45 N (basin side, approx.)")
    corr = {}
    for name, g in [("all", df), *list(df.groupby("side"))]:
        if len(g) >= 5:
            s = spearmanr(g["elev_m"], g["ann_in"])
            corr[name] = {"n": int(len(g)), "spearman_rho": r3(s.statistic), "p": float(s.pvalue)}
    djf = df.dropna(subset=["djf_in"])
    season = {"median_djf_share": r3(float((djf["djf_in"] / djf["ann_in"]).median()))} if len(djf) else {}
    south = df["side"].str.startswith("south")
    band_defs = {"basin side below 150 m": south & (df["elev_m"] < 150),
                 "basin side 500 m and higher": south & (df["elev_m"] >= 500),
                 "north of 34.45 N": ~south}
    bands = {k: {"n": int(m.sum()), "median_ann_in": r3(float(df.loc[m, "ann_in"].median()), 2) if m.any() else None,
                 "min_ann_in": r3(float(df.loc[m, "ann_in"].min()), 2) if m.any() else None,
                 "max_ann_in": r3(float(df.loc[m, "ann_in"].max()), 2) if m.any() else None}
             for k, m in band_defs.items()}
    lo_b, hi_b = bands["basin side below 150 m"]["median_ann_in"], bands["basin side 500 m and higher"]["median_ann_in"]
    bands["ratio_high_to_low_basin"] = r3(hi_b / lo_b, 2) if lo_b and hi_b else None
    jdump({"label": "REAL (NOAA NCEI 1991-2020 normals)", "points": df.to_dict("records"), "elevation_correlation": corr,
           "season": season, "bands": bands, "n": int(len(df))}, EVAL / "normals.json")
    log("normals", len(df), corr)


# ----------------------------------------------------------------------------------------- summary

def part_summary():
    ro = json.loads((EVAL / "station_roses.json").read_text(encoding="utf-8"))["stations"]
    basin = [s for s, e in ro.items() if e["region"] in rx.BASIN_REGIONS]
    es = [s for s in basin if ro[s]["top_facade"] in ("E", "SE")]
    wdirs = [s for s in basin if ro[s]["sector_allhours_share"] and max(ro[s]["sector_allhours_share"], key=lambda k: ro[s]["sector_allhours_share"][k] or 0) in ("W", "SW", "S")]
    out = {"basin_stations": basin, "basin_top_E_or_SE": es, "n_basin": len(basin), "n_basin_E_SE": len(es),
           "headline_ok": len(es) > len(basin) / 2,
           "basin_most_common_allhours_sector_W_SW_S": wdirs,
           "top_by_station": {s: e["top_facade"] for s, e in ro.items()},
           "region_by_station": {s: e["region"] for s, e in ro.items()},
           "rayleigh_p_range": [min(e["storms"]["rayleigh"]["p"] for e in ro.values()),
                                max(e["storms"]["rayleigh"]["p"] for e in ro.values())],
           "perm_p_max": max(e["rain_vs_dry_permutation"]["p"] for e in ro.values()),
           "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")}
    mp = EVAL / "metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        out["temporal_mae"] = {k: v["MAE_L_m2"] for k, v in m["temporal"]["methods"].items()}
        out["loso_mae"] = {k: v["MAE_L_m2"] for k, v in m["loso"]["methods"].items()}
    jdump(out, EVAL / "summary.json")
    log("summary", out["n_basin_E_SE"], "/", out["n_basin"], "basin stations E/SE")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="qc,roses,events,model,map,normals,summary")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    a = ap.parse_args(argv)
    EVAL.mkdir(parents=True, exist_ok=True)
    for p in a.parts.split(","):
        t = time.time()
        log("== part", p)
        if p == "qc":
            part_qc()
        elif p == "roses":
            part_roses(a.n_perm)
        elif p == "events":
            part_events()
        elif p == "model":
            part_model(a.n_boot)
        elif p == "map":
            part_map()
        elif p == "normals":
            part_normals()
        elif p == "summary":
            part_summary()
        else:
            raise SystemExit(f"unknown part {p}")
        log("== done", p, f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
