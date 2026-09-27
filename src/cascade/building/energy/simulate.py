"""Policy replays.

(A) REAL replay on ROBOD held-out test days: policy kWh = sum(recorded 5-min lighting kWh x level). Savings are
    "recoverable kWh vs as-operated" (what the room really used), so a policy can only remove recorded use.
    Policies: as-operated, fixed schedule 07-22, ideal sensor (ground truth + hold: an UPPER BOUND, under-lit 0 by
    construction), motion sensor (INJECTED detections at UCI 864 PIR rates), sensor+ML (ML may only add on-time) and
    ML-only (unsafe counterfactual). The p95-power 24/7 case is kept only as a labelled hypothetical.
(B) SEMI-SYNTHETIC tower year 2025 in America/Los_Angeles: inventory x (ROBOD test-day blocks of presence and M1
    probability, resampled jointly so ML errors carry over) x hypothetical DR days picked from Open-Meteo 2025
    REANALYSIS temperatures. Lighting kW = level x design W ([ASSUMPTION] W). Priced on LADWP A-2 Rate B.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from . import tariff
from .data import LIGHT, Y
from .inventory import CommonZone
from .occupancy import inject_detection
from .policy import lighting_levels, safety_counts, switch_count
from .rules import life_safety, zone_rule

STEP_MIN = 5
STEPS_DAY = 288
SCHEDULE_HOURS = (7, 22)  # fixed-schedule policy: on 07:00-21:59 [team-proposed baseline]
ML_THRESHOLD = 0.5  # fixed a priori, not tuned
WHAT_IF_POLICIES = ("sensor", "sensor_ml", "sensor_ml_union")
POLICY_LABELS = {
    "as_operated": "As operated (recorded)",
    "always_on": "Always on 24/7",
    "schedule": "Fixed schedule 07-22",
    "ideal_sensor": "Ideal sensor (upper bound)",
    "sensor": "Motion sensor + 15 min hold",
    "sensor_ml": "Sensor + ML (ML may only extend a hold)",
    "sensor_ml_union": "Sensor + ML union (ML may also switch on)",
    "ml_only": "ML only (unsafe counterfactual)",
}
PROFILES = {"room_rule": "amenity", "corridor_rule": "corridor"}
REPLAY_DESIGN_FC = 10.0  # [ASSUMPTION] only used for the egress-floor clamp in the replay


def _kw_from_kwh5(kwh: np.ndarray) -> np.ndarray:
    return kwh / (STEP_MIN / 60.0)


PEAK_WINDOWS = [tuple(w) for w in tariff.LADWP_A2B["periods_weekday"]["high_peak"]]  # LADWP weekday high-peak hours


def _peak_kw(kwh: np.ndarray, ts: pd.DatetimeIndex) -> float:
    """Max 15-min mean kW on weekdays inside the LADWP high-peak window(s) read from the tariff table."""
    s = pd.Series(_kw_from_kwh5(kwh), index=ts).resample("15min").mean().dropna()
    h = s.index.hour
    inwin = np.zeros(len(s), dtype=bool)
    for a, b in PEAK_WINDOWS:
        inwin |= (h >= a) & (h < b)
    sel = s[(s.index.dayofweek < 5) & inwin]
    return float(sel.max()) if len(sel) else 0.0


def replay_metrics(level: np.ndarray, e: np.ndarray, occ: np.ndarray, ts: pd.DatetimeIndex, rule, design_fc: float) -> Dict:
    kwh = e * level
    sc = safety_counts(rule, level, occ, design_fc)
    n_occ = int(np.asarray(occ, bool).sum())
    days = max(1, pd.DatetimeIndex(ts).normalize().nunique())
    return {"kwh": float(kwh.sum()), "usd": float((kwh * tariff.energy_rate(ts)).sum()), "peak_kw_wkday_high_peak": _peak_kw(kwh, ts),
            "underlit_occupied_share": sc["underlit_occupied_steps"] / n_occ if n_occ else None,
            "egress_floor_violations": sc["life_safety_violations"], "in_use_floor_violation_steps": sc["in_use_floor_violations"],
            "switches_per_day": switch_count(level) / days}


def replay_room(tf: pd.DataFrame, pir: Dict, *, seeds: Iterable[int] = range(10), what_if_miss: Iterable[float] = (0.1, 0.3)) -> Dict:
    """All policies for one ROBOD room's test days, for both rule profiles."""
    ts = pd.DatetimeIndex(tf["ts"])
    e = tf[LIGHT].fillna(0).to_numpy(float)
    occ = tf[Y].to_numpy().astype(bool)
    ml_on = tf["m1_prob"].to_numpy() >= ML_THRESHOLD
    sched = (ts.hour >= SCHEDULE_HOURS[0]) & (ts.hour < SCHEDULE_HOURS[1])
    rec = {"n_steps": int(len(tf)), "n_days": int(ts.normalize().nunique()), "n_occupied_steps": int(occ.sum()),
           "recorded_kwh": float(e.sum()), "recorded_kwh_while_vacant": float(e[~occ].sum()),
           "recorded_vacant_share": float(e[~occ].sum() / e.sum()) if e.sum() > 0 else None,
           "from": str(ts.min()), "to": str(ts.max())}
    on = e[e > 0]
    p_full = float(np.percentile(on, 95)) if len(on) else 0.0
    rec["hypothetical_247_p95_kwh"] = p_full * len(e)
    out = {"recorded": rec, "profiles": {}}
    for prof, kind in PROFILES.items():
        rule = zone_rule(kind)
        P: Dict[str, Dict] = {}
        base = dict(design_fc=REPLAY_DESIGN_FC, step_min=STEP_MIN, n=len(e))
        P["as_operated"] = replay_metrics(np.ones(len(e)), e, occ, ts, rule, REPLAY_DESIGN_FC)
        P["schedule"] = replay_metrics(lighting_levels(rule, mode="schedule", schedule_on=sched, **base), e, occ, ts, rule, REPLAY_DESIGN_FC)
        P["ideal_sensor"] = replay_metrics(lighting_levels(rule, mode="sensor", detect=occ, **base), e, occ, ts, rule, REPLAY_DESIGN_FC)
        P["ml_only"] = replay_metrics(lighting_levels(rule, mode="ml_only", ml_on=ml_on, **base), e, occ, ts, rule, REPLAY_DESIGN_FC)
        scen = {"measured": pir["miss_rate"]}
        scen.update({f"what_if_miss_{m:g}": m for m in what_if_miss})
        injected = {}
        for sname, miss in scen.items():
            runs = {"sensor": [], "sensor_ml": [], "sensor_ml_union": []}
            for sd in seeds:
                det = inject_detection(occ, miss, pir["false_trigger_rate"], np.random.default_rng(1000 + sd))
                runs["sensor"].append(replay_metrics(lighting_levels(rule, mode="sensor", detect=det, **base), e, occ, ts, rule, REPLAY_DESIGN_FC))
                runs["sensor_ml"].append(replay_metrics(lighting_levels(rule, mode="sensor_ml", detect=det, ml_on=ml_on, **base), e, occ, ts, rule, REPLAY_DESIGN_FC))
                runs["sensor_ml_union"].append(replay_metrics(lighting_levels(rule, mode="sensor_ml_union", detect=det, ml_on=ml_on, **base), e, occ, ts, rule, REPLAY_DESIGN_FC))
            injected[sname] = {"miss_rate": miss, "n_seeds": len(list(seeds)) if not isinstance(seeds, range) else len(seeds),
                               **{p: _mean_range(v) for p, v in runs.items()}}
        P["sensor"] = injected["measured"]["sensor"]["mean"]
        P["sensor_ml"] = injected["measured"]["sensor_ml"]["mean"]
        P["sensor_ml_union"] = injected["measured"]["sensor_ml_union"]["mean"]
        for k, v in P.items():
            v["saved_vs_as_operated_pct"] = 100.0 * (1 - v["kwh"] / rec["recorded_kwh"]) if rec["recorded_kwh"] else None
        out["profiles"][prof] = {"zone_rule": kind, "policies": P, "injected": injected}
    return out


def _mean_range(rows: List[Dict]) -> Dict:
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and rows[0][k] is not None]
    return {"mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
            "min": {k: float(np.min([r[k] for r in rows])) for k in keys},
            "max": {k: float(np.max([r[k] for r in rows])) for k in keys}}


def replay_robod(test_frames: Dict[int, pd.DataFrame], pir: Dict, **kw) -> Dict:
    rooms = {i: replay_room(tf, pir, **kw) for i, tf in test_frames.items()}
    tot = {}
    for prof in PROFILES:
        agg = {}
        for pol in ("as_operated", "schedule", "ideal_sensor", "sensor", "sensor_ml", "sensor_ml_union", "ml_only"):
            kwh = sum(rooms[i]["profiles"][prof]["policies"][pol]["kwh"] for i in rooms)
            usd = sum(rooms[i]["profiles"][prof]["policies"][pol]["usd"] for i in rooms)
            n_occ = sum(rooms[i]["recorded"]["n_occupied_steps"] for i in rooms)
            ul = sum((rooms[i]["profiles"][prof]["policies"][pol]["underlit_occupied_share"] or 0) * rooms[i]["recorded"]["n_occupied_steps"] for i in rooms)
            agg[pol] = {"kwh": kwh, "usd": usd, "underlit_occupied_share": ul / n_occ if n_occ else None}
        rec = sum(rooms[i]["recorded"]["recorded_kwh"] for i in rooms)
        for v in agg.values():
            v["saved_vs_as_operated_pct"] = 100.0 * (1 - v["kwh"] / rec)
        tot[prof] = agg
    return {"label": "REAL (ROBOD test days) + INJECTED (motion-sensor misses at UCI 864 rates)", "rooms": rooms,
            "all_rooms": tot, "pir": pir, "ml_threshold": ML_THRESHOLD, "schedule_hours": list(SCHEDULE_HOURS),
            "policy_labels": POLICY_LABELS, "peak_windows_weekday": [list(w) for w in PEAK_WINDOWS],
            "profiles": {"room_rule": "non-egress room rule: off after a 15 min hold (T24 130.1(c)5 style)",
                         "corridor_rule": "egress corridor rule: 50% setback when vacant, 15 min fail-safe hold (T24 130.1(c)6C, NFPA 101 7.8.1.2.2)"},
            "price_note": "ROBOD clock times (Singapore, +08:00) mapped onto LADWP A-2 Rate B periods as if local; base energy charges only"}


# ---------------------------------------------------------------------------------------------------- tower year
def day_pool(tf: pd.DataFrame) -> Dict[str, List[Dict[str, np.ndarray]]]:
    """Split one room's test frame into 288-step day blocks of (presence, M1 probability), by weekday/weekend."""
    pool = {"weekday": [], "weekend": []}
    for day, g in tf.groupby(tf["ts"].dt.normalize()):
        slot = (g["ts"].dt.hour * 12 + g["ts"].dt.minute // 5).to_numpy()
        occ = np.full(STEPS_DAY, np.nan)
        m1 = np.full(STEPS_DAY, np.nan)
        occ[slot] = g[Y].to_numpy()
        m1[slot] = g["m1_prob"].to_numpy()
        occ = pd.Series(occ).ffill().bfill().to_numpy()
        m1 = pd.Series(m1).ffill().bfill().to_numpy()
        pool["weekend" if pd.Timestamp(day).dayofweek >= 5 else "weekday"].append({"occ": occ.astype(bool), "m1": m1, "day": str(pd.Timestamp(day).date())})
    pool["fallback"] = [k for k in ("weekday", "weekend") if not pool[k]]  # an empty kind borrows the other kind's days
    for k in ("weekday", "weekend"):
        if not pool[k]:
            pool[k] = pool["weekday"] or pool["weekend"]
    return pool


def pool_sizes(pools: Dict[int, Dict], zones: List[CommonZone]) -> Dict[str, Dict]:
    """How many distinct held-out days each mapped ROBOD room contributes to the resampled tower year."""
    out = {}
    for room, pool in pools.items():
        kinds = sorted({z.kind for z in zones if z.robod_room == room})
        if not kinds:
            continue
        out[str(room)] = {"zone_kinds": kinds, "n_zones": sum(1 for z in zones if z.robod_room == room),
                          "weekday_days": 0 if "weekday" in pool.get("fallback", []) else len(pool["weekday"]),
                          "weekend_days": 0 if "weekend" in pool.get("fallback", []) else len(pool["weekend"]),
                          "weekend_borrowed_from_weekday": "weekend" in pool.get("fallback", []),
                          "weekday_borrowed_from_weekend": "weekday" in pool.get("fallback", []),
                          "occupied_share": float(np.mean(np.concatenate([d["occ"] for d in pool["weekday"] + pool["weekend"]]))),
                          "days": sorted({d["day"] for d in pool["weekday"] + pool["weekend"]})}
    return out


def hypothetical_dr_days(weather: pd.DataFrame, year: int, n: int = 10) -> List[str]:
    """SIMULATED DR events: the n hottest weekdays (daily max temperature) inside the LADWP DR season."""
    t = weather["temperature_2m"]
    daily = t.groupby(t.index.date).max()
    daily.index = pd.to_datetime(daily.index)
    a, b = tariff.LADWP_DR["season"]
    season = daily[(daily.index >= pd.Timestamp(f"{year}-{a}")) & (daily.index <= pd.Timestamp(f"{year}-{b}")) & (daily.index.dayofweek < 5)]
    return [str(d.date()) for d in season.sort_values(ascending=False).index[:n]]


def tower_year(zones: List[CommonZone], pools: Dict[int, Dict], pir: Dict, weather: pd.DataFrame, *, year: int = 2025,
               seed: int = 0, dr_window=(13, 17), miss_rates: Optional[Dict[str, float]] = None) -> Dict:
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    n_days = len(days)
    ts = pd.date_range(f"{year}-01-01", periods=n_days * STEPS_DAY, freq=f"{STEP_MIN}min")  # local clock, DST ignored
    hol = set(USFederalHolidayCalendar().holidays(days[0], days[-1]).date)
    weekend = np.array([(d.dayofweek >= 5) or (d.date() in hol) for d in days])
    sched = np.tile((np.arange(STEPS_DAY) // 12 >= SCHEDULE_HOURS[0]) & (np.arange(STEPS_DAY) // 12 < SCHEDULE_HOURS[1]), n_days)
    dr_days = hypothetical_dr_days(weather, year)
    dr_mask = np.isin(ts.normalize(), pd.to_datetime(dr_days)) & (ts.hour >= dr_window[0]) & (ts.hour < dr_window[1])
    rng = np.random.default_rng(seed)
    miss_rates = miss_rates or {"measured": pir["miss_rate"]}
    pols = ["always_on", "schedule", "sensor", "sensor_ml", "sensor_ml_union", "ml_only", "sensor_ml_dr"]
    kw = {(m, p): np.zeros(len(ts)) for m in miss_rates for p in pols}
    stats = {(m, p): {"occ_steps": 0, "underlit": 0, "life": 0, "stair_in_use_viol": 0, "switches": 0} for m in miss_rates for p in pols}
    by_kind = {(m, p, z.kind): 0.0 for m in miss_rates for p in pols for z in zones}
    # Alternative stair design: the lowest design fc at which the code setback (vacant level) still meets the in-use
    # floor, so a missed detection never leaves a stair user below it. Same fixture efficacy [ASSUMPTION]: W scales with fc.
    alt_pols = ("sensor", "sensor_ml")
    alt_kw = {p: np.zeros(len(ts)) for p in alt_pols}
    alt_viol = {p: 0 for p in alt_pols}
    alt_design = {}
    for z in zones:
        rule = zone_rule(z.kind)
        pool = pools[z.robod_room]
        picks_wd = rng.integers(0, len(pool["weekday"]), n_days)
        picks_we = rng.integers(0, len(pool["weekend"]), n_days)
        occ = np.concatenate([pool["weekend"][picks_we[k]]["occ"] if weekend[k] else pool["weekday"][picks_wd[k]]["occ"] for k in range(n_days)])
        m1 = np.concatenate([pool["weekend"][picks_we[k]]["m1"] if weekend[k] else pool["weekday"][picks_wd[k]]["m1"] for k in range(n_days)])
        ml_on = m1 >= ML_THRESHOLD
        u = rng.random(len(ts))
        base = dict(design_fc=z.design_fc, step_min=STEP_MIN, n=len(ts))
        for mname, miss in miss_rates.items():
            det = np.where(occ, u >= miss, u < pir["false_trigger_rate"])
            levels = {
                "always_on": lighting_levels(rule, mode="always_on", **base),
                "schedule": lighting_levels(rule, mode="schedule", schedule_on=sched, **base),
                "sensor": lighting_levels(rule, mode="sensor", detect=det, **base),
                "sensor_ml": lighting_levels(rule, mode="sensor_ml", detect=det, ml_on=ml_on, **base),
                "sensor_ml_union": lighting_levels(rule, mode="sensor_ml_union", detect=det, ml_on=ml_on, **base),
                "ml_only": lighting_levels(rule, mode="ml_only", ml_on=ml_on, **base),
                "sensor_ml_dr": lighting_levels(rule, mode="sensor_ml", detect=det, ml_on=ml_on, dr=dr_mask, **base),
            }
            if mname != "measured":
                levels = {k: v for k, v in levels.items() if k in WHAT_IF_POLICIES}
            for p, lv in levels.items():
                kw[(mname, p)] += lv * z.design_w / 1000.0
                by_kind[(mname, p, z.kind)] += float((lv * z.design_w / 1000.0).sum() * STEP_MIN / 60.0)
                sc = safety_counts(rule, lv, occ, z.design_fc)
                st = stats[(mname, p)]
                st["occ_steps"] += int(occ.sum())
                st["underlit"] += sc["underlit_occupied_steps"]
                st["life"] += sc["life_safety_violations"]
                if z.kind == "stairwell":
                    st["stair_in_use_viol"] += sc["in_use_floor_violations"]
                st["switches"] += switch_count(lv)
            if mname == "measured" and z.kind == "stairwell" and rule.min_in_use_fc > 0 and rule.vacant_level > 0:
                fc_alt = max(z.design_fc, rule.min_in_use_fc / rule.vacant_level)
                w_alt = z.design_w * fc_alt / z.design_fc
                alt_design = {"design_fc": fc_alt, "w_per_ft2": z.w_per_ft2 * fc_alt / z.design_fc,
                              "current_design_fc": z.design_fc, "current_w_per_ft2": z.w_per_ft2}
                for p in alt_pols:
                    lv_alt = lighting_levels(rule, mode=p, detect=det, ml_on=ml_on, design_fc=fc_alt, step_min=STEP_MIN, n=len(ts))
                    alt_kw[p] += lv_alt * w_alt / 1000.0 - levels[p] * z.design_w / 1000.0
                    alt_viol[p] += safety_counts(rule, lv_alt, occ, fc_alt)["in_use_floor_violations"]
    results = {}
    for (mname, p), series in kw.items():
        if mname != "measured" and p not in WHAT_IF_POLICIES:
            continue
        s = pd.Series(series, index=ts)
        kwh = s * STEP_MIN / 60.0
        st = stats[(mname, p)]
        dem = tariff.demand_charges(s)
        hp = s[tariff.is_summer_high_peak(s.index)]
        r = {"annual_kwh": float(kwh.sum()), "energy_usd": tariff.energy_cost(kwh), **dem,
             "peak_kw_summer_high_peak": float(hp.resample("15min").mean().max()) if len(hp) else None,
             "monthly_kwh": [float(v) for v in kwh.groupby(kwh.index.month).sum().to_numpy()],
             "underlit_occupied_share": st["underlit"] / st["occ_steps"] if st["occ_steps"] else None,
             "egress_floor_violations": st["life"], "stair_in_use_violation_minutes": st["stair_in_use_viol"] * STEP_MIN,
             "kwh_by_kind": {k: by_kind[(mname, p, k)] for k in sorted({z.kind for z in zones})}}
        r["total_usd"] = r["energy_usd"] + r["demand_usd"] + r["facilities_usd"]
        results.setdefault(mname, {})[p] = r
    meas = results["measured"]
    dr_trim = (pd.Series(kw[("measured", "sensor_ml")] - kw[("measured", "sensor_ml_dr")], index=ts))[dr_mask]
    ao, sch, rec = meas["always_on"], meas["schedule"], meas["sensor_ml"]
    kpi = {
        "kwh_saved_vs_always_on": ao["annual_kwh"] - rec["annual_kwh"],
        "pct_saved_vs_always_on": 100 * (1 - rec["annual_kwh"] / ao["annual_kwh"]),
        "kwh_saved_vs_schedule": sch["annual_kwh"] - rec["annual_kwh"],
        "pct_saved_vs_schedule": 100 * (1 - rec["annual_kwh"] / sch["annual_kwh"]),
        "usd_saved_vs_always_on": ao["total_usd"] - rec["total_usd"],
        "usd_saved_vs_schedule": sch["total_usd"] - rec["total_usd"],
        "peak_kw_cut_vs_always_on": (ao["peak_kw_summer_high_peak"] or 0) - (rec["peak_kw_summer_high_peak"] or 0),
        "dr_mean_trim_kw": float(dr_trim.mean()) if len(dr_trim) else 0.0,
        "dr_max_trim_kw": float(dr_trim.max()) if len(dr_trim) else 0.0,
        "ladwp_dr_min_kw": tariff.LADWP_DR["min_curtail_kw"],
        "egress_floor_fc": float(life_safety("EGRESS_FLOOR_FC")),
        "egress_floor_violations_recommended": rec["egress_floor_violations"],
        "egress_floor_violations_all_deployable": int(sum(meas[p]["egress_floor_violations"] for p in ("always_on", "schedule", "sensor", "sensor_ml", "sensor_ml_union", "sensor_ml_dr"))),
        "egress_floor_violations_note": "0 by construction: level = max(egress floor, target), and every egress vacant level is at or above the floor",
        "stair_in_use_fc": float(life_safety("STAIR_IN_USE_FC")),
        "stair_in_use_violation_minutes_recommended": rec["stair_in_use_violation_minutes"],
        "stair_in_use_violation_minutes_sensor_only": meas["sensor"]["stair_in_use_violation_minutes"],
        "stair_in_use_violation_minutes_always_on": ao["stair_in_use_violation_minutes"],
        "ml_stair_minutes_avoided_vs_sensor_only": meas["sensor"]["stair_in_use_violation_minutes"] - rec["stair_in_use_violation_minutes"],
        "pct_saved_sensor_only_vs_always_on": 100 * (1 - meas["sensor"]["annual_kwh"] / ao["annual_kwh"]),
        "ml_extra_kwh_vs_sensor_only": rec["annual_kwh"] - meas["sensor"]["annual_kwh"],
        "ml_extra_usd_vs_sensor_only": rec["total_usd"] - meas["sensor"]["total_usd"],
        "what_if": {m: {p: {"stair_in_use_violation_minutes": results[m][p]["stair_in_use_violation_minutes"],
                            "underlit_occupied_share": results[m][p]["underlit_occupied_share"],
                            "annual_kwh": results[m][p]["annual_kwh"]} for p in WHAT_IF_POLICIES}
                    for m in results},
    }
    alt = {"label": "SEMI-SYNTHETIC + INJECTED", **alt_design,
           "note": "stairs designed so the code setback (vacant level) still meets the stair in-use floor; W scales with fc [ASSUMPTION]"}
    for p in alt_pols:
        s_alt = pd.Series(kw[("measured", p)] + alt_kw[p], index=ts)
        kwh_alt = float(s_alt.sum() * STEP_MIN / 60.0)
        dem = tariff.demand_charges(s_alt)
        tot = tariff.energy_cost(s_alt * STEP_MIN / 60.0) + dem["demand_usd"] + dem["facilities_usd"]
        alt[p] = {"annual_kwh": kwh_alt, "total_usd": float(tot),
                  "stair_in_use_violation_minutes": alt_viol[p] * STEP_MIN,
                  "extra_kwh_vs_current_design": kwh_alt - meas[p]["annual_kwh"],
                  "extra_usd_vs_current_design": float(tot - meas[p]["total_usd"]),
                  "pct_saved_vs_always_on": 100 * (1 - kwh_alt / ao["annual_kwh"])}
    return {"label": "SEMI-SYNTHETIC", "year": year, "tz": "America/Los_Angeles (local clock; DST hour shifts ignored)",
            "n_zones": len(zones), "steps": int(len(ts)), "step_min": STEP_MIN, "recommended_policy": "sensor_ml",
            "dr_days": dr_days, "dr_window": list(dr_window), "results": results, "kpi": kpi, "policy_labels": POLICY_LABELS,
            "miss_rates": miss_rates, "holidays_count": len(hol), "stair_alternative": alt, "pools": pool_sizes(pools, zones),
            "assumptions": [
                "zone areas, W/ft2 and design fc are [ASSUMPTION] (see inventory)",
                "occupancy per zone kind is resampled from REAL ROBOD held-out test days of one mapped room ([ASSUMPTION] mapping); weekends and US federal holidays draw weekend days, and a room with no weekend test days reuses its weekdays (see pools)",
                "presence and M1 probability are resampled jointly per day so M1 errors carry over",
                "motion-sensor detections are INJECTED at the UCI 864 PIR miss/false-trigger rates, independent per 5-min step",
                "always-on baseline = every common-area fixture at design power 24/7 (a pre-controls building)",
                "fixed schedule = design power 07:00-21:59, vacant level otherwise",
                "DR days are SIMULATED: the 10 hottest weekdays of the LADWP DR season in Open-Meteo 2025 REANALYSIS, 13:00-17:00, trim 15% on non-egress zones only",
                "$ = LADWP A-2 Rate B base energy + demand + facilities charges (adjustment factors excluded); demand is a coincident-peak estimate",
                "lighting only; HVAC not simulated",
            ]}
