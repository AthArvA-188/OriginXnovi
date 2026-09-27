"""Common-area energy (ask 3): download data, train/evaluate M1-M3, run replays, build the LA calendar.

Usage (origin_hack env; writes eval/energy/*.json, models/energy/*.joblib):
    python scripts/train_energy_models.py            # everything except the optional Chronos comparator
    python scripts/train_energy_models.py --offline  # fail instead of downloading missing raw files
Optional zero-shot comparator (separate torch env, e.g. E:\\conda_envs\\cerebro_ml):
    E:\\conda_envs\\cerebro_ml\\python.exe scripts/train_energy_models.py chronos
Rebuild the tiny test fixtures from data/raw/energy (writes to --out, default a check folder under data/raw/energy, and
compares them with tests/fixtures/energy):
    python scripts/train_energy_models.py fixtures [--out tests/fixtures/energy]

Every number the site page shows comes from the JSON files written here.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ENERGY_SRC = REPO / "src" / "cascade" / "building" / "energy"


def _load_by_path(name: str):
    """Load data.py / forecast.py without importing cascade.building (whose __init__ needs pydantic), for torch envs."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"_energy_{name}", ENERGY_SRC / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


if len(sys.argv) > 1 and sys.argv[1] in ("chronos", "fixtures"):
    data, forecast = _load_by_path("data"), _load_by_path("forecast")
else:
    from cascade.building.energy import calendar_la, data, forecast, inventory, occupancy, rules, simulate, tariff  # noqa: E402
    from cascade.building.energy.policy import Proposal, propose_setting  # noqa: E402

EVAL = REPO / "eval" / "energy"
MODELS = REPO / "models" / "energy"
CAL_YEAR = 2027  # next full planning year
SIM_YEAR = 2025  # year of the Open-Meteo reanalysis weather


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return o


def dump(name: str, obj) -> Path:
    EVAL.mkdir(parents=True, exist_ok=True)
    p = EVAL / name
    p.write_text(json.dumps(clean(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {p.relative_to(REPO)} ({p.stat().st_size:,} B)", flush=True)
    return p


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def best_baseline(models: dict, names=("majority", "schedule_48", "schedule_168")) -> tuple[str, float]:
    k = max(names, key=lambda n: models[n]["f1"])
    return k, models[k]["f1"]


def run_all(offline: bool) -> None:
    import joblib

    t0 = time.time()
    if not offline:
        log("fetch sources (skips files already present)")
        data.fetch_all(log=lambda s: print(s, flush=True))

    # --------------------------------------------------------------------------------------------- occupancy
    log("load ROBOD")
    rooms = data.load_robod()
    log("M1 day-blocked per room")
    m1, test_frames = occupancy.m1_room_eval(rooms)
    log("M1 leave-one-room-out")
    loro = occupancy.m1_loro(rooms)
    dump("occupancy_m1.json", {"label": "REAL (ROBOD, CC BY 4.0)", "features": occupancy.M1_FEATURES,
                               "excluded_as_leakage": occupancy.LEAKY, "threshold": 0.5, "hgb": occupancy.HGB_KW,
                               "room_names": data.ROBOD_ROOMS, "day_blocked": m1, "leave_one_room_out": loro})
    log("M2 1 h ahead")
    m2, _ = occupancy.m2_eval(rooms)
    dump("presence_forecast_m2.json", {"label": "REAL (ROBOD)", "horizon_min": 60, "oracle_features": occupancy.M2_ORACLE,
                                       "deployable_features": occupancy.M2_DEPLOY, "rooms": m2,
                                       "note": "oracle = uses ground-truth presence at t (not available to a deployed system); "
                                               "deployable = uses the M1 probability at t (out-of-fold on training days)"})
    log("UCI 357 leakage check")
    uci = occupancy.uci357_eval(data.load_uci357())
    dump("uci357.json", {"label": "REAL (UCI 357, CC BY 4.0)", **uci})
    log("UCI 864 PIR rates")
    pir = occupancy.pir_rates(data.load_uci864())
    dump("pir_uci864.json", {"label": "REAL (UCI 864, CC BY 4.0); used to INJECT sensor misses", **pir})

    log("M1 deploy model (all rooms, all days)")
    allf = pd.concat([occupancy.add_features(d) for d in rooms.values()], ignore_index=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": occupancy.fit_hgb(allf[occupancy.M1_FEATURES], allf[data.Y]), "features": occupancy.M1_FEATURES,
                 "trained_on": "ROBOD rooms 1-5, all days", "label": "REAL"}, MODELS / "m1_presence_hgb.joblib", compress=3)

    # --------------------------------------------------------------------------------------------- garages
    log("M3 BDG2 parking meters")
    el, meta, wx = data.load_bdg2_parking()
    m3 = forecast.garage_eval(el, meta, wx)
    keep = m3.pop("_keep")
    dump("garage_m3.json", {"label": "REAL (BDG2, CC BY-SA, version ambiguous)", "train": "2016", "test": "2017",
                            "issue": "00:00 local, next 24 h (all lags >= 24 h)", "variants": forecast.VARIANTS,
                            "headline_variant": forecast.DEPLOYABLE_HEADLINE, **m3})
    # small per-meter test slice for the page (first two weeks of 2017 for two US/Pacific meters)
    pac = [r["meter"] for r in m3["meters"] if r["timezone"] == "US/Pacific"]
    ex = {}
    for b in pac:
        k = keep[b]
        idx = k["index"]
        sel = idx < pd.Timestamp("2017-01-15")
        ex[b] = {"ts": [str(t) for t in idx[sel]], "y": k["y"][sel].round(3).tolist(), "hgb": np.round(k["hgb"][sel], 3).tolist()}
    dump("garage_m3_example.json", {"label": "REAL (BDG2)", "variant": forecast.DEPLOYABLE_HEADLINE, "meters": ex})
    np.savez_compressed(data.RAW / "m3_test_cache.npz", **{f"{b}__idx": keep[b]["index"].values.astype("datetime64[s]").astype("int64") for b in keep},
                        **{f"{b}__y": keep[b]["y"] for b in keep}, **{f"{b}__hgb": keep[b]["hgb"] for b in keep})

    # --------------------------------------------------------------------------------------------- replay
    log("REAL ROBOD replay")
    rep = simulate.replay_robod(test_frames, pir)
    log("policy comparison on TRAINING days (M1 out-of-fold), the set a policy choice should use")
    train_frames = occupancy.m1_train_oof_frames(rooms)
    sel = simulate.replay_robod(train_frames, pir, what_if_miss=())
    keep_p = ("schedule", "sensor", "sensor_ml", "sensor_ml_union", "ideal_sensor", "ml_only")
    rep["selection_on_train_days"] = {
        "label": "REAL (ROBOD training days, M1 out-of-fold by day) + INJECTED",
        "note": ("The extend-only rule was first chosen after looking at the test-day replay. This block repeats the same "
                 "comparison on the training days only, so the choice can be checked without the test days."),
        "n_days": {str(i): int(pd.DatetimeIndex(f["ts"]).normalize().nunique()) for i, f in train_frames.items()},
        "recorded_kwh": float(sum(r["recorded"]["recorded_kwh"] for r in sel["rooms"].values())),
        "all_rooms": {prof: {p: sel["all_rooms"][prof][p] for p in keep_p} for prof in sel["all_rooms"]}}
    dump("replay_robod.json", rep)

    # --------------------------------------------------------------------------------------------- tower year
    log("SEMI-SYNTHETIC tower year")
    weather, wmeta = data.load_openmeteo()
    zones = inventory.tower_inventory()
    pools = {i: simulate.day_pool(tf) for i, tf in test_frames.items()}
    tower = simulate.tower_year(zones, pools, pir, weather, year=SIM_YEAR, seed=0,
                                miss_rates={"measured": pir["miss_rate"], "what_if_0.1": 0.1, "what_if_0.3": 0.3})
    inv = inventory.inventory_summary(zones)
    inv.pop("zones")
    tower["inventory"] = inv
    tower["weather"] = {"source": "Open-Meteo archive (REANALYSIS), requested timezone=GMT then converted to America/Los_Angeles",
                        **wmeta, "rows": int(len(weather)), "first_local": str(weather.index[0]), "last_local": str(weather.index[-1])}
    assert tower["results"]["measured"]["sensor_ml"]["egress_floor_violations"] == 0  # 1 fc egress floor (by construction)
    dump("tower_year.json", tower)

    # --------------------------------------------------------------------------------------------- calendar
    log("LA calendar")
    noaa = data.read_noaa_normals(data.RAW / data.SOURCES["noaa_monthly"].file)
    ann = data.read_noaa_normals(data.RAW / data.SOURCES["noaa_annual"].file)
    cal = calendar_la.build_calendar(CAL_YEAR, noaa, weather)
    dump("calendar_la.json", {"year": CAL_YEAR, "label": "REAL (NOAA normals) + computed (NOAA GML) + REANALYSIS 2025 temperature",
                              "stations": {calendar_la.USC: "LOS ANGELES DWTN USC CAMPUS, CA US", calendar_la.LAX: "LOS ANGELES INTL AP, CA US"},
                              "station_names_from_api": sorted(set(noaa.get("NAME", pd.Series(dtype=str)).tolist())),
                              "annual_normals": ann.to_dict(orient="records"),
                              "annual_cdd_sum_usc": float(sum(r["cdd65_usc"] for r in cal)),
                              "annual_cdd_sum_lax": float(sum(r["cdd65_lax"] for r in cal)),
                              "rows": cal, "modes": calendar_la.month_hour_modes(cal), "plan": calendar_la.seasonal_plan(cal),
                              "solar": {"lat": data.LA_LAT, "lon": data.LA_LON, "method": "NOAA GML general solar position equations, zenith 90.833 deg, 15th of each month",
                                        "jun21_day_length_h": calendar_la.solar_day(pd.Timestamp(f"{CAL_YEAR}-06-21").date())["day_length_h"],
                                        "dec21_day_length_h": calendar_la.solar_day(pd.Timestamp(f"{CAL_YEAR}-12-21").date())["day_length_h"]}})
    dump("tariff.json", tariff.tariff_artifact())
    dump("rules.json", {"rules": rules.load_rules(), "table": rules.rules_table()})

    # --------------------------------------------------------------------------------------------- proposals
    log("proposals (pending approval)")
    res = tower["results"]["measured"]
    kw = tower["kpi"]
    props = []
    alt = tower["stair_alternative"]
    stair_fc = float(rules.life_safety("STAIR_IN_USE_FC"))
    for kind in ("corridor", "stairwell", "lobby", "garage", "restroom", "amenity"):
        r = rules.zone_rule(kind)
        a, w, fc, room = inventory.ASSUMED[kind]
        saved = res["always_on"]["kwh_by_kind"][kind] - res["sensor_ml"]["kwh_by_kind"][kind]
        reason = f"Sensor + ML policy (ML only adds light). Simulated saving vs always-on: {saved:,.0f} kWh/yr"
        if r.min_in_use_fc > 0 and r.vacant_level * fc + 1e-9 < r.min_in_use_fc:
            reason += (f". CAUTION: at the assumed {fc:g} fc design the vacant setback gives {r.vacant_level * fc:g} fc, below the "
                       f"{r.min_in_use_fc:g} fc stair in-use level. In the simulated year stairs were below it for "
                       f"{res['sensor_ml']['stair_in_use_violation_minutes']:,} min while someone was on them (a missed detection). "
                       f"See the stair design proposal for the fix")
        p = propose_setting(r, zone=f"all {r.label.lower()} zones", vacant_level=r.vacant_level, hold_min=r.hold_min, design_fc=fc,
                            reason=reason, expected_kwh_per_year=saved)
        props.append(p)
    if alt.get("design_fc"):
        props.append(Proposal(
            zone="all stairwell zones", action=f"review stair lighting design: {alt['design_fc']:g} fc design so the setback holds {stair_fc:g} fc",
            reason=(f"Removes the simulated stair shortfall ({res['sensor_ml']['stair_in_use_violation_minutes']:,} -> "
                    f"{alt['sensor_ml']['stair_in_use_violation_minutes']:,} min/yr below {stair_fc:g} fc while in use) at a cost of "
                    f"{alt['sensor_ml']['extra_kwh_vs_current_design']:,.0f} kWh/yr (${alt['sensor_ml']['extra_usd_vs_current_design']:,.0f}). "
                    f"Needs a lighting designer and a photometric check; W is assumed to scale with fc"),
            citations=[b["cite"] for b in rules.zone_rule("stairwell").basis],
            expected_kwh_per_year=-alt["sensor_ml"]["extra_kwh_vs_current_design"],
            expected_usd_per_year=-alt["sensor_ml"]["extra_usd_vs_current_design"], data_label="SEMI-SYNTHETIC + INJECTED"))
    props.append(Proposal(zone="restrooms + amenity rooms", action="DR playbook: trim 15% during LADWP DR events",
                          reason=f"Simulated mean trim {kw['dr_mean_trim_kw']:.2f} kW on {len(tower['dr_days'])} hypothetical DR days; "
                                 f"LADWP DR needs >= {kw['ladwp_dr_min_kw']} kW building-wide, so this is only a contribution",
                          citations=[rules.load_rules()["demand_response"]["cite"]], data_label="SIMULATED"))
    dump("proposals.json", {"label": "SEMI-SYNTHETIC", "status_note": "every proposal is pending_approval; a named human approves or rejects it",
                            "proposals": [p.__dict__ for p in props]})

    # --------------------------------------------------------------------------------------------- summary
    log("summary + model card")
    card = []
    for i, r in m1.items():
        bn, bf = best_baseline(r["models"])
        mm = r["models"]["m1_hgb"]
        vb = r["vs_best_baseline"]
        card.append({"model": "M1 presence now", "data": f"ROBOD room {i} ({data.ROBOD_ROOMS[i]})", "split": "day-blocked 70/30",
                     "n_test": mm["n"], "n_pos": mm["n_pos"], "prevalence": mm["prevalence"], "metric": "F1 (occupied)",
                     "model_score": mm["f1"], "best_baseline": bn, "baseline_score": bf, "model_wins": mm["f1"] > bf, "label": "REAL",
                     "diff_ci95": vb["ci95"], "verdict": vb["verdict"], "n_days": vb["n_days"]})
    for i, r in m2.items():
        mm = r["models"]["m2_hgb_deployable"]
        cands = ("majority", "schedule_48", "schedule_168", "persistence_m1_deployable")
        bn, bf = best_baseline(r["models"], cands)
        card.append({"model": "M2 presence +1 h (deployable)", "data": f"ROBOD room {i}", "split": "day-blocked 70/30",
                     "n_test": mm["n"], "n_pos": mm["n_pos"], "prevalence": mm["prevalence"], "metric": "F1 (occupied)",
                     "model_score": mm["f1"], "best_baseline": bn, "baseline_score": bf, "model_wins": mm["f1"] > bf, "label": "REAL",
                     "diff_ci95": r["deployable_vs_best_baseline"]["ci95"], "verdict": r["deployable_vs_best_baseline"]["verdict"],
                     "n_days": r["deployable_vs_best_baseline"]["n_days"]})
        mo = r["models"]["m2_hgb_oracle"]
        po = r["models"]["persistence_oracle"]
        card.append({"model": "M2 presence +1 h (oracle)", "data": f"ROBOD room {i}", "split": "day-blocked 70/30",
                     "n_test": mo["n"], "n_pos": mo["n_pos"], "prevalence": mo["prevalence"], "metric": "F1 (occupied)",
                     "model_score": mo["f1"], "best_baseline": "persistence_oracle", "baseline_score": po["f1"],
                     "model_wins": mo["f1"] > po["f1"], "label": "REAL (oracle input)",
                     "diff_ci95": r["oracle_vs_persistence"]["ci95"], "verdict": r["oracle_vs_persistence"]["verdict"],
                     "n_days": r["oracle_vs_persistence"]["n_days"]})
    s = m3["summary"]
    card.append({"model": "M3 garage day-ahead (no weather)", "data": f"BDG2 {s['n_meters']} parking meters", "split": "train 2016 / test 2017",
                 "n_test": int(sum(r["n_test_h"] for r in m3["meters"])), "n_pos": None, "prevalence": None,
                 "metric": "median CV(RMSE) (lower is better)", "model_score": s["median_cvrmse"]["hgb_no_weather"],
                 "best_baseline": "lag24", "baseline_score": s["median_cvrmse"]["lag24"],
                 "model_wins": s["median_cvrmse"]["hgb_no_weather"] < s["median_cvrmse"]["lag24"], "label": "REAL",
                 "meters_beating_best_naive": s["beats_best_naive"]["hgb_no_weather"]})
    m1_wins = sum(1 for c in card if c["model"] == "M1 presence now" and c["model_wins"])
    m2_wins = sum(1 for c in card if c["model"] == "M2 presence +1 h (deployable)" and c["model_wins"])
    m2o_wins = sum(1 for c in card if c["model"] == "M2 presence +1 h (oracle)" and c["model_wins"])
    summary = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "model_card": card,
               "m1_rooms_beating_best_baseline": m1_wins, "m2_deployable_rooms_beating_best_baseline": m2_wins,
               "m2_oracle_rooms_beating_persistence": m2o_wins, "n_rooms": len(m1),
               "replay_all_rooms": rep["all_rooms"], "tower_kpi": tower["kpi"],
               "pir": {"miss_rate": pir["miss_rate"], "false_trigger_rate": pir["false_trigger_rate"], "n_occupied_windows": pir["n_occupied_windows"],
                       "n_vacant_windows": pir["n_vacant_windows"]}}
    dump("summary.json", summary)
    write_model_card(summary, m3, uci)

    man = data.load_manifest()
    dump("manifest.json", {"generated_utc": summary["generated_utc"], "runtime_s": round(time.time() - t0, 1),
                           "python": sys.version.split()[0], "platform": platform.platform(),
                           "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": __import__("sklearn").__version__},
                           "command": "python scripts/train_energy_models.py",
                           "raw_files": data.normalized_manifest(man), "sources": {k: s.__dict__ for k, s in data.SOURCES.items()},
                           "extra_sources": data.EXTRA_SOURCES})
    log(f"done in {time.time() - t0:.0f}s")


def write_model_card(summary: dict, m3: dict, uci: dict) -> None:
    lines = ["# Common-area energy models: model card (generated)", "",
             f"Generated {summary['generated_utc']} by scripts/train_energy_models.py. Every number is copied from eval/energy/*.json.", "",
             "| model | data | split | n test | positives | metric | model | best baseline | baseline | model wins | 95% CI of difference (days) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in summary["model_card"]:
        lines.append(f"| {c['model']} | {c['data']} | {c['split']} | {c['n_test']} | {c['n_pos'] if c['n_pos'] is not None else '-'} | {c['metric']} | "
                     f"{c['model_score']:.3f} | {c['best_baseline']} | {c['baseline_score']:.3f} | {'yes' if c['model_wins'] else 'no'} | "
                     + (f"{c['diff_ci95'][0]:+.3f} to {c['diff_ci95'][1]:+.3f} ({c['verdict']})" if c.get('diff_ci95') else "-") + " |")
    t2 = uci["tests"]["datatest2"]["models"]
    lines += ["", "UCI 357 test2 (forward in time) F1: " + ", ".join(f"{k} {v['f1']:.3f}" for k, v in t2.items()),
              "", "Labels: REAL = public measured data; SEMI-SYNTHETIC / INJECTED / SIMULATED are marked in each JSON."]
    (EVAL / "model_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  wrote eval/energy/model_card.md", flush=True)


def chronos_day_ahead(el: pd.DataFrame, meters, test_index, model_id: str = "amazon/chronos-bolt-small", context_h: int = 512,
                      device=None, batch: int = 64):
    """Zero-shot day-ahead forecasts (median) for each 2017 day, issued at 00:00 from history only.

    Needs torch + chronos-forecasting, so it lives in this script (run in a torch env), not in src/.
    """
    import torch
    from chronos import BaseChronosPipeline

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipe = BaseChronosPipeline.from_pretrained(model_id, device_map=dev, torch_dtype=torch.float32)
    out = {}
    for b in meters:
        days = pd.DatetimeIndex(sorted(set(test_index[b].normalize())))
        starts, arrs = forecast.history_contexts(el[b], days, context_h=context_h, min_h=168)
        ctxs = [torch.tensor(a, dtype=torch.float32) for a in arrs]
        fc = {}
        for k in range(0, len(ctxs), batch):
            q, _mean = pipe.predict_quantiles(ctxs[k:k + batch], prediction_length=24, quantile_levels=[0.5])
            med = q[..., 0].cpu().numpy()
            for d, row in zip(starts[k:k + batch], med):
                for h in range(24):
                    fc[d + pd.Timedelta(hours=h)] = float(row[h])
        out[b] = np.array([fc.get(t, np.nan) for t in test_index[b]])
    return out


def build_fixtures(out_dir: Path) -> None:
    """Re-cut the committed test fixtures from data/raw/energy and compare them with tests/fixtures/energy.

    ROBOD rooms 2 and 5: the first 10 dates of each room, the 9 columns the tests use (CC BY 4.0).
    NOAA normals: byte copies. Open-Meteo 2025 (timezone=GMT): the hours of UTC dates 2025-01-01/02 and 2025-03-09/10.
    """
    import io
    import shutil
    import zipfile

    fx = REPO / "tests" / "fixtures" / "energy"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["timestamp", "indoor_co2 [ppm]", "sound_pressure_level [dba]", "voc [ppb]", "indoor_relative_humidity [%]",
            "air_temperature [Celsius]", "wifi_connected_devices [number]", "lighting_energy [kWh]", "occupant_presence [binary]"]
    with zipfile.ZipFile(data.RAW / data.SOURCES["robod"].file) as z:
        names = {n.rsplit("/", 1)[-1]: n for n in z.namelist()}
        for i in (2, 5):
            d = pd.read_csv(io.BytesIO(z.read(names[f"combined_Room{i}.csv"])))
            date = d["timestamp"].astype(str).str.slice(0, 10)
            first = sorted(date.unique())[:10]
            cut = d.loc[date.isin(first), cols]
            cut.to_csv(out_dir / f"combined_Room{i}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    for key in ("noaa_monthly", "noaa_annual"):
        f = data.SOURCES[key].file
        if (out_dir / f).resolve() != (data.RAW / f).resolve():
            shutil.copyfile(data.RAW / f, out_dir / f)
    om = json.loads((data.RAW / data.SOURCES["openmeteo_2025"].file).read_text(encoding="utf-8"))
    keep_dates = {"2025-01-01", "2025-01-02", "2025-03-09", "2025-03-10"}
    sel = [k for k, t in enumerate(om["hourly"]["time"]) if t[:10] in keep_dates]
    om["hourly"] = {k: [v[j] for j in sel] for k, v in om["hourly"].items()}
    (out_dir / "openmeteo_la_2025_gmt_slice.json").write_text(json.dumps(om), encoding="utf-8")
    # compare with the committed fixtures (content, not bytes)
    for i in (2, 5):
        a = pd.read_csv(out_dir / f"combined_Room{i}.csv.gz")
        b = pd.read_csv(fx / f"combined_Room{i}.csv.gz")
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)
    for key in ("noaa_monthly", "noaa_annual"):
        f = data.SOURCES[key].file
        assert (out_dir / f).read_bytes() == (fx / f).read_bytes(), f
    assert json.loads((out_dir / "openmeteo_la_2025_gmt_slice.json").read_text(encoding="utf-8")) == \
        json.loads((fx / "openmeteo_la_2025_gmt_slice.json").read_text(encoding="utf-8"))
    log(f"fixtures written to {out_dir} and identical in content to tests/fixtures/energy")


def run_chronos(model_id: str) -> None:
    """Optional: Chronos-Bolt zero-shot on the same 2017 test hours as M3 (run in a torch env)."""
    t0 = time.time()
    cache = np.load(data.RAW / "m3_test_cache.npz")
    meters = sorted({k.split("__")[0] for k in cache.files})
    el, meta, wx = data.load_bdg2_parking()
    idx = {b: pd.DatetimeIndex(cache[f"{b}__idx"].astype("datetime64[s]")) for b in meters}
    log(f"chronos {model_id} on {len(meters)} meters")
    preds = chronos_day_ahead(el, meters, idx, model_id=model_id)
    rows = []
    for b in meters:
        y, h, c = cache[f"{b}__y"], cache[f"{b}__hgb"], preds[b]
        ok = np.isfinite(c)
        rows.append({"meter": b, "n_test_h": int(ok.sum()), "cvrmse_chronos": forecast.cvrmse(y[ok], c[ok]),
                     "cvrmse_hgb_no_weather": forecast.cvrmse(y[ok], h[ok]), "nmbe_chronos": forecast.nmbe(y[ok], c[ok])})
    f = pd.DataFrame(rows)
    m3 = json.loads((EVAL / "garage_m3.json").read_text(encoding="utf-8"))
    naive = {r["meter"]: min(r["cvrmse_lag24"], r["cvrmse_lag168"]) for r in m3["meters"]}
    f["best_naive"] = f["meter"].map(naive)
    import torch
    dump("garage_m3_chronos.json", {"label": "REAL (BDG2) zero-shot", "model": model_id, "licence": "Apache-2.0",
                                    "model_url": f"https://huggingface.co/{model_id}", "accessed": "2026-09-26",
                                    "context_fill": "history only: gaps interpolated between past values, then the last past value carried forward; nothing at or after the 00:00 issue time",
                                    "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    "context_h": 512, "device": "cuda" if torch.cuda.is_available() else "cpu",
                                    "runtime_s": round(time.time() - t0, 1),
                                    "median_cvrmse_chronos": float(f["cvrmse_chronos"].median()),
                                    "median_cvrmse_hgb_no_weather": float(f["cvrmse_hgb_no_weather"].median()),
                                    "chronos_beats_hgb": int((f["cvrmse_chronos"] < f["cvrmse_hgb_no_weather"]).sum()),
                                    "chronos_beats_best_naive": int((f["cvrmse_chronos"] < f["best_naive"]).sum()),
                                    "n_meters": int(len(f)), "meters": f.to_dict(orient="records")})
    log(f"chronos done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", nargs="?", default="all", choices=["all", "chronos", "fixtures"])
    ap.add_argument("--out", default=None, help="fixtures step: output folder (default data/raw/energy/fixtures_check)")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--chronos-model", default="amazon/chronos-bolt-small")
    a = ap.parse_args()
    if a.step == "chronos":
        run_chronos(a.chronos_model)
    elif a.step == "fixtures":
        build_fixtures(Path(a.out) if a.out else data.RAW / "fixtures_check")
    else:
        run_all(a.offline)
