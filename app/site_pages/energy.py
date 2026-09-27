"""Site page: common-area energy smart switching (ask 3). Reads only eval/energy/*.json written by
scripts/train_energy_models.py; every number on this page comes from those files."""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "eval" / "energy"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MUTED, CRITICAL = "#898781", "#d03b3b"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal blue, light -> dark
LABEL_COLOR = {"REAL": "green", "SEMI-SYNTHETIC": "orange", "SIMULATED": "violet", "INJECTED": "red",
               "REANALYSIS": "blue", "ASSUMPTION": "gray", "TARIFF APPROXIMATION": "gray"}


def load(name: str):
    return _load_cached(name, _mtime(EV / name))


def _mtime(p) -> float:
    """File modification time, part of every cache key so regenerated artifacts are picked up without a restart."""
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def _load_cached(name: str, mtime: float):
    p = EV / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def badge(*labels: str) -> str:
    return " ".join(f":{LABEL_COLOR.get(x, 'gray')}-badge[{x}]" for x in labels)


def f0(x) -> str:
    return "n/a" if x is None else f"{x:,.0f}"


def f1(x) -> str:
    return "n/a" if x is None else f"{x:,.1f}"


def f3(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def pct(x, d=1) -> str:
    return "n/a" if x is None else f"{x:.{d}f}%"


summary = load("summary.json")
tower = load("tower_year.json")
replay = load("replay_robod.json")
m1 = load("occupancy_m1.json")
m2 = load("presence_forecast_m2.json")
m3 = load("garage_m3.json")
m3ex = load("garage_m3_example.json")
chron = load("garage_m3_chronos.json")
uci = load("uci357.json")
pir = load("pir_uci864.json")
cal = load("calendar_la.json")
tar = load("tariff.json")
rules = load("rules.json")
props = load("proposals.json")
manifest = load("manifest.json")

# ------------------------------------------------------------------------------------------------ 1. title
st.title("Common-area energy: smart switching inside the safety rules")

if not all([summary, tower, replay, cal, rules, m1, m2, m3]):
    st.warning("Energy artifacts are missing. Run `python scripts/train_energy_models.py` to generate eval/energy/*.json.")
    st.stop()

R = rules["rules"]
life = {r["key"]: r for r in R["life_safety"]}
EGRESS_FC = life["EGRESS_FLOOR_FC"]["value"]
STAIR_FC = life["STAIR_IN_USE_FC"]["value"]
zt = R["zone_types"]
k = tower["kpi"]
res = tower["results"]["measured"]
rec_pol = tower["recommended_policy"]
stair_alt = tower.get("stair_alternative") or {}
stair_min = k["stair_in_use_violation_minutes_recommended"]
st.markdown(f"**Shared spaces in a high-rise (corridors, stairs, lobbies, garages, restrooms, amenity rooms) use less "
            f"electricity, lights are never set below the {EGRESS_FC:g} foot-candle exit-route minimum, and a person approves "
            f"every change.**")
st.caption(f"Caveat, stated up front: in our simulated tower, stairs were below the {STAIR_FC:g} foot-candle level required "
           f"while someone is on them for {f0(stair_min)} minutes a year, each time a motion sensor missed a person "
           f"(SEMI-SYNTHETIC tower, INJECTED sensor misses). A fix is proposed below.")
hp = tar["ladwp_a2b"]["periods_weekday"]["high_peak"] if tar else []
hp_txt = ", ".join(f"{a:02d}:00-{b:02d}:00" for a, b in hp) or "high peak"
u_saved = replay["all_rooms"]["room_rule"]["sensor_ml_union"]["saved_vs_as_operated_pct"]
s_saved = replay["all_rooms"]["room_rule"]["sensor"]["saved_vs_as_operated_pct"]
_days = sorted({round((r["n_train"] + r["models"]["m1_hgb"]["n"]) / 288) for r in m1["day_blocked"].values()}) if m1 else []
day_span = (f"{_days[0]}-{_days[-1]}" if len(_days) > 1 else f"{_days[0]}") if _days else "n/a"

# ------------------------------------------------------------------------------------------------ 2. what this means
with st.container(border=True):
    st.markdown("#### What this means")
    st.markdown(
        f"- **Who it helps:** the building manager and chief engineer of a high-rise.\n"
        f"- **The problem:** shared spaces are often lit around the clock. Codes let most of that light dim or switch off "
        f"when nobody is there, but they also set hard minimums: exit routes stay at {EGRESS_FC:g} "
        f"foot-candle or more whenever the building is occupied, stairs reach {STAIR_FC:g} foot-candles "
        f"while someone is on them, and motion timers on exit routes hold for at least "
        f"{life['EGRESS_MOTION_HOLD_MIN']['value']} minutes and fail to ON.\n"
        f"- **What Cerebro does:** the motion sensors and code rules do the switching. The AI can only keep lights on longer "
        f"when it thinks someone is still there. It never turns lights off by itself, never touches emergency or exit-sign "
        f"circuits, and every setting change or demand-response action waits for a person to approve it.\n"
        f"- **What it does not guarantee:** the {EGRESS_FC:g} fc exit-route floor holds by design, but the {STAIR_FC:g} fc "
        f"stair level depends on the motion sensor seeing the person. With the assumed stair design, a missed detection "
        f"leaves the stair at its setback level, below {STAIR_FC:g} fc. Sensor health matters more than the AI.\n"
        f"- **What you do next:** review the pending proposals below (including the stair design fix), approve the ones "
        f"you agree with, and follow the season-by-season plan for Los Angeles.")

# ------------------------------------------------------------------------------------------------ 3. main visual
st.subheader(f"A year in a {tower['inventory']['floors']}-floor LA tower")
st.markdown(badge("SEMI-SYNTHETIC", "INJECTED", "SIMULATED", "TARIFF APPROXIMATION") +
            f" The tower, fixture wattages and areas are assumptions. Occupancy is resampled from real room data. "
            f"Sensor misses are injected. The {len(tower['dr_days'])} demand-response days are simulated. "
            f"{tower['inventory']['n_zones']} common-area zones, {tower['year']} weather.")
def compact(x, unit=""):
    if x is None:
        return "n/a"
    ax = abs(x)
    txt = f"{x / 1e6:.2f}M" if ax >= 1e6 else (f"{x / 1e3:.0f}k" if ax >= 1e4 else f"{x:,.0f}")
    return f"{txt}{unit}"


st.caption(f"Recommended policy: {tower['policy_labels'][rec_pol]}. The first two tiles compare it with a pre-controls building "
           f"whose common-area lights are always on. Motion sensors alone, which the energy code already expects, save more "
           f"({pct(k['pct_saved_sensor_only_vs_always_on'])}); the third tile shows what the AI part costs on top of them.")
c1, c2 = st.columns(2)
c1.metric("Energy saved per year vs a pre-controls building (lights always on)", compact(k["kwh_saved_vs_always_on"], " kWh"),
          pct(k["pct_saved_vs_always_on"]), border=True,
          help=f"{f0(k['kwh_saved_vs_always_on'])} kWh. Motion sensors alone: {pct(k['pct_saved_sensor_only_vs_always_on'])}.")
c2.metric("Bill saved per year vs a pre-controls building", "$" + compact(k["usd_saved_vs_always_on"]),
          "$" + compact(k["usd_saved_vs_schedule"]) + " vs fixed schedule", border=True,
          help=f"${f0(k['usd_saved_vs_always_on'])}; LADWP A-2 Rate B base charges only (adjustment factors excluded)")
c3, c4 = st.columns(2)
c3.metric("Extra energy the AI extension uses vs motion sensors alone", f"+{f0(k['ml_extra_kwh_vs_sensor_only'])} kWh/yr",
          f"+${f0(k['ml_extra_usd_vs_sensor_only'])} per year", delta_color="inverse", border=True,
          help=f"Sensors alone: {f0(res['sensor']['annual_kwh'])} kWh/yr; sensor + AI: {f0(res[rec_pol]['annual_kwh'])} kWh/yr. "
               f"In return, stair minutes below {STAIR_FC:g} fc fall from {f0(k['stair_in_use_violation_minutes_sensor_only'])} "
               f"to {f0(stair_min)} per year.")
c4.metric(f"Stair minutes below {STAIR_FC:g} fc while in use", f"{f0(stair_min)} min/yr",
          f"{-k['ml_stair_minutes_avoided_vs_sensor_only']:+,} vs motion sensors alone", delta_color="inverse", border=True,
          help=(f"SEMI-SYNTHETIC tower + INJECTED sensor misses at the measured UCI 864 rate. "
                + (f"Designing stairs for {stair_alt['design_fc']:g} fc brings this to {f0(stair_alt[rec_pol]['stair_in_use_violation_minutes'])} "
                   f"for +{f0(stair_alt[rec_pol]['extra_kwh_vs_current_design'])} kWh/yr." if stair_alt.get("design_fc") else "")))
c5, c6 = st.columns(2)
c5.metric(f"Exit-route {EGRESS_FC:g} fc floor violations (0 by design)", f"{k['egress_floor_violations_all_deployable']}", border=True,
          help=f"5-min steps below the {EGRESS_FC:g} fc egress floor across all deployable policies and zones. "
               f"{k['egress_floor_violations_note']}, so this tile cannot fail; the stair tile is the real safety test.")
c6.metric("Summer high-peak kW cut vs a pre-controls building", f"{f1(k['peak_kw_cut_vs_always_on'])} kW", border=True,
          help=f"LADWP high season, weekdays {hp_txt}, 15-min mean kW of common-area lighting")
if stair_alt.get("design_fc"):
    st.info(f"**Stair fix on the table (pending approval below):** design stair lighting for {stair_alt['design_fc']:g} fc instead of the "
            f"assumed {stair_alt['current_design_fc']:g} fc, so the code's {100 * (1 - zt['stairwell']['vacant_level']):.0f}% setback still gives {STAIR_FC:g} fc when a sensor misses "
            f"someone. Simulated effect: stair minutes below {STAIR_FC:g} fc {f0(stair_min)} -> "
            f"{f0(stair_alt[rec_pol]['stair_in_use_violation_minutes'])} per year, for +{f0(stair_alt[rec_pol]['extra_kwh_vs_current_design'])} kWh "
            f"(+${f0(stair_alt[rec_pol]['extra_usd_vs_current_design'])}) a year. SEMI-SYNTHETIC; fixture wattage is assumed to scale "
            f"with foot-candles.")
st.caption("$ uses LADWP A-2 Rate B base charges only (adjustment factors excluded), so it understates a real bill. "
           "The 'what if sensors degrade' table below shows how the stair shortfall grows when sensors miss more often.")
with st.expander("Assumptions behind these tower numbers (SEMI-SYNTHETIC, not measured in a real tower)"):
    st.markdown("\n".join(f"- {a}" for a in tower["assumptions"]))
    inv_rows = [{"zone type": kd, "zones": v["count"], "total ft2": v["area_ft2"], "design W": v["design_w"]}
                for kd, v in tower["inventory"]["by_kind"].items()]
    st.dataframe(pd.DataFrame(inv_rows), hide_index=True, width="stretch")
    if tower.get("pools"):
        st.markdown("**How much real occupancy data sits behind each zone type.** Each simulated day copies one held-out "
                    "ROBOD day of the mapped room, so a small pool means the same few days repeat all year.")
        st.dataframe(pd.DataFrame([{"ROBOD room": rm, "zone types": ", ".join(v["zone_kinds"]), "zones": v["n_zones"],
                                    "weekday days in pool": v["weekday_days"], "weekend days in pool": v["weekend_days"],
                                    "weekend borrowed from weekdays": v["weekend_borrowed_from_weekday"],
                                    "occupied share of pool": pct(100 * v["occupied_share"]),
                                    "dates": f"{v['days'][0]} to {v['days'][-1]}" if v["days"] else "-"}
                                   for rm, v in tower["pools"].items()]), hide_index=True, width="stretch")

st.markdown("##### Policy comparison on real rooms")
st.markdown(badge("REAL", "INJECTED") +
            f" Replay on the held-out days of {summary['n_rooms']} real rooms (ROBOD, Singapore): **kWh the rooms really used** "
            "compared with what each policy would have used. Motion-sensor misses are injected at the rate measured on UCI 864.")
prof = st.radio("Rule applied", ["room_rule", "corridor_rule"], horizontal=True,
                format_func=lambda p: {"room_rule": "Room rule (off after hold)", "corridor_rule": f"Corridor rule ({100 * (1 - zt['corridor']['vacant_level']):.0f}% setback, egress)"}[p])
order = ["as_operated", "schedule", "sensor", "sensor_ml", "sensor_ml_union", "ideal_sensor", "ml_only"]
role = {"as_operated": "Reference", "ideal_sensor": "Reference (upper bound)", "ml_only": "Unsafe counterfactual"}
agg = replay["all_rooms"][prof]
rows = []
for p in order:
    v = agg[p]
    lab = replay["policy_labels"][p] + (" (recommended)" if p == rec_pol else "")
    rows.append({"policy": lab, "role": role.get(p, "Deployable"), "saved_pct": v["saved_vs_as_operated_pct"],
                 "underlit_pct": 100 * (v["underlit_occupied_share"] or 0), "kwh": v["kwh"], "usd": v["usd"]})
df = pd.DataFrame(rows)
cscale = alt.Scale(domain=["Deployable", "Reference", "Reference (upper bound)", "Unsafe counterfactual"],
                   range=[BLUE, MUTED, "#c3c2b7", CRITICAL])
ca, cb = st.columns(2)
with ca:
    ch = alt.Chart(df, title="Recorded lighting kWh saved (%)").mark_bar(cornerRadiusEnd=4, size=18).encode(
        y=alt.Y("policy:N", sort=[r["policy"] for r in rows], title=None),
        x=alt.X("saved_pct:Q", title="% of recorded kWh saved"),
        color=alt.Color("role:N", scale=cscale, legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("policy:N"), alt.Tooltip("saved_pct:Q", format=".1f", title="% saved"),
                 alt.Tooltip("kwh:Q", format=",.1f", title="kWh"), alt.Tooltip("usd:Q", format=",.2f", title="$ (LADWP A-2B)")])
    st.altair_chart(ch, width="stretch")
with cb:
    ch2 = alt.Chart(df, title="Occupied time left under-lit (%)").mark_bar(cornerRadiusEnd=4, size=18).encode(
        y=alt.Y("policy:N", sort=[r["policy"] for r in rows], title=None),
        x=alt.X("underlit_pct:Q", title="% of occupied 5-min steps below the occupied level"),
        color=alt.Color("role:N", scale=cscale, legend=None),
        tooltip=[alt.Tooltip("policy:N"), alt.Tooltip("underlit_pct:Q", format=".2f", title="% under-lit")])
    st.altair_chart(ch2, width="stretch")
rec_total = sum(r["recorded"]["recorded_kwh"] for r in replay["rooms"].values())
vac_total = sum(r["recorded"]["recorded_kwh_while_vacant"] for r in replay["rooms"].values())
st.caption(f"Recorded lighting while the rooms were empty: {f1(vac_total)} of {f1(rec_total)} kWh "
           f"({pct(100 * vac_total / rec_total)}) on the test days. The 'ideal sensor' uses ground-truth presence, so it is "
           f"under-lit 0% by construction and is only an upper bound. ML-only switching is shown to make the point that it leaves "
           f"people in the dark; it is never offered as a mode. ROBOD has no motion-sensor column, so the 'motion sensor' here is "
           f"simulated: true presence with misses INJECTED independently at the lab-measured UCI 864 rate, which flatters it.")
sel = replay.get("selection_on_train_days")
if sel:
    sa = sel["all_rooms"]["room_rule"]
    trio = [("sensor_ml_union", "union"), ("sensor_ml", "extend-only"), ("sensor", "sensor-only")]
    agg_room = replay["all_rooms"]["room_rule"]
    te_order = sorted(trio, key=lambda t: agg_room[t[0]]["saved_vs_as_operated_pct"])
    tr_order = sorted(trio, key=lambda t: sa[t[0]]["saved_vs_as_operated_pct"])
    same = [t[0] for t in te_order] == [t[0] for t in tr_order]
    st.caption(
        "How the recommended rule was chosen: we first built the union rule and switched to extend-only after seeing these "
        "test days, so the test days did double duty. Check on the training days only (" +
        ", ".join(f"room {r} {n} d" for r, n in sel["n_days"].items()) + ", M1 out-of-fold): room-rule savings "
        + ", ".join(f"{lab} {pct(sa[key]['saved_vs_as_operated_pct'])} (under-lit {pct(100 * (sa[key]['underlit_occupied_share'] or 0), 3)})"
                    for key, lab in trio)
        + (". Same savings ordering as on the test days." if same else
           ". The savings ordering differs from the test days; treat the choice as provisional.")
        + (" On the training days the extend-only rule did not reduce under-lit time compared with sensor-only, so its "
           "safety benefit is not confirmed there; it only costs energy." if
           (sa["sensor_ml"]["underlit_occupied_share"] or 0) >= (sa["sensor"]["underlit_occupied_share"] or 0) else
           " On the training days the extend-only rule also reduced under-lit time compared with sensor-only."))

st.markdown("##### What if the motion sensors degrade?")
st.markdown(badge("SEMI-SYNTHETIC", "INJECTED") + " Same tower year, with sensor miss rates raised on purpose (what-if values, not measurements).")
wi = []
for m, pols in k["what_if"].items():
    miss = tower["miss_rates"][m]
    for p, v in pols.items():
        wi.append({"sensor miss rate per 5 min": f"{miss:.3f}" + (" (UCI 864 measured)" if m == "measured" else " (what-if)"),
                   "policy": tower["policy_labels"][p], "annual kWh": f0(v["annual_kwh"]),
                   "occupied time under-lit": pct(100 * (v["underlit_occupied_share"] or 0), 3),
                   f"stair minutes below {STAIR_FC:g} fc while in use": f0(v["stair_in_use_violation_minutes"])})
st.dataframe(pd.DataFrame(wi), hide_index=True, width="stretch")

# ------------------------------------------------------------------------------------------------ 4. measured results
st.subheader("Measured results: models against simple baselines")
st.markdown(badge("REAL") + " Every score is on held-out data and sits next to the best naive baseline, "
            "including where the baseline wins. The interval is a 95% bootstrap over whole test days.")
card = pd.DataFrame(summary["model_card"])
card["95% CI of difference"] = card.apply(
    lambda r: (f"{r['diff_ci95'][0]:+.3f} to {r['diff_ci95'][1]:+.3f}" if isinstance(r.get("diff_ci95"), list) else "-"), axis=1)
card["verdict"] = card.get("verdict").fillna("-") if "verdict" in card else "-"
show = card[["model", "data", "split", "n_test", "n_pos", "metric", "model_score", "best_baseline", "baseline_score",
             "95% CI of difference", "verdict", "label"]].rename(columns={"n_test": "n test", "n_pos": "positives",
                                                                         "model_score": "score", "baseline_score": "baseline score",
                                                                         "best_baseline": "best baseline"})
st.dataframe(show, hide_index=True, width="stretch",
             column_config={"score": st.column_config.NumberColumn(format="%.3f"),
                            "baseline score": st.column_config.NumberColumn(format="%.3f")})
nr = summary["n_rooms"]


def _card(model: str):
    return [c for c in summary["model_card"] if c["model"] == model]


def _verdicts(model: str):
    cs = _card(model)
    better = [c for c in cs if c.get("verdict") == "model better"]
    worse = [c for c in cs if c.get("verdict") == "baseline better"]
    names = sorted({c["best_baseline"] for c in cs})
    return cs, better, worse, names


_m1, _m1b, _m1w, _m1n = _verdicts("M1 presence now")
_m2, _m2b, _m2w, _m2n = _verdicts("M2 presence +1 h (deployable)")
_room1 = m1["day_blocked"].get("1", {}).get("models", {}).get("m1_hgb", {}).get("prevalence")
st.markdown(
    f"- **M1 (who is here now):** clearly better than the best simple baseline (per room: {', '.join(_m1n)}) in "
    f"{len(_m1b)} of {len(_m1)} rooms, clearly worse in {len(_m1w)}; in the rest the difference is within noise. "
    + (f"Room 1 was occupied only {pct(100 * _room1)} of test time, so its scores are unstable." if _room1 is not None else "") + "\n"
    f"- **M2 (who will be here in 1 hour):** the deployable version is clearly better than the best baseline (per room: "
    f"{', '.join(_m2n)}) in {len(_m2b)} of {len(_m2)} rooms and clearly worse in {len(_m2w)}"
    + (f" ({', '.join(c['data'] + ': ' + f3(c['model_score']) + ' vs ' + c['best_baseline'] + ' ' + f3(c['baseline_score']) for c in _m2w)})" if _m2w else "")
    + ". The 'oracle' version needs true presence now, which no deployed system has. We use M2 only as an "
    f"input to planning, never for switching.\n"
    f"- **M3 (garage load tomorrow):** on {m3['summary']['n_meters']} real parking-garage meters, the model has a median error "
    f"(CV(RMSE)) of {f3(m3['summary']['median_cvrmse']['hgb_no_weather'])}, against {f3(m3['summary']['median_cvrmse']['lag24'])} for "
    f"'same hour yesterday'. It beats the better naive baseline on {m3['summary']['beats_best_naive']['hgb_no_weather']} of "
    f"{m3['summary']['n_meters']} meters.")
if chron:
    st.markdown(badge("REAL") + f" **Zero-shot comparator** `{chron['model']}` ({chron['licence']}), no training on these meters: "
                f"median CV(RMSE) {f3(chron['median_cvrmse_chronos'])} against {f3(chron['median_cvrmse_hgb_no_weather'])} for our model. "
                f"Better than our model on {chron['chronos_beats_hgb']} of {chron['n_meters']} meters, and better than the best naive "
                f"baseline on {chron['chronos_beats_best_naive']}. Whether BDG2 was in its pretraining data was not checked.")

ga, gb = st.columns(2)
with ga:
    prof_rows = []
    for mtr, vals in m3["profiles_weekday_norm"].items():
        for h, v in enumerate(vals):
            prof_rows.append({"meter": mtr, "hour": h, "load / meter mean": v})
    pf = pd.DataFrame(prof_rows)
    med = pf.groupby("hour", as_index=False)["load / meter mean"].median()
    base = alt.Chart(pf).mark_line(strokeWidth=1, color=MUTED, opacity=0.35).encode(
        x=alt.X("hour:Q", title="hour of day (weekdays)"), y=alt.Y("load / meter mean:Q", scale=alt.Scale(zero=False)),
        detail="meter:N", tooltip=["meter:N", "hour:Q", alt.Tooltip("load / meter mean:Q", format=".2f")])
    top = alt.Chart(med).mark_line(strokeWidth=2.5, color=BLUE).encode(x="hour:Q", y="load / meter mean:Q",
                                                                        tooltip=[alt.Tooltip("load / meter mean:Q", format=".2f", title="median")])
    st.altair_chart((base + top).properties(title=f"BDG2 garage meters draw about as much at night as by day (median "
                                                  f"night/day {m3['summary']['median_night_day_ratio']:.2f})"), width="stretch")
    st.caption("Whole-garage meters (lighting plus ventilation and other loads), mostly outside California; not lighting circuits.")
with gb:
    if m3ex and m3ex.get("meters"):
        mtr = st.selectbox("Garage meter (US/Pacific)", list(m3ex["meters"]))
        e = m3ex["meters"][mtr]
        ef = pd.DataFrame({"ts": pd.to_datetime(e["ts"]), "Actual": e["y"], "Forecast (M3)": e["hgb"]}).melt("ts", var_name="series", value_name="kWh")
        ch = alt.Chart(ef, title="Day-ahead forecast vs actual, first 2 weeks of the 2017 test year").mark_line(strokeWidth=2).encode(
            x=alt.X("ts:T", title=None), y=alt.Y("kWh:Q", title="kWh per hour"),
            color=alt.Color("series:N", scale=alt.Scale(domain=["Actual", "Forecast (M3)"], range=[BLUE, ORANGE]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[alt.Tooltip("ts:T", format="%Y-%m-%d %H:%M"), "series:N", alt.Tooltip("kWh:Q", format=".2f")])
        st.altair_chart(ch, width="stretch")

# ------------------------------------------------------------------------------------------------ 5. calendar
st.subheader(f"Year-round plan for Los Angeles ({cal['year']})")
st.markdown(badge("REAL", "REANALYSIS", "TARIFF APPROXIMATION") +
            " Climate normals: NOAA 1991-2020 (Downtown USC and LAX). Sunrise and sunset: NOAA solar equations. "
            "Tariff and demand-response seasons: LADWP and SCE. Holidays and DST: computed.")
modes = pd.DataFrame(cal["modes"])
_pref = ["Daylight off-peak", "Dark: sensor setback, egress >= 1 fc", "Price shoulder (LADWP low peak)",
         "Evening grid-stress watch", "Peak cap + DR ready"]
_seen = list(dict.fromkeys(modes["mode"].tolist()))
mode_order = [m for m in _pref if m in _seen] + [m for m in _seen if m not in _pref]
hm = alt.Chart(modes, title="Recommended common-area mode on a typical weekday").mark_rect(stroke="white", strokeWidth=1).encode(
    x=alt.X("hour:O", title="hour of day"), y=alt.Y("month:N", sort=[r["name"] for r in cal["rows"]], title=None),
    color=alt.Color("mode:N", scale=alt.Scale(domain=mode_order, range=(RAMP * 2)[:len(mode_order)]), sort=mode_order,
                    legend=alt.Legend(orient="bottom", columns=3, title=None)),
    tooltip=["month:N", "hour:O", "mode:N", alt.Tooltip("ladwp_usd_kwh_weekday:Q", format=".5f", title="LADWP energy $/kWh")])
st.altair_chart(hm, width="stretch")

ct = pd.DataFrame([{"month": r["name"], "CDD65 USC": r["cdd65_usc"], "HDD65 USC": r["hdd65_usc"], "CDD65 LAX": r["cdd65_lax"],
                    "day length h": r["day_length_h"], "sunset (15th)": r["sunset_local"], "LADWP season": r["ladwp_season"],
                    "SCE season": r["sce_season"], "LADWP DR days": r["ladwp_dr_days"],
                    "2025 mean temp C (reanalysis)": r.get("reanalysis_2025_tmean_c"),
                    "holidays": ", ".join(h["name"] for h in r["holidays"]), "DST": ", ".join(d["date"] + " " + d["change"] for d in r["dst"])}
                   for r in cal["rows"]])
st.dataframe(ct, hide_index=True, width="stretch",
             column_config={"2025 mean temp C (reanalysis)": st.column_config.NumberColumn(format="%.1f")})
st.caption(f"Annual CDD65: Downtown USC {f1(cal['annual_cdd_sum_usc'])} against LAX {f1(cal['annual_cdd_sum_lax'])}. "
           f"Where the tower sits matters.")

st.markdown("##### What the building manager does each season")
cols = st.columns(len(cal["plan"]))
for col, s in zip(cols, cal["plan"]):
    with col.container(border=True):
        st.markdown(f"**{s['season']}**")
        st.caption(s["why"])
        st.markdown("\n".join(f"- {d}" for d in s["do"]))
mname = st.selectbox("Month detail", [r["name"] for r in cal["rows"]])
mrow = next(r for r in cal["rows"] if r["name"] == mname)
st.markdown("\n".join(f"- {a}" for a in mrow["actions"]))

# ------------------------------------------------------------------------------------------------ proposals
st.subheader("Pending proposals (a person approves or rejects)")
st.markdown(badge("SEMI-SYNTHETIC", "SIMULATED") + " Expected savings come from the simulated tower year. Decisions made here "
            "are kept for this browser session only; the production path is the append-only proposal log.")
pp = pd.DataFrame(props["proposals"]) if props else pd.DataFrame()
if not pp.empty:
    if "decisions" not in st.session_state:
        st.session_state["decisions"] = {}
    pp["status"] = pp["id"].map(lambda i: st.session_state["decisions"].get(i, "pending_approval"))
    st.dataframe(pp[["id", "zone", "action", "reason", "expected_kwh_per_year", "data_label", "status"]], hide_index=True,
                 width="stretch", column_config={"expected_kwh_per_year": st.column_config.NumberColumn(
                     "kWh/yr saved (negative = costs energy)", format="%.0f")})
    pa, pb, pc, pd_ = st.columns([2, 2, 1, 1])
    pid = pa.selectbox("Proposal", pp["id"].tolist())
    who = pb.text_input("Reviewer name", "")
    if pc.button("Approve", disabled=not who.strip()):
        st.session_state["decisions"][pid] = f"approved by {who.strip()}"
        st.rerun()
    if pd_.button("Reject", disabled=not who.strip()):
        st.session_state["decisions"][pid] = f"rejected by {who.strip()}"
        st.rerun()

# ------------------------------------------------------------------------------------------------ 6. engineers
with st.expander("For engineers: method, metrics, limits"):
    st.markdown(
        "**Policy.** Level = max(code floor, target). Target = occupied level while the motion sensor's hold is active, otherwise "
        "the zone's vacant level. Egress holds are at least the NFPA minimum and fail to full ON on a sensor fault. "
        f"`sensor_ml` lets M1 (probability >= {replay['ml_threshold']:g}, fixed in advance) keep an already-ON zone ON; it cannot switch a zone on or off. "
        "`sensor_ml_union` also lets M1 switch on (reported for transparency). DR trims only non-egress zones and never goes below the floor. "
        f"We first built the union rule; on the room rule the test-day replay gave it {pct(u_saved)} saved against {pct(s_saved)} for "
        f"sensor-only, so the recommended policy became extend-only. That choice used the test days; the training-day check "
        f"is in the caption under the replay chart. Both rules are reported.")
    st.markdown("**Rule table** (from `common_area_rules.json`; not a compliance tool)")
    rt = pd.DataFrame(rules["table"])
    rt["value"] = rt["value"].map(lambda v: "-" if v is None else str(v))  # numbers and words share this column
    st.dataframe(rt, hide_index=True, width="stretch")
    st.markdown(f"**M1 features:** {', '.join(m1['features'])}. **Excluded as leakage:** {', '.join(m1['excluded_as_leakage'])}.")
    rows = []
    for split in ("day_blocked", "leave_one_room_out"):
        for room, r in m1[split].items():
            for mn, mm in r["models"].items():
                rows.append({"split": split, "room": room, "model": mn, "n": mm["n"], "positives": mm["n_pos"], "F1": mm["f1"],
                             "balanced acc": mm["bal_acc"], "ROC AUC": mm["auc"], "false-vacancy rate": mm["false_vacancy_rate"]})
    st.markdown("**M1 all models and splits**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    rows = []
    for room, r in m2["rooms"].items():
        for mn, mm in r["models"].items():
            rows.append({"room": room, "model": mn, "n": mm["n"], "positives": mm["n_pos"], "F1": mm["f1"], "balanced acc": mm["bal_acc"]})
    st.markdown("**M2 (1 h ahead)**: " + m2["note"])
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("**M3 per meter** (CV(RMSE); oracle weather uses observed same-hour temperature, which is not available a day ahead)")
    mt = pd.DataFrame(m3["meters"])[["meter", "timezone", "night_day_ratio", "n_test_h", "cvrmse_lag168", "cvrmse_lag24",
                                     "cvrmse_hgb_oracle_weather", "cvrmse_hgb_lagged_weather", "cvrmse_hgb_no_weather",
                                     "hgb_no_weather_beats_best_naive"]]
    st.dataframe(mt, hide_index=True, width="stretch")
    st.markdown(f"Skipped meters: " + ", ".join(f"{s['meter']} ({s['reason']}, coverage {s['coverage']})" for s in m3["skipped"]))
    if chron:
        st.markdown("**Chronos-Bolt zero-shot per meter**")
        st.dataframe(pd.DataFrame(chron["meters"]), hide_index=True, width="stretch")
    if uci:
        rows = []
        for tn, t in uci["tests"].items():
            for mn, mm in t["models"].items():
                rows.append({"test file": tn, "note": t["note"], "model": mn, "n": mm["n"], "F1": mm["f1"], "balanced acc": mm["bal_acc"]})
        st.markdown("**UCI 357 leakage check.** The Light feature includes the lights being controlled, so any score that uses it is leakage.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if pir:
        st.markdown(f"**Injected sensor rates (UCI 864):** miss rate {pir['miss_rate']:.4f} over {pir['n_occupied_windows']} fully "
                    f"occupied 5-min windows; false-trigger rate {pir['false_trigger_rate']:.4f} over {pir['n_vacant_windows']} vacant windows. "
                    "Misses are injected independently per step, which is optimistic for a real corridor.")
    rr = []
    pk_txt = ", ".join(f"{a:02d}-{b:02d} h" for a, b in replay.get("peak_windows_weekday", [])) or "high peak"
    for room, r in replay["rooms"].items():
        for p, v in r["profiles"][prof]["policies"].items():
            rr.append({"room": room, "policy": p, "kWh": v["kwh"], "% saved vs recorded": v["saved_vs_as_operated_pct"],
                       "under-lit share": v["underlit_occupied_share"], f"peak kW weekday {pk_txt}": v["peak_kw_wkday_high_peak"],
                       "switches/day": v["switches_per_day"]})
        rr.append({"room": room, "policy": "hypothetical 24/7 at p95 power", "kWh": r["recorded"]["hypothetical_247_p95_kwh"]})
    st.markdown(f"**Replay per room** ({replay['price_note']})")
    st.dataframe(pd.DataFrame(rr), hide_index=True, width="stretch")
    st.markdown("**Tower year: all policies**")
    tt = pd.DataFrame([{"policy": tower["policy_labels"].get(p, p + " (with DR trims)"), "annual kWh": v["annual_kwh"],
                        "energy $": v["energy_usd"], "demand $": v["demand_usd"], "facilities $": v["facilities_usd"],
                        "summer high-peak kW": v["peak_kw_summer_high_peak"], "under-lit share": v["underlit_occupied_share"],
                        f"stair minutes < {STAIR_FC:g} fc in use": v["stair_in_use_violation_minutes"],
                        f"egress < {EGRESS_FC:g} fc steps": v["egress_floor_violations"]}
                       for p, v in res.items()])
    st.dataframe(tt, hide_index=True, width="stretch")
    st.markdown("**Tower assumptions**\n" + "\n".join(f"- {a}" for a in tower["assumptions"]))
    st.dataframe(pd.DataFrame(tower["inventory"]["assumptions"]), hide_index=True, width="stretch")
    st.markdown(f"**Demand response:** simulated mean trim {k['dr_mean_trim_kw']:.2f} kW on {len(tower['dr_days'])} hot weekdays "
                f"({', '.join(tower['dr_days'])}). The LADWP program needs at least {k['ladwp_dr_min_kw']} kW building-wide, so common-area "
                "lighting is only a contribution to a building-level bid.")
    st.markdown(
        f"**Limits.** ROBOD is {summary['n_rooms']} university rooms in tropical Singapore ({day_span} days each); no public high-rise corridor or stair "
        "occupancy dataset was found, so the tower traces are resampled. How ROBOD's ground truth was collected is not documented. "
        "The replay can only remove recorded use; it cannot add light that was off. The tower ignores DST hour shifts, HVAC is not "
        "simulated, and LADWP adjustment factors are excluded. Each garage level is one control zone in the tower, although the codes "
        "require smaller zones. BDG2 garages are whole-garage meters, not lighting circuits. NFPA 101 "
        "and CFC values were read from secondary sources. Nothing here certifies code compliance.")
    if tar:
        st.markdown("**Tariff tables used**")
        st.json(tar["ladwp_a2b"], expanded=False)

# ------------------------------------------------------------------------------------------------ 7. sources
st.divider()
st.markdown("#### Sources & licences")
src_lines = []
if manifest:
    for key, s in manifest["sources"].items():
        rf = manifest["raw_files"].get(key, {})
        acc = rf.get("accessed") or "access date not recorded"
        prov = f"; {rf['provenance']}, sha256 {rf['sha256'][:12]}..." if rf.get("provenance") and rf.get("sha256") else ""
        src_lines.append(f"- {s['cite']} [{s['label']}], licence {s['licence']}: {s['url']} (accessed {acc}{prov})")
    for s in manifest.get("extra_sources", []):
        src_lines.append(f"- {s['cite']} [{s['label']}], licence {s['licence']}: {s['url']} (accessed {s['accessed']})")
for key, s in R["sources"].items():
    src_lines.append(f"- {s['title']}: {s['url']} (accessed {s['accessed']})")
if tar:
    for t in ("ladwp_a2b", "sce_tou", "ladwp_dr", "flex_alert"):
        for u in tar[t]["urls"]:
            src_lines.append(f"- {tar[t]['name']}: {u} (accessed {tar[t]['accessed']})")
if chron and chron.get("model_url") and chron.get("accessed"):
    src_lines.append(f"- Hugging Face {chron['model']} ({chron['licence']}): {chron['model_url']} (accessed {chron['accessed']})")
st.markdown("\n".join(dict.fromkeys(src_lines)))
if manifest:
    st.caption(f"Artifacts generated {manifest['generated_utc']} by `{manifest['command']}` in {manifest['runtime_s']} s. "
               "Open-Meteo's free tier is for non-commercial use. BDG2 is CC BY-SA (version ambiguous).")
