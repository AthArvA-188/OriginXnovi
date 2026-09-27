"""Pipes page: pipe-clog testing and prevention (module key: clog).

Standalone Streamlit page script for the multipage site (no st.set_page_config). Reads only precomputed artifacts in
eval/clog/ (made by scripts/clog_*.py) plus the module's rule and source tables. Every number shown comes from those
files; nothing is typed in here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cascade.building.clog import geometry as G  # noqa: E402
from cascade.building.clog import report as RPT  # noqa: E402
from cascade.building.clog import rules as R  # noqa: E402
from cascade.building.clog import svg as SVG  # noqa: E402

EVAL = ROOT / "eval" / "clog"
SRC = ROOT / "src" / "cascade" / "building" / "clog"
SEV = {"0": "clean (false alarm)", "1": "mild", "2": "moderate", "3": "severe"}
SEV_ORDER = list(SEV.values())
BADGE = {"REAL": "green", "SIMULATED": "blue", "SYNTHETIC": "orange", "INJECTED": "violet", "RULE": "gray"}


def load_json(name: str) -> dict:
    return _load_json_cached(name, _mtime(EVAL / name))


def _mtime(p) -> float:
    """File modification time, part of every cache key so regenerated artifacts are picked up without a restart."""
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def _load_json_cached(name: str, mtime: float) -> dict:
    return json.loads((EVAL / name).read_text(encoding="utf-8"))


def load_csv(name: str) -> pd.DataFrame:
    return _load_csv_cached(name, _mtime(EVAL / name))


@st.cache_data(show_spinner=False)
def _load_csv_cached(name: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(EVAL / name)


def badges(*labels: str) -> None:
    st.markdown(" ".join(f":{BADGE[x]}-badge[{x}]" for x in labels))


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def ci_txt(ci) -> str:
    return f"{pct(ci[0])}-{pct(ci[1])}"


def pct1(x) -> str:
    return "" if x is None or pd.isna(x) else f"{100 * x:.1f}%"


needed = ["riser_metrics.json", "drain_test.json", "drain_curves.csv", "bellinge_case.json",
          "bellinge_jul2020_5min.csv", "schedule.json", "data_manifest.json", "sim_check.json"]
missing = [n for n in needed if not (EVAL / n).exists()]
st.title("Pipe clogs: test, catch, prevent")
st.markdown("**A short nightly flow test and a risk-based check calendar are designed to catch clogs in a tower's "
            "water pipes and drains early. In our simulation the test caught nearly all severe clogs; it has not yet "
            "been tried in a real building.**")
if missing:
    st.error("Missing artifacts: " + ", ".join(missing) + ". Run the scripts listed in docs/research/16_pipe_clogging.md.")
    st.stop()

riser, drain, bell, sched = (load_json("riser_metrics.json"), load_json("drain_test.json"),
                             load_json("bellinge_case.json"), load_json("schedule.json"))
manifest, simchk = load_json("data_manifest.json"), load_json("sim_check.json")
qa, qb = riser["design"]["tests_lps"]
base = riser["variants"]["base"]
det = base["detectors"]

with st.container(border=True):
    st.markdown("#### What this means")
    st.markdown(
        f"- **Who it helps:** the building engineer and facilities manager of a high-rise.\n"
        f"- **The problem:** clogs build up quietly in water risers, strainers and drains. A pressure gauge that only "
        f"watches everyday use barely moves until the clog is large, and in-building drains often have no fixed "
        f"inspection interval and are checked after a backup.\n"
        f"- **What Cerebro does:** it proposes a short nightly test: a valve at the top of each pressure zone draws a "
        f"known flow ({qa:.1f} L/s) for {'-'.join(str(x) for x in R.value('ACTIVE_TEST_SECONDS'))} seconds while "
        f"pressure sensors measure how much head the pipe loses. A clog "
        f"shows up as extra loss between two sensors, which also tells you where it is. For drains, a known volume is "
        f"discharged and the time the level takes to fall back is compared with the day the drain was commissioned. "
        f"It also keeps a calendar of legally required and risk-based checks.\n"
        f"- **What the person does next:** approve the standing nightly test once (the building management system runs "
        f"it and aborts on low pressure), review alarms, and approve or reschedule each proposed work order. Cerebro "
        f"never opens a valve itself and never touches fire, sprinkler or standpipe piping.")

# ------------------------------------------------------------------------------------------------ main visual
st.subheader("Where the sensors go, and what the nightly test caught")
badges("SIMULATED", "INJECTED", "REAL")
st.caption(f"Design for a synthetic {G.FLOORS}-floor tower. Riser segments and strainers are coloured by the measured "
           f"share of moderate-or-severe clog nights the {qa:.1f} L/s test flagged (z-score rule) on held-out "
           f"simulated weeks, with clogs injected into a WNTR model driven by real apartment water use. Scenarios per "
           f"location: " + ", ".join(f"{r['loc']} {r['n_scenarios']}" for r in
                                      riser["moderate_plus_by_location"]["act4_z"]) + ".")
loc_rate = {r["loc"]: r["rate"] for r in riser["moderate_plus_by_location"]["act4_z"]}
st.markdown(SVG.riser_svg(loc_rate, "share of moderate-or-severe clog nights flagged"), unsafe_allow_html=True)
with st.expander(f"Static pressure at every floor (design check: every outlet at or below "
                 f"{G.MAX_STATIC_M * G.PSI_PER_M:.0f} psi)"):
    stat = pd.DataFrame(G.static_table())
    st.caption(f"Maximum static pressure {stat.static_psi.max():.1f} psi, minimum {stat.static_psi.min():.1f} psi "
               f"(computed from the zoning rules in geometry.py; UPC 608.2 limit "
               f"{R.value('UPC_608_2_PRV_80PSI')} m).")
    st.dataframe(stat, hide_index=True, height=240)

# ------------------------------------------------------------------------------------------------ results
st.subheader("Measured results")
t_sup, t_drain, t_real = st.tabs(["Supply riser (simulated)", "Drains (simulated)", "Real blockage (Bellinge)"])

with t_sup:
    badges("SIMULATED", "INJECTED", "REAL")
    n_wk = riser["design"]["test"].get("n_weeks", len(riser["design"]["test"]["weeks"]))
    st.markdown(f"**Alarm rate by true clog size, held-out test.** Each bar is the share of nightly tests that raised "
                f"an alarm; the 'clean' bar is the false-alarm rate. Whiskers are 95% intervals from a two-stage "
                f"bootstrap (the {n_wk} test demand weeks, then scenarios within each week). Test set: "
                f"{base['n_scenarios']} scenario-weeks ({base['n_nights']} nights) on a different apartment-to-unit "
                f"map (the same HSB apartments, the same simulated riser) and weeks never used for training. Clogs "
                f"are INJECTED into a SIMULATED riser; demand is REAL (HSB Living Lab).")
    vkeys = list(riser["variants"].keys())
    vsel = st.selectbox("Test condition", vkeys, format_func=lambda k: riser["variants"][k]["label"], index=0)
    vdet = riser["variants"][vsel]["detectors"]
    default = ["bms", "pas_hgb", "act25_z", "act4_z", "act4_hgb"]
    chosen = st.multiselect("Detectors", list(vdet.keys()), default=default,
                            format_func=lambda k: vdet[k]["label"])
    rows = []
    for k in chosen:
        for s, a in vdet[k]["alarm"].items():
            rows.append({"detector": vdet[k]["label"], "true state": SEV[s], "alarm rate": a["rate"],
                         "lo": a["ci95"][0], "hi": a["ci95"][1], "scenarios": a["n_scenarios"],
                         "nights": a["n_nights"]})
    adf = pd.DataFrame(rows)
    if len(adf):
        base_ch = alt.Chart(adf).encode(
            x=alt.X("true state:N", sort=SEV_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("detector:N"),
            color=alt.Color("detector:N", legend=alt.Legend(orient="bottom", columns=2, title=None)))
        bars = base_ch.mark_bar().encode(
            y=alt.Y("alarm rate:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title="nights alarmed"),
            tooltip=["detector", "true state", alt.Tooltip("alarm rate:Q", format=".1%"),
                     alt.Tooltip("lo:Q", format=".1%"), alt.Tooltip("hi:Q", format=".1%"), "scenarios", "nights"])
        whisk = base_ch.mark_rule(color="#111827").encode(y="lo:Q", y2="hi:Q")
        st.altair_chart((bars + whisk).properties(height=340), width="stretch")
    _hs = manifest.get("hsb_stats", {})
    if _hs.get("share_of_volume_without_apartment") is not None:
        st.caption(f"Demand note: HSB is Swedish living-lab apartment use ({_hs['n_apartments']} apartments). The "
                   f"{pct(_hs['share_of_volume_without_apartment'])} of metered volume with no apartment (shared "
                   f"spaces) is left out, so the simulated tower has no shared laundry or other common-area water use.")
    cnt = riser["variants"][vsel]["scenarios_by_severity"]
    st.caption("Scenarios per true state: " + ", ".join(f"{SEV[k]} {v}" for k, v in cnt.items()) +
               f". Alarm thresholds were fixed on out-of-fold clean training nights (target "
               f"{pct(R.value('FALSE_ALARM_TARGET'))}), never on test weeks.")
    st.markdown("**Grading and locating (held-out test)**")
    grows = []
    for k, d in det.items():
        sv = d.get("severity")
        lo = d.get("localisation", {})
        grows.append({
            "detector": d["label"],
            "macro-F1 (4 classes)": f"{sv['macro_f1']:.2f} ({sv['macro_f1_ci95'][0]:.2f}-{sv['macro_f1_ci95'][1]:.2f})"
            if sv else "detection only",
            "QWK": f"{sv['qwk']:.2f}" if sv else "",
            "locate moderate": f"{pct(lo['2']['accuracy'])} ({ci_txt(lo['2']['ci95'])})" if lo else "",
            "locate severe": f"{pct(lo['3']['accuracy'])} ({ci_txt(lo['3']['ci95'])})" if lo else "",
            "persistent alarm, clean weeks": pct(d["alarm"]["0"]["persistent_2of3"]),
            "persistent alarm, moderate weeks": pct(d["alarm"]["2"]["persistent_2of3"]),
        })
    blf = riser.get("baselines_macro_f1", {})
    blank = {"QWK": "", "locate severe": "", "persistent alarm, clean weeks": "",
             "persistent alarm, moderate weeks": ""}
    if blf:
        names = blf["class_names"]
        grows.append({"detector": f"Baseline: always '{blf['majority_class_name']}' (training majority)",
                      "macro-F1 (4 classes)": f"{blf['majority_macro_f1']:.2f}",
                      "locate moderate": f"chance {pct(1 / len(G.LOCATIONS))}", **blank})
        for k, v in blf["constant_macro_f1"].items():
            if int(k) != blf["majority_class"]:
                grows.append({"detector": f"Baseline: always '{names[int(k)]}'", "macro-F1 (4 classes)": f"{v:.2f}",
                              "locate moderate": "", **blank})
        sr = blf["stratified_random"]
        grows.append({"detector": "Baseline: random grade at training class frequencies",
                      "macro-F1 (4 classes)": f"{sr['mean']:.2f} ({sr['range95'][0]:.2f}-{sr['range95'][1]:.2f})",
                      "locate moderate": "", **blank})
    else:
        grows.append({"detector": "Baseline: majority class", "macro-F1 (4 classes)":
                      f"{riser['baseline_majority_macro_f1']:.2f}",
                      "locate moderate": f"chance {pct(1 / len(G.LOCATIONS))}", **blank})
    st.dataframe(pd.DataFrame(grows), hide_index=True)
    if blf:
        sr_mean = blf["stratified_random"]["mean"]
        below = [d["label"] for d in det.values() if d.get("severity") and d["severity"]["macro_f1"] < sr_mean]
        st.caption(f"Macro-F1 is scored on {base['n_nights']} test nights. A random grade drawn with the training "
                   f"class frequencies scores {sr_mean:.2f} (range over {blf['stratified_random']['n_draws']} draws "
                   f"in brackets). " + (f"Below that random baseline: {'; '.join(below)}." if below else
                                        "Every grading detector scored above that random baseline."))
    kk, nn = R.value("PERSISTENCE_2_OF_3")
    st.caption(f"Persistent alarm = at least {kk} alarms in any {nn} consecutive nights of the scenario week (the "
               f"rule that would open a work order).")
    st.markdown("**Stress tests (out of distribution)**: the same detectors and thresholds, conditions they were "
                "not trained on. The constant-offset row is a check, not a stress: a constant offset cancels in the "
                "active test's before/after jump by construction, so its active-test rows equal the base run. A "
                "drifting offset was not tested.")
    orows = []
    for v, vr in riser["variants"].items():
        for k in ("act4_z", "act4_hgb", "pas_z", "bms"):
            a = vr["detectors"][k]["alarm"]
            orows.append({"condition": vr["label"], "detector": vr["detectors"][k]["label"],
                          "false alarms": f"{pct(a['0']['rate'])} ({ci_txt(a['0']['ci95'])})",
                          "moderate": pct(a["2"]["rate"]), "severe": pct(a["3"]["rate"])})
    st.dataframe(pd.DataFrame(orows), hide_index=True, height=300)

with t_drain:
    badges("SIMULATED", "INJECTED")
    dm = drain["model"]
    st.markdown(f"**Known-volume drain-down test in a simulated building drain.** {dm['test']['lps']} L/s for "
                f"{dm['test']['seconds']} s into the stack; the clog is an orifice left partly open (fully open = clean). "
                f"Level sensor noise (sd {dm['level_noise']['sd'] * 100:.1f} cm) and a "
                f"{dm['level_noise']['step'] * 100:.0f} cm logging step are assumptions.")
    curves = load_csv("drain_curves.csv")
    curves["opening"] = curves["opening"].map(lambda o: f"{o:.0%} open")
    st.altair_chart(alt.Chart(curves).mark_line().encode(
        x=alt.X("t_s:Q", title="seconds"), y=alt.Y("cleanout_depth_m:Q", title="cleanout depth (m)"),
        color=alt.Color("opening:N", title="pipe open")).properties(height=280), width="stretch")
    drows = pd.DataFrame(drain["rows"])
    long = drows.melt(id_vars=["opening"], value_vars=["detect_rate_drain_down", "detect_rate_peak"],
                      var_name="signal", value_name="flagged")
    long["signal"] = long["signal"].map({"detect_rate_drain_down": "drain-down time",
                                         "detect_rate_peak": "peak level"})
    st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("opening:Q", scale=alt.Scale(reverse=True), axis=alt.Axis(format="%"), title="share of pipe open"),
        y=alt.Y("flagged:Q", axis=alt.Axis(format="%"), title="tests flagged"),
        color=alt.Color("signal:N", title=None),
        tooltip=["opening", "signal", alt.Tooltip("flagged:Q", format=".0%")]).properties(height=260),
        width="stretch")
    th = drain["thresholds"]
    st.caption(f"Thresholds: 95th percentile of clean calibration draws (drain-down ratio "
               f"{th['drain_down_ratio_p95_clean']:.2f}, peak ratio {th['peak_ratio_p95_clean']:.2f}). Evaluation: "
               f"{drows.n_draws.iloc[0]} level-sensor noise redraws per opening over "
               f"{len(dm['base_flows_lps'])} deterministic SWMM runs (night base flows "
               f"{', '.join(str(b) for b in dm['base_flows_lps'])} L/s), separate from the calibration draws. "
               f"Commissioning drain-down {drain['commissioning']['drain_down_median_s']:.0f} s.")
    show = drows[["opening", "drain_down_median_s", "censored_share", "detect_rate_drain_down", "detect_rate_peak",
                  "share_urgent"]].copy()
    show["opening"] = show["opening"].map(lambda o: f"{o:.0%} open")
    for c in ("censored_share", "detect_rate_drain_down", "detect_rate_peak", "share_urgent"):
        show[c] = show[c].map(pct1)
    show = show.rename(columns={
                      "drain_down_median_s": "drain-down median (s)", "censored_share": "never drained (share)",
                      "detect_rate_drain_down": "flagged by drain-down", "detect_rate_peak": "flagged by peak level",
                      "share_urgent": "graded urgent"})
    st.dataframe(show, hide_index=True)

with t_real:
    badges("REAL")
    st.markdown(f"**The one documented real blockage we could use.** {bell['label']}. Upstream gauge "
                f"{bell['gauges']['upstream']}, downstream gauge {bell['gauges']['downstream']}. The log says: "
                f"*\"{bell['source']['quote']}\"*")
    bj = load_csv("bellinge_jul2020_5min.csv")
    bj["time_local"] = pd.to_datetime(bj["time_local"])
    lines = bj.melt(id_vars=["time_local"], value_vars=["up_m", "dn_m"], var_name="gauge", value_name="depth_m")
    lines["gauge"] = lines["gauge"].map({"up_m": "upstream (G71F04R)", "dn_m": "downstream (G71F06R)"})
    ev = pd.DataFrame({"start": [pd.Timestamp(bell["periods"]["event"][0])],
                       "end": [pd.Timestamp(bell["periods"]["event"][1]) + pd.Timedelta(days=1)]})
    h = bell["headline"]
    marks = [{"t": pd.Timestamp(bell["onset"]["time"]), "what": "visible onset", "col": "#111827"},
             {"t": pd.Timestamp(h["paired"]["first_alarm"]), "what": "paired rule", "col": "#0f766e"},
             {"t": pd.Timestamp(h["residual_only"]["first_alarm"]), "what": "residual only", "col": "#7c3aed"}]
    fixed_cols = ["#f59e0b", "#ea580c", "#dc2626", "#9f1239"]
    for f, c in zip(h["fixed_levels"], fixed_cols):
        if f["first_alarm"]:
            marks.append({"t": pd.Timestamp(f["first_alarm"]), "what": "fixed " + f["label"].replace("train ", ""),
                          "col": c})
    mk = pd.DataFrame([m for m in marks if not pd.isna(m["t"])])
    gauge_scale = alt.Scale(domain=["upstream (G71F04R)", "downstream (G71F06R)"], range=["#334155", "#94a3b8"])
    mark_scale = alt.Scale(domain=mk["what"].tolist(), range=mk["col"].tolist())

    def bell_chart(lines_df, mk_df, ev_df, x_title, fmt, height, legend=True):
        rect = alt.Chart(ev_df).mark_rect(opacity=0.12, color="#d97706").encode(x="start:T", x2="end:T")
        ln = alt.Chart(lines_df).mark_line(strokeWidth=1.2).encode(
            x=alt.X("time_local:T", title=x_title, axis=alt.Axis(format=fmt)), y=alt.Y("depth_m:Q", title="depth (m)"),
            color=alt.Color("gauge:N", title=None, scale=gauge_scale,
                            legend=alt.Legend(orient="bottom") if legend else None))
        rl = alt.Chart(mk_df).mark_rule(strokeDash=[4, 3], strokeWidth=1.6).encode(
            x="t:T", color=alt.Color("what:N", title="first alarm", scale=mark_scale,
                                     legend=alt.Legend(orient="bottom", columns=4, labelLimit=220) if legend else None),
            tooltip=[alt.Tooltip("t:T", format="%d %b %H:%M"), "what"])
        return alt.layer(rect, ln, rl).resolve_scale(color="independent").properties(height=height)

    st.altair_chart(bell_chart(lines, mk, ev, "local time, July 2020", "%d %b", 300), width="stretch")
    z0, z1 = pd.Timestamp(bell["onset"]["time"]).normalize() + pd.Timedelta(hours=17), \
        pd.Timestamp(bell["onset"]["time"]).normalize() + pd.Timedelta(hours=22)
    zl = lines[(lines.time_local >= z0) & (lines.time_local <= z1)]
    zm = mk[(mk.t >= z0) & (mk.t <= z1)]
    ze = pd.DataFrame({"start": [max(z0, ev.start.iloc[0])], "end": [z1]})
    st.markdown(f"Zoom: {z0:%d %b} {z0:%H:%M}-{z1:%H:%M}, the order of the first alarms")
    st.altair_chart(bell_chart(zl, zm, ze, f"local time, {z0:%d %b %Y}", "%H:%M", 220, legend=False),
                    width="stretch")
    st.caption("Shaded: documented blockage window (log gives dates only). Onset: " + bell["onset"]["definition"] + ".")
    brow = [{"detector": "paired rule (upstream above normal AND downstream starved)",
             "first alarm": h["paired"]["first_alarm"],
             "minutes after onset": h["paired"]["delay_min_from_onset"],
             "other episodes, held-out months": h["paired"]["other_test_episodes"],
             "episodes, training months": h["paired"]["train_episodes"]},
            {"detector": "upstream residual only", "first alarm": h["residual_only"]["first_alarm"],
             "minutes after onset": h["residual_only"]["delay_min_from_onset"],
             "other episodes, held-out months": h["residual_only"]["other_test_episodes"],
             "episodes, training months": h["residual_only"]["train_episodes"]},
            {"detector": "downstream drop only", "first alarm": h["downstream_drop_only"]["first_alarm"],
             "minutes after onset": h["downstream_drop_only"]["delay_min_from_onset"],
             "other episodes, held-out months": h["downstream_drop_only"]["other_test_episodes"],
             "episodes, training months": h["downstream_drop_only"]["train_episodes"]}]
    for f in h["fixed_levels"]:
        brow.append({"detector": f"fixed high level > {f['threshold_m']:.3f} m ({f['label']})",
                     "first alarm": f["first_alarm"], "minutes after onset": f["delay_min_from_onset"],
                     "other episodes, held-out months": f["other_test_episodes"],
                     "episodes, training months": f["train_episodes"]})
    for r_, src in zip(brow, [h["paired"], h["residual_only"], h["downstream_drop_only"]] + h["fixed_levels"]):
        n_pre = src.get("pre_onset_episodes", 0)
        r_["note"] = (f"{n_pre} episode(s) in the event window started before onset (first {src['pre_onset_first']}): "
                      f"fired before onset, not attributable; counted as other episodes") if n_pre else ""
    st.dataframe(pd.DataFrame(brow), hide_index=True)
    sw = bell["sweep"]["paired_summary"]
    st.caption(f"Held-out period {bell['test_months']:.1f} months. Paired-rule sweep over {sw['n_configs']} settings "
               f"(threshold quantile, downstream ratio, window): {sw['n_detected']} detected the event, first alarms "
               f"{sw['first_alarm_range'][0]} to {sw['first_alarm_range'][1]}, other held-out episodes "
               f"{sw['other_test_episodes_range'][0]}-{sw['other_test_episodes_range'][1]}. An episode inside the "
               f"documented dates counts as a detection only if it starts no more than "
               f"{h['paired'].get('pre_onset_margin_min', 60):.0f} min before the visible onset. One event: a case "
               f"study, not an accuracy.")

# ------------------------------------------------------------------------------------------------ schedule
st.subheader("Proposed check calendar")
badges("SYNTHETIC", "SIMULATED", "RULE")
st.caption(f"As of {sched['as_of']}. {sched['register_label']}. Every row is a proposal that a person approves.")
srows = pd.DataFrame([{"next check": r["next_check"], "status": r["status"], "asset": f"{r['asset_id']}: {r['asset']}",
                       "why": r["reason"], "who does it": r["performed_by"],
                       "rule basis": ", ".join(f"{k} ({t})" for k, t in zip(r["rule_keys"], r["rule_tags"])),
                       "input": r["input_provenance"],
                       "approval": "requires human approval" if r["requires_human_approval"] else ""}
                      for r in sched["rows"]])
st.dataframe(srows, hide_index=True)

# ------------------------------------------------------------------------------------------------ conclusions
st.subheader("Conclusions")
with st.container(border=True):
    for c in RPT.conclusions(riser, drain, bell, sched):
        st.markdown(f"**{c['topic']}** `{c['label']}`  \n{c['text']}")

# ------------------------------------------------------------------------------------------------ engineers
with st.expander("For engineers: method, metrics, limits"):
    d = riser["design"]
    st.markdown(
        f"**Riser model.** WNTR (EPANET engine), {G.FLOORS} floors, three pressure zones: PRVs feed the low and mid "
        f"zones, the booster feeds the top zone directly. Clog = extra minor-loss K on one pipe or strainer, drawn "
        f"log-uniform; severity bands {d['k_bands']}. Noise: pressure sd {d['noise']['pressure_sd_m']} m per reading "
        f"(assumption); flow sd max({pct(d['noise']['flow']['rel'])} of reading, "
        f"{d['noise']['flow']['abs_velocity_ms']} m/s x pipe area) (vendor spec). Unit of evaluation: "
        f"{d['unit_of_evaluation']}. Thresholds: {d['threshold']}. CIs: {d['ci']}.\n\n"
        f"**Train** weeks {', '.join(d['train']['weeks'])} ({d['train']['n_scenarios']} scenarios); **test** weeks "
        f"{', '.join(d['test']['weeks'])}; commissioning: {d['commissioning']}.\n\n"
        f"**Test window.** EPANET solves the settled flow at one 10-minute step; the proposed "
        f"{'-'.join(str(x) for x in R.value('ACTIVE_TEST_SECONDS'))} s hold and the transient after the valve opens "
        f"are not simulated.\n\n"
        f"**Features.** For each segment, (head-loss jump during the test step) / (zone-flow^2 jump) minus the "
        f"commissioning value. Passive: head-loss residual over the busiest steps of the day against a "
        f"commissioning Hazen-Williams fit. BMS baseline: daily drop of the zone-top minimum pressure.")
    st.markdown("**Thresholds (from training data only)**")
    st.dataframe(pd.DataFrame([{"detector": riser["variants"]["base"]["detectors"][k]["label"], **v}
                               for k, v in riser["thresholds"].items()]), hide_index=True)
    st.markdown("**Commissioning sensitivity**: the test building's reference week redrawn with new sensor noise.")
    cs = riser.get("commissioning_sensitivity", {})
    st.dataframe(pd.DataFrame([{"detector": det[k]["label"], "true state": SEV[s], "median": pct1(v["median"]),
                                "min": pct1(v["min"]), "max": pct1(v["max"]), "draws": v["n_draws"]}
                               for k, dd in cs.items() for s, v in dd.items()]), hide_index=True)
    st.markdown("**Alarm rate by clog size K (held-out test)**")
    kb = pd.DataFrame([{"detector": det[k]["label"], "K": f"{r['k_lo']}-{r['k_hi']}", "k_lo": r["k_lo"],
                        "rate": r["rate"], "scenarios": r["n_scenarios"]}
                       for k, rows_ in riser["rate_by_k"].items() for r in rows_])
    st.altair_chart(alt.Chart(kb).mark_line(point=True).encode(
        x=alt.X("k_lo:Q", scale=alt.Scale(type="log"), title="K (lower edge of bin, log scale)"),
        y=alt.Y("rate:Q", axis=alt.Axis(format="%"), title="nights alarmed"), color=alt.Color("detector:N", title=None),
        tooltip=["detector", "K", alt.Tooltip("rate:Q", format=".0%"), "scenarios"]).properties(height=260),
        width="stretch")
    st.markdown("**Simulation engine check**")
    eng = pd.DataFrame(simchk["engine_check"]["cases"])
    eng["clog_pipe"] = eng["clog_pipe"].map(lambda v: "none (clean)" if v is None or pd.isna(v) else v)
    st.dataframe(eng, hide_index=True)
    st.caption(f"{simchk['n_sims']} simulated weeks; engines {simchk['engine_counts']}; "
               f"{simchk['engine_check']['note']}")
    st.markdown("**Bellinge normal-behaviour model**")
    nm = bell["normal_model"]
    st.markdown(f"{nm['type']} on {', '.join(nm['features'])}. MAE out-of-fold on training months "
                f"{nm['mae_train_oof_m']:.4f} m; on held-out months (event excluded) {nm['mae_test_excl_event_m']:.4f} m. "
                f"Threshold: {nm['threshold_basis']}.")
    st.markdown("**Rule table** (tags: PUBLIC = quoted from a source, team-proposed = our design, Assumption)")
    rules_df = pd.DataFrame(R.rows())
    rules_df["value"] = rules_df["value"].map(lambda v: v if isinstance(v, str) else json.dumps(v))
    st.dataframe(rules_df, hide_index=True, height=260)
    st.markdown("**Limits**")
    hsb = manifest.get("hsb_stats", {})
    shared = hsb.get("share_of_volume_without_apartment")
    target = R.value("FALSE_ALARM_TARGET")
    fa_txt = "; ".join(f"{det[k]['label']}: {100 * det[k]['alarm']['0']['rate']:.1f}%" for k in det if k != "bms")
    lim = ["Riser, clogs, geometry and noise are simulated; K bands do not map to a measured blockage state.",
           f"HSB demand is Swedish living-lab apartment use without shared facilities"
           + (f" ({pct(shared)} of the metered volume has no apartment and is dropped)" if shared is not None else "")
           + f"; the demand x{simchk['la_demand_scale']} stress test stands in for a busier LA tower.",
           "Train and test share the simulator and riser geometry; the test 'building' is the same HSB apartments "
           "on a different apartment-to-unit map. Only that map, the weeks, clog draws and noise differ.",
           f"95% CIs: {d['ci']}. With only {n_wk} test weeks the intervals are coarse.",
           "The constant-offset row checks that the offset cancels in the active test; a drifting offset was not "
           "tested.",
           f"Thresholds target {pct(target)} false alarms on training nights; on the new test building the clean-night "
           f"rates were {fa_txt}. Thresholds do not transfer exactly between buildings.",
           f"Solver: {simchk['engine_counts']} simulated weeks by engine. Weeks where EPANET fails fall back to WNTR's "
           f"own solver, and which weeks fall back can differ between reruns; the engine check above bounds the "
           f"difference at {max(c['max_abs_diff_m'] for c in simchk['engine_check']['cases'])} m of pressure.",
           "Drain model has one geometry and no fixture-event ML (descoped).",
           "No clog ramp / detection-delay experiment was run."] + bell["caveats"]
    st.markdown("\n".join(f"- {x}" for x in lim))

# ------------------------------------------------------------------------------------------------ provenance
st.subheader("Data provenance")
prov = [{"item": f["dataset"], "kind": f["kind"], "licence": f["licence"],
         "bytes": f["bytes"], "sha256": (f["sha256"] or "")[:16], "url": f["url"]} for f in manifest["files"]]
prov += [{"item": f"Riser simulations ({simchk['n_sims']} weeks, WNTR)", "kind": "SIMULATED", "licence": "ours",
          "bytes": None, "sha256": "", "url": "scripts/clog_riser_sim.py"},
         {"item": "Partial clogs (minor-loss K, orifice openings)", "kind": "INJECTED", "licence": "ours",
          "bytes": None, "sha256": "", "url": "scripts/clog_riser_sim.py, scripts/clog_drain_sim.py"},
         {"item": "Demo asset register (dates, backups, FOG readings)", "kind": "SYNTHETIC", "licence": "ours",
          "bytes": None, "sha256": "", "url": "scripts/clog_schedule.py"}]
st.dataframe(pd.DataFrame(prov), hide_index=True)
st.caption("Not used: " + "; ".join(manifest["not_used"]) + ".")

st.markdown("---")
st.markdown("**Sources & licences**")
srcs = json.loads((SRC / "sources.json").read_text(encoding="utf-8"))
st.markdown("\n".join(f"{s['n']}. [{s['title']}]({s['url']}) - {s['licence']}; {s['used_for']}. Accessed "
                      f"{srcs['accessed']}." for s in srcs["sources"]))
st.caption("Advisory only. Cerebro proposes tests, alarms and checks; a person approves every action; the software "
           "never actuates equipment or overrides life-safety systems.")
