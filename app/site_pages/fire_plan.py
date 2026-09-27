"""Fire plan page (Pavilion Cerebro local site, key "fire"): a sourced de-escalation plan card for any floor of the
SYNTHETIC tower, three drawings (building section, flowchart, swimlane), and a REAL-data alarm-verifier result held out
on an unseen room, shown next to naive baselines even where the model loses.

Numbers on this page come from files our scripts wrote (eval/fire/*.json, *.csv) or from the deterministic planner
(cascade.building.fire) run on the synthetic tower with eval/fire/fire_layer.json. None is typed here.
Top-level Streamlit script for st.navigation: no st.set_page_config.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "eval" / "fire"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cascade.building import fire  # noqa: E402
from cascade.building.synthetic import TowerConfig, build_tower_model  # noqa: E402

ZONE_CHOICES = {"East office": "E", "West office": "W", "North office": "N", "South office": "S", "Core (elevators, lobby)": "CORE",
                "Shaft (risers)": "SHAFT", "Electrical room": "ELEC"}
METHOD_ORDER = ["stage1_pm", "co_delta", "ml_alone", "two_stage"]
SHORT = {"stage1_pm": "PM trigger", "co_delta": "CO trigger", "ml_alone": "ML alone", "two_stage": "Two-stage (ours)"}
KIND_COLOR = {"fire": "#c62828", "nuisance": "#1f6fb2", "other": "#8a8984"}
TABLE_LABEL = {"stage1_pm": "PM trigger alone (baseline)", "co_delta": "CO trigger alone (baseline)",
               "ml_alone": "ML alone, no gate (baseline)", "two_stage": "Two-stage (ours)"}


def load_json(name: str):
    return _load_json_cached(name, _mtime(EVAL / name))


def _mtime(p) -> float:
    """File modification time, part of every cache key so regenerated artifacts are picked up without a restart."""
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def _load_json_cached(name: str, mtime: float):
    p = EVAL / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_csv(name: str):
    return _load_csv_cached(name, _mtime(EVAL / name))


@st.cache_data(show_spinner=False)
def _load_csv_cached(name: str, mtime: float):
    p = EVAL / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource(show_spinner="Building the synthetic tower model ...")
def tower():
    b = build_tower_model(TowerConfig())
    p = EVAL / fire.LAYER_FILE
    layer = fire.load_fire_layer(p) if p.exists() else fire.synthetic_fire_layer(b)
    return b, layer, p.exists()


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def count_cell(block: dict, kind: str, exact: dict | None) -> str:
    r = block[kind]
    if not r["n"]:
        return "-"
    ci = (exact or {}).get(kind, {}).get("ci95") or r["ci95"]
    return f"{r['alarmed']}/{r['n']} ({pct(ci[0])}-{pct(ci[1])})"


def num(x, fmt="{:.1f}") -> str:
    return "n/a" if x is None else fmt.format(x)


RESULT_COLUMNS = {  # short header, width, help (the table stays narrow; details go in the tooltip)
    "Method": ("Method", "medium", None),
    "Fires": ("Fires caught", "small", "fire episodes with at least one alarm / all fire episodes (exact 95% interval)"),
    "Nuisances": ("Nuisances called fire", "small", "nuisance episodes with an alarm / all nuisance episodes (exact 95% interval); lower is better"),
    "Other": ("Other called fire", "small", "other releases (ethanol, CO) with an alarm / all such episodes"),
    "Latency": ("Median min to alarm", "small", "median minutes from the labelled start to the first alarm, detected fires only"),
    "Latency CI": ("Latency 95% CI", "small", "bootstrap over episodes"),
    "Bg rate": ("Background / 24 sensor-h", "small", "alarms in clean background per 24 sensor-hours; lower is better"),
    "Bg CI": ("Background 95% CI", "small", "bootstrap over recording days"),
    "Bg count": ("Background alarms (sensor-h)", "small", "alarm count and clean-background sensor-hours behind the rate"),
}


def results_table(metrics: dict, exact_block: dict | None, methods_label: dict) -> pd.DataFrame:
    rows = []
    for m in METHOD_ORDER:
        r = metrics[m]
        ex = (exact_block or {}).get(m, {})
        lat = r["latency_min"]
        bg = r["background"]
        rows.append({
            "Method": TABLE_LABEL.get(m, methods_label.get(m, m)),
            "Fires": count_cell(r, "fire", ex),
            "Nuisances": count_cell(r, "nuisance", ex),
            "Other": count_cell(r, "other", ex),
            "Latency": num(lat["median"]),
            "Latency CI": f"{num(lat['median_ci95'][0])} to {num(lat['median_ci95'][1])}",
            "Bg rate": num(bg["per_24_sensor_h"], "{:.2f}"),
            "Bg CI": f"{num(bg['ci95'][0], '{:.2f}')}-{num(bg['ci95'][1], '{:.2f}')}",
            "Bg count": f"{bg['alarms']} ({bg['sensor_hours']:.0f} h)",
        })
    return pd.DataFrame(rows)


def show_results(df: pd.DataFrame) -> None:
    cfg = {k: st.column_config.TextColumn(lab, width=w, help=h, pinned=(k == "Method")) for k, (lab, w, h) in RESULT_COLUMNS.items()}
    st.dataframe(df, hide_index=True, width="stretch", column_config=cfg)


def separation_phrase(auc) -> str:
    """Plain words for a stage-2 AUROC read from the artifact (wording bins only; the number itself is printed next to it)."""
    if auc is None:
        return "could not be measured"
    if auc < 0.6:
        return "barely separated triggers"
    if auc < 0.75:
        return "separated triggers only weakly"
    if auc < 0.9:
        return "separated triggers moderately"
    return "separated triggers well"


# ------------------------------------------------------------------------------------------ 1. title and promise
st.title("Fire plan: the smallest safe response, drawn for any floor")
st.markdown("Pick where a fire starts in our synthetic demo tower and get a one-page plan: which floors hear the alarm, where "
            "people move, which stair firefighters climb and where their hose water will run. Every rule shows its basis: a cited "
            "code or guideline, or a [team-proposed, validate] tag. Every decision stays with people.")
st.error("**The AI never controls alarms, sprinklers, smoke control or elevators.** It never silences, delays or overrides the "
         "listed fire alarm system, never chooses the evacuation scope and never delays the 911 call. It prepares evidence; the "
         "Fire Safety Director and then the fire department incident commander decide.", icon="🛑")

# ------------------------------------------------------------------------------------------ 2. what this means
st.info("**What this means.** In a high-rise fire, the building's alarm system automatically warns the fire floor and the floors "
        "just above and below. Everyone else usually stays put: fire-safety guidance favours moving only the floors at risk, "
        "which NFPA's high-rise guide says has proved effective. The person in charge on site, the Fire Safety Director, must "
        "then decide quickly who moves where, while "
        "firefighters need to know which stair to climb, where to connect hoses and what hazards wait for them. "
        "This page drafts that plan in advance for every floor, so the director checks it instead of inventing it under "
        "stress. Next step for a building team: print the card for each floor, keep the impairment list current, "
        "and practise the decision points in drills. The plan is advice; the fire department takes command when it arrives.")

st.caption("Data labels: `SYNTHETIC` = our generated demo tower and its fire layer (stairs, elevators, occupants, impairments). "
           "`SIMULATED` = firefighting-water paths predicted on that tower's water graph. `REAL` = public laboratory sensor "
           "recordings (Mendeley, CC BY 4.0) used for the alarm-verifier test. `INJECTED` data are not used on this page.")

b, layer, layer_from_artifact = tower()
checks = load_json("planner_checks.json")

# ------------------------------------------------------------------------------------------ 3. main visual
st.subheader("Plan card for a chosen fire location  `SYNTHETIC`")
floors = [f.floor_id for f in fire.occupied_floors(b)]
c1, c2, c3 = st.columns([1, 1.3, 1.7])
floor_id = c1.selectbox("Fire floor", floors, index=floors.index("F19") if "F19" in floors else 0, key="fire_floor")
zone_label = c2.selectbox("Where on the floor", list(ZONE_CHOICES), index=0, key="fire_zone")
extra = c3.multiselect("Simulate a second alarm on (escalation test)", [f for f in floors if f != floor_id], default=[],
                       key="fire_extra", help="Adds another alarming floor to show how trigger E1 widens the alert zone.")
zone_id = f"{floor_id}-{ZONE_CHOICES[zone_label]}"
card = fire.plan_for_incident(b, layer, zone_id, detections=extra)


def fl(levels) -> str:
    levels = sorted(levels)
    if not levels:
        return "none"
    if len(levels) > 2 and levels == list(range(levels[0], levels[-1] + 1)):
        return f"F{levels[0]:02d}-F{levels[-1]:02d}"
    return ", ".join(f"F{x:02d}" for x in levels)


m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Floors alerted", fl(card.alert_floors))
m2.metric("People move to", fl(card.relocation_floors) if card.relocation_floors else "outside")
m3.metric("Fire dept. staging", f"F{card.staging_floor:02d}" if card.staging_floor else "ground")
m4.metric("Evacuation stair", f"Stair {card.evac_stair}")
m5.metric("Attack stair", f"Stair {card.attack_stair}", delta=f"hose at F{card.standpipe_landing:02d}", delta_color="off", delta_arrow="off")
fired = [t for t in card.triggers if t.status == "fired"]
if fired:
    st.warning("**Escalation triggers fired:** " + "; ".join(f"{t.id}: {t.detail}" for t in fired))
moved = [ln for ln in card.lines if ln.item.startswith("Staging floor") and fire.STAGING_MOVED_TAG in ln.basis]
if moved:
    st.caption(f"Staging: {moved[0].detail}. The sourced rule is {fire.ft('FIRE_STAGING_BELOW')} floors below the fire; "
               "the move is a team rule.")
st.caption(f"Design occupant load on the alert floors: {card.occupants['total_to_relocate']} people (a code design figure, "
           f"not a headcount); people needing help on these floors: {card.occupants['assistance_count']} (SYNTHETIC list). "
           "Suggested stairs are suggestions only: the fire department designates stairs on arrival.")

t_sec, t_flow, t_swim = st.tabs(["Building section", "De-escalation flowchart", "Who does what, in order"])
with t_sec:
    st.markdown(f"<div style='overflow-x:auto'>{fire.svg_section(card, layer)}</div>", unsafe_allow_html=True)
    st.caption("SYNTHETIC tower. Bands follow the sourced rules in the plan card; relocation and smoke-watch bands are "
               "[team-proposed, validate]. The blue arrow is SIMULATED from the tower's water graph.")
with t_flow:
    st.markdown(f"<div style='overflow-x:auto'>{fire.svg_flowchart(card)}</div>", unsafe_allow_html=True)
with t_swim:
    st.markdown(f"<div style='overflow-x:auto'>{fire.svg_swimlane(card)}</div>", unsafe_allow_html=True)

with st.expander("The full plan card, line by line, with the basis for each line", expanded=False):
    st.dataframe(pd.DataFrame([{"Section": ln.section, "Item": ln.item, "Detail": ln.detail,
                                "Approved by": ln.requires_approval_by or "-", "Basis": ln.basis} for ln in card.lines]),
                 hide_index=True, width="stretch")
    st.markdown("**Escalation triggers**")
    st.dataframe(pd.DataFrame([{"Id": t.id, "Trigger": t.text, "Status": t.status, "Detail": t.detail, "Basis": t.basis}
                               for t in card.triggers]), hide_index=True, width="stretch")
    st.markdown("**Verify on the fire alarm panel** (the listed system runs these; the AI does not)")
    st.dataframe(pd.DataFrame([{"Check": a.item, "Condition": a.detail, "Basis": a.basis} for a in card.automatic_sequence]),
                 hide_index=True, width="stretch")
    if card.water_watch:
        st.markdown("**Where firefighting water may go** `SIMULATED` on the `SYNTHETIC` tower's water graph (path score = product "
                    "of edge weights, [team-proposed, validate])")
        st.dataframe(pd.DataFrame([{"Node": w.node, "Floor": w.floor_id, "Kind": w.kind, "Path score": w.score, "Hops": w.hops}
                                   for w in card.water_watch[:15]]), hide_index=True, width="stretch")
    d1, d2 = st.columns(2)
    d1.download_button("Download plan card (Markdown)", fire.plan_card_markdown(card), file_name=f"fire_plan_{zone_id}.md",
                       mime="text/markdown", key="fire_dl_md")
    d2.download_button("Download plan card (JSON)", card.model_dump_json(indent=1), file_name=f"fire_plan_{zone_id}.json",
                       mime="application/json", key="fire_dl_json")

if checks:
    esc_txt = ""
    if "escalated_cards_checked" in checks:
        esc_txt = (f" Escalated cards ({checks['escalated_scenario']}): "
                   f"**{checks['escalated_cards_passing']} of {checks['escalated_cards_checked']}** pass the same checks "
                   f"plus the escalation checks; in {checks['escalated_staging_moved']} of them staging moves below the widened alert "
                   "set, a team rule tagged [team-proposed, validate] on the card.")
    st.success(f"`SYNTHETIC` Planner check: **{checks['cards_passing']} of {checks['cards_checked']}** plan cards "
               f"(every floor x office, core, shaft and electrical room of the {checks['tower_config']['floors']}-floor tower) pass "
               f"all {len(checks['rules_checked'])} rule checks.{esc_txt} This checks rule conformance, not real-world accuracy.")
if not layer_from_artifact:
    st.warning("eval/fire/fire_layer.json is missing; the fire layer was derived on the fly. Run scripts/fire_build_plans.py.")

# ------------------------------------------------------------------------------------------ 4. measured results
st.subheader("Can AI tell a real fire from a nuisance? Measured on an unseen room  `REAL`")
res = load_json("sensor_eval.json")
exact = load_json("exact_intervals.json")
sens = load_json("sensitivity_hall.json")
if not res or "hall" not in res:
    st.warning("Verifier results not found. Run: python scripts/fire_sensor_eval.py --stage prep, then --stage cv, then --stage hall.")
else:
    head = res["frozen_config"]["headline_features"]
    hall = res["hall"][head]["metrics"]
    cv = res["cv_en54"][head]["metrics"]
    ex_h = (exact or {}).get("blocks", {}).get(f"hall/{head}")
    ex_c = (exact or {}).get("blocks", {}).get(f"cv_en54/{head}")
    s1, ts = hall["stage1_pm"], hall["two_stage"]
    auc_h = hall["stage2_auroc"]
    better = ts["nuisance"]["alarmed"] < s1["nuisance"]["alarmed"]
    verdict = ("fewer" if better else "no fewer")
    auc_v = auc_h["auroc"]
    no_transfer = (not better) and auc_v is not None and auc_v < 0.75
    if no_transfer:
        outcome = ("**The model did not transfer to a new room.** That is exactly why the design lets it add evidence only and "
                   "never downgrade or delay an alarm.")
    elif better:
        outcome = (f"In the new room the label cut nuisance calls from {s1['nuisance']['alarmed']} to {ts['nuisance']['alarmed']} "
                   f"of {ts['nuisance']['n']}; with so few episodes the intervals below are wide. The design still lets it add "
                   "evidence only and never downgrade or delay an alarm.")
    else:
        outcome = ("The label did not reduce nuisance calls in the new room. The design lets it add evidence only and never "
                   "downgrade or delay an alarm.")
    st.markdown(
        f"We trained on recordings from one test room (EN54 room, {res['episodes']['en54']['n']} labelled episodes) and tested "
        f"**{'once' if res['hall']['meta']['hall_run_number'] == 1 else str(res['hall']['meta']['hall_run_number']) + ' times (see hall_runs.log)'}** on a different, unseen room (an industrial hall, {res['episodes']['hall']['n']} episodes: "
        f"{hall['two_stage']['fire']['n']} small fires, {hall['two_stage']['nuisance']['n']} nuisance sources such as deodorant, "
        f"exhaust or welding, and {hall['two_stage']['other']['n']} other releases). A conventional particle-rise trigger raised an "
        f"alarm for **{s1['fire']['alarmed']} of {s1['fire']['n']} fires** but also for "
        f"**{s1['nuisance']['alarmed']} of {s1['nuisance']['n']} nuisances**. Our AI label then called "
        f"**{ts['nuisance']['alarmed']} of {ts['nuisance']['n']} nuisances fire-like**: {verdict} than the trigger alone. "
        f"Its fire-versus-not score {separation_phrase(auc_v)} in the new room (AUROC {num(auc_v, '{:.2f}')}, "
        f"95% CI {num(auc_h['ci95'][0], '{:.2f}')}-{num(auc_h['ci95'][1], '{:.2f}')}, {auc_h['n_triggers']} triggers; 0.5 = chance). "
        + outcome)
    tab_h, tab_c = st.tabs(["Held-out room (headline)", "Same room, leave-one-day-out"])
    with tab_h:
        show_results(results_table(hall, ex_h, res["methods"]))
        st.caption(f"`REAL` Industrial Hall, n = {res['episodes']['hall']['n']} episodes, {hall['stage1_pm']['background']['sensor_hours']:.0f} "
                   "clean-background sensor-hours; trained on the EN54 room only; settings frozen before this single run "
                   f"(run {res['hall']['meta']['hall_run_number']}, config hash {res['hall']['meta']['frozen_hash']}). Episode intervals are "
                   "exact binomial (Clopper-Pearson); latency and background intervals are bootstrap (episodes; recording days). "
                   "Methods: " + "; ".join(f"{TABLE_LABEL.get(k, k)} = {v}" for k, v in res["methods"].items()) + ".")
    with tab_c:
        show_results(results_table(cv, ex_c, res["methods"]))
        auc_c = cv["stage2_auroc"]
        st.caption(f"`REAL` EN54 test room, n = {res['episodes']['en54']['n']} episodes, {len(res['cv_en54'][head]['folds'])} recording days, each day held out in turn "
                   f"with thresholds and model fitted on the other days. Stage-2 AUROC {num(auc_c['auroc'], '{:.2f}')} "
                   f"(95% CI {num(auc_c['ci95'][0], '{:.2f}')}-{num(auc_c['ci95'][1], '{:.2f}')}, {auc_c['n_triggers']} triggers).")

    # chart: share of episodes alarmed as fire, by method, fire vs nuisance, both evaluations
    rows = []
    for ev_name, mets, exb in (("Held-out room", hall, ex_h), ("Same room (CV)", cv, ex_c)):
        for m in METHOD_ORDER:
            for kind in ("fire", "nuisance"):
                r = mets[m][kind]
                ci = ((exb or {}).get(m, {}).get(kind, {}) or {}).get("ci95") or r["ci95"]
                rows.append({"evaluation": ev_name, "method": SHORT[m], "episodes": kind,
                             "share called fire": r["rate"], "lo": ci[0], "hi": ci[1], "k/n": f"{r['alarmed']}/{r['n']}"})
    cdf = pd.DataFrame(rows)
    base = alt.Chart(cdf).encode(
        y=alt.Y("method:N", title=None, sort=[SHORT[m] for m in METHOD_ORDER]),
        color=alt.Color("episodes:N", scale=alt.Scale(domain=["fire", "nuisance"], range=[KIND_COLOR["fire"], KIND_COLOR["nuisance"]]),
                        title="Episode type"),
        yOffset="episodes:N")
    chart = (base.mark_rule(strokeWidth=2).encode(x=alt.X("lo:Q", title="Share of episodes called fire (95% CI)",
                                                          axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])), x2="hi:Q")
             + base.mark_point(filled=True, size=70).encode(x="share called fire:Q", tooltip=["evaluation", "method", "episodes", "k/n"])
             ).properties(width=220, height=150).facet(column=alt.Column("evaluation:N", title=None, sort=["Held-out room", "Same room (CV)"]))
    st.altair_chart(chart, width="content")
    st.caption("Wanted: red dots at the right (fires caught), blue dots at the left (nuisances not called fire). Lines are 95% intervals.")

    rep = load_csv("replay_hall.csv")
    if rep is not None and len(rep):
        st.markdown("**Replay one held-out episode** `REAL` (one sensor node: the first to trigger)")
        eps = rep.drop_duplicates("ep")[["ep", "kind", "scenario"]]
        opts = [f"{r.ep}: {r.scenario} ({r.kind})" for r in eps.itertuples()]
        default = next((i for i, o in enumerate(opts) if "Candles (fire)" in o), 0)
        pick = st.selectbox("Episode", opts, index=default, key="fire_replay")
        g = rep[rep["ep"] == pick.split(":")[0]].copy()
        end_min = float(g["end_min"].iloc[0])
        pm = alt.Chart(g).mark_line(color="#5b6474").encode(x=alt.X("min_from_start:Q", title="minutes from labelled start"),
                                                             y=alt.Y("pm_total_delta:Q", title="particle rise (counts)"))
        trig = alt.Chart(g[g["stage1_trigger"]]).mark_point(color="#d35400", size=90, shape="triangle-up", filled=True).encode(
            x="min_from_start:Q", y="pm_total_delta:Q", tooltip=["min_from_start", "pm_total_delta"])
        spans = alt.Chart(pd.DataFrame({"x": [0.0, end_min], "what": ["label start", "label end"]})).mark_rule(strokeDash=[4, 3]).encode(
            x="x:Q", tooltip=["what"])
        pf = alt.Chart(g).mark_line(color="#0b7a75").encode(x=alt.X("min_from_start:Q", title="minutes from labelled start"),
                                                            y=alt.Y("p_fire:Q", title="AI P(fire)", scale=alt.Scale(domain=[0, 1])))
        thr = alt.Chart(pd.DataFrame({"y": [res["frozen_config"]["stage2_p_fire"]]})).mark_rule(color="#b00020", strokeDash=[2, 2]).encode(y="y:Q")
        st.altair_chart(alt.vconcat((pm + trig + spans).properties(height=160, width=600),
                                    (pf + thr + spans).properties(height=120, width=600)), width="content")
        st.caption("Orange triangles: the conventional trigger fired (an alarm in this test). Teal line: the AI's probability of "
                   "fire; above the red dashed line the trigger is labelled 'fire-like evidence', otherwise 'nuisance-like, FSD to "
                   "check'. Neither label silences anything.")
    if sens:
        shown = [m for m in ("stage1_pm", "two_stage") if m in sens.get("methods", {})]
        counts = "; ".join(
            f"{SHORT[m]} {sens['methods'][m]['fire']['alarmed_strict_window']}/{sens['methods'][m]['fire']['n']} fires and "
            f"{sens['methods'][m]['nuisance']['alarmed_strict_window']}/{sens['methods'][m]['nuisance']['n']} nuisances" for m in shown)
        early_only = {m: sum(sens["methods"][m][k]["alarmed_only_in_early_window"] for k in ("fire", "nuisance")) for m in shown}
        early_min = res["frozen_config"]["pre_start_s"] // 60
        if all(v == 0 for v in early_only.values()):
            early_txt = (f"the pre-registered {early_min}-minute early window changes no fire or nuisance count for these methods; "
                         "it only moves the latency numbers.")
        else:
            early_txt = (f"the pre-registered {early_min}-minute early window does change the counts: fire or nuisance episodes "
                         "alarmed only inside it: " + ", ".join(f"{SHORT[m]} {v}" for m, v in early_only.items()) + ".")
        st.caption(f"Post-hoc check (defined after the Hall run): with triggers counted only inside the labelled window, {counts}; "
                   + early_txt
                   + (f" Hall episodes follow each other closely ({sens['episode_gaps_min']['min']:.0f} to "
                      f"{sens['episode_gaps_min']['max']:.0f} min apart, median {sens['episode_gaps_min']['median']:.0f}), so an "
                      "'early' alarm may belong to the previous activity." if "episode_gaps_min" in sens else ""))

# ------------------------------------------------------------------------------------------ lessons
rules = fire.load_rules()
src_by_n = {s["n"]: s for s in rules["sources"]}
st.subheader("Lessons from the 1988 First Interstate Bank fire, Los Angeles")
for row in rules["first_interstate"]:
    links = ", ".join(f"[{n}]({src_by_n[n]['url']})" for n in row["src"])
    st.markdown(f"- {row['text']} ({links})  \n  **Design rule here:** {row['design_rule']}.")

cols = st.columns(2)
with cols[0]:
    st.markdown("**Design scope: what the AI may do** (read-only; not all built)")
    for ph in rules["phases"]:
        st.markdown(f"- Phase {ph['n']} {ph['name']}: {ph['ai']}  \n  *Status:* {ph.get('built', 'not stated')}")
with cols[1]:
    st.markdown("**What the AI never does**")
    for x in rules["ai_never"]:
        st.markdown(f"- {x}")

# ------------------------------------------------------------------------------------------ 5. for engineers
with st.expander("For engineers: method, rules, full metrics, limits"):
    ft = fire.ft
    sa, sb = layer.stair("A"), layer.stair("B")
    fsae = sum(1 for e in layer.elevators if e.kind == "fire_service_access")
    st.markdown(
        "**Planner (Part A, deterministic, SYNTHETIC).** `cascade.building.fire.plan_for_incident(building, fire_layer, zone, "
        f"detections)`. Alert set = fire floor, {ft('FIRE_ALERT_ABOVE')} above and {ft('FIRE_ALERT_BELOW')} below (IBC 907.5.2.2), "
        "widened by trigger E1 for any other alarming floor with the floors in between [team-proposed, validate]; staging "
        f"{ft('FIRE_STAGING_BELOW')} floors below (moved to the first non-alerted floor below a widened alert set: "
        "[team-proposed, validate], tagged on the card); hose connection {ft('FIRE_EQUIP_BELOW')} floor below in the attack stair; "
        f"relocation = {ft('FIRE_RELOCATE_FLOORS')} floors from {ft('FIRE_RELOCATE_MIN_BELOW')} below the fire, skipping staging, "
        "alert, impaired floors and floor 1; evacuation stair = the stair farther from the fire zone; fire-department elevator exit "
        f"{ft('FIRE_LIFT_EXIT_BELOW')} below and only for fires at or above floor {ft('FIRE_LIFT_MIN_FIRE_FLOOR')}; electrical path "
        "from the building's feeds edges; firefighting-water migration = max-product reach on the existing water graph from the fire "
        f"floor's rooms and core (<= {ft('WATER_WATCH_MAX_HOPS')} hops, score >= REACH_MIN). The fire layer (stair enclosures "
        f"{sa.rect_m[2] - sa.rect_m[0]:g} x {sa.rect_m[3] - sa.rect_m[1]:g} m, {fire.stair_separation_m(sa, sb):.1f} m apart at their "
        f"nearest points; {fsae} fire service elevators; low and high passenger banks; FCC and a battery room on floor 1; design "
        f"load from plate area / {ft('OCC_LOAD_SQFT_PER_PERSON')} sq ft; {len(layer.assistance)} assistance entries; "
        f"{len(layer.impairments)} impairment(s)) is derived from the tower geometry and stamped SYNTHETIC.")
    th = pd.DataFrame([{"Key": k, "Value": str(v.value), "Unit": v.unit, "Tag": v.tag, "URL": v.url or "", "Note": v.note}
                       for k, v in fire.fire_thresholds().items()])
    st.dataframe(th, hide_index=True, width="stretch")
    if res:
        fc = res["frozen_config"]
        st.markdown(
            "**Verifier (Part B, REAL).** Stage 1 = PM_Total rise over its 30-min rolling median (shifted one sample) above the "
            f"q{fc['stage1_quantile'] * 100:g} of training clean background for {fc['k_consecutive']} consecutive 10 s samples of one "
            "node (a team-defined proxy, not a listed detector). Stage 2 = scikit-learn HistGradientBoosting, 3 classes, "
            f"max_iter {fc['hgb']['max_iter']}, balanced class weights; label 'fire-like' when P(fire) >= {fc['stage2_p_fire']} within "
            f"{fc['verify_window_s']} s of the trigger. Baselines: the stage-1 trigger alone, a CO-rise trigger, and the ML alone "
            f"without the gate. Latency counts from the label start, with triggers up to {fc['pre_start_s'] // 60} min early; clean "
            f"background = at least {fc['bg_before_s'] // 60} min before and {fc['bg_after_s'] // 60} min after any episode, after a "
            f"{fc['warmup_s'] // 60}-min warm-up. Headline features: '{fc['headline_features']}' "
            f"({res['hall'][fc['headline_features']]['n_features']} features: each sensor channel's value, rise over its rolling "
            f"median and {fc['slope_lag_samples']}-sample slope); pre-registered ablation: '{fc['ablation_features']}' "
            f"({res['hall'][fc['ablation_features']]['n_features']} features: rises and slopes only).")
        abl = res["hall"][fc["ablation_features"]]["metrics"]
        ex_a = (exact or {}).get("blocks", {}).get(f"hall/{fc['ablation_features']}")
        st.markdown(f"Held-out room, ablation '{fc['ablation_features']}' (stage-2 AUROC {num(abl['stage2_auroc']['auroc'], '{:.2f}')}):")
        show_results(results_table(abl, ex_a, res["methods"]))
        folds = pd.DataFrame(res["cv_en54"][fc["headline_features"]]["folds"])
        st.markdown("Within-site folds (thresholds fitted on the training days only):")
        st.dataframe(folds, hide_index=True, width="stretch")
        st.caption(f"Cross-site thresholds from all EN54 clean background: PM rise > {res['hall']['thr_pm']:.1f} counts, CO rise > "
                   f"{res['hall']['thr_co']:.2f}. Model files: {res['hall'][fc['headline_features']]['model_file']} "
                   f"({res['hall'][fc['headline_features']]['model_bytes'] / 1e6:.1f} MB).")
    st.markdown(
        "**Limits.** The tower and its fire layer are SYNTHETIC; no code compliance is claimed. Relocation distance, smoke-watch "
        "band, battery-room distance, the stair choice, the in-between fill and the staging move under trigger E1, and the "
        "water-graph weights are [team-proposed, validate]; the approved Emergency Plan of a "
        "real building overrides them. The staging and elevator rules come from a Sacramento regional guideline, not an LAFD SOP. "
        "Elevator recall is automatic only for lobby or hoistway detectors (LAFD matrix). The sensor data are German laboratory "
        "rooms, not a high-rise office; the small episode counts above give wide intervals; the Industrial Hall episodes are packed "
        "closely, so some first alarms fall inside the pre-registered early window. The stage-1 trigger is a proxy, not a listed "
        "smoke detector. Nothing here has been reviewed by LAFD or an authority having jurisdiction.")

# ------------------------------------------------------------------------------------------ 6. sources and licences
st.subheader("Sources & licences")
man = load_json("dataset_manifest.json")
if man:
    lic = man["licence"]
    for k, d in man["datasets"].items():
        st.markdown(f"- `REAL` {d['citation']}. [{d['page']}]({d['page']}). Licence [{lic['name']}]({lic['url']}). "
                    f"sha256 {d['sha256_measured'][:12]}... ({'matches' if d['sha256_ok'] else 'DOES NOT match'} the Mendeley record). "
                    f"{d['note']} Changes made: {lic['changes']}.")
with st.expander(f"All {len(rules['sources'])} sources (URLs and access dates)"):
    for s in sorted(rules["sources"], key=lambda s: s["n"]):
        st.markdown(f"[{s['n']}] {s['title']}. <{s['url']}> (accessed {s['accessed']})")
st.caption("Figures on this page are drawn by code (cascade.building.fire, SVG; Altair). Copies: docs/figures/fire_*.svg. "
           "Research note: docs/research/15_fire_deescalation.md.")
