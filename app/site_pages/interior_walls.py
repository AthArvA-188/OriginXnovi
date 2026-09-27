"""Interior wall testing page (Streamlit multipage site; no set_page_config here).

Every number shown is read from artifacts written by scripts/interior/*.py
(eval/interior/*.json|csv, models/interior/ndt_strength_v1.json) or computed live from the
readings the user types. Moisture is rule-graded (interior_water rubric), never ML.
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

from cascade.interior.ndt import StrengthModel  # noqa: E402
from cascade.interior.readings import (TEMPLATE_PATH, grade_moisture, pair_strength_locations, parse_csv,  # noqa: E402
                                       route)

MODEL_NAMES = {"training_mean": "Training mean (no readings)", "rn_only_powerlaw": "Rebound-only power law (fitted on SonReb rows)",
               "vp_only_powerlaw": "UPV-only power law (fitted on SonReb rows)", "sonreb_powerlaw": "SonReb power law (shipped)",
               "hgb_rn_vp": "Gradient boosting (challenger)"}


@st.cache_data
def _read_json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(rel: str):
    p = ROOT / rel
    return _read_json(str(p), p.stat().st_mtime) if p.exists() else None


@st.cache_data
def _read_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def f1(x) -> str:
    return "n/a" if x is None or x != x else f"{x:.1f}"


def pct(x) -> str:
    return "n/a" if x is None or x != x else f"{100 * x:.1f}%"


E = load("eval/interior/ndt_strength_v1.json")
SPEC = load("models/interior/ndt_strength_v1.json")
METH = load("eval/interior/methods_table.json")
SRC = load("eval/interior/sources.json")

st.title("Interior wall testing")
st.markdown("**Turn two quick concrete readings (rebound hammer and ultrasonic pulse) into a strength estimate with an honest range, "
            "and grade RH, leak-sensor and wood moisture readings by published EPA / USDA rules; other meter readings show U (cannot be graded) until a dry-reference rule exists.**")
st.caption("Estimates for an engineer to review, never a structural verdict. Calibrate with cores before any decision.")

with st.container(border=True):
    st.markdown("#### What this means")
    st.markdown(
        "- **Who it helps:** building engineers and facility managers checking interior concrete walls and columns, "
        "and anyone logging moisture readings after a leak.\n"
        "- **The problem:** drilling cores to test strength is slow, costly and damages the wall; hammer and ultrasonic readings are quick "
        "but only loosely related to strength, and the relation changes from one building to another.\n"
        "- **What the tool does:** it estimates strength from the readings using a model fitted on a large public database of real tests, "
        "shows a range instead of one number, and narrows the range when you enter a few core results from the same building.\n"
        "- **What the person does next:** if the range falls below the design strength, or straddles it, schedule cores and an engineer's review. "
        "Moisture: RH spot readings below 60 %RH and leak sensors are graded by EPA-based rules and wood moisture content by the USDA Forest Products Laboratory limit; pin or pinless readings on gypsum, concrete or plaster show U (cannot be graded) until a dry-reference rule exists. Nothing is automated.")

if not (E and SPEC):
    st.error("Artifacts missing: run scripts/interior/download_ndt.py then scripts/interior/ndt_strength.py.")
    st.stop()

SM = StrengthModel(SPEC)
cov = SPEC["coverage"]

# ---------------------------------------------------------------- interactive estimator --------
st.header("Estimate strength from field readings")
c1, c2, c3 = st.columns(3)
use_rn = c1.checkbox("I have a rebound number", value=True, key="iw_use_rn")
rn = c1.number_input("Rebound number (median of the impacts)", min_value=6.0, max_value=80.0, value=None, step=0.5,
                     key="iw_rn", disabled=not use_rn, placeholder="e.g. your hammer's median")
use_vp = c2.checkbox("I have a pulse velocity", value=True, key="iw_use_vp")
vp = c2.number_input("Pulse velocity Vp (m/s)", min_value=600.0, max_value=6500.0, value=None, step=10.0,
                     key="iw_vp", disabled=not use_vp, placeholder="from path length / transit time")
design = c3.number_input("Design strength f'c (MPa, optional)", min_value=0.0, max_value=150.0, value=None, step=1.0, key="iw_design")
st.markdown(f"**Optional: cores from the same building**, with the readings taken at each core location. Core strength in the model's convention: {SPEC['target']}.")
cores_df = st.data_editor(pd.DataFrame({"rebound number": pd.Series(dtype=float), "Vp (m/s)": pd.Series(dtype=float),
                                        "core strength (MPa)": pd.Series(dtype=float)}),
                          num_rows="dynamic", key="iw_cores", hide_index=True)
rn_v = float(rn) if (use_rn and rn is not None) else None
vp_v = float(vp) if (use_vp and vp is not None) else None
if rn_v is None and vp_v is None:
    st.info("Enter a rebound number, a pulse velocity, or both to get an estimate.")
else:
    cores = []
    for _, r in cores_df.iterrows():
        fc = r.get("core strength (MPa)")
        if fc is None or pd.isna(fc) or fc <= 0:
            continue
        cr = r.get("rebound number")
        cv = r.get("Vp (m/s)")
        cores.append((None if pd.isna(cr) else float(cr), None if pd.isna(cv) else float(cv), float(fc)))
    try:
        est = SM.estimate(rn=rn_v, vp=vp_v, cores=cores, design_fc_mpa=design)
    except ValueError as e:
        st.error(str(e))
        est = None
    if est:
        m = SPEC["models"][est.model]
        a, b, c = st.columns(3)
        a.metric("Estimated strength", f"{est.fc_mpa:.1f} MPa")
        b.metric(f"{pct(cov)} range", f"{est.lo_mpa:.1f} to {est.hi_mpa:.1f} MPa")
        c.metric("Cores used for calibration", est.k_cores)
        chart = pd.DataFrame([{"what": "estimate", "lo": est.lo_mpa, "hi": est.hi_mpa, "fc": est.fc_mpa}])
        base = alt.Chart(chart).encode(y=alt.Y("what:N", title=None))
        layers = [base.mark_rule(size=6, opacity=0.5).encode(x=alt.X("lo:Q", title="compressive strength (MPa)"), x2="hi:Q"),
                  base.mark_point(size=160, filled=True).encode(x="fc:Q")]
        if design:
            layers.append(alt.Chart(pd.DataFrame({"design": [design]})).mark_rule(strokeDash=[4, 3], color="#b3261e").encode(x="design:Q"))
        st.altair_chart(alt.layer(*layers).properties(height=90), width="stretch")
        if est.below_design is True:
            st.error("The whole range is below the design strength: schedule cores and an engineer's review. This is not a structural verdict.")
        elif design and est.action == "calibrate_with_cores":
            st.warning("The range straddles the design strength: the readings alone cannot tell. Calibrate with cores.")
        st.warning("Calibrate with cores before any decision.")
        st.caption(" ".join(est.notes))
        rng_notes = []
        if rn_v is not None and "rn_range" in m and not (m["rn_range"][0] <= rn_v <= m["rn_range"][1]):
            rng_notes.append(f"rebound number outside the fitted range {m['rn_range'][0]:.0f}-{m['rn_range'][1]:.0f}")
        if vp_v is not None and "vp_range_ms" in m and not (m["vp_range_ms"][0] <= vp_v <= m["vp_range_ms"][1]):
            rng_notes.append(f"velocity outside the fitted range {m['vp_range_ms'][0]:.0f}-{m['vp_range_ms'][1]:.0f} m/s")
        for n in rng_notes:
            st.caption("Extrapolating: " + n)
        st.caption(f"Model: {est.model} ({m['fitted_on']}). Range = split-conformal {pct(cov)} quantile of held-out-study errors"
                   + (f", tightened for {est.k_cores} cores." if est.k_cores else "."))
        fb_key = {"rn_only": "rn_only_on_rebound_db", "vp_only": "vp_only_on_upv_db"}.get(est.model)
        fbm = (E.get("fallbacks") or {}).get(fb_key) if fb_key else None
        if fbm:
            inst = "rebound-only" if est.model == "rn_only" else "UPV-only"
            ins_txt = ""
            if fbm.get("loso_powerlaw_in_situ"):
                ins_txt = (f"; on in-situ rows (real structures, n={fbm['n_in_situ_rows']} from {fbm['n_in_situ_studies']} studies) "
                           f"{f1(fbm['loso_powerlaw_in_situ']['mae'])} MPa vs {f1(fbm['loso_training_mean_in_situ']['mae'])} MPa for the training mean")
            st.info(f"Only one instrument was entered, so this is the {inst} fallback ({m['fitted_on']}). Its leave-one-study-out error: "
                    f"MAE {f1(fbm['loso_powerlaw']['mae'])} MPa on all {fbm['n_rows']} rows from {fbm['n_studies']} studies "
                    f"(training-mean baseline {f1(fbm['loso_training_mean']['mae'])} MPa){ins_txt}. "
                    "The results table below is for the two-instrument SonReb rows; add the second reading to use the shipped model.")

# ---------------------------------------------------------------- measured results -------------
st.header("How good is the estimate? (measured, REAL data)")
st.markdown("Labels: **REAL** data only (Matthews et al. database; BAM drilled cores). No synthetic rows. "
            "Headline split: **leave-one-study-out** - every study (a building or lab campaign) is predicted by a model that never saw it.")
L = E["loso"]
ins = L["in_situ"]["models"]
alln = L["all"]["models"]
a, b, c, d = st.columns(4)
a.metric("Error on real structures, no cores", f"{f1(ins['sonreb_powerlaw']['mae'])} MPa", help=f"MAE, in-situ rows n={L['in_situ']['n_rows']} from {L['in_situ']['n_studies']} studies")
b.metric("Guessing the average instead", f"{f1(ins['training_mean']['mae'])} MPa", help="MAE of the training-mean baseline on the same rows")
k3 = E["calibration"]["in_situ"]["by_k"]["3"]["pooled_mae"]
c.metric("With 3 same-study specimens", f"{f1(k3)} MPa",
         help=f"k=3 calibration, {E['calibration']['in_situ']['n_studies']} in-situ studies, "
              f"{E['calibration']['in_situ']['draws_per_study']} random draws each")
d.metric("External test (BAM cores)", f"{f1(E['bam_external']['global']['mae'])} MPa", help=f"n={E['bam_external']['n_cores']} cores; NDT measured on the cores in labs")

rows = []
for subset, lab in (("all", "all rows"), ("in_situ", "in-situ rows (real structures)")):
    for mk, mv in L[subset]["models"].items():
        rows.append({"rows": lab, "model": MODEL_NAMES.get(mk, mk), "data": "REAL", "MAE (MPa)": f1(mv["mae"]), "RMSE (MPa)": f1(mv["rmse"]),
                     "MAPE": f"{mv['mape_pct']:.1f}%", "R2": f"{mv['r2']:.3f}", "per-study MAE (macro)": f1(mv["macro_mae_by_study"]),
                     "n rows / studies": f"{L[subset]['n_rows']} / {L[subset]['n_studies']}"})
st.dataframe(pd.DataFrame(rows), hide_index=True)
bars = pd.DataFrame([{"rows": s, "model": MODEL_NAMES[mk], "MAE (MPa)": mv["mae"]} for s in ("all", "in_situ") for mk, mv in L[s]["models"].items()])
st.altair_chart(alt.Chart(bars).mark_bar().encode(x=alt.X("MAE (MPa):Q"), y=alt.Y("model:N", title=None, sort=list(MODEL_NAMES.values())),
                                                  color=alt.Color("model:N", legend=None), row=alt.Row("rows:N", title=None)).properties(width=420, height=140))
hgb_i, pl_i = ins["hgb_rn_vp"]["mae"], ins["sonreb_powerlaw"]["mae"]
st.markdown(f"- The shipped power law beats the gradient-boosting challenger on held-out studies ({f1(pl_i)} vs {f1(hgb_i)} MPa on real structures), "
            f"so the simpler model ships.\n"
            f"- Errors are large without cores: on real structures the typical miss is {f1(pl_i)} MPa ({ins['sonreb_powerlaw']['mape_pct']:.0f}% on average). "
            f"The largest in-situ study ({L['in_situ']['largest_study']['study']}) supplies {L['in_situ']['largest_study']['rows']} of {L['in_situ']['n_rows']} rows, "
            f"so the per-study average ({f1(ins['sonreb_powerlaw']['macro_mae_by_study'])} MPa) is shown too.")

pred_path = ROOT / "eval" / "interior" / "loso_predictions.csv"
if pred_path.exists():
    P = _read_csv(str(pred_path), pred_path.stat().st_mtime)
    P = P.assign(where=P.in_situ.map({True: "in-situ (real structure)", False: "laboratory"}))
    mx = float(max(P.fc.max(), P.pred_sonreb.max()))
    sc = alt.Chart(P).mark_circle(size=14, opacity=0.45).encode(
        x=alt.X("fc:Q", title="measured strength (MPa)", scale=alt.Scale(domain=[0, mx])),
        y=alt.Y("pred_sonreb:Q", title="predicted, study held out (MPa)", scale=alt.Scale(domain=[0, mx])),
        color=alt.Color("where:N", title=None), tooltip=["study", "fc", "pred_sonreb", "rn", "vp"])
    diag = alt.Chart(pd.DataFrame({"x": [0, mx], "y": [0, mx]})).mark_line(color="gray", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    st.altair_chart((sc + diag).properties(height=360), width="stretch")
    st.caption(f"Every point is predicted by a model that never saw its study (leave-one-study-out, n={len(P)}).")

cal = E["calibration"]
cal_rows = [{"rows": s, "cores (k)": int(k), "MAE (MPa)": v["pooled_mae"]} for s in ("all", "in_situ") for k, v in cal[s]["by_k"].items()]
st.subheader("Calibrating with a few cores from the same building")
st.altair_chart(alt.Chart(pd.DataFrame(cal_rows)).mark_line(point=True).encode(x=alt.X("cores (k):O"), y="MAE (MPa):Q", color=alt.Color("rows:N", title=None)).properties(height=220),
                width="stretch")
st.caption(f"{cal['in_situ']['note']}. Studies large enough to hold specimens out: {cal['all']['n_studies']} (all), {cal['in_situ']['n_studies']} (in-situ); "
           f"{cal['all']['draws_per_study']} random draws per study.")

conf = E["conformal"]
bam = E["bam_external"]
st.markdown(
    f"**Is the {pct(cov)} range honest?** Measured on held-out studies (the range was set on other studies only): "
    f"{pct(conf['k0']['pooled_row_coverage'])} of rows fell inside without cores (in-situ rows {pct(conf['k0']['in_situ_row_coverage'])}), "
    f"{pct(conf['k3']['pooled_row_coverage'])} after 3-core calibration. Typical width: ×/÷ {conf['k0']['median_interval_factor']:.2f} without cores, "
    f"×/÷ {conf['k3']['median_interval_factor']:.2f} with 3.\n\n"
    f"**External test, BAM drilled cores (REAL, n={bam['n_cores']}):** global model MAE {f1(bam['global']['mae'])} MPa, "
    f"mean signed error {f1(bam.get('mean_signed_error_mpa'))} MPa (a systematic under-estimate that calibration removes) "
    f"(training-mean baseline {f1(bam['training_mean_mae'])} MPa); with 3 cores for calibration {f1(bam['k3_calibrated']['mae'])} MPa on the other "
    f"{bam['k3_calibrated']['n_scored_per_draw']} ({bam['k3_calibrated']['draws']} draws). Rebound and UPV were measured in labs on the drilled cores, "
    f"not on the bridge; core strengths were converted with the database's own factor {bam['core_to_cylinder_factor']['factor']:.3f} "
    f"(n={bam['core_to_cylinder_factor']['n_rows']} core rows). Per lab-pair MAE ranged {f1(bam['per_lab_pair_mae_range'][0])}-{f1(bam['per_lab_pair_mae_range'][1])} MPa.")

# ---------------------------------------------------------------- research table ---------------
st.header("What interior wall testing captures")
if METH:
    mt = pd.DataFrame(METH["methods"])
    mt["sources"] = mt.sources.map(lambda ids: ", ".join(ids))
    st.dataframe(mt.rename(columns={"method": "method", "captures": "what it measures", "data": "data it produces",
                                    "public_labelled_data": "public labelled data we found", "cerebro": "how Cerebro uses it"}), hide_index=True)
    st.caption("Only rebound hammer and UPV have a large public dataset paired with measured strength, so they are the only ones with an ML model here. "
               "Moisture readings are meter-specific and are graded by rules, not ML; only RH, leak-sensor and wood readings have a rule row today.")

# ---------------------------------------------------------------- field readings --------------
st.header("Log field readings")
if TEMPLATE_PATH.exists():
    st.download_button("Download the field-readings CSV template", TEMPLATE_PATH.read_bytes(), file_name="field_readings_template.csv", key="iw_tpl")
up = st.file_uploader("Upload filled readings (CSV)", type=["csv"], key="iw_csv")
if up is not None:
    ok, errs = parse_csv(up.getvalue().decode("utf-8", errors="replace"))
    if errs:
        st.warning(f"{len(errs)} row(s) rejected")
        st.dataframe(pd.DataFrame(errs), hide_index=True)
    parts = route(ok)
    if parts["strength"]:
        out = []
        for loc in pair_strength_locations(parts["strength"]):
            try:
                e = SM.estimate(rn=loc["rn"], vp=loc["vp"], design_fc_mpa=loc["design_fc_mpa"])
                out.append({"wall": loc["wall_id"], "face": loc["face"], "readings": ", ".join(loc["ids"]), "model": e.model,
                            "estimate (MPa)": round(e.fc_mpa, 1), "range (MPa)": f"{e.lo_mpa:.1f}-{e.hi_mpa:.1f}", "action": e.action})
            except ValueError as ex:
                out.append({"wall": loc["wall_id"], "readings": ", ".join(loc["ids"]), "action": f"not estimated: {ex}"})
        st.markdown("**Strength estimates** (uncalibrated; add cores above to calibrate)")
        st.dataframe(pd.DataFrame(out), hide_index=True)
    if parts["moisture"]:
        g = [{"reading": r.reading_id, "method": r.method, "value": r.value, "scale": r.scale, **grade_moisture(r)} for r in parts["moisture"]]
        st.markdown("**Moisture readings, rule-graded** (interior_water rubric; U = cannot be graded, never 'dry')")
        st.dataframe(pd.DataFrame(g), hide_index=True)
    if parts["record_only"]:
        st.caption(f"{len(parts['record_only'])} reading(s) recorded without grading (e.g. IR, GPR, sounding, borescope).")

# ---------------------------------------------------------------- engineers ---------------------
with st.expander("For engineers: method, leakage check, fallbacks, limits"):
    dat = E["data"]
    law = SPEC["models"]["sonreb"]["law"]
    g = E["groupkfold_vs_random"]
    pub = E["published_comparison"]
    st.markdown(
        f"**Data.** SonReb database: {dat['sonreb_rows']} rows from {dat['sonreb_studies']} studies ({dat['sonreb_in_situ_rows']} in-situ rows from "
        f"{dat['sonreb_in_situ_studies']} studies); rebound-only {dat['rebound_rows']} rows / {dat['rebound_studies']} studies; UPV-only {dat['upv_rows']} rows / "
        f"{dat['upv_studies']} studies. Cleaning: {dat['cleaning']}. Target: {SPEC['target']}.\n\n"
        f"**Shipped model.** ln fc = {law['intercept']:.3f} + {law['coef'][0]:.3f} ln RN + {law['coef'][1]:.3f} ln Vp, least squares on all SonReb rows. "
        f"Calibration: add the mean log residual of the k cores. Interval: {SPEC['interval_note']}.\n\n"
        f"**Why not random splits?** The same gradient-boosting model scores R2 {g['random_rows_kfold5']['hgb_rn_vp']['r2']:.3f} with random 5-fold rows "
        f"but {g['groupkfold5_by_study']['hgb_rn_vp']['r2']:.3f} when whole studies are held out (5-fold GroupKFold). "
        f"{pub['claim']} using {pub['split']}; compare it with our random-row figure, not with the study-held-out one ([paper]({pub['url']})).")
    fb = E["fallbacks"]
    fb_rows = []
    for key, lab in (("rn_only_on_rebound_db", "rebound only (rebound DB)"), ("vp_only_on_upv_db", "UPV only (UPV DB)")):
        v = fb[key]
        fb_rows.append({"fallback (shipped when one instrument is entered)": lab, "data": "REAL",
                        "LOSO MAE, all rows": f1(v["loso_powerlaw"]["mae"]), "mean-baseline MAE, all rows": f1(v["loso_training_mean"]["mae"]),
                        "rows / studies": f"{v['n_rows']} / {v['n_studies']}",
                        "LOSO MAE, in-situ rows": f1((v.get("loso_powerlaw_in_situ") or {}).get("mae")),
                        "mean-baseline MAE, in-situ": f1((v.get("loso_training_mean_in_situ") or {}).get("mae")),
                        "in-situ rows / studies": f"{v.get('n_in_situ_rows')} / {v.get('n_in_situ_studies')}"})
    st.markdown("**Single-instrument fallbacks.** These are the laws a user gets with only one reading, each fitted on its own larger database. "
                "The rebound-only and UPV-only rows in the results table above are different fits (on SonReb rows) used as baselines.")
    st.dataframe(pd.DataFrame(fb_rows), hide_index=True)
    st.markdown("**Limits**\n" + "\n".join(f"- {x}" for x in SPEC["limits"]) +
                "\n- The database mixes laboratory cubes and in-situ tests from many countries; a new building can sit far from all of them."
                "\n- The calibration experiment uses labelled specimens from the same study, mostly lab cubes, as a stand-in for cores."
                "\n- No ML moisture model is shipped: meter readings are instrument-specific and no public labelled building set was found.")

# ---------------------------------------------------------------- sources -----------------------
st.divider()
st.markdown("#### Sources and licences")
if SRC:
    for k, s in SRC["sources"].items():
        st.markdown(f"- [{s['title']}]({s['url']}): {s['licence']}; accessed {s['accessed']}.")
st.caption("Estimates and rule grades for a person to review. The AI proposes; an engineer decides. It never actuates equipment.")
