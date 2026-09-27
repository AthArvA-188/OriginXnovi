"""Numeric AI page (Pavilion Cerebro local site): day-ahead meter forecasts with an 80% band, a measured backtest
against seasonal naive, anomaly-scorer results, and the Hugging Face model catalogue.

Every number on this page is read from files our scripts wrote (eval/numeric/*.json, *.parquet); none is typed here.
Top-level Streamlit script for st.navigation: no st.set_page_config.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "eval" / "numeric"
MODELS = ROOT / "models" / "numeric"
LOCAL_LEAD = ROOT / "data" / "raw" / "numeric" / "lead_timeline_local.parquet"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
MODEL_CHOICES = {"Chronos-2 (Hugging Face, zero-shot)": ("chronos_2", "forecasts_chronos_2.parquet"),
                 "Chronos-Bolt-small (Hugging Face, zero-shot)": ("chronos_bolt_small",
                                                                  "forecasts_chronos_bolt_small.parquet"),
                 "scikit-learn HGB + conformal band (runs on CPU)": ("hgb", "forecasts_baselines_hgb.parquet")}


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


def load_parquet(name: str):
    return _load_parquet_cached(name, _mtime(EVAL / name))


@st.cache_data(show_spinner=False)
def _load_parquet_cached(name: str, mtime: float):
    p = EVAL / name
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    for c in ("origin", "ts", "timestamp"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def load_selected():
    return _load_selected_cached(_mtime(EVAL / "bdg2_selected.parquet"))


@st.cache_data(show_spinner=False)
def _load_selected_cached(mtime: float):
    p = EVAL / "bdg2_selected.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df


SHORT = {"snaive168": "Seasonal naive: last week, same hour (baseline)",
         "naive24": "Naive: yesterday, same hour (baseline)",
         "hgb": "HGB + conformal band (scikit-learn)",
         "hgb_oct": "HGB refit to 31 Oct, no band (scikit-learn)",
         "chronos_bolt_small": "Chronos-Bolt-small (Hugging Face, zero-shot)",
         "chronos_2": "Chronos-2 (Hugging Face, zero-shot)"}


def pct(x, nd=1):
    return "n/a" if x is None else f"{100 * x:.{nd}f}%"


def num(x, nd=2):
    return "n/a" if x is None else f"{x:,.{nd}f}"


bt = load_json("backtest_bdg2.json")
lead = load_json("anomaly_lead.json")
cat = load_json("model_catalogue.json")
srcs = load_json("sources.json")
cpu = load_json("chronos_timing_cpu.json")
NOM = round(100 * (1 - bt["config"]["alpha"])) if bt else None  # nominal band level set in the backtest config
HORIZON = bt["horizon_h"] if bt else None
PRIMARY = ((bt or {}).get("paired_wape_diff") or {}).get("primary") or "chronos_2"
_m = (bt or {}).get("models", {})
PRIMARY_COV = _m.get(PRIMARY, {}).get("coverage_80")
BASE_WAPE = _m.get((bt or {}).get("baseline", ""), {}).get("WAPE_pct")
BASE_WAPE_TXT = "n/a" if BASE_WAPE is None else f"{BASE_WAPE:.1f}%"

# ---- (1) title and promise ------------------------------------------------------------------------------------------
st.title("Numeric AI: tomorrow's meter readings, with a likely range")
st.markdown("We tested Hugging Face time-series models on real office electricity meters. The chosen model predicts "
            f"the next {HORIZON} hours for each meter with a likely range, and a simple statistical check flags hours "
            "that look unusual so a person can check them.")

# ---- (2) what this means --------------------------------------------------------------------------------------------
st.info(
    "**What this means**\n\n"
    "- **Who it helps:** building engineers and facility managers who watch dozens of meters and sensors.\n"
    "- **The problem:** a reading is only 'too high' if you know what was expected. Taking last week's value at the "
    f"same hour (the usual rule of thumb) was off by {BASE_WAPE_TXT} of the actual use on average in our test.\n"
    "- **What the tool does:** for every meter it gives tomorrow's expected hourly use and a range meant to hold "
    f"the real value {NOM}% of the time. In our test the chosen model's range held it {pct(PRIMARY_COV)} of the "
    "time, so the range is a guide, not a guarantee (see Measured results). A separate simple check marks unusual "
    "hours.\n"
    "- **What the person does next:** look at hours outside the range or marked unusual, and decide whether to open "
    "a work order (for example: equipment left running, a faulty meter). Nothing here switches equipment or "
    "overrides a life-safety system; a person approves every action."
)
st.markdown(
    "**Data labels on this page:** `REAL` = measured public data (BDG2 electricity meters; LEAD human anomaly "
    "labels). `SYNTHETIC`, `SIMULATED` and `INJECTED` data are **not used** on this page."
)

if bt is None:
    st.warning("Backtest results not found (eval/numeric/backtest_bdg2.json). Run the scripts listed under "
               "'For engineers'.")
    st.stop()

models = bt["models"]
base = models[bt["baseline"]]
sel_info = bt["selection"]
cfg = bt["config"]

# ---- (3) main visual ------------------------------------------------------------------------------------------------
st.header("Which Hugging Face model we chose, and why")
chosen = {r["id"]: r for r in (cat or {}).get("models", [])}
c2 = chosen.get("amazon/chronos-2", {})
cb = chosen.get("amazon/chronos-bolt-small", {})
cols = st.columns(3)
with cols[0]:
    st.markdown(f"**Primary: `amazon/chronos-2`**  \nLicence: {c2.get('hf_licence_tag', 'n/a')}  \n"
                f"Size: {num(c2.get('params_millions'))} M parameters, {num(c2.get('largest_weight_file_MB'), 1)} MB "
                f"weights  \nRevision: `{str(c2.get('revision', ''))[:10]}`")
with cols[1]:
    st.markdown(f"**Second opinion: `amazon/chronos-bolt-small`**  \nLicence: {cb.get('hf_licence_tag', 'n/a')}  \n"
                f"Size: {num(cb.get('params_millions'))} M parameters, {num(cb.get('largest_weight_file_MB'), 1)} MB "
                f"weights  \nRevision: `{str(cb.get('revision', ''))[:10]}`")
with cols[2]:
    st.markdown("**Live fallback: scikit-learn HGB**  \nLicence: BSD-3-Clause  \nRuns on the app's CPU from a "
                "stored model; no GPU, no new install")
st.caption(c2.get("reason", "") + " The Hugging Face models run offline in a separate environment; this page reads "
           "their saved forecasts. Full list of candidates: model catalogue below.")

st.subheader("Forecast for one meter and one day")
fc_hgb = load_parquet("forecasts_baselines_hgb.parquet")
sel = load_selected()
meters = sel_info["chosen"]
c_a, c_b, c_c = st.columns([2, 2, 3])
meter = c_a.selectbox("Meter (REAL, BDG2)", meters, index=0)
origins = sorted(fc_hgb.origin.unique()) if fc_hgb is not None else []
origin = c_b.select_slider("Forecast day", options=[pd.Timestamp(o).date() for o in origins],
                           value=pd.Timestamp(origins[len(origins) // 2]).date() if origins else None)
avail = {k: v for k, v in MODEL_CHOICES.items() if (EVAL / v[1]).exists()}
model_label = c_c.radio("Model", list(avail), index=0)
tag, fname = avail[model_label]
o_ts = pd.Timestamp(origin)

if fc_hgb is not None and sel is not None:
    fdf = load_parquet(fname)
    if tag == "hgb":
        f = fdf[(fdf.meter == meter) & (fdf.origin == o_ts)][["ts", "hgb_q10", "hgb_q50", "hgb_q90"]]
        f.columns = ["ts", "q10", "q50", "q90"]
    else:
        f = fdf[(fdf.meter == meter) & (fdf.origin == o_ts)][["ts", "q10", "q50", "q90"]]
    sn = fc_hgb[(fc_hgb.meter == meter) & (fc_hgb.origin == o_ts)][["ts", "snaive168_q50"]]
    hist = sel[meter].loc[o_ts - pd.Timedelta(days=3): o_ts + pd.Timedelta(hours=23)]
    keys = {"actual": "Actual (REAL)", "forecast": "Model forecast (median)", "baseline": "Last week, same hour (baseline)"}
    lines = pd.concat([
        pd.DataFrame({"ts": hist.index, "kWh": hist.to_numpy(), "key": "actual"}),
        pd.DataFrame({"ts": f.ts, "kWh": f.q50.to_numpy(), "key": "forecast"}),
        pd.DataFrame({"ts": sn.ts, "kWh": sn.snaive168_q50.to_numpy(), "key": "baseline"}),
    ], ignore_index=True)
    lines["series"] = lines.key.map(keys)
    dom = list(keys.values())
    color = alt.Color("series:N", scale=alt.Scale(domain=dom, range=[BLUE, ORANGE, GRAY]),
                      legend=alt.Legend(title=None, orient="top", labelLimit=320))
    x = alt.X("ts:T", title=None)
    band = alt.Chart(f).mark_area(opacity=0.22, color=ORANGE).encode(
        x=x, y=alt.Y("q10:Q", title="kWh per hour"), y2="q90:Q",
        tooltip=[alt.Tooltip("ts:T", title="hour"), alt.Tooltip("q10:Q", format=",.1f", title="band low"),
                 alt.Tooltip("q90:Q", format=",.1f", title="band high")])
    line = alt.Chart(lines).mark_line(strokeWidth=2).encode(
        x=x, y="kWh:Q", color=color,
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=dom, range=[[1, 0], [1, 0], [5, 4]]),
                                  legend=None))
    hover = alt.selection_point(on="pointerover", nearest=True, fields=["ts"], empty=False)
    wide = lines.pivot_table(index="ts", columns="key", values="kWh").reset_index()
    wide.columns = [str(c) for c in wide.columns]
    tips = [alt.Tooltip("ts:T", title="hour")] + [alt.Tooltip(f"{k}:Q", format=",.1f", title=v)
                                                  for k, v in keys.items() if k in wide]
    rule = alt.Chart(wide).mark_rule(color=GRAY).encode(x="ts:T", opacity=alt.condition(hover, alt.value(0.6),
                                                                                         alt.value(0)),
                                                        tooltip=tips).add_params(hover)
    start = alt.Chart(pd.DataFrame({"ts": [o_ts]})).mark_rule(strokeDash=[2, 2], color=GRAY).encode(x="ts:T")
    st.altair_chart((band + line + rule + start).properties(height=320), width="stretch")
    st.caption(f"Shaded: the model's {NOM}% range. Dotted line: forecast start ({origin}, 00:00). The model saw only the "
               f"hours before the dotted line. Data: REAL BDG2 meter `{meter}`.")

    if tag == "hgb" and (MODELS / "hgb_bdg2.joblib").exists():
        if st.button("Re-run the HGB model live for this meter and day"):
            from cascade.numeric import NumericAgent

            agent = NumericAgent()
            ctx = sel[meter].loc[: o_ts - pd.Timedelta(hours=1)]
            live = agent.forecast(ctx, 24, backend="hgb", series_id=meter)
            diff = float((pd.Series(live.q50, index=live.index) - f.set_index("ts").q50).abs().max())
            st.success(f"Live run done on this computer's CPU. Largest difference from the stored forecast: "
                       f"{diff:.6f} kWh. Model: {live.provenance.model_id}, licence {live.provenance.licence}.")

# ---- (4) measured results -------------------------------------------------------------------------------------------
st.header("Measured results")
st.markdown(
    f"`REAL` Held-out backtest on **{bt['n_meters']} BDG2 office electricity meters**, **{bt['n_origins']} daily "
    f"forecasts** each ({cfg['test_start']} to {cfg['last_origin']}), {bt['horizon_h']} hours ahead: "
    f"**n = {bt['n_scored_hours']:,} scored hours**. No model was trained on these weeks. Every model is compared "
    f"with the baseline *last week's value at the same hour* (seasonal naive)."
)
order = ["snaive168", "naive24", "hgb", "hgb_oct", "chronos_bolt_small", "chronos_2"]
rows = []
for m in [m for m in order if m in models] + [m for m in models if m not in order]:
    d = models[m]
    ci = d.get("WAPE_pct_ci95") or [None, None]
    rows.append({"Model": SHORT.get(m, m), "Type": d.get("kind", ""),
                 "Error, WAPE %": num(d["WAPE_pct"]), "WAPE 95% CI (by meter)": f"{num(ci[0])} to {num(ci[1])}",
                 "MASE": num(d["MASE_mean_over_meters"], 3), "CV(RMSE) %": num(d["CVRMSE_pct"]),
                 "NMBE %": num(d["NMBE_pct"]),
                 f"{NOM}% range holds the actual": pct(d.get("coverage_80")),
                 "Meters better than baseline": (f"{d['meters_better_than_baseline']} / {d['meters_total']}"
                                                 if "meters_better_than_baseline" in d else "baseline")})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

def short(m: str) -> str:
    return SHORT.get(m, m)


best = min((m for m in models if m != bt["baseline"]), key=lambda m: models[m]["WAPE_pct"])
best_cv = min(models, key=lambda m: models[m]["CVRMSE_pct"])
covs = {m: d["coverage_80"] for m, d in models.items() if d.get("coverage_80") is not None}
under = [m for m, v in covs.items() if v < NOM / 100]
pairs = (bt.get("paired_wape_diff") or {}).get("pairs", [])
pair = {(p["a"], p["b"]): p for p in pairs}
pb = pair.get((best, bt["baseline"]))
bullets = [f"- **Lowest error:** {short(best)}, WAPE {num(models[best]['WAPE_pct'])}% against "
           f"{num(base['WAPE_pct'])}% for the baseline (lower is better)"
           + (f"; paired difference {num(pb['diff_pp'])} points, 95% interval {num(pb['ci95_pp'][0])} to "
              f"{num(pb['ci95_pp'][1])} (the same meters resampled for both)." if pb else ".")]
if under:
    listing = ", ".join(f"{short(m)} {pct(covs[m])}" for m in under)
    bullets.append(f"- **Where the models fall short:** these {NOM}% ranges held the actual less often than {NOM}% of the "
                   f"time ({listing}), so they are somewhat too narrow. Treat them as a guide, not a guarantee.")
if best_cv != best:
    bullets.append(f"- **The ranking depends on the metric:** on CV(RMSE), which punishes big misses more, "
                   f"{short(best_cv)} is best ({num(models[best_cv]['CVRMSE_pct'])}% against "
                   f"{num(models[best]['CVRMSE_pct'])}% for {short(best)}).")
st.markdown("\n".join(bullets))

wdf = pd.DataFrame([{"model": short(m), "WAPE": models[m]["WAPE_pct"],
                     "lo": models[m]["WAPE_pct_ci95"][0], "hi": models[m]["WAPE_pct_ci95"][1],
                     "group": "baseline" if models[m].get("kind") == "baseline" else "model"} for m in models])
ysort = alt.Y("model:N", sort=wdf.sort_values("WAPE").model.tolist(), title=None, axis=alt.Axis(labelLimit=320))
bars = alt.Chart(wdf).mark_bar(cornerRadiusEnd=4).encode(
    x=alt.X("WAPE:Q", title="WAPE % (lower is better), with 95% interval over meters"), y=ysort,
    color=alt.Color("group:N", scale=alt.Scale(domain=["model", "baseline"], range=[BLUE, GRAY]),
                    legend=alt.Legend(title=None, orient="top", labelLimit=320)),
    tooltip=["model", alt.Tooltip("WAPE:Q", format=".2f"), alt.Tooltip("lo:Q", format=".2f", title="95% low"),
             alt.Tooltip("hi:Q", format=".2f", title="95% high")])
err = alt.Chart(wdf).mark_rule(strokeWidth=2, color=GRAY).encode(x="lo:Q", x2="hi:Q", y=ysort)
st.altair_chart((bars + err).properties(height=260), width="stretch")
st.caption("Each bar's interval resamples meters for that model alone, so overlapping bars do not mean two models are "
           "equal. The paired table below resamples the same meters for both models, which is the right comparison.")
if pairs:
    prow = [{"Comparison (A minus B)": f"{short(p['a'])} minus {short(p['b'])}",
             "WAPE difference, points": num(p["diff_pp"]),
             "95% interval (paired, by meter)": f"{num(p['ci95_pp'][0])} to {num(p['ci95_pp'][1])}",
             "Interval excludes 0": "yes" if p["ci95_pp"][1] < 0 or p["ci95_pp"][0] > 0 else "no"} for p in pairs]
    st.markdown(f"**Paired WAPE differences** (`REAL`, n = {bt['n_scored_hours']:,} hours, {bt['n_meters']} meters). "
                "Negative = A has the lower error.")
    st.dataframe(pd.DataFrame(prow), hide_index=True, width="stretch")

pm = bt["per_meter_mase"]
show = [m for m in ("snaive168", "hgb", "chronos_2") if m in pm]
pmd = pd.DataFrame([{"meter": b, "model": short(m), "MASE": v} for m in show for b, v in pm[m].items()])
dom2 = [short(m) for m in show]
meter_order = sorted(pm[show[0]], key=lambda b: -pm[show[0]][b])
dots = alt.Chart(pmd).mark_point(filled=True, size=70).encode(
    x=alt.X("MASE:Q", title="MASE per meter (below 1 = better than last week's value on the training data)"),
    y=alt.Y("meter:N", title=None, sort=meter_order, axis=alt.Axis(labelLimit=220)),
    color=alt.Color("model:N", scale=alt.Scale(domain=dom2, range=[GRAY, ORANGE, BLUE][:len(dom2)]),
                    legend=alt.Legend(title=None, orient="top", labelLimit=320)),
    tooltip=["meter", "model", alt.Tooltip("MASE:Q", format=".3f")])
one = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(strokeDash=[4, 3], color=GRAY).encode(x="x:Q")
st.altair_chart((dots + one).properties(height=26 * len(meter_order)), width="stretch")
st.caption("One row per meter, sorted by the baseline's error. Grey: last week's value; the other colours: the "
           "compared models. Dashed line: MASE 1.")

st.subheader("Anomaly scoring")
if lead:
    st.markdown(
        f"`REAL` (human anomaly labels) LEAD1.0-small, **{lead['split']['n_test_buildings']} held-out buildings** "
        f"(split by building, seed {lead['split']['seed']}), **n = {lead['n_rows_test']:,} hourly readings**, of which "
        f"{lead['n_anomalies_test']:,} ({pct(lead['prevalence_test'], 2)}) are labelled anomalous. A random scorer's "
        f"PR-AUC equals that share. Aggregate numbers only: LEAD states no licence, so its labelled series are not "
        f"shown here.")
    names = {"robust_z": "Robust z-score (baseline, no training)", "isolation_forest": "IsolationForest (unsupervised)",
             "hgb_supervised": f"HGB classifier (trained on {lead['split']['n_train_buildings']} other buildings' labels)"}
    arows = []
    for m, d in lead["models"].items():
        ci = d["pr_auc_ci95_building_bootstrap"]
        arows.append({"Scorer": names.get(m, m), "ROC-AUC": num(d["roc_auc"], 3), "PR-AUC": num(d["pr_auc"], 3),
                      "PR-AUC 95% CI (by building)": f"{num(ci[0], 3)} to {num(ci[1], 3)}",
                      "Chance PR-AUC": num(lead["chance_pr_auc"], 3),
                      f"Precision in top {pct(d['at_top_1pct']['budget_share'], 0)} alerts":
                          pct(d["at_top_1pct"]["precision"]),
                      f"Recall in top {pct(d['at_top_1pct']['budget_share'], 0)} alerts":
                          pct(d["at_top_1pct"]["recall"])})
    st.dataframe(pd.DataFrame(arows), hide_index=True, width="stretch")
    rz = lead["models"]["robust_z"]
    iso = lead["models"]["isolation_forest"]
    thr = rz.get("at_threshold", {})
    st.markdown(
        f"- **Labels matter most:** the supervised HGB reached PR-AUC {num(lead['models']['hgb_supervised']['pr_auc'], 3)}"
        f" against {num(rz['pr_auc'], 3)} for the robust z baseline.\n"
        f"- **Where a model loses:** IsolationForest scored PR-AUC {num(iso['pr_auc'], 3)}, "
        f"{'below' if iso['pr_auc'] < rz['pr_auc'] else 'above'} the simple baseline.\n"
        f"- **At K = {num(thr.get('threshold'), 0)}** (the building layer's ANOMALY_MAD_K on the same 1.4826 x MAD "
        f"scale, but with a trailing baseline rather than the building layer's fixed first-28-day one), the robust z "
        f"flags {pct(thr.get('flag_share'))} of hours, with precision {pct(thr.get('precision'))} and recall "
        f"{pct(thr.get('recall'))}: most flags are not labelled anomalies, so flags go to a person for review.")

ind = load_parquet("bdg2_indicators.parquet")
if ind is not None:
    st.markdown(f"**Unusual-hour flags on the forecast meter** (`REAL` series, robust z >= "
                f"{num(bt['indicators']['threshold'], 0)} on the 1.4826 x MAD scale). There are no labels for these "
                f"meters, so these are indicators for review, not measured detections.")
    im = ind[ind.meter == meter]
    base_line = alt.Chart(im).mark_line(strokeWidth=1.5, color=BLUE).encode(
        x=alt.X("ts:T", title=None), y=alt.Y("y:Q", title="kWh per hour"),
        tooltip=[alt.Tooltip("ts:T", title="hour"), alt.Tooltip("y:Q", format=",.1f", title="kWh")])
    flags = alt.Chart(im[im.flag]).mark_point(filled=True, size=64, color=ORANGE).encode(
        x="ts:T", y="y:Q", tooltip=[alt.Tooltip("ts:T", title="hour"), alt.Tooltip("y:Q", format=",.1f"),
                                    alt.Tooltip("robust_z:Q", format=".1f", title="robust z")])
    st.altair_chart((base_line + flags).properties(height=220), width="stretch")
    st.caption(f"Blue: hourly reading. Orange: flagged hour ({int(im.flag.sum())} on this meter; "
               f"{bt['indicators']['n_flagged']:,} of {bt['indicators']['n_hours']:,} hours across all "
               f"{bt['n_meters']} meters).")

if os.environ.get("CEREBRO_LOCAL_LEAD") == "1" and LOCAL_LEAD.exists():
    with st.expander("LOCAL ONLY: one LEAD building with human labels (not for publication)"):
        tl = pd.read_parquet(LOCAL_LEAD)
        st.altair_chart(
            alt.Chart(tl).mark_line(strokeWidth=1, color=BLUE).encode(x="timestamp:T", y="y:Q")
            + alt.Chart(tl[tl.anomaly_label == 1]).mark_point(color=ORANGE, filled=True).encode(x="timestamp:T",
                                                                                                y="y:Q"),
            width="stretch")
else:
    st.caption("A LEAD building with its human labels can be viewed locally only (set CEREBRO_LOCAL_LEAD=1); it is "
               "hidden because LEAD states no licence.")

st.subheader("Model catalogue (from our Hugging Face research)")
if cat:
    crows = [{"Model / tool": r["id"], "Licence": r["licence"], "Parameters (M)": r.get("params_millions"),
              "Largest weight file (MB)": r.get("largest_weight_file_MB"),
              "Measured here (only this task)": r.get("measured_scope") or ("yes" if r["measured_here"] else "no"),
              "Possible uses (not measured unless stated)": r["fit"],
              "Decision": r["status"], "Why": r["reason"]}
             for r in cat["models"]]
    st.dataframe(pd.DataFrame(crows), hide_index=True, width="stretch")
    st.caption("Parameters and file sizes are read from the Hugging Face API (decimal MB). Only the task named under "
               "'Measured here' has results on this page; the 'possible uses' were not measured. Unmeasured rows were "
               "assessed from their cards, licences and install requirements.")

st.warning("Advisory only. Forecasts and flags are indicators for a person to review. The software never switches "
           "equipment, never overrides a life-safety system, and never sets an inspection grade on its own.")

# ---- (5) for engineers ----------------------------------------------------------------------------------------------
with st.expander("For engineers: method, full metrics, limits"):
    st.markdown(
        f"**Test pool.** {sel_info['n_office_meters']} BDG2 office meters; {sel_info['n_eligible_before_exclusion']} "
        f"pass the pre-test data rules (missing <= {pct(sel_info['rule']['max_missing_pre_test'], 0)}, median >= "
        f"{sel_info['rule']['min_median_kwh_pre_test']} kWh, zero share <= "
        f"{pct(sel_info['rule']['max_zero_share'], 0)}, flat days <= {pct(sel_info['rule']['max_flat_day_share'], 0)});"
        f" {sel_info['n_eligible_after_exclusion']} remain after excluding sites "
        f"{', '.join(sel_info['excluded_sites'])} ({sel_info['exclusion_reason']}); "
        f"{len(sel_info['chosen'])} drawn with seed {sel_info['rule']['seed']}.\n\n"
        f"**Protocol.** Daily origins at 00:00, {bt['horizon_h']} h ahead. Imputation: {bt['imputation']}. "
        f"{bt['n_dropped_hours']} target hours had no reading and were not scored. HGB: lags "
        f"{bt['hgb_params']['lags_h']} h + hour + day of week, max_iter {bt['hgb_params']['max_iter']}; trained before "
        f"{cfg['hgb_train_end']}, band = split-conformal residual quantiles on {cfg['hgb_train_end']} to "
        f"{cfg['hgb_calib_end']}. Chronos: zero-shot, univariate, context {cfg['context']} h, quantiles "
        f"{'/'.join(str(q) for q in next(iter(bt.get('hf_provenance', {}).values()), {}).get('quantiles', []))} "
        f"from the same call. MASE scale: in-sample mean |y(t) - y(t-{cfg['mase_period']})| before {cfg['test_start']}. "
        f"NMBE sign: positive = over-forecast.")
    full = pd.DataFrame({m: {k: v for k, v in d.items() if not isinstance(v, (list, dict))} for m, d in models.items()}).T
    st.dataframe(full, width="stretch")
    psite = bt.get("per_site_WAPE_pct")
    site_losses = []
    if psite:
        mps = bt.get("meters_per_site", {})
        sdf = pd.DataFrame(psite).rename(columns=SHORT)
        sdf.index = [f"{k} ({mps.get(k, '?')} meters)" for k in sdf.index]
        st.markdown(f"**WAPE % by site** (`REAL`; each site's hours pooled). All {bt['n_meters']} meters come from "
                    f"these {len(psite[PRIMARY])} sites, so the meter-level intervals above do not account for site "
                    "clustering.")
        st.dataframe(sdf.round(2), width="stretch")
        others = [m for m in psite if m not in (bt["baseline"], "naive24")]
        for site_name in psite[PRIMARY]:
            best_here = min(others, key=lambda m: psite[m][site_name])
            if best_here != PRIMARY:
                site_losses.append(f"{site_name}: {short(best_here)} {num(psite[best_here][site_name])}% vs "
                                   f"{short(PRIMARY)} {num(psite[PRIMARY][site_name])}%")
        if site_losses:
            st.markdown(f"- **Sites where {short(PRIMARY)} is not the lowest-error model:** " + "; ".join(site_losses))
    diag = bt.get("selection_diagnostics")
    if diag:
        rem = diag["removed_by_flat_rule"]
        kept_pool = [k for k, v in rem.items() if not v["in_excluded_site"]]
        alt = diag.get("alternative_limits", {})
        alt_txt = "; ".join(f"A {pct(float(lim), 0)} limit would leave {a['after_exclusion']} and also remove "
                            f"{len(a['also_removed_after_exclusion'])} meters "
                            f"({', '.join(sorted({v['site'] for v in a['also_removed_after_exclusion'].values()}))})"
                            for lim, a in alt.items())
        st.markdown(
            f"**Flat-day rule (added before any metric was computed).** Without it, the pool is "
            f"{diag['pool_without_flat_rule']['before_exclusion']} eligible and "
            f"{diag['pool_without_flat_rule']['after_exclusion']} after the site exclusion; with it, "
            f"{diag['pool_at_chosen_limit']['before_exclusion']} and {diag['pool_at_chosen_limit']['after_exclusion']}. "
            f"It removes {len(kept_pool)} meters from the non-excluded pool: "
            + ", ".join(f"{k} ({pct(rem[k]['flat_day_share'], 0)} flat days)" for k in kept_pool)
            + f". {alt_txt}. Source: backtest_setup.json selection_diagnostics.")
    if lead and lead["models"]["robust_z"].get("at_threshold_raw_mads"):
        raw = lead["models"]["robust_z"]["at_threshold_raw_mads"]
        st.markdown(f"**Robust z threshold sensitivity (LEAD).** At K = {num(raw['threshold'], 0)} raw MADs (not the "
                    f"building layer's scale; an earlier version of this page used it) the robust z flags "
                    f"{pct(raw['flag_share'])} of hours, precision {pct(raw['precision'])}, recall "
                    f"{pct(raw['recall'])}. Threshold source: {lead['models']['robust_z']['at_threshold']['source']}")
    prov_rows = []
    for tagk, p in bt.get("hf_provenance", {}).items():
        prov_rows.append({"model": p["model_id"], "revision": p["revision"], "resolved snapshot": p["resolved_snapshot"],
                          "device": p["device"], "forecasts": p["n_forecasts"], "predict seconds": p["predict_seconds"],
                          "torch": p["library_versions"]["torch"], "chronos": p["library_versions"]["chronos"],
                          "created": p["created_utc"]})
    if cpu:
        for mid, t in cpu.items():
            if mid.startswith("_"):
                continue
            prov_rows.append({"model": mid, "revision": t["revision"], "resolved snapshot": "(CPU timing run)",
                              "device": f"cpu, {t['torch_threads']} threads", "forecasts": t["n_forecasts"],
                              "predict seconds": t["predict_seconds"], "torch": "", "chronos": "",
                              "created": f"max |q50 CPU - GPU| = {t['max_abs_q50_diff_vs_stored_kWh']:.2e} kWh"})
    st.markdown("**Provenance of the Hugging Face forecasts** (timings on a shared laptop with other jobs running):")
    st.dataframe(pd.DataFrame(prov_rows), hide_index=True, width="stretch")
    for m in ("chronos_2", "chronos_bolt_small"):
        if m in models and models[m].get("pretraining_overlap"):
            st.markdown(f"- **Pretraining overlap, {m}:** {models[m]['pretraining_overlap']}")
    st.markdown(
        "**Limits.**\n"
        f"- {bt['n_meters']} office meters from {len(set(sel_info['site_of'].values()))} sites, one winter window; "
        "other building types and seasons are not measured.\n"
        "- The 95% intervals resample meters, so they show meter-to-meter spread, not hour-level noise. The meters "
        f"come from {len(set(sel_info['site_of'].values()))} sites and are resampled independently, so the intervals "
        "ignore site clustering and are likely too narrow for new sites.\n"
        f"- The {NOM}% ranges under-cover (see table); a recalibration step would be needed before relying on them.\n"
        "- Anomaly results use LEAD human labels on other buildings; the flags on BDG2 meters have no labels.\n"
        "- No weather or occupancy inputs are used; Chronos-2 can take them but that was not measured.\n\n"
        "**Rerun.** `scripts/numeric_backtest.py`, then `scripts/numeric_chronos_precompute.py` (cerebro_ml env), "
        "then `scripts/numeric_evaluate.py`, `scripts/numeric_anomaly_lead.py`, `scripts/numeric_catalogue.py`. "
        "Details: docs/research/11_numeric_models.md."
    )

# ---- (6) sources and licences ---------------------------------------------------------------------------------------
st.divider()
st.subheader("Sources & licences")
lic = [f"- **{d['name']}** ({d['label']}): {d['licence']} [{d['url']}]({d['url']})"
       + (f". Cite: {d['citation']}" if d.get("citation") else "") for d in (cat or {}).get("datasets", [])]
span = f"{sel.index.min():%b %Y} to {sel.index.max():%b %Y}" if sel is not None else "see file"
st.markdown("**Data**\n" + "\n".join(lic) + f"\n- The derived file `eval/numeric/bdg2_selected.parquet` "
            f"({bt['n_meters']} BDG2 meters, {span}) is shared under the same CC BY-SA terms.")
st.markdown("**Models:** amazon/chronos-2 and amazon/chronos-bolt-small, Apache-2.0; scikit-learn, BSD-3-Clause.")
if srcs:
    with st.expander(f"All {len(srcs['sources'])} research sources (URLs and access dates)"):
        st.markdown("\n".join(f"{s['n']}. [{s['title']}]({s['url']}) (accessed {s['accessed']})"
                              + (f", moved from {s['moved_from']}" if s.get("moved_from") else "")
                              for s in srcs["sources"]))
