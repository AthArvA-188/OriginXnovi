"""Exterior facade screening page (Streamlit multipage site; no set_page_config here).

Every number on this page is read from artifacts written by scripts/facade/*.py
(eval/facade/*.json, models/facade/tilecls_v1.card.json) or computed live from the user's photo.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cascade.facade import heatmap as hm  # noqa: E402
from cascade.facade import review as rv  # noqa: E402

try:  # the console's record of which asset classes have a labelled set to score the grader against
    from cascade.evalmetrics import TRUTH_SOURCE  # noqa: E402
    GRADER_TRUTH = TRUTH_SOURCE.get(rv.ASSET_CLASS)
except Exception:  # pragma: no cover - the page must render even if the console module changes
    GRADER_TRUTH = None
GRADER_UNMEASURED = GRADER_TRUTH is None or str(GRADER_TRUTH).startswith("none")
GRADER_CAPTION = (("Grader accuracy on facade photos is unmeasured"
                   + (f" (cascade.evalmetrics reference labels: {GRADER_TRUTH})" if GRADER_TRUTH else "")
                   + "; treat the class as a suggestion for the inspector.")
                  if GRADER_UNMEASURED else f"Grader reference labels: {GRADER_TRUTH}. Treat the class as a suggestion for the inspector.")

MODEL_NAMES = {"constant_prior": "Constant guess (always 'no crack')", "heuristic_crack_mask": "Repo heuristic (dark-thin-line mask)",
               "resnet18_ozgenel_only": "ResNet-18, Ozgenel only (ablation)", "resnet18_ozgenel_sdnet": "ResNet-18, Ozgenel + SDNET (shipped)"}
SET_NAMES = {"sdnet_test": "SDNET2018 walls, held-out source photos (in-domain)", "bfdd_ood": "BFDD drone facade photos (out-of-domain)"}


@st.cache_data
def _read_json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(rel: str):
    p = ROOT / rel
    return _read_json(str(p), p.stat().st_mtime) if p.exists() else None


def pct(x) -> str:
    return "n/a" if x is None or x != x else f"{100 * x:.1f}%"


def f3(x) -> str:
    return "n/a" if x is None or x != x else f"{x:.3f}"


def ci(v) -> str:
    return "" if not v else f" [{v[0]:.3f}, {v[1]:.3f}]"


@st.cache_resource
def get_classifier():
    return hm.TileClassifier()


@st.cache_data(show_spinner="Scoring every window of the photo on CPU...")
def score_bytes(data: bytes, stride: int) -> hm.HeatmapResult:
    return hm.score_image(Image.open(io.BytesIO(data)), get_classifier(), stride=stride)


R = load("eval/facade/tilecls_v1.json")
OX = load("eval/facade/onnx_rescore.json")  # shipped ONNX re-scored on the same held-out rows
S = load("eval/facade/samples/samples.json")
SRC = load("eval/facade/sources.json")
CARD = hm.load_card()

st.title("Exterior facade screening")
st.markdown("**Shade a facade photo by how likely each patch shows a crack, so an inspector looks at the right spots first.**")
st.caption("AI screening, not a QEWI/FISP finding. Nothing here is filed or approved; a qualified person reviews every result.")

with st.container(border=True):
    st.markdown("#### What this means")
    st.markdown(
        "- **Who it helps:** facade inspectors and building managers who go through hundreds of photos from walk-downs, "
        "lifts or drones (NYC FISP, San Francisco's facade program), on concrete, masonry and render surfaces. "
        "Wood balcony members under California SB 721/326 are out of scope: the screen never saw wood.\n"
        "- **The problem:** most of any facade photo is intact wall; cracks are thin and easy to miss.\n"
        "- **What the screen does:** it cuts the photo into small squares, scores each square for 'crack or not', "
        "colours the photo, and lists the most likely squares.\n"
        "- **What the person does next:** open the listed squares first, optionally ask the AI grader for a FISP-style class "
        "(Safe / SWARMP / Unsafe; its accuracy on facades is not yet measured, so it is only a suggestion), measure width if the photo has a scale, then confirm or reject each one. "
        "FISP says photos and drones do not replace close-up inspection, so a confirmed crack means *schedule a close-up*.")

if R is None or not hm.DEFAULT_MODEL.exists() or not CARD:
    st.error("Model or evaluation artifacts are missing. Run scripts/facade/train_tilecls.py, export_onnx.py and eval_tilecls.py.")
    st.stop()

THR = float(CARD["threshold"])
WIN = int(CARD["input"]["window_px"])

# ---------------------------------------------------------------- main visual -----------------
st.header("Screen a photo")
src_mode = st.radio("Photo source", ["Sample facade photos (REAL drone images from the BFDD test set)", "Upload your own photo"],
                    horizontal=True, key="fx_src")
img = None
image_id = "upload"
sample = None
if src_mode.startswith("Sample") and S:
    labels = [f"Sample {i + 1}: {Path(s['file']).stem} ({s['percentile']}th percentile of labelled crack area)" for i, s in enumerate(S["samples"])]
    choice = st.selectbox("Sample photo", labels, index=0, key="fx_sample")
    sample = S["samples"][labels.index(choice)]
    data = (ROOT / sample["file"]).read_bytes()
    image_id = Path(sample["file"]).stem
    st.caption(f"Photo: {S['attribution']}. These frames are part of the out-of-domain test set, picked by a fixed rule ({S['rule']}).")
else:
    up = st.file_uploader("Facade photo (JPG or PNG)", type=["jpg", "jpeg", "png"], key="fx_upload")
    data = up.getvalue() if up is not None else None
    if up is not None:
        image_id = Path(up.name).stem[:40] or "upload"
    else:
        st.info("Upload a facade photo to screen it, or switch to the sample photos.")

c1, c2 = st.columns(2)
stride = c1.segmented_control("Window placement", options=[WIN, WIN // 2], default=WIN, key="fx_stride",
                              format_func=lambda s: "Side by side (faster)" if s == WIN else "Half-overlapping (finer map, slower)")
stride = stride or WIN
k = c2.slider("Squares to list for review", min_value=3, max_value=12, value=6, key="fx_k")

if data is not None:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    res = score_bytes(data, int(stride))
    n_avail = len(hm.review_candidates(res))
    top = hm.top_k(res, k=int(k))
    left, right = st.columns(2)
    left.image(hm.overlay(img, res, threshold=THR, highlight=top), caption="AI crack likelihood (darker red = more likely); numbered squares = review list", width="stretch")
    if sample:
        right.image(str(ROOT / sample["labelled"]), caption="Cracks as labelled by the BFDD authors (red), for comparison", width="stretch")
    else:
        right.image(img, caption="Original photo", width="stretch")
    st.image(hm.legend_strip(), caption="colour scale: p(crack) from 0 (no tint) to 1 (dark red)")
    above = sum(t.p_crack >= THR for t in res.tiles)
    st.markdown(f"**{res.n_windows}** squares of {WIN} px scored in **{res.seconds:.1f} s** on CPU. "
                f"**{above}** are at or above the operating threshold p = {THR:.3f} "
                f"(chosen on validation photos so that {pct(R['recall_target_on_val'])} of cracked squares are flagged).")
    for n in res.notes:
        st.caption(n)

    st.subheader("Squares to review first")
    st.caption(f"{n_avail} distinct squares available in this photo; listing {len(top)}. "
               + ("Side-by-side placement lists every window, including edge squares that overlap a neighbour."
                  if res.stride_px >= res.window_px else
                  f"Half-overlapping placement drops a square that overlaps a higher-scoring listed one by more than {hm.NMS_IOU:.0%} (IoU)."))
    win_lookup = {tuple(w["box"]): w for w in sample["windows"]} if sample else {}
    cols = st.columns(min(4, len(top)) or 1)
    for i, t in enumerate(top):
        with cols[i % len(cols)]:
            st.image(img.crop(t.box), width="stretch")
            lab = win_lookup.get(tuple(t.box))
            st.markdown(f"**#{t.rank}** p(crack) = {t.p_crack:.3f} " + ("(above threshold)" if t.p_crack >= THR else "(below threshold)")
                        + (f"  \nlabel: {lab['label']}" if lab else ""))

    # ---- optional grader ---------------------------------------------------------------------
    st.subheader("Optional: ask the AI grader for a FISP-style class")
    st.caption(GRADER_CAPTION)
    ok, msg = rv.grader_status()
    ranks = [t.rank for t in top]
    chosen_ranks = st.multiselect("Squares to send", ranks, default=ranks[:2], key="fx_sel")
    chosen = [t for t in top if t.rank in chosen_ranks]
    if not ok:
        st.warning(msg)
    confirm = st.checkbox("Send the selected crops to the Anthropic API (Claude). I understand the class it returns is an unmeasured "
                          "suggestion, not a FISP finding.", value=False, key="fx_grade_ok", disabled=not ok)
    if st.button("Send selected squares to the grader (facade_ll11 rubric)", disabled=not ok or not chosen or not confirm, key="fx_grade"):
        st.session_state["fx_graded"] = {"image_id": image_id, "results": rv.grade_tiles(img, chosen, image_id=image_id)}
    graded = st.session_state.get("fx_graded")
    graded_by_rank = {}
    if graded and graded["image_id"] == image_id:
        rows = []
        for g in graded["results"]:
            f = g.get("finding") or {}
            graded_by_rank[g["rank"]] = g
            rows.append({"square": g["rank"], "p(crack)": round(g["p_crack"], 3), "FISP class": (f.get("native_scale") or {}).get("value"),
                         "level": (f.get("unified") or {}).get("level"), "grader note": (f.get("justification") or g.get("error") or "")[:200]})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(GRADER_CAPTION)

    # ---- crack width -------------------------------------------------------------------------
    st.subheader("Crack width (only when the photo has a scale)")
    mode = st.radio("Scale", ["No scale (pixels only)", "I know the millimetres per pixel", "A reference length is visible"],
                    horizontal=True, key="fx_scale")
    mmpp = ref_px = ref_mm = None
    if mode.startswith("I know"):
        mmpp = st.number_input("Millimetres per pixel", min_value=0.0, value=0.0, step=0.01, format="%.4f", key="fx_mmpp") or None
    elif mode.startswith("A reference"):
        a, b = st.columns(2)
        ref_mm = a.number_input("Known length (mm)", min_value=0.0, value=0.0, step=1.0, key="fx_refmm") or None
        ref_px = b.number_input("Same length measured on the photo (pixels)", min_value=0.0, value=0.0, step=1.0, key="fx_refpx") or None
    if st.button("Measure the selected squares", disabled=not chosen, key="fx_measure"):
        st.session_state["fx_meas"] = {"image_id": image_id, "rows": [
            {"rank": t.rank, **rv.measure_tile(img, t, mm_per_px=mmpp, ref_px=ref_px, ref_mm=ref_mm)} for t in chosen]}
    meas = st.session_state.get("fx_meas")
    meas_by_rank = {}
    if meas and meas["image_id"] == image_id:
        rows = []
        for m in meas["rows"]:
            meas_by_rank[m["rank"]] = m
            w = m["crack_width_mm"]
            rows.append({"square": m["rank"],
                         "width (mm)": (f"{'<= ' if m['upper_bound'] else ''}{w:.2f} +/- {m['crack_width_uncertainty_mm']:.2f}" if w is not None else "no scale / not measurable"),
                         "width p95 (px)": None if m["width_px_p95"] is None else round(m["width_px_p95"], 1),
                         "length (px)": None if m["length_px"] is None else round(m["length_px"], 1), "basis": m["basis"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Width comes from cascade.measure: a dark-thin-line mask, not a trained segmenter; shadows and joints can be counted. "
                   "It has no measured error on facades; the band shows pixel quantisation and scale error only.")

    # ---- human review ------------------------------------------------------------------------
    st.subheader("Human review")
    rsel = st.selectbox("Square", ranks, key="fx_rev_rank")
    decision = st.radio("Reviewer decision", ["pending", "crack: schedule a close-up inspection", "not a crack", "cannot tell from the photo"],
                        key="fx_rev_dec", horizontal=True)
    note = st.text_input("Reviewer note", key="fx_rev_note")
    if st.button("Add to review list", key="fx_rev_add"):
        t = next(t for t in top if t.rank == rsel)
        rec = rv.review_note(image_id, t, graded=graded_by_rank.get(rsel), measured=meas_by_rank.get(rsel), model_id=res.model_id, threshold=THR)
        rec.update({"reviewer_decision": decision, "reviewer_note": note})
        if decision != "pending":
            rec["status"] = "reviewed by a person (not filed)"
        st.session_state.setdefault("fx_reviews", []).append(rec)
    revs = st.session_state.get("fx_reviews", [])
    if revs:
        st.dataframe(pd.DataFrame(revs)[["image_id", "tile_rank", "p_crack", "fisp_class", "crack_width_mm", "reviewer_decision", "status"]], hide_index=True)
        st.download_button("Download review list (JSON)", json.dumps(revs, indent=2), file_name="facade_review_list.json", key="fx_dl")

# ---------------------------------------------------------------- measured results -------------
st.header("How well does the screen work? (measured on held-out photos)")
st.markdown("Labels: **REAL** photos only. No synthetic or injected tiles. Every figure below is read from `eval/facade/tilecls_v1.json`.")
sd = R["sets"]["sdnet_test"]
bf = R["sets"]["bfdd_ood"]
m_sd = sd["models"]["resnet18_ozgenel_sdnet"]
m_bf = bf["models"]["resnet18_ozgenel_sdnet"]
a, b, c, d = st.columns(4)
a.metric("In-domain AUROC (REAL)", f3(m_sd["auroc"]), help=f"SDNET2018 walls, {sd['groups']} held-out source photos, n={sd['n']} squares")
b.metric("In-domain cracks caught", pct(m_sd["recall"]), help=f"recall at the validation-chosen threshold {f3(m_sd['threshold'])}")
c.metric("Drone facade AUROC (REAL, OOD)", f3(m_bf["auroc"]), help=f"BFDD, {bf['groups']} flights, n={bf['n']} squares")
d.metric("Drone facade cracks caught", pct(m_bf["recall"]), help="recall at the same threshold")

rows = []
for sk, sv in R["sets"].items():
    for mk, mv in sv["models"].items():
        cis = mv.get("ci95_group_bootstrap") or {}
        rows.append({"test set": SET_NAMES.get(sk, sk), "data": "REAL", "model": MODEL_NAMES.get(mk, mk),
                     "AUROC [95% CI]": f3(mv["auroc"]) + ci(cis.get("auroc")), "AP [95% CI]": f3(mv["ap"]) + ci(cis.get("ap")),
                     "precision": f3(mv["precision"]), "recall": f3(mv["recall"]), "F1": f3(mv["f1"]),
                     "squares skipped": pct(mv["share_below_threshold"]),
                     "n (crack / no crack)": f"{sv['n']} ({sv['crack']} / {sv['no_crack']})", "groups": sv["groups"]})
    ox = ((OX or {}).get("sets") or {}).get(sk)
    if ox:  # the file this page actually runs, re-scored on the same rows (no bootstrap CI)
        mo = ox["shipped_onnx"]
        rows.append({"test set": SET_NAMES.get(sk, sk), "data": "REAL", "model": "Same model, shipped ONNX file (re-score, no CI)",
                     "AUROC [95% CI]": f3(mo["auroc"]), "AP [95% CI]": f3(mo["ap"]), "precision": f3(mo["precision"]),
                     "recall": f3(mo["recall"]), "F1": f3(mo["f1"]), "squares skipped": pct(mo["share_below_threshold"]),
                     "n (crack / no crack)": f"{ox['n']} ({ox['crack']} / {ox['n'] - ox['crack']})", "groups": ox["groups"]})
st.dataframe(pd.DataFrame(rows), hide_index=True)
if OX:
    ag_sd, ag_bf = OX["sets"]["sdnet_test"]["agreement"], OX["sets"]["bfdd_ood"]["agreement"]
    st.caption(f"Provenance: the model rows with CIs are scored from the PyTorch checkpoint's predictions (training environment). "
               f"The page runs the shipped ONNX file (`{OX['onnx_file']}`); re-scored on the same squares at the same threshold it differs by at most "
               f"{max(ag_sd['max_abs_p_diff'], ag_bf['max_abs_p_diff']):.4f} in p(crack), and {ag_sd['flag_flips'] + ag_bf['flag_flips']} of "
               f"{OX['sets']['sdnet_test']['n'] + OX['sets']['bfdd_ood']['n']:,} squares change side of the threshold (`eval/facade/onnx_rescore.json`).")
else:
    st.caption(f"Provenance: model rows are scored from the PyTorch checkpoint's predictions. The shipped ONNX file differs by at most "
               f"{R['latency']['ort_vs_torch_max_abs_p_diff']:.4f} in p(crack) on {R['latency']['ort_cpu_app_env']['n_tiles']} test squares.")
st.caption(f"Split: training used Ozgenel tiles and SDNET training photos; the threshold was chosen on {R['sdnet_val']['groups']} SDNET validation photos; "
           f"the SDNET test photos and all BFDD flights were never seen in training or threshold choice. CIs: {R['n_boot']} bootstrap resamples of whole "
           "photos (SDNET) or flights (BFDD). 'Squares skipped' = share below the threshold, i.e. grader calls the screen saves.")

chart_df = pd.DataFrame([{"test set": SET_NAMES[sk].split(",")[0], "model": MODEL_NAMES[mk], "AUROC": mv["auroc"]}
                         for sk, sv in R["sets"].items() for mk, mv in sv["models"].items()])
st.altair_chart(alt.Chart(chart_df).mark_bar().encode(
    x=alt.X("AUROC:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("model:N", title=None, sort=list(MODEL_NAMES.values())),
    color=alt.Color("model:N", legend=None, sort=list(MODEL_NAMES.values())), row=alt.Row("test set:N", title=None)).properties(width=420, height=120))

best_base_sd = max((v["auroc"], k) for k, v in sd["models"].items() if k != "resnet18_ozgenel_sdnet")
best_base_bf = max((v["auroc"], k) for k, v in bf["models"].items() if k != "resnet18_ozgenel_sdnet")
heur_bf = bf["models"]["heuristic_crack_mask"]
st.markdown(
    f"- **Where it wins:** on held-out SDNET photos the shipped model's AUROC is {f3(m_sd['auroc'])} against {f3(best_base_sd[0])} for the best baseline "
    f"({MODEL_NAMES[best_base_sd[1]]}). On drone photos it is {f3(m_bf['auroc'])} against {f3(best_base_bf[0])}.\n"
    f"- **Where it is weak:** on SDNET only {pct(m_sd['precision'])} of flagged squares were real cracks, so an inspector dismisses many false alarms; "
    f"it still missed {pct(1 - m_sd['recall'])} of cracked squares at this threshold.\n"
    f"- **Where a baseline fails:** the repo's dark-line heuristic scores AUROC {f3(heur_bf['auroc'])} on drone photos, "
    f"worse than the constant guess ({f3(bf['models']['constant_prior']['auroc'])}).\n"
    f"- **Read AP with care on BFDD:** {pct(bf['prevalence'])} of BFDD squares contain a labelled crack, so a constant guess already gets AP {f3(bf['models']['constant_prior']['ap'])}. "
    f"There the screen skips only {pct(m_bf['share_below_threshold'])} of squares.")

# ---------------------------------------------------------------- engineers ---------------------
with st.expander("For engineers: method, full metrics, limits"):
    tr = R["training"]["main"]
    ex = R["export"]
    st.markdown(
        f"**Model.** `{CARD['model_id']}`: {CARD['architecture']}. Input {WIN} px squares cut at native resolution; ImageNet normalisation. "
        f"Trained {tr['epochs']} epochs (AdamW lr {tr['lr']}, weight decay {tr['weight_decay']}, batch {tr['batch']}, class-weighted cross-entropy, "
        f"GPU augmentation: random resized crop, flips, quarter-turn rotations, brightness/contrast/saturation) on {tr['n_train']} tiles "
        f"({tr['n_train_crack']} crack / {tr['n_train_no_crack']} no crack) on {tr['gpu']}. Checkpoint chosen by validation AP (epoch {tr['selected_epoch']}).\n\n"
        f"**Threshold.** {CARD['threshold_rule']}: p = {THR:.4f}. Never tuned on a test set.\n\n"
        f"**Export.** ONNX opset {ex['opset']} ({ex['exporter']}); fp32 parity with PyTorch: max |logit diff| {ex['fp32_max_abs_logit_diff']:.2e} on {ex['parity_n_tiles']} test tiles. "
        f"Shipped file stores weights as fp16 ({ex['shipped_file_bytes'] / 1e6:.1f} MB) and computes in fp32: max |p diff| {ex['fp16store_max_abs_p_diff']:.4f}. "
        f"Against the PyTorch fp16-autocast predictions used for the headline table: max |p diff| {R['latency']['ort_vs_torch_max_abs_p_diff']:.4f} "
        f"on {R['latency']['ort_cpu_app_env']['n_tiles']} test tiles"
        + (f"; on all {sum(v['n'] for v in OX['sets'].values())} validation and test tiles, see the re-score rows above "
           f"(threshold the ONNX validation scores would pick: {OX['threshold_onnx_val_would_pick']:.4f} vs {THR:.4f} used)." if OX else ".") + "\n\n"
        f"**Latency.** onnxruntime CPU in the app environment: {R['latency']['ort_cpu_app_env']['ms_per_tile']} ms per square "
        f"({R['latency']['ort_cpu_app_env']['threads']} threads, shared laptop); PyTorch GPU fp16: {R['latency']['gpu_torch_fp16_b128_ms_per_tile']} ms per square.")
    cm_rows = []
    for sk, sv in R["sets"].items():
        for mk, mv in sv["models"].items():
            cm_rows.append({"set": sk, "model": mk, "threshold": f3(mv["threshold"]), "TP": mv["tp"], "FP": mv["fp"], "FN": mv["fn"], "TN": mv["tn"],
                            "recall at precision >= 0.9 (oracle)": f3(mv["recall_at_precision_0_9_oracle"])})
    st.markdown("**Confusion matrices at the validation threshold** (the 'oracle' column picks its threshold on the test set itself and is context only).")
    st.dataframe(pd.DataFrame(cm_rows), hide_index=True)
    pr_path = ROOT / "eval" / "facade" / "pr_curves.csv"
    if pr_path.exists():
        pr = pd.read_csv(pr_path)
        st.altair_chart(alt.Chart(pr).mark_line().encode(x="recall:Q", y="precision:Q", color="model:N", column=alt.Column("set:N", title=None)).properties(width=260, height=220))
    st.markdown("**Training log** (validation = SDNET validation photos)")
    st.dataframe(pd.DataFrame(tr["log"]), hide_index=True)
    man = R["manifest"]
    st.markdown("**Data counts** (from `eval/facade/manifest_summary.json`)")
    st.dataframe(pd.DataFrame([{"dataset/split": k, **v} for k, v in man["counts"].items()]), hide_index=True)
    st.markdown("**Rules:** " + " | ".join(f"{k}: {v}" for k, v in man["rules"].items()))
    chk = ROOT / "eval" / "facade" / "bfdd_label_check.jpg"
    if chk.exists():
        st.image(str(chk), caption="BFDD label value 1 drawn in red over the photo: it follows cracks, so we read value 1 as 'crack' (the archive has no README)")
    st.markdown("**Limits**\n" + "\n".join(f"- {x}" for x in CARD.get("limits", [])) +
                "\n- The grader call and the width measurement are optional and never change the screen's score."
                f"\n- {GRADER_CAPTION} The page asks for confirmation before any crop is sent to the API."
                "\n- The review list: side-by-side placement lists every window; half-overlapping placement suppresses overlaps "
                f"above IoU {hm.NMS_IOU} (non-maximum suppression)."
                "\n- BFDD frames were shot by one team on a few days; results on other facades, cameras and distances are unmeasured.")

# ---------------------------------------------------------------- sources -----------------------
st.divider()
st.markdown("#### Sources and licences")
if SRC:
    for s in SRC["sources"]:
        st.markdown(f"- [{s['title']}]({s['url']}): {s['licence']}; accessed {s['accessed']}. Used for: {s['used_for']}.")
st.caption("AI screening, not a QEWI/FISP finding. The AI proposes; a person decides. It never files reports or controls building systems.")
