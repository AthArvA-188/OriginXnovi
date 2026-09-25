"""FR-21 demo UI: pick a dataset or upload images, run the cascade, watch stage counters,
inspect findings with evidence crops, review (accept / override / mark U), see the queue
with its multipliers, run surge mode, export CSV/JSON, and read the eval report.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from cascade.exemplars import exemplar_provider  # noqa: E402
from cascade.export import finding_row  # noqa: E402
from cascade.ingest import ingest_folder, read_manifest, write_manifest  # noqa: E402
from cascade.pipeline import LEVELS, Progress, RunConfig, load_run, run_cascade, save_findings  # noqa: E402
from cascade.prioritize import SEVERITY_WEIGHT, consequence_for, urgency_for  # noqa: E402
from cascade.review import ReviewLog  # noqa: E402
from cascade.schema import Finding, ImageRecord  # noqa: E402
from cascade.surge import surge_counts, write_surge_report  # noqa: E402

RUNS = ROOT / "runs"
DEV_MANIFEST = ROOT / "data" / "dev" / "manifest.jsonl"
EVAL_MANIFEST = ROOT / "data" / "eval_v1" / "manifest.jsonl"
DEMO_DIR = ROOT / "data" / "demo"
DATASET_LABELS = {
    "corrosion_cs": "Steel coating: corrosion condition state (bridge steel)",
    "dacl10k": "Bridge elements: concrete defects (dacl10k)",
    "ir_solar": "PV thermal modules (InfraredSolarModules)",
    "rescuenet": "Post-disaster UAV (RescueNet), surge mode",
}
LEVEL_COLOR = {"S0": "#7a7a7a", "S1": "#3b82f6", "S2": "#f59e0b", "S3": "#f97316", "S4": "#dc2626", "U": "#8b5cf6"}

st.set_page_config(page_title="Inspection grading cascade", layout="wide")


# ---------- helpers ----------


def list_manifests() -> dict:
    """label -> manifest path. Demo manifests first, then the dev set."""
    out = {}
    if DEMO_DIR.exists():
        for p in sorted(DEMO_DIR.glob("*/manifest.jsonl")):
            out[f"demo: {DATASET_LABELS.get(p.parent.name, p.parent.name)}"] = p
    if DEV_MANIFEST.exists():
        out["dev set (all four datasets, 50 images)"] = DEV_MANIFEST
    return out


def records_for(manifest: Path, dataset_filter, limit: int):
    recs = read_manifest(manifest)
    if dataset_filter:
        recs = [r for r in recs if r.source_dataset == dataset_filter]
    return recs[:limit] if limit else recs


def list_runs() -> list:
    if not RUNS.exists():
        return []
    return sorted([p.name for p in RUNS.iterdir() if p.is_dir() and (p / "gate.jsonl").exists()], reverse=True)


def draw_evidence(f: Finding, rec: ImageRecord) -> Image.Image:
    img = Image.open(rec.path).convert("RGB")
    if f.evidence.bbox and f.evidence.tile != "full":
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = f.evidence.bbox
        d.rectangle([x0, y0, x1, y1], outline=LEVEL_COLOR.get(f.unified.level, "#ffffff"), width=max(3, img.width // 300))
    if max(img.size) < 320:  # 24x40 thermal crops
        s = 320 / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)), Image.NEAREST)
    return img


def multipliers(f: Finding, rec) -> dict:
    return {
        "severity_weight": SEVERITY_WEIGHT.get(f.unified.level),
        "criticality": 1.0,
        "consequence": consequence_for(f),
        "urgency": urgency_for(rec.captured_on if rec else None, date.today()),
    }


def render_counters(box, p: Progress, cfg: RunConfig):
    with box.container():
        c = st.columns(7)
        c[0].metric("Images", p.images)
        c[1].metric("Gated", p.gated)
        c[2].metric("Routed", p.routed, help="recall-first: damage, unusable, or low-confidence clean verdicts")
        c[3].metric("Graded", p.graded)
        c[4].metric("Findings", p.findings)
        c[5].metric("Cost USD", f"{p.usd:.3f}", help="API list price from the call log; local models cost $0")
        c[6].metric("Model seconds", f"{p.seconds_gate + p.seconds_grade:.0f}", help=f"gate {p.seconds_gate:.0f} s, grade {p.seconds_grade:.0f} s")
        st.caption(
            f"stage **{p.stage}** · image `{p.current_image}` · gate={cfg.gate} grader={cfg.grader} tiles={cfg.tiles} · levels: "
            + "  ".join(f"{k} {v}" for k, v in p.levels.items())
        )


def findings_df(findings) -> pd.DataFrame:
    cols = ["queue_rank", "queue_score", "image_id", "unified_level", "native_value", "standard", "defect_type", "action_code", "sla_days", "confidence", "flags", "review_status", "finding_id"]
    rows = [finding_row(f) for f in findings]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)


def load_records_for_run(out: Path) -> dict:
    """image_id -> ImageRecord from the run's own manifest, else the dev and eval manifests."""
    recs = []
    for mp in [out / "manifest.jsonl", DEV_MANIFEST, EVAL_MANIFEST]:
        if mp.exists():
            recs += read_manifest(mp)
    return {r.image_id: r for r in recs}


# ---------- sidebar ----------

st.sidebar.title("Inspection grading cascade")
st.sidebar.caption("gate → crop → grade → prioritize → review → export")
mode = st.sidebar.radio("Mode", ["Inspect", "Surge (disaster)"], horizontal=True)
surge = mode.startswith("Surge")
source = st.sidebar.radio("Images", ["Dataset", "Upload"], horizontal=True)

manifests = list_manifests()
records = []
if source == "Dataset":
    if not manifests:
        st.sidebar.error("No manifests found. Run `python scripts/make_demo_manifests.py` first.")
    choice = st.sidebar.selectbox("Dataset", list(manifests) or ["none"])
    dataset_filter = None
    if choice in manifests and manifests[choice] == DEV_MANIFEST:
        pick = st.sidebar.selectbox("Filter dev set", ["all"] + list(DATASET_LABELS), format_func=lambda k: "all" if k == "all" else DATASET_LABELS[k])
        dataset_filter = None if pick == "all" else pick
    limit = st.sidebar.slider("Max images", 1, 50, 10)
    if choice in manifests:
        records = records_for(manifests[choice], dataset_filter, limit)
        if surge:
            records = [r for r in records if r.asset_class == "building_disaster"]
else:
    asset_class = "building_disaster" if surge else st.sidebar.selectbox("Asset class", ["bridge_element", "steel_coating", "pv_module", "building_disaster"])
    files = st.sidebar.file_uploader("JPEG / PNG", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if files:
        up = RUNS / "_uploads" / time.strftime("%Y%m%d_%H%M%S")
        up.mkdir(parents=True, exist_ok=True)
        for fobj in files:
            (up / fobj.name).write_bytes(fobj.getbuffer())
        records = ingest_folder(up, asset_class=asset_class, source_dataset="upload", split="upload", id_prefix="up")
        write_manifest(records, up / "manifest.jsonl")
        st.sidebar.success(f"{len(records)} images ingested; hash, size and EXIF date recorded, missing metadata stays null")

st.sidebar.divider()
gate = st.sidebar.selectbox("Gate (stage A)", ["local", "claude", "none"], help="local = Qwen3-VL-4B via Ollama; claude = Haiku 4.5; none = route everything")
grader = st.sidebar.selectbox("Grader (stage C)", ["claude", "local", "none"], help="claude = " + os.getenv("GRADER_MODEL", "claude-opus-5") + "; local = Qwen3-VL-8B via Ollama; none = gate only")
tiles = st.sidebar.checkbox("Tile large images (FR-8)", value=False, disabled=surge, help="one grader call per 1568 px tile; surge mode always grades the whole frame")
use_exemplars = st.sidebar.checkbox("Few-shot exemplars from dev set (FR-13)", value=False, disabled=not DEV_MANIFEST.exists())
gate_min_conf = st.sidebar.slider("Route clean verdicts below confidence", 0.0, 1.0, 0.7, 0.05)
run_name = st.sidebar.text_input("Run name", value=f"ui_{time.strftime('%m%d_%H%M')}")
reviewer = st.sidebar.text_input("Reviewer name (for the review log)", value=os.getenv("USERNAME", "reviewer"))
run_clicked = st.sidebar.button("Run cascade", type="primary", disabled=not records, width="stretch")

st.sidebar.divider()
open_run = st.sidebar.selectbox("Or open a finished run", ["(none)"] + list_runs())

# ---------- run ----------

cfg = RunConfig(gate=gate, grader=grader, tiles=tiles and not surge, gate_min_conf=gate_min_conf)
active_run = st.session_state.get("active_run")

if run_clicked and records:
    out = RUNS / run_name
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(records, out / "manifest.jsonl")
    st.session_state["active_run"] = run_name
    counters = st.empty()
    log_box = st.empty()
    lines = []

    def on_progress(p: Progress):
        render_counters(counters, p, cfg)
        entry = f"{p.current_image} -> {p.stage}"
        if p.current_image and (not lines or lines[-1] != entry):
            lines.append(entry)
            log_box.code("\n".join(lines[-12:]))

    provider = exemplar_provider(DEV_MANIFEST, k=3) if use_exemplars else None
    with st.spinner("Running. A cold local gate can take a minute; each heavy grade is 10 to 40 s."):
        try:
            summary = run_cascade(records, out, cfg, exemplars=provider, progress=on_progress)
            if surge:
                write_surge_report(surge_counts(load_run(out)["findings"]), summary, out)
            st.success(f"Done: {summary['findings']} findings from {summary['images']} images, ${summary['usd_total']} total, routed {summary['routed_to_grader']} of {summary['gated']}.")
        except Exception as e:  # partial outputs stay on disk; the run is resumable
            st.error(f"Run stopped: {type(e).__name__}: {e}. Outputs so far are in runs/{run_name}; press Run again with the same name to resume.")
    active_run = run_name
elif open_run != "(none)":
    active_run = open_run
    st.session_state["active_run"] = open_run

if not active_run:
    st.title("Inspection grading cascade")
    st.markdown(
        """
Pick a dataset or upload images on the left, choose the gate and grader backends, and press **Run cascade**.

**What happens per image**

1. **Gate** (local small VLM): usable? damage present? Confidence and a one-line reason. Recall-first routing.
2. **Crop**: large images tiled at 1,568 px with overlap; tile coordinates are kept as evidence.
3. **Grade** (heavy VLM, schema-enforced): native scale value with the rubric criterion quoted verbatim, unified S0 to S4, measurements or `not_measurable`, action and justification. Refusals become U, never S0.
4. **Prioritize**: score = severity × criticality × consequence × urgency; any S4 goes to the top with same-day escalation.
5. **Review**: accept, override or mark U; every action is logged with the prior model value.
6. **Export**: queue CSV, findings JSON, bridge entry CSV.

Numbers shown here are measured from this run's call log. Accuracy claims live only in `eval/reports/`.
"""
    )
    st.stop()

# ---------- load the active run ----------

out = RUNS / active_run
run = load_run(out)
findings = run["findings"]
gate_rows, calls, summary = run["gate"], run["calls"], run["summary"]
imgs = load_records_for_run(out)
recs = list(imgs.values())
review_log = ReviewLog(out / "reviews.sqlite")

st.title(f"Run `{active_run}`")
tab_over, tab_find, tab_queue, tab_surge, tab_export, tab_eval = st.tabs(["Overview", "Findings & review", "Work queue", "Surge counts", "Export", "Eval report"])

with tab_over:
    p = Progress(
        images=summary.get("images", len(gate_rows)),
        gated=len(gate_rows),
        routed=sum(1 for g in gate_rows if g["routed"]),
        unusable=sum(1 for g in gate_rows if not g["usable"]),
        graded=len({f.evidence.image_ids[0] for f in findings if f.evidence.image_ids}),
        findings=len(findings),
        usd=round(sum(c["usd"] for c in calls), 4),
        seconds_gate=round(sum(c["seconds"] for c in calls if c["stage"] == "gate"), 1),
        seconds_grade=round(sum(c["seconds"] for c in calls if c["stage"] == "grade"), 1),
        stage="done",
    )
    for f in findings:
        p.levels[f.unified.level] = p.levels.get(f.unified.level, 0) + 1
    shown_cfg = RunConfig(**summary["config"]) if summary.get("config") else cfg
    render_counters(st.empty(), p, shown_cfg)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Gate decisions")
        if gate_rows:
            st.dataframe(pd.DataFrame(gate_rows)[["image_id", "usable", "damage_present", "confidence", "routed", "seconds", "reason"]], width="stretch", hide_index=True, height=320)
        else:
            st.info("No gate rows yet.")
    with c2:
        st.subheader("Cost and latency per call")
        if calls:
            cdf = pd.DataFrame(calls)
            agg = cdf.groupby(["stage", "model"]).agg(calls=("usd", "size"), usd=("usd", "sum"), median_s=("seconds", "median"), in_tok=("input_tokens", "sum"), out_tok=("output_tokens", "sum")).reset_index()
            st.dataframe(agg, width="stretch", hide_index=True)
            st.caption(f"USD per image: {summary.get('usd_per_image')} · API list prices from src/cascade/costlog.py; local models are $0")
        else:
            st.info("No calls logged.")
    if summary:
        with st.expander("summary.json"):
            st.json(summary)

with tab_find:
    if not findings:
        st.info("No findings. Either nothing was routed or the grader was set to none.")
    else:
        df = findings_df(findings)
        st.dataframe(df, width="stretch", hide_index=True, height=260)
        fid = st.selectbox("Finding", df["finding_id"].tolist())
        f = next(x for x in findings if x.finding_id == fid)
        rec = imgs.get(f.evidence.image_ids[0]) if f.evidence.image_ids else None
        c1, c2 = st.columns([1.1, 1])
        with c1:
            if rec and Path(rec.path).exists():
                st.image(draw_evidence(f, rec), caption=f"{rec.image_id} · {rec.width}x{rec.height} · tile {f.evidence.tile} · bbox {f.evidence.bbox}", width="stretch")
                truth = rec.labels.get("grade_native") or rec.labels.get("source_class")
                if truth:
                    st.caption(f"Dataset label, for eval only and never shown to the model: **{truth}** ({rec.labels.get('grade_source')})")
            else:
                st.warning("Source image not on disk for this run.")
        with c2:
            st.markdown(
                f"### {f.native_scale.value} on {f.native_scale.standard} · <span style='color:{LEVEL_COLOR[f.unified.level]}'>**{f.unified.level}**</span> {f.unified.uncertainty}",
                unsafe_allow_html=True,
            )
            sla = f", within {f.action.sla_days} days" if f.action.sla_days is not None else ""
            st.markdown(f"**Defect:** {f.defect_type}  \n**Action:** {f.action.code}{sla}  \n**Basis:** {f.action.basis}")
            st.markdown("**Criteria matched (verbatim from rubric):**")
            for c in f.native_scale.criteria_matched or ["(none quoted)"]:
                st.markdown(f"> {c}")
            st.markdown(f"**Justification:** {f.justification}")
            m = f.measurements
            st.markdown(
                f"**Measurements:** area {m.area_cm2} cm², crack {m.crack_width_mm} mm, ΔT {m.delta_t_k} K, rust {m.percent_area_rusted} %, section loss {m.section_loss_pct} % · confidence **{m.confidence:.2f}** · flags: {', '.join(f.unified.flags) or 'none'}"
            )
            reviewed = f" by {f.review.reviewer} at {f.review.reviewed_at} (prior {f.review.prior_level})" if f.review.reviewer else ""
            st.caption(f"model {f.model} · ${f.usd:.4f} · {f.seconds:.1f} s · review: {f.review.status}{reviewed}")
            st.markdown("**Reviewer decision (FR-18)**")
            b1, b2, b3 = st.columns(3)
            gradable = [lvl for lvl in LEVELS if lvl != "U"]
            new_level = b2.selectbox("Override to", gradable, index=gradable.index(f.unified.level) if f.unified.level in gradable else 0, label_visibility="collapsed")
            action = None
            if b1.button("Accept", width="stretch"):
                action = ("accepted", None)
            if b2.button("Override", width="stretch"):
                action = ("overridden", new_level)
            if b3.button("Mark U", width="stretch"):
                action = ("marked_u", None)
            if action:
                review_log.apply(active_run, f, action[0], reviewer, action[1])
                save_findings(findings, out, recs)
                st.success(f"Logged {action[0]} by {reviewer}; prior level {f.review.prior_level}. Queue re-ranked.")
                st.rerun()
        agg = review_log.agreement(active_run)
        rate = f", agreement {agg['agreement_rate']:.0%}" if agg["agreement_rate"] is not None else ""
        st.caption(f"Review log: {agg['total']} decisions, accepted {agg['accepted']}, overridden {agg['overridden']}, marked U {agg['marked_u']}{rate} · persisted in reviews.sqlite")

with tab_queue:
    st.markdown("`score = severity_weight[S] × criticality × consequence × urgency`. Any **S4** sorts first and is escalated same day. **U** is listed, never scored as S0. Criticality is 1 unless supplied per asset.")
    if findings:
        rows = []
        for f in sorted(findings, key=lambda x: x.queue_rank or 10**9):
            rec = imgs.get(f.evidence.image_ids[0]) if f.evidence.image_ids else None
            rows.append(
                {
                    "rank": f.queue_rank,
                    "score": f.queue_score,
                    "level": f.unified.level,
                    "native": f.native_scale.value,
                    "image": f.evidence.image_ids[0] if f.evidence.image_ids else "",
                    "asset_class": f.asset_class,
                    "action": f.action.code,
                    "sla_days": f.action.sla_days,
                    **multipliers(f, rec),
                    "flags": ";".join(f.unified.flags),
                    "review": f.review.status,
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=480)
    else:
        st.info("Empty queue.")

with tab_surge:
    dis = [f for f in findings if f.asset_class == "building_disaster"]
    if not dis:
        st.info("No building_disaster findings in this run. Choose the RescueNet demo set or Surge mode.")
    else:
        counts = surge_counts(dis)
        c1, c2, c3 = st.columns(3)
        c1.metric("Images assessed", counts["n_images"])
        c2.metric("Unassessable (U)", counts["u_count"])
        c3.metric("Mean confidence", f"{counts['mean_confidence']:.2f}" if counts["mean_confidence"] is not None else "n/a")
        cc1, cc2 = st.columns(2)
        cc1.bar_chart(pd.Series(counts["by_fema_class"], name="FEMA PDA class"))
        cc2.bar_chart(pd.Series(counts["by_level"], name="unified level"))
        st.dataframe(pd.DataFrame(counts["ranked"]), width="stretch", hide_index=True)
        rp = out / "surge_report.md"
        if rp.exists():
            st.download_button("Download surge_report.md", rp.read_bytes(), file_name="surge_report.md")

with tab_export:
    for name in ["queue.csv", "findings.json", "bridge_entry.csv", "gate.jsonl", "calls.jsonl", "summary.json", "surge_counts.json", "surge_report.md"]:
        fp = out / name
        if fp.exists():
            st.download_button(f"Download {name}", fp.read_bytes(), file_name=f"{active_run}_{name}", key=f"dl_{name}")
    st.caption("queue.csv columns are documented in src/cascade/export.py (QUEUE_COLUMNS). bridge_entry.csv carries element, condition state and quantity columns for SNBI-style entry (FR-17).")

with tab_eval:
    reports = sorted((ROOT / "eval" / "reports").glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not reports:
        st.info("No eval report yet. Run `python eval/run_eval.py --manifest data/eval_v1/manifest.jsonl --run runs/<id> --name <name>`.")
    else:
        rp = st.selectbox("Report", reports, format_func=lambda p: p.name)
        st.markdown(rp.read_text(encoding="utf-8"))
