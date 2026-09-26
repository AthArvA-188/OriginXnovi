# Inspection grading cascade

Origin Weekend Fall 2026, Prompt D (Infrastructure & Resilience). A multi-sensor
structural-health platform, software plus a sensor kit, on one grading engine
(`docs/decisions.md` D-014). Inputs are aerial and handheld imagery, thermal heatmaps,
underwater sonar frames, seismic and vibration readings and, on the roadmap, lidar. A small
local vision-language model gates every image frame, a heavy model grades flagged frames on
the industry-native scale with the rubric criterion quoted verbatim, crack widths are
measured in millimetres only when a scale reference is in the frame or a GSD is known (and
always shown with their uncertainty and basis), seismic series are reduced to indicators and
graded against a stored baseline in plain code (U when there is none), and every finding
rolls into one consequence-weighted work queue that a qualified inspector reviews, overrides
and exports. The same pipeline runs in surge mode after a disaster. Imagery is the demo and
the only measured accuracy; the sensor paths are minimal but real; the kit is a bill of
materials from public prices.

Documents: `docs/problem_statement.md`, `docs/PRD.md` (section 5A for modalities),
`docs/decisions.md`, `docs/implementation_plan.md`, `docs/progress.md`. Research notes in
`docs/research/` (R10 covers the sensor scope).

## Asset classes and modalities

| Modality | Asset classes and rubric | Status |
|---|---|---|
| `rgb` (aerial, handheld, video frames) | `bridge_element` (MBEI CS, NBI 0 to 9), `steel_coating` / corrosion CS, `building_disaster` (FEMA PDA) | Demo; measured on `eval_v1` |
| `rgb` with a scale reference | crack width in mm via `cascade.measure` (marker, ruler, camera plus stand-off, or GSD metadata); MBEI and ACI 224R width rows | Demo on marker images; `not_measurable` on `eval_v1` |
| `thermal` | `pv_module` (IEC TS 62446-3 CoA); bridge-deck delamination (ASTM D4788, screening only) and building envelope (ISO 6781-1) | PV demo; deck and envelope rubrics roadmap |
| `sonar` (underwater) | `underwater_sonar`: extent and geometry rows in SNBI B.C.11 style (scour, undermining, exposed footing, debris, missing pile); never crack width; Level I supplement only | Minimal path; labelled sample frames |
| `seismic` (ground and structure vibration) | `structure_vibration` (ShakeMap PGA/PGV, Hazus drift, per-asset frequency-drop as a team assumption), `machinery_vibration` (ISO 20816-3 zones) via `cascade.signals`; U without a baseline | Minimal path; synthetic or public series labelled as such |
| `lidar` | deflection and section-loss differencing | Roadmap |

## Setup (conda, not venv)

```powershell
conda env create -f environment.yml        # Python 3.13, installs the package with dev + ui extras
conda activate origin_hack
copy .env.example .env                     # then fill ANTHROPIC_API_KEY
ollama pull qwen3-vl:4b-instruct           # local gate
ollama pull qwen3-vl:8b-instruct           # optional local grader
# video ingestion (FR-4) shells out to ffmpeg/ffprobe on PATH; no Python video package is needed
```

If the env already exists: `conda activate origin_hack; pip install -e .[dev,ui]`.

Datasets (public, licenses noted in `docs/progress.md`): `python scripts/download_data.py`,
then `python eval/build_eval.py` to build the frozen `data/eval_v1` and `data/dev` manifests,
then `python scripts/make_demo_manifests.py` for the small demo sets used by the app.

## Run

```powershell
# gate only, no API cost
python -m cascade.run --manifest data/dev/manifest.jsonl --out runs/gate_dev --grader none

# full cascade on 10 images with dev-set exemplars in the grader prompt
python -m cascade.run --manifest data/demo/corrosion_cs/manifest.jsonl --out runs/demo_corr --exemplars data/dev/manifest.jsonl

# surge mode over a folder of post-event UAV images (FEMA PDA rubric)
python -m cascade.surge --folder data/raw/rescuenet/images --out runs/surge01 --limit 40

# score a run against the frozen eval set (bootstrap CIs)
python eval/run_eval.py --manifest data/eval_v1/manifest.jsonl --run runs/eval_v1_run1 --name eval_v1_run1

# demo UI
streamlit run app/streamlit_app.py   # tabs: Inspect run, Architecture, Drop & grade, Batch, Reports, Client reports, Eval matrix, Model health, Why this approach

# video to frames (every 2 s, near-duplicates dropped by dHash), then an ordinary run
python -m cascade.video --video data/demo/video/bridge_walkthrough.mp4 --out runs/v1/frames --every 2

# per-client HTML/markdown reports for a finished run
python -m cascade.clientreport --run runs/ui_0925_0856

# model health (zero model calls) is written to runs/<run>/health.json by the app; the canary re-grade refuses without --confirm
python -m cascade.canary --gate claude --grader claude   # prints the cost estimate and exits

# crack width in mm from an in-image scale or GSD, with uncertainty and basis (FR-23); being built, flags per --help
python -m cascade.measure --help

# seismic / vibration series to indicators and a graded finding, U without a baseline (FR-24); being built, flags per --help
python -m cascade.signals --help

# tests (no model, no network)
pytest -q
```

Backends: `--gate local|claude|none`, `--grader claude|local|none`. Model ids come from `.env`.
Runs are resumable: rerun with the same `--out` to continue after an interruption.

## Layout

```
src/cascade/
  schema.py      finding contract (PRD appendix A), pydantic
  ingest.py      folder or manifest -> ImageRecord rows (hash, size, EXIF date, sidecar metadata)
  gate.py        stage A: usable? damage? local Qwen3-VL via Ollama or Claude Haiku
  crop.py        stage B: 1568 px overlapping tiles, model-ready resizing
  grade.py       stage C: rubric-driven grading, schema-enforced (Claude parse / Ollama format)
  exemplars.py   FR-13: few-shot exemplars from the dev set
  rubrics/       standard rows per asset class (MBEI, NBI, corrosion CS, IEC 62446-3, FEMA PDA)
  prioritize.py  stage D: severity x criticality x consequence x urgency; S4 always first
  review.py      stage E: SQLite review log with prior model value
  export.py      queue CSV, findings JSON, bridge entry CSV (element, CS, quantity)
  pipeline.py    run_cascade(): resumable loop with live progress, shared by CLI, surge and UI
  surge.py       FR-20: FEMA triage over a folder, counts per class, U count, ranked list
  run.py         CLI
app/streamlit_app.py   FR-21 demo UI
eval/                  build_eval.py (freeze), run_eval.py (metrics + CIs), rubrics/, reports/
tests/                 unit and pipeline tests with fake backends
```

## Outputs per run (`runs/<id>/`)

| File | Content |
|---|---|
| `gate.jsonl` | one row per image: usable, damage_present, confidence, reason, routed, seconds |
| `findings.jsonl` / `findings.json` | finding contract rows; JSON is ranked |
| `queue.csv` | ranked work queue (columns in `export.QUEUE_COLUMNS`) |
| `bridge_entry.csv` | element, condition state, quantity columns for SNBI-style entry |
| `calls.jsonl` | tokens, dollars and seconds per model call |
| `summary.json` | counts, levels, routing fraction, cost per image |
| `reviews.sqlite` | reviewer decisions with prior model value |
| `surge_counts.json`, `surge_report.md` | surge mode only |

## Honesty rules

Every number in the deck is labelled Sourced, Team-measured, Inference or Assumption
(`docs/decisions.md` D-011). Accuracy claims come only from `eval/reports/` on the frozen
`eval_v1` set with n and confidence intervals. One image proves plumbing, not accuracy.
