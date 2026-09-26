# Changelog

Format follows Keep a Changelog. Two sections per release: **Docs** (planning and research documents) and **Product** (code, data, models, prompts). Dates are YYYY-MM-DD, Pacific.

## [Unreleased]

### Docs (2026-09-25, afternoon)
- Scope widened to multi-sensor structural health after the team whiteboard (2026-09-25 16:22): `docs/decisions.md` D-014 (Proposed) with the per-person map, options, consequences and revisit trigger; D-001 and D-002 marked "superseded in part by D-014" and kept.
- `docs/problem_statement.md` v1.1: scope paragraph and hypotheses (H7 to H10) cover thermal heatmaps, sonar, seismic readings and lidar as inputs and SW+HW as the product; new numbers tagged [PUBLIC: url], [PUBLIC, secondary: url] or [Assumption] from R10; honesty box lists what stays imagery-only in the demo; every v1.0 sourced number kept.
- `docs/PRD.md` v1.1: section 5A Modalities table; FR-22 (modality on every record, default rgb), FR-23 (crack width in mm from an in-image scale or GSD with basis recorded), FR-24 (seismic series to indicators, frequency shift vs baseline, U without baseline), FR-25 (sonar as an asset class with an underwater rubric), FR-26 (lidar, roadmap); v1.1 native mappings in section 7; finding contract v0.1 with `modality` and `measurements.basis`; FR-1 to FR-21 unchanged.
- `README.md`: description, asset-class and modality table, `python -m cascade.measure` and `python -m cascade.signals` listed (being built).
- Research note `docs/research/10_multisensor_scope.md` (R10) added by the research workstream: sonar, seismic/vibration, thermal beyond PV, crack metrology, lidar, SW+HW pricing.
- Pending: team confirmation of D-001, D-002, D-008 and D-014 at the next sync.

### Product (2026-09-25, evening)
- Fixed 21 review findings in `measure.py`, `signals.py`, `schema.py`, `grade.py` and the new rubrics: no-baseline seismic series grade U; frequency shift needs two agreeing channels for S3 and carries one FFT bin of uncertainty; underwater rubric U row; crack width with uncertainty, scale basis and length; grader schema excludes measurement fields; NaN-safe signal parsing; CLI guards. 26 new tests.
- App: Measure crack panel in Findings and review, Sensors tab (CSV upload, baseline, synthetic demo labelled as such, spectrum and time-series charts, save as run), modality chips, Architecture inputs lane with six modalities and whiteboard owners. 198 tests.
- Docs: D-014 records the whiteboard box as two options (SW or SW+HW); eval_v1 size corrected to 189.

### Product (2026-09-25, midday)
- Added `src/cascade/video.py` (FR-4): ffmpeg/ffprobe frame extraction at a fixed interval or on scene changes, dHash near-duplicate suppression, `frames.json` sidecar, `ingest_video`, CLI; `.streamlit/config.toml` raises the upload cap to 1 GB.
- Added `src/cascade/clientreport.py`: per-client HTML + markdown + JSON reports with inline SVG charts and embedded evidence thumbnails; `clients/index.json`; CLI.
- Added `src/cascade/workforce.py`: review-desk workload estimate with sourced public figures, labelled assumptions, sensitivity tornado, per-client split; `render_markdown` for report.md.
- Added `src/cascade/drift.py`, `drift_thresholds.json`, `canary.py`: fingerprint, contract audit, U-rate p-chart, gate / review / ops / input health, baseline cards, promotion checklist, eval ledger and guard, canary CLI (cost estimate, `--confirm` required).
- Changed `schema.ImageRecord` (+ `client_id`, `asset_id`, `source_video`, `frame_time_s`), `ingest_folder` (client and asset passthrough), `pipeline` (fingerprint.json, summary fingerprint), `run.py` (manifest copy), `costlog` (prefix pricing; served model, request id, stop category), `grade.py` (effort pinned high, response metadata), `gate.py` (response metadata), `review.py` (migrate, blind rows, decisions), `export.py` and `surge.py` (U never written as a native grade), `exemplars.py` (eval_v1 excluded), `eval/run_eval.py` (fingerprint guard and ledger).
- App: video uploads and frame strips, client id inputs, Client reports tab, Workforce impact panel, Model health tab, demo video button (ingest only), de-duplicated queue galleries. 123 tests.

### Product (2026-09-25, morning)
- Added the Architecture tab: animated SVG data-flow diagram of the seven stages with live counts from the open run, the inputs lane (capture, metadata, rubric files, exemplars), the feedback lane (cost log, review log, frozen eval, reports), a stage-by-stage table, and a startup-level flow (customers, what they have, engine, outputs, moat).
- Added the mixed demo manifest (`--mixed` in `scripts/make_demo_manifests.py`) as the default dataset; landing preview grouped by asset class.
- UI polish: CSS animations, progress bar with stage and cost, toasts; FR-19 agreement timeline (`ReviewLog.timeline`); in-app bootstrap scoring button and scored-report viewer; demo script on the landing page. 42 tests.
- Rebuilt `app/streamlit_app.py` with six tabs: Inspect run (image galleries with level badges), Drop & grade (drag-and-drop inference), Batch (multi-dataset runs), Reports (stored `report.md`/`report.json` per run, comparison charts), Eval matrix (gate and grading confusion heatmaps), Why this approach (sourced incumbent comparison with measured column).
- Added `src/cascade/evalmetrics.py` (gate 2x2, per-class confusion, exact / within-one / QWK / U rate; bridge MBEI CS predictions now mappable) and `src/cascade/report.py` (`run_metrics`, `write_run_report`); every run now writes `report.md` and `report.json`. 7 new tests (41 total).
- Added `RunConfig.force_route_classes` (default `pv_module`): forced routing to the grader per asset class, gate verdict preserved in `gate.jsonl` with a `[forced: ...]` reason suffix. `--force-route` CLI flag, app sidebar checkbox, two tests (34 total).

### Product (2026-09-25, early morning)
- Added `src/cascade/pipeline.py`: `run_cascade()` extracted from the CLI so the CLI, surge mode and the UI share one resumable loop with a live `Progress` callback (stage counters, cost, latency); `load_run()` and `save_findings()` for the UI.
- Added `src/cascade/exemplars.py` (FR-13): dev-set-only few-shot exemplars spread across grade values, never the image's own label; `--exemplars DEV_MANIFEST` and `--exemplars-k` on the CLI.
- Added `src/cascade/surge.py` (FR-20): `python -m cascade.surge --folder ...` runs the FEMA PDA rubric over a folder, whole-frame only, and writes `surge_counts.json` and `surge_report.md` with counts per class, per level, U count and a ranked list.
- Added `write_bridge_csv` (FR-17): `bridge_entry.csv` with element, condition state, quantity and unit columns; quantity stays blank unless an area was measured.
- Added `app/streamlit_app.py` (FR-21): dataset or upload, backend selection, live counters, findings with evidence crops and verbatim criteria, accept / override / mark U with the SQLite review log and re-ranking, queue with multipliers, surge counts, exports, eval report viewer. Headless `AppTest` smoke test passes.
- Added `scripts/make_demo_manifests.py` (10 images per dataset from the dev set, under `data/demo/`, gitignored), `README.md`, and 15 new tests (`test_pipeline.py`, `test_surge.py`, `test_exemplars.py`); 32 tests pass.
- Changed `environment.yml` to install the `ui` extra; the project now runs in the conda env `origin_hack` (the earlier `.venv` is unused and ignored).
- `run.py` rewritten as a thin CLI over `pipeline.run_cascade`; `--rubric` added for the NBI bridge rubric.

### Product (2026-09-24, late evening)
- Added `src/cascade/`: `schema.py` (finding contract as Pydantic models), `ingest.py`, `gate.py` (local Ollama small VLM with JSON schema; Claude Haiku fallback), `crop.py` (tiling, resizing), `grade.py` (Claude structured-output grader with rubric rows and exemplars; local Ollama grader), `prioritize.py` (PRD section 8 scoring), `review.py` (SQLite review log), `export.py` (queue CSV, findings JSON), `run.py` (CLI), `costlog.py` (per-call tokens, dollars, seconds).
- Added rubric files `src/cascade/rubrics/`: `corrosion_cs.json` (verbatim text from the dataset's annotation guidelines), `bridge_mbei.json`, `bridge_nbi.json`, `pv_iec62446_3.json`, `disaster_fema.json`.
- Added `eval/build_eval.py`, `eval/run_eval.py`, and `eval/rubrics/` class-to-severity maps and the dacl10k team-grading rubric, all written before any model output.
- Added `scripts/download_data.py`, `pyproject.toml`, `requirements.txt`, `.env.example`, `.gitignore`, and 17 unit tests under `tests/` (all passing).
- Changed the corrosion unified mapping to follow the MBEI row (Fair S1, Poor S2, Severe S3) in the rubric and `docs/PRD.md` section 7; the earlier draft had Fair S2 / Severe S4.
- Added `--grader none` to the run CLI so the gate can be evaluated with no API spend.

## [0.1.0] - 2026-09-24

### Docs
- Added `problem.md`: the organizers' Prompt D text, unchanged.
- Added `docs/problem_statement.md` v0.1, then raised to v1.0 with research citations, hypothesis verdicts (H1 to H6), and the wedge reference to D-001.
- Added research notes `docs/research/01` to `09`: competitors (wind/solar; bridges/grid/telecom; underwater/industrial/disaster), market pain and regulation, grading standards with a unified S0 to S4 schema, VLM landscape, datasets and eval plan, customer voice and discovery plan, business models.
- Added `docs/research/00_research_brief.md`: synthesis, slide-safe numbers, numbers to avoid, gaps.
- Added `docs/decisions.md` D-001 to D-013.
- Added `docs/PRD.md` v1.0 with FR-1 to FR-21, grading spec, prioritization model, finding contract.
- Added `docs/implementation_plan.md` v1.0 with the Thursday-to-Monday schedule, repo layout, demo script, Q&A drills, risk register.
- Added `docs/documentation.md` (index, conventions, technical notes, glossary), `docs/progress.md`, and this changelog.

### Product
- None.
