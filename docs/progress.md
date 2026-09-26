# Progress Log

| Field | Value |
|---|---|
| Project | Origin Weekend Fall 2026, Prompt D (Infrastructure & Resilience) |
| Date started | 2026-09-24 |
| Owner | Founding team |
| Rule | Only record what has actually happened. Planned work lives in `docs/implementation_plan.md`. |

Times Pacific. Deadlines: Devpost Sun 2026-09-27 23:59; check-out form Mon 2026-09-28 12:00; pitch Mon 2026-09-28 18:00 at GCS.

## Status board

| Workstream | Status | Latest |
|---|---|---|
| Research (R01 to R09) | Done | Nine sourced notes, about 3,450 lines, completed 2026-09-24 ~22:00 |
| Problem statement | Done, v1.1 | Sourced evidence and hypothesis verdicts added 2026-09-24; multi-sensor scope, H7 to H10 and honesty box 2026-09-25 |
| Decisions, PRD, plan, docs index, changelog, research brief | Done, v1.0 (PRD v1.1) | Written 2026-09-24 evening; PRD section 5A and FR-22 to FR-26 added 2026-09-25 |
| F. Multi-sensor scope (D-014) | Docs done, Proposed; code in progress | R10 research note (355 lines), D-014, problem statement v1.1, PRD v1.1, README updated 2026-09-25 afternoon; `cascade.measure` (FR-23) and `cascade.signals` (FR-24) being built by other workflows; no sensor accuracy number exists or is claimed |
| Team registration form (due Thu 23:00) | Unknown | Verify tonight |
| Devpost accounts and project | Not started | |
| A. Data and eval | Built (Thu night) | Four datasets on disk with license notes; label indexes derived; class-to-severity maps and team-grading rubric written before any model output; `eval/build_eval.py` and `eval/run_eval.py` written; eval_v1 (189 images in the frozen manifest; earlier notes said 184) and dev (50) manifests built |
| B. Pipeline | Complete for the weekend scope (Fri early morning) | Shared resumable `pipeline.run_cascade`, exemplars (FR-13), surge mode (FR-20), bridge entry CSV (FR-17); 32 tests pass in the conda env; local gate run over the full dev set (50 images, 37 routed, 0 API cost); no full grading run yet |
| C. UI and demo | App built, not yet demoed (Fri early morning) | `app/streamlit_app.py` with run, counters, findings and review, queue, surge, export, eval tabs; headless smoke test passes; video not recorded |
| D. Customer discovery | Not started | Zero outreach sent; funnel 0 / 0 / 0 |
| E. Deck, video, Devpost | Not started | |

## Log

### 2026-09-24 (Thursday)

- ~21:10 Prompt saved as `problem.md` (unchanged from the organizers' text).
- ~21:30 `docs/problem_statement.md` v0.1 drafted.
- ~21:40 to 22:00 Research notes R01 to R09 written under `docs/research/`, each with summary, findings, key-numbers table, implications, open questions and sources.
- Evening: `docs/decisions.md` (D-001 to D-013), `docs/PRD.md`, `docs/implementation_plan.md`, `docs/documentation.md`, `docs/changelog.md`, `docs/research/00_research_brief.md` written; `docs/problem_statement.md` raised to v1.0 with citations.
- Wedge recorded as **Proposed**: bridge elements first (D-001), solar PV thermal second (D-002), disaster surge mode on the same engine (D-003). Needs team confirmation Fri 09:00.
- ~22:45 to 23:40 Phase 1 and 2 build. Environment: Python 3.13 venv, Ollama installed via winget, `qwen3-vl:4b` pulled (3.3 GB), GPU RTX 4060 8 GB. Data: Corrosion Condition State (440 images, CC0, LabelMe polygons with Fair/Poor/Severe labels; test split has 4 Fair, 28 Poor, 12 Severe and no all-clean image), InfraredSolarModules test parquet (4,000 images, 12 classes, MIT), dacl10k validation index (975 images) from the Voxel51 mirror, RescueNet building-damage split from the USF-IAE mirrors (three classes Intact/Damaged/Collapsed, not four; images 4000x3000; original license CC BY-NC-ND per the dataset README). Code: `src/cascade/` package, `eval/` scripts, 17 tests passing.
- **[Team-measured, single image each, not a benchmark]** Local gate on a 1072x803 corrosion image: correct (damage, confidence 0.95), 95 s cold start, 5.9 s warm. Claude grader (`claude-opus-5`) on a 1600x1200 corrosion image: returned Fair / S1 with two verbatim criteria, `percent_area_rusted` 3.0, `section_loss_pct` null with `not_measurable`, confidence 0.72; 6,016 input tokens, 921 output tokens, $0.053, 23.2 s. Dataset truth for that image (train/100) is Poor, so the model was one grade under on a picture where the annotated corrosion covers under 1 percent of the frame. One image proves the plumbing, not the accuracy.

### 2026-09-25 (Friday, early morning)

- ~00:00 to 00:50 Code completed to the PRD's P0 and P1 scope without running any model: `src/cascade/pipeline.py` (shared resumable loop with live progress), `exemplars.py` (FR-13), `surge.py` (FR-20), bridge entry CSV (FR-17), `app/streamlit_app.py` (FR-21), `scripts/make_demo_manifests.py`, `README.md`. Environment switched to the conda env `origin_hack` (Python 3.13.15) with `pip install -e .[dev,ui]`; the old `.venv` is unused. 32 tests pass; both CLIs and the app load headless.
- **[Team-measured, gate only, dev set, n=50]** `runs/dev_gate02` scored with `eval/run_eval.py` (`eval/reports/dev_gate02.md`): local gate (`qwen3-vl:4b-instruct`) recall 78.9 percent (95 percent CI 65.9 to 90.5), precision 81.1 percent, routing fraction 74 percent, 0 marked unusable, 304 s for 50 images, $0. Per dataset: corrosion 15/15 recall 1.00, dacl10k recall 1.00 at precision 0.71, RescueNet recall 1.00 at precision 0.63, **InfraredSolarModules recall 0.00: none of the 12 thermal 24x40 px crops was routed**, all 8 damaged ones missed. The gate as prompted does not recognise thermal module anomalies. Options to test on dev only (D-007): a thermal-specific gate prompt, forced routing for `pv_module`, or the Claude gate for thermal. Nothing tuned yet.
- First push to `github.com/AthArvA-188/OriginXnovi` (repo was empty) on the team lead's go-ahead.

### 2026-09-25 (Friday, morning)

- ~10:00 Phase 3 started: gate tuning on dev only (D-007). Added `RunConfig.force_route_classes` (default `("pv_module",)`): images of a listed asset class go to the grader regardless of the gate verdict; the gate still runs and its verdict is kept in `gate.jsonl`, only `routed` is overridden and the reason is suffixed `[forced: asset class ...]`. Exposed as `--force-route` on the CLI (`''` disables) and a sidebar checkbox in the app. This addresses the 0/12 thermal recall in `dev_gate02` by construction, not by prompt change; the cost is that every PV crop is graded. 34 tests pass. No model run yet; the eval_v1 grading run and `eval/reports/eval_v1_run1.md` are still pending the 09:00 sync decision on grader spend.

- ~10:15 to 10:40 UI phase 3b. `app/streamlit_app.py` rebuilt around six top-level tabs: Inspect run (image grid with gate verdict and worst-level badge per image, level legend, findings with evidence crops, queue with a visual top-10), Drop & grade (drag-and-drop inference, one stored run per drop, per-image cards), Batch (several datasets sequentially, one resumable run each, live status table), Reports (per-run `report.md` and `report.json` stored under `runs/<name>/`, comparison charts of levels and cost, visual top findings), Eval matrix (gate 2x2 and per-asset-class confusion heatmaps with exact, within-one, QWK, U rate, missed-image gallery), Why this approach (sourced incumbent comparison from R01 to R09 with a live measured column). New modules `src/cascade/evalmetrics.py` (same ordinal maps as `eval/run_eval.py`, plus MBEI CS for bridge elements) and `src/cascade/report.py`; 7 new tests, 41 pass. Verified in Chrome against the stored run `ui_0925_0856` (Claude gate + Opus grader, 10 dacl10k images, $0.82): gate recall 90 percent on n=10, no grade truth for dacl10k so the grading matrix is empty by design.

- ~10:45 to 11:10 UI polish and FR-19. `scripts/make_demo_manifests.py --mixed 3` writes `data/demo/mixed/manifest.jsonl` (12 rows, round-robin over the four datasets); it is the default dataset and the landing page groups the preview by asset class with count chips. CSS animations (fade-in on images and metrics, hover lift, pulsing S4 badge), a progress bar with stage and cost during runs, toasts on run end and reviewer actions. FR-19 done: `ReviewLog.timeline()` and a running model-vs-reviewer agreement chart in Findings & review. Eval matrix gained a "Score with bootstrap CIs" button (runs `eval/run_eval.py` on the run's manifest, no model calls) and the scored-report viewer. 60-second demo script (plan section 6) shown on the landing page. 42 tests pass. Remaining plan items that are not code: eval_v1 grading run (spend decision), team grading of dacl10k (kappa), interviews, deck, video, Devpost.

- ~11:10 Architecture tab added to the app (animated data-flow diagram with live counts, plus the startup-level flow). Committed locally, not pushed on the team lead's instruction.

- ~11:30 to 13:10 Phase 4, built with a design panel (three drift designs from MLOps, bridge-QA and SPC lenses, two judges, one synthesis), four parallel implementers, three reviewers, refutation of 29 findings (26 confirmed and fixed) and a fix pass. Delivered: `src/cascade/video.py` (FR-4: ffmpeg frame sampling at a fixed interval or on scene changes, PIL dHash near-duplicate suppression, provenance `source_video` / `frame_time_s`, CLI; verified on `data/demo/video/bridge_walkthrough.mp4`, a 12 s clip built from 6 dataset stills: 6 extracted, 6 kept, 0 dropped); `src/cascade/clientreport.py` (one self-contained HTML plus markdown per client, inline SVG charts, embedded evidence thumbnails with bbox, 5 MB budget; `ui_0925_0856` renders 14 of 14 thumbnails at 0.64 MB); `src/cascade/workforce.py` (review desk time with and without the cascade, every figure tagged Measured / Public figure / Assumption; sources: T&D World 3 to 5 min per image, AEP 500+ h for 400k to 500k images; sliders and a tornado; fixed disclaimer: image review only, no field-visit or accuracy claim); `src/cascade/drift.py` + `drift_thresholds.json` + `canary.py` (run fingerprint over rubrics, prompts, model ids, thresholds and exemplar ids; contract audit H1 to H7 against the rubric; U-rate p-chart with cause split; gate health incl. dead gate and template collapse; reviewer loop with blind QC rows and a critical-miss rule; ops and input drift; promotion checklist, eval ledger and `eval_guard` so eval_v1 is scored once per fingerprint; canary refuses to spend without `--confirm`). `ImageRecord` gained `client_id`, `asset_id`, `source_video`, `frame_time_s`. App: video in both uploaders with a Video options expander and a frame strip, Client reports tab, Workforce impact panel on Overview plus a row in Why this approach, Model health tab, demo video button (ingest only; grading is a second, labelled click). 123 tests pass. Measured on `ui_0925_0856` (n=10 dacl10k, labels present): gate recall 0.90, hard contract violations 0, 4 U findings all `model_u`, health status candidate (no baseline yet). One accidental spend during my UI test: one local-gate call and one Opus grade ($0.054) on a demo frame before I stopped the run and deleted that folder.

### 2026-09-25 (Friday, afternoon)

- After 16:22 Two whiteboard photos from the team room (per-person map: Atharva underwater, seismic, sensors, thermal + lidar; Leena buildings and bridges, aerial images only, drone; Jie interior machinery; Runze software; Drew a thermal item; box "OUR PROD: SW + HW") and the team lead's instruction re-scoped the product to a multi-sensor structural-health platform with a sensor kit. Research note R10 (`docs/research/10_multisensor_scope.md`, 355 lines, every number tagged) written; D-014 recorded as Proposed with D-001 and D-002 superseded in part; `docs/problem_statement.md` v1.1, `docs/PRD.md` v1.1 (section 5A, FR-22 to FR-26, contract v0.1 with measurement basis), `README.md` and this log updated. Facts kept honest: `eval_v1` carries no scale, so crack width stays `not_measurable` there; no public sonar set of bridge substructure with condition labels; public seismic sets are licence-gated; the whiteboard's "Reconstruction" and "16%" have no readable context and are not used. No code, data or test changed by the docs pass.

- ~17:00 to 18:30 Scope review of D-014 by five critics (hackathon judge, DOT bridge and underwater inspection manager, hardware seed investor, SHM and sonar engineer, sprint operator) plus a defender and a prosecutor. Verdict: keep the multi-sensor engine, narrow the pitch to bridge inspection with measured crack width, show seismic as one honest second modality that returns U without a baseline, put sonar, lidar, machinery and the kit on one roadmap slide. Odds are critic judgment only: hackathon 26 percent as written, 38 percent with the fixes; paying pilot in 12 months 8 and 15 percent. Review of the new modules confirmed 21 defects, all fixed (seismic no-baseline now U; underwater rubric U row; crack width carries uncertainty and basis; grader no longer sees measurement fields). UI: Measure crack panel, Sensors tab with a labelled synthetic signal, modality lane in Architecture. D-014 now records the whiteboard box as two options, SW or SW+HW. 198 tests pass. Review page published as a private artifact.

## Blockers

- None technical yet. The team roster, owners per workstream and the registration form status are unknown to this log.

## Repository

- Upstream: `https://github.com/AthArvA-188/OriginXnovi.git`. The folder is not yet a git repository; nothing has been committed or pushed. Push only when the team lead says so.

## Gaps (numbers or facts wanted but not in the research)

- Team-measured accuracy, cost and latency: none yet, by definition.
- Willingness-to-pay evidence from bridge consultants: none yet; interview plan in R08 section 5.
- RescueNet license text: verify inside the download (sources conflict per R07).
- Exact Claude structured-output parameter names: confirm from the `claude-api` skill before coding.

## Next up

1. Confirm registration form submitted; create Devpost accounts.
2. Friday 09:00 sync: confirm D-001, D-002, D-008; assign owners; decide whether the eval run uses the Claude grader (about $0.05 per image on Opus 5, roughly $10 for eval_v1 without tiles) or the local 8B grader only.
3. Run the local gate over the dev set, tune the routing threshold on dev only, then run gate plus grader on eval_v1 and produce `eval/reports/eval_v1_run1.md`.
4. Workstream C (Streamlit UI) and D (outreach) per the Friday table in `docs/implementation_plan.md`.
