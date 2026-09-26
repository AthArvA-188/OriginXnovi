# Product Requirements Document

| Field | Value |
|---|---|
| Project | Origin Weekend Fall 2026, Prompt D (Infrastructure & Resilience) |
| Version | 1.1 (multi-sensor scope per D-014: section 5A, FR-22 to FR-26, measurement basis in the contract; FR-1 to FR-21 unchanged; v1 roadmap in section 12) |
| Date | 2026-09-25 |
| Owner | Founding team |
| Status | Draft for team review; depends on D-001, D-002, D-008 and D-014 in `docs/decisions.md` |

**Citation key.** R01..R10 = notes in `docs/research/`; R10 is `10_multisensor_scope.md`. Labels: [Sourced: Rxx], [Inference], [Assumption], [Team-measured]. See `docs/decisions.md` D-011. Numbers new in v1.1 carry R10's tags: **[PUBLIC: url]** read on the cited page, **[PUBLIC, secondary: url]** seen only in a search summary or third-party page (verify before a slide), or **[Assumption]**. Section 5A is lettered rather than numbered so that the section 7 and 8 references in `src/cascade/prioritize.py`, `docs/implementation_plan.md` and `docs/changelog.md` stay valid.

## 1. Summary

A multi-sensor structural-health platform on one grading engine: software plus a sensor kit (D-014). Users drop in image frames from any drone, ROV, handheld camera or thermal sensor, sonar frames with their range metadata, or seismic and vibration time series from ground and structure sensors. Every record carries a `modality`. Image frames go through the cascade: a small local vision-language model (VLM) gates every image for "damage present and image usable"; flagged images are cropped and graded by a heavy VLM on the customer's own industry scale, with the verbatim criterion matched, a unified S0 to S4 severity, a confidence and an evidence crop. When a scale reference is in the frame or the ground sample distance is known, crack width is measured in millimetres with an uncertainty and its basis recorded; otherwise the finding is `not_measurable`. Sonar frames use the same cascade with an extent-and-geometry rubric. Seismic and vibration series are reduced to indicators in plain code and graded on sourced rubric rows against a stored baseline, U when there is none. Findings roll into one prioritized work queue that a qualified inspector reviews, overrides and exports. The same pipeline runs in surge mode after a disaster. Lidar is roadmap.

The hackathon MVP proves the imagery path on real public data for bridge elements (primary), PV thermal modules (secondary) and post-disaster UAV imagery (surge mode), with measured accuracy on a frozen held-out set. The measurement, seismic and sonar paths ship as minimal but real code with sourced rubric rows and labelled sample data, without an accuracy claim.

## 2. Problem

See `docs/problem_statement.md`. In one sentence: capture is cheap and abundant, but grading is slow, inconsistent, rarely carries the number the standard asks for, and is never ranked by consequence, so failures are missed and repairs are reactive.

Evidence the MVP must answer to [Sourced: R04, R08]:

- AEP Ohio's 2025 drone pilot covered about 4 percent of its distribution system, produced 400,000 to 500,000 images, and one person spent over 500 hours reviewing them.
- FHWA found only 68 percent of bridge condition ratings fell within one point of the mean across 49 inspectors; a 2026 Indiana study found 30 percent of ratings matched the expected value.
- NTSB attributed the Fern Hollow Bridge collapse to failure to act on repeated inspection recommendations: a prioritization failure, not a detection failure.
- Solar equipment-driven power loss rose from 2.36 percent in 2021 to 5.08 percent in 2025, up to about $5,070 per MW per year.

Evidence the v1.1 scope must answer to [Sourced: R10]:

- The dacl10k paper: "Damage size is measured with a pocket rule, thus, it's imprecise"; the German minimum crack width to document is 0.2 mm [PUBLIC: https://ar5iv.labs.arxiv.org/html/2309.00460].
- FHWA NBIS Q&A (updated 2024-04-05): imaging technology "can supplement" Level I underwater inspection; "The Level II portion of the UWI is still to be performed by an underwater bridge inspection diver" [PUBLIC: https://www.fhwa.dot.gov/bridge/nbis2022/qanda/08.cfm].
- Modal-frequency drop has no published universal threshold; 5 to 10 percent seasonal swings can be normal [PUBLIC, secondary: https://arxiv.org/pdf/2010.07026].

## 3. Goals and non-goals

**Goals (MVP, by Sun 2026-09-27 23:59 PT)**

1. Grade real inspection imagery on native industry scales for three asset classes with no per-class training.
2. Show the cascade working end to end with live per-image cost and latency.
3. Report measured triage and grading accuracy on `eval_v1` with confidence intervals.
4. Produce a prioritized, exportable work queue with a human review step.
5. (v1.1) Carry a `modality` on every record; measure crack width in mm with uncertainty and basis when a scale exists; run a minimal seismic path and a sonar asset class with sourced rubric rows and U when there is no baseline, with no accuracy claim.

**Non-goals (MVP)**

- Flight planning, autonomy; buying, building or testing hardware (the kit is a bill of materials).
- Fine-tuned models (D-013).
- Integrations beyond CSV and JSON export.
- Any claim of field accuracy; any accuracy figure for crack width, sonar or seismic.
- Any crack width on `eval_v1` (no scale exists there).
- Lidar processing (FR-26, roadmap).

## 4. Users and personas [Sourced: R08 section 4; R02; R09; R10 for the three v1.1 rows]

| Persona | Organization | Job to be done | What they judge us on |
|---|---|---|---|
| Bridge inspection engineer (primary) | Consulting engineering firm under contract to a DOT or county; county bridge engineer | Enter element-level condition states into the state system within deadline, with CS3/CS4 photo documentation, crack widths in inches or mm where the CS turns on width, and a per-structure work list | Fewer hours per bridge, defensible ratings, integration with InspectX / AASHTOWare BrM, liability stays with the inspector |
| Drone service provider owner or chief pilot | Small or mid DSP flying bridges, solar, poles | Deliver "AI-ready, utility-grade" reports to win and keep contracts | Turnaround, price per asset, resellable output |
| Solar O&M lead (secondary) | Regional solar owner or independent O&M firm | Turn a thermal survey into an IEC-classed anomaly list and repair plan | False alarms (there is little patience for even small numbers), production loss quantified |
| Emergency or resilience coordinator (surge mode) | County, utility storm team, insurer CAT team | Triage thousands of post-event images to FEMA classes in hours | Speed, confidence flags, export in the formats responders use |
| Underwater inspection program manager or diving contractor (v1.1, sonar) | DOT underwater program, diving inspection firm, port authority | Level I screening and change detection from sonar frames between the 60-month diver inspections; flag scour, undermining, exposed footings, debris | Never claims to replace the Level II diver; extent classes match SNBI water items; range and frequency shown next to every frame |
| Structural engineer or facilities manager in a seismic region (v1.1, seismic) | Building owner, bridge owner, campus or hospital facilities | After an event, site shaking class and drift or frequency indicators against a stored baseline within hours; between events, a baseline that is actually kept | Says U when there is no baseline; every threshold cites its source or says "team assumption"; no universal frequency-drop rule |
| Plant maintenance lead (v1.1, interior machinery) | Industrial operator | Machinery vibration severity on the ISO 20816-3 zone the plant already uses, with the kW class and support type it depends on | Zone boundaries match the standard's table for the machine group; metadata missing means U, not Zone A |

## 5. Competitive position [Sourced: R01, R02, R03; R10 for structural monitoring]

- Capture is owned by Skydio ($4.4B valuation), Zeitview, Percepto, ROV makers and Gecko Robotics. We do not compete on capture; we ingest anyone's imagery and sensor data.
- Analytics incumbents are single-vertical and closed-vocabulary: SkySpecs and Clobotics on blades, Raptor Maps and Sitemark on solar, Buzz Solutions and Sharper Shape on grid. None found publishes grading definitions or the rationale per finding, none spans asset classes in one engine, and all are quote-only enterprise pricing except Scopito.
- Bridges have no dominant AI-grading vendor. Skydio's DOT case studies stop at 3D twins.
- Disaster triage products (ICEYE, Nearmap, Vexcel) serve insurers with building-level classes from overhead imagery; none grades engineered infrastructure and overhead imagery under-reports damage by at least 20 percent versus drones.
- Structural-monitoring platforms (Move Solutions, Worldsensing, Resensys, Bentley iTwin IoT, ex-sensemetrics) are quote-only and single-sensor; none publishes a price or a grading rule [Sourced: R10 section 6]. Imaging sonar vendors (Sound Metrics ARIS, Blueprint Oculus, Teledyne BlueView) publish specifications but not prices [Sourced: R10 section 1.3].

**Our edge, in order of defensibility [Inference]:** (1) standards-cited grading with the clause next to each finding; (2) one engine across asset classes and modalities via the VLM cascade plus plain-code indicators for time series; (3) an audit trail of human decisions against model output; (4) transparent per-asset pricing for the long tail; (5) surge mode on the same engine; (6) a kit at cost-plus from public component prices with the grading subscription as the margin, the coverage-based pattern Gecko uses without Gecko's capex.

## 5A. Modalities (v1.1) [Sourced: R10 unless tagged]

| Modality | Sensor (public anchor) | Input format | What the engine does with it today | Rubric or indicator standard | Status |
|---|---|---|---|---|---|
| RGB imagery (aerial, handheld, video frames) | Any drone or camera; DJI Matrice 4T RGB plus 640 x 512 thermal, $7,199 [PUBLIC: https://www.dronenerds.com/products/dji-matrice-4t-thermal-enterprise-drone] | JPEG/PNG; MP4 via `cascade.video` (FR-4) | Full cascade: gate, crop, grade, prioritize, review, export | MBEI element CS, NBI 0 to 9, Corrosion CS, ISO 4628-3, FEMA PDA [Sourced: R05] | Demo; measured on `eval_v1` |
| Crack metrology (RGB with a scale reference) | Same camera plus a marker of known size, comparator card or ruler in frame; or camera model plus stand-off; or `gsd_mm_per_px` metadata | Image plus a scale hint `{marker_mm, marker_px}`, `{ruler_px_per_mm}` or `{focal_mm, sensor_w_mm, distance_m}` | `cascade.measure` (FR-23): width in px normal to the crack skeleton, mm via the scale, uncertainty, basis; grade to one CS only if the whole +/- band sits in one state | MBEI RC 0.012 / 0.05 in, PSC 0.004 / 0.009 in [PUBLIC: `src/cascade/rubrics/bridge_mbei.json`, R05]; ACI 224R-01 Table 4.1 widths 0.016 / 0.012 / 0.007 / 0.006 / 0.004 in by exposure [PUBLIC, secondary: https://onlinepubs.trb.org/onlinepubs/nchrp/nchrp_rpt_654AppendixA.pdf] | Demo on marker or ruler images; `not_measurable` on `eval_v1` |
| Thermal heatmap | Radiometric or 8-bit IR; Matrice 4T thermal core | 8-bit PNG/JPEG plus temperature statistics as text (FR-3) | Cascade with forced routing for `pv_module`; IEC CoA grading | IEC TS 62446-3 [Sourced: R05]; ASTM D4788-03(2022) deck delamination, overlays to 4 in, "should not be used for acceptance or rejection" [PUBLIC: https://store.astm.org/d4788-03r22.html]; ISO 6781-1:2023 building envelopes [PUBLIC: https://www.iso.org/standard/79848.html] | PV demo; deck and envelope rubrics roadmap (screening only, never S3 or S4 from thermal alone) |
| Sonar (underwater) | Sound Metrics ARIS Explorer 3000: 3.0 MHz identification to 5 m, 1.8 MHz detection to 15 m, 128 beams, 3 to 19 mm downrange resolution [PUBLIC: https://mfe-is.com/offshore/aris-explorer-3000/]; price not public | Rendered sonar frame (PNG) plus range, frequency and window as text | Cascade as asset class `underwater_sonar` (FR-25) with an extent-and-geometry rubric; crack-width rows excluded | SNBI B.C.11 Scour Condition 0 to 9, critical finding at 3 or below [PUBLIC, secondary: https://www.geocadra.com/en/standards/fhwa-nbis-snbi]; NYC EDC underwater grades [Sourced: R05 section 5.2]; FHWA Q&A: Level I supplement only [PUBLIC: https://www.fhwa.dot.gov/bridge/nbis2022/qanda/08.cfm] | Minimal real path; marine-debris or pipeline frames, or synthetic frames, labelled |
| Seismic and vibration series | Raspberry Shake RS4D $784.99 turnkey / $604.99 DIY [PUBLIC: https://raspberryshake.org/pricing]; ADXL355 25 ug/rtHz [PUBLIC: https://www.eetasia.com/3-axis-mems-accelerometers-detect-structural-defects/]; SM-24 geophone $59.95 [PUBLIC: https://www.spikenzielabs.com/Catalog/sensors/envirnomental/geophone-sm-24] | CSV or JSON time series (time, channels, units, sample rate) plus an optional stored baseline | `cascade.signals` (FR-24): PGA %g, PGV cm/s, RMS velocity mm/s over 10 to 1000 Hz, drift ratio when displacement is available, dominant frequency and its shift versus baseline; graded on rubric rows in plain code; U when no baseline | ShakeMap instrumental intensity PGA/PGV [PUBLIC, secondary: https://www.intensitylab.com/mmi-scale/]; Hazus drift ratios by building type [PUBLIC, secondary: https://www.fema.gov/sites/default/files/documents/fema_hazus-earthquake-model-technical-manual-6-1.pdf]; ISO 20816-3:2022 zones [PUBLIC, secondary: https://vibromera.eu/glossary/iso-20816-3/]; frequency-drop percentage [Assumption, per asset, in the rubric JSON] | Minimal real path; synthetic series or a public record labelled as such |
| Lidar and point clouds | Terrestrial laser scanner: Trimble TX8 +/-2 mm declared [PUBLIC: https://pmc.ncbi.nlm.nih.gov/articles/PMC7215276/]; scanners $100,000 to $500,000, scan services $150 to $500 per hour [PUBLIC: https://iscano.com/laser-scanning-lidar-best-practices/3d-laser-scanning-cost-guide-2025/] | LAS/PLY | Nothing today | Deflection under load and section-loss differencing against a baseline scan | Roadmap (FR-26) |

Interior machinery (Jie's stream) is served by two rows: RGB imagery for visible defects and the seismic and vibration row for ISO 20816-3 zones.

## 6. Functional requirements

Priority: P0 must ship for the demo; P1 ship if on schedule; P2 roadmap.

### 6.1 Ingest

| ID | Requirement | Priority | Acceptance |
|---|---|---|---|
| FR-1 | Accept a folder or upload of JPEG/PNG images; record filename, hash, dimensions, EXIF where present | P0 | 100 images ingested to a run manifest without error |
| FR-2 | Attach optional per-image metadata: asset id, asset class, GSD (mm per px), irradiance (W/m2), capture date | P0 | Missing values stored as null, never defaulted |
| FR-3 | Normalize 8-bit thermal images to a fixed grayscale plus pseudo-color and pass temperature statistics as text when radiometric data exists | P1 | Verified on one radiometric sample or explicitly skipped for the demo |
| FR-4 | Video frame extraction at a fixed interval | P2 | |

### 6.2 Gate (stage A)

| ID | Requirement | Priority | Acceptance |
|---|---|---|---|
| FR-5 | For every image return `{damage_present: bool, usable: bool, confidence: float, reason: string}` from a local small VLM with a JSON schema | P0 | Runs on the IR-solar dev set; output validates against schema 100 percent |
| FR-6 | Recall-first threshold tuned on the dev set only; log the fraction routed to the heavy stage | P0 | Threshold and routing fraction printed in the eval report |
| FR-7 | Cloud fallback (Haiku 4.5) selectable by flag | P1 | Same schema, same tests |

### 6.3 Crop (stage B)

| ID | Requirement | Priority | Acceptance |
|---|---|---|---|
| FR-8 | Tile images above 1,568 px on the long side into overlapping tiles; carry tile coordinates | P0 | dacl10k images (avg 1,950x1,581) tile without loss of coverage |
| FR-9 | Zero-shot detector (Grounding DINO or OWLv2) proposes boxes drawn onto crops for the grader | P1 | Boxes rendered; grader accuracy compared with and without on dev set |

### 6.4 Grade (stage C)

| ID | Requirement | Priority | Acceptance |
|---|---|---|---|
| FR-10 | Per flagged crop, return the finding contract in the appendix, schema-enforced | P0 | 100 percent schema-valid on eval_v1 |
| FR-11 | Rubric files per asset class containing the standard's own table rows and thresholds: MBEI element condition states and NBI 0 to 9 for bridges; IEC TS 62446-3 Annex C rows for PV; FEMA PDA matrix for disaster | P0 | Rubrics reviewed against R05 by a second teammate |
| FR-12 | `criteria_matched` must quote the rubric row verbatim; if no measurement supports a threshold, the finding is U or carries `not_measurable` | P0 | Spot check 20 findings |
| FR-13 | Few-shot exemplars (2 to 5 per class) retrieved from the dev set and included in the prompt | P1 | Toggle on/off compared on dev set |
| FR-14 | Second grader (local Qwen3-VL-8B) for disagreement-based escalation | P2 | |

### 6.5 Prioritize (stage D)

| ID | Requirement | Priority | Acceptance |
|---|---|---|---|
| FR-15 | Deterministic queue score (section 8) with S4 always at the top and same-day action | P0 | Unit test: any S4 outranks any S3 regardless of criticality |
| FR-16 | Per-asset-class consequence lever exposed in the score: production loss for PV, load posting or closure risk for bridges, occupancy for buildings | P1 | Visible in the queue table |
| FR-17 | Export queue as CSV and JSON; bridge export includes MBEI element, CS and quantity columns ready for SNBI-style entry | P0 | File opens in a spreadsheet; columns documented |

### 6.6 Review (stage E)

| ID | Requirement | Priority | Acceptance |
|---|---|---|---|
| FR-18 | Reviewer can accept, override grade, or mark U per finding; every action logged with timestamp and prior model value | P0 (table form) / P1 (full UI) | Review log persists across restarts |
| FR-19 | Show model-versus-reviewer agreement over time as the audit metric | P2 | |

### 6.7 Surge mode

| ID | Requirement | Priority | Acceptance |
|---|---|---|---|
| FR-20 | Batch run over a folder with FEMA PDA rubric, output a map-free ranked list and counts per class with confidence and U counts | P0 | Runs on 40 RescueNet tiles in the eval; counts shown |

### 6.8 Demo UI

| ID | Requirement | Priority | Acceptance |
|---|---|---|---|
| FR-21 | Single-page app: choose dataset or upload, run, watch stage counters, per-image cost and latency, findings table with crops, queue, export, review buttons | P0 | 60-second walkthrough recorded without errors |

### 6.9 Modalities (v1.1, D-014)

| ID | Requirement | Priority | Acceptance |
|---|---|---|---|
| FR-22 | Every `ImageRecord` and every `Finding` carries `modality` in `{rgb, thermal, sonar, seismic, lidar}`, default `rgb`; the queue and findings exports carry the column; existing manifests and runs load unchanged | P0 | All existing tests pass with the default; a manifest row with `modality: "sonar"` round-trips through ingest and export |
| FR-23 | Crack width in mm from an in-image scale (a marker of known size in mm, or ruler graduations) or from GSD (camera model plus measured stand-off, or `gsd_mm_per_px` metadata), via `python -m cascade.measure`. Output `{crack_width_mm, width_px, gsd_mm_per_px, uncertainty_mm, scale_source, not_measurable}`; width is taken normal to the crack skeleton; the basis is recorded on the finding (appendix A); when the +/- band straddles an MBEI boundary both candidate states are reported and the finding is not graded to a single CS; with no scale the finding carries `not_measurable` and is never S0; never run on `eval_v1` | P0 (module and tests) / P1 (UI) | Unit tests on synthetic images with a known scale recover the drawn width within the declared uncertainty; a run without scale yields `not_measurable`; the drift check H6 (width without GSD) never fires on the module's output |
| FR-24 | Seismic readings via `python -m cascade.signals`: ingest a time-series file (sample rate, channels, units), compute indicators (PGA %g, PGV cm/s, RMS velocity mm/s over 10 to 1000 Hz, dominant frequency; drift ratio when displacement is available), compute the frequency shift versus a stored baseline, grade on rubric rows that cite ShakeMap, Hazus, ISO 20816-3 or `source: "team assumption"`, emit U when there is no baseline, and write findings into the same queue and exports | P0 (module and tests) / P1 (UI) | Tests on synthetic sine-plus-noise series: indicators computed within tolerance; shift recovered when a baseline is given; U emitted without one; every rubric row has a non-empty `source` |
| FR-25 | Sonar imagery as asset class `underwater_sonar` through the cascade: range, frequency and window passed as text beside the frame; rubric `underwater_sonar.json` with extent-and-geometry rows (scour hole, undermining, exposed footing, debris, missing pile, gross section loss) mapped to SNBI B.C.11 style codes and S-levels; no crack-width rows; positioned as a Level I supplement, never a diver replacement | P1 | Rubric rows reviewed against R10 sections 1.2 to 1.3; one labelled sample frame runs end to end with the fake backends in tests |
| FR-26 | Lidar point clouds: deflection under load and section-loss differencing against a baseline scan | P2 (roadmap) | Slide only; no code this weekend |

## 7. Grading specification [Sourced: R05; v1.1 rows Sourced: R10]

The unified scale is defined by action semantics so it is comparable across industries.

| Level | Meaning | Queue action |
|---|---|---|
| S0 | No finding | Record; baseline for change detection |
| S1 | Cosmetic or minor | Log; monitor at routine cadence |
| S2 | Moderate | Schedule in next campaign |
| S3 | Major | Engineering review within weeks; consider derate, posting or restriction |
| S4 | Critical or safety | Same-day escalation; stop, isolate, close or red-tag |
| U | Not assessable | Re-image or other NDT; never default to S0 |

MVP native mappings:

| Native scale | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| Bridge NBI component 0 to 9 | 9, 8 | 7, 6 | 5 | 4, 3 (flag `load_posting_review` at 3) | 2, 1, 0 |
| Bridge MBEI element CS | CS1 | CS2 | CS3 | CS4 | CS4 with instability indicators |
| Steel coating ISO 4628-3 / ASTM D610 | Ri0 / 10 | Ri1 / 9 to 8 | Ri2 to Ri3 / 7 to 5 | Ri4 to Ri5 / 4 to 0, flag `section_loss` | only if section loss compromises structure |
| Corrosion Condition State dataset labels (these are MBEI CS1 to CS4 for steel corrosion) | Good (CS1) | Fair (CS2) | Poor (CS3) | Severe (CS4), flag `load_posting_review` | Severe with instability indicators |
| PV IEC TS 62446-3 CoA | CoA 1 | (none) | CoA 2 module or substring, dT 2 to 7 K | CoA 2 hot cell 10 to 40 K; string or inverter outage | CoA 3: over 40 K cell, broken glass, arc |
| Disaster FEMA PDA | undamaged | Affected | Minor | Major | Destroyed; Inaccessible maps to U |

The Corrosion Condition State row is our own mapping of the dataset's four labels onto S-levels and must be fixed before eval [Assumption]. The full multi-industry table (wind, hull, waterfront, weld RT, poles, insulators) is in R05 and ships as roadmap rubrics.

v1.1 native mappings for the sensor modalities. The thresholds are public; the placement of each band on S0 to S4 is our own reading of the standard's descriptors against the action semantics above and is tagged [Assumption] until a second teammate reviews it:

| Native scale | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| Underwater substructure, SNBI B.C.11 Scour Condition 0 to 9 (same ladder as NBI; critical finding at 3 or below [PUBLIC, secondary: https://www.geocadra.com/en/standards/fhwa-nbis-snbi]) | 9, 8 | 7, 6 | 5 | 4, 3 (flag `load_posting_review` at 3) | 2, 1, 0 |
| Site shaking, ShakeMap instrumental intensity by PGA in %g (bins 2.76 Light, 6.2 Moderate / very light damage, 11.5 Strong / light, 21.5 Very strong / moderate, 40.1 Severe / moderate-heavy) [PUBLIC, secondary: https://www.intensitylab.com/mmi-scale/] | below 6.2 | 6.2 to 11.5 | 11.5 to 21.5 | 21.5 to 40.1 | 40.1 and above |
| Building inter-storey drift ratio, Hazus (example W1 High-Code: Slight 0.004, Moderate 0.012, Extensive 0.040, Complete 0.100; values differ per building type and code level, so the rubric holds one row per type) [PUBLIC, secondary: https://www.fema.gov/sites/default/files/documents/fema_hazus-earthquake-model-technical-manual-6-1.pdf] | below Slight | Slight | Moderate | Extensive | Complete |
| Machinery vibration, ISO 20816-3:2022 broadband RMS velocity (example Group 2 rigid: A below 1.4, B 1.4 to 2.8, C 2.8 to 4.5, D above 4.5 mm/s; Zone C "Not suitable for continuous long-term operation", Zone D "Vibration severe enough to cause damage") [PUBLIC, secondary: https://vibromera.eu/glossary/iso-20816-3/] | Zone A | Zone B | (none) | Zone C | Zone D |
| Structure frequency shift versus stored baseline | within the per-asset tolerance | (none) | beyond tolerance, temperature-compensated | beyond tolerance and confirmed by a second indicator | (never from frequency alone) |

Frequency-shift tolerances are per asset and recorded in the rubric JSON with `source: "team assumption"`; no universal percentage exists in the literature we could read (a 5 percent model deviation is used as a flag in one 2025 paper while 5 to 10 percent seasonal swings can be normal) [PUBLIC, secondary: https://www.sciencedirect.com/science/article/pii/S2772991525000477 and https://arxiv.org/pdf/2010.07026]. With no stored baseline the finding is U. Machinery rows require the machine power class and support type; missing metadata is U, not Zone A.

Uncertainty is reported as +/-1 level following EPRI's guidance on blade categories. For measured crack widths the uncertainty is in millimetres and the grade is stated only when the whole band sits in one condition state; otherwise both candidate states are listed (FR-23).

## 8. Prioritization model [Assumption: weights are ours; the levers are sourced from R05]

```
score = severity_weight[S] * criticality * consequence * urgency
severity_weight = {S1: 1, S2: 3, S3: 9, S4: 27}; U is listed separately, never scored as S0
criticality   = 1 to 3, user-supplied per asset (e.g. daily traffic band, customers served, MW)
consequence   = per asset class: PV = estimated production loss; bridge = 2 if load_posting_review flag else 1; building = occupancy class
urgency       = 1 + (days since finding / 30), capped at 2
rule          = any S4 sorts above every non-S4 regardless of score and gets action "escalate", sla_days = 0
```

Weights and multipliers are printed in the UI so a reviewer can see why an item ranks where it does. Sensor-derived findings (FR-24, FR-25) enter the same score with the same levers; a finding from a modality with no baseline is U and listed separately.

## 9. Non-functional requirements

| Area | Requirement | Basis |
|---|---|---|
| Cost | Show measured dollars per image per stage in the UI. Expected order of magnitude: about $10 per 1,000 1080p frames heavy-only on Sonnet 5, about $3 with a local gate at 30 percent pass-through | [Sourced: R06] |
| Latency | Show measured median seconds per image per stage | [Team-measured] |
| Privacy | Gate stage runs locally so only flagged frames leave the machine; a fully local mode exists for customers who cannot use a public API; seismic indicators are computed locally with no model call | [Sourced: R06, R09] |
| Honesty | Every accuracy figure carries n and a CI; U and confidence are always visible | D-007 |
| Measurement basis | A crack width is always shown as width +/- uncertainty with its `scale_source`; a bare millimetre value never appears; a width without a basis is a contract violation (drift check H6) | D-004, [Sourced: R10 section 4.3] |
| Secondary thresholds | Rubric rows whose numbers come only from a search summary or a third-party page carry `secondary: true` and are verified against the primary document before a slide | D-011, [Sourced: R10] |
| Robustness | Resolution normalization and tiling because resolution changes caused 82.6 percent label flips in a field deployment | [Sourced: R03] |
| Licenses | Non-commercial datasets are demo-only and labeled on the data slide | [Sourced: R07, R10] |

## 10. Evaluation and success metrics

Protocol: `docs/decisions.md` D-007 and R07.

| Metric | Target for the demo [Assumption] | Why |
|---|---|---|
| Gate recall on `damage_present` | at least 0.95 on eval_v1 with CI shown | A miss is a skipped inspection |
| Gate routing fraction | reported, no target | Drives the cost story |
| Grader macro-F1 on classes present | reported, no target | Literature ceiling is modest: GPT-4o 74.9 percent on MMAD (R06) |
| Grading within-one-grade accuracy (corrosion, RescueNet) | reported, no target | EPRI's own +/-1 tolerance |
| Schema validity | 100 percent | Product contract |
| Cost and latency per image | reported | Unit economics |
| Crack width on synthetic scale images (v1.1) | recovered within the declared uncertainty in unit tests | Proves the plumbing, not field accuracy; published methods reach 0.16 to 0.22 mm with a scale reference [Sourced: R10 section 4.2] |
| Seismic and sonar accuracy (v1.1) | none claimed | No labelled public data of the right kind (R10 sections 1.4, 2.2); the deck says so |

Success for the hackathon means the numbers exist, are honest, and the story survives Q&A.

## 11. Data and licensing

See `docs/decisions.md` D-006. Commercial bootstrapping is limited to permissively licensed sets (Corrosion Condition State CC0, InfraredSolarModules MIT, SDNET2018 CC BY 4.0, COCO-Bridge CC0, TTPLA Apache-2.0) [Sourced: R07].

v1.1 sensor data [Sourced: R10 sections 1.4 and 2.2]: Marine Debris FLS Datasets (ARIS Explorer 3000 frames of debris in a tank and a quarry; CC BY-NC-SA 4.0 on Zenodo, CC0 stated on the arXiv page; treated as non-commercial) [PUBLIC: https://zenodo.org/records/15101686]; PEER NGA-West2 (21,336 records, institutional e-mail, about 200 downloads per two weeks) [PUBLIC: https://ngawest2.berkeley.edu/]; Z24 bridge benchmark (non-commercial research only) [PUBLIC: https://bwk.kuleuven.be/bwm/z24]; CESMD strong-motion records [PUBLIC: https://www.strongmotioncenter.org/]. None of these carries bridge-substructure condition labels or a per-image crack scale; the demo labels every sonar or seismic sample as public-non-commercial or synthetic.

## 12. Roadmap after the weekend [Inference]

1. Wind blades (EPRI five-category and IEA Task 46 erosion levels), poles and insulators, hull and waterfront, weld radiography: rubrics already mapped in R05.
2. Radiometric thermal and 16-bit radiograph preprocessing (R06); bridge-deck delamination (ASTM D4788, screening only) and building-envelope (ISO 6781-1) thermal rubrics (R10 section 3).
3. Lidar geometry layer (FR-26): deflection and section-loss differencing; needs a point-cloud stack not in the current environment (R10 section 5).
4. Sonar change detection against a stored baseline frame set; customer-labelled substructure frames to replace debris sets.
5. Seismic baselines per asset with temperature compensation; verification of the ShakeMap, Hazus and ISO 20816-3 rows against the primary documents.
6. Multi-vendor grader abstraction and disagreement escalation.
7. Exports to InspectX and AASHTOWare BrM, CMMS.
8. Asset registry with baselines for change detection and true post-event triage; `exposure_class` (ACI 224R), `element_type` (RC or PSC), machine power class and support type on the asset record.
9. Compliance: SOC 2 roadmap, self-hosted option (R09).

## 13. Business model hypothesis

See `docs/decisions.md` D-008. Per-asset pricing, free gate-only tier, consultants and DSPs first, utilities and DOTs via programs and grants. v1.1 adds the kit: a bill of materials from public prices of about $16,300 (budget: RS4D DIY $604.99, Matrice 4T $7,199, Deep Trekker DTG3 $8,500) to about $38,800 (standard: RS4D turnkey $784.99, Matrice 4T $7,199, Blueye X3 $30,788 ex VAT), sonar excluded because no sonar price is public [Inference, R10 section 6]; kit at cost-plus, grading subscription as the margin. No subscription price goes on a slide until an interview supports it.

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Hallucinated defects or memorized answers regardless of image (R06) | Grounded crops, schema-enforced criteria quotes, U state, human review, report our own error rate |
| Mid-scale grades are the hardest (xBD Major F1 0.0094) (R05) | Show within-one-grade accuracy and confidence; never claim exact-match dominance |
| Thermal domain gap (R06) | Demo on 8-bit modules; radiometric handling is roadmap |
| Model availability and pricing: Haiku 4.5 may retire after 2026-10-15 (R06) | Vendor-agnostic schema and prompt template |
| Judges ask "why not focus?" | One beachhead (D-001), engine generality shown as next steps; sensor paths shown as the same contract, not new products (D-014) |
| Judges ask for customers | Honest funnel from the discovery plan in R08 section 5 |
| A crack width is read as more precise than it is (R10) | Width always with uncertainty and `scale_source`; both candidate states when the band spans a boundary; never on `eval_v1` |
| Sensor thresholds come from secondary sources (R10) | `secondary: true` on the rubric row; verified before a slide; frequency-drop rule labelled team assumption per asset |
| High U rate on sensor findings because baselines do not exist | Say so; U is the honest answer and the reason a customer buys the kit and keeps a baseline |
| No public sonar or seismic data with the right labels (R10) | Labelled sample data, no accuracy claim, customer-supplied data as the path to a measured number |

## 15. Open questions

1. D-001, D-002 and D-014 confirmation at the next team sync.
2. Willingness to pay from at least three interviews; whether an owner would buy a kit plus subscription.
3. Whether the zero-shot detector improves grader accuracy on the dev set enough to keep.
4. Which scale reference the drone team carries (comparator card, ArUco board, tape); the demo images must contain one (R10 section 9).
5. Frequency-drop tolerance per asset class and whether temperature compensation is in weekend scope (team assumption to be recorded in the rubric JSON).
6. Real seismic series (CESMD or NGA-West2 registration lead time; 2026-07-02 membership reset) versus a synthetic series labelled as such.
7. ACI 224R exposure class default per asset; ISO 20816-3 metadata (power class, support type) on the asset schema.
8. Sonar price points for the kit BOM; Level II cleaning percentage (FHWA-NHI-23-027); SNBI B.C.15 code text; the Hazus 6.1 table row for the demo building types.

## Appendix A. Finding contract (v0.1) [Sourced: R05; measurement basis per R10 section 10]

Every numeric measurement carries its basis: how the scale was obtained (`scale_source`), the pixel width and GSD it came from, the uncertainty in the measured unit and, when the +/- band spans a rubric boundary, the candidate states. Seismic findings carry their indicators and the baseline they were compared against. A width without a basis is a contract violation (drift check H6). `modality` defaults to `rgb` so v0 findings remain valid.

```json
{
  "modality": "rgb | thermal | sonar | seismic | lidar",
  "asset_class": "bridge_element | steel_coating | pv_module | building_disaster | underwater_sonar | structure_vibration | machinery_vibration",
  "defect_type": "<native taxonomy term>",
  "native_scale": {
    "standard": "NBI-0-9 | MBEI-CS | ISO-4628-3 | IEC-62446-3-CoA | FEMA-PDA | CorrosionCS | SNBI-BC11 | ShakeMap-PGA | Hazus-drift | ISO-20816-3 | team-assumption",
    "value": "<code>",
    "criteria_matched": ["<verbatim threshold matched>"],
    "candidate_values": ["<second code when the measurement band spans two states, else empty>"]
  },
  "unified": {
    "level": "S0|S1|S2|S3|S4|U",
    "uncertainty": "+/-1",
    "flags": ["fire_shock_pathway", "load_posting_review", "section_loss", "not_measurable", "no_baseline"]
  },
  "measurements": {
    "area_cm2": null, "crack_width_mm": null, "delta_t_k": null,
    "percent_area_rusted": null, "section_loss_pct": null, "confidence": 0.0,
    "basis": {
      "scale_source": "marker | ruler | camera_standoff | gsd_metadata | none",
      "width_px": null, "gsd_mm_per_px": null, "uncertainty_mm": null,
      "indicators": {"pga_pct_g": null, "pgv_cm_s": null, "drift_ratio": null, "rms_velocity_mm_s": null, "f_dominant_hz": null, "f_baseline_hz": null, "f_shift_pct": null},
      "baseline_id": null,
      "source": "<rubric row source, or 'team assumption'>"
    }
  },
  "action": {"code": "record|monitor|schedule|prioritize|escalate", "sla_days": null, "basis": "<standard clause>"},
  "evidence": {"image_ids": [], "bbox": [], "tile": null, "gsd_mm_per_px": null, "irradiance_wm2": null, "sonar_range_m": null, "sonar_frequency_mhz": null, "series_id": null},
  "review": {"status": "pending|accepted|overridden|marked_u", "reviewer": null, "reviewed_at": null, "prior_level": null}
}
```
