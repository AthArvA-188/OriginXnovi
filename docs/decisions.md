# Decision Log

| Field | Value |
|---|---|
| Project | Origin Weekend Fall 2026, Prompt D (Infrastructure & Resilience) |
| Version | 1.0 |
| Date | 2026-09-24 |
| Owner | Founding team |
| Status | Living document. Decisions marked **Proposed** need a team "yes" at the Friday 2026-09-25 09:00 sync; until then the plan proceeds on them by default. |

**Citation key.** R01..R09 = the nine notes in `docs/research/` (R01 wind/solar competitors, R02 bridges/power lines/telecom, R03 underwater/industrial/disaster, R04 market pain and regulation, R05 grading standards, R06 VLM landscape, R07 datasets, R08 customer voice, R09 business models); R10 = `10_multisensor_scope.md` (2026-09-25: sonar, seismic/vibration, thermal beyond PV, crack metrology, lidar, SW+HW pricing), whose numbers carry **[PUBLIC: url]**, **[PUBLIC, secondary: url]** or **[Inference]** tags. Every number below traces to one of them. Our own reasoning is tagged **[Inference]**; planning guesses are tagged **[Assumption]**.

## Index

| ID | Decision | Status |
|---|---|---|
| D-001 | Primary wedge: bridge elements (steel and concrete), graded to AASHTO/NBIS scales, sold first to inspection consultants and county owners | Proposed; superseded in part by D-014 |
| D-002 | Secondary asset class for the platform proof: solar PV thermal, graded to IEC TS 62446-3 | Proposed; superseded in part by D-014 |
| D-003 | Disaster surge mode as a feature of the same engine, demoed on UAV building-damage imagery | Accepted |
| D-004 | Grading contract: native scale first, unified S0 to S4 overlay, explicit U state, +/-1 uncertainty | Accepted |
| D-005 | Architecture: three-stage cascade (gate, crop, grade) with deterministic prioritization and mandatory human sign-off | Accepted |
| D-006 | Demo datasets and licenses | Accepted |
| D-007 | Evaluation protocol and honesty rules | Accepted |
| D-008 | Business model hypothesis: per-asset pricing, self-serve free triage tier, consultants and DSPs first | Proposed |
| D-009 | Weekend scope and cut lines | Accepted |
| D-010 | Tech stack | Accepted |
| D-011 | Documentation conventions and evidence labels | Accepted |
| D-012 | Positioning: AI-assisted, human-certified decision support, never the inspector of record | Accepted |
| D-013 | Do not fine-tune this weekend; use retrieval and few-shot with the standards' own tables | Accepted |
| D-014 | Scope widened to multi-sensor structural health, product is software plus a sensor kit | Proposed |

---

## D-001. Primary wedge: bridge elements, graded to AASHTO/NBIS scales

**Status:** Proposed (confirm Fri 2026-09-25 09:00). **Superseded in part by D-014 (2026-09-25):** bridge elements remain the primary imagery wedge and the demo lead, but the product is no longer imagery-only; see D-014 for the widened scope. **Date:** 2026-09-24.

**Context.** The problem statement lists seven segments. The hackathon rubric rewards measured demos, customer evidence and a credible first-100-customers story, and the rules forbid fabricated performance claims. We need one asset class where all three can be shown by Sunday.

**Options considered.**

| Option | For | Against |
|---|---|---|
| A. Bridge elements (steel corrosion condition state, concrete defects) | Most codified grading in any sector: NBI 0 to 9 component scale, MBEI element condition states CS1 to CS4 with numeric thresholds such as RC crack width under 0.012 in / 0.012 to 0.05 in / over 0.05 in (R05). The only public dataset with real inspector-assigned grades is a bridge set: Corrosion Condition State, 440 images, CC0, AASHTO Good/Fair/Poor/Severe (R07). No dominant AI-grading vendor; Skydio's own bridge case studies show no defect grading (R02). Demand is legally recurring: routine inspection at most every 24 months, and SNBI element-level data due 15 Mar 2028 (R02, R04). Inspector inconsistency is documented by FHWA: only 68 percent of ratings within one point of the mean (R08). RFPs ask for exactly our output: CS3/CS4 photo documentation and a per-structure work list (R08). LA-local discovery targets exist (R08 section 5). | Ratings must be assigned by qualified inspectors; drones cannot replace arm's-length inspection (R02, R08). State DOT sales cycles are long. IIJA funding expires 30 Sep 2026 (R04). |
| B. Solar PV thermal | IEC TS 62446-3 has numeric Class of Abnormality thresholds (R05). InfraredSolarModules is MIT-licensed with 20,000 images (R07). Solar DSPs and O&M firms have shorter sales cycles (R09). Pain is growing: 5.08 percent average power loss (R04). | Raptor Maps, Sitemark and Above already own this exact thermal-classification workflow (R01). No public solar set carries a severity label, so grading accuracy cannot be measured against ground truth (R07). |
| C. Wind blades | Highest repair-cost leverage: $30k minor repair versus $500k structural (R04). | Crowded and consolidating: SkySpecs, Zeitview, Clobotics, Cornis, Nearthlab (R01). Public blade data is small and non-commercially licensed (R07). |
| D. Utility poles and lines | Largest asset counts, 180M poles (R04). | Most mature AI-analytics segment: Buzz Solutions, Sharper Shape, Hepta (R02). Tier-1 utilities take about three years from trial to award (R09). |

**Decision.** Option A is the primary wedge. First customers are consulting engineering firms that perform NBIS inspections under contract and county or municipal bridge owners, not state DOTs directly (R02, R04). The product is positioned as inspector decision support that pre-fills MBEI condition states and SNBI fields with evidence, and the inspector of record signs off (see D-012).

**Rationale.** Option A is the only option where the weekend demo can report grading accuracy against real inspector labels, where the grading rules are public and numeric enough to prompt a model with, and where no incumbent already sells the grading layer. **[Inference]**

**Consequences.** The demo leads with steel corrosion condition states and concrete defect classes. Solar is the second class (D-002). Wind, underwater and weld radiography move to the roadmap slide, backed by the standards mapped in R05 rather than by demo data.

**Revisit trigger.** If two or more Friday interviews say bridge consultants would not pay for pre-filled condition states, swap to Option B as primary before Saturday noon.

---

## D-002. Secondary asset class: solar PV thermal

**Status:** Proposed. **Superseded in part by D-014 (2026-09-25):** PV thermal remains the secondary imagery class and the thermal demo; thermal as a modality widens to bridge-deck delamination and building envelopes as roadmap rubrics (PRD section 5A). **Date:** 2026-09-24.

**Decision.** Solar PV thermal modules graded to IEC TS 62446-3 Classes of Abnormality (CoA 1 no abnormality, CoA 2 thermal abnormality, CoA 3 safety-relevant) using the InfraredSolarModules dataset (20,000 images, 12 classes, MIT license, 50 percent No-Anomaly) (R05, R07).

**Rationale.** It proves the cross-asset engine and the thermal modality with a permissively licensed dataset. The 50 percent No-Anomaly share makes it the ideal set to measure the gate stage's false-negative rate (R07). Solar DSPs are the second-fastest channel after bridge consultants (R09). **[Inference]**

**Consequences.** The class-to-CoA mapping table must be written before any results are seen (R07 eval plan). Because the source has no severity label, we report classification accuracy, not grading accuracy, for solar.

---

## D-003. Disaster surge mode on the same engine

**Status:** Accepted. **Date:** 2026-09-24.

**Decision.** Disaster response is a mode of the same pipeline, not a separate product: gate every image, grade flagged ones to FEMA PDA classes (Affected / Minor / Major / Destroyed), rank by consequence. Demo on the RescueNet UAV test split (450 images, four building-damage levels) (R07). Stretch: one slide using CAL FIRE DINS Palisades/Eaton structure status (CC BY) with Vantor open imagery (R03, R07).

**Rationale.** Prompt D names disasters as the additional high-value application. FEMA's July 2025 PDA guide explicitly invites analytics on imagery (R03). Insurers already get building-level triage in 24 to 48 hours from ICEYE, Nearmap and Vexcel, but none grades engineered infrastructure (R03), so we frame it as complementary, not head-on.

**Consequences.** The mid-scale classes are where models fail (xBD baseline F1 for Major damage was 0.0094) (R05), so the demo must show confidence and the U state, not just a verdict.

---

## D-004. Grading contract

**Status:** Accepted. **Date:** 2026-09-24.

**Decision.** Every finding carries, in this order: the native scale and the verbatim criterion matched; the unified level S0 to S4 or U; a +/-1 uncertainty band; measurements when scale metadata allows; an action code with the standard clause as basis; and evidence (image id, crop, GSD, irradiance). The JSON contract in R05 is adopted verbatim as the v0 schema (see `docs/PRD.md` appendix).

**Rationale.** No incumbent publishes grading definitions or rationale per finding (R01, R02). EPRI documents that blade categories are disputed between owners, OEMs and service providers and recommends a one-category uncertainty (R05). Measurability must gate severity: IEC requires 600 W/m2 irradiance for a valid thermal grade and MBEI thresholds need crack width in inches (R05).

**Consequences.** When GSD or irradiance is missing, the pipeline must emit U or the `not_measurable` flag rather than guess.

---

## D-005. Architecture: three-stage cascade with deterministic prioritization

**Status:** Accepted. **Date:** 2026-09-24.

**Decision.**

| Stage | What | Default model | Fallback |
|---|---|---|---|
| A. Gate | "Damage present? Image usable?" on every image, recall-first threshold, JSON output | Qwen3-VL-4B via Ollama, Apache-2.0, 3.3 GB (R06) | Claude Haiku 4.5 via API (R06) |
| B. Crop | Fixed tiling for large images; zero-shot detector (Grounding DINO or OWLv2, both Apache-2.0) as stretch | Tiling | Whole image |
| C. Grade | Native scale + unified level + action + evidence, schema-enforced JSON, prompted with the standard's own tables and a few exemplars | Claude Sonnet 5 with structured outputs (R06) | Qwen3-VL-8B via Ollama for a fully local run (R06) |
| D. Prioritize | Deterministic score from severity, criticality and consequence; S4 always escalates | Plain code | none |
| E. Review | Human accept / override per finding, logged against the model's output | UI | none |

**Rationale.** The 2025-2026 literature converged on exactly this shape: a cheap pre-filter that cut VLM calls 240x in one deployed system, and a detector plus constrained model that produced a 4 percent hallucination rate versus 65 percent for a zero-shot VLM (R02, R06). Frontier VLMs are good at image-level "is there damage" and weak at localizing small defects (R06), so localization is not delegated to the grader. Prioritization in plain code keeps the ranking auditable. **[Inference]**

**Cost basis.** Heavy-only grading of 1,000 1080p frames on Sonnet 5 is about $10; a local gate passing 30 percent brings it to about $3; Batch API halves the Claude figures (R06). Weekend spend for a 180-image eval set, a 50-image dev set and demo runs is well under $20. **[Inference]** If the team prefers zero API spend, stage C runs on the local fallback at lower expected accuracy.

**Consequences.** One schema and one prompt template across vendors so a second grader can be swapped in; disagreement between graders becomes an escalation signal (R06).

**Amendment 2026-09-24 (implemented).** The grader's default model id follows the Claude API reference loaded in-session, `claude-opus-5` ($5 in / $25 out per million tokens), with `claude-sonnet-5` selectable through the `GRADER_MODEL` variable for cost. One measured call on a 1600x1200 corrosion image cost $0.053 and took 23 s (6,016 input tokens, of which about 2,350 are the image and the rest the rubric and instructions; 921 output tokens) **[Team-measured, n=1]**. The gate runs locally on Qwen3-VL-4B at about 6 s per image warm on an RTX 4060 laptop GPU **[Team-measured, n=1]**. A fully local grader (`qwen3-vl:8b`) is pulled as the zero-API fallback. The run CLI grades whole images downscaled to 1,568 px by default for cost; tiling is available with a flag.

---

## D-006. Demo datasets and licenses

**Status:** Accepted. **Date:** 2026-09-24. Source: R07.

| Dataset | Asset class | Size | Label | License | Use |
|---|---|---|---|---|---|
| Corrosion Condition State | Steel bridge elements | 440 images, 333 MB | AASHTO Good/Fair/Poor/Severe masks | CC0 | Flagship grading demo and eval (real inspector grades) |
| dacl10k, validation split only | Concrete bridge elements | 975 images of 9,920 | 19 classes, polygons, no grade | CC BY-NC 4.0 | Defect-class demo and eval; team-graded severity |
| InfraredSolarModules, HF parquet mirror | PV thermal | 20,000 images, 24x40 px | 12 classes incl. 10,000 No-Anomaly | MIT | Gate-stage eval, solar grading demo |
| RescueNet, test split via the USF-IAE building-damage mirrors on Hugging Face | Post-disaster UAV | 675 test scenes, 4000x3000 | 3 scene-level classes (Intact / Damaged / Collapsed), coarser than the paper's 4 building levels | Dataset README says CC BY-NC-ND 4.0; demo only | Disaster surge demo and eval at three levels |

Stretch only after the four above are done by Saturday noon: 2026 wind-blade set (1,065 images, CC BY-NC-ND), InsPLAD fault crops, CAL FIRE DINS plus Vantor LA-fire imagery.

**Consequences.** Non-commercial sets are demo-only and are named as such on the data slide. The startup story is "customers bring imagery, we bring rubrics, evaluation and workflow" (R07).

---

## D-007. Evaluation protocol and honesty rules

**Status:** Accepted. **Date:** 2026-09-24. Source: R07 held-out plan.

1. Freeze `eval_v1` of about 180 images from official test or validation splits (40 corrosion, 60 dacl10k, 40 IR solar, 40 RescueNet) with a `manifest.jsonl` of file hashes and labels before the first end-to-end run. Keep a separate 50-image dev set for prompt tuning. Never tune on eval.
2. Ground truth is derived by script where the source carries it. For dacl10k severity, two teammates grade independently with a written rubric, report Cohen's kappa, and mark those rows "team-graded".
3. Report per stage: gate precision, recall and F1 (recall is the headline); grader per-class precision, recall and macro-F1; grading exact-match, within-one-grade accuracy and quadratic-weighted kappa; measured latency and dollars per image.
4. Every accuracy number on a slide carries n and a 95 percent bootstrap confidence interval. At n of about 180 that is roughly +/-5 to +/-7 points; say so.
5. Never extrapolate to field performance. Public sets are not customer imagery. Literature numbers are quoted as feasibility evidence only.

---

## D-008. Business model hypothesis

**Status:** Proposed. **Date:** 2026-09-24.

**Decision.** Per-asset pricing with a self-serve free tier that runs only the gate stage. Anchors from public prices: Scopito charges EUR 160 per turbine analysed and EUR 80 self-analysed, EUR 20 per solar asset, EUR 1 per image for expert review; the analytics add-on for wind runs $100 to $300 per turbine; SCE's pole contract implies about $313 per pole; a large-bridge conventional inspection costs about $59,000 (R01, R04, R09). Proposed price points **[Inference, per R02]**: low hundreds of dollars per bridge element set, tens of dollars per pole or tower report, under the $100 per turbine analytics floor.

**Who pays first.** Consulting engineering firms doing NBIS inspections and drone service providers who already hold imagery and must deliver graded reports; regional solar owners and O&M firms second; Tier-1 utilities and state DOTs as later reference accounts entered through programs (EPRI Incubatenergy, CEC EPIC, USDOT DIIG with an LA-area agency partner before the 2026-11-15 close) rather than RFPs (R09).

**Traction calibration.** Buzz Solutions raised a $20M Series A on three utility logos and 400 percent revenue growth; Voliro reached $23M with 40 customers (R09). A believable 18-month goal is 30 to 100 consultant and DSP accounts plus one or two grant-backed public pilots (R09).

**Unit economics.** Gate inference costs fractions of a cent per image and heavy grading about a cent per full-resolution image, so gross margin on compute exceeds 95 percent; real costs are human QA, storage, support and acquisition (R09). **[Inference]**

**Revisit trigger.** Interview evidence on willingness to pay by Saturday.

---

## D-009. Weekend scope and cut lines

**Status:** Accepted. **Date:** 2026-09-24.

**In scope (must ship by Sun 2026-09-27 23:59 PT):** working cascade on the four datasets in D-006; grading in MBEI condition states, IEC CoA and FEMA PDA classes; prioritized work queue with CSV/JSON export; measured eval_v1 numbers; six-slide deck PDF; 30 to 60 second demo video; Devpost entry; at least the customer outreach funnel logged honestly.

**Cut lines, in order, if behind at the Saturday 12:00 checkpoint:** (1) drop the zero-shot detector, keep tiling; (2) drop RescueNet to a single pre-graded example; (3) drop the human-review UI to a table with accept/override buttons; (4) drop dacl10k team-grading, report classes only. Never cut: eval_v1 freeze, measured numbers, the deck.

**Out of scope:** flight operations, hardware, custom training, CMMS integrations beyond export, any accuracy claim not measured by us.

---

## D-010. Tech stack

**Status:** Accepted. **Date:** 2026-09-24. **[Assumption]** throughout; change freely if the team's skills differ.

| Layer | Choice | Why |
|---|---|---|
| Language and environment | Python 3.13 in a conda environment named `origin_hack` (`environment.yml`; team lead's instruction 2026-09-25: conda, not venv) | Fastest path for VLM APIs, image handling and metrics |
| Local models | Ollama (Qwen3-VL-4B gate, Qwen3-VL-8B fallback grader); structured JSON via Ollama `format` (R06) | Runs on a laptop; Apache-2.0 |
| Cloud grader | Anthropic Python SDK, Claude Sonnet 5, structured outputs with a JSON schema (R06). Load the `claude-api` skill for exact parameter names before writing this code. | Schema-guaranteed output |
| Schema | Pydantic models mirroring the R05 contract | Validation and export |
| Storage | JSONL per run plus SQLite for the review log | No server to run |
| UI | Streamlit single page: upload or pick a dataset folder, watch the cascade, review findings, export queue | Buildable in hours |
| Eval | Scripts under `eval/` producing a markdown report with bootstrap CIs | Feeds the deck directly |
| Hosting | Local laptop for the live pitch, recorded video as backup; optional Replit deployment of the UI using the event's one-month Core credit | Event guidance recommends screenshots or embedded video to avoid day-of failures |
| Thermal preprocessing | FlirImageExtractor (MIT) if radiometric files are used; InfraredSolarModules is already 8-bit so none needed for the demo (R06) | |

---

## D-011. Documentation conventions

**Status:** Accepted. **Date:** 2026-09-24.

- All planning docs live in `docs/`; research in `docs/research/`; the index is `docs/documentation.md`.
- Evidence labels: **[Sourced: Rxx]** for anything traceable to a research note; **[Team-measured]** for our own eval numbers; **[Inference]** for our reasoning from sources; **[Assumption]** for planning guesses.
- No number, quote or company fact appears in a doc or slide without one of those labels. If a number is not in the research, write "TBD" and log it in `docs/progress.md` under gaps.
- `problem.md` (the prompt as given) is never edited.
- Dates are absolute (YYYY-MM-DD) and times are Pacific.

---

## D-012. Positioning

**Status:** Accepted. **Date:** 2026-09-24.

**Decision.** "AI-assisted, human-certified grading for any inspection imagery." The product pre-fills the native form (MBEI condition states and SNBI items, IEC report content, FEMA matrix) with evidence and a ranked queue; a qualified person accepts or overrides each finding and that decision is logged against the model's.

**Rationale.** NBIS requires qualified inspectors, class societies keep a surveyor in the loop, and radiography codes say an AI output does not discharge the interpreter's responsibility (R02, R03). Every credible vendor positions as assist, not replace. The review log itself becomes a compliance feature incumbents lack (R03).

---

## D-013. No fine-tuning this weekend

**Status:** Accepted. **Date:** 2026-09-24.

**Decision.** Use retrieval of the standards' own tables (IEC Annex C rows, MBEI defect rows, FEMA matrix) plus a handful of labeled exemplars in the grader prompt. Keep a LoRA run as a stretch goal only.

**Rationale.** RAG-grounded VLMs beat the same VLM without retrieval on blade defects, and a bridge-priority paper found accuracy fell when noisy training data grew from 3k to 4k samples (R06). Most quality datasets are non-commercial, so a product model cannot be trained on them anyway (R07). Weekend fine-tuning notebooks exist if needed (R06).

---

## D-014. Scope widened to multi-sensor structural health, product is software plus a sensor kit

**Status:** Proposed (needs a team "yes" at the next sync; the documents and the two new code paths proceed on it by default). **Date:** 2026-09-25. **Source:** team whiteboard, two photos taken in the team room 2026-09-25 16:22, transcribed by the orchestrator; the team lead's instruction quoted below; research in R10.

**Context.** The whiteboard maps each person to a sensing domain and states the product shape:

| Person | Whiteboard entry (transcribed) |
|---|---|
| Atharva | underwater exploration, seismic readings, sensors, thermal + lidar, SW |
| Leena | buildings + bridges, ONLY aerial images, DRONE + SW |
| Jie | interior machinery, SW |
| Runze | software |
| Drew | a thermal item (illegible) |
| Box | OUR PROD: SW + HW |

The words "Reconstruction" and "16%" also appear with no readable context; neither is assigned a meaning here or in R10. The team lead's instruction (verbatim): "also can we have the crack measurements with the data if there is any scale present in the image? to get the dimentions of the crack as a number? Also the scope of the total AI model is changed a lot we are focusing on the cesmic data modelling, and other sencors like heatmaps, sonar data etc. ... go through the images, and change the problem scope as well".

**Options considered.**

| Option | For | Against |
|---|---|---|
| A. Keep the imagery-only scope of D-001 and D-002; add sensors after the weekend | Nothing built changes; `eval_v1` stays the only evidence | Contradicts the whiteboard and the instruction; the "SW + HW" product would have no path in the code or the deck |
| B. One engine, many modalities: each modality is an input adapter plus a rubric file; imagery stays the demo; seismic and sonar get a minimal but real path; lidar is roadmap; the product is software plus a sensor kit | Engine, contract, review log and honesty rules unchanged; every person on the whiteboard has a lane; public thresholds exist at the extremes (ShakeMap PGA and PGV, Hazus drift, ISO 20816-3 zones) and public hardware prices exist (R10 sections 2, 6) | Two new code paths in two days; no public sonar imagery of bridge substructure with condition labels (R10 section 1.4); frequency-drop rules are per structure, not universal (R10 section 2.1); the extreme-only thresholds are [PUBLIC, secondary] until verified |
| C. Pivot to seismic-first and drop imagery | Matches "we are focusing on the seismic data modelling" read literally | Discards the only measured evidence (`eval_v1`) and the only public labelled datasets; PEER NGA-West2 and Z24 are licence-gated (R10 section 2.2); Leena's stream is "ONLY aerial images" |
| D. A separate product per sensor | Each stream ships on its own | Five pipelines, five rubrics, no shared review log: the vertical-silo pattern R01 and R02 criticise in incumbents |

**Decision.** Option B. The product is a multi-sensor structural-health platform: the gate-crop-grade-prioritize-review-export engine is unchanged; inputs widen from aerial RGB imagery to thermal heatmaps, sonar (underwater), seismic and vibration readings and lidar, with interior machinery as a further image and vibration domain; the offer is software plus a sensor hardware kit whose public component prices are in R10 section 6. Bridge elements (D-001) remain the primary imagery wedge and the demo lead; PV thermal (D-002) remains the secondary imagery class.

**Rationale.** Every structural-monitoring incumbent found is quote-only and single-sensor, the gap R01 and R02 found in imagery analytics repeated for sensors (R10 section 6). FHWA's position that imaging "can supplement" Level I underwater inspection while "The Level II portion of the UWI is still to be performed by an underwater bridge inspection diver" [PUBLIC: https://www.fhwa.dot.gov/bridge/nbis2022/qanda/08.cfm] is the assist-not-replace posture of D-012, so sonar fits without changing positioning. Seismic and vibration indicators with published thresholds (ShakeMap PGA and PGV, Hazus drift ratio, ISO 20816-3 velocity zones) map onto S0 to S4 with U for "no baseline" (R10 section 2.1). Crack width from imagery reaches 0.16 to 0.22 mm with a scale reference in the frame [PUBLIC: https://www.mdpi.com/2673-8244/4/1/5, https://pmc.ncbi.nlm.nih.gov/articles/PMC10007411/], the same order as the 0.30 mm MBEI CS1/CS2 boundary, so a measured width is useful only with its uncertainty and basis attached (R10 section 4.3). **[Inference]**

**Consequences.**

1. Engine unchanged: one finding contract, one review log, one queue. A `modality` field is added to every record with default `rgb` (PRD FR-22) so every existing run, manifest and test stays valid.
2. Modalities are new input adapters plus rubric JSON files, not new pipelines. Sonar frames are an asset class through the cascade with an extent-and-geometry rubric (FR-25). Seismic and vibration series go through `cascade.signals`, which computes indicators and grades on rubric rows in plain code (FR-24). Crack width in mm comes from `cascade.measure` only when an in-image scale or a GSD exists, with uncertainty and basis recorded (FR-23).
3. Imagery remains the demo and the only measured accuracy. `data/eval_v1` is frozen and carries no scale, so crack width stays `not_measurable` there (R10 section 8); no accuracy figure is claimed for crack width, sonar or seismic this weekend.
4. Seismic and sonar get a minimal but real path: real code with tests, rubric rows that cite a source or say `team assumption`, synthetic or public sample data labelled as such, U when there is no baseline.
5. Lidar is roadmap (FR-26): no OpenCV, scikit-image or Open3D in the environment and the rule is no new packages; scanners cost $100,000 to $500,000 [PUBLIC: https://iscano.com/laser-scanning-lidar-best-practices/3d-laser-scanning-cost-guide-2025/].
6. Any frequency-drop percentage is a team assumption per asset class, written into the rubric JSON with `source: "team assumption"`; no single percentage goes on a slide (R10 section 8).
7. The kit is a bill of materials from public prices, about $16,300 (budget: Raspberry Shake RS4D DIY $604.99, DJI Matrice 4T $7,199, Deep Trekker DTG3 $8,500) to about $38,800 (standard: RS4D turnkey $784.99, Matrice 4T $7,199, Blueye X3 $30,788 ex VAT), sonar excluded because no sonar price is public [Inference, R10 section 6]. Nothing is bought, built or tested this weekend; D-009's hardware exclusion stands.
8. D-001 and D-002 are superseded in part: their wedge, dataset and rubric choices stand; their imagery-only framing does not. D-003 to D-013 are unchanged.

**Revisit trigger.** If by the Saturday 12:00 cut-line review (D-009) the seismic or the sonar path has no rubric with sourced rows or no passing test, that path drops to a roadmap slide and the deck says so. If the team states a different per-person split or product shape at the next sync, this record is amended with the date, not rewritten.
