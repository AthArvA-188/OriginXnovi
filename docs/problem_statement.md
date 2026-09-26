# Problem Statement

| Field | Value |
|---|---|
| Project | Origin Weekend Fall 2026, Prompt D (Infrastructure & Resilience) |
| Version | 1.1 (multi-sensor scope per D-014; v1.0 of 2026-09-24 superseded in part, every sourced number kept) |
| Date | 2026-09-25 |
| Owner | Founding team |
| Status | For team review. Wedge choice is Proposed in `docs/decisions.md` D-001; the widened scope is Proposed in D-014 (team whiteboard, 2026-09-25 16:22). |

**Citation key.** R01..R10 = the notes in `docs/research/`; R10 is `10_multisensor_scope.md` (2026-09-25). Labels per `docs/decisions.md` D-011: [Sourced: Rxx], [Inference], [Assumption]. Numbers new in v1.1 carry R10's own tags: **[PUBLIC: url]** read on the cited page, **[PUBLIC, secondary: url]** seen only in a search summary or a third-party page (verify before a slide), or **[Assumption]**.

## 1. The prompt we are answering

> **Prompt D:** How can we detect infrastructure damage and failures before they become expensive, and rapidly prioritize recovery when they occur?

The prompt's own framing: utilities, telecoms, transportation networks, insurers, industrial operators and infrastructure owners spend billions inspecting, maintaining and repairing physical assets. Much of that work is manual and reactive. Disasters force organizations to assess thousands of assets while every hour of downtime compounds losses. Meanwhile satellites, drones, vehicles, phones, cameras and sensors produce more physical-world imagery than ever. The strongest solutions serve **everyday inspection and maintenance workflows first**, with disasters as an additional high-value application.

## 2. The problem in one sentence

**Infrastructure owners and the inspection providers who serve them now capture far more imagery and sensor data than they can analyze, so damage is graded late, inconsistently, without a number where the standard asks for one, and without a defensible priority order. The result is avoidable failures, expensive reactive repairs, and slow disaster recovery.**

## 3. Where the bottleneck actually is

Capture is no longer the constraint. Drone inspection of a wind turbine takes 15 to 45 minutes at $300 to $600 versus 3 to 6 hours at $1,500 to $3,000 by rope access; helicopter line patrol costs $1,200 to $1,600 per mile versus $200 to $300 by drone [Sourced: R04]. The constraint has moved downstream:

1. **Image overload.** AEP Ohio's 2025 drone pilot covered about 4 percent of its distribution system, produced 400,000 to 500,000 images, and one person spent more than 500 hours reviewing them [Sourced: R08]. Post-storm drone campaigns generate 47 to 369 GB per day [Sourced: R03].
2. **Inconsistent grading.** In FHWA's study of 49 inspectors across 25 states, only 68 percent of condition ratings fell within one point of the mean; in a 2026 Indiana study only 30 percent of ratings matched the expected value; manual crack-area measurements vary from 18 percent to over 100 percent [Sourced: R08]. EPRI found agreement on any one blade damage category was rare and recommends treating every category as uncertain by one level [Sourced: R05].
3. **Slow, unstructured reporting.** Public solar guidance quotes 24 to 72 hours for an AI report and 3 to 5 business days end to end; most vendors publish no turnaround at all [Sourced: R01]. Bridge inspectors describe redundant manual data entry into disparate systems and difficulty locating defects from narrative descriptions [Sourced: R08].
4. **No consequence-aware prioritization.** NTSB found the Fern Hollow Bridge collapsed because the city failed to act on repeated maintenance recommendations from inspection reports; the I-40 Hernando de Soto fracture was visible in 2019 drone footage but not acted on until 2021 [Sourced: R08]. Bridges alone carry a $467B repair backlog across 624,167 structures, 41,677 of them in poor condition [Sourced: R04].
5. **Vertical silos.** Every wind vendor found is blade-only, every solar vendor PV-only, and the cross-sector players are service companies with per-vertical pipelines; none spans wind, solar, bridges, underwater and radiography in one engine [Sourced: R01].
6. **Disasters break the process entirely.** Duke Energy's 2024 hurricane season cost about $2.8B net of insurance and Hurricane Milton alone required 16,000 workers and 1,560 pole replacements [Sourced: R04]. The 2025 Los Angeles fires damaged or destroyed more than 18,189 structures; a FEMA declaration for the March 2025 Oklahoma wildfires took two months [Sourced: R03].
7. **The number the standard asks for is rarely measured.** MBEI condition states for concrete cracking turn on width (0.012 and 0.05 in for reinforced concrete) [Sourced: R05], yet the largest public bridge-defect dataset states "Damage size is measured with a pocket rule, thus, it's imprecise" and documents no scale in its images [PUBLIC: https://ar5iv.labs.arxiv.org/html/2309.00460]. Published image methods reach 0.22 mm precision with planar markers [PUBLIC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10007411/] and 0.16 mm mean absolute error with a laser calibration [PUBLIC: https://www.mdpi.com/2673-8244/4/1/5], which is the same order as the 0.30 mm CS1/CS2 boundary, so a width is only useful with its uncertainty attached [Inference, R10 section 4.3].
8. **Sensor data has the same bottleneck in a different shape.** Underwater bridge inspection is due at most every 60 months, every 24 months when the Scour Condition Rating is 3 or less, and up to 72 months with FHWA approval when it is 6 or better (23 CFR 650.311(b)) [PUBLIC: https://ecfr.io/Title-23/Section-650.311], and FHWA states that imaging technology "can supplement" Level I inspection while "The Level II portion of the UWI is still to be performed by an underwater bridge inspection diver" [PUBLIC: https://www.fhwa.dot.gov/bridge/nbis2022/qanda/08.cfm]. Vibration monitoring has published thresholds only at the extremes: modal-frequency drop has no universal published threshold, 5 to 10 percent seasonal swings can be normal, and controlled-damage drops of 0.37 to 1.4 percent have been reported [PUBLIC, secondary: https://arxiv.org/pdf/2010.07026 and https://pmc.ncbi.nlm.nih.gov/articles/PMC9227402/]. Every structural-monitoring vendor found (Move Solutions, Worldsensing, Resensys, Bentley iTwin IoT) is quote-only and single-sensor [Sourced: R10 section 6].

## 4. Who experiences it

| Segment | Role feeling the pain | What they need | Evidence |
|---|---|---|---|
| Transportation agencies and their inspection consultants (**primary wedge, D-001**) | Bridge inspection engineer at a consulting firm or county | Element-level condition states that fit federal reporting, with CS3/CS4 photo documentation, crack widths in the units the standard uses, and a per-structure work list, entered into the state system on deadline | RFPs from Dane County WI, Kansas DOT, INDOT and NJDOT specify exactly this; one fines $100 per day for late entry [Sourced: R08] |
| Wind and solar owners and O&M providers (**secondary, D-002**) | O&M lead, reliability engineer | Faster blade and thermal anomaly triage, graded on the scales OEMs and insurers accept | Solar equipment-driven loss rose from 2.36 percent (2021) to 5.08 percent (2025); a technician now covers 70 percent more MW than five years ago [Sourced: R04, R08] |
| Drone and inspection service providers | Owner-operator, chief pilot | An analytics layer they can resell so they compete on more than flight time | DSPs must now deliver "AI-ready, utility-grade" data; 15 to 25 percent of DSP imagery needs remediation [Sourced: R09] |
| Utilities (electric, telecom) | Asset manager, wildfire mitigation lead | Pole, conductor, insulator and tower defect triage across hundreds of thousands of assets | SCE runs 200,000+ inspections a year; PG&E aerially inspected 220,000 poles in 2024-25 [Sourced: R02] |
| Bridge owners with underwater elements; ports, offshore, marine owners (**sonar, D-014**) | Underwater inspection program manager, diving contractor, port or marine engineer | Level I screening and change detection from sonar and ROV frames between diver inspections; hull condition from ROV or diver video without weeks of review | Underwater inspection at most every 60 months, 24 when scour is rated 3 or less [PUBLIC: https://ecfr.io/Title-23/Section-650.311]; FHWA keeps Level II with the diver [PUBLIC: https://www.fhwa.dot.gov/bridge/nbis2022/qanda/08.cfm]; hull report AI exists at about 83 to 90 percent accuracy from one vendor; class surveyors stay in the loop [Sourced: R03] |
| Building and bridge owners in seismic regions (**seismic and vibration, D-014**) | Structural engineer, facilities or emergency manager | Site shaking class and drift or frequency indicators against a stored baseline within hours of an event, with "no baseline" stated as such | ShakeMap instrumental intensity: PGA about 11.5 %g = "Strong / Light damage", about 40 %g = "Severe / Moderate-heavy" [PUBLIC, secondary: https://www.intensitylab.com/mmi-scale/]; Hazus W1 High-Code drift ratios 0.004 / 0.012 / 0.040 / 0.100 for Slight / Moderate / Extensive / Complete [PUBLIC, secondary: https://www.fema.gov/sites/default/files/documents/fema_hazus-earthquake-model-technical-manual-6-1.pdf] |
| Industrial operators (**interior machinery, D-014**) | Plant maintenance lead; inspection or NDT lead | Machinery vibration severity on the ISO zone the plant already uses; weld radiograph screening tied to code acceptance criteria | ISO 20816-3:2022 Group 2 rigid zones A < 1.4, B 1.4 to 2.8, C 2.8 to 4.5, D > 4.5 mm/s RMS [PUBLIC, secondary: https://vibromera.eu/glossary/iso-20816-3/]; a certified interpreter must still sign; about 30 percent of NDT personnel are over 55 [Sourced: R03, R04] |
| Insurers, emergency managers | Claims, CAT and resilience teams | Rapid, consistent post-event grading | FEMA's 2025 PDA guide invites analytics on imagery; existing products grade buildings, not infrastructure [Sourced: R03] |

The initial wedge and the order of the other segments are recorded in `docs/decisions.md` D-001, D-002 and D-003; the sensor segments are added by D-014, with the rationale and the alternatives rejected.

## 5. Why existing solutions leave the problem open (hypotheses and verdicts)

Verdicts H1 to H6 come from the competitor research; details in `docs/research/00_research_brief.md` section 2. H7 to H10 come from R10.

- **H1. Single-vertical, closed-model tools.** *Supported.* Incumbents sell closed-vocabulary detectors per asset class [Sourced: R01, R02].
- **H2. Detection without industry-native grading.** *Supported.* Vendors advertise "severity ratings" without publishing definitions or per-finding rationale; no vendor found ships MBEI condition states with the clause cited [Sourced: R01, R02].
- **H3. No prioritization tied to consequence.** *Partially supported.* SkySpecs and Raptor Maps express findings in energy or dollar terms, but no cross-asset, consequence-ranked work queue was found [Sourced: R01].
- **H4. Enterprise-only economics.** *Supported.* Every analytics incumbent is quote-only except Scopito; small providers are left with AI-less generic tools at about $300 per month [Sourced: R01, R09].
- **H5. Poor fit for thermal, radiographic, and underwater imagery.** *Partially supported.* Thermal is well served inside solar; radiography software is sold on-prem to manufacturers; underwater AI is a thin bolt-on to hardware; nobody spans modalities [Sourced: R01, R03].
- **H6. Nothing built for the disaster surge.** *Refined.* Insurers get building-level classes within 24 to 48 hours from satellite and aerial providers; nobody grades engineered infrastructure after an event [Sourced: R03].
- **H7. Sensor-analytics incumbents repeat the imagery pattern.** *Supported.* Every structural-monitoring vendor found is quote-only and single-sensor; Worldsensing claims "up to 30%" material and "up to 40%" installation savings without a base [Sourced: R10 section 2.3, section 6].
- **H8. Sonar can screen substructure but cannot grade cracks.** *Supported.* ARIS Explorer 3000: 3.0 MHz identification mode to 5 m, 1.8 MHz detection mode to 15 m, downrange resolution 3 to 19 mm [PUBLIC: https://mfe-is.com/offshore/aris-explorer-3000/]; FHWA: sonars "have not demonstrated the ability to identify some smaller scale elements of substructure condition" [PUBLIC: https://www.fhwa.dot.gov/bridge/nbis2022/qanda/08.cfm]. No public sonar imagery of bridge piers, piles or abutments with condition labels was found [Sourced: R10 section 1.4]. Inference: the sonar rubric is extent and geometry (scour, undermining, exposed footing, debris, missing pile), never crack width.
- **H9. Seismic and vibration grading is promptable only at the extremes; the middle needs a per-structure baseline.** *Supported.* ShakeMap PGA and PGV classes, Hazus drift ratios and ISO 20816-3 zones are published [PUBLIC, secondary: see section 4 rows]; any frequency-drop percentage must be per structure and temperature-compensated, never a universal constant [Sourced: R10 section 2.1]. Public series are licence-gated: PEER NGA-West2 (21,336 records from 599 events) allows about 200 downloads per two weeks with an institutional e-mail and reset all accounts on 2026-07-02 [PUBLIC: https://ngawest2.berkeley.edu/]; the Z24 bridge benchmark (15 progressive damage scenarios, 100 Hz) is "available for non-commercial research" only [PUBLIC: https://bwk.kuleuven.be/bwm/z24].
- **H10. Crack width from imagery is feasible to about 0.2 mm when a scale reference is in the frame.** *Supported by the literature, not yet measured by us.* 0.22 mm with planar markers plus total station at 1 m stand-off [PUBLIC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10007411/]; 0.16 mm MAE with laser calibration [PUBLIC: https://www.mdpi.com/2673-8244/4/1/5]; 0.1 mm pixel size via ground-control block triangulation on a Taiwan bridge [PUBLIC: https://www.mdpi.com/2504-446X/7/6/342]; a 2026 UAV laser-cross module cut the coefficient of variation of fine-crack width from 0.36 to 0.10 and maps to MBEI states [PUBLIC: https://www.mdpi.com/2673-8244/6/3/58]. No public dataset with a per-image scale was found; `data/eval_v1` carries none [Sourced: R10 section 4.2].

## 6. Why now

- **Vision-language models (VLMs) changed the cost of specialization.** Structured JSON output is generally available on the major model APIs and on local models; a grader can be prompted with the standard's own tables [Sourced: R06].
- **Small VLMs make a cheap first pass possible.** Apache-2.0 models of 2 to 8 billion parameters run on a laptop; a deployed pre-filter cut expensive VLM calls by 240x; grading 1,000 frames costs about $10 heavy-only or about $3 with a local gate [Sourced: R06].
- **The literature validates the shape but not the shortcut.** A grounded detector-plus-model pipeline produced a 4 percent hallucination rate versus 65 percent for a zero-shot VLM; frontier models are weak at localizing small defects [Sourced: R02, R06]. That is why the product grounds, constrains and audits rather than trusting a raw model.
- **Capture hardware is commoditized.** Skydio is valued at $4.4B and 49 of 50 state DOTs use its drones, yet its bridge case studies contain no defect grading [Sourced: R02].
- **Sensor hardware is cheap enough for a kit.** Raspberry Shake RS4D (geophone plus 3-axis MEMS accelerometer) $784.99 turnkey or $604.99 DIY [PUBLIC: https://raspberryshake.org/pricing]; ADXL355 MEMS accelerometer, 25 ug/rtHz noise density, $28.25 each at 1,000 units (2016 price) [PUBLIC: https://www.eetasia.com/3-axis-mems-accelerometers-detect-structural-defects/]; SM-24 geophone element $59.95 [PUBLIC: https://www.spikenzielabs.com/Catalog/sensors/envirnomental/geophone-sm-24]; DJI Matrice 4T thermal drone with a 640 x 512 core $7,199 [PUBLIC: https://www.dronenerds.com/products/dji-matrice-4t-thermal-enterprise-drone]; Deep Trekker DTG3 mini ROV $8,500 [Sourced: R03]. Imaging sonar prices are not public [Sourced: R10 section 1.3].
- **Regulatory and climate pressure is rising.** NBIS routine inspections at most every 24 months; underwater inspections at most every 60 months [PUBLIC: https://ecfr.io/Title-23/Section-650.311]; SNBI element-level data due 2028-03-15; FAA Part 108 expected by end of 2026 will raise image volumes; California wildfire plans mandate more frequent inspections [Sourced: R02, R04]. About 30 percent of NDT personnel are over 55 and solar jobs grew 12 percent while capacity grew 286 percent [Sourced: R04].

## 7. What we are building (one paragraph)

A multi-sensor structural-health platform: software plus a sensor kit, on one grading engine (D-014). Inputs are aerial and handheld RGB imagery of bridges and buildings, thermal heatmaps (PV modules today; bridge-deck delamination and building envelopes as roadmap rubrics), sonar frames from ROVs for underwater substructure, seismic and vibration time series from ground and structure sensors (buildings, bridges, interior machinery) and, on the roadmap, lidar point clouds. Every input record carries a `modality`. Image frames go through the existing cascade: a small local VLM gates every frame for usability and the presence of damage; flagged frames are cropped and graded by a heavy VLM on the customer's industry-native scale with the verbatim criterion matched, a unified S0 to S4 severity, a confidence and an evidence crop. When a scale reference is in the frame (a marker of known size, a ruler) or a ground sample distance is known, crack width is measured in millimetres and reported as width plus or minus uncertainty with the basis recorded; without one, the finding says `not_measurable` and is never S0. Sonar frames go through the same cascade with range and frequency passed as text and an extent-and-geometry rubric. Seismic and vibration series skip the VLM: indicators (PGA, PGV, RMS velocity, drift, dominant frequency and its shift against a stored baseline) are computed in plain code and graded on rubric rows whose thresholds cite a source or are labelled team assumptions, with U when there is no baseline. All findings share one contract, one consequence-weighted queue, one review log and one export, which a qualified inspector reviews and overrides. In a disaster the same pipeline runs in surge mode. The kit (seismic node, thermal drone, mini ROV) is a bill of materials from public prices. Requirements are in `docs/PRD.md`.

## 8. Scope for this hackathon

**In scope (by Sunday 2026-09-27, 11:59 PM PT):**
- A working pipeline on real public imagery for bridge elements (primary) and PV thermal modules (secondary), plus post-disaster UAV imagery in surge mode, demonstrating the cascade, industry-native grading and a prioritized queue (D-001 to D-003, D-006).
- Measured gate and grading accuracy on a frozen held-out set with confidence intervals, reported honestly (D-007).
- A `modality` on every record (FR-22); crack width in mm from an in-image scale or GSD with uncertainty and basis (FR-23, `python -m cascade.measure`), demonstrated on images that contain a scale reference; a minimal but real seismic path (FR-24, `python -m cascade.signals`) on a synthetic or public series labelled as such; sonar as an asset class with a sourced extent rubric (FR-25) on debris or pipeline sonar frames labelled as such.
- A kit bill of materials from public prices on one slide; about $16,300 (budget) to about $38,800 (standard), sonar excluded because no sonar price is public [Inference, R10 section 6].
- Customer evidence: public procurement documents and practitioner quotes from R08, plus any interviews completed over the weekend, clearly separated from inference.
- The 6-slide deck, a 30 to 60 second demo video, and a Devpost submission.

**Out of scope for the weekend:** flight operations, building or testing any hardware, lidar processing (FR-26, roadmap), custom model training (D-013), integrations beyond CSV or JSON export, any accuracy claim we did not measure, and any crack width on `data/eval_v1` (the frozen set carries no scale, so a width there would be fabricated).

### Honesty box: what is still imagery-only in the demo

| Item | State on 2026-09-25 |
|---|---|
| Measured accuracy | Exists only for imagery, on `eval_v1` (184 images per `docs/progress.md`). No accuracy number exists or will be claimed for crack width, sonar or seismic this weekend. |
| Crack width | No public dataset ships a per-image scale; `eval_v1` carries none, so width stays `not_measurable` there. The demo uses team-photographed or synthetic images with a marker or ruler in frame; the width is shown as width plus or minus uncertainty with `scale_source`, never a bare number. Which scale reference the drone team will carry is open (section 10). |
| Sonar | No public sonar imagery of bridge piers, piles or abutments with condition labels was found. The demo uses marine-debris or pipeline sonar sets (Marine Debris FLS: CC BY-NC-SA 4.0 on Zenodo, non-commercial) or synthetic frames, labelled as such; the rubric is extent and geometry only. |
| Seismic and vibration | Public series are licence-gated (NGA-West2 institutional e-mail and download caps; Z24 non-commercial). The demo uses a synthetic series or a CESMD record labelled as such. Indicators are computed; the frequency-drop percentage is a team assumption per asset; ShakeMap, Hazus and ISO 20816-3 rows are [PUBLIC, secondary] until verified against the primary documents. |
| Thermal beyond PV | Rubric rows only. ASTM D4788 states it "should not be used for acceptance or rejection" [PUBLIC: https://store.astm.org/d4788-03r22.html]; ISO 6781-1:2023 is qualitative. No demo data. |
| Interior machinery | ISO 20816-3 zones need machine power class and support type as metadata; those fields do not exist on the asset record yet; no data. |
| Lidar | Slide only. No point-cloud code; no OpenCV, scikit-image or Open3D in the environment. |
| Hardware kit | Bill of materials from public prices; nothing bought, built or tested; sonar excluded because no price is public. |

## 9. How we will know the problem is worth solving

- Asset owners or service providers confirm that image review and inconsistent grading, not capture, is their bottleneck. Public evidence already points this way (section 3); interviews will test it.
- Public RFPs and industry standards show buyers ask for graded, standards-aligned outputs, not raw detections. *Confirmed for bridges* [Sourced: R08].
- Our measured cascade cost per 1,000 images is a fraction of manual review at prevailing rates. Expected order of magnitude is single-digit dollars per 1,000 frames [Sourced: R06]; the measured figure comes from eval_v1.
- At least one segment shows a sales cycle a student team could realistically close within months. Consultants and DSPs, not Tier-1 utilities, whose trial-to-award has run about three years [Sourced: R09].
- For the sensor scope: a discovery interview confirms that an owner would pay for a kit plus a grading subscription rather than a quote-only monitoring platform. No subscription price goes on a slide until an interview supports it [Sourced: R10 section 6].

## 10. Open questions

1. ~~Which wedge?~~ Proposed in D-001 and D-002; scope widened by D-014; team confirms at the next sync.
2. ~~Which grading scales?~~ Mapped in R05 and `docs/PRD.md` section 7; MVP speaks MBEI/NBI, ISO 4628-3, IEC TS 62446-3 CoA and FEMA PDA; v1.1 adds SNBI B.C.11 style extent rows for sonar, ShakeMap, Hazus and ISO 20816-3 rows for seismic and vibration (secondary until verified).
3. How do frontier VLMs actually perform on thermal inputs versus RGB for our data? Answered by eval_v1 on InfraredSolarModules.
4. What is the honest measured accuracy on eval_v1, and which failure modes must the UI surface? Answered Saturday.
5. Who pays first? Hypothesis in D-008: consultants and DSPs per asset; tested in interviews.
6. Which scale reference will the drone team (Leena) carry: comparator card, ArUco board or tape? The measurement module supports all three, but the demo images must contain one [Sourced: R10 section 9].
7. Which frequency-drop percentage does the team adopt per asset class, and is temperature compensation in scope for the weekend? To be written into the rubric JSON with `source: "team assumption"` [Sourced: R10 section 9].
8. Real seismic series (CESMD or NGA-West2, given registration lead time and the 2026-07-02 membership reset) or a synthetic series labelled as such?
9. ACI 224R exposure class default per asset (de-icing 0.007 in / 0.18 mm proposed as the strictest plausible bridge default) [Assumption, R10 section 4.3].
10. Sonar price points (ARIS, Oculus, BlueView, Kongsberg) for the kit BOM; Level II cleaning percentage from FHWA-NHI-23-027; SNBI B.C.15 code text; the Hazus 6.1 table row for the demo building types [Sourced: R10 section 9].
11. The whiteboard's "Reconstruction" and "16%": no readable context; not assigned a meaning until the team explains them.

## Related documents

- `docs/PRD.md` - product requirements
- `docs/decisions.md` - decision log (D-014 for the v1.1 scope)
- `docs/implementation_plan.md` - build plan for the weekend
- `docs/progress.md` - running progress log
- `docs/changelog.md` - document and product change history
- `docs/documentation.md` - documentation index and technical notes
- `docs/research/00_research_brief.md` - synthesis of the research notes
- `docs/research/10_multisensor_scope.md` - multi-sensor scope research (R10)
- `docs/research/` - sourced research notes R01 to R10
