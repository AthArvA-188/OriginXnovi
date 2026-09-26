# 10 - Multi-Sensor Scope: Underwater Sonar, Seismic/Vibration, Thermal, Crack Metrology, Lidar, SW+HW Pricing

| Field | Value |
|---|---|
| Version | 1.0 |
| Date | 2026-09-25 (Friday, Los Angeles) |
| Trigger | Two whiteboard photos (team room, 2026-09-25 16:22) and the user instruction quoted in section 0 |
| Inputs already sourced | R00 brief, R03 (underwater/NDT/disaster competitors), R05 (grading standards), R06 (VLM landscape), R07 (datasets); nothing in those notes is repeated here except by cross-reference |
| Method | 30 web searches, about 60 page fetches on 2026-09-25. Federal Register / eCFR, MDPI, SAGE, Wiley, Springer and Nature returned 403 or auth redirects; where a mirror or the Semantic Scholar API was used it is named. PDFs over 10 MB (Hazus 6.1 manual, SNBI) could not be fetched; those items are marked secondary. |
| Tags | [PUBLIC] = number read on the cited page; [PUBLIC, secondary] = number seen only in a search-engine summary or a third-party page, verify before a slide; [Inference] = our derivation or assumption |

---

## 0. What changed and what this note adopts

User instruction (verbatim): "also can we have the crack measurements with the data if there is any scale present in the image? to get the dimentions of the crack as a number? Also the scope of the total AI model is changed a lot we are focusing on the cesmic data modelling, and other sencors like heatmaps, sonar data etc. ... go through the images, and change the problem scope as well".

Whiteboard transcription (by the orchestrator, from the two photos): ATHARVA - underwater exploration, seismic readings, sensors, thermal + lidar, SW. LEENA - buildings + bridges, ONLY aerial images, DRONE + SW. JIE - interior machinery, SW. RUNZE - software. DREW - a thermal item (illegible). Box "OUR PROD: SW + HW". The words "Reconstruction" and "16%" appear without readable context and are **not** assigned a meaning in this note.

Scope adopted for the rest of this note: the product becomes a **multi-sensor structural-health platform**. The existing gate-crop-grade-prioritize-review-export engine stays; inputs widen from aerial RGB to thermal heatmaps, sonar (underwater), seismic/vibration time series and lidar, with interior machinery as a further image domain; the product is software plus a sensor hardware kit. Every number below is tagged.

Summary of what the research supports:

1. **Underwater is regulated and interval-driven, and FHWA has already ruled on sonar's role**: underwater inspections at most every 60 months, 24 months when the Scour Condition Rating is 3 or less, up to 72 months with FHWA approval when it is 6 or better (23 CFR 650.311(b)) [1]. FHWA's NBIS Q&A (updated 2024-04-05) says imaging technology can supplement Level I but "The Level II portion of the UWI is still to be performed by an underwater bridge inspection diver", and sonars "have not demonstrated the ability to identify some smaller scale elements of substructure condition" [5]. Inference: sonar AI is a Level I screening and change-detection aid, not a replacement for the diver, which is exactly the human-verified posture the rest of the product uses.
2. **Imaging sonar delivers centimetre geometry, not millimetre cracks.** ARIS Explorer 3000: 3.0 MHz identification mode to 5 m, 1.8 MHz detection mode to 15 m, 128 beams at 0.25 deg spacing, downrange resolution 3-19 mm, 30 x 15 deg field of view, 4-15 fps [6]. No public sonar dataset of bridge piers or piles with condition labels was found; the closest public sets are marine-debris and pipeline sonar sets (section 1.4) [8][9][10][11].
3. **Seismic/vibration SHM has published, promptable thresholds only at the extremes.** Hazus drift-ratio damage states (e.g., W1 High-Code Slight 0.004 / Moderate 0.012 / Extensive 0.040 / Complete 0.100) [17, secondary], ISO 20816-3:2022 machinery velocity zones (Group 2 rigid: A < 1.4, B 1.4-2.8, C 2.8-4.5, D > 4.5 mm/s RMS) [23], and the ShakeMap instrumental-intensity table (PGA about 11.5 %g = "Strong / Light damage", about 40 %g = "Severe / Moderate-heavy") [16, secondary]. Modal-frequency drop has **no** published universal threshold: a >5 % deviation from a model is used as a flag in one 2025 digital-twin paper, while 5-10 % seasonal swings can be normal, and controlled-damage drops of 0.37-1.4 % have been reported [13][14, secondary]. Inference: a frequency-drop rule must be per-structure and temperature-compensated; do not put a single percentage on a slide.
4. **Public seismic/vibration data exists but is licence-gated.** PEER NGA-West2: 21,336 three-component records from 599 events, M 3.0-7.9 [19]; downloads capped at about 200 records per two weeks, institutional e-mail only, all accounts reset on 2026-07-02 to a one-year membership [18]. Z24 benchmark: one year of monitoring plus 15 progressive damage scenarios, 100 Hz, "available for non-commercial research" only [20]. Tianjin Yonghe: 14 accelerometers, 100 Hz, 24 h healthy (2008-01-17) and damaged (2008-07-31) sets [22]. CESMD (USGS + CGS) serves strong-motion records including instrumented structures [21].
5. **Sensor hardware is cheap enough for a kit.** Raspberry Shake RS4D (geophone + 3-axis MEMS) $784.99 turnkey / $604.99 DIY; RS1D $584.99 / $294.99 board-only [24]. ADXL355 MEMS accelerometer 25 ug/rtHz noise density, $28.25 each at 1,000 units (2016 price) [26]. SM-24 geophone element $59.95 [27]. DJI Matrice 4T thermal drone $7,199 (640 x 512 radiometric core) [40]. Commercial SHM vendors (Move Solutions, Worldsensing, Resensys, Bentley iTwin IoT) publish no prices [28][29][30][31].
6. **Thermal outside PV has references but weak quantitative content.** ISO 6781-1:2023 (50 pages) replaces ISO 6781:1983 for building envelopes [32][33]; the 1983 text required a 10 degC air-to-air difference for 24 h [34, secondary]. ASTM C1153 is night-time roof wet-insulation location [34, secondary]. ASTM D4788-03(2022) covers vehicle-mounted IR for deck delamination, overlays up to 4 in (100 mm), and states "A Precision and Bias statement has not been developed ... should not be used for acceptance or rejection" [35]; secondary sources give 0.5 degC as the minimum sound/defective contrast [36, secondary].
7. **Crack width from imagery is feasible to about 0.2 mm with a scale reference, and the limits it feeds are already in our rubric.** Published: 0.22 mm precision with planar markers plus total station at 1 m (Sensors, 2023-02-25) [43]; 0.16 mm MAE with a laser calibration (Metrology, 2024-02-05) [44]; 0.1 mm pixel size via GCP block triangulation on a Taiwan bridge (Drones, 2023-05-25) [45]; a 2026 UAV laser-cross system that cut the coefficient of variation of fine-crack width from 0.36 to 0.10 and maps to AASHTO MBEI states (Metrology, 2026-08-21) [46]. MBEI CS thresholds in `src/cascade/rubrics/bridge_mbei.json`: RC <0.012 / 0.012-0.05 / >0.05 in; PSC <0.004 / 0.004-0.009 / >0.009 in [47]. ACI 224R-01 Table 4.1 tolerable widths: 0.016 in dry air, 0.012 humid, 0.007 de-icing, 0.006 seawater, 0.004 water-retaining [48, secondary]. **dacl10k has no documented rulers**; its paper says "Damage size is measured with a pocket rule, thus, it's imprecise" and the German minimum crack width to document is 0.2 mm [49]. The "rulers in some photos" claim is therefore **not verified**.
8. **Lidar adds millimetre geometry at five-figure cost and is roadmap.** Trimble TX8 TLS +/-2 mm declared, consistent with tachymetry (1-2 mm) on a 165 m suspension span; photogrammetry 18-20 % high [52]. Scanner purchase "$100,000 and $500,000"; scan service $150-500 per hour; QC inspection $1,000-4,000 each (iScano, 2025-08-10) [53]. Inference: with no OpenCV/scikit-image in the env and a 2-day window, point-cloud geometry is a slide, not a demo.
9. **Market framing SW+HW**: every structural-monitoring incumbent is quote-only, exactly like the imagery incumbents in R01/R02. The only public anchors are the component prices in item 5 and the RaaS pattern ("tied to asset coverage rather than hourly rates", Gecko, R03). A bottom-up kit BOM from public prices is about $16k (budget: RS4D DIY + Matrice 4T + Deep Trekker DTG3) to about $39k (RS4D turnkey + Matrice 4T + Blueye X3), sonar excluded because no sonar price is public [Inference, section 6].

---

## 1. Underwater bridge inspection

### 1.1 Regulation: 23 CFR 650.311(b) (2022 NBIS) [1]

Read on the ecfr.io mirror because ecfr.gov redirected to a block page.

- (b)(1)(i): "Each bridge must be inspected at regular intervals not to exceed 60 months".
- (b)(1)(ii): reduced interval "not to exceed 24 months" when NBI criteria are met, including (B)(3) "The observed scour condition is three (3) or less, as recorded by the Scour Condition Rating item".
- (b)(1)(iii): extended interval "not to exceed 72 months" with written FHWA approval when all criteria are met, including (A)(3) "Scour Condition Rating item is satisfactory or better, coded six (6) or greater".
- Owners "must develop and document criteria used to determine when intervals must be reduced below 60 months"; factors include "structure type, design, materials, age, condition ratings, scour, environment, ... history of vehicle/vessel impact damage" [2, search summary of the same section].
- Routine inspection baseline stays at 24 months (Method 1) - already in R05.
- FHWA Q&A (updated 2024-04-05): underwater portions not rehabilitated "do not need an underwater or NSTM inspection within 12 months"; after rehabilitation of underwater sections, inspection "within 12 months of rehabilitation work being completed"; subsequent inspections must include "all underwater portions" [5].

### 1.2 Inspection levels and NBI/SNBI items

TxDOT Bridge Inspection Manual, section 9 (Underwater Inspections) [3]:

| Level | Definition (verbatim) |
|---|---|
| I | "A simple visual or tactile (by feel) inspection, without the aid of tools or measuring devices." |
| II | "A detailed inspection which involves physically cleaning or removing growth from portions of the structure to assess hidden defects." |
| III | "A highly detailed inspection of a structure which is warranted if extensive repair or replacement is being considered." |

TxDOT triggers: water depth "at least 4 feet year-round"; frequency 60 months, reduced to 12 or 24 months, extended to 72; legacy NBI items used for the interval decision are Item 60 Substructure (<=4 reduce, >=6 extend), Item 61 Channel and Channel Protection (same thresholds) and Item 113 Scour Critical (<=3 reduce; 5 or 8 extend). Instruments named: "black-and-white fathometer", "color fathometer", "ground penetrating radar" for shallow water [3]. The Level II cleaning percentage (commonly 10 % of members) is in FHWA-NHI-23-027 (Sept 2024) [4]; the 8 MB PDF could not be text-extracted here, so the percentage is **not verified in this session**.

SNBI (March 2022) adds rated water items on the same 0-9 scale as R05 Table 20: B.C.09 Channel Condition, B.C.10 Channel Protection, B.C.11 Scour Condition, B.C.14 NSTM Inspection Condition, B.C.15 Underwater Inspection Condition [7, secondary: NYSDOT tool-tips PDF and Geocadra summary; the SNBI PDF itself exceeded the fetch limit]. Geocadra: "Critical Findings can also be triggered by extreme scour conditions (Scour Condition B.C.11 rated 3 or below)" [7]. Element-level underwater damage grades (Minor/Moderate/Advanced/Severe with section-loss and crack-width numbers) are already in R05 section 5.2 (NYC EDC WFMMS).

### 1.3 What imaging sonar delivers

FHWA position (NBIS Q&A Q313-11, updated 2024-04-05): imaging technology "can supplement" Level I; Level II remains a diver task; sonars "have not demonstrated the ability to identify some smaller scale elements of substructure condition" [5]. WSDOT/CDOT ROV-camera approvals for depths >120 ft are in R03 (HIF-18-049).

| Sonar | Specs (verbatim where quoted) | Price | Source |
|---|---|---|---|
| Sound Metrics ARIS Explorer 3000 | "1.8 MHz" detection, "15m @ 15 degC"; "3.0 MHz" identification, "5m @ 15 degC"; beams "128 or 64"; spacing "0.25 deg nominal"; FOV "30 deg horizontal x 15 deg vertical"; downrange resolution "3mm to 19mm"; "4-15 frames/sec (128 beams)"; depth "300m"; "5.12 kg (11.0 lb)" in air; "18 Watts typical" | not published ("contact") | [6] |
| Blueprint Oculus M750d | dual frequency 750 kHz / 1.2 MHz; range 0.1-120 m; "Integrated velocimeter" | not displayed on Deep Trekker shop page | [12] |
| Teledyne BlueView M900 Mk2 | 900 kHz and 2250 kHz, "130 degree field of view" each | not found | [12, search summary] |
| Kongsberg (M3 / Flexview) | not fetched | not found | gap |

Inference: at 3-19 mm downrange resolution and centimetre-class cross-range at 5 m, sonar can score **presence/extent** of scour holes, exposed footings, undermining, debris, missing piles and gross section loss (the SNBI B.C.11 / NYC EDC "Advanced/Severe" classes), but not the 1/32-1/8 in crack-width classes. The product's sonar rubric should therefore be an extent/geometry rubric, and the VLM sees a rendered sonar frame plus range metadata as text (same pattern as radiometric thermal in R06 section 5).

### 1.4 Public sonar datasets (structures)

| Dataset | Content | Sonar | Size | Licence | Source |
|---|---|---|---|---|---|
| Marine Debris FLS Datasets (Zenodo 15101686, 2025-03-28) | marine debris objects in "watertank, turntable, flooded quarry"; classification, detection, segmentation, patch matching | ARIS Explorer 3000 | 7.1 GB (quarry 6.5 GB) | CC BY-NC-SA 4.0 on Zenodo (arXiv page said CC0 - conflict, treat Zenodo as controlling) | [8][9] |
| UATD (multibeam FLS) | "basic shapes that make up engineering structures, such as cubes and cylinders", proposed as pre-training for "bridge piers and abutments" | MFLS | not fetched | not fetched | [10, search summary] |
| Portoroz 2025 | marina sequences, 1,452 and 4,080 images | Oculus 750d | 2 sequences | not fetched | [10, search summary] |
| Survey arXiv 2510.03353 (2025-10-07) master table | ROSAR (synthetic aperture sonar, "pipeline inspection and subsea structure detection", Zenodo); SubPipe (side-scan, pipelines, GitHub); Shipwreck (side-scan); NNSSS; SeabedObjects-KLSG (Kaggle) | SAS / side-scan | varies | "publicly available through academic repositories" | [11] |
| Ge, Singh, Sadhu 2024 (SHM journal) | YOLOv7 adaptation on MFLS; "engineering structures have similar geometric shapes to the objects tested ... potential applicability to underwater structural inspection" | MFLS | "a dedicated public dataset" (not named in abstract) | - | [10] |

Verdict: **no public sonar imagery of bridge piers, piles or abutments with condition labels was found.** Any sonar demo must use debris/pipeline sets or synthetic frames, labelled as such. Optical underwater sets (LIACi hull segmentation, SUIM) are in R07.

---

## 2. Seismic and vibration-based SHM

### 2.1 Indicators practitioners use, with the published thresholds we could find

| Indicator | Published threshold | Status | Source |
|---|---|---|---|
| Peak ground acceleration / velocity (site shaking) | ShakeMap instrumental intensity: PGA <0.0464 %g "Not felt"; 0.297 "Weak"; 2.76 "Light"; 6.2 "Moderate / Very light damage"; 11.5 "Strong / Light"; 21.5 "Very strong / Moderate"; 40.1 "Severe / Moderate-heavy"; 74.7 "Violent / Heavy"; >139 "Extreme / Very heavy" (PGV cm/s: 0.0215 / 0.135 / 1.41 / 4.65 / 9.64 / 20 / 41.4 / 85.8 / >178) | [PUBLIC, secondary] - values from a search summary attributed to USGS ShakeMap V4 documentation (Worden et al. 2012 GMICE); the docs page returned 404 and the USGS image page had no legend text | [16] |
| Inter-storey drift ratio (building damage state) | Hazus "Typical Drift Ratios Used to Define Median Values of Structural Damage": W1 High-Code Slight 0.004 / Moderate 0.012 / Extensive 0.040 / Complete 0.100 (Table 5-12 in Hazus 4.2; Table 5-19 in Hazus 6.1, July 2024); values vary by building type and code level | [PUBLIC, secondary] - the 6.1 PDF exceeds the 10 MB fetch limit; verify page before slide | [17] |
| ATC-20 rapid evaluation | "Building or story leaning" rated Minor/Moderate/Severe; estimated damage bins 0-1 / 1-10 / 10-30 / 30-60 / 60-100 / 100 %; "Severe conditions endangering the overall building are grounds for an Unsafe posting" - no numeric drift threshold on the form | [PUBLIC] (R05 section 7.3) | R05 |
| Modal (natural) frequency shift | One 2025 digital-twin paper flags a measured frequency that "deviates from the FEM-predicted value by more than a set tolerance (e.g. > 5 %)"; a crowdsourced-bridge paper warns "large fluctuations, e.g., 5-10%, of a bridge's modal frequency can be normal in some climates"; controlled-damage studies report drops "ranging from 0.37% to 1.4%" | [PUBLIC, secondary] - all three from search summaries; pages not fetched | [13][14][15] |
| Frequency and damping anomaly (statistical) | Sensors 2020 (Kostic-type study): thresholds set in sigma, "around 1 sigma" for mild damage to "beyond 6 sigma" for severe, on a 20 m two-span RC bridge (UNR shake table) and a Columbia University steel pedestrian bridge; frequency gave a "sharp distinction" versus damping | [PUBLIC] | [13] |
| Machinery vibration severity (interior machinery) | ISO 20816-3:2022 broadband RMS velocity, 10-1000 Hz (2-1000 Hz below 600 rpm). Group 2 (15-300 kW) rigid: A <1.4, B 1.4-2.8, C 2.8-4.5, D >4.5 mm/s; flexible: <2.3 / 2.3-4.5 / 4.5-7.1 / >7.1. Group 1 (>300 kW) rigid: <2.3 / 2.3-4.5 / 4.5-7.1 / >7.1; flexible: <3.5 / 3.5-7.1 / 7.1-11.0 / >11.0. Zone C "Not suitable for continuous long-term operation"; Zone D "Vibration severe enough to cause damage" | [PUBLIC, secondary] - vendor glossary reproducing the standard | [23] |
| Salawu 1997 review ("5 % frequency change" rule often cited) | abstract not retrievable (ScienceDirect and Semantic Scholar returned no abstract) | **not verified** - do not quote a number | [25] |

Inference for the rubric: seismic/vibration grading in the product should be (a) site shaking class from PGA/PGV via the ShakeMap table, (b) building drift class via Hazus per building type, (c) machinery zone A-D via ISO 20816-3, and (d) a per-structure frequency-drop alarm whose percentage is a **team assumption per asset**, temperature-compensated, never a universal constant. All four map cleanly onto S0-S4 with U for "no baseline".

### 2.2 Public datasets

| Dataset | What it is | Access / licence | Source |
|---|---|---|---|
| PEER NGA-West2 | "21,336 (mostly) three-component records from 599 events", M 3.0-7.9, distance 0.05-1,533 km, Vs30 94-2,100 m/s; time series plus spectra at 111 periods | ngawest2.berkeley.edu: "approximately 200 records every two weeks, 400 every month"; "Only university or corporate email addresses"; "on July 2, 2026, all user accounts will be disabled, and all users will need to register for a one-year membership"; released "as is" | [18][19] |
| CESMD (strongmotioncenter.org) | USGS + CGS strong-motion records; "Search for Data from Specific Structures or Station Types"; "Data Attribution and Use Policy" | public, policy page not fetched | [21] |
| Z24 bridge benchmark (KU Leuven) | post-tensioned box girder, 30 m main span, built 1963, demolished 1998; one year of monitoring; 16 accelerometers at 100 Hz, 48 environmental sensors hourly; 15 progressive damage scenarios (pier settlement 4 stages, foundation tilt, spalling 2, landslide, hinge failure, anchor heads 2, tendon ruptures 3) | "Available for non-commercial research", cite KU Leuven, no third-party transfer | [20] |
| Z24 reported effect | second mode (first transverse) "ranging from 5.02 to 4.72 Hz" across damage; "stiffness degradation up to 30% due to 95mm settlement" | [PUBLIC, secondary] - search summaries of ResearchGate/Academia pages | [20, secondary] |
| Tianjin Yonghe (SMC benchmark) | cable-stayed, 510 m long, 260 m main span; 14 uniaxial deck accelerometers at 100 Hz; 24 h healthy set 2008-01-17 and damaged set 2008-07-31; damage found Aug 2008: closure-segment crack and girder/auxiliary-pier detachment | UCF IASCM page (licence not fetched) | [22] |
| KW51 railway bridge (Leuven) | ML damage identification (Stacked GRU, kNN, CNN) on a real bridge retrofit; frequency numbers not in abstract | arXiv 2408.03002 (2024-08-06, rev. 2024-09-25) | [15] |

### 2.3 Sensor hardware costs

| Item | Price / spec | Date | Source |
|---|---|---|---|
| Raspberry Shake RS1D (vertical geophone) | "$584.99" turnkey indoor; "$294.99" board and sensor only; "$834.99" outdoor | page fetched 2026-09-25 | [24] |
| Raspberry Shake RS4D (geophone + 3-axis MEMS accelerometer) | "$784.99" turnkey; "$604.99" DIY; "$1,054.99" outdoor | same | [24] |
| Raspberry Shake RS3D | "$1,134.99" turnkey; "$954.99" DIY; "$1,404.99" outdoor | same | [24] |
| RS&Boom (seismo + infrasound) | "$934.99" / "$794.99" / "$1,254.99" | same | [24] |
| RS1D vs RS4D | "the RS4D has a 3 dimensional accelerometer built into the board"; RS1D "uses a single, vertical geophone" | - | [24, search summary] |
| ADXL355 MEMS accelerometer | noise density 25 ug/rtHz; +/-2/4/8 g; <200 uA; "$28.25" at 1,000 units (2016) | 2016 | [26] |
| SM-24 geophone element | "$59.95" (SpikenzieLabs); bandwidth 10-240 Hz, 28.8 V/m/s | 2026 page | [27] |
| Commercial SHM (Move Solutions DECKAXE-SHM, DECK002-X; Worldsensing Loadsensing; Resensys SenSpot 10-year CR123A battery; Bentley iTwin IoT, ex-sensemetrics, acquired 2021-04-29) | **no public prices**; Move: "customized quotations"; Worldsensing claims "up to 30%" material and "up to 40%" installation savings without a base | 2021-2026 | [28][29][30][31] |

---

## 3. Thermal beyond PV

IEC TS 62446-3 (PV) is in R05 section 2 and radiometric preprocessing in R06 section 5. References for the two new thermal domains:

| Standard | Scope (verbatim where quoted) | Numbers | Status | Source |
|---|---|---|---|---|
| ISO 6781-1:2023 "Performance of buildings - Detection of heat, air and moisture irregularities in buildings by infrared methods - Part 1: General procedures" | "specifies methodologies by thermographic examination for detecting thermal irregularities in building envelopes"; residential, commercial, institutional | 50 pages; published 2023 | [PUBLIC] (ISO catalogue page blocked; NBS/SIS/BSI listings fetched via search) | [32][33] |
| ISO 6781:1983 (superseded) | "qualitative method, by thermographic examination (infrared method), for detecting thermal irregularities in building envelopes" | "air-air temperature drop across the building envelope shall be at least 10 degC for at least 24 hours prior to investigation"; inspect from the low-pressure side | [PUBLIC, secondary] (Snell Group summary) | [34] |
| ASTM C1153 "Standard Practice for the Location of Wet Insulation in Roofing Systems Using Infrared Imaging" | roofs, "only addresses inspections of roofs at night" | current edition C1153-10(2015) per ASTM store search listing; store page returned 403 | [PUBLIC, secondary] | [34] |
| ASTM C1060 | thermographic inspection of insulation in envelope cavities of frame buildings | - | listed by Snell Group; not fetched | [34] |
| ASTM D4788-03(2022) "Standard Test Method for Detecting Delaminations in Bridge Decks Using Infrared Thermography" | "imaging infrared scanner and video recorder, mounted on a vehicle"; exposed and overlaid decks; "asphalt or concrete overlays as thick as 4 in. (100 mm)"; "A Precision and Bias statement has not been developed at this time. Therefore, this standard should not be used for acceptance or rejection of a material." | adopted 2003, reapproved 2022 | [PUBLIC] | [35] |
| D4788 minimum contrast | "0.5 degC (0.9 degF) is the minimum contrast to distinguish between sound and defective concrete" | - | [PUBLIC, secondary] (search summary citing D4788-03; ACI 18-JI paper and WisDOT NDE chapter PDFs not text-extractable) | [36] |
| MBEI delamination classes the thermal output must land in | CS2 "Delaminated. Spall 1 in. or less deep or 6 in. or less in diameter"; CS3 "Spall greater than 1 in. deep or greater than 6 in. diameter" | - | [PUBLIC] | R05 section 3.2 |

Hardware anchor: DJI Matrice 4T, "$7,199.00", thermal "640 x 512", uncooled VOx, "Supports High-Res Mode" (Drone Nerds page; radiometric capability not stated on that page) [40]. Other dealers list $7,100-7,849 (July 2026) [40, search summary].

Inference: ASTM D4788's own disclaimer means a thermal delamination map is a **screening** output that must be labelled "not for acceptance/rejection", and the product should require a ground-truth hammer-sound or chain-drag sample before a CS3 spall/delamination is written to the export. Building-envelope thermography under ISO 6781-1 is qualitative; grade it S0-S2 (anomaly present / extent) and never S3-S4 from thermal alone.

---

## 4. Crack width measurement from imagery

### 4.1 The limits the number must be compared against

From `src/cascade/rubrics/bridge_mbei.json` [47]:

- line 7: `"cracking (reinforced concrete)", "CS1", "Width less than 0.012 in (insignificant cracks or moderate-width cracks that have been sealed)." -> S0, record`
- line 8: `"CS2", "Unsealed moderate-width cracks 0.012 to 0.05 in wide, or unsealed moderate pattern (map) cracking." -> S1, monitor`
- line 9: `"CS3", "Wide cracks greater than 0.05 in or heavy pattern (map) cracking." -> S2, schedule`
- lines 10-12: prestressed concrete CS1 "Width less than 0.004 in.", CS2 "Width 0.004 to 0.009 in.", CS3 "Width greater than 0.009 in." -> S0 / S1 / S2
- line 22 `unified_note`: "Crack-width rows require crack_width_mm from gsd_mm_per_px; without GSD, set not_measurable and grade from pattern, spall size class or exposed rebar only."

Metric equivalents (arithmetic): 0.004 in = 0.10 mm; 0.009 in = 0.23 mm; 0.012 in = 0.30 mm; 0.05 in = 1.27 mm.

ACI 224R-01 Table 4.1 "Guide to reasonable crack widths, reinforced concrete under service loads" [48, secondary: NCHRP Report 654 Appendix A summary; corroborated by the bold ACI graduations printed on the CRACKMON 224R comparator: .004/.006/.007/.012/.016 in and 0.10/0.15/0.18/0.30/0.40 mm [50]]:

| Exposure condition | Crack width |
|---|---|
| Dry air or protective membrane | 0.016 in (0.41 mm) |
| Humidity, moist air, soil | 0.012 in (0.30 mm) |
| De-icing chemicals | 0.007 in (0.18 mm) |
| Seawater and seawater spray, wetting and drying | 0.006 in (0.15 mm) |
| Water-retaining structures | 0.004 in (0.10 mm) |

ACI's caveat (as summarised): these values "are not always a reliable indication of steel corrosion and deterioration of concrete to be expected" [48, secondary]. ACI FAQ 855: ACI 224.1R-07 names two field devices, "a crack comparator (hand-held microscope with scale) and a clear card with various width lines"; "A crack comparator card generally has sufficient accuracy to determine crack widths for repairs since the recommended limit for epoxy injecting a crack is 0.01 in. (0.25 mm)"; feeler gauges rejected because "Concrete cracks typically form an irregular random surface" [51]. Comparator-card accuracy claims of "5-10um" are Amazon listings, i.e. marketing [50, search summary]. NYC EDC underwater/topside vocabulary (Hairline <1/32 in, Fine 1/32-1/16, Medium 1/16-1/8, Wide >1/8 in) is in R05 section 5.2.

### 4.2 Published image-based methods and their accuracy

| Method (scale reference) | Reported accuracy | Setup | Date | Source |
|---|---|---|---|---|
| Planar markers on the surface + total station coordinates; YOLOv4 + morphological edge detection | "width measurements as precise as 0.22 mm"; YOLOv4 mAP 92 % | DJI Mavic 2 Pro, 5472 x 3648, aerial shots "from a distance of 1 m", phone shots 50-70 cm; 379 site photos + SDNET2018 | 2023-02-25 (Sensors) | [43] |
| Laser calibration (known-geometry laser spots) + CNN/U-Net | "mean absolute error observed for crack width measurement was 0.16 mm"; classification 99.22 %, segmentation 96.54 % | lab and field images | 2024-02-05 (Metrology) | [44] |
| Ground control points + tie points, block triangulation, image registration | "0.1 mm pixel size for crack size measurements" on close-ups | Ai-He bridge, Taiwan; UAV | 2023-05-25 (Drones) | [45] |
| UAV-mounted laser cross projector + 3 ToF distance sensors (no external scale) | coefficient of variation of width: fine "from 0.36 to 0.10", medium "from 0.27 to 0.11", larger "from 0.22 to 0.07"; mapped to AASHTO MBEI condition states; field tests Peru and Canada | drone-agnostic module | 2026-08-21 (Metrology) | [46] |
| Skeleton pruning + edge-OrthoBoundary width | "smallest RMS, MAE" vs OP/OrthoBoundary/ESD; numbers not in abstract | real + synthetic cracks | 2025-07-16 (Buildings) | [41] |
| Parallel laser line-camera, sub-pixel "Equal Area" width | accuracy claimed, numbers not in abstract | handheld, non-perpendicular shots | 2025-01-16 (CACIE) | [42] |
| Fiducial markers via Harris corners, GSD-based | fine cracks "approximately 0.3 mm or less" estimated; "the number of pixels rather than the ground sample distance was the dominant factor" | - | [PUBLIC, secondary] search snippet, paper not identified | [43, search summary] |
| Planar markers (other UAV study) | "precise down to 0.53 mm" | - | [PUBLIC, secondary] search snippet | [46, search summary] |
| Neighbourhood shortest distance (synthetic cracks) | "RMSE of 0.24 mm with average absolute deviation of 0.21 mm" | synthetic | [PUBLIC, secondary] search snippet (ScienceDirect, not fetched) | [41, search summary] |

Public data with a pixel-to-mm scale: **none found.** dacl10k (CC BY-NC 4.0, 9,920 images, >100 bridges, average 1950 x 1581 px, images 2000-2020, half from engineering offices and half from German authorities) documents no rulers or scale bars; the paper states "the minimum crack width, which must be documented, is 0.2mm" and "Damage size is measured with a pocket rule, thus, it's imprecise" [49]. SDNET2018 lists crack widths 0.06-25 mm but ships no per-image scale (R06). So the team's `data/eval_v1` remains scale-less and crack width stays `not_measurable` on it (the frozen set is not touched).

### 4.3 What the numbers imply for the implementation [Inference]

- Resolution budget: to place a crack in the RC CS1/CS2 band (0.30 mm boundary) the width must span several pixels. With 3 px minimum, GSD must be <= 0.10 mm/px; a 5472 px-wide sensor then covers about 0.55 m of wall per frame. The PSC boundary (0.10 mm) needs GSD <= 0.033 mm/px, i.e. macro or comparator-card imaging. A 20 MP drone frame at 5 m stand-off (GSD about 1 mm/px) cannot measure any MBEI crack class; it can only detect presence. These are arithmetic consequences of the thresholds above, not measured results.
- Scale sources in priority order: (1) fiducial marker of known size (ArUco/checkerboard/comparator card) detected in-frame; (2) a ruler or tape with legible graduations; (3) camera model plus measured stand-off (EXIF focal length and a distance reading); (4) none -> `not_measurable`. Published scale-referenced methods report 0.22 mm precision (planar markers, 1 m range [43]) and 0.16 mm mean absolute error (laser calibration, not a marker [44]); neither is an error figure for this pipeline's heuristic mask, which has none measured. Both are the same order as the 0.30 mm CS1/CS2 boundary, so every width must carry an uncertainty and the grade must be "CS2 (0.35 +/- 0.2 mm, spans CS1/CS2)" style, never a bare number.
- Widths measured perpendicular to the crack skeleton (orthogonal boundary / shortest edge distance) are the accepted practice [41]; oblique photographs need the pixel scale corrected for the viewing angle [42], otherwise widths read high (the reviewed studies note "cracks being generally measured as wider than they actually are" [46, search summary]).
- Environment is not in the image: ACI 224R's exposure class (dry / humid / de-icing / seawater) must be a field on the asset record, defaulting to the strictest plausible class for a bridge in a de-icing state (0.18 mm) - team assumption.

---

## 5. Lidar and 3D reconstruction

| Claim | Value | Source |
|---|---|---|
| TLS deflection accuracy vs reference | Trimble TX8 "+/-2 mm" declared, results "quite consistent with the reference method (tachymetry)"; tachymetry "1-2 mm real accuracy"; photogrammetry about 1 px (4 mm at 120 m) and "higher by 18-20%"; FARO Focus 3D point cloud "weak" at 120 m; 165 m suspension span over the Odra (Materials, 2020-04-18) | [52] |
| TLS vs LVDT on beams | "less than 1 mm and within 1.6% of those measured directly by LVDT"; scanner "accuracy of +/- 2 mm for distances between 10 and 25 m" (2026 Scientific Reports beam study) | [PUBLIC, secondary] search snippets; Springer/Nature blocked | [54] |
| Scanner purchase | "between $100,000 and $500,000" (high-end terrestrial); another iScano page: "$20,000 to $50,000" for typical terrestrial systems | [53] (2025-08-10); [53, search summary] |
| Scan services | "hourly rates typically range from $150 to $500"; QC inspections "$1,000 and $4,000 per inspection"; progress monitoring "$1,500 and $5,000 per monitoring session"; scan-to-BIM "$0.50 and $10.00 per square foot" | [53] |
| Underwater 3D | Voyis VSLAM 2.5 cm voxels (R03) | R03 |

What lidar adds that imagery cannot: vertical deflection under load vs no-load, global geometry drift, and volumetric section loss from repeated scans (differencing) [52][54]. Why it is roadmap for this team [Inference]: (i) the demo engine grades 2D crops with a VLM; point clouds need registration, differencing and meshing, and the environment has no OpenCV/scikit-image/Open3D and the rule says no new packages; (ii) the sensor is a five-figure capital item that no customer in the discovery list (R08) mentioned; (iii) Skydio's DOT case studies already stop at 3D twins (R00 finding 5), so a twin is table stakes, not a differentiator. The slide can say "geometry layer (lidar/photogrammetry) planned; today's demo grades imagery, thermal, sonar frames and vibration series".

---

## 6. Market framing for SW + HW

Public price anchors (everything else is quote-only):

| Component | Public price | Date | Source |
|---|---|---|---|
| Seismic/vibration node | Raspberry Shake RS4D $604.99 (DIY) / $784.99 (turnkey); RS1D $294.99 board | 2026-09-25 | [24] |
| Bare sensors | ADXL355 $28.25 @1k (2016); SM-24 geophone $59.95 | 2016 / 2026 | [26][27] |
| Thermal + RGB drone | DJI Matrice 4T $7,199 (range $7,100-7,849 across dealers) | 2026 | [40] |
| Mini ROV | Deep Trekker DTG3 $8,500; Blueye X3 from $30,788 ex VAT | 2025-26 | R03 |
| Imaging sonar | ARIS / Oculus / BlueView: not published | - | [6][12] |
| Terrestrial lidar | $100k-500k (high-end), $20k-50k (typical) | 2025 | [53] |
| Structural monitoring platforms | Move Solutions, Worldsensing, Resensys, Bentley iTwin IoT (sensemetrics, acquired 2021-04-29): no prices | 2021-26 | [28]-[31] |
| Service anchors already sourced | Notilo hull report from EUR 1,000 per inspection; UWILD $5k-15k (blog); Gecko RaaS "tied to asset coverage" | R03 | R03 |

Bottom-up kit BOM [Inference, from the public prices above; excludes sonar, lidar, mounts, cases, shipping, integration labour]:

| Kit | Items | Sum |
|---|---|---|
| Budget | RS4D DIY $604.99 + Matrice 4T $7,199 + DTG3 $8,500 | about $16,300 |
| Standard | RS4D turnkey $784.99 + Matrice 4T $7,199 + Blueye X3 $30,788 | about $38,800 |

Inference on the business framing: every SHM incumbent behaves like the imagery incumbents in R01/R02 (quote-only, single-sensor). A SW-plus-kit offer priced per asset per year, with the kit at cost-plus and the grading/prioritisation subscription as the margin, mirrors Gecko's coverage-based pricing (R03) without Gecko's capex. Do not put a subscription price on a slide until a discovery interview supports it (R08 gap list).

---

## 7. Key numbers safe for slides

| Number | Use on slide | Tag | Source |
|---|---|---|---|
| Underwater inspection at most every 60 months; 24 months when Scour Condition Rating <= 3; up to 72 months with FHWA approval when >= 6 | Why now / cadence | [PUBLIC] | [1] |
| FHWA 2024-04-05: imaging "can supplement" Level I; "The Level II portion of the UWI is still to be performed by an underwater bridge inspection diver" | Positioning (assist, not replace) | [PUBLIC] | [5] |
| ARIS Explorer 3000: 3 MHz to 5 m, 1.8 MHz to 15 m, 128 beams, 3-19 mm downrange resolution | Sensor capability | [PUBLIC] | [6] |
| No public sonar dataset of bridge piers/piles with condition labels found (2025 survey lists debris, pipeline, wreck sets) | Data moat / honesty | [PUBLIC] | [8][11] |
| PEER NGA-West2: 21,336 records, 599 events; 200 records per two weeks; accounts reset 2026-07-02 | Data availability | [PUBLIC] | [18][19] |
| Z24: 15 progressive damage scenarios, 100 Hz, non-commercial licence | Benchmark | [PUBLIC] | [20] |
| ISO 20816-3:2022 Group 2 rigid zones 1.4 / 2.8 / 4.5 mm/s | Machinery rubric | [PUBLIC, secondary] | [23] |
| Hazus W1 High-Code drift 0.004 / 0.012 / 0.040 / 0.100 | Building rubric | [PUBLIC, secondary] | [17] |
| ShakeMap PGA about 11.5 %g = light damage, about 40 %g = moderate/heavy | Site shaking rubric | [PUBLIC, secondary] | [16] |
| Raspberry Shake RS4D $784.99 turnkey / $604.99 DIY; DJI Matrice 4T $7,199 | Kit cost | [PUBLIC] | [24][40] |
| ASTM D4788: overlays to 4 in; "should not be used for acceptance or rejection" | Thermal caveat | [PUBLIC] | [35] |
| ISO 6781-1:2023 replaces ISO 6781:1983 for building envelopes | Standards coverage | [PUBLIC] | [32][33] |
| MBEI RC crack 0.012 / 0.05 in (0.30 / 1.27 mm); PSC 0.004 / 0.009 in | Crack rubric | [PUBLIC] | [47], R05 |
| ACI 224R-01 tolerable widths 0.016 / 0.012 / 0.007 / 0.006 / 0.004 in by exposure | Crack rubric | [PUBLIC, secondary] | [48][50] |
| Image crack width with a scale reference: 0.22 mm (markers, 2023), 0.16 mm MAE (laser, 2024) | Feasibility | [PUBLIC] | [43][44] |
| dacl10k: "Damage size is measured with a pocket rule, thus, it's imprecise"; 0.2 mm minimum documented width | Problem | [PUBLIC] | [49] |
| TLS +/-2 mm vs tachymetry 1-2 mm; scanners $100k-500k | Lidar roadmap | [PUBLIC] | [52][53] |

## 8. Numbers to avoid

- Any single "x % frequency drop = damage" figure: the >5 % tolerance, the 5-10 % normal swing and the 0.37-1.4 % controlled drops are all search-snippet values from different papers and contexts [13][14][15]; Salawu 1997 could not be read [25].
- Sonar or SHM vendor prices: none are public; do not infer from ROV prices.
- Comparator-card "5-10 um accuracy": Amazon listing text [50].
- The Z24 "5.02 to 4.72 Hz" and "30 % stiffness" figures until the primary paper is read [20, secondary].
- ASTM C1153 edition year and ASTM D4788's 0.5 degC contrast: secondary only [34][36].
- "dacl10k has rulers in some photos": not supported by the paper [49].
- The whiteboard's "16%" and "Reconstruction": no readable context; not a number.
- Any crack width from `data/eval_v1`: the frozen set carries no scale, so widths there would be fabricated.

## 9. Gaps and open questions for the team

1. Level II cleaning percentage and band size from FHWA-NHI-23-027 (Sept 2024) - PDF not text-extractable here [4].
2. SNBI B.C.15 Underwater Inspection Condition code descriptions - SNBI PDF exceeds fetch limit; read Table for B.C.15 locally [7].
3. Sonar price points (ARIS, Oculus, BlueView, Kongsberg) - ask a distributor; needed for the kit BOM.
4. Hazus 6.1 Table 5-19 - open the manual locally and copy the row for the building types the demo will show [17].
5. ShakeMap intensity table - confirm against the ShakeMap manual (page moved) [16].
6. Which frequency-drop percentage the team adopts per asset class, and whether temperature compensation is in scope for the weekend (team assumption to be written into the rubric JSON with a `source: "team assumption"` field).
7. Whether the demo will show a real seismic series (CESMD/NGA-West2 registration lead time and the 2026-07-02 membership reset) or a synthetic series labelled as such [18][21].
8. Which scale reference the drone team (Leena) will actually carry: comparator card, ArUco board, or tape; the module below supports all three but the demo images must contain one.
9. Exposure class per asset for ACI 224R - field on the asset record; default to be agreed.
10. Interior machinery (Jie): ISO 20816-3 needs machine power class and support type as metadata; confirm those fields exist in the asset schema.
11. Marine Debris FLS licence conflict (CC0 on arXiv vs CC BY-NC-SA 4.0 on Zenodo) - treat as NC for the demo [8][9].

## 10. Notes for implementers

- Crack module contract (Inference from sections 4.1-4.3): input = image + optional scale hint {marker_mm, marker_px} or {ruler_px_per_mm} or {focal_mm, sensor_w_mm, distance_m}; output = {crack_width_mm, width_px, gsd_mm_per_px, uncertainty_mm, scale_source, not_measurable}. Grade only when `crack_width_mm - uncertainty_mm` and `+ uncertainty_mm` fall in the same MBEI state; otherwise report the two candidate states and keep U out of S0 (project rule).
- Store `exposure_class` (ACI 224R) and `element_type` (RC vs PSC) on the asset; the PSC ladder is three times tighter.
- Sonar frames: pass range, frequency and window as text next to the rendered frame (same pattern as R06 thermal statistics); rubric is extent/geometry, not width.
- Vibration series: the rubric fields are PGA %g, PGV cm/s, drift ratio, RMS velocity mm/s (10-1000 Hz), frequency-drop % vs stored baseline; each row cites [16][17][23] or "team assumption".
- UI: a "scale detected: yes/no" badge on every image card, and a "width +/- uncertainty" chip that never shows a bare millimetre value.

---

## Sources

1. https://ecfr.io/Title-23/Section-650.311 - 23 CFR 650.311 mirror (eCFR/Federal Register blocked); paragraphs (b)(1)(i)-(iii).
2. https://www.ecfr.gov/current/title-23/chapter-I/subchapter-G/part-650/subpart-C/section-650.311 - official eCFR page (redirected to a block page in this session; cited via search summary).
3. https://www.txdot.gov/manuals/brg/ins/chapter-4-field-inspection-requirements-and-proced/section-9-underwater-inspections.html - TxDOT Bridge Inspection Manual, Section 9 Underwater Inspections (Level I/II/III, NBI items 60/61/113, fathometer).
4. https://www.fhwa.dot.gov/bridge/nbis/pubs/nhi23027.pdf - FHWA-NHI-23-027 Underwater Bridge Inspection Reference Manual (Sept 2024); predecessor https://www.fhwa.dot.gov/bridge/nbis/pubs/nhi10027.pdf (June 2010). Not text-extractable here.
5. https://www.fhwa.dot.gov/bridge/nbis2022/qanda/08.cfm - FHWA 2022 NBIS Q&A section 313 (Q313-8 to Q313-11), updated 2024-04-05.
6. https://mfe-is.com/offshore/aris-explorer-3000/ - ARIS Explorer 3000 specifications (fetched 2026-09-25). Manufacturer page http://www.soundmetrics.com/products/aris-sonars/aris-explorer-3000 refused connection.
7. https://www.geocadra.com/en/standards/fhwa-nbis-snbi - SNBI summary (B.C.09, B.C.11, critical-finding trigger); NYSDOT SNBI tool tips PDF (2024-11-18) https://www.dot.ny.gov/divisions/engineering/structures/repository/manuals/inspection/SNBI%20Component%20Condition%20Ratings%20Tool%20Tips%20and%20AASHTO%20Railings%20over%20buried%20structures%20(11-18-24).pdf (not text-extractable); SNBI PDF https://www.fhwa.dot.gov/bridge/snbi/snbi_march_2022_publication.pdf (R05).
8. https://zenodo.org/records/15101686 - Marine Debris FLS Datasets (2025-03-28, ARIS Explorer 3000, CC BY-NC-SA 4.0, 7.1 GB).
9. https://arxiv.org/abs/2503.22880 - The Marine Debris Forward-Looking Sonar Datasets (Oceans Brest 2025; arXiv page states CC0).
10. https://journals.sagepub.com/doi/10.1177/14759217241235637 - Ge, Singh, Sadhu, "Advanced deep learning framework for underwater object detection with multibeam forward-looking sonar", SHM journal, 2024-03-24 (abstract via Semantic Scholar API). UATD and Portoroz 2025 from the same search results (https://www.researchgate.net/publication/365942113_A_Dataset_with_Multibeam_Forward-Looking_Sonar_for_Underwater_Object_Detection ; https://arxiv.org/pdf/2606.23006).
11. https://arxiv.org/pdf/2510.03353 - Sonar Image Datasets: A Comprehensive Survey (2025-10-07), master table.
12. https://www.deeptrekker.com/shop/products/blueprint-oculus-multibeam-sonars-m750d - Oculus M750d (no price shown); BlueView M900 Mk2 leaflet https://www.teledynemarine.com/en-us/products/SiteAssets/BlueView/PLD20590-3%20Blueview%20M900-Mk2%20product%20leaflet.pdf (search summary only).
13. https://pmc.ncbi.nlm.nih.gov/articles/PMC7506568/ - Automated and Model-Free Bridge Damage Indicators with Simultaneous Multiparameter Modal Anomaly Detection, Sensors 2020-08-22 (sigma thresholds; UNR and Columbia bridges).
14. https://www.sciencedirect.com/science/article/pii/S2772991525000477 - digital-twin SHM framework for bridges (2025) - ">5 %" tolerance, search summary only; https://arxiv.org/pdf/2010.07026 - Crowdsourcing Bridge Vital Signs (5-10 % normal fluctuation, search summary only).
15. https://arxiv.org/abs/2408.03002 - Damage identification for bridges using machine learning: KW51 (2024-08-06, rev. 2024-09-25); https://pmc.ncbi.nlm.nih.gov/articles/PMC9227402/ - FRF + FE model updating (0.37-1.4 % figure attributed by search summary; not fetched).
16. USGS ShakeMap instrumental intensity table (Worden et al. 2012 GMICE) - values from search summary; pages tried: https://ghsc.code-pages.usgs.gov/esi/shakemap/manual4_0/tg_parameters.html (404), https://www.usgs.gov/media/images/instrumental-intensity-shakemap (no legend text), https://www.intensitylab.com/mmi-scale/ (cites "USGS ShakeMap V4 Technical Documentation (Worden et al.)").
17. https://www.fema.gov/sites/default/files/documents/fema_hazus-earthquake-model-technical-manual-6-1.pdf - Hazus Earthquake Model Technical Manual 6.1 (July 2024), drift-ratio tables (PDF >10 MB, not fetched; values via search summary; Hazus 4.2 https://www.fema.gov/sites/default/files/2020-10/fema_hazus_earthquake_technical_manual_4-2.pdf Table 5-12).
18. https://ngawest2.berkeley.edu/ - PEER NGA-West2 online database: download limits, institutional e-mail, 2026-07-02 account reset, terms.
19. https://journals.sagepub.com/doi/10.1193/070913EQS197M - Ancheta et al., "NGA-West2 Database", Earthquake Spectra 2014 (21,336 records / 599 events; via search summary and https://daveboore.com/pubs_online/ngaw2_paper_ancheta_etal_database_eqs_2014.pdf).
20. https://bwk.kuleuven.be/bwm/z24 - Z24 Bridge benchmark, KU Leuven Structural Mechanics (description, sensors, damage scenarios, non-commercial terms). Frequency 5.02-4.72 Hz and 30 % stiffness figures: search summaries of https://www.researchgate.net/publication/2470889_Damage_Identification_On_The_Z24-Bridge_Using_Vibration_Monitoring and related pages (secondary).
21. https://www.strongmotioncenter.org/ - CESMD (USGS + CGS), structure-specific search, data attribution and use policy.
22. https://arxiv.org/pdf/2008.06724 - Damage Detection in Bridge Structures: An Edge Computing Approach (Tianjin Yonghe description, 14 accelerometers, 100 Hz, 2008 dates; via search summary); data page https://www.cece.ucf.edu/IASCM/events/smc-benchmark-problems-for-condition-assessment-and-damage-detection.
23. https://vibromera.eu/glossary/iso-20816-3/ - ISO 20816-3:2022 zone boundaries (vendor glossary; secondary).
24. https://raspberryshake.org/pricing - Raspberry Shake pricing page (fetched 2026-09-25).
25. https://www.sciencedirect.com/science/article/abs/pii/S0141029696001496 - Salawu 1997, Engineering Structures 19(9) 718-723 (abstract not retrievable; Semantic Scholar API returned no abstract).
26. https://www.eetasia.com/3-axis-mems-accelerometers-detect-structural-defects/ - ADXL354/355 launch article (25 ug/rtHz; $25.42 / $28.25 at 1k, 2016); datasheet https://www.analog.com/en/products/adxl355.html.
27. https://www.spikenzielabs.com/Catalog/sensors/envirnomental/geophone-sm-24 - SM-24 geophone $59.95; datasheet https://cdn.sparkfun.com/datasheets/Sensors/Accelerometers/SM-24%20Brochure.pdf.
28. https://www.movesolutions.it/solutions/bridges and https://www.movesolutions.it/sensors/deck002-x - Move Solutions (customized quotations; no prices).
29. https://www.worldsensing.com/structural-monitoring/ - Worldsensing (savings claims; no prices).
30. https://resensys.com/senspot.html - Resensys SenSpot (10-year battery; request quote).
31. https://www.businesswire.com/news/home/20210429006193/en/ - Bentley acquires sensemetrics and Vista Data Vision, 2021-04-29; https://www.bentley.com/software/itwin-iot/ (no prices).
32. https://www.iso.org/standard/79848.html - ISO 6781-1:2023 catalogue entry (page blocked; details via NBS/SIS listings https://www.thenbs.com/PublicationIndex/documents/details?Pub=BSI&DocId=340259 and https://quae.sis.se/en/produkter/construction-materials-and-building/protection-of-and-in-buildings/thermal-insulation-of-buildings/ss-en-iso-6781-12023/).
33. https://www.iso.org/standard/13277.html - ISO 6781:1983 (superseded).
34. https://www.thesnellgroup.com/featured-tips/infrared-related-standards-evaluating-building-thermal-performance - Snell Group summary of ISO 6781, ASTM C1060, ASTM C1153 (10 degC / 24 h; roofs at night); ASTM C1153 store page https://www.astm.org/c1153-10r15.html returned 403.
35. https://store.astm.org/d4788-03r22.html - ASTM D4788-03(2022) scope, overlay limit, precision-and-bias disclaimer.
36. https://www.concrete.org/Portals/0/Files/PDF/18-JI-Paper.pdf and https://wisconsindot.gov/dtsdManuals/strct/inspection/insp-fm-pt5ch11.pdf - sources for the 0.5 degC minimum contrast (search summary; PDFs not text-extractable).
37. (reserved)
38. (reserved)
39. (reserved)
40. https://www.dronenerds.com/products/dji-matrice-4t-thermal-enterprise-drone - DJI Matrice 4T $7,199.00, 640 x 512 thermal; other dealer prices https://globaldronehq.com/products/dji-matrice-4t-enterprise ($7,849).
41. https://www.mdpi.com/2075-5309/15/14/2489 - Buildings 2025-07-16, skeleton pruning + edge-OrthoBoundary crack width (abstract via Semantic Scholar API; MDPI 403). Neighbourhood shortest distance method: https://www.sciencedirect.com/science/article/abs/pii/S0141029624020819 (search summary).
42. https://onlinelibrary.wiley.com/doi/10.1111/mice.13420 - Li et al., parallel laser line-camera tiny crack width, CACIE 2025-01-16 (abstract via Semantic Scholar API).
43. https://pmc.ncbi.nlm.nih.gov/articles/PMC10007411/ - YOLOv4 + UAV crack quantization, Sensors 2023-02-25 (planar markers, total station, 0.22 mm, Mavic 2 Pro, 1 m).
44. https://www.mdpi.com/2673-8244/4/1/5 - Deep Learning for Concrete Crack Detection and Measurement, Metrology 2024-02-05 (laser calibration, MAE 0.16 mm; abstract via Semantic Scholar API).
45. https://www.mdpi.com/2504-446X/7/6/342 - Measurement of Cracks in Concrete Bridges by UAV and Image Registration, Drones 2023-05-25 (GCP block triangulation, 0.1 mm pixel size; abstract via Semantic Scholar API).
46. https://www.mdpi.com/2673-8244/6/3/58 - A Lightweight UAV-Mounted Metrology System for Standards-Aligned Metric Crack Width Measurement, Metrology 2026-08-21 (laser cross + ToF, CoV 0.36 to 0.10, MBEI-aligned; abstract via Semantic Scholar API). "0.53 mm" marker figure: https://link.springer.com/article/10.1007/s11227-025-07903-6 (search summary).
47. E:\origin_hack\src\cascade\rubrics\bridge_mbei.json - lines 7-12 (crack rows) and line 22 (unified_note); MBEI 2019 thresholds also in R05 section 3.2.
48. https://onlinepubs.trb.org/onlinepubs/nchrp/nchrp_rpt_654AppendixA.pdf - NCHRP Report 654 Appendix A literature review quoting ACI 224R-01 Table 4.1 (search summary; PDF not text-extractable); ACI 224R-01 chapter PDF https://wiki.opensourceecology.org/images/8/8c/ACI_224R-01_Control_of_Cracking_in_Concrete_Structures_f224R(01)Chap3.pdf (not text-extractable).
49. https://ar5iv.labs.arxiv.org/html/2309.00460 - dacl10k paper (WACV 2024) HTML rendering: licence, counts, resolution, "pocket rule" quote, 0.2 mm documentation minimum.
50. https://www.buildera.com/crackmon-224r-eim-crack-comparator - CRACKMON 224R comparator (bold ACI 224R-01 graduations .004/.006/.007/.012/.016 in; 88 graduations; no price). Amazon "5-10um" listings: https://www.amazon.com/Transparent-Comparator-Inspection-Precision-Measuring/dp/B0H33B8W79 (marketing).
51. https://www.concrete.org/tools/frequentlyaskedquestions.aspx?faqid=855 - ACI FAQ 855, alternative methods for measuring crack width (ACI 224.1R-07 devices, 0.01 in epoxy limit, feeler gauges).
52. https://pmc.ncbi.nlm.nih.gov/articles/PMC7215276/ - Comparison of Non-Destructive Techniques for Technological Bridge Deflection Testing, Materials 2020-04-18 (tachymetry 1-2 mm; Trimble TX8 +/-2 mm; photogrammetry 18-20 % high; 165 m span).
53. https://iscano.com/laser-scanning-lidar-best-practices/3d-laser-scanning-cost-guide-2025/ - 3D Laser Scanning Cost Guide 2025 (2025-08-10); equipment comparison https://iscano.com/laser-scanning-lidar-technology/lidar-equipment-comparison-guide/ ($20k-50k, search summary).
54. https://www.nature.com/articles/s41598-026-68044-1 - TLS on high-strength concrete beams, Scientific Reports 2026 (+/-2 mm at 10-25 m; auth redirect, search summary only); https://www.researchgate.net/publication/220373604_A_New_Approach_for_Health_Monitoring_of_Structures_Terrestrial_Laser_Scanning ("within 1.6% of LVDT", search summary).
