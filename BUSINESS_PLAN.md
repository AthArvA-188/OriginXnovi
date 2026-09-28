# Pavilion Cerebro business plan

> **Source of truth:** the pitch website (`index.html`, NOVI-INFRA / Pavilion Cerebro) has the full narrative, market sizing, competitive table, pricing and valuation math, with sources linked in its footer. This document is a prose summary of that site plus one reconciliation: **what the pitch describes vs. what's actually built** in this repo, so nobody presents a claim the code doesn't back up.

## The pitch, in one paragraph

The market sells point tools for pieces of a building — a BMS for HVAC, sensors for structure, drone inspection, elevator IoT — and no single product covers the whole building or acts on what it finds. Pavilion Cerebro is that missing layer: one swarm of AI agents (the "senses"), one coordinator (the "brain") that fuses every agent's report into one risk-ranked picture and triggers a response. The technical thesis carries over unchanged from the README: one shared agent interface, so adding a subsystem is writing one more agent, not rebuilding the platform.

## The full vision: 11 problem areas

The pitch names eleven problem areas a complete Pavilion Cerebro would watch, in three stages. Today **3 of them (STR, PWR, HVAC) are covered by 4 working agents on real public data**. Nothing beyond those agents is simulated with fake data.

| Stage | Code | Problem area | Status in this repo |
|---|---|---|---|
| **Built** | STR | Structure & materials (cracks, concrete strength) | Agent 0 (crack imagery) + Agent 1's UPV channel (internal concrete strength) |
| **Built** | PWR | Power & batteries | Agent 2 (battery SOH, RUL, fire-risk precursors); whole-building energy and demand peaks are future scope |
| **Built** | HVAC | HVAC & mechanical | Agent 3 (chiller fault detection & diagnosis); Agent 1 also reads BMS zone data |
| **Next** | H2O | Plumbing & water | Roadmap stub (agent-4). No public dataset, so it needs a partner building's data. Clog Watch on the smart-building site is a simulated prototype |
| **Next** | ENV | Envelope: facade and rain intrusion | Drone facade imagery and rain exposure prototypes on the smart-building site; not yet an agent |
| **Next** | ELEC | Panels & switchgear (heat, current, power quality) | Electrical checks in the smart-building console; not yet an agent |
| **Next** | WIRE | Wiring in walls (arc-fault, insulation) | Not yet an agent |
| **Future** | FIRE | Fire & life safety (sprinkler pressure, valve tamper, panel signals) | Future scope. The camera-based fire/smoke stub (agent-5) is a different sensing approach |
| **Future** | LIFT | Elevators (door cycles, motor, vibration) | Future scope |
| **Future** | AIR | Indoor air quality (CO₂, PM2.5, VOCs, radon) | Future scope |
| **Future** | CYBER | Building-network security | Future scope |
| — | — | Laser scan digital twin (LiDAR) | Roadmap stub (agent-6, hardware-dependent); a sensing *modality* that would mainly feed STR (see README for the design) |

Each new row is the same size of work: one loader, reusing the same `AgentReport` interface and, for numeric sensors, the same `agents/generic.py` scoring engine Agents 1–3 already share. None of this needs new coordinator or prioritization logic.

## Pricing (from the pitch)

- **Software only: $599 per building per month** ($7,188 a year), for buildings that already run building automation and cameras. Cerebro reads what is there.
- **Full kit, 3-year plan: $1,400 per building per month**, with 69 wireless sensors, 2 gateways, an edge computer and installation included.
- **Reference building** (100,000 sq ft office, 7 floors): the pitch estimates it ends up **$15,792 a year ahead** after the fee, from checks replaced, HVAC upkeep and energy savings, with a range of $2,727 (cautious) to $20,742 (strong). Our own figure comes from the pilot.

## Market (from the pitch; sources in its footer)

- **TAM:** $47.6B total addressable market.
- **SAM:** ~590,000 US commercial buildings that already run some automation × $7,188 a year → **$4.2B a year** (PNNL adoption study; EIA CBECS building count).
- **SOM:** 3,000 buildings by Year 5 (0.5% of SAM) → **$21.6M ARR**.
- Markets cited for context: smart buildings $121.6B → $204.4B (2026–2032, 9.0%/yr); predictive maintenance $13.9B → $23.8B (11.4%/yr); facility management software $3.8B → $9.6B (11.1%/yr); structural health monitoring $2.5B → $4.1B (10.4%/yr).

## Why the gap is real, not assumed

- **~10%** of US commercial buildings run any building automation at all today (PNNL). Most small and mid-size buildings run nothing.
- **~39,000** US fires a year start with electrical faults, about 31,600 in homes and 7,400 in commercial buildings (NFPA, via FM). A system that watches wiring and panels continuously addresses a named, sourced failure mode.
- Regulatory tailwinds are real and dated: Florida requires structural milestone inspections at 30 years, then every 10, for condos of 3+ stories; NFPA 70B (2023) requires every piece of electrical equipment to be inspected at least once a year. Both create a recurring, mandated need for condition data, which an always-on swarm produces as a byproduct.

## Customers (from the pitch)

Real estate owners and REITs, facility management firms, condo and HOA associations (Florida milestone inspections are a concrete forcing function), hospitals (downtime, fire and water are patient-safety issues), data centers (battery rooms and NFPA 70B), universities, government and public buildings, and insurers (State Farm already gives customers free electrical-fire sensors, a signal insurers will pay to prevent claims, not just price them).

## Valuation estimate (from the pitch; an internal estimate for discussion, not financial advice)

- **Today, pre-seed target:** $12M post-money, inside the $10–15M median 2025 SAFE cap range (Carta).
- **Year 5 target:** $97M–$175M, from $21.6M ARR × 4.5–8.1× (private SaaS median to top-quartile multiples, Aventis Advisors).
- **Comparables:** BrainBox AI (autonomous HVAC AI, 14,000+ buildings, acquired by Trane in 2025 after ~$82M raised, price undisclosed) and PassiveLogic (autonomous building platform, $74M Series C in September 2025, $125M+ total raised, Johnson Controls as an investor).

## What this repo can honestly claim today vs. what the pitch claims

- The pitch's competitive-gap table, market sizing, pricing and valuation math are the website's claims, sourced there. This repo doesn't independently re-derive them, and nobody should present them as measured by our own pipeline.
- What *is* measured by our own pipeline — detection accuracy, false-alarm rates, lead time before failure, re-triage speed — is in `data/samples/metrics.json` and the README's "Self-check results" table. Keep these two kinds of numbers clearly separate in a pitch: **market and business numbers are sourced from external research; product numbers are measured from our own code.**
- Only 3 of the 11 problem areas are covered today, by 4 agents on real public data. Say that plainly rather than let "11 problem areas" imply all 11 are built. The architecture supports it; the data and code for the other 8 don't exist yet as agents.

## Existing team research

`docs/research/04_market_pain_size_regulation.md` and `docs/research/09_business_models_gtm.md` were written for the original infrastructure-inspection cascade (bridges, wind, solar, utilities), not for the building-swarm pivot. Re-check any figure from there against the building market before reusing it in a Pavilion Cerebro pitch.
