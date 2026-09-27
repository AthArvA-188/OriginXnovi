# Fire plan card: F19-E (F19 of 32)

**DECISION SUPPORT ONLY. Cerebro never controls, silences or overrides the fire alarm system, sprinklers, fire pumps, smoke control, elevator recall or Phase II, stair locks or HVAC. The Fire Safety Director (FSD) and then the fire department incident commander (IC) decide.**

**SYNTHETIC building** (generated tower and fire layer).


## Incident

| Item | Detail | Approved by | Basis |
|---|---|---|---|
| F19-E on F19 | F19 E office; floor 19 of 32 | - | SYNTHETIC building |

## Protect

| Item | Detail | Approved by | Basis |
|---|---|---|---|
| Alert floors: F18-F20 | code minimum is the fire floor, the floor above and the floor below | FSD | FIRE_ALERT_ABOVE: https://up.codes/s/emergency-voice-alarm-communication-systems |
| Relocation floors: F14-F16 | relocate occupants of F18-F20 down the evacuation stair to F14-F16; the approved Emergency Plan's predetermined relocation floors override this | FSD | FIRE_RELOCATE_MIN_BELOW: [team-proposed, validate] https://www.nist.gov/system/files/documents/el/fire_research/TN1664.pdf; FIRE_RELOCATE_FLOORS: [team-proposed, validate] |
| Evacuation stair: Stair A | the stair farther from the fire zone; suggestion only: the fire department designates stairs on arrival | IC | [team-proposed, validate] |
| Smoke watch: F21-F24 | read shaft and core sensors above the alert zone (building sensors, not listed smoke detectors) | - | FIRE_SMOKE_WATCH_ABOVE: [team-proposed, validate] https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir89-4035.pdf |

## Fire department

| Item | Detail | Approved by | Basis |
|---|---|---|---|
| Staging floor: F17 | staging at least two floors below the fire | IC | FIRE_STAGING_BELOW: [PUBLIC, secondary] https://srfecc.ca.gov/files/bf8df60b1/High+Rise+Operations.pdf |
| Attack stair: Stair B, standpipe hose connection at the F18 landing | equipment one floor below the fire; suggestion only: the fire department designates stairs on arrival | IC | FIRE_EQUIP_BELOW: [PUBLIC, secondary] https://srfecc.ca.gov/files/bf8df60b1/High+Rise+Operations.pdf; https://up.codes/s/location-of-class-i-standpipe-hose-connections |
| Fire service elevators FSAE-1, FSAE-2: Phase II, exit at F17 | fire department only; crews leave the car at least two floors below the fire | IC | FIRE_LIFT_EXIT_BELOW: [PUBLIC, secondary] https://srfecc.ca.gov/files/bf8df60b1/High+Rise+Operations.pdf |

## Elevators

| Item | Detail | Approved by | Basis |
|---|---|---|---|
| Passenger banks serving F19: HIGH | Phase I recall expected ONLY IF an elevator lobby or hoistway detector activates (LAFD matrix, footnote g); verify on the FACP. Never send staff up in an elevator to check an alarm. | - | https://lafd.org/fire-prevention/fire-development-services/policy-fire-life-safety-sequence-high-rise-buildings; https://lafire.com/famous_fires/1988-0504_1stInterstateFire/ExSummary/LAFD-ExecutiveSummary.htm |

## Electrical

| Item | Detail | Approved by | Basis |
|---|---|---|---|
| Panel P-F19; circuits C-F19-1, C-F19-2, C-F19-3, C-F19-4 | circuits feeding F19-E: C-F19-2; de-energize only by a qualified person on IC order | IC | building model feeds edges |
| Feeder path: P-F19 -> SWB-1 | panel to main switchboard | - | building model feeds edges |

## Water

| Item | Detail | Approved by | Basis |
|---|---|---|---|
| Risers on F19: R1@F19, L1@F19 | domestic riser and storm leader (not fire protection) | - | building model |
| Sprinkler floor control valve for F19: in Stair A | supervised valve per floor; operate only on IC order | IC | https://up.codes/s/floor-control-valves |
| Standpipes: Stair A, Stair B | hose connection at every floor landing | - | https://up.codes/s/location-of-class-i-standpipe-hose-connections |
| Firefighting water may migrate to 27 node(s) below | highest path scores: F18-E (0.60), F18-N (0.60), F18-S (0.60), F18-W (0.60), F17-E (0.36); plan salvage covers | - | WATER_WATCH_MAX_HOPS: [team-proposed, validate]; REACH_MIN [team-proposed, validate] |

## Hazards

| Item | Detail | Approved by | Basis |
|---|---|---|---|
| ess | Battery room ESS-1 (lithium-ion (SYNTHETIC)) in F01-ELEC: nobody opens the door; remote gas readings only (trigger E5). | - | ESS_NEAR_FLOORS: [team-proposed, validate] https://fsri.org/research-update/report-four-firefighters-injured-lithium-ion-battery-energy-storage-system |
| water near electrical | Firefighting water may reach electrical room F18-ELEC below the fire; isolation or salvage only on IC order by a qualified person. | - | WATER_WATCH_MAX_HOPS: [team-proposed, validate]; REACH_MIN [team-proposed, validate] |
| impairment | Open impairment elsewhere: sprinkler at F23 since 2026-09-20. SYNTHETIC: floor control valve at F23 closed for tenant fit-out; fire watch assigned | - | https://lafire.com/famous_fires/1988-0504_1stInterstateFire/ExSummary/LAFD-ExecutiveSummary.htm; https://www.nist.gov/el/interstate-bank-building-fire-los-angeles-1988 |
| roof | roof: emergency helicopter landing facility or approved equivalency (SYNTHETIC; LAFD Requirement No. 10) Helicopter use is a fire department decision. | - | https://lafd.org/sites/default/files/pdf_files/EHLF-Reg10.pdf |

## Occupants

| Item | Detail | Approved by | Basis |
|---|---|---|---|
| Design load on alert floors: 258 | F18 86, F19 86, F20 86 | - | OCC_LOAD_SQFT_PER_PERSON: [PUBLIC, secondary] https://www.houstonpermittingcenter.org/media/9311/download?inline= (design figure, not a headcount; use the warden count) |
| People needing help on alert floors: 2 | wardens confirm each one (trigger E6) | FSD | https://cityclerk.lacity.org/onlinedocs/2008/08-2476_ord_180648.pdf |

## Human decisions

| Item | Detail | Approved by | Basis |
|---|---|---|---|
| D0 (FSD) | AI early alert with no listed alarm: check by camera; if staff must look, they go by stair, never by elevator. Fire seen: pull the alarm and call 911. AI role: evidence only. | FSD | https://lafire.com/famous_fires/1988-0504_1stInterstateFire/ExSummary/LAFD-ExecutiveSummary.htm |
| D1 (FSD) | Call 911 (or designate someone). The AI never delays or replaces this call. AI role: evidence only. | FSD | https://cityclerk.lacity.org/onlinedocs/2008/08-2476_ord_180648.pdf |
| D2 (FSD) | Approve the protect strategy (alert and relocation floors, evacuation stair) and read one consistent voice message live. AI role: evidence only. | FSD | https://cityclerk.lacity.org/onlinedocs/2008/08-2476_ord_180648.pdf; https://content.nfpa.org/-/media/project/storefront/catalog/files/building-and-life-safety/highrise/emergencyactionplanhighrise.pdf?rev=9f5900973b264cc486ed9efa8643eb30; https://www.nist.gov/system/files/documents/el/fire_research/TN1664.pdf |
| D3 (IC) | Fire department incident commander takes command; building systems move only on IC orders; IC designates the attack and evacuation stairs. AI role: evidence only. | IC | https://content.nfpa.org/-/media/project/storefront/catalog/files/building-and-life-safety/highrise/emergencyactionplanhighrise.pdf?rev=9f5900973b264cc486ed9efa8643eb30; https://up.codes/s/fire-fighter-s-smoke-control-panel; https://srfecc.ca.gov/files/bf8df60b1/High+Rise+Operations.pdf |
| D4 (IC) | Widen the alert zone or order a total evacuation when an escalation trigger fires. AI role: evidence only. | IC | https://content.nfpa.org/-/media/project/storefront/catalog/files/building-and-life-safety/highrise/emergencyactionplanhighrise.pdf?rev=9f5900973b264cc486ed9efa8643eb30 |
| D5 (IC) | Authorize resets, fire watch where systems are impaired, and re-occupancy. AI role: evidence only. | IC | https://cityclerk.lacity.org/onlinedocs/2008/08-2476_ord_180648.pdf; https://content.nfpa.org/-/media/project/storefront/catalog/files/building-and-life-safety/highrise/emergencyactionplanhighrise.pdf?rev=9f5900973b264cc486ed9efa8643eb30 |

## Escalation triggers

| Id | Trigger | Status | Detail |
|---|---|---|---|
| E1 | An alarm or waterflow signal on a floor outside the alert set: add that floor and the floors next to it. | armed |  |
| E2 | An alarm on the fire floor +2 or higher: possible upward spread [team-proposed, validate]. | armed |  |
| E3 | Smoke reported in the evacuation stair: FSD and IC choose another stair. | armed |  |
| E4 | An open system impairment on any alert or relocation floor. | armed |  |
| E5 | An alarm in the battery (ESS) room: nobody opens the door; remote gas readings only. | armed |  |
| E6 | A person who needs help on an alert floor is not accounted for. | watch | 2 person(s) needing help on the alert floors |

## Verify on the fire alarm panel (automatic, listed controls; the AI does not run these)

- [ ] A1 Voice alarm on F18-F20 (alarm floor, floor above, floor below): any listed initiating device (https://lafd.org/fire-prevention/fire-development-services/policy-fire-life-safety-sequence-high-rise-buildings; https://up.codes/s/emergency-voice-alarm-communication-systems)
- [ ] A2 Elevator Phase I recall: ONLY IF an elevator lobby or hoistway detector activates (LAFD matrix row 'Recall All Elevators', footnote g); manual pull, area smoke, duct detector and waterflow do not recall elevators (https://lafd.org/fire-prevention/fire-development-services/policy-fire-life-safety-sequence-high-rise-buildings)
- [ ] A3 Smoke control and stair pressurization start: per the building's approved smoke-control matrix (https://lafd.org/fire-prevention/fire-development-services/policy-fire-life-safety-sequence-high-rise-buildings; https://up.codes/s/stairway-and-ramp-pressurization-alternative)
- [ ] A4 Air handlers shut down and dampers close: per the approved matrix (https://lafd.org/fire-prevention/fire-development-services/policy-fire-life-safety-sequence-high-rise-buildings)
- [ ] A5 Held doors release and stair doors unlock: per the approved matrix (https://lafd.org/fire-prevention/fire-development-services/policy-fire-life-safety-sequence-high-rise-buildings)
- [ ] A6 Annunciation at the fire control room and the 24-hour remote station: any listed initiating device (https://lafd.org/fire-prevention/fire-development-services/policy-fire-life-safety-sequence-high-rise-buildings)

Generated by cascade.building.fire.plan_for_incident (deterministic rules, no ML). Every action line has executes=False.