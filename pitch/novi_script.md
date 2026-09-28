# Pavilion Cerebro · 3-minute pitch script and presenter briefing

Six slides from `NOVI_infra.pdf`. The presenter is the business lead, so the spoken script uses plain words and only light technical terms.
Every technical word is explained under "Explain it simply" for that slide. Read those before rehearsing.

- **Timing:** the spoken script is about 415 words, which is about 2:52 at 145 words a minute.
- **Markup:** words in **bold** get the stress, and `[pause]` means one beat.
- **Presenter view:** `novi_pitch.html` shows the slides with this script. Press N to hide the notes and E to show the explainer.

---

## Slide 1 · Title (0:00 to 0:25)

### Say

Buildings fail **quietly**. Wiring overheats. A chiller slowly leaks refrigerant. A crack keeps growing. Nobody notices until the **yearly inspection**, or until something **breaks**.
Electrical faults alone start about **39,000 fires** a year in the US. And only about **one in ten** commercial buildings has any automation.
We're NOVI-INFRA, and this is **Pavilion Cerebro**: one brain for the whole building.

### Explain it simply

- **What Cerebro is.** Cerebro is software. It takes data the building already produces (photos, meters, the control system) and hands it to small AI specialists called **agents**. The agents report to one **coordinator**, the "brain", which ranks every problem and triggers a response.
- **Chiller.** A large machine that makes the cold water for a building's air conditioning. If it fails, the whole building gets hot.
- **"Automated."** This means the building has a **BMS** (building management system), the computer that runs heating, cooling and lighting. About 90% of US commercial buildings don't have one (source: PNNL, a US national lab). So most buildings have no digital eyes at all.
- **The picture.** It shows the demo building with 11 kinds of issues tagged:
  - STR: structure (columns, slabs)
  - ENV: envelope, meaning the facade and outer walls
  - ELEC: switchgear, the big electrical panels
  - WIRE: circuits
  - HVAC: heating, ventilation and air conditioning
  - H2O: pipes and leaks
  - FIRE: sprinklers
  - LIFT: elevators
  - AIR: indoor air quality
  - PWR: backup batteries
  - CYB: cyber-security of the building's controls

---

## Slide 2 · What we heard and found (0:25 to 0:55)

### Say

We talked to **eleven** practitioners, engineers and architects. We heard three things.
Buildings get checked **once a year**, and problems grow in between. Every system has its **own tool**, so nobody sees the whole building. And the rules now want **proof**: the NFPA 70B electrical-maintenance standard, and Florida's milestone inspection law.
Smart buildings are a **121-billion-dollar** market growing 9% a year, yet each big player watches only its own slice.

### Explain it simply

- **NFPA 70B.** This is the US standard for maintaining electrical equipment. Since its 2023 edition it is a mandatory standard, not just advice. Buildings must run a documented electrical maintenance program, so inspectors and insurers can ask for **proof**.
- **Florida milestone law.** Florida passed it after the 2021 Surfside condo collapse. Older condo and co-op buildings of three or more storeys must pass structural "milestone" inspections, and then repeat them every 10 years.
- **TAM, SAM and SOM.** These are the three sizes of the market, from biggest to smallest:
  - **TAM, $121.6B (total market):** the whole smart-building market worldwide, according to MarketsandMarkets. It is forecast to grow about 9% a year to 2032.
  - **SAM, $7.1B a year (the part we can serve):** about 5.9 million US commercial buildings (EIA CBECS survey) × the roughly 10% that are automated × $12,000 a year each.
  - **SOM, $36M a year (what we can realistically win):** about 3,000 buildings × $12,000.
- **Competitors in one line each:**
  - Johnson Controls (JCI), Siemens and Honeywell sell building control systems, tied to their own equipment.
  - BrainBox AI does AI for heating and cooling energy. Trane now owns it.
  - Eaton sells electrical equipment and monitors it.
  - Ting is a plug-in fire-risk sensor for homes.
  - None of them looks across every system in a building.

---

## Slide 3 · Agents sense, one brain acts (0:55 to 1:40)

### Say

So this weekend we built it. **Agents sense. One brain acts.** Each agent is a specialist for one system.
Agent zero spots **cracks** in photos. Agent one checks **concrete health** and the building's control data. Agent two watches **batteries** wear out, and it warned about failing cells more than **30 charge cycles** early. Agent three catches **chiller** faults, and it names the right fault **97%** of the time.
They report to **one brain**, which ranks every problem by **risk, severity and urgency**. When something turns critical, it acts: slow a battery's charging, switch to a backup chiller, page the team.
The data is **real**, replayed live. The building and the actions are **simulated**.

### Explain it simply

- **Agent.** A small program that watches one kind of data and outputs a **risk score** from 0 to 1, with the evidence behind it.
- **Agent 0, cracks.** It looks at a photo and decides whether there is a crack. The offline version uses classic image processing and was 92.9% accurate on 1,600 test photos. Those photos are lab concrete, not real building walls yet.
- **Agent 1, concrete and BMS.** Two data sources:
  - **UPV** (ultrasonic pulse velocity) sends a sound pulse through concrete. A slower pulse means weaker or damaged concrete, a bit like knocking on a melon. The agent caught 10 out of 10 clearly weak readings.
  - **BMS data** means temperatures and air-conditioning power. On real 2018 data from an office in Bangkok, the agent found a real AC failure without being told where to look.
- **Agent 2, batteries.** NASA published data from batteries that were charged and drained until they wore out. The agent predicts when a battery reaches end of life, and it flagged 3 out of 3 failing cells 32 to 43 cycles early. These batteries never caught fire, so it detects **warning signs**, not fires.
- **Agent 3, chiller.** ASHRAE is the US engineering society for heating and cooling. Its public test data comes from a real chiller with faults deliberately introduced.
  - It raised false alarms 1.5% of the time and named the right fault 96.9% of the time.
  - It caught 20 of 28 faulty test runs. The ones it misses are the mildest leaks.
- **How the brain ranks problems.** Priority = risk × consequence × time:
  - **Risk:** how likely the problem is.
  - **Consequence:** how bad it would be. A battery fire outranks a warm office. These weights are our own assumptions.
  - **Time:** how urgent it is. A problem that stays open gets more urgent.
- **"Acts" today.** The actions are entries in the dashboard log. No real building is connected. In a real building, Cerebro would send commands through the building's control system.
- **"Replayed live."** Real recorded data played back as if it were streaming right now.

---

## Slide 4 · One brain for the whole building (1:40 to 2:05)

### Say

Why does this matter? Today, building-control software sees heating, cooling and power. Drones see the facade. Electrical tools see the wiring. The yearly inspection sees everything, but only **once a year**.
Cerebro gives one **ranked to-do list**, catches problems **early**, and keeps an **audit-ready** record. Every new system is just **one more agent**.

### Explain it simply

- **How to read the grid.** Each row is a type of tool, and each column is a building system.
  - A solid square means the tool watches that system all the time.
  - A half square means it watches partly, or once a year.
  - An empty square means it is blind to that system.
- **The Cerebro row.** It shows where the platform is **going**, not what is built today. Today there are 4 agents: cracks, concrete, batteries and the chiller. If someone asks, say: "Every extra system is one more agent plugged into the same brain. We don't have to rebuild anything."
- **Audit-ready.** Every alert and action is saved with a time and its evidence. That is the proof NFPA 70B and insurers ask for. Don't call it "tamper-proof", because it isn't yet.

---

## Slide 5 · Business model (2:05 to 2:30)

### Say

Our hypothesis is a subscription at about **a thousand dollars per building per month**, a price we still need to test. A hundred buildings means **1.2 million dollars** a year in recurring revenue.
Owners, REITs and facility-management firms pay for **fewer emergency repairs** and **one team covering more buildings**. We reach our first hundred through **five pilot partners**, then building-automation installers, Florida inspection firms and insurers.

### Explain it simply

- **ARR (annual recurring revenue).** $1,000 × 12 months × 100 buildings = $1.2M a year.
- **REIT.** A company that owns many buildings and is listed like a stock.
- **Facility-management (FM) firm.** A company paid to run buildings for owners.
- **BMS integrators ("building-automation installers").** Companies that install and service building control systems. They already have the data and the customer relationship, so we share revenue with them.
- **Florida inspection firms.** They do the legally required milestone inspections. Cerebro gives them data between visits.
- **Insurers.** Fewer fires and water leaks mean fewer claims, so they may offer premium discounts to buildings that run Cerebro.
- **Watch the unit.** The slide says "first 100 customers", but the price is per building, and one customer may own many buildings. If asked, say "100 buildings".

---

## Slide 6 · What we'd validate next (2:30 to 2:50)

### Say

Next is a **30-day pilot** on one real building, a price test with ten facility managers, and civil engineers checking our crack model.
We know the open questions: data access, real-world accuracy and liability.
This weekend we built the brain. [pause] Now we want to put it in a **real building**.
We're NOVI-INFRA. **Thank you.**

### Explain it simply

- **Pilot.** Connect Cerebro to one real building for 30 days, then compare what it catches with what the facility team already knew.
- **Plumbing and fire agents** (on the slide, not spoken). Plumbing has no public dataset, so it needs a partner building's data.
- **Open assumptions:**
  - **Data access:** will owners share their control-system data?
  - **Real-world accuracy:** everything so far was measured on public test data.
  - **Who buys:** the owner or the FM firm?
  - **Liability:** who is responsible if Cerebro misses something or takes a wrong action?
  - **Long sales cycles:** commercial real estate buys slowly.

---

## Q&A cheat sheet (plain answers)

- **"Is this tested on a real building?"** Not yet. Every agent was measured on real public data. The building and the actions are simulated. The 30-day pilot is the next step.
- **"What if it's wrong?"** It ranks problems and raises alerts, and a person decides. Automatic actions would be limited to safe, reversible steps, like slowing a battery's charging.
- **"Why won't Honeywell or Siemens just do this?"** They sell their own equipment and watch their own slice. We are vendor-neutral and sit on top of everything.
- **"Does it predict fires?"** No. It spots **warning signs** early, such as a worn battery or hot equipment, so people can act before a fire.
- **"How do you get the data?"** From the building's control system, existing sensors and inspection photos. Integrators help us connect.
- **"How long does setup take? What does it cost to run?"** We don't know yet. The pilot will measure that. Don't guess a number.

## Don't say

- "We prevent fires" or "fire predicted".
- "Fully self-healing building" (the actions are simulated).
- "Tested in real buildings".
- "Customers said they'd pay" (price not yet tested).
- "It watches all 11 systems today" (4 agents are built).
- "Tamper-proof records".

## If you run long, cut these first

1. **Slide 2:** drop "yet each big player watches only its own slice" (saves about 3 s).
2. **Slide 3:** drop the battery and chiller numbers (saves about 8 s).
3. **Slide 4:** drop the sentence about control software, drones and electrical tools (saves about 7 s).

## Sources behind the numbers

- **Agent metrics:** the README on branch `origin/madhulika-feat`.
- **Smart-building TAM:** MarketsandMarkets, $121.57B in 2026 to $204.43B by 2032 at a 9.0% CAGR.
- **US commercial building count:** EIA CBECS 2018.
- **The 39K fires and $1.85B:** these are **not yet traced**. See the review notes.
