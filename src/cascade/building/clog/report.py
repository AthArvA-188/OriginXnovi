"""Conclusions for the pipe-clog page, computed ONLY from this module's artifacts (no typed statistics).

Each conclusion is chosen by comparing measured values, so if a result changes on a rerun the wording follows it
(for example, which detector detects more, or whether passive monitoring beats its own false-alarm rate).
"""
from __future__ import annotations

from typing import Dict, List

from . import rules as R

DETECT_CRITERION = 0.9  # wording criterion: "flags at least 90% of tests"


def pct(x: float) -> str:
    return f"{100.0 * x:.0f}%"


def _span(r) -> str:
    return f"{r[0]}" if r[0] == r[1] else f"{r[0]}-{r[1]}"


def _rate(riser: dict, variant: str, det: str, sev: int) -> float:
    return riser["variants"][variant]["detectors"][det]["alarm"][str(sev)]["rate"]


def _overlap(a, b) -> bool:
    """True when two [lo, hi] intervals overlap; False when either is missing."""
    return bool(a and b and len(a) == 2 and len(b) == 2 and a[0] <= b[1] and b[0] <= a[1])


def baseline_text(riser: dict) -> str:
    """Naive grading baselines, named from the artifact (the majority class is the TRAINING majority)."""
    bl = riser.get("baselines_macro_f1")
    if not bl:
        return f"majority-class baseline {riser['baseline_majority_macro_f1']:.2f}"
    names = bl.get("class_names", ["clean", "mild", "moderate", "severe"])
    const = bl["constant_macro_f1"]
    others = ", ".join(f"always '{names[int(k)]}' {v:.2f}" for k, v in const.items() if int(k) != bl["majority_class"])
    return (f"always '{bl['majority_class_name']}' (the training majority) {bl['majority_macro_f1']:.2f}; "
            f"{others}; a random grade drawn with the training class frequencies "
            f"{bl['stratified_random']['mean']:.2f}")


def supply(riser: dict) -> List[Dict[str, str]]:
    out = []
    b = "base"
    qa, qb = riser["design"]["tests_lps"]
    n_loc = len(riser["design"]["locations"])
    fa4, m4, s4 = (_rate(riser, b, "act4_z", s) for s in (0, 2, 3))
    pas = {s: _rate(riser, b, "pas_hgb", s) for s in (0, 2, 3)}
    bms = {s: _rate(riser, b, "bms", s) for s in (0, 2, 3)}
    passive_blind = max(pas[2], pas[3]) <= pas[0] + 0.05 and max(bms[2], bms[3]) <= bms[0] + 0.05
    noise = riser["design"].get("noise", {}).get("pressure_sd_m", R.value("PRESSURE_NOISE_M"))
    busy = riser["variants"].get("ood_demand")
    busy_txt = ""
    if busy:
        bp0, bp3 = (busy["detectors"]["pas_hgb"]["alarm"][k]["rate"] for k in ("0", "3"))
        busy_txt = (f" In the stress test '{busy['label']}' passive (gradient boosting) flagged {pct(bp3)} of "
                    f"severe nights at {pct(bp0)} false alarms.")
    out.append({"topic": "Earliest supply-side signal", "text": (
        f"A short nightly flow test at a known flow is the signal that worked. On held-out simulated weeks the "
        f"{qa:.1f} L/s test flagged {pct(s4)} of severe-clog nights and {pct(m4)} of moderate-clog nights at "
        f"{pct(fa4)} false alarms per clean night. "
        + (f"In our simulation (10-minute readings, assumed pressure noise sd {noise} m, apartment-only demand) "
           f"passive 24-hour monitoring and a BMS low-pressure alarm did no better than their own false-alarm rate "
           f"(severe: {pct(pas[3])} and {pct(bms[3])}): everyday flows are too small to make a clog's head loss "
           f"stand out." if passive_blind else
           f"Passive monitoring flagged {pct(pas[3])} of severe nights and the BMS low-pressure alarm "
           f"{pct(bms[3])}.") + busy_txt), "label": "SIMULATED"})
    dz, dh = (riser["variants"][b]["detectors"][k] for k in ("act4_z", "act4_hgb"))
    zf1, hf1 = dz["severity"]["macro_f1"], dh["severity"]["macro_f1"]
    zm, hm = m4, _rate(riser, b, "act4_hgb", 2)
    zl, hl = dz["localisation"]["2"]["accuracy"], dh["localisation"]["2"]["accuracy"]
    ov_detect = _overlap(dz["alarm"]["2"].get("ci95"), dh["alarm"]["2"].get("ci95"))
    ov_grade = _overlap(dz["severity"].get("macro_f1_ci95"), dh["severity"].get("macro_f1_ci95"))
    ov_loc = _overlap(dz["localisation"]["2"].get("ci95"), dh["localisation"]["2"].get("ci95"))
    better_detect = "the plain z-score rule" if zm >= hm else "gradient boosting"
    better_grade = "gradient boosting" if hf1 > zf1 else "the plain z-score rule"
    better_loc = "the z-score rule's largest-deviation segment" if zl >= hl else "the gradient-boosting localiser"
    hfa = _rate(riser, b, "act4_hgb", 0)
    fa_hi, fa_lo = (fa4, hfa) if zm >= hm else (hfa, fa4)
    if ov_detect:
        detect_txt = (f"Detection: the plain z-score rule and gradient boosting caught moderate clogs at rates whose "
                      f"95% CIs overlap ({pct(zm)} vs {pct(hm)}, at {100 * fa4:.1f}% vs {100 * hfa:.1f}% false "
                      f"alarms per clean night). ")
    else:
        detect_txt = (f"Detection: {better_detect} caught more moderate clogs ({pct(max(zm, hm))} vs "
                      f"{pct(min(zm, hm))}, at {100 * fa_hi:.1f}% vs {100 * fa_lo:.1f}% false alarms per clean night). ")
    if ov_grade:
        grade_txt = (f"Grading into clean / mild / moderate / severe: gradient boosting and the z-score rule were about "
                     f"the same (macro-F1 {hf1:.2f} vs {zf1:.2f}, overlapping 95% CIs). ")
    else:
        grade_txt = (f"Grading into clean / mild / moderate / severe: {better_grade} scored higher (macro-F1 "
                     f"{max(hf1, zf1):.2f} vs {min(hf1, zf1):.2f}). ")
    grade_txt += f"Naive grading baselines: {baseline_text(riser)}. "
    if ov_loc:
        loc_txt = (f"Locating a moderate clog to one of {n_loc} places: the z-score rule's largest-deviation segment "
                   f"and the gradient-boosting localiser were about the same ({pct(zl)} vs {pct(hl)}, overlapping 95% "
                   f"CIs; chance {pct(1 / n_loc)}). ")
    else:
        loc_txt = (f"Locating a moderate clog to one of {n_loc} places: {better_loc} was right more often "
                   f"({pct(max(zl, hl))} vs {pct(min(zl, hl))}; chance {pct(1 / n_loc)}). ")
    act = [k for k in riser["variants"][b]["detectors"] if k.startswith("act")]
    mild_gap = max(_rate(riser, b, k, 1) - _rate(riser, b, k, 0) for k in act)
    mild_txt = (f"No method flagged mild clogs (K below {riser['design'].get('k_bands', R.value('K_BANDS'))['mild'][1]}) much more often than "
                f"clean weeks (at most {100 * mild_gap:.0f} percentage points above its own false-alarm rate)."
                if mild_gap <= 0.05 else
                f"The best method flagged mild clogs {100 * mild_gap:.0f} percentage points above its false-alarm "
                f"rate.")
    out.append({"topic": "What machine learning adds", "text": detect_txt + grade_txt + loc_txt + mild_txt,
                "label": "SIMULATED"})
    fa25, m25 = _rate(riser, b, "act25_z", 0), _rate(riser, b, "act25_z", 2)
    out.append({"topic": "Test flow", "text": (
        f"Head loss grows with flow squared, so the stack-safe {qb:.1f} L/s test is weaker: {pct(m25)} of moderate "
        f"clogs at {pct(fa25)} false alarms, against {pct(m4)} at {pct(fa4)} for {qa:.1f} L/s. A draw above "
        f"{R.value('DN100_STACK_MAX_LPS'):.1f} L/s should go to a break tank, not into a DN100 stack."), "label": "SIMULATED"})
    n2, off_a, off_p, aged = (_rate(riser, "ood_noise2", "act4_z", 0), _rate(riser, "ood_offset", "act4_z", 0),
                              _rate(riser, "ood_offset", "pas_z", 0), _rate(riser, "ood_aged", "act4_z", 0))
    cs = riser.get("commissioning_sensitivity", {}).get("act4_z", {}).get("0")
    cs_txt = (f" Redrawing the one-week commissioning reference {cs['n_draws']} times moved the {qa:.1f} L/s "
              f"false-alarm rate between "
              f"{pct(cs['min'])} and {pct(cs['max'])}.") if cs else ""
    def _move(new: float) -> str:
        if new - fa4 > 0.05:
            return f"pushed its false alarms from {pct(fa4)} to {pct(new)} per clean night"
        return f"barely moved its false alarms ({pct(fa4)} to {pct(new)} per clean night)"

    aged25, base25 = _rate(riser, "ood_aged", "act25_z", 0), fa25
    off_txt = (f"A constant offset on one pressure sensor cancels in the test's before/after jump by construction "
               f"(false alarms {pct(off_a)}, {'identical to' if abs(off_a - fa4) < 1e-12 else 'against'} "
               f"{pct(fa4)} without it), so that row checks the algebra rather than stressing the detector; a "
               f"drifting offset was not tested. The passive z-score rule, which has no such cancellation, jumped "
               f"to {pct(off_p)}.")
    out.append({"topic": "What breaks it (stress tests)", "text": (
        f"For the {qa:.1f} L/s test, twice the sensor noise {_move(n2)}; pipes aged since commissioning "
        f"{_move(aged)}, while the {qb:.1f} L/s test went from {pct(base25)} to {pct(aged25)}. "
        f"{off_txt}{cs_txt} So: recommission after "
        f"pipe or sensor work; a longer commissioning window may help but was not tested."), "label": "SIMULATED"})
    return out


def drains(drain: dict, bell: dict) -> List[Dict[str, str]]:
    rows = sorted(drain["rows"], key=lambda r: -r["opening"])
    first_dd = next((r for r in rows if r["detect_rate_drain_down"] >= DETECT_CRITERION), None)
    first_pk = next((r for r in rows if r["detect_rate_peak"] >= DETECT_CRITERION), None)
    vol = drain["model"]["test"]["lps"] * drain["model"]["test"]["seconds"]
    clean = next(r for r in rows if r["opening"] == 1.0)
    half = next((r for r in rows if r["opening"] == 0.5), None)
    out = []
    if first_dd and first_pk:
        txt = (f"In the simulated drain, drain-down time after a known {vol:.0f} L discharge flagged at least "
               f"{pct(DETECT_CRITERION)} of tests "
               f"once the clog left {pct(first_dd['opening'])} of the pipe open; peak level needed "
               f"{pct(first_pk['opening'])} open. ")
    else:
        txt = "In the simulated drain, "
    if half:
        txt += (f"At 50% open the model drain behaved like a clean one (flagged {pct(half['detect_rate_drain_down'])} "
                f"vs {pct(clean['detect_rate_drain_down'])} for a clean drain), so in this model the test cannot see "
                f"a clog that leaves half the pipe open.")
    out.append({"topic": "Earliest drain-side signal", "text": txt, "label": "SIMULATED"})
    h = bell["headline"]
    p = h["paired"]
    r = h.get("residual_only")
    fixed = sorted(h["fixed_levels"], key=lambda x: x["first_alarm"] or "9999")
    early = fixed[0]
    sw = bell["sweep"]["paired_summary"]
    res_txt = ""
    if r and r.get("detected"):
        res_txt = (f" The upstream level checked against its learned normal alone (residual only) alarmed "
                   f"{r['delay_min_from_onset']:.0f} minutes after onset with {r['other_test_episodes']} other held-out "
                   f"episodes and {r['train_episodes']} in the training months (out-of-fold).")
    out.append({"topic": "Real blockage (Bellinge, July 2020)", "text": (
        f"On the one documented real blockage, the paired rule (upstream above normal AND downstream starved) "
        f"alarmed {p['delay_min_from_onset']:.0f} minutes after the visible onset with {p['other_test_episodes']} "
        f"other alarm episodes in {bell['test_months']:.1f} held-out months and {p['train_episodes']} in the "
        f"training months.{res_txt} The fastest fixed high-level alarm ({early['label']}) was "
        f"{early['delay_min_from_onset']:.0f} minutes after onset but raised {early['other_test_episodes']} other "
        f"held-out episodes and {early['train_episodes']} in the training months. {sw['n_detected']} of "
        f"{sw['n_configs']} paired-rule settings caught it, with {_span(sw['other_test_episodes_range'])} other "
        f"held-out episodes. The learned-normal signals won on false alarms. One event in a municipal combined "
        f"sewer: a case study, not an accuracy."),
        "label": "REAL"})
    return out


def placement(riser: dict) -> Dict[str, str]:
    rows = riser["moderate_plus_by_location"]["act4_z"]
    lo = min(rows, key=lambda r: r["rate"])
    hi = max(rows, key=lambda r: r["rate"])
    return {"topic": "Where to put sensors", "text": (
        f"Supply: a clamp-on flow meter on each zone header; pressure at the booster, at each PRV station's strainer "
        f"inlet and outlet, and at the header, middle and top floor of every zone; one test valve at each zone top. "
        f"With this layout the {riser['design']['tests_lps'][0]:.1f} L/s test flagged {pct(lo['rate'])}-"
        f"{pct(hi['rate'])} of moderate-or-severe clog nights at each location (lowest: {lo['name']}; highest: "
        f"{hi['name']}; simulated, per night). Drains: a level sensor at each stack "
        f"base and main cleanout (for the drain-down test), a FOG probe at the grease interceptor, and a downstream "
        f"partner gauge where a pair is possible."), "label": "SIMULATED + design"}


def cadence(sched: dict) -> Dict[str, str]:
    rows = sched["rows"]
    k, n = R.value("PERSISTENCE_2_OF_3")
    soon = [r for r in rows if r["days_from_as_of"] <= 7]
    return {"topic": "How often to check", "text": (
        f"Regulatory maximums first: backflow assemblies every {R.value('BACKFLOW_ANNUAL')} days by an LA "
        f"County-licensed tester; grease interceptors before FOG and solids reach {R.value('FOG_25PCT')}% of the "
        f"liquid depth; grease traps daily. Supply risers: the nightly test, with a work order only after {k} alarms "
        f"in {n} nights. Everything else: a base interval "
        "shortened by consequence, backups and test results, as LA Sanitation does for city sewers. In the demo "
        f"register, {len(soon)} of {len(rows)} checks fall due within 7 days of {sched['as_of']}."),
        "label": "rules + SYNTHETIC register"}


def conclusions(riser: dict, drain: dict, bell: dict, sched: dict) -> List[Dict[str, str]]:
    return supply(riser) + drains(drain, bell) + [placement(riser), cadence(sched)]
