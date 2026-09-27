"""Home page of the Pavilion Cerebro local site.

Every number here is read at runtime from the result files the module scripts wrote (eval/<module>/*.json).
Nothing is typed in; a missing file shows "not yet run". Top-level Streamlit page: no st.set_page_config.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "eval"

LABEL_COLORS = {"REAL": "green", "SIMULATED": "orange", "SYNTHETIC": "orange", "INJECTED": "violet", "RULES": "blue"}


def _load(rel: str):
    p = EVAL / rel
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _get(d, *path):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x


# Each headline returns [(text, label), ...] built only from artifact values, or [] when the artifact is absent.
def facade_headline():
    t = _get(_load("facade/tilecls_v1.json"), "sets", "sdnet_test")
    m, h = _get(t, "models", "resnet18_ozgenel_sdnet"), _get(t, "models", "heuristic_crack_mask")
    if not (m and h and _num(m.get("recall")) and _num(m.get("auroc"))):
        return []
    return [(f"Finds {m['recall']:.0%} of cracked wall tiles it never saw (precision {m['precision']:.0%}, "
             f"AUROC {m['auroc']:.2f}; a simple rule-based detector scores AUROC {h['auroc']:.2f}). "
             f"Test: {t['n']:,} tiles from {t['groups']} held-out photos.", "REAL")]


NDT_NAMES = {"sonreb_powerlaw": "SonReb physics formula", "hgb_rn_vp": "gradient boosting",
             "rn_only_powerlaw": "rebound-only formula", "vp_only_powerlaw": "ultrasonic-only formula"}


def interior_headline():
    a = _get(_load("interior/ndt_strength_v1.json"), "loso", "all")
    models = _get(a, "models") or {}
    base = _get(models, "training_mean", "mae")
    cands = {k: v["mae"] for k, v in models.items() if k != "training_mean" and _num(v.get("mae"))}
    if not (cands and _num(base)):
        return []
    best = min(cands, key=cands.get)
    return [(f"Estimates concrete strength from rebound-hammer and ultrasonic readings with a typical error of "
             f"{cands[best]:.1f} MPa on {a['n_studies']} studies it never saw (guessing the average: {base:.1f} MPa). "
             f"Best method: {NDT_NAMES.get(best, best)}. Always calibrate with drilled cores.", "REAL")]


def energy_headline():
    r = _get(_load("energy/summary.json"), "replay_all_rooms", "room_rule")
    lab = _get(_load("energy/replay_robod.json"), "label") or "REAL"
    s, ml = _get(r, "sensor"), _get(r, "ml_only")
    if not (s and ml):
        return []
    return [(f"Motion sensors inside code rules cut lighting energy {s['saved_vs_as_operated_pct']:.0f}% on real rooms, "
             f"leaving people under-lit {s['underlit_occupied_share']:.2%} of occupied time. ML switching alone saved "
             f"{ml['saved_vs_as_operated_pct']:.0f}% but under-lit {ml['underlit_occupied_share']:.1%} of occupied time, "
             f"so ML plans and forecasts while code-safe rules switch.", "REAL + INJECTED" if "INJECTED" in lab else "REAL")]


def clog_headline():
    out = []
    p = _get(_load("clog/bellinge_case.json"), "headline", "paired")
    if p and p.get("detected") and _num(p.get("delay_min_from_onset")):
        months = _get(_load("clog/bellinge_case.json"), "test_months")
        tail = f", with {p['other_test_episodes']} false alarms in {months:.1f} test months" if _num(months) else ""
        out.append((f"Caught a real, documented sewer blockage {p['delay_min_from_onset']:.0f} minutes after it began"
                    f"{tail} (Bellinge, Denmark, 2020; one case).", "REAL"))
    sev = _get(_load("clog/riser_metrics.json"), "variants", "base", "detectors", "act4_z", "alarm", "3")
    if sev and _num(sev.get("rate")):
        out.append((f"A short nightly flow test flags {sev['rate']:.0%} of nights with a severe simulated riser clog "
                    f"({sev['n_scenarios']} held-out scenarios).", "SIMULATED"))
    return out


def numeric_headline():
    m = _load("numeric/backtest_bdg2.json")
    c, b = _get(m, "models", "chronos_2", "WAPE_pct"), _get(m, "models", "snaive168", "WAPE_pct")
    if not (_num(c) and _num(b)):
        return []
    return [(f"Hugging Face Chronos-2 forecasts tomorrow's electricity use with {c:.1f}% error on "
             f"{m['n_meters']} real office meters (same hour last week: {b:.1f}%).", "REAL")]


def rain_headline():
    s = _load("rain/summary.json")
    k, n = _get(s, "n_basin_E_SE"), _get(s, "n_basin")
    if not (_num(k) and _num(n)):
        return []
    out = [(f"In {k} of {n} LA-basin weather stations, east- or south-east-facing walls take the most "
            f"wind-driven rain, although the wind usually blows from the west.", "REAL")]
    hgb, phys = _get(s, "loso_mae", "HGB_model"), _get(s, "loso_mae", "B1_iso_physics_on_grid")
    if _num(hgb) and _num(phys):
        out.append((f"Per-storm wall-wetting model error on stations it never saw: {hgb:.2f} "
                    f"(physics-only: {phys:.2f}).", "REAL"))
    return out


def fire_headline():
    t = _get(_load("fire/sensor_eval.json"), "hall", "full", "metrics", "two_stage")
    if not t:
        return []
    f, n = t["fire"], t["nuisance"]
    return [(f"In a test hall it never trained on, the early-fire check flagged {f['alarmed']} of {f['n']} real test "
             f"fires, and also {n['alarmed']} of {n['n']} nuisance events (welding, exhaust, sprays), "
             f"so a person always verifies before anyone acts.", "REAL")]


CARDS = [
    ("Exterior inspection", "exterior_inspection.py", ":material/apartment:",
     "Upload a facade photo: a crack heatmap shows where to look, then an AI grader explains the worst spots.",
     facade_headline),
    ("Interior walls", "interior_walls.py", ":material/format_paint:",
     "Turn hammer and ultrasonic readings from inside walls into a strength estimate with an honest range.",
     interior_headline),
    ("Common-area energy", "energy.py", ":material/bolt:",
     "Common-area lighting follows real use inside code minimums; HVAC and seasonal calendar actions are proposals "
     "a person approves.",
     energy_headline),
    ("Clog Watch (pipes)", "pipes.py", ":material/plumbing:",
     "Find blocked pipes and drains early from flow and water-level data, and plan the right periodic checks.",
     clog_headline),
    ("Numeric AI", "numeric_ai.py", ":material/insights:",
     "One engine for every sensor number: forecasts with a range, and alerts when a reading looks wrong.",
     numeric_headline),
    ("Rain exposure (LA)", "rain_la.py", ":material/rainy:",
     "Which side of your building gets hit by rain in Los Angeles, and how wet the next storm will make it.",
     rain_headline),
    ("Fire plan", "fire_plan.py", ":material/local_fire_department:",
     "Pick where a fire starts: see which floors to alert, which stairs to use and what is nearby, for the "
     "fire safety director to decide.", fire_headline),
]

FLOW_SVG = """
<svg viewBox="0 0 900 150" width="100%" role="img" aria-label="Photos, sensors and public data feed specialist
models, which feed one ranked list that a person approves" style="max-width:900px;font-family:sans-serif">
  <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0,0 L10,5 L0,10 z" fill="#8a8984"/></marker></defs>
  <g font-size="14" fill="currentColor" text-anchor="middle">
    <rect x="10" y="30" width="190" height="90" rx="12" fill="#2a78d6" fill-opacity="0.15" stroke="#2a78d6"/>
    <text x="105" y="62" font-weight="600">What we see</text>
    <text x="105" y="84">photos, sensors, meters,</text><text x="105" y="102">weather, public records</text>
    <rect x="245" y="30" width="190" height="90" rx="12" fill="#1baf7a" fill-opacity="0.15" stroke="#1baf7a"/>
    <text x="340" y="62" font-weight="600">Specialist models</text>
    <text x="340" y="84">facade, walls, energy,</text><text x="340" y="102">pipes, rain, fire</text>
    <rect x="480" y="30" width="190" height="90" rx="12" fill="#eb6834" fill-opacity="0.15" stroke="#eb6834"/>
    <text x="575" y="62" font-weight="600">One ranked list</text>
    <text x="575" y="84">what matters most,</text><text x="575" y="102">with its evidence</text>
    <rect x="715" y="30" width="175" height="90" rx="12" fill="#8a8984" fill-opacity="0.15" stroke="#8a8984"/>
    <text x="802" y="62" font-weight="600">A person decides</text>
    <text x="802" y="84">approves the first</text><text x="802" y="102">safe step, logged</text>
  </g>
  <g stroke="#8a8984" stroke-width="2" marker-end="url(#a)">
    <line x1="202" y1="75" x2="241" y2="75"/><line x1="437" y1="75" x2="476" y2="75"/>
    <line x1="672" y1="75" x2="711" y2="75"/></g>
</svg>
"""


def _link(page_file: str, title: str, icon: str) -> None:
    try:
        st.page_link(f"site_pages/{page_file}", label=f"Open {title}", icon=icon)
    except Exception:  # standalone run without st.navigation: page links are unavailable
        st.caption(f"Open **{title}** from the sidebar.")


st.title("Pavilion Cerebro")
st.subheader("Your building, watched by specialist AI, ranked by what matters, decided by people.")

st.info(
    "**What this means.** Tall buildings send warning signs every day: a crack on the facade, a damp wall, a "
    "blocked drain, lights burning in an empty garage, rain driving into one side, a smell of smoke. Each sign "
    "sits with a different trade. Cerebro reads them all, tells the building manager which one to deal with "
    "first and why, and waits for a person to approve every action. It never controls alarms, sprinklers, "
    "elevators or valves."
)

st.markdown(FLOW_SVG, unsafe_allow_html=True)

st.markdown("#### What each part does, and what we measured")
legend = " ".join(f":{c}-badge[{k}]" for k, c in LABEL_COLORS.items() if k != "RULES")
st.caption(f"Every result below is read from our evaluation files, on data the model did not train on. {legend} "
           "REAL = public measured data; SIMULATED/SYNTHETIC = generated by our simulators; INJECTED = faults we "
           "added on purpose. No result here comes from a customer building yet.")

cols = st.columns(2)
for i, (title, page, icon, promise, headline) in enumerate(CARDS):
    with cols[i % 2].container(border=True):
        st.markdown(f"##### {title}")
        st.write(promise)
        try:
            lines = headline()
        except Exception:  # a malformed artifact must never break the home page
            lines = []
        if not lines:
            st.caption("Result: not yet run.")
        for text, label in lines:
            badges = " ".join(f":{LABEL_COLORS.get(part.strip(), 'gray')}-badge[{part.strip()}]"
                              for part in label.split("+"))
            st.markdown(f"{badges} {text}")
        _link(page, title, icon)

with st.expander("For investors and engineers: how it fits together"):
    st.markdown(
        "- **One building model.** Floors, zones, pipes, risers, panels and circuits live in one graph, so a leak "
        "is scored by what it could reach (the electrical room one floor down) and a fire plan knows which risers "
        "and panels are nearby.\n"
        "- **Specialist models, one queue.** Each module scores its own problem with its own measured method; "
        "the building console ranks everything in one list by risk, consequence and time left.\n"
        "- **Honest by construction.** Every page separates real, simulated and injected data, shows the naive "
        "baseline next to the model (including where the model loses), and cites its sources and licences.\n"
        "- **People decide.** Cerebro proposes; a named person approves; the decision is logged. Life-safety "
        "systems stay under their own listed controls.\n"
        "- **Runs locally.** This site runs on a laptop from precomputed results; the crack model runs on CPU."
    )

st.caption("Sources and licences for each result are listed at the bottom of its page.")
