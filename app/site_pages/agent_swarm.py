"""Live agent swarm page: the four-agent swarm console (dashboard/) inside the Pavilion Cerebro site.

The console's standard-library server starts once per Streamlit process on daemon threads, and this page
embeds it, so `streamlit run app/site.py` shows every part of the product in one place. If a standalone
`python -m dashboard.server` already serves the port, the page reuses it. Agent 0 runs the offline OpenCV
crack check here (no API calls during a demo); set CEREBRO_VISUAL_BACKEND to change that, and
CEREBRO_DASHBOARD_PORT to move the console off port 8765. Top-level Streamlit page: no st.set_page_config.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "data" / "samples" / "metrics.json"
VIDEO = ROOT / "site" / "assets" / "pavilion_cerebro_demo.mp4"
PORT = int(os.getenv("CEREBRO_DASHBOARD_PORT", "8765"))


def _alive(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/api/state", timeout=1) as r:
            return r.status == 200
    except OSError:
        return False


@st.cache_resource(show_spinner="Starting the agent swarm (loading five public datasets)...")
def start_swarm(port: int) -> str | None:
    """Start the console server once per process; return its URL, or None when another program holds the port."""
    url = f"http://127.0.0.1:{port}"
    if _alive(url):
        return url
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from dashboard.server import start_background

    try:
        start_background(port, visual_backend=os.getenv("CEREBRO_VISUAL_BACKEND") or "opencv")
    except OSError:
        return None
    return url


def _metrics() -> dict:
    try:
        return json.loads(METRICS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


st.html('<div class="pc-eyebrow"><span class="pc-dot"></span>Agents · live replay</div>')
st.title("Live agent swarm")
st.html('<div class="pc-tagline">Four agents. One coordinator.</div>'
        '<p class="pc-lead">Every problem ranked by risk × consequence × time, as five real datasets replay.</p>')

st.info(
    "**What this means.** Each agent watches one part of the building on real public data: cracks in photos, "
    "concrete strength and building-system readings, battery wear, and chiller faults. They all report in one "
    "format to one coordinator, which ranks every problem by risk × consequence × time. The building and the "
    "actions in the console are simulated; the readings and the detection logic are real, replayed live."
)

m = _metrics()
h, a0, a2, a3 = (m.get(k) or {} for k in ("headline", "agent0", "agent2", "agent3"))
held = a0.get("heldout") or {}
lead = list((a2.get("lead_cycles") or {}).values())
cols = st.columns(4)
if h.get("real_fault_cases_flagged") is not None:
    b = h.get("real_fault_cases_breakdown") or {}
    cols[0].metric("Real faults flagged", h["real_fault_cases_flagged"],
                   border=True, help=f"{b.get('ashrae_fault_runs_detected_and_diagnosed')} chiller fault runs, "
                        f"{b.get('nasa_cells_flagged_before_eol')} battery cells before end of life, "
                        f"{b.get('umn_upv_low_readings_flagged')} weak concrete readings and "
                        f"{b.get('cu_bems_ac1_collapse_flagged')} real air-conditioning failure, "
                        f"across {h.get('public_datasets')} public datasets.")
if a3.get("diagnosis_accuracy_detected") is not None:
    cols[1].metric("Chiller diagnosis", f"{a3['diagnosis_accuracy_detected']:.1%}",
                   border=True, help=f"Of {a3.get('n_detected_windows'):,} detected fault windows. False alarms: "
                        f"{a3.get('false_alarm_rate_holdout_normal', 0):.1%} of {a3.get('n_holdout_normal_windows')} "
                        f"windows from fault-free runs the agent never saw.")
if lead:
    cols[2].metric("Failing cells caught", f"{len(lead)} of {len(lead)}",
                   border=True, help=f"{min(lead)} to {max(lead)} charge cycles before end of life (NASA Li-ion aging data). "
                        "These are wear warning signs, not fire predictions.")
if held.get("accuracy") is not None:
    cols[3].metric("Crack check accuracy", f"{held['accuracy']:.1%}",
                   border=True, help=f"Offline OpenCV method on {held.get('n_eval'):,} held-out lab concrete photos, "
                        "not yet on real building facades.")
st.caption(":green-badge[REAL] Measured by `python scripts/selfcheck.py` on public data. "
           ":orange-badge[SIMULATED] The building and its actions: five datasets replayed on one clock.")

url = start_swarm(PORT)
if url is None:
    start_swarm.clear()
    st.error(f"Port {PORT} is taken by another program, so the console could not start. Stop that program, "
             "or set CEREBRO_DASHBOARD_PORT to a free port, then reload this page.")
else:
    st.link_button("Open the console full screen", url, icon=":material/open_in_new:")
    st.iframe(url, height=2440)  # the console's full height at the page's content width (860-1300 px layout)
    st.caption("Console controls: pause, 1x / 2x / 4x speed, restart, and disaster mode, which switches the "
               "ranking to pure triage by severity.")

with st.expander("What each agent does"):
    st.markdown(
        "- **Agent 0, structural visual.** Checks photos for cracks. It wraps the team's inspection cascade; "
        "this page runs its offline OpenCV method.\n"
        "- **Agent 1, internal sensors.** Scores concrete pulse-velocity tests (UPV) and building-management "
        "data with one generic engine that infers its own baseline.\n"
        "- **Agent 2, batteries.** Tracks state of health and remaining life on NASA Li-ion aging data.\n"
        "- **Agent 3, HVAC.** Detects and diagnoses chiller faults on ASHRAE RP-1043 data.\n"
        "- **Coordinator.** Fuses every report into one building health score and one ranked list. Adding a "
        "system means adding one more agent, not rebuilding the platform."
    )

if VIDEO.exists():
    with st.expander("Demo video (about 78 seconds, rendered from a real run)"):
        st.video(str(VIDEO))

st.caption("Sources and licences: UMN DRUM UHPC UPV dataset (CC0), CU-BEMS building data (CC BY 4.0), NASA PCoE "
           "Li-ion aging data, ASHRAE RP-1043 chiller data (CC0), and concrete crack images (Özgenel, CC BY 4.0). "
           "Details and caveats are in the README.")
