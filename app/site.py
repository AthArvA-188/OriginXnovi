"""Pavilion Cerebro local website: one Streamlit multipage site over every module page.

Run locally (no deploy):
    conda activate origin_hack
    streamlit run app/site.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

APP = Path(__file__).resolve().parent
PAGES = APP / "site_pages"

st.set_page_config(page_title="Pavilion Cerebro", layout="wide", page_icon="\U0001F3E2")

# (file, title, icon); a page is listed only if its file exists, so a missing module never breaks the site.
SECTIONS = {
    "Overview": [("home.py", "Home", ":material/home:")],
    "Agents": [("agent_swarm.py", "Live agent swarm", ":material/hub:")],
    "Inspect": [
        ("exterior_inspection.py", "Exterior inspection", ":material/apartment:"),
        ("interior_walls.py", "Interior walls", ":material/format_paint:"),
    ],
    "Operate": [
        ("energy.py", "Common-area energy", ":material/bolt:"),
        ("pipes.py", "Clog Watch (pipes)", ":material/plumbing:"),
        ("numeric_ai.py", "Numeric AI", ":material/insights:"),
    ],
    "Protect": [
        ("rain_la.py", "Rain exposure (LA)", ":material/rainy:"),
        ("fire_plan.py", "Fire plan", ":material/local_fire_department:"),
    ],
}


def build_nav() -> dict:
    nav = {}
    for section, items in SECTIONS.items():
        pages = [
            st.Page(str(PAGES / f), title=title, icon=icon, default=(f == "home.py"))
            for f, title, icon in items
            if (PAGES / f).exists()
        ]
        if pages:
            nav[section] = pages
    console = APP / "streamlit_app.py"
    if console.exists():
        nav["Console"] = [st.Page(str(console), title="Building console", icon=":material/dashboard:")]
    return nav


st.navigation(build_nav()).run()
