"""Pavilion Cerebro look for the local site: one CSS layer on top of the theme in .streamlit/config.toml.

The theme sets the palette and the brand fonts (IBM Plex Sans and Mono, Big Shoulders Display). This layer adds
what a theme cannot express: the page glow, card hover, metric tiles, and the classes the pages use for the
hero, the numbers strip and the problem-area chips. Colors match the swarm console (dashboard/static/style.css),
so the embedded console sits in the page with no seam. app/site.py calls apply() once per run, before the page.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

STATIC = Path(__file__).resolve().parent / "static"
LOGO = STATIC / "cerebro_logo.svg"
MARK = STATIC / "cerebro_mark.svg"

CSS = """
:root {
  --pc-bg: #0b0d0e; --pc-surface: #131618; --pc-surface-2: #1a1e21; --pc-line: #262b2f; --pc-line-2: #33393e;
  --pc-ink: #eef0f1; --pc-ink-2: #b7bec3; --pc-muted: #7d868c; --pc-accent: #3987e5;
  --pc-built: #19a974; --pc-next: #3987e5; --pc-future: #ec835a;
  --pc-display: 'Big Shoulders Display', 'Arial Narrow', sans-serif;
  --pc-mono: 'IBM Plex Mono', ui-monospace, Consolas, monospace;
}

/* page: a soft glow behind the header, flat console black everywhere else */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 460px at 80% -170px, rgba(57, 135, 229, .20), transparent 70%),
    radial-gradient(760px 420px at -5% -140px, rgba(25, 169, 116, .10), transparent 70%),
    var(--pc-bg);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 1380px; padding-top: 3.2rem; }
[data-testid="stMain"] h1 { text-transform: uppercase; letter-spacing: .005em; line-height: .95; }

/* sidebar */
[data-testid="stSidebar"] { border-right: 1px solid var(--pc-line); }
[data-testid="stNavSectionHeader"] {
  font-family: var(--pc-mono) !important; font-size: .68rem !important; letter-spacing: .16em;
  text-transform: uppercase; color: var(--pc-muted) !important;
}
[data-testid="stSidebarNavLink"] { border-radius: 6px; transition: background .15s; }
[data-testid="stSidebarNavLink"]:hover { background: rgba(57, 135, 229, .10); }

/* metric tiles */
[data-testid="stMetric"] { background: linear-gradient(180deg, #161b1f, #111416); }
[data-testid="stMetricValue"] { font-family: var(--pc-mono) !important; letter-spacing: -.02em; }
[data-testid="stMetricLabel"] p {
  font-family: var(--pc-mono); font-size: .72rem !important; letter-spacing: .08em; text-transform: uppercase;
  color: var(--pc-ink-2) !important;
}

/* cards: st.container(border=True, key="pc-card-...") */
[class*="st-key-pc-card"] {
  background: linear-gradient(180deg, rgba(26, 30, 33, .92), rgba(17, 20, 22, .92));
  transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
}
[class*="st-key-pc-card"]:hover {
  transform: translateY(-2px); border-color: rgba(57, 135, 229, .75) !important;
  box-shadow: 0 14px 34px rgba(0, 0, 0, .45), 0 0 0 1px rgba(57, 135, 229, .25);
}
[class*="st-key-pc-card"] h5 { text-transform: uppercase; letter-spacing: .02em; }
[data-testid="stExpander"] details { background: rgba(19, 22, 24, .85); }
[data-testid="stIFrame"] { border: 1px solid var(--pc-line); border-radius: 10px; box-shadow: 0 22px 60px rgba(0, 0, 0, .5); }

/* hero */
.pc-eyebrow {
  display: inline-flex; align-items: center; gap: 10px; padding: 6px 13px; border: 1px solid var(--pc-line-2);
  border-radius: 999px; background: rgba(19, 22, 24, .75); color: var(--pc-ink-2);
  font: 600 .72rem/1 var(--pc-mono); letter-spacing: .18em; text-transform: uppercase;
}
.pc-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--pc-built); animation: pc-pulse 1.8s infinite; }
@keyframes pc-pulse {
  0% { box-shadow: 0 0 0 0 rgba(25, 169, 116, .7); }
  70% { box-shadow: 0 0 0 9px rgba(25, 169, 116, 0); }
  100% { box-shadow: 0 0 0 0 rgba(25, 169, 116, 0); }
}
.pc-tagline {
  margin: .1rem 0 .7rem; color: var(--pc-accent); text-transform: uppercase; letter-spacing: .01em;
  font: 800 clamp(1.9rem, 3.4vw, 2.8rem)/1 var(--pc-display);
}
.pc-lead { max-width: 46em; margin: 0 0 1.5rem; font-size: 1.12rem; line-height: 1.55; color: var(--pc-ink-2); }
.pc-kicker {
  margin: 2.2rem 0 .35rem; color: var(--pc-accent); font: 600 .7rem/1 var(--pc-mono); letter-spacing: .2em;
  text-transform: uppercase;
}

/* numbers strip */
.pc-kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 0 0 .4rem; }
.pc-kpi {
  position: relative; overflow: hidden; padding: 15px 16px 14px 18px; border: 1px solid var(--pc-line);
  border-radius: 10px; background: linear-gradient(180deg, #161b1f, #111416);
}
.pc-kpi::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--pc-kpi, var(--pc-accent)); }
.pc-kpi b { display: block; color: var(--pc-ink); font: 500 2.3rem/1 var(--pc-mono); letter-spacing: -.03em; }
.pc-kpi span {
  display: block; margin-top: 7px; color: var(--pc-muted); font: 500 .7rem/1.35 var(--pc-mono);
  letter-spacing: .08em; text-transform: uppercase;
}

/* problem areas by stage */
.pc-scope { display: grid; grid-template-columns: 3fr 4fr 4fr; gap: 12px; margin-bottom: .4rem; }
.pc-stage {
  --pc-stage: var(--pc-accent); padding: 14px 16px 16px; border: 1px solid var(--pc-line);
  border-top: 3px solid var(--pc-stage); border-radius: 10px; background: rgba(19, 22, 24, .88);
}
.pc-stage.built { --pc-stage: var(--pc-built); }
.pc-stage.next { --pc-stage: var(--pc-next); }
.pc-stage.future { --pc-stage: var(--pc-future); }
.pc-stage-h { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; margin-bottom: 11px; }
.pc-stage-h b { color: var(--pc-stage); font: 800 1.35rem/1 var(--pc-display); letter-spacing: .03em; text-transform: uppercase; }
.pc-stage-h em {
  color: var(--pc-muted); font: 500 .66rem/1.25 var(--pc-mono); font-style: normal; letter-spacing: .06em;
  text-align: right; text-transform: uppercase;
}
.pc-chips { display: flex; flex-wrap: wrap; gap: 6px; }
.pc-chip {
  display: inline-flex; align-items: center; gap: 7px; padding: 5px 9px; border: 1px solid var(--pc-line-2);
  border-radius: 6px; background: var(--pc-surface-2); color: var(--pc-ink); font-size: .84rem;
}
.pc-chip i {
  padding: 2px 4px; border: 1px solid currentColor; border-radius: 3px; color: var(--pc-stage);
  font: 600 .64rem/1 var(--pc-mono); font-style: normal; letter-spacing: .06em;
}

@media (max-width: 900px) {
  .pc-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .pc-scope { grid-template-columns: 1fr; }
}
"""


def apply() -> None:
    """Inject the brand CSS and the sidebar logo for the current run."""
    st.html(f"<style>{CSS}</style>")
    if LOGO.exists():
        st.logo(str(LOGO), size="large", icon_image=str(MARK) if MARK.exists() else None)
