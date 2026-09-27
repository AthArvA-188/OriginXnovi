"""Inline SVG: riser-and-stack sensor diagram for the SYNTHETIC tower (pure string building, no dependencies).

Segments can be coloured by a measured value per location (for example, the share of moderate or severe clogs the
nightly test detected); the caller passes values it read from an artifact, the diagram never invents numbers.
Unrated plumbing (express riser, connectors, PRVs) is drawn in slate so it cannot be confused with a colour bin.
"""
from __future__ import annotations

from html import escape
from typing import Dict, Optional

from . import geometry as G
from . import rules as R

W, H = 780, 636
TOP, FLOOR_PX = 48, 13.5
X_EXP, X_STR, X_PRV, X_RISER = 70, 100, 128, 172
X_STACK, X_DRAIN_END = 560, 740
INK, MUTED, PIPE, DRAIN = "#374151", "#9ca3af", "#64748b", "#92400e"
SENS, TEST, ALERT = "#0f766e", "#b45309", "#b91c1c"
BINS = (("80% or more", 0.8, "#0f766e"), ("60-80%", 0.6, "#2563eb"), ("below 60%", 0.0, "#d97706"))


def y_of(f: float) -> float:
    return TOP + (G.FLOORS - f) * FLOOR_PX


def _color(v: Optional[float]) -> str:
    if v is None:
        return PIPE
    for _lab, lo, col in BINS:  # high = teal, mid = blue, low = amber
        if v >= lo:
            return col
    return BINS[-1][2]


def _text(x, y, s, size=11, anchor="start", color=INK, weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{color}" '
            f'font-weight="{weight}" font-family="system-ui, sans-serif">{escape(s)}</text>')


def _p(x, y, label="P", r=6.0):
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="#ffffff" stroke="{SENS}" stroke-width="2"/>'
            + _text(x, y + 3.3, label, 8, "middle", SENS, "bold"))


def _q(x, y):
    return (f'<rect x="{x - 6:.1f}" y="{y - 6:.1f}" width="12" height="12" transform="rotate(45 {x:.1f} {y:.1f})" '
            f'fill="#ffffff" stroke="{SENS}" stroke-width="2"/>' + _text(x, y + 3.4, "Q", 8, "middle", SENS, "bold"))


def _valve(x, y, color=TEST):
    return (f'<path d="M{x - 7:.1f},{y - 5:.1f} L{x + 7:.1f},{y + 5:.1f} L{x + 7:.1f},{y - 5:.1f} '
            f'L{x - 7:.1f},{y + 5:.1f} Z" fill="{color}" stroke="{color}" stroke-width="1"/>')


def _level(x, y):
    return (f'<rect x="{x - 7:.1f}" y="{y - 7:.1f}" width="14" height="14" rx="3" fill="#ffffff" stroke="{SENS}" '
            f'stroke-width="2"/>' + _text(x, y + 3.5, "L", 8.5, "middle", SENS, "bold"))


def riser_svg(detect_by_loc: Optional[Dict[str, float]] = None, value_label: str = "") -> str:
    d = detect_by_loc or {}
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px;height:auto" role="img" '
         f'aria-label="Riser and drainage stack sensor diagram" xmlns="http://www.w3.org/2000/svg">',
         f'<rect x="0" y="0" width="{W}" height="{H}" fill="#f8fafc" rx="10"/>',
         _text(16, 24, "SUPPLY: domestic cold-water riser (3 pressure zones)", 12.5, weight="bold"),
         _text(X_STACK - 60, 24, "DRAINS: stack, building drain, grease line", 12.5, weight="bold")]
    zone_tops = {zd["floors"][1] for zd in G.ZONES.values()}
    # floor ticks (zone-top floors are named in the test-valve labels instead)
    for f in (1, 6, 11, 12, 17, 22, 23, 28, 32):
        if f in zone_tops:
            continue
        o.append(f'<line x1="{X_RISER + 4}" y1="{y_of(f):.1f}" x2="{X_RISER + 60}" y2="{y_of(f):.1f}" '
                 f'stroke="{MUTED}" stroke-width="0.6" stroke-dasharray="2 3"/>')
        o.append(_text(X_RISER + 64, y_of(f) + 3.5, G.fid(f), 9, color=MUTED))
    yb = y_of(0) + 26
    # booster + express riser (unrated plumbing in slate)
    o.append(f'<rect x="{X_EXP - 34}" y="{yb - 12}" width="68" height="24" rx="4" fill="#eef2f7" stroke="{PIPE}"/>')
    o.append(_text(X_EXP, yb + 4, "booster", 10, "middle", PIPE, "bold"))
    o.append(f'<line x1="{X_EXP}" y1="{yb - 12}" x2="{X_EXP}" y2="{y_of(23):.1f}" stroke="{PIPE}" stroke-width="5"/>')
    o.append(_text(X_EXP - 8, y_of(20), "express", 9, "end", PIPE))
    o.append(_text(X_EXP - 8, y_of(19.1), "riser", 9, "end", PIPE))
    o.append(f'<line x1="{X_EXP - 10}" y1="{yb - 24:.1f}" x2="{X_EXP}" y2="{yb - 24:.1f}" stroke="{SENS}" '
             f'stroke-width="1.5"/>')
    o.append(_p(X_EXP - 16, yb - 24))
    # backflow on the service entry
    o.append(f'<line x1="8" y1="{yb:.1f}" x2="{X_EXP - 34}" y2="{yb:.1f}" stroke="{PIPE}" stroke-width="4"/>')
    o.append(_valve(20, yb, INK))
    o.append(_text(8, yb + 22, "backflow assembly", 9, color=INK))
    # zone risers, coloured by location; labels to the right of the riser, between the sensor floors
    for loc in ("L1", "L2", "M1", "M2", "H1", "H2"):
        a, b = G.SEGMENT_SENSORS[loc]
        fa, fb = int(a[1:]), int(b[1:])
        col = _color(d.get(loc))
        o.append(f'<line x1="{X_RISER}" y1="{y_of(fa):.1f}" x2="{X_RISER}" y2="{y_of(fb):.1f}" stroke="{col}" '
                 f'stroke-width="6" stroke-linecap="round"><title>{escape(G.LOCATION_NAMES[loc])}</title></line>')
        if loc in d:
            ym = (y_of(fa) + y_of(fb)) / 2
            o.append(_text(X_RISER + 12, ym + 4, f"{loc} {d[loc] * 100:.0f}%", 9.5, "start", col, "bold"))
    # PRV stations (L at F1, M at F12) and the direct feed at F23
    for zone, f, sloc in (("L", 1, "STR_L"), ("M", 12, "STR_M")):
        y = y_of(f)
        o.append(f'<line x1="{X_EXP}" y1="{y:.1f}" x2="{X_RISER}" y2="{y:.1f}" stroke="{PIPE}" stroke-width="4"/>')
        col = _color(d.get(sloc))
        o.append(f'<rect x="{X_STR - 8}" y="{y - 7:.1f}" width="16" height="14" rx="2" fill="#ffffff" stroke="{col}" '
                 f'stroke-width="2.5"><title>strainer</title></rect>')
        o.append(f'<path d="M{X_STR - 5},{y - 4:.1f} L{X_STR + 5},{y + 4:.1f} M{X_STR + 5},{y - 4:.1f} '
                 f'L{X_STR - 5},{y + 4:.1f}" stroke="{col}" stroke-width="1.2"/>')
        o.append(_valve(X_PRV, y, PIPE))
        o.append(_text(X_PRV, y + 19, "PRV", 8.5, "middle", PIPE))
        o.append(_text(X_STR, y + 19, "strainer", 8.5, "middle", col))
        o.append(_p(X_STR - 15, y - 12))
        o.append(_p(X_STR + 15, y - 12))
        if sloc in d:
            o.append(_text(X_STR, y - 24, f"{d[sloc] * 100:.0f}%", 9.5, "middle", col, "bold"))
        o.append(_q(X_RISER - 18, y + 12))
    y23 = y_of(23)
    o.append(f'<line x1="{X_EXP}" y1="{y23:.1f}" x2="{X_RISER}" y2="{y23:.1f}" stroke="{PIPE}" stroke-width="4"/>')
    o.append(_q(X_RISER - 30, y23 + 12))
    # pressure sensors at header/mid/top and test valves at zone tops
    for zone, zd in G.ZONES.items():
        lo, hi = zd["floors"]
        for f in (lo, zd["mid"], hi):
            o.append(_p(X_RISER + 16, y_of(f)))
        yt = y_of(hi)
        o.append(f'<line x1="{X_RISER + 22}" y1="{yt:.1f}" x2="{X_RISER + 118}" y2="{yt:.1f}" stroke="{TEST}" '
                 f'stroke-width="2"/>')
        o.append(_valve(X_RISER + 124, yt))
        o.append(_text(X_RISER + 136, yt + 4, f"test valve {zone} at {G.fid(hi)} (to break tank)", 9.5, color=TEST))
        o.append(_text(X_RISER + 136, (y_of(lo) + y_of(hi)) / 2 + 4, zd["name"], 10, color=INK, weight="bold"))
    # ---- drainage side
    ys_top, ys_base = y_of(32) - 10, y_of(0) + 4
    o.append(f'<line x1="{X_STACK}" y1="{ys_top:.1f}" x2="{X_STACK}" y2="{ys_base:.1f}" stroke="{DRAIN}" '
             f'stroke-width="6"/>')
    o.append(_text(X_STACK + 10, ys_top + 10, "DN100 stack", 10, color=DRAIN, weight="bold"))
    o.append(_text(X_STACK + 10, ys_top + 24, f"(max {R.value('DN100_STACK_MAX_LPS'):.1f} L/s, square entries)", 9,
                   color=DRAIN))
    for f in range(2, G.FLOORS + 1, 3):
        o.append(f'<line x1="{X_STACK - 22}" y1="{y_of(f) - 5:.1f}" x2="{X_STACK}" y2="{y_of(f):.1f}" '
                 f'stroke="{DRAIN}" stroke-width="1.5"/>')
    o.append(f'<path d="M{X_STACK},{ys_base:.1f} Q{X_STACK},{ys_base + 18:.1f} {X_STACK + 24},{ys_base + 18:.1f} '
             f'L{X_DRAIN_END},{ys_base + 18:.1f}" fill="none" stroke="{DRAIN}" stroke-width="6"/>')
    o.append(_level(X_STACK - 22, ys_base - 6))
    o.append(_text(X_STACK - 34, ys_base - 18, "stack base (level)", 9.5, "end", SENS))
    xc = X_STACK + 70
    o.append(f'<line x1="{xc}" y1="{ys_base + 18:.1f}" x2="{xc}" y2="{ys_base - 8:.1f}" stroke="{DRAIN}" '
             f'stroke-width="3"/>')
    o.append(_level(xc, ys_base - 16))
    o.append(_text(xc + 12, ys_base - 26, "cleanout level", 9.5, color=SENS))
    o.append(_text(xc + 12, ys_base - 13, "(drain-down test)", 9.5, color=SENS))
    xg = X_STACK + 130
    o.append(f'<rect x="{xg - 26}" y="{ys_base + 36:.1f}" width="52" height="26" rx="3" fill="#fef3c7" '
             f'stroke="{DRAIN}"/>')
    o.append(_text(xg, ys_base + 53, "grease int.", 9, "middle", DRAIN))
    o.append(f'<line x1="{xg}" y1="{ys_base + 36:.1f}" x2="{xg}" y2="{ys_base + 18:.1f}" stroke="{DRAIN}" '
             f'stroke-width="3"/>')
    o.append(_level(xg + 38, ys_base + 49))
    o.append(_text(xg + 4, ys_base + 78, "kitchen branch, FOG probe", 9, "middle", DRAIN))
    o.append(_text(X_DRAIN_END, ys_base + 9, "to sewer", 9.5, "end", DRAIN))
    # legend rows: symbols, colour bins, safety rule
    ly = H - 62
    o.append(_p(24, ly))
    o.append(_text(36, ly + 4, "pressure", 9.5))
    o.append(_q(98, ly))
    o.append(_text(110, ly + 4, "zone flow meter", 9.5))
    o.append(_level(206, ly))
    o.append(_text(218, ly + 4, "level", 9.5))
    o.append(_valve(262, ly))
    o.append(_text(274, ly + 4, "test valve: proposed by Cerebro, approved by a person, run by the BMS", 9.5,
                   color=TEST))
    if d:
        ly2 = ly + 20
        o.append(_text(16, ly2 + 4, f"Segment / strainer colour = {value_label}:", 9.5, color=INK))
        x = 300
        for lab, _lo, col in BINS:
            o.append(f'<rect x="{x}" y="{ly2 - 4}" width="18" height="8" rx="1" fill="{col}"/>')
            o.append(_text(x + 24, ly2 + 4, lab, 9.5, color=INK))
            x += 110
        o.append(f'<line x1="{x}" y1="{ly2:.1f}" x2="{x + 18}" y2="{ly2:.1f}" stroke="{PIPE}" stroke-width="5"/>')
        o.append(_text(x + 24, ly2 + 4, "not rated", 9.5, color=INK))
    o.append(_text(16, H - 12, "Domestic cold water only. Never on fire, sprinkler or standpipe piping. "
                               f"Every outlet at or below {G.MAX_STATIC_M * G.PSI_PER_M:.0f} psi static.", 9.5,
                   color=ALERT))
    o.append("</svg>")
    return "".join(o)
