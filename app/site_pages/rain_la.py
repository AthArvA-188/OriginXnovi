"""LA rain exposure: which side of a building gets the most wind-driven rain (key: rain).

Standalone Streamlit page for the local multipage site. Reads only precomputed artifacts in eval/rain/ (built by
scripts/build_la_rain.py from real IEM ASOS, GHCN-Daily, NCEI normals, ERA5 and GFS data). Every number shown comes
from those files. No network access; no st.set_page_config (the site entry owns it).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "eval" / "rain"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cascade.building import rainexposure as rx  # noqa: E402

FAC = list(rx.FACADES)
FAC_NAME = {"N": "north", "NE": "north-east", "E": "east", "SE": "south-east", "S": "south", "SW": "south-west",
            "W": "west", "NW": "north-west"}
FAC_COLOR = {"N": "#4c78a8", "NE": "#72b7b2", "E": "#e45756", "SE": "#f58518", "S": "#eeca3b", "SW": "#54a24b",
             "W": "#9d755d", "NW": "#b279a2"}


@st.cache_data(show_spinner=False)
def _load(name: str):
    p = EVAL / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def fmt_p(p) -> str:
    if p is None:
        return "n/a"
    p = float(p)
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * float(x):.0f}%"


def num(x, k=2) -> str:
    return "n/a" if x is None else f"{float(x):.{k}f}"


roses = _load("station_roses.json")
qa = _load("qa_station_wy.json")
metrics = _load("metrics.json")
grid = _load("grid_exposure.json")
normals = _load("normals.json")
evsum = _load("events_summary.json")
manifest = _load("sources_manifest.json")

st.title("Which side of an LA building gets the most rain?")
st.markdown("We measured two decades of real hourly airport weather around Los Angeles to find which walls take the most "
            "wind-driven rain, tested whether rain direction really differs from everyday wind, and built a model that "
            "predicts it storm by storm.")

if roses is None:
    st.warning("Artifacts are missing. Run `python scripts/fetch_la_rain_data.py` then `python scripts/build_la_rain.py`.")
    st.stop()

S = roses["stations"]
stations = list(S)
basin = [s for s in stations if S[s]["region"] in rx.BASIN_REGIONS]
basin_es = [s for s in basin if S[s]["top_facade"] in ("E", "SE")]
coast = [s for s in stations if S[s]["region"] == "Coast & basin"]


def modal_sector(d: dict) -> str:
    return max(d, key=lambda k: d[k] or 0)


coast_w = [s for s in coast if modal_sector(S[s]["sector_allhours_share"]) in ("W", "SW")]
years = sorted({S[s]["years_used"] for s in stations})

# ----------------------------------------------------------------------------------------- one-line answer
if len(basin_es) > len(basin) / 2:
    answer = (f"In most of the LA basin, east- and south-east-facing walls get the most wind-driven rain: "
              f"{len(basin_es)} of {len(basin)} basin and San Fernando Valley stations.")
else:
    answer = (f"The most-exposed wall differs across the LA basin: only {len(basin_es)} of {len(basin)} basin and valley "
              f"stations point to east or south-east.")
st.success(f"**{answer}**")
if coast_w:
    st.markdown(f"At {len(coast_w)} of {len(coast)} coastal and basin stations the everyday wind blows most often from the "
                f"west or south-west, so the usual wind rose points at the wrong wall. :green-badge[REAL] measured, "
                f"{min(years)}-{max(years)} water years per station after quality control.")

with st.container(border=True):
    st.markdown("#### What this means")
    top_novmar = [S[s]["novmar_share_of_driving_rain"] for s in basin if S[s]["novmar_share_of_driving_rain"] is not None]
    st.markdown(
        "- **Who it helps:** owners, property managers and facade inspectors of LA buildings, from mid-rise to towers.\n"
        "- **The problem:** water gets in through sealant joints, windows and weep holes on the walls that storms hit. "
        "Checking every wall equally wastes crews; guessing from the everyday (sea-breeze) wind picks the wrong side.\n"
        f"- **What to do next:** before the Nov-Mar storm season (it carries {pct(min(top_novmar))}-{pct(max(top_novmar))} "
        "of driving rain at basin stations), inspect the most-exposed side for your area first, and the upper floors and "
        "corners of that side before the lower middle. Use *Check your building* below for your own wall angle.\n"
        "- **Who decides:** the page only ranks walls. A person approves any inspection or work order; nothing here "
        "controls building equipment.")

# ----------------------------------------------------------------------------------------- map + station view
st.subheader("Where the rain comes from, station by station")
names = {s: f"{s} - {S[s]['name']} ({S[s]['region']})" for s in stations}
default = stations.index("CQT") if "CQT" in stations else 0
sid = st.selectbox("Station", stations, index=default, format_func=lambda s: names[s], key="rain_station")
E = S[sid]

col_map, col_rose = st.columns([1.15, 1])
with col_map:
    import pydeck as pdk

    show_grid = st.checkbox("Show modelled ERA5 grid cells (MODELLED)", value=bool(grid and grid.get("cells")),
                            disabled=not (grid and grid.get("cells")), key="rain_grid")
    show_norm = st.checkbox("Show NCEI 1991-2020 annual rainfall normals (REAL)", value=False,
                            disabled=normals is None, key="rain_norm")
    layers = []

    def hex_rgb(h, a=255):
        h = h.lstrip("#")
        return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [a]

    if show_grid and grid:
        polys = []
        for c in grid["cells"]:
            la, lo, d = c["lat"], c["lon"], 0.125
            polys.append({"polygon": [[lo - d, la - d], [lo + d, la - d], [lo + d, la + d], [lo - d, la + d]],
                          "color": hex_rgb(FAC_COLOR[c["top_facade_model"]], 70),
                          "tip": f"MODELLED cell {la:.2f}, {lo:.2f}: most-exposed {c['top_facade_model']} "
                                 f"({pct(c['top_share_model'])} of the 8-facade total)"
                                 + ("; outside training elevation" if c["outside_training_elevation"] else "")})
        layers.append(pdk.Layer("PolygonLayer", polys, get_polygon="polygon", get_fill_color="color",
                                get_line_color=[120, 120, 120, 80], line_width_min_pixels=1, pickable=True))
    if show_norm and normals:
        pts = [{"lon": p["lon"], "lat": p["lat"], "r": 600 + 150 * p["ann_in"],
                "tip": f"REAL NCEI normal: {p['name']}: {p['ann_in']:.2f} in/yr, elevation {p['elev_m']:.0f} m"}
               for p in normals["points"]]
        layers.append(pdk.Layer("ScatterplotLayer", pts, get_position=["lon", "lat"], get_radius="r",
                                get_fill_color=[30, 110, 200, 90], pickable=True))
    arrows, dots = [], []
    for s in stations:
        e = S[s]
        az = math.radians(rx.AZ[e["top_facade"]])
        la, lo = e["lat"], e["lon"]
        L = 0.11
        tip = (la + L * math.cos(az), lo + L * math.sin(az) / math.cos(math.radians(la)))
        col = hex_rgb(FAC_COLOR[e["top_facade"]])
        arrows.append({"from": [lo, la], "to": [tip[1], tip[0]], "color": col})
        for side in (-1, 1):
            b = az + math.pi + side * math.radians(28)
            h = 0.035
            arrows.append({"from": [tip[1], tip[0]], "to": [tip[1] + h * math.sin(b) / math.cos(math.radians(la)),
                                                            tip[0] + h * math.cos(b)], "color": col})
        dots.append({"lon": lo, "lat": la, "label": f"{s} {e['top_facade']}", "color": col,
                     "tip": f"REAL {s} {e['name']}: most-exposed wall faces {FAC_NAME[e['top_facade']]} "
                            f"({e['IA_L_m2_yr'][e['top_facade']]:.0f} L/m2/yr airfield index, {e['years_used']} water years)"})
    layers.append(pdk.Layer("LineLayer", arrows, get_source_position="from", get_target_position="to",
                            get_color="color", get_width=4, width_min_pixels=3))
    layers.append(pdk.Layer("ScatterplotLayer", dots, get_position=["lon", "lat"], get_radius=1800,
                            get_fill_color="color", get_line_color=[20, 20, 20], stroked=True, line_width_min_pixels=1,
                            pickable=True))
    layers.append(pdk.Layer("TextLayer", dots, get_position=["lon", "lat"], get_text="label", get_size=13,
                            get_color=[20, 20, 20], get_pixel_offset=[0, -16]))
    deck = pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=34.2, longitude=-118.1, zoom=7.6),
                    map_style=pdk.map_styles.CARTO_LIGHT, tooltip={"text": "{tip}"})
    st.pydeck_chart(deck, height=470)
    st.caption("Arrows point the way the most-exposed wall faces (towards where storm rain comes from). Dots and arrows: "
               ":green-badge[REAL] ASOS measurements. Shaded cells: :orange-badge[MODELLED] from ERA5 reanalysis, "
               "not measured; their expected accuracy is the leave-one-station-out score below. The basemap needs an "
               "internet connection; the data layers do not.")
    st.markdown("Colour = most-exposed wall: " + " ".join(
        f"<span style='color:{FAC_COLOR[f]};font-size:1.1em'>&#9632;</span>&nbsp;{f}" for f in FAC),
        unsafe_allow_html=True)

with col_rose:
    rows = []
    for kind, key in (("Rain (weighted by mm)", "sector_rain_share"), ("All hours (everyday wind)", "sector_allhours_share")):
        for i, f in enumerate(FAC):
            v = E[key][f] or 0.0
            rows.append({"kind": kind, "sector": f, "share": v, "t0": math.radians(i * 45 - 22.5),
                         "t1": math.radians(i * 45 + 22.5), "pct": f"{100 * v:.0f}%"})
    rdf = pd.DataFrame(rows)
    lab = pd.DataFrame({"sector": ["N", "E", "S", "W"], "t": [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]})

    def rose(kind, color):
        d = rdf[rdf["kind"] == kind]
        arc = alt.Chart(d).mark_arc(stroke="white", color=color).encode(
            theta=alt.Theta("t0:Q", scale=None), theta2="t1:Q",
            radius=alt.Radius("share:Q", scale=alt.Scale(type="sqrt", zero=True, domain=[0, max(0.6, d["share"].max())],
                                                         range=[0, 80])),
            tooltip=["sector", "pct"])
        txt = alt.Chart(lab).mark_text(radius=92, fontSize=12, fontWeight="bold", color="#808080").encode(
            theta=alt.Theta("t:Q", scale=None), text="sector")
        return (arc + txt).properties(width=195, height=195, title=kind)

    st.altair_chart(alt.hconcat(rose("Rain (weighted by mm)", "#1f77b4"), rose("All hours (everyday wind)", "#9e9e9e")),
                    width="content")
    st.caption(f":green-badge[REAL] {sid}: share of rain (left) and of all non-calm hours (right) by the direction the "
               f"wind blows FROM, {E['years_used']} QC'd water years. Rain-weighted mean direction "
               f"{num(E['rain_w_mean_dir'], 0)} deg (Rbar {num(E['rain_w_rbar'])}), all hours {num(E['allhours_mean_dir'], 0)} deg. "
               f"{pct(E['calm_share_of_rain'])} of rain fell in calm hours (no direction).")
    bdf = pd.DataFrame({"facade": FAC, "I_A": [E["IA_L_m2_yr"][f] for f in FAC],
                        "top": ["most exposed" if f == E["top_facade"] else "other" for f in FAC]})
    bars = alt.Chart(bdf).mark_bar().encode(
        x=alt.X("facade:N", sort=FAC, title="Wall facing", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("I_A:Q", title="L/m2 per year", scale=alt.Scale(domain=[0, 1.15 * max(bdf["I_A"].max(), 1e-9)])),
        color=alt.Color("top:N", scale=alt.Scale(domain=["most exposed", "other"], range=["#e45756", "#9ecae9"]),
                        legend=None),
        tooltip=["facade", alt.Tooltip("I_A:Q", format=".1f")])
    st.altair_chart((bars + bars.mark_text(dy=-6, fontSize=10).encode(text=alt.Text("I_A:Q", format=".0f"))).properties(
        height=230, title=f"{sid}: yearly driving-rain index per wall"), width="stretch")
    st.caption(f":green-badge[REAL] ISO 15927-3 airfield annual index at 10 m on open airfield terrain, averaged over "
               f"{E['years_used']} water years; most-exposed wall {E['top_facade']} gets {num(E['max_min_ratio'], 1)} times "
               f"the least-exposed ({E['least_facade']}).")

wy = E["top_facade_by_wy"]
if wy:
    wdf = pd.DataFrame({"water year": [int(k) for k in wy], "facade": list(wy.values())})
    wdf["row"] = "most exposed"
    strip = alt.Chart(wdf).mark_rect(stroke="white").encode(
        x=alt.X("water year:O", title=None, axis=alt.Axis(labelAngle=0, labelFontSize=10)),
        y=alt.Y("row:N", title=None, axis=None),
        color=alt.Color("facade:N", scale=alt.Scale(domain=FAC, range=[FAC_COLOR[f] for f in FAC]), legend=None),
        tooltip=["water year", "facade"])
    txt = alt.Chart(wdf).mark_text(fontSize=11, color="black").encode(x="water year:O", y=alt.Y("row:N", axis=None),
                                                                     text="facade")
    st.altair_chart((strip + txt).properties(height=40, title=f"{sid}: most-exposed wall in each water year (REAL)"),
                    width="stretch", height=150)
    cnt = E["top_facade_wy_counts"]
    st.caption("Counts over water years: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items(), key=lambda x: -x[1]))
               + f". Water years excluded by QC: {', '.join(map(str, qa['stations'][sid]['failed_wy'])) or 'none'}.")

# ----------------------------------------------------------------------------------------- measured results
st.subheader("Measured results")
hm = pd.DataFrame([{"station": f"{s} ({S[s]['region']})", "wall": f, "share": S[s]["IA_share"][f] or 0.0,
                    "top": f == S[s]["top_facade"]} for s in stations for f in FAC])
hm["label"] = (100 * hm["share"]).round().astype(int).astype(str) + "%"
heat = alt.Chart(hm).mark_rect(stroke="white").encode(
    x=alt.X("wall:N", sort=FAC, title="Wall facing", axis=alt.Axis(labelAngle=0, orient="top")),
    y=alt.Y("station:N", sort=[f"{s} ({S[s]['region']})" for s in stations], title=None,
            axis=alt.Axis(labelLimit=300)),
    color=alt.Color("share:Q", scale=alt.Scale(scheme="blues", domain=[0, max(0.35, float(hm["share"].max()))]),
                    legend=alt.Legend(title="share", format=".0%")),
    tooltip=["station", "wall", alt.Tooltip("share:Q", format=".1%")])
htxt = alt.Chart(hm).mark_text(fontSize=11).encode(
    x=alt.X("wall:N", sort=FAC), y=alt.Y("station:N", sort=[f"{s} ({S[s]['region']})" for s in stations]),
    text="label", color=alt.condition("datum.share > 0.2", alt.value("white"), alt.value("#333333")))
st.altair_chart((heat + htxt).properties(height=34 * len(stations) + 20,
                                          title="Share of each station's wind-driven rain that each wall receives (REAL)"),
                width="stretch")
_by_top: dict = {}
for s in stations:
    _by_top.setdefault(S[s]["top_facade"], []).append(s)
st.caption(":green-badge[REAL] Each row sums to 100% over the 8 walls (ISO 15927-3 airfield index, QC'd ASOS water "
           "years). Most-exposed wall: " + "; ".join(f"{f} at {', '.join(v)}" for f, v in
                                                    sorted(_by_top.items(), key=lambda kv: FAC.index(kv[0]))) + ".")
st.markdown("**Most-exposed wall at every station** :green-badge[REAL] (ASOS hourly, QC'd against GHCN-Daily; airfield "
            "terrain at 10 m; values are relative exposure, not litres on a real wall)")
tab = []
for s in stations:
    e = S[s]
    tab.append({"station": s, "region": e["region"], "water years": e["years_used"], "most exposed": e["top_facade"],
                "index top (L/m2/yr)": e["IA_L_m2_yr"][e["top_facade"]], "least exposed": e["least_facade"],
                "max/min": e["max_min_ratio"], "top-3 share": e["top3_share"],
                "rain dir (deg)": e["rain_w_mean_dir"], "rain Rbar": e["rain_w_rbar"],
                "all-hours dir (deg)": e["allhours_mean_dir"], "storms >=5 mm": e["storms"]["n"],
                "Rayleigh p": fmt_p(e["storms"]["rayleigh"]["p"]), "rain vs dry p": fmt_p(e["rain_vs_dry_permutation"]["p"]),
                "Nov-Mar share": e["novmar_share_of_driving_rain"]})
st.dataframe(pd.DataFrame(tab), hide_index=True, width="stretch", height=36 * (len(tab) + 1) + 4)

st.markdown("#### Is there a correlation between rain and wind direction?")
rp = [S[s]["storms"]["rayleigh"]["p"] for s in stations]
pp = [S[s]["rain_vs_dry_permutation"]["p"] for s in stations]
nperm = S[stations[0]]["rain_vs_dry_permutation"]["n_perm"]
nst = [S[s]["storms"]["n"] for s in stations]
c1, c2, c3 = st.columns(3)
c1.metric("Clustered storm directions",
          f"{sum(1 for p in rp if p < 1e-3)} of {len(rp)}", help=f"Rayleigh test p < 0.001; p from {fmt_p(min(rp))} to {fmt_p(max(rp))}; "
          f"{min(nst)}-{max(nst)} storms per station")
c2.metric("Rain wind differs from dry",
          f"{sum(1 for p in pp if p < 0.01)} of {len(pp)}",
          help=f"p < 0.01, block permutation test, {nperm} permutations of whole storms and dry days; p floor {fmt_p(1 / (nperm + 1))}")
if normals and "elevation_correlation" in normals:
    ec = normals["elevation_correlation"]
    key = next((k for k in ec if k.startswith("south")), "all")
    c3.metric("Rain vs elevation (rho)", num(ec[key]["spearman_rho"]),
              help=f"Spearman rho, basin side; NCEI 1991-2020 normals, n = {ec[key]['n']}, p = {fmt_p(ec[key]['p'])}; all {ec['all']['n']} stations: "
                   f"rho {num(ec['all']['spearman_rho'])} (the desert behind the mountains is dry)")
st.markdown(
    f"Yes. At every station the rain-weighted storm directions are far from random (Rayleigh test, p between "
    f"{fmt_p(min(rp))} and {fmt_p(max(rp))}), and the wind during rain comes from a different direction than the wind in "
    f"dry weather (block permutation test, largest p {fmt_p(max(pp))} with {nperm} permutations, so every station sits at "
    f"or near the test's floor). Storms are the unit of the test, so hours inside one storm are not counted as independent. "
    f":green-badge[REAL]")

if normals:
    ndf = pd.DataFrame(normals["points"])
    sc = alt.Chart(ndf).mark_circle(size=60, opacity=0.8).encode(
        x=alt.X("elev_m:Q", title="Station elevation (m)"),
        y=alt.Y("ann_in:Q", title="Annual rainfall normal (in)", scale=alt.Scale(domain=[0, float(ndf["ann_in"].max()) * 1.1])),
        color=alt.Color("side:N", title=None, legend=alt.Legend(orient="bottom-right", labelLimit=400, fillColor="rgba(128,128,128,0.15)", padding=6)),
        tooltip=["name", alt.Tooltip("ann_in:Q", format=".2f"), alt.Tooltip("elev_m:Q", format=".0f")])
    st.altair_chart(sc.properties(height=320, title=f"Rain coverage: NCEI 1991-2020 annual normals at {normals['n']} "
                                                    f"LA-area stations (REAL)"), width="stretch")
    bd = normals.get("bands") or {}
    lo_b, hi_b, no_b = (bd.get("basin side below 150 m") or {}), (bd.get("basin side 500 m and higher") or {}), \
        (bd.get("north of 34.45 N") or {})
    if lo_b.get("median_ann_in") and hi_b.get("median_ann_in"):
        st.caption(f"Basin-side stations at 500 m and higher have a median normal of {hi_b['median_ann_in']:.1f} in/yr "
                   f"(n = {hi_b['n']}) against {lo_b['median_ann_in']:.1f} in/yr below 150 m (n = {lo_b['n']}), "
                   f"{bd.get('ratio_high_to_low_basin') or float('nan'):.1f} times as much. North of the crest (Antelope "
                   f"Valley side) the median is {num(no_b.get('median_ann_in'), 1)} in/yr (n = {no_b.get('n', 0)}): a "
                   "rain shadow. The 34.45 N split is a rough stand-in for the mountain crest.")

# ----------------------------------------------------------------------------------------- model evaluation
st.markdown("#### Can AI predict which wall a coming storm will hit?")
METH = {"HGB_model": "AI model (gradient boosting)", "B1_iso_physics_on_grid": "Baseline: ISO physics on gridded wind",
        "B2_station_coef_x_grid_rain": "Baseline: station share x gridded rain",
        "B0_station_climatology": "Baseline: station average"}
TESTS = {"temporal": ("Later years", "ERA5"), "loso": ("New station", "ERA5"),
         "lspo": ("New station + years", "ERA5"), "forecast_lead": ("Day-ahead forecast", "GFS day-1 forecast")}
if metrics is None:
    st.info("Model metrics not built yet (run the 'model' part of scripts/build_la_rain.py).")
else:
    def mtable(block):
        out = []
        boot = block.get("bootstrap_mae_diff_vs_hgb", {})
        for k, name in METH.items():
            m = block["methods"].get(k)
            if not m:
                continue
            b = boot.get(k)
            out.append({"method": name, "error per wall-storm (L/m2)": round(m["MAE_L_m2"], 2),
                        "error on the wettest wall (L/m2)": round(m["peak_MAE_L_m2"], 2),
                        "wettest wall right": round(m["top1_acc"], 3), "within 45 deg": round(m["within45_acc"], 3),
                        "rank agreement (Spearman)": round(m["median_spearman"], 3),
                        "AI better by (95% CI)": (f"{b['diff']:.2f} [{b['lo95']:.2f}, {b['hi95']:.2f}]" if b else "-")})
        return pd.DataFrame(out)

    def _cmp(a, b, lower_better, tol):
        """'better' / 'about the same as' / 'worse' for the AI value a against the baseline value b."""
        if abs(a - b) <= tol:
            return "about the same as"
        return "better than" if ((a < b) if lower_better else (a > b)) else "worse than"

    def verdict(blk, src):
        """Plain-language comparison of the AI model with the best baseline of one evaluation block."""
        ms = blk["methods"]
        hg = ms["HGB_model"]
        base = [k for k in ms if k != "HGB_model"]
        best = min(base, key=lambda k: ms[k]["MAE_L_m2"])
        bb = blk["bootstrap_mae_diff_vs_hgb"][best]
        word = ("beats" if bb["lo95"] > 0 else "is not clearly better than" if bb["hi95"] >= 0 else "is worse than")
        best_top = max(base, key=lambda k: ms[k]["top1_acc"])
        best_peak = min(base, key=lambda k: ms[k]["peak_MAE_L_m2"])
        top_word = _cmp(hg["top1_acc"], ms[best_top]["top1_acc"], False, 0.005)
        peak_word = _cmp(hg["peak_MAE_L_m2"], ms[best_peak]["peak_MAE_L_m2"], True, 0.05)
        bname = METH[best].replace("Baseline: ", "").replace("gridded", src)
        return (f"the AI model **{word}** the best simple baseline ({bname}): error "
                f"{hg['MAE_L_m2']:.2f} vs {ms[best]['MAE_L_m2']:.2f} L/m2 per wall per storm (95% CI of the gap "
                f"{bb['lo95']:.2f} to {bb['hi95']:.2f}, {blk['n_events']} events in {bb['n_clusters']} storm clusters). "
                f"It names the wettest wall in {pct(hg['top1_acc'])} of storms, {top_word} the best baseline "
                f"({pct(ms[best_top]['top1_acc'])}); its error on the wettest wall ({hg['peak_MAE_L_m2']:.2f} L/m2) is "
                f"{peak_word.replace('better than', 'lower than').replace('worse than', 'higher than')} the best "
                f"baseline's ({ms[best_peak]['peak_MAE_L_m2']:.2f}).")

    st.markdown("- **Later years, ERA5 inputs:** " + verdict(metrics["temporal"], "ERA5"))
    if metrics.get("loso"):
        st.markdown("- **A station the model never saw:** " + verdict(metrics["loso"], "ERA5"))
    if metrics.get("forecast_lead"):
        st.markdown("- **Real day-ahead forecasts (GFS, 2024-2025):** " + verdict(metrics["forecast_lead"], "GFS forecast"))
    st.markdown("So the model is a modest bias corrector on top of the physics, not a replacement for it. "
                ":green-badge[REAL] held-out evaluations on measured airport data; ERA5 inputs are reanalysis, a "
                "stand-in for a forecast.")
    crow, torder = [], []
    for key, (tname, _src) in TESTS.items():
        blk = metrics.get(key)
        if not blk:
            continue
        tl = f"{tname} (n={blk['n_events']})"
        torder.append(tl)
        for k, name in METH.items():
            if k in blk["methods"]:
                crow.append({"test": tl, "method": name.replace("Baseline: ", "B: "),
                             "err": blk["methods"][k]["MAE_L_m2"], "top1": blk["methods"][k]["top1_acc"]})
    cdf_m = pd.DataFrame(crow)
    msort = [v.replace("Baseline: ", "B: ") for v in METH.values()]
    mcol = alt.Color("method:N", sort=msort, scale=alt.Scale(domain=msort, range=["#e45756", "#4c78a8", "#72b7b2", "#bab0ac"]),
                     legend=alt.Legend(orient="bottom", columns=2, labelLimit=320, title=None))
    ch = []
    for ycol, ytitle, ctitle in (("err", "L/m2", "Error per wall per storm (lower is better)"),
                                 ("top1", "share of storms", "Wettest wall named right (higher is better)")):
        ch.append(alt.Chart(cdf_m).mark_bar().encode(
            x=alt.X("test:N", sort=torder, title=None, axis=alt.Axis(labelAngle=0, labelLimit=240)),
            xOffset=alt.XOffset("method:N", sort=msort),
            y=alt.Y(f"{ycol}:Q", title=ytitle), color=mcol,
            tooltip=["test", "method", alt.Tooltip(f"{ycol}:Q", format=".3f")]).properties(height=200, title=ctitle))
    st.altair_chart(alt.vconcat(*ch).resolve_scale(color="shared"), width="stretch")
    st.caption(":green-badge[REAL] held-out results; the red bar is the AI model, the others are simple baselines.")
    tabs = st.tabs([TESTS[k][0] for k in ("temporal", "loso", "lspo", "forecast_lead")])
    blocks = [("temporal", tabs[0]), ("loso", tabs[1]), ("lspo", tabs[2]), ("forecast_lead", tabs[3])]
    for key, tb in blocks:
        with tb:
            blk = metrics.get(key)
            if not blk:
                st.info("Not available.")
                continue
            st.caption(f"{blk['split']}. n = {blk['n_events']} station-storm events in {blk['n_clusters']} storm clusters"
                       + (f"; trained on {blk['n_train']}" if "n_train" in blk else "") + ".")
            st.dataframe(mtable(blk), hide_index=True, width="stretch")
    st.caption("Events are found in the gridded/forecast rain (the same way the model would run live), so false alarms "
               "count. Target: the ISO storm index each wall got, computed from the airport's own measurements. "
               "'AI better by' = baseline error minus AI error; an interval above 0 means the AI is better.")

# ----------------------------------------------------------------------------------------- check your building
st.subheader("Check your building")
st.markdown("Pick the nearest station and the direction one of your walls faces. You get where that wall ranks for "
            "wind-driven rain and how height and nearby buildings change it. :violet-badge[ADVISORY]")
pick = st.radio("Station", ["Pick the nearest station", "Find it from my building's coordinates"], horizontal=True,
                key="rain_b_mode", label_visibility="collapsed")
k1, k2, k3 = st.columns(3)
with k1:
    if pick.startswith("Pick"):
        b_sid = st.selectbox("Nearest station", stations, index=stations.index(sid), format_func=lambda s: names[s],
                             key="rain_b_station")
    else:
        la_in = st.number_input("Latitude", min_value=33.3, max_value=35.2, value=34.0522, step=0.01, format="%.4f",
                                key="rain_b_lat")
        lo_in = st.number_input("Longitude", min_value=-119.2, max_value=-117.2, value=-118.2437, step=0.01,
                                format="%.4f", key="rain_b_lon")
        b_sid, b_km = rx.nearest_station(la_in, lo_in, stations)
        st.caption(f"Nearest measured station: **{names[b_sid]}**, {b_km:.0f} km away. The farther away it is, or "
                   "if a ridge lies between, the more your local pattern can differ.")
    az = st.slider("Wall faces (degrees clockwise from north; 90 = east)", 0, 359, 90, key="rain_b_az")
with k2:
    z = st.slider("Height on the wall (m above ground)", 2, 300, 30, key="rain_b_z")
    terr = st.selectbox("Terrain around the building (ISO category)", list(rx.TERRAIN), index=3,
                        format_func=lambda k: rx.TERRAIN_LABEL[k], key="rain_b_terrain")
with k3:
    obs_on = st.checkbox("A building at least as tall faces this wall", value=False, key="rain_b_obs_on")
    obs_m = st.slider("Distance to it (m)", 4, 200, 30, key="rain_b_obs", disabled=not obs_on)

B = S[b_sid]
rk = rx.exposure_rank(B, az)
cr, cr_ref = rx.c_r(z, terr), rx.c_r(10, "II")
o = rx.obstruction_factor(obs_m if obs_on else None)
iwa = rk["I_A"] * cr * o
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rank among 8 sides", f"{rk['rank_of_8']} of 8")
m2.metric("Share of the peak", pct(rk["share_of_max"]),
          help=f"most-exposed direction here: {rk['max_azimuth']} deg")
m3.metric("Airfield index", f"{rk['I_A']:.0f} L/m2/yr")
m4.metric("Wall index here", f"{iwa:.0f} L/m2/yr",
          help=f"I_A x C_R(z) {cr:.2f} x O {o:.1f}; C_T and W not applied")
cdf = pd.DataFrame({"azimuth": np.arange(360), "I_A": B["IA_curve_1deg"]})
line = alt.Chart(cdf).mark_area(opacity=0.35, line=True).encode(
    x=alt.X("azimuth:Q", title="Direction the wall faces (deg; 0 N, 90 E, 180 S, 270 W)", scale=alt.Scale(domain=[0, 359], nice=False),
              axis=alt.Axis(values=list(range(0, 360, 45)))),
    y=alt.Y("I_A:Q", title="L/m2 per year"))
rule_ = alt.Chart(pd.DataFrame({"azimuth": [az]})).mark_rule(color="#e45756", size=2).encode(x="azimuth:Q")
st.altair_chart((line + rule_).properties(height=180, title=f"{b_sid}: measured driving-rain index for every wall direction"),
                width="stretch", height=260)
st.caption(f"Height factor C_R: {cr:.2f} at {z} m in terrain {terr} (ISO 15927-3 Table 1). For comparison, open "
           f"airfield at 10 m gives {cr_ref:.2f}. Literature (Blocken & Carmeliet 2004) finds the top corners and side "
           f"edges of the windward wall get the most water; that pattern is not computed here. "
           f"Advisory only: a person approves any inspection.")

four = []
for k in range(4):
    a_k = (az + 90 * k) % 360
    r_k = rx.exposure_rank(B, a_k)
    o_k = o if k == 0 else 1.0          # the obstruction control describes the selected wall only
    four.append({"wall faces (deg)": int(a_k), "nearest compass side": FAC[int(((a_k + 22.5) % 360) // 45)],
                 "airfield index (L/m2/yr)": round(r_k["I_A"], 1),
                 f"wall index at {z} m (relative)": round(r_k["I_A"] * cr * o_k, 1),
                 "share of station max": pct(r_k["share_of_max"])})
four_df = pd.DataFrame(four).sort_values(f"wall index at {z} m (relative)", ascending=False).reset_index(drop=True)
four_df.insert(0, "inspect order", np.arange(1, len(four_df) + 1))
st.markdown(f"**If your building is a rectangle with one wall facing {az} deg**, its four walls rank like this at "
            f"{b_sid} :green-badge[REAL] measured curve :violet-badge[ADVISORY] order")
st.dataframe(four_df, hide_index=True, width="stretch")

# ----------------------------------------------------------------------------------------- engineers
with st.expander("For engineers: method, full metrics, QC and limits"):
    st.markdown(
        "**Physics (ISO 15927-3:2009 method 1).** Hourly airfield index h(theta) = (2/9) U10 Rh^(8/9) cos(D - theta) for "
        "cos > 0, with U10 wind speed at 10 m (m/s, power 1), Rh hourly rain (mm), D the direction the wind blows from. "
        "I_A = sum h / N years. Wall index I_WA = I_A C_R(z) C_T O W with C_R(z) = K_R ln(max(z, z_min)/z0) (Table 1) and "
        "O from Table 2; C_T and W are set to 1 and labelled (W needs ISO Figure 1, not read). Formulas as given by "
        "Blocken & Carmeliet 2010, eq. 5-9.\n\n"
        "**Data rules.** IEM routine METAR only (minutes 40-59), hour = ceil(obs time). p01i 'T' = 0; 'M' = 0 before "
        f"{rx.rule('P01I_M_SWITCH')[:10]} (empirical switch in this archive) and missing after. CQT ends "
        f"{rx.rule('CQT_END')[:10]} (site moved). Daily QC against GHCN-Daily (local standard day): IEM days above "
        f"{rx.rule('SPIKE_DAY_MM')} mm where GHCN-D = 0 are zeroed; station water years with an IEM/GHCN-D ratio outside "
        f"{rx.rule('QC_RATIO_BAND')} are excluded.\n\n"
        "**Model.** Events are segmented on ERA5 (or GFS day-1) precipitation: wet hour >= "
        f"{rx.rule('WET_HOUR_MM')} mm, split by >= 6 dry hours, total >= 5 mm; window +/- {rx.rule('EVENT_PAD_H')} h. "
        "Features: gridded event rain total/peak/duration, rain-weighted wind vectors and speeds at 10 m (and 100 m), "
        "month, station lat/lon/elevation, and the ISO physics index from gridded wind for all 8 walls. One sklearn "
        "HistGradientBoostingRegressor per wall on log1p(target). Bootstrap resamples storm clusters (starts within 36 h "
        "across stations), 2,000 resamples.")
    if metrics:
        rows = []
        for key in ("temporal", "temporal_10m", "loso", "lspo", "forecast_lead", "forecast_period_era5_reference"):
            blk = metrics.get(key)
            if not blk:
                continue
            for k, m in blk["methods"].items():
                rows.append({"test": key, "method": k, "n events": m["n_events"], "n clusters": blk["n_clusters"],
                             "MAE": m["MAE_L_m2"], "bias": m["bias_L_m2"], "peak MAE": m["peak_MAE_L_m2"],
                             "top-1": m["top1_acc"], "within45": m["within45_acc"], "Spearman": m["median_spearman"],
                             "mean target": m["mean_target_L_m2"]})
        st.markdown("**All evaluations** (L/m2; REAL held-out)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        ps = metrics["loso"].get("per_station", {})
        if ps:
            st.markdown("**Leave-one-station-out, per station** (wettest wall right)")
            st.dataframe(pd.DataFrame([{"station": s, **{METH[k]: v["top1_acc"] for k, v in d.items() if k in METH}}
                                       for s, d in ps.items()]), hide_index=True, width="stretch")
    if evsum:
        st.markdown("**ERA5 vs airport rain per station** (events on ERA5; REAL vs reanalysis)")
        st.dataframe(pd.DataFrame([{"station": s, **v} for s, v in evsum["era5_vs_asos"].items()]), hide_index=True,
                     width="stretch")
    if qa:
        st.markdown("**QC: IEM hourly vs GHCN-Daily per water year**")
        qrows = [{"station": s, "WY": w["wy"], "IEM mm": w["iem_mm"], "GHCN-D mm": w["ghcnd_mm"], "ratio": w["ratio"],
                  "spike days": w["spikes"], "used": w["passed"]} for s, e in qa["stations"].items() for w in e["wy"]]
        st.dataframe(pd.DataFrame(qrows), hide_index=True, width="stretch", height=260)
    st.markdown("**Sensitivity of the most-exposed wall** (all 20 water years without the QC exclusion; WY2013-2025 only)")
    st.dataframe(pd.DataFrame([{"station": s, "main": S[s]["top_facade"], **S[s]["sensitivity"]} for s in stations]),
                 hide_index=True, width="stretch")
    st.markdown("**Direction by rain intensity** (rain-weighted mean direction, deg)")
    st.dataframe(pd.DataFrame([{"station": s, **{k: v["mean_dir"] for k, v in S[s]["by_intensity"].items()}}
                               for s in stations]), hide_index=True, width="stretch")
    shared = " and ".join("/".join(x) for x in (grid or {}).get("shared_cells", [])) or "none"
    max_el = max(v["elev"] for v in rx.STATIONS.values())
    st.markdown(
        "**Limits.** Airport sites at 10 m on open terrain, not measurements on any building. ASOS wind is a 2-minute "
        "average to 10 deg, calm at 2 kt or less, while ISO assumes an hourly mean. ERA5 is 0.25 deg reanalysis; stations "
        f"sharing one cell: {shared}. Map cells are modelled and those well above the highest training station "
        f"({max_el:.0f} m) are outside the training range; ISO does not "
        "apply to mountains with sheer cliffs or gorges. Winds above 100 m, street canyons and tower effects are not "
        "resolved. The ISO 3-year-return spell index is not computed. The forecast-lead test covers only 2024-01 to "
        "2025-09. W (wall factor) and C_T are not applied, so absolute litres on a real wall need on-site calibration.")

# ----------------------------------------------------------------------------------------- sources
st.divider()
st.markdown("##### Sources & licences")
n_files = len(manifest) if manifest else 0
st.markdown(
    "1. Iowa Environmental Mesonet (IEM) ASOS archive, https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py and "
    "station metadata https://mesonet.agron.iastate.edu/geojson/network/CA_ASOS.geojson - public domain "
    "(https://mesonet.agron.iastate.edu/disclaimer.php), accessed 2026-09-26.\n"
    "2. NOAA NCEI Access Data Service: GHCN-Daily PRCP and 1991-2020 normals, https://www.ncei.noaa.gov/access/services/data/v1 "
    "- US federal data, accessed 2026-09-26.\n"
    "3. Weather data by Open-Meteo.com (CC BY 4.0, https://open-meteo.com/en/terms; free tier is non-commercial): archive "
    "API (ERA5) https://archive-api.open-meteo.com/v1/archive and Previous Runs API (GFS day-1) "
    "https://previous-runs-api.open-meteo.com/v1/forecast, accessed 2026-09-26. ERA5: Hersbach et al. 2023, "
    "Copernicus Climate Change Service / ECMWF, https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels "
    "(DOI 10.24381/cds.adbb2d47).\n"
    "4. Blocken B, Carmeliet J (2010) Overview of three state-of-the-art wind-driven rain assessment models, Building and "
    "Environment (ISO 15927-3 formulas and Tables 1-2), http://www.urbanphysics.net/2010_BAE_BB_JC_2010_WDRcomp_review__Preprint.pdf, "
    "accessed 2026-09-26. ISO 15927-3:2009, https://www.iso.org/standard/44281.html (paywalled; not read).\n"
    "5. Blocken B, Carmeliet J (2004) A review of wind-driven rain research in building science, J Wind Eng Ind Aerodyn "
    "92:1079-1130, https://urbanphysics.net/2004_JWEIA_WDRreview_preprint.pdf, accessed 2026-09-26.\n"
    "6. NWS SCN 24-47 (Downtown LA site move), https://www.weather.gov/media/notification/pdf_2023_24/scn24-47_downtown_los_angeles_observing_site_move.pdf; "
    "NWS ASOS wind sensor, https://www.weather.gov/asos/WindSensor.html; accessed 2026-09-26.\n"
    "7. scikit-learn (BSD-3-Clause), https://github.com/scikit-learn/scikit-learn/blob/main/COPYING.\n\n"
    f"{n_files} cached source files are listed with URL, parameters, access date and licence in "
    "eval/rain/sources_manifest.json.")
