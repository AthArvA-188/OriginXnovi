"""SYNTHETIC tower generator: a seeded skyscraper with five labelled injected scenarios (building_spec section 11).

Everything written here is synthetic: building.synthetic is True, every CSV row carries synthetic=True, plans and
facade drawings carry a SYNTHETIC stamp, and truth.json names each injected scenario. Synthetic data drives the demo
and the tests. It is never used to claim accuracy. Analysis code never reads truth.json.

Generator parameters (gains, noise, profiles) are choices of this generator, not thresholds. Every threshold used
by the analysis lives in cascade.building.rules.

CLI:  python -m cascade.building.synthetic --out data/demo/building --seed 7
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import lfilter

from ..ingest import sha256_of, write_manifest
from ..schema import ImageRecord
from .blueprint import FILES, TS_FORMAT, draw_north_arrow, draw_synthetic_stamp, load_building_dir, polygon_area, save_building
from .model import AZIMUTH, Building, BuildingData, Edge, Element, Floor, Sensor, Zone

SIDES: Tuple[str, ...] = ("N", "E", "S", "W")
WIND_CYCLE: Tuple[str, ...] = ("E", "W", "S", "N", "E", "SW", "NW", "SE")
ROOF_QUADRANTS: Dict[str, Tuple[str, Tuple[str, str]]] = {  # quadrant -> (roof drain, rooms under it on the top floor)
    "ROOF-NW": ("RD-1", ("N", "W")),
    "ROOF-NE": ("RD-2", ("N", "E")),
    "ROOF-SE": ("RD-3", ("S", "E")),
    "ROOF-SW": ("RD-4", ("S", "W")),
}
CIRCUIT_SIDE: Dict[int, str] = {1: "N", 2: "E", 3: "S", 4: "W"}

# generator parameters [synthetic generator choice, not a threshold]
S1_GAIN = 0.12  # %RH per (mm * m/s) of hourly wind-driven rain on the east facade, through the unit-peak kernel
S2_GAIN_N = 0.9  # %RH per mm of hourly rain after the drain blocks, top-floor north room
S2_GAIN_E = 0.6  # same, top-floor east room (also under ROOF-NE)
S3_CORE_RISE = 20.0  # %RH added to F(m-2)-CORE-RH by the riser leak
KERNEL_PEAK_H = 8.0
KERNEL_LEN_H = 72
TICKET_RISE = 15.0  # %RH: an east storm with a larger measured rise produces a tenant ticket
PLATE_M = (40.0, 30.0)


@dataclass(frozen=True)
class TowerConfig:
    floors: int = 32
    days: int = 365
    seed: int = 7
    start: str = "2025-10-01T00:00:00Z"
    plan_size: Tuple[int, int] = (800, 600)
    floor_height_m: float = 4.0
    storm_every_days: float = 5.0


@dataclass
class ScenarioTruth:
    scenario_id: str
    kind: str
    root_nodes: List[str]  # any listed node counts as a correct top-1 root
    affected_nodes: List[str]
    onset_ts: str
    expected_level_min: str
    note: str


def fid(n: int) -> str:
    return f"F{n:02d}"


def _pos(frac: float, floors: int, lo: int = 2) -> int:
    return int(min(max(round(frac * floors), lo), floors - 1))


def positions(cfg: TowerConfig) -> Dict[str, int]:
    """Scenario floors and onset hours. Depend only on cfg."""
    n = cfg.floors
    return {
        "top": n,
        "c": _pos(0.6, n),
        "m": _pos(0.4, n, lo=3),  # the riser leak needs two floors below it
        "l": _pos(0.3, n),
        "o": _pos(0.8, n),
        "s2_onset_h": int(0.5 * cfg.days * 24),
        "s3_onset_h": int(0.7 * cfg.days * 24),
        "s4_onset_h": int(0.3 * cfg.days * 24),
        "s5_onset_h": int((cfg.days - 0.3 * cfg.days) * 24),
    }


def _hours(cfg: TowerConfig) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(cfg.start), periods=cfg.days * 24, freq="h", name="ts")


def _ts(cfg: TowerConfig, hour: int) -> str:
    return (pd.Timestamp(cfg.start) + pd.Timedelta(hours=hour)).strftime(TS_FORMAT)


# ----------------------------------------------------------------------------------------- geometry

def _plan_polygons(size: Tuple[int, int]) -> Dict[str, List[Tuple[int, int]]]:
    """Zone polygons of one typical floor and the roof, in plan pixels (plate 40 x 30 m)."""
    w, h = size
    sx, sy = w / PLATE_M[0], h / PLATE_M[1]
    f = int(round(sx))  # 1 m facade strip
    fy = int(round(sy))
    cx0, cx1 = int(round(15 * sx)), int(round(25 * sx))
    cy0, cy1 = int(round(10 * sy)), int(round(20 * sy))
    s1, s2 = int(round(18 * sx)), int(round(22 * sx))
    return {
        "FD-N": [(0, 0), (w, 0), (w - f, fy), (f, fy)],
        "FD-E": [(w, 0), (w, h), (w - f, h - fy), (w - f, fy)],
        "FD-S": [(w, h), (0, h), (f, h - fy), (w - f, h - fy)],
        "FD-W": [(0, h), (0, 0), (f, fy), (f, h - fy)],
        "N": [(f, fy), (w - f, fy), (cx1, cy0), (cx0, cy0)],
        "E": [(w - f, fy), (w - f, h - fy), (cx1, cy1), (cx1, cy0)],
        "S": [(w - f, h - fy), (f, h - fy), (cx0, cy1), (cx1, cy1)],
        "W": [(f, h - fy), (f, fy), (cx0, cy0), (cx0, cy1)],
        "SHAFT": [(cx0, cy0), (s1, cy0), (s1, cy1), (cx0, cy1)],
        "CORE": [(s1, cy0), (s2, cy0), (s2, cy1), (s1, cy1)],
        "ELEC": [(s2, cy0), (cx1, cy0), (cx1, cy1), (s2, cy1)],
        "ROOF-NW": [(0, 0), (w // 2, 0), (w // 2, h // 2), (0, h // 2)],
        "ROOF-NE": [(w // 2, 0), (w, 0), (w, h // 2), (w // 2, h // 2)],
        "ROOF-SE": [(w // 2, h // 2), (w, h // 2), (w, h), (w // 2, h)],
        "ROOF-SW": [(0, h // 2), (w // 2, h // 2), (w // 2, h), (0, h)],
    }


def build_tower_model(cfg: TowerConfig) -> Building:
    """The tower as a Building. Pure. synthetic=True on the building and every sensor."""
    w, h = cfg.plan_size
    ppm = w / PLATE_M[0]
    polys = _plan_polygons(cfg.plan_size)
    m2 = (w / PLATE_M[0]) * (h / PLATE_M[1])
    top = cfg.floors
    floors: List[Floor] = []
    zones: List[Zone] = []
    elements: List[Element] = []
    sensors: List[Sensor] = []
    edges: List[Edge] = []
    shaft_x = (polys["SHAFT"][0][0] + polys["SHAFT"][1][0]) // 2
    elec_x0, elec_x1 = polys["ELEC"][0][0], polys["ELEC"][1][0]
    cy0, cy1 = polys["CORE"][0][1], polys["CORE"][2][1]

    for n in range(1, top + 1):
        F = fid(n)
        floors.append(Floor(floor_id=F, level=n, elevation_m=(n - 1) * cfg.floor_height_m, plan_image=f"plans/{F}.png",
                            plan_size_px=cfg.plan_size, px_per_m=ppm))
        for s in SIDES:
            zones.append(Zone(zone_id=f"{F}-FD-{s}", floor_id=F, kind="facade_drop", name=f"{F} {s} facade drop", polygon=polys[f"FD-{s}"],
                              orientation=s, drop_id=f"FD-{s}", area_m2=round(polygon_area(polys[f"FD-{s}"]) / m2, 1)))
            zones.append(Zone(zone_id=f"{F}-{s}", floor_id=F, kind="room", name=f"{F} {s} office", polygon=polys[s],
                              area_m2=round(polygon_area(polys[s]) / m2, 1)))
        for key, kind in (("SHAFT", "shaft"), ("CORE", "core"), ("ELEC", "electrical_room")):
            zones.append(Zone(zone_id=f"{F}-{key}", floor_id=F, kind=kind, name=f"{F} {key.lower()}", polygon=polys[key],
                              area_m2=round(polygon_area(polys[key]) / m2, 1)))
        elements.append(Element(element_id=f"R1@{F}", kind="riser", zone_id=f"{F}-SHAFT", name=f"domestic water riser R1 at {F}",
                                xy=(shaft_x, cy0 + (cy1 - cy0) // 4), attrs={"system": "domestic"}))
        elements.append(Element(element_id=f"L1@{F}", kind="riser", zone_id=f"{F}-SHAFT", name=f"storm leader L1 at {F}",
                                xy=(shaft_x, cy0 + 3 * (cy1 - cy0) // 4), attrs={"system": "storm"}))
        ex = (elec_x0 + elec_x1) // 2
        elements.append(Element(element_id=f"P-{F}", kind="electrical_panel", zone_id=f"{F}-ELEC", name=f"panelboard {F}",
                                xy=(ex, cy0 + 25), rating_a=100.0, voltage_v=208.0))
        for k in range(1, 5):
            elements.append(Element(element_id=f"C-{F}-{k}", kind="circuit", zone_id=f"{F}-ELEC", name=f"circuit {k} of P-{F}",
                                    xy=(ex, cy0 + 60 + 25 * (k - 1)), rating_a=20.0, voltage_v=120.0))
            elements.append(Element(element_id=f"LD-{F}-{k}", kind="load", zone_id=f"{F}-{CIRCUIT_SIDE[k]}",
                                    name=f"office receptacles {F} {CIRCUIT_SIDE[k]}"))
        # sensors
        for s in SIDES:
            sensors.append(Sensor(sensor_id=f"{F}-{s}-RH", type="humidity", zone_id=f"{F}-{s}", unit="%RH", synthetic=True))
        sensors.append(Sensor(sensor_id=f"{F}-CORE-RH", type="humidity", zone_id=f"{F}-CORE", unit="%RH", synthetic=True))
        sensors.append(Sensor(sensor_id=f"{F}-CORE-T", type="temperature", zone_id=f"{F}-CORE", unit="degC", synthetic=True))
        sensors.append(Sensor(sensor_id=f"{F}-SHAFT-LK", type="leak", zone_id=f"{F}-SHAFT", unit="wet", synthetic=True))
        sensors.append(Sensor(sensor_id=f"{F}-ELEC-LK", type="leak", zone_id=f"{F}-ELEC", unit="wet", synthetic=True))
        for k in range(1, 5):
            sensors.append(Sensor(sensor_id=f"C-{F}-{k}-A", type="circuit_current", zone_id=f"{F}-ELEC", unit="A",
                                  element_id=f"C-{F}-{k}", synthetic=True))
        # water edges
        for s in SIDES:
            edges.append(Edge(src=f"{F}-FD-{s}", dst=f"{F}-{s}", kind="drains_to"))
            edges.append(Edge(src=f"{F}-CORE", dst=f"{F}-{s}", kind="adjacent"))
            if n > 1:
                edges.append(Edge(src=f"{F}-FD-{s}", dst=f"{fid(n - 1)}-FD-{s}", kind="above"))
                edges.append(Edge(src=f"{F}-{s}", dst=f"{fid(n - 1)}-{s}", kind="above"))
        edges.append(Edge(src=f"R1@{F}", dst=f"{F}-SHAFT", kind="drains_to"))
        edges.append(Edge(src=f"{F}-SHAFT", dst=f"{F}-CORE", kind="adjacent"))
        edges.append(Edge(src=f"{F}-CORE", dst=f"{F}-ELEC", kind="adjacent"))
        if n > 1:
            edges.append(Edge(src=f"{F}-SHAFT", dst=f"{fid(n - 1)}-SHAFT", kind="above"))
            edges.append(Edge(src=f"R1@{F}", dst=f"R1@{fid(n - 1)}", kind="drains_to", weight=0.3,
                              basis="[team-proposed, validate] weak: a riser leak mostly exits into its own shaft"))
        # power edges
        edges.append(Edge(src="SWB-1", dst=f"P-{F}", kind="feeds"))
        for k in range(1, 5):
            edges.append(Edge(src=f"P-{F}", dst=f"C-{F}-{k}", kind="feeds"))
            edges.append(Edge(src=f"C-{F}-{k}", dst=f"LD-{F}-{k}", kind="feeds"))

    elements.append(Element(element_id="SWB-1", kind="switchboard", zone_id=f"{fid(1)}-ELEC", name="main switchboard",
                            xy=((elec_x0 + elec_x1) // 2, cy1 - 20), rating_a=1200.0, voltage_v=208.0))
    RF = "RF"
    floors.append(Floor(floor_id=RF, level=top + 1, elevation_m=top * cfg.floor_height_m, plan_image=f"plans/{RF}.png",
                        plan_size_px=cfg.plan_size, px_per_m=ppm))
    for q, (rd, under) in ROOF_QUADRANTS.items():
        poly = polys[q]
        zones.append(Zone(zone_id=q, floor_id=RF, kind="roof", name=f"roof quadrant {q[5:]}", polygon=poly,
                          area_m2=round(polygon_area(poly) / m2, 1)))
        cx = (poly[0][0] + poly[1][0]) // 2
        cy = (poly[0][1] + poly[2][1]) // 2
        elements.append(Element(element_id=rd, kind="roof_drain", zone_id=q, name=f"roof drain {rd}", xy=(cx, cy)))
        edges.append(Edge(src=q, dst=rd, kind="drains_to"))
        edges.append(Edge(src=rd, dst=f"L1@{fid(top)}", kind="drains_to"))
        for s in under:
            edges.append(Edge(src=q, dst=f"{fid(top)}-{s}", kind="above"))

    return Building(building_id="SYN-TOWER-01", name="Synthetic Tower (SYNTHETIC)", address="synthetic building, no real address",
                    height_m=top * cfg.floor_height_m, floors=top, synthetic=True, floor_list=floors, zones=zones,
                    elements=elements, sensors=sensors, edges=edges)


_ZONE_FILL = {"room": (245, 245, 240), "facade_drop": (205, 220, 235), "roof": (225, 225, 215), "shaft": (215, 225, 245),
              "electrical_room": (250, 235, 205), "core": (230, 230, 230)}


def draw_floor_plan(b: Building, floor_id: str) -> Image.Image:
    """Plan PNG: zone polygons, labels, core, north arrow, 10 m scale bar and a SYNTHETIC stamp."""
    f = b.floor(floor_id)
    if f is None:
        raise KeyError(floor_id)
    img = Image.new("RGB", tuple(f.plan_size_px), (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for z in b.zones_on(floor_id):
        d.polygon(z.polygon, fill=_ZONE_FILL.get(z.kind, (240, 240, 240)), outline=(70, 70, 70))
    for z in b.zones_on(floor_id):
        xs = [p[0] for p in z.polygon]
        ys = [p[1] for p in z.polygon]
        cx, cy = sum(xs) // len(xs), sum(ys) // len(ys)
        if z.kind == "facade_drop":
            continue
        label = z.zone_id.replace(f"{floor_id}-", "")
        d.text((cx - 3 * len(label), cy - 5), label, fill=(40, 40, 40), font=font)
    for s in SIDES:
        z = b.zone(f"{floor_id}-FD-{s}")
        if z is None:
            continue
        xs = [p[0] for p in z.polygon]
        ys = [p[1] for p in z.polygon]
        pos = {"N": ((min(xs) + max(xs)) // 2 + 60, 4), "S": ((min(xs) + max(xs)) // 2 + 60, max(ys) - 16),
               "E": (max(xs) - 26, (min(ys) + max(ys)) // 2 + 40), "W": (4, (min(ys) + max(ys)) // 2 + 40)}[s]
        d.text(pos, f"FD-{s}", fill=(30, 60, 110), font=font)
    w, h = img.size
    if f.px_per_m:
        bar = int(round(10 * f.px_per_m))
        d.rectangle([w - 40 - bar, h - 50, w - 40, h - 44], fill=(20, 20, 20))
        d.text((w - 40 - bar, h - 42), "10 m", fill=(20, 20, 20), font=font)
    draw_north_arrow(d, 38, 30)
    d.text((60, 34), f"{floor_id}  level {f.level}  elev {f.elevation_m:.0f} m", fill=(20, 20, 20), font=font)
    return draw_synthetic_stamp(img)


# ----------------------------------------------------------------------------------------- weather

def synth_weather(cfg: TowerConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Hourly rain and wind (index ts, UTC). Storms every storm_every_days +/- 1 day, 4-14 h, 6-40 mm, 4-14 m/s, wind
    from the fixed cycle E, W, S, N, E, SW, NW, SE +/- 20 degrees. Light drizzle only away from storms. The storm list
    is in df.attrs["storms"] (generator truth, used by tests and truth.json)."""
    idx = _hours(cfg)
    n = len(idx)
    rain = np.zeros(n)
    speed = np.clip(3.0 + lfilter([1.0], [1.0, -0.95], rng.normal(0, 0.5, n)), 0.3, 12.0)
    wdir = np.mod(90.0 + np.cumsum(rng.normal(0, 8.0, n)), 360.0)
    storms: List[dict] = []
    near = np.zeros(n, dtype=bool)
    t_h = cfg.storm_every_days * 12.0 + rng.uniform(-12, 12)
    i = 0
    while True:
        start = int(round(t_h))
        dur = int(rng.integers(4, 15))
        if start + dur + 1 >= n:
            break
        total = float(rng.uniform(6.0, 40.0))
        sp = float(rng.uniform(4.0, 14.0))
        label = WIND_CYCLE[i % len(WIND_CYCLE)]
        dirn = float(np.mod(AZIMUTH[label] + rng.uniform(-20.0, 20.0), 360.0))
        prof = np.sin(np.pi * (np.arange(dur) + 0.5) / dur)
        hourly = np.round(total * prof / prof.sum(), 1)
        hourly[hourly < 0.1] = 0.1
        rain[start:start + dur] = hourly
        speed[start:start + dur] = np.clip(sp * (1 + rng.normal(0, 0.08, dur)), 0.5, None)
        wdir[start:start + dur] = np.mod(dirn + rng.normal(0, 4.0, dur), 360.0)
        near[max(0, start - 12):min(n, start + dur + 12)] = True
        storms.append({"storm_id": f"ST-{i + 1:03d}", "start": _ts(cfg, start), "end": _ts(cfg, start + dur - 1),
                       "start_h": start, "end_h": start + dur - 1, "total_mm": round(float(hourly.sum()), 1),
                       "wind_from": label, "wind_dir_deg": round(dirn, 1), "wind_speed_ms": round(sp, 1)})
        t_h += (cfg.storm_every_days + rng.uniform(-1.0, 1.0)) * 24.0
        i += 1
    drizzle = (rng.random(n) < 0.01) & ~near
    rain[drizzle] = np.round(rng.uniform(0.1, 0.6, int(drizzle.sum())), 1)
    df = pd.DataFrame({"rain_mm": np.round(rain, 1), "wind_speed_ms": np.round(speed, 1), "wind_dir_deg": np.round(wdir, 1),
                       "synthetic": True}, index=idx)
    df.attrs["storms"] = storms
    return df


# ----------------------------------------------------------------------------------------- scenarios

def scenarios(cfg: TowerConfig) -> List[ScenarioTruth]:
    """The five injected, labelled scenarios. Positions depend only on cfg."""
    p = positions(cfg)
    top, c, m, l, o = fid(p["top"]), fid(p["c"]), fid(p["m"]), fid(p["l"]), fid(p["o"])
    m1, m2 = fid(p["m"] - 1), fid(p["m"] - 2)
    return [
        ScenarioTruth("S1", "facade_crack_wdr", [f"{c}-FD-E"], [f"{c}-E", f"{c}-FD-E"], cfg.start, "S2",
                      f"East facade crack at {c}: {c}-E-RH rises with east wind-driven rain only; tenant tickets after big east storms; "
                      f"one drawn facade image of {c}-FD-E with a crack and a stain."),
        ScenarioTruth("S2", "blocked_roof_drain", ["RD-2", "ROOF-NE"], [f"{top}-N", f"{top}-E", "ROOF-NE", "RD-2"],
                      _ts(cfg, p["s2_onset_h"]), "S1",
                      f"RD-2 blocks at onset: after it {top}-N-RH (and {top}-E-RH, also under ROOF-NE) rise with every storm's rain, "
                      "whatever the wind direction; no response before onset."),
        ScenarioTruth("S3", "riser_leak", [f"R1@{m}"], [f"{m}-SHAFT", f"{m1}-SHAFT", f"{m2}-CORE", f"R1@{m}"],
                      _ts(cfg, p["s3_onset_h"]), "S3",
                      f"Domestic riser leak at {m}: {m}-SHAFT-LK wet from onset + 2 h, {m1}-SHAFT-LK from onset + 36 h, "
                      f"{m2}-CORE-RH ramps +20 %RH from onset + 72 h; not rain-linked."),
        ScenarioTruth("S4", "loose_lug", [f"C-{l}-2", f"P-{l}"], [f"C-{l}-2", f"P-{l}"], _ts(cfg, p["s4_onset_h"]), "S3",
                      f"Loose lug on C-{l}-2: monthly IR delta-T grows linearly 1 -> 25 K from onset to the end; other circuits N(0.3, 0.3) K."),
        ScenarioTruth("S5", "overloaded_circuit", [f"C-{o}-3"], [f"C-{o}-3", f"P-{o}"], _ts(cfg, p["s5_onset_h"]), "S2",
                      f"C-{o}-3 carries 17.0-19.5 A (85-97 % of 20 A) on weekdays 09-18 in the last 30 % of the period."),
    ]


def _kernel() -> np.ndarray:
    """Unit-peak gamma-shaped response kernel: peak at 8 h, tail to about 36 h."""
    x = np.arange(KERNEL_LEN_H, dtype=float)
    return (x / KERNEL_PEAK_H) ** 2 * np.exp(2.0 - x / (KERNEL_PEAK_H / 2.0))


def _respond(driver: np.ndarray) -> np.ndarray:
    return np.convolve(driver, _kernel())[: len(driver)]


def wdr_hourly(weather: pd.DataFrame, orientation: str) -> np.ndarray:
    """Hourly wind-driven-rain proxy (rules WDR_PROXY) on a facade of the given orientation."""
    ang = np.deg2rad(weather["wind_dir_deg"].to_numpy(float) - AZIMUTH[orientation])
    return weather["rain_mm"].to_numpy(float) * weather["wind_speed_ms"].to_numpy(float) * np.clip(np.cos(ang), 0.0, None)


def synth_sensors(b: Building, weather: pd.DataFrame, cfg: TowerConfig, rng: np.random.Generator,
                  truth: Sequence[ScenarioTruth]) -> pd.DataFrame:
    """Wide hourly sensor frame (index ts, one column per sensor in building order). Baselines plus injected scenarios."""
    idx = weather.index
    n = len(idx)
    hours = np.arange(n, dtype=float)
    hod = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    doy0 = pd.Timestamp(cfg.start).dayofyear
    season = np.sin(2 * np.pi * ((doy0 + hours / 24.0) - 105.0) / 365.0)
    diurnal = np.sin(2 * np.pi * (hod - 9) / 24.0)
    active = {s.scenario_id for s in truth}
    p = positions(cfg)
    cols: Dict[str, np.ndarray] = {}
    phi = 0.9
    for s in b.sensors:
        if s.type == "humidity":
            noise = lfilter([1.0], [1.0, -phi], rng.normal(0, 1.2 * math.sqrt(1 - phi ** 2), n))
            cols[s.sensor_id] = 42.0 + rng.uniform(-2, 2) + 6.0 * season + 3.0 * diurnal + noise
        elif s.type == "temperature":
            noise = lfilter([1.0], [1.0, -phi], rng.normal(0, 0.4 * math.sqrt(1 - phi ** 2), n))
            cols[s.sensor_id] = 22.0 + 1.0 * diurnal + noise
        elif s.type == "leak":
            cols[s.sensor_id] = np.zeros(n)
        elif s.type == "circuit_current":
            day_level = rng.uniform(6.0, 11.0)
            night_level = rng.uniform(2.0, 3.0)
            office = (dow < 5) & (hod >= 8) & (hod < 19)
            cols[s.sensor_id] = np.clip(np.where(office, day_level + rng.normal(0, 0.4, n), night_level + rng.normal(0, 0.15, n)), 0.0, None)
        else:
            cols[s.sensor_id] = np.full(n, np.nan)

    if "S1" in active:
        cols[f"{fid(p['c'])}-E-RH"] = cols[f"{fid(p['c'])}-E-RH"] + S1_GAIN * _respond(wdr_hourly(weather, "E"))
    if "S2" in active:
        after = weather["rain_mm"].to_numpy(float) * (hours >= p["s2_onset_h"])
        resp = _respond(after)
        cols[f"{fid(p['top'])}-N-RH"] = cols[f"{fid(p['top'])}-N-RH"] + S2_GAIN_N * resp
        cols[f"{fid(p['top'])}-E-RH"] = cols[f"{fid(p['top'])}-E-RH"] + S2_GAIN_E * resp
    if "S3" in active:
        on = p["s3_onset_h"]
        cols[f"{fid(p['m'])}-SHAFT-LK"] = (hours >= on + 2).astype(float)
        cols[f"{fid(p['m'] - 1)}-SHAFT-LK"] = (hours >= on + 36).astype(float)
        ramp = np.clip((hours - (on + 72)) / 48.0, 0.0, 1.0)
        cols[f"{fid(p['m'] - 2)}-CORE-RH"] = cols[f"{fid(p['m'] - 2)}-CORE-RH"] + S3_CORE_RISE * ramp
    if "S5" in active:
        col = f"C-{fid(p['o'])}-3-A"
        hot = (hours >= p["s5_onset_h"]) & (dow < 5) & (hod >= 9) & (hod < 18)
        cols[col] = np.where(hot, rng.uniform(17.0, 19.5, n), cols[col])

    for s in b.sensors:
        if s.type == "humidity":
            cols[s.sensor_id] = np.clip(cols[s.sensor_id], 15.0, 99.0)
    df = pd.DataFrame({sid: np.round(v, 1) for sid, v in cols.items()}, index=idx)
    return df


def synth_thermal(b: Building, cfg: TowerConfig, rng: np.random.Generator, truth: Sequence[ScenarioTruth]) -> pd.DataFrame:
    """Monthly IR readings (day 10, 40, 70, ... at 10:00 UTC) for every circuit. delta N(0.3, 0.3) K, except the S4
    circuit whose delta grows linearly from 1 K at onset to 25 K at the end. load_pct 45-70."""
    p = positions(cfg)
    s4 = f"C-{fid(p['l'])}-2" if any(s.scenario_id == "S4" for s in truth) else None
    onset_d = p["s4_onset_h"] / 24.0
    circuits = [e.element_id for e in b.elements if e.kind == "circuit"]
    rows = []
    for d in range(10, cfg.days, 30):
        ts = _ts(cfg, d * 24 + 10)
        for cid in circuits:
            ref = 30.0 + rng.normal(0, 1.0)
            delta = rng.normal(0.3, 0.3)
            if cid == s4 and d >= onset_d:
                delta = 1.0 + 24.0 * (d - onset_d) / (cfg.days - onset_d)
            rows.append({"ts": ts, "element_id": cid, "t_element_c": round(ref + delta, 1), "t_reference_c": round(ref, 1),
                         "t_ambient_c": round(22.0 + rng.normal(0, 0.5), 1), "load_pct": round(float(rng.uniform(45.0, 70.0)), 1),
                         "synthetic": True})
    return pd.DataFrame(rows, columns=["ts", "element_id", "t_element_c", "t_reference_c", "t_ambient_c", "load_pct", "synthetic"])


def synth_schedule(b: Building) -> pd.DataFrame:
    parent = {e.dst: e.src for e in b.edges if e.kind == "feeds"}
    load_of = {e.src: e.dst for e in b.edges if e.kind == "feeds" and e.src.startswith("C-")}
    rows = []
    for e in b.elements:
        if e.kind != "circuit":
            continue
        panel = parent[e.element_id]
        ld = b.element(load_of[e.element_id])
        rows.append({"panel_id": panel, "fed_from": parent.get(panel, ""), "circuit_id": e.element_id, "breaker_a": e.rating_a,
                     "poles": 1, "voltage_v": e.voltage_v, "load_desc": ld.name if ld else "", "zone_id": ld.zone_id if ld else ""})
    return pd.DataFrame(rows, columns=["panel_id", "fed_from", "circuit_id", "breaker_a", "poles", "voltage_v", "load_desc", "zone_id"])


def synth_tickets(b: Building, weather: pd.DataFrame, sensors: pd.DataFrame, cfg: TowerConfig, rng: np.random.Generator,
                  truth: Sequence[ScenarioTruth]) -> pd.DataFrame:
    """S1 tenant tickets: one "brown stain under east window" 1 day after each east storm whose measured rise in the
    S1 room exceeds 15 %RH (rise = max over storm + 48 h minus the 24 h pre-storm median)."""
    rows = []
    if any(s.scenario_id == "S1" for s in truth):
        p = positions(cfg)
        zone = f"{fid(p['c'])}-E"
        series = sensors[f"{zone}-RH"].to_numpy(float)
        n = len(series)
        k = 0
        for st in weather.attrs.get("storms", []):
            if st["wind_from"] != "E":
                continue
            s0, s1 = st["start_h"], st["end_h"]
            pre = np.median(series[max(0, s0 - 24):s0]) if s0 > 0 else series[0]
            post = series[s0:min(n, s1 + 49)].max()
            if post - pre > TICKET_RISE and s1 + 24 < n:
                k += 1
                rows.append({"ts": _ts(cfg, s1 + 24), "ticket_id": f"T-{k:04d}", "zone_id": zone, "text": "brown stain under east window",
                             "category": "water", "status": "open", "synthetic": True})
    return pd.DataFrame(rows, columns=["ts", "ticket_id", "zone_id", "text", "category", "status", "synthetic"])


# ----------------------------------------------------------------------------------------- facade drawings

def _facade_drawing(zone_id: str, defect: bool, size: Tuple[int, int] = (800, 600)) -> Image.Image:
    """A drawn (not photographed) curtain-wall facade: sky, spandrel bands, glass panels, mullions. With defect=True a
    jagged crack in a spandrel and a brown stain streak under the window. Deterministic from zone_id."""
    w, h = size
    seed = sum(ord(ch) for ch in zone_id)
    r = np.random.default_rng(seed)
    img = Image.new("RGB", size, (190, 205, 220))
    d = ImageDraw.Draw(img)
    for y in range(0, 90):
        c = int(170 + y * 0.6)
        d.line([(0, y), (w, y)], fill=(c - 30, c - 10, c + 20 if c + 20 < 256 else 255))
    band_h, glass_h = 60, 150
    y = 90
    while y < h:
        d.rectangle([0, y, w, y + band_h], fill=(176, 170, 160))
        y += band_h
        for x in range(0, w, 160):
            shade = int(r.integers(95, 125))
            d.rectangle([x + 6, y, x + 154, y + glass_h], fill=(shade - 30, shade, shade + 40))
            d.line([x + 20, y + 10, x + 60, y + glass_h - 10], fill=(200, 215, 230), width=2)
            d.rectangle([x, y, x + 6, y + glass_h], fill=(60, 60, 64))
        y += glass_h
    if defect:
        x0, y0 = 250, 100
        pts = [(x0, y0)]
        for _ in range(14):
            x0 += int(r.integers(10, 22))
            y0 = int(np.clip(y0 + r.integers(-6, 9), 92, 148))
            pts.append((x0, y0))
        d.line(pts, fill=(35, 30, 28), width=3)
        for yy in range(300, 440):
            a = (yy - 300) / 140.0
            col = (int(150 + 40 * a), int(110 + 50 * a), int(70 + 60 * a))
            d.line([(262 + int(6 * math.sin(yy / 9)), yy), (300 + int(8 * math.sin(yy / 13)), yy)], fill=col)
    d.text((10, h - 20), f"{zone_id}  drawn facade, not a photo", fill=(255, 255, 255), font=ImageFont.load_default())
    return draw_synthetic_stamp(img)


def synth_facade_images(b: Building, out: Path, truth: Sequence[ScenarioTruth]) -> List[ImageRecord]:
    """6 drawn facade images, 800 x 600 PNG, labels {"synthetic": True, ...}. One shows the S1 drop with a crack and a
    stain. captured_on is the last day of the period."""
    out = Path(out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    s1 = next((s for s in truth if s.scenario_id == "S1"), None)
    s1_zone = s1.root_nodes[0] if s1 else None
    top = b.floors
    picks = [z for z in [s1_zone, f"{s1_zone[:3]}-FD-W" if s1_zone else None, f"{fid(top)}-FD-N", f"{fid(2)}-FD-S",
                         f"{fid(max(2, top // 2))}-FD-W", f"{fid(3)}-FD-E"] if z]
    seen: List[str] = []
    for z in picks:
        if z not in seen and b.zone(z) is not None:
            seen.append(z)
    k = 4
    while len(seen) < 6 and k <= top:
        z = f"{fid(k)}-FD-N"
        if z not in seen:
            seen.append(z)
        k += 1
    records: List[ImageRecord] = []
    for z in seen[:6]:
        defect = z == s1_zone
        img = _facade_drawing(z, defect)
        p = img_dir / f"syn_{z}.png"
        img.save(p, format="PNG")
        records.append(ImageRecord(image_id=f"syn_{z}", path=str(p.resolve()), sha256=sha256_of(p), width=img.width, height=img.height,
                                   asset_class="facade_element", modality="rgb", captured_on=None, source_dataset="synthetic_tower",
                                   split="building_demo", asset_id=z,
                                   labels={"synthetic": True, "scenario": "S1" if defect else None,
                                           "drawn": "crack in spandrel and brown water stain" if defect else "clean curtain wall"}))
    return records


# ----------------------------------------------------------------------------------------- writer

def _write_csv(df: pd.DataFrame, path: Path, index_ts: bool = False) -> None:
    out = df.copy()
    if index_ts:
        out.insert(0, "ts", out.index.strftime(TS_FORMAT))
    out.to_csv(path, index=False, float_format="%.1f", lineterminator="\n", encoding="utf-8")


def generate_tower(out: Path, cfg: TowerConfig = TowerConfig()) -> BuildingData:
    """Write the full building folder (section-5 layout plus truth.json and ground_truth.json). Same cfg gives
    byte-identical CSVs (gzip with mtime 0)."""
    out = Path(out)
    (out / FILES["plans"]).mkdir(parents=True, exist_ok=True)
    seeds = np.random.SeedSequence(cfg.seed).spawn(4)
    r_weather, r_sensors, r_thermal, r_tickets = (np.random.default_rng(s) for s in seeds)
    b = build_tower_model(cfg)
    truth = scenarios(cfg)
    for f in b.floor_list:
        draw_floor_plan(b, f.floor_id).save(out / f.plan_image, format="PNG")
    weather = synth_weather(cfg, r_weather)
    sensors = synth_sensors(b, weather, cfg, r_sensors, truth)
    thermal = synth_thermal(b, cfg, r_thermal, truth)
    schedule = synth_schedule(b)
    tickets = synth_tickets(b, weather, sensors, cfg, r_tickets, truth)
    last_day = (pd.Timestamp(cfg.start) + pd.Timedelta(days=cfg.days - 1)).strftime("%Y-%m-%d")
    records = [r.model_copy(update={"captured_on": last_day}) for r in synth_facade_images(b, out, truth)]
    write_manifest(records, out / FILES["images"])

    _write_csv(weather, out / FILES["weather"], index_ts=True)
    s = sensors.copy()
    s.insert(0, "ts", s.index.strftime(TS_FORMAT))
    s["synthetic"] = True
    csv_bytes = s.to_csv(index=False, float_format="%.1f", lineterminator="\n").encode("utf-8")
    (out / FILES["sensors"]).write_bytes(gzip.compress(csv_bytes, compresslevel=6, mtime=0))
    _write_csv(thermal, out / FILES["thermal"])
    _write_csv(schedule, out / FILES["schedule"])
    _write_csv(tickets, out / FILES["tickets"])

    storms = [{k: v for k, v in st.items() if not k.endswith("_h")} for st in weather.attrs["storms"]]
    truth_doc = {
        "synthetic": True,
        "note": "SYNTHETIC tower: generated data with injected scenarios. Ground truth for tests and the UI truth overlay only; "
                "analysis code never reads this file; no accuracy is claimed from it.",
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
        "generator": {"S1_GAIN": S1_GAIN, "S2_GAIN_N": S2_GAIN_N, "S2_GAIN_E": S2_GAIN_E, "S3_CORE_RISE": S3_CORE_RISE,
                      "kernel_peak_h": KERNEL_PEAK_H, "ticket_rise": TICKET_RISE},
        "scenarios": [asdict(t) for t in truth],
        "storm_count": len(storms),
        "storms": storms,
    }
    text = json.dumps(truth_doc, indent=1)
    (out / FILES["truth"]).write_text(text, encoding="utf-8")
    (out / FILES["ground_truth"]).write_text(text, encoding="utf-8")
    save_building(b, out)
    return load_building_dir(out)


def load_truth(root: Path) -> dict:
    """truth.json of a synthetic building (tests and the UI truth overlay only)."""
    p = Path(root) / FILES["truth"]
    if not p.exists():
        p = Path(root) / FILES["ground_truth"]
    return json.loads(p.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the SYNTHETIC demo tower (labelled synthetic; no accuracy claims).")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--floors", type=int, default=32)
    ap.add_argument("--days", type=int, default=365)
    a = ap.parse_args(argv)
    cfg = TowerConfig(floors=a.floors, days=a.days, seed=a.seed)
    data = generate_tower(a.out, cfg)
    print(f"SYNTHETIC tower written to {a.out}: {cfg.floors} floors, {len(data.building.zones)} zones, "
          f"{len(data.building.sensors)} sensors x {len(data.sensors)} hours, {len(data.thermal)} IR readings, "
          f"{len(data.tickets)} tickets, {len(data.images)} facade drawings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
