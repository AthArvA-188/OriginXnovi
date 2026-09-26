"""Building folder store, floor plans and tickets (building_spec section 5).

One folder per building:
    building.json          Building.model_dump_json(indent=1)
    plans/F01.png ... RF.png
    weather.csv            ts, rain_mm, wind_speed_ms, wind_dir_deg, synthetic
    sensors.csv.gz         ts, <sensor_id> ..., synthetic   (wide, hourly, 1 decimal)
    thermal.csv            ts, element_id, t_element_c, t_reference_c, t_ambient_c, load_pct, synthetic
    panel_schedule.csv     panel_id, fed_from, circuit_id, breaker_a, poles, voltage_v, load_desc, zone_id
    tickets.csv            ts, ticket_id, zone_id, text, category, status, synthetic
    images/manifest.jsonl  ImageRecord rows (asset_id = zone_id or element_id)
    truth.json             synthetic only: scenario ground truth, never read by analysis code
    analysis/              findings.json, queue.csv (pipeline.save_findings), observations.jsonl
    problems.json          ProblemStore
    grading/               run_cascade output for building photos
IFC 4.3 and COBie import are roadmap: ifcopenshell is not installed (ifc_supported() is False). Zones can come
from a zones CSV instead (load_zones_csv).
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from ..clientreport import LEVEL_COLOR
from ..ingest import read_manifest
from .model import Building, BuildingData, Observation, Zone

FILES: Dict[str, str] = {
    "building": "building.json",
    "plans": "plans",
    "weather": "weather.csv",
    "sensors": "sensors.csv.gz",
    "thermal": "thermal.csv",
    "schedule": "panel_schedule.csv",
    "tickets": "tickets.csv",
    "images": "images/manifest.jsonl",
    "truth": "truth.json",
    "ground_truth": "ground_truth.json",  # same content as truth.json, kept for the demo folder contract
    "analysis": "analysis",
    "observations": "analysis/observations.jsonl",
    "problems": "problems.json",
    "grading": "grading",
}

COLUMNS: Dict[str, List[str]] = {
    "weather": ["ts", "rain_mm", "wind_speed_ms", "wind_dir_deg", "synthetic"],
    "thermal": ["ts", "element_id", "t_element_c", "t_reference_c", "t_ambient_c", "load_pct", "synthetic"],
    "schedule": ["panel_id", "fed_from", "circuit_id", "breaker_a", "poles", "voltage_v", "load_desc", "zone_id"],
    "tickets": ["ts", "ticket_id", "zone_id", "text", "category", "status", "synthetic"],
}

TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
TICKET_GREY = "#9ca3af"
_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    (r"leak|drip|water|stain|mold|mould|damp|wet", "water"),
    (r"crack|spall|facade|brick|sealant", "facade"),
    (r"burning|hot|smell|trip|breaker|spark|flicker", "electrical"),
)


# ------------------------------------------------------------------------------------------ store

def save_building(b: Building, root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    p = root / FILES["building"]
    p.write_text(b.model_dump_json(indent=1), encoding="utf-8")
    return p


def load_building(path: Path) -> Building:
    """Read building.json (a folder or the file itself). Plan paths stay relative to the building folder."""
    p = Path(path)
    if p.is_dir():
        p = p / FILES["building"]
    return Building.model_validate_json(p.read_text(encoding="utf-8"))


def _utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True)


def _with_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    """Real uploads get synthetic=False when the column is missing; the generator always writes True."""
    if "synthetic" not in df.columns:
        df["synthetic"] = False
    else:
        df["synthetic"] = df["synthetic"].astype(str).str.lower().isin(["true", "1"])
    return df


def read_wide_series(path: Path) -> pd.DataFrame:
    """A wide hourly CSV (optionally .gz): index = UTC DatetimeIndex named ts, numeric float columns only.
    df.attrs["synthetic"] is True when any row of the file's synthetic column is true (the column itself is dropped)."""
    df = pd.read_csv(path)
    if "ts" not in df.columns:
        raise ValueError(f"{path}: needs a 'ts' column (ISO 8601 UTC)")
    df.index = pd.DatetimeIndex(_utc(df.pop("ts")), name="ts")
    syn = bool(_with_synthetic(df[["synthetic"]].copy())["synthetic"].any()) if "synthetic" in df.columns else False
    df = df.drop(columns=[c for c in ("synthetic",) if c in df.columns])
    out = df.apply(pd.to_numeric, errors="coerce").astype(float)
    out.attrs["synthetic"] = syn
    return out


def _empty(kind: str) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS[kind]})


def _read_weather(path: Path) -> pd.DataFrame:
    if not path.exists():
        idx = pd.DatetimeIndex([], tz="UTC", name="ts")
        return pd.DataFrame({"rain_mm": [], "wind_speed_ms": [], "wind_dir_deg": [], "synthetic": []}, index=idx).astype(
            {"rain_mm": float, "wind_speed_ms": float, "wind_dir_deg": float, "synthetic": bool})
    df = _with_synthetic(pd.read_csv(path))
    df.index = pd.DatetimeIndex(_utc(df.pop("ts")), name="ts")
    for c in ("rain_mm", "wind_speed_ms", "wind_dir_deg"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df


def _read_table(path: Path, kind: str, ts: bool) -> pd.DataFrame:
    if not path.exists():
        df = _empty(kind)
        if ts:
            df["ts"] = pd.Series(dtype="datetime64[ns, UTC]")
        if "synthetic" in df.columns:
            df["synthetic"] = pd.Series(dtype=bool)
        return df
    df = pd.read_csv(path, dtype={"zone_id": str, "ticket_id": str, "element_id": str, "panel_id": str, "circuit_id": str})
    if ts:
        df["ts"] = _utc(df["ts"])
    if "synthetic" in COLUMNS[kind]:
        df = _with_synthetic(df)
    for c in COLUMNS[kind]:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def load_building_dir(root: Path) -> BuildingData:
    """Everything in one building folder. Missing CSVs become empty frames with the section-5 columns.

    weather and sensors are indexed by a UTC DatetimeIndex named ts; thermal and tickets keep a UTC ts column."""
    root = Path(root)
    b = load_building(root)
    sensors_path = root / FILES["sensors"]
    if sensors_path.exists():
        sensors = read_wide_series(sensors_path)
    else:
        sensors = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="ts"), dtype=float)
    images_path = root / FILES["images"]
    images = read_manifest(images_path) if images_path.exists() else []
    weather = _read_weather(root / FILES["weather"])
    thermal = _read_table(root / FILES["thermal"], "thermal", ts=True)
    tickets = _read_table(root / FILES["tickets"], "tickets", ts=True)
    sources = {"sensors": bool(sensors.attrs.get("synthetic", False)),
               "weather": bool(len(weather) and weather["synthetic"].astype(bool).any()),
               "thermal": bool(len(thermal) and thermal["synthetic"].astype(bool).any()),
               "tickets": bool(len(tickets) and tickets["synthetic"].astype(bool).any())}
    if any(sources.values()) and not b.synthetic:
        b = b.model_copy(update={"synthetic": True})  # a synthetic input file makes every output of this building synthetic
    return BuildingData(
        root=root,
        building=b,
        weather=weather,
        sensors=sensors,
        thermal=thermal,
        schedule=_read_table(root / FILES["schedule"], "schedule", ts=False),
        tickets=tickets,
        images=images,
        synthetic_sources=sources,
    )


def load_zones_csv(path: Path, floor_ids: Sequence[str]) -> List[Zone]:
    """Zones from a CSV: floor_id,zone_id,name,kind,orientation,polygon with polygon "x y;x y;...". Rows on an unknown
    floor raise ValueError, so a typo never silently drops a zone."""
    known = set(floor_ids)
    out: List[Zone] = []
    with Path(path).open(encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            fid = (row.get("floor_id") or "").strip()
            if fid not in known:
                raise ValueError(f"{path} line {i}: unknown floor_id {fid!r}")
            pts = []
            for pair in (row.get("polygon") or "").split(";"):
                if pair.strip():
                    x, y = pair.split()
                    pts.append((int(round(float(x))), int(round(float(y)))))
            orient = (row.get("orientation") or "").strip() or None
            out.append(Zone(zone_id=row["zone_id"].strip(), floor_id=fid, kind=row["kind"].strip(), name=(row.get("name") or "").strip(),
                            polygon=pts, orientation=orient, drop_id=f"FD-{orient}" if row["kind"].strip() == "facade_drop" and orient else None))
    return out


# ------------------------------------------------------------------------------------------ geometry

def point_in_polygon(xy: Tuple[float, float], poly: Sequence[Tuple[int, int]]) -> bool:
    """Ray casting. Points on the left or top edge count as inside."""
    x, y = xy
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside


def polygon_area(poly: Sequence[Tuple[int, int]]) -> float:
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def centroid(poly: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    """Area centroid of a simple polygon (vertex mean when the area is zero)."""
    a = 0.0
    cx = cy = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        c = x1 * y2 - x2 * y1
        a += c
        cx += (x1 + x2) * c
        cy += (y1 + y2) * c
    if abs(a) < 1e-9:
        return (int(round(sum(p[0] for p in poly) / len(poly))), int(round(sum(p[1] for p in poly) / len(poly))))
    a *= 0.5
    return (int(round(cx / (6 * a))), int(round(cy / (6 * a))))


def zone_at(b: Building, floor_id: str, xy: Tuple[int, int]) -> Optional[str]:
    hits = [z for z in b.zones_on(floor_id) if point_in_polygon(xy, z.polygon)]
    if not hits:
        return None
    return min(hits, key=lambda z: polygon_area(z.polygon)).zone_id  # the smallest zone wins where outlines touch


def place(b: Building, obs: Observation) -> Optional[Tuple[str, Tuple[int, int]]]:
    """(floor_id, xy) for a pin: the observation's plan_xy, else its element's xy, else its zone centroid."""
    node = obs.element_id or obs.zone_id
    floor = b.floor_of(node) if node else None
    if floor is None and obs.zone_id:
        floor = b.floor_of(obs.zone_id)
    if floor is None:
        return None
    if obs.plan_xy is not None:
        return floor.floor_id, (int(obs.plan_xy[0]), int(obs.plan_xy[1]))
    if obs.element_id:
        e = b.element(obs.element_id)
        if e is not None and e.xy is not None:
            return floor.floor_id, (int(e.xy[0]), int(e.xy[1]))
        if e is not None:
            z = b.zone(e.zone_id)
            if z is not None:
                return floor.floor_id, centroid(z.polygon)
    z = b.zone(obs.zone_id) if obs.zone_id else None
    if z is not None:
        return floor.floor_id, centroid(z.polygon)
    return None


# ------------------------------------------------------------------------------------------ drawing

def plan_image(b: Building, root: Path, floor_id: str) -> Optional[Image.Image]:
    f = b.floor(floor_id)
    if f is None or not f.plan_image:
        return None
    p = Path(root) / f.plan_image
    if not p.exists():
        return None
    with Image.open(p) as im:
        return im.convert("RGB")


def _rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def draw_north_arrow(d: ImageDraw.ImageDraw, x: int, y: int, size: int = 26, color: Tuple[int, int, int] = (20, 20, 20)) -> None:
    """North arrow (FISP report item: location diagram with a north arrow). North is up on every plan."""
    d.polygon([(x, y), (x - size // 3, y + size), (x, y + int(size * 0.72)), (x + size // 3, y + size)], fill=color)
    d.text((x - 4, y + size + 2), "N", fill=color, font=ImageFont.load_default())


def draw_synthetic_stamp(img: Image.Image, text: str = "SYNTHETIC") -> Image.Image:
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    font = ImageFont.load_default()
    w, h = img.size
    for i in range(3):
        d.text((w // 2 - 40, 40 + i * (h // 3)), text, fill=(200, 30, 30, 90), font=font)
    d.rectangle([w - 110, h - 26, w - 6, h - 6], fill=(200, 30, 30, 200))
    d.text((w - 102, h - 22), text, fill=(255, 255, 255, 255), font=font)
    return Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")


def render_plan(b: Building, root: Path, floor_id: str, observations: Sequence[Observation],
                risk_by_zone: Optional[Dict[str, float]] = None, highlight: Sequence[str] = ()) -> Image.Image:
    """Floor plan with zone outlines, an optional risk tint, element glyphs and observation pins.

    Pins take clientreport.LEVEL_COLOR (U purple, never grey-as-S0); tickets are grey rings. A zone missing from
    risk_by_zone is left untinted (not measured, not zero risk). SYNTHETIC watermark when the building is synthetic."""
    f = b.floor(floor_id)
    if f is None:
        raise KeyError(f"unknown floor {floor_id!r}")
    base = plan_image(b, root, floor_id)
    if base is None or base.size != tuple(f.plan_size_px):
        base = Image.new("RGB", tuple(f.plan_size_px), (250, 250, 248))
    img = base.convert("RGBA")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    font = ImageFont.load_default()
    zones = b.zones_on(floor_id)
    risk_by_zone = risk_by_zone or {}
    hl = set(highlight)
    for z in zones:
        r = risk_by_zone.get(z.zone_id)
        if r is not None and not (isinstance(r, float) and math.isnan(r)):
            a = int(40 + 150 * max(0.0, min(1.0, float(r))))
            d.polygon(z.polygon, fill=(234, 88, 12, a))
    for z in zones:
        width = 4 if z.zone_id in hl else 1
        color = (220, 38, 38, 255) if z.zone_id in hl else (60, 60, 60, 255)
        d.line(list(z.polygon) + [z.polygon[0]], fill=color, width=width)
    for e in b.elements:
        z = b.zone(e.zone_id)
        if z is None or z.floor_id != floor_id:
            continue
        x, y = e.xy if e.xy is not None else centroid(z.polygon)
        ring = (220, 38, 38, 255) if e.element_id in hl else (30, 30, 30, 255)
        if e.kind in ("electrical_panel", "switchboard"):
            d.rectangle([x - 6, y - 6, x + 6, y + 6], outline=ring, fill=(255, 255, 255, 230), width=2 if e.element_id in hl else 1)
            d.text((x - 3, y - 5), "P" if e.kind == "electrical_panel" else "S", fill=(30, 30, 30, 255), font=font)
        elif e.kind == "roof_drain":
            d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=ring, width=2)
            d.line([x - 4, y, x + 4, y], fill=ring)
        elif e.kind == "riser":
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(37, 99, 235, 255) if e.attrs.get("system") != "storm" else (14, 116, 144, 255), outline=ring)
        elif e.kind == "circuit":
            d.rectangle([x - 2, y - 2, x + 2, y + 2], fill=ring)
    for o in observations:
        pl = place(b, o)
        if pl is None or pl[0] != floor_id:
            continue
        x, y = pl[1]
        if o.kind == "ticket" or o.level is None:
            d.ellipse([x - 8, y - 8, x + 8, y + 8], outline=_rgb(TICKET_GREY) + (255,), width=3)
        else:
            c = _rgb(LEVEL_COLOR.get(o.level, "#8b5cf6"))
            d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=c + (235,), outline=(255, 255, 255, 255), width=2)
    img = Image.alpha_composite(img, over)
    d2 = ImageDraw.Draw(img)
    draw_north_arrow(d2, 38, 30)
    d2.text((60, 50), f"{b.name}  {floor_id}", fill=(20, 20, 20), font=font)
    out = img.convert("RGB")
    return draw_synthetic_stamp(out) if b.synthetic else out


# ------------------------------------------------------------------------------------------ tickets

def ticket_category(text: str, category: Optional[str] = None) -> str:
    """The ticket's own category when it is one of water, facade, electrical, other; else a keyword map. LLM ticket
    triage is roadmap: no API call here."""
    c = (category or "").strip().lower()
    if c in ("water", "facade", "electrical", "other"):
        return c
    low = (text or "").lower()
    for pattern, cat in _KEYWORDS:
        if re.search(pattern, low):
            return cat
    return "other"


def _iso(ts) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime(TS_FORMAT) if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts).strftime(TS_FORMAT)


def tickets_to_observations(b: Building, tickets: pd.DataFrame) -> List[Observation]:
    """One ungraded observation (level None) per ticket row whose zone exists in the building."""
    out: List[Observation] = []
    if tickets is None or len(tickets) == 0:
        return out
    for row in tickets.itertuples(index=False):
        zid = str(getattr(row, "zone_id", "") or "")
        tid = str(getattr(row, "ticket_id", "") or "")
        if not zid or b.zone(zid) is None or not tid:
            continue
        text = str(getattr(row, "text", "") or "")
        cat_raw = getattr(row, "category", None)
        cat = ticket_category(text, None if cat_raw is None or (isinstance(cat_raw, float) and math.isnan(cat_raw)) else str(cat_raw))
        st_raw = getattr(row, "status", None)
        status = None if st_raw is None or st_raw is pd.NA or (isinstance(st_raw, float) and math.isnan(st_raw)) or not str(st_raw).strip()             else str(st_raw).strip().lower()
        out.append(Observation(obs_id=f"ticket:{tid}", kind="ticket", ts=_iso(row.ts), zone_id=zid, source=f"ticket:{tid}",
                               level=None, text=text, synthetic=bool(getattr(row, "synthetic", False)), category=cat, status=status))
    return out


def ifc_supported() -> bool:
    """False: ifcopenshell is not installed, so IFC 4.3 import and COBie sheets are roadmap. Use building.json or a
    zones CSV."""
    return False
