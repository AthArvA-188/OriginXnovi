"""REAL demand for the riser simulation: HSB Living Lab apartment water use (Zenodo 22076411, CC BY 4.0).

* ``value`` is m3 per 10 min (its sum matches the stated 3,271 m3 total; DATA_DICTIONARY.txt says litres).
* Timestamps are UTC; they are converted to Europe/Stockholm local time so the 03:00 test lands at local night.
* Per-apartment totals include hot and cold fixture meters; rows without an apartment (shared spaces) are dropped.
* A "good week" is a Monday-start local week in which at least 20 apartments have records on every day.

A synthetic building maps each of its 32 x 6 units to one HSB apartment plus a day shift (0-6), so floor demand is a
sum of REAL apartment weeks. The mapping is a building property: the train and test buildings use different maps.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .geometry import FLOORS, UNITS_PER_FLOOR

HSB_MEMBER = "HSB_Living_Lab_Water_Consumption_Anonymized.csv"
STEPS_PER_DAY = 144
STEPS_PER_WEEK = 7 * STEPS_PER_DAY


def load_hsb_apartments(zip_path: Path) -> pd.DataFrame:
    """Wide frame: local-time 10-min index x apartment, m3 per 10 min (zeros where no record)."""
    with zipfile.ZipFile(zip_path) as z:
        df = pd.read_csv(z.open(HSB_MEMBER), sep=";", usecols=["timestamp", "apartment", "value"])
    df = df.dropna(subset=["apartment"])
    ts = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC").dt.tz_convert("Europe/Stockholm").dt.tz_localize(None)
    df = df.assign(timestamp=ts.dt.floor("10min"))
    apt = df.groupby(["timestamp", "apartment"])["value"].sum().unstack(fill_value=0.0)  # DST fall-back: summed
    full = pd.date_range(apt.index.min().floor("D"), apt.index.max().ceil("D"), freq="10min")
    return apt.reindex(full, fill_value=0.0)


def good_weeks(apt: pd.DataFrame, min_apartments: int = 20) -> List[pd.Timestamp]:
    daily = apt.resample("D").sum()
    active = (daily > 0).sum(axis=1)
    good = set(active[active >= min_apartments].index)
    return [d for d in sorted(good) if d.dayofweek == 0 and all(d + pd.Timedelta(days=k) in good for k in range(7))]


def split_weeks(weeks: List[pd.Timestamp]) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
    """Calendar split: train = weeks starting 2019-2021, test = weeks starting 2022 or later."""
    return [w for w in weeks if w.year <= 2021], [w for w in weeks if w.year >= 2022]


def building_map(apartments: List[str], seed: int) -> List[Tuple[str, int]]:
    """One (apartment, day-shift) pair per unit, floors 1..32 x UNITS_PER_FLOOR units."""
    rng = np.random.default_rng(seed)
    n = FLOORS * UNITS_PER_FLOOR
    apts = rng.choice(np.array(apartments), size=n, replace=True)
    shifts = rng.integers(0, 7, size=n)
    return [(str(a), int(s)) for a, s in zip(apts, shifts)]


def week_matrix(apt: pd.DataFrame, week_start: pd.Timestamp) -> Dict[str, np.ndarray]:
    idx = pd.date_range(week_start, periods=STEPS_PER_WEEK, freq="10min")
    sub = apt.reindex(idx, fill_value=0.0)
    return {c: sub[c].to_numpy(dtype=float) for c in sub.columns}


def floor_demands(week: Dict[str, np.ndarray], bmap: List[Tuple[str, int]], scale: float = 1.0) -> np.ndarray:
    """floors x steps demand in m3/s (m3 per 10 min / 600) for one week of one building."""
    out = np.zeros((FLOORS, STEPS_PER_WEEK))
    for u, (a, shift) in enumerate(bmap):
        f = u // UNITS_PER_FLOOR
        out[f] += np.roll(week[a], shift * STEPS_PER_DAY)
    return out * scale / 600.0
