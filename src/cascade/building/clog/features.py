"""Features for riser clog detection (pure numpy/pandas; runtime-safe).

Physics: a partial clog adds a local head loss h = K v^2 / 2g (EPANET manual), which is only large at high flow.
Residential risers are almost still at night, so we compare two nearby 10-min readings: the step with a known test
draw and the step before it. For each sensor segment the jump in head loss, divided by the jump in (zone flow)^2,
is a resistance in m per (L/s)^2; subtracting the commissioning value leaves the added resistance of a clog.
Constant sensor offsets cancel in the jump; they do not cancel in passive features.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import geometry as G

STEPS_PER_DAY = 144
TESTS = {"A": (4.0, {"L": 18, "M": 19, "H": 20}, 17), "B": (2.5, {"L": 22, "M": 23, "H": 24}, 21)}
ZONE_IDX = {"L": 0, "M": 1, "H": 2}
METER_AREA = np.array([np.pi * (G.ZONE_DIAM / 2) ** 2, np.pi * (G.ZONE_DIAM / 2) ** 2,
                       np.pi * (G.EXPRESS_DIAM / 2) ** 2])  # m2 of the pipe each zone meter is clamped on
ELEV = np.array([G.node_elevation(n) for n in G.PRESSURE_SENSORS])
SEG_IDX = [(G.PRESSURE_SENSORS.index(a), G.PRESSURE_SENSORS.index(b)) for a, b in
           (G.SEGMENT_SENSORS[loc] for loc in G.LOCATIONS)]
TOP_IDX = [G.PRESSURE_SENSORS.index(n) for n in ("F11", "F22", "F32")]


def test_steps() -> List[int]:
    out = []
    for _lps, zsteps, _pre in TESTS.values():
        out += list(zsteps.values())
    return sorted(out)


def add_noise(p: np.ndarray, q: np.ndarray, rng: np.random.Generator, p_sd: float = 0.15, rel: float = 0.01,
              abs_v: float = 0.0046, mult: float = 1.0, offset: Optional[Tuple[int, float]] = None):
    """Pressure noise sd p_sd (m) per reading; flow sd = max(rel*|Q|, abs_v*area) (AW-Lake CUTT spec form).
    ``offset`` = (sensor index, metres) adds a constant drift to one pressure sensor."""
    pn = p.astype(float) + rng.normal(0.0, p_sd * mult, p.shape)
    if offset is not None:
        pn[:, offset[0]] += offset[1]
    sd_q = np.maximum(rel * np.abs(q), abs_v * METER_AREA[None, :]) * mult
    qn = q.astype(float) + rng.normal(0.0, 1.0, q.shape) * sd_q
    return pn, qn


def segment_dh(p: np.ndarray) -> np.ndarray:
    """steps x 8 head loss (m) across each location's sensor pair (piezometric head = pressure + elevation)."""
    h = p + ELEV[None, :]
    return np.stack([h[:, a] - h[:, b] for a, b in SEG_IDX], axis=1)


def nightly_resistance(p: np.ndarray, q: np.ndarray, test: str = "A") -> Tuple[np.ndarray, np.ndarray]:
    """Per night (rows) and location (cols): jump in head loss / jump in zone flow^2, m per (L/s)^2.
    Also returns the measured test-flow jump per night and zone (L/s)."""
    _lps, zsteps, pre = TESTS[test]
    dh = segment_dh(p)
    ql = q * 1000.0
    n_days = p.shape[0] // STEPS_PER_DAY
    r = np.full((n_days, len(G.LOCATIONS)), np.nan)
    dq = np.full((n_days, 3), np.nan)
    for d in range(n_days):
        t0 = d * STEPS_PER_DAY + pre
        for zone, zi in ZONE_IDX.items():
            t = d * STEPS_PER_DAY + zsteps[zone]
            dq[d, zi] = ql[t, zi] - ql[t0, zi]
        for j, loc in enumerate(G.LOCATIONS):
            zi = ZONE_IDX[G.LOCATION_ZONE[loc]]
            t = d * STEPS_PER_DAY + zsteps[G.LOCATION_ZONE[loc]]
            dq2 = ql[t, zi] ** 2 - ql[t0, zi] ** 2
            if dq2 > 0.25:  # (L/s)^2; guards against a failed test draw
                r[d, j] = (dh[t, j] - dh[t0, j]) / dq2
    return r, dq


def passive_fit(p: np.ndarray, q: np.ndarray) -> List[Tuple[float, float]]:
    """Commissioning fit dh = a * Q^1.852 + b per location over all non-test steps (Hazen-Williams exponent)."""
    mask = np.ones(p.shape[0], bool)
    for d in range(p.shape[0] // STEPS_PER_DAY):
        for s in test_steps():
            mask[d * STEPS_PER_DAY + s] = False
    dh = segment_dh(p)[mask]
    fits = []
    for j, loc in enumerate(G.LOCATIONS):
        qz = np.clip(q[mask, ZONE_IDX[G.LOCATION_ZONE[loc]]] * 1000.0, 0.0, None) ** 1.852
        A = np.vstack([qz, np.ones_like(qz)]).T
        a, b = np.linalg.lstsq(A, dh[:, j], rcond=None)[0]
        fits.append((float(a), float(b)))
    return fits


def passive_daily(p: np.ndarray, q: np.ndarray, fits, n_busy: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Per day and location: mean head-loss residual over the zone's n_busy busiest non-test steps."""
    dh = segment_dh(p)
    ql = np.clip(q * 1000.0, 0.0, None)
    n_days = p.shape[0] // STEPS_PER_DAY
    tests = set(test_steps())
    keep = np.array([s not in tests for s in range(STEPS_PER_DAY)])
    res = np.zeros((n_days, len(G.LOCATIONS)))
    qbusy = np.zeros((n_days, 3))
    for d in range(n_days):
        sl = slice(d * STEPS_PER_DAY, (d + 1) * STEPS_PER_DAY)
        for zone, zi in ZONE_IDX.items():
            qq = np.where(keep, ql[sl, zi], -1.0)
            busy = np.argsort(qq)[-n_busy:]
            qbusy[d, zi] = ql[sl, zi][busy].mean()
            for j, loc in enumerate(G.LOCATIONS):
                if G.LOCATION_ZONE[loc] != zone:
                    continue
                a, b = fits[j]
                res[d, j] = float((dh[sl, j][busy] - (a * ql[sl, zi][busy] ** 1.852 + b)).mean())
    return res, qbusy


def zone_top_min(p: np.ndarray) -> np.ndarray:
    """Per day: minimum pressure at each zone's top floor over non-test steps (what a BMS low-pressure alarm sees)."""
    n_days = p.shape[0] // STEPS_PER_DAY
    tests = set(test_steps())
    keep = np.array([s not in tests for s in range(STEPS_PER_DAY)])
    out = np.zeros((n_days, 3))
    for d in range(n_days):
        blk = p[d * STEPS_PER_DAY:(d + 1) * STEPS_PER_DAY][keep]
        out[d] = blk[:, TOP_IDX].min(axis=0)
    return out


def commissioning(p: np.ndarray, q: np.ndarray) -> Dict[str, object]:
    """Reference values from one clean commissioning week (already noised)."""
    rA, _ = nightly_resistance(p, q, "A")
    rB, _ = nightly_resistance(p, q, "B")
    return {"rA": np.nanmean(rA, axis=0), "rB": np.nanmean(rB, axis=0), "fits": passive_fit(p, q),
            "top_min": zone_top_min(p).min(axis=0)}


def night_rows(p: np.ndarray, q: np.ndarray, ref: Dict[str, object]) -> Dict[str, np.ndarray]:
    """All per-night feature blocks for one noised scenario week."""
    rA, dqA = nightly_resistance(p, q, "A")
    rB, dqB = nightly_resistance(p, q, "B")
    pas, qbusy = passive_daily(p, q, ref["fits"])
    top = zone_top_min(p)
    return {"actA": rA - ref["rA"][None, :], "actB": rB - ref["rB"][None, :],
            "dqA": dqA, "dqB": dqB,
            "pas": pas, "pas_q": qbusy, "bms_drop": (ref["top_min"][None, :] - top).max(axis=1)}


# ----------------------------------------------------------------------------------------------- drain-down test
def drain_down_seconds(t_s: np.ndarray, depth: np.ndarray, rise_m: float = 0.02,
                       base_window: Tuple[float, float] = (0.0, 55.0)) -> Tuple[float, bool]:
    """Seconds from peak depth until depth falls back below baseline + rise_m. Returns (seconds, censored):
    censored=True when the level never returns within the record (right-censored at the record end)."""
    t_s = np.asarray(t_s, float)
    depth = np.asarray(depth, float)
    base_mask = (t_s >= base_window[0]) & (t_s <= base_window[1])
    base = float(np.median(depth[base_mask])) if base_mask.any() else float(depth[0])
    ipk = int(np.argmax(depth))
    below = np.nonzero(depth[ipk:] < base + rise_m)[0]
    if len(below) == 0:
        return float(t_s[-1] - t_s[ipk]), True
    return float(t_s[ipk + below[0]] - t_s[ipk]), False


def quantise(depth: np.ndarray, rng: np.random.Generator, sd: float = 0.005, step: float = 0.01) -> np.ndarray:
    """Level sensor model: Gaussian noise then rounding to the logging step (Bellinge iFix stores 1 cm changes)."""
    return np.round((np.asarray(depth, float) + rng.normal(0.0, sd, np.shape(depth))) / step) * step


def fit_recession_tau(t_s: np.ndarray, depth: np.ndarray) -> float:
    """Exponential recession time constant (s) from a log-linear fit to the falling limb above baseline."""
    t_s = np.asarray(t_s, float)
    depth = np.asarray(depth, float)
    ipk = int(np.argmax(depth))
    base = float(np.min(depth[ipk:]))
    y = depth[ipk:] - base
    tt = t_s[ipk:] - t_s[ipk]
    m = y > 0.1 * y[0]
    if m.sum() < 3 or y[0] <= 0:
        return float("nan")
    slope = np.polyfit(tt[m], np.log(y[m]), 1)[0]
    return float(-1.0 / slope) if slope < 0 else float("inf")
