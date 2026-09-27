"""Gravity-drain clog test: known-volume nightly drain-down test in a SYNTHETIC building drain (pure Python).

The pyswmm model and simulator live in scripts/clog_sim_lib.py (cerebro_ml env only), so that nothing under src/
imports pyswmm. Model: stack base -> 10 m DN100 building drain -> cleanout (level sensor) -> ORIFICE "CLOG" -> 15 m
DN100 -> sewer. A partial clog is the orifice opening (1.0 = clean). The test discharges a known volume (2.5 L/s for
60 s, below the 4.0 L/s DN100 stack limit) on top of a small night base flow; the cleanout level is logged with noise
and a 1 cm step. Signals: drain-down time (peak -> baseline + 2 cm; right-censored when the level never returns) and
peak depth.
"""
from __future__ import annotations

from typing import Dict

DURATION_S = 1200
TEST_START_S = 60


def grade_ratio(ratio: float, censored: bool, bands: Dict[str, float]) -> str:
    """Transparent drain-down rule against the commissioning value (team-proposed bands)."""
    if censored or ratio >= bands["urgent"]:
        return "urgent"
    if ratio >= bands["watch"]:
        return "watch"
    return "normal"
