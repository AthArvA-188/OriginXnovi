"""Nightly test schedule for the SYNTHETIC riser simulation (pure numpy; runtime-safe).

The WNTR network builder and simulator live in scripts/clog_sim_lib.py (cerebro_ml env only), so that nothing under
src/ imports wntr. This module keeps the parts both environments share.

One simulation = one building-week at 10-min steps (1,008 steps). Each night the BMS runs two short tests per zone
at the zone's top floor (4.0 L/s at 03:00/03:10/03:20 for zones L/M/H, then 2.5 L/s at 03:40/03:50/04:00); the step
before each block (02:50, 03:30) is the pre-test reference.
"""
from __future__ import annotations

import numpy as np

from . import geometry as G

TESTS = {  # test id -> (flow L/s, {zone: step}, pre-test step)
    "A": (4.0, {"L": 18, "M": 19, "H": 20}, 17),
    "B": (2.5, {"L": 22, "M": 23, "H": 24}, 21),
}
FLOW_LINKS = ["STR_L", "STR_M", "X3"]  # zone meters L, M, H


def add_tests(dem: np.ndarray) -> np.ndarray:
    """Add the nightly test draws (m3/s) at the top floor of each zone for every day of the week."""
    d = dem.copy()
    steps_per_day = 144
    for lps, zsteps, _pre in TESTS.values():
        for zone, step in zsteps.items():
            f = G.TEST_FLOOR[zone]
            for day in range(d.shape[1] // steps_per_day):
                d[f - 1, day * steps_per_day + step] += lps / 1000.0
    return d
