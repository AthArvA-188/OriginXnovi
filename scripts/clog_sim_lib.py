"""Offline simulation library for the clog module (cerebro_ml env ONLY: needs wntr 1.5.0 and pyswmm 2.1.0).

Moved out of src/cascade/building/clog/ so runtime code imports only runtime packages. Used by
scripts/clog_riser_sim.py, scripts/clog_drain_sim.py and tests/test_clog_sims.py. The pure-numpy pieces stay in the
package: clog.sim_riser (test schedule, add_tests), clog.drain (drain-down grading rule).

Riser: one simulation = one building-week at 10-min steps (1,008 steps), demand-driven, Hazen-Williams, SYNTHETIC
32-floor riser. A clog is an extra minor-loss coefficient K on one pipe (or on a PRV-station strainer).
Drain: stack base -> 10 m DN100 building drain -> cleanout (level sensor) -> ORIFICE "CLOG" -> 15 m DN100 -> sewer;
a partial clog is the orifice opening (1.0 = clean).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import Dict, Optional

import numpy as np

from cascade.building.clog import geometry as G
from cascade.building.clog.drain import TEST_START_S
from cascade.building.clog.sim_riser import FLOW_LINKS


# ------------------------------------------------------------------------------------------------ riser (WNTR)
def build_network(dem: np.ndarray, clog_pipe: Optional[str] = None, K: float = 0.0,
                  roughness: float = G.ROUGHNESS_C):
    import wntr

    wn = wntr.network.WaterNetworkModel()
    h = wn.options.hydraulic
    h.headloss = "H-W"
    h.demand_model = "DD"
    # zero-flow night steps make the PRVs flap; 10 extra trials then continue (checked against WNTRSimulator in
    # scripts/clog_riser_sim.py --check, written to eval/clog/sim_check.json)
    h.unbalanced = "CONTINUE"
    h.unbalanced_value = 10
    n = dem.shape[1]
    t = wn.options.time
    t.duration = (n - 1) * 600
    t.hydraulic_timestep = 600
    t.pattern_timestep = 600
    t.report_timestep = 600

    def k_of(name: str, base: float = 0.0) -> float:
        return base + (K if clog_pipe == name else 0.0)

    wn.add_reservoir("BOOST", base_head=G.BOOSTER_HEAD)
    for node in ["B0", "XL", "SL_OUT", "XM", "SM_OUT"]:
        wn.add_junction(node, base_demand=0.0, elevation=G.node_elevation(node))
    for f in range(1, G.FLOORS + 1):
        wn.add_pattern(f"p{f}", list(dem[f - 1]))
        wn.add_junction(G.fid(f), base_demand=1.0, demand_pattern=f"p{f}", elevation=G.z_of_floor(f))
    E, Z = G.EXPRESS_DIAM, G.ZONE_DIAM
    wn.add_pipe("X0", "BOOST", "B0", length=1.0, diameter=E, roughness=roughness)
    wn.add_pipe("X1", "B0", "XL", length=G.z_of_floor(1), diameter=E, roughness=roughness)
    wn.add_pipe("X2", "XL", "XM", length=G.z_of_floor(12) - G.z_of_floor(1), diameter=E, roughness=roughness)
    wn.add_pipe("X3", "XM", "F23", length=G.z_of_floor(23) - G.z_of_floor(12), diameter=E, roughness=roughness)
    # PRV stations (strainer modelled as a short pipe with a minor loss, then the PRV)
    wn.add_pipe("STR_L", "XL", "SL_OUT", length=0.5, diameter=Z, roughness=roughness,
                minor_loss=k_of("STR_L", G.STRAINER_K_CLEAN))
    wn.add_valve("PRV_L", "SL_OUT", "F01", diameter=Z, valve_type="PRV", initial_setting=G.PRV_SET)
    wn.add_pipe("STR_M", "XM", "SM_OUT", length=0.5, diameter=Z, roughness=roughness,
                minor_loss=k_of("STR_M", G.STRAINER_K_CLEAN))
    wn.add_valve("PRV_M", "SM_OUT", "F12", diameter=Z, valve_type="PRV", initial_setting=G.PRV_SET)
    # zone risers: pipe Pnn ends at floor nn
    for zone in G.ZONES.values():
        lo, hi = zone["floors"]
        for f in range(lo + 1, hi + 1):
            name = f"P{f:02d}"
            wn.add_pipe(name, G.fid(f - 1), G.fid(f), length=G.FLOOR_H, diameter=Z, roughness=roughness,
                        minor_loss=k_of(name))
    return wn


def simulate(dem: np.ndarray, clog_pipe: Optional[str] = None, K: float = 0.0, roughness: float = G.ROUGHNESS_C,
             tmp_root: Optional[str] = None) -> Dict[str, object]:
    """Run one week. Returns pressures (steps x sensors), zone flows (steps x 3, m3/s), engine and seconds."""
    import warnings

    import wntr

    wn = build_network(dem, clog_pipe, K, roughness)
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="clogsim_", dir=tmp_root)
    engine = "EPANET"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(tmp, "run"))
            except Exception:
                engine = "WNTRSimulator"
                res = wntr.sim.WNTRSimulator(build_network(dem, clog_pipe, K, roughness)).run_sim(
                    convergence_error=True)
        nonconv = sum("converge" in str(w.message) for w in caught)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    p = res.node["pressure"][G.PRESSURE_SENSORS].to_numpy(dtype=np.float32)
    q = res.link["flowrate"][FLOW_LINKS].to_numpy(dtype=np.float32)
    return {"p": p, "q": q, "engine": engine, "nonconverged_steps": int(nonconv), "seconds": time.time() - t0}


# ------------------------------------------------------------------------------------------------ drain (SWMM)
INP = """[TITLE]
Cerebro building drain clog test (SYNTHETIC)
[OPTIONS]
FLOW_UNITS LPS
INFILTRATION HORTON
FLOW_ROUTING DYNWAVE
START_DATE 01/01/2026
START_TIME 00:00:00
REPORT_START_DATE 01/01/2026
REPORT_START_TIME 00:00:00
END_DATE 01/01/2026
END_TIME 00:20:00
REPORT_STEP 00:00:05
ROUTING_STEP 0:00:01
ALLOW_PONDING NO
[JUNCTIONS]
;name elev maxdepth initdepth surdepth aponded
STACKBASE 10.00 1.5 0 0 0
CLEANOUT  9.90  1.5 0 0 0
J3        9.85  1.5 0 0 0
[OUTFALLS]
SEWER 9.70 FREE NO
[CONDUITS]
;name from to length n inoff outoff
C1 STACKBASE CLEANOUT 10 0.011 0 0
C2 J3 SEWER 15 0.011 0 0
[ORIFICES]
;name from to type offset qcoeff gated closetime
CLOG CLEANOUT J3 SIDE 0 0.65 NO 0
[XSECTIONS]
C1 CIRCULAR 0.10 0 0 0 1
C2 CIRCULAR 0.10 0 0 0 1
CLOG CIRCULAR 0.10 0 0 0
"""


def simulate_drain(opening: float, base_lps: float, test_lps: float = 2.5, test_s: int = 60,
                   tmp_root: Optional[str] = None) -> Dict[str, np.ndarray]:
    """One 20-min SWMM run at 1-s routing. Returns time (s), cleanout depth (m) and stack-base depth (m)."""
    from pyswmm import Links, Nodes, Simulation

    tmp = tempfile.mkdtemp(prefix="clogswmm_", dir=tmp_root)
    inp = os.path.join(tmp, "drain.inp")
    with open(inp, "w", encoding="ascii") as f:
        f.write(INP)
    t, co, sb = [], [], []
    try:
        with Simulation(inp) as sim:
            clog = Links(sim)["CLOG"]
            nodes = Nodes(sim)
            stack, clean = nodes["STACKBASE"], nodes["CLEANOUT"]
            start = sim.start_time
            for _ in sim:
                el = (sim.current_time - start).total_seconds()
                clog.target_setting = opening
                q = base_lps + (test_lps if TEST_START_S <= el < TEST_START_S + test_s else 0.0)
                stack.generated_inflow(q)
                t.append(el)
                co.append(clean.depth)
                sb.append(stack.depth)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"t": np.asarray(t), "cleanout": np.asarray(co), "stackbase": np.asarray(sb)}
