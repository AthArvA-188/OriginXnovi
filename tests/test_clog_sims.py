"""Simulator checks. They need wntr / pyswmm (cerebro_ml env) and are skipped in the origin_hack runtime env.

    E:\\conda_envs\\cerebro_ml\\python.exe -m pytest tests/test_clog_sims.py -q -p no:cacheprovider
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if SCRIPTS not in sys.path:  # the WNTR / pyswmm code lives in scripts/clog_sim_lib.py (offline only)
    sys.path.insert(0, SCRIPTS)
if importlib.util.find_spec("pydantic") is None:  # cerebro_ml has no pydantic; load clog without building/__init__
    import clog_common

    clog_common.bootstrap()


def _demand():
    from cascade.building.clog import geometry as G
    from cascade.building.clog import sim_riser as S

    dem = np.zeros((G.FLOORS, 144))
    dem[:, 60:70] = 2e-5  # small daytime draw on every floor
    return S.add_tests(dem)


def test_riser_statics_and_clog_headloss():
    pytest.importorskip("wntr")
    from cascade.building.clog import geometry as G
    from cascade.building.clog import sim_riser as S

    dem = _demand()
    import clog_sim_lib as L

    clean = L.simulate(dem)
    assert clean["engine"] == "EPANET"
    i = G.PRESSURE_SENSORS.index
    for node in ("F01", "F11", "F12", "F22", "F23", "F32"):
        assert abs(clean["p"][0, i(node)] - G.static_pressure_m(int(node[1:]))) < 0.05
    clog = L.simulate(dem, "P20", 100.0)
    step = S.TESTS["A"][1]["M"]
    drop = clean["p"][step, i("F22")] - clog["p"][step, i("F22")]
    assert 2.0 < drop < 5.0  # K v^2/2g at 4 L/s in DN80 is about 3.3 m
    assert abs(clean["p"][step, i("F17")] - clog["p"][step, i("F17")]) < 0.05  # clog above F17 is invisible below


def test_drain_down_grows_as_the_orifice_closes():
    pytest.importorskip("pyswmm")
    import clog_sim_lib as L

    from cascade.building.clog import features as F

    dd = []
    for op in (1.0, 0.3, 0.15):
        r = L.simulate_drain(op, 0.05)
        dd.append(F.drain_down_seconds(r["t"], r["cleanout"])[0])
    assert dd[0] < dd[1] < dd[2]
