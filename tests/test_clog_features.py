"""Feature maths on analytic inputs (no simulator needed)."""
import numpy as np

from cascade.building.clog import features as F
from cascade.building.clog import geometry as G


def _flat_week():
    n = 7 * F.STEPS_PER_DAY
    p = np.tile(np.array([G.BOOSTER_HEAD - G.node_elevation(s) for s in G.PRESSURE_SENSORS]), (n, 1)).astype(float)
    q = np.zeros((n, 3))
    return p, q


def test_nightly_resistance_recovers_added_headloss():
    p, q = _flat_week()
    lps, zsteps, _pre = F.TESTS["A"]
    j = G.LOCATIONS.index("M1")
    down = G.PRESSURE_SENSORS.index(G.SEGMENT_SENSORS["M1"][1])
    extra = 0.8  # m of extra head loss across M1 during the M test
    for d in range(7):
        t = d * F.STEPS_PER_DAY + zsteps["M"]
        q[t, F.ZONE_IDX["M"]] = lps / 1000.0
        p[t, down] -= extra
    r, dq = F.nightly_resistance(p, q, "A")
    assert np.allclose(r[:, j], extra / lps ** 2)
    assert np.allclose(dq[:, F.ZONE_IDX["M"]], lps)
    m2 = G.LOCATIONS.index("M2")
    assert np.allclose(r[:, m2], -extra / lps ** 2)  # the downstream sensor is shared by both M segments
    assert np.isnan(r[:, G.LOCATIONS.index("L1")]).all()  # no test flow in zone L -> undefined, not zero


def test_constant_offset_cancels_in_active_features_but_not_passive():
    p, q = _flat_week()
    for zone, step in F.TESTS["A"][1].items():
        for d in range(7):
            q[d * F.STEPS_PER_DAY + step, F.ZONE_IDX[zone]] = 0.004
    for zone, step in F.TESTS["B"][1].items():
        for d in range(7):
            q[d * F.STEPS_PER_DAY + step, F.ZONE_IDX[zone]] = 0.0025
    q += 1e-5
    ref = F.commissioning(p, q)
    shifted = p.copy()
    shifted[:, G.PRESSURE_SENSORS.index("F17")] += 0.3
    a = F.night_rows(p, q, ref)
    b = F.night_rows(shifted, q, ref)
    assert np.allclose(np.nan_to_num(a["actA"]), np.nan_to_num(b["actA"]))
    assert not np.allclose(a["pas"], b["pas"])


def test_noise_model_flow_floor():
    rng = np.random.default_rng(1)
    p = np.zeros((20000, len(G.PRESSURE_SENSORS)))
    q = np.zeros((20000, 3))
    pn, qn = F.add_noise(p, q, rng, p_sd=0.15, rel=0.01, abs_v=0.0046)
    assert abs(pn.std() - 0.15) < 0.01
    floor = 0.0046 * F.METER_AREA[0]
    assert abs(qn[:, 0].std() - floor) / floor < 0.05


def test_drain_down_and_tau_on_analytic_recession():
    t = np.arange(0, 600.0)
    depth = np.where(t < 60, 0.01, 0.01 + 0.05 * np.exp(-(t - 60) / 40.0))
    dd, cens = F.drain_down_seconds(t, depth, rise_m=0.02)
    assert not cens and abs(dd - 40.0 * np.log(0.05 / 0.02)) <= 1.0
    assert abs(F.fit_recession_tau(t, depth) - 40.0) < 1.0
    stuck = np.where(t < 60, 0.01, 0.2)
    assert F.drain_down_seconds(t, stuck)[1] is True


def test_quantise_steps():
    rng = np.random.default_rng(2)
    x = F.quantise(np.full(100, 0.034), rng, sd=0.0, step=0.01)
    assert np.allclose(x, 0.03)
