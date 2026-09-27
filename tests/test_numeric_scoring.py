"""Review fixes for the numeric layer (SYNTHETIC series only; no accuracy claims):
robust z on the building layer's 1.4826 x MAD scale, paired bootstrap differences and per-site WAPE in score(),
and the flat-day selection diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cascade.numeric import datasets as D
from cascade.numeric import metrics as M
from cascade.numeric.backtest import score
from cascade.numeric.baselines import MAD_TO_SD, RobustZ


def noisy_weekly(weeks: int = 8, seed: int = 0) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=168 * weeks, freq="h")
    rng = np.random.default_rng(seed)
    return pd.Series(np.tile(rng.uniform(10, 50, 168), weeks) + rng.normal(0, 1.0, 168 * weeks), index=idx)


def test_robust_z_scale_matches_the_building_layer_constant():
    electrical = pytest.importorskip("cascade.building.electrical")
    assert MAD_TO_SD == electrical.MAD_TO_SD


def test_robust_z_default_is_raw_mads_divided_by_1_4826():
    y = noisy_weekly()
    sd = RobustZ().scores(y)
    raw = RobustZ(mad_to_sd=1.0).scores(y)
    ok = sd.notna() & raw.notna()
    assert ok.sum() > 500
    np.testing.assert_allclose((sd[ok] * MAD_TO_SD).to_numpy(), raw[ok].to_numpy(), rtol=1e-5)
    r = RobustZ(threshold=5.0).anomaly_score(y)
    assert "1.4826" in r.provenance.params["threshold_unit"]
    assert RobustZ(mad_to_sd=1.0).unit() == "raw MADs"
    # the stricter scale flags no more hours than raw MADs at the same K
    assert (sd[ok] >= 5).sum() <= (raw[ok] >= 5).sum()


def _frame() -> tuple:
    """3 meters on 2 sites; model 'good' is closer to y than 'base' on every meter."""
    rng = np.random.default_rng(1)
    rows = []
    for b, level in (("A1", 20.0), ("A2", 40.0), ("B1", 80.0)):
        y = level + rng.normal(0, 2.0, 240)
        rows.append(pd.DataFrame({"meter": b, "y": y, "base_q50": y + rng.normal(0, 6.0, 240),
                                  "good_q50": y + rng.normal(0, 1.0, 240),
                                  "other_q50": y + rng.normal(0, 3.0, 240)}))
    return pd.concat(rows, ignore_index=True), {"A1": "A", "A2": "A", "B1": "B"}


def test_paired_differences_use_the_same_draws_and_match_point_values():
    fr, site_of = _frame()
    scales = {"A1": 1.0, "A2": 1.0, "B1": 1.0}
    res = score(fr, scales, baseline="base", n_boot=300, primary="good", site_of=site_of)
    plain = score(fr, scales, baseline="base", n_boot=300)
    for m in res["models"]:  # the marginal intervals do not move when paired differences are added
        assert res["models"][m]["WAPE_pct_ci95"] == plain["models"][m]["WAPE_pct_ci95"]
    pairs = {(p["a"], p["b"]): p for p in res["paired_wape_diff"]["pairs"]}
    assert set(pairs) == {("good", "base"), ("other", "base"), ("good", "other")}
    for (a, b), p in pairs.items():
        assert p["diff_pp"] == pytest.approx(res["models"][a]["WAPE_pct"] - res["models"][b]["WAPE_pct"])
        lo, hi = p["ci95_pp"]
        assert lo <= hi
    g = pairs[("good", "base")]
    assert g["ci95_pp"][1] < 0 and g["share_of_resamples_a_lower"] == 1.0
    a_rows = fr[fr.meter.isin(["A1", "A2"])]
    assert res["per_site_WAPE_pct"]["good"]["A"] == pytest.approx(M.wape(a_rows.y, a_rows.good_q50))
    assert res["meters_per_site"] == {"A": 2, "B": 1}
    assert "clustering" in res["paired_wape_diff"]["method"]


def test_flat_day_stats_counts_flat_days_only_from_the_window_start():
    idx = pd.date_range("2016-12-30", periods=24 * 12, freq="h")
    y = pd.Series(np.random.default_rng(0).uniform(10, 20, len(idx)), index=idx)
    y.loc["2017-01-03":"2017-01-05 23:00"] = 12.0  # 3 constant days (test only)
    share, cnt, days = D.flat_day_stats(y, "2017-01-01")
    assert (cnt, days) == (3, 10) and share == pytest.approx(0.3)


def test_selection_diagnostics_on_a_tiny_synthetic_pool():
    idx = pd.date_range("2017-01-01", "2017-11-30 23:00", freq="h")
    rng = np.random.default_rng(0)
    ok = rng.uniform(10, 50, len(idx))
    stuck = rng.uniform(10, 50, len(idx))
    stuck[(idx >= "2017-04-01") & (idx < "2017-11-01")] = 7.0  # stuck meter (test only)
    elec = pd.DataFrame({"Aa_office_ok": ok, "Aa_office_stuck": stuck, "Bear_office_x": rng.uniform(10, 50, len(idx))},
                        index=idx)
    meta = pd.DataFrame({"building_id": list(elec.columns), "site_id": ["Aa", "Aa", "Bear"],
                         "primaryspaceusage": ["Office"] * 3})
    chosen, info = D.select_offices(meta, elec, n=1, seed=0)
    diag = D.selection_diagnostics(meta, elec, n=1, seed=0)
    assert diag["pool_without_flat_rule"] == {"before_exclusion": 3, "after_exclusion": 2}
    assert diag["pool_at_chosen_limit"] == {"before_exclusion": info["n_eligible_before_exclusion"],
                                            "after_exclusion": info["n_eligible_after_exclusion"]} \
        == {"before_exclusion": 2, "after_exclusion": 1}
    r = diag["removed_by_flat_rule"]["Aa_office_stuck"]
    assert r["min_kwh_in_window"] == r["max_kwh_in_window"] == 7.0 and not r["in_excluded_site"]
    assert chosen == ["Aa_office_ok"] and len(diag["draw_without_flat_rule_same_seed"]) == 1
