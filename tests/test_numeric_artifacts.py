"""The committed numeric artifacts are internally consistent: the headline numbers in backtest_bdg2.json can be
recomputed from the committed forecast parquets, the baseline is present, labels are stated, pins match.
Reads only eval/numeric (committed); skipped when the artifacts have not been generated."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cascade.numeric import HF_MODELS
from cascade.numeric import metrics as M

EVAL = Path(__file__).resolve().parents[1] / "eval" / "numeric"
pytestmark = pytest.mark.skipif(not (EVAL / "backtest_bdg2.json").exists(), reason="numeric artifacts missing")


@pytest.fixture(scope="module")
def bt():
    return json.loads((EVAL / "backtest_bdg2.json").read_text(encoding="utf-8"))


def test_backtest_has_baseline_n_and_real_label(bt):
    assert bt["baseline"] == "snaive168" and "snaive168" in bt["models"]
    assert bt["n_meters"] == len(bt["selection"]["chosen"]) and bt["n_scored_hours"] > 0
    assert bt["data"]["label"] == "REAL"
    assert not set(bt["selection"]["site_of"].values()) & set(bt["selection"]["excluded_sites"])


def test_headline_wape_recomputes_from_committed_forecasts(bt):
    fr = pd.read_parquet(EVAL / "forecasts_baselines_hgb.parquet")
    fr["ts"] = pd.to_datetime(fr["ts"])
    for name, spec in HF_MODELS.items():
        p = EVAL / f"forecasts_{spec.tag}.parquet"
        if p.exists() and spec.tag in bt["models"]:
            c = pd.read_parquet(p)
            c["ts"] = pd.to_datetime(c["ts"])
            fr = fr.merge(c[["meter", "ts", "q50"]].rename(columns={"q50": f"{spec.tag}_q50"}), on=["meter", "ts"])
    cols = ["y"] + [f"{m}_q50" for m in bt["models"]]
    ok = fr[cols].notna().all(axis=1)
    assert int(ok.sum()) == bt["n_scored_hours"]
    f = fr[ok]
    for m, d in bt["models"].items():
        assert M.wape(f.y, f[f"{m}_q50"]) == pytest.approx(d["WAPE_pct"], rel=1e-9)


def test_hf_provenance_pins_match_the_agent(bt):
    for tag, p in bt["hf_provenance"].items():
        spec = [s for s in HF_MODELS.values() if s.tag == tag][0]
        assert p["model_id"] == spec.model_id and p["revision"] == spec.revision
        assert p["resolved_snapshot"] == spec.revision


def test_lead_json_is_aggregate_only():
    lead = json.loads((EVAL / "anomaly_lead.json").read_text(encoding="utf-8"))
    assert set(lead["models"]) == {"robust_z", "isolation_forest", "hgb_supervised"}
    assert lead["chance_pr_auc"] == pytest.approx(lead["prevalence_test"])
    text = json.dumps(lead)
    assert "timestamp" not in text and "meter_reading" not in text  # no labelled series published
    assert not any(p.name.startswith("lead") for p in EVAL.glob("*.parquet"))


def test_catalogue_has_no_pilot_numbers_and_lists_chosen_models():
    cat = json.loads((EVAL / "model_catalogue.json").read_text(encoding="utf-8"))
    ids = {r["id"]: r for r in cat["models"]}
    assert ids["amazon/chronos-2"]["hf_licence_tag"] == "apache-2.0"
    assert ids["amazon/chronos-2"]["revision"] == HF_MODELS["chronos-2"].revision
    for r in cat["models"]:
        assert "WAPE" not in r["reason"] and "pilot" not in r["reason"].lower()


def test_artifact_sizes_within_budget():
    total = sum(p.stat().st_size for p in EVAL.rglob("*") if p.is_file())
    assert total < 25e6
    assert all(p.stat().st_size < 10e6 for p in EVAL.rglob("*") if p.is_file())


def test_indicator_share_uses_scored_hours_and_matches_the_parquet(bt):
    ind = bt["indicators"]
    assert ind["flag_share"] == pytest.approx(ind["n_flagged"] / ind["n_hours"])
    assert "1.4826" in ind["threshold_unit"] and "different baseline" in ind["threshold_source"]
    df = pd.read_parquet(EVAL / "bdg2_indicators.parquet")
    scored = df.robust_z.notna()
    assert int(scored.sum()) == ind["n_hours"] and int(df.flag[scored].sum()) == ind["n_flagged"]
    assert bool((df.robust_z[df.flag] >= ind["threshold"]).all())


def test_paired_differences_match_the_model_wapes(bt):
    pw = bt["paired_wape_diff"]
    assert pw["pairs"] and "clustering" in pw["method"]
    for p in pw["pairs"]:
        assert p["diff_pp"] == pytest.approx(bt["models"][p["a"]]["WAPE_pct"] - bt["models"][p["b"]]["WAPE_pct"])
        assert p["ci95_pp"][0] <= p["ci95_pp"][1]
    assert set(bt["per_site_WAPE_pct"]["snaive168"]) == set(bt["selection"]["site_of"].values())
    assert sum(bt["meters_per_site"].values()) == bt["n_meters"]


def test_selection_diagnostics_agree_with_the_selection(bt):
    d, sel = bt["selection_diagnostics"], bt["selection"]
    assert d["pool_at_chosen_limit"] == {"before_exclusion": sel["n_eligible_before_exclusion"],
                                         "after_exclusion": sel["n_eligible_after_exclusion"]}
    assert d["pool_without_flat_rule"]["after_exclusion"] >= sel["n_eligible_after_exclusion"]
    assert not set(sel["chosen"]) & set(d["removed_by_flat_rule"])


def test_lead_threshold_is_on_the_building_layer_scale():
    lead = json.loads((EVAL / "anomaly_lead.json").read_text(encoding="utf-8"))
    rz = lead["models"]["robust_z"]
    assert "1.4826" in rz["at_threshold"]["unit"]
    assert "raw MADs" in rz["at_threshold_raw_mads"]["unit"]
    # the sd-equivalent scale is stricter than raw MADs at the same K
    assert rz["at_threshold"]["flag_share"] <= rz["at_threshold_raw_mads"]["flag_share"]


def test_catalogue_states_what_was_measured():
    cat = json.loads((EVAL / "model_catalogue.json").read_text(encoding="utf-8"))
    for r in cat["models"]:
        assert (r["measured_scope"] != "not measured") == bool(r["measured_here"])
