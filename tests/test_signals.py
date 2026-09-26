"""Seismic/vibration module tests on synthetic numpy series: no model, no network, no real data."""

import json

import numpy as np
import pandas as pd
import pytest

from cascade import signals
from cascade.pipeline import load_run, save_findings
from cascade.schema import Action, Evidence, Finding, Measurements, NativeScale, SignalRecord, Unified

FS = 200.0


def make_csv(path, freq, seconds=20.0, amp=1.0, noise=0.05, seed=0, zeta=None, channels=("z",), time_col=True):
    """Sinusoid (plus white noise) per channel; `zeta` turns it into a free decay exp(-zeta*w*t) sin(w*t)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    cols = {}
    if time_col:
        cols["time_s"] = t
    w = 2 * np.pi * freq
    for i, ch in enumerate(channels):
        env = np.exp(-zeta * w * t) if zeta is not None else 1.0
        cols[ch] = amp * env * np.sin(w * t + 0.3 * i) + noise * rng.standard_normal(t.size)
    pd.DataFrame(cols).to_csv(path, index=False)
    return path


def rubric():
    return signals.load_seismic_rubric()


def row_for(family, v):
    return signals._find_row(rubric(), family, v)


# ------------------------------------------------------------------------------------------------ ingest


def test_ingest_detects_time_column_and_rate(tmp_path):
    p = make_csv(tmp_path / "base.csv", 3.2)
    rec = signals.ingest_signal_csv(p, "accelerometer")
    assert isinstance(rec, SignalRecord)
    assert abs(rec.sample_rate_hz - FS) < 1e-6
    assert rec.channels == ["z"] and rec.n_samples == 4000 and abs(rec.duration_s - 20.0) < 1e-9
    assert len(rec.sha256) == 64 and rec.signal_id.startswith("base_")
    assert rec.labels["time_column"] == "time_s" and rec.labels["rate_source"].startswith("time column")
    assert rec.labels["units"] is None and rec.baseline_id is None and rec.asset_class == "bridge_element"


def test_ingest_without_time_column_needs_rate(tmp_path):
    p = make_csv(tmp_path / "raw.csv", 3.2, time_col=False, channels=("x", "y"))
    with pytest.raises(ValueError):
        signals.ingest_signal_csv(p, "accelerometer")
    rec = signals.ingest_signal_csv(p, "accelerometer", sample_rate_hz=FS, asset_id="B-1", client_id="acme", units="m/s2")
    assert rec.channels == ["x", "y"] and rec.labels["rate_source"] == "given"
    assert rec.asset_id == "B-1" and rec.client_id == "acme" and rec.labels["units"] == "m/s2"


# -------------------------------------------------------------------------------------------- indicators


def test_indicators_frequency_pga_rms_and_resolution(tmp_path):
    rec = signals.ingest_signal_csv(make_csv(tmp_path / "b.csv", 3.2, amp=1.0), "accelerometer")
    ind = signals.indicators(rec)
    assert abs(ind["fft_resolution_hz"] - FS / 4000) < 1e-9  # 0.05 Hz stated
    assert abs(ind["dominant_frequency_hz"]["z"] - 3.2) < ind["fft_resolution_hz"]
    assert 0.9 < ind["pga"]["z"] < 1.3
    assert abs(ind["rms"]["z"] - 1 / np.sqrt(2)) < 0.05
    assert ind["fft_window"] == "hann" and ind["n_samples_used"] == 4000
    assert ind["frequency_shift_pct"] is None and ind["rms_change_pct"] is None
    assert ind["pga_pct_g"] is None  # units unknown: never assumed
    assert any("no baseline" in n for n in ind["notes"]) and any("units" in n for n in ind["notes"])


def test_frequency_shift_vs_baseline_is_about_minus_9_4_pct(tmp_path):
    base = signals.ingest_signal_csv(make_csv(tmp_path / "b.csv", 3.2), "accelerometer")
    dmg = signals.ingest_signal_csv(make_csv(tmp_path / "d.csv", 2.9, seed=1), "accelerometer")
    ind = signals.indicators(dmg, base)
    assert ind["baseline_id"] == base.signal_id
    assert abs(ind["frequency_shift_pct"]["z"] - (-9.375)) < 0.5
    assert abs(ind["baseline_dominant_frequency_hz"]["z"] - 3.2) < 0.05
    assert ind["rms_change_pct"]["z"] is not None and abs(ind["rms_change_pct"]["z"]) < 10


def test_damping_from_free_decay_and_none_for_steady_tone(tmp_path):
    dec = signals.ingest_signal_csv(make_csv(tmp_path / "decay.csv", 3.0, seconds=10.0, zeta=0.05, noise=0.005), "accelerometer")
    ind = signals.indicators(dec)
    assert ind["damping_ratio"]["z"] is not None and abs(ind["damping_ratio"]["z"] - 0.05) < 0.01
    assert ind["damping_detail"]["z"]["n_peaks"] >= signals.DECAY_MIN_PEAKS
    steady = signals.ingest_signal_csv(make_csv(tmp_path / "steady.csv", 3.0), "accelerometer")
    assert signals.indicators(steady)["damping_ratio"]["z"] is None


def test_pga_pct_g_and_pgv_need_known_units(tmp_path):
    rec = signals.ingest_signal_csv(make_csv(tmp_path / "a.csv", 3.2, amp=1.0, noise=0.0), "accelerometer", units="m/s2")
    ind = signals.indicators(rec)
    assert abs(ind["pga_pct_g"]["z"] - 100 / signals.G_MS2) < 0.2  # 1 m/s2 = 10.2 %g
    geo = signals.ingest_signal_csv(make_csv(tmp_path / "v.csv", 3.2, amp=2.0, noise=0.0), "geophone", units="mm/s")
    assert abs(signals.indicators(geo)["pgv_cm_s"]["z"] - 0.2) < 0.01
    strain = signals.ingest_signal_csv(make_csv(tmp_path / "s.csv", 3.2), "strain", units="ue")
    ind_s = signals.indicators(strain)
    assert ind_s["pga_pct_g"] is None and ind_s["pgv_cm_s"] is None


def test_window_s_limits_samples(tmp_path):
    rec = signals.ingest_signal_csv(make_csv(tmp_path / "b.csv", 3.2), "accelerometer")
    ind = signals.indicators(rec, window_s=5.0)
    assert ind["n_samples_used"] == 1000 and abs(ind["fft_resolution_hz"] - 0.2) < 1e-9


# ------------------------------------------------------------------------------------------------ grading


def test_seismic_rubric_rows_are_sourced_and_cover_the_axis():
    rb = rubric()
    assert rb["standard"] == "SHM-Seismic"
    assert any(r["family"] == "baseline" and r["unified"] == "U" for r in rb["rows"])
    for r in rb["rows"]:
        assert r["criterion"] and r["unified"] in ("S0", "S1", "S2", "S3", "S4", "U")
        assert "source" in r and ("[team-proposed, validate]" in r["criterion"] + r["source"] or "R10" in r["source"])
        assert r["value"] in rb["allowed_values"]
    for fam in ("frequency_shift", "pga", "pgv", "rms_change"):
        for v in (0.0, 1.0, 7.0, 15.0, 30.0, 1000.0):
            assert row_for(fam, v) is not None, (fam, v)


def test_grade_is_u_without_baseline(tmp_path):
    rec = signals.ingest_signal_csv(make_csv(tmp_path / "d.csv", 2.9), "accelerometer")
    f = signals.grade_signal(rec, signals.indicators(rec), rubric())
    assert f.unified.level == "U" and f.native_scale.value == "U"
    assert "not_measurable" in f.unified.flags and f.measurements.confidence == 0.0
    base_row = next(r for r in rubric()["rows"] if r["family"] == "baseline")
    assert f.native_scale.criteria_matched == [base_row["criterion"]]
    assert f.modality == "seismic" and f.asset_class == "bridge_element" and f.native_scale.standard == "SHM-Seismic"
    assert f.evidence.signal_id == rec.signal_id and f.evidence.image_ids == [rec.signal_id] and f.evidence.baseline_id is None
    assert f.action.code == "monitor" and f.queue_score is None


def test_grade_with_baseline_uses_frequency_shift_row(tmp_path):
    base = signals.ingest_signal_csv(make_csv(tmp_path / "b.csv", 3.2), "accelerometer")
    dmg = signals.ingest_signal_csv(make_csv(tmp_path / "d.csv", 2.9, seed=1), "accelerometer", asset_class="building_disaster")
    ind = signals.indicators(dmg, base)
    f = signals.grade_signal(dmg, ind, rubric())
    expected = row_for("frequency_shift", abs(ind["frequency_shift_pct"]["z"]))
    assert expected["value"] == "df 5-10%"  # 9.4 % sits in the 5-10 band
    assert f.unified.level == expected["unified"] and f.native_scale.value == expected["value"]
    assert expected["criterion"] in f.native_scale.criteria_matched
    assert f.action.code == expected["action"] and "seismic_shm.json" in f.action.basis
    assert f.defect_type == "seismic_frequency_shift" and f.asset_class == "building_disaster"
    assert f.evidence.baseline_id == base.signal_id and f.measurements.confidence == 1.0
    assert "shift -9." in f.justification and "3.200 Hz" in f.justification
    assert "not_measurable" in f.unified.flags  # PGA family skipped: units unknown


def test_grade_takes_worst_family_and_pga_row_with_units(tmp_path):
    base = signals.ingest_signal_csv(make_csv(tmp_path / "b.csv", 3.2, amp=1.0, noise=0.0), "accelerometer", units="m/s2")
    same = signals.ingest_signal_csv(make_csv(tmp_path / "s.csv", 3.2, amp=1.0, noise=0.0, seed=2), "accelerometer", units="m/s2")
    ind = signals.indicators(same, base)
    f = signals.grade_signal(same, ind, rubric())
    pga_row = row_for("pga", ind["pga_pct_g"]["z"])
    assert pga_row["value"] == "PGA 6.2-11.5%g" and f.unified.level == pga_row["unified"]  # 10.2 %g; shift ~0 gives S0
    assert f.native_scale.value == pga_row["value"] and pga_row["criterion"] in f.native_scale.criteria_matched
    assert "not_measurable" not in f.unified.flags
    big = signals.ingest_signal_csv(make_csv(tmp_path / "big.csv", 3.2, amp=5.0, noise=0.0), "accelerometer", units="m/s2")
    fb = signals.grade_signal(big, signals.indicators(big, base), rubric())
    assert fb.unified.level == "S4" and fb.action.code == "escalate"  # 51 %g: 'Severe / Moderate-heavy' row


# ------------------------------------------------------------------------------------------------- export


def fake_image_finding(iid="img_0", level="S2"):
    return Finding(
        finding_id=f"{iid}/full", asset_class="steel_coating", defect_type="corrosion",
        native_scale=NativeScale(standard="CorrosionCS", value="Poor", criteria_matched=["row"]),
        unified=Unified(level=level, uncertainty="+/-1", flags=[]),
        measurements=Measurements(area_cm2=None, crack_width_mm=None, delta_t_k=None, percent_area_rusted=10.0, section_loss_pct=None, confidence=0.8),
        action=Action(code="schedule", sla_days=90, basis="row"), justification="fake", evidence=Evidence(image_ids=[iid]), model="fake",
    )


def test_write_signal_findings_is_readable_by_load_run_and_merges_with_image_findings(tmp_path):
    out = tmp_path / "run"
    save_findings([fake_image_finding()], out)
    (out / "summary.json").write_text(json.dumps({"findings": 1, "levels": {"S2": 1}}), encoding="utf-8")
    base = signals.ingest_signal_csv(make_csv(tmp_path / "b.csv", 3.2), "accelerometer")
    dmg = signals.ingest_signal_csv(make_csv(tmp_path / "d.csv", 2.9, seed=1), "accelerometer")
    f = signals.grade_signal(dmg, signals.indicators(dmg, base), rubric())
    ranked = signals.write_signal_findings(out, [f])
    assert {x.finding_id for x in ranked} == {"img_0/full", f.finding_id}
    loaded = load_run(out)["findings"]
    sig = next(x for x in loaded if x.finding_id == f.finding_id)
    assert sig.modality == "seismic" and sig.unified.level == "S2" and sig.queue_rank in (1, 2) and sig.evidence.signal_id == dmg.signal_id
    assert next(x for x in loaded if x.finding_id == "img_0/full").modality == "rgb"
    assert (out / "queue.csv").read_text(encoding="utf-8").count(dmg.signal_id) >= 1
    assert (out / "findings.jsonl").read_text(encoding="utf-8").count("\n") == 1
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["findings"] == 2 and summary["levels"]["S2"] == 2 and summary["signal_findings"] == 1
    # rewriting the same finding replaces it, no duplicate line
    signals.write_signal_findings(out, [f])
    assert len(load_run(out)["findings"]) == 2 and (out / "findings.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_cli_end_to_end(tmp_path, capsys):
    b = make_csv(tmp_path / "b.csv", 3.2)
    d = make_csv(tmp_path / "d.csv", 2.9, seed=1)
    out = tmp_path / "run"
    rc = signals.main(["--csv", str(d), "--sensor", "accelerometer", "--rate", "200", "--baseline", str(b), "--out", str(out), "--asset-id", "B-1"])
    assert rc == 0
    findings = load_run(out)["findings"]
    assert len(findings) == 1 and findings[0].unified.level == "S2" and findings[0].modality == "seismic"
    rows = [json.loads(l) for l in (out / "signals.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["record"]["asset_id"] == "B-1" and rows[0]["indicators"]["baseline_id"] == rows[0]["record"]["baseline_id"]
    assert "S2" in capsys.readouterr().out
