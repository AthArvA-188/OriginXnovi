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
    assert f.evidence.baseline_id == base.signal_id
    # -9.4 % +/- one FFT bin (0.05 Hz / 3.2 Hz = 1.56 %) crosses the 10 % row edge: confidence is lowered
    assert abs(ind["frequency_shift_uncertainty_pct"]["z"] - 100 * 0.05 / ind["baseline_dominant_frequency_hz"]["z"]) < 1e-6
    assert f.measurements.confidence == signals.STRADDLE_CONFIDENCE and "straddles" in f.justification
    assert "shift -9." in f.justification and "3.200 Hz" in f.justification
    assert "not_measurable" in f.unified.flags  # PGA family skipped: units unknown


def test_grade_takes_worst_family_and_pga_row_with_units(tmp_path):
    base = signals.ingest_signal_csv(make_csv(tmp_path / "b.csv", 3.2, amp=1.0, noise=0.0), "accelerometer", units="m/s2", mount="free_field")
    same = signals.ingest_signal_csv(make_csv(tmp_path / "s.csv", 3.2, amp=1.0, noise=0.0, seed=2), "accelerometer", units="m/s2", mount="free_field")
    ind = signals.indicators(same, base)
    f = signals.grade_signal(same, ind, rubric())
    pga_row = row_for("pga", ind["pga_pct_g"]["z"])
    assert pga_row["value"] == "PGA 6.2-11.5%g" and f.unified.level == pga_row["unified"]  # 10.2 %g; shift ~0 gives S0
    assert f.native_scale.value == pga_row["value"] and pga_row["criterion"] in f.native_scale.criteria_matched
    assert "not_measurable" not in f.unified.flags
    big = signals.ingest_signal_csv(make_csv(tmp_path / "big.csv", 3.2, amp=5.0, noise=0.0), "accelerometer", units="m/s2", mount="free_field")
    fb = signals.grade_signal(big, signals.indicators(big, base), rubric())
    assert fb.unified.level == "S4" and fb.action.code == "escalate"  # 51 %g: 'Severe / Moderate-heavy' row


def test_no_baseline_with_known_units_and_low_pga_is_u_not_s0(tmp_path):
    # ambient shaking 0.02 m/s2 (about 0.2 %g) matches 'PGA<6.2%g' (S0), but site shaking says nothing about the structure
    for mount in ("free_field", None):
        rec = signals.ingest_signal_csv(make_csv(tmp_path / f"amb_{mount}.csv", 3.2, amp=0.02, noise=0.0), "accelerometer", units="m/s2", mount=mount)
        ind = signals.indicators(rec)
        assert ind["pga_pct_g"]["z"] < 6.2
        f = signals.grade_signal(rec, ind, rubric())
        assert f.unified.level == "U" and f.native_scale.value == "U" and f.measurements.confidence == 0.0
        assert "not_measurable" in f.unified.flags
        base_row = next(r for r in rubric()["rows"] if r["family"] == "baseline")
        assert f.native_scale.criteria_matched == [base_row["criterion"]]  # the S0 PGA row is not quoted
        assert "never S0" in f.justification or "mount" in f.justification


def test_no_baseline_high_pga_on_free_field_can_still_raise_the_level(tmp_path):
    rec = signals.ingest_signal_csv(make_csv(tmp_path / "strong.csv", 3.2, amp=1.0, noise=0.0), "accelerometer", units="m/s2", mount="free_field")
    f = signals.grade_signal(rec, signals.indicators(rec), rubric())
    assert f.unified.level == "S1" and f.native_scale.value == "PGA 6.2-11.5%g"  # 10.2 %g
    assert "not_measurable" in f.unified.flags


def test_structure_mounted_sensor_skips_shakemap_rows(tmp_path):
    # 0.13 g on a deck would read 'PGA 11.5-21.5%g' (S2); ShakeMap describes free-field ground shaking
    rec = signals.ingest_signal_csv(make_csv(tmp_path / "deck.csv", 2.0, amp=0.13, noise=0.0), "accelerometer", units="g", mount="structure")
    ind = signals.indicators(rec)
    assert abs(ind["pga_pct_g"]["z"] - 13.0) < 0.2 and ind["mount"] == "structure"
    f = signals.grade_signal(rec, ind, rubric())
    assert f.unified.level == "U" and "free_field" in f.justification


def test_seismic_pga_pgv_rows_carry_the_free_field_tag():
    rb = rubric()
    assert "Unified mapping is the team's own" in rb["source_note"]
    for r in rb["rows"]:
        if r["family"] in ("pga", "pgv"):
            assert "structure-mounted sensor" in r["source"] and "[team-proposed, validate]" in r["source"]


def _two_mode_csv(path, a1, a2, f1=1.1, f2=3.4, seconds=60.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    z = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t) + 0.01 * rng.standard_normal(t.size)
    pd.DataFrame({"time_s": t, "z": z}).to_csv(path, index=False)
    return path


def test_mode_jump_between_records_is_not_graded_as_a_frequency_shift(tmp_path):
    # same structure (modes 1.1 and 3.4 Hz unchanged); only the excitation changes which mode dominates
    base = signals.ingest_signal_csv(_two_mode_csv(tmp_path / "wind.csv", 1.0, 0.3), "accelerometer")
    rec = signals.ingest_signal_csv(_two_mode_csv(tmp_path / "traffic.csv", 0.3, 1.0, seed=1), "accelerometer")
    ind = signals.indicators(rec, base)
    assert abs(ind["dominant_frequency_hz"]["z"] - 3.4) < 0.05  # the global peak jumped mode
    assert abs(ind["tracked_frequency_hz"]["z"] - 1.1) < 0.02 and abs(ind["frequency_shift_pct"]["z"]) < 2.0
    f = signals.grade_signal(rec, ind, rubric())
    assert f.unified.level == "S0" and f.native_scale.value == "df<2%" and "load_posting_review" not in f.unified.flags


def test_mode_not_found_makes_frequency_family_not_assessable(tmp_path):
    base = signals.ingest_signal_csv(make_csv(tmp_path / "b.csv", 3.2), "accelerometer")
    far = signals.ingest_signal_csv(make_csv(tmp_path / "far.csv", 6.0, seed=1), "accelerometer")
    ind = signals.indicators(far, base)
    assert ind["frequency_shift_pct"]["z"] is None and any("mode not found" in s for s in ind["frequency_shift_skipped"])
    f = signals.grade_signal(far, ind, rubric())
    assert f.unified.level == "U" and f.measurements.confidence == 0.0


def test_single_channel_s3_is_kept_but_not_full_confidence(tmp_path):
    base = signals.ingest_signal_csv(make_csv(tmp_path / "b.csv", 3.2), "accelerometer")
    dmg = signals.ingest_signal_csv(make_csv(tmp_path / "d.csv", 2.72, seed=1), "accelerometer")  # -15 %
    f = signals.grade_signal(dmg, signals.indicators(dmg, base), rubric())
    assert f.unified.level == "S3" and f.native_scale.value == "df>=10%"
    assert f.measurements.confidence == signals.UNCORROBORATED_CONFIDENCE and "not corroborated" in f.justification


def test_coarse_fft_bin_makes_frequency_family_not_assessable(tmp_path):
    base = signals.ingest_signal_csv(make_csv(tmp_path / "b.csv", 3.2), "accelerometer")
    dmg = signals.ingest_signal_csv(make_csv(tmp_path / "d.csv", 3.1, seed=1), "accelerometer")
    ind = signals.indicators(dmg, base, window_s=5.0)  # 0.2 Hz bin = 6.25 % of 3.2 Hz, above the 2 % first edge
    assert ind["frequency_shift_uncertainty_pct"]["z"] > 2.0
    f = signals.grade_signal(dmg, ind, rubric())
    assert f.unified.level == "U" and "FFT bin" in f.justification


def test_baseline_with_no_matching_channel_is_not_reported_as_no_baseline(tmp_path):
    base = signals.ingest_signal_csv(make_csv(tmp_path / "b.csv", 3.2, channels=("z",)), "accelerometer")
    rec = signals.ingest_signal_csv(make_csv(tmp_path / "x.csv", 3.2, channels=("x",)), "accelerometer", baseline_id=base.signal_id)
    f = signals.grade_signal(rec, signals.indicators(rec, base), rubric())
    assert f.unified.level == "U" and f.evidence.baseline_id == base.signal_id
    assert f.native_scale.criteria_matched == []  # the 'No stored baseline series' row is not quoted
    assert "No baseline series" not in f.justification and "baseline present" in f.justification and "channel missing" in f.justification
    # channel names match case-insensitively
    upper = signals.ingest_signal_csv(make_csv(tmp_path / "Z.csv", 3.2, channels=("Z",)), "accelerometer")
    assert signals.indicators(upper, base)["frequency_shift_pct"]["Z"] is not None


def test_blank_sample_is_interpolated_and_nan_channel_cannot_hide_a_shift(tmp_path):
    p = make_csv(tmp_path / "b.csv", 3.2)
    df = pd.read_csv(p)
    df.loc[100, "z"] = np.nan
    df.to_csv(p, index=False)
    base = signals.ingest_signal_csv(p, "accelerometer")
    rec = signals.ingest_signal_csv(make_csv(tmp_path / "d.csv", 2.9, seed=1), "accelerometer")
    ind = signals.indicators(rec, base)  # used to raise ValueError: cannot convert float NaN to integer
    assert ind["frequency_shift_pct"]["z"] is not None
    assert any("interpolated" in n for n in signals.indicators(base)["notes"])
    for vals in ({"x": float("nan"), "y": 12.5}, {"y": 12.5, "x": float("nan")}):
        assert signals._worst_channel(vals) == ("y", 12.5)


def test_time_column_gaps_and_milliseconds(tmp_path):
    t = np.arange(4000) / FS
    t = np.where(t >= 10.0, t + 5.0, t)  # 5 s telemetry gap
    pd.DataFrame({"time_s": t, "z": np.sin(2 * np.pi * 3.2 * t)}).to_csv(tmp_path / "gap.csv", index=False)
    rec = signals.ingest_signal_csv(tmp_path / "gap.csv", "accelerometer")
    assert rec.labels["time_irregular"] is True and rec.labels["time_gaps"] == 1
    assert abs(rec.duration_s - 25.0) < 0.01
    assert any("irregular" in n for n in signals.indicators(rec)["notes"])
    tm = np.arange(4000) / FS * 1000.0
    pd.DataFrame({"timestamp": tm, "z": np.sin(2 * np.pi * 3.2 * tm / 1000.0)}).to_csv(tmp_path / "ms.csv", index=False)
    with pytest.raises(ValueError, match="time-units"):
        signals.ingest_signal_csv(tmp_path / "ms.csv", "accelerometer")
    ms = signals.ingest_signal_csv(tmp_path / "ms.csv", "accelerometer", time_units="ms")
    assert abs(ms.sample_rate_hz - FS) < 1e-6
    assert abs(signals.indicators(ms)["dominant_frequency_hz"]["z"] - 3.2) < 0.05


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


def test_regraded_signal_replaces_its_findings_jsonl_row_so_resume_keeps_it(tmp_path):
    from cascade.pipeline import _read_jsonl, RunConfig, run_cascade

    out = tmp_path / "run"
    base = signals.ingest_signal_csv(make_csv(tmp_path / "b.csv", 3.2), "accelerometer")
    d = make_csv(tmp_path / "d.csv", 2.9, seed=1)
    rec_u = signals.ingest_signal_csv(d, "accelerometer")
    f_u = signals.grade_signal(rec_u, signals.indicators(rec_u), rubric())
    signals.write_signal_findings(out, [f_u])
    rec_b = signals.ingest_signal_csv(d, "accelerometer", baseline_id=base.signal_id)
    f_b = signals.grade_signal(rec_b, signals.indicators(rec_b, base), rubric())
    assert f_u.finding_id == f_b.finding_id and f_u.unified.level == "U" and f_b.unified.level == "S2"
    signals.write_signal_findings(out, [f_b])
    rows = _read_jsonl(out / "findings.jsonl")
    assert len(rows) == 1 and rows[0]["unified"]["level"] == "S2"
    run_cascade([], out, RunConfig())  # a resumed run rebuilds findings.json from findings.jsonl
    assert [x.unified.level for x in load_run(out)["findings"]] == ["S2"]


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
