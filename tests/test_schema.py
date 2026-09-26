import json

from cascade.schema import Evidence, Finding, GateOutput, GraderOutput, unassessable_finding


def test_grader_output_schema_has_all_fields_required():
    schema = GraderOutput.model_json_schema()
    assert set(schema["required"]) == {"defect_type", "native_scale", "unified", "measurements", "action", "justification"}
    meas = schema["$defs"]["GraderMeasurements"]
    assert set(meas["required"]) == {"area_cm2", "crack_width_mm", "delta_t_k", "percent_area_rusted", "section_loss_pct", "confidence"}
    # the metrology fields are filled by the crack module only, so the model is never offered them
    assert set(meas["properties"]) == set(meas["required"])
    assert "Measurements" not in schema["$defs"]


def test_from_grader_drops_metrology_fields_a_model_might_send():
    raw = {
        "defect_type": "crack",
        "native_scale": {"standard": "MBEI-CS", "value": "CS2", "criteria_matched": ["row"]},
        "unified": {"level": "S1", "uncertainty": "+/-1", "flags": []},
        "measurements": {"area_cm2": None, "crack_width_mm": 0.4, "delta_t_k": None, "percent_area_rusted": None, "section_loss_pct": None, "confidence": 0.7,
                         "measurement_basis": "scale_object", "crack_length_mm": 120.0, "crack_width_uncertainty_mm": 0.1},
        "action": {"code": "monitor", "sla_days": None, "basis": "row"},
        "justification": "x",
    }
    out = GraderOutput.model_validate(raw)
    f = Finding.from_grader(out, finding_id="i/full", asset_class="bridge_element", evidence=Evidence(image_ids=["i"]), model="t", usd=0.0, seconds=0.0)
    assert f.measurements.crack_width_mm == 0.4
    assert f.measurements.measurement_basis is None and f.measurements.crack_length_mm is None and f.measurements.crack_width_uncertainty_mm is None
    assert f.modality == "rgb"


def test_round_trip_contract():
    raw = {
        "defect_type": "corrosion",
        "native_scale": {"standard": "CorrosionCS", "value": "Poor", "criteria_matched": ["Rust with pitting"]},
        "unified": {"level": "S3", "uncertainty": "+/-1", "flags": ["section_loss"]},
        "measurements": {"area_cm2": None, "crack_width_mm": None, "delta_t_k": None, "percent_area_rusted": 35.0, "section_loss_pct": None, "confidence": 0.8},
        "action": {"code": "prioritize", "sla_days": 90, "basis": "CorrosionCS row Poor"},
        "justification": "Laminar rust across the flange.",
    }
    out = GraderOutput.model_validate_json(json.dumps(raw))
    f = Finding.from_grader(out, finding_id="img_0/full", asset_class="steel_coating", evidence=Evidence(image_ids=["img_0"]), model="test", usd=0.01, seconds=1.2)
    assert f.unified.level == "S3"
    assert f.review.status == "pending"
    assert Finding.model_validate_json(f.model_dump_json()).finding_id == "img_0/full"


def test_unassessable_is_u_not_s0():
    f = unassessable_finding(finding_id="x", asset_class="pv_module", standard="IEC-62446-3-CoA", evidence=Evidence(image_ids=["x"]), reason="blurred")
    assert f.unified.level == "U"
    assert "not_measurable" in f.unified.flags


def test_multisensor_defaults_keep_old_rows_valid():
    from typing import get_args

    from cascade.schema import AssetClass, ImageRecord, Measurements, Modality, SignalRecord, Standard

    assert {"underwater_structure", "interior_machinery"} <= set(get_args(AssetClass))
    assert set(get_args(Modality)) == {"rgb", "thermal", "sonar", "seismic", "lidar"}
    assert {"NBIS-UW", "SHM-Seismic", "ISO-20816-3"} <= set(get_args(Standard))
    rec = ImageRecord(image_id="i", path="i.jpg", sha256="0" * 64, width=1, height=1, asset_class="underwater_structure")
    assert rec.modality == "rgb"
    m = Measurements(area_cm2=None, crack_width_mm=None, delta_t_k=None, percent_area_rusted=None, section_loss_pct=None, confidence=0.5)
    assert m.measurement_basis is None and m.crack_length_mm is None and m.crack_width_uncertainty_mm is None
    m2 = Measurements.model_validate({**m.model_dump(), "measurement_basis": "scale_object", "crack_width_mm": 0.35, "crack_width_uncertainty_mm": 0.2, "crack_length_mm": 120.0})
    assert m2.measurement_basis == "scale_object" and m2.crack_length_mm == 120.0
    # an old finding row (no modality, no signal evidence) still loads and reads as rgb
    f = unassessable_finding(finding_id="x", asset_class="pv_module", standard="IEC-62446-3-CoA", evidence=Evidence(image_ids=["x"]), reason="blurred")
    old = json.loads(f.model_dump_json())
    old.pop("modality")
    old["evidence"].pop("signal_id")
    old["evidence"].pop("baseline_id")
    f2 = Finding.model_validate(old)
    assert f2.modality == "rgb" and f2.evidence.signal_id is None and f2.evidence.baseline_id is None
    # SignalRecord round trip with defaults
    s = SignalRecord(signal_id="s", path="s.csv", sha256="0" * 64, sensor="accelerometer", sample_rate_hz=200.0, channels=["z"], n_samples=10, duration_s=0.05)
    s2 = SignalRecord.model_validate_json(s.model_dump_json())
    assert s2.baseline_id is None and s2.asset_class == "bridge_element" and s2.labels == {} and s2.captured_on is None


def test_gate_output_bounds():
    GateOutput(usable=True, damage_present=False, confidence=0.5, reason="ok")
    try:
        GateOutput(usable=True, damage_present=False, confidence=1.5, reason="bad")
        assert False, "confidence above 1 must fail"
    except Exception:
        pass
