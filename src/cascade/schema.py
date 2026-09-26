"""Pydantic models for the finding contract (docs/PRD.md appendix A) and pipeline records.

GraderOutput is exactly what a model must return; Finding adds evidence, review and
provenance that the pipeline fills in. Keep GraderOutput free of defaults so every
field is required in the JSON schema sent to the model (nullable where allowed).
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# Multi-sensor scope (docs/research/10_multisensor_scope.md section 0): the engine stays, inputs widen.
# underwater_structure = piers, piles, abutments seen by ROV camera or imaging sonar; interior_machinery = plant
# and rotating equipment (whiteboard 2026-09-25). Seismic readings are a modality on an existing asset class.
AssetClass = Literal["bridge_element", "steel_coating", "pv_module", "building_disaster", "underwater_structure", "interior_machinery"]
Modality = Literal["rgb", "thermal", "sonar", "seismic", "lidar"]
Sensor = Literal["accelerometer", "geophone", "strain", "tilt", "other"]
Level = Literal["S0", "S1", "S2", "S3", "S4", "U"]
Standard = Literal["NBI-0-9", "MBEI-CS", "ISO-4628-3", "IEC-62446-3-CoA", "FEMA-PDA", "CorrosionCS", "NBIS-UW", "SHM-Seismic", "ISO-20816-3"]
ActionCode = Literal["record", "monitor", "schedule", "prioritize", "escalate"]
Flag = Literal["fire_shock_pathway", "load_posting_review", "section_loss", "not_measurable"]
# how a crack dimension was obtained (R10 section 4.3 scale-source priority); None when no dimension was measured
MeasurementBasis = Literal["gsd_metadata", "scale_object", "manual_two_points", "model_estimate"]

LEVEL_ORDER = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4}


class GateOutput(BaseModel):
    """Stage A output. Two questions only."""

    usable: bool
    damage_present: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class NativeScale(BaseModel):
    standard: Standard
    value: str
    criteria_matched: List[str]


class Unified(BaseModel):
    level: Level
    uncertainty: str
    flags: List[Flag]


class GraderMeasurements(BaseModel):
    """The six measurement fields a model may return. The metrology fields below live only on the
    Finding-side Measurements, so the JSON schema sent to the grader never offers them."""

    area_cm2: Optional[float]
    crack_width_mm: Optional[float]
    delta_t_k: Optional[float]
    percent_area_rusted: Optional[float]
    section_loss_pct: Optional[float]
    confidence: float = Field(ge=0.0, le=1.0)


GRADER_MEASUREMENT_FIELDS = frozenset(GraderMeasurements.model_fields)


class Measurements(GraderMeasurements):
    # crack metrology (R10 section 4): filled by the scale-aware crack module, never guessed by the grader.
    # Not part of GraderOutput; Finding.from_grader resets them to None whatever the model sent.
    measurement_basis: Optional[MeasurementBasis] = None
    crack_length_mm: Optional[float] = None
    crack_width_uncertainty_mm: Optional[float] = None  # UI shows width +/- uncertainty, never a bare mm value


class Action(BaseModel):
    code: ActionCode
    sla_days: Optional[int]
    basis: str


class GraderOutput(BaseModel):
    """What the heavy grader returns for one crop. All fields required."""

    defect_type: str
    native_scale: NativeScale
    unified: Unified
    measurements: GraderMeasurements
    action: Action
    justification: str


class Evidence(BaseModel):
    image_ids: List[str] = []
    bbox: List[int] = []  # x0, y0, x1, y1 in original image pixels
    tile: Optional[str] = None
    gsd_mm_per_px: Optional[float] = None
    irradiance_wm2: Optional[float] = None
    # signal findings (cascade.signals): the graded series and the baseline it was compared with
    signal_id: Optional[str] = None
    baseline_id: Optional[str] = None


class Review(BaseModel):
    status: Literal["pending", "accepted", "overridden", "marked_u"] = "pending"
    reviewer: Optional[str] = None
    reviewed_at: Optional[str] = None
    prior_level: Optional[Level] = None


class Finding(BaseModel):
    finding_id: str
    asset_class: AssetClass
    defect_type: str
    native_scale: NativeScale
    unified: Unified
    measurements: Measurements
    action: Action
    justification: str
    evidence: Evidence
    review: Review = Review()
    modality: Modality = "rgb"  # older findings.json rows carry no modality and read back as rgb
    model: str = ""
    usd: float = 0.0
    seconds: float = 0.0
    queue_score: Optional[float] = None
    queue_rank: Optional[int] = None

    @classmethod
    def from_grader(
        cls,
        out: GraderOutput,
        *,
        finding_id: str,
        asset_class: AssetClass,
        evidence: Evidence,
        model: str,
        usd: float,
        seconds: float,
        modality: Modality = "rgb",
    ) -> "Finding":
        # only the six model-facing fields are copied: measurement_basis, crack_length_mm and
        # crack_width_uncertainty_mm come from the crack module, never from a model reply
        meas = Measurements(**out.measurements.model_dump(include=set(GRADER_MEASUREMENT_FIELDS)))
        return cls(
            finding_id=finding_id,
            asset_class=asset_class,
            defect_type=out.defect_type,
            native_scale=out.native_scale,
            unified=out.unified,
            measurements=meas,
            action=out.action,
            justification=out.justification,
            evidence=evidence,
            modality=modality,
            model=model,
            usd=usd,
            seconds=seconds,
        )


def unassessable_finding(
    *,
    finding_id: str,
    asset_class: AssetClass,
    standard: Standard,
    evidence: Evidence,
    reason: str,
    model: str = "",
) -> Finding:
    """A U finding for images that are unusable, refused, or otherwise not gradable."""

    return Finding(
        finding_id=finding_id,
        asset_class=asset_class,
        defect_type="not_assessable",
        native_scale=NativeScale(standard=standard, value="U", criteria_matched=[]),
        unified=Unified(level="U", uncertainty="+/-1", flags=["not_measurable"]),
        measurements=Measurements(
            area_cm2=None,
            crack_width_mm=None,
            delta_t_k=None,
            percent_area_rusted=None,
            section_loss_pct=None,
            confidence=0.0,
        ),
        action=Action(code="monitor", sla_days=None, basis=f"Not assessable: {reason}. Re-image or use other NDT."),
        justification=reason,
        evidence=evidence,
        model=model,
    )


class ImageRecord(BaseModel):
    """One row of a run or eval manifest."""

    image_id: str
    path: str
    sha256: str
    width: int
    height: int
    asset_class: AssetClass
    modality: Modality = "rgb"  # thermal heatmap, rendered sonar frame or lidar view are graded as images too
    gsd_mm_per_px: Optional[float] = None
    irradiance_wm2: Optional[float] = None
    captured_on: Optional[str] = None  # YYYY-MM-DD
    source_dataset: str = ""
    split: str = ""
    labels: dict = {}
    # optional customer context (FR-2): who owns the asset and which asset the image shows; null when unknown
    client_id: Optional[str] = None
    asset_id: Optional[str] = None
    # video provenance (FR-4): source file and timestamp of the extracted frame; null for still images
    source_video: Optional[str] = None
    frame_time_s: Optional[float] = None


class SignalRecord(BaseModel):
    """One ingested time series (accelerometer, geophone, strain or tilt CSV), the signal analogue of ImageRecord.

    `labels` carries what the CSV cannot declare: units, the detected time column, how the rate was found.
    `baseline_id` names the healthy-state record the indicators are compared with; None means "no baseline",
    which grades U, never S0 (rubrics/seismic_shm.json).
    """

    signal_id: str
    path: str
    sha256: str
    sensor: Sensor
    sample_rate_hz: float
    channels: List[str]
    n_samples: int
    duration_s: float
    captured_on: Optional[str] = None  # YYYY-MM-DD
    asset_id: Optional[str] = None
    client_id: Optional[str] = None
    baseline_id: Optional[str] = None
    asset_class: AssetClass = "bridge_element"
    labels: dict = {}


class GateRecord(BaseModel):
    image_id: str
    usable: bool
    damage_present: bool
    confidence: float
    reason: str
    routed: bool
    model: str
    seconds: float
    usd: float
