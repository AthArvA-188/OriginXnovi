"""Threshold registry and rubric-row helpers for the building layer (building_spec section 3).

Every numeric threshold used in cascade.building reads from THRESHOLDS by key. Each entry carries a public URL or
the tag [team-proposed, validate] or [Assumption]. URLs are from the research of 2026-09-25.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence

from ..grade import RUBRIC_FOR_CLASS
from ..schema import (Action, AssetClass, Evidence, Finding, Measurements, Modality, NativeScale, Unified,
                      unassessable_finding)

TagKind = Literal["PUBLIC", "PUBLIC, secondary", "team-proposed, validate", "Assumption"]
MODEL_TAG = "deterministic:cascade.building"

EPA_HOME = "https://www.epa.gov/mold/brief-guide-mold-moisture-and-your-home"
EPA_CH4 = "https://www.epa.gov/mold/mold-remediation-schools-and-commercial-buildings-guide-chapter-4"
FISP_RULE = "https://www.nyc.gov/assets/buildings/rules/1_RCNY_103-04.pdf"


@dataclass(frozen=True)
class Threshold:
    key: str
    value: Any
    unit: str
    meaning: str
    tag: TagKind
    url: Optional[str] = None
    note: str = ""

    def label(self) -> str:
        """Short printable form: "60 %RH (EPA_RH_MAX, https://...)" or "5.0 mm [team-proposed, validate]"."""
        v = self.value
        if isinstance(v, (tuple, list)):
            vs = "-".join(str(x) for x in v)
        elif isinstance(v, dict):
            vs = ", ".join(f"{k} {x}" for k, x in v.items())
        else:
            vs = str(v)
        head = f"{vs} {self.unit}".strip()
        if self.tag == "PUBLIC" and self.url:
            return f"{head} ({self.key}, {self.url})"
        tail = f" [{self.tag}]"
        return f"{head}{tail}" + (f" ({self.url})" if self.url else "")


def _th(key: str, value: Any, unit: str, meaning: str, tag: TagKind, url: Optional[str] = None, note: str = "") -> Threshold:
    return Threshold(key=key, value=value, unit=unit, meaning=meaning, tag=tag, url=url, note=note)


_ROWS = [
    _th("EPA_RH_MAX", 60.0, "%RH", "indoor RH should stay below this; ideally 30-50 %", "PUBLIC", EPA_HOME),
    _th("EPA_DRY_WINDOW_H", (24.0, 48.0), "h", "wet materials dried within 24-48 h usually do not grow mold", "PUBLIC", EPA_HOME),
    _th("EPA_TABLE1", "ceiling tile: discard; wallboard: dry in place only if no swelling and seams intact", "",
        "EPA Table 1 water-damage responses", "PUBLIC", EPA_CH4),
    _th("ASHRAE160_RH_30D", 80.0, "%RH", "30-day running mean RH with surface temperature 5-40 degC (original ASHRAE 160 criterion)",
        "PUBLIC, secondary", "https://doi.org/10.1520/stp159920160106", "later replaced by a mold index"),
    _th("ASHRAE160_T_RANGE_C", (5.0, 40.0), "degC", "temperature range in which the ASHRAE 160 RH criterion applies",
        "PUBLIC, secondary", "https://doi.org/10.1520/stp159920160106"),
    _th("ASHRAE160_WINDOW_D", 30, "days", "running-mean window of the original ASHRAE 160 RH criterion", "PUBLIC, secondary",
        "https://doi.org/10.1520/stp159920160106"),
    _th("RH_TREND_DAYS", 14, "days", "window for the daily-mean RH trend used in risk and escalation clocks", "team-proposed, validate"),
    _th("WOOD_MC_MAX", 20.0, "%MC", "wood moisture content above which decay risk starts (decay above about 30 %)", "PUBLIC",
        "https://www.fpl.fs.usda.gov/documnts/fplgtr/fplgtr282/chapter_14_fpl_gtr282.pdf"),
    _th("RH_SUSTAINED_H", 72.0, "h", "duration that makes RH above EPA_RH_MAX 'sustained'", "team-proposed, validate"),
    _th("STORM_MIN_MM", 5.0, "mm", "minimum rain total for a storm event", "team-proposed, validate"),
    _th("STORM_GAP_H", 6.0, "h", "dry gap that splits two storm events", "team-proposed, validate"),
    _th("RESPONSE_WINDOW_H", 48.0, "h", "window after storm end in which an indoor response is counted", "team-proposed, validate",
        note="research found no published lag window"),
    _th("RESPONSE_PRE_H", 24.0, "h", "pre-storm baseline window", "team-proposed, validate"),
    _th("RESPONSE_MIN_R", 0.6, "", "minimum Pearson r for a significant rain response", "team-proposed, validate"),
    _th("RESPONSE_MIN_EVENTS", 3, "events", "minimum storm responses for a significant fit", "team-proposed, validate"),
    _th("RESPONSE_MIN_RISE", 8.0, "%RH", "RH rise that counts as a storm response event", "team-proposed, validate"),
    _th("WDR_PROXY", "rain_mm * wind_speed_ms * max(0, cos(wind_from - facade_azimuth))", "mm*m/s",
        "wind-driven rain proxy per facade orientation", "team-proposed, validate", "https://www.iso.org/standard/41323.html",
        "method reference ISO 15927-3; its coefficients were not read this session"),
    _th("EDGE_WEIGHT", {"drains_to": 0.9, "above": 0.6, "adjacent": 0.3, "feeds": 1.0}, "",
        "default water or power transfer weight per edge kind", "team-proposed, validate"),
    _th("REACH_MIN", 0.05, "", "minimum path score for a node to count as reached", "team-proposed, validate"),
    _th("SOURCE_SHARE_MIN", 0.5, "", "a shared water source joins two observations only when its path score is at least this "
        "share of each node's best source score (one source must be a strong explanation for both)", "team-proposed, validate"),
    _th("RISK_WEIGHTS", {"exposure": 0.15, "upstream_defects": 0.25, "rain_response": 0.30, "trend": 0.15, "open_tickets": 0.15}, "",
        "additive zone water-risk weights", "team-proposed, validate"),
    _th("FORECAST_Z", 1.96, "", "forecast band = +/- z * residual sd", "team-proposed, validate"),
    _th("DT_BANDS_K", (1.0, 4.0, 15.0), "K", "delta-T band edges, element minus similar element", "team-proposed, validate",
        "https://www.netaworld.org/standards/ansi-neta-mts", "validate against the ANSI/NETA MTS-2023 table, NOT read this session"),
    _th("IR_MIN_LOAD_PCT", 40.0, "%", "minimum load at scan time for a valid IR reading", "team-proposed, validate"),
    _th("LOAD_CONT_PCT", 80.0, "%", "continuous load limit as percent of breaker rating", "Assumption",
        note="NEC 210.20(A) 125 % continuous-load sizing; NEC text not read this session"),
    _th("CONTINUOUS_H", 3.0, "h", "duration of a continuous load", "Assumption", note="NEC Article 100 'continuous load'; not read this session"),
    _th("LOAD_RECENT_D", 7, "days", "an overload is 'ongoing' when a CONTINUOUS_H-hour mean at or above LOAD_CONT_PCT occurred "
        "within this many days before the last reading (office loads cycle daily and weekly)", "team-proposed, validate"),
    _th("IR_INTERVAL_DAYS", 365, "days", "maximum interval between infrared scans", "PUBLIC, secondary",
        "https://www.fluke.com/en-us/learn/blog/thermal-imaging/nfpa-70b-how-thermal-imaging-supports-electrical-equipment-maintenance",
        "vendor says NFPA 70B 2023 requires annual IR; primary not read"),
    _th("ANOMALY_BASELINE_DAYS", 28, "days", "baseline length for hour-of-week current profiles", "team-proposed, validate"),
    _th("ANOMALY_MAD_K", 5.0, "MAD", "anomaly threshold in median absolute deviations", "team-proposed, validate"),
    _th("FISP_CYCLE_YEARS", 5, "years", "FISP inspection cycle", "PUBLIC", FISP_RULE),
    _th("FISP_UNSAFE_FIX_DAYS", 90, "days", "time to fix a QEWI-confirmed Unsafe condition, counted from the filed report",
        "team-proposed, validate", FISP_RULE,
        "no section of 1 RCNY 103-04 was cited for 90 days this session; the facade_ll11.json Unsafe criterion quotes "
        "'repair within one (1) year of completion of critical examinations'. The clock never starts from a model grade"),
    _th("FISP_MIN_PHOTO_PX", (800, 600), "px", "minimum photo size for FISP reports", "PUBLIC", FISP_RULE),
    _th("PROBLEM_WINDOW_H", 72.0, "h", "observations within this gap chain into one problem", "team-proposed, validate"),
    _th("ROOT_WEIGHTS", {"coverage": 0.35, "path_strength": 0.25, "signature": 0.25, "own_evidence": 0.15, "own_zone_quiet": -0.20}, "",
        "additive root-cause ranking weights", "team-proposed, validate"),
    _th("URGENCY", {"numerator_h": 72.0, "floor_h": 6.0, "min": 1.0, "max": 3.0}, "",
        "urgency = 1 + 72 / max(hours, 6), clipped to [1, 3]", "team-proposed, validate"),
    _th("WATER_NEAR_PANEL_HOPS", 2, "hops", "water path this close to electrical equipment sets fire_shock_pathway", "team-proposed, validate"),
]

THRESHOLDS: Dict[str, Threshold] = {r.key: r for r in _ROWS}


def t(key: str) -> Any:
    """Value of a registered threshold. KeyError on an unknown key, so no bare literal slips in."""
    return THRESHOLDS[key].value


def find_band_row(rubric: dict, family: str, value: float, **match: Any) -> Optional[dict]:
    """First row of `family` whose [min, max) band holds `value` (None bound = open). Extra keyword arguments must
    equal the row's fields (e.g. group=2). None when no row matches or the value is NaN."""
    if value is None or value != value:  # NaN
        return None
    for row in rubric.get("rows", []):
        if row.get("family") != family:
            continue
        if any(row.get(k) != v for k, v in match.items()):
            continue
        lo, hi = row.get("min"), row.get("max")
        if lo is not None and value < lo:
            continue
        if hi is not None and value >= hi:
            continue
        return row
    return None


def _rubric_file(rubric: dict, asset_class: AssetClass) -> str:
    return rubric.get("_file") or RUBRIC_FOR_CLASS.get(asset_class) or str(rubric.get("standard", "rubric"))


def _empty_measurements(confidence: float) -> Measurements:
    return Measurements(area_cm2=None, crack_width_mm=None, delta_t_k=None, percent_area_rusted=None, section_loss_pct=None, confidence=confidence)


def finding_from_row(row: dict, rubric: dict, *, finding_id: str, asset_class: AssetClass, modality: Modality,
                     evidence: Evidence, justification: str, measurements: Optional[dict] = None,
                     flags: Sequence[str] = (), confidence: float = 1.0) -> Finding:
    """A deterministic Finding from one rubric row. The row criterion is quoted verbatim in criteria_matched."""
    meas = _empty_measurements(confidence)
    if measurements:
        known = {k: v for k, v in measurements.items() if k in Measurements.model_fields}
        meas = meas.model_copy(update=known)
    all_flags = list(dict.fromkeys(list(flags) + ([row["flag"]] if row.get("flag") else [])))
    return Finding(
        finding_id=finding_id,
        asset_class=asset_class,
        defect_type=str(row.get("family", "reading")),
        native_scale=NativeScale(standard=rubric["standard"], value=row["value"], criteria_matched=[row["criterion"]]),
        unified=Unified(level=row["unified"], uncertainty="+/-1", flags=all_flags),
        measurements=meas,
        action=Action(code=row["action"], sla_days=None, basis=f"{_rubric_file(rubric, asset_class)} row '{row['value']}': {row.get('source', '')}"),
        justification=justification,
        evidence=evidence,
        modality=modality,
        model=MODEL_TAG,
    )


def u_finding(rubric: dict, *, finding_id: str, asset_class: AssetClass, modality: Modality,
              evidence: Evidence, reason: str) -> Finding:
    """A U finding (never S0) for a reading that cannot be graded. Wraps schema.unassessable_finding."""
    f = unassessable_finding(finding_id=finding_id, asset_class=asset_class, standard=rubric["standard"], evidence=evidence,
                             reason=reason, model=MODEL_TAG)
    return f.model_copy(update={"modality": modality})
