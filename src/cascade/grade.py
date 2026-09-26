"""Stage C: the heavy grader.

Builds a prompt from the asset class's rubric file (rows copied from the standard, see
docs/research/05_damage_grading_standards.md), optional exemplar crops, the target crop
and any measurement metadata, and asks for the GraderOutput contract. Cloud backend is
Claude through `client.messages.parse` (schema-enforced). Local backend is Ollama with the
same JSON schema. Refusals and parse failures become U findings, never S0.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests
from PIL import Image

from .costlog import CallLog, Timer
from .crop import fit_for_model, to_base64_jpeg, upscale_small
from .schema import AssetClass, Evidence, Finding, GraderOutput, Modality, Standard, unassessable_finding

RUBRIC_DIR = Path(__file__).parent / "rubrics"

RUBRIC_FOR_CLASS = {
    "steel_coating": "corrosion_cs.json",
    "bridge_element": "bridge_mbei.json",
    "pv_module": "pv_iec62446_3.json",
    "building_disaster": "disaster_fema.json",
    # multi-sensor scope (docs/research/10_multisensor_scope.md): underwater piers/piles from ROV or sonar frames,
    # interior machinery graded on ISO 20816-3 zones from RMS-velocity metadata (an image alone grades U)
    "underwater_structure": "underwater_nbis.json",
    "interior_machinery": "machinery_iso.json",
    # smart-building scope (building_spec section 1): graded images reuse this grader unchanged
    "facade_element": "facade_ll11.json",
    "interior_zone": "interior_water.json",
    "electrical_equipment": "electrical_thermal.json",
}
# seismic/vibration series are graded by cascade.signals, not by the image grader; kept here so the drift
# fingerprint and client reports can find the file by name
SEISMIC_RUBRIC_FILE = "seismic_shm.json"


def load_rubric(asset_class: AssetClass, rubric_file: Optional[str] = None) -> dict:
    name = rubric_file or RUBRIC_FOR_CLASS.get(asset_class)
    if name is None:
        raise KeyError(f"no rubric for asset class {asset_class!r}: add a rubrics/*.json file and map it in RUBRIC_FOR_CLASS")
    return json.loads((RUBRIC_DIR / name).read_text(encoding="utf-8"))


@dataclass
class Exemplar:
    image: Image.Image
    native_value: str
    note: str = ""
    image_id: str = ""  # so the pipeline never shows an image its own label


SYSTEM_TEMPLATE = """You are an inspection grading assistant. You grade ONE image crop of a {asset_label} against the rubric below and return the JSON contract exactly.

Rules:
- Emit the native scale first: set native_scale.standard to "{standard}" and native_scale.value to one of the allowed values. Put the verbatim rubric criterion text you matched into native_scale.criteria_matched (copy it, do not paraphrase).
- Then set unified.level using the mapping in the rubric. uncertainty is always "+/-1".
- If a criterion needs a measurement (width, area, temperature, percent area, section loss) that you cannot make from the image and the metadata provided, do NOT guess a value: set that measurement to null, add the flag "not_measurable", and grade only from criteria that need no measurement. If no criterion can be applied without a measurement, set unified.level to "U".
- Never describe damage that is not visible. If the crop shows no defect and the rubric's no-defect row needs no measurement, use that row and unified.level "S0". If the rubric has a "U" value and its criterion applies (a reading missing from the metadata, or a frame that cannot show the element surface), use "U" with the flag "not_measurable", never S0.
- action.code follows the rubric's action mapping; action.basis must name the rubric row or standard clause used.
- measurements.confidence is your confidence in the native grade, 0 to 1.
- justification: two or three sentences describing exactly what is visible and why it meets the criterion.

Rubric ({standard}):
{rubric_json}
"""


def build_system(asset_class: AssetClass, rubric: dict) -> str:
    return SYSTEM_TEMPLATE.format(
        asset_label=rubric.get("asset_label", asset_class),
        standard=rubric["standard"],
        rubric_json=json.dumps({k: v for k, v in rubric.items() if k not in ("asset_label",)}, indent=1),
    )


def build_user_content(img: Image.Image, metadata: dict, exemplars: Optional[List[Exemplar]] = None) -> list:
    content: list = []
    for i, ex in enumerate(exemplars or []):
        content.append({"type": "text", "text": f"Exemplar {i + 1}: native grade {ex.native_value}. {ex.note}".strip()})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": to_base64_jpeg(fit_for_model(upscale_small(ex.image)))}})
    content.append({"type": "text", "text": "Target image to grade:"})
    content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": to_base64_jpeg(fit_for_model(upscale_small(img)))}})
    meta_lines = [f"{k}: {v if v is not None else 'unknown'}" for k, v in metadata.items()]
    content.append({"type": "text", "text": "Metadata:\n" + ("\n".join(meta_lines) if meta_lines else "none") + "\nReturn the JSON contract."})
    return content


EFFORT = "high"  # pinned: the Opus 5 default; an id swap to Opus 5.5 (default medium) cannot silently change behaviour
GRADE_MAX_TOKENS = 4096  # hashed into the run fingerprint (drift.fingerprint), so a change is a config change
OLLAMA_OPTIONS = {"temperature": 0}  # likewise hashed; the local grader is deterministic by request


def response_meta(response) -> dict:
    """Served model id, request id and refusal category from a Claude response, with getattr defaults so a fake
    or older SDK object never raises. Logged per call so drift.check_models can pin the identity (M1)."""
    stop_details = getattr(response, "stop_details", None)
    return {
        "served_model": getattr(response, "model", None),
        "request_id": getattr(response, "_request_id", None),
        "stop_category": getattr(stop_details, "category", None) if getattr(response, "stop_reason", None) == "refusal" else None,
    }


def grade_claude(img: Image.Image, asset_class: AssetClass, rubric: dict, metadata: dict, exemplars=None, model: Optional[str] = None):
    """Returns (GraderOutput or None, usage dict, stop_reason, meta dict). No sampling parameters: Opus 5 and
    Sonnet 5 reject temperature/top_p/top_k; effort is pinned through output_config."""
    import anthropic

    model = model or os.getenv("GRADER_MODEL", "claude-opus-5")
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=model,
        max_tokens=GRADE_MAX_TOKENS,
        system=build_system(asset_class, rubric),
        messages=[{"role": "user", "content": build_user_content(img, metadata, exemplars)}],
        output_format=GraderOutput,
        output_config={"effort": EFFORT},
    )
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    meta = response_meta(response)
    if response.stop_reason == "refusal":
        return None, usage, "refusal", meta
    return response.parsed_output, usage, response.stop_reason, meta


def grade_ollama(img: Image.Image, asset_class: AssetClass, rubric: dict, metadata: dict, exemplars=None, model: Optional[str] = None, url: Optional[str] = None, timeout: int = 600):
    model = model or os.getenv("GRADER_LOCAL_MODEL", "qwen3-vl:8b-instruct")
    url = (url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
    content = build_user_content(img, metadata, exemplars)
    text = "\n".join(c["text"] for c in content if c["type"] == "text")
    images = [c["source"]["data"] for c in content if c["type"] == "image"]
    payload = {
        "model": model,
        "stream": False,
        "format": GraderOutput.model_json_schema(),
        "options": dict(OLLAMA_OPTIONS),
        "messages": [
            {"role": "system", "content": build_system(asset_class, rubric)},
            {"role": "user", "content": text, "images": images},
        ],
    }
    r = requests.post(f"{url}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    usage = {"input_tokens": int(data.get("prompt_eval_count", 0)), "output_tokens": int(data.get("eval_count", 0))}
    meta = {"served_model": data.get("model"), "request_id": None, "stop_category": None}
    try:
        return GraderOutput.model_validate_json(data["message"]["content"]), usage, "end_turn", meta
    except Exception as e:  # invalid JSON from a local model becomes U, not a crash
        return None, usage, f"parse_error: {e}", meta


MACHINE_SUPPORT_TYPES = ("rigid", "flexible")


def _finite_number(v) -> bool:
    try:
        return v is not None and not isinstance(v, bool) and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def machinery_reading_missing(metadata: dict) -> Optional[str]:
    """Why an interior_machinery image cannot be zoned (machinery_iso.json U row), or None when the metadata
    holds a numeric rms_velocity_mm_s, a machine class (machine_group 1/2 or numeric rated_kw) and a support type."""
    missing = []
    if not _finite_number(metadata.get("rms_velocity_mm_s")):
        missing.append("rms_velocity_mm_s")
    group = metadata.get("machine_group")
    if not (str(group) in ("1", "2") or _finite_number(metadata.get("rated_kw"))):
        missing.append("machine_group or rated_kw")
    if str(metadata.get("support_type", "")).strip().lower() not in MACHINE_SUPPORT_TYPES:
        missing.append("support_type (rigid or flexible)")
    return ", ".join(missing) if missing else None


def grade_image(
    img: Image.Image,
    *,
    finding_id: str,
    image_id: str,
    asset_class: AssetClass,
    backend: str = "claude",
    rubric: Optional[dict] = None,
    metadata: Optional[dict] = None,
    exemplars: Optional[List[Exemplar]] = None,
    evidence: Optional[Evidence] = None,
    log: Optional[CallLog] = None,
    modality: Optional[Modality] = None,
) -> Finding:
    """Grade one crop. Two deterministic U guards enforce 'U is never S0' whatever a model says:
    interior_machinery without an RMS-velocity reading, machine class and support type is U without a model
    call (machinery_iso.json U row), and an S0 on a sonar frame (underwater_nbis.json U row) becomes U."""
    rubric = rubric or load_rubric(asset_class)
    metadata = metadata or {}
    evidence = evidence or Evidence(image_ids=[image_id])
    standard: Standard = rubric["standard"]
    modality = modality or metadata.get("modality") or "rgb"
    if asset_class == "interior_machinery":
        missing = machinery_reading_missing(metadata)
        if missing is not None:
            f = unassessable_finding(finding_id=finding_id, asset_class=asset_class, standard=standard, evidence=evidence, reason=f"no vibration reading in metadata (missing {missing}); a photograph alone cannot place an ISO 20816-3 zone", model="guard:machinery_no_reading")
            f.modality = modality
            return f
    with Timer() as t:
        if backend == "claude":
            model = os.getenv("GRADER_MODEL", "claude-opus-5")
            out, usage, stop, meta = grade_claude(img, asset_class, rubric, metadata, exemplars)
        elif backend == "local":
            model = os.getenv("GRADER_LOCAL_MODEL", "qwen3-vl:8b-instruct")
            out, usage, stop, meta = grade_ollama(img, asset_class, rubric, metadata, exemplars)
        else:
            raise ValueError(f"unknown grader backend {backend}")
    usd = 0.0
    if log is not None:
        row = log.record(stage="grade", model=model, image_id=image_id, input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"], seconds=t.seconds, note=str(stop), **meta)
        usd = row["usd"]
    if out is None:
        f = unassessable_finding(finding_id=finding_id, asset_class=asset_class, standard=standard, evidence=evidence, reason=f"grader returned no contract ({stop})", model=model)
        f.modality = modality
        return f
    if modality == "sonar" and out.unified.level == "S0":
        # imaging sonar cannot resolve the crack classes or show a cleaned surface (underwater_nbis.json U row)
        f = unassessable_finding(finding_id=finding_id, asset_class=asset_class, standard=standard, evidence=evidence, reason=f"sonar frame graded S0 ('{out.native_scale.value}'): imaging sonar cannot show the element surface at crack resolution, so no-defect is not assessable", model=model)
        f.modality, f.usd, f.seconds = modality, usd, t.seconds
        return f
    return Finding.from_grader(out, finding_id=finding_id, asset_class=asset_class, evidence=evidence, model=model, usd=usd, seconds=t.seconds, modality=modality)
