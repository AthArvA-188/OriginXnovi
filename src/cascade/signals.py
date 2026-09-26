"""Seismic and vibration time series: ingest, indicators, rubric grading, findings export.

The sensor kit on the whiteboard (docs/research/10_multisensor_scope.md section 2.3: Raspberry Shake RS4D
geophone plus MEMS accelerometer, or a bare ADXL355 / SM-24) produces CSV time series. This module turns one
series into a `SignalRecord`, computes the indicators that `rubrics/seismic_shm.json` grades on, and writes
`Finding` rows in the same findings.json format as the image cascade so the app, the queue and client reports
read them unchanged. No model call: every number is computed from the samples with numpy and every threshold
lives in the rubric JSON with its source tag.

Indicator definitions (stated so the numbers are reproducible):
- pga: max |x| per channel after mean removal, in the record's units (labels["units"]). pga_pct_g is filled
  only for an accelerometer whose units are known (m/s2, g, mg, gal, cm/s2); pgv_cm_s only for a geophone
  with known velocity units (m/s, cm/s, mm/s). Unknown units leave both None: no unit is ever assumed.
- rms: sqrt(mean(x^2)) after mean removal.
- dominant_frequency_hz: peak of |rfft(hann * x)| above DC, refined by parabolic interpolation over the three
  bins around the peak. Resolution = sample_rate / n_samples_used, reported as fft_resolution_hz.
- damping_ratio: logarithmic decrement on a free-decay window after the global peak: the envelope is the max
  of |x| in consecutive one-period bins, ln(envelope) is fitted against cycle index, and zeta =
  delta / sqrt(4 pi^2 + delta^2). None unless at least DECAY_MIN_PEAKS decaying peaks fit with R^2 >= DECAY_MIN_R2.
- frequency_shift_pct: (f - f_baseline) / f_baseline * 100 per channel; rms_change_pct likewise. None without a
  baseline or when the channel is missing from the baseline.

Usage: python -m cascade.signals --csv x.csv --sensor accelerometer --rate 200 [--baseline y.csv] --out runs/<run>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, get_args

import numpy as np
import pandas as pd

from .grade import RUBRIC_DIR, SEISMIC_RUBRIC_FILE
from .ingest import sha256_of
from .pipeline import LEVELS, load_run, save_findings
from .schema import LEVEL_ORDER, Action, AssetClass, Evidence, Finding, Measurements, NativeScale, Sensor, SignalRecord, Unified

TIME_COLUMNS = ("time", "t", "time_s", "timestamp", "seconds", "sec", "s")
G_MS2 = 9.80665  # standard gravity, m/s2
# unit -> factor to m/s2 (accelerations) or to cm/s (velocities); anything else is "unknown" and grades no PGA/PGV row
ACCEL_TO_MS2: Dict[str, float] = {"m/s2": 1.0, "m/s^2": 1.0, "g": G_MS2, "mg": G_MS2 / 1000.0, "gal": 0.01, "cm/s2": 0.01, "cm/s^2": 0.01}
VEL_TO_CM_S: Dict[str, float] = {"m/s": 100.0, "cm/s": 1.0, "mm/s": 0.1}
DECAY_MIN_PEAKS = 4
DECAY_MIN_R2 = 0.9
DECAY_FLOOR = 0.05  # a bin whose peak is below 5 % of the global peak ends the decay window (team assumption)
DECAY_MAX_CYCLES = 40
MODEL_TAG = "deterministic:cascade.signals"


# ------------------------------------------------------------------------------------------------ ingest


def _numeric_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        raise ValueError(f"{path}: no numeric columns")
    return num


def _time_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        name = str(c).strip().lower()
        if name in TIME_COLUMNS or name.startswith("time"):
            return str(c)
    return None


def ingest_signal_csv(
    path: Path,
    sensor: Sensor,
    sample_rate_hz: Optional[float] = None,
    asset_id: Optional[str] = None,
    client_id: Optional[str] = None,
    *,
    units: Optional[str] = None,
    asset_class: AssetClass = "bridge_element",
    captured_on: Optional[str] = None,
    baseline_id: Optional[str] = None,
    signal_id: Optional[str] = None,
) -> SignalRecord:
    """One CSV in, one SignalRecord out. A time column (time, t, timestamp, seconds, ...) is detected by name
    and used for the sample rate when `sample_rate_hz` is not given; every other numeric column is a channel.
    `units` is stored, never inferred: without it PGA/PGV rows of the rubric cannot be applied."""
    path = Path(path)
    df = _numeric_frame(path)
    tcol = _time_column(df)
    channels = [str(c) for c in df.columns if str(c) != tcol]
    if not channels:
        raise ValueError(f"{path}: no signal channels besides the time column")
    n = int(len(df))
    if n < 2:
        raise ValueError(f"{path}: fewer than two samples")
    rate_source = "given"
    if sample_rate_hz is None:
        if tcol is None:
            raise ValueError(f"{path}: no time column found and no sample_rate_hz given")
        t = df[tcol].to_numpy(dtype=float)
        dt = float(np.median(np.diff(t)))
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError(f"{path}: time column '{tcol}' is not increasing")
        sample_rate_hz = 1.0 / dt
        rate_source = f"time column '{tcol}'"
    sha = sha256_of(path)
    return SignalRecord(
        signal_id=signal_id or f"{path.stem}_{sha[:8]}",
        path=str(path),
        sha256=sha,
        sensor=sensor,
        sample_rate_hz=float(sample_rate_hz),
        channels=channels,
        n_samples=n,
        duration_s=n / float(sample_rate_hz),
        captured_on=captured_on,
        asset_id=asset_id,
        client_id=client_id,
        baseline_id=baseline_id,
        asset_class=asset_class,
        labels={"units": units, "time_column": tcol, "rate_source": rate_source},
    )


def load_samples(record: SignalRecord) -> np.ndarray:
    """Samples as float array of shape (n_samples, n_channels), channel order as in the record."""
    df = _numeric_frame(Path(record.path))
    return df[record.channels].to_numpy(dtype=float)


# ---------------------------------------------------------------------------------------------- indicators


def dominant_frequency(x: np.ndarray, fs: float) -> Tuple[Optional[float], float]:
    """(peak frequency in Hz or None, bin resolution in Hz). Hann window, rfft, DC bin excluded, parabolic
    interpolation of the peak over its neighbours (sub-bin estimate; exact when the tone sits on a bin)."""
    n = int(x.size)
    res = fs / n if n else float("nan")
    if n < 8:
        return None, res
    spec = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(n)))
    if spec.size < 3:
        return None, res
    k = int(np.argmax(spec[1:])) + 1
    if spec[k] <= 0:
        return None, res
    delta = 0.0
    if 1 <= k < spec.size - 1:
        a, b, c = float(spec[k - 1]), float(spec[k]), float(spec[k + 1])
        denom = a - 2 * b + c
        if denom != 0:
            delta = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
    return float((k + delta) * res), res


def log_decrement_damping(x: np.ndarray, fs: float, f_dom: Optional[float]) -> Optional[dict]:
    """Damping ratio from the free decay after the global peak, or None when no clean decay exists.

    Envelope = max |x| in consecutive bins one period (fs / f_dom) long starting at the peak; bins stop at
    DECAY_MAX_CYCLES or when a bin drops below DECAY_FLOOR of the peak. ln(envelope) vs cycle index must fit
    a decreasing line with R^2 >= DECAY_MIN_R2 over at least DECAY_MIN_PEAKS bins. delta = -slope;
    zeta = delta / sqrt(4 pi^2 + delta^2) (log-decrement relation for viscous damping)."""
    if f_dom is None or f_dom <= 0:
        return None
    x = x - x.mean()
    i0 = int(np.argmax(np.abs(x)))
    peak = float(abs(x[i0]))
    if peak <= 0:
        return None
    period = int(round(fs / f_dom))
    if period < 2:
        return None
    seg = np.abs(x[i0:])
    amps: List[float] = []
    for j in range(DECAY_MAX_CYCLES):
        s = seg[j * period : (j + 1) * period]
        if s.size < period:
            break
        a = float(s.max())
        if a < DECAY_FLOOR * peak:
            break
        amps.append(a)
    if len(amps) < DECAY_MIN_PEAKS:
        return None
    y = np.log(np.asarray(amps))
    j = np.arange(len(amps), dtype=float)
    slope, intercept = np.polyfit(j, y, 1)
    fit = slope * j + intercept
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if slope >= 0 or r2 < DECAY_MIN_R2:
        return None
    delta = float(-slope)
    zeta = delta / math.sqrt(4 * math.pi**2 + delta**2)
    return {"damping_ratio": zeta, "log_decrement": delta, "n_peaks": len(amps), "r2": r2}


def _window(x: np.ndarray, fs: float, window_s: Optional[float]) -> np.ndarray:
    """First `window_s` seconds of the series (None = whole series)."""
    if window_s is None:
        return x
    n = max(2, int(round(window_s * fs)))
    return x[:n]


def _pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return (a - b) / b * 100.0


def indicators(record: SignalRecord, baseline: Optional[SignalRecord] = None, window_s: Optional[float] = None) -> dict:
    """Per-channel PGA, RMS, dominant frequency, damping ratio, and shifts against `baseline` (same
    indicators on the baseline record, matched by channel name). `notes` lists what could not be computed."""
    fs = record.sample_rate_hz
    data = _window(load_samples(record), fs, window_s)
    units = (record.labels or {}).get("units")
    notes: List[str] = []
    pga: Dict[str, float] = {}
    rms: Dict[str, float] = {}
    fdom: Dict[str, Optional[float]] = {}
    damping: Dict[str, Optional[float]] = {}
    damping_detail: Dict[str, Optional[dict]] = {}
    res = fs / data.shape[0]
    for i, ch in enumerate(record.channels):
        x = data[:, i] - data[:, i].mean()
        pga[ch] = float(np.max(np.abs(x)))
        rms[ch] = float(np.sqrt(np.mean(x**2)))
        fdom[ch], res = dominant_frequency(x, fs)
        if fdom[ch] is None:
            notes.append(f"{ch}: dominant frequency not computed (too few samples or flat signal)")
        d = log_decrement_damping(x, fs, fdom[ch])
        damping[ch] = d["damping_ratio"] if d else None
        damping_detail[ch] = d
    if all(v is None for v in damping.values()):
        notes.append("damping_ratio: no free-decay window found (no monotone decay after the peak); None")

    pga_pct_g: Optional[Dict[str, float]] = None
    pgv_cm_s: Optional[Dict[str, float]] = None
    if record.sensor == "accelerometer" and units in ACCEL_TO_MS2:
        pga_pct_g = {ch: v * ACCEL_TO_MS2[units] / G_MS2 * 100.0 for ch, v in pga.items()}
    elif record.sensor == "geophone" and units in VEL_TO_CM_S:
        pgv_cm_s = {ch: v * VEL_TO_CM_S[units] for ch, v in pga.items()}
    elif record.sensor in ("accelerometer", "geophone"):
        notes.append(f"PGA/PGV band not applied: units {units!r} unknown (pass units=m/s2, g, mg, gal, cm/s2 or m/s, cm/s, mm/s)")
    else:
        notes.append(f"PGA/PGV band not applied: sensor {record.sensor!r} does not measure acceleration or velocity")

    freq_shift: Optional[Dict[str, Optional[float]]] = None
    rms_change: Optional[Dict[str, Optional[float]]] = None
    base_fdom: Optional[Dict[str, Optional[float]]] = None
    baseline_id = baseline.signal_id if baseline is not None else record.baseline_id
    if baseline is None:
        notes.append("no baseline: frequency_shift_pct and rms_change_pct not computed (rubric row 'U')")
    else:
        base = indicators(baseline, None, window_s)
        base_fdom = base["dominant_frequency_hz"]
        freq_shift, rms_change = {}, {}
        for ch in record.channels:
            freq_shift[ch] = _pct(fdom.get(ch), base_fdom.get(ch))
            rms_change[ch] = _pct(rms.get(ch), base["rms"].get(ch))
            if freq_shift[ch] is None:
                notes.append(f"{ch}: frequency shift not computed (channel missing in baseline or no dominant frequency)")
        if base["fft_resolution_hz"] != res:
            notes.append(f"baseline FFT resolution {base['fft_resolution_hz']:.4f} Hz differs from this record's {res:.4f} Hz")
    return {
        "signal_id": record.signal_id,
        "baseline_id": baseline_id,
        "sensor": record.sensor,
        "units": units,
        "sample_rate_hz": fs,
        "n_samples_used": int(data.shape[0]),
        "window_s": window_s,
        "fft_window": "hann",
        "fft_resolution_hz": res,
        "channels": list(record.channels),
        "pga": pga,
        "pga_pct_g": pga_pct_g,
        "pgv_cm_s": pgv_cm_s,
        "rms": rms,
        "dominant_frequency_hz": fdom,
        "baseline_dominant_frequency_hz": base_fdom,
        "damping_ratio": damping,
        "damping_detail": damping_detail,
        "frequency_shift_pct": freq_shift,
        "rms_change_pct": rms_change,
        "notes": notes,
    }


# ------------------------------------------------------------------------------------------------- grading


def load_seismic_rubric(name: str = SEISMIC_RUBRIC_FILE) -> dict:
    return json.loads((RUBRIC_DIR / name).read_text(encoding="utf-8"))


def _in_band(row: dict, v: float) -> bool:
    lo, hi = row.get("min"), row.get("max")
    return (lo is None or v >= lo) and (hi is None or v < hi)


def _find_row(rubric: dict, family: str, v: float) -> Optional[dict]:
    for row in rubric["rows"]:
        if row.get("family") == family and _in_band(row, v):
            return row
    return None


def _worst_channel(values: Optional[Dict[str, Optional[float]]], key=abs) -> Optional[Tuple[str, float]]:
    if not values:
        return None
    present = [(ch, v) for ch, v in values.items() if v is not None]
    if not present:
        return None
    return max(present, key=lambda cv: key(cv[1]))


def grade_signal(record: SignalRecord, ind: dict, rubric: dict) -> Finding:
    """One Finding from the indicators and the seismic rubric.

    Families evaluated: frequency_shift (worst |shift| over channels; the rubric's 'baseline' row and level U
    when there is no baseline), pga or pgv (worst channel, only when units are known), rms_change (largest
    increase). The finding takes the worst level over evaluable families, quotes every matched row verbatim
    in criteria_matched, and is U when nothing could be evaluated. confidence is 1.0 when a level was
    assigned (the arithmetic is deterministic; the uncertainty is the rubric's own source tag) and 0.0 for U.
    """
    matched: List[Tuple[dict, str]] = []  # (row, sentence for the justification)
    skipped: List[str] = []

    fs_shift = _worst_channel(ind.get("frequency_shift_pct"))
    if fs_shift is None:
        row = next((r for r in rubric["rows"] if r.get("family") == "baseline"), None)
        if row is not None:
            matched.append((row, "No baseline series: frequency shift and RMS change not assessable."))
        skipped.append("frequency_shift")
    else:
        ch, v = fs_shift
        row = _find_row(rubric, "frequency_shift", abs(v))
        f_now = (ind.get("dominant_frequency_hz") or {}).get(ch)
        f_base = (ind.get("baseline_dominant_frequency_hz") or {}).get(ch)
        sent = f"Channel {ch}: dominant frequency {f_now:.3f} Hz vs baseline {f_base:.3f} Hz, shift {v:+.1f} % (FFT resolution {ind.get('fft_resolution_hz', float('nan')):.3f} Hz)."
        if row is None:
            skipped.append(f"frequency_shift (no row covers {abs(v):.1f} %)")
        else:
            matched.append((row, sent))

    for family, key, unit in (("pga", "pga_pct_g", "%g"), ("pgv", "pgv_cm_s", "cm/s")):
        worst = _worst_channel(ind.get(key))
        if worst is None:
            continue
        ch, v = worst
        row = _find_row(rubric, family, v)
        if row is None:
            skipped.append(f"{family} (no row covers {v:.2f} {unit})")
        else:
            matched.append((row, f"Channel {ch}: {family.upper()} {v:.2f} {unit}."))
    if ind.get("pga_pct_g") is None and ind.get("pgv_cm_s") is None:
        skipped.append("pga/pgv (units unknown or sensor type without PGA/PGV)")

    rc = _worst_channel(ind.get("rms_change_pct"), key=lambda v: v)
    if rc is not None:
        ch, v = rc
        row = _find_row(rubric, "rms_change", v)
        if row is None:
            skipped.append(f"rms_change (no row covers {v:+.1f} %)")
        else:
            matched.append((row, f"Channel {ch}: RMS {ind['rms'][ch]:.4g} vs baseline, change {v:+.1f} %."))

    assessable = [(r, s) for r, s in matched if r["unified"] != "U"]
    flags: List[str] = []
    if assessable:
        worst_row, _ = max(assessable, key=lambda rs: LEVEL_ORDER[rs[0]["unified"]])
        level = worst_row["unified"]
        value = worst_row["value"]
        family = worst_row["family"]
        action = Action(code=worst_row["action"], sla_days=None, basis=f"{SEISMIC_RUBRIC_FILE} row '{value}': {worst_row.get('source', '')}".strip())
        for r, _ in assessable:
            if r.get("flag") and r["flag"] not in flags:
                flags.append(r["flag"])
        confidence = 1.0
    else:
        level, value, family = "U", "U", "not_assessable"
        action = Action(code="monitor", sla_days=None, basis="Not assessable: no evaluable indicator (no baseline and no PGA/PGV units). Collect a baseline series.")
        confidence = 0.0
    if skipped or level == "U":
        flags.append("not_measurable")
    damping = {ch: v for ch, v in (ind.get("damping_ratio") or {}).items() if v is not None}
    sentences = [s for _, s in matched]
    if damping:
        sentences.append("Damping ratio (log decrement): " + ", ".join(f"{ch} {v:.3f}" for ch, v in damping.items()) + " (indicator only, no threshold row).")
    if skipped:
        sentences.append("Not evaluated: " + "; ".join(skipped) + ".")
    return Finding(
        finding_id=f"{record.signal_id}/signal",
        asset_class=record.asset_class,
        modality="seismic",
        defect_type=f"seismic_{family}",
        native_scale=NativeScale(standard=rubric["standard"], value=value, criteria_matched=[r["criterion"] for r, _ in matched]),
        unified=Unified(level=level, uncertainty="+/-1", flags=flags),
        measurements=Measurements(area_cm2=None, crack_width_mm=None, delta_t_k=None, percent_area_rusted=None, section_loss_pct=None, confidence=confidence),
        action=action,
        justification=" ".join(sentences) if sentences else "No indicator could be evaluated.",
        evidence=Evidence(image_ids=[record.signal_id], tile="signal", signal_id=record.signal_id, baseline_id=ind.get("baseline_id")),
        model=MODEL_TAG,
    )


# -------------------------------------------------------------------------------------------------- export


def write_signal_findings(run_dir: Path, findings: List[Finding]) -> List[Finding]:
    """Merge `findings` into a run folder the way pipeline.save_findings lays it out: findings.jsonl gets one
    line per new finding (so a resumed run_cascade keeps them), findings.json / queue.csv / bridge_entry.csv
    are re-ranked over image and signal findings together, and summary.json counts are patched if present.
    A finding with an id already in the run replaces the old one."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = load_run(run_dir)["findings"]
    by_id = {f.finding_id: f for f in existing}
    jsonl = run_dir / "findings.jsonl"
    with jsonl.open("a", encoding="utf-8") as fh:
        for f in findings:
            if f.finding_id not in by_id:
                fh.write(f.model_dump_json() + "\n")
            by_id[f.finding_id] = f
    ranked = save_findings(list(by_id.values()), run_dir)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["findings"] = len(ranked)
        summary["levels"] = {lvl: sum(1 for f in ranked if f.unified.level == lvl) for lvl in LEVELS}
        summary["signal_findings"] = sum(1 for f in ranked if f.modality == "seismic")
        summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return ranked


def append_signal_row(run_dir: Path, record: SignalRecord, ind: dict, finding: Finding) -> None:
    """signals.jsonl: the record, its indicators and the finding id, for the UI's metadata strip."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "signals.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"record": record.model_dump(), "indicators": ind, "finding_id": finding.finding_id}) + "\n")


# ----------------------------------------------------------------------------------------------------- CLI


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cascade.signals", description="Grade one seismic/vibration CSV against rubrics/seismic_shm.json.")
    ap.add_argument("--csv", required=True, help="time series CSV (numeric columns; a time column is detected by name)")
    ap.add_argument("--sensor", required=True, choices=list(get_args(Sensor)))
    ap.add_argument("--rate", type=float, default=None, help="sample rate in Hz; taken from the time column when omitted")
    ap.add_argument("--baseline", default=None, help="healthy-state CSV of the same asset and channels")
    ap.add_argument("--out", required=True, help="run folder (findings.json, queue.csv, signals.jsonl)")
    ap.add_argument("--units", default=None, help="sample units, e.g. m/s2, g, mg, gal, cm/s2 (accelerometer) or m/s, cm/s, mm/s (geophone); unknown when omitted")
    ap.add_argument("--asset-class", default="bridge_element", choices=list(get_args(AssetClass)))
    ap.add_argument("--asset-id", default=None)
    ap.add_argument("--client-id", default=None)
    ap.add_argument("--captured-on", default=None, help="YYYY-MM-DD")
    ap.add_argument("--window-s", type=float, default=None, help="analyse only the first N seconds")
    ap.add_argument("--rubric", default=SEISMIC_RUBRIC_FILE)
    args = ap.parse_args(argv)

    baseline = None
    if args.baseline:
        baseline = ingest_signal_csv(Path(args.baseline), args.sensor, args.rate, args.asset_id, args.client_id, units=args.units, asset_class=args.asset_class)
    record = ingest_signal_csv(
        Path(args.csv), args.sensor, args.rate, args.asset_id, args.client_id,
        units=args.units, asset_class=args.asset_class, captured_on=args.captured_on, baseline_id=baseline.signal_id if baseline else None,
    )
    ind = indicators(record, baseline, args.window_s)
    finding = grade_signal(record, ind, load_seismic_rubric(args.rubric))
    out = Path(args.out)
    write_signal_findings(out, [finding])
    append_signal_row(out, record, ind, finding)
    fdom = ", ".join(f"{ch} {v:.3f} Hz" if v is not None else f"{ch} n/a" for ch, v in ind["dominant_frequency_hz"].items())
    print(f"{record.signal_id}: {record.n_samples} samples at {record.sample_rate_hz:g} Hz, dominant {fdom}; {finding.unified.level} ({finding.native_scale.value}) -> {out / 'findings.json'}")
    for note in ind["notes"]:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
