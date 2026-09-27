"""Core types for the NumericAgent layer: data cards, provenance, forecast and anomaly results.

Every result carries a Provenance (model id, pinned revision, licence, data card with a REAL / SYNTHETIC /
SIMULATED / INJECTED label). Results are indicators only: they never set a rubric level and never actuate anything.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO_ROOT / "eval" / "numeric"
MODELS_DIR = REPO_ROOT / "models" / "numeric"
RAW_DIR = REPO_ROOT / "data" / "raw" / "numeric"

DATA_LABELS = ("REAL", "SYNTHETIC", "SIMULATED", "INJECTED")
ADVISORY = "indicator only - human review; never switches equipment or overrides life-safety systems"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_head(root: Path = REPO_ROOT) -> Optional[str]:
    """Commit SHA of the working tree (read from .git, no subprocess); None outside a git checkout."""
    head = root / ".git" / "HEAD"
    if not head.exists():
        return None
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref:"):
        ref = root / ".git" / text.split(" ", 1)[1].strip()
        if ref.exists():
            return ref.read_text(encoding="utf-8").strip()
        packed = root / ".git" / "packed-refs"
        if packed.exists():
            name = text.split(" ", 1)[1].strip()
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(" " + name):
                    return line.split(" ", 1)[0]
        return None
    return text


@dataclass(frozen=True)
class DataCard:
    """Where a series came from. label must be one of DATA_LABELS."""

    name: str
    label: str
    source_url: str = ""
    licence: str = ""
    sha256: Optional[str] = None
    accessed: Optional[str] = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.label not in DATA_LABELS:
            raise ValueError(f"DataCard {self.name}: label {self.label!r} not in {DATA_LABELS}")


@dataclass(frozen=True)
class Provenance:
    model_id: str
    backend: str
    revision: Optional[str]
    licence: str
    data: DataCard
    params: Dict[str, Any] = field(default_factory=dict)
    created_utc: str = field(default_factory=utc_now)
    code_git_sha: Optional[str] = field(default_factory=git_head)
    note: str = ADVISORY

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForecastResult:
    """q50 is the model's point forecast; q10 and q90 are the edges of its 80% band (NaN when it has none)."""

    index: pd.DatetimeIndex
    q10: np.ndarray
    q50: np.ndarray
    q90: np.ndarray
    provenance: Provenance

    def __post_init__(self) -> None:
        n = len(self.index)
        for name in ("q10", "q50", "q90"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"ForecastResult.{name} has {len(getattr(self, name))} values for {n} timestamps")

    @property
    def has_band(self) -> bool:
        return bool(np.isfinite(self.q10).all() and np.isfinite(self.q90).all())

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"q10": self.q10, "q50": self.q50, "q90": self.q90}, index=self.index)


@dataclass(frozen=True)
class AnomalyResult:
    """scores: higher = more unusual. flags = scores >= threshold. NaN scores are never flagged."""

    index: pd.DatetimeIndex
    scores: np.ndarray
    threshold: float
    method: str
    provenance: Provenance

    @property
    def flags(self) -> np.ndarray:
        s = np.asarray(self.scores, dtype=float)
        return np.where(np.isfinite(s), s >= self.threshold, False)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"score": self.scores, "flag": self.flags}, index=self.index)


def hourly(y: pd.Series) -> pd.Series:
    """Regular hourly float series (missing hours become NaN); the index must be a DatetimeIndex."""
    if not isinstance(y.index, pd.DatetimeIndex):
        raise TypeError("series needs a DatetimeIndex")
    y = y[~y.index.duplicated(keep="last")].sort_index()
    return y.asfreq("h").astype(float)


def causal_fill(y: pd.Series, limit: int = 6) -> pd.Series:
    """Forward fill only (never backward, never two-sided interpolation), so a value never borrows the future."""
    return y.ffill(limit=limit)
