"""Numeric AI layer: forecasts with a q10/q50/q90 band and anomaly scores for meter and sensor series.

NumericAgent (agent.py) gives one interface over a seasonal-naive baseline, a scikit-learn HGB forecaster with a
split-conformal band, robust-z / IsolationForest / supervised-HGB anomaly scorers, and Chronos-2 / Chronos-Bolt
forecasts precomputed offline from pinned Hugging Face revisions. Every output carries provenance and is an
indicator only: it never sets a rubric level, switches equipment or touches a life-safety system.
"""

from .agent import HF_MODELS, NumericAgent
from .base import AnomalyResult, DataCard, ForecastResult, Provenance

__all__ = ["NumericAgent", "HF_MODELS", "DataCard", "Provenance", "ForecastResult", "AnomalyResult"]
