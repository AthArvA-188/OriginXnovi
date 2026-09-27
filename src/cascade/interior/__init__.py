"""Interior wall testing: NDT strength estimate (rebound + UPV) and field-reading schema.

- ndt: SonReb power law with per-building core calibration and conformal intervals
- readings: pydantic FieldReading schema, CSV template, and routing of each reading to the
  strength estimator (rebound/UPV) or the rule-graded interior_water rubric (moisture/RH).
Moisture is rule-graded, never ML. Every output is an estimate for an engineer to review.
"""
